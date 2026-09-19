"""Label readers: pds4_tools + pvl -> SceneMeta. Owner: Member A (Part 1). Features: DATA-01, DATA-02.

PDS4 (Chandrayaan-2, and later LRO) is read in two layers:

* the standard PDS4 parts -- which file holds the pixels, where they start, the
  pixel type and the array axes -- come from NASA's pds4_tools;
* ISRO's mission-specific fields (sun angles, corner coordinates, pixel
  resolution) live in the ``isda:`` namespace, which pds4_tools does not model,
  so they are read with targeted lxml XPath lookups.

If pds4_tools cannot handle a label at all, the array layout is read with lxml
instead (PLAN.md risk R12), so an unusual label degrades to a slower path, never
to a guess.

Honesty rule: ``SceneMeta.label_fields_verified[field]`` is True only when that
value was read from the label. Anything taken from configs/instruments.yaml or
left empty is False. A label missing a required field raises PdsParseError --
there is no fallback to loading pixels with a generic image reader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from lxml import etree

from chandralign.contracts import SceneMeta
from chandralign.io.instruments import UnknownInstrumentError, detect_instrument, get_spec

# PDS4 data_type -> numpy dtype. Endianness is explicit, never assumed.
PDS4_DTYPES = {
    "UnsignedByte": "u1", "SignedByte": "i1",
    "UnsignedMSB2": ">u2", "UnsignedLSB2": "<u2", "SignedMSB2": ">i2", "SignedLSB2": "<i2",
    "UnsignedMSB4": ">u4", "UnsignedLSB4": "<u4", "SignedMSB4": ">i4", "SignedLSB4": "<i4",
    "IEEE754MSBSingle": ">f4", "IEEE754LSBSingle": "<f4",
    "IEEE754MSBDouble": ">f8", "IEEE754LSBDouble": "<f8",
}

MISSION_NAMES = {"chandrayaan-2": "CH2", "lunar reconnaissance orbiter": "LRO"}

CORNER_ORDER = ("upper_left", "upper_right", "lower_right", "lower_left")  # clockwise


class PdsParseError(ValueError):
    """The label is unreadable or lacks a field we cannot work without."""


@dataclass(frozen=True)
class ArrayLayout:
    """How the pixel file is laid out on disk, exactly as the label states it."""
    raster_path: Path
    offset_bytes: int
    dtype: str                      # numpy dtype string with explicit byte order, e.g. "<u2"
    axis_names: tuple[str, ...]     # e.g. ("Line", "Sample") or ("BAND", "LINE", "SAMPLE")
    shape: tuple[int, ...]          # same order as axis_names
    array_type: str                 # e.g. "Array_2D_Image", "Array_3D_Spectrum"
    file_size_bytes: Optional[int]  # as declared in the label
    md5: Optional[str]              # as declared in the label
    parsed_with: str                # "pds4_tools" | "lxml"


# ----------------------------------------------------------------------------- xml helpers

def _load_xml(xml_path: Path) -> etree._ElementTree:
    try:
        return etree.parse(str(xml_path))
    except (OSError, etree.XMLSyntaxError) as exc:
        raise PdsParseError(f"{xml_path.name}: not a readable XML label ({exc})") from exc


def _find(tree, local_name: str, within: Optional[str] = None):
    """First element with this local name (namespace-agnostic), optionally under a parent."""
    path = f"//*[local-name()='{within}']" if within else ""
    hits = tree.xpath(f"{path}//*[local-name()='{local_name}']")
    return hits[0] if hits else None


def _text(tree, local_name: str, within: Optional[str] = None) -> Optional[str]:
    el = _find(tree, local_name, within)
    return el.text.strip() if el is not None and el.text and el.text.strip() else None


def _float(tree, local_name: str, within: Optional[str] = None) -> Optional[float]:
    value = _text(tree, local_name, within)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise PdsParseError(f"<{local_name}> is not a number: {value!r}") from exc


# ----------------------------------------------------------------------------- array layout

def _layout_from_pds4_tools(xml_path: Path) -> ArrayLayout:
    import pds4_tools

    structures = pds4_tools.read(str(xml_path), lazy_load=True, quiet=True, no_scale=True)
    arrays = [s for s in structures if s.type.startswith("Array")]
    if not arrays:
        raise PdsParseError(f"{xml_path.name}: no Array structure in File_Area_Observational")
    st = arrays[0]
    md = st.meta_data
    axes = md.get_axis_arrays(sort=True)
    return _layout(
        xml_path,
        file_name=Path(st.parent_filename).name,  # pds4_tools returns a full path
        offset=int(md["offset"]),
        data_type=md["Element_Array"]["data_type"],
        axis_names=tuple(a["axis_name"] for a in axes),
        shape=tuple(int(a["elements"]) for a in axes),
        array_type=st.type,
        parsed_with="pds4_tools",
    )


def _layout_from_lxml(xml_path: Path, tree) -> ArrayLayout:
    arrays = tree.xpath("//*[local-name()='File_Area_Observational']/*[starts-with(local-name(),'Array')]")
    if not arrays:
        raise PdsParseError(f"{xml_path.name}: no Array_* element in File_Area_Observational")
    arr = arrays[0]
    axes = sorted(
        arr.xpath("*[local-name()='Axis_Array']"),
        key=lambda a: int(a.xpath("string(*[local-name()='sequence_number'])") or 0),
    )
    offset = arr.xpath("string(*[local-name()='offset'])")
    data_type = arr.xpath("string(*[local-name()='Element_Array']/*[local-name()='data_type'])")
    file_name = tree.xpath("string(//*[local-name()='File_Area_Observational']/*[local-name()='File']/*[local-name()='file_name'])")
    if not (offset and data_type and file_name and axes):
        raise PdsParseError(f"{xml_path.name}: array description is incomplete")
    return _layout(
        xml_path,
        file_name=file_name.strip(),
        offset=int(offset),
        data_type=data_type.strip(),
        axis_names=tuple(a.xpath("string(*[local-name()='axis_name'])").strip() for a in axes),
        shape=tuple(int(a.xpath("string(*[local-name()='elements'])")) for a in axes),
        array_type=etree.QName(arr).localname,
        parsed_with="lxml",
    )


def _layout(xml_path, *, file_name, offset, data_type, axis_names, shape, array_type, parsed_with):
    if data_type not in PDS4_DTYPES:
        raise PdsParseError(f"{xml_path.name}: unsupported data_type {data_type!r}")
    tree = _load_xml(xml_path)
    size = _text(tree, "file_size", "File")
    return ArrayLayout(
        raster_path=xml_path.parent / file_name,
        offset_bytes=offset,
        dtype=np.dtype(PDS4_DTYPES[data_type]).str,
        axis_names=axis_names,
        shape=shape,
        array_type=array_type,
        file_size_bytes=int(size) if size else None,
        md5=_text(tree, "md5_checksum", "File"),
        parsed_with=parsed_with,
    )


def read_array_layout(xml_path: str | Path) -> ArrayLayout:
    """The pixel file's on-disk layout, from the label. pds4_tools first, lxml if it fails."""
    xml_path = Path(xml_path)
    tree = _load_xml(xml_path)  # fail fast, with our own error, on broken XML
    try:
        return _layout_from_pds4_tools(xml_path)
    except PdsParseError:
        raise
    except Exception:  # pds4_tools rejects some mission labels (PLAN.md risk R12)
        return _layout_from_lxml(xml_path, tree)


# ----------------------------------------------------------------------------- corners

def read_corner_sets(xml_path: str | Path) -> dict[str, list[tuple[float, float]]]:
    """Every corner-coordinate set in an ISRO label, keyed by its quality level.

    Returns e.g. {"system": [...4 corners...], "refined": [...]}, each clockwise from
    upper-left, as (lat, lon). ``refined`` corners in CH-2 labels were adjusted
    against the reference named in ``isda:reference_data_used`` (often SELENE), so
    they are NOT independent of our references -- see parse_pds4 for why the
    default is ``system``.
    """
    tree = _load_xml(Path(xml_path))
    sets = {}
    for key, block in (("system", "System_Level_Coordinates"), ("refined", "Refined_Corner_Coordinates")):
        if _find(tree, block) is None:
            continue
        corners = []
        for corner in CORNER_ORDER:
            lat = _float(tree, f"{corner}_latitude", block)
            lon = _float(tree, f"{corner}_longitude", block)
            if lat is None or lon is None:
                raise PdsParseError(f"{Path(xml_path).name}: {block} lacks {corner}")
            corners.append((lat, lon))
        sets[key] = corners
    return sets


# ----------------------------------------------------------------------------- main entry

def parse_pds4(xml_path: str | Path, corners: str = "system") -> SceneMeta:
    """Read a PDS4 label into SceneMeta.

    corners: which ISRO corner set fills ``corner_latlon``. Default "system", because
    "refined" corners were adjusted against a reference image and would leak that
    reference into any evaluation against it.
    """
    xml_path = Path(xml_path)
    tree = _load_xml(xml_path)
    layout = read_array_layout(xml_path)
    verified: dict[str, bool] = {}

    # identity ---------------------------------------------------------------
    product_id = layout.raster_path.stem
    verified["product_id"] = True
    try:
        instrument = detect_instrument(product_id)
    except UnknownInstrumentError as exc:
        raise PdsParseError(f"{xml_path.name}: {exc}") from exc
    verified["instrument"] = True           # derived from the label's own file_name
    spec = get_spec(instrument)

    mission_name = _text(tree, "name", "Investigation_Area")
    mission = MISSION_NAMES.get((mission_name or "").lower())
    verified["mission"] = mission is not None
    if mission is None:
        mission = spec.mission

    # array ------------------------------------------------------------------
    names = [n.lower() for n in layout.axis_names]
    try:
        lines = layout.shape[names.index("line")]
        samples = layout.shape[names.index("sample")]
    except ValueError as exc:
        raise PdsParseError(f"{xml_path.name}: axes {layout.axis_names} lack Line/Sample") from exc
    n_bands = layout.shape[names.index("band")] if "band" in names else 1
    verified["array_shape"] = verified["dtype"] = verified["n_bands"] = True

    # resolution -------------------------------------------------------------
    gsd = _float(tree, "pixel_resolution")
    verified["gsd_m"] = gsd is not None
    if gsd is None:
        gsd = spec.gsd_m

    # corners ----------------------------------------------------------------
    corner_sets = read_corner_sets(xml_path) if _find(tree, "Geometry_Parameters") is not None else {}
    if corners not in ("system", "refined"):
        raise ValueError("corners must be 'system' or 'refined'")
    corner_latlon = corner_sets.get(corners, [])
    verified["corner_latlon"] = bool(corner_latlon)

    # illumination and time ----------------------------------------------------
    sun_azimuth = _float(tree, "sun_azimuth")
    incidence = _float(tree, "solar_incidence")
    start = _text(tree, "start_date_time", "Time_Coordinates")
    verified["sub_solar_azimuth_deg"] = sun_azimuth is not None
    verified["solar_incidence_deg"] = incidence is not None
    verified["acquisition_utc"] = start is not None
    # Not present in any CH-2 label we hold; computed later from geometry (GEO-02).
    verified["emission_deg"] = verified["phase_deg"] = False
    # Only described in prose in CH-2 labels; the value comes from the registry.
    verified["wavelength_nm"] = False

    verified["raster_path"] = True          # the label's File/file_name

    return SceneMeta(
        product_id=product_id,
        instrument=instrument,
        mission=mission,
        gsd_m=gsd,
        n_bands=n_bands,
        wavelength_nm=spec.wavelength_nm,
        array_shape=(lines, samples),
        dtype=layout.dtype,
        corner_latlon=corner_latlon,
        sub_solar_azimuth_deg=sun_azimuth,
        solar_incidence_deg=incidence,
        emission_deg=None,
        phase_deg=None,
        acquisition_utc=start,
        label_path=xml_path,
        raster_path=layout.raster_path,
        label_fields_verified=verified,
    )


def parse_pds3(lbl_path: str | Path) -> SceneMeta:
    """PDS3 / ODL labels (SELENE). Implemented in Step 6 (DATA-02)."""
    raise NotImplementedError("PDS3 reader lands in Step 6 (DATA-02)")
