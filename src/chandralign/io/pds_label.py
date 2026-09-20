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

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
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
    """The pixel file's on-disk layout, from the label.

    PDS3 labels go through pvl. PDS4 labels go through pds4_tools, or lxml if it fails.
    Results are cached per label file and reused only while the file's size and
    modification time are unchanged -- re-parsing a 77 kB IIRS label on every
    single-pixel read cost up to ~230 ms.
    """
    xml_path = Path(xml_path)
    try:
        st = xml_path.stat()
    except OSError as exc:
        raise PdsParseError(f"{xml_path.name}: cannot read ({exc})") from exc
    return _cached_layout(str(xml_path.resolve()), st.st_mtime_ns, st.st_size)


def clear_layout_cache() -> None:
    """Forget cached layouts (e.g. after swapping libraries in a test)."""
    _cached_layout.cache_clear()


@lru_cache(maxsize=256)
def _cached_layout(path: str, mtime_ns: int, size: int) -> ArrayLayout:
    return _read_array_layout_uncached(Path(path))


def _read_array_layout_uncached(xml_path: Path) -> ArrayLayout:
    if is_pds3(xml_path):
        return _pds3_layout(xml_path, _pvl_label(xml_path))
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


def _wavelength_range(tree) -> Optional[tuple[float, float]]:
    """(min, max) nm from a stated centre wavelength and bandwidth, if the label has both."""
    centre = _float(tree, "center_filter_wavelength")
    width = _float(tree, "bandwidth")
    if centre is None or width is None:
        return None
    return (centre - width / 2.0, centre + width / 2.0)


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
    # Not present in any CH-2 or LRO label we hold; computed later from geometry (GEO-02).
    verified["emission_deg"] = verified["phase_deg"] = False
    # LRO states centre + bandwidth (img: namespace); CH-2 only describes it in prose,
    # so CH-2 falls back to the registry value, marked unverified.
    wavelength = _wavelength_range(tree)
    verified["wavelength_nm"] = wavelength is not None

    verified["raster_path"] = True          # the label's File/file_name

    return SceneMeta(
        product_id=product_id,
        instrument=instrument,
        mission=mission,
        gsd_m=gsd,
        n_bands=n_bands,
        wavelength_nm=wavelength or spec.wavelength_nm,
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


# ============================================================================= PDS3
#
# SELENE / Kaguya (and many older NASA products) use PDS3: a plain-text ODL label,
# parsed with pvl. Pixels are read with the same memory-mapped window reader as
# PDS4, using the layout below. planetaryimage is deliberately not used: it loads
# the whole array into memory, so it cannot serve windowed reads, and its own
# README calls it alpha quality (PLAN.md section 3.2).

# (SAMPLE_TYPE, SAMPLE_BITS) -> numpy dtype. Byte order is explicit.
_PDS3_KIND = {
    "MSB_UNSIGNED_INTEGER": ">u", "UNSIGNED_INTEGER": ">u", "MAC_UNSIGNED_INTEGER": ">u",
    "SUN_UNSIGNED_INTEGER": ">u", "LSB_UNSIGNED_INTEGER": "<u", "PC_UNSIGNED_INTEGER": "<u",
    "VAX_UNSIGNED_INTEGER": "<u",
    "MSB_INTEGER": ">i", "INTEGER": ">i", "MAC_INTEGER": ">i", "SUN_INTEGER": ">i",
    "LSB_INTEGER": "<i", "PC_INTEGER": "<i", "VAX_INTEGER": "<i",
    "IEEE_REAL": ">f", "REAL": ">f", "FLOAT": ">f", "MAC_REAL": ">f", "SUN_REAL": ">f",
    "PC_REAL": "<f",
}
_PDS3_STORAGE = {
    "BAND_SEQUENTIAL": ("BAND", "LINE", "SAMPLE"),
    "LINE_INTERLEAVED": ("LINE", "BAND", "SAMPLE"),
    "SAMPLE_INTERLEAVED": ("LINE", "SAMPLE", "BAND"),
}
_PDS3_MISSIONS = {"selene": "SELENE", "kaguya": "SELENE", "lunar reconnaissance orbiter": "LRO"}
# INSTRUMENT_ID values that name a camera family rather than one camera.
_PDS3_INSTRUMENT_IDS = {"LROC": {"NAC", "WAC"}}


_ODL_END = re.compile(rb"(?m)^END[ \t\r]*$")   # tolerate CR / CRCRLF line endings
_MAX_LABEL_BYTES = 1 << 20


def _pvl_label(lbl_path: Path):
    """Parse an ODL label. Reads only the text up to END, so a label attached to the
    front of a multi-GB image file never pulls the pixels into memory."""
    import pvl

    try:
        with lbl_path.open("rb") as fh:
            head = fh.read(_MAX_LABEL_BYTES)
    except OSError as exc:
        raise PdsParseError(f"{lbl_path.name}: cannot read ({exc})") from exc
    end = _ODL_END.search(head)
    if end is None:
        raise PdsParseError(f"{lbl_path.name}: no END statement in the first {_MAX_LABEL_BYTES:,} bytes")
    try:
        return pvl.loads(head[: end.end()].decode("ascii", errors="replace"))
    except (pvl.exceptions.LexerError, pvl.exceptions.ParseError, ValueError) as exc:
        raise PdsParseError(f"{lbl_path.name}: not a readable PDS3 label ({exc})") from exc


def _pvl_value(value):
    """Strip pvl units: Quantity(3.0, 'deg') -> 3.0. 'UNK' / 'N/A' -> None."""
    value = getattr(value, "value", value)
    if isinstance(value, str) and value.strip().upper() in ("UNK", "N/A", "NULL", ""):
        return None
    return value


def _pvl_number(block, key) -> Optional[float]:
    if key not in block:
        return None
    value = _pvl_value(block[key])
    return None if value is None else float(value)


def _iso_utc(value) -> Optional[str]:
    """PDS3 times (UTC by definition) as ISO-8601 with a trailing Z, matching PDS4 labels.

    pvl turns them into datetime objects, whose str() is '2022-09-09 11:07:18+00:00'.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value.isoformat(timespec="microseconds") + "Z"
    return str(value)


def _pds3_find(label, key: str):
    """A top-level key, or the same key one level down inside a FILE-type object.

    SELENE labels put IMAGE and ^IMAGE at the top level; LOLA nests them inside
    OBJECT = UNCOMPRESSED_FILE. Both are valid PDS3.
    """
    if key in label:
        return label[key]
    for _, value in label.items():
        if hasattr(value, "keys") and key in value:
            return value[key]
    return None


def _resolve_file(parent: Path, name: str) -> Path:
    """`name` in `parent`, matching case-insensitively when the exact name is absent.

    PDS3 labels often say 'LDEM_..._IMG' while the archive serves 'ldem_....img';
    Windows would not notice, Linux would fail to find the file.
    """
    exact = parent / name
    if exact.exists():
        return exact
    lowered = name.lower()
    for candidate in parent.iterdir() if parent.is_dir() else []:
        if candidate.name.lower() == lowered:
            return candidate
    return exact


def _pds3_image_offset(label, lbl_path: Path) -> tuple[Path, int]:
    """Resolve ^IMAGE to (pixel file, byte offset). Every PDS3 pointer form is handled:

    ^IMAGE = 12                          record 12 of this file (1-based)
    ^IMAGE = 1024 <BYTES>                byte 1024 of this file (1-based)
    ^IMAGE = ("x.img")                   start of x.img
    ^IMAGE = ("x.img", 5)                record 5 of x.img
    ^IMAGE = ("x.img", 1 <BYTES>)        byte 1 of x.img
    """
    pointer = _pds3_find(label, "^IMAGE")
    if pointer is None:
        raise PdsParseError(f"{lbl_path.name}: no ^IMAGE pointer")
    file_name, location = None, pointer
    if isinstance(pointer, (list, tuple)):
        file_name = pointer[0]
        location = pointer[1] if len(pointer) > 1 else 1
    elif isinstance(pointer, str):
        file_name, location = pointer, 1

    units = str(getattr(location, "units", "") or "").upper()
    position = int(getattr(location, "value", location))
    if position < 1:
        raise PdsParseError(f"{lbl_path.name}: ^IMAGE position must be >= 1, got {position}")
    if units == "BYTES":
        offset = position - 1
    else:
        record_bytes = _pvl_value(_pds3_find(label, "RECORD_BYTES"))
        if not record_bytes:
            raise PdsParseError(f"{lbl_path.name}: ^IMAGE is a record number but RECORD_BYTES is missing")
        offset = (position - 1) * int(record_bytes)

    raster = _resolve_file(lbl_path.parent, file_name) if file_name else lbl_path
    return raster, offset


def _pds3_layout(lbl_path: Path, label) -> ArrayLayout:
    image = _pds3_find(label, "IMAGE")
    if image is None:
        raise PdsParseError(f"{lbl_path.name}: no IMAGE object")
    try:
        lines, samples = int(image["LINES"]), int(image["LINE_SAMPLES"])
        sample_type = str(image["SAMPLE_TYPE"]).upper()
        sample_bits = int(image["SAMPLE_BITS"])
    except KeyError as exc:
        raise PdsParseError(f"{lbl_path.name}: IMAGE object lacks {exc}") from exc
    if sample_type not in _PDS3_KIND:
        raise PdsParseError(f"{lbl_path.name}: unsupported SAMPLE_TYPE {sample_type!r}")
    if sample_bits % 8:
        raise PdsParseError(f"{lbl_path.name}: SAMPLE_BITS={sample_bits} is not whole bytes")
    kind = _PDS3_KIND[sample_type]
    dtype = np.dtype(("|" if sample_bits == 8 else kind[0]) + kind[1] + str(sample_bits // 8)).str

    bands = int(_pvl_value(image.get("BANDS")) or 1)
    if bands == 1:
        axis_names, shape = ("LINE", "SAMPLE"), (lines, samples)
    else:
        storage = str(_pvl_value(image.get("BAND_STORAGE_TYPE")) or "BAND_SEQUENTIAL").upper()
        if storage not in _PDS3_STORAGE:
            raise PdsParseError(f"{lbl_path.name}: unsupported BAND_STORAGE_TYPE {storage!r}")
        axis_names = _PDS3_STORAGE[storage]
        sizes = {"BAND": bands, "LINE": lines, "SAMPLE": samples}
        shape = tuple(sizes[a] for a in axis_names)

    raster, offset = _pds3_image_offset(label, lbl_path)
    return ArrayLayout(
        raster_path=raster,
        offset_bytes=offset,
        dtype=dtype,
        axis_names=axis_names,
        shape=shape,
        array_type="PDS3_IMAGE",
        file_size_bytes=None,   # PDS3 labels do not declare it
        md5=None,
        parsed_with="pvl",
    )


def is_pds3(label_path: str | Path) -> bool:
    """PDS3 labels are ODL text starting with PDS_VERSION_ID; PDS4 labels are XML."""
    with Path(label_path).open("rb") as fh:
        head = fh.read(64).lstrip()
    return head.upper().startswith(b"PDS_VERSION_ID")


def parse_pds3(lbl_path: str | Path) -> SceneMeta:
    """Read a PDS3 (ODL) label into SceneMeta. Same honesty rule as parse_pds4."""
    lbl_path = Path(lbl_path)
    label = _pvl_label(lbl_path)
    layout = _pds3_layout(lbl_path, label)
    verified: dict[str, bool] = {}

    product_id = _pvl_value(label.get("PRODUCT_ID"))
    verified["product_id"] = product_id is not None
    product_id = str(product_id or lbl_path.stem)
    try:
        instrument = detect_instrument(product_id)
    except UnknownInstrumentError as exc:
        raise PdsParseError(f"{lbl_path.name}: {exc}") from exc
    declared = _pvl_value(label.get("INSTRUMENT_ID"))
    if declared is not None and instrument not in _PDS3_INSTRUMENT_IDS.get(str(declared).upper(), {str(declared).upper()}):
        raise PdsParseError(
            f"{lbl_path.name}: product ID says {instrument} but INSTRUMENT_ID says {declared}"
        )
    verified["instrument"] = True
    spec = get_spec(instrument)

    mission = _PDS3_MISSIONS.get(str(_pvl_value(label.get("MISSION_NAME")) or "").lower())
    verified["mission"] = mission is not None
    mission = mission or spec.mission

    names = [a.lower() for a in layout.axis_names]
    lines, samples = layout.shape[names.index("line")], layout.shape[names.index("sample")]
    n_bands = layout.shape[names.index("band")] if "band" in names else 1
    verified["array_shape"] = verified["dtype"] = verified["n_bands"] = True

    projection = label.get("IMAGE_MAP_PROJECTION", {})
    scale_km = _pvl_number(projection, "MAP_SCALE")
    verified["gsd_m"] = scale_km is not None
    gsd = scale_km * 1000.0 if scale_km is not None else spec.gsd_m

    corners = []
    for corner in ("UPPER_LEFT", "UPPER_RIGHT", "LOWER_RIGHT", "LOWER_LEFT"):   # clockwise
        lat = _pvl_number(label, f"{corner}_LATITUDE")
        lon = _pvl_number(label, f"{corner}_LONGITUDE")
        if lat is None or lon is None:
            corners = []
            break
        corners.append((lat, lon))
    verified["corner_latlon"] = bool(corners)

    # Mosaics (e.g. TC ortho maps) have START_TIME = UNK and no single sun position.
    # They are photometrically normalised to STANDARD_GEOMETRY, but their shadows
    # still come from the original passes, so we do NOT report that as a sun angle.
    start = _pvl_value(label.get("START_TIME"))
    sun_azimuth = _pvl_number(label, "SOLAR_AZIMUTH") if "SOLAR_AZIMUTH" in label else None
    incidence = _pvl_number(label, "INCIDENCE_ANGLE") if "INCIDENCE_ANGLE" in label else None
    emission = _pvl_number(label, "EMISSION_ANGLE") if "EMISSION_ANGLE" in label else None
    phase = _pvl_number(label, "PHASE_ANGLE") if "PHASE_ANGLE" in label else None
    verified["acquisition_utc"] = start is not None
    verified["sub_solar_azimuth_deg"] = sun_azimuth is not None
    verified["solar_incidence_deg"] = incidence is not None
    verified["emission_deg"] = emission is not None
    verified["phase_deg"] = phase is not None
    centre = _pvl_number(label, "CENTER_FILTER_WAVELENGTH")
    width = _pvl_number(label, "BANDWIDTH")
    wavelength = (centre - width / 2.0, centre + width / 2.0) if centre is not None and width is not None else None
    verified["wavelength_nm"] = wavelength is not None
    verified["raster_path"] = True

    return SceneMeta(
        product_id=product_id,
        instrument=instrument,
        mission=mission,
        gsd_m=gsd,
        n_bands=n_bands,
        wavelength_nm=wavelength or spec.wavelength_nm,
        array_shape=(lines, samples),
        dtype=layout.dtype,
        corner_latlon=corners,
        sub_solar_azimuth_deg=sun_azimuth,
        solar_incidence_deg=incidence,
        emission_deg=emission,
        phase_deg=phase,
        acquisition_utc=_iso_utc(start),
        label_path=lbl_path,
        raster_path=layout.raster_path,
        label_fields_verified=verified,
    )


def read_pds3_image_info(lbl_path: str | Path) -> dict:
    """Value-handling fields from a PDS3 IMAGE object: no-data value, valid range,
    scaling to physical units, and the producer's own statistics (if stated)."""
    image = _pds3_find(_pvl_label(Path(lbl_path)), "IMAGE")
    keys = ("DUMMY", "VALID_MINIMUM", "VALID_MAXIMUM", "SCALING_FACTOR", "OFFSET",
            "MINIMUM", "MAXIMUM", "AVERAGE", "STDEV", "IMAGE_VALUE_TYPE", "UNIT")
    return {k.lower(): _pvl_value(image[k]) for k in keys if k in image}


@dataclass(frozen=True)
class SpecialValues:
    """Pixel values the label says are not real measurements."""
    nodata: tuple[float, ...]       # missing / null / dummy / saturation codes
    valid_min: Optional[float]
    valid_max: Optional[float]
    declared: bool                  # False = the label states none of these


_PDS4_SPECIAL = ("missing_constant", "invalid_constant", "unknown_constant", "not_applicable_constant",
                 "high_instrument_saturation", "high_representation_saturation",
                 "low_instrument_saturation", "low_representation_saturation")
_PDS3_SPECIAL = ("DUMMY", "NULL", "MISSING_CONSTANT", "LOW_REPR_SATURATION", "LOW_INSTR_SATURATION",
                 "HIGH_REPR_SATURATION", "HIGH_INSTR_SATURATION")


def read_special_values(label_path: str | Path) -> SpecialValues:
    """No-data and saturation codes plus the valid range, exactly as the label states them.

    Saturated pixels are listed with no-data: a clipped value is not a measurement
    and must not be matched. A label that states nothing yields declared=False --
    callers then treat only NaN/inf as invalid, rather than guessing a fill value.
    """
    label_path = Path(label_path)
    if is_pds3(label_path):
        image = _pds3_find(_pvl_label(label_path), "IMAGE")
        codes = [_pvl_value(image[k]) for k in _PDS3_SPECIAL if k in image]
        vmin = _pvl_value(image.get("VALID_MINIMUM"))
        vmax = _pvl_value(image.get("VALID_MAXIMUM"))
    else:
        tree = _load_xml(label_path)
        codes = [_float(tree, k, "Special_Constants") for k in _PDS4_SPECIAL]
        vmin = _float(tree, "valid_minimum", "Special_Constants")
        vmax = _float(tree, "valid_maximum", "Special_Constants")
    nodata = tuple(sorted({float(c) for c in codes if c is not None}))
    vmin = float(vmin) if vmin is not None else None
    vmax = float(vmax) if vmax is not None else None
    return SpecialValues(nodata, vmin, vmax, declared=bool(nodata) or vmin is not None or vmax is not None)


def parse_label(label_path: str | Path, **kwargs) -> SceneMeta:
    """PDS3 or PDS4, decided by the label's content, not its file extension."""
    return parse_pds3(label_path) if is_pds3(label_path) else parse_pds4(label_path, **kwargs)


# ----------------------------------------------------------------------------- viewing geometry

@dataclass(frozen=True)
class ViewingGeometry:
    """Where the spacecraft was and how it was pointing, as the label states it.

    These are SCENE-level values -- one per product -- and they are the only
    spacecraft information Chandrayaan-2 labels carry. They are kept out of
    SceneMeta because SceneMeta is a frozen contract shared by all three parts;
    geometry/solar.py reads them directly for the per-pixel layers (GEO-02).

    Every field is None when the label does not state it. Nothing is defaulted.
    """
    altitude_km: Optional[float]
    focal_length_mm: Optional[float]
    detector_pixel_width_um: Optional[float]
    roll_deg: Optional[float]
    pitch_deg: Optional[float]
    yaw_deg: Optional[float]
    orbit_limb_direction: Optional[str]      # "Ascending" | "Descending"
    line_exposure: Optional[float] = None    # AS STATED, in the unit the label declares
    line_exposure_unit: Optional[str] = None  # the declared unit, which may be wrong
    fields_verified: dict[str, bool] = field(default_factory=dict)

    @property
    def line_period_s(self) -> Optional[float]:
        """The line period in seconds, TAKING THE DECLARED UNIT AT FACE VALUE.

        Deliberately not corrected: OHRC labels declare unit="ms" on a value that is
        really microseconds, and evaluate/groundtruth.py detects that by checking this
        against orbital mechanics. Silently fixing it here would hide the defect.
        """
        scale = {"s": 1.0, "sec": 1.0, "ms": 1e-3, "millisecond": 1e-3,
                 "us": 1e-6, "microsecond": 1e-6, "micros": 1e-6}
        if self.line_exposure is None or self.line_exposure_unit is None:
            return None
        factor = scale.get(self.line_exposure_unit.strip().lower())
        return None if factor is None else self.line_exposure * factor


_VIEWING_FIELDS = {
    "altitude_km": "spacecraft_altitude",
    "focal_length_mm": "focal_length",
    "detector_pixel_width_um": "detector_pixel_width",
    "roll_deg": "roll",
    "pitch_deg": "pitch",
    "yaw_deg": "yaw",
}


def read_viewing_geometry(label_path: str | Path) -> ViewingGeometry:
    """Spacecraft altitude and pointing from a PDS4 label.

    PDS3 labels (our LRO and SELENE references) do not carry these ISRO-specific
    fields, so every value comes back None with fields_verified all False -- which
    is the honest answer, not an error.
    """
    label_path = Path(label_path)
    if is_pds3(label_path):
        return ViewingGeometry(None, None, None, None, None, None, None,
                               {name: False for name in _VIEWING_FIELDS})
    tree = _load_xml(label_path)
    values = {name: _float(tree, tag) for name, tag in _VIEWING_FIELDS.items()}
    limb = _text(tree, "orbit_limb_direction")
    exposure_el = _find(tree, "line_exposure_duration")
    exposure = _float(tree, "line_exposure_duration")
    unit = exposure_el.get("unit") if exposure_el is not None else None
    return ViewingGeometry(orbit_limb_direction=limb, line_exposure=exposure,
                           line_exposure_unit=unit,
                           fields_verified={n: v is not None for n, v in values.items()},
                           **values)
