"""LRO NAC/WAC and SELENE TC/MI reference loading. Owner: Member A (Part 1).
Features: DATA-06 (NAC), DATA-07 (WAC), DATA-08 (TC), DATA-09 (MI).

One call loads any of the four references, because the pipeline must not branch on
which camera a reference came from:

    ref = load_reference(path)          # -> Reference
    ref.meta                            # SceneMeta, same contract as a CH-2 product
    ref.footprint                       # ground outline (label corners or ODE record)
    ref.band_names, ref.band_wavelengths_nm
    plane = ref.read(Window(0, 0, 512, 512))   # ImagePlane, masked, 0..1

WHAT EACH REFERENCE ACTUALLY GIVES US, measured on the products we hold rather
than taken from the plan:

    NAC   0.5 m   1 band    PDS3 attached label; corners in the label
    WAC   100 m   1 band    PDS4; NO corners in the label -- the footprint comes
                            from the saved ODE record and is tagged ode_catalogue
    TC    7.4 m   1 band    PDS3 map-projected tile; corners from the projection
    MI    14.8 m  9 bands   PDS3 map-projected tile; a 16 MB elevation backplane
                            sits AHEAD of the image in the same file

THREE THINGS REAL PRODUCTS BROKE, all found here and fixed in io/pds_label.py:

  1. WAC declares its special constants as HEX BIT PATTERNS (0xFF7FFFFB), not
     decimal. Read as an integer that is 4287102971; reinterpreted as the array's
     float32 it is -3.4028227e+38, which is exactly the fill value in the pixels.
     A reader that does not reinterpret the bits masks nothing at all.
  2. MI states NINE CENTER_FILTER_WAVELENGTH values, one per band. A reader
     assuming a scalar raises TypeError on the real label.
  3. Two of the four WAC products covering our test box are NIGHT-SIDE (ODE
     incidence 166 deg and 179 deg). Their reflectance is ~0.000 where a sunlit
     product reads 0.016-0.095. `sunlit` reports this, because a black reference
     is worse than no reference: it matches nothing and explains nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from chandralign.contracts import ImagePlane, SceneMeta
from chandralign.geometry.footprint import Footprint, FootprintError, footprint_of
from chandralign.io.ode_client import catalogue_number, find_saved_record
from chandralign.io.pds_label import (
    SpecialValues,
    is_pds3,
    parse_label,
    read_special_values,
)
from chandralign.io.pds_raster import Window
from chandralign.io.tiling import read_tile

REFERENCE_INSTRUMENTS = ("NAC", "WAC", "TC", "MI")

# A reference whose mean reflectance is below this is not usefully illuminated.
# Measured: our two night-side WAC products read ~0.000 where the sunlit pair of
# the same camera over the same ground read 0.035.
DARK_REFLECTANCE = 0.005


class NotAReferenceError(ValueError):
    """The product is not one of the four reference cameras."""


@dataclass(frozen=True)
class Reference:
    """A reference product, ready to pair with a Chandrayaan-2 scene."""
    meta: SceneMeta
    special: SpecialValues
    footprint: Optional[Footprint]
    footprint_error: Optional[str]
    band_names: tuple[str, ...]
    band_wavelengths_nm: tuple[float, ...]
    catalogue: Optional[dict]

    @property
    def instrument(self) -> str:
        return str(self.meta.instrument)

    @property
    def independent_of_selene(self) -> bool:
        """False for SELENE's own cameras: never judge a SELENE registration with these."""
        return self.meta.mission != "SELENE"

    @property
    def incidence_deg(self) -> Optional[float]:
        """Sun angle, from the label where stated, else the ODE catalogue."""
        if self.meta.solar_incidence_deg is not None:
            return self.meta.solar_incidence_deg
        if self.catalogue is None:
            return None
        # A saved record is {"pdsid": ..., "record": {...ODE fields...}}.
        fields = self.catalogue.get("record", self.catalogue)
        return catalogue_number(fields, "Incidence_angle")

    def read(self, window: Window, band: Optional[int] = None) -> ImagePlane:
        """One window as an ImagePlane, masked with the label's own special values."""
        return read_tile(self.meta, window, band=band, special=self.special)

    def raw(self, window: Window, band: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """(values, valid_mask) in the product's OWN units, unnormalised.

        read()/read_tile stretches each window to 0..1, which is right for matching
        and wrong for asking how bright something is: a black night-side scene
        stretches to full range just as a sunlit one does.
        """
        from chandralign.io.pds_raster import read_raster
        from chandralign.io.tiling import valid_mask

        array = read_raster(self.meta, window, bands=band)
        if array.ndim == 3:
            array = array[0]
        return array, valid_mask(array, self.special)

    def sunlit(self, window: Optional[Window] = None) -> dict:
        """Is this reference actually lit? Measured from RAW pixels, not assumed.

        A night-side product parses perfectly and is useless: it has no features to
        match. The incidence angle is not enough on its own -- it comes from the
        catalogue for WAC and can be absent entirely -- so the pixels decide.

        This MUST read raw values. An earlier version used the normalised plane and
        reported our two night-side WAC products as BRIGHTER than the sunlit pair of
        the same camera over the same ground, because a min-max stretch hides exactly
        the thing being asked about.
        """
        rows, samples = self.meta.array_shape
        if window is None:
            size = min(256, rows, samples)
            window = Window(max(0, rows // 2 - size // 2), max(0, samples // 2 - size // 2),
                            size, size)
        values, mask = self.raw(window, band=0 if self.meta.n_bands > 1 else None)
        kept = values[mask].astype(np.float64)
        mean = float(kept.mean()) if kept.size else 0.0
        return {
            "product_id": self.meta.product_id,
            "mean_raw": mean,
            "valid_fraction": float(mask.mean()),
            "incidence_deg": self.incidence_deg,
            "reflectance_units": self.meta.dtype.lstrip("<>").startswith("f"),
            "lit": bool(kept.size) and mean > DARK_REFLECTANCE,
        }


def _band_information(meta: SceneMeta) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Per-band names and filter centres, where the label states them.

    MI names its nine filters (MV1..MV5, MN1..MN4) and gives a centre for each. The
    single-band references state neither, and empty tuples say so rather than
    inventing 'band 1'.
    """
    if not is_pds3(meta.label_path):
        return (), ()
    from chandralign.io.pds_label import _pvl_label, _pvl_numbers, _pvl_value, _pds3_find

    label = _pvl_label(meta.label_path)
    image = _pds3_find(label, "IMAGE") or {}
    # MI states FILTER_NAME at the label's top level, not inside the IMAGE object,
    # so both places are checked before concluding there is no band information.
    source = image if "FILTER_NAME" in image or "CENTER_FILTER_WAVELENGTH" in image else label
    names = _pvl_value(source.get("FILTER_NAME")) or _pvl_value(source.get("BAND_NAME"))
    if names is None:
        names = ()
    elif isinstance(names, (list, tuple)):
        names = tuple(str(_pvl_value(n)).strip() for n in names)
    else:
        names = (str(names).strip(),)
    centres = tuple(_pvl_numbers(source, "CENTER_FILTER_WAVELENGTH"))
    return names, centres


def load_reference(label_path: str | Path, catalogue: Optional[dict] = None) -> Reference:
    """Load any LRO or SELENE reference product from its label.

    The footprint is not fatal: a WAC product with no saved ODE record still loads
    and reports why it has no outline, because it can still be read and inspected.
    """
    meta = parse_label(label_path)
    instrument = str(meta.instrument)
    if instrument not in REFERENCE_INSTRUMENTS:
        raise NotAReferenceError(
            f"{meta.product_id} is {instrument}, not one of {REFERENCE_INSTRUMENTS}")

    record = catalogue if catalogue is not None else find_saved_record(meta.raster_path.parent)
    footprint = None
    error = None
    try:
        footprint = footprint_of(meta, record)
    except FootprintError as exc:
        error = str(exc)

    names, centres = _band_information(meta)
    return Reference(meta=meta, special=read_special_values(meta.label_path),
                     footprint=footprint, footprint_error=error,
                     band_names=names, band_wavelengths_nm=centres, catalogue=record)


def find_references(root: str | Path) -> list[Path]:
    """Every reference label under a directory, browse and pyramid sidecars excluded."""
    root = Path(root)
    skip = ("_PYR.", "_BROWSE.", "_THUMB.")
    found = []
    for pattern in ("lro/nac/*/*.XML", "lro/wac/*/*.XML", "selene/tc/*.lbl", "selene/mi/*.lbl"):
        found.extend(p for p in sorted(root.glob(pattern))
                     if not any(m in p.name.upper() for m in skip))
    return found


def load_all(root: str | Path) -> list[Reference]:
    """Every reference we hold, in one list. Anything unreadable is left out loudly."""
    out = []
    for path in find_references(root):
        out.append(load_reference(path))
    return out
