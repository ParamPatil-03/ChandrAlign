"""Scene sun/view geometry and the illumination difference between two scenes.
Owner: Member A (Part 1). Features: GEO-01 (GEO-02, per-pixel layers, comes later).

Where each number comes from depends on the product, and every value records it:

    Chandrayaan-2   label: sun azimuth + incidence (isda:), no emission/phase
    LRO NAC CDR     no geometry in the label; incidence/emission/phase from ODE's
                    catalogue, no sun azimuth
    SELENE TC map   a mosaic of many passes: no single sun position exists

    ill = scene_illumination(meta)                  # finds a saved ODE record if there is one
    ill.incidence_deg, ill.sources["incidence_deg"] # 82.73, "label"
    illumination_delta(ohrc_meta, nac_meta)         # {"d_incidence_deg": 75.02, "d_azimuth_deg": None, ...}

Unknown stays None. Nothing is filled with a typical value, because the regime
selector (Part 2) routes on these numbers and a made-up angle would silently
send a pair down the wrong path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from chandralign.contracts import SceneMeta
from chandralign.io.ode_client import catalogue_number, find_saved_record

ANGLES = ("incidence_deg", "emission_deg", "phase_deg", "sub_solar_azimuth_deg")
# Label and catalogue values further apart than this are reported, not averaged.
DISAGREEMENT_TOLERANCE_DEG = 1.0


@dataclass(frozen=True)
class SceneIllumination:
    """Scene-level sun/view geometry, with the origin of every value."""
    product_id: str
    incidence_deg: Optional[float]           # sun angle from the local vertical (0 = overhead)
    emission_deg: Optional[float]            # camera angle from the local vertical
    phase_deg: Optional[float]               # sun-ground-camera angle
    sub_solar_azimuth_deg: Optional[float]   # direction of the sun, degrees
    sources: dict[str, Optional[str]] = field(default_factory=dict)   # "label" | "ode_catalogue" | None
    notes: tuple[str, ...] = ()

    @property
    def sun_elevation_deg(self) -> Optional[float]:
        return None if self.incidence_deg is None else 90.0 - self.incidence_deg

    @property
    def known(self) -> bool:
        return self.incidence_deg is not None


def scene_illumination(meta: SceneMeta, catalogue: Optional[dict] = None,
                       use_saved_catalogue: bool = True) -> SceneIllumination:
    """Sun/view geometry for one scene.

    Label values always win. A catalogue record (an ODE record as saved by
    io/ode_client) only fills fields the label lacks; where both exist and differ by
    more than DISAGREEMENT_TOLERANCE_DEG, the disagreement is recorded in `notes`.
    With use_saved_catalogue, a record saved next to the product is picked up.
    """
    if catalogue is None and use_saved_catalogue:
        catalogue = find_saved_record(meta.raster_path.parent)
    record = None
    if catalogue is not None:
        pdsid = str(catalogue.get("pdsid", ""))
        if pdsid.lower().split(".")[-1] != meta.product_id.lower():
            raise ValueError(f"catalogue record {pdsid!r} does not belong to {meta.product_id!r}")
        record = catalogue["record"]

    verified = meta.label_fields_verified
    label_values = {
        "incidence_deg": meta.solar_incidence_deg if verified.get("solar_incidence_deg") else None,
        "emission_deg": meta.emission_deg if verified.get("emission_deg") else None,
        "phase_deg": meta.phase_deg if verified.get("phase_deg") else None,
        "sub_solar_azimuth_deg": meta.sub_solar_azimuth_deg if verified.get("sub_solar_azimuth_deg") else None,
    }
    catalogue_values = {
        "incidence_deg": catalogue_number(record, "Incidence_angle") if record else None,
        "emission_deg": catalogue_number(record, "Emission_angle") if record else None,
        "phase_deg": catalogue_number(record, "Phase_angle") if record else None,
        "sub_solar_azimuth_deg": None,   # ODE's product record does not carry it
    }

    values, sources, notes = {}, {}, []
    for angle in ANGLES:
        lab, cat = label_values[angle], catalogue_values[angle]
        if lab is not None:
            values[angle], sources[angle] = lab, "label"
            if cat is not None and abs(lab - cat) > DISAGREEMENT_TOLERANCE_DEG:
                notes.append(f"{angle}: label {lab:.2f} vs catalogue {cat:.2f}")
        elif cat is not None:
            values[angle], sources[angle] = cat, "ode_catalogue"
        else:
            values[angle], sources[angle] = None, None

    if values["incidence_deg"] is None:
        if meta.acquisition_utc is None:
            notes.append("no acquisition time: a mosaic of several passes has no single sun position")
        else:
            notes.append("no sun geometry in the label and no catalogue record")

    return SceneIllumination(product_id=meta.product_id, sources=sources, notes=tuple(notes), **values)


def azimuth_difference(a: float, b: float) -> float:
    """Smallest angle between two directions, 0..180 degrees (350 vs 10 -> 20)."""
    d = abs(a - b) % 360.0
    return 360.0 - d if d > 180.0 else d


Scene = Union[SceneMeta, SceneIllumination]


def illumination_delta(src: Scene, ref: Scene) -> dict:
    """How different the lighting is between two scenes (PLAN.md P1-T07 step 3).

    Each difference is None when either side lacks that angle -- the caller sees
    the gap instead of a zero. `max_incidence_deg` is the lower of the two suns
    (largest incidence), the other quantity the regime selector needs.
    """
    a = src if isinstance(src, SceneIllumination) else scene_illumination(src)
    b = ref if isinstance(ref, SceneIllumination) else scene_illumination(ref)

    def diff(x, y):
        return None if x is None or y is None else abs(x - y)

    az = (None if a.sub_solar_azimuth_deg is None or b.sub_solar_azimuth_deg is None
          else azimuth_difference(a.sub_solar_azimuth_deg, b.sub_solar_azimuth_deg))
    incidences = [i for i in (a.incidence_deg, b.incidence_deg) if i is not None]
    return {
        "src": a.product_id,
        "ref": b.product_id,
        "d_incidence_deg": diff(a.incidence_deg, b.incidence_deg),
        "d_azimuth_deg": az,
        "d_phase_deg": diff(a.phase_deg, b.phase_deg),
        "max_incidence_deg": max(incidences) if len(incidences) == 2 else None,
        "sources": {"src": dict(a.sources), "ref": dict(b.sources)},
        "complete": all(v is not None for v in (a.incidence_deg, b.incidence_deg,
                                                a.sub_solar_azimuth_deg, b.sub_solar_azimuth_deg)),
    }
