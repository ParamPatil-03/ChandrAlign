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

import numpy as np

from chandralign.contracts import GeometryLayers, SceneMeta
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


def azimuth_difference(a, b):
    """Smallest angle between two directions, 0..180 degrees (350 vs 10 -> 20).

    Works on scalars and on arrays; a scalar pair gives a plain float back, so the
    per-pixel layers (GEO-02) and the scene-level delta share one definition.
    """
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    out = np.where(d > 180.0, 360.0 - d, d)
    return float(out) if out.ndim == 0 else out


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


# ----------------------------------------------------------------------------- GEO-02: per-pixel layers

MOON_RADIUS_KM = 1737.4
# Beyond this roll the camera is not looking along the nadir track and the derived
# emission would be quietly wrong, so it is withheld instead.
ROLL_TOLERANCE_DEG = 1.0


def sub_solar_point(lat_deg, lon_deg, incidence_deg, azimuth_deg):
    """Where the sun stands overhead, from one point's incidence and sun azimuth.

    The sun's incidence angle at a point IS the angular distance from that point to
    the sub-solar point, and the sun azimuth is the bearing to it. So one labelled
    (incidence, azimuth) pair fixes the sub-solar point exactly -- and from there the
    incidence at EVERY other pixel follows, with no SPICE kernel and no ephemeris.
    """
    la, lo, d, b = (np.radians(np.asarray(v, float))
                    for v in (lat_deg, lon_deg, incidence_deg, azimuth_deg))
    lat2 = np.arcsin(np.sin(la) * np.cos(d) + np.cos(la) * np.sin(d) * np.cos(b))
    lon2 = lo + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la),
                           np.cos(d) - np.sin(la) * np.sin(lat2))
    return np.degrees(lat2), (np.degrees(lon2) + 180.0) % 360.0 - 180.0


def angular_separation(lat1, lon1, lat2, lon2):
    """Great-circle angle between two points, in degrees. Haversine, for small angles."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    h = (np.sin((la2 - la1) / 2) ** 2
         + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2)
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


def bearing(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, degrees clockwise from north."""
    la1, lo1, la2, lo2 = (np.radians(np.asarray(v, float)) for v in (lat1, lon1, lat2, lon2))
    y = np.sin(lo2 - lo1) * np.cos(la2)
    x = np.cos(la1) * np.sin(la2) - np.sin(la1) * np.cos(la2) * np.cos(lo2 - lo1)
    return np.degrees(np.arctan2(y, x)) % 360.0


def emission_from_altitude(separation_deg, altitude_km: float):
    """Angle from the local vertical to the spacecraft, given how far off nadir the pixel is.

    G is the ground pixel, N the point directly under the spacecraft, S the
    spacecraft at altitude h. With psi the angle G-centre-N, the zenith angle of S
    seen from G is atan2((R+h) sin psi, (R+h) cos psi - R). Exact on a sphere; at
    psi = 0 it gives 0, which is nadir.
    """
    psi = np.radians(np.asarray(separation_deg, float))
    r = MOON_RADIUS_KM + float(altitude_km)
    return np.degrees(np.arctan2(r * np.sin(psi), r * np.cos(psi) - MOON_RADIUS_KM))


def phase_from_angles(incidence_deg, emission_deg, azimuth_difference_deg):
    """Sun-ground-camera angle from the two zenith angles and their azimuth difference."""
    i, e, d = (np.radians(np.asarray(v, float))
               for v in (incidence_deg, emission_deg, azimuth_difference_deg))
    cos_phase = np.cos(i) * np.cos(e) + np.sin(i) * np.sin(e) * np.cos(d)
    return np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))


def geometry_layers(meta: SceneMeta, rows=None, cols=None, step: int = 1,
                    model=None, viewing=None) -> GeometryLayers:
    """Per-pixel incidence, emission and phase for one scene (GEO-02).

    WHY THIS EXISTS. The label states ONE incidence and ONE sun azimuth for a whole
    product. Measured on our own three products, that is fine for OHRC and wrong for
    the other two, because incidence varies with position and these strips are long:

        OHRC     25 km long    incidence varies 0.14 deg across the scene
        TMC-2   812 km long    incidence varies 7.80 deg
        IIRS   1042 km long    incidence varies 8.30 deg

    An 8 degree error in incidence is not a rounding detail to a physics-based
    outlier filter that is asked whether two patches can be the same ground.

    HOW. No Chandrayaan-2 label carries angle backplanes -- only the scene-level
    numbers above -- so the layers are DERIVED, and `source` says so:

        incidence   angular distance from each pixel to the sub-solar point, which
                    the labelled (incidence, azimuth) pair fixes exactly
        emission    from the spacecraft altitude and how far the pixel lies off the
                    nadir track, taken as the centre column of its own row
        phase       from the two zenith angles and the difference of their azimuths

    ASSUMPTIONS, all of them consequences of what the label does not say:
      * the labelled scene angles refer to the scene CENTRE (ISRO does not state
        the reference point; this is the natural reading and is recorded in `notes`)
      * the sun does not move during the scene -- true to well under a degree, since
        the Moon turns 0.55 deg/hour and these scenes last minutes
      * the camera looks along the nadir track. Roll, pitch and yaw are read from the
        label and reported; where roll exceeds ROLL_TOLERANCE_DEG the emission layer
        is withheld rather than quietly biased.

    Returns layers on the sampled grid (`step` > 1 subsamples, which is how a
    whole 80,000-line strip is summarised cheaply). Anything unknown stays None.
    """
    from chandralign.geometry.projection import GeolocationUnavailable, geolocation_model
    from chandralign.io.pds_label import read_viewing_geometry

    ill = scene_illumination(meta)
    notes: list[str] = []
    if ill.incidence_deg is None or ill.sub_solar_azimuth_deg is None:
        return GeometryLayers(source="unknown")

    if model is None:
        try:
            model = geolocation_model(meta)
        except GeolocationUnavailable as exc:
            return GeometryLayers(source="unknown")
    if viewing is None:
        viewing = read_viewing_geometry(meta.label_path)

    n_rows, n_cols = meta.array_shape
    if rows is None:
        rows = np.arange(0, n_rows, step)
    if cols is None:
        cols = np.arange(0, n_cols, step)
    rows = np.asarray(rows); cols = np.asarray(cols)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    lat, lon = model.pixel_to_latlon(rr, cc)

    # The sub-solar point, from the scene-centre reading of the labelled angles.
    centre_lat, centre_lon = model.pixel_to_latlon(np.array(n_rows / 2.0), np.array(n_cols / 2.0))
    sun_lat, sun_lon = sub_solar_point(centre_lat, centre_lon,
                                       ill.incidence_deg, ill.sub_solar_azimuth_deg)
    notes.append("scene angles taken to refer to the scene centre (the label does not say)")

    incidence = angular_separation(lat, lon, sun_lat, sun_lon)
    sun_azimuth = bearing(lat, lon, sun_lat, sun_lon)

    emission = phase = None
    if viewing.altitude_km is None:
        notes.append("no spacecraft_altitude in the label: emission and phase withheld")
    elif viewing.roll_deg is not None and abs(viewing.roll_deg) > ROLL_TOLERANCE_DEG:
        notes.append(f"roll {viewing.roll_deg:.3f} deg exceeds {ROLL_TOLERANCE_DEG} deg: "
                     "the nadir-track assumption fails, so emission and phase are withheld")
    else:
        # Nadir track: the spacecraft sits above the centre column of each row.
        track_lat, track_lon = model.pixel_to_latlon(rr, np.full_like(cc, n_cols // 2))
        off_nadir = angular_separation(lat, lon, track_lat, track_lon)
        emission = emission_from_altitude(off_nadir, viewing.altitude_km)
        sc_azimuth = bearing(lat, lon, track_lat, track_lon)
        # Directly under the track there is no direction to the spacecraft; the
        # azimuth is meaningless there, but emission is 0 so phase = incidence.
        d_az = np.where(off_nadir > 0, azimuth_difference(sun_azimuth, sc_azimuth), 0.0)
        phase = phase_from_angles(incidence, emission, d_az)

    return GeometryLayers(incidence_deg=incidence, emission_deg=emission, phase_deg=phase,
                          source="derived")
