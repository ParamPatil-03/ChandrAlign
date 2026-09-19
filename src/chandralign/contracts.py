"""The integration contract between Part 1, Part 2 and Part 3.

Transcribed from PLAN.md section 5. Frozen after Phase 0: changing anything here
requires all three members to agree, in writing, in the PR description.

ONE PROPOSED DEVIATION FROM PLAN.md section 5, flagged for sign-off:
    Metrics.source accepts "synthetic" in addition to "measured" and "external".
    Reason: until GATE A opens, every number Part 2 produces is measured on
    synthetic imagery. Under the original two-value contract we would tag those
    "measured", which reads as "measured on real Chandrayaan-2 data" -- exactly
    the honesty failure rule H5 and failure mode #19 exist to prevent. One extra
    value now costs nothing; retrofitting it after C's report and B's benchmark
    both depend on it is a migration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

Instrument = Literal["OHRC", "TMC2", "IIRS", "NAC", "WAC", "TC", "MI"]
Tier = Literal["HIGH", "MEDIUM", "LOW", "REJECTED"]
Regime = Literal["same_modal_normal", "same_modal_polar",
                 "cross_modal", "extreme_scale"]

# Where a number came from. "synthetic" means honestly measured, but on
# generated imagery -- never to be presented as a real-data result (rule H5).
MetricSource = Literal["measured", "external", "synthetic"]


@dataclass(frozen=True)
class SceneMeta:
    """Everything the label actually told us. Part 1 -> everyone."""

    product_id: str
    instrument: Instrument
    mission: str                       # "CH2" | "LRO" | "SELENE"
    gsd_m: float
    n_bands: int
    wavelength_nm: tuple[float, float] | None
    array_shape: tuple[int, int]       # (lines, samples)
    dtype: str
    corner_latlon: list[tuple[float, float]]   # 4 corners, may be empty
    sub_solar_azimuth_deg: Optional[float]
    solar_incidence_deg: Optional[float]
    emission_deg: Optional[float]
    phase_deg: Optional[float]
    acquisition_utc: Optional[str]
    label_path: Path
    raster_path: Path
    # ANTI-STUB PROOF: which fields genuinely came from the label
    label_fields_verified: dict[str, bool] = field(default_factory=dict)


@dataclass
class GeometryLayers:
    """Per-pixel physics. Part 1 -> Part 2 (geometry_filter, regime)."""

    incidence_deg: Optional[np.ndarray] = None
    emission_deg: Optional[np.ndarray] = None
    phase_deg: Optional[np.ndarray] = None
    dem_elev_m: Optional[np.ndarray] = None
    slope_deg: Optional[np.ndarray] = None
    aspect_deg: Optional[np.ndarray] = None
    source: str = "unknown"            # "label" | "spice" | "dem" | "derived"


@dataclass
class ImagePlane:
    """THE handoff object: Part 1 -> Part 2. One tile, ready to match."""

    array: np.ndarray                  # float32, 2-D, normalised 0..1
    valid_mask: np.ndarray             # bool
    shadow_mask: np.ndarray            # bool -- True = shadowed, exclude
    gsd_m: float
    meta: SceneMeta
    geo: Optional[GeometryLayers] = None
    tile_origin: tuple[int, int] = (0, 0)   # (row, col) in the full product
    preprocess_chain: list[str] = field(default_factory=list)
    texture_score: Optional[float] = None
    repetitiveness_score: Optional[float] = None


@dataclass
class MatchSet:
    """Part 2 internal -> Part 3 export."""

    src_pts: np.ndarray                # (N,2) float64, SUB-PIXEL, source frame
    ref_pts: np.ndarray                # (N,2) float64, reference frame
    confidence: np.ndarray             # (N,) float32
    method: str                        # "lightglue" | "rift" | "sift" | ...
    regime: Regime
    stage: str                         # cascade stage, e.g. "TMC2->TC"


@dataclass
class TransformModel:
    kind: Literal["affine", "homography", "tps"]
    matrix: Optional[np.ndarray] = None       # 3x3 for affine/homography
    tps_params: Optional[dict] = None
    scale_estimated: Optional[float] = None
    scale_expected: Optional[float] = None    # from known GSD ratio (failure mode #13)


@dataclass
class Metrics:
    """Every field is Optional. Unmeasured is None, NEVER a made-up number (rule H1)."""

    rmse_px: Optional[float] = None
    rmse_m: Optional[float] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    spatial_coverage: Optional[float] = None
    max_delaunay_gap_px: Optional[float] = None
    subpixel_recovery_err_px: Optional[float] = None
    keypoints_src: Optional[int] = None
    keypoints_ref: Optional[int] = None
    runtime_s: Optional[float] = None
    source: MetricSource = "measured"   # rule H5


@dataclass
class RegistrationResult:
    """THE handoff object: Part 2 -> Part 3."""

    matches: MatchSet
    inlier_mask: np.ndarray
    model: Optional[TransformModel]
    metrics: Metrics
    confidence_tier: Tier
    gates: dict[str, bool]             # control gates -- must be non-empty (rule H4)
    failure_modes: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
