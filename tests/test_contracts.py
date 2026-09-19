"""P0-T02 acceptance: every contract dataclass constructs and round-trips through asdict.

Also enforces the freeze: src/chandralign/contracts.py must stay identical to the
code block in PLAN.md section 5, so neither can drift without the other.
"""
import dataclasses
import re
from pathlib import Path

import numpy as np
import pytest

from chandralign.contracts import (
    GeometryLayers,
    ImagePlane,
    MatchSet,
    Metrics,
    RegistrationResult,
    SceneMeta,
    TransformModel,
)

ROOT = Path(__file__).resolve().parents[1]


def make_meta(**overrides) -> SceneMeta:
    values = dict(
        product_id="ch2_ohr_ncp_20240330T0035085365_d_img_d18",
        instrument="OHRC",
        mission="CH2",
        gsd_m=0.25,
        n_bands=1,
        wavelength_nm=(500.0, 800.0),
        array_shape=(79796, 12000),
        dtype="uint8",
        corner_latlon=[(0.372, 23.45), (0.372, 23.59), (-0.444, 23.59), (-0.444, 23.45)],
        sub_solar_azimuth_deg=None,
        solar_incidence_deg=None,
        emission_deg=None,
        phase_deg=None,
        acquisition_utc="2024-03-30T00:35:08.536Z",
        label_path=Path("a.xml"),
        raster_path=Path("a.img"),
        label_fields_verified={"array_shape": True, "dtype": True},
    )
    values.update(overrides)
    return SceneMeta(**values)


def make_matchset(n: int = 5) -> MatchSet:
    rng = np.random.default_rng(0)
    return MatchSet(
        src_pts=rng.random((n, 2)) * 100,
        ref_pts=rng.random((n, 2)) * 100,
        confidence=rng.random(n).astype(np.float32),
        method="sift",
        regime="same_modal_normal",
        stage="OHRC->NAC",
    )


def test_scenemeta_roundtrip_and_frozen():
    meta = make_meta()
    assert SceneMeta(**dataclasses.asdict(meta)) == meta
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.gsd_m = 99.0


def test_scenemeta_verified_defaults_empty():
    meta = make_meta(label_fields_verified={})
    assert meta.label_fields_verified == {}
    # default_factory, not a shared dict between instances
    del_kwargs = {k: v for k, v in dataclasses.asdict(make_meta()).items() if k != "label_fields_verified"}
    a, b = SceneMeta(**del_kwargs), SceneMeta(**del_kwargs)
    assert a.label_fields_verified is not b.label_fields_verified


def test_imageplane_with_geometry_roundtrip():
    tile = np.zeros((8, 8), dtype=np.float32)
    plane = ImagePlane(
        array=tile,
        valid_mask=np.ones_like(tile, dtype=bool),
        shadow_mask=np.zeros_like(tile, dtype=bool),
        gsd_m=0.25,
        meta=make_meta(),
        geo=GeometryLayers(slope_deg=np.full((8, 8), 3.0), source="dem"),
        tile_origin=(1024, 2048),
        preprocess_chain=["clahe"],
    )
    d = dataclasses.asdict(plane)
    assert d["meta"]["product_id"] == plane.meta.product_id
    assert d["geo"]["source"] == "dem"
    assert np.array_equal(d["array"], tile)
    assert d["tile_origin"] == (1024, 2048)
    # the handoff masks must be distinct objects (failure mode #19, shared-mask bug)
    assert plane.valid_mask is not plane.shadow_mask


def test_registration_result_roundtrip():
    matches = make_matchset()
    result = RegistrationResult(
        matches=matches,
        inlier_mask=np.array([True, True, False, True, False]),
        model=TransformModel(kind="homography", matrix=np.eye(3), scale_expected=2.0),
        metrics=Metrics(inlier_count=3, inlier_ratio=0.6),
        confidence_tier="LOW",
        gates={"null_constant_grey": True},
        failure_modes=[12],
    )
    d = dataclasses.asdict(result)
    assert d["metrics"]["inlier_count"] == 3
    assert d["model"]["kind"] == "homography"
    assert np.array_equal(d["matches"]["src_pts"], matches.src_pts)
    assert d["gates"] == {"null_constant_grey": True}


def test_metrics_default_to_none():
    """Rule H1: an unmeasured metric is None, never a plausible-looking number."""
    m = Metrics()
    for f in dataclasses.fields(Metrics):
        if f.name != "source":
            assert getattr(m, f.name) is None, f.name
    assert m.source == "measured"


def _code_after_imports(text: str) -> str:
    start = text.index("Instrument =")
    return text[start:].strip()


def test_contracts_match_plan_section_5():
    plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
    block = re.search(r"```python\n# src/chandralign/contracts\.py\n(.*?)```", plan, re.S)
    assert block, "PLAN.md section 5 contract block not found"
    source = (ROOT / "src" / "chandralign" / "contracts.py").read_text(encoding="utf-8")
    assert _code_after_imports(source) == _code_after_imports(block.group(1)), (
        "contracts.py and PLAN.md section 5 have drifted. The contract is frozen: "
        "change both together, with all three members agreeing in the PR."
    )
