"""DATA-10 acceptance: all 7 cameras resolve from product IDs; scale_gap('OHRC','IIRS') ~= 320."""
import json
from pathlib import Path

import pytest

from chandralign.io.instruments import (
    UnknownInstrumentError,
    detect_instrument,
    get_spec,
    scale_gap,
    load_registry,
)

ROOT = Path(__file__).resolve().parents[1]

# Real product names from our downloads (see data/manifest.json).
REAL_IDS = {
    "ch2_ohr_ncp_20240330T0035085365_d_img_d18.zip": "OHRC",
    "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml": "OHRC",
    "ch2_tmc_nca_20250207T1102039417_d_img_d18.img": "TMC2",
    "ch2_iir_nci_20240523T1600301891_d_img_d18.qub": "IIRS",
    "M1417360906LC.IMG": "NAC",
    "M102000149RC.XML": "NAC",
    "nac.m1417360906lc": "NAC",          # ODE's pdsid form; the dot is not an extension
    "TCO_MAP_02_N03E021N00E024SC.lbl": "TC",
}

# Pattern examples not yet checked against a downloaded product.
UNVERIFIED_IDS = {
    "wac.m119962003ce": "WAC",
    "MI_MAP_03_N00E021S03E024SC": "MI",
}


def test_registry_has_all_seven_cameras():
    assert set(load_registry()) == {"OHRC", "TMC2", "IIRS", "NAC", "WAC", "TC", "MI"}


@pytest.mark.parametrize("product, expected", REAL_IDS.items())
def test_detects_real_products(product, expected):
    assert detect_instrument(product) == expected


@pytest.mark.parametrize("product, expected", UNVERIFIED_IDS.items())
def test_detects_pattern_examples(product, expected):
    assert detect_instrument(product) == expected


def test_accepts_full_paths():
    p = Path("data/raw/ch2/ohrc/products/x/ch2_ohr_ncp_20240330T0035085365_d_img_d18.img")
    assert detect_instrument(p) == "OHRC"


@pytest.mark.parametrize("product", ["holiday.png", "ch2_xyz_foo.img", "", "M12"])
def test_unknown_products_raise_instead_of_guessing(product):
    with pytest.raises(UnknownInstrumentError):
        detect_instrument(product)


def test_every_image_product_in_manifest_is_recognised():
    """Every image product we actually hold resolves to the camera its folder says."""
    folder_to_camera = {"ch2/ohrc": "OHRC", "ch2/tmc2": "TMC2", "ch2/iirs": "IIRS",
                        "lro/nac": "NAC", "selene/tc": "TC"}
    files = json.loads((ROOT / "data" / "manifest.json").read_text(encoding="utf-8"))["files"]
    checked = dem_rasters = 0
    for rel in files:
        name = rel.rsplit("/", 1)[-1]
        is_image = ("_d_img_" in name and name.endswith((".img", ".qub"))) or \
                   name.upper().endswith(".IMG")
        if not is_image:
            continue
        camera_folder = next((cam for folder, cam in folder_to_camera.items() if rel.startswith(folder)), None)
        if camera_folder is None:
            # Elevation maps are .img rasters too, but they come from no camera:
            # detection must refuse them rather than guess one.
            assert rel.startswith("dem/"), f"unexpected raster outside the camera and DEM folders: {rel}"
            with pytest.raises(UnknownInstrumentError):
                detect_instrument(name)
            dem_rasters += 1
            continue
        assert detect_instrument(name) == camera_folder, rel
        checked += 1
    assert checked >= 7, f"expected >=7 image products in the manifest, found {checked}"
    assert dem_rasters >= 2, f"expected the LOLA/SLDEM tiles in the manifest, found {dem_rasters}"


@pytest.mark.parametrize("a, b, expected", [
    ("OHRC", "IIRS", 320.0),   # PLAN.md 1.2: known-hard pair
    ("OHRC", "NAC", 2.0),
    ("IIRS", "WAC", 1.25),
    ("TMC2", "TC", 2.0),       # the open gap
    ("TMC2", "NAC", 10.0),
    ("IIRS", "NAC", 160.0),
    ("TMC2", "TMC2", 1.0),
])
def test_scale_gap_matches_plan(a, b, expected):
    assert scale_gap(a, b) == pytest.approx(expected)
    assert scale_gap(b, a) == pytest.approx(expected)


def test_scale_gap_rejects_unknown_camera():
    with pytest.raises(UnknownInstrumentError):
        scale_gap("OHRC", "HUBBLE")


def test_spec_values_come_from_plan():
    ohrc = get_spec("OHRC")
    assert ohrc.gsd_m == 0.25 and ohrc.wavelength_nm == (500.0, 800.0) and ohrc.mission == "CH2"
    assert get_spec("IIRS").n_bands == 256
    assert get_spec("TMC2").gsd_range_m == (4.41, 5.0)   # 4.41 on OUR label; PLAN quotes 4.43 from another product
    assert get_spec("MI").n_bands is None     # not stated in the plan -> read from label


def test_config_missing_a_camera_is_rejected(tmp_path):
    bad = tmp_path / "instruments.yaml"
    bad.write_text("OHRC:\n  mission: CH2\n  role: source\n  kind: pan\n  gsd_m: 0.25\n"
                   "  id_patterns: ['^ch2_ohr_']\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must define exactly"):
        load_registry(bad)
