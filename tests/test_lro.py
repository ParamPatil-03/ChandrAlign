"""DATA-06: LRO NAC reference products read into the same SceneMeta as Chandrayaan-2.

NASA's calibrated NAC products carry NO geometry in either label -- no sun angles,
corners or resolution. These tests pin that down, so nobody later "fills" those
fields from a search catalogue and marks them as label facts.
"""
from pathlib import Path

import numpy as np
import pytest

from chandralign.io.pds_label import PdsParseError, parse_label, parse_pds3, parse_pds4, read_array_layout
from chandralign.io.pds_raster import Window, check_attached_pds3_header, read_raster, verify_raster

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "labels"
NAC_LOW_SUN = FIXTURES / "M102000149RC.XML"      # 2009, ODE lists incidence 79.8 deg
NAC_HIGH_SUN = FIXTURES / "M1417360906LC.XML"    # 2022, ODE lists incidence 7.7 deg
ATTACHED = FIXTURES / "M1417360906LC_attached_header.lbl"


@pytest.mark.parametrize("label, product, utc", [
    (NAC_HIGH_SUN, "M1417360906LC", "2022-09-09T11:07:18.714000Z"),
    (NAC_LOW_SUN, "M102000149RC", "2009-07-12T01:08:02.537000Z"),
])
def test_real_nac_label_fields(label, product, utc):
    meta = parse_pds4(label)
    assert meta.product_id == product
    assert meta.instrument == "NAC" and meta.mission == "LRO"
    assert meta.array_shape == (52224, 5064)
    assert meta.dtype == "<i2"                        # SignedLSB2
    assert meta.n_bands == 1
    assert meta.acquisition_utc == utc
    assert meta.wavelength_nm == (450.0, 750.0)       # centre 600 nm, bandwidth 300 nm
    assert meta.raster_path.name == f"{product}.IMG"


def test_nac_verification_is_honest():
    meta = parse_pds4(NAC_HIGH_SUN)
    v = meta.label_fields_verified
    assert sum(v.values()) >= 6
    for f in ("product_id", "instrument", "mission", "array_shape", "dtype", "acquisition_utc", "wavelength_nm"):
        assert v[f] is True, f
    # Absent from NASA's CDR labels. Values come from elsewhere (registry / later SPICE or ODE),
    # so they must be reported as NOT read from the label.
    for f in ("gsd_m", "corner_latlon", "sub_solar_azimuth_deg", "solar_incidence_deg", "emission_deg", "phase_deg"):
        assert v[f] is False, f
    assert meta.corner_latlon == []
    assert meta.solar_incidence_deg is None
    assert meta.gsd_m == 0.5                          # registry nominal, unverified


def test_nac_and_ch2_share_one_contract():
    """Source and reference arrive in the same SceneMeta shape (REQ-01)."""
    nac = parse_label(NAC_HIGH_SUN)
    ohrc = parse_label(FIXTURES / "ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
    assert type(nac) is type(ohrc)
    assert set(nac.label_fields_verified) == set(ohrc.label_fields_verified)


def test_nac_pixel_layout_skips_attached_header():
    layout = read_array_layout(NAC_HIGH_SUN)
    assert layout.offset_bytes == 5064                # one PDS3 header record before the pixels
    assert layout.file_size_bytes == 528_929_736 == 5064 + 52224 * 5064 * 2
    assert layout.md5 == "03baccfcccfe047c68af5dc0bc693e97"


def test_attached_pds3_header_agrees_with_pds4():
    """The PDS3 header inside the .IMG, parsed on its own, describes the same pixels."""
    pds3 = parse_pds3(ATTACHED)
    pds4 = parse_pds4(NAC_HIGH_SUN)
    assert pds3.instrument == "NAC"                   # INSTRUMENT_ID = LROC is accepted for NAC
    assert pds3.product_id == pds4.product_id
    assert pds3.array_shape == pds4.array_shape and pds3.dtype == pds4.dtype
    assert pds3.acquisition_utc == "2022-09-09T11:07:18.714000Z" == pds4.acquisition_utc
    assert pds3.wavelength_nm == pds4.wavelength_nm
    assert read_array_layout(ATTACHED).offset_bytes == 5064   # ^IMAGE = 2, RECORD_BYTES = 5064


def test_family_instrument_id_does_not_admit_other_cameras(tmp_path):
    """LROC may stand for NAC or WAC -- not for a SELENE camera."""
    raw = ATTACHED.read_bytes().replace(b"PRODUCT_ID                         = M1417360906LC",
                                        b'PRODUCT_ID                         = "TCO_FAKE"')
    fake = tmp_path / "fake.lbl"
    fake.write_bytes(raw)
    with pytest.raises(PdsParseError, match="INSTRUMENT_ID"):
        parse_pds3(fake)


# ----------------------------------------------------------------------------- real products

REAL = {p.stem: p for p in sorted((ROOT / "data" / "raw" / "lro" / "nac").glob("*/M*[CE].XML"))}
needs_real = pytest.mark.skipif(len(REAL) < 2, reason="NAC products not downloaded (data/raw is gitignored)")


@needs_real
@pytest.mark.parametrize("product", ["M1417360906LC", "M102000149RC"])
def test_real_nac_window_matches_direct_bytes(product):
    label = REAL[product]
    layout = read_array_layout(label)
    w = Window(row=30000, col=2000, height=48, width=48)
    tile = read_raster(label, w)
    samples = layout.shape[1]
    with layout.raster_path.open("rb") as fh:
        for r, c in [(0, 0), (47, 47), (20, 9)]:
            fh.seek(layout.offset_bytes + ((w.row + r) * samples + w.col + c) * 2)
            assert tile[r, c] == np.frombuffer(fh.read(2), "<i2")[0]
    assert verify_raster(label)["size_ok"]
    assert check_attached_pds3_header(label)["layout_agrees"]


@needs_real
@pytest.mark.slow
@pytest.mark.parametrize("product", ["M1417360906LC", "M102000149RC"])
def test_real_nac_both_checksums(product):
    """NASA's whole-file MD5 (PDS4) and the pixels-only MD5 (attached PDS3 header) both match."""
    assert verify_raster(REAL[product], check_md5=True)["md5_checked"]
    assert check_attached_pds3_header(REAL[product], check_md5=True)["pixel_md5_checked"]
