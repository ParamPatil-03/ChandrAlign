"""MATCH-11: the best-known geolocation prior, from committed registrations."""
import numpy as np

from chandralign.geometry import geoprior


def test_fit_is_linear_within_the_measured_lines_and_flat_beyond():
    g = geoprior.GeoPrior("x", np.array([0.0, 1000, 2000]), np.array([100.0, 300, 500]),
                          np.array([10.0, 10, 10]), "test")
    e, n = g.offset_at([1000, 5000, -50])
    assert np.allclose(e, [300, 500, 100]) and np.allclose(n, 10)


def test_an_unmeasured_product_gets_a_labelled_zero():
    g = geoprior.load("ch2_ohr_ncp_29990101T0000000000_d_img_d18")
    assert g.source == "system" and np.allclose(g.offset_at(123), 0)


def test_held_products_reproduce_the_measured_system_errors():
    tmc = geoprior.load("ch2_tmc_nca_20250207T1102039417_d_img_d18")
    iirs = geoprior.load("ch2_iir_nci_20240523T1600301891_d_img_d18")
    ohrc = geoprior.load("ch2_ohr_ncp_20240330T0035085365_d_img_d18")
    assert tmc.source.startswith("SELENE TC") and -5000 < float(tmc.offset_at(16000)[1]) < -4400
    assert iirs.source.startswith("LRO WAC") and 12700 < float(iirs.offset_at(3000)[1]) < 12900
    assert ohrc.source.startswith("LRO NAC") and 1500 < float(ohrc.offset_at(20000)[1]) < 2600


import pytest  # noqa: E402

_TMC = "ch2_tmc_nca_20250207T1102039417_d_img_d18"
_HAVE_LABELS = geoprior._find_label(_TMC) is not None


@pytest.mark.skipif(not _HAVE_LABELS, reason="CH-2 labels not in data/raw")
def test_a_product_nobody_registered_gets_isros_refined_label_prior():
    """Audit I-17: without a measured table entry, the search starts from ISRO's refined corners
    (~5.1 km north for TMC-2), not from zero -- 88% of the measured 4.7 km correction."""
    g = geoprior.load(_TMC, use_measured=False)
    assert g.source.startswith("ISRO refined corners") and "never evaluation" in g.source
    assert -5400 < float(g.offset_at(16000)[1]) < -4900
    measured = geoprior.load(_TMC)
    resid = np.hypot(*(np.array(g.offset_at(16000)) - np.array(measured.offset_at(16000))))
    assert resid < 1000


@pytest.mark.skipif(not _HAVE_LABELS, reason="CH-2 labels not in data/raw")
def test_a_label_whose_refined_corners_equal_the_system_ones_adds_nothing():
    g = geoprior.load("ch2_ohr_ncp_20240330T0035085365_d_img_d18", use_measured=False)
    assert g.source == "system" and np.allclose(g.offset_at(100), 0)
