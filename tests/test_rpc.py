"""geometry/rpc.py: the terrain model as a first-order RPC00B is exact (no GDAL needed)."""
import numpy as np

from chandralign.geometry import rpc


def test_first_order_rpc_reproduces_the_ground_height_model():
    A = np.array([[0.98, -0.05, 12.3], [0.04, 1.01, -7.9], [0, 0, 1.0]])
    G = np.array([[1 / 4096, 0, 23.1], [0, -1 / 4096, -0.2], [0, 0, 1.0]])      # ref px -> lon, lat
    p, h0 = np.array([0.001, -0.066]), -1500.0
    R = rpc.from_parallax(A, p, h0, G, (1000, 900), (23.1, 23.35, -0.45, -0.2), (-1900.0, -1100.0))
    rng = np.random.default_rng(0)
    r = rng.uniform([0, 0], [900, 1000], (500, 2))
    h = rng.uniform(-1900, -1100, 500)
    lon, lat = (G @ np.c_[r, np.ones(500)].T)[:2]
    s = (np.linalg.inv(A) @ np.c_[r - (h - h0)[:, None] * p, np.ones(500)].T)[:2].T
    samp, line = rpc.evaluate(R, lon, lat, h)
    assert np.abs(np.c_[samp, line] - s).max() < 1e-6
    assert len(R["LINE_NUM_COEFF"]) == len(rpc.TERMS) == 20 and R["LINE_DEN_COEFF"][0] == 1.0
