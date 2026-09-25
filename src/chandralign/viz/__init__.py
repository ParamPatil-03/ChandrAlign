"""Figures for the report and demo UI. Owner: Member C (Part 3)."""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=False)
except ImportError:  # pragma: no cover
    pass
