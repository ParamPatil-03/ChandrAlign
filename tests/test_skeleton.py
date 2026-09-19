"""P0-T01 acceptance: the package and every planned module import cleanly."""
import importlib
import pkgutil

import chandralign


def test_package_imports():
    assert chandralign.__version__


def test_every_module_imports():
    names = [m.name for m in pkgutil.walk_packages(chandralign.__path__, "chandralign.")]
    # PLAN.md section 4 lays out 9 subpackages and ~45 modules; guard against the tree vanishing.
    assert len(names) >= 50, names
    for name in names:
        importlib.import_module(name)
