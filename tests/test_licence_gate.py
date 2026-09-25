"""CHECK-10: the licence gate, for models and for packages (scripts/check_licences.py).

done_when: superpoint-lightglue is blocked when ship_mode is true and available
when false; adding a GPL package fails CI. Both halves are proved here, the
second by planting a GPL distribution and a restricted model and requiring the
check to fail -- a check that cannot fail proves nothing.
"""
import sys
from pathlib import Path

import pytest

from chandralign import config
from chandralign.matching import licence

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def check():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_licences
        yield check_licences
    finally:
        sys.path.remove(str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
def test_superpoint_is_blocked_in_ship_mode():
    with pytest.raises(licence.LicenceRestrictedError):
        licence.assert_allowed("superpoint-lightglue", ship_mode=True)


def test_superpoint_is_available_for_benchmarking_outside_ship_mode():
    licence.assert_allowed("superpoint-lightglue", ship_mode=False)     # no raise


def test_the_routed_defaults_are_clean():
    cfg = config.load("regimes")
    for name in (cfg["default_matcher"], cfg["cross_modal_matcher"]):
        licence.assert_allowed(name, ship_mode=True)


# ---------------------------------------------------------------------------
# package licences
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, verdict", [
    ("MIT", "ok"),
    ("BSD-3-Clause", "ok"),
    ("Apache Software License", "ok"),
    ("GPL-3.0-only", "forbidden"),
    ("AGPL-3.0-or-later", "forbidden"),
    ("GNU General Public License v3 (GPLv3)", "forbidden"),
    ("GNU Affero General Public License v3", "forbidden"),
    ("LGPL-2.1-or-later", "weak-copyleft"),
    ("GNU Lesser General Public License v2 or later (LGPLv2+)", "weak-copyleft"),
    ("MIT OR GPL-3.0-only", "ok"),          # the licensee may choose MIT
    ("", "unknown"),
    ("UNKNOWN", "unknown"),
])
def test_classify(check, text, verdict):
    assert check.classify(text) == verdict


class _FakeDist:
    def __init__(self, name, licence_text):
        self.version = "9.9"
        self.metadata = _Meta({"Name": name, "License-Expression": licence_text})
        self.requires = []


class _Meta(dict):
    def get_all(self, key):
        return None


def test_a_gpl_dependency_fails_the_check(check, monkeypatch, capsys):
    real = check.closure

    def with_gpl(roots):
        found = real(roots)
        found["evil-gpl-lib"] = _FakeDist("evil-gpl-lib", "GPL-3.0-only")
        return found

    monkeypatch.setattr(check, "closure", with_gpl)
    monkeypatch.setattr(sys, "argv", ["check_licences.py"])
    assert check.main() == 1
    assert "evil-gpl-lib" in capsys.readouterr().out


def test_a_restricted_model_in_the_config_fails_the_check(check, monkeypatch):
    real_load = config.load

    def load(name):
        cfg = dict(real_load(name))
        if name == "regimes":
            cfg["shippable_matchers"] = list(cfg.get("shippable_matchers") or []) + ["superpoint-lightglue"]
        return cfg

    monkeypatch.setattr(config, "load", load)
    leaked, _ = check.check_models()
    assert leaked == ["superpoint-lightglue"]


def test_the_real_install_passes(check, monkeypatch):
    """The shipped dependency set, as installed here, is clean. (CI runs the
    script itself; this keeps a local run honest too.)"""
    monkeypatch.setattr(sys, "argv", ["check_licences.py"])
    assert check.main() == 0


# ---------------------------------------------------------------------------
# Audit 2026-09-26 I-06: the gate was a 3-component denylist; four non-redistributable
# vismatch models passed it in ship mode, and so would any model nobody had audited.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["master", "duster", "gim-lightglue", "omniglue"])
def test_the_models_the_denylist_missed_are_blocked(name):
    from chandralign.matching.licence import LicenceRestrictedError, assert_allowed, is_restricted
    assert is_restricted(name)
    with pytest.raises(LicenceRestrictedError):
        assert_allowed(name, ship_mode=True)


@pytest.mark.parametrize("name", ["roma", "minima-roma", "ufm", "romav2", "tiny-roma"])
def test_a_model_nobody_audited_is_refused_in_ship_mode(name):
    from chandralign.matching.licence import LicenceRestrictedError, assert_allowed
    with pytest.raises(LicenceRestrictedError, match="allowlist"):
        assert_allowed(name, ship_mode=True)
    assert_allowed(name, ship_mode=False)          # benchmarking stays possible


def test_every_shippable_model_is_allowed():
    from chandralign.matching.licence import assert_allowed, shippable
    assert shippable()
    for name in shippable():
        assert_allowed(name, ship_mode=True)


def test_every_installed_vismatch_model_off_the_allowlist_is_refused():
    vismatch = pytest.importorskip("vismatch")
    from chandralign.matching.licence import LicenceRestrictedError, assert_allowed, shippable
    allowed = set(shippable())
    for name in vismatch.available_models:
        if name in allowed:
            continue
        with pytest.raises(LicenceRestrictedError):
            assert_allowed(name, ship_mode=True)
