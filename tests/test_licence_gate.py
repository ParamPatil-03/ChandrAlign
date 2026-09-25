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
# Audit I-06: an ALLOWLIST, not a denylist
# ---------------------------------------------------------------------------
def test_every_vismatch_model_off_the_allowlist_is_refused_in_ship_mode():
    import pytest as _pt
    vismatch = _pt.importorskip("vismatch")
    from chandralign.matching import licence
    names = vismatch.available_models
    names = names() if callable(names) else names
    allowed = {n.lower() for n in licence.allowlist()}
    for n in names:
        if n.lower() in allowed:
            licence.assert_allowed(n, ship_mode=True)            # audited: must pass
        else:
            with _pt.raises(licence.LicenceRestrictedError):
                licence.assert_allowed(n, ship_mode=True)


def test_the_four_models_the_old_denylist_passed_are_refused_with_their_reason():
    from chandralign.matching import licence
    for n, word in (("master", "NC-SA"), ("duster", "NC-SA"), ("gim-lightglue", "superpoint"), ("omniglue", "SuperPoint")):
        assert word.lower() in (licence.restriction_reason(n) or "").lower(), n
    assert licence.restriction_reason("eloftr") is None and licence.restriction_reason("rift2") is None


def test_benchmark_mode_is_explicit_and_recorded(monkeypatch):
    """G-03: unaudited models run only with an explicit switch, and every report then says ship_mode false."""
    import pytest as _pt
    from chandralign import config
    from chandralign.evaluate.run_record import run_record
    from chandralign.matching import licence
    with _pt.raises(licence.LicenceRestrictedError):
        licence.assert_allowed("minima-roma")
    monkeypatch.setitem(config.load("default"), "ship_mode", True)
    with _pt.warns(UserWarning):
        licence.enable_benchmark_mode()
    licence.assert_allowed("minima-roma")
    assert run_record()["ship_mode"] is False


def test_dense_models_get_the_larger_sample_budget():
    from chandralign.matching import adapter
    assert adapter.is_dense("minima-roma") and adapter.is_dense("gim-dkm") and adapter.is_dense("ufm")
    assert not adapter.is_dense("eloftr") and not adapter.is_dense("minima-loftr")
