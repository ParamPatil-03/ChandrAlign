"""Camera registry, auto-detection and GSD ratios. Owner: Member A (Part 1). Features: DATA-10.

    >>> detect_instrument("ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml")
    'OHRC'
    >>> round(scale_gap("OHRC", "IIRS"))
    320

Settings live in configs/instruments.yaml. Nothing here guesses: an unrecognised
product raises UnknownInstrumentError instead of falling back to a default camera.

Two different scale numbers exist, deliberately named apart:
    scale_gap(a, b)             coarser/finer, symmetric, always >= 1 (this module)
    config.gsd_ratio(src, ref)  gsd(src)/gsd(ref), directional -- the scale a correct
                                transform should recover (Part 2's scale check)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, get_args

import yaml

from chandralign.contracts import Instrument

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "instruments.yaml"

# Extensions stripped before matching. Only these: ODE ids such as
# "nac.m1417360906lc" contain a dot that is NOT an extension.
_KNOWN_EXTENSIONS = (".zip", ".xml", ".img", ".lbl", ".qub", ".hdr", ".tif", ".tiff", ".cub")


class UnknownInstrumentError(ValueError):
    """The product ID matches no camera in the registry."""


@dataclass(frozen=True)
class InstrumentSpec:
    name: Instrument
    mission: str
    role: str                                   # "source" | "reference"
    kind: str
    gsd_m: float
    gsd_range_m: Optional[tuple[float, float]]
    n_bands: Optional[int]
    wavelength_nm: Optional[tuple[float, float]]
    swath_km: Optional[float]
    id_patterns: tuple[re.Pattern, ...]


def _tuple_or_none(value):
    return tuple(float(v) for v in value) if value is not None else None


@lru_cache(maxsize=None)
def load_registry(path: Path = DEFAULT_CONFIG) -> dict[str, InstrumentSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    # configs/instruments.yaml nests the cameras under `instruments:` alongside
    # `pairings:`; a bare mapping of cameras is also accepted.
    raw = raw.get("instruments", raw)
    expected = set(get_args(Instrument))
    if set(raw) != expected:
        raise ValueError(
            f"{path} must define exactly {sorted(expected)}; "
            f"missing {sorted(expected - set(raw))}, unexpected {sorted(set(raw) - expected)}"
        )
    registry = {}
    for name, cfg in raw.items():
        if not cfg.get("gsd_m") or cfg["gsd_m"] <= 0:
            raise ValueError(f"{name}: gsd_m must be a positive number")
        if not cfg.get("id_patterns"):
            raise ValueError(f"{name}: at least one id_pattern is required")
        registry[name] = InstrumentSpec(
            name=name,
            mission=cfg["mission"],
            role=cfg["role"],
            kind=cfg["kind"],
            gsd_m=float(cfg["gsd_m"]),
            gsd_range_m=_tuple_or_none(cfg.get("gsd_range_m")),
            n_bands=cfg.get("n_bands"),
            wavelength_nm=_tuple_or_none(cfg.get("wavelength_nm")),
            swath_km=cfg.get("swath_km"),
            id_patterns=tuple(re.compile(p, re.IGNORECASE) for p in cfg["id_patterns"]),
        )
    return registry


def get_spec(instrument: str) -> InstrumentSpec:
    registry = load_registry()
    if instrument not in registry:
        raise UnknownInstrumentError(f"{instrument!r} is not one of {sorted(registry)}")
    return registry[instrument]


def _product_stem(product: str | Path) -> str:
    name = Path(product).name
    lowered = name.lower()
    for ext in _KNOWN_EXTENSIONS:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return name


def detect_instrument(product: str | Path) -> Instrument:
    """Camera that produced `product` (a product ID, file name or path)."""
    stem = _product_stem(product)
    matches = [
        spec.name
        for spec in load_registry().values()
        if any(p.search(stem) for p in spec.id_patterns)
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise UnknownInstrumentError(f"no camera in the registry matches {stem!r}")
    raise UnknownInstrumentError(f"{stem!r} is ambiguous: matches {matches}")


def scale_gap(a: str, b: str) -> float:
    """How many times coarser the coarser camera is than the finer one (always >= 1).

    Symmetric: scale_gap("OHRC", "IIRS") == scale_gap("IIRS", "OHRC") == 320, exactly
    as PLAN.md section 1.2 tabulates the pairings. Uses nominal registry GSDs; the
    regime selector and cascade use this for pre-flight decisions (feature GEO-06),
    before any label has been read. For the DIRECTIONAL scale a transform should
    recover, use config.gsd_ratio(src, ref).
    """
    ga, gb = get_spec(a).gsd_m, get_spec(b).gsd_m
    return max(ga, gb) / min(ga, gb)


# ----------------------------------------------------------------------------- GEO-06: scale pre-check

DEFAULT_CASCADE_RATIO = 4.0      # fallback if configs/regimes.yaml does not say


@dataclass(frozen=True)
class ScaleCheck:
    """Whether two products may be matched directly, decided before any matcher runs."""
    instrument_a: str
    instrument_b: str
    gsd_a_m: float
    gsd_b_m: float
    ratio: float                 # coarser / finer, always >= 1
    threshold: float
    extreme: bool                # True -> must go through the cascade (MATCH-10)
    source: str                  # "label" | "nominal" | "mixed"
    reason: str

    @property
    def route(self) -> str:
        return "cascade" if self.extreme else "direct"


def cascade_threshold(config_name: str = "regimes") -> float:
    """The scale ratio at which routing switches to the cascade, from Member B's config.

    Read rather than hard-coded so that when the ablation tunes the threshold, the
    pre-check and the regime selector cannot drift apart.
    """
    from chandralign import config as _config
    try:
        rules = _config.load(config_name).get("rules", []) or []
    except Exception:
        return DEFAULT_CASCADE_RATIO
    for rule in rules:
        value = (rule.get("when") or {}).get("scale_ratio_gte")
        if value is not None:
            return float(value)
    return DEFAULT_CASCADE_RATIO


def _gsd_of(scene) -> tuple[str, float, str]:
    """(instrument, gsd_m, source) from a SceneMeta, or from a bare instrument name."""
    if isinstance(scene, str):
        return scene, get_spec(scene).gsd_m, "nominal"
    return str(scene.instrument), float(scene.gsd_m), "label"


def scale_precheck(a, b, threshold: Optional[float] = None) -> ScaleCheck:
    """Can these two be matched directly, or must they go through the cascade? (GEO-06)

    Accepts SceneMeta objects or bare instrument names. A SceneMeta carries the GSD
    the LABEL states, which is preferred, because the registry's nominal figures are
    the mission's design values and our real products differ from them substantially:

        OHRC   nominal 0.25   label  0.30 m   (20.0% out)
        TMC-2  nominal 5.00   label  4.41 m   (11.8% out)
        IIRS   nominal 80.0   label 97.15 m   (21.4% out)

    MEASURED HONESTLY: across all 21 instrument pairings in the registry, using the
    label GSD instead of the nominal one changes NO routing decision -- every ratio
    sits far from the threshold. It still matters for two reasons. TMC-2 <-> MI is
    nominally 4.00 against a threshold of 4.00, which is a knife-edge that a 20%
    error could tip either way; with the label GSD it is 4.54 and no longer marginal.
    And the reported ratio is what the cascade uses to plan its steps, so being 20%
    wrong about it matters even when the yes/no answer does not change.

    This runs on metadata alone -- no pixels are read -- so an OHRC <-> IIRS request
    is refused before a matcher is ever constructed.
    """
    threshold = cascade_threshold() if threshold is None else float(threshold)
    name_a, gsd_a, src_a = _gsd_of(a)
    name_b, gsd_b, src_b = _gsd_of(b)
    if gsd_a <= 0 or gsd_b <= 0:
        raise ValueError(f"a GSD must be positive, got {gsd_a} and {gsd_b}")
    ratio = max(gsd_a, gsd_b) / min(gsd_a, gsd_b)
    extreme = ratio >= threshold
    source = src_a if src_a == src_b else "mixed"
    reason = (f"{name_a} {gsd_a:g} m vs {name_b} {gsd_b:g} m is {ratio:.2f}x, "
              + (f"at or past the {threshold:g}x limit: no matcher has documented "
                 "evaluation at this scale gap, so route to the cascade"
                 if extreme else
                 f"within the {threshold:g}x limit: direct matching is allowed"))
    return ScaleCheck(name_a, name_b, gsd_a, gsd_b, ratio, threshold, extreme, source, reason)
