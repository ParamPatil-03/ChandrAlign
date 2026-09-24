"""Turn a regime decision into the matcher the pipeline actually runs (MATCH-09 wiring).

WHY THIS MODULE EXISTS
`matching/regime.py` decides a route, a representation and a matcher from scene
metadata -- and until this module, NOTHING outside the tests called it. The only
real registration script named its matcher directly, and `default_matcher` in
configs/regimes.yaml was read only as a fallback no caller ever reached. The
routing logic was, in effect, dead code with a config value attached.

`choose()` is the one place a pipeline asks "which matcher for this pair?". Part
3's CLI and API (UI-01/UI-02, not yet built) are meant to call it too, so the
decision lives here rather than being re-implemented at each entry point.

WHAT IT ROUTES, AND WHAT IT DELIBERATELY DOES NOT
  extreme scale gap   -> route "cascade", NO matcher name. The caller must hand
                         the pair to cascade.plan(), which decides BY FOOTPRINT
                         whether one step suffices or an intermediate product is
                         needed -- TMC-2 -> NAC, for instance, is a single step,
                         because a NAC swath spans ~400-1300 TMC-2 pixels.
                         NOTHING chooses a matcher per cascade step: the step is
                         run by cascade.register_step_dense (MIND template
                         search, the method the feasibility threshold was
                         measured with) unless the caller picks
                         cascade.register_step with a named matcher. Returning a
                         matcher here would invite a caller to match directly
                         across a gap routing has flagged.
  cross-modality      -> `cross_modal_matcher` (xoftr), tiled in the fine stage.
                         Measured: xoftr is the only candidate at 10/10 on
                         cross-modal, and tiled it runs 9 real windows in 80 s.
  everything else     -> `default_matcher` (eloftr).

It does NOT send "hard illumination" to a different matcher. The illumination
bands are validated on real data for incidence only, not for sun azimuth (see
docs/part2_evidence_summary.md); routing on them would act on an unvalidated
claim. A pair the selector marks `unsolved` or `unknown` still gets the default
matcher, and carries that expectation forward for the quality gate and report.

TILING IS A FINE-STAGE OPTION. `tile_px` is only valid on a PRE-ALIGNED pair of
equal size; `pair_tiling.match_tiled` raises otherwise. Callers pass
`fine_stage_options` after the coarse lock, never on raw scenes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .. import config
from ..contracts import SceneMeta
from . import licence, regime


@dataclass(frozen=True)
class MatcherChoice:
    """What the pipeline should run for one pair, and why."""

    route: str                          # "direct" | "cascade"
    model_name: Optional[str]           # None exactly when route == "cascade"
    fine_stage_options: dict[str, Any] = field(default_factory=dict)
    regime: str = ""
    expectation: str = ""               # passed through from regime.select, never upgraded
    reason: str = ""
    fallbacks: tuple[str, ...] = ()     # tried in order ONLY if model_name's result is rejected

    def candidates(self) -> list[str]:
        """The matchers to try, in order: the routed one, then its fallbacks."""
        return [m for m in (self.model_name, *self.fallbacks) if m]

    def as_provenance(self) -> dict[str, Any]:
        return {"route": self.route, "matcher": self.model_name,
                "fine_stage_options": dict(self.fine_stage_options),
                "regime": self.regime, "expectation": self.expectation,
                "reason": self.reason, "fallbacks": list(self.fallbacks),
                "chosen_by": "matching.routing.choose"}


def choose(src: SceneMeta | str, ref: SceneMeta | str) -> MatcherChoice:
    """The matcher for this pair, decided by regime.select and the routing config."""
    decision = regime.select(src, ref)
    cfg = config.load("regimes")

    if decision.route == "cascade":
        return MatcherChoice(
            route="cascade", model_name=None, regime=decision.regime,
            expectation=decision.expectation,
            reason="scale gap at or above the routing threshold; cascade.plan() decides "
                   "by footprint how many steps, and each runs register_step_dense. "
                   + decision.reason)

    if decision.regime == "cross_modal":
        name = str(cfg.get("cross_modal_matcher", "xoftr"))
        tile = cfg.get("cross_modal_tile_px")
        opts = {"tile_px": int(tile)} if tile else {}
        licence.assert_allowed(name)
        return MatcherChoice(
            route="direct", model_name=name, fine_stage_options=opts,
            regime=decision.regime, expectation=decision.expectation,
            reason=f"cross-modal pair: {name}, the only candidate measured at 10/10 on "
                   f"cross-modal" + (f", tiled at {tile} px in the fine stage" if tile else ""))

    name = str(decision.matcher or cfg.get("default_matcher", "eloftr"))
    licence.assert_allowed(name)
    fallbacks = tuple(m for m in (cfg.get("fallback_matchers") or []) if m != name)
    for m in fallbacks:
        licence.assert_allowed(m)
    return MatcherChoice(
        route="direct", model_name=name, regime=decision.regime,
        expectation=decision.expectation, fallbacks=fallbacks,
        reason=f"same-modality pair: default matcher {name}"
               + (f", then {', '.join(fallbacks)} if it is rejected" if fallbacks else ""))
