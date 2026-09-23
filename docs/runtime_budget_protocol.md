# Real-scale runtime criterion (MATCH-03/04/05, blocking the default switch)

**Status: FROZEN. Committed before the measurement was written or run.**
Check `git log` on this file against `reports/runtime_budget.json`. If a
threshold below is edited after results exist, the experiment is void.

---

## Why this exists

`docs/default_matcher_protocol.md` section 4.4 set a runtime gate of **under 5 s
per 512 px pair**, sourced from the research document's section 38 three-minute
demo. `xoftr` passed it at 0.53 s.

On **real 1536 px windows it takes 24.7 s**, against `eloftr`'s 1.6 s — about
15x. That is not a protocol violation. It is a gate that measured the wrong
thing: a per-pair figure at a tile size we do not actually use, never checked
against the end-to-end budget it was derived from.

So the default switch is blocked on this, and the question is deliberately NOT
"is xoftr too slow". It is:

> **Does the recommended pipeline fit the real application budget?**

A candidate is not replaced for being slower. It is replaced only if the budget
is actually violated and cannot be recovered.

## What is measured

Every configuration runs the same real data, same machine, same GPU.

| # | Workload | Why |
|---|---|---|
| 1 | one 1536 px real window | the unit the pipeline actually processes |
| 2 | several consecutive real windows | reveals per-call overhead vs warm cost |
| 3 | the complete TMC-2 -> SELENE demo path | the thing section 38's budget is about |
| 4 | a realistic full-strip workload | what a real product run costs |
| 5 | tiling / batching, if implemented | whether throughput is recoverable |

Recorded for each: total wall clock, per-window time, peak VRAM, and the split
between matching and everything else — because if the coarse MIND search or the
control gates dominate, the matcher is not the thing to optimise.

Candidates: `xoftr` (recommended), `aliked-lightglue` (incumbent), `eloftr`
(fastest measured). Others only if these three leave the question open.

**The control-gate cost is measured separately and reported separately.** In the
frozen run the gates cost 20-140 s per window against a 1.6-52 s match, so a
naive end-to-end number would be dominated by them and would say nothing about
the matcher. Gates are a per-window verification cost, not a per-product one,
and the demo path may legitimately run them once rather than per window.

## The criterion, fixed now

**PASS** — the complete TMC-2 -> SELENE demo path (workload 3) completes within
**180 s** on the demo machine.
*Source:* research document section 38, the three-minute demo. External to this
experiment and unchanged.

**CONDITIONAL PASS** — workload 3 exceeds 180 s, but a stated, implemented
change (tiling, batching, running gates once per product rather than per window,
or a reduced demo window count) brings it under 180 s, and that change is
measured rather than asserted.

**FAIL** — workload 3 cannot be brought under 180 s by any of the above.

**Only on FAIL** does the default question reopen, and then the options are, in
order: a faster eligible candidate, or a routed default (fast matcher where it
suffices, `xoftr` where it is needed). Quietly accepting an over-budget default
is not an option.

Per-window and full-strip figures (workloads 1, 2, 4) are **reported, not
gated**. There is no external source for a full-strip budget, so inventing a
threshold for it here would be exactly the "measurably" problem this project has
already been bitten by twice.

## What this cannot establish

- One machine, one GPU (RTX 4050 laptop). The demo machine may differ; the
  research document's section 34 assumes RTX 2080Ti-class or better, so these
  numbers are a lower bound on available hardware, not an upper bound on cost.
- Nine windows over three TC tiles is not a full product run; workload 4
  extrapolates and is labelled as such.
- First-call cost includes weight loading and CUDA warm-up. Reported separately
  from steady-state, because a demo pays it once and a product run does not pay
  it per window.
