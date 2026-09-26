# Protocol: illumination-invariant per-point refinement (audit G-05, part 1). Frozen before measuring.

## Question

Under an opposite-sun proxy (the harness's "inverted" shading) the intensity refinement has nothing to lock
on. Its score gate correctly refuses every move, so the points keep the matcher's error (dev: TMC-2 r110k
inverted, p50 0.79 / p95 2.51 ref px). Can refinement on **MIND channels** (the descriptor the pipeline already
uses to survive lighting change) refine there, without costing accuracy where intensity already works?

## Change under test

`subpixel.representation: auto` (vs `intensity`, today's). Each point first goes through the current
intensity path. Only if no intensity candidate passes the score gate, the SAME warped patches are converted to
MIND channels, a shift is estimated on the channel-averaged NCC surface (the integer peak, then the Gaussian
3-point fit, iterated), and the move is gated by the mean per-channel NCC (it must rise), within the same
max_move. Intensity points are therefore untouched by construction; only points that intensity refused can
change.

## Measurement (known-warp harness, `scripts/known_warp_harness.py`)

- dev, held-out and fresh sets; every case that registers, INCLUDING the inverted proxy (so far 1 dev and
  1 held-out inverted case register; the rest do not register and are reported as such).
- **Bars:** (1) on every registered inverted case, the refined points beat unrefined at p50 AND p95;
  (2) on the 15 scored cases of each set, p95 is no worse than `intensity` by more than 10% (the null test
  of G-07 showed re-selection noise near 5%; here the points are identical unless intensity refused them).

Evidence will be thin: 1-2 inverted cases per set. A pass is reported as "consistent with", not "shown".

## Not done: part 2, rendered references

Rendering the reference under OHRC's sun needs a DTM at OHRC-comparable resolution (LROC NAC DTMs near the
scenes). None is held (`data/raw/dem`: LOLA and SLDEM2015 at ~60 m; SELENE TC DTM at ~10 m, not over the OHRC
scenes). It is blocked on data, not attempted.

## Result (2026-09-26): bar 2 met, bar 1 NOT met. Not adopted; ships switchable, default off.

`scripts/known_warp_harness.py --set <set> --no-legacy --mind-arm` -> `reports/known_warp_<set>_g05.json`.

| set | scored cases where MIND fallback p95 is > 10% worse | inverted proxy (registered cases): unrefined / intensity / MIND fallback, p50 / p95 ref px |
|---|---|---|
| dev | 0 / 15 | TMC-2 r110k: 0.791 / 2.508, 0.791 / 2.508 (0 moved), **0.341** / 2.508 (10 moved) |
| held-out | 0 / 15 | TMC-2 r90k: 0.917 / 2.623, 0.917 / 2.623 (0 moved), **0.692** / 2.623 (6 moved) |
| fresh | 0 / 15 | no inverted case registered |

- The fallback never costs accuracy where intensity works (by construction, and measured).
- On the opposite-sun proxy it moves only 6-10 points: those where MIND finds a clear peak within the
  1.5 px cap. The median improves, but the p95 (points 2+ px off) is out of reach of a local refinement, so
  bar 1 (beat unrefined at p50 AND p95) is not met. The remaining error there belongs to the MATCHER, not the
  refiner: G-03 (RoMa) is the lever for it.
- `subpixel.representation` stays `intensity`; `auto` is implemented and tested for anyone who wants it.
