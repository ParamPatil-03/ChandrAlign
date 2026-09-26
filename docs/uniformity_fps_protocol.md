# Protocol: uniform match points INSIDE grid cells (audit G-07). Frozen before measuring.

## Question

`refine/uniformity.enforce` keeps the top-k points per grid cell BY CONFIDENCE, so inside a cell the
kept points cluster wherever the matcher was most confident. The audit measured a nearest-neighbour
coefficient of variation (NN-CV: std/mean of each point's distance to its nearest neighbour) of
0.73-0.92 on delivered points. Uniform random scores about 0.52, and farthest-point selection 0.13. The
problem statement asks for match points "uniformly distributed". Does quality x spacing selection make
them uniform without costing accuracy?

## Change under test

`uniformity.within_cell: spread` (vs `confidence`, today's rule): cells are visited round-robin, one
pick per cell per round, up to top_k. Each pick maximises
`d_min x (confidence / max confidence) ^ uniformity.spread_conf_power`, where d_min is the distance to
EVERY point kept so far (any cell), so spacing is global and not only per cell. The first pick is the most
confident point. The grid, top_k and coverage are unchanged, so tiers and coverage cannot move.

## Measurement

`scripts/known_warp_harness.py`, dev set for the choice of `spread_conf_power` from {0, 0.5, 1}, fresh set
for confirmation. On the delivered points of each scored case:
- NN-CV and the max Delaunay gap, in the SOURCE frame and in the REFERENCE frame (over valid overlap);
- the delivered points' true error p50/p95 (refined).

**Accept (both sets):** median NN-CV <= 0.45 (under uniform random); the max Delaunay gap not larger in
more than 3/15 cases; delivered-point p95 no more than 10% worse than `confidence` in every case.

## Result (2026-09-26, dev set): NOT ACCEPTED as the default. The option ships switchable.

Delivered points, 15 dev cases (per-case NN-CV, max Delaunay gap, true error of refined points):

| selection | median NN-CV (src / ref) | gap larger than `confidence` | p95 > 10% worse | median p95 ratio vs `confidence` |
|---|---|---|---|---|
| confidence (today) | 0.865 / 0.865 | - | - | 1 |
| spread, power 0 | 0.415 / 0.414 | 1/15 | 6/15 | 1.070 |
| spread, power 0.5 | 0.422 / 0.422 | 1/15 | 3/15 | 1.072 |
| spread, power 1 | 0.447 / 0.446 | 1/15 | 4/15 | 1.057 |
| spread, pool 2-3 x top_k | 0.413-0.433 | 1-2/15 | 3-5/15 | - |
| **null:** confidence rule, confidences shuffled | 0.751-0.756 | 2-7/15 | **3/15** | 1.041-1.050 |

- Uniformity: every spread variant meets the bar (NN-CV <= 0.45; about 2x more regular than today), coverage
  is identical, and the largest hole shrinks in 13-14/15 cases.
- Accuracy guard ("p95 no more than 10% worse in every case"): failed by every variant. The null row shows
  the guard is at the sampling-noise level. An arbitrary re-selection of equally good points "fails" it in
  3/15 cases, because a p95 over ~384 points rests on ~19 points. Even so, spread's median p95 ratio
  (1.06-1.07) is above the null's (1.04-1.05), so there is a small real accuracy cost: about +0.005 px
  on a p95 of about 0.08 px.
- The bar is not amended after the fact. The frozen rule says no, so the fresh set was not run, and
  `uniformity.within_cell` stays `confidence`.
- `spread` is implemented, tested (`tests/test_uniformity.py`) and available. Switching the default is a
  product trade-off (PS "uniformly distributed" vs ~+0.005 px at p95) for the team to decide. If they
  switch, it should be confirmed on the fresh set first.
