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
