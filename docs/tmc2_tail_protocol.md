# Protocol: the TMC-2 -> SELENE TC accuracy tail (audit G-06). Frozen before measuring.

## Where it stands (Track B batch 1, snapshot `db3a350`)

After C-02/G-02 the matcher-free probe error of the delivered geometry is p50 0.18-0.35 TMC-2 px on 13/15
windows. Still, **8/15 windows have p95 > 1 TMC-2 px** (0.94-1.47). These are the hilly N00 windows and
their column offsets. Two windows delivered the parallax model, because TPS beat it by less than the 5%
`min_gain`, and they had the worst p50 (0.48-0.53). On relief neither model alone is right: the DEM explains
the parallax it can predict, and it misses what the DEM gets wrong or cannot resolve.

## Change under test (in order; stop at the first that meets the bar)

1. **`parallax_tps` candidate** in `estimate/selection.py`: the Huber-refit parallax model, then a TPS fitted
   on ITS residuals (smoothing by the same CV), scored by the same stratified CV as every other candidate. It is
   richer than both parents, so it must beat the best of them by `min_gain`. It is delivered as a composite
   model: prediction = parallax(p) + tps_residual(p).
2. If 1 is not enough: `geometry.max_fit_points` 1000 -> 3000, a denser fit set for the TPS parts.

A dense optical-flow residual field (DIS/Farneback on MIND channels) is deliberately NOT first. It fits the
geometry to image intensity, which is what the probes also measure, so the probes would stop being
independent of the fit. It is attempted only if 1-2 fail, and then scored on held-out probes only.

## Measurement

The 15 committed TMC-2 -> TC windows (`register_tmc2_tc.py`, same rows and columns as the batch), recording
`probe_check`. Metric: per-window probe p95 in TMC-2 px (fine px x factor_worst 1.501), plus p50.

**Bar (the audit's):** p95 < 1 TMC-2 px on **every** window. Also report: the windows moved from > 1 to < 1;
that no window's p50 gets worse by more than 10%; tiers unchanged.

Disclosure: the probes are also the diagnostic that motivated this change (batch 1). The bar is the audit's,
fixed before this protocol. The change is judged on a new run, not on batch 1's numbers.
