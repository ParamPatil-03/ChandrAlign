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

## Result (2026-09-26)

Probe p50 / p95 of the delivered geometry, TMC-2 px (fine px x 1.501), the 15 committed windows. Runs: batch 1
(before G-06, snapshot `db3a350`), step 1 (batch 2, `74171b3`, parallax_tps on), step 2 (batch 4 = batch 2 code
with `geometry.max_fit_points` 3000 and `geometry.parallax_tps` false).

| window | before G-06 | step 1 | **step 2** |
|---|---|---|---|
| 1562 | 0.306 / 1.048 | 0.356 / 1.124 (parallax_tps) | **0.231 / 0.663** |
| 3125 | 0.323 / 1.214 | 0.323 / 1.214 | **0.249 / 0.931** |
| 4687 | 0.294 / 1.162 | 0.294 / 1.162 | **0.221 / 0.830** |
| 6250 | 0.290 / 1.069 | 0.410 / 1.160 (parallax_tps) | **0.212 / 0.687** |
| 16000 | 0.204 / 0.623 | same | **0.156 / 0.407** |
| 18750 | 0.212 / 0.503 | same | **0.161 / 0.423** |
| 21500 | 0.203 / 0.633 | same | **0.159 / 0.432** |
| 52000 | 0.176 / 0.452 | same | **0.135 / 0.369** |
| 54750 | 0.206 / 0.549 | same | **0.156 / 0.414** |
| 57500 | 0.207 / 0.638 | same | **0.161 / 0.455** |
| 1562 c768 | 0.525 / 1.400 | 0.393 / 1.142 (parallax_tps) | **0.257 / 0.913** |
| 3125 c768 | 0.267 / 1.022 | same | **0.219 / 0.875** |
| 1562 c3232 | 0.482 / 1.249 | 0.366 / 1.169 (parallax_tps) | **0.219 / 0.773** |
| 3125 c3232 | 0.264 / 0.938 | same | **0.215 / 0.689** |
| 4687 c3232 | 0.354 / 1.474 | same | **0.267 / 1.333** |
| **windows with p95 > 1** | 8 | 8 | **1** |

- **Step 1: not adopted.** It was chosen on 4 windows. It improved 2 of them (c768, c3232) and made 2 worse at p50
  (6250 +41%, 1562 +16%), which breaks the 10% guard, and it moved no window below p95 1 px. The check-point CV
  preferred it where the independent probes disagree. `geometry.parallax_tps` is off; the code stays, tested.
- **Step 2: adopted.** p50 improves on every window (0.135-0.267 TMC-2 px), and p95 drops below 1 px on 14/15.
  Tiers are unchanged, and there are 0 cross-check flags. Cost: the TMC-2 -> TC run takes 1141 s instead of 487 s
  for 10 windows (the TPS on 3000 points).
- **The bar (p95 < 1 TMC-2 px on EVERY window) is NOT met: 14/15.** 4687 c3232 (hilly N00, column offset +1232) is
  at 1.333. The next step in this protocol is the dense residual field, which it deliberately postpones
  (it would share the probes' intensity signal); it stays a follow-up.
- Fresh windows (I-16, batch 3, step-1 code): 5/5 below p95 1 px (0.45-0.68), before step 2.
