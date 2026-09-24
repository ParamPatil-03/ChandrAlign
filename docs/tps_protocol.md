# Protocol: does real data need TPS (ALIGN-02)? Frozen before measuring.

## Question

The pipeline fits one affine model per window. On the N00 TMC-2 -> SELENE TC windows the
residual to that model is ~1.8 px (others 0.5-1.3), which is what ground an affine map
cannot describe (relief, pushbroom attitude drift) would look like. Does a thin-plate
spline (TPS, `estimate/models.fit_tps`) describe the real geometry better -- on points it
was NOT fitted to?

## Data

The nine committed TMC-2 -> SELENE TC windows (`scripts/register_tmc2_tc.py --tile all
--windows 3`, current defaults), with `--dump-points` saving per window: the delivered
control points (uniform, sub-pixel refined, <= 384) and all inliers of the first robust
estimate, both in the fine frame.

## Method (`scripts/tps_heldout.py`)

- TPS fitted on the CONTROL points only. Smoothing chosen by 5-fold cross-validation on
  the control points alone, from {0.1, 1, 10, 100, 1000}.
- Held-out set: every first-estimate inlier that is not a control point (thousands per
  window).
- Metric: RMS residual on the held-out set, TPS vs the pipeline's affine model.

## Decision

Among windows whose affine held-out RMS exceeds 1.0 px, the median improvement from TPS
must be >= 20%, and on NO window may TPS be more than 5% worse. If so, TPS is added as a
pipeline stage (`tps`, on), fitted on the delivered control points and carried alongside
the affine model (which still drives the gates and scale check, as TPS cannot be composed
as a matrix). Otherwise TPS stays unused and the result is recorded.

Held-out residuals include matcher noise, so this bounds, not measures, the non-affine part.
