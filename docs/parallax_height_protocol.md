# Protocol: where the parallax stage samples the height (frozen before running)

Found by the RPC check (docs/rpc_export_protocol.md): the ALIGN-08 stage samples the DEM at each
match's SOURCE pixel position (in the coarse-aligned TC frame), but the physical model is

    TC px r of a ground point = A . (TMC-2 px s) + (h(ground) - h0) . p,   ground = r (TC is ortho)

so h belongs at the REFERENCE point r, up to ~25 px along-track from s. On window 4687 the two
conventions put pixels 0.91 px RMS / 2.67 px p99 apart. Question: which predicts the images better?

## Change under test
`parallax.height_at: ref` (config; `--parallax-height-at ref`): the stage samples h at each match's
reference point. The model then reads ref = A.src + (h(ref) - h0).p, applied by fixed-point
iteration forward and DIRECTLY backward (s = A^-1 (r - (h(r) - h0) p), the RPC form). Everything
else as adopted (amendment 3: TC DTM for parallax only, SLDEM for the terrain filter).

## Measurement
The 15 windows of amendment 3 (9 hilly N00, 6 flat N03/N09), rerun with `height_at: ref`, scored
with the same NCC probes (`scripts/parallax_probe_eval.py`, which applies each model in its own
convention). Baseline: amendment 3's run (`height_at: src`, `reports/tmc2_tc_parallax_a3.json`).

## Decision (all must hold to switch the default to `ref`)
(a) Hilly: the median over the 9 windows of the per-window median probe error falls by >= 10%,
    and no hilly window's median rises by > 0.05 px.
(b) Flat: no window's median rises by > 0.05 px.
(c) No window loses registration, a gate or tier (vs the SLDEM parallax-off baselines), and every
    hilly window keeps <= 1 empty cell.
Otherwise `src` stays, and the RPC/remap delivery documents that the model's h is sampled at the
source position.
