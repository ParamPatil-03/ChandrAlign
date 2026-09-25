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

## Result (2026-09-25): `ref` ADOPTED (the default is now the ground point)

`reports/tmc2_tc_parallax_height_ref.json` (15 windows, height_at ref), against amendment 3's
`reports/tmc2_tc_parallax_a3.json` (height_at src). Median NCC-probe error of affine + parallax:

| | src (before) | ref (now) |
|---|---|---|
| hilly, median of 9 windows | 0.415 px | **0.352 px (-15%)** |
| hilly, per window | 0.359-0.618 | 0.282-0.507 (9/9 lower, by 0.02-0.13 px) |
| flat, per window | 0.240-0.293 | 0.238-0.298 (within +0.005) |

(a) pass, (b) pass, (c) pass: registration, gates and tier unchanged on all 15; 0 empty cells on
every hilly window. p / 26 deg prediction on hilly windows 1.01-1.03 (0.97-1.04 with src).
`configs/default.yaml parallax.height_at: ref`. This is also the RPC's convention, so the model,
our remap and a GDAL RPC export now describe the same geometry (docs/rpc_export_protocol.md).
Tests built on a source-height synthetic world now say so (parallax_height_at="src").
