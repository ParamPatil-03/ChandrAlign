# Track B -> Track A: interface changes on `track-b/accuracy`

Written for the Track A (product wiring) owner. Two cross-session messages expired before approval, so the
contract points are recorded here instead. Branch `track-b/accuracy`, worktree
`C:\Users\524ta\OneDrive\Desktop\sih\chandrayana2-trackB`, based on `8f87746` (the same base as Track A).

## `register_bundle`: signature UNCHANGED (as the contract in audit section 2 requires)

Only its internals and outputs changed:

| output | before | now |
|---|---|---|
| `result.metrics.rmse_px` | the affine's in-sample residual on the delivered points | **check-point RMSE of the DELIVERED geometry**, ref px (C-03) |
| `result.metrics.rmse_m` | always None | `rmse_px` x `ref.gsd_m` |
| `result.metrics.max_delaunay_gap_px`, `subpixel_recovery_err_px` | never filled | filled (I-10) |
| `result.provenance["accuracy"]` | - | `rmse_px_ref`, `rmse_px_src`, `rmse_m`, `fit_residual_px` (the old number), `model`, `point_set`, `n_check`, `geometry_error_lower_px_ref`, `probes` (I-08) |
| `result.provenance["crosscheck"]` | - | C-04: `verdict` agree / flag / inconclusive / no checker, `gap_px`, `affine_gap_px` |
| `result.gates["independent_crosscheck"]` | - | present when the checker had an answer; a flag -> REJECTED (mode 12) |
| tier | inliers / ratio / coverage / scale | + `accuracy` signal (I-08) and the cross-check cap (inconclusive -> at most MEDIUM) |
| `RegistrationBundle.geometry`, `.geometry_model` | - | **the geometry to warp with**: "affine" / "tps" / "parallax" / "parallax_tps" and its model |

**Runtime:** `register_bundle` now also runs RIFT2 (~7 s per 1024 px tile, CPU) and the probe check (<1 s).

## Things Track A should do in its own files

1. **`cli.py:160` and `product/report.py:155`** guess the geometry name from `bundle.parallax` / `bundle.tps`.
   Use `product.warp.best_geometry(bundle)[0]` instead, or the affine-vs-TPS choice is misreported.
2. **`register_products` (C-01)** should forward to `fine_stage`: `ground_model=<src>`,
   **`ref_ground_model=<ref>`** (new, I-10), `dem`, `parallax_dem`, `parallax_height_at`, `rematch`,
   `expected_scale`. Add those parameters to `register_bundle` and pass them through. Without
   `ref_ground_model`, the source model serves both frames, which is only valid after the coarse lock.
3. **Fallbacks:** use `matching.routing.run_candidates(choice, evaluate, ok)`, the one shared loop (I-09).
4. **Source px:** `provenance["accuracy"]["rmse_px_src"]` is in the frame `fine_stage` saw. If
   `register_products` resamples the source into a fine frame (as the scripts do), compose back with the
   fine -> source scale (`evaluate/source_px.to_source_px`).
5. **UI (C-05):** show the tier vocabulary together with `provenance["crosscheck"]["verdict"]` and
   `provenance["accuracy"]` (`rmse_px_ref`, `p50_px_ref`/`p95_px_ref`, and the probe p50/p95 in source px).

## Shared files Track B touched (keep these hunks on merge)

- `product/warp.py`: `best_geometry` honours `bundle.geometry` / `geometry_model`. The TPS export uses
  `tps_params["inverse"]`. The parallax / parallax_tps source maps.
- `matching/licence.py` (I-06): ship mode is an ALLOWLIST (licence_audit.json `pass` rows +
  shippable_matchers + our own methods). `--benchmark-models` / `licence.enable_benchmark_mode()` for
  benchmarks only; `run_record()["ship_mode"]` records it.
- `configs/default.yaml`: `subpixel:`, `geometry:`, `tiers.accuracy`, `gates.crosscheck*`,
  `matching.dense_*`, `uniformity.within_cell`.
- `configs/instruments.yaml`: `matched_as` for WAC and MI (I-09).
