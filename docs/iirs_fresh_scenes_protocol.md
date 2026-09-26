# IIRS -> LRO WAC mosaic on fresh scenes: protocol (frozen before any result)

**Question.** The IIRS -> WAC result (`docs/iirs_wac_results.md`: xoftr 5/5 HIGH) comes from ONE IIRS
scene. Does the same pipeline, with nothing tuned, register IIRS scenes it has never seen, over
different terrain and latitudes?

This file is committed before any of the scenes below has been registered. Nothing here may be
changed after the first result exists except by a dated amendment that says why; a success rate
quoted after an amendment is not the pre-registered one.

## Scenes (chosen by region from the IIRS footprint index, before any data was opened)

From `ch2_iir_cal.shp` (PRADAN `IIRS_ShapeFiles.zip`), calibrated `_d_img_d18` products, one per
region the evidence scene does not cover (the evidence scene is 26 S - 8 N, 23-24 E, nearside mare):

| scene | footprint | region |
|---|---|---|
| `ch2_iir_nci_20260201T0533429523_d_img_d18` | 23-57 N, ~20 W | nearside northern mare (Imbrium) |
| `ch2_iir_nci_20240605T2337002684_d_img_d18` | 36-48 S, ~154 W | far-side southern highlands |
| `ch2_iir_nci_20231227T1021279603_d_img_d18` | 54-72 N, ~6 W | high northern latitude |
| `ch2_iir_nci_20260217T0308068424_d_img_d18` | 22 S - 18 N, ~133 E | far-side equatorial highlands |

**Exclusion (decided now, not after seeing results):** a scene whose label solar incidence at the
scene centre is >= 90 deg (night side) is excluded as unusable data, not counted as a failure. A
scene whose product cannot be read is reported as such, not counted.

## Method (unchanged from the adopted IIRS -> WAC path)

- Reference: the LROC WAC global 100 m mosaic, clip cut for the scene by
  `chandralign fetch-wac-clip --iirs <label>` (`io.wac_mosaic`, verified to reproduce the evidence
  clip byte for byte).
- Registration: `workflows.products.register_products(iirs_label, clip_json, windows=5)` --
  `workflows.iirs_wac.run_window`, the code the evidence came from, with the routed matcher (xoftr,
  tiled 640), the MIND coarse lock, `pipeline.fine_stage` defaults, all control gates, the tier.
  No parameter is changed for any scene.
- Windows: 5 per scene, placed by `workflows.iirs_wac.mosaic_windows` (inside the clip, 1.2 deg
  latitude margin).

## Success (per window) -- the evidence rule, `docs/iirs_wac_protocol.md`

Registered; all control gates pass; tier >= LOW; and the implied IIRS offset within **240 m** of
the median over that scene's gate-passing windows (>= 3 required). Verdict per scene: solved >= 90%,
degraded >= 60%, else unsolved.

## What each outcome allows us to claim

- **>= 3 of 4 scenes solved:** "IIRS -> WAC registers on fresh scenes it was not tuned on" (with
  the per-scene table).
- **Fewer:** the claim stays "one scene", and the failing scenes are reported with their
  failure modes as a limit.
- Accuracy is reported per scene as the evidence reported it (known-shift error, fit RMS, MI peak
  in IIRS px), p50 and max over windows -- never only the best scene.
