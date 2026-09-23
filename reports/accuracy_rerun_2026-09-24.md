> **Re-run 2026-09-24 on a second machine, kept beside `accuracy.md`, not replacing it.** Two rows differ from `accuracy.md` for stated reasons: row 3 covers 7 product IDs instead of 25,575 because the PRADAN archive shapefiles (login required) are not on this machine; row 13e now renders each product's sun AT the DEM patch (TMC-2 92.6/46.3 deg) instead of the label's scene-centre value (104.3/44.0), the fix recorded in docs/AUDIT_2026-09-23.md. Every other row reproduces.

# Part 1 accuracy report

Generated 2026-09-23T18:29:30+00:00 by `scripts/accuracy_report.py --repeats 10 --samples 500 --seed 0`. Every number below is measured; regenerate rather than edit.

Each check compares our code with an independent reference and was run 10 times with the same random samples; **Repeatable** counts runs whose result was identical to the first.

| Step | Check | Method | Correct | Accuracy | Other measurements | Repeatable | Time / run |
|---|---|---|---|---|---|---|---|
| 3 | Camera detection | Every product ID in ISRO's archive index + every product we hold | 7 / 7 | 100.00% | – | 10/10 | 0.0 s |
| 4 | CH-2 label fields | Parsed fields vs a separate plain-text read of the same XML | 24 / 24 | 100.00% | – | 10/10 | 0.1 s |
| 5 | Pixel reading | Random pixels vs hand-computed byte offsets (all 7 products) | 3,500 / 3,500 | 100.00% | median_ms = 1.58; p95_ms = 2.11 | 10/10 | 6.3 s |
| 5, 7 | Publisher checksums | ISRO / NASA MD5 of whole files and NASA pixels-only MD5 | 7 / 7 | 100.00% | – | 10/10 | 21.6 s |
| 6 | SELENE pixel statistics | Our min/max/mean/std vs the values JAXA wrote in the label | 8 / 8 | 100.00% | max_abs_error = 4.9e-07 | 10/10 | 5.4 s |
| 8 | Tiling | Grid covers every pixel of every product; point round-trip error | 7 / 7 | 100.00% | roundtrip_max_error_px = 1.46e-11 | 10/10 | 0.2 s |
| 9 | Sun geometry | Incidence vs 90 - elevation (two label fields); azimuth maths | – | – | incidence_plus_elevation_max_error_deg = 0.0; azimuth_difference_symmetric = True; ohrc_vs_tmc2_d_azimuth = 165.553592 | 10/10 | 0.1 s |
| 10 | Footprint overlap | check_overlap vs area-weighted Monte-Carlo, all 11 real pairs | 11 / 11 | 100.00% | max_error_pp = 0.78; mean_error_pp = 0.22; monte_carlo_se_pp = 0.5 | 10/10 | 5.0 s |
| 11 | Geolocation / projection | SELENE corners vs label; every ISRO grid node; ground->pixel inverse; map round trip | 169,529 / 169,529 | 100.00% | grid_nodes_checked = 164021; reprojection_max_error_deg = 1.78e-14; median_m_independent_vs_refined = {'OHRC': 0.2, 'TMC2': 5146.6, 'IIRS': 13295.0} | 10/10 | 0.7 s |
| 12b | Per-pixel sun/camera angles | Label value at scene centre; spherical triangle inequality; nadir column vs swath edge | 9 / 9 | 100.00% | incidence_spread_deg = {'OHRC': 0.142, 'TMC2': 7.8, 'IIRS': 9.279} | 10/10 | 0.1 s |
| 12c | Label cross-validation | Optics, corner geometry and orbital mechanics vs what the label states. NOTE: the 2 disagreements are real defects FOUND in ISRO labels, not our errors | 7 / 9 | 77.78% | per_product = {'OHRC': {'checks': 3, 'failed': ['along_track_gsd_m'], 'corner_disagreement_m': 0.0}, 'TMC2': {'checks': 3, 'failed': ['cross_track_gsd_m'], 'corner_disagreement_m': 5388.5}, 'IIRS': {'checks': 3, 'failed': [], 'corner_disagreement_m': 14577.4}} | 10/10 | 0.1 s |
| 13b | Shadow detection | Detected shadow vs a mask traced by hand on a real OHRC crater crop | 251,986 / 262,144 | 96.13% | iou = 0.8937; hand_only_median_DN = 25.0; code_only_median_DN = 5.0 | 10/10 | 0.0 s |
| 13c | IIRS composite | Repeatable features (independent-noise pair) vs the best single band | 1,042 / 1,769 | 58.90% | bands_kept = 125; composite_matches = 1042; best_single_band_matches = 727; raw_keypoints_composite = 1318; raw_keypoints_single = 1362 | 10/10 | 6.5 s |
| 13d | Terrain scores | Texture and repetitiveness on real crops of known character | 3 / 3 | 100.00% | crater_field = {'texture': 0.01153, 'raw': 0.01156, 'repetitive': 0.107}; smooth = {'texture': 0.0, 'raw': 0.0005, 'repetitive': 0.0}; noisy_dark = {'texture': 0.00173, 'raw': 0.00189, 'repetitive': 0.0} | 10/10 | 0.1 s |
| 13e | Illumination invariance | Real relief under our two real suns: raw vs phase congruency vs MIND | 3 / 3 | 100.00% | raw_brightness = -0.9804; phase_congruency = 0.923; mind = 0.9712; sun_at_patch_az_elev_deg = {'OHRC': [269.8, 7.3], 'TMC2': [92.6, 46.3]} | 10/10 | 0.6 s |
