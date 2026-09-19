# Part 1 accuracy report

Generated 2026-09-19T18:45:01+00:00 by `scripts/accuracy_report.py --repeats 10 --samples 200 --seed 0`. Every number below is measured; regenerate rather than edit.

Each check compares our code with an independent reference and was run 10 times with the same random samples; **Repeatable** counts runs whose result was identical to the first.

| Step | Check | Method | Correct | Accuracy | Other measurements | Repeatable | Time / run |
|---|---|---|---|---|---|---|---|
| 3 | Camera detection | Every product ID in ISRO's archive index + every product we hold | 25,575 / 25,575 | 100.00% | – | 10/10 | 0.7 s |
| 4 | CH-2 label fields | Parsed fields vs a separate plain-text read of the same XML | 24 / 24 | 100.00% | – | 10/10 | 0.1 s |
| 5 | Pixel reading | Random pixels vs hand-computed byte offsets (all 7 products) | 1,400 / 1,400 | 100.00% | median_ms = 5.19; p95_ms = 228.59 | 10/10 | 54.3 s |
| 5, 7 | Publisher checksums | ISRO / NASA MD5 of whole files and NASA pixels-only MD5 | 7 / 7 | 100.00% | – | 10/10 | 35.5 s |
| 6 | SELENE pixel statistics | Our min/max/mean/std vs the values JAXA wrote in the label | 8 / 8 | 100.00% | max_abs_error = 4.9e-07 | 10/10 | 11.1 s |
| 8 | Tiling | Grid covers every pixel of every product; point round-trip error | 7 / 7 | 100.00% | roundtrip_max_error_px = 1.46e-11 | 10/10 | 0.6 s |
| 9 | Sun geometry | Incidence vs 90 - elevation (two label fields); azimuth maths | – | – | incidence_plus_elevation_max_error_deg = 0.0; azimuth_difference_symmetric = True; ohrc_vs_tmc2_d_azimuth = 165.553592 | 10/10 | 0.2 s |
| 10 | Footprint overlap | check_overlap vs area-weighted Monte-Carlo, all 11 real pairs | 11 / 11 | 100.00% | max_error_pp = 0.74; mean_error_pp = 0.29; monte_carlo_se_pp = 0.79 | 10/10 | 4.0 s |
