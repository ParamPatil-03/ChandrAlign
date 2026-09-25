# Our results next to the published baseline (arXiv:2509.04775)

The research doc's credibility floor: reproduce the published OHRC <-> NAC and IIRS <-> WAC results
next to a SIFT baseline. What the paper measures (read from the PDF, 2026-09-25):
- RMSE = RMS distance of their control points under their fitted (perspective) transform,
  reported per axis (X, Y), in the resampled grid (OHRC resampled to NAC's resolution, IIRS to
  WAC's 100 m) -- i.e. a FIT RESIDUAL, not accuracy against ground truth;
- ONE image pair per region (equatorial, polar) per pairing; inlier threshold not stated.

Our comparable number is the same kind of residual (`inlier_rmse_px`, 2-D, on the fine frame that
sits on the NAC / WAC grid), over MORE and HARDER data (OHRC: 5 NACs incl. 55, 80 deg and
opposite-sun illumination). Paper values combined as sqrt(X^2 + Y^2).

| Pairing | Method | Data | Fit RMSE (2-D, reference px) |
|---|---|---|---|
| OHRC -> NAC | paper SuperGlue | 1 pair equatorial / 1 polar | 0.85 / 1.19 |
| | paper SIFT | 1 pair equatorial | 6.96 (polar: failed) |
| | **ours (routed: eloftr, minima-loftr fallback)** | 24 windows, 5 NACs; 20 pass all gates | **median 1.24 (0.51-2.02)** |
| | ours SIFT | 14 windows registered; **5 pass gates** | median 0.54 (few points) |
| IIRS -> WAC | paper SuperGlue | 1 equatorial / 1 polar | 0.80 / 1.21 |
| | paper SIFT | 1 equatorial / 1 polar | 1.30 / 2.05 |
| | **ours (xoftr)** | 5 windows, 5/5 HIGH | **median 0.26 (0.26-0.48)** |
| | ours SIFT | 5 windows | median 0.24 (0.20-0.60) |

Reading, honestly:
- IIRS -> WAC: our residual is ~3x lower than the paper's best, on every window.
- OHRC -> NAC: our median residual (1.24) is between the paper's equatorial (0.85) and polar (1.19)
  SuperGlue values, over 24 windows that include illumination the paper did not attempt.
- A residual REWARDS few, self-consistent points: our SIFT has the lowest OHRC residual (0.54) yet
  fails the control gates on 9 of 14 windows. That is why we also report what the paper does not:
  the known-shift control gate, matcher-free accuracy (NCC probes, the MATCH-07 MI check) and
  errors in SOURCE pixels (docs/source_pixel_accuracy.md). By MI, the OHRC model's global offset is
  0.29-0.95 OHRC px on 10/10 coarse-NAC windows.
- Not a reproduction on their pairs: their OHRC products (ch2_ohr_ncp_20210401T235737..., NAC
  M1350459544RE) are not held. Same pairings, same resampling convention, different images.
