# IIRS -> LRO WAC on real data: results

Protocol frozen before measuring: `docs/iirs_wac_protocol.md` (three amendments, each
recorded before the run it affected). Result = mosaic run 3, `reports/iirs_wac_mosaic.json`.
**The first real registration of IIRS in this project.**

| matcher | windows | verdict | tiers | known-shift error |
|---|---|---|---|---|
| **xoftr (routed, tiled 640)** | **5/5** | **solved** | all HIGH | 0.021-0.063 px |
| minima-loftr | 5/5 | solved | all HIGH | 0.11-0.47 px |
| sift | 5/5 | solved | LOW-HIGH | 0.002-0.075 px |
| rift2 | 0/5 | unsolved | -- | 2 windows passed the gates; the rule needs 3 |

Coarse lock (MIND on the IIRS PREP-06 composite, block-averaged to the mosaic's 100 m):
z 52-69 on all 5 windows. Every success is location-consistent (within 240 m of the
product median; in practice within ~150 m, drifting smoothly along the strip).

## What it shows

1. **The credibility floor holds.** IIRS <-> WAC registers on every window with the
   routed cross-modal matcher at HIGH confidence. Prediction 1 (xoftr solved): confirmed.
   Prediction 2 (sift below xoftr): NOT confirmed -- sift also registers 5/5 (lower tiers);
   on this pairing the matcher is not the hard part.
2. **The reference product matters more than the matcher.** Against raw WAC CDR frames
   (run 1) no window locked (z 9-10): a ~90 deg field-of-view camera's frames cannot be
   described by one affine map over 258 km. Against the map-projected WAC global mosaic,
   every window locks at z > 50. **Use map-projected WAC, not raw CDR frames.**
3. **IIRS's system geolocation is ~12.8 km off** (north; ~1.2-1.5 km east), consistently
   along the strip -- consistent with GEO-07's finding of system/refined corners up to
   14.6 km apart. Window placement must allow for it (amendments 2-3), the same lesson
   as OHRC <-> NAC.

## Limits

One IIRS scene, one reference (the mosaic, which is itself a composite of 2010 WAC images),
5 windows along ~160 km of strip. The known-shift gate measures lock precision, not
absolute accuracy.
