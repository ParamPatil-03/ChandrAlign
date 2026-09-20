# CHANDRALIGN

**SIH26166 (ISRO):** register Chandrayaan-2 images (OHRC, TMC-2, IIRS) onto external lunar references (LRO NAC/WAC, SELENE TC/MI) with sub-pixel accuracy, despite different sun angles, resolutions and sensor types.

- **Plan:** [`PLAN.md`](PLAN.md) is the single source of truth: architecture, tasks, honesty rules.
- **Features:** [`FEATURES.csv`](FEATURES.csv) lists all 85 features with owner, task, status and "done when".
- **Plain-language guide:** [`docs/FEATURE_GUIDE.html`](docs/FEATURE_GUIDE.html).

> This README is updated after every completed step. Last update: **Step 13c, IIRS band quality and composite** (PREP-05, PREP-06).

---

## Status at a glance

**Gate A is open:** real Chandrayaan-2 and real LRO products can be read and cut into `ImagePlane` tiles. **Member B can start on real data.**

| Step | What | Features | Status |
|---|---|---|---|
| 1 | Project skeleton (`pyproject.toml`, package tree, smoke test) | P0-T01 | ✅ |
| 2 | Shared contract `contracts.py` (frozen) | P0-T02 | ✅ needs B + C sign-off |
| 3 | Camera registry and auto-detection | DATA-10 | 🟡 WAC/MI patterns untested on real files |
| 4 | Chandrayaan-2 PDS4 label reader and anti-stub test | DATA-01, 03, 04, 05 | ✅ |
| 5 | Windowed pixel reader and integrity checks | DATA-01 | ✅ |
| 6 | SELENE PDS3 reader | DATA-02, DATA-08 | ✅ reader · 🟡 needs ≥3 TMC-2↔SELENE pairs |
| 7 | LRO NAC reader | DATA-06 | ✅ |
| 8 | Tiling → `ImagePlane` | DATA-11 | ✅ **Gate A** |
| 9 | Sun angles and lighting difference between two scenes (source of every value recorded) | GEO-01, DATA-12 | 🟡 scene level done · per-pixel layers with GEO-02 |
| 10 | Footprint overlap and pre-filter (rejects a pair before any matcher runs) | GEO-04 | ✅ |
| 11 | Moon map projection (IAU 2015) and pixel ↔ Moon-position models | GEO-05 | ✅ |
| 12 | Elevation maps (LOLA, SLDEM2015) and slope / aspect | DATA-13, GEO-03 | ✅ on real LOLA tiles |
| 13a | Contrast preparation: percentile stretch + CLAHE | PREP-01 | ✅ |
| 13b | Shadow detection → `shadow_mask` | PREP-04 | ✅ IoU 0.894 vs a hand-traced mask (plan needs > 0.7) |
| 13c | IIRS: score 256 bands, blend the good ones into one plane | PREP-05, PREP-06 | ✅ +43% repeatable features vs the best single band |
| next | 13d flat / repetitive terrain · 13e phase congruency, MIND | PREP-07, 08, 02, 03 | ⏳ |

**Tests:** 249 passing, 0 skipped (`pytest -m ""`), including checks on every real product we hold.

---

## Measured accuracy

Every completed step is checked against an **independent** reference (a different code path, the publisher's own checksums or statistics, or a Monte-Carlo estimate). Each check runs 10 times to prove repeatability.

- **[`reports/accuracy.md`](reports/accuracy.md)**: per-step correct / total, errors and timing. Regenerate with `python scripts/accuracy_report.py`; never edit the numbers by hand.
- **[`reports/test_repeats.md`](reports/test_repeats.md)**: the full test suite run 10 times back to back.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest                 # quick run (~20 s)
pytest -m ""           # everything, including the multi-GB checksum checks (~40 s)
```

Python ≥ 3.11. Only the data-layer dependencies are installed so far. Heavier groups (matching, geo, product) are listed in `pyproject.toml` and get added by the step that first needs them.

**Without the data:** every test that needs a real image **skips** with a message. Label tests still run everywhere, because the real labels (a few KB each) are committed under `tests/fixtures/labels/`.

---

## Data

Images are **never committed** (`data/` is gitignored). `data/manifest.json` **is** committed: it records the size and SHA-256 of every file, so any result traces back to exact bytes. After downloading anything, run `python scripts/make_manifest.py`.

### What we hold: one equatorial test region

Latitude −0.44 to +0.37, longitude 23.45 to 23.59. It's the one spot where all three Chandrayaan-2 cameras and both references overlap.

| Product | Camera | Size | Detail | Sun elevation | Source |
|---|---|---|---|---|---|
| `ch2_ohr_ncp_20240330T0035085365` | OHRC | 79,796 × 12,000, uint8 | 0.30 m | **7.3°** (near sunset) | PRADAN |
| `ch2_tmc_nca_20250207T1102039417` | TMC-2 (aft) | 160,269 × 4,000, uint16 | 4.41 m | 44.0° | PRADAN |
| `ch2_iir_nci_20240523T1600301891` | IIRS | 256 bands × 13,101 × 250, float32 | 97.15 m | 57.3° | PRADAN |
| `M1417360906LC` | LRO NAC | 52,224 × 5,064, int16 | 0.5 m | 82.3° (ODE catalogue; incidence 7.7°) | NASA ODE |
| `M102000149RC` | LRO NAC | 52,224 × 5,064, int16 | 0.5 m | 10.3° (ODE catalogue; incidence 79.75°) | NASA ODE |
| `TCO_MAP_02_N03E021N00E024SC` + `…N00E021S03E024SC` | SELENE TC ortho mosaic | 12,288 × 12,288, uint16 BE | 7.40 m | *mosaic, none* | JAXA DARTS |

Each NAC folder also holds `ode_metadata.json`, NASA's catalogue record (angles and footprint) saved at download time so nothing depends on ODE being reachable later.

Every product's pixels match the publisher's own MD5 checksum (NAC: both the whole-file and the pixels-only checksum). The SELENE tiles also reproduce JAXA's label statistics exactly.

### Getting the data

| Source | How | Login |
|---|---|---|
| Chandrayaan-2 (PRADAN) | Manual download, since SSO can't be scripted. Download the **calibrated** zips (`ncp` / `nca` / `nci`, not `nr*`), put them in `data/raw/ch2/`, and they get extracted to `data/raw/ch2/<camera>/products/<id>/`. | yes |
| Footprints of every CH-2 product | `python scripts/ch2_footprints.py`, which reads the PRADAN shapefiles. It finds 18 OHRC scenes with TMC-2 + IIRS overlap. | – |
| Elevation maps (LOLA, SLDEM2015) | `python scripts/fetch_dem.py --bbox <minlat maxlat minlon maxlon> --dem lola\|sldem2015 --download` (whole tiles, 0.9–1.4 GB each) | no |
| LRO NAC / WAC | `python scripts/fetch_lro.py --scene <ch2_id> [--product NAC\|WAC] --ids <pdsid>…` | no |
| SELENE TC | `python scripts/fetch_selene.py --scene <ch2_id> --download` | no |

---

## Using what exists (for Members B and C)

### Read any product: same code for every mission

```python
from chandralign.io.pds_label import parse_label          # PDS3 or PDS4, detected by content
from chandralign.io.pds_raster import read_raster, Window

meta = parse_label("data/raw/lro/nac/nac.m1417360906lc/M1417360906LC.XML")
meta.instrument, meta.gsd_m, meta.array_shape              # 'NAC', 0.5, (52224, 5064)
meta.label_fields_verified                                 # which fields genuinely came from the label

pixels = read_raster(meta, Window(row=30000, col=2000, height=512, width=512))   # native dtype, ~25 ms
band   = read_raster(iirs_meta, Window(6000, 0, 256, 250), bands=100)  # iirs_meta = parse_label(<IIRS .xml>); one band
```

### Get tiles ready to match: the Gate A handoff

```python
from chandralign.io.tiling import iter_tiles, read_tile, tile_to_global, TilingStats

stats = TilingStats()
for plane in iter_tiles(meta, tile=1024, overlap=128, stats=stats):   # lazy, ~38 MiB peak for any product
    plane.array          # float32, 0..1
    plane.valid_mask     # True = real measurement (label's no-data / saturation codes excluded)
    plane.shadow_mask    # all False for now; PREP-04 fills it
    plane.tile_origin    # (row, col) of this tile in the full product
    ...
global_pts = tile_to_global(tile_pts, plane.tile_origin)              # (x, y) tile → (x, y) product
```

**Conventions you can rely on:**
- **Points are `(x, y)` = `(column, row)`, as sub-pixel floats. `tile_origin` is `(row, col)`.** Use `tile_to_global` and `global_to_tile`; don't add offsets by hand.
- `valid_mask` and `shadow_mask` are always **separate arrays** (the shared-mask bug is failure mode #19).
- A multi-band product (IIRS) needs `band=`. Matching a whole cube is refused; the IIRS composite (PREP-06) will produce one plane.
- **Tiles come out with a per-tile min/max placeholder** (`"tile_minmax"`). Call `prepare_plane()` (PREP-01) before matching; it replaces that step.

### Sun angles and lighting difference (for the regime selector)

```python
from chandralign.geometry.solar import scene_illumination, illumination_delta

ill = scene_illumination(meta)            # picks up a saved ODE record next to the product
ill.incidence_deg, ill.sun_elevation_deg  # 82.73, 7.27 for our OHRC
ill.sources                               # {'incidence_deg': 'label', 'emission_deg': None, ...}

d = illumination_delta(ohrc_meta, nac_meta)
d['d_incidence_deg'], d['d_azimuth_deg']  # None where either side is unknown -- never 0
d['max_incidence_deg'], d['complete']     # lower sun of the two; False if any angle is missing
```

Every value records where it came from: `"label"`, `"ode_catalogue"` or `None` (unknown). Label values always win; a catalogue value that disagrees with the label by more than 1° is listed in `ill.notes`.

| Scene | Incidence | Sun direction | Emission / phase |
|---|---|---|---|
| CH-2 (all three) | label | label | **unknown** |
| LRO NAC | ODE catalogue | **unknown** | ODE catalogue |
| SELENE TC mosaic | **none** (many passes) | **none** | **none** |

### Do two products show the same ground? (run this before matching)

```python
from chandralign.geometry.footprint import check_overlap, run_if_overlapping

c = check_overlap(src_meta, ref_meta, min_overlap=0.10)
c.ok, c.reason, c.overlap_km2, c.fraction_of_smaller

run_if_overlapping(src_meta, ref_meta, my_matcher)   # raises InsufficientOverlapError, matcher never called
```

Outlines come from label corners (CH-2, SELENE) or the saved ODE record (NAC), recorded in `Footprint.source`. Areas are equal-area km², correct at any latitude. A pair passes when the overlap covers at least `min_overlap` of the **smaller** footprint.

**Overlaps of the pairs we hold** (system-level CH-2 corners):

| Pair | Overlap (share of smaller) | |
|---|---|---|
| OHRC ↔ TMC-2 | 100% | OHRC lies inside the TMC-2 strip |
| OHRC ↔ NAC M102000149RC | 92% | Easy tier: similar sun |
| OHRC ↔ NAC M1417360906LC | 21% | stress pair: sun 75° apart |
| OHRC ↔ SELENE TC (north / south tile) | 45% / 54% | |
| TMC-2 ↔ NAC M1417360906LC | 99% | |
| TMC-2 ↔ SELENE TC (north / south tile) | 22% / 12% | the headline pairing |
| OHRC ↔ IIRS | 27%, but **0% with ISRO's refined corners** | see finding 8 |
| NAC M1417360906LC ↔ SELENE TC north | 0% → rejected | |

### Where is a pixel on the Moon? (and a metric map)

```python
from chandralign.geometry.projection import geolocation_model, scene_crs, to_map, from_map

model = geolocation_model(meta)                 # reference-independent by default
lat, lon = model.pixel_to_latlon(rows, cols)    # 0-based line/sample, floats OK
rows, cols = model.latlon_to_pixel(lat, lon)
model.source, model.independent_of_references

precise = geolocation_model(meta, prefer="precise")   # ISRO's dense grid; may be reference-tuned!
crs = scene_crs(lat0, lon0)                     # IAU 2015 Moon sphere; equirectangular, polar above 60°
x, y = to_map(lat, lon, crs)
```

| Product | Model | Independent of our references? |
|---|---|---|
| CH-2 (default) | 4 system corners, bilinear | ✅, but coarse for long strips (straight edges) |
| CH-2 `prefer="precise"` | ISRO geometry grid (every 100 px; 50 for IIRS) | OHRC ✅ (`System`) · **TMC-2, IIRS ❌ (tuned against SELENE)** |
| SELENE TC | label map projection (exact) | ✅ |
| LRO NAC | **none**: labels have no geometry, needs SPICE | – |

**Never evaluate a registration against SELENE using a model with `independent_of_references == False`.**

### Elevation, slope and aspect

```python
from chandralign.io.dem import dem_patch, find_tiles
from chandralign.geometry.dem_terrain import slope_aspect

patch = dem_patch(find_tiles("data/raw/dem/lola"), bbox=(min_lat, max_lat, min_lon, max_lon))
patch.heights_m                  # metres, north-up; tiles stitched across boundaries
patch.sample(lat, lon)           # bilinear; NaN outside -- never extrapolated
t = slope_aspect(patch)          # t.slope_deg, t.aspect_deg (direction the ground faces, 0 = N)
```

| DEM | Detail | Built from | Independent of our references? |
|---|---|---|---|
| LOLA `ldem_1024` | ~30 m, but **mostly interpolated** between laser tracks, so slopes come out smoothed | laser altimetry | ✅ |
| SLDEM2015 `_512` | ~59 m, with real stereo detail | LOLA **+ SELENE TC stereo** | ❌ (SELENE-derived) |

**Measured over our test region:** the two agree on absolute height (median difference −0.41 m, typical scatter 3.7 m, worst 157 m on steep walls), but SLDEM shows **nearly twice the slope** (median 2.26° vs LOLA's 1.24°) because LOLA's grid is interpolated. So: **SLDEM for slope detail, LOLA whenever independence from SELENE matters.**

Heights are converted to metres from each label's `UNIT` (LOLA stores DN × 0.5 m, SLDEM stores km). The label's `OFFSET` is the Moon's reference radius and is deliberately **not** added. Aspect is NaN on flat ground and slope is NaN wherever the DEM has no height.

### Contrast preparation (run on tiles before matching)

```python
from chandralign.preprocess.radiometric import prepare_plane

ready = prepare_plane(plane)     # new ImagePlane; the input tile is not modified
ready.preprocess_chain           # [..., 'stretch_p1-99', 'clahe_c2_t8x8']
```

A 1st–99th percentile stretch of the valid pixels, then CLAHE in 16 bits (so 16-bit sensors keep their precision). No-data pixels can't affect the result and come back as 0 with the same valid mask. Measured on real mid-strip tiles, local contrast rises ×2.1 (OHRC) to ×4.2 (SELENE), and the spread between cameras narrows from 2.7× to 1.7×. On the darkest OHRC area (sun 7.3° up) it rises ×3.4.

### Shadow mask

```python
from chandralign.preprocess.shadow_mask import with_shadow_mask

plane = with_shadow_mask(read_tile(meta, window))   # run BEFORE prepare_plane (CLAHE is non-linear)
ready = prepare_plane(plane)                         # the shadow mask is carried through
ready.shadow_mask                                    # True = shadow: exclude from matching
```

Per image: the black level (1st percentile) and sunlit level (median) set a threshold 15% of the way up. Dark **and** uniform pixels seed a shadow, and each seed grows over all connected dark pixels, so shadow **edges** (which move with the sun) are included. Dark textured material is left out. On a real OHRC crater crop (sun 7.3° up), 33.9% of pixels are flagged; deep shadow only, not half-lit small crater floors. Against a mask **traced by hand** on that crop: **IoU 0.894**, 96.1% of pixels agree. Both kinds of disagreement trace to hand-drawing limits rather than detector error — pixels only the human marked have median brightness 25 DN (sunlit ground is 38, shadow is 4: brush overspill on a jagged edge), while pixels only the detector marked sit at 5 DN, the sensor's black floor.

### IIRS: 256 bands → one image

```python
from chandralign.preprocess.iirs_composite import product_band_selection, iirs_composite_plane

selection = product_band_selection(iirs_meta)      # score the bands ONCE per product
plane = iirs_composite_plane(iirs_meta, window, selection)
selection.bands, selection.rejected                # which bands, and why the others went
plane.preprocess_chain[0]                          # 'iirs_composite_125of256_snr>=30_refl'
```

Every band is scored from the data (signal = median, noise = robust spread of a diagonal double difference). On our real cube **125 of 256 bands are kept**: 3 are dropped for dead pixels (bands 0–2, 9.7%/4.5%/1.6% dead), 20 for low SNR, and 108 for being past 3,000 nm, where the Moon's own **thermal glow** dominates and the image shows temperature rather than surface appearance. Kept bands are blended with inverse-variance weights.

**Use one selection for the whole product.** If each tile scored its own bands, neighbouring tiles would be built from different data and stop being comparable.

**Two things this is not:** it is not a panchromatic response (pan cameras see 400–800 nm, IIRS starts at 800 — there is no overlap; it is a spatial-structure proxy), and per-band wavelengths are **not in the label** — only the overall 800–5,000 nm range is stated, so band→wavelength is interpolated and marked approximate.

### What is NOT there yet (don't build on it)

| Missing | Why | Arrives with |
|---|---|---|
| **Sun direction (azimuth)** for NAC | Neither the labels nor ODE's product record have it | GEO-02 (SPICE) or the LROC index table |
| Emission and phase angles for **CH-2** | Not in ISRO's labels (sun azimuth and incidence are) | GEO-02 |
| Pixel geolocation for **NAC** | Not in the label (ODE gives only an outline) | SPICE camera model |
| Per-pixel geometry (`plane.geo`) | Not built yet | GEO-02, GEO-03 |
| Phase congruency, MIND, terrain scores | Not built yet | PREP-02, 03, 07, 08 |

---

## Things the real data taught us

Worth knowing before designing anything downstream:

1. **ISRO labels do carry sun angles** (`isda:sun_azimuth`, `sun_elevation`, `solar_incidence`). The PRADAN shapefiles have them as all zeros, so don't use the shapefiles for sun angles.
2. **Real resolution differs from nominal:** OHRC 0.30 m (not 0.25), IIRS 97.15 m (not 80), TMC-2 4.41 m. `meta.gsd_m` holds the label's value; `configs/instruments.yaml` holds the nominal one.
3. **CH-2 "refined" corners were adjusted against SELENE** (`reference_data_used = SELENE`). Using them would leak a reference into its own evaluation, so `corner_latlon` defaults to the **system-level** corners. `read_corner_sets()` returns both.
4. **The SELENE TC ortho map is a mosaic:** no acquisition time, brightness normalised to a standard 30° sun, but shadows still from the original passes. It has no sun angle, and the reader claims none.
5. **The OHRC scene is very dark** (sun 7.3° up). Mid-strip pixels only reach ~83 out of 255, so illumination handling isn't optional.
6. **One OHRC scene gives both an easy pair and a stress pair.** OHRC vs NAC M102000149RC differ by only **3°** in incidence (Easy tier). OHRC vs NAC M1417360906LC differ by **75°** (illumination stress test). Same ground, real data.
7. **OHRC and TMC-2 had the sun on opposite sides** (azimuth 270° vs 104°, 166° apart), so shadows point the opposite way. Expect this pair to be hard for classical matchers.
8. **IIRS's position is uncertain by up to ~13 km.** ISRO's refined corners (adjusted against SELENE) move IIRS 0.435° from the system-level ones; TMC-2 moves 0.176° (~5 km); OHRC doesn't move. For IIRS this changes which pairs overlap at all (OHRC ↔ IIRS: 27% → 0%). OHRC ↔ IIRS goes through the TMC-2 cascade anyway, but don't trust IIRS geolocation to better than ~13 km until GEO-07 cross-checks it.
10. **Real terrain under the OHRC scene** (LOLA, 29.6 m/px, stitched across the equator from two tiles): heights −1,905 to −1,585 m, 320 m of relief, no gaps. Slopes: median 1.2°, 95th percentile 5.4°, max 28°. The tile join is invisible — the height step across it (0.51 m) is smaller than the typical step elsewhere (0.58 m).
11. **Raw keypoint count is a misleading quality measure.** On the IIRS cube the noisiest single band scores the *most* SIFT keypoints (1,413 at noise 0.048) while the quietest composite scores fewer (1,319 at 0.042) — SIFT fires on noise. Measured on pairs with independent noise, where only real features can match, the composite gives **+43%** more genuine matches (1,042 vs 727). PLAN.md P1-T14's keypoint-count acceptance wording is superseded for PREP-06; the change is recorded in FEATURES.csv.
12. **LOLA and SLDEM agree on height but not on slope.** Median height difference −0.41 m (scatter 3.7 m), yet median slope 1.24° (LOLA) vs 2.26° (SLDEM). LOLA's 30 m grid is interpolated between laser tracks, so it understates roughness; SLDEM carries real SELENE stereo detail but is not independent of SELENE. Pick per purpose, and never judge a SELENE registration with SLDEM.
9. **Independent vs refined geolocation, measured over the whole strip:** OHRC 0.2 m, TMC-2 ~5.1 km, IIRS ~13.4 km (median). Four-corner models are also coarse on long strips: straight edges are up to ~1 km off for TMC-2 and ~2.5 km for IIRS. SELENE's map formula, written for 0-based pixel indices, reproduces the label's own corners; it is the standard PDS3 relation, which has a `+1` for 1-based indices. **Dropping that `+1` is off by one pixel** (an earlier version of this README wrongly called our convention a departure from the standard).

---

## Repository map

| Path | What | Owner |
|---|---|---|
| `src/chandralign/contracts.py` | Shared dataclasses. **Frozen**, and a test fails if it drifts from PLAN.md §5. | all |
| `src/chandralign/io/` | Readers, camera registry, tiling (**working**); DEM, ODE client, cache (stubs) | A |
| `src/chandralign/geometry/`, `preprocess/` | Sun geometry, terrain, footprints, projection, image preparation (stubs) | A |
| `src/chandralign/matching/`, `estimate/`, `refine/` | Matching, robust estimation, uniformity, sub-pixel (stubs) | B |
| `src/chandralign/evaluate/` | Metrics (C), control gates and benchmark (B), ground truth (A) (stubs) | shared |
| `src/chandralign/product/`, `viz/`, `cli.py`, `api.py` | Exports, report, figures, CLI, API (stubs) | C |
| `configs/instruments.yaml` | The 7 cameras: values from PLAN.md only, `null` where unstated | A |
| `scripts/` | Data fetchers, footprint index, manifest | A |
| `tests/` | One test file per module; real-label fixtures in `tests/fixtures/labels/` | everyone |

Every stub module's docstring names its owner and feature IDs.

---

## Working rules

- **Never report a number we did not measure.** An unmeasured metric is `None`. `label_fields_verified[f]` is `True` only if the value came from the label. (PLAN.md §2.1)
- **One step, one commit**, with the message starting with the task ID (`P1-T05: …`). Update `FEATURES.csv` status in the same commit.
- **Tests go with code.** `pytest -m ""` must be green before pushing. Tests on multi-GB files are marked `slow`.
- **`contracts.py` changes need all three members** to agree, in writing, in the PR.

---

## Data credits

Chandrayaan-2 data courtesy of ISRO / ISSDC (PRADAN), used for non-profit scientific purposes. LRO LROC data courtesy of NASA / Arizona State University (PDS). SELENE / Kaguya data courtesy of JAXA (DARTS).
