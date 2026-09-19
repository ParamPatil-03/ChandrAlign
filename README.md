# CHANDRALIGN

**SIH26166 (ISRO):** register Chandrayaan-2 images (OHRC, TMC-2, IIRS) onto external lunar references (LRO NAC/WAC, SELENE TC/MI) with sub-pixel accuracy, despite different sun angles, resolutions and sensor types.

- **Plan:** [`PLAN.md`](PLAN.md) is the single source of truth: architecture, tasks, honesty rules.
- **Features:** [`FEATURES.csv`](FEATURES.csv) lists all 85 features with owner, task, status and "done when".
- **Plain-language guide:** [`docs/FEATURE_GUIDE.html`](docs/FEATURE_GUIDE.html).

> This README is updated after every completed step. Last update: **Step 9, sun angles and lighting difference.**

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
| next | Footprint overlap, projection, elevation maps, slope/aspect, preprocessing | GEO-04, GEO-05, DATA-13, GEO-03, PREP-* | ⏳ |

**Tests:** 144 passing (`pytest -m ""`), including checks on every real product we hold.

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
- **Normalisation is a per-tile min/max placeholder**, recorded as `"tile_minmax"` in `preprocess_chain`. Real radiometric preparation (CLAHE and more) arrives with PREP-01.

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

### What is NOT there yet (don't build on it)

| Missing | Why | Arrives with |
|---|---|---|
| **Sun direction (azimuth)** for NAC | Neither the labels nor ODE's product record have it | GEO-02 (SPICE) or the LROC index table |
| Emission and phase angles for **CH-2** | Not in ISRO's labels (sun azimuth and incidence are) | GEO-02 |
| Corner lat/lon for **NAC** | Not in the label | GEO-01 / ODE footprint |
| Per-pixel geometry (`plane.geo`) | Not built yet | GEO-02, GEO-03 |
| Elevation maps, slope, aspect | Not fetched yet | DATA-13, GEO-03 |
| Shadow mask, CLAHE, phase congruency, IIRS composite | Not built yet | PREP-01…08 |

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
