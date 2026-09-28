# 🌕 ChandrAlign — Sub-Pixel Lunar Image Registration for Chandrayaan-2

**ISRO Smart India Hackathon — Problem Statement SIH26166**

ChandrAlign is a research-grade, production-ready Python system that **registers Chandrayaan-2 satellite images** (OHRC, TMC-2, IIRS) onto external lunar reference maps (NASA LRO NAC/WAC, JAXA SELENE TC/MI) with **sub-pixel accuracy**. It handles the hardest unsolved cases in planetary image registration: extreme resolution differences (up to 290×), inverted illumination from opposed sun angles, multi-modal sensors (hyperspectral vs. optical), and metadata that is wrong by kilometres.

This project is backed by a 49-page deep technical research report analysing 100+ academic papers, 25 patents, and 49 competitor repositories. Every pipeline parameter is derived from measurement on real orbital data, not guessed. Every metric that was not measured is reported as `None`, never fabricated.

---

## Table of Contents

- [Why This Exists — The Core Problem](#why-this-exists--the-core-problem)
- [Key Capabilities](#-key-capabilities)
- [System Architecture](#-system-architecture)
- [The Registration Pipeline — Step by Step](#-the-registration-pipeline--step-by-step)
- [The Matching Regimes](#-the-matching-regimes)
- [The Fine Stage — From Raw Matches to Certified Results](#-the-fine-stage--from-raw-matches-to-certified-results)
- [Trust & Safety — Control Gates and Failure Modes](#-trust--safety--control-gates-and-failure-modes)
- [Supported Cameras and Instruments](#-supported-cameras-and-instruments)
- [Validated Pairings and Results](#-validated-pairings-and-results)
- [Installation & Setup](#-installation--setup)
- [Usage — CLI](#-usage--cli)
- [Usage — REST API](#-usage--rest-api)
- [Usage — Web UI (Lunar Mission Control)](#-usage--web-ui-lunar-mission-control)
- [Configuration Reference](#-configuration-reference)
- [Data Handling & Provenance](#-data-handling--provenance)
- [Scientific Findings](#-scientific-findings)
- [Repository Structure](#-repository-structure)
- [Design Principles & Honesty Rules](#-design-principles--honesty-rules)
- [Credits & Data Sources](#-credits--data-sources)

---

## Why This Exists — The Core Problem

Lunar image registration is fundamentally harder than terrestrial image registration because of four compounding challenges:

| Challenge | What Happens | Example from Our Data |
|---|---|---|
| **Extreme resolution gaps** | A feature visible at 0.3 m is invisible at 97 m | OHRC (0.30 m/px) vs IIRS (97.15 m/px) = **290× scale ratio** |
| **Inverted illumination** | The sun moving doesn't weaken brightness matching — it **inverts** it | OHRC sun at 7.3° from west, TMC-2 at 44° from east → raw correlation = **−0.96** |
| **Unreliable metadata** | Labels carry design values, not measurements; positions are off by kilometres | TMC-2 label says 4.41 m/px; measured from pixels = **4.92 m** (11.6% error). IIRS position uncertain by **~13 km** |
| **Multi-modal sensors** | Hyperspectral, panchromatic, and optical sensors see completely different things | IIRS records 256 spectral bands (800–5000 nm); NAC records a single visible band |

**The result:** a naive SIFT or SuperPoint matcher applied to this data doesn't just fail weakly — it returns **confidently wrong** answers. RANSAC finds fits with many "inliers", all of which are false. This project exists to solve that.

---

## 🚀 Key Capabilities

### Sub-Pixel Accuracy
Median registration error of **0.07 pixels** on delivered match points, achieved through geometry-aware least-squares matching (LSM) refinement with Lanczos-scored NCC gating. Every refinement move must *improve* the match score or it is rejected.

### Regime-Routed Matching
The system doesn't use one matcher for everything. A **regime selector** (`matching/regime.py`) examines the sun-angle difference, scale ratio, and sensor modality of each pair *before loading a single pixel*, and routes to the optimal combination of:
- **Representation**: raw pixels, phase congruency, or MIND descriptors
- **Route**: direct match or scale-bridging cascade
- **Matcher**: ELoFTR, XoFTR, SIFT, MIM RIFT, or a fallback chain

### Scale-Bridging Cascade
When the scale gap is too large for a direct match (e.g., OHRC → IIRS at 290×), the system plans a chain through intermediate cameras: OHRC → TMC-2 (16×) → IIRS (16×). Feasibility is decided by **footprint in coarser pixels** (not the raw ratio), and errors propagate in quadrature across steps.

### Illumination Invariance
Two preprocessing descriptors survive the sun moving:
- **Phase Congruency**: Detects structural edges regardless of brightness. Correlation: **+0.86** where raw = −0.96.
- **MIND (Modality Independent Neighbourhood Descriptor)**: Survives contrast inversion. Correlation: **+0.90** where raw = −0.96.

Critically, these are *not always applied*. Below ~60° sun-angle difference, raw brightness is the strongest signal, and both descriptors throw it away. The regime selector knows this.

### Trust & Safety
Five brutal control gates prevent false-positive matches. A result that passes all gates but has too few inliers or too low coverage is graded `REJECTED` with named failure modes. The system will *refuse to answer* rather than give a wrong answer.

### Native PDS4/PDS3 Ingestion
Reads ISRO (PDS4), NASA (PDS3/PDS4), and JAXA (PDS3) data natively — no ISIS3, no GDAL, no SPICE kernels required for the core pipeline. Every label field records whether it genuinely came from the label (`label_fields_verified`).

---

## 🏗️ System Architecture

ChandrAlign is structured as a Python package (`src/chandralign/`) with clearly separated layers:

```
┌────────────────────────────────────────────────────────────────┐
│                     Entry Points                               │
│  CLI (cli.py) ─── REST API (api.py) ─── Web UI (ui/)           │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────────────────────┐
│                   Workflow Layer                               │
│  workflows/products.py → dispatches to validated workflows:    │
│    tmc2_tc.py  │  ohrc_nac.py  │  iirs_wac.py                  │
└──────────────────────┬─────────────────────────────────────────┘
                       │
┌──────────────────────▼────────────────────────────────────────┐
│                    Core Pipeline                              │
│  pipeline.py → fine_stage (terrain filter → MAGSAC → model    │
│  selection → uniform points → sub-pixel refinement)           │
└──────────┬───────────┬──────────┬──────────┬──────────────────┘
           │           │          │          │
    ┌──────▼──┐  ┌─────▼──┐  ┌────▼───┐  ┌───▼────┐
    │ matching│  │estimate│  │ refine │  │evaluate│
    │         │  │        │  │        │  │        │
    │ regime  │  │ MAGSAC │  │subpixel│  │ gates  │
    │ cascade │  │ models │  │uniform.│  │metrics │
    │ adapter │  │ scale  │  │        │  │quality │
    │ rift    │  │geo_filt│  │        │  │probes  │
    │ routing │  │selectn.│  │        │  │failure │
    └─────────┘  └────────┘  └────────┘  └────────┘
           │
    ┌──────▼──────────────────────────────────────────────────────┐
    │                   Foundation Layer                          │
    │  io/       → PDS readers, tiling, DEM, ODE client, camera   │
    │  geometry/ → solar angles, footprint, projection, terrain   │
    │  preprocess/ → CLAHE, shadow mask, IIRS composite, phase    │
    │               congruency, MIND, terrain scores              │
    └─────────────────────────────────────────────────────────────┘
```

### The Integration Contract (`contracts.py`)

All data flows through **frozen dataclasses** defined in `contracts.py`. These are the handoff objects between pipeline stages:

| Dataclass | Flow | Purpose |
|---|---|---|
| `SceneMeta` | Reader → everyone | Everything the label told us about a product |
| `ImagePlane` | Part 1 → Part 2 | One tile, float32, normalised 0..1, with valid/shadow masks |
| `GeometryLayers` | Part 1 → Part 2 | Per-pixel sun/camera angles and DEM |
| `MatchSet` | Part 2 internal | Matched point pairs with confidence |
| `TransformModel` | Pipeline output | Affine, homography, or TPS model |
| `Metrics` | Pipeline → export | Every field is `Optional` — unmeasured = `None`, never fabricated |
| `RegistrationResult` | Part 2 → Part 3 | The final result with tier, gates, failure modes, and notes |

---

## 📋 The Registration Pipeline — Step by Step

### 1. Ingestion
```python
from chandralign.io.pds_label import parse_label
from chandralign.io.pds_raster import read_raster, Window

meta = parse_label("path/to/label.XML")       # PDS3 or PDS4, auto-detected
meta.instrument       # 'OHRC', 'TMC2', 'IIRS', 'NAC', 'WAC', 'TC', 'MI'
meta.gsd_m            # ground sample distance (from the label)
meta.array_shape      # (lines, samples)
meta.corner_latlon    # 4 corners, may be empty for NAC
```
- **PDS4** (Chandrayaan-2): XML labels parsed with `pds4_tools` + `lxml`
- **PDS3** (SELENE, LRO): ODL labels parsed with `pvl`
- **All sensors**: a unified `SceneMeta` output regardless of source

### 2. Pre-Filtering (Before Loading Pixels)
```python
from chandralign.geometry.footprint import check_overlap

c = check_overlap(src_meta, ref_meta, min_overlap=0.10)
# c.ok, c.overlap_km2, c.fraction_of_smaller
```
- **Footprint overlap**: Reject pairs that don't share ground before any matcher runs
- **Scale pre-check**: Refuse impossible scale ratios (OHRC↔IIRS = 290× → cascade required)
- **Illumination delta**: Compute sun-angle difference for regime routing

### 3. Preprocessing (Per Tile)
```python
from chandralign.preprocess.radiometric import prepare_plane
from chandralign.preprocess.shadow_mask import with_shadow_mask

plane = with_shadow_mask(read_tile(meta, window))  # shadow detection
ready = prepare_plane(plane)                        # percentile stretch + CLAHE
```
Each tile goes through:
- **Shadow masking**: Seed-and-grow from dark uniform pixels. IoU **0.894** vs hand-traced ground truth
- **Contrast preparation**: 1st–99th percentile stretch, then 16-bit CLAHE. Local contrast rises 2.1× (OHRC) to 4.2× (SELENE)
- **IIRS composite**: 256 bands → 125 usable bands (3 dead, 20 low-SNR, 108 thermal), inverse-variance blended. **+43%** more genuine matches vs best single band
- **Terrain scores**: Texture score (real structure vs noise) and repetitiveness score (crater fields vs varied terrain)
- **Phase congruency / MIND**: Applied only when the regime selector calls for illumination-invariant descriptors

### 4. Coarse Lock
For pairs with unreliable geolocation (e.g., IIRS with ~13 km position error), a MIND dense template search corrects the gross displacement before fine matching begins. The source is then resampled onto the reference grid at matched scale.

### 5. Fine Matching
The regime selector chooses the matcher:
- **ELoFTR** (default for same-modal)
- **XoFTR** (default for cross-modal, tiled for large images)
- **SIFT** (classical fallback)
- **MIM RIFT** (multi-modal independent method for cross-checking)

### 6. The Fine Stage
See [dedicated section below](#-the-fine-stage--from-raw-matches-to-certified-results).

### 7. Export
- **GeoTIFF**: Source warped onto the reference grid
- **Match points**: GeoJSON with pixel and geographic coordinates
- **HTML report**: Self-contained, shows side-by-side imagery, match overlays, coverage maps, metrics
- **Provenance**: Full record of what ran, on what data, with what checksums

---

## 🎯 The Matching Regimes

The regime selector (`matching/regime.py`) makes three independent decisions before any pixel is read:

| Decision | What it controls | Based on |
|---|---|---|
| **Route** | Direct match or cascade | Footprint in coarser pixels |
| **Representation** | Raw, phase congruency, or MIND | Sun-angle difference |
| **Matcher** | Which algorithm runs | What's been benchmarked for this regime |

### Illumination Bands (Measured on 600 Synthetic Runs)

| Sun Azimuth Δ | Raw Brightness | Phase Congruency | MIND | Best |
|---|---|---|---|---|
| 30° | +0.867 | +0.705 | +0.634 | **Raw wins** |
| 90° | +0.010 | **+0.465** | +0.038 | **PC wins** (12× MIND) |
| 180° | −0.991 | +0.860 | **+0.975** | **MIND wins** |

**Two counterintuitive results:**
1. Preprocessing is **not free**. Below ~60° sun difference, raw brightness is the strongest signal.
2. 90° is the **hardest** case, not 180°. At 180° the scene is close to a contrast inversion (which MIND handles); at 90° lit and shadowed facets swap unpredictably.

### Matcher Difficulty by Illumination

| Sun Δ | Success Rate (Best Licence-Clean Matcher) |
|---|---|
| 0–75° | 10/10 solved |
| 90–120° | 6–8/10 **degraded** (the hard band) |
| 135–180° | 10/10 solved |

---

## ⚙️ The Fine Stage — From Raw Matches to Certified Results

The fine stage (`pipeline.py`) turns raw match points into a certified, audited registration model. Every stage is **independently toggleable** via `configs/default.yaml`:

### Stage 1: Terrain Filter (`geometry_filter`)
Maps each match to the SLDEM digital elevation model and drops matches whose two endpoints land on physically different terrain. This is not a polish — it's a **rescue**:
- Without filter: RANSAC finds **13 inliers, 0 correct**, reports success
- With filter: RANSAC finds **24 inliers, all correct**

### Stage 2: Robust Estimation (`estimate`)
MAGSAC++ with 100,000 max iterations. The estimated scale is checked against the expected GSD ratio from the instrument registry (failure mode #13). At 90% outliers, MAGSAC succeeds 10/10 where a 10,000-cap succeeds only 4/10.

### Stage 3: Model Selection (`model_selection`)
Cross-validates three model families (affine, parallax+DEM, thin-plate spline) on a held-out stratified grid. The model whose check-point RMSE is lowest wins, but a complex model must beat the simpler one by at least 5% (`min_gain: 0.05`).

### Stage 4: Uniform Distribution (`uniformity`)
The Problem Statement mandates uniformly distributed match points. Top 6 matches per cell in an 8×8 grid. Refill searches empty cells with a looser ratio test. The grading tier is **always based on the first robust estimate** (before thinning), so the grade reflects all evidence, not just the delivered subset.

### Stage 5: Sub-Pixel Refinement (`subpixel`)
Geometry-aware LSM (ECC affine) with NCC Gaussian-iterative fallback. The source patch is resampled through the model's local Jacobian. A move is applied **only if it raises the Lanczos-scored NCC** — the system cannot make a point worse.

**Measured result**: median error **0.32 → 0.07 px** on real TMC-2 → TC data.

### Stage 6: TPS Fitting (`tps`)
A thin-plate spline is fit through the delivered points (after sub-pixel refinement) with Huber-reweighted IRLS. On real TMC-2 → TC, held-out residuals drop **−40%** at the median where the affine leaves >1 px.

---

## 🛡️ Trust & Safety — Control Gates and Failure Modes

### The 5 Control Gates

Every registration automatically runs these tests. **All must pass, or the result is REJECTED:**

| Gate | Test | What It Catches |
|---|---|---|
| **CHECK-01** | Match against constant grey | Matcher fires on noise or bias |
| **CHECK-02** | Match against random noise | Same, with high-frequency content |
| **CHECK-03** | Shift source by exactly (3, 4) px; pipeline must recover it within 1.5 px | Pipeline ignores an axis, or the model is degenerate |
| **CHECK-04** | Register image against itself | Identity must give RMSE < 0.25 px, >90% inlier ratio |
| **CHECK-05** | Valid mask ≠ shadow mask | Shared-mask bug (failure mode #19) |

Additionally, an independent **RIFT2 cross-check** is run on every result whose primary matcher is from a different family. If RIFT2 disagrees by >2.0 px, the result is rejected (failure mode #12).

### The 20 Failure Modes

Every `REJECTED` result carries a list of named failure modes, never a blank response:

| # | Mode | Example |
|---|---|---|
| 1 | Featureless terrain | Dark mare with no craters |
| 2 | Repetitive pattern | Crater field where all craters look alike |
| 3 | Severe shadow | Sun at 7.3° — 34% of pixels are in deep shadow |
| 7 | Coarse lock failure | MIND z-score < 10 |
| 10 | No overlap | Footprints don't intersect |
| 12 | Cross-check disagreement | RIFT2 says the model is wrong |
| 13 | Scale mismatch | Estimated scale contradicts instrument registry |
| 19 | Mask corruption | Valid mask and shadow mask are the same array |

---

## 📡 Supported Cameras and Instruments

ChandrAlign supports all 7 cameras across 3 missions:

| Camera | Mission | GSD (Nominal → Measured) | Bands | Pixel Shape | Notes |
|---|---|---|---|---|---|
| **OHRC** | Chandrayaan-2 | 0.25 → **0.30 m** | 1 (panchromatic) | 0.300 × 0.309 m | Highest resolution; sun often very low |
| **TMC-2** | Chandrayaan-2 | 5.0 → **4.41 m** (label) / **4.92 m** (measured) | 1 (panchromatic) | 4.41 × 5.065 m | 812 km long strips; incidence varies 7.8° |
| **IIRS** | Chandrayaan-2 | 80 → **97.15 m** | 256 (hyperspectral, 800–5000 nm) | 97.15 × 79.52 m | 2×2 binned; **22% non-square**; position ±13 km |
| **NAC** | LRO | 0.5 m (nominal) | 1 | ~square | Labels have no geolocation; varies up to 2.33× |
| **WAC** | LRO | 100 m (mosaic) | 7 | ~square | Global mosaic; some products are night-side |
| **TC** | SELENE | 7.40 m | 1 | ~square | Ortho mosaic; no acquisition time; shadows from mixed passes |
| **MI** | SELENE | 14.806 m (not 20 m nominal) | 9 (404–1572 nm) | ~square | 16 MB elevation backplane ahead of image |

---

## 📊 Validated Pairings and Results

Every result below was measured on real orbital data and is reproducible:

| Source → Reference | Scale Ratio | Windows | Success | Median Error | Notes |
|---|---|---|---|---|---|
| **TMC-2 → SELENE TC** | ~1.7× | 18 | **18/18 HIGH** | 0.46 px (TMC-2) | Headline pairing. No published prior attempt. |
| **TMC-2 → LRO NAC** | variable | 45 | **43/45** | — | 177° opposed sun gives no penalty |
| **OHRC → LRO NAC** (control) | ~2× | 5 | **5/5** | 2.1 px (OHRC) | Easy tier: similar sun angles |
| **OHRC → LRO NAC** (opposed sun) | ~2× | varies | ✅ located | — | 55° and 178° sun diff solved via geodetic bridge |
| **OHRC → LRO NAC** (75° incidence) | ~2× | — | ❌ REJECTED | — | **Documented limit.** Coarse lock breaks. Honest refusal. |
| **IIRS → LRO WAC mosaic** | ~1× | 5 | **5/5 HIGH** | 0.09–0.30 px (IIRS) | Hyperspectral to visible; 125-band composite |
| **IIRS → LRO NAC** | ~200× | 3 | **1/3** | — | Dense lock works for large NACs; starves on small ones |
| **OHRC → IIRS** (cascade) | 290× | — | routed | — | Cascade via TMC-2; direct match impossible (37 px sliver) |

---

## 📦 Installation & Setup

### Prerequisites
- Python ≥ 3.11
- GPU (RTX 2080Ti-class or better) recommended for learned matchers (ELoFTR, XoFTR, RoMa)

### Install

```bash
# Clone and create virtual environment
git clone <repo-url>
cd chandrayana2
python -m venv .venv

# Activate
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install with all extras
pip install -e ".[dev,product,api,learned]"
```

### Install Groups

| Extra | Packages | When needed |
|---|---|---|
| (none) | numpy, scipy, opencv, scikit-image, phasepack, lxml, pyproj | Always (core pipeline) |
| `dev` | pytest, pytest-cov | Running tests |
| `product` | rasterio, matplotlib, jinja2 | GeoTIFF export + HTML report |
| `api` | fastapi, uvicorn, httpx | REST API server |
| `learned` | torch, vismatch≤1.3.2 | AI matchers (LightGlue, RoMa, XoFTR) |
| `fast` | pyfftw | ~2× faster phase congruency |

> **⚠️ Licence gate**: `vismatch` is capped at **1.3.2** because later versions may vendor code relicensed from Apache-2.0 to a non-OSI licence. A test fails the build if a newer version is detected.

### Without Data
Every feature works without downloading multi-GB orbital products. The system includes:
- A **synthetic pair generator** (`synth.py`) for offline demonstrations
- **Mock mode** (`--mock`) in both CLI and API
- Pre-computed overlap pairs committed in `data/pairs/`

---

## 💻 Usage — CLI

The `chandralign` command provides four subcommands:

### Register Real Products
```bash
chandralign register \
  --src data/raw/ch2/tmc2/products/ch2_tmc_nca_20250207.xml \
  --ref data/raw/selene/tc/TCO_MAP_02_N03E021N00E024SC.lbl \
  --out runs/tmc2_tc_run1 \
  --windows 3
```
Registers 3 windows across the overlap using the validated TMC-2 → TC workflow. The matcher is chosen by the routing, not by the user. Unsupported pairings are **refused with the reason**.

### Register a Synthetic Pair (Offline Demo)
```bash
chandralign register --mock --out runs/mock_test --seed 7
```

### Generate an HTML Report
```bash
chandralign report --run runs/tmc2_tc_run1
# → runs/tmc2_tc_run1/report.html
```

### Full Offline Demo (Synthetic + Report)
```bash
chandralign demo --out runs/demo
```

### Fetch a WAC Reference for IIRS
```bash
chandralign fetch-wac-clip --iirs path/to/iirs_label.xml
# Downloads only the needed part of the LROC WAC global 100m mosaic
```

---

## 🌐 Usage — REST API

Start the FastAPI backend:

```bash
uvicorn chandralign.api:app --reload
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/register` | Queue a registration job (runs in background) |
| `GET` | `/runs/{run_id}` | Job status, progress log, metrics, assets |
| `GET` | `/runs/{run_id}/assets/{name}` | Download a file produced by the run |
| `GET` | `/pairs` | List curated benchmark pairs with evidence |

### Register a Curated Pair

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"pair_id": "tmc2_selene_tc", "windows": 3}'
```

### Register with Custom Labels

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"src": "path/to/source.xml", "ref": "path/to/reference.lbl"}'
```

### Run a Synthetic Pair via API

```bash
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"mock": true, "seed": 42}'
```

### Poll Job Status

```bash
curl http://localhost:8000/runs/<run_id>
```

Returns: `QUEUED` → `RUNNING` (with live progress log) → `DONE` or `FAILED`.

### Curated Pairs Available

| `pair_id` | Description |
|---|---|
| `tmc2_selene_tc` | TMC-2 → SELENE TC (headline pairing) |
| `ohrc_nac` | OHRC → LRO NAC M102014464RC (credibility pairing) |
| `ohrc_nac_opposed_sun` | OHRC → NAC M175124932LC (opposed sun, finest NAC) |
| `ohrc_nac_75deg_sun` | OHRC → NAC M1417360906LC (75° incidence — **expect REJECTED**) |
| `iirs_wac` | IIRS → LRO WAC global mosaic (hyperspectral source) |

---

## 🖥️ Usage — Web UI (Lunar Mission Control)

ChandrAlign includes a **4-quadrant web console** called "Lunar Mission Control" (`src/chandralign/ui/`). It's a self-contained HTML/CSS/JS application served by the FastAPI backend.

### Launch

```bash
uvicorn chandralign.api:app --reload
# Open http://localhost:8000 in your browser
```

### Features
- **Pair selector**: Pick from the curated benchmark pairs or enter custom labels
- **Live progress**: Watch the registration run in real-time with timestamped logs
- **4-quadrant view**: Source image, reference image, match overlay, and coverage heatmap
- **Metrics panel**: Confidence tier (colour-coded), inlier count/ratio, RMSE, coverage, runtime
- **Gate results**: Pass/fail status of all 5 control gates
- **Failure modes**: Named reasons when a result is REJECTED
- **Asset downloads**: GeoTIFF, match points, provenance JSON

> The UI is **strictly honest**: a test (`test_ui_honesty.py`) fails the build if the HTML/JS contains hard-coded metric values or random number generation (`Math.random`). Every number displayed comes from the actual backend response.

---

## ⚙️ Configuration Reference

All system behaviour is governed by YAML files in `configs/`:

### `configs/default.yaml` — Pipeline Defaults

| Section | Key Settings | Purpose |
|---|---|---|
| `ship_mode` | `true` | Licence gate: blocks non-OSI matchers in production |
| `device` | `auto` | GPU selection: `auto`, `cpu`, `cuda` |
| `matching.max_num_keypoints` | `2048` | Cap for sparse matchers |
| `matching.ratio_test` | `0.80` | Lowe ratio test (tightened to 0.70 on repetitive terrain) |
| `matching.mutual_nn` | `true` | Mutual nearest neighbour filtering |
| `estimate.method` | `magsac` | Robust estimator: `magsac`, `ransac`, `usac_accurate` |
| `estimate.max_iters` | `100000` | MAGSAC iteration cap (adaptive termination) |
| `estimate.scale_tolerance` | `0.25` | Reject transforms whose scale disagrees with GSD ratio |
| `pipeline.geometry_filter` | `true` | Terrain-based outlier filter |
| `pipeline.uniformity` | `true` | Uniform grid distribution of match points |
| `pipeline.subpixel` | `true` | Per-point sub-pixel refinement |
| `pipeline.tps` | `true` | Thin-plate spline fit |
| `pipeline.model_selection` | `true` | Cross-validated model choice |
| `pipeline.parallax` | `false` | DEM parallax correction (ON only for TMC-2→TC) |
| `subpixel.method` | `lsm` | LSM with ECC affine + NCC Gaussian fallback |
| `subpixel.max_move_px` | `1.5` | Maximum refinement shift per point |
| `tiers.low.min_inliers` | `8` | **Accept/reject boundary** (fitted from 60 measured runs) |
| `tiers.low.min_inlier_ratio` | `0.325` | Must be met simultaneously with inlier count |
| `tiers.low.min_coverage` | `0.15` | Fraction of grid cells with a match |
| `gates.perturbation_shift_xy` | `[3, 4]` | The known shift used in CHECK-03 |
| `gates.perturbation_tolerance_px` | `1.5` | How close the pipeline must recover it |
| `gates.crosscheck` | `true` | Run independent RIFT2 cross-check |
| `cascade.min_footprint_px` | `112` | Minimum footprint for a cascade step (measured) |

### `configs/instruments.yaml` — Camera Registry

Nominal specifications for all 7 cameras (OHRC, TMC-2, IIRS, NAC, WAC, TC, MI). Values from the PLAN document only; `null` where unstated.

### `configs/regimes.yaml` — Matcher Routing

Defines the illumination band thresholds and matcher preferences for each regime. Based on 600 measured runs across 10 seeds and multiple sun geometries.

### `configs/measured_scales.yaml` — Verified Pixel Sizes

Per-axis pixel sizes verified from two independent sources (labels + measurement). Unverified values are marked and carry a 3× slack tolerance.

---

## 📂 Data Handling & Provenance

### Images Are Never Committed
All orbital data lives in `data/raw/` which is **gitignored**. `data/manifest.json` records the SHA-256 of every file so results trace back to exact bytes.

### Getting Data

| Source | Command | Login |
|---|---|---|
| Chandrayaan-2 (PRADAN) | Manual download (SSO can't be scripted) | Yes |
| LRO NAC/WAC | `chandralign fetch-wac-clip --iirs <label>` for WAC mosaic clips | No |
| SELENE TC/MI | Via JAXA DARTS | No |
| Elevation (LOLA, SLDEM) | Via NASA PDS | No |

### Provenance Tracking
Every run records:
- Input file checksums (SHA-256)
- Matcher used, device, configuration snapshot
- ML framework versions (PyTorch, vismatch)
- Git commit hash and dirty-tree status
- Wall-clock runtime per stage

---

## 🔬 Scientific Findings

During development, 29 critical findings about real lunar data were documented. Key highlights:

1. **The sun moving inverts brightness matching, not just weakening it.** Raw brightness correlation on real relief under our two sun geometries: **−0.96**. A brightness-based matcher is *actively misled*, not merely uninformed.

2. **RANSAC can hallucinate confidently wrong answers.** On a 3.7%-precision match pool over real relief, RANSAC found 13 inliers (0 correct) and reported success. The terrain filter rescued the fit to 24 inliers (all correct).

3. **ISRO labels carry design values, not measurements.** TMC-2's GSD is 4.41 m in the label but **4.92 m when measured from pixels** — an 11.6% error. The label computes GSD as `design_GSD × altitude / 100km`, which is circular with the optics check.

4. **Non-square pixels matter.** IIRS pixels are 97.15 × 79.52 m — a **22% scale error in one axis** if assumed square. No matcher recovers from that.

5. **The noisiest IIRS band scores the *most* SIFT keypoints.** Classical detectors fire on noise. The 125-band composite gives **+43%** more genuine matches despite fewer raw keypoints.

6. **90° sun difference is harder than 180°.** At 180° the scene approaches a contrast inversion (MIND handles this). At 90° lit and shadowed facets swap unpredictably — the worst case for all methods.

7. **One sun angle per product is honest for OHRC and wrong for the others.** Incidence varies 0.14° across OHRC (25 km), but **7.80° across TMC-2** (812 km) and **8.30° across IIRS** (1,042 km).

8. **OHRC's label declares the wrong unit on its line period.** Labelled as milliseconds, it's actually microseconds — a factor of **1,009×** error.

9. **LOLA and SLDEM agree on height but not slope.** Median height difference −0.41 m, but LOLA's median slope is 1.24° vs SLDEM's 2.26° because LOLA is interpolated between laser tracks.

10. **Preprocessing is not always better.** Below ~60° sun difference, raw brightness is the strongest signal. Phase congruency and MIND both throw it away. Always preprocessing is worse than never.

---

## 📁 Repository Structure

```
chandrayana2/
├── src/chandralign/                  # Python package (88 files)
│   ├── __init__.py                   # Package init
│   ├── contracts.py                  # FROZEN integration contract
│   ├── pipeline.py                   # Core fine-stage pipeline runner
│   ├── api.py                        # FastAPI REST backend
│   ├── cli.py                        # CLI entry point
│   ├── config.py                     # YAML config loader
│   ├── compute.py                    # Device selection (CPU/CUDA)
│   ├── synth.py                      # Synthetic pair generator
│   │
│   ├── io/                           # Data Layer
│   │   ├── pds_label.py              #   PDS3 + PDS4 label reader
│   │   ├── pds_raster.py             #   Windowed pixel reader
│   │   ├── tiling.py                 #   Tile → ImagePlane
│   │   ├── instruments.py            #   7-camera registry
│   │   ├── dem.py                    #   LOLA/SLDEM elevation
│   │   ├── dem_subtile.py            #   DEM sub-tile cutting
│   │   ├── ode_client.py             #   NASA ODE product search
│   │   ├── reference.py              #   Reference product search
│   │   ├── wac_mosaic.py             #   LROC WAC global mosaic
│   │   ├── evidence.py               #   Evidence file reader
│   │   └── cache.py                  #   Label layout cache
│   │
│   ├── geometry/                     # Physics Layer
│   │   ├── solar.py                  #   Sun/camera angles
│   │   ├── footprint.py              #   Footprint overlap
│   │   ├── projection.py             #   IAU 2015 Moon CRS
│   │   ├── dem_terrain.py            #   Slope/aspect from DEM
│   │   ├── geoprior.py               #   Geometric prior
│   │   ├── nac.py                    #   NAC geometry helpers
│   │   └── rpc.py                    #   RPC model export
│   │
│   ├── preprocess/                   # Image Preparation
│   │   ├── radiometric.py            #   Percentile stretch + CLAHE
│   │   ├── shadow_mask.py            #   Shadow detection
│   │   ├── iirs_composite.py         #   256 bands → 1 image
│   │   ├── phase_congruency.py       #   Illumination-invariant edges
│   │   ├── texture.py                #   Terrain matchability scores
│   │   └── resample.py               #   Resampling utilities
│   │
│   ├── matching/                     # Matching Engine
│   │   ├── regime.py                 #   Automatic matcher routing
│   │   ├── cascade.py                #   Scale-bridging cascade
│   │   ├── adapter.py                #   vismatch adapter
│   │   ├── classical.py              #   SIFT / ORB fallback
│   │   ├── rift.py                   #   MIM RIFT
│   │   ├── filters.py                #   Match filtering
│   │   ├── routing.py                #   Pairing → route
│   │   ├── similarity.py             #   Similarity transforms
│   │   ├── rotation.py               #   Rotation search
│   │   ├── pair_tiling.py            #   Tile pairing
│   │   ├── licence.py                #   Licence gate (CHECK-10)
│   │   └── craters.py                #   Crater matching (stub)
│   │
│   ├── estimate/                     # Estimation
│   │   ├── models.py                 #   Affine, homography, TPS
│   │   ├── robust.py                 #   MAGSAC/RANSAC wrapper
│   │   ├── geometry_filter.py        #   Terrain-based outlier filter
│   │   ├── scale.py                  #   Pixel scale estimation
│   │   └── selection.py              #   Model selection (CV)
│   │
│   ├── refine/                       # Refinement
│   │   ├── subpixel.py               #   Per-point NCC/LSM refinement
│   │   └── uniformity.py             #   Uniform grid distribution
│   │
│   ├── evaluate/                     # Evaluation & Quality
│   │   ├── control_gates.py          #   5 control gates
│   │   ├── metrics.py                #   Accuracy metrics
│   │   ├── groundtruth.py            #   Ground truth methods
│   │   ├── quality.py                #   Quality/uniformity scoring
│   │   ├── failure_log.py            #   20 failure modes
│   │   ├── probes.py                 #   Diagnostic probes
│   │   ├── selftest.py               #   Self-test utilities
│   │   ├── source_px.py              #   Source-pixel accuracy
│   │   ├── run_record.py             #   Run record dataclass
│   │   ├── ablation.py               #   Ablation study support
│   │   └── benchmark.py              #   Benchmark harness
│   │
│   ├── product/                      # Output & Export
│   │   ├── matchpoints.py            #   Match point GeoJSON
│   │   ├── warp.py                   #   Warped GeoTIFF
│   │   ├── report.py                 #   HTML report generation
│   │   ├── run_export.py             #   Run directory writer
│   │   └── provenance.py             #   Provenance tracking
│   │
│   ├── workflows/                    # Validated End-to-End Workflows
│   │   ├── products.py               #   Product registration dispatcher
│   │   ├── tmc2_tc.py                #   TMC-2 → SELENE TC
│   │   ├── ohrc_nac.py               #   OHRC → LRO NAC
│   │   └── iirs_wac.py               #   IIRS → LRO WAC mosaic
│   │
│   ├── viz/                          # Visualizations
│   │   ├── match_plot.py             #   Match overlay plots
│   │   ├── coverage_plot.py          #   Coverage heatmaps
│   │   ├── sidebyside.py             #   Side-by-side comparison
│   │   └── swipe.py                  #   Swipe comparison
│   │
│   └── ui/                           # Web UI
│       ├── index.html                #   Lunar Mission Control console
│       ├── style.css                 #   Styling
│       └── app.js                    #   Frontend logic
│
├── configs/                          # Configuration (4 files)
│   ├── default.yaml                  #   All pipeline settings
│   ├── instruments.yaml              #   7-camera specifications
│   ├── regimes.yaml                  #   Matcher routing rules
│   └── measured_scales.yaml          #   Verified pixel sizes
│
├── data/                             # Data (gitignored except metadata)
│   ├── manifest.json                 #   SHA-256 of every data file
│   └── pairs/                        #   Pre-computed overlap pairs
│
├── .github/workflows/                # CI/CD
│   ├── ci.yml                        #   Core test suite
│   └── part3.yml                     #   Part 3 integration tests
│
├── FEATURES.csv                      # 85 features with status
├── PLAN.md                           # Architecture & task plan
├── pyproject.toml                    # Build & dependency config
└── README.md                         # This file
```

---

## 🏛️ Design Principles & Honesty Rules

This project follows strict rules to prevent the common failure mode in academic/hackathon image registration: **reporting fabricated accuracy**.

| Rule | What It Means |
|---|---|
| **H1: Never report an unmeasured number** | `Metrics.rmse_px` is `Optional[float]`. If it wasn't measured, it's `None`. |
| **H3: REJECTED must say why** | Every rejection carries a `failure_modes` list and `notes`. Never a blank response. |
| **H4: Gates must be non-empty** | A result without control gate results is not considered valid. |
| **H5: Synthetic data is tagged** | `MetricSource = "synthetic"` — never presented as a real-data result. |
| **Labels that disagree are reported, not silently corrected** | OHRC's wrong unit (ms vs μs, factor 1009×) stays visible in `line_period_s`. |
| **Tuned corners are not used for evaluation** | CH-2 "refined" corners were tuned against SELENE — using them to evaluate against SELENE would be circular. |
| **Every experiment is pre-registered** | The acceptance criterion is frozen before the measurement exists. |

---

## 📜 Credits & Data Sources

- **Chandrayaan-2 data**: ISRO / ISSDC (PRADAN), used for non-profit scientific purposes
- **LRO LROC data**: NASA / Arizona State University (PDS)
- **SELENE / Kaguya data**: JAXA (DARTS)
- **Elevation**: LOLA (NASA Goddard), SLDEM2015 (LOLA + SELENE stereo)

---

## 📄 Licence

Core components use permissive licences (Apache-2.0, MIT, BSD). Optional AI matchers (`vismatch`) are capped at version 1.3.2 under `ship_mode: true` to enforce licence compliance. The licence gate (`CHECK-10`) blocks construction of any matcher from the `benchmark_only` list in `regimes.yaml` when `ship_mode` is active.
