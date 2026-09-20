# PLAN.md — CHANDRALIGN

**SIH26166 — Multi-modal, Sun-angle and scale-invariant image correspondence using Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)**
Organisation: ISRO / Department of Space · Theme: Space Technology · Category: Software

> This file is the **single source of execution truth** for Claude Code and for all 3 team members.
> Companion file: `FEATURES.csv` — the feature register: 85 capabilities in 9 sections, each with a plain-English description, owner, phase, task ID, code location and acceptance test (see §21).

---

## 0. HOW TO USE THIS FILE (read this first)

### 0.1 For Claude Code

When you are asked to work on this repo:

1. Read `PLAN.md` (this file) **and** `FEATURES.csv` before writing any code.
2. Find the task ID you were given (e.g. `P1-T04`). Work **only** inside that task's declared `module path`. Do not touch another member's modules.
3. Every task below states: **Goal → Inputs → Outputs → Files to create → Steps → Acceptance criteria**. A task is done only when the acceptance criteria run green in `pytest`.
4. Update `FEATURES.csv` `status` column (`TODO` → `WIP` → `DONE` → `VERIFIED`) as you go. Never delete rows.
5. **Never fabricate a number.** If a metric is not measured, write `null`, not a plausible-looking value. See §2 Ground Rules — this is a hard rule with a test that enforces it.
6. If a task is blocked by another member's module, stub **only** against the frozen contract in §5, mark it `BLOCKED` in the CSV, and move on.

### 0.2 For the 3 humans

| | Owner | Part | One-line scope |
|---|---|---|---|
| **Member A** | Data & Geometry Engineer | **Part 1** | Raw PDS bytes → clean, geometry-annotated, preprocessed tile pairs |
| **Member B** | Matching & Registration Engineer | **Part 2** | Tile pairs → match points → transform → registered product → metrics |
| **Member C** | Product, Evidence & Demo Engineer | **Part 3** | Two features only: (1) Deliverable export + evaluation report, (2) Demo UI |

**Part 2 is strictly downstream of Part 1.** Part 3 is deliberately light and can start on mock data from Day 1.

---

## 1. WHAT WE ARE BUILDING, AND WHY THIS EXACT THING

### 1.1 The problem statement, decoded

The PS's own **Dataset** field is the decisive evidence for scoping:

> "Chandrayaan-2 orbiter optical payload: OHRC, TMC-2, IIRS lunar images … **Reference:** LRO NAC Images … SELENE Images"

So:

- **Source (moving)** = one Chandrayaan-2 instrument image — OHRC, TMC-2 or IIRS.
- **Reference (fixed)** = an **external** lunar reference — LRO NAC (named), LRO WAC (reachable via the linked LROC QuickMap), or SELENE/Kaguya.
- Internal CH-2 cross-instrument matching (OHRC↔TMC-2↔IIRS) is a **useful engineering technique and a bonus capability**, but it is *not* the PS's stated primary target — zero dataset links are given for it.

**Deliverables the PS explicitly names:** software + registered product + corresponding match points + evaluation metrics (RMSE, inlier match count, inlier ratio), at **sub-pixel accuracy**, with **uniform match-point distribution** across the image.

### 1.2 Instrument facts (hard-code these into `configs/instruments.yaml`)

| Instrument | Type | GSD | Swath | Spectral |
|---|---|---|---|---|
| **OHRC** | Panchromatic pushbroom, TDI CCD | 0.25 m/px | narrow strip | 500–800 nm |
| **TMC-2** | Panchromatic pushbroom, 3 look angles (stereo) | ~5 m/px (4.43 m/px on a real product label) | 20 km | panchromatic |
| **IIRS** | Hyperspectral pushbroom | ~80 m/px | 20 km | 0.8–5.0 µm, ~250–256 bands |
| **LRO NAC** | Panchromatic | 0.5 m/px | — | — |
| **LRO WAC** | Multiband | ~100 m/px | — | — |
| **SELENE TC** | Panchromatic stereo | ~7.4–10 m/px | — | — |
| **SELENE MI** | Multiband imager | ~20–62 m/px | — | — |

**Scale ratios that drive the entire architecture:**

| Pair | Ratio | Status in the literature |
|---|---|---|
| OHRC ↔ LRO NAC | ~2:1 | **SOLVED** (2 independent sources) |
| IIRS ↔ LRO WAC | ~1.25:1 | **SOLVED** (2 independent sources) |
| TMC-2 ↔ SELENE TC | ~2:1 | **OPEN — zero prior art, despite being the best-matched pair in the matrix** |
| TMC-2 ↔ LRO NAC | ~10:1 | **OPEN** — best competitor measured only 2.8% inlier ratio |
| IIRS ↔ LRO NAC | ~160:1 | **OPEN** |
| OHRC ↔ LRO WAC | ~333:1 | **OPEN** (nobody has published an attempt) |
| OHRC ↔ IIRS | ~320:1 | **KNOWN-HARD** — two independent teams measured 0 inliers |

### 1.3 Our defensible claim (say exactly this, nothing more)

> A **generic, instrument-agnostic** registration pipeline that registers **any** Chandrayaan-2 instrument against **either** named external reference, validated honestly on the **two neglected axes — TMC-2 and SELENE** — while **reproducing** the two already-solved pairs (OHRC↔NAC, IIRS↔WAC) as a credibility floor.

We do **not** claim a new matching algorithm. Nothing in a hackathon timeline out-innovates SuperGlue / LightGlue / LoFTR / RIFT at the algorithm level, and claiming so is the fastest way to lose a judge. What is genuinely new is the **combination + the validation focus + the honesty discipline**:

1. A **real, tested PDS3/PDS4 ingestion layer** — the single most commonly claimed-but-stubbed component across the ~49 competing repos found.
2. **Genuine coverage of TMC-2 and SELENE** with real downloaded products, not synthetic stand-ins.
3. A **regime-selected, geometry-guided** combination of existing methods (classical RIFT-family where cross-modal radiometry dominates; SuperGlue/LightGlue-class where illumination is extreme; **staged coarse-to-fine bridging** across the ~320:1 scale gap).
4. **Control-gated, honest validation** — confidence tiers, null tests, and published failure reporting.

### 1.4 What we deliberately do NOT build

- ❌ A third independent attempt at **direct OHRC↔IIRS matching without staged bridging**. Two teams already burned real effort failing at exactly this.
- ❌ A "lunar fine-tuned" model trained for 1 epoch and marketed as fine-tuned.
- ❌ SuperPoint / SuperGlue / R2D2 **in the shipped deliverable** (non-commercial licences — see §2.3).
- ❌ Vanilla LoFTR, unmodified, for extreme-illumination cases.
- ❌ A beautiful README over thin code. The graded competitor field is full of these.

---

## 2. GROUND RULES (non-negotiable, enforced by tests)

### 2.1 The honesty rules

| # | Rule | Enforcement |
|---|---|---|
| H1 | **No fabricated metrics.** Unmeasured = `null`, never a plausible number. | `tests/test_honesty.py::test_no_hardcoded_metrics` greps for numeric literals in metric return paths |
| H2 | **Every metric must be reproducible** by re-running one documented command. | `scripts/reproduce.sh` must regenerate every number in the report |
| H3 | **Failures are reported, not hidden.** A pipeline that fails emits `confidence_tier="REJECTED"` + reason, never a silent fallback. | `test_quality_gate_rejects` |
| H4 | **Control gates run on every benchmark execution** (null test + perturbation test). A run without gates is not a valid result. | `Metrics.gates` dict must be non-empty or the report refuses to render |
| H5 | **Cited numbers are labelled as cited.** External baselines from published papers appear in the report tagged `source="external"`, never as our own measurement. | `report.py` requires a `source` field on every number |
| H6 | **Training depth is stated honestly.** If we fine-tune, log epochs + loss curves. If we do not fine-tune, say "pretrained weights, no fine-tuning". | `docs/REPRODUCIBILITY.md` |

> **Why this matters concretely:** one of the strongest competitor repos found and documented its *own* bug that produced fake 100% inlier rates from a shared zero-mask. That bug class is our failure mode #19. We design the control gate in from day one instead of catching it at demo time.

### 2.2 The "no stub" rule

The most repeated finding across the competitor field: PDS4 parsers that silently fall back to `cv2.imread()` / generic TIFF loading. We enforce the opposite:

```python
# tests/test_pds_real.py
def test_parser_reads_real_label_fields():
    meta = parse_pds4(REAL_OHRC_XML)
    assert meta.label_fields_verified["gsd_m"] is True      # came from the label
    assert meta.label_fields_verified["corner_latlon"] is True
    assert meta.gsd_m == pytest.approx(0.25, rel=0.2)
    assert meta.array_shape[0] > 10_000                     # real strip, not a thumbnail
```

A parser that loads pixels but not label fields **fails the test**.

### 2.3 The licence rule

| Component | Licence | Usable in shipped deliverable? |
|---|---|---|
| PyTorch, OpenCV, GDAL, rasterio, **kornia** | BSD / MIT / **Apache-2.0** | ✅ yes |
| **LightGlue** (incl. weights) | Apache-2.0 | ✅ **yes — this is our default matcher** |
| RoMa (MIT), XFeat (Apache-2.0), D2-Net (Clear BSD), ALIKE/ALIKED (BSD-3) | permissive | ✅ yes |
| AROSICS, AFIDS-POMM, ISIS3, VICAR | Apache-2.0 / public-domain-style | ✅ yes (as comparison baselines) |
| **SuperPoint / SuperGlue** | Magic Leap, **non-commercial research only** | ⚠️ **internal benchmark ONLY**, behind a licence flag, never in the shipped path |
| **R2D2** | CC BY-NC-SA 3.0 (share-alike) | ❌ **do not use at all** — most restrictive in the stack |

**Implementation — note this changed when we adopted `vismatch`.** `vismatch` bundles 50+ matchers in one package, SuperPoint and SuperGlue among them, so the gate can no longer sit at import level. It sits at **model-name level instead**:

```python
# matching/licence.py
NON_COMMERCIAL = {"superpoint", "superglue", "r2d2", "sp-sg", "superpoint-superglue"}

def guard(model_name: str, ship_mode: bool) -> None:
    if ship_mode and any(t in model_name.lower() for t in NON_COMMERCIAL):
        raise LicenceRestrictedError(
            f"{model_name} is non-commercial; unavailable while ship_mode is true"
        )
```

`configs/default.yaml` has `ship_mode: true` by default, and every `get_matcher()` call routes through `guard()`. Only `scripts/bench_external.py` sets `ship_mode: false`; its outputs go to `reports/external_benchmark/`, which is **excluded from the deliverable bundle**. Same protection as before, enforced one layer lower.

### 2.4 Git discipline

- `main` is always green. Nobody pushes to `main` directly.
- Branches: `part1/<task-id>-slug`, `part2/<task-id>-slug`, `part3/<task-id>-slug`.
- One PR per task ID. PR title = task ID + goal.
- `src/chandralign/contracts.py` is **frozen after Phase 0**. Changing it requires all 3 members to agree, in writing, in the PR description.

---

## 3. TECH STACK

### 3.0 Build-vs-reuse policy *(read before writing any module)*

**Default: reuse. Build only what does not exist.** A hackathon's scarcest resource is attention, and every hour spent re-implementing a solved problem is an hour not spent on TMC-2 and SELENE, which is the only place we can actually win. Judges do not award points for hand-rolling an XML parser.

Before starting any task, the question is not "how do I build this?" but **"has someone already built this, and is it permissively licensed?"** If yes, integrate it and spend the saved time on validation. If it exists but is fragile or unmaintained, wrap it behind our own interface so we can swap it out. Only build from scratch when nothing exists.

This does **not** weaken our differentiation claim. The research finding was never "competitors fail to write parsers" — it was that competitors ship parsers **that do not actually parse**. Proving a real, maintained library works on real ISRO products is the same differentiator at a fraction of the cost, and it is more defensible, not less.

### 3.1 What we reuse *(all verified to exist and be permissively licensed)*

| Need | Library we use | Licence | What it saves us |
|---|---|---|---|
| **PDS4 label + array read** | **`pds4_tools`** (NASA PDS Small Bodies Node) — `read()` returns a StructureList, each Structure carrying `.data` and `.meta_data`, with lazy-loading for large arrays | open (NASA) | the whole namespace-aware XML parser |
| **PDS3 / ODL labels** | **`pvl`** — PlanetaryPy affiliate; decodes PVL, ODL, PDS3 labels **and ISIS cube labels** | BSD | the whole ODL keyword parser |
| **PDS3 pixel read** | `planetaryimage.PDS3Image` where it works (handles `^PTR` pointer forms, `SAMPLE_TYPE`→dtype, `start_byte`) — **but see the caveat in §3.2** | BSD | most of the raw reader |
| **IIRS hyperspectral cubes** | **`spectral`** (SPy) — pure-Python ENVI reader covering BSQ/BIL/BIP + wavelength metadata. IIRS ships `.hdr`, which is ENVI. GDAL as fallback | MIT | the interleave handling |
| **Phase congruency / Log-Gabor** | **`phasepack`** — `phasecong`, `phasecongmono`, `phasesym`; phase-based detection explicitly invariant to brightness and contrast | open | the entire filter-bank implementation |
| **Feature matching (all of it)** | **`vismatch`** (formerly image-matching-models) — unified API over 50+ matchers, weights auto-downloaded, one `get_matcher(name)` call returning `num_inliers`, `H`, all/matched/inlier keypoints, plus `plot_matches` | BSD-3 | the matcher interface **and** every per-matcher adapter |
| **RIFT cross-modal matcher** | **`RIFT2-python`** — a Python implementation of RIFT2 built on PhasePack (vendored, see §3.2) | open | the MATLAB→Python port the research said we'd need |
| **Robust estimation** | OpenCV `USAC_MAGSAC` / `USAC_ACCURATE` | Apache-2.0 | MAGSAC from scratch |
| **Sub-pixel shift** | `skimage.registration.phase_cross_correlation(upsample_factor=…)`; `cv2.cornerSubPix` | BSD | two of the four sub-pixel methods |
| **Tiling / windowed reads** | `rasterio` `Window`, `WarpedVRT` | MIT | memory-safe reads on 191k×4k products |
| **Slope / aspect** | `gdaldem slope` / `gdaldem aspect` (or `richdem`) | MIT | the terrain maths |
| **Reprojection / lunar CRS** | `rasterio` + `pyproj`; `shapely` for footprints | MIT/BSD | projection plumbing |
| **Baseline comparison** | **AROSICS** — purpose-built sub-pixel co-registration, named in the PS's own competitor context | Apache-2.0 | a credible external baseline for free |
| **Solar geometry (primary)** | **`spiceypy`** + public NAIF kernels | MIT | SPICE maths |
| **Solar geometry (optional)** | ISIS3 `phocube` — per-pixel incidence/emission/phase. **Optional enhancement only**, see Risk R3 | CC0 | rigour where it works |

### 3.2 Reuse decisions we deliberately did *not* take

| Candidate | Verdict | Why |
|---|---|---|
| `planetaryimage` as the **primary** pixel reader | ⚠️ **opportunistic only** | Its own README states it is *"Alpha quality software that is being actively developed, use at your own risk"*, and there is a documented `^IMAGE` KeyError on detached labels. **Use `pvl` for labels (solid, mature) and our own `np.memmap` for pixels using pvl-parsed fields.** Try `planetaryimage` first, fall through on exception. |
| `RIFT2-python` as a **pip dependency** | ⚠️ **vendor it** | Small repo, no tagged releases. Copy into `third_party/rift2/` with attribution and a pinned commit SHA, and validate its output against the reference implementation on a fixture. A dependency that can vanish mid-hackathon is a risk we control by copying. |
| `image-matching-webui` for the demo UI | ❌ **no** | It is a generic *matching* UI. It cannot show coverage heatmaps, confidence tiers, cascade path, or the REJECTED state — and that display **is** the point of ours. Reusing it would delete our differentiator to save a day. |
| `AFIDS-POMM` as a dependency | ❌ **cite only** | Lists LRO NAC/WAC support, but 4 stars, untouched since April 2023, and needs external ISIS + SPICE + DEMs. Cite as prior art in `JUDGE_QA.md`; do not depend. |
| SRIF's 8-matcher harness | ❌ **superseded** | `vismatch` covers the same ground with 50+ matchers and active maintenance. |
| Ames Stereo Pipeline | ❌ **out of scope** | Stereo/DEM generation. We consume DEMs, we do not produce them. |
| ISIS3 as a **required** dependency | ❌ **optional only** | Risk R3: `spiceinit` is documented to fail on stock releases for OHRC. The pipeline must pass every test with ISIS absent. |

### 3.3 Stack summary

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11** (C++ only for a *proven* bottleneck) | entire ecosystem is Python-first |
| Matching | **`vismatch`** (primary) + **kornia** (geometry utilities, `ImageRegistrator`) | one API over 50+ matchers; kornia stays for its homography/RANSAC/epipolar helpers |
| Classical CV | **OpenCV** (`feature2d`, `findHomography`, `USAC_MAGSAC`, `cornerSubPix`) | Apache-2.0, the substrate everything sits on |
| Raster / GIS I/O | **GDAL + rasterio + pyproj + shapely** | MIT; windowed reads, reprojection, geotransform, footprints |
| PDS ingestion | **`pds4_tools` + `pvl` + `spectral`**, behind our thin ISRO adapter | reuse the parsing, own the validation |
| Illumination | **`phasepack`** + OpenCV CLAHE | the RIFT-family front end, already implemented |
| Planetary geometry | **`spiceypy`**; ISIS3 optional | see Risk R3 |
| API / demo | FastAPI + a static JS front-end | reproducible and offline-capable |
| Testing | pytest + pytest-cov | acceptance criteria are tests, not opinions |
| Packaging | `uv` or `pip-tools`, lockfile committed | reproducibility checklist item |

**Hardware target:** consumer GPU, ~8 GB VRAM (RTX 2080Ti-class or better) is sufficient for every learned matcher here. **CPU-only fallback is viable** via LightGlue / XFeat / the classical path — build and test this path, because the demo machine may not have a GPU.

**Tiling is mandatory, not optional.** Full-resolution products are enormous: a real TMC-2 label shows 191,483 × 4,000; a real OHRC raw `.img` loads at 79,796 × 12,000; full CDR NAC products run to 104,448 × 5,000. Nothing goes through a matcher whole.

**Published inference speeds (external, for planning only — measure our own at build time):** LightGlue 44.2 ms/pair (GPU) up to 150 FPS @1k keypoints; SuperGlue ~69 ms/pair; LoFTR 116–130 ms/pair; XFeat real-time webcam-capable.

---

## 4. REPOSITORY LAYOUT

Create this in Phase 0. Directory ownership is marked — **stay in your lane**.

```
chandralign/
├── PLAN.md                         # this file
├── FEATURES.csv                    # feature register (all 3 members update)
├── README.md
├── pyproject.toml / requirements.lock
├── Makefile                        # make setup | test | bench | ablate | demo
├── configs/                        # [A owns, B+C read]
│   ├── default.yaml                # ship_mode, thresholds, paths
│   ├── instruments.yaml            # GSD, bands, swath per instrument
│   ├── regimes.yaml                # regime-selector routing rules
│   └── benchmark_tiers.yaml        # Easy / Medium / Hard / Extreme definitions
├── src/chandralign/
│   ├── contracts.py                # ⚠️ FROZEN after Phase 0 — all 3 members
│   ├── io/                         # ── PART 1 · Member A ──────────────
│   │   ├── pds_label.py            #    adapter: pds4_tools + pvl → SceneMeta
│   │   ├── pds_raster.py           #    raw .IMG reader (shape/dtype from label)
│   │   ├── instruments.py          #    instrument registry
│   │   ├── reference.py            #    LRO NAC/WAC, SELENE TC/MI loaders
│   │   ├── dem.py                  #    LOLA / SLDEM2015 / Kaguya ARD (STAC+S3)
│   │   ├── ode_client.py           #    PDS ODE REST footprint search
│   │   ├── cache.py                #    local cache + checksum manifest
│   │   └── tiling.py               #    tile/patch generator
│   ├── geometry/                   # ── PART 1 · Member A ──────────────
│   │   ├── solar.py                #    incidence / emission / phase angles
│   │   ├── footprint.py            #    overlap intersection, pair pre-filter
│   │   ├── projection.py           #    GDAL reprojection, selenographic CRS
│   │   └── dem_terrain.py          #    slope / aspect rasters
│   ├── preprocess/                 # ── PART 1 · Member A ──────────────
│   │   ├── radiometric.py          #    CLAHE, normalisation
│   │   ├── phase_congruency.py     #    Log-Gabor / PC (RIFT front-end)
│   │   ├── iirs_composite.py       #    per-band SNR + panchromatic synthesis
│   │   ├── shadow_mask.py          #    shadow-aware masking
│   │   └── texture.py              #    low-texture & repetitive-terrain detection
│   ├── matching/                   # ── PART 2 · Member B ──────────────
│   │   ├── adapter.py              #    thin vismatch wrapper → MatchSet
│   │   ├── licence.py              #    ⚠️ model-name licence gate (§2.3)
│   │   ├── rift.py                 #    wrapper over vendored third_party/rift2
│   │   ├── regime.py               #    regime selector
│   │   ├── cascade.py              #    scale-bridging cascade
│   │   └── filters.py              #    ratio test, MNN, adaptive thresholds
│   ├── estimate/                   # ── PART 2 · Member B ──────────────
│   │   ├── robust.py               #    RANSAC / MAGSAC++ / USAC
│   │   ├── models.py               #    affine / homography / TPS
│   │   └── geometry_filter.py      #    ⭐ DEM-slope-aware outlier rejection
│   ├── refine/                     # ── PART 2 · Member B ──────────────
│   │   ├── uniformity.py           #    grid top-k, empty-cell pass, Delaunay, FPS
│   │   └── subpixel.py             #    NCC quad-fit, phase-corr, cornerSubPix, LSQ
│   ├── evaluate/                   # ── PART 2 · Member B (C consumes) ─
│   │   ├── metrics.py              #    [C] RMSE, inliers, ratio, coverage
│   │   ├── control_gates.py        #    null test, perturbation test
│   │   ├── groundtruth.py          #    [A] label geometry × DEM cross-check
│   │   ├── benchmark.py            #    4-tier runner
│   │   ├── ablation.py             #    [C] 6-stage runner
│   │   └── failure_log.py          #    [C] observed failure → 20-mode register
│   ├── product/                    # ── PART 3 · Member C ──────────────
│   │   ├── warp.py                 #    registered GeoTIFF product
│   │   ├── matchpoints.py          #    CSV + GeoJSON match-point export
│   │   ├── provenance.py           #    manifest: inputs, versions, commit
│   │   └── report.py               #    HTML evaluation report
│   ├── viz/                        # ── PART 3 · Member C ──────────────
│   │   ├── sidebyside.py           #    illumination-difference panel
│   │   ├── match_plot.py           #    match overlay
│   │   ├── coverage_plot.py        #    grid-occupancy heatmap
│   │   └── swipe.py                #    before/after checkerboard + swipe
│   ├── api.py                      # ── PART 3 · Member C — FastAPI
│   └── cli.py                      # ── PART 3 · Member C — `chandralign ...`
├── ui/                             # ── PART 3 · Member C — demo front-end
├── third_party/                    # vendored, pinned by SHA (see §3.2)
│   └── rift2/                      #    RIFT2-python, with attribution + SHA
├── tests/                          # everyone writes tests for their own modules
│   ├── test_pds_real.py            # anti-stub gate (A)
│   ├── test_contracts.py           # contract conformance (all)
│   ├── test_honesty.py             # H1–H6 enforcement (B)
│   └── ...
├── scripts/
│   ├── fetch_lro.py                # A
│   ├── fetch_selene.py             # A
│   ├── fetch_dem.py                # A
│   ├── pradan_runbook.md           # A — manual SSO download steps
│   ├── bench_external.py           # B — ship_mode=false, SuperGlue + AROSICS
│   ├── check_licences.py           # C — CI licence audit + vendored-SHA check
│   ├── run_demo.py                 # C — the scripted 3-minute demo
│   └── reproduce.sh                # C — regenerates every reported number
├── data/                           # gitignored; manifest.json IS committed
├── reports/                        # generated; committed at milestones
└── docs/
    ├── ARCHITECTURE.md             # A+B
    ├── JUDGE_QA.md                 # C — the 40 questions
    ├── REPRODUCIBILITY.md          # C
    ├── LICENSE_AUDIT.md            # C
    └── FAILURE_MODES.md            # B — the 20-mode register
```

---

## 5. THE INTEGRATION CONTRACT (frozen after Phase 0)

This is what lets 3 people work in parallel without blocking each other. **Write this file first, in Phase 0, together.** Everything else stubs against it.

> **PENDING SIGN-OFF (proposed by Member B, integrate/part2).** `Metrics.source` now also accepts `"synthetic"`, for numbers honestly measured on generated imagery before GATE A. Under the original two values those would have to be tagged `"measured"`, which reads as "measured on real Chandrayaan-2 data" -- the exact confusion rule H5 exists to prevent. **This block is updated to match the merged `contracts.py`; it is not agreed until all three members sign off in the PR** (PLAN.md 2.4).

```python
# src/chandralign/contracts.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import numpy as np

Instrument = Literal["OHRC", "TMC2", "IIRS", "NAC", "WAC", "TC", "MI"]
Tier = Literal["HIGH", "MEDIUM", "LOW", "REJECTED"]
Regime = Literal["same_modal_normal", "same_modal_polar",
                 "cross_modal", "extreme_scale"]

# Where a number came from. "synthetic" means honestly measured, but on
# generated imagery -- never to be presented as a real-data result (rule H5).
MetricSource = Literal["measured", "external", "synthetic"]


@dataclass(frozen=True)
class SceneMeta:
    """Everything the label actually told us. Part 1 -> everyone."""

    product_id: str
    instrument: Instrument
    mission: str                       # "CH2" | "LRO" | "SELENE"
    gsd_m: float
    n_bands: int
    wavelength_nm: tuple[float, float] | None
    array_shape: tuple[int, int]       # (lines, samples)
    dtype: str
    corner_latlon: list[tuple[float, float]]   # 4 corners, may be empty
    sub_solar_azimuth_deg: Optional[float]
    solar_incidence_deg: Optional[float]
    emission_deg: Optional[float]
    phase_deg: Optional[float]
    acquisition_utc: Optional[str]
    label_path: Path
    raster_path: Path
    # ANTI-STUB PROOF: which fields genuinely came from the label
    label_fields_verified: dict[str, bool] = field(default_factory=dict)


@dataclass
class GeometryLayers:
    """Per-pixel physics. Part 1 -> Part 2 (geometry_filter, regime)."""

    incidence_deg: Optional[np.ndarray] = None
    emission_deg: Optional[np.ndarray] = None
    phase_deg: Optional[np.ndarray] = None
    dem_elev_m: Optional[np.ndarray] = None
    slope_deg: Optional[np.ndarray] = None
    aspect_deg: Optional[np.ndarray] = None
    source: str = "unknown"            # "label" | "spice" | "dem" | "derived"


@dataclass
class ImagePlane:
    """THE handoff object: Part 1 -> Part 2. One tile, ready to match."""

    array: np.ndarray                  # float32, 2-D, normalised 0..1
    valid_mask: np.ndarray             # bool
    shadow_mask: np.ndarray            # bool -- True = shadowed, exclude
    gsd_m: float
    meta: SceneMeta
    geo: Optional[GeometryLayers] = None
    tile_origin: tuple[int, int] = (0, 0)   # (row, col) in the full product
    preprocess_chain: list[str] = field(default_factory=list)
    texture_score: Optional[float] = None
    repetitiveness_score: Optional[float] = None


@dataclass
class MatchSet:
    """Part 2 internal -> Part 3 export."""

    src_pts: np.ndarray                # (N,2) float64, SUB-PIXEL, source frame
    ref_pts: np.ndarray                # (N,2) float64, reference frame
    confidence: np.ndarray             # (N,) float32
    method: str                        # "lightglue" | "rift" | "sift" | ...
    regime: Regime
    stage: str                         # cascade stage, e.g. "TMC2->TC"


@dataclass
class TransformModel:
    kind: Literal["affine", "homography", "tps"]
    matrix: Optional[np.ndarray] = None       # 3x3 for affine/homography
    tps_params: Optional[dict] = None
    scale_estimated: Optional[float] = None
    scale_expected: Optional[float] = None    # from known GSD ratio (failure mode #13)


@dataclass
class Metrics:
    """Every field is Optional. Unmeasured is None, NEVER a made-up number (rule H1)."""

    rmse_px: Optional[float] = None
    rmse_m: Optional[float] = None
    inlier_count: Optional[int] = None
    inlier_ratio: Optional[float] = None
    spatial_coverage: Optional[float] = None
    max_delaunay_gap_px: Optional[float] = None
    subpixel_recovery_err_px: Optional[float] = None
    keypoints_src: Optional[int] = None
    keypoints_ref: Optional[int] = None
    runtime_s: Optional[float] = None
    source: MetricSource = "measured"   # rule H5


@dataclass
class RegistrationResult:
    """THE handoff object: Part 2 -> Part 3."""

    matches: MatchSet
    inlier_mask: np.ndarray
    model: Optional[TransformModel]
    metrics: Metrics
    confidence_tier: Tier
    gates: dict[str, bool]             # control gates -- must be non-empty (rule H4)
    failure_modes: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
```

**The three handoff objects, and who produces them:**

```
  Member A (Part 1)              Member B (Part 2)              Member C (Part 3)
  ─────────────────              ─────────────────              ─────────────────
  raw .IMG + .xml/.lbl
        ↓
     SceneMeta ─────────────────────────────────────────────────────→ (report header)
        ↓
     ImagePlane  ──── GATE A ───→ match → estimate → refine
     (src + ref)                        ↓
                                 RegistrationResult ── GATE B ──→ warp + export
                                        ↓                          + viz + report
                                  Metrics, gates                   + demo UI
```

---

## 6. SYSTEM ARCHITECTURE (the pipeline we are building)

Recommended architecture = **hybrid classical + deep (F)** × **coarse-to-fine cascade (H)**, with **geometry-guidance (G) as an enhancement layer, not a wholesale replacement**. G is used as a *feature and a filter*, because no working geometry-guided transformer precedent exists to build on — claiming one outright would be overclaiming.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  RAW INPUT: CH-2 product (.IMG + PDS4 .xml)  +  Reference product            │
│             (LRO NAC/WAC PDS4, or SELENE TC/MI PDS3 .img+.lbl)               │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │                                  ╔═══════════╗
┌───────────────────────────────▼──────────────────────────────┐   ║  PART 1   ║
│ 1. REAL PDS3/PDS4 INGESTION                                  │   ║ Member A  ║
│    • genuine label parsing (NOT cv2.imread fallback)         │   ╚═══════════╝
│    • raw .IMG read with shape/dtype from the label
│    • instrument auto-detect → instruments.yaml registry
│    • tiling (mandatory: products reach 191k × 4k pixels)
├──────────────────────────────────────────────────────────────┤
│ 2. GEOMETRY EXTRACTION                                       │
│    • incidence / emission / phase angle (label + SPICE)
│    • DEM elevation → slope / aspect (LOLA / SLDEM / Kaguya ARD)
│    • footprint intersection → reject non-overlapping pairs early
│    • scale-ratio pre-flight from known GSDs
├──────────────────────────────────────────────────────────────┤
│ 3. ILLUMINATION-AWARE PREPROCESSING                          │
│    • CLAHE radiometric normalisation
│    • phase-congruency / Log-Gabor transform (RIFT-family front end)
│    • IIRS: per-band SNR → panchromatic-equivalent composite
│      (NEVER match raw hyperspectral cubes — 2 teams failed doing that)
│    • shadow-aware masking; low-texture & repetitive-terrain scoring
└───────────────────────────────┬──────────────────────────────┘
                                │  ══════ GATE A: ImagePlane ══════
┌───────────────────────────────▼──────────────────────────────┐   ╔═══════════╗
│ 4. SCALE-BRIDGING CASCADE  (never direct across 320:1)       │   ║  PART 2   ║
│      OHRC ──(~20:1, precedent exists)──→ TMC-2               │   ║ Member B  ║
│      TMC-2 ──(~2:1, THE open gap)──────→ SELENE TC           │   ╚═══════════╝
│      TMC-2 ──(~16:1)───────────────────→ IIRS
│      any ─────────────────────────────→ LRO NAC / WAC
│      OHRC↔NAC and IIRS↔WAC are used as CALIBRATION CHECKPOINTS,
│      not reinvented.
├──────────────────────────────────────────────────────────────┤
│ 5. REGIME-SELECTED MATCHING  (routing, not hardcoding)       │
│      incidence angle + scale ratio + modality  →  matcher
│      ├ same-modal, normal illumination  → LightGlue (kornia)
│      ├ same-modal, polar / low sun      → LightGlue/LoFTR + PC preprocessing
│      ├ cross-modal (pan ↔ hyperspectral)→ RIFT / MIM classical descriptor
│      └ extreme scale                    → cascade + affine simulation
│      (SIFT+RANSAC always available as the documented baseline/floor)
├──────────────────────────────────────────────────────────────┤
│ 6. ROBUST GEOMETRIC ESTIMATION — RANSAC / MAGSAC++ / USAC     │
│      affine | homography | TPS (TPS for crater-driven local relief)
│      estimated scale is sanity-checked against the known GSD ratio
├──────────────────────────────────────────────────────────────┤
│ 7. ⭐ GEOMETRY-GUIDED OUTLIER FILTERING  (the novel layer)     │
│      reject matches inconsistent with known terrain relief /
│      DEM slope / incidence geometry — not just RANSAC's purely
│      photometric consensus. No general algorithmic precedent exists.
├──────────────────────────────────────────────────────────────┤
│ 8. SPATIAL-UNIFORMITY ENFORCEMENT  (PS mandate)               │
│      N×N grid → per-cell top-k → empty-cell secondary pass →
│      Delaunay gap diagnostic → coverage metric
├──────────────────────────────────────────────────────────────┤
│ 9. SUB-PIXEL REFINEMENT  (PS mandate)                         │
│      quadratic NCC peak fit · phase-correlation residual shift ·
│      cornerSubPix · iterative least-squares homography refinement
├──────────────────────────────────────────────────────────────┤
│10. CONTROL GATES + QUALITY GATING                             │
│      constant-grey null test · perturbation sensitivity ·
│      shared-mask guard · confidence tiers HIGH/MED/LOW/REJECTED
│      → NEVER silently degrade into a fabricated "success"
└───────────────────────────────┬──────────────────────────────┘
                                │  ══════ GATE B: RegistrationResult ══════
┌───────────────────────────────▼──────────────────────────────┐   ╔═══════════╗
│11. DELIVERABLE                                                │   ║  PART 3   ║
│    • registered product (GeoTIFF + geotransform + sidecar)    │   ║ Member C  ║
│    • corresponding match points (CSV + GeoJSON)               │   ╚═══════════╝
│    • metrics: RMSE · inlier count · inlier ratio · coverage
│    • HTML evaluation report with provenance manifest
│    • demo UI: side-by-side · match overlay · coverage heatmap ·
│      before/after swipe · live metrics
└──────────────────────────────────────────────────────────────┘
```

---

## 7. WORK SPLIT — THE THREE PARTS

> **Rebalanced after the §3.1 reuse decisions.** Adopting `vismatch`, `pds4_tools`, `pvl`, `spectral`, `phasepack` and `RIFT2-python` removed roughly a third of the hand-written code, and it came disproportionately out of Part 2's matcher layer. Two blocks of work moved to keep the three loads even: **ground truth + benchmark dataset construction moved to Member A** (it is data work, and A owns the data layer), and **metrics + failure logging moved to Member C** (C owns everything that displays those numbers, so owning their computation removes a handoff).

### PART 1 — Member A · **Data, Geometry & Benchmark Foundation**

> **Everything from raw bytes to a clean, geometry-annotated, preprocessed tile pair — plus the benchmark data and its ground truth.**
> Nobody else can start on real data until the first half lands. Front-loaded on purpose.

| Owns | Modules |
|---|---|
| Ingestion (thin adapters over `pds4_tools` / `pvl` / `spectral`) | `io/pds_label.py`, `io/pds_raster.py`, `io/instruments.py`, `io/reference.py`, `io/cache.py`, `io/tiling.py`, `io/ode_client.py`, `io/dem.py` |
| Geometry | `geometry/solar.py`, `geometry/footprint.py`, `geometry/projection.py`, `geometry/dem_terrain.py` |
| Preprocessing | `preprocess/radiometric.py`, `preprocess/phase_congruency.py`, `preprocess/iirs_composite.py`, `preprocess/shadow_mask.py`, `preprocess/texture.py` |
| **Ground truth & benchmark data** ← *moved from B* | `evaluate/groundtruth.py`, `configs/benchmark_tiers.yaml`, the tier pair manifests |
| Config | all of `configs/` |
| Scripts | `scripts/fetch_lro.py`, `fetch_selene.py`, `fetch_dem.py`, `pradan_runbook.md` |

**Produces:** `SceneMeta`, `GeometryLayers`, `ImagePlane`, the four benchmark tiers with derived ground truth
**Signature deliverable:** ingestion that passes `tests/test_pds_real.py` against **real downloaded products**, and the TMC-2 + SELENE pair set nobody else has built.

---

### PART 2 — Member B · **Registration Core & Differentiators**
> **Strictly downstream of Part 1.** Consumes `ImagePlane`, produces `RegistrationResult`.
> Now concentrated on the things that do not exist off the shelf.

| Owns | Modules |
|---|---|
| Matching (integration + our logic) | `matching/licence.py`, `matching/adapter.py` (thin `vismatch` wrapper), `rift.py` (vendored RIFT2), **`regime.py`**, **`cascade.py`**, `filters.py` |
| Estimation | `estimate/robust.py`, `models.py`, **`geometry_filter.py`** |
| Refinement | `refine/uniformity.py`, `refine/subpixel.py` |
| Validation | `evaluate/control_gates.py`, `benchmark.py` (runner), `ablation.py` |

**Produces:** `MatchSet`, `TransformModel`, `RegistrationResult`
**Signature deliverables (all bolded above):** the geometry-guided filter (no precedent to copy), the scale-bridging cascade, the regime selector, and the control-gate discipline. **Everything B now writes is something that does not already exist** — the matcher plumbing is `vismatch`'s problem.

---

### PART 3 — Member C · **Product, Evidence, Metrics & Demo**

> Starts Day 1 against **mock** `RegistrationResult` objects, switches to real data at GATE B.

**Feature 3.1 — Deliverable Export & Evaluation Report**
The PS's literal ask: *"Software and registered product with corresponding match points" + "Evaluation metric"*.
- registered GeoTIFF with correct geotransform
- match points → CSV + GeoJSON
- metrics table → HTML report
- provenance manifest (inputs, versions, git SHA) embedded in every output

**Feature 3.2 — Demo UI**
- pick/upload a source + reference pair
- run pipeline, stream progress
- side-by-side illumination-difference panel · match overlay · coverage heatmap · before/after swipe · live metric readout
- an explicit "this run was REJECTED because …" state (honesty rule H3 rendered in the UI, not hidden)

**Feature 3.3 — Metrics & failure logging** ← *moved from B*
`evaluate/metrics.py` and `evaluate/failure_log.py`. Pure computation from a `MatchSet` plus an inlier mask, with no dependency on B's internals. C already owns every surface that renders these numbers, so owning their computation removes a handoff and lets the honesty rules (H1, H5) be enforced in one place.

| Also owns | Files |
|---|---|
| CLI + pipeline orchestration | `cli.py` |
| API | `api.py` |
| Visuals | `viz/*` (`match_plot.py` now wraps `vismatch.viz.plot_matches`) |
| Docs | `JUDGE_QA.md`, `REPRODUCIBILITY.md`, `LICENSE_AUDIT.md`, `FAILURE_MODES.md` |
| Demo | `scripts/run_demo.py`, `scripts/reproduce.sh`, `scripts/check_licences.py` |

---

## 8. PHASE PLAN — who does what, when

Baseline is a **30-day** window. A compressed 10-day column is given for a short internal hackathon; the ordering and gates never change.

| Phase | 30-day | 10-day | Theme | Exit gate |
|---|---|---|---|---|
| **0** | D1–D2 | D1 | Setup & contracts | `contracts.py` frozen; repo green; data accounts live |
| **1** | D3–D7 | D2–D3 | Skeleton & baseline | **GATE A** — real `ImagePlane` from a real OHRC + real NAC product |
| **2** | D8–D14 | D4–D5 | Core capability | **GATE B** — real `RegistrationResult` on OHRC↔NAC |
| **3** | D15–D21 | D6–D7 | Differentiators | **GATE C** — cascade + geometry filter + TMC-2/SELENE axis running |
| **4** | D22–D26 | D8–D9 | Benchmark, ablation, control gates | Full 4-tier + 6-stage numbers, all measured |
| **5** | D27–D30 | D10 | Demo, docs, freeze | Demo runs end-to-end twice in a row from a clean clone |

### Phase-by-phase assignment matrix

| Phase | Member A (Part 1) | Member B (Part 2) | Member C (Part 3) |
|---|---|---|---|
| **0** | repo skeleton, `configs/`, data-account setup | `contracts.py` draft, matcher interface sketch | `Makefile`, CI, mock-data generator |
| **1** | PDS4+PDS3 parsers, raster reader, tiling, instrument registry | classical baseline (SIFT/AKAZE + RANSAC) on **synthetic** pairs, metrics skeleton | CLI skeleton, match-point export, mock report |
| **2** | solar geometry, DEM, footprint, CLAHE, PC transform, IIRS composite, shadow mask | LightGlue/LoFTR paths, uniformity, sub-pixel, ground truth | real report, `viz/*`, API skeleton |
| **3** | SELENE PDS3 + TMC-2 products; **then joins B** on benchmark data prep | **cascade**, **regime selector**, **geometry-guided filter**, RIFT path | demo UI build-out, `JUDGE_QA.md` |
| **4** | benchmark tier dataset construction + ground-truth cross-checks | benchmark runner, ablation runner, control gates, failure log | report polish, `REPRODUCIBILITY.md`, `LICENSE_AUDIT.md` |
| **5** | data manifest freeze, `ARCHITECTURE.md` | numbers freeze, `FAILURE_MODES.md`, external-baseline comparison | **`run_demo.py`**, rehearsal ×2, slide-ready exports |

> **Balance after the reuse decisions — the real numbers, not a flattering round-off.** Counting shared features for both owners: **A = 40 (33 net-new), B = 45 (40 net-new), C = 26 (24 net-new).** B is still the heaviest, and that is not a scheduling failure — B owns the four things that do not exist off the shelf (regime selector, cascade, geometry-guided filter, control gates), and those are irreducible. What *has* changed is the shape of B's work: the entire matcher-adapter layer went to `vismatch`, ground truth went to A, and metrics, failure logging, the ablation runner and external-validity reporting went to C.
>
> **The relief valve is real and must not be skipped:** Member A's Part 1 is now front-loaded and largely finished by the end of Phase 2. **From Phase 3, A works alongside B on the cascade and the geometry-guided filter** — A already owns the geometry layer those two consume, so the ramp-up cost is near zero. Phase 4 fails without this reassignment.

> **Phase 1 is now materially shorter for Member A.** Ingestion that was a five-day parser build is a one-to-two-day adapter build. Spend the recovered time on **P1-T17 (TMC-2 + SELENE acquisition)**, pulling it forward from Phase 3 into Phase 2. That work has a long human-latency tail and it is the differentiator; starting it early is the single highest-value schedule change available.

---

## 9. PHASE 0 — SETUP & CONTRACTS *(all three, D1–D2)*

### P0-T01 · Repo skeleton — **Member A**
**Goal** Create the full tree in §4 with `__init__.py` everywhere and empty-but-importable modules.
**Steps** `uv init` (or poetry/pip-tools) → add deps → create dirs → `.gitignore` for `data/`, `reports/*.html`, `*.IMG`, `*.tif` → commit.
**Deps** (core) `numpy scipy opencv-python-headless torch kornia rasterio gdal pyproj shapely pyyaml pytest pytest-cov fastapi uvicorn jinja2 matplotlib requests tqdm scikit-image`
**Deps** (reuse layer, §3.1) `pds4_tools pvl planetaryimage spectral phasepack vismatch spiceypy arosics`
**Vendored** `third_party/rift2/` — pinned copy of `RIFT2-python` with attribution and its commit SHA recorded in `docs/LICENSE_AUDIT.md`
**Accept also** a `scripts/check_licences.py` run in CI lists every installed package with its licence and fails on any non-permissive entry.
**Accept** `python -c "import chandralign"` works; `pytest` collects 0 tests without error.

### P0-T02 · Freeze `contracts.py` — **all three, together**
**Goal** Paste §5 verbatim into `src/chandralign/contracts.py`, review as a group, merge, then treat as frozen.
**Accept** `tests/test_contracts.py` constructs every dataclass with dummy values and round-trips it through `dataclasses.asdict`.

### P0-T03 · Config files — **Member A**
**Files** `configs/instruments.yaml` (GSD/bands/swath/dtype per instrument from §1.2), `configs/default.yaml` (`ship_mode: true`, grid size, thresholds, data paths), `configs/regimes.yaml` (routing rules, §11.2), `configs/benchmark_tiers.yaml` (§12.1).
**Accept** `load_config()` returns typed objects; `instruments.yaml` covers all 7 instruments.

### P0-T04 · Data access — **Member A** *(do this on Day 1, it has a human-latency tail)*
1. Register on **PRADAN** (`pradan.issdc.gov.in/ch2/`). Mandatory Keycloak SSO; free for non-profit scientific use; ISRO retains ownership and requires acknowledgement. **There may be a manual approval step — start today.**
2. Verify the no-login sources work: LROC PDS Data Node, PDS ODE REST (`oderest.rsl.wustl.edu`), JAXA DARTS SELENE PDS3 (`data.darts.isas.jaxa.jp/pub/pds3`), USGS Astrogeology ARD (`aws s3 ls --no-sign-request s3://astrogeo-ard/moon/kaguya/terrain_camera/`).
3. Write `scripts/pradan_runbook.md` documenting the exact click-path, because it cannot be scripted.
**Accept** at least one real product from each of: CH-2 (any instrument), LRO NAC, SELENE TC, and one DEM, sitting in `data/raw/` with checksums in `data/manifest.json`.

### P0-T05 · Mock-data generator — **Member C**
**Goal** Unblock Parts 2 and 3 before real data lands.
**File** `tests/factories.py` — builds synthetic `SceneMeta` / `ImagePlane` / `RegistrationResult`, plus a synthetic image pair with a **known** homography and a **known** sub-pixel shift (this doubles as the sub-pixel validation fixture in P2-T14).
**Accept** `make_pair(scale=2.0, rot_deg=15, shift=(0.37, -0.62))` returns two arrays plus the exact ground-truth transform.

### P0-T06 · Makefile + CI — **Member C**
```make
setup:   ## install deps + pre-commit
test:    ## pytest -q --cov=chandralign
bench:   ## python -m chandralign.evaluate.benchmark --tiers all
ablate:  ## python -m chandralign.evaluate.ablation
demo:    ## python scripts/run_demo.py
report:  ## python -m chandralign.product.report --latest
```
**Accept** GitHub Actions runs `make test` on every PR and blocks merge on red.

---

## 10. PART 1 — MEMBER A · DETAILED TASKS

### ── Phase 1 ──────────────────────────────────────────────────

### P1-T01 · PDS4 ingestion adapter *(the signature task — now an adapter, not a parser)*
**Goal** Turn a real Chandrayaan-2 PDS4 label into `SceneMeta`. **We do not write an XML parser.** `pds4_tools` does the parsing; we write the ISRO-specific field mapping and, critically, the validation that proves it worked.
**File** `io/pds_label.py` → `parse_pds4(xml_path) -> SceneMeta`
**Steps**
1. `structures = pds4_tools.read(xml_path, lazy_load=True, quiet=True)` — lazy so multi-GB arrays are not pulled into memory.
2. Map `structures[0].meta_data` onto `SceneMeta`: product ID, instrument, `Array_2D_Image` axis lengths, `data_type`, GSD, acquisition UTC, corner lat/lon (OHRC carries corners at **two quality levels** — capture both), sub-solar azimuth, incidence/emission/phase where present.
3. **ISRO fallback.** `pds4_tools` expects labels conforming to the PDS4 standard; ISRO products carry mission-specific dictionaries. If a field is absent from `meta_data`, fall back to a targeted `lxml` XPath for that field only. Roughly 100 lines, not a parser.
4. Populate `label_fields_verified[field] = True` **only** for fields genuinely read from the label, whichever path found them. Anything defaulted stays `False`.
5. Raise `PdsParseError` on a missing required field — **never** fall back to `cv2.imread`.
**Why this is still the differentiator:** the research finding was that competitors ship parsers that do not parse, not that parsers are hard to write. The proof is P1-T01's test, not its line count.
**Reference products to test against** (real, known to exist):
`ch2_tmc_ncn_20231026T0943001971_d_img_d18.xml` (orbit 18569, 88.55 km, 4.43 m/px, 191,483×4,000, ~1.5 GB) · `ch2_ohr_ncp_20211228T2209123959_d_img_d18.img` (OHRC, 79,796×12,000) · `ch2_iir_nri_20231003T2152304115_d_img_d18.hdr/.xml`
**Accept** `tests/test_pds_real.py` passes on ≥1 real OHRC, ≥1 real TMC-2, ≥1 real IIRS label; `label_fields_verified` has ≥6 `True` entries; a deliberately truncated label raises `PdsParseError`.

### P1-T02 · PDS3 ingestion adapter (SELENE / older products)
**File** `io/pds_label.py` → `parse_pds3(lbl_path) -> SceneMeta`
**Why** SELENE/Kaguya ships PDS3 `.img + .lbl`; Chandrayaan-1 and several NASA archives do too.
**Steps** `label = pvl.load(lbl_path)` handles the ODL grammar, unit suffixes (`<DEG>`, `<M>`) and `OBJECT = IMAGE` nesting. We map `LINES`, `LINE_SAMPLES`, `SAMPLE_BITS`, `SAMPLE_TYPE`, `RECORD_BYTES` and the `^IMAGE` pointer onto `SceneMeta`. `pvl` also reads ISIS cube labels, which we get free if we ever touch ISIS output.
**Accept** round-trips a real SELENE TC `.lbl`; byte offset from `^IMAGE` correctly locates pixel data.

### P1-T03 · Raster reader
**File** `io/pds_raster.py` → `read_raster(meta, window=None, bands=None) -> np.ndarray`
**Steps — three backends, tried in order:**
1. **IIRS / any ENVI `.hdr`** → `spectral.open_image(hdr)`; SPy handles BSQ/BIL/BIP interleave and exposes wavelengths. Use `.read_subregion()` for windows.
2. **PDS3 `.img`** → try `planetaryimage.PDS3Image`; **on any exception fall through to step 3**, because that package self-describes as alpha quality and has a known `^IMAGE` failure on detached labels.
3. **Fallback (and the PDS4 path)** → `np.memmap` with dtype/shape/offset taken from the label fields P1-T01/T02 already parsed. Handle big/little-endian. This is ~30 lines and it is the one we trust.
Windowed reads throughout so we never load 1.5 GB.
**Accept** reads a 1 KB window from a multi-GB product in <100 ms; shape assertion matches the label; the memmap fallback is exercised by a test that forces the `planetaryimage` path to fail.

### P1-T04 · Instrument registry
**File** `io/instruments.py`
**Steps** Load `instruments.yaml`; `detect_instrument(product_id) -> Instrument` from the naming convention (`ch2_ohr_*`, `ch2_tmc_*`, `ch2_iir_*`, `M*RC.IMG` for NAC…); expose `gsd_ratio(a, b)`.
**Accept** `gsd_ratio("OHRC","IIRS") ≈ 320`; every instrument in §1.2 resolves.

### P1-T05 · Tiling
**File** `io/tiling.py` → `iter_tiles(meta, tile=1024, overlap=128)`
**Why** Tiling is mandatory: real products reach 191k × 4k and 104k × 5k pixels. No whole-scene inference, ever.
**Steps** Use `rasterio.windows.Window` for the read primitive rather than hand-rolled slicing. We write only the tile-grid generator with overlap, the >X%-invalid skip, `tile_origin` propagation into `ImagePlane`, and `tile_to_global(pts, origin)` for stitching match points back. Small, but ours — no library knows our contract.
**Accept** tiles of a 191k×4k product enumerate lazily with bounded memory; `tile_to_global` round-trips exactly.

### P1-T06 · Reference loaders
**File** `io/reference.py`
**Steps** LRO NAC/WAC (PDS4 XML+IMG for EDR/CDR, GeoTIFF for RDR); SELENE TC/MI via PDS3; normalise all of them into the same `SceneMeta`.
**Test product** `M1404744695RC.IMG` (real NAC product, resolvable via the ODE REST API).
**Accept** a NAC product and a SELENE TC product both load into identical `SceneMeta` shape.

### ── Phase 2 ──────────────────────────────────────────────────

### P1-T07 · Solar / viewing geometry
**File** `geometry/solar.py`
**Steps**
1. Read per-scene solar azimuth / incidence / emission / phase from the label where present.
2. Where absent, derive per-pixel geometry from SPICE (mirroring what ISIS3's `phocube` does) or interpolate from scene corners as a documented approximation — **label which path was used** in `GeometryLayers.source`.
3. Expose `illumination_delta(src_meta, ref_meta) -> dict` (Δincidence, Δazimuth, Δphase) — this is the input the regime selector needs.
**Honest caveat to carry into the docs:** archive documentation confirms this metadata exists; verifying it byte-for-byte in a raw OHRC/TMC-2 XML label is a build-time step, not something to assert in advance.
**Accept** on a real label, incidence/azimuth are read (not defaulted) and `GeometryLayers.source == "label"`.

### P1-T08 · DEM ingestion
**File** `io/dem.py`
**Steps** Fetch/window LOLA (118 m/px global GeoTIFF), SLDEM2015 (59 m/px, ±60° lat only), Kaguya ARD Cloud-Optimized GeoTIFF from `s3://astrogeo-ard/moon/kaguya/terrain_camera/` (no AWS account needed, `--no-sign-request`); reproject/crop to a scene footprint.
**Accept** returns an elevation array co-registered to a given scene footprint, with the DEM source recorded.

### P1-T09 · Slope / aspect
**File** `geometry/dem_terrain.py` → `slope_aspect(dem, gsd_m)`
**Why** This is the input to **Part 2's geometry-guided filter** — the novel layer. Get it right.
**Steps** Shell out to `gdaldem slope` / `gdaldem aspect` (or use `richdem`) rather than implementing Horn's method by hand. Wrap both behind one function so the backend is swappable.
**Accept** slope of a synthetic constant-gradient DEM equals the analytic value within 0.1°.

### P1-T10 · Footprint intersection & pair pre-filter
**File** `geometry/footprint.py`
**Steps** Build shapely polygons from corner lat/lon; intersect; reject pairs below a minimum overlap fraction **before** any matching is attempted (failure mode #10); `ode_client.py` queries the PDS ODE REST API for reference products overlapping a CH-2 footprint.
**Accept** a deliberately non-overlapping pair is rejected with a clear reason and zero matcher calls.

### P1-T11 · Projection
**File** `geometry/projection.py`
**Steps** Selenographic CRS handling (`IAU_2015:30100`-style lunar CRS via pyproj), explicit GDAL reprojection step. **Raw/calibrated CH-2 products carry corner lat/lon only — no pre-applied map projection.** Never assume one is implicit (failure mode #9).
**Accept** reprojecting a scene and back returns corner coordinates within 1e-6°.

### P1-T12 · Radiometric preprocessing
**File** `preprocess/radiometric.py`
**Steps** CLAHE (tunable clip/tile), percentile stretch, per-tile normalisation to float32 0..1, optional ECC-style brightness/contrast normalisation (ECC is purpose-built for illumination variation but needs a reasonable initial alignment, so it belongs in refinement, not initial correspondence).
**Accept** CLAHE measurably raises local contrast in a shadowed crop; output dtype/range conform to `ImagePlane`.

### P1-T13 · Phase-congruency / Log-Gabor transform
**File** `preprocess/phase_congruency.py`
**Why** This is the RIFT-family front end: phase-congruency features survive **nonlinear radiometric differences** between sensors, which is exactly the PS's headline challenge, and they do not depend on strong local gradients (mitigating featureless-mare failure mode #1).
**Steps** Use **`phasepack`**: `phasecong()` for the oriented multi-scale bank, `phasecongmono()` for the faster monogenic variant. We write only the maximum-index map (MIM) on top of the returned per-orientation responses, plus caching. **Pin `phasepack` specifically** rather than the newer `phasecongruency` package, because the vendored `RIFT2-python` (P2-T13) is built against PhasePack and we want one phase-congruency implementation in the process, not two.
**Note** `phasepack` uses `pyFFTW` when available for a significant speedup. Install it.
**Accept** PC map of an image and of the same image with a strong gamma shift correlate > 0.9, where raw-intensity correlation is much lower.

### P1-T14 · IIRS panchromatic-equivalent composite
**File** `preprocess/iirs_composite.py`
**Why** No single IIRS band corresponds to OHRC/TMC-2's panchromatic response. Two independent teams measured **zero inliers** matching raw IIRS cubes. We synthesise instead.
**Steps** Per-band SNR estimation → drop low-SNR bands (failure mode #4) → weighted composite over high-SNR bands approximating a panchromatic response → single 2-D plane into `ImagePlane`.
**Accept** composite from a real IIRS cube has higher keypoint count than the best single band; the selected band list is logged in `preprocess_chain`.

### P1-T15 · Shadow masking
**File** `preprocess/shadow_mask.py`
**Steps** Region-level intensity-variance check + near-zero-illumination thresholding, optionally cross-checked against incidence angle; emit `shadow_mask` so the matcher can exclude those regions from primary matching and treat them as a separate low-confidence tier (failure mode #2).
**Accept** on a crater-interior crop, the mask covers the shadowed region with IoU > 0.7 against a hand-drawn mask.

### P1-T16 · Texture & repetitiveness scoring
**File** `preprocess/texture.py`
**Steps** `texture_score` (local gradient energy / entropy) → triggers the featureless-terrain fallback (failure mode #1). `repetitiveness_score` (autocorrelation peak structure) → triggers the stricter ratio-test threshold in Part 2 (failure mode #11).
**Accept** flat mare crop scores low texture; a crater-field crop scores high repetitiveness.

### ── Phase 3–5 ────────────────────────────────────────────────

### P1-T17 · SELENE + TMC-2 acquisition push *(the differentiator's fuel)*
**Goal** Actually download and parse real **TMC-2** and real **SELENE TC** products covering overlapping ground. This unglamorous work is precisely what most of the field skips — and it is what makes our headline claim real rather than aspirational.
**Accept** ≥3 genuinely overlapping TMC-2 ↔ SELENE TC pairs and ≥3 TMC-2 ↔ LRO NAC pairs in `data/pairs/`, with footprint overlap computed and recorded.

### P1-T18 · Benchmark tier dataset construction *(Phase 4 — now Member A's alone)*
Build the four tiers of §12.1 as concrete, manifested pair lists with derived ground truth. Member B consumes the manifest through `configs/benchmark_tiers.yaml` and never touches the data itself.

### P1-T20 · Ground-truth derivation ← *moved from Member B (was P2-T09)*
**File** `evaluate/groundtruth.py`
**Why it moved here** It is data work built on label geometry and DEM reprojection, both of which Member A already owns. Keeping it with B meant B needed deep knowledge of the ingestion layer.
**Why** No public ground-truth benchmark exists for this instrument/reference combination — every team that reported real metrics computed its own. So we derive ours, and we **never trust a single source** (failure mode #8).
**Steps** Transform from each product's PDS4 label geometry (OHRC's two corner-quality levels both used) → **cross-validate against an independent DEM re-projection** (LOLA / SLDEM / Kaguya ARD) → hand-verify match points on a small held-out subset → record the disagreement between the two sources as an explicit uncertainty bound.
**Accept** label-derived and DEM-derived transforms are reported together with their disagreement; a synthetic injected label error is caught by the DEM cross-check.

### P1-T21 · Photoclinometry-style shadow assistance *(Phase 3, with B — P2 priority)*
**Files** `geometry/solar.py`, `estimate/geometry_filter.py`
**Why** The document separates Class 2 geometry-**guided** matching (sun geometry actively drives correspondence) from Class 1 illumination-**invariant** description, with photoclinometry-assisted matching between differently-illuminated LROC NAC images as the precedent. Judge answer #13 commits us to using known sun geometry to *assist* rather than merely tolerate the shadow, where geometry permits.
**Accept** where incidence angle and a DEM both exist, predicted shadow geometry assists matching; falls back to plain masking otherwise, and the report says which path ran.

### P1-T22 · Real-data-only validation gate *(Phase 4)*
**File** `tests/test_real_data.py`
**Why** Opportunity #11 in the document: real (not synthetic-only) validation against actual PRADAN-downloaded products, scored as a genuine gap because most competitors test on synthetic or generic images. **Every headline number must come from a real product.** Synthetic data is confined to unit fixtures.
**Accept** a test asserts every benchmark-tier pair resolves to a real product ID in `data/manifest.json`; a synthetic input inside a reported tier fails the build.

### P1-T23 · Publish the benchmark *(Phase 5, P2 priority)*
**File** `docs/BENCHMARK.md`
**Why** Opportunity #12: no reusable public ground-truth benchmark exists for this instrument/reference combination, and every competitor with real metrics computed its own internally. Publishing ours is a contribution that outlives the hackathon.
**Accept** pair manifests, ground-truth derivation and evaluation protocol are runnable by a third party from the repo alone.

### P1-T19 · `ARCHITECTURE.md` + data manifest freeze *(Phase 5)*
Diagram of §6, module ownership table, `data/manifest.json` with checksums for every product used in a reported number.

---

## 11. PART 2 — MEMBER B · DETAILED TASKS

### ── Phase 1 ──────────────────────────────────────────────────

### P2-T01 · `vismatch` adapter + licence gate *(was: build a matcher interface)*
**Files** `matching/adapter.py`, `matching/licence.py`
**We no longer write a matcher interface.** `vismatch` already provides one: `get_matcher(name, device)` returns a callable whose result carries `num_inliers`, `H`, `all_kpts0/1`, `matched_kpts0/1`, `inlier_kpts0/1`. Over 50 matchers, weights auto-downloaded, BSD-3.
**What we write** is a ~60-line adapter that (a) converts `ImagePlane` → the tensor form `vismatch` expects, (b) converts its result → our `MatchSet`, (c) routes every call through the model-name licence gate from §2.3, and (d) pins `skip_ransac=True` so **our** estimator (P2-T06) does the geometry, not theirs — we need the inlier mask and the scale sanity check under our control.
**Accept** three different matchers run through the identical adapter call path; `tests/test_licence_gate.py` proves `superpoint-lightglue` is unavailable while `ship_mode` is true and available when it is false.

### P2-T02 · Classical baseline — SIFT + RANSAC
**File** `matching/filters.py` (the detectors come from `vismatch` / OpenCV)
**Why** This is the field's own documented baseline and our ablation floor. Every real lunar paper uses it as the comparison floor. We must have it working before anything else.
**Steps** `get_matcher("sift-lg")` or plain OpenCV SIFT for the true classical floor — no adapter code of our own beyond P2-T01. **What we do write** is `filters.py`: the Lowe ratio test with our adaptive threshold (ADD-24) and the mutual-nearest-neighbour filter, because the adaptive part is driven by our `repetitiveness_score` and no library knows about that.
**Accept** on the synthetic pair from P0-T05 with a known homography, recovers it to < 1 px RMSE.

### P2-T03 · Metrics skeleton → **reassigned to Member C (see P3-T13)**
Member B still *consumes* `Metrics`, but no longer owns the module. This removes a circular dependency: C's report was blocked on B's metrics, while B's benchmark runner was blocked on nothing C had. See §7 Part 3, Feature 3.3.

### ── Phase 2 ──────────────────────────────────────────────────

### P2-T04 · Learned matcher selection *(was: write adapters)*
**File** `configs/regimes.yaml` (config, not code)
**Steps** No adapter code — P2-T01 covers all of them. This task is now **choosing and benchmarking** which `vismatch` model names we ship: LightGlue variants (Apache-2.0 incl. weights → our default), LoFTR / EfficientLoFTR, XFeat for the CPU path, RoMa if dense matching helps on low-texture mare. Record the chosen names in `regimes.yaml`, and add tile-aware batching to the adapter if throughput demands it.
**CPU fallback must work** — LightGlue and XFeat are both CPU-capable, and the demo machine may have no GPU.
**Accept** the chosen default runs on CPU and GPU, returns a conforming `MatchSet`, and beats SIFT on the synthetic illumination-shifted pair.

### P2-T05 · SuperGlue benchmark run *(licence-gated, never shipped)*
**File** `scripts/bench_external.py` (no adapter needed — `vismatch` already carries SuperGlue)
**Why** SuperGlue has the strongest direct real-Chandrayaan-2 evidence of any method — lowest RMSE and fastest runtime among SIFT/ASIFT/AKAZE/RIFT2/SuperGlue on real OHRC/IIRS/DFSAR data, and it did not degrade at the poles where the classical methods did. We must benchmark honestly against the field's strongest known baseline. But its licence is non-commercial research only, so it can never be in the deliverable.
**Accept** importable only under `ship_mode: false`; its outputs land in `reports/external_benchmark/`, which the deliverable bundler excludes.

### P2-T06 · Robust estimation
**File** `estimate/robust.py`, `estimate/models.py`
**Steps** MAGSAC comes free: `cv2.findHomography(..., method=cv2.USAC_MAGSAC)` and `cv2.USAC_ACCURATE`. TPS via `cv2.createThinPlateSplineShapeTransformer` or `scipy.interpolate.RBFInterpolator`. **What we write** is the model-selection logic (affine vs homography vs TPS by residual structure) and the **scale sanity check against the known GSD ratio** (failure mode #13), rejecting an estimate that disagrees by more than a configured factor. That check exists nowhere off the shelf because it needs our instrument registry.
**Accept** MAGSAC beats plain RANSAC on a synthetic set with 70% outliers; a deliberately wrong-scale match set is rejected.

### P2-T07 · Spatial uniformity enforcement *(PS mandate)*
**File** `refine/uniformity.py`
**The 5-step algorithm — implement exactly:**
1. Partition the source image into an **N×N grid** (N adaptive to image size and expected match density).
2. Within each cell, keep only the **top-k highest-confidence** matches; discard the excess so dense regions cannot dominate the RANSAC consensus.
3. For cells with **zero** matches after the primary pass, optionally run a **lower-threshold secondary pass restricted to that cell's local region**.
4. Report **spatial coverage** = fraction of grid cells containing ≥1 valid inlier — a PS-named evaluation metric.
5. Optionally run **Delaunay triangulation** over the final inlier set and flag any triangle above an area threshold as an under-covered region (a second, complementary coverage diagnostic).
Provide farthest-point sampling as an alternative selector.
**Accept** on a synthetic match set deliberately clustered in one corner, coverage rises from <0.15 to >0.6 after enforcement, and the visual matches the PS's illustrated *target* (evenly spaced with local structure) rather than its illustrated *failure* (a dense cluster plus empty regions).

### P2-T08 · Sub-pixel refinement *(PS mandate)*
**File** `refine/subpixel.py` — **four methods, two of them free:**
1. **Quadratic NCC peak fitting** — ours, ~40 lines: fit a parabola to the NCC surface around each integer-pixel match, interpolate the sub-pixel peak.
2. **Phase-correlation residual-shift correction** — `skimage.registration.phase_cross_correlation(ref, src, upsample_factor=100)` returns sub-pixel shift directly via upsampled DFT. Add log-polar for the scale residual.
3. **`cv2.cornerSubPix`** — straight OpenCV, no work.
4. **Iterative least-squares homography refinement** — ours, `scipy.optimize.least_squares` over the 8 homography parameters. Refines the *global transform*, not individual point locations.
**Accept** on the P0-T05 fixture with known shift `(0.37, -0.62)`, recovery error < 0.1 px for at least two of the four methods.

### P2-T09 · Ground truth derivation → **reassigned to Member A (see P1-T20)**
It is data work built on label geometry and DEM reprojection, both already owned by A. B consumes the derived ground truth through the tier manifest.

### P2-T21 · AROSICS baseline comparison *(new — Phase 4)*
**File** `scripts/bench_external.py`
**Why** AROSICS is Apache-2.0, purpose-built for sub-pixel co-registration, and **named directly in the PS's own competitor context**. Running against it costs a day and buys a real answer to judge question #4: we benchmarked it rather than assuming it does not apply. It has no lunar validation and no documented hyperspectral support, which is exactly the gap we report.
**Accept** AROSICS runs on the Easy tier and its numbers appear in the report tagged as an external baseline.

### ── Phase 3 — THE DIFFERENTIATORS ────────────────────────────

### P2-T10 · Regime selector ⭐
**File** `matching/regime.py`, `configs/regimes.yaml`
**Why** The evidence is genuinely regime-dependent, and saying so is a stronger position than "we used the newest model". SuperGlue-class methods won on real lunar polar data where SIFT/AKAZE degraded; an independent 2025 SAR-optical benchmark found classical RIFT beating the best deep method at high resolution. Neither result generalises universally. So: route, don't hardcode.
**Routing rules (start here, tune with the ablation):**

| Condition | Regime | Matcher |
|---|---|---|
| same modality, Δincidence < 30°, scale ratio < 4 | `same_modal_normal` | LightGlue |
| same modality, Δincidence ≥ 30° **or** incidence > 70° (polar-like) | `same_modal_polar` | LightGlue/LoFTR **on phase-congruency input** |
| cross-modality (pan ↔ hyperspectral/multiband) | `cross_modal` | RIFT / MIM classical descriptor |
| scale ratio ≥ 4 | `extreme_scale` | cascade (P2-T11) + affine simulation |

Selection is driven by the **incidence/phase angle extracted upstream**, not by a hardcoded instrument pair.
**Accept** a polar-illumination pair routes to the polar path; an OHRC↔IIRS request routes to the cascade, never to a direct match.

### P2-T11 · Scale-bridging cascade ⭐
**File** `matching/cascade.py`
**Why** No matcher in the literature has documented evaluation at 100×+ scale ratios — it is a field-wide gap. Every real Chandrayaan-2 registration paper deliberately chose resolution-matched pairs. Two independent teams measured **0 inliers** attempting direct IIRS correspondence. So we stage.
**Stages**
```
OHRC  ──(~20:1, precedent exists)──→  TMC-2
TMC-2 ──(~2:1,  the open gap)─────→  SELENE TC
TMC-2 ──(~16:1)───────────────────→  IIRS
any   ─────────────────────────────→  LRO NAC / WAC
```
**Steps** Plan the stage graph from GSD ratios; run each hop; compose transforms; **propagate uncertainty across hops** (a 3-hop chain must not report a 1-hop error bar); expose the **OHRC-as-geodetic-bridge** mode — register OHRC↔NAC first for absolute geolocation control, then tie TMC-2/IIRS into that same frame via their internal geometric relationship to OHRC (same spacecraft, same orbit, near-simultaneous acquisition). **Flag this bridge as a design hypothesis to validate during the build, not as a literature-backed finding.**
**Accept** a direct OHRC↔IIRS request is intercepted and routed through TMC-2; composed transform on synthetic data matches the analytic composition; per-hop and composed error are both reported.

### P2-T12 · Geometry-guided outlier filtering ⭐⭐ *(the novel layer)*
**File** `estimate/geometry_filter.py`
**Why** This is the most defensible technical opportunity in the whole analysis: no source demonstrates a general "compute per-pixel incidence/emission/phase angle and DEM slope, then use it to directly constrain feature-descriptor search or reweight candidate matches" pipeline for cross-sensor lunar images. Precedents exist where geometry *drives* correspondence (photoclinometry-assisted matching between differently-illuminated NAC images; multi-view shape-from-shading; DEM/terrain-shape correlation), and separately where geometry only *predicts* matchability (stereo-pair acquisition planning; crater-detectability studies). Nobody has built the general runtime version.
**Steps**
1. For each candidate match, sample DEM slope/aspect and incidence/emission/phase at both endpoints.
2. **Reject** matches whose local terrain geometry is inconsistent between source and reference beyond a tolerance (e.g. a match linking a steep sun-facing slope to a flat shadowed plain).
3. **Reweight** (don't only reject) surviving matches by geometric plausibility, feeding the weight into MAGSAC.
4. Use DEM consistency to **disambiguate repetitive terrain** (failure mode #11) where photometric evidence alone is ambiguous.
5. Degrade gracefully: with no DEM available, become a no-op and say so in `notes`.
**Accept** on a repetitive-crater-field subset, false-positive inliers drop measurably versus RANSAC-only, and **inlier precision** (not just count) is the reported metric. This experiment is Ablation Experiment 3 — it is the novel-contribution measurement, so it must be isolated cleanly.

### P2-T22 · Mutual information cross-modal similarity *(Phase 3)*
**File** `matching/similarity.py`
**Why** The document names MI as the classical gold standard for cross-modal similarity, with a real precedent in Sentinel SAR/optical sub-pixel registration at 1.0–2.3 px. It has **no inherent scale or rotation invariance**, so it is a *validation score* for cross-modal alignment quality where no ground truth exists, not an initial correspondence method. Do not confuse the two roles.
**Accept** MI on an aligned cross-modal pair exceeds MI on a deliberately misaligned one; exposed as a post-hoc alignment-quality metric.

### P2-T23 · ECC alignment refinement *(Phase 2)*
**File** `refine/subpixel.py` → `cv2.findTransformECC`
**Why** The document states ECC's brightness/contrast normalisation is purpose-built for exactly the PS's illumination-variation challenge — but that it needs a reasonable initial alignment to converge, so it belongs in **refinement, not initial correspondence**.
**Accept** reduces residual error on an illumination-shifted pair that already has a coarse alignment; degrades gracefully when the initial alignment is poor.

### P2-T24 · Crater-based structural prior *(Phase 3, P2 priority)*
**File** `matching/craters.py`
**Why** The document flags crater-based structural matching as a recurring **lunar-specific** technique absent from general remote-sensing literature: a 2025 crater-neighbourhood-structure paper plus multiple competitor repos independently converge on craters as stable, illumination-robust landmarks. A domain prior terrestrial matchers do not have.
**Accept** crater-neighbourhood consistency rejects matches RANSAC alone accepts on a repetitive-terrain subset; no-ops cleanly when too few craters are detected.

### P2-T13 · RIFT / cross-modal classical path *(the port is already done for us)*
**Files** `third_party/rift2/` (vendored), `matching/rift.py` (thin wrapper)
**Steps** **`RIFT2-python` exists** — a Python implementation of RIFT2 built on PhasePack, optimised for high-resolution acquisitions. The research file said "MATLAB, needs porting"; that is out of date. **Vendor it** into `third_party/rift2/` with attribution and a pinned commit SHA (it has no tagged releases, so a pip dependency could shift under us). Our `matching/rift.py` wraps it to the `MatchSet` contract and feeds it the phase-congruency maps from P1-T13.
**Validate before trusting it:** run it on a fixture and confirm the descriptor behaves as the RIFT2 paper describes. A small unvalidated port is a risk; a small *validated* one is a gift.
**Accept** beats SIFT on a synthetic cross-modal pair (strong nonlinear intensity remap) by inlier ratio; vendored SHA recorded in `docs/LICENSE_AUDIT.md`.

### P2-T14 · Extreme-rotation & large-displacement handling
**File** `matching/filters.py`
**Steps** Orientation-histogram sanity check pre-RANSAC → if extreme, run **ASIFT-style affine simulation** or a rotation-invariant descriptor (failure mode #6). On match-count collapse, widen the search using the **DEM/SPICE-derived coarse geolocation prior** before fine matching (failure mode #7). Adaptive Lowe-ratio threshold driven by `repetitiveness_score` (failure mode #11).
**Accept** a 60°-rotated synthetic pair that plain SIFT fails on is recovered.

### ── Phase 4 — VALIDATION ─────────────────────────────────────

### P2-T15 · Control gates ⭐ *(the honesty infrastructure)*
**File** `evaluate/control_gates.py`
**Why** A real, documented failure in this exact competitive field: a team caught its own bug producing **fake 100% inlier rates from a shared zero-mask**. We design the gate in as a pipeline stage, not an afterthought.
**Gates — all run on every benchmark execution:**

| Gate | Test | Pass condition |
|---|---|---|
| `null_constant_grey` | match a constant-grey image against the reference | near-zero inliers; a "success" here means the pipeline is broken |
| `null_pure_noise` | match random noise against the reference | near-zero inliers |
| `perturbation_sensitivity` | apply a known 5 px shift | recovered transform moves by ~5 px, not 0 and not 50 |
| `shared_mask_guard` | assert src and ref masks are distinct objects | no accidental aliasing |
| `identity_sanity` | match an image against itself | RMSE ≈ 0, inlier ratio ≈ 1 |
| `scale_consistency` | estimated scale vs known GSD ratio | within tolerance |

**Accept** `RegistrationResult.gates` is non-empty on every run; `report.py` refuses to render a result with empty gates (rule H4).

### P2-T16 · Quality gating & confidence tiers
**File** integrated into the pipeline runner
**Tiers** `HIGH` / `MEDIUM` / `LOW` / `REJECTED`, thresholded on inlier count, inlier ratio, coverage and gate results. A RANSAC "success" with an inlier count below threshold is **not** trustworthy and must be tiered down (failure mode #12). Known-hard regimes (direct IIRS) are flagged explicitly rather than silently degrading.
**Accept** a deliberately impossible pair returns `REJECTED` with a populated `failure_modes` list — and the pipeline exits 0, because a correct rejection is a correct result.

### P2-T17 · Benchmark runner
**File** `evaluate/benchmark.py` — implements §12.1's four tiers.
**Accept** `make bench` produces `reports/benchmark_<date>.json` with per-tier RMSE / inlier count / inlier ratio / coverage / runtime, plus every gate result.

### P2-T18 · Ablation runner → **reassigned to Member C (see P3-T16)**
The runner composes B's stages through config and tabulates the result, which is reporting work. B exposes each stage as an independently toggleable config flag; C drives them.

### P2-T19 · Failure-mode register & logger → **reassigned to Member C (see P3-T14)**
B still raises failure-mode IDs from inside the pipeline; C owns the register, the logger and the document that renders them.

### P2-T20 · External-validity comparison *(Phase 5, B + C)*
B runs the Easy and Medium tiers; **C renders the comparison** (see P3-T16). Our measured numbers sit next to the published external baselines and the AROSICS run from P2-T21, each clearly tagged `source="external"` vs `source="measured"` (rule H5). Easy/Medium are the **credibility floor** — approaching the published numbers there is what earns the right to be believed on Hard/Extreme, where no external precedent exists to compare against.

---

## 12. BENCHMARK & ABLATION PROTOCOL

### 12.1 The four-tier benchmark *(no existing benchmark fits — we build one)*

No public benchmark combines planetary/lunar imagery, genuine panchromatic-to-hyperspectral registration, this instrument set, and reusable ground truth. HPatches/MegaDepth/IMC/ETH3D are terrestrial; SEN1-2/3MOS/MultiResSAR are Earth SAR-optical; the closest real lunar precedent was internal to one paper and never published as reusable. So a custom benchmark is required, not optional.

| Tier | Pairing | Scale ratio | Illumination Δ | What it proves | Ground truth |
|---|---|---|---|---|---|
| **Easy** | OHRC ↔ LRO NAC, near-equatorial, similar local time | ~2:1 | < 20° | Sanity check — should reproduce the published equatorial result (~0.6 px RMSE class) | SPICE/label geometry + DEM cross-check |
| **Medium** | IIRS ↔ LRO WAC, moderate latitude | ~1.25:1 | 20–60° | Sanity check — should reproduce the published sub-pixel result | Same, plus the published residuals as a calibration anchor |
| **Hard** | **TMC-2 ↔ LRO NAC** or **TMC-2 ↔ SELENE TC** | 10:1 / ~2:1 | 60°+ | **This tier IS the contribution** — genuinely unattempted in real literature | SPICE geometry + independent DEM check (no prior ground truth exists) |
| **Extreme** | OHRC ↔ IIRS (cross-modal, extreme scale) | ~320:1 | any | Included **specifically to report degradation honestly**, not to claim success — two independent teams already failed here with real data | SPICE geometry only |

For every pair: ground truth derived from PDS4 label geometry, **cross-validated against an independent DEM reprojection** rather than trusted from one source; match points hand-verified on a small held-out subset; error bounds set conservatively from the closest real precedent per tier (Hard tier's bound is deliberately left open/exploratory since no precedent exists); metrics = **RMSE + inlier count + inlier ratio + spatial coverage**, exactly as the PS specifies.

### 12.2 The six-stage ablation

| Stage | Configuration | What it isolates | Metrics |
|---|---|---|---|
| **Baseline** | SIFT + RANSAC | the field's own documented floor; expected to degrade sharply at poles | RMSE, inlier count, inlier ratio, runtime, coverage, failure rate |
| **Exp 1** | + illumination normalisation (CLAHE / phase congruency) | does radiometric preprocessing help? | same, **split equatorial vs polar** |
| **Exp 2** | + deep features (learned matcher in place of SIFT) | does the learned path help, and by how much? | same, compared against the published SuperGlue-vs-SIFT gap as an external check |
| **Exp 3** | + **geometry constraints** (DEM-slope / incidence-aware filtering) | ⭐ **the novel-contribution measurement** — no comparable published ablation exists | **inlier precision** (not just count), false-positive rate on a repetitive-terrain subset |
| **Exp 4** | + **coarse-to-fine cascade** | ⭐ expected to be the difference between "0 inliers" and a working result on the hard tiers | success rate on **Hard/Extreme tiers specifically**, not easy-tier averages |
| **Exp 5** | + sub-pixel refinement | sub-pixel RMSE reduction vs integer-pixel-only | RMSE at sub-pixel resolution; synthetic-shift recovery error |
| **Final** | complete system | should exceed every individual experiment on Hard/Extreme, while at minimum matching the published numbers on Easy/Medium | all metrics, all tiers, explicit floor-vs-target labelling |

**Experiments 3 and 4 are expected to matter most** — they target the two largest genuine gaps found (geometry-guided filtering, scale bridging). Design them to be isolable and clean.

### 12.3 Proving sub-pixel accuracy honestly

The PS demands *demonstrated* sub-pixel accuracy, which needs ground truth finer than one pixel — and no public sub-pixel ground truth exists for this combination. Two converging approaches, both used:

1. **Synthetic self-validation** — apply a *known* sub-pixel shift to a real Chandrayaan-2 image, then measure recovery error against that exactly-known shift.
2. **DEM-based independent geometric cross-validation** — as in §12.1.

Two independent competitor teams arrived at this same pattern separately, which is reasonable convergent evidence that it is the sound approach rather than one team's preference.

---

## 13. PART 3 — MEMBER C · DETAILED TASKS *(two features)*

> Start Day 1 against the mock objects from P0-T05. Switch to real data at **GATE B**.

### ══ FEATURE 3.1 — Deliverable Export & Evaluation Report ══

### P3-T01 · CLI skeleton *(Phase 1)*
**File** `cli.py`
```bash
chandralign register --src <ch2_product> --ref <reference_product> \
                     --out runs/<name> [--config configs/default.yaml] [--cpu]
chandralign benchmark --tiers easy,medium,hard,extreme
chandralign ablate    --stages all
chandralign report    --run runs/<name>
chandralign demo
```
**Accept** `--help` documents every flag; `register` runs end-to-end on mocks and writes a run directory.

### P3-T02 · Match-point export *(Phase 1)* — **PS deliverable**
**File** `product/matchpoints.py`
**Outputs**
- `matches.csv` — `idx, src_x, src_y, ref_x, ref_y, src_lat, src_lon, ref_lat, ref_lon, confidence, is_inlier, method, regime, cascade_stage, grid_cell`
- `matches.geojson` — reference-frame points with properties, so they drop straight into QGIS
Coordinates are written at **sub-pixel precision** (float, not rounded) — that is the whole point.
**Accept** CSV round-trips into a `MatchSet` without loss; GeoJSON opens in QGIS with correct lunar CRS.

### P3-T03 · Registered product export *(Phase 2)* — **PS deliverable**
**File** `product/warp.py`
**Steps** Apply `TransformModel` to the source (affine/homography via OpenCV or TPS via a dense displacement field) → write a **GeoTIFF with a correct geotransform and lunar CRS** via rasterio/GDAL → write a sidecar `.json` carrying `SceneMeta` for both inputs, the transform, and the metrics.
**Accept** the registered product opens in QGIS aligned to the reference; a round-trip warp-then-inverse-warp returns within sub-pixel error.

### P3-T04 · Provenance manifest *(Phase 2)*
**File** `product/provenance.py`
Every output carries: input product IDs + checksums, config hash, package versions, git commit SHA, UTC timestamp, host/GPU, and the `ship_mode` flag. This is the reproducibility backbone.
**Accept** two runs from the same commit and inputs produce identical manifests except the timestamp.

### P3-T05 · Visualisations *(Phase 2)*
**Files** `viz/sidebyside.py`, `viz/match_plot.py`, `viz/coverage_plot.py`, `viz/swipe.py`
- **side-by-side** with Δsun-angle annotated (this is the visual that makes the problem obvious in 5 seconds)
- **match overlay** — inliers vs outliers colour-coded
- **coverage heatmap** — the N×N grid occupancy, which *shows* the uniformity mandate being met
- **swipe / checkerboard** before-vs-after registration
**Accept** all four render from a mock `RegistrationResult` and from a real one.

### P3-T06 · HTML evaluation report *(Phase 2, polished Phase 4)*
**File** `product/report.py` (Jinja2)
**Sections** Inputs (with `SceneMeta`) → pipeline path taken (regime, cascade stages) → **the four PS metrics** → coverage figure → sub-pixel validation result → **control-gate table** → confidence tier → failure modes triggered → provenance.
**Hard requirements**
- refuses to render if `gates` is empty (rule H4)
- every number is tagged `measured` or `external` (rule H5)
- `None` renders as `not measured`, never as a blank or a zero (rule H1)
**Accept** `make report` produces a self-contained HTML; a mocked empty-gates result raises rather than rendering.

### ══ FEATURE 3.2 — Demo UI ══

### P3-T07 · FastAPI backend *(Phase 2–3)*
**File** `api.py` — `POST /register` (async job), `GET /runs/{id}` (status + metrics), `GET /runs/{id}/assets/{name}`, `GET /pairs` (curated benchmark pairs).
**Accept** returns a valid result for a curated pair in under the configured timeout; streams progress.

### P3-T08 · Web UI *(Phase 3, polished Phase 5)*
**File** `ui/`
**Screens** (1) pick a curated pair or upload, (2) live pipeline progress showing which regime/cascade path was chosen, (3) results — side-by-side, match overlay, coverage heatmap, swipe, metric cards, confidence-tier badge.
**Non-negotiable UI state:** a `REJECTED` run renders a clear **"REJECTED — reason"** panel. Hiding failures in the UI would undo the entire honesty discipline the project is built on.
**Accept** full flow works offline from cached runs (no live network at demo time — assume the venue Wi-Fi fails).

### ══ FEATURE 3.3 — Metrics & Failure Logging *(moved from Member B)* ══

### P3-T13 · Metrics module *(was P2-T03)*
**File** `evaluate/metrics.py`
**Why it moved here** Pure computation from a `MatchSet` plus an inlier mask, with no dependence on B's internals — and C owns every surface that renders these numbers. Owning the computation lets honesty rules H1 and H5 be enforced in one place instead of across a handoff.
**Steps** `rmse_px`, `rmse_m` (= `rmse_px × gsd_m`), `inlier_count`, `inlier_ratio`, `spatial_coverage`, `max_delaunay_gap_px`, `runtime_s`. **Every unmeasured field returns `None`** (rule H1). `scipy.spatial.Delaunay` for the gap diagnostic.
**Accept** `tests/test_honesty.py::test_metrics_default_none` — a `Metrics()` with no computation has every field `None`, and `report.py` refuses to render a number without a `source` tag.

### P3-T14 · Failure-mode register & logger *(was P2-T19)*
**Files** `evaluate/failure_log.py`, `docs/FAILURE_MODES.md` — the 20-mode table in §15, with every observed runtime failure tagged to a mode ID and surfaced in `RegistrationResult.failure_modes`. Member B raises the IDs from inside the pipeline; C owns the register and its rendering.
**Accept** each of the 20 modes has a detection hook or an explicit "detected manually only" note.

### P3-T15 · Dependency licence checker *(new)*
**File** `scripts/check_licences.py`
**Why** We now depend on ten-plus third-party packages and one vendored repo. Failure mode #14 (licence contamination) is no longer theoretical.
**Steps** Walk the installed environment, emit package → licence → ship/no-ship, fail CI on any non-permissive entry, and assert the vendored `third_party/rift2/` SHA matches the one recorded in `docs/LICENSE_AUDIT.md`.
**Accept** runs in CI on every PR; a deliberately added GPL package fails the build.

### P3-T16 · Ablation runner & comparison tables *(was P2-T18 / P2-T20)*
**Files** `evaluate/ablation.py`, plus the comparison section of `product/report.py`
**Why it moved here** The ablation runner composes Member B's stages through config flags and tabulates the output. That is reporting work, and C already owns every table a judge will read. B's job is to make each stage independently toggleable; C's job is to drive them and render the result.
**Steps** Implement §12.2's six stages as a config-driven sweep. Render three tables: per-stage ablation, per-tier benchmark, and the external-validity comparison placing our measured numbers beside the published baselines and the AROSICS run, each tagged `measured` or `external` (rule H5).
**Accept** `make ablate` produces a per-stage table isolating each component's contribution. **The ablation exists to test whether each component helps, not to assume it does** — a stage that does not help must be reported as not helping, and the report must render that plainly rather than omitting the row.

### ══ Docs & demo *(Phase 4–5)* ══

### P3-T09 · `docs/JUDGE_QA.md`
The 40-question defence set, answered with **this project's own evidence**, not generic CV answers. §14 below is the seed content — expand each with the measured number once it exists.

### P3-T10 · `docs/REPRODUCIBILITY.md` + `scripts/reproduce.sh`
Dataset list with access notes (PRADAN needs SSO; NASA/JAXA/USGS need none), pinned environment, full licence-audited dependency list, model/weights provenance, training procedure (**"pretrained weights, no fine-tuning"** unless we actually fine-tune, in which case epochs and loss curves go here), evaluation scripts, ground-truth derivation, expected results per tier.
**Accept** `bash scripts/reproduce.sh` regenerates every number in the report from a clean clone.

### P3-T11 · `docs/LICENSE_AUDIT.md`
The §2.3 table, plus the explicit statement: the deliverable is buildable **entirely from permissively-licensed components** (Apache-2.0 / MIT / BSD); the non-commercial matchers appear only in an internal benchmark that is excluded from the shipped bundle. This is a deliberate design choice, made to remove licence risk from anything ISRO might redistribute.

### P3-T12 · `scripts/run_demo.py` — the 3-minute demo *(Phase 5)*

```
0:00–0:20  Input: real CH-2 OHRC + real LRO NAC of the same ground region,
           visibly different Sun angle, shown side-by-side.
0:20–0:45  Traditional SIFT on screen → few, scattered, partly wrong matches.
           State the documented reason (classical detectors degrade at lunar
           poles) as a cited finding, not an unsupported claim.
0:45–1:30  Our method on the SAME pair → dense, spatially distributed,
           high-confidence matches; visibly uniform grid coverage, not
           clustered; sub-pixel refinement pass shown; warped output
           overlaid on the reference.
1:30–2:00  Live metrics, all measured at build time:
             RMSE [X.XX] px   (floor to beat: the published SuperGlue
                               result, 0.57–0.92 px equatorial→polar)
             Inliers [XXXX] · Inlier ratio [XX]% · Spatial coverage [XX]%
2:00–2:40  ⭐ THE DIFFERENTIATOR: the same demo on the neglected axis —
           TMC-2 ↔ SELENE Terrain Camera, a pairing with ZERO prior
           published attempt — with honest reporting if the result is
           weaker than the solved pairs.
2:40–3:00  Close: one slide — what's already solved (OHRC↔NAC, IIRS↔WAC,
           matched against published literature) vs what this system adds
           (TMC-2 / SELENE coverage no prior work demonstrates), plus the
           licence + reproducibility summary.
```
**Every bracketed placeholder is replaced with our own measured number from the exact same evaluation protocol — never invented.**
**Accept** runs twice in a row from a clean clone with no network, in under 3 minutes.

---

## 14. JUDGE DEFENCE — the 40 questions *(seed for `docs/JUDGE_QA.md`)*

| # | Question | Answer |
|---|---|---|
| 1 | Why isn't SIFT enough? | Documented: SIFT degrades specifically at lunar poles in the only paper that tested it on real Chandrayaan-2 data. We keep it as the baseline floor, not the primary matcher. |
| 2 | Why isn't LoFTR enough? | Vanilla LoFTR is documented to fail near-completely under extreme sun-angle/altitude change on Mars-analog terrain. It needs the geometry-aided path our architecture provides, not blind off-the-shelf use. |
| 3 | Why do you need AI? | We don't, universally — a 2025 benchmark found classical RIFT beating the best deep method at high resolution. Our architecture is **regime-selected**: AI where the evidence shows it wins (polar illumination), classical elsewhere. |
| 4 | Why can't AROSICS solve this? | Real, permissive, purpose-built for sub-pixel co-registration — but no lunar/planetary validation found and no documented hyperspectral support. We use it as a benchmark comparison, not an assumed solution. |
| 5 | Why can't ArcGIS solve this? | Generic Earth-imagery co-registration; no lunar sensor models, no illumination-invariance design. |
| 6 | Why can't USGS ISIS solve this? | Closest architectural precedent, but confirmed fragile on stock releases for OHRC — `spiceinit` fails without manual template files. It's a component we integrate, not a turnkey solution. |
| 7 | Why can't OpenCV solve this? | OpenCV provides the primitives everything here (including us) is built on. It has zero domain-specific illumination robustness or hyperspectral handling on its own. |
| 8 | What exactly is novel? | A generic pipeline validated specifically on the two neglected axes (TMC-2, SELENE), which no real publication or competitor demonstrates, plus geometry-guided outlier filtering with no general algorithmic precedent. |
| 9 | What exactly is multimodal? | OHRC (pan, 0.25 m), TMC-2 (pan, ~5 m), IIRS (hyperspectral, ~80 m, ~250–256 bands) against LRO NAC/WAC (pan, 0.5–100 m) or SELENE (optical/multiband, ~7–62 m) — five distinct sensor types across a ~320:1 scale range. |
| 10 | How do you handle IIRS? | Synthesise a panchromatic-equivalent composite from high-SNR bands before matching, rather than matching raw hyperspectral cubes — informed by two independent teams' real failures doing the latter. |
| 11 | How do you handle spectral differences? | Band-composite synthesis plus modality-independent descriptors (MIND/RIFT-family), which are purpose-built for nonlinear radiometric differences. |
| 12 | How do you handle Sun-angle differences? | Regime-selected matching driven by the extracted incidence/phase angle, routing polar/high-incidence cases to the path with documented favourable lunar evidence. |
| 13 | How do you handle shadows? | Shadow-aware masking, and where geometry permits, photoclinometry-style use of known sun geometry to actively assist rather than merely tolerate the shadow. |
| 14 | How do you handle huge resolution differences? | Hierarchical staged bridging; never direct OHRC↔IIRS — informed by two independent 0-inlier failures at exactly that pairing. |
| 15 | How do you obtain ground truth? | SPICE/PDS-label geometry cross-validated against an independent DEM source, never trusted from a single source, because no public ground-truth benchmark exists for this combination. |
| 16 | How do you prove sub-pixel accuracy? | Synthetic self-validation (known sub-pixel shift, measured recovery error) plus DEM cross-validation — the honest answer given no public sub-pixel ground truth exists. |
| 17 | How do you prevent false matches? | Spatial-uniformity-aware filtering, geometry-guided outlier rejection using terrain relief, and stricter ratio thresholds in detected repetitive terrain. |
| 18 | How do you guarantee spatial uniformity? | Grid-based retention with per-cell top-k plus an empty-cell secondary pass, reported as a quantitative coverage metric. |
| 19 | What if there are no keypoints? | Featureless-terrain fallback to phase-congruency descriptors; explicit low-confidence tier rather than silent failure. |
| 20 | What if the terrain is repetitive? | Stricter ratio-test threshold plus DEM-consistency disambiguation. |
| 21 | What if metadata is wrong? | Independent DEM cross-validation catches it rather than trusting metadata blindly. |
| 22 | What if overlap is small? | Pre-filter candidate pairs by footprint intersection via PDS/ODE geometry before attempting to match at all. |
| 23 | How much GPU is required? | ~8 GB VRAM class per the recommended methods' own published benchmarks; classical/lightweight path runs on CPU. |
| 24 | Can it run on a laptop? | Yes for the classical and LightGlue/XFeat paths (both confirmed CPU-capable); GPU strongly preferred for the learned regime path. |
| 25 | Can it process full-resolution images? | Only via tiled processing — full CDR NAC products reach 104,448 × 5,000 px, exceeding any single-pass inference budget. |
| 26 | How long does inference take? | Published per-method figures: LightGlue 44.2 ms–150 FPS, SuperGlue ~69 ms/pair, LoFTR 116–130 ms/pair. **End-to-end pipeline time including tiling is measured at build time**, not estimated. |
| 27 | What datasets did you actually use? | PRADAN CH-2 (OHRC/TMC-2/IIRS), LROC PDS Data Node (NAC/WAC), JAXA DARTS SELENE PDS3, LOLA/SLDEM/Kaguya-ARD for DEM ground truth. |
| 28 | Are they actually free? | Yes. PRADAN requires SSO login but is free for non-profit scientific use; NASA/JAXA/USGS sources need no login at all. |
| 29 | What is the licence? | Buildable entirely from Apache-2.0 / MIT / BSD components; explicitly avoids the non-commercial restrictions on SuperPoint/SuperGlue/R2D2 in the shipped path. |
| 30 | Can ISRO legally use it? | Yes for the permissive default path. Non-commercial matchers, if used, are internal-benchmark-only and excluded from the deliverable. A deliberate design choice, not an oversight. |
| 31 | Can another team reproduce it? | Yes — `docs/REPRODUCIBILITY.md` plus `scripts/reproduce.sh`; every dataset, licence and dependency is individually verified. |
| 32 | What happens outside your benchmark? | Honestly: unknown for regimes outside the four tiers. Stated as a real limitation rather than concealed. |
| 33 | What is your baseline? | SIFT + RANSAC, matching the field's own documented convention. |
| 34 | What is your ablation study? | The six stages in §12.2, isolating illumination normalisation, deep features, geometry constraints, cascade bridging, and sub-pixel refinement individually. |
| 35 | What is your biggest failure case? | Direct OHRC↔IIRS correspondence without staged bridging. Two independent teams confirm it fails on real data. We report it as a known hard limit, mitigated but not eliminated by bridging. |
| 36 | Why should ISRO use this? | It targets the PS's actual stated dataset, on the two axes the entire competitive field — including its strongest teams — has not solved, with licensing that permits ISRO reuse. |
| 37 | What happens when it fails? | Explicit confidence-tier reporting and quality gating. Never a silent or fabricated success. |
| 38 | Is it explainable? | Partially. The classical/RIFT path is fully interpretable (gradient/phase-based). The learned path is not inherently interpretable, and we do not overclaim there. |
| 39 | Will it work on future missions? | The geometry layer is built around generically-available SPICE/PDS metadata patterns rather than CH-2-specific hacks — a reasonable expectation, not a verified claim. |
| 40 | What makes this more than a hackathon demo? | Reproducibility discipline, honest failure-mode reporting matched to real observed failures, and a specific checkable claim about what is actually new rather than a vague novelty assertion. |

---

## 15. THE 20 FAILURE MODES *(→ `docs/FAILURE_MODES.md`, owner: Member B)*

Each must have a **detection hook** in code and a **mitigation** in the pipeline. Mode IDs appear in `RegistrationResult.failure_modes`.

| # | Failure | Cause | Detection | Mitigation | Owner |
|---|---|---|---|---|---|
| 1 | Featureless terrain (mare) | low-relief, weak texture | keypoint count below threshold pre-matching | fall back to phase-congruency/self-similarity descriptors; widen search window | A+B |
| 2 | Severe shadows / crater interiors | near-total intensity loss | region intensity-variance check | shadow-aware masking; separate low-confidence tier | A |
| 3 | High-res vs low-res (~320:1) | scale mismatch beyond any validated matcher range | pre-flight scale-ratio check vs known GSDs | hierarchical staged bridging, never direct | B |
| 4 | Hyperspectral noise | IIRS band-level SNR variation | per-band SNR estimation | band selection/averaging before feature extraction | A |
| 5 | Spectral mismatch (pan vs hyperspectral band) | no single IIRS band matches a pan response | known from prior work | synthesise a panchromatic-equivalent composite | A |
| 6 | Extreme rotation | pushbroom geometry / orbit-track differences | orientation-histogram check pre-RANSAC | ASIFT-style affine simulation or rotation-invariant descriptors | B |
| 7 | Large displacement | metadata geolocation error | match-count collapse at default radius | widen search via DEM/SPICE coarse geolocation prior | B |
| 8 | Incorrect / stale metadata | inaccurate PDS label geometry | cross-check against independent DEM geometry | never trust one metadata source | A+B |
| 9 | Wrong/missing map projection | CH-2 raw products carry corner lat/lon only | projection-consistency check on output | explicit GDAL reprojection step, never implicit | A |
| 10 | Insufficient overlap | partial ground coverage | footprint intersection via SPICE/PDS geometry | pre-select genuinely overlapping pairs via ODE/PDS footprint APIs | A |
| 11 | Repetitive terrain | ambiguous self-similarity | high ratio of near-equal top-2 descriptor distances | stricter ratio threshold + DEM/geometric consistency check | B |
| 12 | False correspondences surviving RANSAC | degenerate configurations, low true-inlier count | inlier count below threshold despite "success" | confidence tiers; never treat RANSAC success as unconditional | B |
| 13 | Cross-instrument scale confusion | ambiguous scale search without a prior | estimated scale vs known GSD ratio | constrain scale search using PDS-derived GSDs, not blind pyramid search | B |
| 14 | Non-commercial licence contamination | using restricted weights in a redistributable | dependency licence audit | ship only permissive components; gate the rest | C |
| 15 | PDS ingestion silently falling back to generic loading | stubbed parser | **unit test asserting real label fields**, not just a successful pixel load | build and test a genuine parser against real products | A |
| 16 | Polar illumination collapse (classical) | documented SIFT/AKAZE polar degradation | solar-elevation check from metadata | regime-aware routing to the learned path | B |
| 17 | Deep-matcher collapse under extreme sun angle | vanilla LoFTR fails at very low elevation | same solar-elevation check | don't deploy vanilla LoFTR for extreme illumination; prefer the documented-strong path | B |
| 18 | ISIS/ASP camera-model fragility | `spiceinit` fails on stock releases for OHRC | build/CI failure on the planetary-tooling step | budget explicit integration time; containerise; keep the pipeline functional without ISIS | A |
| 19 | Fabricated / inflated evaluation numbers | e.g. a shared zero-mask producing fake 100% inlier rates | **control gates**: null test, perturbation sensitivity | control-gate discipline as a designed pipeline stage | B |
| 20 | Token "fine-tuning" presented as a real capability | 1-epoch training described in marketing terms | compare claimed capability against committed training logs | report training depth honestly, or state "no fine-tuning" | C |

---

## 16. DEPENDENCY GRAPH & HANDOFF GATES

```
PHASE 0 ─── contracts.py frozen ──────────────────────────────────────┐
                                                                      │
 A: P1-T01 PDS4 ─┬─ P1-T03 raster ─┬─ P1-T05 tiling ─┐                │
    P1-T02 PDS3 ─┘  P1-T04 registry┘                 │                │
                                                     ▼                │
                                            ╔══════════════════╗      │
                                            ║  GATE A          ║      │
                                            ║  real ImagePlane ║      │
                                            ║  (OHRC + NAC)    ║      │
                                            ╚════════╤═════════╝      │
 A: P1-T07 solar ─┐                                  │                │
    P1-T08 DEM ───┼─→ P1-T09 slope/aspect ───────────┤                │
    P1-T10 footprint                                 │                │
    P1-T12..16 preprocessing ───────────────────────►┤                │
                                                     ▼                │
 B: P2-T01 iface ─→ P2-T02 SIFT ─→ P2-T04 LightGlue ─→ P2-T06 RANSAC  │
                                                     │                │
                    P2-T07 uniformity ───────────────┤                │
                    P2-T08 subpixel ─────────────────┤                │
                    P2-T09 groundtruth ──────────────┤                │
                                                     ▼                │
                                            ╔══════════════════════╗  │
                                            ║  GATE B              ║  │
                                            ║  RegistrationResult  ║  │
                                            ║  on real OHRC↔NAC    ║  │
                                            ╚════════╤═════════════╝  │
 B: P2-T10 regime ──┐                                │                │
    P2-T11 cascade ─┼─ needs P1-T07 (angles)         │                │
    P2-T12 geo-filt ┘  needs P1-T09 (slope/aspect)   │                │
    P2-T13 RIFT ────── needs P1-T13 (phase congruency)                │
 A: P1-T17 TMC-2 + SELENE products ──────────────────┤                │
                                                     ▼                │
                                            ╔══════════════════════╗  │
                                            ║  GATE C              ║  │
                                            ║  TMC-2 ↔ SELENE TC   ║  │
                                            ║  running end-to-end  ║  │
                                            ╚════════╤═════════════╝  │
 A+B: P1-T18 tiers · P2-T15 gates · P2-T17 bench · P2-T18 ablation    │
                                                     ▼                │
 C: P3-T01..06 (mock-first, real after GATE B) ──────┴────────────────┘
    P3-T07..08 UI · P3-T09..11 docs · P3-T12 demo
```

**Gate rules**
- **GATE A** blocks Member B's *real-data* work only. B works on synthetic pairs until it opens. B must not wait idle.
- **GATE B** blocks Member C's *real* report/UI. C works on mocks until it opens. C must not wait idle.
- **GATE C** blocks Phase 4 entirely. If GATE C slips, cut scope from Extreme tier first, then from the RIFT path — **never** from the control gates or the honest reporting, which are the project's spine.

---

## 17. RISK REGISTER

| ID | Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| **R1** | **PRADAN SSO approval delays CH-2 data access** — mandatory Keycloak login, possibly with a manual approval step | High | **Critical — blocks everything** | **Register on Day 1.** Meanwhile build against LRO NAC / SELENE / DEM (all no-login) and against competitor-committed real PDS4 label samples so the parser is testable regardless | A |
| **R2** | No public ground-truth benchmark exists for this combination | Certain | High | Build our own 4-tier benchmark with dual-source (label × DEM) ground truth, and say clearly that we built it | A+B |
| **R3** | ISIS3 / ASP fragility — `spiceinit` documented to fail on stock releases for OHRC without manual template files | High | Medium | Containerise; budget explicit integration time; **keep the whole pipeline functional without ISIS** — treat rigorous camera models as an enhancement, not a dependency | A |
| **R4** | Hard tier (TMC-2 ↔ external) simply doesn't work — the best competitor measured 2.8% inlier ratio | Medium | High | It is a *known-hard* tier, which is why it's the contribution. Report the real number whatever it is; the cascade and geometry filter are the two levers. A weak but honest Hard-tier number still beats a fabricated strong one | B |
| **R5** | Scope creep on Part 2 | High | Medium | Member A joins Part 2 from Phase 3 (§8). Cut order if needed: Extreme tier → RIFT path → SELENE MI → TPS | all |
| **R6** | Demo machine has no GPU / venue network fails | Medium | High | CPU path (LightGlue/XFeat/classical) tested every phase; demo runs fully offline from cached runs | C |
| **R7** | Full-resolution products blow memory | Medium | Medium | Tiling from Phase 1, not retrofitted. `memmap` + windowed reads only | A |
| **R8** | Licence contamination in the deliverable | Low | **Critical** | `ship_mode` gate + `LICENSE_AUDIT.md` + a CI dependency-licence check | C |
| **R9** | A metric turns out to be a bug, not a result | Medium | **Critical** (credibility) | Control gates on every run (P2-T15). This exact bug class has already bitten a competitor publicly | B |
| **R10** | Crowded field — ~49 competing teams on this PS | Certain | Medium | Differentiate on the parts actually still open (TMC-2, SELENE, validated ingestion, honest gating), not on UI polish everyone has | all |
| **R11** | **Third-party dependency breaks or shifts mid-build** — `vismatch` is our matcher layer, `RIFT2-python` has no tagged releases | Medium | High | Pin every version in the lockfile on Day 1; **vendor `RIFT2-python`** into `third_party/` at a fixed SHA; keep the `matching/adapter.py` boundary thin so `vismatch` is swappable for kornia in a day if needed | B |
| **R12** | **`pds4_tools` rejects ISRO labels** — it expects labels conforming to the PDS4 standard, and CH-2 uses mission-specific dictionaries | Medium | Medium | The `lxml` per-field fallback in P1-T01 is designed for exactly this. Test against a real ISRO label in Phase 1, not Phase 4 | A |
| **R13** | **`planetaryimage` is alpha-quality** — its own README says so, and a `^IMAGE` failure on detached labels is documented | Medium | Low | Never the primary path. `pvl` for labels + our own memmap for pixels is the trusted route; `planetaryimage` is opportunistic only (P1-T03) | A |
| **R14** | "You just glued libraries together" from a judge | Medium | Medium | True and correct, and the answer is prepared: §3.0's policy plus the Tier-3 list of what is genuinely ours (regime selector, cascade, geometry-guided filter, uniformity enforcement, control gates, the benchmark). Add this to `JUDGE_QA.md` as question #41 | C |

---

## 18. DEFINITION OF DONE

**Minimum viable (must have — without these we have not answered the PS):**
- [ ] Real PDS4 + PDS3 parsing on real downloaded products, passing the anti-stub test
- [ ] OHRC ↔ LRO NAC registration working end-to-end on real data
- [ ] Match points exported (CSV + GeoJSON) at sub-pixel precision
- [ ] Registered product exported as a georeferenced GeoTIFF
- [ ] RMSE, inlier count, inlier ratio, spatial coverage — all measured, none invented
- [ ] Spatial uniformity enforcement demonstrably working (before/after coverage figure)
- [ ] Sub-pixel refinement with synthetic-shift recovery error reported
- [ ] Control gates running on every benchmark execution
- [ ] CLI + HTML report

**Target (the differentiators — this is what wins):**
- [ ] IIRS ↔ LRO WAC reproducing the published sub-pixel class result
- [ ] **TMC-2 ↔ SELENE TC** registered with real products and honest numbers
- [ ] **TMC-2 ↔ LRO NAC** attempted with a real, reported inlier ratio (whatever it is)
- [ ] Scale-bridging cascade working; direct OHRC↔IIRS intercepted and routed
- [ ] Geometry-guided filter measurably improving inlier **precision** (Ablation Exp 3)
- [ ] All 6 ablation stages measured on all 4 tiers
- [ ] Demo UI running offline; 3-minute demo rehearsed twice
- [ ] `JUDGE_QA.md`, `REPRODUCIBILITY.md`, `LICENSE_AUDIT.md`, `FAILURE_MODES.md` complete

**Stretch (only if everything above is green):**
- [ ] IIRS ↔ LRO NAC attempted
- [ ] OHRC ↔ LRO WAC attempted
- [ ] SELENE MI reference support
- [ ] Cross-instrument internal consistency (OHRC↔TMC-2↔IIRS) as a bonus capability
- [ ] Our 4-tier benchmark published as a reusable public artifact (nothing like it exists)

---

## 19. DAILY OPERATING RHYTHM

- **Daily, 15 min:** each member states — task ID finished, task ID in progress, blocker (naming the gate).
- **End of each phase:** run `make test && make bench` on `main`. A phase does not end on a red build.
- **Commit hygiene:** commit message starts with the task ID (`P1-T13: log-gabor phase congruency`).
- **`FEATURES.csv` is updated in the same PR as the code.** A feature marked `DONE` with no passing test is not done.
- **Numbers freeze:** Day 27 (or 10-day Day 10 morning). After the freeze, no metric changes without re-running the full benchmark and re-generating the report.

---

## 20. QUICK REFERENCE — who owns what

| Question | Answer |
|---|---|
| Who owns `io/`, `geometry/`, `preprocess/`, `configs/`, `evaluate/groundtruth.py`? | **Member A (Part 1)** |
| Who owns `matching/`, `estimate/`, `refine/`, `evaluate/{control_gates,benchmark,ablation}.py`? | **Member B (Part 2)** |
| Who owns `product/`, `viz/`, `cli.py`, `api.py`, `ui/`, `docs/`, `evaluate/{metrics,failure_log}.py`? | **Member C (Part 3)** |
| Who can edit `contracts.py`? | Nobody, after Phase 0, without all three agreeing in the PR |
| What blocks Part 2's real-data work? | GATE A — real `ImagePlane` |
| What blocks Part 3's real work? | GATE B — real `RegistrationResult` |
| Do I write this module or reuse a library? | **Check §3.1 first.** Reuse is the default; build only what is not there |
| What is the one thing we must not do? | Report a number we did not measure |
| What is the headline differentiator? | Real TMC-2 and SELENE coverage, the geometry-guided filter, the cascade, and ingestion validated against real products |
| If a judge says "you just glued libraries together"? | Correct — and §3.0 explains why that was the right call. What is ours: regime selector, cascade, geometry-guided filter, uniformity enforcement, control gates, the 4-tier benchmark |

---

## 21. `FEATURES.csv` — the feature register

**85 features, 15 columns, grouped into 9 sections in pipeline order.** Every row says in plain English what the feature does and why it exists, then who owns it, which task delivers it, where the code lives and the test that closes it.

The register lists **capabilities only**. Documents, scripts and project activities are tracked by their task IDs in this file, not as feature rows: data download scripts + PRADAN runbook (P0-T04), local cache + manifest, benchmark tiers and runner (P1-T18, P2-T17), ablation runner (P3-T16), external-baseline comparison (P2-T20, P2-T21), demo runner (P3-T12), `JUDGE_QA.md`, `REPRODUCIBILITY.md`, `LICENSE_AUDIT.md`, `ARCHITECTURE.md`, CI, honesty test suite, mock-data factories (P0-T05) and optional ISIS3 integration.

**Sections**

| Prefix | Section | Rows | Main owner |
|---|---|---|---|
| `DATA` | Data ingestion | 13 | Member A |
| `GEO` | Geometry & physics | 7 | Member A |
| `PREP` | Image preparation | 8 | Member A |
| `MATCH` | Matching | 15 | Member B |
| `ALIGN` | Filtering & alignment | 7 | Member B |
| `PREC` | Precision | 6 | Member B |
| `CHECK` | Trust & safety checks | 10 | Member B (C for failure log) |
| `OUT` | Outputs | 10 | Member C |
| `UI` | Interface | 9 | Member C |

Features marked `*` in their name are the headline differentiators: the regime selector, the scale-bridging cascade, physics-based outlier filtering and the blank-image null test.

**Columns**

| Column | Meaning |
|---|---|
| `id` | section prefix + number, e.g. `DATA-01` |
| `section` | one of the 9 sections above |
| `feature` | short name |
| `what_it_does` | plain-English description of the capability |
| `why_it_matters` | the problem it solves, with the failure-mode number from §15 where one applies |
| `owner` | Member A / B / C, or a pair — matches the reassignments in §7 |
| `priority` | `P0` must ship · `P1` should ship · `P2` stretch |
| `phase` | phase number(s) from §8 |
| `task_ids` | the task(s) in this file that deliver it |
| `code_location` | module path(s) under `src/chandralign/` (or repo-root paths for `configs/`, `tests/`, `scripts/`, `ui/`, `third_party/`) |
| `depends_on` | other feature `id`s that must exist first (`-` = none) |
| `done_when` | the acceptance test that closes it |
| `origin` | `Problem statement` · `Research` (the deep technical research document) · `Team addition` (our own scaffolding, in neither document) |
| `old_ids` | the `REQ-nn` / `ADD-nn` IDs this row replaces, for traceability to older discussions |
| `status` | `TODO` → `WIP` → `DONE` → `VERIFIED` (or `BLOCKED`) |

**Useful filters**

```bash
# everything Member A owns
grep ',Member A' FEATURES.csv

# everything that must ship
grep ',P0,' FEATURES.csv

# one section
grep '^GEO-' FEATURES.csv
```

Edit `FEATURES.csv` directly, in the same PR as the code it describes. Never delete a row; mark it `BLOCKED` with a reason instead.

---

*End of PLAN.md — see `FEATURES.csv` for the full feature register (§21 explains its columns).*
