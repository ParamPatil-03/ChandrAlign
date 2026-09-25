"""FastAPI demo backend. Owner: Member C (Part 3). Feature: UI-02 (P3-T07).

Endpoints
---------
POST /register                  → queue a background registration job
GET  /runs/{run_id}             → job status, progress log, and metrics
GET  /runs/{run_id}/assets/{name} → serve a file produced by that run
GET  /pairs                     → list curated benchmark pairs

Design constraints (PLAN.md P3-T07 & UI-02 in FEATURES.csv)
  - POST /register runs as a background job (non-blocking).
  - Progress is streamed via a lightweight log list, polled by GET /runs/{id}.
  - Every metric value is tagged with a ``source`` field (rule H5).
  - A REJECTED result returns a run record with ``confidence_tier="REJECTED"``
    and a ``failure_modes`` list — never a blank response (rule H3).
  - All measured fields that are unknown stay ``null`` (rule H1).
  - The ``gates`` dict must be non-empty for a result to be considered valid
    (rule H4), and the route exposes it plainly.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Ensure headless Agg backend for matplotlib in background tasks & API server
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg", force=False)
except ImportError:  # pragma: no cover
    pass

# ---------------------------------------------------------------------------
# FastAPI imports — installed under the ``product`` extra.
# We guard the import so that importing *this module* in a context where
# fastapi is absent (e.g. during the base-only unit tests) does not crash.
# ---------------------------------------------------------------------------
try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    # Provide placeholder types so the module-level code below still parses.
    class BaseModel:  # type: ignore[no-redef]
        pass
    def Field(*a, **k):  # type: ignore[override]
        return None


# ---------------------------------------------------------------------------
# In-process job store (replaces a database for the demo; suitable for
# single-process demo server; not for multi-worker production).
# ---------------------------------------------------------------------------
class JobStatus(str, Enum):
    QUEUED   = "QUEUED"
    RUNNING  = "RUNNING"
    DONE     = "DONE"
    FAILED   = "FAILED"


class RunRecord:
    """Mutable record for a single registration job."""

    def __init__(self, run_id: str, pair_id: Optional[str], out_dir: Path,
                 request_payload: dict) -> None:
        self.run_id: str = run_id
        self.pair_id: Optional[str] = pair_id
        self.out_dir: Path = out_dir
        self.request_payload: dict = request_payload
        self.status: JobStatus = JobStatus.QUEUED
        self.progress_log: list[str] = []
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.result_summary: Optional[dict] = None  # extracted from result.json
        self.error: Optional[str] = None
        self._lock = threading.Lock()

    def log(self, message: str) -> None:
        ts = time.strftime("%H:%M:%S", time.gmtime())
        with self._lock:
            self.progress_log.append(f"[{ts}] {message}")

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "run_id":          self.run_id,
                "pair_id":         self.pair_id,
                "status":          self.status.value,
                "progress_log":    list(self.progress_log),
                "started_at":      self.started_at,
                "finished_at":     self.finished_at,
                "elapsed_s":       (
                    (self.finished_at or time.time()) - self.started_at
                    if self.started_at else None
                ),
                "result":          self.result_summary,
                "error":           self.error,
                "assets":          self._list_assets(),
            }

    def _list_assets(self) -> list[str]:
        """Names of files present in the run directory."""
        if not self.out_dir.exists():
            return []
        return [f.name for f in self.out_dir.iterdir() if f.is_file()]


_RUNS: dict[str, RunRecord] = {}
_RUNS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Curated benchmark pairs (offline-capable; relative paths from repo root)
# ---------------------------------------------------------------------------
CURATED_PAIRS: list[dict] = [
    {
        "pair_id":     "ohrc_nac_easy",
        "label":       "OHRC ↔ LRO NAC (easy — similar sun angle, ~2:1 scale)",
        "tier":        "easy",
        "src_product": "ch2_ohr_ncp_20240330T0035085365",
        "ref_product": "M102000149RC",
        "src_camera":  "OHRC",
        "ref_camera":  "NAC",
        "scale_ratio": 1.8,
        "sun_delta_deg": 3.0,
        "note":        "Published: solved. Our credibility check.",
        "mock_seed":   1,
    },
    {
        "pair_id":     "ohrc_nac_stress",
        "label":       "OHRC ↔ LRO NAC (stress — sun 75° apart)",
        "tier":        "hard",
        "src_product": "ch2_ohr_ncp_20240330T0035085365",
        "ref_product": "M1417360906LC",
        "src_camera":  "OHRC",
        "ref_camera":  "NAC",
        "scale_ratio": 1.8,
        "sun_delta_deg": 75.0,
        "note":        "Illumination stress test on our own products.",
        "mock_seed":   2,
    },
    {
        "pair_id":     "tmc2_selene_tc",
        "label":       "TMC-2 ↔ SELENE TC (headline — zero prior art, ~1.68:1 scale)",
        "tier":        "medium",
        "src_product": "ch2_tmc_nca_20250207T1102039417",
        "ref_product": "TCO_MAP_02_N03E021N00E024SC",
        "src_camera":  "TMC2",
        "ref_camera":  "TC",
        "scale_ratio": 1.679,
        "sun_delta_deg": None,
        "note":        "Our headline contribution: no published attempt prior to this work.",
        "mock_seed":   3,
    },
    {
        "pair_id":     "iirs_wac",
        "label":       "IIRS ↔ LRO WAC (medium — ~1.25:1 scale)",
        "tier":        "medium",
        "src_product": "ch2_iir_nci_20240523T1600301891",
        "ref_product": "wac_mosaic",
        "src_camera":  "IIRS",
        "ref_camera":  "WAC",
        "scale_ratio": 1.03,
        "sun_delta_deg": None,
        "note":        "Published: solved. Reproduced at 5/5 HIGH tier (docs/iirs_wac_results.md).",
        "mock_seed":   4,
    },
]

_PAIR_INDEX: dict[str, dict] = {p["pair_id"]: p for p in CURATED_PAIRS}


# ---------------------------------------------------------------------------
# Worker function (runs in a thread so the event loop is not blocked)
# ---------------------------------------------------------------------------
def _run_registration(record: RunRecord) -> None:
    """Execute registration for *record* in a background thread."""
    record.started_at = time.time()
    record.status = JobStatus.RUNNING
    record.log("Job started.")

    payload  = record.request_payload
    mock     = bool(payload.get("mock", False))
    matcher  = payload.get("matcher", "sift")
    pair_id  = record.pair_id
    seed     = payload.get("seed")
    if seed is None:   # the pair's own seed unless the caller set one explicitly
        seed = _PAIR_INDEX[pair_id]["mock_seed"] if pair_id in _PAIR_INDEX else 7
    seed = int(seed)

    try:

        record.log(f"Matcher={matcher}  mock={mock}  seed={seed}")

        out = record.out_dir
        out.mkdir(parents=True, exist_ok=True)

        if mock:
            from . import synth
            from .pipeline import register_bundle

            record.log("Generating synthetic pair …")
            src, ref, _ = synth.make_pair(
                out_shape=(256, 256),
                shift=(3.4, -2.2),
                seed=seed,
                n_craters=35,
                shadows=False,
            )
            record.log("Running registration pipeline …")
            bundle = register_bundle(src, ref, matcher=matcher)
        else:
            src_label = Path(payload["src"])
            ref_label = Path(payload["ref"])

            record.log(f"Parsing source label: {src_label.name}")
            from .io.pds_label import parse_label
            from .io.tiling import iter_tiles

            src_meta = parse_label(src_label)
            ref_meta = parse_label(ref_label)

            record.log("Reading first tiles …")
            tile_size = int(payload.get("tile_size", 1024))
            src_plane = next(iter_tiles(src_meta, tile=tile_size, overlap=128))
            ref_plane = next(iter_tiles(ref_meta, tile=tile_size, overlap=128))

            record.log(f"Running registration (matcher={matcher}) …")
            from .pipeline import register_bundle
            bundle = register_bundle(src_plane, ref_plane, matcher=matcher)

        # ---- exports -------------------------------------------------------
        record.log("Exporting match points …")
        from .product import matchpoints, warp
        from .viz import coverage_plot, match_plot, sidebyside, swipe

        src_model = ref_model = None
        if mock:
            # minimal ground model so exports don't crash on absent raster
            from .cli import _MockGroundModel
            src_model = ref_model = _MockGroundModel(src.gsd_m)

        matchpoints.export_bundle(
            out, bundle,
            src_model=src_model, ref_model=ref_model,
            grid=8,
        )

        record.log("Building provenance …")
        if mock:
            # Synthetic pairs have no real files on disk, so we write a
            # lightweight marker instead of calling provenance.build()
            # (which requires file checksums). This is not a skip — it is
            # the honest record: the inputs are synthetic, not real data.
            prov_data = {
                "synthetic": True,
                "seed": seed,
                "matcher": matcher,
                "note": "Mock run — no real PDS data; provenance checksums not applicable.",
            }
            (out / "provenance.json").write_text(
                json.dumps(prov_data, indent=2) + "\n", encoding="utf-8"
            )
        else:
            from .product import provenance
            manifest = provenance.build(bundle, config_data={}, ship_mode=True)
            provenance.write(out / "provenance.json", manifest)

        record.log("Rendering figures …")
        sidebyside.render(bundle, out / "side-by-side.png")
        match_plot.render(bundle, out / "matches.png")
        coverage_plot.render(bundle, out / "coverage.png", grid=8)
        registered_arr, _ = warp.warp_array(bundle)
        swipe.render_checkerboard(bundle.ref.array, registered_arr,
                                  out / "checkerboard.png")

        record.log("Writing result.json …")
        result = bundle.result
        metrics_d = asdict(result.metrics) if is_dataclass(result.metrics) \
                    else result.metrics.__dict__

        result_record = {
            "confidence_tier": result.confidence_tier,
            "metrics":         _jsonable(result.metrics),
            "gates":           result.gates,
            "failure_modes":   result.failure_modes,
            "notes":           result.notes,
            "matcher":         result.matches.method,
            "regime":          result.matches.regime,
            "match_stage":     result.matches.stage,
        }
        (out / "result.json").write_text(
            json.dumps(result_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        record.result_summary = result_record
        record.status = JobStatus.DONE
        record.log(
            f"Done — tier={result.confidence_tier}  "
            f"inliers={result.metrics.inlier_count}  "
            f"rmse={result.metrics.rmse_px}"
        )

    except Exception as exc:  # noqa: BLE001
        record.error  = f"{type(exc).__name__}: {exc}"
        record.status = JobStatus.FAILED
        record.log(f"ERROR: {record.error}")

    finally:
        record.finished_at = time.time()


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
if _FASTAPI_AVAILABLE:
    class RegisterRequest(BaseModel):
        pair_id:   Optional[str]  = Field(None,    description="ID from GET /pairs (uses mock data)")
        src:       Optional[str]  = Field(None,    description="Path to source PDS label (real data)")
        ref:       Optional[str]  = Field(None,    description="Path to reference PDS label (real data)")
        matcher:   str            = Field("sift",  description="sift | rift2 | learned")
        mock:      bool           = Field(False,   description="Run a SYNTHETIC pair instead of real labels")
        seed:      Optional[int]  = Field(None,    description="RNG seed for a synthetic pair (default: the pair's own)")
        tile_size: int            = Field(1024,    description="Tile edge length in pixels")
        cpu:       bool           = Field(False,   description="Force CPU (no GPU)")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app(runs_root: Optional[Path] = None) -> "FastAPI":
    """Build and return the FastAPI application.

    Parameters
    ----------
    runs_root:
        Directory under which per-job run folders are created.
        Defaults to ``runs/`` next to the current working directory.
    """
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "fastapi and uvicorn are required for the API server.\n"
            "Install them with:  pip install -e '[product,api]'"
        )

    _runs_root = Path(runs_root or "runs").resolve()

    app = FastAPI(
        title="CHANDRALIGN API",
        description=(
            "SIH26166 — Multi-modal Chandrayaan-2 image registration backend. "
            "Implements UI-02 (P3-T07)."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # The UI is served by this same app, so it needs no CORS at all. A wildcard would let any
    # page on the venue network drive the API and read run files (audit C-06); allow only the
    # local origins, or an explicit comma-separated CHANDRALIGN_CORS_ORIGINS.
    origins = os.environ.get("CHANDRALIGN_CORS_ORIGINS", "")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in origins.split(",") if o.strip()] or
                      ["http://127.0.0.1:8000", "http://localhost:8000"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # POST /register
    # ------------------------------------------------------------------ #
    @app.post("/register", status_code=202, summary="Queue a registration job")
    def post_register(req: RegisterRequest, background_tasks: BackgroundTasks):
        """Queue a registration job and return immediately with a ``run_id``.

        Poll ``GET /runs/{run_id}`` for status and results.
        """
        # A request for real data must never be answered with a synthetic run (audit C-08).
        if req.mock and (req.src or req.ref):
            raise HTTPException(status_code=422,
                                detail="src/ref are real labels; they cannot be combined with mock=true")
        if not req.mock and not (req.src and req.ref) and not req.pair_id:
            raise HTTPException(status_code=422,
                                detail="give src and ref labels, a pair_id, or mock=true for a synthetic pair")
        if req.pair_id is not None and req.pair_id not in _PAIR_INDEX:
            raise HTTPException(status_code=422, detail=f"unknown pair_id {req.pair_id!r}; see GET /pairs")
        run_id  = str(uuid.uuid4())
        out_dir = _runs_root / run_id

        payload = req.model_dump()
        record  = RunRecord(
            run_id=run_id,
            pair_id=req.pair_id,
            out_dir=out_dir,
            request_payload=payload,
        )
        with _RUNS_LOCK:
            _RUNS[run_id] = record

        background_tasks.add_task(_run_registration_bg, record)
        return {"run_id": run_id, "status": "QUEUED", "poll": f"/runs/{run_id}"}

    # ------------------------------------------------------------------ #
    # GET /runs/{run_id}
    # ------------------------------------------------------------------ #
    @app.get("/runs/{run_id}", summary="Poll job status, metrics and logs")
    def get_run(run_id: str):
        """Return the current state of a registration job.

        The ``result`` field is present only when ``status == "DONE"``.
        The ``progress_log`` list grows while the job runs — poll repeatedly.
        If ``confidence_tier == "REJECTED"`` the ``failure_modes`` list
        explains why (honesty rule H3).
        """
        record = _get_or_404(run_id)
        return record.to_dict()

    # ------------------------------------------------------------------ #
    # GET /runs/{run_id}/assets/{name}
    # ------------------------------------------------------------------ #
    @app.get("/runs/{run_id}/assets/{name}",
             summary="Download a file produced by a run")
    def get_asset(run_id: str, name: str):
        """Serve a named asset (PNG, GeoTIFF, CSV, JSON …) from the run directory."""
        record = _get_or_404(run_id)
        root   = record.out_dir.resolve()
        asset  = (root / name).resolve()
        # `name` comes from the URL: "..\..\x" must not leave the run folder (audit C-06).
        if not asset.is_relative_to(root) or not asset.exists() or not asset.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Asset '{name}' not found in run '{run_id}'. "
                       f"Available: {record._list_assets()}",
            )
        return FileResponse(
            path=str(asset),
            filename=name,
            media_type=_guess_media_type(name),
        )

    # ------------------------------------------------------------------ #
    # GET /pairs
    # ------------------------------------------------------------------ #
    @app.get("/pairs", summary="List curated benchmark pairs")
    def get_pairs():
        """Return the list of pre-selected benchmark pairs that can be
        submitted to ``POST /register`` without supplying real data
        (``mock=true`` uses synthetic imagery with the pair's recommended seed).
        """
        return {"pairs": CURATED_PAIRS}

    # ------------------------------------------------------------------ #
    # GET /runs  (convenience — list all jobs in this server session)
    # ------------------------------------------------------------------ #
    @app.get("/runs", summary="List all runs in this session")
    def list_runs():
        with _RUNS_LOCK:
            return {
                "runs": [
                    {
                        "run_id":    r.run_id,
                        "pair_id":   r.pair_id,
                        "status":    r.status.value,
                        "started_at": r.started_at,
                        "finished_at": r.finished_at,
                    }
                    for r in _RUNS.values()
                ]
            }

    # ------------------------------------------------------------------ #
    # GET /health
    # ------------------------------------------------------------------ #
    @app.get("/health", summary="Health check")
    def health():
        return {"status": "ok", "feature": "UI-02", "task": "P3-T07"}

    # ------------------------------------------------------------------ #
    # Web UI Static Files & Root  (UI-03 / P3-T08)
    # index.html served at GET /
    # style.css, app.js served at GET /style.css, GET /app.js
    # ------------------------------------------------------------------ #
    ui_dir = Path(__file__).parent / "ui"
    if ui_dir.exists():
        from fastapi.staticfiles import StaticFiles

        # Serve the entire ui/ folder under /static  (for future assets)
        app.mount("/static", StaticFiles(directory=str(ui_dir)), name="static")

        # Expose the three key files at the root level so index.html can
        # reference them simply as  src="app.js"  and  href="style.css"
        @app.get("/style.css", include_in_schema=False)
        def serve_css():
            return FileResponse(str(ui_dir / "style.css"),
                                media_type="text/css")

        @app.get("/app.js", include_in_schema=False)
        def serve_js():
            return FileResponse(str(ui_dir / "app.js"),
                                media_type="application/javascript")

        @app.get("/", include_in_schema=False)
        def root_ui():
            return FileResponse(str(ui_dir / "index.html"),
                                media_type="text/html")

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_or_404(run_id: str) -> RunRecord:
    with _RUNS_LOCK:
        record = _RUNS.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return record


def _run_registration_bg(record: RunRecord) -> None:
    """Thin wrapper so BackgroundTasks can call a plain function."""
    _run_registration(record)


def _guess_media_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    return {
        ".png":     "image/png",
        ".jpg":     "image/jpeg",
        ".tif":     "image/tiff",
        ".tiff":    "image/tiff",
        ".csv":     "text/csv",
        ".geojson": "application/geo+json",
        ".json":    "application/json",
        ".html":    "text/html",
        ".jsonl":   "application/jsonlines",
    }.get(suffix, "application/octet-stream")


def _jsonable(value: Any) -> Any:
    """Recursively convert a dataclass / ndarray / Path to JSON-safe types."""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# ---------------------------------------------------------------------------
# Module-level ``app`` instance — used by uvicorn directly:
#   uvicorn chandralign.api:app --reload
# ---------------------------------------------------------------------------
try:
    app = create_app()
except RuntimeError:
    app = None  # fastapi not installed; API unavailable


# ---------------------------------------------------------------------------
# Entry point for ``python -m chandralign.api``
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    try:
        import uvicorn
    except ImportError:
        raise SystemExit("uvicorn is required: pip install uvicorn")
    # Local only by default; set CHANDRALIGN_HOST=0.0.0.0 deliberately to expose it.
    uvicorn.run("chandralign.api:app", host=os.environ.get("CHANDRALIGN_HOST", "127.0.0.1"),
                port=int(os.environ.get("CHANDRALIGN_PORT", "8000")))
