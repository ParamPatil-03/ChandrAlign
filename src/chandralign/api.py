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
        """Files of the run, as paths relative to the run folder (a product run has window_NN/)."""
        if not self.out_dir.exists():
            return []
        return sorted(f.relative_to(self.out_dir).as_posix() for f in self.out_dir.rglob("*") if f.is_file())


_RUNS: dict[str, RunRecord] = {}
_RUNS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Curated pairs: real products with committed evidence (audit 2026-09-26 C-05, C-08)
# ---------------------------------------------------------------------------
# Every claim about a pair is READ from the committed report it names (`_evidence`), never typed
# here: the previous list carried scale ratios, sun angles and "Published: solved" notes that
# did not match the evidence. `note` states what the evidence does NOT show, from the audit.
CURATED_PAIRS: list[dict] = [
    {
        "pair_id":     "tmc2_selene_tc",
        "label":       "TMC-2 -> SELENE TC (headline: no published prior attempt)",
        "src_camera":  "TMC2", "ref_camera": "TC",
        "src_product": "ch2_tmc_nca_20250207T1102039417",
        "ref_product": "TCO_MAP_02_N03E021N00E024SC",
        "evidence":    ("tmc2_tc", "reports/tmc2_tc_registration_height_ref.json"),
        "note":        "Independent accuracy (NCC probes): p50 0.46, p95 1.32 TMC-2 px -- sub-pixel at "
                       "the median, not at p95 (docs/AUDIT_2026-09-26.md section 7).",
        "mock_seed":   3,
    },
    {
        "pair_id":     "ohrc_nac",
        "label":       "OHRC -> LRO NAC M102014464RC (credibility pairing)",
        "src_camera":  "OHRC", "ref_camera": "NAC",
        "src_product": "ch2_ohr_ncp_20240330T0035085365",
        "ref_product": "M102014464RC",
        "evidence":    ("ohrc_nac", "reports/ohrc_nac_q8_auto_bridge.json"),
        "note":        "Tier is capped at LOW (NAC pixel size unverified). Accuracy on this coarser NAC: "
                       "p50 2.1, p95 6.6 OHRC px -- located, not sub-pixel in OHRC pixels.",
        "mock_seed":   1,
    },
    {
        "pair_id":     "ohrc_nac_opposed_sun",
        "label":       "OHRC -> LRO NAC M175124932LC (opposed sun, finest NAC)",
        "src_camera":  "OHRC", "ref_camera": "NAC",
        "src_product": "ch2_ohr_ncp_20240330T0035085365",
        "ref_product": "M175124932LC",
        "evidence":    ("ohrc_nac", "reports/ohrc_nac_q8_auto_bridge.json"),
        "note":        "Located on all windows, but the matcher-free MI check flags all of them "
                       "(2.4-2.9 OHRC px): unconfirmed, see audit I-07.",
        "mock_seed":   5,
    },
    {
        "pair_id":     "ohrc_nac_75deg_sun",
        "label":       "OHRC -> LRO NAC M1417360906LC (75 deg incidence: a known limit)",
        "src_camera":  "OHRC", "ref_camera": "NAC",
        "src_product": "ch2_ohr_ncp_20240330T0035085365",
        "ref_product": "M1417360906LC",
        "evidence":    ("ohrc_nac", "reports/ohrc_nac_q8_auto_bridge.json"),
        "note":        "Expect REJECTED: the documented illumination limit. Shown to demonstrate "
                       "honest refusal, not success.",
        "mock_seed":   2,
    },
    {
        "pair_id":     "iirs_wac",
        "label":       "IIRS -> LRO WAC global mosaic (hyperspectral source)",
        "src_camera":  "IIRS", "ref_camera": "WAC_MOSAIC",
        "src_product": "ch2_iir_nci_20240523T1600301891",
        "ref_product": "WAC_GLOBAL_MOSAIC_100M",
        "evidence":    ("iirs_wac", "reports/iirs_wac_mosaic.json"),
        "note":        "Accuracy by the matcher-free MI check: 0.09-0.30 IIRS px, sub-pixel -- but from one "
                       "IIRS scene (5 windows).",
        "mock_seed":   4,
    },
]

_PAIR_INDEX: dict[str, dict] = {p["pair_id"]: p for p in CURATED_PAIRS}
_ROOT = Path(__file__).resolve().parents[2]


def _label_for(product_id: str) -> Optional[Path]:
    """The label of a held product, or None (data/raw is not committed)."""
    base = _ROOT / "data" / "raw"
    if not base.exists():
        return None
    if product_id == "WAC_GLOBAL_MOSAIC_100M":            # the mosaic clip: its .json is the "label"
        hit = base / "lro" / "wac_mosaic" / "wac_mosaic_100m_clip.json"
        return hit if hit.is_file() else None
    for pattern in (f"{product_id}_d_img_d18.xml", f"{product_id}.XML", f"{product_id}.xml", f"{product_id}.lbl"):
        hit = next((p for p in sorted(base.rglob(pattern)) if "_PYR" not in p.name), None)
        if hit is not None:
            return hit
    return None


def _evidence(pair: dict) -> dict:
    """What the committed report says about this pair -- read, not restated."""
    kind, rel = pair["evidence"]
    path = _ROOT / rel
    if not path.is_file():
        return {"report": rel, "available": False}
    data = json.loads(path.read_text(encoding="utf-8"))
    if kind == "tmc2_tc":
        tiers = [r.get("tier") for r in data.get("rows", [])]
        done = [t for t in tiers if t]
        return {"report": rel, "available": True, "windows": len(tiers),
                "accepted": sum(t in ("HIGH", "MEDIUM", "LOW") for t in done),
                "tiers": {t: done.count(t) for t in sorted(set(done))}}
    if kind == "ohrc_nac":
        s = (data.get("summary") or {}).get(pair["ref_product"], {}).get("routed", {})
        return {"report": rel, "available": True, "windows": s.get("windows"),
                "accepted": s.get("success"), "verdict": s.get("verdict")}
    s = (data.get("summary") or {}).get(pair["ref_product"], {}).get("xoftr", {})
    return {"report": rel, "available": True, "windows": s.get("windows"),
            "accepted": s.get("success"), "verdict": s.get("verdict"), "matcher": "xoftr"}


def pair_catalogue() -> list[dict]:
    """GET /pairs: each curated pair with its committed evidence and whether it can run here."""
    from .workflows.products import SUPPORTED
    out = []
    for pair in CURATED_PAIRS:
        entry = {k: v for k, v in pair.items() if k != "evidence"}
        entry["evidence"] = _evidence(pair)
        src, ref = _label_for(pair["src_product"]), _label_for(pair["ref_product"])
        if (pair["src_camera"], pair["ref_camera"]) not in SUPPORTED:
            entry["runnable"], entry["why_not"] = False, "no product workflow for this pairing yet"
        elif src is None or ref is None:
            entry["runnable"], entry["why_not"] = False, "the real products are not in data/raw on this machine"
        else:
            entry["runnable"], entry["why_not"] = True, None
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Worker function (runs in a thread so the event loop is not blocked)
# ---------------------------------------------------------------------------
def _run_registration(record: RunRecord) -> None:
    """Execute registration for *record* in a background thread.

    Real products (src/ref labels, or a curated pair_id) go through the SAME path as the CLI:
    workflows.products.register_products and cli.write_product_run. mock=true runs a synthetic
    pair, labelled synthetic everywhere (audit C-01, C-08).
    """
    record.started_at = time.time()
    record.status = JobStatus.RUNNING
    record.log("Job started.")

    payload  = record.request_payload
    mock     = bool(payload.get("mock", False))
    pair_id  = record.pair_id
    seed     = payload.get("seed")
    if seed is None:   # the pair's own seed unless the caller set one explicitly
        seed = _PAIR_INDEX[pair_id]["mock_seed"] if pair_id in _PAIR_INDEX else 7
    seed = int(seed)

    try:
        out = record.out_dir
        out.mkdir(parents=True, exist_ok=True)
        if mock:
            record.result_summary = _run_mock(record, out, seed, payload.get("matcher") or "sift")
        else:
            record.result_summary = _run_real(record, out, payload)
        record.status = JobStatus.DONE
    except Exception as exc:  # noqa: BLE001
        record.error  = f"{type(exc).__name__}: {exc}"
        record.status = JobStatus.FAILED
        record.log(f"ERROR: {record.error}")
    finally:
        record.finished_at = time.time()


def _run_mock(record: RunRecord, out: Path, seed: int, matcher: str) -> dict:
    from . import synth
    from .cli import _MockGroundModel
    from .pipeline import register_bundle
    from .product import run_export
    record.log(f"SYNTHETIC pair (seed={seed}), matcher={matcher} ...")
    src, ref, _ = synth.make_pair(out_shape=(256, 256), shift=(3.4, -2.2), seed=seed, n_craters=35, shadows=False)
    bundle = register_bundle(src, ref, matcher=matcher)
    model = _MockGroundModel(bundle.src.gsd_m)
    # Synthetic pairs have no files on disk, so provenance is an honest marker, not checksums.
    manifest = {"synthetic": True, "seed": seed, "matcher": matcher,
                "note": "Mock run -- no real PDS data; provenance checksums not applicable."}
    result = run_export.write_run(out, bundle, manifest=manifest, src_model=model, ref_model=model, grid=8)
    record.log(f"Done (synthetic) -- tier={result['confidence_tier']}  "
               f"inliers={bundle.result.metrics.inlier_count}  rmse={bundle.result.metrics.rmse_px}")
    return result


def _run_real(record: RunRecord, out: Path, payload: dict) -> dict:
    from .cli import write_product_run
    from .workflows.products import register_products
    if record.pair_id:
        pair = _PAIR_INDEX[record.pair_id]
        src, ref = _label_for(pair["src_product"]), _label_for(pair["ref_product"])
        if src is None or ref is None:
            raise FileNotFoundError(f"{record.pair_id}: the real products are not in data/raw on this machine")
    else:
        src, ref = Path(payload["src"]), Path(payload["ref"])
        for label in (src, ref):
            if not label.is_file():
                raise FileNotFoundError(f"label not found: {label}")
    windows = int(payload.get("windows") or 1)
    device = "cpu" if payload.get("cpu") else None
    run = register_products(src, ref, windows=windows, device=device, progress=record.log)
    arguments = {"command": "api POST /register", "pair_id": record.pair_id, "src": str(src), "ref": str(ref),
                 "windows": windows, "device": device or "auto"}
    summary = write_product_run(out, run, arguments)
    record.log(f"Done -- {summary['accepted']}/{summary['windows']} windows accepted: {summary['tiers']}")
    return summary


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------
if _FASTAPI_AVAILABLE:
    class RegisterRequest(BaseModel):
        pair_id:   Optional[str]  = Field(None,    description="ID from GET /pairs: its real products")
        src:       Optional[str]  = Field(None,    description="Path to source PDS label (real data)")
        ref:       Optional[str]  = Field(None,    description="Path to reference PDS label (real data)")
        matcher:   Optional[str]  = Field(None,    description="mock only (default sift); real products use routing")
        windows:   int            = Field(1,       ge=1, le=10, description="windows across the overlap (real data)")
        mock:      bool           = Field(False,   description="Run a SYNTHETIC pair instead of real labels")
        seed:      Optional[int]  = Field(None,    description="RNG seed for a synthetic pair (default: the pair's own)")
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
        if req.matcher and not req.mock:
            raise HTTPException(status_code=422, detail="matcher applies to mock runs only: real products "
                                                        "use the matcher routing chose, as the evidence did")
        if req.pair_id is not None and not req.mock:
            entry = next(p for p in pair_catalogue() if p["pair_id"] == req.pair_id)
            if not entry["runnable"]:
                raise HTTPException(status_code=422, detail=f"{req.pair_id} cannot run here: {entry['why_not']}")
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
    @app.get("/runs/{run_id}/assets/{name:path}",
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
    @app.get("/failure-modes", summary="The canonical failure-mode register (CHECK-09)")
    def get_failure_modes():
        """The 20 failure modes a result's `failure_modes` ids refer to, with detection and
        mitigation -- so a client shows names, never invented codes."""
        from .evaluate.failure_log import FAILURE_MODES
        return {"failure_modes": [{"id": m.id, "name": m.name, "detection": m.detection,
                                   "mitigation": m.mitigation} for m in FAILURE_MODES.values()]}

    @app.get("/pairs", summary="List curated benchmark pairs")
    def get_pairs():
        """The curated pairs: real products, their committed evidence (read from the report), and
        whether they can run on this machine. POST /register {pair_id} runs the real products;
        add mock=true for a synthetic stand-in with the pair's seed (labelled synthetic).
        """
        return {"pairs": pair_catalogue()}

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
