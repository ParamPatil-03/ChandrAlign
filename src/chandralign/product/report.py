"""Self-contained, control-gated HTML evaluation report (OUT-09)."""
from __future__ import annotations

import base64
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from jinja2 import Environment, StrictUndefined

from ..evaluate import control_gates, failure_log

_SOURCES = {"measured", "external", "synthetic"}
_METRICS = (
    ("RMSE", "rmse_px", "px"),
    ("Inliers", "inlier_count", ""),
    ("Inlier ratio", "inlier_ratio", ""),
    ("Spatial coverage", "spatial_coverage", ""),
)
_EXTRA_METRICS = (
    ("Sub-pixel recovery error", "subpixel_recovery_err_px", "px"),
    ("Maximum Delaunay gap", "max_delaunay_gap_px", "px"),
    ("Runtime", "runtime_s", "s"),
)

_TEMPLATE = Environment(autoescape=True, undefined=StrictUndefined).from_string(r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ChandraAlign evaluation report</title><style>
:root{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f8fafc;--ok:#16794b;--bad:#b42318;--brand:#3157d5}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,sans-serif;color:var(--ink);background:#eef2f7}.page{max-width:1120px;margin:auto;background:white;min-height:100vh;padding:36px}
h1{margin:0;font-size:30px}h2{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}h3{margin-bottom:6px}.muted{color:var(--muted)}.badge,.source{display:inline-block;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:700}.badge{background:#e8eefc;color:#2345ad}.badge.REJECTED{background:#fee4e2;color:var(--bad)}.source{margin-left:6px;background:#e2e8f0;color:#475569}.source.synthetic{background:#fff1c2;color:#854d0e}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}.card{border:1px solid var(--line);border-radius:10px;padding:14px;background:var(--panel)}.value{font-size:24px;font-weight:750}.unit{font-size:14px;color:var(--muted)}table{border-collapse:collapse;width:100%}th,td{text-align:left;border-bottom:1px solid var(--line);padding:9px;vertical-align:top}th{background:var(--panel)}.pass{color:var(--ok);font-weight:700}.fail{color:var(--bad);font-weight:700}.figures img{width:100%;border:1px solid var(--line);border-radius:8px}.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px}.reason{border-left:4px solid var(--bad);padding:8px 12px;background:#fff4f2;margin:8px 0}code{overflow-wrap:anywhere}footer{margin-top:40px;color:var(--muted);font-size:12px}
</style></head><body><main class="page">
<header><h1>ChandraAlign evaluation report</h1><p class="muted">Control-gated lunar image registration evidence</p><span class="badge {{ tier }}">{{ tier }}</span></header>
<h2>Inputs</h2><table><thead><tr><th>Role</th><th>Product</th><th>Mission / instrument</th><th>Raster</th></tr></thead><tbody>{% for x in inputs %}<tr><td>{{ x.role }}</td><td>{{ x.product_id }}</td><td>{{ x.mission }} / {{ x.instrument }}</td><td><code>{{ x.raster }}</code></td></tr>{% endfor %}</tbody></table>
<h2>Pipeline path</h2><p>Matcher: <strong>{{ matcher }}</strong> · regime: <strong>{{ regime }}</strong> · delivered stage: <strong>{{ match_stage }}</strong> · geometry: <strong>{{ geometry }}</strong></p><table><thead><tr><th>Stage</th><th>Applied</th><th>Details</th></tr></thead><tbody>{% for s in stages %}<tr><td>{{ s.name }}</td><td class="{{ 'pass' if s.passed else 'muted' }}">{{ s.applied }}</td><td>{{ s.details }}</td></tr>{% endfor %}</tbody></table>
<h2>Problem-statement metrics</h2><div class="grid">{% for m in metrics %}<article class="card"><div class="muted">{{ m.label }}</div><div class="value">{{ m.display }}{% if m.unit %} <span class="unit">{{ m.unit }}</span>{% endif %}</div><span class="source {{ m.source }}">{{ m.source }}</span></article>{% endfor %}</div>
<h2>Precision diagnostics</h2><div class="grid">{% for m in extra_metrics %}<article class="card"><div class="muted">{{ m.label }}</div><div class="value">{{ m.display }}{% if m.unit %} <span class="unit">{{ m.unit }}</span>{% endif %}</div><span class="source {{ m.source }}">{{ m.source }}</span></article>{% endfor %}</div>
{% if figures %}<h2>Visual evidence</h2><div class="figures">{% for f in figures %}<figure><img src="{{ f.uri }}" alt="{{ f.label }}"><figcaption>{{ f.label }} <span class="source measured">measured</span></figcaption></figure>{% endfor %}</div>{% endif %}
<h2>Control gates</h2><table><thead><tr><th>Gate</th><th>Result</th></tr></thead><tbody>{% for g in gates %}<tr><td>{{ g.name }}</td><td class="{{ 'pass' if g.passed else 'fail' }}">{{ 'PASS' if g.passed else 'FAIL' }}</td></tr>{% endfor %}</tbody></table>
<h2>Confidence and failure modes</h2><p>Final tier: <span class="badge {{ tier }}">{{ tier }}</span></p>{% if failures %}{% for f in failures %}<div class="reason"><strong>FM-{{ '%02d'|format(f.id) }}: {{ f.name }}</strong><br>{{ f.detection }}<br><span class="muted">Mitigation: {{ f.mitigation }}</span></div>{% endfor %}{% else %}<p>No canonical failure mode was triggered.</p>{% endif %}
<h2>Provenance</h2><table><tbody>{% for p in provenance %}<tr><th>{{ p.name }}</th><td><code>{{ p.value }}</code></td></tr>{% endfor %}</tbody></table>
<footer>Generated from recorded evidence. Missing measurements are shown as “not measured”; synthetic evidence is never labelled as real measurement.</footer>
</main></body></html>""")


def render_bundle(bundle: Any, path: str | Path, *, provenance: dict | None = None,
                  grid: int = 8) -> Path:
    """Render a live RegistrationBundle, generating and embedding its figures."""
    from ..product import warp
    from ..viz import coverage_plot, match_plot, sidebyside, swipe

    control_gates.require_gates(bundle.result)
    path = Path(path)
    with TemporaryDirectory(prefix="chandralign-report-") as folder:
        tmp = Path(folder)
        sidebyside.render(bundle, tmp / "side.png")
        match_plot.render(bundle, tmp / "matches.png")
        coverage_plot.render(bundle, tmp / "coverage.png", grid=grid)
        registered, _ = warp.warp_array(bundle)
        swipe.render_checkerboard(bundle.ref.array, registered, tmp / "checker.png")
        figures = [("Input images", tmp / "side.png"), ("Evidence matches", tmp / "matches.png"),
                   ("Delivered-point coverage", tmp / "coverage.png"),
                   ("Reference / registered checkerboard", tmp / "checker.png")]
        context = _bundle_context(bundle, provenance or {}, figures)
        return _write(path, context)


def render_run(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Regenerate a report from a persisted CLI run directory."""
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"run result not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    control_gates.require_gates(SimpleNamespace(gates=result.get("gates")))
    provenance_path = run_dir / "provenance.json"
    provenance = (json.loads(provenance_path.read_text(encoding="utf-8"))
                  if provenance_path.is_file() else {})
    figures = [(label, run_dir / name) for label, name in (
        ("Input images", "side-by-side.png"), ("Evidence matches", "matches.png"),
        ("Delivered-point coverage", "coverage.png"),
        ("Reference / registered checkerboard", "checkerboard.png")) if (run_dir / name).is_file()]
    context = _saved_context(result, provenance, figures)
    return _write(Path(output) if output else run_dir / "report.html", context)


def _metric_rows(metrics: Any, definitions) -> list[dict[str, str]]:
    raw = asdict(metrics) if is_dataclass(metrics) else dict(metrics or {})
    source = str(raw.get("source", ""))
    if source not in _SOURCES:
        raise ValueError("metrics.source must be measured, external, or synthetic")
    rows = []
    for label, key, unit in definitions:
        value = raw.get(key)
        if value is None:
            display = "not measured"
        elif key in ("inlier_ratio", "spatial_coverage"):
            display = f"{float(value):.1%}"
        elif isinstance(value, float):
            display = f"{value:.4g}"
        else:
            display = str(value)
        rows.append({"label": label, "display": display, "unit": unit, "source": source})
    return rows


def _inputs_from_planes(bundle) -> list[dict[str, str]]:
    rows = []
    for role in ("src", "ref"):
        meta = getattr(getattr(bundle, role), "meta")
        rows.append({"role": role, "product_id": str(meta.product_id), "mission": str(meta.mission),
                     "instrument": str(meta.instrument), "raster": str(meta.raster_path)})
    return rows


def _inputs_from_provenance(provenance: dict) -> list[dict[str, str]]:
    return [{"role": str(x.get("role", "unknown")), "product_id": str(x.get("product_id", "unknown")),
             "mission": str(x.get("mission", "unknown")), "instrument": str(x.get("instrument", "unknown")),
             "raster": str((x.get("raster") or {}).get("path", "not recorded"))}
            for x in provenance.get("inputs", [])]


def _stage_rows(stages: dict) -> list[dict[str, str]]:
    rows = []
    for name, detail in stages.items():
        detail = detail if isinstance(detail, dict) else {"value": detail}
        applied = detail.get("applied", "not recorded")
        rest = {k: v for k, v in detail.items() if k != "applied"}
        rows.append({"name": str(name), "applied": str(applied), "passed": applied is True,
                     "details": json.dumps(_jsonable(rest), sort_keys=True)})
    return rows


def _provenance_rows(provenance: dict) -> list[dict[str, str]]:
    return [{"name": str(key), "value": json.dumps(_jsonable(value), sort_keys=True)}
            for key, value in provenance.items() if key != "inputs"]


def _figures(items) -> list[dict[str, str]]:
    rows = []
    for label, path in items:
        data = Path(path).read_bytes()
        rows.append({"label": label, "uri": "data:image/png;base64," + base64.b64encode(data).decode("ascii")})
    return rows


def _bundle_context(bundle, provenance: dict, figures) -> dict:
    result, match = bundle.result, bundle.result.matches
    return _context(result.metrics, result.gates, result.failure_modes, result.confidence_tier,
                    _inputs_from_planes(bundle), match.method, match.regime, match.stage,
                    "parallax" if bundle.parallax is not None else "tps" if bundle.tps is not None
                    else result.model.kind if result.model else "not measured",
                    bundle.stages, provenance, figures)


def _saved_context(result: dict, provenance: dict, figures) -> dict:
    return _context(result.get("metrics", {}), result["gates"], result.get("failure_modes", []),
                    result.get("confidence_tier", "not measured"), _inputs_from_provenance(provenance),
                    result.get("matcher", "not recorded"), result.get("regime", "not recorded"),
                    result.get("match_stage", "not recorded"), result.get("geometry_used", "not measured"),
                    result.get("stages", {}), provenance, figures)


def _context(metrics, gates, failure_ids, tier, inputs, matcher, regime, match_stage,
             geometry, stages, provenance, figures) -> dict:
    return {"tier": tier, "inputs": inputs, "matcher": matcher, "regime": regime,
            "match_stage": match_stage, "geometry": geometry, "stages": _stage_rows(stages),
            "metrics": _metric_rows(metrics, _METRICS), "extra_metrics": _metric_rows(metrics, _EXTRA_METRICS),
            "figures": _figures(figures), "gates": [{"name": k, "passed": bool(v)} for k, v in gates.items()],
            "failures": [asdict(x) for x in failure_log.modes(failure_ids)],
            "provenance": _provenance_rows(provenance)}


def _write(path: Path, context: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE.render(**context), encoding="utf-8")
    return path


def _jsonable(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
