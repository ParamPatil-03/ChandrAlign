"""Cut a reference clip from the LROC WAC global 100 m mosaic for any IIRS scene.

The IIRS -> WAC evidence registers IIRS against the map-projected WAC global mosaic (the standard
reference in the literature; raw WAC frames do not lock -- docs/iirs_wac_results.md). The project
held ONE clip, cut for one IIRS scene; this module cuts a clip for any scene, from the public
USGS copy of the mosaic (a 6 GB uncompressed GeoTIFF on S3, one image row per strip).

Reading a small window through GDAL issues thousands of tiny requests, and whole rows are 109 KB
each, so neither is practical on a slow link. Instead the TIFF header is parsed for the strip
offsets and, for every needed row, only the needed columns are fetched with an HTTP Range request
(many in parallel). The clip is written in the same .npy + .json format as the existing clip, so
workflows.iirs_wac reads it unchanged.
"""
from __future__ import annotations

import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MOSAIC_URL = ("https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/"
              "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif")
PRODUCT = "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013 (LROC WAC global morphologic mosaic, 643 nm)"
WIDTH, HEIGHT = 109164, 54582
DEG_PER_PX = 360.0 / WIDTH
M_PER_DEG = 30323.333333333332


def _get(session, url, start, end, tries=6):
    for k in range(tries):
        try:
            r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
            if r.status_code == 206 and len(r.content) == end - start + 1:
                return r.content
        except Exception:                                  # network hiccup: retry
            if k == tries - 1:
                raise
    raise IOError(f"range {start}-{end} failed")


def strip_layout(session, url=MOSAIC_URL):
    """(offsets per row, bytes per sample, dtype) from the (Big)TIFF header."""
    head = _get(session, url, 0, 65535)
    bo = {b"II": "<", b"MM": ">"}[head[:2]]
    big = struct.unpack(bo + "H", head[2:4])[0] == 43
    if not big:
        raise NotImplementedError("classic TIFF")
    ifd = struct.unpack(bo + "Q", head[8:16])[0]
    ifd_bytes = _get(session, url, ifd, ifd + 8 + 20 * 64)
    n = struct.unpack(bo + "Q", ifd_bytes[:8])[0]
    tags = {}
    for i in range(n):
        e = ifd_bytes[8 + 20 * i: 28 + 20 * i]
        tag, typ, count = struct.unpack(bo + "HHQ", e[:12])
        tags[tag] = (typ, count, e[12:20])
    size = {3: 2, 4: 4, 16: 8}
    typ, count, val = tags[273]                            # StripOffsets
    raw = val if size[typ] * count <= 8 else _get(session, url, struct.unpack(bo + "Q", val)[0],
                                                  struct.unpack(bo + "Q", val)[0] + size[typ] * count - 1)
    offsets = np.frombuffer(raw, dtype=bo + {3: "u2", 4: "u4", 16: "u8"}[typ], count=count).astype(np.int64)
    bits = struct.unpack(bo + "H", tags[258][2][:2])[0]
    rows_per_strip = struct.unpack(bo + ("I" if tags[278][0] == 4 else "H"), tags[278][2][:4 if tags[278][0] == 4 else 2])[0]
    if rows_per_strip != 1 or len(offsets) != HEIGHT:
        raise NotImplementedError(f"expected one row per strip, got {rows_per_strip}")
    return offsets, bits // 8


def fetch_clip(lat_max, lat_min, lon_min, lon_max, out_json: str | Path, *, url=MOSAIC_URL,
               workers=32, progress=None) -> Path:
    """Cut the mosaic between the given lat/lon (deg) and write <out>.npy + <out>.json."""
    import requests
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))
    offsets, bps = strip_layout(s, url)
    row0 = max(0, int(np.ceil((90.0 - lat_max) / DEG_PER_PX - 0.5 - 1e-6)))           # first centre inside
    row1 = min(HEIGHT, int(np.floor((90.0 - lat_min) / DEG_PER_PX - 0.5 + 1e-6)) + 1)   # last centre inside
    col0 = max(0, int(np.ceil((lon_min + 180.0) / DEG_PER_PX - 0.5 - 1e-6)))
    col1 = min(WIDTH, int(np.floor((lon_max + 180.0) / DEG_PER_PX - 0.5 + 1e-6)) + 1)
    w = col1 - col0

    def row(r):
        return _get(s, url, int(offsets[r]) + col0 * bps, int(offsets[r]) + col1 * bps - 1)
    out = np.empty((row1 - row0, w), dtype=np.uint8 if bps == 1 else np.uint16)
    with ThreadPoolExecutor(workers) as ex:
        for i, b in enumerate(ex.map(row, range(row0, row1))):
            out[i] = np.frombuffer(b, dtype=out.dtype)
            if progress and i % 500 == 0:
                progress(f"mosaic rows {i}/{row1 - row0}")
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_json.with_suffix(".npy"), out)
    meta = {"source": url, "product": PRODUCT,
            "projection": "simple cylindrical (equirectangular), north up, 100 m/px",
            "m_per_deg": M_PER_DEG, "row0": row0, "col0": col0, "shape": [int(out.shape[0]), int(out.shape[1])],
            "pixel_centre_lat_of_row0": 90.0 - (row0 + 0.5) * DEG_PER_PX,
            "pixel_centre_lon_of_col0": -180.0 + (col0 + 0.5) * DEG_PER_PX,
            "deg_per_px": DEG_PER_PX, "nodata": 0,
            "sha256": hashlib.sha256(out.tobytes()).hexdigest()}
    out_json.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out_json


# The evidence clip spans 2425 rows = 8.0 deg of latitude; the IIRS -> WAC success rule (implied
# offset within 240 m of the scene median) was defined for windows spread inside such a span. IIRS
# system geolocation drifts smoothly along a ~34 deg strip (~900 m east on the evidence scene), so
# a whole-strip clip would fail that rule on correct registrations (docs/iirs_fresh_scenes_protocol.md
# amendment 1).
EVIDENCE_LAT_SPAN_DEG = 2425 * DEG_PER_PX


def clip_for_iirs(label, out_dir: str | Path, *, lat_span=EVIDENCE_LAT_SPAN_DEG, lon_margin=0.8,
                  progress=None) -> Path:
    """A mosaic clip for an IIRS scene: the evidence clip's latitude span, centred on the scene, the
    scene's longitude range plus a margin for its ~13 km system error."""
    from .pds_label import parse_label
    meta = parse_label(label)
    lats = [c[0] for c in meta.corner_latlon]
    lons = np.unwrap(np.radians([c[1] for c in meta.corner_latlon]))
    lon_lo, lon_hi = np.degrees(lons.min()), np.degrees(lons.max())
    lat_c = (max(lats) + min(lats)) / 2.0
    return fetch_clip(min(90.0, lat_c + lat_span / 2), max(-90.0, lat_c - lat_span / 2),
                      lon_lo - lon_margin, lon_hi + lon_margin,
                      Path(out_dir) / f"wac_mosaic_clip_{meta.product_id}.json", progress=progress)
