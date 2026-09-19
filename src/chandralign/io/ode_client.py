"""PDS ODE REST client: catalogue metadata for reference products. Owner: Member A (Part 1). Features: DATA-12.

NASA's Orbital Data Explorer (https://oderest.rsl.wustl.edu, no login) publishes
catalogue values that some product labels lack -- for LRO NAC CDR products, the
incidence / emission / phase angles and the ground footprint. These are ODE's own
catalogue values (derived by the LROC team from SPICE), NOT label contents, so
everything read through this module is tagged as coming from "ode_catalogue".

A record is fetched once and saved next to the product as ode_metadata.json, so
later runs (and the demo) never depend on ODE being reachable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ODE_URL = "https://oderest.rsl.wustl.edu/live2/"
METADATA_FILE = "ode_metadata.json"


class OdeError(RuntimeError):
    """ODE returned no product, an error, or an unexpected payload."""


def fetch_product_record(pdsid: str, timeout: float = 60.0) -> dict:
    """The full ODE catalogue record for one product id (e.g. 'nac.m1417360906lc')."""
    import requests

    params = {"query": "product", "target": "moon", "results": "mf", "output": "JSON", "pdsid": pdsid}
    r = requests.get(ODE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    payload = r.json().get("ODEResults", {})
    if str(payload.get("Status", "")).lower() != "success":
        raise OdeError(f"ODE query for {pdsid} failed: {payload.get('Error', payload)}")
    products = payload.get("Products")
    if not isinstance(products, dict):
        raise OdeError(f"ODE has no product {pdsid!r} ({products})")
    record = products.get("Product")
    if isinstance(record, list):
        if len(record) != 1:
            raise OdeError(f"ODE returned {len(record)} products for {pdsid!r}")
        record = record[0]
    return record


def save_product_record(pdsid: str, dest_dir: str | Path, record: Optional[dict] = None) -> Path:
    """Fetch (unless given) and save a record as <dest_dir>/ode_metadata.json."""
    record = record if record is not None else fetch_product_record(pdsid)
    out = Path(dest_dir) / METADATA_FILE
    out.write_text(json.dumps({
        "pdsid": pdsid,
        "source": "ode_catalogue",
        "query_url": f"{ODE_URL}?query=product&target=moon&results=mf&output=JSON&pdsid={pdsid}",
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record": record,
    }, indent=2) + "\n", encoding="utf-8")
    return out


def load_product_record(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "record" not in data or "pdsid" not in data:
        raise OdeError(f"{path} is not a saved ODE record")
    return data


def find_saved_record(product_dir: str | Path) -> Optional[dict]:
    """The saved record in a product's folder, or None if it was never fetched."""
    path = Path(product_dir) / METADATA_FILE
    return load_product_record(path) if path.exists() else None


def catalogue_number(record: dict, key: str) -> Optional[float]:
    """A numeric catalogue field, or None when ODE leaves it empty."""
    value = record.get(key)
    if value in (None, "", "N/A", "NULL"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise OdeError(f"ODE field {key}={value!r} is not a number") from exc
