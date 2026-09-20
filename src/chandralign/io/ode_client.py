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


# ----------------------------------------------------------------------------- DATA-12: area search

# ODE product types for target=moon, as listed by its own query=iipy. CDR is
# radiometrically calibrated and is what a reference image should be; EDR (raw)
# stays reachable because the anti-stub tests need a product with a detached label.
PRODUCT_TYPES = {
    "NAC": ["CDRNAC4"],
    "NAC-EDR": ["EDRNAC4"],
    "WAC": ["CDRWAM4"],
    "WAC-EDR": ["EDRWAM4"],
}
IHID, IID = "LRO", "LROC"
PRODUCT_FILE_SUFFIXES = (".IMG", ".LBL", ".XML")


def to_360(lon: float) -> float:
    """ODE expresses longitude in 0..360 east; our shapefiles use -180..180."""
    return lon % 360.0


def _as_list(value) -> list:
    """ODE collapses a single-element list to a bare object. Undo that."""
    if value is None:
        return []
    return [value] if isinstance(value, dict) else list(value)


def parse_products(payload: dict, product_type: str) -> list[dict]:
    """The Products block of an ODE response, flattened to the fields we use.

    Split out from the network call so it can be tested against a saved payload
    without reaching ODE -- the shape of this response is the part that breaks.
    """
    products = _as_list(payload.get("Products", {}).get("Product"))
    results = []
    for p in products:
        files = [
            {"name": f.get("FileName"), "url": f.get("URL"), "kbytes": f.get("KBytes")}
            for f in _as_list(p.get("Product_files", {}).get("Product_file"))
            if str(f.get("FileName", "")).upper().endswith(PRODUCT_FILE_SUFFIXES)
        ]
        results.append({
            "product_id": p.get("pdsid"),
            "type": product_type,
            "start_utc": p.get("UTC_start_time"),
            "incidence_deg": p.get("Incidence_angle"),
            "emission_deg": p.get("Emission_angle"),
            "phase_deg": p.get("Phase_angle"),
            "bbox": [p.get("Minimum_latitude"), p.get("Maximum_latitude"),
                     p.get("Westernmost_longitude"), p.get("Easternmost_longitude")],
            "files": files,
        })
    return results


def search_area(bbox: tuple[float, float, float, float], product: str = "NAC",
                limit: int = 20, timeout: float = 60.0) -> list[dict]:
    """Reference products whose footprint covers a ground box (DATA-12).

    bbox is (min_lat, max_lat, min_lon, max_lon) in degrees, longitudes in
    -180..180 as our shapefiles give them; they are converted to ODE's 0..360.

    Lives here rather than in scripts/fetch_lro.py so the pair-finding work can
    call it directly and so the response parsing is testable without a network.
    Raises OdeError when ODE reports a failure; an empty result is NOT an error,
    because "nothing covers this box" is a real answer.
    """
    import requests

    if product not in PRODUCT_TYPES:
        raise OdeError(f"unknown product {product!r}; known: {sorted(PRODUCT_TYPES)}")
    min_lat, max_lat, min_lon, max_lon = bbox
    if min_lat > max_lat:
        raise OdeError(f"bbox latitudes are inverted: {min_lat} > {max_lat}")

    results: list[dict] = []
    for ode_type in PRODUCT_TYPES[product]:
        params = {
            "query": "product", "target": "moon", "results": "fmp", "output": "JSON",
            "ihid": IHID, "iid": IID, "pt": ode_type, "limit": limit,
            "minlat": min_lat, "maxlat": max_lat,
            "westernlon": to_360(min_lon), "easternlon": to_360(max_lon),
        }
        response = requests.get(ODE_URL, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json().get("ODEResults", {})
        status = str(payload.get("Status", "")).lower()
        if status != "success":
            # ODE says "Success" with zero products for an empty box, so a
            # non-success status is a real failure and not simply "none found".
            if "no products" in str(payload.get("Error", "")).lower():
                continue
            raise OdeError(f"{ode_type}: {payload.get('Error', 'unexpected ODE response')}")
        results.extend(parse_products(payload, ode_type))
    return results
