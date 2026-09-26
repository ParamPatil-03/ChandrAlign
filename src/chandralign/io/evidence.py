"""The exact Chandrayaan-2 products the committed evidence was measured on.

Scripts used to pick "the" product with next(rglob("*_d_img_d18.xml")) -- whichever label the
filesystem returned first. That was harmless while one product per instrument was held, and wrong
the moment a second one arrives (more IIRS scenes for the fresh-data check): a committed result
could silently be re-run on a different scene. Evidence scripts name their product instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .. import config

EVIDENCE_PRODUCTS = {
    "OHRC": "ch2_ohr_ncp_20240330T0035085365_d_img_d18",
    "TMC2": "ch2_tmc_nca_20250207T1102039417_d_img_d18",
    "IIRS": "ch2_iir_nci_20240523T1600301891_d_img_d18",
}
_FOLDER = {"OHRC": "ohrc", "TMC2": "tmc2", "IIRS": "iirs"}


def evidence_label(instrument: str, root: Optional[Path] = None) -> Path:
    """The PDS4 label of the product the committed `instrument` evidence used."""
    key = instrument.upper().replace("-", "")
    pid = EVIDENCE_PRODUCTS[key]
    base = Path(root or config.ROOT) / "data" / "raw" / "ch2" / _FOLDER[key]
    hit = next(iter(sorted(base.rglob(f"{pid}.xml"))), None) if base.exists() else None
    if hit is None:
        raise FileNotFoundError(f"{pid}.xml (the {key} evidence product) is not under {base}")
    return hit
