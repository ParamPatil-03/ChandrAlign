# Real PDS4 label fixtures

Unmodified PDS4 labels of three real Chandrayaan-2 calibrated products, copied
from the zips downloaded from ISRO's PRADAN archive (checksums of the full
products are in `data/manifest.json`). They let `tests/test_pds_real.py` run
the anti-stub test on real labels on any machine, without the multi-GB pixel files.

| File | Camera |
|---|---|
| `ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml` | OHRC |
| `ch2_tmc_nca_20250207T1102039417_d_img_d18.xml` | TMC-2 |
| `ch2_iir_nci_20240523T1600301891_d_img_d18.xml` | IIRS |

Data courtesy of ISRO / ISSDC, Chandrayaan-2 mission, via PRADAN
(https://pradan.issdc.gov.in), used for non-profit scientific purposes.
Do not edit these files: tests compare parsed values against their exact contents.
