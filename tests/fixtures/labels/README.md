# Real PDS3 / PDS4 label fixtures

Unmodified PDS4 labels of three real Chandrayaan-2 calibrated products, copied
from the zips downloaded from ISRO's PRADAN archive (checksums of the full
products are in `data/manifest.json`). They let `tests/test_pds_real.py` run
the anti-stub test on real labels on any machine, without the multi-GB pixel files.

| File | Camera |
|---|---|
| `ch2_ohr_ncp_20240330T0035085365_d_img_d18.xml` | OHRC |
| `ch2_tmc_nca_20250207T1102039417_d_img_d18.xml` | TMC-2 |
| `ch2_iir_nci_20240523T1600301891_d_img_d18.xml` | IIRS |
| `TCO_MAP_02_N03E021N00E024SC.lbl` | SELENE TC (PDS3) |
| `TCO_MAP_02_N00E021S03E024SC.lbl` | SELENE TC (PDS3) |
| `M1417360906LC.XML` | LRO NAC (PDS4) |
| `M102000149RC.XML` | LRO NAC (PDS4) |
| `M1417360906LC_attached_header.lbl` | first 5064 bytes of `M1417360906LC.IMG`: its embedded PDS3 header |

Data courtesy of ISRO / ISSDC, Chandrayaan-2 mission, via PRADAN
(https://pradan.issdc.gov.in), used for non-profit scientific purposes.
Do not edit these files: tests compare parsed values against their exact contents.

The SELENE labels are from JAXA DARTS (https://data.darts.isas.jaxa.jp/pub/pds3/),
SELENE/Kaguya Terrain Camera ortho map v2.0, data courtesy of JAXA.

The LRO NAC labels are from NASA's PDS (LROC CDR, via the PDS Orbital Data Explorer),
Lunar Reconnaissance Orbiter Camera, NASA / Arizona State University.
