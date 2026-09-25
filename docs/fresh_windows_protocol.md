# Protocol: score the adopted settings on FRESH windows (audit I-16). Frozen before running.

## Why

Several adoption rules were amended after results were seen (ALIGN-08 on its 4th try, map-pairings amendment
1, TC-reference amendments 1-2, OHRC Q1/Q2 chosen and scored on the same 25 windows). The audit asks that success
rates be re-scored on data the choices never saw before they are quoted.

## What "fresh" can mean with held data

Only one product per instrument is held (and one TC / NAC set), so fresh PRODUCTS are impossible here. Fresh
WINDOWS of the same products are possible: ground that no adopted choice was tuned on. This is weaker (same
orbit, same illumination, same sensor state), and it is reported as such.

## Windows, by rule (fixed now, before any result)

- **OHRC -> NAC** (`register_ohrc_nac.py --auto-bridge --matchers routed --fresh`): per NAC, the midpoints
  between consecutive default picks, snapped to the script's own candidate grid, dropping any within one
  window (2048 rows) of a default pick. Up to 4 per NAC.
- **TMC-2 -> TC** (`register_tmc2_tc.py --rows ...`): the committed rows are 1562 apart in N00, closer than
  2 windows, so there are no midpoints. Each tile's committed span is extended by one spacing on each side:
  N00 7812; N03 13250, 24250; N09 49250, 60250. A window that falls off the tile is reported as skipped,
  not replaced.
- **IIRS -> WAC**: none possible. The five committed windows already tile the held mosaic clip at the
  window length (512 lines), so every other window overlaps them. Reported as "no fresh window available".

## Settings and report

Current HEAD settings (everything adopted up to the snapshot). Per pairing: the success rate on fresh windows
next to the committed rate, with the same success rule. For OHRC, the rule of the I-07 amendment (success /
unconfirmed / failed). No bar: the purpose is to report how far the quoted rates generalise.
