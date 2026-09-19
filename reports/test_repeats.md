# Repeated full test-suite runs

Command: `pytest -m ""` (all tests including slow ones), run 10 times back to back on the same code, 2026-09-19, at commit a99ab27.

```
run 1: ======================= 188 passed, 1 warning in 35.24s =======================
run 2: ======================= 188 passed, 1 warning in 35.48s =======================
run 3: ======================= 188 passed, 1 warning in 37.83s =======================
run 4: ======================= 188 passed, 1 warning in 39.20s =======================
run 5: ======================= 188 passed, 1 warning in 39.44s =======================
run 6: ======================= 188 passed, 1 warning in 40.68s =======================
run 7: ======================= 188 passed, 1 warning in 42.02s =======================
run 8: ======================= 188 passed, 1 warning in 40.30s =======================
run 9: ======================= 188 passed, 1 warning in 33.47s =======================
run 10: ======================= 188 passed, 1 warning in 33.33s =======================
```

The 1 warning is pvl's own PendingDeprecationWarning (see Step 6 notes).
