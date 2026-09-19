# Repeated full test-suite runs

Command: `pytest -m ""` (all tests including slow ones), run 10 times back to back on the same code, 2026-09-19.

```
run 1: ================== 162 passed, 1 warning in 68.45s (0:01:08) ==================
run 2: ================== 162 passed, 1 warning in 69.06s (0:01:09) ==================
run 3: ================== 162 passed, 1 warning in 61.73s (0:01:01) ==================
run 4: ================== 162 passed, 1 warning in 60.80s (0:01:00) ==================
run 5: ================== 162 passed, 1 warning in 63.42s (0:01:03) ==================
run 6: ================== 162 passed, 1 warning in 66.35s (0:01:06) ==================
run 7: ================== 162 passed, 1 warning in 68.84s (0:01:08) ==================
run 8: ================== 162 passed, 1 warning in 67.14s (0:01:07) ==================
run 9: ================== 162 passed, 1 warning in 66.93s (0:01:06) ==================
run 10: ================== 162 passed, 1 warning in 67.44s (0:01:07) ==================
```

The 1 warning is pvl's own PendingDeprecationWarning (see Step 6 notes).
