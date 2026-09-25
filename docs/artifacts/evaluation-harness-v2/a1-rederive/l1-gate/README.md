# L1 gate — raw cells

Written by the L1 gate line of
evaluation-harness-v2.md §11.7 / §11.8.
The frozen A1 harness in `/workspace/wt/golden` with its library subtree
swapped to `dev/l1-library-layout` @ `df04e74` (tree `624459b6`).

| file | what |
|---|---|
| `goodreads-d128-c0_genre-silvertorch-triton.json` | the gate cell, L1 run 1 — bit-identical to `evaluation/golden/` |
| `repeat2-silvertorch-triton.json` | the same cell again, same tree — bit-identical, so the cell is deterministic |
| `goodreads-d128-c0_genre-linr_v3-triton.json`, `repeat2-…`, `repeat3-…` | three runs on the identical tree; spread 6.8e-5 |
| `run1-linr_v2-triton.json`, `repeat2-…` | two runs on the identical tree; spread 2.0e-6 |

The `linr_v*` files exist to show that those two cells do not reproduce
themselves, so their mismatch against the golden says nothing about L1. Do not
read them as golden candidates — `evaluation/golden/` is the baseline.
