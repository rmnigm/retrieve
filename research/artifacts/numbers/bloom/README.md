# Bloom legs on pod 1 (bloom-runner): drivers

The user's bloom directive (2026-10-12): SilverTorch on the adapted backend with its bloom filter and the recheck off (false positives count as
misses against the exact filtered oracle), LiNR V1 / V2 / V3 with the exact clause filter only. Results and their state are in
[validation](../../../validation.md#numbers-adapted-silvertorch-over-the-study-grid); raw outputs on the Hub ([hub-index](../../hub-index.md)).

| file | what |
|---|---|
| `run_bloom.impl.sh`, `run_bloom.sh` | one leg on pod 1 GPU 0: [`run_leg`](../README.md)'s runner plus a pause below 3 GB free and the leg's own `_parity/*.npz` deleted while it runs (single backend, never uploaded) |
| `queue_b1.sh`, `queue_b1b.sh`, `queue_b1c.sh` | B1 = `numbers/A1-bloom`, the quality-only bloom curves (`n-bloom-curve-d*`), as run: first pass, resume after a 1-min hold, reorder (synth moved to pod e) |
| `queue_b2.sh`, `queue_b2r.sh`, `chain_b2r.sh` | B2 = `numbers/A2-bloom`, timed at the picks of [`pick_bloom.py`](../pick_bloom.py), gated by `check_b2.py` (dev/numbers-bloom-picks); `queue_b2r` re-runs an OOM-failed dataset's cells split by algo (V1 + V2 / V3 / SilverTorch; laion30m one algo each) |
| `hold_k1000.sh` | holds pod 1's k 1000 calls once k 100 is uploaded (k 1000 ran on pod e) |
| `queue_grade_rest_rerun.sh`, `queue_rerun.sh` | the grading points the d256 OOM left (fresh processes), then `numbers/A2-bloom-rerun`: A2-bloom's 15 unstable k 100 cells, stable records seeded; stopped by the controller after goodreads, arXiv, LAION 1M (the final run supersedes it) |
| `grade_points.py` | B2's SilverTorch points per dim, the input of [`grade_bloom.py`](../../results-draft/grade_bloom.py) |
| `copy_laion30m.sh` | pod 1 → pod e copy of `data/laion30m` with sha256 on both sides (laion30m is not on the Hub) |
