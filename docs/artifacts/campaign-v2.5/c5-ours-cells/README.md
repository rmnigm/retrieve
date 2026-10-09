# C5-OURS cells — co-design vs full mask inside our Triton SilverTorch, at campaign-v2.5

Roadmap C5-OURS cells (claim **C5**: "co-design 1.79-2.15× lower latency than full mask then IVF"). Meta's official
backend reverses C5 because its partial path is host-bound (V-PROF3). These cells run both paths on our own fused kernels,
which separates the co-design idea from Meta's implementation of it. NOT CITABLE (no gate).
State: [validation](../../../validation.md) row *C5-OURS cells*. Option and gates:
[kernels](../../../system/kernels.md#bloom_full_mask--the-full-n-mask-bloom_pathfull).

## How it ran
- **Where:** pod b, A100-SXM4-80GB GPU 0.
- **Code:** library **472f2fc6** (tag campaign-v2.5), from a worktree at the tag via PYTHONPATH.
  [`driver.sh`](driver.sh) refuses to start unless `bench env` reports campaign.yaml's code_version.
- **Run:** one `bench run --backend triton --filter-kind bloom --interleave --resume` chunk per sweep, each under the GPU
  flock.
- **Grid:** `bloom_path` {partial, full} in one interleave group per (cell, seed), k 100, bs {1, 16}, eager and graph.

| dataset | sweeps | n_lists | n_probe | seeds | cells |
|---|---|---|---|---|---|
| arXiv d128 | `c3_nversions`, `c0_maincat`, `all4` | 1664 | {8, 32, 128} | 0-2 | 54 / 54 ok |
| goodreads d128 | `c0_genre` | 1024 | {8, 32, 128} | 0-2 | 18 / 18 ok |
| PubMed 10 M d768 | `c0_mesh` (the `filter` suite's bloom slot) | 4096 | {24, 1024} | 0 | 4 / 4 ok |

  Goodreads and PubMed were trimmed by the claims-first rule (controller 2026-10-10):
  - goodreads dropped `c2_format` and `c3_year`;
  - PubMed runs seed 0 only.

  On arXiv the sweep does not move the ratio, and seeds agree within about ±0.01. The PubMed suite is
  [`suite-pubmed.yaml`](suite-pubmed.yaml), appended to the v2.5 suites in a one-off config dir.
- **Summaries:** [`c5_summary.py`](c5_summary.py) gives the full / partial ratio paired per seed, `peak_fwd_mib`, and
  whether the two paths returned the same ids. Output: [arXiv](summary-arxiv.txt), [goodreads](summary-goodreads.txt),
  [PubMed](summary-pubmed.txt).
- **Records:** Hub `campaign-v2.5/{arxiv,goodreads,pubmed}-codesign-ours`. Logs and clocks: `artifacts/c5-ours-cells`.

## Results (median over seeds; full / partial)
| | eager | graph, narrow probe | graph, widest probe bs 16 | peak MiB bs 16, partial → full |
|---|---|---|---|---|
| arXiv (≈ 2.6 M) | 1.28-1.29 | 0.99-1.21 (n_probe ≤ 32) | 0.85-0.93 (128) | 2.2 → 8.4 (8), 23 → 29 (128) |
| goodreads | 1.28-1.29 | 1.00-1.04 (≤ 32) | 0.97 (128) | 2.6 → 4.4 (8), 11 → 13 (128) |
| PubMed 10 M | 1.12-1.31 | 1.05-1.12 (24) | 1.12 (1024) | 11.5 → 30.6 (24), 242 → 261 (1024) |

Ids (`ids_sha256_canon`) and recall@100 are identical between the two paths in every cell.

## What it means for C5
**Direction: holds on our kernels at 10 M, and mostly at 2-3 M.** The co-designed (partial) path is faster at PubMed 10 M
at every point (graph 1.05-1.12×). At 2-3 M it is faster or level at narrow probes. At the widest probe it loses (full
0.85-0.97×): the fused scorer evaluates about `C · k_hash` bloom words per slot, where the full path reads one mask word.

**Magnitude: not reproduced.** Our largest co-design gain is 1.12× in graph mode and 1.31× eager. Meta reports 1.79-2.15×
at 20 M. The eager gap here is launch overhead (the mask kernel and a few torch ops), not the filter work.

**Memory grows with N, as C5 says.** The full mask adds about B · N / 8 bytes: +19 MiB at 10 M bs 16, +6 MiB at arXiv.
The co-design path adds none.

So the C5 reversal measured on Meta's official backend comes from its host-bound partial path, not from the co-design
idea. On fused kernels the idea is right in direction and memory, but worth ≤ 1.12× here, not ~2×.
