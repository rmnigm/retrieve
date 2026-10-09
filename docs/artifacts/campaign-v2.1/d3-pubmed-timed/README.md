# D3 PubMed `bloomwidth-timed` at campaign-v2.1, and the surprise gate it tripped

Roadmap D3, the PubMed timed piece, on pod `a100-x1-b` (1x A100-SXM4-80GB, GPU 0 shared through `flock`), code_version
`f01255f1`. State: [validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | `bench oracle --suite bloomwidth-timed` (rebuilt at v2.1), then the timed campaign into `/scratch/campaign-v2.1/results` |
| [`surprise-run.sh`](surprise-run.sh), [`surprise-bw.yaml`](surprise-bw.yaml) | the surprise gate's one-cell rerun: a scratch suite appended to a copy of `config/suites.yaml`, Triton and official at the shared bloom default in one interleave group, `--interleave --profile` |

## `bloomwidth-timed` (21 / 21 ok)

PubMed d768, bloom `c0_mesh`, seeds 0-2, bs 16, k 100, `n_lists` 1024, `n_probe` 24; Triton `m_bits` 64-2048 at
`k_hash` 5, official at its own width. Hub `campaign-v2.1/pubmed-bloomwidth-timed`, MANIFEST sha256
`da0d5b0e5e3a5559ec6c2a62e1c8c89a03c792e51856551812eefd30d1fa6beb`. Triton 1,112 s + official 148 s + oracle 45 s =
**0.36 GPU-h**. Clocks: 117 / 117 windows at 1410 MHz, 0 / 21 records `unstable`.

| arm (median over seeds) | eager ms | graph ms | recall_oracle@100 |
|---|---|---|---|
| Triton 64 / 128 / 256 / 512 / 1024 / 2048 bits | 2.983 / 2.949 / 2.913 / 2.918 / 2.903 / 2.915 | 2.887 / 2.830 / 2.814 / 2.803 / 2.804 / 2.807 | 0.4646 / 0.6400 / 0.6690 / 0.6699 / 0.6699 / 0.6699 |
| official (k_hash 5) | **1.986** | not capturable | 0.6698 |

Latency is flat in width; recall saturates at 512 bits.

## Surprise gate: official is faster than Triton at d768

Official below Triton end to end contradicts H2H-FINAL and D3 arXiv / goodreads (d128: official / Triton eager 1.5-2.2), the
roadmap's surprise-gate case. The `filter` leg was stopped before any timed cell, and the one cell was rerun interleaved
(Hub `artifacts/d3-pubmed-surprise`, MANIFEST sha256 `a6037ea582e322236aa926c9f037904dc83bb3bbbf5433c8984a505ff00e79fe`):
seed 0, bloom default 1024 / 5 for both arms, interleave group `d75528625a422c80`, all windows 1410 MHz.

| arm | eager ms (3 windows) | graph ms |
|---|---|---|
| official | **1.995** (1.994 / 1.995 / 1.995) | not capturable |
| Triton | 2.841 (2.843 / 2.840 / 2.835) | 2.725 |

**official / Triton = 0.702 eager, 0.732 against Triton graph**; recall@100 0.664 on both (Jaccard@100 0.984, the known
bloom false-positive divergence). `--profile` (eager; the profiler fix is unmerged, so tail kernels may be missing): Triton
2,663 of 2,840 µs in the top 8, of which **`_codesigned_probe_score_kernel` 2,311 µs**, then the `[B, N]` top-k (~0.33 ms);
official 795 of 1,995 µs (about 1.2 ms not captured), its scorer `st::ops::fused_kmean_ann::process_cluster<int8, half,
768, …>` 72 µs as profiled, the same top-k. Triton's tile at d768 (`_host.tile_for_width`, `D_PAD` 1024): `BLOCK_P` 64 ×
4 warps for none / bloom, 128 × 4 for exact (the per-width tile; the `[BLOCK_P, D_PAD]` code tile sits in registers with no
loop over D). The controller's decision: roadmap ST-DLOOP (the probe scorers loop over D), V-PUBMED at campaign-v2.2.

**`bloomwidth-timed` cannot interleave Triton with official.** It declares no `interleave:` group, and `by: backend` could
not pair its arms anyway: a group needs equal keys but for the `by` fields, and Triton's build carries `m_bits` where
official's does not ([evaluation](../../../system/evaluation.md#interleaved-groups)). The rerun's first attempt hit exactly
that (`build: {m_bits, k_hash}` vs `{k_hash}`: two units, run back to back: official 2.157 vs Triton 2.838 / 2.723 ms); the
interleaved run above gives both arms no build params, so both take the suite's bloom default.
