# D3 arXiv: `bloomwidth` and `bloomwidth-timed`

Roadmap Phase V, D3, the arXiv piece. The records are on the Hub as `campaign-v2/arxiv-bloomwidth` and
`campaign-v2/arxiv-bloomwidth-timed`. The state is in
[validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | one suite per run, under `flock /scratch/gpu0.lock`. It checks that `retrieve` is imported from the main checkout, that that checkout's library tree is identical to this worktree's, and that code_version is `408b1188`. Then `bench oracle`, then `bench campaign --resume` (`--interleave` on the timed suite) |
| [`run.sh`](run.sh) | waits until the GPU has no compute process, then runs both suites, releasing the lock between them |
| [`summary.py`](summary.py) | per (suite, backend, sweep, width): FPR, recall, index size and median latency. With d1's `filter` JSONL it also prints the default-width check against `d1/arxiv` |

## How it ran

The run used GPU 0 of the one-GPU A100-SXM4-80GB pod c, `/venvs/retrieve`, and `taskset -c 64-127` (GPU 0's NUMA-local cores on this pod).
Code was at worktree `dev/d3-arxiv` (staging `c3f1894`) with code_version `408b1188`, on arXiv d128, sweeps `c3_nversions`, `c0_maincat` and `all4`, seeds 0-2.
A second worker on the pod (`v-graph-ids`) used the GPU between the two suites, through the same lock, never concurrently.

| suite | cells | rc | wall | notes |
|---|---|---|---|---|
| `bloomwidth` (quality only) | 126 / 126 ok | 0 | 3,010 s | triton `m_bits` {64..2048} × `k_hash` {3, 5}; official `k_hash` {3, 5}; k {100, 1000}, bs 16 |
| `bloomwidth-timed` | 63 / 63 ok | 0 | 1,023 s | triton `m_bits` {64..2048} × `k_hash` 5; official `k_hash` 5; k 100, bs 16, interleaved |

Total wall time was 4,033 s (1.12 GPU-h), against the ≈ 3-5 GPU-h estimate for all three datasets.

In the timed leg, 15 of 351 timing windows ran below 1410 MHz (minimum 1170), and 3 of 63 records are `unstable`.
Official's graph entries are empty, because official cannot be captured.

## Results (median over seeds)

Bloom false-positive rate (`bloom_fp_rate`) for triton. It is 0 at every width ≥ 256 and for official on every sweep:

| sweep (pass rate) | 64 / k 3 | 64 / k 5 | 128 / k 3 | 128 / k 5 |
|---|---|---|---|---|
| `c3_nversions` (0.444) | 1.7e-3 | 4.5e-4 | 0 | 0 |
| `c0_maincat` (0.136) | 1.6e-3 | 3.6e-5 | 7.0e-5 | 0 |
| `all4` (0.0094) | 1.3e-4 | 3.0e-6 | 3.1e-6 | 0 |

- **Recall.** The FPR shows up only at 64 bits: recall_oracle@100 drops 0.0013 on `c3_nversions` at k 5 (0.8755 vs 0.8768), and ≤ 0.0037 at k 3.
  From 128 bits up, triton recall is identical at every width: @100 0.8768 / 0.8834 / 0.7611.
- **Index size.** Triton index_mib 433.8 / 456.6 / 502.2 / 593.4 / 775.9 / 1140.7 at 64-2048 bits. Official is 453.8 (k 3) and 482.3 (k 5); it ignores `m_bits`.
- **Latency** at bs 16, k 100 does not depend on width:
  - triton eager 0.831-0.837 ms, graph 0.259-0.263 ms on every sweep and width;
  - official eager 1.52 ms (`c3_nversions`, `c0_maincat`) and 1.64 ms (`all4`).

## Check against d1/arxiv (surprise gate)

At the default width (`m_bits` 1024, `k_hash` 5, `n_probe` 24) the `bloomwidth` cells reproduce d1/arxiv's `filter` silvertorch bloom cells (code_version `72e5a90`) per seed:
- FPR is 0 in both, and `index_mib` is equal (775.9 triton, 482.3 official).
- recall_oracle@100 is equal on 13 of 14 (sweep, seed, backend) pairs.
- The exception is official `c0_maincat` seed 0, at 0.883735 vs 0.883757: Δ 2.2e-5, a single neighbour, with recall@1000 equal.
  That is within the official fp16 path's known near-tie effect ([validation](../../../validation.md)).

The timed leg's official / triton eager ratio, about 1.82-1.97, sits within or just below H2H-FINAL's bloom range of 1.87-2.18. So there is no surprise.
