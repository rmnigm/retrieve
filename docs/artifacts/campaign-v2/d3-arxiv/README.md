# D3 arXiv: `bloomwidth` (campaign-v2) and `bloomwidth-timed` (campaign-v2.1)

Roadmap Phase V, D3, the arXiv piece. Records on the Hub:
- `campaign-v2/arxiv-bloomwidth`, quality only, library `408b1188`. It stays valid: v2.1 changes no quality output.
- `campaign-v2.1/arxiv-bloomwidth-timed`, library `f01255f1`. This replaces `campaign-v2/arxiv-bloomwidth-timed`, which is stale because
  the quantize fix adds ≈ 30 µs to every SilverTorch eager path.

The state is in [validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | the `bloomwidth` driver (408b1188), one suite per run under `flock /scratch/gpu0.lock` |
| [`run.sh`](run.sh) | waited for an idle GPU, then ran both suites at 408b1188, releasing the lock between them |
| [`driver_v21.sh`](driver_v21.sh) | the v2.1 driver (pod c's generic one-leg driver): see the checks below |
| [`upload_v21.sh`](upload_v21.sh) | copies the logs into the tree, then `bench upload --verify` under campaign.yaml's `hub` pattern |
| [`oracle_v21.sh`](oracle_v21.sh), [`compare_oracles.py`](compare_oracles.py) | rebuild arXiv's kept-sweep oracle blobs at v2.1 (408b1188 blobs moved to `gt_d128/_408b1188/`) and `torch.equal` them against the old ones |
| [`summary.py`](summary.py) | per (suite, backend, sweep, width): FPR, recall, index size, median eager / graph latency. With d1's `filter` JSONL it also prints the default-width check against `d1/arxiv` |

## How it ran

Both legs ran on GPU 0 of the one-GPU A100-SXM4-80GB pod c, with `/venvs/retrieve`, `taskset -c 64-127`,
on arXiv d128, sweeps `c3_nversions` / `c0_maincat` / `all4`, seeds 0-2.
Another worker shared the GPU through the lock, never concurrently.

`driver_v21.sh` refuses to start unless:
- `retrieve` is imported from the main checkout;
- that checkout's library tree is identical to this worktree's;
- `bench env` equals campaign.yaml's `default.perf.code_version`, and that value is not 408b1188.

| suite | library | cells | rc | wall | notes |
|---|---|---|---|---|---|
| `bloomwidth` (quality only) | `408b1188` | 126 / 126 ok | 0 | 3,010 s | triton `m_bits` {64..2048} × `k_hash` {3, 5}; official `k_hash` {3, 5}; k {100, 1000}, bs 16 |
| `bloomwidth-timed` | `f01255f1` | 63 / 63 ok | 0 | 980 s (oracle 6 s, campaign 974 s) | triton `m_bits` {64..2048} × `k_hash` 5; official `k_hash` 5; k 100, bs 16, interleaved |

- **Timed-leg clocks:** 10 of 351 windows below 1410 MHz (minimum 1200); 3 of 63 records `unstable`.
- **Official graph entries are empty**, because official cannot be captured.
- **Oracle at v2.1.** The timed leg read the k 100 blobs built at 408b1188. `oracle_v21.sh` then rebuilt all three at v2.1, and every tensor
  (`topk`, `pass_counts`, `targets_in_filter`, `target_in_filter`) is `torch.equal`; only the provenance fields differ.
  This is as expected: the oracle calls none of the files v2.1 changed (`quantize.py`, `fused_masked_knn_topk`, `oporp_1bit_match_topk`).

## Results (median over seeds)

Bloom false-positive rate (`bloom_fp_rate`) for triton. It is 0 at every width ≥ 256 and for official on every sweep:

| sweep (pass rate) | 64 / k 3 | 64 / k 5 | 128 / k 3 | 128 / k 5 |
|---|---|---|---|---|
| `c3_nversions` (0.444) | 1.7e-3 | 4.5e-4 | 0 | 0 |
| `c0_maincat` (0.136) | 1.6e-3 | 3.6e-5 | 7.0e-5 | 0 |
| `all4` (0.0094) | 1.3e-4 | 3.0e-6 | 3.1e-6 | 0 |

- **Recall.** The FPR shows up only at 64 bits: recall_oracle@100 drops 0.0013 on `c3_nversions` at k 5 (0.8755 vs 0.8768), and ≤ 0.0037 at k 3.
  From 128 bits up, triton recall is identical at every width: @100 0.8768 / 0.8834 / 0.7611.
- **Index size.** Triton index_mib 433.8 / 456.6 / 502.2 / 593.4 / 775.9 / 1140.7 at 64-2048 bits. Official is 453.8 (k 3) and 482.3 (k 5).
- **Latency** at bs 16, k 100, v2.1, does not depend on width:
  - triton eager 0.866-0.870 ms, graph 0.260-0.262 ms;
  - official eager 1.54 ms (`c3_nversions`, `c0_maincat`) and 1.67 ms (`all4`).
- **Against the stale 408b1188 timed leg:** triton eager +≈ 34 µs (0.834 → 0.868), official eager +≈ 24 µs (1.52 → 1.55),
  graph unchanged. This is the quantize fix's expected eager cost. Quality, FPR and index size are identical.

## Check against d1/arxiv (surprise gate)

At the default width (`m_bits` 1024, `k_hash` 5, `n_probe` 24) the `bloomwidth` cells reproduce d1/arxiv's `filter` silvertorch bloom cells (code_version `72e5a90`) per seed:
- FPR is 0 in both, and `index_mib` is equal (775.9 triton, 482.3 official).
- recall_oracle@100 is equal on 13 of 14 (sweep, seed, backend) pairs.
- The exception is official `c0_maincat` seed 0, at 0.883735 vs 0.883757: Δ 2.2e-5, a single neighbour, with recall@1000 equal.
  That is within the official fp16 path's known near-tie effect.

At v2.1 the official / triton eager ratio is about 1.78-1.92, against H2H-FINAL's 1.87-2.18 bloom range at 408b1188.
H2H-FINAL is itself stale at v2.1 and reruns.
