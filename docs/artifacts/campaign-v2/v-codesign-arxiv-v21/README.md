# V-CODESIGN, arXiv half, at campaign-v2.1

Roadmap Phase V, V-CODESIGN. This is the arXiv half; pod 1 runs the goodreads half.
Records on the Hub as `campaign-v2.1/arxiv-codesign`. They replace the stale 408b1188 arXiv records in `artifacts/v-codesign-408b`.
The state is in [validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`summary.txt`](summary.txt) | the per-point table below, from [`codesign_summary.py`](../v-codesign/codesign_summary.py) over `codesign/arxiv-d128.jsonl` |

## How it ran
- **Pod and code.** Pod c, A100-SXM4-80GB GPU 0, `taskset -c 64-127`, `/venvs/retrieve`. Library **`f01255f1`**, worktree `dev/v-codesign-arxiv-v21` (staging fa3a730).
- **Driver.** Pod c's one-leg driver, published as `docs/artifacts/campaign-v2/d3-arxiv/driver_v21.sh` with the D3 arXiv rerun. Under `flock /scratch/gpu0.lock`.
  Before any cell it checks: `retrieve` imported from the main checkout, identical library trees, and `bench env` equal to campaign.yaml's code_version.
  Then `bench oracle`, then `bench campaign --suite codesign --dataset arxiv --resume --interleave --timeout 48`,
  into the fresh tree `/scratch/campaign-v2.1/results-arxiv-codesign`.
- **Oracle at v2.1.** Just before this leg, the arXiv k 100 blobs it reads were rebuilt at v2.1 and are `torch.equal` to the 408b1188 ones
  (`oracle_v21.sh` in `docs/artifacts/campaign-v2/d3-arxiv/`).
- **Grid.** SilverTorch official, bloom, `n_lists` 1664, `bloom_path` {partial, full} interleaved, `n_probe` {8, 32, 128}, bs {1, 16}, k 100,
  sweeps `c3_nversions` / `c0_maincat` / `all4`, seeds 0-2.
- **Totals.** 54 / 54 ok, rc 0, 869 s (oracle 7 s, campaign 862 s) = 0.24 GPU-h.
- **Clocks.** 242 of 324 timing windows below 1410 MHz (minimum 1155); 39 of 54 records `unstable`. The ratios are paired within interleave groups.
- **Graph mode.** Official is eager-only, so its graph entries are empty.

## Results (median over seeds; full / partial paired per interleave group, 95 % CI)

| | bs 1 | bs 16, n_probe 8 / 32 | bs 16, n_probe 128 |
|---|---|---|---|
| full / partial | 0.818-0.823 | 0.847-0.855 | 0.876-0.880 |
| partial ms | 1.29-1.30 | 1.55-1.67 | 1.88-1.94 |
| full ms | 1.06-1.07 | 1.31-1.42 | 1.65-1.71 |

- **Full is faster than partial** at all 18 (sweep, n_probe, bs) points, and every CI excludes 1. Recall is identical between the two paths:
  @100 0.52 / 0.75 / 0.90 (`all4`), 0.74 / 0.88 / 0.94 (`c0_maincat`), 0.71 / 0.88 / 0.94 (`c3_nversions`) at n_probe 8 / 32 / 128.
- **Against the stale 408b1188 arXiv records:**
  - same direction and size (full / partial 0.80-0.87 there);
  - absolute times are ≈ 40-130 µs higher, which includes the quantize fix's eager cost.
