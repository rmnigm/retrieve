---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-06-180000000-a1-golden-baseline-record.md"
nextStep: "Decide gate 1 for linr_v2 / linr_v3 triton (exclude, interval, or fix the non-determinism: roadmap L3 chose the fix), then run the C4 gate on the L2 layout."
created: "2026-09-15T08:00:00Z"
---

# §11 record: A1 golden re-derive, 2026-09-15 (roadmap eval-queue item 1)

Why: §10's cells predate the three C4 library fixes (atomic k-means, compiled-eager V2/V3 graph cells, no `-1` sentinel), so C4's 1e-6 comparison would have measured the library change. Fix: run the old harness unchanged against the current library, everything else fixed.

Environment: A100-SXM4-80GB, driver 580.159.04, nvcc 12.4, torch 2.10.0+cu128, triton 3.6.0, Python 3.11, `silvertorch` 1.0.0 at `21aa35e`; fresh rental. Worktree `/workspace/wt/golden`, branch `tmp/golden-rederive` off `origin/dev/a1-golden` (`1ccdb27`) with `git checkout 4f52972 -- retrieve/`; library tree `28fda5ae` (A1's `6dc72aac`); rows carry `extra.commit = 87a9b38`; throwaway, never merged. `/venvs/golden`. Datasets to `/workspace/data` (goodreads 2.6 GB d128, arxiv 1.3 GB `content_d128`), confirmed still the 1-indexed Hub layout. GPU shared with the L1 worker under `flock /workspace/gpu.lock`. Clocks: 1155 MHz median under load (A1: 1140), 1410 peak, 210 idle, 26-39 °C. Runbook `docs/artifacts/evaluation-harness-v2/a1-rederive/a1_golden_rerun.sh`, `STAGES=golden` (flock, `uv run --no-sync`, `NO_CLOCK_LOCK=1`, a private `TORCHINDUCTOR_CACHE_DIR`).

## 11.1 Inputs provably A1's
Logs show `loaded encoded_queries from cache` and `loaded oracle from cache` (the checkpoint mtime set back to the blob's recorded `ckpt_mtime`); kept users 9,859 / 10,000 and 10,000 / 10,000.

## 11.2 Cells, 11/11 green (29.4 min cell time)
| cell | vs A1 | max abs delta | direction |
|---|---|---|---|
| linr_v1_filter_mask triton / torch, linr_v2-torch, linr_v3-torch, linr_v4 triton / torch | identical | 0 | - |
| linr_v2-triton | moved | 5.07e-6 | mixed |
| linr_v3-triton | moved | 2.43e-5 | up |
| arxiv silvertorch-triton | moved | 3.50e-5 | mixed |
| goodreads silvertorch-triton | moved | 7.79e-5 | mixed |
| goodreads silvertorch-torch | moved | 1.57e-4 | up |
Row-level diff: `docs/artifacts/evaluation-harness-v2/a1-rederive/quality-diff.md`. Moves identical at all three batch sizes.

## 11.3 The 2026-09-06 prediction was wrong in three of four claims
1. The `-1` sentinel cannot move a goodreads number: of 10,000 `c0_genre` queries exactly 141 have zero survivors and 9,859 the full 1,000, and the 141 are exactly the dropped users. On arxiv it can act on only 4 / 6 / 11 of 10,000 rows at k = 100 / 500 / 1000.
2. `torch` moved (silvertorch-torch by 1.57e-4, the largest delta): deterministic k-means changes the centroids on every backend.
3. Moves go up as well as down.
4. The two LiNR triton moves: withdrawn by 11.8 (run-to-run noise).

Replacement result: SilverTorch torch vs triton now agree exactly on goodreads:
| k | A1 torch vs triton | re-derived |
|---|---|---|
| 100 | recall 1.156e-4, ndcg 8.523e-5 | 0.0 / 0.0 |
| 500 | 1.550e-4, 1.279e-4 | 0.0 / 0.0 |
| 1000 | 1.717e-4, 1.444e-4 | 0.0 / 0.0 |
A1's gap was the non-deterministic k-means, not tie order.

## 11.4 Consequences for C4
Five cells moved 5x-160x the tolerance; a gate against the old cells would have failed unattributably. Carry `--golden-sm-mhz 1155` (later shown to be the wrong estimator) and a stricter gate 4 for SilverTorch.

## 11.5 Findings outside the cells
1. The old harness does not import against the current library: `retrieve.interfaces.Backend` was split at B4; the throwaway worktree declares the literal locally (`a1-rederive/harness_compat_backend.py`).
2. The harness CPU suite is only green with CUDA hidden: 3 failed, 98 passed, 3 skipped on the GPU box vs 103 passed, 1 skipped under `CUDA_VISIBLE_DEVICES=""` (`test_bench.py::test_latency_windows_and_keys`, `test_run.py::test_end_to_end_records`, `test_run.py::test_perf_times_with_the_plan_cache_off_and_records_it` assert CPU-only behaviour).
3. The Hub still publishes the 1-indexed artifacts and the `gt_d128/gt_topk_v3_*` oracle blobs (which kept this run to 33 minutes).
4. `/tmp/torchinductor_root` is shared between workers: every concurrent GPU job needs its own `TORCHINDUCTOR_CACHE_DIR`.

## 11.7 L1's gate cell
Library subtree swapped to `dev/l1-library-layout` @ `df04e74` (tree `624459b6`), committed as `52ee677` on `tmp/golden-rederive`. The `torchretrieve` distribution rename broke the frozen workspace metadata (root and `evaluation/pyproject.toml` depended on `retrieve`); fixed by taking L1's root `pyproject.toml` and renaming one dependency line; the `retrieve.kernels` shim warned as designed. `goodreads c0_genre silvertorch triton`: PASS, bit-identical, 0 of 36 quality columns differ, on two runs. `linr_v3-triton` (3 runs, 27 of 36 differ) and `linr_v2-triton` (2 runs, 24 of 36 differ): not evaluable, see 11.8.

## 11.8 linr_v2-triton and linr_v3-triton are non-deterministic run to run
| recall@k, bs=1 | golden | L1 run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|---|
| linr_v3-triton k=100 | 0.877002740643 | 0.877019983864 | 0.876952025615 | 0.876981440444 | 6.796e-05 |
| linr_v3-triton k=500 | 0.724373871613 | 0.724341413849 | 0.724360888599 | 0.724337356573 | 2.353e-05 |
| linr_v3-triton k=1000 | 0.617545795329 | 0.617553200089 | 0.617552388708 | 0.617545694154 | 7.506e-06 |
| linr_v2-triton k=100 | 0.999273760709 | 0.999275789310 | 0.999273760709 | - | 2.029e-06 |
| linr_v2-triton k=1000 | 0.999391020874 | 0.999389803697 | 0.999389296547 | - | 5.072e-07 |
Same commit, data, seed, cached queries and oracle; the golden lies inside the run-to-run range. (1) The mismatches are not attributable to L1; `silvertorch-triton` (deterministic, bit-identical across three runs on two trees) carries the gate. (2) §11.2's two LiNR moves withdrawn. (3) C4's gate 1 cannot be met on these two cells by any harness (noise 2x and 68x the tolerance): blocker for queue item 2. Hypothesis at the time (not tested): per-process Triton autotuning. The real cause, found by roadmap L3: `clause_compact` / `bloom_compact` claimed their row base with `atomic_add`, ordering each row by tile completion.

## 11.9 Ruling: do not add `Backend` to the L shim
`Backend` was deleted at B4, not moved by L; re-exporting it would resurrect a retired API (coding-guidelines D2). The alias lives in the frozen A1 harness on `tmp/golden-rederive`. Supersedes 11.5 item 1's closing sentence.
