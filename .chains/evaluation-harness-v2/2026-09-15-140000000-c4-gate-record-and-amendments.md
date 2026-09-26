---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-15-080000000-a1-golden-re-derive-and-l1-gate-cell.md"
nextStep: "C4 closed under the user's steer; carry L4 (linr_v2 backend divergence), L4-b (linr_v4 chunk-64), L4-c (clocks_drift / clocks_locked); then C5 and D4."
created: "2026-09-15T14:00:00Z"
---

# §12 record: C4 / WP-4 GPU gate, 2026-09-15, and the gate amendments

Branch `dev/c4-gate-rerun` off `development` @ `afc2ab9`. Verdict as run: not green. Gates 3 and 6 pass; gate 1 9 of 11; gate 4 passes on the roadmap's cell and the SilverTorch pair; gates 2 and 5 fail as written for measurement reasons. Merged as `32d4877` at `c1ae1b5`; closed under the steer below.

Environment: A100-SXM4-80GB, driver 580.159.04, nvcc 12.4, torch 2.10.0+cu128, triton 3.6.0, Python 3.11, `silvertorch` 1.0.0; venv `/venvs/retrieve` used with `PYTHONPATH` (never `uv run`: the shared venv's editable pointer may belong to another worktree); private `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-c4`; sole GPU worker, `flock`. Every record `code_version = 0fe440d4bc9013097737e225af59b71ccec94a6d`, `dirty: false`. Runbook `docs/artifacts/evaluation-harness-v2/c4_gate_run.sh` (stages goodreads -> arxiv -> resume -> gate; `GOLDEN_SM_MHZ` env default 1155). goodreads 08:50:26-11:15:28 14/14 ok; arxiv 11:20:06-11:51:47 6/6 ok; resume pass; 20 records, 360 perf entries, 0 failed; all 11 golden cells matched exactly one record.

## 12.1 Inputs checked
v2 re-encoded all 313,178 queries (its own cache key): `item_embs [797084, 128]`, `queries[:10000]`, `targets[:10000]`, `n_targets[:10000]` bit-identical to A1's blob. goodreads oracle cache hit (`oracle_v4_c0_genre_2634b8ef02f0f10c.pt`). arxiv's published `gt_d128/` had only `gt_topk_v3_c0_maincat.pt`, so one oracle built in ~40 s for 10,000 x 2,988,996; its `topk` exactly equal to the golden's blob.

## 12.2 Per-cell gate table
| cell | 1 quality | 2 latency | 3 graph | 4 parity | 5 stable |
|---|---|---|---|---|---|
| gr linr_v1_filter_mask-triton | PASS | FAIL | PASS | ref | PASS |
| gr linr_v1_filter_mask-torch | PASS | FAIL | PASS | PASS | FAIL |
| gr linr_v2-triton | PASS | FAIL | PASS | ref | FAIL |
| gr linr_v2-torch | PASS | FAIL | PASS | FAIL | FAIL |
| gr linr_v3-triton / torch | PASS | FAIL | PASS | INFO | FAIL |
| gr linr_v4-triton | FAIL | FAIL | PASS | ref | FAIL |
| gr linr_v4-torch | FAIL | FAIL | PASS | INFO | PASS |
| gr silvertorch-triton np24 | PASS | FAIL | PASS | ref | FAIL |
| gr silvertorch-torch np24 | PASS | FAIL | PASS | INFO | FAIL |
| gr silvertorch-official np24 / np32 | INFO | INFO | PASS | PASS | FAIL |
| ax silvertorch-triton np24 | FAIL | FAIL | PASS | ref | FAIL |
| ax silvertorch-official np24 / np32 | INFO | INFO | PASS | FAIL | FAIL |
(np32 triton / torch rows INFO except graph PASS and stable FAIL.) Full table `c4/gate.txt`: PASS 88, FAIL 127, INFO 23.

## 12.3 Gate 1 (quality within 1e-6): 9 of 11
52 of 66 rows pass; worst passing delta 7.5e-9; six cells exact to 0.0. The two cells §11.8 called unmeetable now pass (L3): `linr_v2-triton` 3.6e-11 (own noise before L3 2.0e-6), `linr_v3-triton` 1.6e-9 (was 6.8e-5). Two fail:
| cell | k=100 | k=500 | k=1000 |
|---|---|---|---|
| linr_v4 (both backends) recall@k | 7.3e-5 | 9.3e-6 | 2.1e-6 |
| linr_v4 ndcg@k | 5.2e-5 | 7.3e-6 | 1.6e-6 |
| ax silvertorch-triton np24 recall@k | 2.0e-6 | 6.0e-12 | 6.0e-12 |
| ax silvertorch-triton np24 ndcg@k | 1.4e-6 | 7.5e-9 | 6.5e-9 |
Not the `k_max` slice: rerun at `--k 100` gives identical v2 numbers (`gr linr_v4-triton` 0.982127004293 at both, golden 0.982053974462; `ax silvertorch-triton` 0.884044068151 at both, golden 0.884042068159; `c4/kmax-diag/`). `linr_v4` probe (`c4_linr_v4_probe.py`): not the library (bit-identical A1 vs re-derive), not eager vs compiled (0/2048 rows differ), but the query-batch shape: chunk 16 vs 64 on 2048 rows: `linr_v1_filter_mask` 0 rows differ; `linr_v4` 1536 rows differ, max |Δscore| 196, set overlap@100 0.992065430 (`PostfilterKNNInt8`'s `>>5` compression ties; pads below `_PAD_M = 17`, and v2's `QUALITY_CHUNK = 16` always pads). (This attribution was later falsified by C5's L4-b.) The arxiv 2.0e-6 (~2 single-hit changes in 10,000 queries) unattributed: not the slice, not batch shape (0/4096 rows differ).

## 12.4 Gate 2 (graph latency within 5 %): an estimator artifact
At `--golden-sm-mhz 1155`, 92 of 99 rows fail. The golden's clock is a 30 s-cadence whole-run median dominated by build / quality / oracle phases; v2's `perf[].sm_mhz` is one under-load sample (1410 on all 360 entries). Traces identical: golden 65 samples median 1155 (1155x55, 1410x9, 210x1); C4 41 samples median 1155 (1155x31, 1410x9, 210x1). The normalisation injects a flat 1410/1155 = 1.221. At `--golden-sm-mhz 1410` (`c4/gate-matched-clock.txt`):
| bs | PASS | FAIL | ratio v2/golden |
|---|---|---|---|
| 1 | 17 | 16 | 0.799-1.210 |
| 8 | 33 | 0 | 0.957-1.026 |
| 16 | 33 | 0 | 0.975-1.030 |
14 of 16 bs=1 failures have v2 faster (the no-flush direction). Golden's own repeat noise (`c4_latency_evidence.py --golden-noise`): bs=1 max 21.1 %, median 14.1 %; bs=8 0.4 %; bs=16 0.1 %. v2 windows <= 0.3 % at bs=1. `--flush-l2` not run; bs 8/16 bound any flush effect at <= 3 %.

## 12.5 Gate 3: PASS 20/20 (official null with `not_capturable` only).
## 12.6 Gate 4
SilverTorch torch vs triton jaccard@100 1.000000, diff 0.0 on both datasets and both `n_probe`. Roadmap clause: `gr silvertorch-official` 0.999849 (np24) / 0.999805 (np32) >= 0.99, `cache_plans: false` on all 9 timed entries. Fails: `gr linr_v2-torch` vs triton 0.998743, diff 9.766e-03 (also in the golden: recall@100 0.9996946954935145 torch vs 0.9992737607088252 triton; v2 reproduces each to 1e-11; became L4); `ax silvertorch-official` 0.985033 / 0.984882, diff 4.827e-04 (arxiv not named by the roadmap; later explained by B3 as fp16).
## 12.7 Gate 5: not evaluable
18 of 20 flagged: 8 real window spread > 5 % (11 of 14 outliers at bs=1; worst `linr_v3-triton` k=100 bs=1 eager 22.8 %, `silvertorch-triton` k=500 / 1000 bs=1 graph 14.9 % / 14.3 %), 10 `clocks_drift` only (a 1155 MHz process-start idle sample vs 1410 under load: the flag fired on boosting), 2 clean. `clocks_locked` read `true` on arxiv cells by coincidence. Precondition "at locked clocks" unavailable.
## 12.8 Gate 6: PASS
SIGTERM at 12:01:03 after the first record; `--resume` logged `all 1 cells done`, finished `{'skipped': 1, 'ok': 1}`, no duplicate (`c4/resume/`).
## 12.9 Not done
No checkbox, no merge by the worker; `--flush-l2`; the arxiv 2.0e-6; seed 0 only; no bloom cells. The 2026-09-06 partial run moved to `c4/2026-09-06-partial/`. Parity spills gitignored.

## Superseding decision 2026-09-15 (user) and the WP-4 amendments
"We don't care about reproducing old results now, we're improving all code and rewriting, then testing and profiling, then running the full evals step by step." The golden is informational, not a gate; clauses (1) and (2) no longer block; what blocks is (3), (4), (6), the parity spill and the official cell. The paper may not claim equivalence with the pre-v2 harness (P G9). Amendments recorded rather than edited into the gate text:
- (1) kept, two residuals: `linr_v4` 7.3e-5 (C5's L4-b later falsified the chunk attribution: chunk 64 lands 2.9e-4 from the golden, four times further; k=500/1000 move 2.3-2.6e-4; unexplained) and arxiv `silvertorch` 2.0e-6 (bounded, unexplained).
- (2) re-specified as matched-estimator, bs >= 8: 66/66 pass, ratios 0.957-1.030; bs=1 excluded (baseline noise up to 21.1 %).
- (3), (6) passed.
- (4) split: "exact algo" means exact filtering, not bit-identical arithmetic across two fp16 dot implementations. `linr_v2` divergence carried as roadmap L4.
- (5) not evaluable here; `clocks_drift` and `clocks_locked` carried as L4-c (fixed in C5).
