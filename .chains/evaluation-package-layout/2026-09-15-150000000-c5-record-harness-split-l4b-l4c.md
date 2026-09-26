---
chain: "evaluation-package-layout"
branch: "main"
parent: "2026-09-15-070000000-plan-v-three-packages-one-direction.md"
nextStep: "None in this plan (C5 merged 6900306 at d9a3200). Open: L4-b (what the old harness's quality pass did to linr_v4's batch) needs the golden worktree; write_query_set / merge_prep_log and the ETL helper dedup land with the first E-phase loader that needs them."
created: "2026-09-15T15:00:00Z"
---

# §11 record: WP-V1 + WP-V2 (roadmap C5), 2026-09-15, plus L4-b and L4-c

Branch `dev/c5-harness-split` off `development` @ `71430d7`, worktree `/workspace/wt/c5`. Driver 580.159.04, nvcc 12.4, Python 3.11, torch 2.10.0+cu128, triton 3.6.0, ruff 0.15.6 (uvx), `/venvs/c5` with the official extra, `TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-c5`, datasets at `/workspace/data` (`evaluation/data` a symlink); GPU shared with the L4 worker under `flock`. `retrieve/` untouched: library subtree `0fe440d4bc90...`, the same as every C4 record, so `code_version` carried across the rename. Artifacts: `docs/artifacts/evaluation-package-layout/` (`c5/`, `c5_gpu_run.sh`, `c5_compare.py`, `l4b_chunk64.py`).

## Done
`retrieval/` -> `bench/` (`bench.py` -> `measure.py`, `data.py` -> `inputs.py` minus readers, `encode.py` -> `training/encode.py`); `records.py` (`SCHEMA_VERSION`, `KEY_FIELDS`, `resume_key`, `record_path`, `samples_path`, `append_record`, `read_records` / `read_keys`, `flatten -> flat.csv`); `algos.py` as the X §3 table (`ALGOS` name -> library class, `PATHS` derived from `DISPATCH`, `filter_backend()`, the library's own `capturable`; `build()` constructs and calls `register_index`; `run.perf` switches the plan cache through `OfficialConfig`); `eval_datasets/layout.py` (readers and checks, `atomic_write`, `apply_users_limit` the one site, `validate_layout` behind `bench check`); `hf_io.py` -> `hub.py`; seven ETL scripts under `etl/` behind `eval-data`; `training/train.py` + `checkpoints.py` behind `train`; `training/evaluate.py` with its own `hits_at` / `recall_at_k` / `ndcg_at_k` (the cycle gone); 12 scripts -> 3; one test tree with root `conftest.py`; new `test_dependency_direction.py`, `test_paths.py`, `test_records.py`, `test_layout.py`, `test_encode.py`; pre-v2 results -> `results/archive/`; `config/{pubmed,yfcc10m}.yaml` in v2 shape (in no suite); docs and CLAUDE.md commands; L4-c. Code commit: 109 files, +2,356 / -2,013, 65 renames.

## Gates, CPU
ruff clean (the 11 pre-existing E501s cleared; `format --check` still names 7 pre-existing files, left alone); `CUDA_VISIBLE_DEVICES="" uv run pytest tests/` 164 passed, 4 skipped (`test_yfcc.py::TestRealSlice`, yfcc10m not staged); dependency test green; `test_paths.py` green (the markdown-parsing test deleted); links 0 (9 broken by the moves, re-pointed); 20 / 20 `--help` render; `bench check` goodreads d64 / d128 / d256 and arxiv d128 ok; `flatten` 108 rows (6 records x 18 perf entries), 83 columns.

## Gate, GPU: the one cell
`bench run --dataset goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre --algo silvertorch --backend triton`, both modes: without `--seed 0` the headline block expanded to seeds {0, 1, 2} x `n_probe` {24, 32} = 6 cells, all ok, 480-509 s each, 13:54-14:43 UTC; no `reason` on any of 54 graph variants.
| cell | key block vs C4 | quality (31 fields) | code_version |
|---|---|---|---|
| silvertorch/triton c0_genre n_probe=24 | equal | equal to the digit | 0fe440d4..., both |
| silvertorch/triton c0_genre n_probe=32 | equal | equal to the digit | 0fe440d4..., both |
Record identity survives the rename (a C5 `resume_key` finds the C4 record).

## 11.2 L4-b: the linr_v4 chunk-64 experiment, attribution not confirmed
`l4b_chunk64.py` sets `run.QUALITY_CHUNK = 64` (stays 16 in the code). `linr_v4/triton`, goodreads d128 `c0_genre`, quality only, seeds {0, 1, 2} identical:
| k | metric | v2 chunk 64 | golden | |Δ| vs golden | v2 chunk 16 (C4) | |Δ| vs C4 |
|---|---|---|---|---|---|---|
| 100 | recall | 0.981765912959 | 0.982053974462 | 2.88e-4 | 0.982127004293 | 3.61e-4 |
| 100 | ndcg | 0.986845961920 | 0.987055010100 | 2.09e-4 | 0.987106857471 | 2.61e-4 |
| 500 | recall | 0.985861852325 | 0.986093519232 | 2.32e-4 | 0.986102850663 | 2.41e-4 |
| 500 | ndcg | 0.988789057119 | 0.988973464071 | 1.84e-4 | 0.988980759199 | 1.92e-4 |
| 1000 | recall | 0.987059843378 | 0.987315650142 | 2.56e-4 | 0.987317780281 | 2.58e-4 |
| 1000 | ndcg | 0.989421225299 | 0.989631064621 | 2.10e-4 | 0.989632714459 | 2.11e-4 |
Chunk 64 lands four times further from the golden at recall@100 than chunk 16 and moves k=500 / 1000 by 2.3-2.6e-4 (all down). Batch shape moves `linr_v4`, but "golden at 64, v2 at 16" is not the separation; the open question needs the golden worktree. Held-out recall@100 0.212285 at chunk 64 vs 0.211894 at 16.

## 11.3 L4-c, fixed
`measure.clocks()` is one `nvidia-smi` sample; `expected_sm_mhz` / `clocks_locked` / `--expected-sm-mhz` gone; `env.sm_mhz_idle` (+ `mem_mhz`, `sm_max_mhz`, `power_limit_w`) is the process-start sample; `clocks_drift` compares only under-load perf samples against the process's first under-load sample (`CLOCK_DRIFT` 5 %); `env.sm_mhz_load` the median; `SCHEMA_VERSION` 2. On the gate run 4 of 6 cells `clocks_drift: false` (all 1410 MHz; the old rule would have flagged all six); 2 flagged each carry one variant at 1185 / 1290 MHz, a real dip. `unstable` still true on 5 of 6 from window spread (seed 0 n_probe 24: `graph k=100 bs=1` 14.8 %, `eager k=1000 bs=8` 26.6 %).

## 11.4 Deviations and not done
`Dataset` stays in `bench/config.py` (its fields are harness config; parity is `validate_layout` over paths); `write_query_set` / `merge_prep_log` and the `retry_download` / `checksum` / `year_to_bucket` dedup not done; ETL CLIs stay argparse, mounted as pass-through subcommands (~600 lines of wiring for no behaviour otherwise); `training.encode`'s cache key unchanged (holds the full split); `SCHEMA_VERSION` 1 -> 2; `validate_layout`'s sequential shape asks only for `item_id_map.json` + `test.parquet`; the layer-level `k` tests dropped (library's `test_boundary.py` owns them); the gate ran 6 cells (50 min GPU); the L shim untouched (the library belonged to L4 that day). Not run: `bench campaign` on the GPU, ETL bodies, `train sasrec` beyond `--help`, `bench upload`, `official` through the new `run.perf` path on the GPU. `.gitignore` gained `evaluation/data` and the C5 logs.
