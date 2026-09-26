---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-15-230000000-results-storage-and-bench-upload-record.md"
nextStep: "Ask the user before running more of D1 (arxiv leg, D1-b..e). Re-run bench report over all 126 goodreads cells; move the 73 MB samples sidecar to the Hub; run clause 5 (ids across mode)."
created: "2026-09-16T12:00:00Z"
---

# WP-5 staged, the grid narrowed, and §15 record: D1-a goodreads leg

## WP-5 staged (2026-09-15, orchestrator, on "running the full evals step by step")
Five resumable stages; no kernel or library change between stages (the tree hash is in the resume key).
| stage | what | rough wall (as planned) |
|---|---|---|
| D1-a | goodreads + arxiv `filter`, d128, seed 0, `{triton, torch, official}` | ~4-6 h |
| D1-b | same cells at seeds 1, 2 (headline sweeps) | ~8-12 h |
| D1-c | `quality` suite, all four datasets, all dims | ~3-4 h |
| D1-d | `deep` suite (2 builds x 6 query configs, seeds 0-2) | ~6-8 h |
| D1-e | S9 ablation cells (`OfficialConfig(bloom_path="full")`) | ~1 h |
Gate per stage, unchanged from WP-5: `bench report` with no missing cells; `median_ms(bs=16) < 16·median_ms(bs=1)`; ids identical across `mode`; a rerun byte-identical in quality (meetable only because L3 and L5 landed). Report `sm_mhz_load`, never `sm_mhz_idle`; treat any bs=1 comparison narrower than ~21 % as noise.

Wall time corrected 2026-09-16 from measurement: D1-a's first job took 4,390 s over nine sweeps at 365-535 s each plus ~65 s per new oracle; the unit is ~490 s per (algo, backend, sweep, params). The d128 matrix is 833 jobs (goodreads filter 165 / quality 7 / deep 216; arxiv 198 / 7 / 240); D1-a ~25-30 h, the campaign days. Value per GPU hour: the goodreads leg alone (~13 h) gives the primary dataset's full filtered picture; cheapest trims are the six non-headline sweeps at seed 0, then D1-d (456 of 833 jobs).

Grid narrowed 2026-09-16 (user), mid-D1-a: `torch` leaves the perf grid ("torch backends are less interesting now, priority for triton (fastest?) or official meta version"; ~45 % of wall time; parity lives in the library suite and B3 measured torch vs triton 1.0 / 0.0 on SilverTorch); modes narrow to `eager` plus `graph` on `triton` for headline sweeps (graph-only rejected: official cannot be captured, and eager is the papers' comparable mode). Consequence: `run.py` stamps a narrowed mode set `status: partial, partial_reasons: ["modes"]`, which `report.py` treats as NOT CITABLE; whether a deliberate narrowing should read differently is the user's decision.

## §15 D1-a goodreads leg, 2026-09-16
Branch `dev/d1a-campaign` off `development` `5fd05a6`. A100-SXM4-80GB, host `96ef99fba44c`, torch 2.10.0+cu128, triton 3.6.0, CUDA 12.4, Python 3.11; `code_version 0e6778056238de3c921cd2a14beec28b98a1d41c` constant; `dirty: false`; `git_branch: dev/d1a-campaign`. Artifacts `docs/artifacts/evaluation-harness-v2/d1a/`. Stopped twice by the orchestrator (after ~8 h: the stage was 25-30 h, cut at the dataset boundary; after ~10.75 h: the grid change).

Command (`bench campaign` has no `--seed`, so `stage_a.py` replays its loop with `--seed 0`):
`flock /workspace/gpu.lock -c 'uv run --no-sync python ../docs/plans/evaluation-harness-v2-artifacts/d1a/stage_a.py'` with `UV_PROJECT_ENVIRONMENT=/venvs/d1a RETRIEVE_DATA_ROOT=/workspace/data TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-d1a`; each child `python -m bench.cli run --dataset goodreads --dim 128 --suite filter --algo <a> --backend <b> --seed 0 --out ... --config-dir config --resume`. Gate `d1a/d1a_gate.py`.

Per job (first 8 jobs, 72 cells):
| algo | backend | cells | job wall | s/cell | timed s | other s | other % | recall@100 min..max |
|---|---|---|---|---|---|---|---|---|
| linr_v1_filter_mask | triton | 9 | 4390 | 450 | 108 | 342 | 76 % | 0.999423..0.999695 |
| linr_v1_filter_mask | torch | 9 | 4479 | 493 | 146 | 347 | 70 % | 0.999423..0.999695 |
| linr_v2 | triton | 9 | 4123 | 455 | 123 | 332 | 73 % | 0.999677..0.999729 |
| linr_v2 | torch | 9 | 6849 | 756 | 420 | 337 | 45 % | 0.999423..0.999695 |
| linr_v3 | triton | 9 | 3895 | 429 | 113 | 316 | 74 % | 0.694572..0.952268 |
| linr_v3 | torch | 9 | 6592 | 727 | 381 | 346 | 48 % | 0.694571..0.952175 |
| linr_v4 | triton | 9 | 3866 | 425 | 110 | 315 | 74 % | 0.980697..0.984766 |
| linr_v4 | torch | 9 | 4556 | 501 | 152 | 349 | 70 % | 0.980697..0.984766 |
| total | | 72 | 38,750 s | 530 | 170 | 330 | 62 % | |
Driver 2026-09-15T22:39:01Z to 2026-09-16T09:24:51Z; 98.4 % of wall inside cells (the process boundary ~1.6 %). `build_s` 0.0 (cached assets); `index_mib` 292 (v1/v2), 304 (v3), 195 (v4). Quality not validated.

Wall-time finding: 530 s per cell (H §2.8 said ~2 min, off 4.4x; stage table off 7-10x). The non-timed remainder is 315-349 s per cell, nearly independent of algo and backend: 9 `torch.compile` + cudagraph captures per cell (3 bs x 3 k, `_dynamo.reset()` per bs) plus quality and setup; timed windows only 125.7 s/cell for nine eager entries and 68.4 s for nine graph entries. An eager-only cell should cost ~150-200 s (~3x speedup). Graph is 1.84x faster than eager at the median over 648 paired comparisons (min 1.01x, max 6.50x). `torch` is slow in execution, not compilation (timed 1.35x v1/v4 to 3.4x v2/v3 of triton; 40 % of this run's wall).

`linr_v2` is not an exact algo: across nine sweeps torch vs triton jaccard 0.999005..0.999751, `score_max_abs_diff` 3.903e-03..7.818e-03 (one fp16 ULP: 2^-8 in [4, 8), 2^-7 in [8, 16)); `linr_v1_filter_mask` and `linr_v4` 1.0 / 0.0 on 9/9; `linr_v3` 0.999017..0.999797. On C4's cell 0.999751 / 3.904e-03 vs C4's 0.998743 / 9.766e-03 (post-L5 better). Recommendation, not acted on: move `linr_v2` out of `EXACT_ALGOS` or chase the one-ULP difference; torch is the only arm that yields this comparison.

Clocks: `env.sm_mhz_load` 1410.0 on all 72 cells; `sm_mhz_idle` 1155.0 (unused); per-entry min 1155, median 1410, max 1410; `clocks_drift` on 6 of 72. Instability (72-cell cut): 37 of 1296 perf entries unstable; bs=1 31, bs=8 5, bs=16 1; all eager; by algo v3 16, v2 9, v4 8, v1 4.

Harness bug recorded, not fixed (no change during a campaign): `bench/report.py:848-849` unconditionally appends "These records predate the D1 campaign; they come from C4's gate run and C5's one-cell check..." to every non-citable report; false here, prose only. And `partial` is stamped per process (`run.py:402-404`, line 490): an eager-only pass marks every record partial, including `official` whose graph would be `not_capturable` anyway.

Could not stop the driver (`kill -TERM` refused by the sandbox as "Interfere With Workloads").

§15.11 addendum (90 cells): `silvertorch triton` 18 cells (9 sweeps x `n_probe` 24 / 32), rc=0 in 8105 s, finished 2026-09-16T11:39:56Z. Gate at 90: 90 ok; 90/90; 540 batch-scaling comparisons, worst 12.406 (`linr_v3 torch all4 k=100 eager`, 1.6847 -> 20.9008 ms); 810/810 graph measured. `bench report` 20 artifacts. SilverTorch triton 447 s/cell (100 timed, 347 not); across nine jobs the remainder is 315-349 s vs timed 100-420 s. Corrections: graph unstable is 1 of 810 (not 0); instability 50 / 21 / 1 at bs 1 / 8 / 16 (72 total; 71 of 72 at bs in {1, 8}); SilverTorch bs=1 eager instability 33 % (18/54) vs 14 % (31/216) for LiNR; `clocks_drift` 11 of 90. A concurrent E1 `yfcc10m d192` CPU rehearsal from 10:17:28Z held zero `nvidia*` fds and never appeared in `nvidia-smi`; bs=1 eager unstable rate 5/18 before vs 13/36 after, no detectable contention.

## §15.12 final state (orchestrator, 2026-09-16)
The goodreads leg finished at 11/11 jobs, 126/126 cells, all `ok`. A watcher killed the driver 26 s after `silvertorch official rc=0`; the orphaned first arxiv job died mid-cell with no record.

Clause 6 (byte-identical rerun) PASSES on all 12 rerun records (five algos x three backends x both filter kinds), verified by the orchestrator from the raw JSONL in `d1a/rerun-records/`: `linr_v2` triton (was 2.029e-06 spread) identical; `linr_v3` triton (was 6.796e-05) identical, oracle block equal to 16 significant digits; the other 10 identical. L3 and L5 validated at campaign scale; §11.8's objection retired. `linr_v2 torch bloom/c2_format` differs only on parity fields (it became its own group's reference); the worker committed both raw and corrected diffs. Clause 5 (ids across mode) NOT RUN (executing when the session stopped; scratch only).

Full-leg numbers from `d1a/gate.txt`: worst batch-scaling ratio 14.429 (`silvertorch torch c3_year k=100 eager`); graph 972 measured + 162 `official:not_capturable` (first time the exemption fired); SilverTorch torch vs triton 1.000000 / 0.000e+00 on all 18 cells; official vs triton 0.999321 / 7.792e-03 (C4's arxiv official failure is dataset-specific).

Open: the 73 MB samples sidecar in git belongs on the Hub (unblocked); re-run `bench report` over all 126 cells (last run at 90).
