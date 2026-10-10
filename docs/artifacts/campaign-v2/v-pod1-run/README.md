# v-pod1-run: pod 1's legs at `campaign-v2.1`

Roadmap Phase V reruns and legs on pod 1, GPU 0, at the `campaign-v2.1` tag's code_version (each
driver refuses unless `bench env` equals both the tag's `retrieve/src/retrieve` tree hash and
*evaluation/campaign.yaml*'s default). Oracle blobs of the leg's datasets built at another
code_version move to `<gt_dir>/before-<code_version[:8]>/` first, so `bench oracle` rebuilds them at the tag. Results tree
`/scratch/campaign-v21/results`; uploads `campaign-v2.1/<dataset>-<suite>` ([hub-index](../../hub-index.md)).
State per leg in [validation](../../../validation.md). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`common.sh`](common.sh) | sourced: env, code_version check, 1 Hz clock trace, `step` (a pinned `bench` subcommand), `stream` (one `bench run --resume --interleave` with its log), `old_oracles` |
| [`move_old_oracles.py`](move_old_oracles.py) | moves blobs whose `code_version` is not the tag's into `before-<code_version[:8]>/` (never deletes; v2.1 legs used `pre-v21/`) |
| [`h2h.sh`](h2h.sh) | H-PROFILE + H2H-FINAL: `h2h` on goodreads + arXiv, whole, `--interleave --profile`, one pass |
| [`h-ksum.sh`](h-ksum.sh) | H-KSUM's profile-only `h2h` pass (goodreads + arXiv, `--interleave --profile --skip-quality`, every kernel summed) into `/scratch/h-ksum/results`; artifact `artifacts/h-ksum-h2h`, not a leg |
| [`v-v3bits.sh`](v-v3bits.sh), [`v3bits_quality.py`](v3bits_quality.py) | V-V3BITS: `v3bits` on goodreads-synth + goodreads, a `--skip-perf` quality pass and its recall table, then the whole suite timed |
| [`v-codesign.sh`](v-codesign.sh) | V-CODESIGN, goodreads half (pod c runs arXiv): `codesign` on goodreads, whole |
| [`gr-reruns.sh`](gr-reruns.sh) | V-RERUN-V21: goodreads `filter` and goodreads-synth `synth` SilverTorch arms (triton + official, torch), V1 + V2 Triton (paired, one interleave group), V3 Triton |
| [`v-gr-deep.sh`](v-gr-deep.sh) | V-GR-DEEP: goodreads `deep` |
| [`v-yfcc-deep.sh`](v-yfcc-deep.sh), [`yfcc_deep_config.py`](yfcc_deep_config.py) | V-YFCC deep, claims-first: SilverTorch triton n_lists 4096 × n_probe {24, 256, 1024} (a cut of `deep`) + V2 bs 16 on `filter`; 4 cells |
| [`boundary-gates.sh`](boundary-gates.sh), [`gate_smoke.py`](gate_smoke.py) | the V-V3BITS boundary: GPU library suite on the main checkout (H-INDCACHE's gate), then the campaign-v2.3 candidate (dev/ctl-v23, library 1258a63e): its GPU library suite on a fresh inductor cache and goodreads `filter` smoke cells per changed path, eager + graph, capture checked |
| [`v-router.sh`](v-router.sh) | V-ROUTER: the `router` suite on goodreads at seed 0, k 100, clause (18 cells; claims-first) |
| [`c3-pair.sh`](c3-pair.sh) | C3's bs-1 pair: V1 Triton vs `torch.compile` V1 on goodreads-synth / arxiv-synth `p01` (records with quality; interleaved timing via `../v-prof3/prof3.py`) |
| [`v-gr-deep-triton.sh`](v-gr-deep-triton.sh) | V-GR-DEEP's trimmed end: the SilverTorch-triton arm only, at campaign-v2.3 |
| [`v-yfcc-synth.sh`](v-yfcc-synth.sh) | V-YFCC synth (D): the SYNTH-TRIM `synth` suite on yfcc10m-synth, whole (25 cells) |
| [`gr-synth-mid.sh`](gr-synth-mid.sh) | goodreads-synth's middle rates p 0.05 / 0.2 / 0.5 (24 cells) |
| [`gr-synth-mid-q.sh`](gr-synth-mid-q.sh) | the same 24 cells quality only (`--skip-perf`) at campaign-v2.6: nothing timed on clause kernels over 10-clause synth tables before CLAUSE-SKIP; done, Hub `campaign-v2.6/goodreads-synth-synth` |
| [`ax-synth-v1v2.sh`](ax-synth-v1v2.sh) | arXiv-synth V1 + V2 interleaved, the synth suite's 7 rates (14 cells): C1's 3 M crossover on pod 1; 8 cells ran at v2.6 (pre-CLAUSE-SKIP), the rest after v2.7 |
| [`h2h-v28.sh`](h2h-v28.sh) | T3 / C7 redo at campaign-v2.8 (staging f24f077 harness, library 78cfbc72; own venv from `scripts/build_official_o3.sh`, Meta's extension at `-O3`, OF-11): the `h2h` suite on goodreads `c0_genre` + arXiv `c0_maincat`, none + bloom, seed 0, `--interleave --profile`, from the tag's own worktree (`REPO`, which `common.sh` puts on `PYTHONPATH`); Hub `campaign-v2.8/{goodreads,arxiv}-h2h` |
| [`gr-deep-v29.sh`](gr-deep-v29.sh) | goodreads `deep` re-time at campaign-v2.9 (ST-WIDE): SilverTorch triton clause, n_lists {1024, 4096} x n_probe {8 ... 128}, bs {1, 16}, seed 0, k 100, the three kept sweeps (c0_genre first); C6 at 0.8 M and the QPS@0.95 bands; Hub `campaign-v2.9/goodreads-deep` |
| [`codesign-v29.sh`](codesign-v29.sh) | C5 ours below 30 M at campaign-v2.9: `codesign` triton only, `bloom_path` partial vs full interleaved, goodreads `c0_genre` + arXiv `c0_maincat`, n_probe {8, 32, 128}, bs {1, 16}, k 100, seed 0; Hub `campaign-v2.9/{goodreads,arxiv}-codesign-ours` |
| [`codesign-v29-ax.sh`](codesign-v29-ax.sh) | the same leg's arXiv `c3_nversions` + `all4` (appended to `campaign-v2.9/arxiv-codesign-ours`; upload with `HUB=campaign-v2.9/arxiv-codesign-ours`) |
| [`yfcc-deep-v29.sh`](yfcc-deep-v29.sh) | V-YFCC deep's SilverTorch cells (n_lists 4096, n_probe {24, 256, 1024}, bs {1, 16}) re-timed at campaign-v2.9 (ST-WIDE) + the V2 bs-16 reference, timed only; Hub `campaign-v2.9/yfcc10m-deep` |
| [`yfcc-synth-v29.sh`](yfcc-synth-v29.sh), [`yfcc_synth_config.py`](yfcc_synth_config.py), [`yfcc-synth-v29-hi.sh`](yfcc-synth-v29-hi.sh) | AFTER-QUEUE 2 on YFCC at campaign-v2.9: SilverTorch at nine rates and V1 + V2 at 0.001 / 0.003 / 0.03 / 0.05, then V1 + V2 at 0.01 / 0.1 / 0.2 / 0.5 / 1.0 (the full nine-rate curve on pod 1); Hub `campaign-v2.9/yfcc10m-synth-synth` |
| [`synth_cut_config.py`](synth_cut_config.py) | scratch `synth` config for one dataset: its sweep list at k 100 and optional seeds 0-2 at given rates |
| [`ax-synth-v29.sh`](ax-synth-v29.sh) | arXiv-synth 3 M V1 + V2 at nine rates at campaign-v2.9 (pod 1's 3 M crossover); Hub `campaign-v2.9/arxiv-synth-v1v2-pod1` |
| [`yfcc-seeds-v29.sh`](yfcc-seeds-v29.sh) | yfcc10m-synth V1 + V2 seeds 1-2 at p 0.05 / 0.1 / 0.2 (a CI on the 10 M crossover); appended to `campaign-v2.9/yfcc10m-synth-synth` |
| [`filter-v29.sh`](filter-v29.sh) | night-queue item 4: the `filter` grid's V1 / V2 / SilverTorch triton + official bloom cells (52), then linr_v3 triton (11), on goodreads / arXiv / YFCC at campaign-v2.9, seed 0, `/venvs/v29-o3`; Hub `campaign-v2.9/{goodreads,arxiv,yfcc10m}-filter` |
| [`tune_config.py`](tune_config.py), [`tune-v29.sh`](tune-v29.sh) | night-queue item 8: YFCC 10 M ANN tuning at campaign-v2.9; phase 1 quality only (`tune-q`: SilverTorch triton n_lists {2048 ... 16384} x n_probe {8 ... 4096}, real `tags_and` + synth p 0.01 / 0.1 / 1, clause + synth bloom, 273 cells), phase 2 the timed frontier (`tune`); Hub `campaign-v2.9/yfcc10m-tune` |
| [`tune_bench_config.py`](tune_bench_config.py), [`tune-bench-v210.sh`](tune-bench-v210.sh) | night-queue item 8, quality phase on goodreads 0.8 M + arXiv 3 M at campaign-v2.10 (1,372 cells: n_lists {256 ... 16384} x n_probe ≤ n_lists / 4, real + uniform synth + arXiv correlated synth, clause + bloom); Hub `campaign-v2.10/{goodreads,arxiv}-tune` |
| [`v3bits-gr-v210.sh`](v3bits-gr-v210.sh) | V-V3BITS remainder at campaign-v2.10: the `v3bits` suite's goodreads half (16 cells, seed 0); Hub `campaign-v2.10/goodreads-v3bits` |
| [`../../campaign-v2.10/ceiling-int8/ceiling-v210.sh`](../../campaign-v2.10/ceiling-int8/ceiling-v210.sh) | the C6 recall-ceiling check on goodreads + arXiv at campaign-v2.10 (12 sweeps, quality only); Hub `artifacts/ceiling-int8-gr-ax` |
| [`c3_real_config.py`](c3_real_config.py), [`c3-real-v210.sh`](c3-real-v210.sh) | C3 coverage on real filters at campaign-v2.10: V1 Triton vs torch.compile max-autotune on goodreads / arXiv / YFCC kept sweeps and PubMed 10 M d768 (20 cells); Hub `campaign-v2.10/<dataset>-c3-real` |
| [`yfcc-seeds-filter-v210.sh`](yfcc-seeds-filter-v210.sh) | V-SEEDS YFCC half at campaign-v2.10: yfcc10m `filter` V1 / V2 / V3 triton, seeds 0-2 (9 cells); Hub `campaign-v2.10/yfcc10m-filter-seeds` |
| [`yfcc-int8.sh`](yfcc-int8.sh) | the YFCC int8-precision check (quality only; [`../../campaign-v2.5/yfcc-int8/int8_check.py`](../../campaign-v2.5/yfcc-int8/int8_check.py)) |
| [`chain-legs.sh`](chain-legs.sh) | the queue as arguments (`v-gr-deep`, `v-router`, `v-yfcc-deep`, `v-yfcc-synth`, `gr-synth-mid`, `yfcc-int8`, `ax-synth-v1v2`, `gr-synth-mid-q`), one lock hold, stop file between legs |
| [`chain.sh`](chain.sh) | the queue after V-V3BITS (at the tag common.sh defaults to, now campaign-v2.7): V-GR-DEEP, V-YFCC deep, the synth chunks, one lock hold, stop file between legs |
| [`v-seeds-yfcc.sh`](v-seeds-yfcc.sh) | V-SEEDS, YFCC half: yfcc10m `filter`, every arm |
| [`h2h-diag.sh`](h2h-diag.sh) | H2H protocol diagnostic (controller addendum): goodreads `c0_genre` bloom bs 16 k 100 seed 0, the three h2h arms (a) interleaved + `--profile`, (b) interleaved, (c) one process per arm, orders abc / cba / bac; scratch trees, not a leg |
| [`h2h_diag_config.py`](h2h_diag_config.py), [`h2h_diag_summary.py`](h2h_diag_summary.py) | its scratch config dirs (the real `h2h` suite cut to that cell) and its per-arm table |
| [`stage-upload.sh`](stage-upload.sh) | copies one `(suite, dataset)` slice and the leg's logs and runs `bench upload --verify` |

Launch: `setsid nohup flock -n /scratch/gpu0.lock bash <driver> > /scratch/v21/<leg>/driver.log 2>&1 &`;
after a crash rerun the same command (every step resumes).

## Expansion against a fresh tree (tag `campaign-v2.1` + IVF-TUNE part A, dev/ivf-tune `153ee26`)

Order (controller, exploration phase 2026-10-09): goodreads `codesign`, the H2H diagnostic + H-PROFILE
GPU checks, h2h with the profiler fix, V-GR-DEEP, V-YFCC, V-SEEDS' YFCC half. V-RERUN-V21
(`gr-reruns.sh`) is dropped to the controller's redo ledger and does not run now.

| leg | dataset / suite (arms) | cells |
|---|---|---|
| V-CODESIGN | goodreads / codesign (arXiv on pod c) | 54 |
| h2h | goodreads / h2h; arxiv / h2h | 30; 30 |
| V-RERUN-V21 | goodreads / filter: silvertorch (all backends; n_lists 4096, triton / official n_probe {24, 64}); V1 + V2 triton; V3 triton | 54; 24; 12 |
| V-RERUN-V21 | goodreads-synth / synth: silvertorch; V1 + V2 triton; V3 triton | 210; 84; 84 |
| V-GR-DEEP | goodreads / deep | 210 |
| V-YFCC | yfcc10m-synth / synth; yfcc10m / deep | 285; 45 |
| V-SEEDS | yfcc10m / filter (before part B's n95 slot) | 18 |

## Outcomes

- **V-CODESIGN goodreads half** (2026-10-08, staging `69dd3d6`, `f01255f1`): 54 / 54 ok (44 `unstable`),
  0.79 GPU-h; full / partial 0.81-0.86, recall identical; Hub `campaign-v2.1/goodreads-codesign`
  ([validation](../../../validation.md)). Summary via [`../v-codesign/codesign_summary.py`](../v-codesign/codesign_summary.py).
- **H2H protocol diagnostic** (2026-10-09, `f01255f1`, 0.35 GPU-h): (a) = (b) = (c) within noise on every arm
  (official ≈ 1.44 ms, Triton eager ≈ 0.68, graph ≈ 0.19; official / Triton 2.13); Hub `artifacts/h2h-diag-v21`.
- **H2H-FINAL at v2.1** (2026-10-09, staging `cb1dbc3`, `f01255f1`, 1.33 GPU-h): 60 / 60 ok, kernel lists complete,
  eager = graph ids 80 / 80; Hub `campaign-v2.1/{goodreads,arxiv}-h2h` ([validation](../../../validation.md)).
  Summary via [`../h2h-final/h2h_summary.py`](../h2h-final/h2h_summary.py).
- **H-KSUM profile-only pass** (2026-10-09, staging `0627961`, `f01255f1`, 0.91 GPU-h): 60 records with the full kernel sum;
  official / Triton device time 1.5-2.2 everywhere; Hub `artifacts/h-ksum-h2h` ([validation](../../../validation.md)).
- **V-V3BITS, goodreads-synth** (2026-10-09, staging `0693399`, `0d23c615`, 4.06 GPU-h): 168 / 168 ok; `k_bits` 64 loses 10-30 points
  of recall@100 at p ≥ 0.01 for ≤ 3 % latency; goodreads (48 cells) deferred; Hub `campaign-v2.2/goodreads-synth-v3bits`.
  The first run's `--skip-perf` pass was stopped at 128 / 168 (duplicate work); `v-v3bits.sh` keeps it behind `QUALITY=1`.
