# v-pod1-run: pod 1's legs at `campaign-v2.1`

Roadmap Phase V reruns and legs on pod 1, GPU 0, at the `campaign-v2.1` tag's code_version (each
driver refuses unless `bench env` equals both the tag's `retrieve/src/retrieve` tree hash and
*evaluation/campaign.yaml*'s default). Oracle blobs of the leg's datasets built at another
code_version move to `<gt_dir>/pre-v21/` first, so `bench oracle` rebuilds them at the tag. Results tree
`/scratch/campaign-v21/results`; uploads `campaign-v2.1/<dataset>-<suite>` ([hub-index](../../hub-index.md)).
State per leg in [validation](../../../validation.md). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`common.sh`](common.sh) | sourced: env, code_version check, 1 Hz clock trace, `step` (a pinned `bench` subcommand), `stream` (one `bench run --resume --interleave` with its log), `old_oracles` |
| [`move_old_oracles.py`](move_old_oracles.py) | moves blobs whose `code_version` is not the tag's into `pre-v21/` (never deletes) |
| [`h2h.sh`](h2h.sh) | H-PROFILE + H2H-FINAL: `h2h` on goodreads + arXiv, whole, `--interleave --profile`, one pass |
| [`h-ksum.sh`](h-ksum.sh) | H-KSUM's profile-only `h2h` pass (goodreads + arXiv, `--interleave --profile --skip-quality`, every kernel summed) into `/scratch/h-ksum/results`; artifact `artifacts/h-ksum-h2h`, not a leg |
| [`v-codesign.sh`](v-codesign.sh) | V-CODESIGN, goodreads half (pod c runs arXiv): `codesign` on goodreads, whole |
| [`gr-reruns.sh`](gr-reruns.sh) | V-RERUN-V21: goodreads `filter` and goodreads-synth `synth` SilverTorch arms (triton + official, torch), V1 + V2 Triton (paired, one interleave group), V3 Triton |
| [`v-gr-deep.sh`](v-gr-deep.sh) | V-GR-DEEP: goodreads `deep` |
| [`v-yfcc.sh`](v-yfcc.sh) | V-YFCC: yfcc10m-synth `synth`, then yfcc10m `deep` |
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
