# v-pod1-run: pod 1's legs at `campaign-v2.1`

Roadmap Phase V reruns and legs on pod 1, GPU 0, at the `campaign-v2.1` tag's code_version (each
driver refuses unless `bench env` equals the tag's `retrieve/src/retrieve` tree hash). Results tree
`/scratch/campaign-v21/results`; uploads `campaign-v2.1/<dataset>-<suite>` ([hub-index](../../hub-index.md)).
State per leg in [validation](../../../validation.md). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`common.sh`](common.sh) | sourced: env, code_version check, 1 Hz clock trace, `step` (a pinned `bench` subcommand) and `stream` (one `bench run --resume --interleave` with its log) |
| [`v-codesign.sh`](v-codesign.sh) | leg 1: `codesign` on arXiv + goodreads, whole |
| [`gr-reruns.sh`](gr-reruns.sh) | leg 2: goodreads `filter` and goodreads-synth `synth` SilverTorch arms (triton + official, torch), V1 + V2 Triton (paired, one interleave group), V3 Triton; goodreads `bloomwidth-timed` |
| [`v-gr-deep.sh`](v-gr-deep.sh) | leg 3: goodreads `deep` |
| [`v-yfcc.sh`](v-yfcc.sh) | leg 4: yfcc10m-synth `synth`, then yfcc10m `deep` |
| [`v-seeds-yfcc.sh`](v-seeds-yfcc.sh) | leg 5: yfcc10m `filter`, every arm |
| [`stage-upload.sh`](stage-upload.sh) | copies one `(suite, dataset)` slice and the leg's logs and runs `bench upload --verify` |

Launch: `setsid nohup flock -n /scratch/gpu0.lock bash <driver> > /scratch/v21/<leg>/driver.log 2>&1 &`;
after a crash rerun the same command (every step resumes).

## Expansion against a fresh tree (staging `2957a0f`, before the tag and IVF-TUNE)

| leg | dataset / suite (arms) | cells |
|---|---|---|
| 1 | arxiv / codesign; goodreads / codesign | 54; 54 |
| 2 | goodreads / filter: silvertorch (all backends); V1 + V2 triton; V3 triton | 39; 24; 12 |
| 2 | goodreads-synth / synth: silvertorch; V1 + V2 triton; V3 triton | 210; 84; 84 |
| 2 | goodreads / bloomwidth-timed | 21 |
| 3 | goodreads / deep | 210 |
| 4 | yfcc10m-synth / synth; yfcc10m / deep | 285; 45 |
| 5 | yfcc10m / filter | 18 |
