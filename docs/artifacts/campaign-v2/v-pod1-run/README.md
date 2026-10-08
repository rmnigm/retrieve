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
| [`h2h.sh`](h2h.sh) | H-PROFILE + H2H-FINAL: `h2h` on goodreads + arXiv, whole, `--interleave --profile` |
| [`v-codesign.sh`](v-codesign.sh) | V-CODESIGN: `codesign` on arXiv + goodreads, whole |
| [`gr-reruns.sh`](gr-reruns.sh) | V-RERUN-V21: goodreads `filter` and goodreads-synth `synth` SilverTorch arms (triton + official, torch), V1 + V2 Triton (paired, one interleave group), V3 Triton |
| [`v-gr-deep.sh`](v-gr-deep.sh) | V-GR-DEEP: goodreads `deep` |
| [`v-yfcc.sh`](v-yfcc.sh) | V-YFCC: yfcc10m-synth `synth`, then yfcc10m `deep` |
| [`v-seeds-yfcc.sh`](v-seeds-yfcc.sh) | V-SEEDS, YFCC half: yfcc10m `filter`, every arm |
| [`stage-upload.sh`](stage-upload.sh) | copies one `(suite, dataset)` slice and the leg's logs and runs `bench upload --verify` |

Launch: `setsid nohup flock -n /scratch/gpu0.lock bash <driver> > /scratch/v21/<leg>/driver.log 2>&1 &`;
after a crash rerun the same command (every step resumes).

## Expansion against a fresh tree (tag `campaign-v2.1`, before IVF-TUNE)

Order (controller): goodreads `bloomwidth-timed` (v-short-legs' driver), h2h after dev/h-profile
merges, V-CODESIGN, then the IVF-dependent legs after dev/ivf-tune merges (recounted then).

| leg | dataset / suite (arms) | cells |
|---|---|---|
| h2h | goodreads / h2h; arxiv / h2h | 30; 30 |
| V-CODESIGN | arxiv / codesign; goodreads / codesign | 54; 54 |
| V-RERUN-V21 | goodreads / filter: silvertorch (all backends); V1 + V2 triton; V3 triton | 39; 24; 12 |
| V-RERUN-V21 | goodreads-synth / synth: silvertorch; V1 + V2 triton; V3 triton | 210; 84; 84 |
| V-GR-DEEP | goodreads / deep | 210 |
| V-YFCC | yfcc10m-synth / synth; yfcc10m / deep | 285; 45 |
| V-SEEDS | yfcc10m / filter | 18 |
