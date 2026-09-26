---
chain: "campaign-d1"
branch: "main"
parent: "2026-09-26-000421063-filter-suite-launched.md"
nextStep: "Nothing until the orchestrator relaunches D1 after the code steps (Q3, G-a, G-d, G-e, Q4). At relaunch: resolve the four decision points below first, then `bench campaign --suite filter --resume` from staging with TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1 HF_HOME=/scratch/hf."
created: "2026-09-26T00:07:40Z"
---

# D1 campaign paused by the orchestrator

## Why
The orchestrator is re-sequencing the roadmap: library and harness code changes (Q3, G-a, G-d, G-e, Q4) land before the full campaign, because D1's only real tie to them is the code_version invalidation. Every change under retrieve/src/retrieve bumps code_version, so records from this run would need rerunning anyway.

## State at pause (2026-09-26T00:07Z)
- Campaign (pid 8063, pgid 8063) killed with SIGTERM to the process group about 3 min in, during the first group (goodreads d128 linr_v1_filter_mask triton). No bench or inductor compile-worker processes remain. GPU at 0 MiB, 0 %.
- Monitor task blzwqoxsi stopped.
- The run appended 1 record (`ok`, goodreads linr_v1_filter_mask c0_genre seed 1, git_branch staging) + 18 samples lines. **Not committed, not pushed.** Stashed so the shared staging tree is clean: `stash@{0}` "D1 aborted filter run 2026-09-26: 1 goodreads seed-1 record + samples, not to commit". Disposable: drop it once the code steps bump code_version.
- `evaluation/results/_parity/` deleted (2 spills from the aborted group; a stale spill would have become the next run's parity reference).
- Nothing in docs/validation.md changed; no commits made by this worker.

## Still in place (reusable at relaunch)
- Datasets staged in /data (ephemeral container disk): goodreads-work-id d128 + checkpoints, arxiv-papers d128, yfcc10m d192. `bench check`: goodreads, arxiv ok; yfcc10m reports only the missing optional `content_d192/{text,query}_emb.meta.json`.
- Symlink `evaluation/data -> /data` (gitignored), needed because bench config resolves `data/<ds>` relative to evaluation/ and ignores RETRIEVE_DATA_ROOT (roadmap Q2 defect; Q2 may make the symlink unnecessary).
- /scratch/inductor/d1 holds the inductor cache compiled under the old code; delete it before relaunch (stale-kernel risk, storage.md D-7).

## Decision points left for when the campaign launches
1. **S9 ablation not encoded.** evaluation/config/suites.yaml has no `OfficialConfig(bloom_path="full")` combo in either suite. Adding one is a config change for the orchestrator.
2. **The goodreads seed-0 records are uncitable by branch.** The 126 committed records carry `env.git_branch: dev/d1a-campaign`, and report.py vetoes non-staging/development/main branches. Moot if the code steps bump code_version (then everything reruns on staging); otherwise it takes a `--force` rerun of about 18 h, which doubles as the "rerun byte-identical" gate.
3. **Samples storage.** goodreads-d128.samples.jsonl is 73 MB and tracked in git; D1 will push it past GitHub's 100 MB limit. Plan: commit only the `*.jsonl` records and publish the samples with `bench upload` (needs authorization), or untrack them.
4. **yfcc10m exact-algorithm failures are expected.** linr_v1_filter_mask and linr_v2 raise QualityGateError (fp16 scoring, recall_oracle@1000 ≈ 0.964). If a code step switches exact algorithms to fp32 scoring before the relaunch, this changes.
