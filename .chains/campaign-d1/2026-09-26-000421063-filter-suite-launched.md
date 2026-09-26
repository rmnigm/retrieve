---
chain: "campaign-d1"
branch: "main"
nextStep: "Tail evaluation/results/_logs/campaign.log; when `bench campaign --suite filter` (pid 8063) exits, run `bench report results`, update validation.md Campaign section + Datasets table, commit ONLY the *.jsonl records (not *.samples.jsonl) + validation.md on staging, push, then launch `--suite deep --resume`."
created: "2026-09-26T00:04:21Z"
---

# D1 filter suite launched on staging

## Primary request
Worker for roadmap D1: run `filter` then `deep` suites (goodreads < arxiv < yfcc10m) on branch `staging`, checkpoint commits of data + docs/validation.md (Campaign section + Datasets table only), push to origin/staging. User explicitly confirmed "full grid" (the roadmap "Needs the user: Run the campaign?" item) on 2026-09-26.

## Work completed
- GPU confirmed idle (A100 80GB, 0 MiB used).
- Datasets fetched from the Hub into /data (`eval-data fetch`): goodreads-work-id d128 + checkpoints (6.5G), arxiv-papers d128 (3.0G), yfcc10m d192 (9.2G). Log: /scratch/tmp/d1-fetch.log.
- `bench check`: goodreads d128 ok, arxiv d128 ok, yfcc10m d192 only flags missing optional `content_d192/{text,query}_emb.meta.json` (prefix sidecars; CLIP has no prefix) — treated as non-blocking.
- Launched: `nohup uv run --no-sync --directory evaluation bench campaign --suite filter --resume` with TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/d1 HF_HOME=/scratch/hf, pid 8063, started 2026-09-26T00:03:55Z. stdout: /scratch/campaigns/d1-filter.out. goodreads/filter = 75 jobs, 105 cells.

## Decisions and rationale
- `evaluation/data -> /data` symlink (gitignored path, no code change): bench config resolves `data_dir: data/<ds>` relative to evaluation/ and ignores RETRIEVE_DATA_ROOT (the roadmap Q2 "two RETRIEVE_DATA_ROOT defaults" defect).
- Parity spills left at evaluation/results/_parity (on /workspace; ~680 MB, deleted per group). Not symlinked to /scratch: `shutil.rmtree(..., ignore_errors=True)` on a symlink silently fails and stale spills would become the next group's reference.
- Commit only `evaluation/results/<suite>/*.jsonl`, NOT `*.samples.jsonl`: goodreads-d128.samples.jsonl is already 73 MB tracked in git and will exceed GitHub's 100 MB limit; storage rule says samples go to the Hub via `bench upload` (not authorized in this brief). Samples stay on /workspace (persistent).

## Unresolved — orchestrator decisions
1. **S9 ablation not encoded**: evaluation/config/suites.yaml has no `bloom_path="full"` / OfficialConfig combo in either suite. Needs a suites.yaml change (config, outside a pure run step) — orchestrator's call.
2. **Existing 126 goodreads seed-0 records carry `env.git_branch: dev/d1a-campaign`** (code_version 0e67780… matches HEAD, so `--resume` skips them). report.py vetoes citability for non-staging/development/main branches → the goodreads seed-0 leg stays uncitable unless re-run with `--force` on staging (~126 × 530 s ≈ 18 h), which would also serve as the "rerun byte-identical" gate.
3. Samples sidecars → Hub via `bench upload` needs authorization (shared-state action).
4. Expected: yfcc10m linr_v1_filter_mask / linr_v2 groups fail with QualityGateError (fp16, recall_oracle@1000 ≈ 0.964). Not to be fixed.
