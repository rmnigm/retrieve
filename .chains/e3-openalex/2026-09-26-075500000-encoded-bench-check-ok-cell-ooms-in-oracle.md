---
chain: "e3-openalex"
branch: "main"
parent: "2026-09-26-021500000-streamed-and-prepped-at-15m-awaiting-gpu.md"
nextStep: "Orchestrator/user decides how the filter cell fits: (A) shrink the catalog to ~10M (prep --keep-items 10000000 on CPU; the vectors already exist in the 15M shards but need a gather step or a 2.9 h re-encode; queries re-encode in seconds), or (B) a bench/oracle.py change that scores against item_embs.t() without .contiguous() (saves 43 GB, but the oracle is gated, so it needs torch.equal against the current blobs). Then re-add openalex to suites.yaml's filter suite and run the one linr_v1_filter_mask/triton field_era cell."
created: "2026-09-26T07:55:00Z"
---

# E3 OpenAlex: 15M encoded, bench check clean; the filter cell OOMs in the harness oracle

## Done (dev/e3-openalex, commit 9f58984 on top of the 0036ef1 staging merge; not pushed)

- **Staging merge check:** 77b376e refactored openalex.py (shared helpers moved to `common.py`,
  type hints). I checked it is behaviour-preserving: the tests pass, and `select_pmids`
  reproduces the on-disk 15M catalog.
- **Encode chain:**
  - `encode_text`: 15,663 s = 4.35 A100-h, 958 docs/s average. The first shards ran at
    1.00–1.02k docs/s.
  - Output: 15 × 1M fp16 shards, 23 GB, plus `shard_index.json` and the meta sidecars.
  - `encode_queries`: 10,000 × 768.
  - `attrs`: `[15M, 5, 4]`, 2.4 GB. Vocabularies: 26 fields, 252 subfields, 5,000 sources.
  - Coverage: field 99.85 %, source 51.6 %.
  - The target passes field and era 100 %, subfield 75 %.
- **`bench check --dataset openalex`: `openalex d768: ok`.** Run from the worktree with
  `evaluation/data -> /data`.
- **Filter cell** (`linr_v1_filter_mask`/triton, clause `field_era`, eager, `--skip-perf`,
  `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/e3`): it did not produce a record. Two OOMs:
  1. `layout.load_text_items` held three whole-matrix copies: fp16, `.float()` and
     `F.normalize`, 107 GB in total.
     - **Fixed:** `load_sharded(..., normalize=True)` normalises shard by shard into one fp32
       buffer, and `load_text_items` uses it for sharded layouts.
     - `torch.equal` to the old path on 3M real rows on the GPU, plus a CPU test in
       `test_layout.py`.
     - The harness suite: 237 passed, 1 skipped.
  2. `bench/oracle.py:156` does `item_embs_t = item_embs.t().contiguous()`, a second
     42.9 GB fp32 copy. Items (45.2 GB allocated) plus that copy exceed the 79 GB GPU.
     - **Not touched:** `bench/` is outside my boundary, and the oracle is gated.
- **The corrected harness ceiling at 768-d is ~11–12M items**, from 2 × N × 768 × 4 B plus
  attrs and workspace. The "~15M" behind the 15M decision counted only the resident fp32
  copy. PubMed's 10M needs 57 GB and fits.
- `openalex` was added to `suites.yaml`'s filter suite for the run, then **reverted**. While
  listed, a D1 `bench campaign --suite filter` would launch a child that OOMs. The
  `config/openalex.yaml` header says it is in no suite, and why.
- Docs: `datasets.md` § openalex has the status, the budget and a "What fits the harness"
  paragraph. `validation.md` and `roadmap.md` are untouched.
- Artifacts: `run.sh`, `run15m-encode.log`, `run15m-bench-check.log`,
  `cell-linr_v1-oom.log` (loader OOM), `cell-linr_v1.log` (oracle OOM).

## State on disk
- `/data/openalex` holds the full 15M layout.
- `/data/_raw/openalex/staging` (16 GB) is still kept: a 10M re-prep needs it.
- Disk: about 50 % of the overlay used.

## Options for the cell (decision needed)
- **(A) ~10M catalog.** `prep --keep-items 10000000` on CPU takes about 7 min.
  - The 10M smallest hashes are a subset of the 15M, so no encode is needed if a small
    gather-by-work-id reshard is written.
  - Otherwise, re-encode for about 2.9 A100-h.
  - Queries are re-drawn from the larger pool and re-encoded in seconds.
- **(B) Oracle without the copy.** `q @ item_embs.t()` as a view. The GEMM layout may change
  the scores' summation order, so it needs a bit-exactness check against the existing oracle
  blobs and golden cells. Owner: whoever holds `bench/`.
- **(C) fp16 items in the harness.** A bigger change.
