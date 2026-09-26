---
chain: "e3-openalex"
branch: "main"
nextStep: "Orchestrator: merge dev/e3-openalex (9f64057) into staging, apply the hub.py lines in §5b by hand, fix docs/system/evaluation.md's config count, update docs/validation.md's datasets row and the E3 roadmap entry; then have the user decide the catalog size (50M does not fit the harness at 768-d fp32, §6.1) before scheduling the ~15.5 A100-hour encode."
created: "2026-09-26T01:15:34Z"
---

# E3 OpenAlex: streaming ETL written and dry-run on real snapshot rows

## 1. Primary request and intent
Worker for roadmap E3, OpenAlex-fallback branch (no Semantic Scholar key, not waiting for one).
Scope: write the ingestion plan, implement `evaluation/eval_datasets/etl/openalex.py`, add
`evaluation/config/openalex.yaml`, and validate against a small real prefix. Explicitly **not**:
the full 707 GB stream, the 50M encode, `hub.py`, `retrieve/`, `evaluation/bench/`,
`docs/system/{evaluation,testing}.md`, `docs/validation.md`, `docs/roadmap.md`. Commit locally,
do not push or merge.

## 2. Verified vs inferred

**Verified live, 2026-09-26:**
- The snapshot layout changed. `s3://openalex/README.txt` names `data/jsonl/` and `data/parquet/`
  as two complete copies, partitioned by `updated_date`, each with a `manifest.json` that is
  written last. The old `data/works/manifest` is now an S3 delete marker; `legacy-data/` is
  frozen. Anonymous HTTPS and pyarrow `S3FileSystem(anonymous=True, region="us-east-1")` both
  work.
- Release 2026-09-23 has **476,196,327 works in 2,040 files**: 659.0 GB of JSONL.gz and
  **707.1 GB of parquet**. The largest file is about 1.05 GB / 400k records.
- The parquet schema (full dump in this session): `abstract_inverted_index` is a **JSON string**
  (an object in JSONL and the API). A missing abstract is `""` or null. IDs are URLs
  (`https://openalex.org/W…`, `…/fields/25`, `…/subfields/2505`, `…/S…`).
- Nested-leaf projection works (`primary_topic.field.id` etc.). The projected columns total
  **297.1 GB (42.0 %)**, summed from the footers of all 2,040 files.
- Filter rates (`plan`, 64 seeded row groups, 5.5M rows): after year 83 %, English 62 %,
  type 23 %, flags 18 %, abstract 11.46 %, giving an estimated **54.6M eligible** works.
  Row groups cluster by type; an earlier 32-group sample with fewer types gave 13.1 %, so take
  it as ±10 %. Among English works, `dataset` is the largest type (CCDC boilerplate abstracts).
- About 1 abstract index in 7.1M is cut off near 32.7k characters (e.g. W4214716959): the
  parquet copy caps string length. It crashed the first probe; it is now counted as
  `abstract_truncated` and dropped.
- The abstract rebuilt from the parquet string equals the one rebuilt from the API object (3/3
  random works checked).
- Throughput of the real stream path (spawned processes doing the projected read plus
  `stage_table`): **176 MB/s with 32 workers, 229.5 MB/s with 64**
  (`docs/artifacts/e3-openalex/stream_probe-*.json`). That makes the full stream about 25–30 min
  plus a tail.
- Encode on this A100, nomic-v1.5 at 768-d, 512 tokens, mean 228 tokens, 20k real texts:
  **898 docs/s** with bf16 weights at batch 256; 755 docs/s with bf16 autocast; forward alone
  954 docs/s. `flash_attn` is not installed.

**Inferred / not verified:**
- In-catalog citation density at 50M. The dry run only proves the mechanism, because its
  citations were topped up through the API. `prep_log.json` will report the real number.
- That sustained S3 throughput over 297 GB matches the 20 s probes.
- `prep` and `attrs` wall time and RAM at 50M. Not measured; the design is streaming or
  numpy-vectorised, and the box has 1 TB of RAM.

## 3. Work completed (all on dev/e3-openalex, commit 9f64057, not pushed)
- `etl/openalex.py`: subcommands `plan, download, convert, prep, encode_text, encode_queries,
  attrs, all` (arxiv's shape plus pubmed's `plan`). Lint clean.
- `tests/eval_datasets/test_openalex.py`: 9 CPU tests, all pass. The whole harness suite:
  225 passed, 1 skipped.
- Dry run under `RETRIEVE_DATA_ROOT=/scratch/e3-dryrun`:
  - `convert --files 421,482,533,540,570,634,670,1393` read 50,304 real records and staged
    4,581.
  - `api_citation_topup.py` added the 1,060 eligible works that 60 pool papers really cite.
  - `prep --keep-items 5073` held out 75 queries with 994 qrels (405 pass field + era).
  - `verify_prep.py`: 0 mismatched.
  - `encode_text --shard-rows 2000` plus `encode_queries` ran on the GPU. A deleted shard
    re-encoded bit-identically. The params guard refused a changed config.
  - `attrs` produced `[5073, 5, 4]`, reverse `[F,F,F,T,F]`, and the target passes field+era
    100 %.
  - `bench check --config-dir <scratch>` (data_dir pointed at the dry-run output): `openalex
    d768: ok`. It flags a removed `eval_split.parquet`.
  - All four `layout` readers load the output.
  - Filtered top-10 contains a cited paper for 72/75 queries. This is inflated by the top-up;
    it is a sanity check, not a result.
- `einops>=0.7` added to `evaluation/pyproject.toml` plus `uv.lock`. nomic's remote code needs
  it, so **arxiv's `encode_text` could not have run in the current venv either**. einops 0.8.2
  was installed into `/venvs/retrieve` (additive only).
- The real release is pinned at `/data/_raw/openalex/manifest.json`. `/scratch/e3-dryrun` is
  throwaway.

## 4. Ingestion plan (the deliverable) and its decisions
1. **`plan`**: reads the manifest, the footers of all files, and a seeded row-group sample,
   and prints `--sample-rate` plus the disk and time budget (`--report` writes JSON).
2. **`download`**: pins `manifest.json` into `_raw/openalex/`. Every step walks those files in
   (date, part) order.
3. **`convert`**: a true stream, one spawned process per file (the spawn start method avoids
   the forked-S3 deadlock). It reads row group by row group with only the 14 projected leaves,
   runs `stage_table`, and atomically writes `_raw/openalex/staging/<date>_<part>.parquet`.
   - Filters, in order: year in [2000, release year]; `en`; type in {article, preprint,
     review, conference-paper, book-chapter, dissertation}; not paratext/retracted/xpac;
     `pmid_hash(work_id, seed) < rate·2^64`; abstract present. Only the rows that survive
     are JSON-parsed.
   - Resume works per file. `staging/params.json` refuses a resume under changed parameters.
     A file that fails on OSError/Arrow is reported and skipped (rc 1); a rerun picks it up.
   - The snapshot is never landed.
   - Rejected: the JSONL copy (full-width bytes, JSON-parsing every record) and the API
     (metered, and 476M records).
4. **`prep`**:
   - The catalog is the N smallest work-id hashes, via `pubmed.select_pmids`. That is
     order-free, and a smaller N is a subset of a larger one. A duplicate id keeps its newest
     copy.
   - Item ids are dense in (file, row) order; `papers.parquet` is written in item-id order.
   - The staged works ranked above N form the **held-out pool**, which is never in the
     catalog. This is a real hold-out, unlike arxiv, whose held-out items stay indexed.
   - A query is a pool paper with ≥1 in-catalog reference of the same field and a strictly
     earlier era. `--n-heldout` 10k are drawn with the seed.
   - Relevance: `qrels.parquet` holds all in-catalog references. The target in
     `heldout.item_id` is one seeded draw among the references that pass field + era, because
     the harness text path takes one target.
   - Rejected: item-as-query targets. Under "year < query year" the query paper fails its own
     filter.
5. **`encode_text` / `encode_queries`**:
   - Encoder: nomic-embed-text-v1.5 at native 768-d, the same encoder, prefixes and text
     template as arxiv, so the prefix policy and `layout` asserts apply unchanged.
   - Rejected: SPECTER2. It would need the `adapters` library, has no prefix-sidecar
     precedent, and the survey's 9–14 h estimate assumed nomic.
   - Output: bf16 weights, fp16 L2-normalised, 1M-row shards written atomically.
   - Resume: skip finished shards. `encode_params.json` guards the settings.
     `shard_index.json` and the meta are written only when complete, so a partial encode is
     never loadable.
6. **`attrs`**: `[N, 5, 4]` = C0 field (26), C1 earlier_era, C2 subfield, C3 source (top
   5,000, **reverse**), C4 is_oa (null → 0, which also guarantees no all-(−1) row).
   - The query side is the held-out paper's own field, era, subfield and venue; C4 = 1.
   - "year < query year" becomes an **era set**. An item of era b lists eras b+1..4, and the
     query sends its own era. Edges are [2010, 2015, 2019, 2022]; the eligible sample splits
     25/21/16/15/23 %.
   - This is subtractive: it tests "earlier era", not the exact year, and is documented as
     such. Why eras at all: clauses are equality-only, and A_max = 4 allows 5 eras.
   - Rejected: a wider A_max (it costs device memory on every clause).
7. **Config**: `config/openalex.yaml` is named after the data it holds (OpenAlex + nomic), not
   the S2 plan. `field_era` = [0, 1] is the roadmap filter. Bloom sweeps exclude C3. It is in
   no suite.

## 5. Files
- Created: `evaluation/eval_datasets/etl/openalex.py`, `evaluation/config/openalex.yaml`,
  `evaluation/tests/eval_datasets/test_openalex.py`, and under `docs/artifacts/e3-openalex/`:
  `plan-2026-09-23.{json,log}`, `stream_probe.py`, `stream_probe-{32w,64w}.json`,
  `api_citation_topup.py`, `verify_prep.py`, `dryrun-{prep_log.json,convert_log.jsonl,staging-params.json}`.
- Modified: `evaluation/eval_datasets/cli.py` (ETL entry `openalex`),
  `evaluation/pyproject.toml` and `uv.lock` (einops), `docs/system/datasets.md` (new
  `### openalex` section before `### synth_arxiv`, plus the shape table, module table, layout
  note and tests note), `docs/system/storage.md` (the E3 queued-datasets bullet).

### 5b. hub.py lines for the orchestrator to apply by hand (NOT applied)

```python
# EVAL_REPOS, after "pubmed":
    "openalex":          "pinkmeme/eval-openalex",
# RAW_REPOS, after the yfcc10m comment:
    # openalex is streamed from the public snapshot bucket (s3://openalex, anonymous, CC0)
    # by `openalex convert`, not from the Hub — listed here only so that
    # `raw_dir("openalex")` has a documented home next to the others.
```

With it, update `docs/system/datasets.md` § HuggingFace I/O: add `"openalex": "pinkmeme/eval-openalex",`
to the EVAL_REPOS block and a line "`pinkmeme/eval-openalex` is **registered but not
published**; nothing is pushed to it before roadmap E3." Optionally add `"qrels.parquet"` is
fine to publish; `queries.parquet` (text) could join `papers.parquet` in EVAL_IGNORE_PATTERNS.

## 6. Unresolved issues (for the orchestrator or the user)
1. **50M does not fit the harness**: 50M × 768 × 4 B = 153.6 GB of fp32 items on the device,
   against 80 GB. It is the same ceiling that made pubmed a 10M slice (~15M max at 768-d).
   Options:
   - (a) run E3 at ≤15M via `prep --keep-items`, a nested subset of the same hash order;
   - (b) a harness change: fp16 items plus a chunked oracle;
   - (c) Matryoshka truncation, which conflicts with the no-PCA / native-width decision.
   This is a user decision. The ETL supports any N.
2. `docs/system/evaluation.md:246` says "Seven files under `evaluation/config/`" and lists
   them. It needs `openalex.yaml` (8 files). Not edited: the file is owned by another worker.
3. `docs/validation.md` datasets row "Semantic Scholar, KuaiRand | not started" needs updating
   to "E3 OpenAlex: ETL written + dry-run on real rows (not staged), not citable". Not edited,
   per the brief.
4. The E3 roadmap entry's cost line ("~670 GB snapshot pass plus 9-14 A100 hours") is now
   measured as a 297 GB projected read (~0.5 h) plus ~15.5 A100-h of encode.
5. The latent einops gap for arxiv's encode is fixed by this branch's lock change. It may
   touch a sibling's `uv.lock` at merge; it is one package, so re-run `uv lock` if it
   conflicts.
6. Observation outside scope: pubmed's `convert` writes `shard_index.json` incrementally
   (after each shard), so a partially converted pubmed dir is loadable with
   uninitialised rows (`load_sharded` allocates `torch.empty`). openalex writes the index last.

## 7. Full-scale cost estimate (to schedule)

| step | resource | estimate |
|---|---|---|
| `convert --sample-rate 1.0 --workers 64` | network + CPU, **no GPU** | 297.1 GB read; 176–230 MB/s measured, so ~25–30 min; budget 1 h |
| `prep --keep-items 50000000` | CPU / RAM | not measured; minutes to ~30 min |
| `encode_text` | **GPU, exclusive** | 50M ÷ 898 docs/s ≈ **15.5 A100-h** (at 15M: 4.6 h); resumable per 1M-row shard, so a crash loses ≤ ~19 min |
| `encode_queries` + `attrs` | GPU seconds / CPU minutes | — |
| disk | overlay (200 GB, 169 free today) | staging ~29 GB + papers ~26 GB + fp16 shards 76.8 GB → **peak ~131 GB**; staging can be deleted after `prep` |

Use `--sample-rate 1.0` rather than plan's 0.962. The eligible estimate is ±10 %, and `prep`
cuts exactly N either way.

## 8. Next step
See `nextStep`. For the eventual run:

```bash
eval-data openalex download
eval-data openalex convert --sample-rate 1.0 --workers 64
eval-data openalex prep --keep-items <N> && python docs/artifacts/e3-openalex/verify_prep.py <out>
eval-data openalex encode_text
eval-data openalex encode_queries
eval-data openalex attrs
bench check --dataset openalex
```
