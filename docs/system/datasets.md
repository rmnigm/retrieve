---
title: datasets
created: 2026-09-26
updated: 2026-10-11
type: entity
tags: [datasets, training]
sources: [evaluation/eval_datasets/, evaluation/training/]
---

# Datasets and training

Everything upstream of the benchmark: how the datasets are fetched and
reshaped into the on-disk layout the harness expects, and how the SASRec
checkpoints that produce query embeddings are trained.

For what happens *after* — the sweep driver, measurement methodology,
output schema — see [evaluation.md](evaluation.md). For the trained
checkpoint inventory and the HF Hub workflow, see
[checkpoints.md](checkpoints.md).

## The three shapes of a dataset

The harness reads two different dataset shapes, and which one a config
gets is decided by whether it sets `checkpoint`:

| shape | datasets | query embeddings come from | filters |
|---|---|---|---|
| **sequential** | goodreads, yambda-500m (unfiltered) | a trained SASRec checkpoint ([which one](checkpoints.md#what-the-harness-reads)), encoded at eval time | goodreads |
| **text** | arxiv, arxiv-synth, arxiv-corr-synth, pubmed, laion30m | pre-encoded embeddings on disk | yes |

A third variant, **synthetic**, is a text dataset grown to arbitrary `N`
by interpolating between real embeddings — used for scale sweeps where a
real catalog that size doesn't exist.

Data lives under `$RETRIEVE_DATA_ROOT` (default `<repo>/evaluation/data`;
the pod image sets it, see [storage.md](storage.md)), read only by
`eval_datasets.hub.data_root()`; every ETL default path goes through it. Where it points
elsewhere, `evaluation/data` must be a gitignored symlink to it in every
worktree (`ln -s "$RETRIEVE_DATA_ROOT" evaluation/data`), because the
harness resolves `config/*.yaml`'s `data_dir: data/<dataset>` against
`evaluation/`, not against `$RETRIEVE_DATA_ROOT`. Raw downloads go to
`data/_raw/<dataset>/`; bench-side outputs to `data/<dataset>/`.

## `eval_datasets/` — what is on disk

The package that owns the on-disk layout: the contract as code
(`layout.py`), the Hub registry (`hub.py`), the shared numerics, and one
level down the per-dataset ETL scripts (`etl/`) behind one console script,
`eval-data`. It imports neither `bench` nor `training`
(`tests/test_dependency_direction.py`). The package is named
`eval_datasets` rather than `datasets` because the latter shadows
HuggingFace's `datasets` in the shared venv.

| module | `eval-data` subcommand | source |
|---|---|---|
| [`layout.py`](../../evaluation/eval_datasets/layout.py) | (`bench check`) | the layout contract — readers, checks, `validate_layout` |
| [`hub.py`](../../evaluation/eval_datasets/hub.py) | `fetch`, `publish`, `publish-checkpoint` | HF Hub push/pull, `data_root()` |
| [`etl/yambda.py`](../../evaluation/eval_datasets/etl/yambda.py) | `yambda` | HF `yandex/yambda`, Listen+ branch |
| [`etl/goodreads.py`](../../evaluation/eval_datasets/etl/goodreads.py) | `goodreads` | UCSD Book Graph mirror (HTTPS) |
| [`etl/arxiv.py`](../../evaluation/eval_datasets/etl/arxiv.py) | `arxiv` | HF `open-index/open-arxiv` (~2.99M papers) |
| [`etl/pubmed.py`](../../evaluation/eval_datasets/etl/pubmed.py) | `pubmed` | NCBI FTP MedCPT embeddings + MEDLINE baseline (~36M articles) |
| [`ingest.py`](../../evaluation/eval_datasets/ingest.py) | `ingest` | vectors + attribute tables, no ETL ([below](#the-generic-ingest-eval-data-ingest)) |
| [`etl/laion.py`](../../evaluation/eval_datasets/etl/laion.py) | `laion` | HF `laion/relaion2B-en-research-safe` (gated, auto-approved), first two parquet parts |
| [`etl/synth_arxiv.py`](../../evaluation/eval_datasets/etl/synth_arxiv.py) | `synth-arxiv` | an already-encoded arxiv directory |
| [`common.py`](../../evaluation/eval_datasets/common.py) | — | shared attribute synthesis, id-hash sampling, `prep_log.json` merge, range specs |
| [`timesplit.py`](../../evaluation/eval_datasets/timesplit.py) | — | vendored sequential time-split |

The ETL modules are `argparse` programs; `eval-data <name> …` forwards
its arguments to that module's `main(argv)`, so `uv run eval-data arxiv
--help` is arxiv's own subcommand list (`download`, `convert`, `prep`,
`encode_text`, `encode_queries`, `attrs`, `all`).

### The layout contract (`layout.py`)

[`layout.py`](../../evaluation/eval_datasets/layout.py) is the one place
that says what a dataset directory must contain, shared by the writers
here and the readers in `bench/inputs.py`: `load_text_items` /
`load_text_queries` (the pre-encoded shape, prefix sidecars asserted, fp16
→ fp32, L2-normalised unless the dataset config sets `normalize: false`, in which
case both sidecars must say `normalization: "none"` or the load raises), `load_query_attrs` (row count checked against the
full test split), `load_item_attrs` (the legacy pad row dropped, rows
checked against the items), `apply_users_limit` (one prefix over every
query-side tensor), `atomic_write`, and `validate_layout(data_dir,
content_dir) -> list[str]` — every way the directory can be wrong for the
harness (missing files, a missing, keyless or swapped prefix sidecar, `query_emb`
vs `heldout` rows, attrs vs items, `eval_split` vs queries) — which
`bench check --dataset <name>` runs at every dim. Run it on a freshly
staged dataset before a campaign: a loader that drops query rows or
writes no prefix sidecars is exactly what it reports.

**The prefix policy** (`layout.prefix_problem`, shared by the loader's
`assert_prefixes` and by `validate_layout`): every `text_emb.meta.json` /
`query_emb.meta.json` must carry a `prefix` key. It is either the nomic
prefix the harness expects (`search_document: ` / `search_query: `) or
**explicitly `null`**, which declares that the encoder has no prefix
concept — LAION's CLIP vectors, MedCPT's precomputed vectors. A sidecar
*without* the key is a problem, not a pass, because it cannot be told apart
from a forgotten prefix; a missing sidecar is a problem in `validate_layout`
and a logged warning in the loader.

### Tests

```bash
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/eval_datasets -q     # CPU-only, no GPU
```

[`tests/eval_datasets/`](../../evaluation/tests/eval_datasets/) is CPU-only
and needs no network: `test_layout.py` runs the contract on the shared
tiny-dataset writer of `tests/conftest.py` (both item layouts clean, each
breakage flagged). Give each new loader a fixture test that writes the upstream format into `tmp_path`
and runs `validate_layout` on
what it wrote.
`test_synth_filter.py` is G-synth ([below](#synthetic-selectivity-attrs)).

### Shared conventions

- **Item ids are 1-indexed dense ints** in `item_id_map.json`, because
  id `0` is the sequence-padding token on the trainer side. The harness
  drops the padding row when loading (`item_embs[1:]`) and shifts target
  ids by −1, so the `retrieve` library is 0-indexed over real items
  throughout. This is why no filter mask needs an "item 0" fixup.
- **Attribute tensors are 0-indexed dense**: row `i` of
  `item_attrs_narrow.pt` describes `item_id i+1`.
- **Legacy `[N+1, …]` artifacts are accepted, not assumed.** Both layouts
  exist: the ETL writes `[N, …]`, but the copies *published on the Hub* —
  what `eval-data fetch` pulls — are 1-indexed tensors with a padding row
  at index 0, as their own README and `text_emb.meta.json` say.
  `eval_datasets.layout.drop_legacy_padding_row` recognises that row by its
  content (all-zero for embeddings, all `-1` for attributes) and drops it
  from every per-item tensor the harness loads (the pre-encoded `text_emb`
  and `item_attrs_narrow`; the SASRec path's `nn.Embedding` pad row is
  dropped by construction), and `load_inputs` then requires the attrs row
  count to equal `item_embs`'s, raising with both counts otherwise
  (`check_items_aligned`). Held-out ids are 1-indexed on disk in both
  layouts, so the −1 shift is the same; the drop fixes the rows they
  index. Getting this wrong is not always loud: on the arxiv path
  attrs and embeddings are *both* 1-indexed, so they agree with each other
  and only the held-out target shift is wrong — `cos(query, target)` falls
  from 0.99 to 0.62 with no error anywhere.
- Every subcommand writes a `prep_log.json` with row counts and
  filtering statistics next to its outputs.
- Subcommands are individually re-runnable; `all` chains them.

### yambda

yambda-500m is in the study unfiltered only: it has no attributes, so its
config has no `filters` block and every cell is `none`; `users_limit: 10000`
keeps the first 10,000 of 45,932 test users ([decisions](../../research/decisions.md#datasets)).
The harness reads the E1c checkpoints `checkpoints/sasrec-ssm-logq-d{64,128,256}`
and `test.parquet`, which is the Hub's `trainer/test.parquet` (the split
the checkpoints were scored on; the Hub's root `test.parquet` has the
same rows without timestamps).

`uv run eval-data yambda prep --variant {50m,500m} --output-dir data/yambda/<v>`

Downloads `<variant>/sequential/listens.parquet`, runs `preprocess()`
(Listen+ branch: `played_ratio ≥ 50%`), and writes the four artifacts the
trainer consumes:

```
<output>/train.parquet      item_ids, timestamps: list[int64]
<output>/val.parquet        item_ids, timestamps, targets
<output>/test.parquet       item_ids, timestamps, targets
<output>/item_id_map.json   {raw_yandex_id: dense_int}
```

Validation history is the train portion (already sliced to the last 200
items in `preprocess`); test history is train ++ val, last 200.
`timestamps` is the event time of each `item_ids` entry, same length and
order; it covers the input history only, never `targets`. Yambda ships
its time as **seconds since an anonymized dataset epoch** (UInt32,
0..26,000,000, about 301 days), not unix time, so no conversion to unix
is possible: the ETL only widens it to int64. The val/test row order is
not reproducible run to run (it comes out of a `uid` join that is then
dropped); the row contents are. Adapted
from the Yambda paper's reference `sasrec/data.py`; only the
listens-Listen+ branch is kept, the rest is replaced by the `training`
package ([`evaluation/training/`](../../evaluation/training/)).

### goodreads

`download` → `convert` → `prep` → `attrs`.

`download` mirrors the UCSD top-level files over HTTPS (resumable via
`Range`, sha256 manifest); `convert` streams them into ZSTD parquet.

**`prep`** turns the staged parquets into yambda-shaped trainer inputs:
filters `is_read=true`, parses `date_added`, collapses editions to a
`work_id` catalog, runs an iterative n-core, then time-splits via
`timesplit.sequential_split_train_val_test`. Outputs `train/val/test.parquet`,
`item_id_map.json`, `book_to_work.parquet` (reused by the filter eval),
and `prep_log.json`. The parquets have the yambda columns: `item_ids`,
`timestamps` (unix seconds of `date_added`, int64, same length and order
as `item_ids`, the input history only) and, for val/test, `targets`.
Nothing breaks ties between one user's events at the same second, and
goodreads has many (bulk shelving), so the order inside a tie, and which
tied items survive the 200-item cut, change from run to run; a re-run
differs from the Hub copy only there
([validation](../../research/validation.md#trainer-inputs-with-timestamps-data-gates-not-citable)).

**Vocab order (`attrs`).** Every vocab and top-k cut is a total order of its inputs: language,
author and wide-shelf vocab by count descending, **ties by name**; per-book and per-work genre
top-4 and per-work shelf top-32 by count descending, ties by genre id / shelf name; the joins
that feed a `head(k)` keep the left order (`maintain_order="left"`); `prep`'s `book_to_work`
keeps the first row per `book_id`. Before this, `group_by` handed back tied values in a
thread-dependent order, so their ids (C1 language, C4 author, the wide shelves) and the genre
picked among tied counts changed from build to build. Two builds are byte-identical
(`tests/eval_datasets/test_goodreads.py`). The staged copy stays canonical, pinned by Hub
revision `pinkmeme/eval-goodreads-work-id@4714e9b5` (`item_attrs_narrow.pt` sha256
`584993bd…`, `eval_split.parquet` `f1208e12…`); a rebuild with this code is **not byte-equal** to it.

> `prep` deliberately passes `drop_non_train_items=False`, mirroring
> `yambda.preprocess`. Setting it `True` makes polars re-evaluate an
> imploded set per row and blows the job past a 129 GB cgroup limit.

**`attrs`** builds the filter-bench tensors from `prep`'s outputs plus
the catalog parquets:

```
item_attrs_narrow.pt        [N, 5, 4] int64   (genre, lang, format, year, author)
item_attrs_wide.pt          [N, 1, 32] int64  (shelves; built, never read)
clause_is_reverse_narrow.pt [5] bool = [F, T, F, F, F]
lang_vocab.json / format_vocab.json / author_vocab.json / wide_shelf_vocab.json
wide_shelf_global_freq.pt   [V_wide] int64
eval_split.parquet          aligned 1:1 with test.parquet rows
```

Clause 1 (language) is the **reverse** clause — the one sweep that
exercises negated predicates end-to-end. It is bloom-incompatible, so
goodreads bloom sweeps exclude it.

**The staged file is the published one, not this layout.** The Hub copy the
harness reads (`@4714e9b5`) has **four** clauses in the legacy `[N + 1, 4, 4]`
layout (pad row 0, dropped by the loader): C0 genre (top 4), C1 language
(reverse), C2 format, C3 year bucket. It has no author clause, and its
`clause_is_reverse_narrow.pt` is `[F, T, F, F]`.
[`config/goodreads.yaml`](../../evaluation/config/goodreads.yaml) indexes
those four (`all4` = all of them); re-running `attrs` writes the five-clause
`[N, 5, 4]` file above, which the configs do not read.

### arxiv

`download` → `convert` → `prep` → `encode_text` → `encode_queries` →
`attrs`, or `uv run eval-data arxiv all --output-dir data/arxiv-papers` (the
`data_dir` of [`config/arxiv.yaml`](../../evaluation/config/arxiv.yaml)).

**Vocab order (`attrs`).** Main-category, author and wide-leaf vocab by count descending, **ties
by name**; the author join before each paper's `head(2)` keeps the left order. Before this the
tied values (about 1.5 M authors, most of them tied) got thread-dependent ids, so C4 author ids
and, through the query side, `eval_split.parquet` changed from build to build. Two builds are
byte-identical (`tests/eval_datasets/test_arxiv.py`). The staged copy stays canonical, pinned by
Hub revision `pinkmeme/eval-arxiv-papers@354e6937` (`item_attrs_narrow.pt` sha256 `7f5f2f4d…`,
`eval_split.parquet` `91b58ab2…`); a rebuild with this code is **not byte-equal** to it.

Arxiv has **no user sequences**, so `prep` does no interactions and no
time-split — it emits `item_id_map.json`, `papers.parquet`, and a
sampled `heldout.parquet`. Held-out items stay in the index.
Each query is therefore its own target's text, so `config/arxiv.yaml` (and arxiv-synth,
arxiv-corr-synth, which share these queries) sets `self_match: true`: oracle metrics drop the
query's own item, and held-out recall here is self-retrieval, not end-task recall
([evaluation § Self-match](evaluation.md#self-match)).

The two encode steps use the same model (`nomic-embed-text-v1.5`,
Matryoshka-truncated) with **different prefixes** — `"search_document: "`
for items, `"search_query: "` for queries. That asymmetry is
nomic-specific and load-bearing: it is what makes the unfiltered
cross-check sweep meaningful rather than an identity lookup. The loader
asserts the prefix recorded in each `*.meta.json` sidecar, so a
prefix mismatch fails at load instead of silently producing garbage
recall.

Embeddings are written per dimension:

```
<output>/content/       text_emb.pt + query_emb.pt at D=256
<output>/content_d128/  same, Matryoshka-truncated to 128
<output>/content_d64/   same, truncated to 64
```

A dataset config selects one per dim through its `content_dir` mapping
([evaluation](evaluation.md#config-one-yaml-per-dataset--suitesyaml)).

**Filter attributes.** The published `item_attrs_narrow.pt` is the legacy
`[N+1, 4, 4]` layout (pad row 0, dropped by the loader): C0 main category,
C1 license bucket (`license_vocab.json`, 11 buckets), C2 update-year
bucket, C3 version-count bucket. The current `attrs` step writes
`[N, 5, 4]` instead (no pad row, a C4 top-2 author clause), so re-running
it does not reproduce the published file; the published one is what
`config/arxiv.yaml`'s `all4` sweep reads.

Every `map_elements` over a nullable column passes `skip_nulls=False`.
polars (1.40.1 here) skips nulls by default, so the helper's `None`
branch never ran and the result stayed null, which `.to_numpy()
.astype(np.int64)` turns into INT64_MIN. That hit C1: the 452,732 papers
with a null license (15.1 %) carried INT64_MIN in the published file, and
so did the 1,528 `eval_split.parquet` queries whose target is one of them,
instead of bucket 9 `none`. C2 and C3 had the same latent pattern, but
the source has no nulls there. The corrected C1 is re-derived by
[`patch_license_clause.py`](../../research/artifacts/arxiv-etl-fix/patch_license_clause.py),
which rewrites only C1 and the matching query column and asserts that every
non-null item's bucket is unchanged.

### pubmed

**Status: the 10 M slice** (`--keep-items 10000000 --seed 0`) under
`$RETRIEVE_DATA_ROOT/pubmed-medcpt-dot`, **MedCPT as shipped: items and queries not
normalised**, scored by dot product as MedCPT is trained (`normalize: false` in
[`config/pubmed.yaml`](../../evaluation/config/pubmed.yaml); user, 2026-10-13). Its attrs,
`eval_split.parquet`, `heldout.parquet`, `queries.parquet`, vocabularies and article parquets are
copies of the canonical `pubmed-medcpt` (vocab order below), its `item_id_map.json` is byte-equal to
it, and only `content_d768/` was rebuilt. The older `pubmed-medcpt` holds the same vectors
L2-normalised (cosine). `bench check` passes. A rebuild from source on a fresh pod gives
the same slice, embeddings included; state in [validation](../../research/validation.md#datasets).
The dataset is not on the Hub.

`plan` → `download --what pmids` → `convert --fetch --delete-raw` →
`medline --stream` → `attrs` → `queries` → `encode_queries`. Source is the
NCBI FTP MedCPT article-embedding release, public domain, no registration:
`https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/` — 38 chunks
of

- `embeds_chunk_{i}.npy` — `(N_i, 768)` **float32** (verified from the npy
  header, `descr='<f4'`; **110.35 GB** as served on 2026-09-16),
- `pmids_chunk_{i}.json` — the row-aligned PMID list (0.42 GB),
- `pubmed_chunk_{i}.json` — `{pmid: {"d": date, "t": title, "a": abstract,
  "m": mesh}}` (**52.89 GB**).

That is **35,920,666 articles** (the PMID lists and the npy headers agree),
and chunk *i* holds exactly PMIDs `i,000,000 … i,999,999` — the 38 ranges
are disjoint with no duplicate PMID anywhere, which `convert` re-checks and
records as `pmid_ranges_disjoint`. `plan` re-derives every size above from
the server (`HEAD` per file, a `Range` read of each npy header) and prints
the disk and wall-time budget for a given `--keep-items` — run it before
any download.

NCBI publishes **no** checksums for that directory, so `verify` checks the
shards structurally instead: the npy header must parse, dtype must be float32,
width 768, and the row count must equal `len(pmids_chunk_i.json)`. The MEDLINE
baseline *does* publish `.md5` and `verify --medline` checks those.

**No dimensionality reduction.** Every dataset is benchmarked at its encoder's
native dim ([decisions](../../research/decisions.md#datasets): no PCA), so pubmed has
exactly one content dir, `content_d768`.
[`config/pubmed.yaml`](../../evaluation/config/pubmed.yaml) is the dataset
file, listed in the `filter` suite at 768.

#### The streaming `convert`

The raw mirror (163.7 GB) plus the fp16 item matrix (55.2 GB) plus the
MEDLINE baseline (53.9 GB) is 273 GB — more than the 300 GB overlay can
give one dataset next to the others. So `convert` never holds the
mirror:

1. **PMID lists first** (0.42 GB, `download --what pmids` or `--fetch`).
   They fix the id map: item ids are 1-indexed dense in *(shard order,
   ascending PMID within the shard)* — equal to ascending PMID order given
   the disjoint ranges — and with `--keep-items N` the map covers only the
   `N` articles `select_pmids` picks: the `N` smallest values of a seeded
   splitmix64 hash of the PMID, so the slice is the same set whatever the
   shard order or the fetch history, and is spread uniformly over
   1781–2024 rather than being the oldest `N`.
2. **Shard by shard**, `--prefetch` shards downloading ahead: parse the
   chunk JSON into `staging/articles_chunk_{i}.parquet` (`pmid, year,
   has_abstract, mesh, title` — the title stays so `queries` works after
   the JSON is gone), gather the kept rows of the npy in PMID order,
   cast to fp16 (not normalised) and `torch.save` them as
   `content_d768/text_emb_shard_{i:02d}.pt`; append the `{filename,
   start_id, n_rows}` entry to `shard_index.json`; `--delete-raw` the
   JSON and npy. A killed run resumes at the first shard whose `.pt` or
   parquet is missing. A shard whose article parquet already exists does not
   fetch its JSON, so re-folding the vectors into a directory seeded with
   `staging/` downloads only the npys (110 GB).

The output is the **sharded layout** `layout.load_sharded` already reads
for the synthetic arXiv catalogs (`shard_index.json` + contiguous
`text_emb_shard_*.pt` blocks); there is no monolithic `text_emb.pt`, no
accumulator and no second on-disk copy. `text_emb.meta.json` carries
`prefix: null` (MedCPT has no prefix concept) and the `keep_items` / `seed`
of the slice. Peak disk = the finished shards + the parquets + the PMID
lists + `1 + prefetch` raw shards in flight — the numbers are under
*Disk budget and the slice* below.

#### Attribute semantics

The `m` field is a `|`-separated list of `descriptor!qualifier` entries with a
trailing `*` marking a major topic:

```
"humans!|rectal neoplasms!|rectal neoplasms*|rectal neoplasms!therapy|"
```

`parse_mesh_field` keeps the descriptor only, lower-cased and de-duplicated in
first-seen order — so the three `rectal neoplasms` forms above collapse to one.

Journal and language are **not** in the chunk JSON. They come from a join
against the MEDLINE baseline (`https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/`,
1334 × `pubmed26n*.xml.gz`, **51.8 GB**), which `medline` streams into
`medline/*.parquet` (`pmid`, `journal` = `MedlineTA`, `language`). An
article with no baseline row (0.27 % of the 10 M slice) gets journal and
language `-1`, the pad, like an out-of-vocab journal. MeSH tree-top
category letters come from the MeSH descriptor file
(`xmlmesh/desc2026.gz`, 17 MB).

`item_attrs_narrow.pt` is `[N, 5, 4]`, the same shape as goodreads and arxiv:

| clause | attribute | cardinality | notes |
|---|---|---|---|
| C0 | MeSH descriptor | top-30k vocab | multi-valued OR, K=4 |
| C1 | MeSH tree-top category | 16 (A–N, V, Z) | from C0's first descriptor |
| C2 | year bucket | 7 | `<1975 … ≥2020` |
| C3 | journal (`MedlineTA`) | top-5k | **reverse clause** |
| C4 | has-abstract flag | 2 | old citations often have none |

`clause_is_reverse_narrow.pt` is `[F, F, F, T, F]` — C3 is the negated
predicate, the pubmed analogue of goodreads' C1 language clause. It is
bloom-incompatible, so pubmed bloom sweeps exclude it.

**The MeSH cap.** An article carries 10–15 headings but the narrow tensor holds
four, so `cap_mesh_by_rarity` keeps the **four globally rarest** in-vocab
descriptors, *rarest first*, ties broken on vocab id. The order is load-bearing:
`common.synthesize_qa_narrow` takes the first non-pad entry as the query-side
clause value, so a frequency-descending order would hand every query "humans"
(a ~40 % pass rate) instead of a selective heading.

Language is parsed and stored in `articles.parquet` + `lang_vocab.json` but is
*not* one of the five clauses.

**Vocab order.** Every vocab id is a total, seed-independent order of the inputs:
MeSH by count descending, ties in first-seen order over the item-id-ordered article
rows (`Counter.most_common`); journal and language by count descending, **ties by
name** (`group_by` returns groups in a thread-dependent order, so a count-only sort
gave tied journals different ids, and a different top-5k cut, on every build). Two
builds of the same inputs are byte-identical (`test_attrs_two_builds_are_byte_identical`).
The staged `pubmed-medcpt` predates the tie rule: its journal / language ids are
one draw of the old order and stay canonical (controller, 2026-10-10), pinned by sha256
(`item_attrs_narrow.pt` `8a3bce0f…`, `eval_split.parquet` `3a83c6f9…`); the dot-product build
`pubmed-medcpt-dot` copies these files, and `pinkmeme/eval-pubmed@10d48d05` holds them. Pods that need the old one copy it; a rebuild with this code is **not byte-equal** to it.

#### Query sets

`queries` builds `queries.parquet` (`query_id`, `text`, `target_id`) from two
sources:

- `heldout` — item-as-query: a held-out article's title (from the article
  parquets) is the query and the article itself is the single relevant item;
  the article stays in the catalogue, so `config/pubmed.yaml` sets `self_match: true`
  and held-out recall is self-retrieval
  ([evaluation § Self-match](evaluation.md#self-match)). Always available. **Every held-out row keeps its query row** — an article
  without a title gets an empty string, it is not dropped — so
  `query_emb.pt` stays aligned 1:1 with `heldout.parquet` and
  `eval_split.parquet`, which `bench check` verifies.
- `nfcorpus` — the NFCorpus (BEIR) biomedical query set. NFCorpus document ids
  *are* PMIDs, so its qrels map straight onto our item ids. BEIR asks that its
  corpus not be redistributed, so nothing is downloaded automatically: stage
  `queries.jsonl` + `qrels/test.tsv` under `--nfcorpus-dir` yourself, or the set
  is skipped with a warning. Its rows go to a separate
  `queries_nfcorpus.parquet`, never appended to the held-out set, for the
  same alignment reason.

`encode_queries` runs `ncbi/MedCPT-Query-Encoder` ([CLS] pooling) and writes
`content_d768/query_emb.pt`, fp16, not normalised (`normalization: "none"` in the sidecar). MedCPT's query and article encoders are
*asymmetric* — the same load-bearing property as nomic's prefixes on arxiv — so
the unfiltered cross-check sweep is meaningful rather than an identity lookup.

#### Disk budget and the slice

Computed by `eval-data pubmed plan --medline --keep-items N`
from the server's own sizes (the JSON reports are in
[dataset-candidates-artifacts/pubmed/](../../research/artifacts/dataset-candidates/pubmed/)),
at the 23.9 MB/s a single-stream 64 MB probe measured that day (an earlier
`curl` probe saw 41 MB/s; the range is the honest number):

| | full catalog | `--keep-items 10000000` |
|---|---|---|
| articles | 35,920,666 | 10,000,000 |
| raw to download (MedCPT 163.66 GB + MEDLINE 53.93 GB) | 217.6 GB | 217.6 GB — every shard is scanned; the slice is a hash, not a prefix |
| fp16 item shards on disk | 55.17 GB | 15.36 GB |
| article parquet (60 B/row; 55 measured) + MEDLINE parquet (bound) | ~2.2 + 1.2 GB | ~0.6 + 1.2 GB |
| raw in flight (`1 + prefetch` largest shards, chunks 31 + 34) | 9.6 GB | 9.6 GB |
| **peak disk** | **~68.6 GB** | **~27.2 GB** |
| wall time (download-bound) | ~2.5 h at 23.9 MB/s, ~1.5 h at 41 MB/s | same |
| items as the harness holds them: fp32 on the device | **110.3 GB** | 30.7 GB |

The 55 B/row is chunk 37's article parquet, 20,947,498 B for 380,761 rows;
`etl/pubmed.py`'s `ARTICLE_PARQUET_BYTES_PER_ROW = 60` keeps a margin over it.

The disk is not the obstacle: both fit beside goodreads + arxiv on the
overlay. **The GPU is.** `bench.inputs.load_inputs` holds the item matrix
fp32 on the device, so the full catalog at 768-d needs 110 GB on an 80 GB
A100 before any index exists — and even an fp16-items harness change would
put 55 GB of items next to SilverTorch's 27.6 GB of int8 codes. **The 10 M
slice is therefore the E2 target**, not a stopgap: 30.7 GB of items + 7.7 GB
of int8 codes + 1.6 GB of attrs leaves the working set the oracle and the
perf pools need, it is the papers' 10 M pool size, and the
largest catalog the harness as written can take at 768-d is ~15 M. Going
above that is a harness decision (fp16 items + a chunked oracle), not an
ETL one.

**Measured on the staged slice.** `convert` took 57 min for
all 38 shards, with `medline --stream` (21 min, 39,994,988 rows) running
beside it. On disk: 17 GB under `pubmed-medcpt/` (15 GB of fp16 shards,
1.5 GB of attrs, 0.7 GB of `staging/` article parquets, 0.1 GB of MEDLINE
parquet), plus 0.4 GB of PMID lists kept in `_raw/pubmed/`. The exact
`c0_mesh` oracle (k_gt 1000, 10,000 queries) builds in 23 s
([artifacts/e2-pubmed/](../../research/artifacts/e2-pubmed/)). The item budget above
leaves little room for the index builds' own temporaries: SilverTorch's
one-shot int8 quantize (two more 28.6 GiB fp32 copies) and LiNR V3's
one-shot 1-bit build both OOMed here. Both now quantize chunk by chunk in
the library, with `bench/` and the slice unchanged
([validation](../../research/validation.md#library-gates), rows *Build-time int8
quantization* and *Build-time 1-bit quantization*).

### laion30m

**Status: built, `bench check` ok; not on the Hub, so a pod that needs it rebuilds it with the
steps below (Re-LAION may be shared as vectors only, and the user decides whether:
[roadmap](../../research/roadmap.md#needs-the-user)).** Roadmap V-LAION30: a 30 M scale point at d256, `filter` only
([decisions](../../research/decisions.md#datasets)). Nothing here is citable. The first dataset staged
through the [generic ingest](#the-generic-ingest-eval-data-ingest).

**Order.** The clause codes come from the generic ingest (count descending, ties by value), so
they are deterministic. `prep` picks items (`head(keep_items)`) and queries (`df[chosen]`) by row
position after the domain join, which now keeps the left (rank) order explicitly. polars 1.40
already kept it, so this guards a future polars rather than changing today's output. Two `prep`
runs are byte-identical (`test_prep_two_builds_are_byte_identical`). The pod-d copy stays canonical.
It is not on the Hub, so it is pinned by its files' sha256 at copy time; byte-equality of a
rebuild with it is expected but not verified.

`download` → `prep` → `encode_text` → `encode_queries` → `ingest`
([`etl/laion.py`](../../evaluation/eval_datasets/etl/laion.py)). The LAION-specific steps write
to `_raw/relaion/laion30m/`; `ingest` stages `data/laion30m/` and writes
[`config/laion30m.yaml`](../../evaluation/config/laion30m.yaml). Source: Re-LAION-2B-en-research-safe
on the Hub, gated with automatic approval (the pod's token reads it); `download` takes parts
`0-1`, 2 × 3.6 GB, 32,776,409 rows (16,388,209 + 16,388,200), with url, caption, similarity
(CLIP image-caption cosine), pwatermark, punsafe, original width / height, key and a 64-bit
row hash. It has no embeddings; we encode the captions.

**`prep`** (70 s on 56 threads): strip the captions and drop the blank ones (108); keep
**one row per distinct caption**, because an identical text is an identical vector. Of the
rows sharing a caption, the one with the smallest `common.pmid_hash(hash, seed)` survives:
30,928,288 distinct. The catalog is the 30,000,000 smallest ranks, in source order
(`items.parquet`, `item_id` 1..N). The remaining 928,288 rows are the held-out pool, never in
the catalog; 10,000 are drawn with the seed (`queries.parquet`). Both files carry each row's
`key`, `url`, registered `domain`, the raw tag values and the four tag buckets. They stay
under `_raw/` and are never published (the captions and urls are the gated upstream's).

**Domains.** `domain` is the url host's registrable domain under the public suffix list that
`tldextract` bundles (offline, no fetch: deterministic per tldextract version), so
`cdn.shopify.com` becomes `shopify.com`. A host with no registrable domain (an IP address)
keeps the host. The catalog has 1,430,979 domains; the largest is shopify.com at 5.9 %, then
pinimg.com, wp.com and cloudfront.net. Image CDNs count as sites. 97.2 % of the queries' domains
occur in the catalog. A query's own domain passes, at the 10 / 25 / 50 / 75 / 90 % quantiles,
5e-7 / 1.7e-5 / 1.6e-3 / 1.1e-2 / 4.1e-2 of the items: this is the selectivity range that
"search within a site" spans.

#### Attribute semantics

Every tag is single-valued, so the ingest writes `[N, 6, 1]` int64 (1.4 GB at 30 M, against
5.8 GB padded to the other datasets' 4 slots). Bucket edges are upper-exclusive
(`np.searchsorted(..., side="right")`), picked near the quintiles; a missing value is null,
which the ingest codes as `-1`. The ingest codes every column by count, so a bucket's code
is not its bucket index; `vocab.json` maps codes back to values. The shares are measured on
the 30 M items:

| clause (`laion.CLAUSES`) | column | values | item shares by bucket |
|---|---|---|---|
| C0 `domain` | registered domain | 1,430,979 | — |
| C1 `size` | `max(original_width, original_height)`, edges 200 / 300 / 500 / 800 px | 5 | 11.8 / 19.4 / 29.4 / 19.7 / 16.3 %, 3.45 % missing |
| C2 `similarity` | similarity, edges 0.30 / 0.315 / 0.33 / 0.35 | 5 | 7.8 / 26.1 / 22.5 / 21.2 / 22.2 % |
| C3 `domain_reverse` | `domain`, **reverse** ("other sites") | as C0 | — |
| C4 `watermark` | pwatermark, edges 0.1 / 0.2 / 0.35 / 0.6 | 5 | 19.9 / 24.6 / 22.8 / 18.6 / 13.4 %, 0.68 % missing |
| C5 `unsafe` | punsafe, edges 1e-5 / 1e-4 / 1e-3 / 1e-2 | 5 | 15.9 / 28.2 / 29.3 / 16.9 / 9.6 % |

Every query value is the held-out caption's own tag; a domain absent from the catalog is
`-1`, so that row has no live clause in a domain sweep and is skip-masked. C0 and C3 read
one column, so no sweep contains both (C0 ∧ C3 is empty). The sweeps (`laion.SWEEPS`):
`c0_domain`, `c3_domain_reverse` (bloom-incompatible), `tags4` = C1 ∧ C2 ∧ C4 ∧ C5, and
`all_fwd` = C0 ∧ `tags4`.

#### Encoding

`nomic-embed-text-v1.5` with the model card's Matryoshka recipe: bf16 weights, `layer_norm`
over the 768 output dims, keep the first 256, L2-normalise, store fp16. Prefixes
`search_document: ` (items) and `search_query: ` (queries), text = the caption,
`max_seq_length` 128 (the 99th-percentile caption is 295 characters). arxiv's d256 truncates
without the `layer_norm`, so the two datasets' vectors are not built the same way.
`encode_text` writes `_raw/relaion/laion30m/items_NNN.npy` (1 M rows each), resumable per
shard and pinned by `encode_params.json`; `encode_queries` writes `queries.npy`. The ingest
re-normalises them in fp32 into the staged fp16 shards (the measured rates are in
[validation](../../research/validation.md#datasets)). Targets: the ingest's default, each
query's exact unfiltered top-1 item (a caption query has no relevant item of its own). The
headline is `recall_oracle`, as everywhere.

On the device: items fp32 30.7 GB, attrs 1.4 GB, SilverTorch's int8 codes 7.7 GB.

### synth_arxiv

Grows an encoded arxiv directory to arbitrary `N` (15M / 30M / 50M) by
spherical k-means clustering the source embeddings (`cluster`, cached as
`clusters.pt` and reused across target sizes) and then SLERPing between
same-cluster parents (`synth`). Recall is measured against an exact-kNN
oracle over the *same* synthetic catalog, so synthesis needs no
anti-shadowing guarantees.

Output is a drop-in for the normal arxiv loader, except the embeddings
are sharded (`text_emb_shard_*.pt` + `shard_index.json`) and reassembled
at load. Vocabs, queries, held-out rows, and `eval_split` are copied
verbatim; wide attributes are not synthesized (the arxiv bench is
narrow-only).

There is a hard cap on `N`: above it the attribute tensors exceed
comfortable host RAM during construction.

### Subsets

`eval-data subset PARENT NAME --n-items N [--seed S]` ([`subset.py`](../../evaluation/eval_datasets/subset.py)) stages
`data/NAME` as a seeded row subset of a staged dataset, plus `config/NAME.yaml` and `config/NAME-synth.yaml` (the
parent's configs on the new data dir). It exists for the fixed-d N-sweep: LAION 30 M d256 at 1 / 3 / 10 M (`laion1m`,
`laion3m`, `laion10m`, seed 20261013).

- **Rows:** the sorted first N of one `randperm(N_parent)` from `torch.Generator().manual_seed(seed)`, so the subsets
  of one seed nest (1 M ⊂ 3 M ⊂ 10 M) and two builds are byte-identical
  (`tests/eval_datasets/test_subset.py`).
- **Copied as the parent's rows:** the fp16 item vectors (re-sharded at 1 M rows, not renormalised), `item_attrs_narrow.pt`,
  `item_attrs_synth.pt` and `synth_u.pt` (each item keeps its `u_i`, so its pass / fail at every rate; the sidecar's
  pass counts are recomputed).
- **Copied whole:** the queries, `query_attrs_synth.pt`, the reverse flags, `vocab.json` (codes and counts are the
  parent's) and the queries' real attrs in `eval_split.parquet`.
- **Recomputed:** the held-out targets, by ingest's rule over the subset (exact unfiltered top-1 by inner product, fp32,
  ties to the lowest id). The oracle is built per sweep by `bench oracle` where the legs run, as for any dataset.
- **Provenance:** `prep_log.json`'s `subset` entry (parent, seed, N, the sha256 of the row ids) and `MANIFEST.sha256`
  over every written file; a copy to another pod is checked against it.

### Synthetic selectivity attrs

`eval-data synth-filter --dataset <name> [--rates …]
[--draw 0]` (any dataset with a `config/<name>.yaml`; built for
goodreads, arxiv, laion30m and the PubMed dot-product build, `config/pubmed-synth.yaml`) ([`synth_filter.py`](../../evaluation/eval_datasets/synth_filter.py))
gives a dataset a controllable pass rate on its real embeddings. It draws
one `u_i ~ U(0, 1)` (float64) per catalog item from
`torch.Generator().manual_seed(20261008 + draw)`, in item-row order: N is the row
count of `item_attrs_narrow.pt` with the legacy pad row dropped, the N
`layout.load_item_attrs` checks. Item `i` passes the clause of rate `p`
iff `u_i < p`, so the pass sets nest across rates (the `p = 0.01` set is
inside the `p = 0.03` set) and `p = 1.0` is the unfiltered point. It
writes four files into the dataset's `data_dir`; draw `d > 0` writes its
own copies beside them (`layout.synth_draw_name`: `item_attrs_synth_s<d>.pt`,
`synth_u_s<d>.pt`, `synth_filter_s<d>.json`; the all-ones query file is
shared), so draw 0 keeps the names and digests every earlier record used.
Each draw has its own config, `<name>-synth-s<d>.yaml`, whose
`filters.attrs` names its file; a run pairs draw `d` with job seed `d`:

| file | content |
|---|---|
| `item_attrs_synth.pt` | `[N, 10, 1]` int64, 1 = pass, 0 = fail; clause `j` is rate `j` of `layout.SYNTH_RATES` = `(0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)` |
| `synth_u.pt` | the `[N]` `u` vector |
| `query_attrs_synth.pt` | `[U_full, 10]` int64, all ones, aligned 1:1 with the full test split like `eval_split.parquet` |
| `synth_filter.json` | rates, draw, seed, `n_items`, `n_queries`, achieved `pass_counts` / `pass_rates` per rate |

All ten columns are always written, and the harness refuses attrs whose `synth_filter.json` names another rate list (`layout.check_synth_rates`: the YAMLs index columns, so attrs built before SYNTH-TRIM's ten rates must be rebuilt). The YAMLs name every column (`p0001` … `p1`, `p005` = 0.05, `p02` = 0.2, `p05` = 0.5); the `synth` suite picks the rates per dataset:
[`goodreads-synth.yaml`](../../evaluation/config/goodreads-synth.yaml),
[`arxiv-synth.yaml`](../../evaluation/config/arxiv-synth.yaml) (the suite runs nine:
0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0);
[`laion30m-synth.yaml`](../../evaluation/config/laion30m-synth.yaml) runs nine (0.001 … 1.0 without 0.3, C1's points; SilverTorch at 0.1 / 0.2 / 0.5 / 1;
clause only) in its own `laion30m-synth` suite. Each keeps the parent's `data_dir`, encoder and `users_limit`, so
it shares the parent's encode cache. Its sweeps are named `p0001` … `p1`,
so they cannot collide with the parent's names in the shared `gt_d{dim}`,
and each oracle blob's name also carries the attrs digest. The builder
never touches `item_attrs_narrow.pt`, whose digest keys every existing
oracle and record. With one value per clause a bloom filter has almost no
false positives here, so synth bloom cells measure only what the bloom
path costs as the pass rate changes. Measured pass counts per rate:
[validation](../../research/validation.md#harness-gates). laion30m's files are built
the same way (seed 20261008, 30 M items, 10,000 queries); `laion1m`,
`laion3m` and `laion10m` do not rebuild them: `eval-data subset` copies the
parent's `item_attrs_synth.pt` and `synth_u.pt` rows ([Subsets](#subsets)),
so each item keeps its pass / fail at every rate, every draw the parent has,
and writes their `config/laion{1,3,10}m-synth*.yaml`. A draw built on the
parent after the subset is copied on with `eval-data subset laion30m laion10m
--n-items 10000000 --synth-only` (`subset.sync_synth`: the rows are re-derived
from the seed and must match the subset's recorded `rows_sha256`; the manifest
is rewritten). The files are not on the Hub.

**Cluster-correlated variant.** `eval-data synth-filter --dataset arxiv
--correlated [--rates 0.01,0.03,0.1] [--dim 128] [--draw 0]`
(`build_correlated`, seed `20261008 + draw`, draw `d > 0` files suffixed
`_s<d>` as above) is the uniform variant's opposite: IVF's best case,
where the pass set sits around the query. For each rate `p` it runs a
spherical k-means (`etl/synth_arxiv.spherical_kmeans`, random init from the
seed, at most 25 Lloyd iterations, CPU) with `round(1/p)` centroids over the
item embeddings the harness loads from the dataset's `content_dir` at
`--dim`. Items are then re-assigned to the final centroids. An item's value
is its cluster id, and a query's value is the id of its nearest centroid by
inner product, the same rule, computed from the full test split's
`query_emb.pt`. A query's pass rate is its cluster's size over N: about `p`
on average, varying per query, and not nested across rates. It writes three
files beside the uniform ones and touches no existing file:

| file | content |
|---|---|
| `item_attrs_corr.pt` | `[N, R, 1]` int64 cluster ids, clause `j` = rate `j` |
| `query_attrs_corr.pt` | `[U_full, R]` int64 nearest-centroid ids, aligned like `query_attrs_synth.pt` |
| `synth_corr.json` | rates, seed, k-means settings, `content_dir`, `n_items`, `n_queries`; per rate `n_lists`, the centroids, cluster-size stats and per-query pass-rate stats (mean, std, min, median, max) |

[`arxiv-corr-synth.yaml`](../../evaluation/config/arxiv-corr-synth.yaml)
reads them at d128 only (the clusters are d128's) with sweeps `c001`,
`c003` and `c01`. Achieved pass rates:
[validation](../../research/validation.md#harness-gates).

### On-disk layout the harness expects

Sequential datasets:

```
data/<dataset>/<variant>/
├── item_id_map.json
├── train.parquet / val.parquet / test.parquet
└── (checkpoints live in the checkpoint dir, not here)
```

Text datasets:

```
data/arxiv-papers/                       # config/arxiv.yaml: data_dir
├── item_id_map.json
├── papers.parquet
├── heldout.parquet
└── content{,_d128,_d64}/                # content_dir per dim: {256: content, 128: content_d128, 64: content_d64}
    ├── text_emb.pt + text_emb.meta.json
    └── query_emb.pt + query_emb.meta.json
```

`pubmed` is the same shape with one content dir
(`content_d768`) whose item matrix is **sharded** — `shard_index.json` +
`text_emb_shard_*.pt`, the synth-arxiv layout — instead of one
`text_emb.pt`; `load_text_items` and `validate_layout` take either.

Filter sweeps additionally need:

```
├── item_attrs_narrow.pt
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
└── ... per-clause vocab JSONs
```

plus, for a `-synth` config, the four `synth-filter` files, or for
`arxiv-corr-synth` the three `--correlated` ones ([above](#synthetic-selectivity-attrs)).

## HuggingFace I/O

[`hub.py`](../../evaluation/eval_datasets/hub.py) holds the repo
registry and the fetch/publish helpers, so no path or repo id is
hard-coded in the ETL modules:

```python
EVAL_REPOS = {
    "arxiv-papers":      "pinkmeme/eval-arxiv-papers",
    "yambda-500m":       "pinkmeme/eval-yambda-500m",
    "goodreads-work-id": "pinkmeme/eval-goodreads-work-id",
    "pubmed":            "pinkmeme/eval-pubmed",
    "laion30m":          "pinkmeme/eval-laion30m",
}
```

What may be
published from each repo is a release decision for the user
([roadmap](../../research/roadmap.md#needs-the-user)).

`pinkmeme/eval-pubmed` (private, revision `10d48d05`, 17.2 GB) holds the dot-product 10 M slice
(`pubmed-medcpt-dot`, [pubmed](#pubmed)): `content_d768/`, attrs, `eval_split.parquet`, `heldout.parquet`,
`item_id_map.json` and vocabularies, no titles, abstracts or ETL intermediates (published from a directory of
hardlinks to exactly those files). `eval-data fetch pubmed --target data/pubmed-medcpt-dot` stages it. `pinkmeme/eval-laion30m` (private) is to hold the
staged laion30m (vectors, attrs, vocab; the captions and urls stay under `_raw/`); not yet published. The old A100
`checkpoints/gsasrec-d128-shared` was deleted from it to free space (user decision).

**Trainer inputs** sit under `trainer/` in the private eval repos, uploaded with
`upload_folder` ([script](../../research/artifacts/seqrec-encoder/hub-upload/trainer_upload.py);
`hub.py` has no helper for them). The trainer's `data_dir` needs
`train/val/test.parquet` and `item_id_map.json` in one directory:

- `pinkmeme/eval-goodreads-work-id` `trainer/`: all of `data/goodreads-work-id/trainer/`
  (`train/val/test.parquet`, `item_id_map.json`, `prep_log.json`, `book_to_work.parquet`);
- `pinkmeme/eval-yambda-500m` `trainer/`: all of `data/yambda-500m/trainer/`
  (`train/val/test.parquet`, `item_id_map.json`, the re-prep with `timestamps`).

```bash
hf download pinkmeme/eval-goodreads-work-id --repo-type dataset \
    --include 'trainer/*' --local-dir data/goodreads-work-id
```

`eval-data fetch` pulls a prepared dataset (optionally a subset of dims),
`eval-data publish` pushes one, `eval-data publish-checkpoint` pushes a
trained checkpoint under `<repo>/checkpoints/<ckpt-id>/` (`train
upload-checkpoint` is the same upload with an `all` selector). Downloading
a prepared dataset is the fast path — the full arxiv encode is hours of GPU
time.

## Training — `evaluation/training/`

One sequence `Encoder` (the gSASRec body), two losses (gBCE and sampled
softmax with logQ, the default), plus the history → query-vector encoder the
harness uses at eval time. It exists to produce the embeddings
the sequential benchmarks need. It imports `eval_datasets.{hub,layout}` and
nothing from `bench`; the console script is `train`.

| file | role |
|---|---|
| [`config.py`](../../evaluation/training/config.py) | `TrainConfig` dataclass (defaults = the E1c recipe) + `save` / `load` (unknown keys dropped; no `loss` key means `gbce`) / `num_items` |
| [`model.py`](../../evaluation/training/model.py) | `Encoder`, `SASRecBlock`, `build_encoder(cfg, num_items)` |
| [`dataset.py`](../../evaluation/training/dataset.py) | `load_sequences` (the train split as left-padded `[U, L+1]` tensors on the device), `load_val_transitions` (val rows for `train_on_val`), `target_mask` (the trained target positions), `train_batches` (`randperm` slices) |
| [`losses.py`](../../evaluation/training/losses.py) | `gbce_loss`, `sampled_softmax_loss` |
| [`evaluate.py`](../../evaluation/training/evaluate.py) | chunked full-catalog scoring with its **own** recall / ndcg (`hits_at`, `recall_at_k`, `ndcg_at_k`): a checkpoint's reported quality must not move with the harness's metric code; `tests/training/test_encode.py` pins them to `bench.metrics` at 1e-9 |
| [`encode.py`](../../evaluation/training/encode.py) | `load_model_for_eval`, `encode_queries`, `encode_split` — the eval-time encode of the test split with its cache (`<ckpt-dir>/encoded_queries_v2.pt`, keyed on ckpt mtime + `max_seq_length`, the full split); what `bench.inputs.load_inputs` calls on a `checkpoint` dataset |
| [`train.py`](../../evaluation/training/train.py) | `train()`, `step_loss()`, `target_frequencies()` + the `train run` command |
| [`checkpoints.py`](../../evaluation/training/checkpoints.py) | `train upload-checkpoint`: push a checkpoint dir (or all) to HF through `hub.upload_checkpoint` |
| [`cli.py`](../../evaluation/training/cli.py) | the `train` group |

### Running a training job

```bash
uv run --directory evaluation train run \
    data_dir=/data/yambda-500m/trainer checkpoint_dir=/scratch/ckpt/yambda-d64
uv run --directory evaluation train run data_dir=/data/yambda-500m/trainer \
    checkpoint_dir=/scratch/ckpt/yambda-d128 embedding_dim=128 ffn_hidden_dim=512
uv run --directory evaluation train run data_dir=/data/yambda-500m/trainer loss=gbce num_negatives=256 warmup_steps=0
```

The defaults are the E1c recipe
([artifact](../../research/artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/config.json)):
d64, 2 blocks, 2 heads, ffn 256, dropout 0.5; `loss=sampled_softmax` with
`normalize`, `temperature=0.05`, `num_negatives=8192`,
`inbatch_negatives=4096`, `logq`; lr 1e-3, batch 256, L 200, warmup 1000,
`compile`; 100 epochs, eval every 2, early stop on `ndcg@10` with patience
10. The published gSASRec recipe (Gate B) is `loss=gbce num_negatives=256
warmup_steps=0`.

Arguments are `TrainConfig` `FIELD=VALUE` pairs, the value parsed as JSON
when it parses and taken as a string otherwise; an unknown field is a usage
error. `--config <json>` starts from a saved config instead of the defaults
and the pairs override it (`TrainConfig.load(path, **overrides)`); a pair
that changes `loss` drops the saved `normalize` and `logq`, so they resolve
for the new loss unless given as pairs too. `--resume` picks up from
`<ckpt_dir>/_resume.pt`, holding model, optimizer, scheduler, epoch, best
metric and the Python / numpy / torch / CUDA RNG states, so a resumed run is
reproducible, not merely restarted.

`resume_every` (default 1) is the number of epochs between `_resume.pt`
writes (`resume_due`); the last epoch, including an early stop, always
writes it, so a finished run stays resumable. The file is model + both AdamW
moments, three times `best_model.pt`: 49.2 GB and ~45 s per write with
two 32 M × 64 item tables, about a fifth of a ~210 s epoch. The
trade-off: a crash loses up to `resume_every − 1` finished epochs, which
`--resume` retrains. A superseded best snapshot is deleted at once unless
the current `_resume.pt` names it; that one goes at the next `_resume.pt`
write, so resuming always finds its best, looked up by file name in
`checkpoint_dir`. On `--resume`, every other snapshot (from epochs about to
be retrained, or all of them when no `_resume.pt` was written yet) is
deleted. Disk peak: one extra best snapshot.

`train_on_val=true` is the final fit, run after `num_epochs` has been chosen
on val by an ordinary run. It adds one training row per `val.parquet` row:
the last `max_seq_length + 1` items of `item_ids ++ targets`, with the val
day's clicks in time order (`load_val_transitions`). Only the target
positions whose target is a val-day item are trained (`target_mask`, from
the per-row `first`), because the history transitions are already train
rows. A row with more than `max_seq_length` targets keeps only its last
`max_seq_length`. The logQ frequencies count these positions too. The
run evaluates nothing on val and never stops early: it trains exactly
`num_epochs`, and the last epoch is the model saved as `best_model.pt` and
scored on test (`eval_quality.json`; `best_val_metric` is `{}`). `patience`,
`eval_every` and `early_stop_metric` then have nothing to act on.

### The encoder

`Encoder(items [B, L]) -> [B, L, D]`: `item_embedding [N+1, D]` (row 0 is
padding) `+ position_embedding` → dropout → `blocks` (`num_blocks` ×
`SASRecBlock`) → `final_norm` (LayerNorm). `SASRecBlock` is
`nn.TransformerEncoderLayer` (gelu, `norm_first`, `batch_first`), the
retired `GSASRec` body, so its state dicts load with `encoder.layers.N.`
renamed to `blocks.N.`. `predict_last` takes position −1 (sequences are
left-padded). `scoring_table()` is the `[N+1, D]` output table queries are
scored against; with `normalize` both it and `predict_last` return unit
vectors, so a dot product of the stored vectors is the cosine the loss
trained on. The trainer reads only `item_ids` (and `targets` at eval); a
`timestamps` column in the parquet is ignored.

The attention mask is built once per forward: `causal & (key_valid | eye)`,
`True` = attend. Real positions see exactly their real causal prefix; a
left-padding row sees only itself, so no row is empty and nothing
softmaxes to NaN (`tests/training/test_encoder.py`).

### The losses

`loss` picks one of two; `sampled_softmax` with logQ is the default.
Negatives are uniform over `1..N`, drawn on the GPU inside the step.

- `gbce` — gBCE as gSASRec publishes it (the published checkpoints' loss): `num_negatives` negatives **per
  position** (`[P, K]`, a `[P, K, D]` gather), the positive logit through
  the float64 calibration transform (`gbce_t`, `alpha = K / (N − 1)`), BCE
  over positive + K negatives. The `pow(−beta)` and `1/(x−1)` steps lose
  the positive class in fp32, hence float64. One `[K]` vector shared by the
  batch does not train: only K table rows get a negative gradient per step,
  and on yambda-500m d64 the user queries collapse onto a few popular items
  ([validation](../../research/validation.md#seqrec-encoder)).
- `sampled_softmax` — cross-entropy of the positive against one candidate
  vector shared by the batch: `inbatch_negatives` positives of the batch (a
  random subset) plus `num_negatives` uniform ids, at `temperature` (default 0.05), in fp32 outside
  autocast. A candidate equal to the row's own positive is masked to
  `−inf`. `normalize` (default: on for this loss, off for gbce; recorded in
  `config.json`) L2-normalizes queries and items first. `TrainConfig` rejects
  `normalize=true` with `gbce` and any other `loss` value.
  `logq` (default: on for this loss, off for gbce; `TrainConfig` rejects
  it with `gbce`) subtracts
  `log q_j` from every candidate column after the temperature scaling,
  where `q_j = M·p_train(j) + K/N` is the expected number of times item j
  is drawn among the M + K candidates (M the in-batch candidates actually
  used, K = `num_negatives`, N the item count; `logq_correction`, float64
  until the log). `p_train` is each item's share of
  the trained target positions (`target_frequencies` over `target_mask`,
  one GPU bincount at startup). The positive column is not corrected (arXiv 2507.09331);
  accidental hits stay `−inf`. With M = 0, `q_j = K/N` for every
  candidate, so each moves by `+log(N/K)` against the uncorrected positive:
  the plain uniform sampled-softmax correction.

### Loop shape

Per epoch: `randperm` over the device-resident train tensors → bf16
autocast forward → loss → grad-clip at 1.0 over all parameters → fused
AdamW → linear warmup over `warmup_steps` (0 = none, default 1000). `compile=true` wraps
`Encoder.body` (blocks + final norm) in `torch.compile`; the
embedding lookup and the loss stay eager. Every `eval_every` epochs it
evaluates on `val.parquet` by scoring the full catalog in
`[B, eval_score_chunk]` chunks and merging top-k across chunks, optionally
masking history. Checkpoints on improvement in `early_stop_metric`, stops
after `patience` evaluations without one. With `train_on_val` there is no
val eval, no snapshot and no early stop.

TF32 is **enabled** here (`_enable_tf32`) — the opposite of the
benchmark harness, which pins it off for measurement determinism.

### What a finished run leaves behind

```
<checkpoint_dir>/
├── best_model.pt          # reloaded, then re-saved from the best epoch (train_on_val: the last epoch)
├── config.json            # the TrainConfig — the eval loader reads this
├── item_embs.pt           # scoring_table(), row 0 zeroed
├── item_id_map.json       # copied from data_dir
├── item_attrs.parquet     # copied from data_dir when present
├── eval_quality.json      # test-split metrics
├── train_metrics.json     # losses, val curve, test metrics, epoch_time_sec, samples_per_sec, gpu_name, sm_mhz, peak_gpu_mem_bytes
├── sasrec-ep{N}-{metric}{v}.pt   # the current best-epoch snapshot (plus the one _resume.pt names, until its next write)
└── _resume.pt
```

`samples_per_sec` counts training sequences over the training part of the
epochs (eval excluded); `sm_mhz` is one SM clock sample per epoch taken
mid-epoch under load (clocks cannot be locked on this box), from the
training GPU: the first `CUDA_VISIBLE_DEVICES` entry, `0` when unset.

`config.json` is what makes a checkpoint self-describing at eval time.
A checkpoint without it falls back to
`D128_DROP05_DEFAULTS` in
[`encode.py`](../../evaluation/training/encode.py) — see
[checkpoints.md](checkpoints.md) for which runs those are.

## Adding a dataset

### The generic ingest (`eval-data ingest`)

When a dataset is vectors plus attributes (the text shape), no ETL module is needed:
[`ingest.py`](../../evaluation/eval_datasets/ingest.py) stages it, writes its
`config/<name>.yaml` and runs `validate_layout`. The user-facing walkthrough is
[`retrieve/docs/adding-a-dataset.md`](../../retrieve/docs/adding-a-dataset.md).

```bash
uv run --directory evaluation eval-data ingest mydata \
  --items items_000.npy --items items_001.npy \
  --item-attrs item_attrs.parquet --queries queries.npy --query-attrs query_attrs.parquet \
  --clause site --clause year --clause other_site=site! \
  --sweep c0_site=site --sweep site_year=site,year
```

| input | shape | what the ingest does |
|---|---|---|
| `--items` (repeat) | `.npy [n_i, D]`, fp16 or fp32, concatenated in order | L2-normalised in fp32, stored fp16 as `content_d<D>/text_emb_shard_NNN.pt` (`--shard-rows`, 1 M) + `shard_index.json` |
| `--item-attrs` | parquet, row `i` = item `i`; any scalar or list column | each column dictionary-coded, codes by count descending (ties by value), null `-1` → `item_attrs_narrow.pt [N, C, A]`, `A` = the longest list; `vocab.json` |
| `--queries` | `.npy [U, D]` | `query_emb.pt`, normalised fp16 |
| `--query-attrs` | parquet with the same columns, scalars | coded with the items' vocabulary; a value no item has is `-1` (that clause is off for that query) → `eval_split.parquet` |
| `--targets` (optional) | `.npy [U]`, 0-indexed item rows | `heldout.parquet`. Without it, each query's exact unfiltered top-1 item (fp32, TF32 off, ties to the lowest id; `--device cuda` for large catalogs) |
| `--clause` (repeat, in order) | `NAME[=COLUMN][!]` | `!` = reverse; several clauses may read one column → `clause_is_reverse_narrow.pt` |
| `--sweep` (repeat) | `NAME=CLAUSE,CLAUSE` | the yaml's `filters.clause`; `filters.bloom` = the sweeps with no reverse clause |
| `--doc-prefix` / `--query-prefix` | the encoder's prefixes | written to the meta sidecars; default `null` (no prefix concept). The prefix policy above applies |

`prep_log.json["ingest"]` records the counts, vocab sizes, the share of queries with each
clause live, and each clause's mean pass rate over the first 100 queries. Then add the
dataset to a suite in `config/suites.yaml`, and `EVAL_REPOS` in `hub.py` to publish it.
laion30m is the first user: its ETL does only what is LAION-specific (download, dedup and
split, domains and buckets, caption encode), then calls `ingest`.

### A dataset that needs its own ETL

For the sequential shape, or when the source needs work before it is vectors + tables:

1. Write `eval_datasets/etl/<name>.py` with `download` / `convert` / `prep`
   subcommands emitting the layout above (`main(argv)` over an `argparse`
   parser) and add it to `ETL` in
   [`eval_datasets/cli.py`](../../evaluation/eval_datasets/cli.py); give it
   a fixture test under `tests/eval_datasets/` that runs
   `layout.validate_layout` on what it wrote.
2. If it has attributes, add an `attrs` subcommand producing
   `item_attrs_narrow.pt`, `clause_is_reverse_narrow.pt`, the vocab
   JSONs, and `eval_split.parquet` aligned 1:1 with `test.parquet`.
3. Add an entry to `EVAL_REPOS` in `hub.py`.
4. Add one `evaluation/config/<name>.yaml` (one YAML per
   dataset, see [evaluation.md](evaluation.md#config-one-yaml-per-dataset--suitesyaml))
   and list the dataset in the suites it belongs to in
   `evaluation/config/suites.yaml`. Set `checkpoint` for the sequential
   shape, `content_dir` for the text shape.
5. Train a checkpoint if sequential; otherwise encode embeddings into
   `content*/` with meta sidecars.

Reuse `common.py` for attribute synthesis (`synthesize_qa_narrow`,
`sample_rare_biased_wide`) and id-hash slicing (`select_pmids`) rather than
re-deriving the sampling — the rare-biased draw in particular is what
keeps filter selectivity in an interesting range.
