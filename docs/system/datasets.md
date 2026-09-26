---
title: datasets
created: 2026-09-26
updated: 2026-09-26
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
| **sequential** | yambda-500m, yambda-5b, goodreads, kuairand | a trained SASRec checkpoint, encoded at eval time | goodreads, kuairand |
| **text** | arxiv, arxiv-synth, yfcc10m, pubmed, openalex | pre-encoded embeddings on disk | yes |

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
| [`etl/yfcc.py`](../../evaluation/eval_datasets/etl/yfcc.py) | `yfcc` | `dl.fbaipublicfiles.com` (NeurIPS'23 Big-ANN filtered track) |
| [`etl/yfcc_check_gt.py`](../../evaluation/eval_datasets/etl/yfcc_check_gt.py) | `yfcc-check-gt` | — (validates the shipped GT) |
| [`etl/pubmed.py`](../../evaluation/eval_datasets/etl/pubmed.py) | `pubmed` | NCBI FTP MedCPT embeddings + MEDLINE baseline (~36M articles) |
| [`etl/kuairand.py`](../../evaluation/eval_datasets/etl/kuairand.py) | `kuairand` | Zenodo KuaiRand-27K + category supplement (32M videos) |
| [`etl/openalex.py`](../../evaluation/eval_datasets/etl/openalex.py) | `openalex` | OpenAlex snapshot on public S3, parquet copy (476M works, streamed) |
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
→ fp32 + L2-normalise), `load_query_attrs` (row count checked against the
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
concept — YFCC's CLIP descriptors, MedCPT's precomputed vectors. A sidecar
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
breakage flagged); the fixture writers at the top of
[`test_yfcc.py`](../../evaluation/tests/eval_datasets/test_yfcc.py)
(`write_u8bin`, `write_knn_result`, `write_spmat`) emit the upstream
binary formats into `tmp_path`, so the parsers are tested against bytes
rather than against a downloaded file. The last class, `TestRealSlice`,
round-trips a 1,000-item slice of the *real* prepared dataset against the
uncapped tag CSR and skips itself when `data_root()/yfcc10m` is not on the
machine; its query-predicate test also reads the raw
`_raw/yfcc10m/query.metadata.public.100K.spmat` and skips without it. Reuse those writers when adding the E2–E4 loaders,
and give each new loader a fixture test that runs `validate_layout` on
what it wrote. `test_openalex.py` builds its rows in the snapshot's projected
parquet schema (nested `primary_topic.field.id` etc.) and runs `prep` →
`attrs` → `validate_layout` on a 60-paper citation-connected staging dir.

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

Yambda is out of the study ([decisions](../decisions.md#datasets)); its
ETL, configs and checkpoints stay.

`uv run eval-data yambda prep --variant {50m,500m,5b} --output-dir data/yambda/<v>`

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
([validation](../validation.md#trainer-inputs-with-timestamps-data-gates-not-citable)).

> `prep` deliberately passes `drop_non_train_items=False`, mirroring
> `yambda.preprocess`. Setting it `True` makes polars re-evaluate an
> imploded set per row and blows the job past a 129 GB cgroup limit.

**`attrs`** builds the filter-bench tensors from `prep`'s outputs plus
the catalog parquets:

```
item_attrs_narrow.pt        [N, 5, 4] int64   (genre, lang, format, year, author)
item_attrs_wide.pt          [N, 1, 32] int64  (shelves — see "dead artifacts")
clause_is_reverse_narrow.pt [5] bool = [F, T, F, F, F]
lang_vocab.json / format_vocab.json / author_vocab.json / wide_shelf_vocab.json
wide_shelf_global_freq.pt   [V_wide] int64
eval_split.parquet          aligned 1:1 with test.parquet rows
```

Clause 1 (language) is the **reverse** clause — the one sweep that
exercises negated predicates end-to-end. It is bloom-incompatible, so
goodreads bloom sweeps exclude it.

### arxiv

`download` → `convert` → `prep` → `encode_text` → `encode_queries` →
`attrs`, or `uv run eval-data arxiv all --output-dir data/arxiv-papers` (the
`data_dir` of [`config/arxiv.yaml`](../../evaluation/config/arxiv.yaml)).

Arxiv has **no user sequences**, so `prep` does no interactions and no
time-split — it emits `item_id_map.json`, `papers.parquet`, and a
sampled `heldout.parquet`. Held-out items stay in the index.

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

### yfcc10m

`download` → `convert` → `prep` → `attrs`, or
`uv run eval-data yfcc all --output-dir data/yfcc10m`.

The NeurIPS'23 Big-ANN **filtered-search** track set: 10M CLIP image
descriptors, 192-d uint8, plus a bag of tags per image drawn from a
200,386-word vocabulary (description words, camera model, year, country),
plus 100,000 queries that each carry 1–2 tags. Six files, 2.97 GB, no
registration, from
`https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/`
(exact names and sizes in `etl/yfcc.py`'s `RAW_FILES`). `download` is size-verified and resumable;
`convert` re-parses every header and writes
`data/_raw/yfcc10m/processed/manifest.json` with the sha256 of each file.

**This is the one dataset whose filtered ground truth is not ours.** For
every dataset above, `bench/oracle.py` computes the filtered
top-K itself. Here the organisers ship `GT.public.ibin`: per query, the
10 nearest base vectors *by squared L2* among the items whose tag bag
contains **every** query tag (conjunctive AND). `prep` stores it verbatim
as `gt_shipped.pt`, together with the shipped 100-deep unfiltered GT:

```python
{"format": "yfcc-shipped-gt-v1",
 "ids": int64 [100000, 10],   "dists": float32 [100000, 10],   # filtered
 "unfiltered_ids": int64 [100000, 100], "unfiltered_dists": float32 [...],
 "k": 10, "unfiltered_k": 100,
 "metric": "squared_l2",
 "id_space": "0-indexed base row == item_id - 1 == item_embs row",
 "predicate": "item tag bag contains every query tag (conjunctive AND)",
 "source": ".../GT.public.ibin", "provenance": "shipped by the organisers"}
```

The current harness has **no precomputed-oracle input** — it always
builds its own — so nothing reads `gt_shipped.pt` at sweep time. It is
consumed by
[`etl/yfcc_check_gt.py`](../../evaluation/eval_datasets/etl/yfcc_check_gt.py),
and it is the format a harness input for shipped ground truth would read
(not built).

**Three things about this dataset differ from the others**, all forced by
the upstream data:

1. **The metric is Euclidean, not inner product.** The shipped GT ranks
   by squared L2 over raw uint8 vectors; the harness's text path
   L2-normalises and scores inner product, i.e. cosine. Base-vector norms
   have a coefficient of variation of 1.1 % (`prep_log.json` →
   `prep.base_norm`), so the two orders are close but not equal: on a
   100-query sample, an exact *cosine* filtered top-10 has mean recall
   0.951 against the shipped squared-L2 GT and reproduces it exactly on
   59 % of queries (`yfcc_check_gt --metric ip`). Harness recall on this
   dataset is therefore measured against the harness's own cosine oracle;
   agreement with the shipped GT is checked separately.
2. **fp16 is lossless here.** uint8 values 0..255 are exact in fp16, and
   a 192-term squared-L2 sum over them stays below 2²⁴, so fp32 arithmetic
   reproduces the shipped integer distances bit-for-bit. `content_d192/`
   holds fp16 only; a separate int8 code file would be a redundant copy of
   the same integers, and SilverTorch quantises internally at build time.
   The sidecars `text_emb.meta.json` / `query_emb.meta.json` carry
   `"prefix": null` — the declared "no prefix concept" of the prefix policy
   above — plus the provenance (source URL, raw dtype, both metrics, the
   base-norm statistics); `bench check --dataset yfcc10m` is clean.
3. **The narrow clause tensor is a capped approximation of the tag
   predicate** — see below.

**`attrs` and the tag cap.** `ExactAttributeFilter` matches a clause when
the query's value for it appears anywhere in that clause's `A_max` slots,
and ANDs the clauses. So the tag predicate maps onto **two clauses that
both hold the item's tag bag**: the query's first tag goes in clause 0,
its second in clause 1 (`-1`, "always pass", for the 61,626 single-tag
queries). What does not fit is the bag itself: items carry 10.8 tags on
average with a 1,517-tag tail, and `[10M, 2, 1517]` int64 is 243 GB.
`attrs` therefore

- drops every tag no query ever asks for — 192,476 of 200,386 tags, which
  removes 29 % of the tag entries and cannot change any answer, then
- keeps the `--max-tags` (default 32) most query-frequent of what remains
  and pads with `-1`.

Capping is **subtractive only**: the capped predicate passes a subset of
what the true predicate passes, never a superset. At K=32 that leaves
98.51 % of items uncapped, and 74.21 % of the 100k queries keep their
entire shipped-GT row (84.86 % of individual GT entries survive) — the
numbers land in `prep_log.json` → `attrs.gt_fidelity`, recomputed on every
`attrs` run. The harness builds its oracle from these same capped attrs,
so its recall stays internally exact; what the cap changes is *which*
filter is being benchmarked, not whether the measurement is right.

The **uncapped** bags are shipped as `item_tags_csr.pt` (CSR: `indptr`
int64 `[N+1]`, `indices` int32 `[nnz]`, upstream tag ids), 513 MB. That
is the authoritative attribute source: the gate check reads it, and a
future sparse-set filter could too.

```
data/yfcc10m/
├── item_id_map.json            identity map, 10M entries (row i → i+1)
├── heldout.parquet             query_row, item_id (unfiltered-GT rank 1), n_query_tags
├── content_d192/
│   ├── text_emb.pt             [10M, 192] fp16   (uint8 values, lossless)
│   ├── query_emb.pt            [100k, 192] fp16
│   ├── text_emb.meta.json      prefix: null + provenance
│   └── query_emb.meta.json     prefix: null + provenance
├── item_tags_csr.pt            full uncapped tag bags, CSR
├── item_attrs_narrow.pt        [10M, 2, 32] int64 — dense tag ids, -1 pad
├── clause_is_reverse_narrow.pt [2] bool = [F, F]  (no negated predicate here)
├── tag_vocab.json              7,910 queried tags: upstream id, query freq, doc freq
├── eval_split.parquet          target_id, query_attrs_narrow [2], n_query_tags
├── gt_shipped.pt               the organisers' filtered + unfiltered GT
└── prep_log.json
```

`eval_split.parquet` has no `query_attrs_wide_*` columns — YFCC has one
attribute (tags) and no wide-bloom bag, so the dead artifacts below do
not exist for it.

**Validating the shipped GT** (the dataset's staging gate; its state is
in [validation](../validation.md#datasets)).

```bash
uv run --directory evaluation eval-data yfcc-check-gt \
    --data-dir $RETRIEVE_DATA_ROOT/yfcc10m --device cuda \
    --report $RETRIEVE_DATA_ROOT/yfcc10m/gt_check.json
```

It rebuilds the exact filtered oracle — conjunctive AND over the uncapped
CSR, squared L2 in fp32 with TF32 pinned off — and compares to
`gt_shipped.pt` id-by-id, falling back to an exact distance-vector
comparison where equal distances make the ordering ambiguous. Exit 0 iff
every query is reproduced. `--limit N` checks a random subset (`--device
cpu --limit 200` is a ~2-minute sanity run), `--tags narrow` measures the
cap instead of the true predicate, and `--metric ip` reports the cosine
drift; the last two are diagnostics and always exit 0.

**Running it.** [`config/yfcc10m.yaml`](../../evaluation/config/yfcc10m.yaml)
is the dataset file (one dim, 192; one clause sweep `tags_and`
= clauses `[0, 1]`; no bloom block, because a bloom's false positives would
make the cross-check against the shipped GT meaningless), and the
dataset is listed in the `filter` suite of
[`config/suites.yaml`](../../evaluation/config/suites.yaml) with 192 added
to the suite's dims. So

```bash
uv run eval-data yfcc all --output-dir $RETRIEVE_DATA_ROOT/yfcc10m   # 12.5 GB with the raw
uv run bench check --dataset yfcc10m                       # yfcc10m d192: ok
uv run bench run --dataset yfcc10m --suite filter          # every filter cell
```

is the whole path. Three things to keep in mind when reading its records:
the suite's `ks` (100 / 500 / 1000) are deeper than the shipped GT's
k = 10, which is fine because recall is measured against the harness's own
cosine oracle over the capped attrs (deviations 1 and 3 above); the
`none` cells are in no suite of `suites.yaml` (unfiltered cells come back
with roadmap E5); and **the
exact-algo cell fails the harness's quality gate on this dataset** — the
library's `PostfilterKNN` scores in fp16, whose 4.9 × 10⁻⁴ spacing is
coarser than YFCC's score density (a median 0.0072 cosine between rank 1
and rank 1000, plus 5 % exact-duplicate vectors), so `linr_v1_filter_mask`
reaches only `recall_oracle@1000 ≈ 0.96` against the fp32 oracle and
`QualityGateError` ends the run before the other algos. The oracle is right
(0.9998 against fp64); the module's fp16 is the cause; it is
backend-independent. What to do about it is an open decision on the
[roadmap](../roadmap.md#needs-the-user).

### pubmed

**Status: staged as the 10 M slice** (`--keep-items 10000000 --seed 0`,
roadmap E2) under `$RETRIEVE_DATA_ROOT/pubmed-medcpt`: `bench check`
passes, the `c0_mesh` oracle is built, and one exact filter cell runs.
SilverTorch cannot build at this size yet (see *Disk budget and the
slice*); state in [validation](../validation.md#datasets). Not on the Hub.
Nothing below is citable.

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
native dim ([decisions](../decisions.md#datasets): no PCA), so pubmed has
exactly one content dir, `content_d768`.
[`config/pubmed.yaml`](../../evaluation/config/pubmed.yaml) is the dataset
file, listed in the `filter` suite at 768.

#### The streaming `convert`

The raw mirror (163.7 GB) plus the fp16 item matrix (55.2 GB) plus the
MEDLINE baseline (53.9 GB) is 273 GB — more than the 300 GB overlay can
give one dataset next to the campaign's. So `convert` never holds the
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
   L2-normalise, cast to fp16 and `torch.save` them as
   `content_d768/text_emb_shard_{i:02d}.pt`; append the `{filename,
   start_id, n_rows}` entry to `shard_index.json`; `--delete-raw` the
   JSON and npy. A killed run resumes at the first shard whose `.pt` or
   parquet is missing.

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

#### Query sets

`queries` builds `queries.parquet` (`query_id`, `text`, `target_id`) from two
sources:

- `heldout` — item-as-query: a held-out article's title (from the article
  parquets) is the query and the article itself is the single relevant item.
  Always available. **Every held-out row keeps its query row** — an article
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
`content_d768/query_emb.pt`. MedCPT's query and article encoders are
*asymmetric* — the same load-bearing property as nomic's prefixes on arxiv — so
the unfiltered cross-check sweep is meaningful rather than an identity lookup.

#### Disk budget and the slice

Computed by `eval-data pubmed plan --medline --keep-items N` on 2026-09-16
from the server's own sizes (the JSON reports are in
[dataset-candidates-artifacts/pubmed/](../artifacts/dataset-candidates/pubmed/)),
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
perf pools need, it is the papers' 10 M pool size and YFCC's, and the
largest catalog the harness as written can take at 768-d is ~15 M. Going
above that is a harness decision (fp16 items + a chunked oracle), not an
ETL one.

**Measured on the staged slice (2026-09-26).** `convert` took 57 min for
all 38 shards, with `medline --stream` (21 min, 39,994,988 rows) running
beside it. On disk: 17 GB under `pubmed-medcpt/` (15 GB of fp16 shards,
1.5 GB of attrs, 0.7 GB of `staging/` article parquets, 0.1 GB of MEDLINE
parquet), plus 0.4 GB of PMID lists kept in `_raw/pubmed/`. The exact
`c0_mesh` oracle (k_gt 1000, 10,000 queries) builds in 23 s. The item
budget above does **not** cover SilverTorch's build.
`retrieve.indexing.quantize.quantize_int8_global_codes` computes
`(embs / abs_max * 127.0).round()` over the whole fp32 matrix. That makes
two more 28.6 GiB fp32 temporaries next to the 28.6 GiB of items, so
`register_index` OOMs on the 80 GB A100 on both the `triton` and
`official` backends. A chunked or in-place quantize in the library fixes
it, and `bench/` and the slice stay as they are. The exact
`linr_v1_filter_mask` cell does run
([artifacts/e2-pubmed/](../artifacts/e2-pubmed/)).

### kuairand

**Status: staged and layout-clean, no checkpoint.** `download` → `convert` →
`prep` → `attrs` have run on the real data, and `bench check --dataset
kuairand` is clean. The gSASRec checkpoint, the Hub publish and the filter
cell need the GPU and have not run (roadmap E4). Nothing below is citable.
The run records are in
[artifacts/e4-kuairand/](../artifacts/e4-kuairand/).

`uv run eval-data kuairand all --output-dir data/kuairand`, or the four
subcommands in turn. Sources, both CC BY 4.0 on Zenodo and md5-checked by
`download`:

- `KuaiRand-27K.tar.gz` (9.89 GB): four standard logs, one random-exposure
  log, `video_features_basic_27k.csv`, `user_features_27k.csv` and three
  video statistics CSVs;
- `kuairand_video_categories.csv` (3.69 GB, the 2026 supplement): a
  four-level category path per video, keyed on `final_video_id`, `-124`
  for an unknown level.

`convert` streams the tarball once, straight into ZSTD parquet under
`data/_raw/kuairand/processed/`. It unpacks nothing to disk and never
writes the 21.7 GB of statistics CSVs, which nothing reads.
`final_video_id` is the 27K `video_id`: the supplement has exactly one row
for each of the 32,038,725 ids `0..N-1`, and its level-1 category equals
the basic features' first `tag` on 64 % of videos, far above chance.

**`prep`.** The positives are `is_click == 1` and `is_rand == 0`.
`is_rand` is 0 on every standard-log row and 1 on every row of the random
log, so the filter drops exactly the 1,186,059 random exposures. Those
exposures come from a uniform intervention, not from the user's choice,
and KuaiRand ships them for unbiased evaluation. They are therefore not
training signal. The harness measures recall against its own oracle and
does not use them.

That leaves 122,052,542 clicks from 27,285 users, a median of 3,806 per
user.

- **Catalog.** The full catalog is the item space, whether or not a video
  was ever clicked. `item_id = video_id + 1`, and `item_id_map.json` is
  the identity map over all 32,038,725 videos, 12.06 M of them clicked in
  train.
- **Split.** `timesplit.sequential_split_train_val_test` runs on the local
  dates of the logs (Asia/Shanghai, a fixed UTC+8). Test is 2022-05-07 and
  05-08, val is 05-06, with 30-minute gaps.
- **`test.parquet`.** One row per user: the last 200 items of
  train ++ val as the history, and every test click as the targets. That
  is 26,221 rows with a median of 246 targets; `test_users.parquet` holds
  their `user_id`s.
- **`train.parquet`.** Each user's train history is cut from the end into
  non-overlapping 200-transition windows (`train_windows`), because the
  trainer reads only the last `max_seq_length + 1` items of a row. That
  gives 561,486 rows covering all 109.6 M train transitions, instead of the
  27k × 200 that one row per user would train.
- **`timestamps`.** Every train/val/test row carries `timestamps`, the unix
  seconds (`time_ms // 1000`) of each `item_ids` entry, cut into the same
  windows and tails; it covers the history, not `targets`.

**`attrs`.** It writes `item_attrs_narrow.pt` `[32,038,725, 7, 4]` int64
(7.2 GB), `clause_is_reverse_narrow.pt`, `attr_vocab.json` (every
vocabulary, the thresholds and the reference date) and
`eval_split.parquet`. It takes 4 min and peaks at 37 GB RSS. The
**two filter protocols** occupy disjoint clause slots, because the harness
reads one `query_attrs_narrow` per query and a sweep only chooses which
clauses are live:

| clause | attribute | item slots | query value | mean pass rate |
|---|---|---|---|---|
| C0 | level-1 category (38) | 1 | **target-derived** | 3.6 % |
| C1 | finer category, levels 4 → 3 → 2 in one space (801) | up to 3, deepest first | **target-derived** | 1.0 % |
| C2 | `tag` (58) | up to 4, upstream order | **target-derived** | 4.5 % |
| C3 | `upload_type` (38) | 1 | **target-derived** | 14.2 % |
| C4 | `video_type` — **reverse** | 1 | **business**: "no ads" | 99.9 % |
| C5 | duration ≤ {15, 30, 60, 180} s | every cap it fits under | **business**: the tightest cap at or above the user's median history duration | 52.9 % |
| C6 | uploaded ≤ {3, 7, 14, 30} days before 2022-05-07 | every window it fits | **business**: "within 7 days" | 25.8 % |

The pass rates are exact means over all 26,221 queries (`prep_log.json` →
`attrs.mean_pass_rate`).

- **Target-derived** values come from the held-out target's own
  attributes, through `common.synthesize_qa_narrow`.
- **Business-rule** values never look at the target. They are the rules a
  feed would apply: hide ads, cap the length at what this user watches, and
  keep it fresh.
- **Ranges as equality.** C5 and C6 encode range predicates as equality
  clauses: an item carries every threshold id it satisfies, so the query
  value `j` means "≤ threshold `j`". The clause kernels support equality
  only ([filtering](filtering.md#linr-paper-31--fixed-clause-schema-no-dsl)).
- **Reverse and bloom.** C4 is the reverse clause, so it is excluded from
  bloom sweeps.

What the data ruled out:

- **Upload month.** 90 % of videos were uploaded within 42 days of the
  test start, so a month bucket is near-constant. Recency relative to the
  test start replaces it.
- **Author.** There are 8.8 M authors and the median author has one video,
  so "not the target's author" would pass about 100 %.
- **`music_type`.** Six values, one of them 65 %.

The sweeps of [`config/kuairand.yaml`](../../evaluation/config/kuairand.yaml)
are `t_*` over C0–C3 and `b_*` over C4–C6. Bloom runs `t_cat1`, `t_tag`,
`b_short` and `b_fresh`. The dataset is in no suite yet (roadmap E5).

**Training** runs over the full 32 M table with `reuse_item_embeddings`,
the one item table the trainer already supports. The table is 32,038,726
× 128 fp32 = 16.4 GB. With its dense gradient and AdamW's two moments that
is 65.6 GB before activations on the 80 GB A100; two separate tables would
need 131 GB. That estimate is arithmetic, not a measurement. If it does not
fit, lower `batch_size`. Every epoch also scores
the full catalog for the val users, so `eval_max_users` bounds that
cost. The command, not yet run:

```bash
uv run --directory evaluation train run data_dir=data/kuairand \
    checkpoint_dir=data/kuairand/checkpoints/gsasrec-d128-shared \
    embedding_dim=128 dropout=0.5 reuse_item_embeddings=true eval_max_users=4096
```

```
data/kuairand/
├── item_id_map.json            identity over 32,038,725 videos (video_id v → v+1), 619 MB
├── train.parquet               item_ids, timestamps: 200-transition windows, 561,486 rows
├── val.parquet / test.parquet  item_ids (last 200), timestamps, targets
├── test_users.parquet          user_id of each test row
├── item_attrs_narrow.pt        [32,038,725, 7, 4] int64, -1 pad
├── clause_is_reverse_narrow.pt [7] bool = [F, F, F, F, T, F, F]
├── attr_vocab.json
├── eval_split.parquet          target_id, query_attrs_narrow [7]
└── prep_log.json
```

### openalex

**Status: staged at 10 M, `bench check` clean, one filter cell run.** Roadmap E3's
OpenAlex fallback (no Semantic Scholar key), a 10 M catalog like pubmed's. On 2026-09-26
`convert --sample-rate 0.35 --workers 64` streamed all 2,040 files of release 2026-09-23
(297.1 GB read in 572 s, 0 failures); a 15 M catalog was prepped and encoded
(`encode_text` 15,663 s on the A100, 958 docs/s), did not fit the harness (below), and was
cut to 10 M: `prep --keep-items 10000000` (374 s), `reshard --from-dir` the 15 M vectors
(34 s, no encode; all 10 M rows `torch.equal` to their source rows,
[`check_reshard.py`](../artifacts/e3-openalex/check_reshard.py)), `encode_queries`,
`attrs`. `verify_prep.py` re-derives every query with no mismatch, and
`bench check --dataset openalex` reports `openalex d768: ok`. The dataset is in the `filter`
suite. One cell, `linr_v1_filter_mask`/triton clause `field_era`, eager, `--skip-perf` (so
`partial`): pass rate 0.0526, `recall_oracle@1000` **0.9959**, held-out (a cited paper)
`recall@100` 0.505 / `@1000` 0.741, n = 10,000, 64 GB reserved
([record](../artifacts/e3-openalex/filter-openalex-d768-cell.jsonl)). SilverTorch-triton
and LiNR V2 / V3 cannot run at D = 768 (the power-of-two limit on the
[roadmap](../roadmap.md)); official and V1 can. Nothing here is citable yet.

The full-scan filter counts (`convert_log.jsonl`, summed): 476,196,327 rows → year
373,119,082 → English 261,736,236 → type 154,189,252 → flags 130,910,232 → hash sample
(0.35) 45,821,266 → abstract 29,944,334 staged, 11 of them dropped as `abstract_truncated`.
So **~85.6 M works are eligible**, not the 54.6 M `plan`'s row-group sample estimated: the
sample under-counts badly because row groups cluster by type. Use `plan` for the projected
bytes (exact, from every footer), not for the eligible count.

The 10 M prep: 29,944,334 staged → 10,000,000 items, 19,944,334 in the held-out pool;
7,658,677 pool papers cite the catalog, 4,890,174 have a same-field earlier-era reference,
10,000 drawn; 35,673 qrels. The attrs: field 99.85 % covered, source 51.6 %; the target
passes the query's field and era clauses for every query (subfield 75 %). The superseded
15 M run (`run15m-*` logs) held 4.77 in-catalog references per query, 2.85 same field +
earlier era.

The dry run (2026-09-26, artifacts in [e3-openalex/](../artifacts/e3-openalex/)):
`convert` on 8 real snapshot files (50,304 records) staged 4,581 works; a few thousand
rows barely cite each other, so [`api_citation_topup.py`](../artifacts/e3-openalex/api_citation_topup.py)
added the 1,060 eligible works that 60 pool papers really cite (OpenAlex API, same
`stage_table`). `prep --keep-items 5073` held out 75 papers with 994 in-catalog
references (405 same field, earlier era), `verify_prep.py` re-derived all of it with no
mismatch, `encode_text` / `encode_queries` ran on the A100 (a deleted shard re-encoded
bit-identically), `attrs` built the clauses, and `bench check` against that directory
reports `openalex d768: ok` (and flags a removed `eval_split.parquet`). Every harness
reader (`layout.load_text_items` / `load_text_queries` / `load_item_attrs` /
`load_query_attrs`) loads it.

`download` → `convert` → `prep` → `encode_text` → `encode_queries` → `attrs`, with
`plan` first; for a smaller catalog of the same staging, `reshard --from-dir <larger>`
replaces `encode_text` (the N smallest hashes are a subset of any larger N, so the rows are
copied, not encoded). Source: the OpenAlex quarterly snapshot on public S3, anonymous, CC0.
Verified 2026-09-26 against release **2026-09-23**: the 2026 layout is
`s3://openalex/data/{jsonl,parquet}/<entity>/updated_date=*/part_*` with a
`manifest.json` per format, written last (the old `data/works/` path is an S3 delete
marker; `legacy-data/` is frozen). `works` is **476,196,327 records in 2,040 files**:
659.0 GB of gzipped JSONL, **707.1 GB of parquet**. `download` pins that manifest in
`_raw/openalex/manifest.json`; every later step walks its files in (date, part) order.

**Why the parquet copy.** Column projection. `convert` reads 14 leaves (`id`,
`publication_year`, `language`, `type`, the three flags, `primary_topic.{field,subfield}.id`,
`primary_location.source.id`, `open_access.is_oa`, `title`, `abstract_inverted_index`,
`referenced_works`), which the footers of all 2,040 files put at **297.1 GB (42.0 %)**. The
abstract is a JSON *string* in the parquet copy (an object in JSONL and the API); only rows
that survive the cheap filters are JSON-parsed.

#### The streaming `convert`

The snapshot is never landed. `convert --workers W` runs one spawned process per file
(`stage_file`): row group by row group, read the projected columns over S3, filter
(`stage_table`), and write the survivors to `_raw/openalex/staging/<date>_<part>.parquet`
atomically. A killed run resumes at the first file without a staging parquet; the
parameters (release, year range, types, sample rate, seed) are pinned in
`staging/params.json` and a resume under different ones is refused. A file that fails with
an S3 / Arrow error is reported and skipped; rerunning picks it up.

The filters, in order, each counted into `convert_log.jsonl`:

1. `publication_year` in [2000, release year];
2. `language == "en"`;
3. `type` in `article, preprint, review, conference-paper, book-chapter, dissertation` —
   `dataset` is 29 % of English works and mostly boilerplate (every CCDC crystal-structure
   entry shares one abstract, i.e. exact-duplicate vectors);
4. not `is_paratext` / `is_retracted` / `is_xpac`;
5. the **hash sample**: `common.pmid_hash(work_id, seed) < sample_rate · 2⁶⁴`;
6. abstract present. The text is the inverted index's words in position order
   (`abstract_text`). The parquet copy caps strings just under 32,767 chars, which cuts
   about one index in 7 M mid-JSON (1 in 7,108,348 sampled, W4214716959); such a row is
   counted as `abstract_truncated` and dropped.

`plan` measures these rates on a seeded sample of row groups (weighted by records) and
turns `--keep-items` into the `--sample-rate` that stages `keep_items × (1 + margin)`
works. The parser was checked row-for-row against the API: the abstract rebuilt from the
parquet string equals the one rebuilt from the API object.

#### The slice, the queries, the relevance

`prep` keeps the **N smallest work-id hashes** of the staged rows (`common.select_pmids`,
so the catalog is the same set whatever the file order, and a smaller N is a subset of a
larger one); a work id staged twice keeps its newest copy. Item ids are 1-indexed dense in
(file, row) order; `papers.parquet` is written in item-id order, which `encode_text` and
`attrs` rely on (`attrs` checks it).

The staged works above rank N — the `margin` — are the **held-out pool**: never in the
catalog. A pool paper is a query candidate when at least one work it references is in the
catalog *with the same field and a strictly earlier era* (below). `--n-heldout` (10,000)
of them are drawn with the seed. Per query:

- `qrels.parquet` — every in-catalog reference (`query_row, item_id, passes_field_era`):
  the relevance signal, "the papers this paper cites";
- `heldout.parquet` — `item_id` = **the target**, one reference drawn with the seed among
  those passing field + era (the harness text path takes one target per query), plus the
  query's `work_id / year / era / field / subfield / source / is_oa` and its
  `n_relevant` / `n_relevant_field_era`;
- `queries.parquet` — its title and abstract, the query text.

Recall is measured against the harness's exact filtered oracle, as everywhere; the qrels
carry the citation relevance for the multi-target metrics the harness does not yet read.
[`verify_prep.py`](../artifacts/e3-openalex/verify_prep.py) re-derives every query's qrels,
flags and target from `item_id_map.json`, `papers.parquet` and the staged `refs`.

#### Attribute semantics

`item_attrs_narrow.pt` is `[N, 5, 4]`, the shape of the other text datasets:

| clause | attribute | cardinality | query value |
|---|---|---|---|
| C0 | `primary_topic.field` | 26 | the query's field |
| C1 | earlier era (below) | 5 eras | the query's era |
| C2 | `primary_topic.subfield` | ~250 | the query's subfield |
| C3 | `primary_location.source` | top `--source-vocab` (5,000) | the query's own venue — **reverse** |
| C4 | `open_access.is_oa` (null → 0) | 2 | `1` for every query |

`clause_is_reverse_narrow.pt` is `[F, F, F, T, F]`; C3 is bloom-incompatible and
excluded from the bloom sweeps of [`config/openalex.yaml`](../../evaluation/config/openalex.yaml).
The roadmap's predicate "year < query year, same field" is the `field_era` sweep, C0 ∧ C1.

**Why eras.** Clauses are equality-only
([filtering](filtering.md#linr-paper-31--fixed-clause-schema-no-dsl)), so an order
predicate is encoded as a set: an item of era *b* lists the eras *b+1 … 4* in its C1
slots, and a query sends its own era, which matches exactly the items of strictly earlier
eras (`earlier_era_slots`). Four slots allow five eras: 2000–09, 2010–14, 2015–18,
2019–21, ≥2022 (the sampled eligible works split 25 / 21 / 16 / 15 / 23 %). This is
**subtractive** against the exact year predicate — a reference from earlier in the
query's own era is excluded — so it benchmarks "published in an earlier era", and the
harness oracle is built from the same attrs, so recall stays exact for that filter.

#### Encoding

`nomic-embed-text-v1.5` at its native 768 dims (no truncation, [decisions](../decisions.md#datasets)),
`"search_document: "` / `"search_query: "` prefixes as on arxiv, text
`"{prefix}{title}. {abstract[:1500]}"`, 512 tokens, bf16 weights, fp16 L2-normalised
output. `encode_text` writes `content_d768/text_emb_shard_NNN.pt` of `--shard-rows`
(1 M) items each, atomically; a killed run skips every finished shard, and
`encode_params.json` refuses a resume under different settings. `shard_index.json` and
`text_emb.meta.json` are written only once every shard exists, so a partial encode is
never loadable. The encoder's remote code needs `einops` (now a dependency).

Measured on the A100 on 20,292 real texts (mean 228 tokens, 1.7 % at 512): **898 docs/s**
with bf16 weights at batch 256 (755 under bf16 autocast, 813 at batch 512; min cosine
between the two dtypes 0.99989); the forward alone runs 954 docs/s, so tokenization in
the main thread costs ~6 %. `flash_attn` is not installed.

#### Budget

As measured on 2026-09-26; `plan` from its
[report](../artifacts/e3-openalex/plan-2026-09-23.json):

| | 10 M catalog |
|---|---|
| read over S3 | **297.1 GB** of projected columns (all files; the slice is a hash, not a prefix) |
| `convert` wall time | **572 s** at `--workers 64`, 526 MB/s average (the 20 s probes of [`stream_probe.py`](../artifacts/e3-openalex/stream_probe.py) saw 176–230 MB/s; `plan`'s threaded rate, 20 MB/s, is GIL-bound) |
| sample rate | 0.35 → 29.9 M staged; any rate that stages N × (1 + margin) gives the same catalog |
| staging parquet | 16 GB (`_raw/openalex/staging`, deletable once no smaller catalog is wanted) |
| `prep` | 374 s; `papers.parquet` 6.3 GB, `item_id_map.json` 209 MB |
| fp16 item shards | 15 GB (10 × 1 M) |
| attrs | 1.5 GB (`[10 M, 5, 4]` int64) |
| encode | a fresh encode is ~2.9 A100-hours at 958 docs/s (the 15 M one took 15,663 s); this catalog was `reshard`ed from it in 34 s |
| on the device | items fp32 30.7 GB, plus the oracle's transposed copy; the cell reserved 64 GB |

**What fits the harness.** `load_inputs` holds items fp32 on the device and the oracle
adds a transposed fp32 copy, so a run needs ~2 × N × D × 4 B plus attrs and workspace: at
768-d that is ~11–12 M items on the 80 GB A100. 50 M (307 GB) and 15 M (92 GB) do not
fit; pubmed's 10 M (57 GB) does. Before that, the loader itself peaked at three copies
(fp16 + `.float()` + `F.normalize`, 107 GB at 15 M); `layout.load_text_items` now
normalises a sharded matrix shard by shard into one fp32 buffer (`load_sharded(...,
normalize=True)`, `torch.equal` to the old path on 3 M real rows), which is what got the
15 M run as far as the oracle. So E3 runs at 10 M: `prep --keep-items 10000000`, then
`reshard --from-dir` the 15 M catalog's vectors.

```
data/openalex/
├── item_id_map.json            {"W<id>": item_id}
├── papers.parquet              item-id order: work_id, year, field, subfield, source, is_oa, type, title, abstract, refs
├── heldout.parquet             target item_id + the query's own attributes and relevance counts
├── queries.parquet             query_work_id, title, abstract (heldout row order)
├── qrels.parquet               query_row, item_id, passes_field_era
├── content_d768/
│   ├── text_emb_shard_NNN.pt + shard_index.json + text_emb.meta.json
│   ├── query_emb.pt + query_emb.meta.json
│   └── encode_params.json
├── item_attrs_narrow.pt        [N, 5, 4] int64
├── clause_is_reverse_narrow.pt [5] bool = [F, F, F, T, F]
├── {field,subfield,source,era}_vocab.json
├── eval_split.parquet          target_id, query_attrs_narrow [5]
└── prep_log.json
```

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

`yfcc10m` is the same shape with one content dir (`content_d192`) and no
`papers.parquet` — the full listing is under
[yfcc10m](#yfcc10m) above. `pubmed` is the same shape with one content dir
(`content_d768`) whose item matrix is **sharded** — `shard_index.json` +
`text_emb_shard_*.pt`, the synth-arxiv layout — instead of one
`text_emb.pt`; `load_text_items` and `validate_layout` take either.
`openalex` is the pubmed shape plus `queries.parquet` and `qrels.parquet`
(listing under [openalex](#openalex)).

Filter sweeps additionally need:

```
├── item_attrs_narrow.pt
├── clause_is_reverse_narrow.pt
├── eval_split.parquet
└── ... per-clause vocab JSONs
```

### Dead artifacts

`item_attrs_wide.pt`, `wide_shelf_vocab.json`,
`wide_shelf_global_freq.pt`, and the `query_attrs_wide_1shelf` /
`_2shelf` columns of `eval_split.parquet` are **built but never read**:
`bench.inputs.load_inputs` does not load them and `layout.load_query_attrs`
reads only the `query_attrs_narrow` column. They exist for a wide-bloom
sweep that no suite runs. Keep or drop them as a unit — they are only meaningful
together.

## HuggingFace I/O

[`hub.py`](../../evaluation/eval_datasets/hub.py) holds the repo
registry and the fetch/publish helpers, so no path or repo id is
hard-coded in the ETL modules:

```python
EVAL_REPOS = {
    "arxiv-papers":      "pinkmeme/eval-arxiv-papers",
    "yambda-500m":       "pinkmeme/eval-yambda-500m",
    "yambda-5b":         "pinkmeme/eval-yambda-5b",
    "goodreads-work-id": "pinkmeme/eval-goodreads-work-id",
    "yfcc10m":           "pinkmeme/eval-yfcc10m",
    "pubmed":            "pinkmeme/eval-pubmed",
    "kuairand":          "pinkmeme/eval-kuairand",
}
```

`yfcc10m`'s repo is registered but **not published** — the upstream files
are already public and unauthenticated, so `yfcc download` is the fetch
path; the entry exists so the local directory layout resolves like every
other dataset's.

`pinkmeme/eval-pubmed` is **registered but not published**; nothing is
pushed to it before roadmap E2. `pinkmeme/eval-kuairand` is likewise
registered and not yet published (roadmap E4 publishes it with its
checkpoint).

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
| [`dataset.py`](../../evaluation/training/dataset.py) | `load_sequences` (the train split as left-padded `[U, L+1]` tensors on the device), `train_batches` (`randperm` slices) |
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
uv run --directory evaluation train run --config base.json loss=gbce num_negatives=256 warmup_steps=0
```

The defaults are the E1c recipe
([artifact](../artifacts/seqrec-encoder/e1c-yambda-d64-sasrec-ssm-logq/config.json)):
d64, 2 blocks, 2 heads, ffn 256, dropout 0.5; `loss=sampled_softmax` with
`normalize`, `temperature=0.05`, `num_negatives=8192`,
`inbatch_negatives=4096`, `logq`; lr 1e-3, batch 256, L 200, warmup 1000,
`compile`; 100 epochs, eval every 2, early stop on `ndcg@10` with patience
10. The published gSASRec recipe (Gate B) is `loss=gbce num_negatives=256
warmup_steps=0`.

Arguments are `TrainConfig` `FIELD=VALUE` pairs, the value parsed as JSON
when it parses and taken as a string otherwise; an unknown field is a usage
error. `--config <json>` starts from a saved config instead of the defaults
and the pairs override it. `--resume` picks up from
`<ckpt_dir>/_resume.pt`, written every epoch with model, optimizer,
scheduler, epoch, best metric and the Python / numpy / torch / CUDA RNG
states, so a resumed run is reproducible, not merely restarted.

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
  ([validation](../validation.md#seqrec-encoder-rewrite)).
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
  the train target positions (`target_frequencies`, one GPU bincount at
  startup). The positive column is not corrected (arXiv 2507.09331);
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
after `patience` evaluations without one.

TF32 is **enabled** here (`_enable_tf32`) — the opposite of the
benchmark harness, which pins it off for measurement determinism.

### What a finished run leaves behind

```
<checkpoint_dir>/
├── best_model.pt          # reloaded, then re-saved from the best epoch
├── config.json            # the TrainConfig — the eval loader reads this
├── item_embs.pt           # scoring_table(), row 0 zeroed
├── item_id_map.json       # copied from data_dir
├── item_attrs.parquet     # copied from data_dir when present
├── eval_quality.json      # test-split metrics
├── train_metrics.json     # losses, val curve, test metrics, epoch_time_sec, samples_per_sec, gpu_name, sm_mhz, peak_gpu_mem_bytes
├── sasrec-ep{N}-{metric}{v}.pt   # the current best-epoch snapshot
└── _resume.pt
```

`samples_per_sec` counts training sequences over the training part of the
epochs (eval excluded); `sm_mhz` is one SM clock sample per epoch taken
mid-epoch under load (clocks cannot be locked on this box).

`config.json` is what makes a checkpoint self-describing at eval time.
A checkpoint without it falls back to
`D128_DROP05_DEFAULTS` in
[`encode.py`](../../evaluation/training/encode.py) — see
[checkpoints.md](checkpoints.md) for which runs those are.

## Adding a dataset

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
