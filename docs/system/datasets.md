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
| **sequential** | yambda-500m, yambda-5b, goodreads | a trained SASRec checkpoint, encoded at eval time | goodreads only |
| **text** | arxiv, arxiv-synth, yfcc10m, pubmed | pre-encoded embeddings on disk | yes |

A third variant, **synthetic**, is a text dataset grown to arbitrary `N`
by interpolating between real embeddings — used for scale sweeps where a
real catalog that size doesn't exist.

Data lives under `$RETRIEVE_DATA_ROOT` (default `<repo>/data`). Raw
downloads go to `data/_raw/<dataset>/`; bench-side outputs to
`data/<dataset>/<variant>/`.

## `eval_datasets/` — the ETL package

One module per dataset, each an `argparse` CLI exposed as a console
script. The package is named `eval_datasets` rather than `datasets`
because the latter shadows HuggingFace's `datasets` in the shared venv.

| module | console script | source |
|---|---|---|
| [`yambda.py`](../../evaluation/eval_datasets/yambda.py) | `yambda` | HF `yandex/yambda`, Listen+ branch |
| [`goodreads.py`](../../evaluation/eval_datasets/goodreads.py) | `goodreads` | UCSD Book Graph mirror (HTTPS) |
| [`arxiv.py`](../../evaluation/eval_datasets/arxiv.py) | `arxiv` | HF `open-index/open-arxiv` (~2.99M papers) |
| [`yfcc.py`](../../evaluation/eval_datasets/yfcc.py) | `yfcc` | `dl.fbaipublicfiles.com` (NeurIPS'23 Big-ANN filtered track) |
| [`yfcc_check_gt.py`](../../evaluation/eval_datasets/yfcc_check_gt.py) | `yfcc-check-gt` | — (validates the shipped GT) |
| [`pubmed.py`](../../evaluation/eval_datasets/pubmed.py) | `pubmed` | NCBI FTP MedCPT embeddings + MEDLINE baseline (~36M articles) |
| [`synth_arxiv.py`](../../evaluation/eval_datasets/synth_arxiv.py) | — (run as a module) | an already-encoded arxiv directory |
| [`hf_io.py`](../../evaluation/eval_datasets/hf_io.py) | `eval-fetch`, `eval-publish`, `eval-publish-checkpoint` | HF Hub push/pull |
| [`common.py`](../../evaluation/eval_datasets/common.py) | — | shared attribute-synthesis numerics |
| [`timesplit.py`](../../evaluation/eval_datasets/timesplit.py) | — | vendored sequential time-split |

`synth_arxiv` has **no** console script despite its docstring examples —
invoke it as `uv run python -m eval_datasets.synth_arxiv`.

### Tests

```bash
cd evaluation && uv run pytest eval_datasets/tests -q     # CPU-only, no GPU
```

[`eval_datasets/tests/`](../../evaluation/eval_datasets/tests/) is CPU-only
and needs no network: the fixture writers at the top of
[`test_yfcc.py`](../../evaluation/eval_datasets/tests/test_yfcc.py)
(`write_u8bin`, `write_knn_result`, `write_spmat`) emit the upstream
binary formats into `tmp_path`, so the parsers are tested against bytes
rather than against a downloaded file. The last class, `TestRealSlice`,
round-trips a 1,000-item slice of the *real* prepared dataset against the
uncapped tag CSR and skips itself when `$RETRIEVE_DATA_ROOT/yfcc10m` is
not on the machine. Reuse those writers when adding the E2–E4 loaders.

### Shared conventions

- **Item ids are 1-indexed dense ints** in `item_id_map.json`, because
  id `0` is the sequence-padding token on the trainer side. The harness
  drops the padding row when loading (`item_embs[1:]`) and shifts target
  ids by −1, so the `retrieve` library is 0-indexed over real items
  throughout. This is why no filter mask needs an "item 0" fixup.
- **Attribute tensors are 0-indexed dense**: row `i` of
  `item_attrs_narrow.pt` describes `item_id i+1`.
- Every subcommand writes a `prep_log.json` with row counts and
  filtering statistics next to its outputs.
- Subcommands are individually re-runnable; `all` chains them.

### yambda

`uv run yambda prep --variant {500m,5b} --output-dir data/yambda/<v>`

Downloads `<variant>/sequential/listens.parquet`, runs `preprocess()`
(Listen+ branch: `played_ratio ≥ 50%`), and writes the four artifacts the
trainer consumes:

```
<output>/train.parquet      item_ids: list[int64]
<output>/val.parquet        item_ids, targets
<output>/test.parquet       item_ids, targets
<output>/item_id_map.json   {raw_yandex_id: dense_int}
```

Validation history is the train portion (already sliced to the last 200
items in `preprocess`); test history is train ++ val, last 200. Adapted
from the Yambda paper's reference `sasrec/data.py`; only the
listens-Listen+ branch is kept, the rest is replaced by
`evaluation.training`.

### goodreads

`download` → `convert` → `prep` → `attrs`.

`download` mirrors the UCSD top-level files over HTTPS (resumable via
`Range`, sha256 manifest); `convert` streams them into ZSTD parquet.

**`prep`** turns the staged parquets into yambda-shaped trainer inputs:
filters `is_read=true`, parses `date_added`, collapses editions to a
`work_id` catalog, runs an iterative n-core, then time-splits via
`timesplit.sequential_split_train_val_test`. Outputs `train/val/test.parquet`,
`item_id_map.json`, `book_to_work.parquet` (reused by the filter eval),
and `prep_log.json`.

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
`attrs`, or `uv run arxiv all --output-dir data/arxiv-papers` (the
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

Configs select one via `content_subdir`.

### yfcc10m

`download` → `convert` → `prep` → `attrs`, or
`uv run yfcc all --output-dir data/yfcc10m`.

The NeurIPS'23 Big-ANN **filtered-search** track set: 10M CLIP image
descriptors, 192-d uint8, plus a bag of tags per image drawn from a
200,386-word vocabulary (description words, camera model, year, country),
plus 100,000 queries that each carry 1–2 tags. Six files, 2.97 GB, no
registration, from
`https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/`
(exact names and sizes in
[dataset-candidates.md §3.4](../plans/dataset-candidates.md) and in
`yfcc.py`'s `RAW_FILES`). `download` is size-verified and resumable;
`convert` re-parses every header and writes
`data/_raw/yfcc10m/processed/manifest.json` with the sha256 of each file.

**This is the one dataset whose filtered ground truth is not ours.** For
every dataset above, `retrieval/oracle.py` computes the filtered
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
[`yfcc_check_gt.py`](../../evaluation/eval_datasets/yfcc_check_gt.py),
and it is the format harness v2 should grow an input for
([evaluation-harness-v2.md §7](../plans/evaluation-harness-v2.md#7-risks--open-questions)).

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
   The sidecars are named `emb_provenance.json`, **not** `*.meta.json`:
   `loaders.assert_arxiv_prefixes` treats a `*.meta.json` as a nomic
   encode and demands the `search_document: ` / `search_query: ` prefixes,
   which YFCC has no concept of.
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
│   └── emb_provenance.json
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

**Validating the shipped GT (roadmap E1's gate).**

```bash
export RETRIEVE_DATA_ROOT=/workspace/data
uv run --directory evaluation python -m eval_datasets.yfcc_check_gt \
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
### pubmed

**Status: skeleton only.** Roadmap E2 is deferred (2026-09-06) — no PubMed data
is staged and none of the numbers below have been produced, let alone gated.

`download` → `convert` → `medline` → `attrs` → `queries` → `encode_queries`.
Source is the NCBI FTP MedCPT article-embedding release, public domain, no
registration:
`https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/` — 38 chunks of

- `embeds_chunk_{i}.npy` — `(N_i, 768)` **float32** (verified from the npy
  header, `descr='<f4'`; ~102 GB total),
- `pmids_chunk_{i}.json` — the row-aligned PMID list (~400 MB total),
- `pubmed_chunk_{i}.json` — `{pmid: {"d": date, "t": title, "a": abstract,
  "m": mesh}}` (~44 GB total).

NCBI publishes **no** checksums for that directory, so `verify` checks the
shards structurally instead: the npy header must parse, dtype must be float32,
width 768, and the row count must equal `len(pmids_chunk_i.json)`. The MEDLINE
baseline *does* publish `.md5` and `verify --medline` checks those.

**No dimensionality reduction.** Every dataset is benchmarked at its encoder's
native dim (user decision 2026-09-06), so pubmed has exactly one content dir,
`content_d768`, and `dataset-candidates.md` §4.1's PCA-to-256/128/64 plan is
**not** implemented.

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
`medline/*.parquet` (`pmid`, `journal` = `MedlineTA`, `language`). MeSH tree-top
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
(a ~40 % pass rate) instead of the selective heading
`dataset-candidates.md` §4.1 asks for.

Language is parsed and stored in `articles.parquet` + `lang_vocab.json` but is
*not* one of the five clauses — §4.1 fixes the layout above.

#### Query sets

`queries` builds `queries.parquet` (`query_id`, `text`, `target_id`) from two
sources:

- `heldout` — item-as-query: a held-out article's title is the query and the
  article itself is the single relevant item. Always available.
- `nfcorpus` — the NFCorpus (BEIR) biomedical query set. NFCorpus document ids
  *are* PMIDs, so its qrels map straight onto our item ids. BEIR asks that its
  corpus not be redistributed, so nothing is downloaded automatically: stage
  `queries.jsonl` + `qrels/test.tsv` under `--nfcorpus-dir` yourself, or the set
  is skipped with a warning.

`encode_queries` runs `ncbi/MedCPT-Query-Encoder` ([CLS] pooling) and writes
`content_d768/query_emb.pt`. MedCPT's query and article encoders are
*asymmetric* — the same load-bearing property as nomic's prefixes on arxiv — so
the unfiltered cross-check sweep is meaningful rather than an identity lookup.

#### Disk budget

This is the binding constraint and the reason E2 is deferred. Raw is ~198 GB
(102 GB embeddings + 44 GB chunk JSON + 52 GB MEDLINE baseline) and the fp16
item matrix at native 768-d is another 55 GB. `convert --delete-raw` folds each
shard into the matrix and drops it, so the raw mirror never has to be held
whole, and `--emb-scratch` puts the fp16 accumulator on a disk that can take
55 GB while only the finished `content_d768/text_emb.pt` lands next to the rest
of the dataset. The accumulator is a `.npy` memmap, so a killed run resumes
rather than restarting.

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
[yfcc10m](#yfcc10m) above.

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
`retrieval.data.load_inputs` does not load them and `load_query_attrs`
reads only the `query_attrs_narrow` column. They exist for a wide-bloom sweep that was
never run. Keep or drop them as a unit — they are only meaningful
together.

## HuggingFace I/O

[`hf_io.py`](../../evaluation/eval_datasets/hf_io.py) holds the repo
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
}
```

`yfcc10m`'s repo is registered but **not published** — the upstream files
are already public and unauthenticated, so `yfcc download` is the fetch
path; the entry exists so the local directory layout resolves like every
other dataset's.

`pinkmeme/eval-pubmed` is **registered but not published** — E2 is deferred and
nothing has been pushed to it.

`eval-fetch` pulls a prepared dataset (optionally a subset of dims),
`eval-publish` pushes one, `eval-publish-checkpoint` pushes a trained
checkpoint under `<repo>/checkpoints/<ckpt-id>/`. Downloading a prepared
dataset is the fast path — the full arxiv encode is hours of GPU time.

## Training — `evaluation/training/`

A compact gSASRec trainer. It exists to produce the item embeddings and
query encoder the sequential benchmarks need; it is not a research
surface of its own.

| file | role |
|---|---|
| [`config.py`](../../evaluation/training/config.py) | `GSASRecConfig` dataclass + `save` / `load` / `num_items` |
| [`model.py`](../../evaluation/training/model.py) | `GSASRec` — `nn.TransformerEncoder`, `norm_first`, separate output embedding, `predict_last` |
| [`dataset.py`](../../evaluation/training/dataset.py) | `SequenceDataset`, negative-sampling collate, train dataloader |
| [`losses.py`](../../evaluation/training/losses.py) | `gbce_loss` |
| [`evaluate.py`](../../evaluation/training/evaluate.py) | chunked full-catalog scoring + metrics (shares `retrieval.metrics`) |
| [`train_sasrec.py`](../../evaluation/training/train_sasrec.py) | `train()` + the click CLI |
| [`upload_checkpoints.py`](../../evaluation/training/upload_checkpoints.py) | push a checkpoint dir to HF |

### Running a training job

```bash
uv run --directory evaluation python -m training.train_sasrec \
    --data-dir data/yambda/500m-listens \
    --checkpoint-dir checkpoints/yambda-500m-d128 \
    --embedding-dim 128 --dropout 0.5
```

Every `GSASRecConfig` field has a matching flag; alternatively pass
`--config <json>` to load a saved config verbatim. `--resume` picks up
from `<ckpt_dir>/_resume.pt`, which is written every epoch and carries
model, optimizer, epoch, best metric, and the Python / numpy / torch /
CUDA RNG states — so a resumed run is reproducible, not merely
restarted.

### The loss

`gbce_loss` implements gBCE: uniform negatives per positive, with the
positive logit passed through a calibration transform parameterized by
`gbce_t` (`0.75` default). The transform runs in **float64** — the
`pow(-beta)` and `1/(x-1)` steps lose the positive class entirely in
fp32. The rest of the step runs under bf16 autocast.

### Loop shape

Per epoch: bf16 autocast forward → `gbce_loss` → grad-clip at 1.0 →
AdamW (fused on CUDA). Every `eval_every` epochs it evaluates on
`val.parquet` by scoring the full catalog in `[B, eval_score_chunk]`
chunks and merging top-k across chunks, optionally masking history.
Checkpoints on improvement in `early_stop_metric` (`ndcg@10`), stops
after `patience` epochs without one.

TF32 is **enabled** here (`_enable_tf32`) — the opposite of the
benchmark harness, which pins it off for measurement determinism.
Training throughput matters more than bit-reproducibility of a matmul.

### What a finished run leaves behind

```
<checkpoint_dir>/
├── best_model.pt          # reloaded, then re-saved from the best epoch
├── config.json            # the GSASRecConfig — the eval loader reads this
├── item_embs.pt           # output embedding matrix, row 0 zeroed
├── item_id_map.json       # copied from data_dir
├── item_attrs.parquet     # copied from data_dir when present
├── eval_quality.json      # test-split metrics
├── train_metrics.json
├── gsasrec-ep{N}-{metric}{v}.pt   # per-improvement snapshots
└── _resume.pt
```

`config.json` is what makes a checkpoint self-describing at eval time.
Checkpoints trained before it was written fall back to
`D128_DROP05_DEFAULTS` in
[`encode.py`](../../evaluation/retrieval/encode.py) — see
[checkpoints.md](checkpoints.md) for which runs those are.

## Adding a dataset

1. Write `eval_datasets/<name>.py` with `download` / `convert` / `prep`
   subcommands emitting the layout above; register a console script in
   [`evaluation/pyproject.toml`](../../evaluation/pyproject.toml).
2. If it has attributes, add an `attrs` subcommand producing
   `item_attrs_narrow.pt`, `clause_is_reverse_narrow.pt`, the vocab
   JSONs, and `eval_split.parquet` aligned 1:1 with `test.parquet`.
3. Add an entry to `EVAL_REPOS` in `hf_io.py`.
4. Add one `evaluation/config/<name>.yaml` (harness v2: one YAML per
   dataset, see [evaluation.md](evaluation.md#config-one-yaml-per-dataset--suitesyaml))
   and list the dataset in the suites it belongs to in
   `evaluation/config/suites.yaml`. Set `checkpoint` for the sequential
   shape, `content_dir` for the text shape.
5. Train a checkpoint if sequential; otherwise encode embeddings into
   `content*/` with meta sidecars.

Reuse `common.py` for attribute synthesis (`dense_remap_ids`,
`synthesize_qa_narrow`, `sample_rare_biased_wide`) rather than
re-deriving the sampling — the rare-biased draw in particular is what
keeps filter selectivity in an interesting range.
