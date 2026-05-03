# Goodreads filter-bench harness

## Why this exists

Goodreads is the first benchmark dataset in this repo with **item
attributes**, so it's the first that can exercise `BloomFilter` and
`ClauseIndex` end-to-end. The current eval harness on `main` is
[benchmark.py](../../evaluation/retrieval/benchmark.py).
Its purpose is filter-**absent** retrieval benchmarking: encode the
test set once into a query cache, then run each algorithm in
[registry.py](../../evaluation/retrieval/algo_registry.py) at `bs=1`
against a single shared `item_embs`, recording recall@K / NDCG@K /
latency / peak GPU memory. Per the existing comment
([registry.py L13-L17](../../evaluation/retrieval/algo_registry.py#L13-L17)):

> `silvertorch` here is `IVF_INT8_ANN` (no bloom) since Yambda has no
> item attributes — running the bloom-fused `SilverTorch` against zero
> signatures would measure only the degenerate code path.

The natural step would be to retrofit the yambda harness with filter
support, but the user wants a **second** harness instead — keeps the
yambda numbers reproducible without conditional branches, and lets
the filter harness assume attributes are always present (cleaner
contract, no `if attrs is None` everywhere).

This doc depends on [goodreads-training.md](./goodreads-training.md)
for the trained gSASRec checkpoints (`item_embs.pt` source) and on
the unified [filtering API](./filtering-api.md) for the
`FilterModule` contract that this harness consumes.

## What the existing harness does (research notes)

Reading
[benchmark.py](../../evaluation/retrieval/benchmark.py)
end-to-end, the contract has six pieces:

1. **Config** ([config.py](../../evaluation/retrieval/config.py)):
   YAML → `EvalConfig` with `checkpoint`, `data_dir`, `algorithms`,
   `algo_params`, `ks`, `encode` (batch_size / num_workers /
   max_seq_length).
2. **Model load** (`load_model`, L94-L112): falls back to a hardcoded
   d128-drop0.5 hyperparam set when `config.json` is missing
   (because the 500M ckpts predate the config-saving change).
   Goodreads ckpts always ship `config.json`, so the fallback is
   dead code on this dataset.
3. **Query encoding once** (`encode_queries`, L117-L160): runs
   `model.predict_last` over the full eval split → `(queries [N, D],
   targets [N, T_max], num_targets [N])` cached on CPU. The
   transformer is freed before the perf passes.
4. **Index build** (`build_algorithm`, [registry.py](../../evaluation/retrieval/algo_registry.py)):
   one of `torch_fullscan / triton_knn / linr_v3_then_v2 /
   silvertorch (IVF_INT8_ANN)`; returns `(forward, modules)`.
5. **Quality pass** (`quality_pass_cached`, L165-L186): streams
   cached queries through `forward(q)` at bs=1, accumulates
   `recall@K / ndcg@K`.
6. **Perf pass** (`perf_pass_cached`, L189-L206): same `forward` on a
   fixed query, `triton.testing.do_bench` for `(median, p20, p80) ms`,
   custom mem hooks for `(peak_mib, scratch_mib)`.

Output is one JSON row per `(algorithm, K)` cell.

The **filter integration gap** is exactly the one called out in
[filtering-api.md](./filtering-api.md):
the current `forward = lambda q: idx(q)` lambdas in `registry.py`
do not accept `query_attrs`, so attributes never reach the index
even though `LiNR_V*` and `SilverTorch` already accept them.

## What the new harness adds

`evaluation/retrieval/eval_goodreads_retrieval.py` — a peer of the
yambda harness. Same pipeline shape (encode-once, then per-algo
quality + perf), but with a filter dimension:

```
for filter_kind in {"none", "clause", "bloom", "combined"}:
  for sweep in cfg.filters[filter_kind].sweeps:
    for algo in cfg.algorithms:
      for k in cfg.ks:
        # one row of output
```

The new dimensions:

- **`filter_kind`** — chooses which `FilterModule` instance(s) the
  forward wrapper will use. `none` reproduces the yambda harness
  output for cross-validation.
- **`sweep`** — a named active-clause set (narrow) or a query-attrs
  field (wide); see "Predicate shapes" below.
- **`algo`** wrappers in a new
  `evaluation/retrieval/filter_registry.py` module: thin adapters
  around the existing `build_algorithm` that thread `mask` /
  `candidate_ids` into LiNR forward and `query_clause_attrs` into
  SilverTorch forward. **The yambda `registry.py` is not modified**
  — it's the no-filter codepath, full stop.

### Filter wrapper contract

Pseudo-code for `filter_registry.py`:

```python
def build_filtered_algorithm(
    algo_name: str,
    item_embs: Tensor,
    item_attrs_narrow: Tensor | None,        # [N, C, A_max] int64
    item_attrs_wide: Tensor | None,          # [N, 1, A_max_wide] int64
    clause_is_reverse: Tensor | None,        # [C] bool
    *, k: int, params: dict[str, Any], filter_kind: str,
) -> tuple[FilteredForwardFn, list[nn.Module]]:
    """FilteredForwardFn = (q, query_attrs_narrow, query_attrs_wide) -> (ids, scores)."""
```

The wrapper:

1. Builds the underlying retrieval module via the existing yambda
   `build_algorithm` (so all the kernel paths are unchanged).
2. Builds the `FilterModule`(s):
   - `clause`: one `ClauseIndex` registered on `item_attrs_narrow`,
     `clause_is_reverse_narrow`.
   - `bloom`: one `BloomFilter` registered on `item_attrs_wide` —
     paper-strict, errors at register time if a reverse clause is
     passed.
   - `combined`: both, AND'd via `combine_indices` (sparse cascade,
     [filters/__init__.py L26-67](../../retrieve/src/retrieve/layers/filters/__init__.py#L26-L67)).
3. Routes per algo:
   - `LiNR_V1` / `FullScan`: pass `mask=filter.evaluate_mask(qa)` into
     `forward` — needs a small extension to FullScan to accept a mask
     kwarg, or fall back to `scores.masked_fill_` outside the kernel.
   - `LiNR_V2` / `LiNR_V3`: pass
     `candidate_ids=filter.evaluate_indices(qa)[0]` into `forward`.
   - `SilverTorch`: switch from `IVF_INT8_ANN` to `SilverTorch` and
     thread `query_clause_attrs=qa` directly (its bloom is fused
     into `codesigned_probe_score`).

`SilverTorch` only consumes the wide signature in its fused path; for
narrow the harness uses `SilverTorch + ClauseIndex` mask combo (the
[filtering-api.md cross-compat table](./filtering-api.md#cross-compatibility))
or skips the `silvertorch` cell on the narrow sweep. **Decision:** skip
SilverTorch on narrow sweeps; the report will note this as a gap, not
a metric loss. The bench thesis is "narrow → ClauseIndex; wide →
bloom-fused SilverTorch" — running SilverTorch on narrow doesn't add
signal to that comparison.

### Encoding once with attr-aware queries

The query cache from `encode_queries` is reused unchanged: the encoder
doesn't see attributes. The filter side gathers per-user query attrs
from the held-out target row at quality-pass time:

```python
target_id = targets[i, 0]
qa_narrow = item_attrs_narrow[target_id]   # [C, A_max] → reduce to [C] (first non-pad per clause)
qa_narrow = mask_inactive_clauses(qa_narrow, sweep.active_clauses)
qa_narrow = flip_reverse_clause(qa_narrow, clause_is_reverse, target_id)
```

This is the **target-attribute** sampling pattern that needs to be
implemented from scratch for this harness. The new harness lifts that
logic out of the per-row dataset into a
reusable `synthesize_query_attrs(targets, sweep, item_attrs,
clause_is_reverse) -> [N, C]` function in `filter_registry.py`.

### Ground-truth oracle

Per sweep, compute filtered top-`K_GT=1000` once via `item_embs @ q.T`
with the corresponding mask, cache as `gt/gt_topk_<sweep>.pt`. Recall
is then `|index_top_k ∩ gt_top_k[:, :K]| / K`. On 1.5M items × 100k
test users this is ~30 s / sweep on an A100.

The full-scan oracle is the **ground truth** (perfect recall by
construction); the bench algorithms are graded against it. The reason
this differs from the yambda harness — which uses the held-out target
id directly — is that filtered recall against a single target
degenerates: many sweeps will mask the target out, making "did we
find target?" undefined. The oracle approach gives a top-K-set
comparison that's defined for every sweep.

## Predicate shapes

### Narrow, exact — `item_attrs_narrow.pt`, `[N, 5, 4]` int64

Targets `ClauseIndex`.

| C | clause | A_max | reverse | source | cardinality |
|---|---|---|---|---|---|
| 0 | genre | 4 | no | `goodreads_book_genres_initial.parquet` top-4 of 10 buckets by count | 10 |
| 1 | language_code | 1 | **yes** | `goodreads_books.language_code` top-30 + `-1` for rest/missing | ~30 |
| 2 | format_bucket | 1 | no | `format` regex-mapped → {paperback, hardcover, ebook, audio, other} | 5 |
| 3 | year_bucket | 1 | no | `publication_year` → {<1990, 1990-2000, 2001-2010, 2011-2017, missing→-1} | 4 |
| 4 | author | 2 | no | first 2 of `authors[*].author_id` dense-id-remapped | ~830k |

`clause_is_reverse_narrow.pt = [False, True, False, False, False]`.
Coverage requirements (`prep_log.json` "attrs" section): C0 ≥ 60 %,
C1 ≥ 40 %, C2 ≥ 50 %, C3 ≥ 70 %, C4 ≥ 95 %.

### Wide, free-form — `item_attrs_wide.pt`, `[N, 1, 32]` int64

Targets `BloomFilter`. `popular_shelves[*].name` after stopword filter:

- blocklist: `{to-read, currently-reading, owned, owned-books,
  books-i-own, default, favorites, favourites, kindle, ebook,
  audiobook, library, library-book, wishlist, want-to-read, dnf,
  did-not-finish, unread, read, my-books, my-library, all-books,
  books, fiction, non-fiction}`
- regex drops: `r"^read-(in-)?\d{4}$"`, `r"^\d-?stars?$"`,
  `r"^[a-z]{2}-\d{4}$"`, `r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)$"`.

Per work: top-32 surviving shelves by aggregated `count` across all
its book editions (sum across the `book_id`s mapping to the same
`work_id`). Vocabulary persisted as `wide_shelf_vocab.json`. Books
with zero surviving shelves drop from the wide-eval split (still
indexed; still in every narrow sweep). Coverage requirement: ≥ 70 %
of items have ≥ 1 surviving shelf.

No reverse on wide (paper-strict bloom).

### Attribute-build job (`evaluation/data/goodreads.py attrs`)

Adds an `attrs` subcommand to the same script that
[goodreads-training.md](./goodreads-training.md) introduced for `prep`:

```
uv run python -m data.goodreads attrs \
  --processed-dir <path-to-goodreads-output> \
  --output-dir data/goodreads/work-id
```

Same `--processed-dir` semantics as the `prep` subcommand in
[goodreads-training.md](./goodreads-training.md): required, no default;
must point at the directory `goodreads.py convert` wrote into.
`--output-dir` must be the same `data/goodreads/work-id` that `prep`
populated, so `attrs` can reuse `book_to_work.parquet` and
`item_id_map.json` for the dense-id-remap.

Reads `goodreads_books.parquet` + `goodreads_book_genres_initial.parquet`
+ the training doc's `book_to_work.parquet` and `item_id_map.json`;
writes:

```
data/goodreads/work-id/
├── item_attrs_narrow.pt                # [N, 5, 4] int64
├── item_attrs_wide.pt                  # [N, 1, 32] int64
├── clause_is_reverse_narrow.pt         # [5] bool
├── lang_vocab.json
├── format_vocab.json
├── author_vocab.json
├── wide_shelf_vocab.json
├── eval_split.parquet                  # (uid, query_attrs_narrow,
│                                       #  query_attrs_wide_1shelf,
│                                       #  query_attrs_wide_2shelf)
└── gt/                                 # filled in by the bench harness
                                        #   on first run
```

Wide query sampling for `eval_split.parquet`:

- 1-shelf: pick one shelf from the target's bag, biased toward rare
  (probability ∝ `1 / sqrt(global_freq)`). Drop users whose target
  has 0 surviving shelves.
- 2-shelf: pick one common + one rare from the bag. Drop users whose
  target has < 2 surviving shelves. Composed at bench time as two
  passes of `BloomFilter.evaluate_indices` AND'd through
  `combine_indices` (until `[B, C, Q_max]` query-side multi-value
  lands per [filtering-api.md "Out of scope"](./filtering-api.md)).

## New eval configs

Three YAML files under `evaluation/conf/`, mirroring the yambda
`500m-d{64,128,256}.yaml` shape but with a `filters:` block:

```
evaluation/conf/
├── goodreads-d64.yaml
├── goodreads-d128.yaml
└── goodreads-d256.yaml
```

`goodreads-d128.yaml`:

```yaml
checkpoint: checkpoints/goodreads-gsasrec-d128-drop0.5/best_model.pt
data_dir: data/goodreads/work-id
output: null                  # null → <ckpt-dir>/eval_goodreads_retrieval.json
split: test
device: cuda
ks: [10, 100, 500]

encode:
  batch_size: 512
  num_workers: 8
  max_seq_length: 200

algorithms:
  - torch_fullscan
  - linr_v3_then_v2
  - silvertorch                 # used only on wide sweeps; skipped on narrow

# N≈1.5M → sqrt(N)≈1225; 1024 lists × ~1465 items/cluster avg.
algo_params:
  linr_v3_then_v2: {candidate_pool: 8000, v3_seed: 0}
  silvertorch:     {n_lists: 1024, n_probe: 24, n_iter: 10, m_bits: 1024, k_hash: 5}

filters:
  none:
    sweeps: [{name: full_scan}]
  clause:
    attrs_path: data/goodreads/work-id/item_attrs_narrow.pt
    reverse_path: data/goodreads/work-id/clause_is_reverse_narrow.pt
    sweeps:
      - {name: c0_genre,        active_clauses: [0]}
      - {name: c1_lang_reverse, active_clauses: [1]}
      - {name: c3_year,         active_clauses: [3]}
      - {name: c0c1,            active_clauses: [0, 1]}
      - {name: c0c3,            active_clauses: [0, 3]}
      - {name: c0c4_author,     active_clauses: [0, 4]}
      - {name: all5,            active_clauses: [0, 1, 2, 3, 4]}
  bloom:
    attrs_path: data/goodreads/work-id/item_attrs_wide.pt
    reverse_path: null
    sweeps:
      - {name: 1shelf, query_attrs_field: query_attrs_wide_1shelf}
      - {name: 2shelf, query_attrs_field: query_attrs_wide_2shelf}
  combined:
    attrs_narrow: data/goodreads/work-id/item_attrs_narrow.pt
    attrs_wide:   data/goodreads/work-id/item_attrs_wide.pt
    reverse_path: data/goodreads/work-id/clause_is_reverse_narrow.pt
    sweeps:
      - {name: c0_genre_x_1shelf, active_clauses: [0], query_attrs_field: query_attrs_wide_1shelf}
```

The d64 / d256 variants change only `checkpoint:` (no other deltas).
Three configs × three filter kinds × average 4 sweeps × 3 algos × 3 Ks
≈ **108 cells / dim** ≈ **324 cells total**. Each cell: ~30 s build +
~30 s quality + 1 s perf, so ~5 hr / dim end-to-end. Run in series.

## Implementation surface

**Create:**
- `evaluation/data/goodreads.py` — `attrs` subcommand (the `prep`
  subcommand lands with [goodreads-training.md](./goodreads-training.md)).
- `evaluation/conf/goodreads-d64.yaml` /
  `goodreads-d128.yaml` /
  `goodreads-d256.yaml` — three bench configs.
- `evaluation/retrieval/filter_registry.py` — wrapper around
  `registry.py`'s `build_algorithm` that adds `FilterModule`
  plumbing.
- `evaluation/retrieval/eval_goodreads_retrieval.py` — peer of
  `benchmark.py`, copy-with-filter-loop.
- `evaluation/retrieval/_bench_primitives.py` — small helper module
  holding `quality_pass_cached` / `perf_pass_cached` / `encode_queries`
  / `measure` / `forward_memory_mib`. Both harnesses import from here.
  This is a zero-behaviour-change extraction from the existing yambda
  harness (line-for-line move; no logic edits).

**Modify:**
- [evaluation/retrieval/config.py](../../evaluation/retrieval/config.py)
  — extend `EvalConfig` with an optional
  `filters: dict[str, FilterCfg]` field; the existing yambda harness
  ignores it. Bad keys still raise `TypeError` from the dataclass
  constructor.
- [evaluation/retrieval/benchmark.py](../../evaluation/retrieval/benchmark.py)
  — switch its private helpers (`measure`, `forward_memory_mib`,
  `cuda_allocated_mib`, `quality_pass_cached`, `perf_pass_cached`,
  `encode_queries`) to imports from `_bench_primitives.py`.
  Pure refactor — output identical to pre-refactor.

**Reuse unchanged:**
- [retrieve/src/retrieve/layers/filters/](../../retrieve/src/retrieve/layers/filters/)
  — `ClauseIndex`, `BloomFilter`, `combine_masks`, `combine_indices`
  already implement the contract this plan needs.
- [evaluation/retrieval/algo_registry.py](../../evaluation/retrieval/algo_registry.py)
  — yambda harness untouched; filter harness wraps it from outside.

## Verification

1. **Attr coverage** — `prep_log.json["attrs"]` per-clause coverages
   match the floors above.
2. **Cross-check vs yambda harness** — with `filter_kind="none"`,
   `eval_goodreads_retrieval.py` produces the same recall@K / NDCG@K
   for `torch_fullscan` as the yambda-harness regression number
   (within ±0.0005 — noise from order of arithmetic is OK).
3. **Quality smoke (narrow)** — `--filter clause --sweep c0_genre` on
   `linr_v3_then_v2` ⇒ recall@10 ≥ unfiltered. Filtering to the
   target's own genre can only help SASRec retrieval.
4. **Quality smoke (wide)** — `--filter bloom --sweep 1shelf` on
   `silvertorch` ⇒ recall@10 ≥ unfiltered. Bloom FPR within paper
   bounds for `(m_bits=1024, k_hash=5)`.
5. **Reverse-clause hard-error** — `--filter bloom` against a sweep
   that touches `c1_lang_reverse` exits non-zero with a clear
   message.
6. **Bench thesis** — across the full sweep matrix
   `(d64, d128, d256) × (clause, bloom, combined) × (linr_v3_then_v2,
   silvertorch)`, predict + verify per
   [filtering.md](../system/filtering.md): `ClauseIndex` wins narrow,
   `BloomFilter` wins wide. Numbers go into the per-checkpoint
   `benchmark.json` artifact and a follow-up writeup.

## Out of scope

- Modifying `benchmark.py`'s output (peer harness, not a
  retrofit; only the helper-extraction refactor touches it).
- A `bloom_compact` Triton kernel — first cut uses
  `compact_mask(bloom_match)` per [filtering-api.md](./filtering-api.md).
- Multi-value query (`[B, C, Q_max]`) — out of scope per the API plan;
  2-shelf wide queries compose via two `combine_indices` passes.
- Per-genre subset benchmarks — the full work-id catalog is the
  headline.
- Reviews / spoiler subset — orthogonal to filter bench.
- IVF-aware ClauseIndex (filter-while-probing) — research follow-up
  per [filtering-api.md "Out of scope"](./filtering-api.md#out-of-scope-explicit-non-goals).

## Implementation order

1. `evaluation/data/goodreads.py attrs` (one-shot, ~5 min on the
   user's hardware).
2. `_bench_primitives.py` extraction + yambda-harness refactor.
   Confirm yambda numbers unchanged before going further.
3. `filter_registry.py` — unit-tested against `LiNR_V*` and
   `SilverTorch` on a small synthetic catalog.
4. `eval_goodreads_retrieval.py` — smoke with `filter_kind=none`
   first to prove the cross-check vs yambda.
5. Full sweep across `(d64, d128, d256)`.
