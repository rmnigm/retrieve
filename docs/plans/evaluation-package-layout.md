# `evaluation/` — three packages, one dependency direction

> **Status:** planned 2026-09-15 on `development` at `76f8985` (nothing
> implemented, no code touched). Authored from a file-by-file inventory of
> `evaluation/` taken the same day against that commit, *after* the harness
> v2 rewrite (C1–C3) and the 2026-09-06
> [evaluation review](architecture-review-2026-09-06-evaluation.md) had
> landed. Every "today" claim cites a path.
>
> Question this plan answers: *now that the measurement harness is rewritten
> (H §3), where does each remaining concern in `evaluation/` live — data on
> disk, models that make embeddings, query encoding, algorithm wiring,
> measurement, ground truth, metrics, record storage, reporting — so that a
> reader names the package from the concern, the packages depend on each
> other in one direction, and the harness depends on the library through one
> narrow import surface?*
>
> **Scope.** Packaging only: which package, which file, which CLI, which test
> tree. The measurement protocol, record schema, config matrix and gates of
> [evaluation-harness-v2.md](evaluation-harness-v2.md) (**H**) are as built
> and unchanged; §8 lists the places where **H** and
> [../system/evaluation.md](../system/evaluation.md) read differently after
> this plan. The library side is [library-api-refactor.md](library-api-refactor.md)
> (**L**); the contract between the two is
> [library-harness-boundary.md](library-harness-boundary.md) (**X**). This
> plan absorbs the review's §1.11 layout contract
> ([dataset-candidates.md §6](dataset-candidates.md), plan text only today),
> §1.12 (training / metrics ownership) and §1.4 (`resume_key` placement).
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 Phase C (C5).
> There is no Mac: every step runs on the A100 box; "CPU" means "needs no
> GPU time".

> **Steer 2026-09-15 (user), recorded in every plan it touches.**
> *"We don't care about reproducing old results now, we're improving all code
> and rewriting, then testing and profiling, then running the full evals step
> by step."* The A1 golden baseline stops being a gate and becomes
> information; roadmap **C4 closed** on what the harness proves about itself
> (graph capture 20/20, kill-and-resume, cross-backend parity, the official
> cell end to end), not on equality with the pre-v2 harness. Order of work:
> **code first, then tests and profiling, then the evals one step at a time.**
> Authority: [00-roadmap.md](00-roadmap.md) §1 status block;
> [evaluation-harness-v2.md](evaluation-harness-v2.md) WP-4's amendment block.
>
> **Dependency satisfied 2026-09-15:** WP-V1's "after C4 flips; needs L WP-2"
> is met — L1, L2 and L3 are merged (`ea1243f`) and C4 is closed. §9's one-cell
> gate stands, but read it as written: it compares a `bench run` record against
> a record produced by **this** harness at the same `code_version`, not against
> the old harness. The shim deletion that L D10 scheduled for this step is
> **deferred** — the library was owned by another worker on the day C5 ran, and
> it needs its own scheduling.

## 1. Where it starts

Three top-level packages in one distribution (`retrieve-evaluation`,
[evaluation/pyproject.toml](../../evaluation/pyproject.toml)), 12 console
scripts, no pytest configuration, one package-level cycle.

| package | lines | what is in it |
|---|---:|---|
| `retrieval/` | 2,958 + 2,412 tests | **the harness v2 as built** — `config.py` (`Dataset`, `Job`, `load_matrix`), `data.py` (`load_inputs`, legacy layout, sweeps, filters, pool), `encode.py` (SASRec query encoding + cache; imports `training.*`), `oracle.py` (blob v4, `attrs_digest`, **`resume_key`**), `metrics.py`, `algos.py` (five wrappers + `ALGOS` / `PATHS` / `FILTER_BACKEND` / `CAPTURABLE`), `bench.py` (latency, graph, provenance, `atomic_write`), `run.py` (the loop, `append_record`, `read_keys`, `SCHEMA_VERSION`), `cli.py` (`bench run` / `campaign` / `report` stub), `upload.py` (argparse); tests under `retrieval/tests/` (10 files, CPU); stale `retrieval/algos/__pycache__/`, `retrieval/cli/__pycache__/` |
| `eval_datasets/` | 6,300 + 849 tests | per-dataset ETL CLIs (`arxiv.py` 1,047, `goodreads.py` 1,336, `yambda.py`, `synth_arxiv.py`, `yfcc.py` 786 + `yfcc_check_gt.py`, `pubmed.py` 1,203), `hf_io.py` (485, the Hub registry and `data_root()`), `common.py`, `timesplit.py`, `constants.py`; tests only for yfcc and pubmed |
| `training/` | 962 | gSASRec model, loss, dataset, `train_sasrec.py`, `evaluate.py` (imports `retrieval.metrics`), `upload_checkpoints.py` |

What is still wrong after C3 and the review, concretely:

1. **The cycle.** `retrieval/encode.py` imports `training.model` and
   `training.evaluate`; `training/evaluate.py` imports `retrieval.metrics`.
   Both docstrings now name each other as "the one import the other way";
   review §1.12 parked the fix as "later". Two packages that import each
   other are one package with two names.
2. **The on-disk layout is a contract nobody owns.** `eval_datasets/` writes
   it (six loaders, each by hand, with different extra columns), `retrieval/data.py`
   reads it and re-derives the checks (`_pre_encoded`'s row-alignment
   asserts, the legacy `[N+1]` sniffing, the prefix sidecars). Review §1.11
   found two E-branches already breaking it (PubMed's `cmd_queries` drops
   rows; YFCC omits the sidecars) and specified `eval_datasets/layout.py` +
   `bench check` — plan text only, "lands with E1". E1 landed without it.
3. **Query encoding is split three ways**: history → vector in
   `retrieval/encode.py` (with its cache in `data._sasrec`), text → vector
   inside `eval_datasets/arxiv.py` and `pubmed.py` (`cmd_encode_queries`).
   Phase E's SPECTER2 encoder has nowhere obvious to go.
4. **The algo wrappers are library code in the wrong package** —
   `LinrV1`–`LinrV4` and `Silvertorch` (362 lines) whose content is which
   filter path feeds which layer, plus `set_query_params` and `capturable`,
   which are properties of the retriever, not of the benchmark. **L** D5
   moves them into `retrieve.modules`; `algos.py` then is a table.
5. **The record's identity is spread over three files**: `Job.key` in
   `config.py`, `KEY_FIELDS` and `resume_key` in `oracle.py` (review §1.4:
   "a `Job`-key concern, misplaced"), `SCHEMA_VERSION`, `append_record`,
   `read_keys` in `run.py`. There is no `report.py` (D4; `bench report` exits 2).
6. **Twelve console scripts** (`bench`, `upload-results`, `yambda`, `arxiv`,
   `goodreads`, `yfcc`, `yfcc-check-gt`, `pubmed`, `upload-checkpoints`,
   `eval-fetch`, `eval-publish`, `eval-publish-checkpoint`) in three naming
   conventions; `synth_arxiv` has none; `upload.py` is argparse while
   everything else is click.
7. **Tests are in three places with no shared configuration**:
   `retrieval/tests/` (one conftest), `eval_datasets/tests/` (none), no
   `training/` tests, no `[tool.pytest.ini_options]`, no GPU marker (the
   review's E-phase list wants fixture tests for every loader).
8. **Dead and stale files**: `retrieval/algos/` and `retrieval/cli/` are
   `__pycache__`-only leftovers of the C3 rename; `config/pubmed/d768-filter.yaml`
   and `config/yfcc10m/d192-filter.yaml` are old-harness-schema files
   `load_matrix` cannot read; `results/` holds only pre-v2 outputs that H §5
   said move to `results/archive/`; `checkpoints.md` still says the data root
   is `evaluation/eval_datasets/` (it is `evaluation/data/`, `hf_io.data_root`).

## 2. Decisions

- **D1 — Three packages by concern, one dependency direction:**
  `eval_datasets` (what is on disk) ← `training` (models that produce
  embeddings from it) ← `bench` (measures algorithms over it). `bench` may
  import both; `training` may import `eval_datasets`; `eval_datasets` imports
  neither. A test enforces this (§4).
- **D2 — `retrieval/` is renamed `bench/`.** "retrieval" names the field, not
  the concern, and reads as a sibling of the `retrieve` library; `bench` is
  what the CLI is already called. Ten files, a `git mv`, and import rewrites;
  the harness's `code_version` hashes the *library* subtree, so no recorded
  cell is invalidated.
- **D3 — `eval_datasets/` keeps its name** (it was renamed from `datasets` to
  stop shadowing Hugging Face's package, and `data/` is the data root on
  disk) **and owns the layout contract**: `layout.py` per review §1.11, with
  the *read* side moved out of `bench/data.py` so writer and reader share one
  `Dataset` path object. The ETL scripts move under `eval_datasets/etl/` so
  the package top level is the contract (layout, hub, common) and the
  per-dataset scripts — two more arrive in E3 / E4 — are one level down.
- **D4 — Query encoding with a trained model belongs to `training`**
  (`training/encode.py`: today's `retrieval/encode.py` plus the cache logic
  from `data._sasrec`); text encoding at ETL time stays in
  `eval_datasets/etl/<name>.py` next to the item encoding it must match. The
  cycle disappears because `training` no longer imports the harness (D5).
- **D5 — `training/evaluate.py` gets its own ≈ 30-line recall / ndcg** instead
  of importing `bench.metrics`. Training-time validation is part of what a
  checkpoint *is*; the harness's metric code may change and a checkpoint's
  reported quality must not change with it. A test pins the two to agree on
  random data to 1e-9 (the existing `test_metrics.py` pattern, which already
  pins v2 metrics against the pre-v2 code).
- **D6 — `bench/` is the as-built harness plus one file and one rename:**
  `records.py` gathers the record's identity and storage (`SCHEMA_VERSION`,
  `KEY_FIELDS`, `resume_key`, `append_record`, `read_keys`, the samples
  sidecar, `flatten() → flat.csv` for D4) out of `run.py`, `oracle.py` and
  `config.py`; `bench.py` becomes `measure.py` (a module named like its
  package). `algos.py` becomes the table **X** §3 describes once **L** WP-2
  lands. Everything **H** §4 refuses to abstract stays refused.
- **D7 — Three console scripts:** `bench` (`run`, `campaign`, `report`,
  `check`, `upload`), `eval-data` (`arxiv`, `goodreads`, `yambda`,
  `synth-arxiv`, `yfcc`, `yfcc-check-gt`, `pubmed`, `fetch`, `publish`,
  `publish-checkpoint`, and E3 / E4's), `train` (`sasrec`,
  `upload-checkpoint`). Click groups; the existing command bodies move
  unchanged; `upload.py` becomes a click subcommand.
- **D8 — One test tree, `evaluation/tests/`, mirroring the packages**, with a
  root `conftest.py` (the tiny-dataset writer, a `gpu` marker that skips
  without CUDA) and `[tool.pytest.ini_options]` in the pyproject; the
  dependency-direction test lives there. `eval_datasets` fixture tests get a
  shared writer so E3 / E4 loaders arrive with tests.
- **D9 — Config and results directories are as built (H §3.3 / §3.2)** plus
  the two housekeeping moves H §5 already mandated: pre-v2 `results/**` to
  `results/archive/`, and the two old-schema YAMLs rewritten to the v2 shape
  (`config/yfcc10m.yaml`, `config/pubmed.yaml`) or deleted until their step
  runs.
- **D10 — The harness imports the library through `retrieve` and
  `retrieve.functional` only** (**X** §2). No `retrieve.ops`, no
  `retrieve.indexing`, no deep module paths.
- **D11 — Sequenced after the C4 gate flips and before D1.** The rename and
  the split touch every harness import; the C4 rerun should not have to
  chase them, and D1's campaign must run on the final package (the campaign
  log, child command lines and `bench upload` paths are recorded artifacts).

## 3. Target layout

```
evaluation/
├── pyproject.toml                 scripts: bench, eval-data, train; [tool.pytest.ini_options] with the gpu marker
├── config/                        H §3.3 as built: arxiv.yaml, goodreads.yaml, yambda-500m.yaml, yambda-5b.yaml, suites.yaml
│                                  (+ yfcc10m.yaml, pubmed.yaml in v2 shape when E5 / E2 run)
├── golden/                        A1's JSONs and logs (unchanged)
├── results/                       H §3.2 as built; pre-v2 outputs under archive/
├── data/                          the data root on disk (gitignored; hub.data_root)
├── eval_datasets/                 ── what is on disk ──
│   ├── __init__.py
│   ├── layout.py                  the on-disk contract as code (review §1.11 + the read side of bench/data.py):
│   │                              Dataset paths, write_query_set, validate_layout, load_items / load_queries / load_query_attrs
│   │                              / load_filter_assets, drop_legacy_padding_row, check_items_aligned, users_limit applied ONCE
│   ├── hub.py                     hf_io.py renamed: EVAL_REPOS / RAW_REPOS, download_* / upload_*, data_root()
│   ├── common.py, timesplit.py, constants.py   unchanged (+ retry_download / checksum / year_to_bucket deduped here, review §1.11)
│   └── etl/
│       ├── __init__.py
│       ├── arxiv.py, goodreads.py, yambda.py, synth_arxiv.py, yfcc.py, yfcc_check_gt.py, pubmed.py   moved verbatim
│       └── (E3 s2.py, E4 kuairand.py)
├── training/                      ── models that make embeddings ──
│   ├── __init__.py
│   ├── config.py, dataset.py, model.py, losses.py, train.py (was train_sasrec.py)   unchanged
│   ├── evaluate.py                training-time validation with its own recall / ndcg (D5)
│   ├── encode.py                  retrieval/encode.py + the encode cache from data._sasrec
│   └── checkpoints.py             upload_checkpoints.py renamed (thin over eval_datasets.hub)
├── bench/                         ── measures algorithms ── (renamed from retrieval/)
│   ├── __init__.py
│   ├── config.py                  Dataset, Job, load_matrix                                (as built)
│   ├── inputs.py                  load_inputs over eval_datasets.layout + training.encode; sweep_qa, build_filters,
│   │                              exact_filter, query_pool                                (data.py minus the readers)
│   ├── oracle.py                  as built minus resume_key / KEY_FIELDS
│   ├── metrics.py                 as built
│   ├── algos.py                   ALGOS name → retrieve class, FILTERS, build(job, inputs), PATHS derived   (≈ 80 lines after L WP-2)
│   ├── measure.py                 bench.py renamed: setup, provenance, clocks, timed_build, index_bytes, latency,
│   │                              graph_callable, profile_once, atomic_write
│   ├── records.py                 SCHEMA_VERSION, KEY_FIELDS, resume_key, append_record, read_keys, samples sidecar,
│   │                              flatten() → flat.csv                                     (new; from run.py / oracle.py / config.py)
│   ├── run.py                     the loop (as built, minus the storage functions)
│   ├── report.py                  flat.csv → tables, figures, methodology paragraph      (roadmap D4)
│   ├── upload.py                  results/ → HF mirror (click subcommand)
│   └── cli.py                     `bench` group: run, campaign, report, check, upload
└── tests/
    ├── conftest.py                write_tiny_dataset (from retrieval/tests/conftest.py), gpu marker + skip
    ├── test_dependency_direction.py
    ├── eval_datasets/             test_layout.py (round-trip, users_limit once, writer/reader path parity), test_yfcc.py,
    │                              test_pubmed.py (moved), fixture tests for arxiv / goodreads / yambda writers (E-phase list)
    ├── training/                  test_encode.py (cache key; D5 metric agreement)
    └── bench/                     today's ten files, moved; test_records.py; test_paths.py (PATHS == derive(DISPATCH))
```

Move table (old → new). *moved* rows are `git mv` + import rewrites; *split*
rows move functions between files with no behaviour change.

| today | after | how |
|---|---|---|
| `retrieval/*.py` | `bench/*.py` | moved (D2); `bench.py` → `measure.py` |
| `retrieval/data.py`: `drop_legacy_padding_row`, `check_items_aligned`, `load_query_attrs`, `_assert_prefixes`, `_load_sharded`, `_pre_encoded`'s reading half | `eval_datasets/layout.py` | split; `load_inputs`, `sweep_qa`, `build_filters`, `exact_filter`, `query_pool` stay as `bench/inputs.py` |
| `retrieval/encode.py` + the encode cache in `data._sasrec` | `training/encode.py` | moved / split |
| `oracle.py::{KEY_FIELDS, resume_key}`, `run.py::{SCHEMA_VERSION, record_path, append_record, read_keys}` | `bench/records.py` | split (review §1.4) |
| `training/evaluate.py` import of `retrieval.metrics` | local `_recall_at_k`, `_ndcg_at_k` | D5 |
| `eval_datasets/hf_io.py` | `eval_datasets/hub.py` | moved |
| `eval_datasets/{arxiv,goodreads,yambda,synth_arxiv,yfcc,yfcc_check_gt,pubmed}.py` | `eval_datasets/etl/*.py` | moved; `main()` bodies become `eval-data` subcommands |
| `training/train_sasrec.py`, `training/upload_checkpoints.py` | `training/train.py`, `training/checkpoints.py` | moved |
| `retrieval/tests/*`, `eval_datasets/tests/*` | `tests/bench/*`, `tests/eval_datasets/*` | moved |
| `retrieval/algos/`, `retrieval/cli/` (`__pycache__` only) | deleted | housekeeping |
| `results/{arxiv,goodreads,yambda,deep_sweeps}/` | `results/archive/` | H §5 |
| `config/pubmed/`, `config/yfcc10m/` (old schema) | `config/pubmed.yaml`, `config/yfcc10m.yaml` in v2 shape | D9 |
| `docs/plans/evaluation-harness-v2-artifacts/c4_gate.py` | unchanged; `tests/bench/test_c4_gate.py` keeps its path | — |

## 4. The dependency rule, and how it is enforced

```
retrieve (library)  ←  bench  →  training  →  eval_datasets
                        └──────────────────→  eval_datasets
```

Allowed import edges: `bench → {retrieve, retrieve.functional, training.encode,
eval_datasets.layout, eval_datasets.hub}`; `training → {eval_datasets.hub,
eval_datasets.layout, eval_datasets.timesplit}`; `eval_datasets → {}`; the
library is imported by `bench` only. `tests/test_dependency_direction.py`
walks each package's `import` / `from` statements with `ast` (≈ 25 lines) and
asserts every cross-package and library edge is in the allow-list; a new edge
is a deliberate PR that edits the list.

## 5. Package detail

### 5.1 `eval_datasets` — what is on disk

`layout.py` is review §1.11's specification plus the reading half of
`bench/data.py`: `Dataset` (the path object `bench/config.py` builds today,
moved here so the ETL writers use the same attribute names), `write_query_set`,
`merge_prep_log`, `validate_layout(data_dir, content_dir) -> list[str]`
exposed as `bench check --dataset X`, and the readers with the checks
`data.py` carries today (`_assert_prefixes`, the sharded loader,
`drop_legacy_padding_row`, `check_items_aligned`, `load_query_attrs`).
`users_limit` is one function, `apply_users_limit(queries, targets,
n_targets, qa, n)`, called from `bench/inputs.py` exactly once (as
`load_inputs` does today, kept). The multi-target `item_ids` column, the
explicit prefix policy (`prefix: null` means none; a missing sidecar is an
error) and the deduplicated `retry_download` / `checksum` / `year_to_bucket`
helpers are the review's list, unchanged. Phase E's loaders are
`etl/<name>.py` modules whose fixture tests build a tiny layout on
`tmp_path` and run `validate_layout` on it.

### 5.2 `training` — models that make embeddings

Unchanged except: `encode.py` owns "history → query vector" with its cache
(keyed on checkpoint mtime and `max_seq_length` as today, plus `users_limit`
per review §2.6); `evaluate.py` owns its metrics (D5); `checkpoints.py` is
the upload command body. Text encoders for Phase E queries run at ETL time in
`eval_datasets/etl/<name>.py` next to the item encoding they must match (same
model, same prefix) — the invariant D §4 cares about.

### 5.3 `bench` — measures algorithms

The as-built harness with three packaging changes:

- `records.py` is the file a reader opens to learn what a record *is*: the
  schema version, the key block (`KEY_FIELDS` and `Job.key`'s field order),
  `resume_key`, the JSONL append with its `fsync`, the samples sidecar, the
  tolerant `read_keys`, and `flatten(results_dir) -> flat.csv` (H §8.2 G) that
  `report.py` reads. `run.py` writes through it; `oracle.py` no longer knows
  about resume keys.
- `algos.py` shrinks to the table once **L** WP-2 ships the composites:
  `ALGOS = {"linr_v1_filter_mask": LiNRV1, "linr_v2": LiNRV2, "linr_v3": LiNRV3,
  "linr_v4": LiNRV4, "silvertorch": SilverTorch}`, `FILTERS`, `build(job,
  inputs)` (≈ 25 lines: class, filter by filter backend, `register_index`,
  `.k = k_max`), `PATHS` derived from `retrieve.interfaces.DISPATCH` and
  checked by `tests/bench/test_paths.py`. `FILTER_BACKEND` and `CAPTURABLE`
  go: the first is a one-line function, the second is the module's own
  `capturable` attribute (**X** §4).
- `report.py` is roadmap D4 (H WP-6): `flat.csv` first, then the thesis /
  paper tables, the QPS-vs-recall Pareto and the recall-at-budget table
  (H §8.2 H), the methodology paragraph; it honours `partial`, enforces the
  subtree `dirty` flag, tolerates a torn samples line (review §(d)).

## 6. CLIs

```
bench run|campaign|report|check|upload
eval-data arxiv|goodreads|yambda|synth-arxiv|yfcc|yfcc-check-gt|pubmed <subcommand> [...]
eval-data fetch|publish|publish-checkpoint
train sasrec [...]   |   train upload-checkpoint [...]
```

`[project.scripts]` goes from 12 entries to 3. Command bodies move
unchanged; only the click group wiring is new. `docs/system/{evaluation,
datasets,checkpoints}.md` and CLAUDE.md's commands block show the new
invocations; the 2026-09-06 status block's `bench` commands stay valid.

## 7. Tests

`evaluation/tests/conftest.py`: the tiny-dataset writer (moved from
`retrieval/tests/conftest.py`), a `gpu` marker registered in
`[tool.pytest.ini_options]`, a `pytest_collection_modifyitems` hook that
skips `gpu`-marked tests without CUDA (the library's gate, copied). Today's
ten `retrieval/tests/` files and the two `eval_datasets/tests/` files move;
new: `test_dependency_direction.py` (§4), `tests/eval_datasets/test_layout.py`
(round-trip through the writer; `users_limit` applied once; writer / reader
path-name parity; `validate_layout` flags the two review-found breakages on
synthetic inputs), `tests/training/test_encode.py` (cache key changes with
each input; D5 metric agreement to 1e-9), `tests/bench/test_records.py`
(append / read / flatten round-trip; resume key stable across field order),
`tests/bench/test_paths.py`. Command: `cd evaluation && uv run pytest tests/`;
CLAUDE.md's command loses its `--ignore` clause.

## 8. Where **H** and the system doc read differently after this plan

| H / evaluation.md says | read as |
|---|---|
| `evaluation/retrieval/` (every path) | `evaluation/bench/` |
| `bench.py` | `bench/measure.py` |
| `data.py` "arxiv / SASRec loaders, encode cache, …, filter modules by filter backend" | readers → `eval_datasets/layout.py`; encode cache → `training/encode.py`; the rest → `bench/inputs.py` |
| `algos.py` five wrappers + `PATHS` / `FILTER_BACKEND` / `CAPTURABLE` | the table of **X** §3 (≈ 80 lines), `PATHS` derived, the two dicts gone |
| `oracle.resume_key`, `run.append_record` / `read_keys` / `SCHEMA_VERSION` | `bench/records.py` |
| `upload-results` console script | `bench upload` |
| `bench report` "roadmap D4, exits 2" | `bench/report.py` (D4, unchanged scope) |
| `retrieval/tests/` | `evaluation/tests/bench/` |
| review §(d) "E-phase: `eval_datasets/layout.py` + `bench check`" | C5 (this plan), before E3 / E4 rather than with them |
| review §(d) "later: 1.12 training / metrics ownership, 1.4 `resume_key` → `config.py`" | D5 and `records.py` here |

## 9. Work packages

CPU gates for every WP: `ruff check evaluation`, `cd evaluation && uv run
pytest tests/` green (CPU tests run, `gpu` tests skip), `python3
scripts/check_doc_links.py` at zero. GPU gate: the C4 gate script re-run on
the same 14 cells is *not* required — no library code changes — but one
`bench run` cell on goodreads-d128 `c0_genre` (`silvertorch`, `triton`) must
produce a record whose key block and `quality` equal the pre-rename record
for the same `code_version`.

- **WP-V1 — the rename, the split, the tests tree (CPU, 1.5 d). After C4
  flips; needs L WP-2 for the `algos.py` table.** `bench/` rename; `records.py`;
  `measure.py`; `eval_datasets/layout.py` + `hub.py` + `etl/`; `training/encode.py`
  + D5; `evaluation/tests/` with conftest, marker, dependency test and the new
  test files; the housekeeping rows of §3 (dead dirs, `results/archive/`, the
  two old-schema YAMLs). Gate: the CPU gates; the dependency test green with
  the §4 allow-list; the one-cell record equality above.
- **WP-V2 — CLIs and docs (CPU, 0.5 d; same branch).** `eval-data` and
  `train` groups, `bench check` and `bench upload`, pyproject scripts 12 → 3,
  `docs/system/{evaluation,datasets,checkpoints}.md` invocations and the
  `checkpoints.md` data-root fix, CLAUDE.md's commands block. Gate: each
  subcommand's `--help` renders; CPU gates.
- **WP-V3 — `report.py` (= roadmap D4, H WP-6).** Unchanged scope; lands in
  `bench/` and reads `records.flatten()`.

WP-V1 + WP-V2 is one roadmap step (C5), two CPU days, no GPU time beyond
one cell.

## 10. Risks

- **The rename touches every harness import and `c4_gate.py`'s test.** The
  gate script itself reads JSONL and is path-free; only
  `test_c4_gate.py`'s `bench.ROOT` reference moves. D11 puts the step after
  the C4 flip so the gate day never chases a rename.
- **D5 duplicates metric code.** Two ≈ 15-line functions, pinned by a test to
  agree. The alternative — a fourth package for shared math — is a package
  for two functions.
- **`eval_datasets` grows by ≈ 300 lines of readers.** It is still the right
  home: the writer / reader parity test is only possible when both sides
  share `Dataset`'s path attributes, and review §1.11 reached the same
  conclusion from the writer side.
- **Three click groups hide twelve scripts a runbook may still name.** The
  system docs and CLAUDE.md are updated in WP-V2; archived plans and the
  2026-09-06 status block keep the old names (records are not rewritten).
- **The old-harness golden worktree** (`/workspace/wt/main-golden`) is a
  separate checkout and is untouched by this plan.

## 11. Validation record

### 11.1 WP-V1 + WP-V2 (roadmap C5) — 2026-09-15, A100-SXM4-80GB, `dev/c5-harness-split`

> Model: [library-api-refactor.md §12.1](library-api-refactor.md). Worker job under
> [agent-orchestration.md](agent-orchestration.md): **no roadmap checkbox was flipped and
> nothing was merged into `development`.** Branch `dev/c5-harness-split` off `development`
> @ `71430d7`, worktree `/workspace/wt/c5`. Nothing here is citable (CLAUDE.md rule 2).

**Environment.** The A100 box (driver 580.159.04, nvcc 12.4), Python 3.11, torch
2.10.0+cu128, triton 3.6.0, ruff 0.15.6 (`uvx`; not in the lock), Meta's `silvertorch`
built into `/venvs/c5` with `uv sync --extra official --all-packages`. Private
`TORCHINDUCTOR_CACHE_DIR=/tmp/inductor-c5`; datasets at `/workspace/data`
(`RETRIEVE_DATA_ROOT`, `evaluation/data` a symlink). The GPU was shared with the L4 worker:
both GPU stages ran under `flock /workspace/gpu.lock`, one at a time; clocks cannot be
locked here. **`retrieve/` was not touched**: the library subtree is `0fe440d4bc90…`, the
same tree hash as every C4 record, so `code_version` carried across the rename unchanged.

**What was done.** The §3 move table: `retrieval/` → `bench/` (`bench.py` → `measure.py`,
`data.py` → `inputs.py` with its readers split out, `encode.py` → `training/encode.py`),
`records.py` (`SCHEMA_VERSION`, `KEY_FIELDS`, `resume_key`, `record_path`, `samples_path`,
`append_record`, `read_records` / `read_keys`, `flatten → flat.csv`), `algos.py` as the
**X** §3 table — `ALGOS` name → library class, `PATHS` *derived* from
`retrieve.interfaces.DISPATCH`, `filter_backend()` and the library's own `capturable` in
place of the `FILTER_BACKEND` / `CAPTURABLE` dicts, `build()` constructing the module and
calling `register_index` (the `Silvertorch` wrapper is gone; `run.perf` switches the plan
cache through `OfficialConfig` directly) — `eval_datasets/layout.py` (the readers and their
checks, `atomic_write`, `apply_users_limit` as the one `users_limit` site,
`validate_layout` behind `bench check`), `hf_io.py` → `hub.py` with module-level click
commands, the seven ETL scripts under `etl/` with `main(argv)` behind the `eval-data`
group, `training/train.py` + `checkpoints.py` behind the `train` group, `training/evaluate.py`
with its own `hits_at` / `recall_at_k` / `ndcg_at_k` (D5; the `retrieval` ↔ `training`
cycle is gone), 12 console scripts → 3, one test tree `evaluation/tests/{bench,
eval_datasets,training}` with a root `conftest.py` (`write_tiny_dataset`, the `gpu` marker
skip) and `[tool.pytest.ini_options]`, the new `test_dependency_direction.py`,
`test_paths.py`, `test_records.py`, `test_layout.py`, `test_encode.py`; pre-v2 `results/**`
→ `results/archive/`, `config/{pubmed,yfcc10m}.yaml` in v2 shape (in no suite yet); the
system docs, CLAUDE.md's command block and five plan links re-pointed. **L4-c** in
`measure.clocks` / `run.run` (§11.3). `git diff --stat -M` of the code commit: 109 files,
2,356 (+) / 2,013 (−), 65 rename-detected paths.

**Gates — CPU.**

| gate | result |
|---|---|
| `ruff check evaluation` (0.15.6) | **clean** — the 11 pre-existing E501s in `eval_datasets/` and `training/` are cleared (line wraps only). `ruff format --check evaluation` still names 7 files whose *pre-existing* style it would change (`etl/{pubmed,synth_arxiv,yfcc,yfcc_check_gt}.py`, `hub.py`, `train.py`, `tests/eval_datasets/test_yfcc.py`); left alone — cosmetic, and CLAUDE.md's command formats `retrieve` only |
| `cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/` | **164 passed, 4 skipped** (the 4: `test_yfcc.py::TestRealSlice`, yfcc10m not staged), 1 min 46 s. Was 103 passed / 1 skipped in `retrieval/tests/` + 2 files under `eval_datasets/tests/` |
| `tests/test_dependency_direction.py` | green: `bench` → `training.encode`, `eval_datasets.{layout,hub}`; `training` → `eval_datasets.{hub,layout,timesplit}`; `eval_datasets` → nothing; the library only through `retrieve` (the **X** §2 names), `retrieve.functional`, `retrieve.interfaces` (`DISPATCH`, `FilterModule`, the two backend literals) |
| `tests/bench/test_paths.py` | green: the derived `PATHS` equals the expected table on the full `ALGOS × FILTER_KINDS × BACKENDS` grid; the markdown-parsing test of `architecture.md` is deleted |
| `python3 scripts/check_doc_links.py` | 0 broken links (9 were broken by the moves; all re-pointed) |
| every `--help` | 20 / 20 render with rc 0: `bench {,run,campaign,check,upload,report}`, `eval-data {,arxiv,goodreads,yambda,synth-arxiv,yfcc,yfcc-check-gt,pubmed,fetch,publish,publish-checkpoint}`, `train {,sasrec,upload-checkpoint}` |
| `bench check` on the staged data | `goodreads` d64 / d128 / d256 ok, `arxiv` d128 ok |
| `records.flatten` on the gate run's records | `flat.csv`, 108 rows (6 records × 18 perf entries), 83 columns |

**Gate — GPU: the one cell** ([c5/](evaluation-package-layout-artifacts/README.md);
runbook [c5_gpu_run.sh](evaluation-package-layout-artifacts/c5_gpu_run.sh), comparison
[c5_compare.py](evaluation-package-layout-artifacts/c5_compare.py)). `bench run --dataset
goodreads --dim 128 --suite filter --filter-kind clause --sweep c0_genre --algo silvertorch
--backend triton`, both modes, full perf. Without `--seed 0` the suite's headline block
expands to seeds {0, 1, 2} × `n_probe` {24, 32} = **6 cells, all `status: ok`**, 480–509 s
each, 13:54–14:43 UTC. Every graph entry measured (no `reason` on any of the 54 graph
variants → `cudagraph_skips == 0`). The two **seed-0** cells are the gate:

| cell | key block vs the C4 record | `quality` (31 fields) | `code_version` |
|---|---|---|---|
| `silvertorch/triton c0_genre n_probe=24` | equal | **equal to the digit** | `0fe440d4…`, both |
| `silvertorch/triton c0_genre n_probe=32` | equal | **equal to the digit** | `0fe440d4…`, both |

The record identity survives the rename: `resume_key` of a C5 record finds the C4 record.
The seed-1 / seed-2 cells have no pre-rename counterpart (C4 ran seed 0) and are
information only.

### 11.2 L4-b — the `linr_v4` chunk-64 experiment: **the attribution is not confirmed**

[l4b_chunk64.py](evaluation-package-layout-artifacts/l4b_chunk64.py) sets
`run.QUALITY_CHUNK = 64` and calls `bench run` — nothing else differs; `QUALITY_CHUNK`
**stays 16**. `linr_v4/triton`, goodreads-d128 `c0_genre`, quality only, seeds {0, 1, 2}
(identical to the digit, as V4 must be):

| k | metric | v2 at chunk 64 | golden | \|Δ\| vs golden | v2 at chunk 16 (C4) | \|Δ\| vs C4 |
|---|---|---|---|---|---|---|
| 100 | recall | 0.981765912959 | 0.982053974462 | **2.88e-4** | 0.982127004293 | 3.61e-4 |
| 100 | ndcg | 0.986845961920 | 0.987055010100 | 2.09e-4 | 0.987106857471 | 2.61e-4 |
| 500 | recall | 0.985861852325 | 0.986093519232 | 2.32e-4 | 0.986102850663 | 2.41e-4 |
| 500 | ndcg | 0.988789057119 | 0.988973464071 | 1.84e-4 | 0.988980759199 | 1.92e-4 |
| 1000 | recall | 0.987059843378 | 0.987315650142 | 2.56e-4 | 0.987317780281 | 2.58e-4 |
| 1000 | ndcg | 0.989421225299 | 0.989631064621 | 2.10e-4 | 0.989632714459 | 2.11e-4 |

Chunk 64 does **not** reproduce the golden; it lands **four times further** from it than
chunk 16 does at `recall@100` (2.9e-4 vs C4's 7.3e-5) and moves `k = 500 / 1000` by
2.3–2.6e-4 where chunk 16 was within 9e-6 / 2e-6, all in the same direction (down). So the
batch shape does move `linr_v4` (C4's probe was right about that: 1536 / 2048 rows differ
between the two shapes), but "the golden batched at 64 and v2 at 16" is **not** what
separates the two harnesses — chunk 16 is the closer of the two at every `k`. What the old
harness's quality pass actually did to the batch (its chunking, padding, or the compiled
forward it scored through) is the open question; it needs the golden worktree, which this
job did not own. Carried forward as still-open L4-b, with the number that kills the
current hypothesis. One thing the run adds: at chunk 64 the `heldout` recall@100 is
0.212285 vs 0.211894 at chunk 16 — the int8 tie order moves held-out recall by 4e-4 too.

### 11.3 L4-c — the `unstable` artifacts, fixed

`measure.clocks()` is one `nvidia-smi` sample and nothing more: `expected_sm_mhz` /
`clocks_locked` are gone (nvidia-smi exposes no lock flag; "within 2 % of an expectation"
read `true` on a box that cannot lock, which is what C4 found) and `bench run` loses
`--expected-sm-mhz`. `run.run` records the process-start sample as `env.sm_mhz_idle`
(with `mem_mhz`, `sm_max_mhz`, `power_limit_w`), and `clocks_drift` compares **only** the
perf entries' under-load `sm_mhz` samples against the process's first under-load sample
(`CLOCK_DRIFT` 5 %); `env.sm_mhz_load` is the median of a cell's under-load samples.
`SCHEMA_VERSION` is 2 for the field changes. On the gate run: 4 of 6 cells
`clocks_drift: false` with every sample at 1410 MHz — the old rule (idle 1155 vs 1410)
would have flagged all six; the 2 flagged cells each carry one variant sampled at 1185 and
1290 MHz, i.e. a real dip, which is what the flag is for. `unstable` is still `true` on 5 of
6 cells from **window spread** — seed 0 `n_probe=24` has `graph k=100 bs=1` at 14.8 % and
`eager k=1000 bs=8` at 26.6 % — the bs=1 launch-latency noise C4 §12.7 recorded, plus one
bs=8 eager outlier this run adds; both are measurements, not artifacts.

### 11.4 Deviations from the plan text, and what is not done

- **`Dataset` stays in `bench/config.py`** (§3 put it in `layout.py`): its fields are
  harness config (`clauses`, `encode`, `users_limit`), no ETL writer constructs it, and the
  readers in `layout.py` take paths. Writer / reader parity is `validate_layout` over the
  paths, not a shared path object.
- **`write_query_set` / `merge_prep_log` and the `retry_download` / `checksum` /
  `year_to_bucket` dedup (review §1.11) are not done**: the ETL scripts moved verbatim.
  They land with the first E-phase loader that needs them.
- **The ETL CLIs stay `argparse`**, mounted on `eval-data` as pass-through subcommands
  (`main(argv)`); `eval-data <name> --help` is argparse's help. Rewriting seven parsers to
  click would have changed ~600 lines of command wiring for no behaviour.
- **`training.encode`'s cache key is unchanged** (`ckpt_mtime`, `max_seq_length`); §5.2's
  `users_limit` is not in it because the cache holds the full split and the limit is
  applied after loading.
- **`SCHEMA_VERSION` 1 → 2** (not in the plan): the clock fields of §11.3. Resume is
  unaffected (the key does not carry it); `c4_gate.py` reads schema-1 `env.sm_mhz`.
- **`validate_layout`'s sequential shape asks for `item_id_map.json` + `test.parquet`
  only** — `eval-data fetch` ships no `train` / `val`, and the harness reads neither.
- **The old `test_algos.py` layer-level `k` tests** (`PostfilterKNN` etc. called directly)
  are dropped: the library's `test_boundary.py` owns "k mutation without re-registration"
  and those names are outside **X** §2. The harness keeps the check on what `build()`
  returns, every `(algo, filter_kind)` on the torch path.
- **`retrieval/algos/`, `retrieval/cli/`** were untracked `__pycache__` leftovers in the
  main checkout only; a fresh worktree has nothing to delete.
- **The gate cell ran 6 cells, not 2** (no `--seed 0` narrow); 50 min of GPU instead of
  17, and four extra records.
- **The `retrieve.layers` / `retrieve.kernels` shim** (L D10 scheduled its deletion here)
  is untouched — the library belonged to the L4 worker today. Needs its own slot.
- **Not run:** `bench campaign` on the GPU (the CPU one-child test only); any `eval-data`
  ETL body (unchanged code, no raw data staged); `train sasrec` beyond `--help`; `bench
  upload` (dry-run logic exercised by no test); `official` through the new `run.perf`
  plan-cache path on the GPU (CPU test with a stand-in module only — the gate cell was
  `triton`). `.gitignore` gained `evaluation/data` (the symlink) and the C5 logs.

