# Architecture review — `evaluation/` harness v2 (2026-09-06)

> **Status:** review of `dev/c1-harness-v2` at `595869d` (C1–C3 as built) plus the untouched
> `eval_datasets/` and `training/`, and the E1/E2 loaders on `dev/e1-yfcc` / `dev/e2-pubmed`
> (both branched from `c75b481`, *before* the harness rewrite). Read-only; every claim below
> was checked against the code, not the docs — where the two disagree that is itself a
> finding. Plan under review: [evaluation-harness-v2.md](evaluation-harness-v2.md) (H);
> as-built doc: [../system/evaluation.md](../system/evaluation.md). What ran here: the CPU
> suite (`CUDA_VISIBLE_DEVICES=""`, durations in §4), `scripts/check_doc_links.py` (0 broken).
> `ruff` is not installed in this worktree's venv, so lint was not re-run. No GPU code ran.
> Effort estimates are focused hours (h) / days (d); risk is to *existing* behaviour.

## (a) Executive summary

The rewrite does what H §3 set out to do. The module map is the one in H §3.1, every item on
H §4's "do not abstract" list is honoured (the only classes are `Dataset`, `Job` and the five
`nn.Module` wrappers), the process model is H §8.2 K, and the measurement protocol of H §2 is
implemented rather than approximated — event-per-call windows, three windows with a spread
flag, graph capture with the skip *and* launch-count assertion, `Σ buffers` memory, one
quality pass at `k_max`, oracle blob v4 with `target_in_filter` / `pass_rate`, provenance with
a library-subtree hash, records appended with `fsync`. The 1.8× code overrun is substance,
not framework: docstrings (≈ ⅓), `config.py`'s validation messages, and `run.py`'s record
assembly; there is no layer to delete. The 4.9× test overrun is partly one-off (locking the
old harness's numbers and cell sets, which become redundant once C4 and A1's golden agree)
and partly fixtures duplicated across files. Cut ≈ 300 test lines after C4, no code.

Five changes, in value order:

1. **Provenance can lie about itself: `dirty` and `code_version` (before C4, 1 h, low risk).**
   `bench.provenance` sets `dirty = bool(git status --porcelain)` over the whole tree
   (`bench.py:119`), and the JSONL the harness writes under `evaluation/results/<suite>/` is
   *not* gitignored (only `_logs/` and `_parity/` are). So the first child of a campaign
   records `dirty: false` and every later one `dirty: true` — and H §8.2 F wants `report.py`
   to *refuse* dirty rows. Conversely `code_version` is `git rev-parse HEAD:retrieve/src/retrieve`
   (`bench.py:75`): an uncommitted kernel edit leaves it unchanged, so `--resume` will skip a
   cell measured with different kernel code. Fix: scope `dirty` to the library subtree
   (`git status --porcelain -- retrieve/src/retrieve`), keep a separate `repo_dirty` with
   `--untracked-files=no`, and make `code_version` fall back to the existing `files:` hash
   whenever the subtree is dirty. §2.1.
2. **The oracle fingerprint omits the item side of the predicate (before C4, 1 h, low).**
   `oracle.fingerprint` hashes `item_embs, queries, targets, qa_sweep, clauses, k_gt`
   (`oracle.py:52`) — never `item_attrs` or `clause_is_reverse`. Regenerating
   `item_attrs_narrow.pt` at the same shape (every E-phase re-ETL, the YFCC `--max-tags` cap
   change) silently reuses a stale oracle, and the docstring claims the opposite. Hash both
   tensors (full bytes, once per `(dataset, dim)`; ≈ 1 s). Do it before any v4 blob exists. §2.2.
3. **Narrowed runs are recorded as done (before C4, 1 h, low).** `--k`, `--bs` and `--mode`
   *replace* the suite's lists (`config.py:298–300`, `cli.py`) but are in neither the key nor
   the status: a `bench run --mode eager --k 100` during C4 iteration writes `status: ok`, and
   the next full run skips the cell. Set `status: partial` whenever `ks`, `batch_sizes` or
   `modes` differ from the suite's; and exit non-zero when `load_matrix` yields zero jobs
   (today a `--sweep` typo exits 0 with `{}`). §2.4.
4. **Held-out recall on filter cells uses unreachable targets (before C4, 0.5 h, low).**
   `run.quality` restricts *rows* to `target_in_filter` but scores every target of the row
   (`run.py:228`), including those the mask excludes. Goodreads targets are lists (all
   test-period items), so its filter-cell held-out recall is biased down. The blob already
   carries `targets_in_filter [U, T]`; mask targets to `-1` and set `nt` to the count. §2.3.
5. **A 24-hour campaign has no watchdogs (after C4, before D1, 0.5 d, low).** `subprocess.call`
   has no timeout (`cli.py:168`); a sticky CUDA error (illegal access, device-side assert)
   makes every remaining cell of the child a fast `failed` record instead of ending the
   process; `torch.save` / `np.savez` of the oracle blob, encode cache and parity file are not
   atomic (a crash mid-save leaves an unreadable file at the fingerprint path, and
   `load_or_build` does not catch it); `read_keys` raises on a truncated trailing line. §2.9, §5.

**C4 verdict: not as-is — fix 1–3 first (≈ 3 h, no protocol change), take 4 while there.**
The gate itself (oracle quality vs golden, graph within 5 %, skips = 0, jaccard) runs on the
current code; but C4 is an iterate-and-rerun day, and 1 and 3 corrupt exactly the resume and
provenance data that day relies on. Run the gate with `--seed 0`: the `filter` suite's
headline rule gives `c0_genre` at d128 three seeds, tripling the gate's cells for no gate value.

## (b) Findings

Tags: **[C4]** before C4 · **[D1]** after C4, before D1 · **[E]** before the E-phase loaders
land on the harness · **[later]**.

### 1. Architecture

**1.1 H §3 / §4 conformance — matches.** Modules, ownership and the two surviving class kinds
are as planned; `PATHS` collapses same-code backends (`config.py` `seen`), `official` is
SilverTorch-only, Triton-filtered, `capturable=False` (`algos.py:314`) and, until B1, a
`NotImplementedError` recorded as `failed` (`algos.py:296`). One deviation from §4's letter is
deliberate and right: "no exception swallowing" is overridden by §8.2 D's `status: failed`
records (`run.py:458, 547`). `run(device=)` exists for tests only; fine.

**1.2 `inputs` / `assets` are context bags with a lowercase name [later, 1 h].** H §4 allows
"one local dict"; the as-built has two (9 keys each) threaded through six functions
(`sweep_assets`, `build_module`, `quality`, `perf`, `query_pool`, `exact_filter`). Not a
violation, but the field lists live only in `data.py`'s docstring and `sweep_assets`'s return.
Two `TypedDict`s (`Inputs`, `Assets`, ≈ 20 lines) give the reader and the type checker the
schema without an object model. Do not turn them into classes.

**1.3 Budget: where the 2,657 lines went, and what to cut.** `run.py` 582 = the loop (200 in
`run()` alone, §3.1) + quality/parity/perf + JSONL; `config.py` 365 = a strict validator with
named errors (worth it: a typo is a `ConfigError` naming the file, not an empty JSON);
`bench.py` 349 = the §2.5 protocol; `data.py` 291 = two loaders. Cuttable code: none worth
the churn. Cuttable tests (≈ 300 lines, after C4 passes against golden): `test_config.py`'s
`_OLD_FILTER` / `_OLD_ALGOS` tables and the two "matches the deleted d128 YAML" tests (golden
JSON supersedes them); `test_algos.py`'s five per-layer `k`-slice tests (the wrapper-level
parametrised test exercises the same layers through `build`); the second copy of the tiny
dataset writer (§4.2). Keep `test_metrics.py`'s verbatim old reference — it is the only
thing that ties `metrics.py` to the thesis numbers.

**1.4 Module boundaries — two misplacements [D1, 0.5 h].** `resume_key` and `KEY_FIELDS` live
in `oracle.py` (`oracle.py:233–250`) and have nothing to do with the oracle; `read_keys` in
`run.py` rebuilds the key from `KEY_FIELDS` while `Job.key` is the authority — H §3.1's
"the field list would be written three times" is now true again (`Job.key`, `KEY_FIELDS`,
the system-doc table). Move `resume_key` next to `Job` in `config.py` and derive
`KEY_FIELDS = tuple(Job.key(...))`'s keys. `oracle.py` also imports the private `bench._git`.

**1.5 Cell record and resume key — sound, two holes.** The key is
`Job.key(params) + code_version` as canonical JSON; the last record per key wins; `failed` /
`partial` re-run. Holes: (i) `bloom` (`m_bits`, `k_hash`) is merged into build params at
`run.py:174` but is not in `params`, so editing `bloom:` in `suites.yaml` does not invalidate
bloom cells [D1, 0.5 h — put them into `params` on bloom cells; they *are* build params];
(ii) the narrows of item 3 above.

**1.6 Campaign process model — as specified, one omission [D1, 1 h].** One
`python -m retrieval.cli run` per `(dataset, dim, algo, backend)`, per-child log with the
command line first, summary line, worst-rc exit. `bench campaign` cannot narrow by
`filter_kind` / `sweep` / `seed` / `expected-sm-mhz`, so D1's "headline sweeps only" and
"seeds {0,1,2} on headline cells" reruns need `bench run` by hand per group. Add pass-through
of those five narrows. Also: the parent imports torch anyway (`config → algos → retrieve`), so
`cli.py:73`'s "torch import only when running" comment is moot.

**1.7 Parity spill — correct wiring, wrong docstring, unbounded size.** `.npz` of ids+scores
keyed by `sha1(key − backend)`, compared by `jaccard_at_k` + `score_max_abs_diff`, cleaned per
algo group by the campaign. The `run.py:24` docstring says a missing reference records
`parity: "no_reference"`; the code (`run.py:254–277`) writes a *new* reference labelled by
whatever backend ran — so with `--resume` after triton is done, `torch` becomes the reference
and `official` is compared against torch. Acceptable, but say so. Size is
`n_kept × k_max × 8 B`: 80 MB on a 10k-user prefix, **800 MB on YFCC's 100k queries** with no
`users_limit` (the docstring says ≤ 40 MB). Cap the spill at the first 10k kept rows — it is
a wiring check [E, 0.5 h].

**1.8 Oracle cache and fingerprint.** Blob v4 with the hash in the name is the right design
(portable, never read stale *for the inputs it hashes*). Beyond item 2: the 64-row linspace
sample catches whole-tensor changes but not row-subset edits; for `item_attrs` (the tensor
ETL edits piecemeal) hash the full bytes. `load_or_build` does not catch a corrupt file
(§5.3). The oracle materialises `item_embs.t().contiguous()` (`oracle.py:136`) — a second
full copy of the index on device — and a `[64, N]` fp32 score block; fine at 3 M × 256,
impossible at PubMed's 36 M × 768 (§2.8).

**1.9 Filter modules keyed by backend — right key, wrong cache level [later, 1 h].**
`build_filters` is called inside `sweep_assets` (`run.py:130`), i.e. once per
`(filter_kind, sweep, filter backend, k_max, bloom)`, although the module depends on none of
`sweep` / `k_max`. Every sweep re-registers the `ExactAttributeFilter` (and, on bloom cells, a
`BloomFilter` over 3 M items *plus* a fresh exact filter for the oracle). Seconds per sweep,
not wrong; hoist to the `(dataset, dim, filter_kind, filter backend, bloom)` level.

**1.10 Exact-algo gate.** `EXACT_ALGOS = (linr_v1_filter_mask, linr_v2)` at
`recall_oracle@k_max ≥ 0.99`, recorded as `failed` then raised (`run.py:521`). In a campaign
the child dies with rc = 1 and the loop continues with the next group — "kills the run" means
"kills the group"; the doc should say that. Fine.

**1.11 `eval_datasets/` loaders — the shape, and what YFCC / PubMed reveal [E].** Each loader
is an `argparse` CLI (`download → convert → prep → attrs [→ encode]`) that writes the harness
layout by hand; `common.py` shares three numeric helpers. The E1/E2 loaders copy the shape
faithfully and, in doing so, show what is *not* shared:

- **The layout contract is implicit and re-derived per loader.** `data._pre_encoded` asserts
  `query_emb.pt` rows == `heldout.parquet` rows == `eval_split.parquet` rows, 1-indexed
  `item_id`, `-1`-coded attrs, nomic prefixes in `*.meta.json`. arXiv, YFCC and PubMed each
  hand-build `heldout.parquet` / `eval_split.parquet` with different extra columns
  (`arxiv_id`; `query_row, n_query_tags`; `pmid`). **PubMed breaks the contract**:
  `cmd_queries` drops rows with empty titles and may append NFCorpus rows
  (`pubmed.py:958–1040`), so `query_emb.pt` will not align with `heldout.parquet` and
  `load_inputs` will raise `query_emb rows != heldout rows`. YFCC deliberately omits the
  `meta.json` sidecars and relies on `_assert_prefixes`'s *warning* path (`data.py:52`,
  `yfcc.py:514`) — an implicit API.
- **Multi-target text queries have no home.** `heldout.parquet` carries one `item_id`
  (`data.py:88` builds `[U, 1]`), while the SASRec path supports `[U, T]`. PubMed's NFCorpus
  qrels and E3's citation relevance are multi-target.
- **Duplicated utilities**: year bucketing ×3 (`arxiv.py:86`, `goodreads.py:731`,
  `pubmed.py:165`), retrying downloader ×3 (`goodreads.download_one`, `yfcc._download_one`,
  `pubmed._fetch_one`), checksum ×3, `prep_log.json` merge ×2 + inline ×2, the
  `print("STEP …", flush=True)` idiom in four loaders vs loguru in `yambda.py`.
  `goodreads.ROOT = Path.home()/"datasets"/…` ignores `hf_io.data_root()`; the other four use
  `raw_dir()`.
- **Both branches predate the harness** (merge base `c75b481`): `yfcc.py` documents
  `loaders.assert_arxiv_prefixes()`, which no longer exists. They must rebase onto C3.

Verdict: not over-shared — `common.py` is right-sized — but the *contract* is the missing
shared thing, not a base class. Recommend `eval_datasets/layout.py` (≈ 80 lines):
`write_query_set(output, item_ids | item_id_lists, qa, extra_cols)`, `merge_prep_log`, and
`validate_layout(data_dir, content_dir) -> list[str]` (the same checks `data.py` makes, plus
row alignment), exposed as `bench check --dataset X` so a loader author sees the harness's
verdict without a GPU; `retry_download` / `checksum` / `year_to_bucket(edges)` into
`common.py`; an optional `item_ids` list column in `heldout.parquet` read by
`_pre_encoded` into `[U, T]`; make the prefix policy explicit (`prefix: null` in `meta.json`
= "no prefix", a *missing* sidecar an error). Then rebase E1/E2.

**1.12 `training/` coupling — now bidirectional [later, 0.5 h].** `encode.py` is documented as
the one file importing `training.*`, and it is; but `training/evaluate.py:14` now imports
`retrieval.metrics`, so the packages depend on each other (no runtime cycle: `metrics.py`
imports only torch). Its docstring still says "No `retrieve` framework". Either accept and
document the shared `metrics.py`, or give `training` its own 30-line copy. Not worth a third
package.

**1.13 HF I/O.** `hf_io.py` is a good registry. `upload.py` re-implements folder upload with
*two* ignore semantics (`IGNORE` globs for HF, `files()`'s `_`-prefix walk for the listing,
`upload.py:16–19`) that can disagree, and it will mirror the old `results/{arxiv,goodreads,
yambda,deep_sweeps}` campaign dirs until they move to `results/archive/` — which, not being
`_`-prefixed, would be mirrored too. Use `hf_io._list_files_filtered` with one pattern list
and an explicit `archive/` exclusion [D1, 0.5 h]. `hf_io`'s docstring still names the
pre-rename `data/` package.

### 2. Measurement-correctness risks (code, not numbers)

**2.1 `dirty` / `code_version` [C4]** — summary item 1. Sketch (`bench.py`):

```python
def code_version() -> str:
    sub_dirty = bool(_git("status", "--porcelain", "--", LIB_SUBTREE))
    tree = None if sub_dirty else _git("rev-parse", f"HEAD:{LIB_SUBTREE}")
    return tree or _files_hash()          # "files:<sha256>" — disjoint namespace, already there
# provenance(): "dirty": sub_dirty, "repo_dirty": bool(_git("status","--porcelain","--untracked-files=no"))
```

**2.2 Oracle fingerprint [C4]** — summary item 2. Add `item_attrs` and `clause_is_reverse` to
`fingerprint(...)` (full `sha256` of `.numpy().tobytes()`; compute once in `load_inputs` and
pass the digest down so per-sweep calls stay < 10 ms). No blob exists yet, so no migration.

**2.3 Held-out targets on filter cells [C4]** — summary item 4. In `run.quality`, for
`blob is not None`: `t = t.masked_fill(~blob["targets_in_filter"][sel][m].to(device), -1)`,
`nt = (t != -1).sum(1)`. Record `n_targets_in_filter` next to `n_queries_heldout`.

**2.4 Narrows recorded as `ok` [C4]** — summary item 3. In `run()`, `partial` iff
`skip_quality or skip_perf or set(modes) != set(MODES) or ks/batch_sizes narrowed`; the
narrows are visible because `Job.ks` / `Job.batch_sizes` carry the override — compare with
the suite's lists (pass them through `Job`, or add `narrowed: bool` to `Job`).

**2.5 Mutation order can fail a cell on the deep suite [C4, 1 line].** `perf` leaves
`module.k` at the last `k` (`run.py:317`); the next cell calls `set_query_params(n_probe=…)`
(`run.py:474`) *before* `quality` resets `module.k = k_max` (`run.py:513`).
`Silvertorch.set_query_params` validates `n_probe · max_cluster ≥ k` against the stale `k`
(`algos.py:200–212`), so a small `n_probe` after a large-`k` perf pass can raise for a
combination that is legal at `k_max`'s order of application. Set `module.k = k_max` first.

**2.6 SASRec encode cache key is incomplete [D1, 0.5 h].** `{ckpt_mtime, max_seq_length}`
(`data.py:102`); a regenerated `test.parquet` (E4's ETL, a goodreads re-prep) or a changed
`item_id_map.json` reuses stale queries with no error. Add `test.parquet` size + mtime and
`num_items` to the key.

**2.7 `bloom_fp_rate` describes the standalone filter, not the measured index [D3].**
`oracle.pass_counts(filter_mod=BloomFilter(triton|torch))` — for `silvertorch` bloom cells
the index has its own fused bloom (same params, same hash today) and for `official` a
*different* hash (O's finding). Name the field `bloom_fp_rate_standalone` now, and have D3
compute the fused bloom's rate from the module where it exposes one.

**2.8 The text loader keeps fp32 items on device — PubMed cannot load [E, 1 d].**
`_pre_encoded` does `F.normalize(item_embs.float())` on device (`data.py:85`): 36 M × 768 × 4 B
= 110 GB, plus the oracle's transposed copy. Every layer quantises from what it is given
(fp16 for V1/V2/V3 — `PrefilterKNN.register_index` casts to fp16 itself — int8 for V4 and
SilverTorch), so hand them fp16 and let the oracle run its `q @ Eᵀ` in fp32 *chunks* over the
fp16 matrix (it already batches queries; add item chunking as `training/evaluate.py` does).
Also size the oracle batch by a byte budget rather than the fixed 64 (`oracle.py:124`).

**2.9 Sticky CUDA errors [D1, 1 h].** After an illegal memory access the context is dead;
`except Exception` (`run.py:547`) records the cell and continues, so the group's remaining
cells all fail in seconds with the same traceback and `--resume` will happily re-run them
later. Re-raise when the message contains `CUDA error` / `illegal memory access` /
`device-side assert` (after recording), so the child exits and the campaign moves on.

**2.10 Timing semantics to state, not change.** Per-call CUDA-event ms spans the GPU
timeline between the two `record`s; when the GPU is idle waiting for the host (eager, bs = 1)
the interval *includes* launch latency and `host_gap_ms ≈ 0`. That is the closed-loop
latency the papers report, but `evaluation.md` should say it so nobody reads `median_ms` as
kernel time (`--profile` is the kernel-time view). Graph capture is per `(bs, k)` — 9
compiles per cell at H §7's ≈ 20 s each, i.e. ≈ 3 min of the "2 min per cell" budget; record
`compile_s` per graph entry [D1, 0.5 h] so D1's cost model is measured, not estimated. The
`set_sync_debug_mode("warn")` probe (`bench.py:240`) prints to stderr and leaves nothing on
the record; catch the warnings and record `hidden_syncs: int` [D1, 0.5 h]. Warm-up (50),
`N = clamp(2 s / est, 1000, 5000)`, three windows, median-median window, seeds via
`bench.setup(job.seed)` + a per-pool `Generator`, no L2 flush (by design, H §2.5): all as
specified. Clocks are sampled between cells with the GPU idle; without `-lgc` that reads the
idle clock and trips the 5 % drift warning spuriously — with locked clocks (the runbook) it
is right; note it.

**2.11 `users_limit` single site — confirmed.** `data.py:169` is the only trim, applied
after the `eval_split` row-count check against the full split (the A1 bug is fixed in v2).
`k_gt = max(ks)` per suite; a suite with a larger `k_max` gets its own blob — fine.

**2.12 Tie handling.** Oracle `-inf` ties → `-1` (`oracle.py:141`); `jaccard_at_k` is
set-based; `per_row`'s `argmax` picks the first hit (documented torch behaviour); C4's
"ids equal up to ties" is the right gate statement. Pass-rate arithmetic (`pass_counts /
n_items` over kept rows; rows with 0 survivors count in `pass_rate` but not in
`oracle_rows`) and the FPR definition `(bloom − exact)/(N − exact)` over rows with ≥ 1
negative are correct and unit-tested.

**2.13 Provenance completeness.** Present: gpu, driver, cuda, torch, triton, commit, dirty,
branch, code_version, host, python, started, config_sha, sm/mem/max clocks, power limit,
`clocks_locked`, drift. Missing and cheap [D1]: `uv.lock` sha (the environment), the
`retrieve` package version, the official package sha once B1 lands, `torch.backends.cuda.
matmul.allow_fp16_reduced_precision_reduction` (affects LinR V1's cuBLAS fp16 sums —
record it rather than assume the default).

### 3. Code design and readability

**3.1 `run.run` is 200 lines with four nested caches and a 90-line `try` [D1, 2 h, low].**
Split by the things it already names: `_todo(job, existing, code_version, resume)`;
`_inputs_for(job)` / `_assets_for(job)` (the two keyed caches, as a tiny `_Cache` closure or
two locals — no class); `_cell(job, params, module, inputs, assets, opts) -> (rec, samples)`
containing the try body with `stage` as its local; `run()` becomes the 60-line loop H §3.1
promised. `perf()` and `quality()` are already the right size.

**3.2 Booleans and options.** `run(resume, skip_quality, skip_perf, profile)` is within H §4's
tolerance; if 3.1 lands, bundle them as one `opts: dict` (or a `RunOptions` `NamedTuple`) so
`_cell` does not take nine arguments.

**3.3 Error handling — what is swallowed.** Deliberate and recorded: cell / build exceptions
(`status: failed` with traceback and `stage`). Silently `None`: `nvidia-smi` and `git`
failures (`bench._git`, `_nvidia_smi`) — correct, they land as `null`. Wrong: `_assert_prefixes`
downgrades a *missing* sidecar to a warning (`data.py:52`) — YFCC now depends on it (1.11);
`load_or_build` does not catch a corrupt blob (5.3); `read_keys` (5.4).

**3.4 CLI ergonomics [C4/D1, 1 h].** `bench run --out results` is relative to the *CWD*
(`cli.py:92`) while `bench campaign --out results` is relative to `evaluation/`
(`cli.py:121`): run from the repo root, they write to different places. Resolve both against
`EVAL_DIR`. `--out` (a directory) next to `--output` (a file) is a trap — rename the latter
`--jsonl`. Zero jobs → exit 0 (2.4). `--expected-sm-mhz 0` meaning "none" is fine but
undocumented in `--help`. `bench report` exiting 2 with a clear message is right.

**3.5 Config validation.** Good: unknown *and* missing keys in one message naming the file,
`build:`/`query:` misplacement named, `--k 0` rejected, `disabled:` logged once. Nit:
`_DATASET_KEYS |= {...}` / `_SUITE_KEYS |= {...}` on the next line (`config.py:35, 38`) read
like leftovers from a line-length fight; write the sets in one literal. `is_valid_combo`'s
`n_lists` default of `1 << 30` (`algos.py:246`) lets `{n_probe: 2048}` with the wrapper's
default `n_lists = 1024` through to a library `ValueError` at build — use the wrapper default.

**3.6 Stale docstrings / docs (all cheap, [C4] since C4 will read them).** `encode.py:7`
cites `queries_cache.py` (deleted); `run.py:24` `no_reference` (1.7); `data.py:11` "old
`loaders.py`" is fine as history; `training/evaluate.py` "No `retrieve` framework" (1.12);
`hf_io.py` module docstring names `data/arxiv.py`; `docs/system/datasets.md:200` cites
`load_filter_assets` (deleted) and `:308` says "configs under `evaluation/config/<name>/`"
(v2 is one YAML per dataset); its "On-disk layout" block shows `data/arxiv/papers/` while
`config/arxiv.yaml` uses `data/arxiv-papers`. `evaluation.md` is accurate against the code
except the `no_reference` wording and the "kills the run" nuance (1.10).

**3.7 Naming.** `SilverTorch` vs `QuantizedIVF`: no `QuantizedIVF` anywhere in `evaluation/`
or `docs/system/` — the rename is complete. `algo` is used consistently (no `impl`). Nits:
the wrapper class `Silvertorch` differs from the library's `SilverTorch` by case only —
`SilverTorchAlgo` (or `IVFInt8`) would read; `k_gt` (oracle) and `k_max` (run) are one number
under two names; `filter_mod` / `filter` / `filters` for three different things in `data.py`.

**3.8 Type hints.** `-> dict`, `inputs: dict`, `assets: dict` throughout `run.py` (1.2);
`Job.query: tuple[dict[str, Any], ...]` holds dicts inside a `frozen=True` dataclass, so
`Job` is not hashable despite looking like a value (`config.py:69`) — nobody hashes it today;
either drop `frozen` or freeze the params to `tuple[tuple[str, Any], ...]`.

### 4. Tests

**4.1 What 1,699 lines buy.** Real guarantees: metrics running sums == old per-row means to
1e-9 (fixed and ranked); `PATHS` == architecture.md's dispatch table (parsed, so the doc
cannot drift); `k`-slice invariance of every layer and wrapper on the torch path; matrix
expansion counts, the build/query split, seeds rule, narrows, `ConfigError` texts; the real
d128 cell sets vs the deleted YAMLs; the loader's normalisation / `-1` shift / row checks /
`users_limit`; oracle padding, v4 arithmetic, fingerprint-in-name, FPR; `stats` vs numpy;
end-to-end record schema, resume, `code_version` invalidation, failed-and-continue, the
gate, the parity spill, a real one-child campaign. That is the right set for a CPU-only
authoring box.

**4.2 Duplicated fixtures [D1, 0.5 h].** `conftest.write_tiny_dataset` (N=24, U=8) and
`test_data._write_dataset` (N=12, U=6) are the same 20 lines; `test_algos._data` and
`test_oracle._v4_inputs` are legitimately different. Parametrise the conftest writer
(`n, u, doc_prefix, n_split`) and return the `Dataset` too.

**4.3 Slow tests [D1, 1 h].** 82 passed in 319 s here (table in §4.6). Three sinks account
for ≈ 290 s: (i) `test_cli_run_campaign_and_report`, 109 s — two real children each importing
torch + `retrieve` from cold, plus the in-process run; (ii) seven separate `SilverTorch`
builds on the torch path across `test_algos.py` (`test_wrapper_k_setter_slices[silvertorch-*]`
3 × 22 s, `test_index_bytes_includes_filter_submodule` 20 s,
`test_layer_silvertorch_k_not_baked` 9–17 s, `test_silvertorch_query_params_revalidate` 5 s)
≈ 130 s — the build, not the assertions, is the cost (the LinR wrappers take 0.7–3 s); (iii)
`test_run.py` ≈ 43 s, of which `test_failed_cell_is_recorded_and_the_loop_continues` is 29 s
(two full runs over four cells with V4's `_int_mm` on CPU). Without losing a guarantee: build
each SilverTorch mode once per session (`@pytest.fixture(scope="session")`, three builds, then
`k`-slice / `index_bytes` / `set_query_params` assert on the same module — every one of these
tests is read-only on the module after `.k` is restored) and pass `n_iter=2`; give the
campaign test one algo (one child — the second proves nothing the first does not) and
`--skip-quality` as well as `--skip-perf`; in `test_failed_cell…` narrow to `sweeps=[NONE_SWEEP]`.
Expected: ≈ 100 s. The remaining 60 tests take < 2 s each and are fine.

**4.4 Untested paths that have a CPU shape (no emulation needed).**
- The graph null entry: `run.run(modes=("eager", "graph"))` on CPU must yield entries with
  `reason: cuda_unavailable` and every `PERF_STAT_KEYS` null — the exact code path the
  `official` backend takes on the A100. Not covered (`test_run` is eager-only).
- **The §8.2 A path through `run`** — one build, several `query:` combos applied by
  `set_query_params`, some already `ok` on resume — is untested end to end; `test_algos`
  checks the setters in isolation only. This is the deep suite's whole cost model.
- Bloom cells end to end (`bloom_fp_rate`, `exact_filter` on bloom, the `bloom` merge into
  `silvertorch` params) — the `e2e` suite has `filter_kinds: [none, clause]`.
- `silvertorch`, `linr_v2`, `linr_v3` through `run` (only V1 and V4 run e2e).
- A failing campaign child (rc ≠ 0 recorded, loop continues); clock drift via a monkeypatched
  `bench.clocks`; a truncated trailing JSONL line; the narrow-status rule of 2.4.

**4.5 Layout and naming.** Files mirror modules; names read as sentences. Smells:
`test_parity_spill_compares_the_second_backend` mutates the module-global `FILTER_BACKEND`
dict and relabels a job `backend="torch2"` — it works, but a second real backend on CPU does
not exist, so say so in a comment rather than in a monkeypatch; the architecture.md parser in
`test_algos` will need a touch when B4 renames the column (it already anticipates
`official`/`cuda`). H §9 reports 82 passed — matches the collection here (§4.6).

**4.6 Durations (measured, CPU, this box).** See the table at the end of this file.

### 5. Bad patterns

- **5.1 Mutable defaults / bare excepts / globals.** None found; `except Exception` ×2 is
  deliberate and annotated. Process-wide state (`_dynamo.reset`, matmul precision,
  sync-debug mode) is acceptable under one-process-per-group.
- **5.2 Path handling.** `cli.py` `--out` inconsistency (3.4); `EVAL_DIR = parents[1]` in
  `cli.py` / `upload.py` and `ROOT = parents[2]` in `bench.py` assume the source layout — in
  an installed wheel `code_version` silently becomes `files:` (acceptable, disjoint).
- **5.3 Atomic writes [D1, 1 h].** `append_record` (one `write` + `fsync`) is fine for the
  ~5 KB record; the samples line is ~100 KB and can be torn by a crash. `torch.save` of the
  oracle blob (`oracle.py:225`) and encode cache (`data.py:143`) and `np.savez` of the parity
  file (`run.py:269`) are not atomic — write to `path.with_suffix(".tmp")` and `os.replace`.
- **5.4 JSONL on resume [D1, 0.5 h].** `read_keys` (`run.py:113`) raises on a torn last line,
  which blocks resume exactly when it is needed; tolerate (and log) one malformed trailing
  line. Samples are appended *after* the record, so a crash in between loses the vector for a
  cell `--resume` will then skip — append samples first, or write both lines in one call.
- **5.5 Subprocess handling [D1].** No `timeout` on `subprocess.call` (`cli.py:168`); add one
  (e.g. 6 h) and record `rc=timeout` in `campaign.log`. `Ctrl-C` propagates correctly (the
  child shares the process group; the parent's `KeyboardInterrupt` ends the loop).
- **5.6 Serialisation.** `_clean` (`run.py:77`) handles tensors, `np.floating`/`np.integer`,
  non-finite floats, `Path`; it misses `np.ndarray` and `np.bool_` (not an `np.integer`
  subclass) — nothing produces them today, but `--profile` and future perf keys will; add both.
  `allow_nan=False` + NaN→`null` is the right JSON contract and is tested.

## (c) Keep as is

- One appended JSONL record per cell with `fsync`, nested `perf`, samples as a JSONL sidecar
  (parquet cannot be appended; D4 converts) — H §8.1's field consensus, correctly applied.
- `Job` as the only config class; `Job.key(params)` as the record's key block; suites.yaml's
  `build:`/`query:` split and `disabled:`; the `PATHS` table with the logged collapse.
- One process per `(dataset, dim, algo, backend)` and the spill-file parity (with the cap of
  1.7). The child on the same interpreter, not a second `uv run`.
- `bench.latency`: events per call, 3 windows, median-median window, spread/`unstable`,
  IQR + both outlier counts never dropped, `peak_fwd_mib` from the first eager window only,
  no L2 flush. `graph_callable`'s skip *and* launch-count assertion → `NotCapturable` with
  the reason on the record instead of a mislabelled number.
- `index_bytes = Σ buffers` (every layer registers its index as buffers — verified in
  `silvertorch/main.py`, `postfilter_knn*.py`, `prefilter_knn.py`, `one_bit_knn.py`, both
  filters) and `filter_mib` from the submodule.
- Oracle: `-inf` ties → `-1`, exact filter even on bloom cells, `targets_in_filter` /
  `pass_rate` in the blob, fingerprint in the file name (a shippable artifact for F4).
- `metrics.py` as device-side float64 running sums with the old per-row arithmetic verbatim
  and the 1e-9 test against it.
- `_clean` + `allow_nan=False`; `Counter` as `run()`'s return; loguru; `ConfigError` messages.
- `encode.py` as the single `retrieval → training` import; `hf_io.py`'s registry and
  `data_root()`; `common.py` as numeric helpers only; loaders as plain `argparse` CLIs with
  `download/convert/prep/attrs` — no loader framework.
- `test_metrics.py`'s verbatim old reference and `test_algos.py`'s markdown-parsed dispatch
  table: both are the cheapest possible drift detectors.

## (d) Sequencing against the roadmap

| when | what | effort |
|---|---|---|
| **before C4** | 2.1 `dirty`/`code_version` scoping · 2.2 fingerprint gains `item_attrs` + `clause_is_reverse` · 2.4 narrows → `partial`, zero jobs → non-zero exit · 2.3 held-out target masking · 2.5 `module.k = k_max` before `set_query_params` · 3.4 `--out` resolved against `EVAL_DIR` · 3.6 docstring fixes. Run C4 with `--seed 0`. | ≈ 4 h |
| **after C4, before D1** | 5.3–5.5 atomic saves, tolerant `read_keys`, samples-first, child timeout · 2.9 sticky-CUDA abort · 2.6 encode cache key · 1.5 bloom params in `params` · 1.6 campaign pass-through narrows · 2.13 provenance additions, `compile_s`, `hidden_syncs` · 3.1 `run.run` split · 4.2/4.3/4.4 tests (fixture dedupe, graph-null path, multi-combo build, bloom e2e, failing child) · 1.4 `resume_key` to `config.py` · 1.13 `upload.py` on `hf_io` filters, `results/archive/` move · 2.7 rename `bloom_fp_rate_standalone`. | ≈ 2 d |
| **D4 `report.py`** | Read last record per key, honour `partial`, enforce subtree `dirty` (not `repo_dirty`), tolerate a torn samples line, write `flat.csv` first (H §8.2 G). Nothing in the record schema blocks it. | in D4's 1.5 d |
| **E-phase loaders (E1–E4, before E5)** | Rebase E1/E2 onto C3 · 1.11 `eval_datasets/layout.py` + `bench check` · multi-target `item_ids` column · explicit prefix policy · 2.8 fp16 items on device + chunked oracle (E2 blocker at 36 M × 768) · 1.7 parity cap · 1.9 hoist filter modules · `goodreads.ROOT` via `data_root()` · dedupe year/download/checksum helpers. | ≈ 2 d |
| **later** | 1.2 `TypedDict`s · 1.12 training/metrics ownership · 3.7 naming nits · 3.8 `Job` hashability · 1.3's test trims (after golden agrees). | opportunistic |

## Appendix — §4.6 measured test durations

`cd evaluation && CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest retrieval/tests/ -q
--durations=25` on this box (Linux, torch 2.10.0+cu128 running on CPU): **82 passed in
319.05 s**. Top entries (call phase):

| s | test |
|---:|---|
| 109.4 | `test_cli.py::test_cli_run_campaign_and_report` |
| 29.3 | `test_run.py::test_failed_cell_is_recorded_and_the_loop_continues` |
| 23.2 / 22.8 / 22.2 | `test_algos.py::test_wrapper_k_setter_slices[silvertorch-{clause,bloom,none}]` |
| 20.3 | `test_algos.py::test_index_bytes_includes_filter_submodule` |
| 17.4 / 10.2 / 9.6 | `test_algos.py::test_layer_silvertorch_k_not_baked[{bloom,none,exact}]` |
| 6.5 | `test_run.py::test_resume_skips_ok_cells_and_code_version_change_reruns` |
| 5.1 | `test_algos.py::test_silvertorch_query_params_revalidate` |
| 3.5 | `test_run.py::test_end_to_end_records` |
| 2.6–3.0 | `test_algos.py::test_wrapper_k_setter_slices[linr_v{2,3}-*]` (5 cases) |
| 2.4 | `test_run.py::test_parity_spill_compares_the_second_backend` |
| 1.6 | `test_run.py::test_quality_gate_kills_the_run_after_recording` |
| ≤ 1.5 | everything else (`test_bench`, `test_config`, `test_data`, `test_metrics`, `test_oracle`, the LinR V1/V4 cases) |
