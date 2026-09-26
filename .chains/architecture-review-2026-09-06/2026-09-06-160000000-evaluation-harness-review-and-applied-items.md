---
chain: "architecture-review-2026-09-06"
branch: "main"
parent: "2026-09-06-100000000-library-review-and-applied-items.md"
nextStep: "Harness items still open at 2026-09-26 HEAD (checked against the code): 2.5 module.k = k_max before set_query_params (run.py still calls set_query_params first), 3.4 bench run --out resolved against the CWD (campaign resolves against EVAL_DIR), 2.6 encode-cache key, 1.5 bloom params in params, 2.7 bloom_fp_rate naming, 2.13 provenance additions, compile_s / hidden_syncs, 5.6 np.ndarray / np.bool_ in _clean, 2.8 fp16 items + chunked oracle (E2 blocker)."
created: "2026-09-06T16:00:00Z"
---

# `evaluation/` harness v2 architecture review, 2026-09-06

Source: `docs/plans/architecture-review-2026-09-06-evaluation.md`. Review of `dev/c1-harness-v2` at `595869d` (C1-C3 as built) plus `eval_datasets/`, `training/` and the E1 / E2 loaders (branched from `c75b481`, before the rewrite). Read-only; CPU suite and link checker run; no GPU.

## Summary
The rewrite does what H §3 set out to do; every item of H §4's "do not abstract" list honoured; H §2 implemented rather than approximated. The 1.8x code overrun is substance (docstrings ~1/3, validation messages, record assembly); the 4.9x test overrun partly one-off.

Five changes by value:
1. Provenance lied about itself: `dirty` over the whole tree (results JSONL not gitignored, so every child after the first was dirty) and `code_version` unchanged by uncommitted kernel edits (resume would skip a cell measured with different code). Fix: `dirty` scoped to `retrieve/src/retrieve`, separate `repo_dirty`, `code_version` falls back to a `files:` hash when the subtree is dirty. Applied `64c1508`.
2. The oracle fingerprint omitted `item_attrs` and `clause_is_reverse` (regenerated attrs at the same shape reused a stale oracle). Fix: `attrs_digest` full bytes once per `(dataset, dim)`. Applied `1191e6d`.
3. Narrowed runs (`--k`, `--bs`, `--mode`) recorded `ok`, so the next full run skipped them; a zero-job selection exited 0. Fix: `Job.narrowed`, `status: partial` + `partial_reasons`, exit 1 on zero cells. Applied `2d53edc`.
4. Held-out recall on filter cells scored unreachable targets (goodreads targets are lists). Fix: mask targets the exact filter excludes; record `n_targets_in_filter`. Applied `26b4d78`.
5. No watchdogs for a 24-hour campaign: no child timeout, sticky CUDA errors turned every remaining cell into a fast failure, non-atomic saves, `read_keys` raising on a torn line. Applied `23228c5` (`--timeout`, rc 124; `STICKY_CUDA` re-raise; `atomic_write`; corrupt blob rebuilt; samples before record; tolerant `read_keys`).

C4 verdict at the time: fix 1-3 first, take 4; run C4 with `--seed 0`.

## Other findings (condensed)
- 1.2 `inputs` / `assets` dicts are context bags without a schema: `TypedDict`s later.
- 1.4 `resume_key` / `KEY_FIELDS` misplaced in `oracle.py`: done at C5 (`bench/records.py`).
- 1.5 bloom `m_bits` / `k_hash` merged into build params but not into `params`, so editing `bloom:` does not invalidate bloom cells: open.
- 1.6 `bench campaign` cannot narrow by filter_kind / sweep / seed: open (D1-a needed a replayed loop for `--seed 0`).
- 1.7 parity spill correct; docstring wrong about a "no_reference" state (fixed); size `n_kept x k_max x 8 B` = 800 MB on YFCC's 100k queries: cap at 10k rows (open).
- 1.8 oracle: full-byte hash for attrs (done), corrupt-file handling (done), a transposed full copy of the index on device (E2 problem).
- 1.9 filter modules rebuilt per sweep: hoist (open, cosmetic).
- 1.10 the exact-algo gate kills the group, not the campaign: documented.
- 1.11 `eval_datasets` loaders: the layout contract was implicit (PubMed's `cmd_queries` dropped empty-title rows and appended NFCorpus rows, breaking `query_emb` / `heldout` alignment; YFCC relied on a missing-sidecar warning); multi-target text queries had no home; utilities duplicated three times (year bucketing, retrying download, checksum); `goodreads.ROOT` ignored `data_root()`; E1 / E2 branches predated the harness. Recommended `eval_datasets/layout.py` + `bench check` + explicit prefix policy (done at C5 and 2026-09-16); helper dedup and multi-target column open.
- 1.12 `training` <-> `retrieval` cycle: resolved at C5 (D5 local metrics).
- 1.13 `upload.py` ignore semantics: rewritten 2026-09-15.
- 2.5 mutation order: `perf` leaves `module.k` at the last k; the next cell calls `set_query_params(n_probe=...)` before quality resets `module.k = k_max`, so a small `n_probe` after a large-k perf pass can raise at validation. Still present at HEAD (`bench/run.py:486` before `:517`).
- 2.6 SASRec encode cache key `{ckpt_mtime, max_seq_length}` ignores a regenerated `test.parquet` or id map: open.
- 2.7 `bloom_fp_rate` describes the standalone filter, not SilverTorch's fused bloom or the official hash: rename or compute from the module (D3): open.
- 2.8 the text loader keeps fp32 items on device: PubMed full catalog cannot load (36 M x 768 x 4 B = 110 GB); fp16 items + a chunked oracle: open (E2 targets the 10 M slice instead).
- 2.10 timing semantics stated in the system doc (closed-loop latency incl. launch latency; `--profile` for kernel time); `compile_s` and `hidden_syncs` recording: open.
- 2.11 `users_limit` single site confirmed; 2.12 tie handling correct; 2.13 provenance missing `uv.lock` sha, package version, the fp16 reduced-precision flag: open.
- 3.1 `run.run` 200 lines with four nested caches: split (open). 3.4 `bench run --out` CWD-relative vs `campaign` EVAL_DIR-relative: campaign fixed, run still CWD-relative. 3.7 naming nits; 3.8 `Job` frozen but unhashable.
- 4.3 slow tests: session-scoped SilverTorch builds, one-child campaign test: 82 passed in 206 s -> 94 passed in 99.8 s.
- 5.6 `_clean` misses `np.ndarray` / `np.bool_`: open.

## Keep as is
One appended JSONL record per cell with `fsync`; `Job` as the only config class; the build / query split; one process per group with spill-file parity; `bench.latency`'s protocol and `graph_callable`'s skip and launch-count assertions; `index_bytes = Σ buffers`; oracle design; float64 device metrics with the old per-row arithmetic pinned at 1e-9; `_clean` + `allow_nan=False`; loaders as plain argparse CLIs.

## Deliberate departures when applying
JSONL appends stay one `write` + `fsync` (tmp + replace would rewrite the file per cell); subtree `dirty` counts untracked files (a new kernel module is measured code); the ~300 lines of one-off tests kept until C4 and the golden agree.
