---
chain: "retrieve-review"
branch: "main"
nextStep: "Orchestrator: when the GPU is free, run `UV_PROJECT_ENVIRONMENT=/venvs/retrieve-review uv run --no-sync --directory /scratch/wt/retrieve-review/retrieve pytest tests/ -x -q` on dev/retrieve-review @ 57f86ed. Watch in particular tests/compile (oporp grid change under triton_op/export) and tests/parity/test_oporp_1bit_match_topk.py, test_clause_compact.py, test_bloom_compact.py, test_codesigned_probe_score*.py. If green, merge into staging; if red, send the failure back to this worker."
created: "2026-09-26T08:13:04Z"
---

# retrieve/ review pass (python-review + deslop): committed, GPU suite pending

## Request
Constrained worker. Bounded, behaviour-preserving cleanup of retrieve/ (src + tests), focused on the kernel-opt diff (5fa1c99..HEAD).
Scope excluded evaluation/, docs/roadmap.md and docs/validation.md. Not pushed or merged.

## State
- Branch `dev/retrieve-review`, worktree `/scratch/wt/retrieve-review`, commit **57f86ed** (on top of 204a7f6). 22 files, +114/−152.
- Verified without the GPU: `ruff check retrieve` clean, `ruff format --check retrieve` clean, pre-commit hooks passed, `scripts/check_doc_links.py` 0 problems. The package and tune.py import OK. Collection with `CUDA_VISIBLE_DEVICES=""` gathers 755 tests.
- **Not run: the GPU pytest suite** (the GPU was held by e4's training). Nothing is claimed green.

## Findings and what happened to each
Fixed (host Python or dead values only; no numerics, addressing, tolerances or launch config changed):
1. `ops/triton/oporp_1bit_match_topk.py`: `_oporp_prep` took an unused `k`, which is now removed. The two `@triton_op` wrappers had nested `def grid(meta)` closures computing `cdiv(n_kernel, meta["BLOCK_N"])`. The grid is now a tuple `(cdiv(n_kernel, cfg.block_n), b)` stored on `_OporpLaunch`. BLOCK_N equals cfg.block_n, so the launch is the same, and the other triton_ops already pass tuple grids. Fields `n_kernel` and `b` were dropped from the record. **Needs the GPU compile/export tests to confirm.**
2. `_ClauseCompactLaunch` and `_BloomCompactLaunch` were identical. They are now one `_host.CompactLaunch`, and `_host.compact_finish(launch, n, *, block_n, num_warps)` takes the record instead of 4 loose args.
3. `common.probe_tile` returned `total`, which neither probe scorer read. It now returns `(pos, slot, valid, tail)`. This is a dead value in a @triton.jit helper, so the compiled code is unchanged after DCE, but it still needs the parity run.
4. `modules/silvertorch._lap` had a nested `def lap`. It is now `partial(_synced_clock, device)`.
5. `ops/tune.py`: comments/docstrings still described a deleted "CUDA scorer" `(block_p, num_warps, unroll)` grid, now fixed. `_entry` now reuses `_config_fields`. The late-binding `lambda c=cfg, i=inputs:` is now `partial(spec.run, inputs, cfg)`.
6. `tests/correctness/test_op_boundary.py` had a **real bug**: `parametrize(..., list(_odd_dim_calls()))` allocated CUDA tensors at collection time. On a box without a GPU, collection errored with "No CUDA GPUs" instead of skipping (reproduced against HEAD). The ids are now a static `_ODD_DIM_CASES` tuple.
7. Nested defs in tests were flattened with the same behaviour:
   - `parity/conftest.poison_empty` → `_poisoned_empty` + partial
   - `parity/test_codesigned_probe_score` rowwise → `_bloom_rowwise`
   - `correctness/test_silvertorch` fresh → `_fresh` method
   - `compile/test_silvertorch_compile` inputs → `_inputs`
8. Stale or residue comments:
   - `ops/official/adapter.py` had a "layout: padded IVF → official CSR" block. The padded layout was deleted in kernel-opt; the block is removed.
   - `clause_compact` module docstring had a history clause (the "atomic_add row base this replaced"); removed.
   - "THE single place input checking happens" appeared in 7 docstrings; toned down.
   - Banners carried session/plan residue ("kernel-opt Phase 5, batch 1", "roadmap B5 / official-integration plan §8 TF-2", "00-roadmap.md §2.1"); the residue is stripped.
   - `tests/conftest.require_official` referred to "the Mac"; removed.
   - `test_large_offsets` fixtures used `yield` with no teardown; changed to `return`.

Not applicable:
- No `except Exception`, `global`/`nonlocal` (except the pre-existing `_LOAD_MEMO` in ops/official/__init__.py), TODO/FIXME or debug prints in the diff. `print` appears only in test_official.py's measurement tests.
- test_large_offsets.py and test_op_boundary.py do not duplicate the parity tests. One covers >2³¹ addressing classes, the other boundary rejections; neither is an oracle comparison.
- No leftover dead helper from the failed Phase-1 attempts was found. padded_layout, words_per_cluster and the old build_transposed_sigs are gone; `rg` for padded/atomic/unroll only hit the items fixed above.
- docs/system: no code/doc mismatch found. The probe_tile signature in kernels.md does not list the return tuple, and `n_kernel` is still the local name kernels.md:525 uses.

Flagged for the orchestrator (they touch kernel source, launch config or public API, so not done):
- A. The two probe-scorer kernels (`codesigned_probe_score.py`, `_exact.py`) duplicate about 25 lines each: the tail branch and the int8 dot → dequant → -inf → store epilogue. This could become one `common.py` @triton.jit helper. It is a kernel change and needs parity + timing.
- B. Six identical `*Config(block, num_warps, num_stages=3)` dataclasses exist (one per kernel file). They could be one config type. tune.py prints the class name in its paste line, so this changes the API: Tier C.
- C. `N: tl.constexpr` in `_bloom_match_kernel` and `_oporp_1bit_match_topk_kernel` (full scan) compiles once per N. This is pre-existing and deliberate for oporp (commented), but bloom_match has no such justification. It is a perf/launch-config question.
- D. `tune._register_subcommand` has a nested `callback` plus import-time registration of subcommands. This is pre-existing. Using partial breaks click help (`__doc__`), so any fix needs a small restructure: consider.
- E. There are inline imports for the optional official extension (`ops/__init__.available_backends`, `ops/official/__init__._try_load`, `modules/official.__getattr__`, `tests/conftest.require_official`). They are pre-existing and are the deliberate lazy loading of an optional dep. The house rule says restructure: consider.
- F. `silvertorch._register_filter_buffers` has `assert attrs is not None` as type narrowing. It is left as is; the invariant is established by `_validate_register_args`.
- G. Section banners are pervasive in test_linr/test_quantize/test_kmeans/test_official/adapter.py (pre-existing file convention). A sweep would be cosmetic, so it was not done.
