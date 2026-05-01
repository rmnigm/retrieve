# Filtering API + test-suite refactors — GPU verification plan

Two stacked refactors landed on a no-GPU host and need a CUDA box to
verify end-to-end. This plan is the handoff: run the suite, confirm
everything passes, file fixes inline if anything breaks.

The repo uses `uv` — every command below assumes you `cd retrieve && uv …`.
The first `uv run` will install `torch` + `triton` from the lockfile.

## What changed

### Phase A — filter API refactor (per [filtering-api.md](filtering-api.md))

`retrieve.layers.filters` is the new home for all standalone filters.
`ClauseIndex` moved there from `layers/utils/filters.py`. `BloomFilter`
is new — paper-strict (no reverse, no DSL) and built on the existing
`bloom_match` Triton kernel that previously had no production caller.
The old `BloomIndex` and the silvertorch-private `bloom.py` file were
deleted; the helpers `_build_signatures` / `_generate_seeds` / `_mix64`
moved to `layers/filters/bloom.py`. Both filters subclass an extended
`FilterModule` ABC (`evaluate_mask`, `evaluate_indices`, `evaluate_subset`,
`forward = evaluate_mask`). `combine_masks` and `combine_indices` ship in
`layers/filters/__init__.py`. Tests were rewritten in this shape — there
is no back-compat shim. (`evaluation/retrieval/eval_perf.py` was rewritten
here as part of Phase A; in a later refactor it was deleted entirely
along with `eval_quality.py` — `evaluation/retrieval/benchmark.py` is now
the sole eval driver.) SilverTorch's
fused in-cluster bloom path (`codesigned_probe_score`) is untouched;
only the helper import path moved.

### Phase B — test-suite refactor (drop benches, deepen correctness)

`retrieve/tests/bench/` was deleted in full. Performance characterization
moves to `evaluation/`; `tests/` is correctness only. The `bench` pytest
marker, the `bench_*.py` glob in `tool.pytest.ini_options.python_files`,
and the `tests/bench/*.py` ruff per-file-ignore are all gone from
`retrieve/pyproject.toml`. In their place:

- Three new correctness files for previously-untested public exports:
  `test_scorers.py` (`DotProductScorer`), `test_retrieval_utils.py`
  (`FullScanKNN` direct + `post_filter_topk`), `test_compact.py`
  (`compact_mask`).
- One new parity test for the only production Triton kernel without
  one: `test_clause_compact.py` (mirrors `test_bloom_match.py`'s shape).
- Edge-case extensions on every existing correctness file (empty
  candidate sets, all-pass / all-fail masks, `K=N`, `B=1`, single-item
  index, `query_clause_attrs=None ≡ IVF_INT8_ANN` for SilverTorch, etc.).

Docs (`architecture.md`, `kernels.md`, `filtering-api.md`) were updated
to drop bench references and document the two-tree test layout.

## Step 1 — Smoke (covers Phase A)

```bash
cd retrieve

uv run python -c "
from retrieve import (
    BloomFilter, ClauseIndex, FilterModule,
    combine_masks, combine_indices,
    DotProductScorer, FullScanKNN, post_filter_topk,
)
from retrieve.layers.filters import ClauseIndex as CI2, BloomFilter as BF2
from retrieve.layers.filters.bloom import _build_signatures, _generate_seeds
from retrieve.layers.utils.compact import compact_mask
from retrieve.layers.utils.quantize import (
    popcount_int64, project_oporp_1bit_query, quantize_int8, quantize_oporp_1bit,
)
assert ClauseIndex is CI2 and BloomFilter is BF2

import importlib
for path in (
    'retrieve.layers.silvertorch.bloom',
    'retrieve.layers.utils.filters',
):
    try:
        importlib.import_module(path)
    except ImportError:
        continue
    raise AssertionError(f'{path} should have been deleted')

try:
    from retrieve import BloomIndex  # noqa: F401
except ImportError:
    pass
else:
    raise AssertionError('BloomIndex should no longer be exported')

print('SMOKE OK')
"
```

Expected: `SMOKE OK`. If the import fails, stop and inspect — do not
proceed to the test suite.

## Step 2 — Pytest collection (covers Phase B layout)

This is a no-CUDA-required check that the test tree is well-formed and
that nothing under `tests/bench/` lingers.

```bash
cd retrieve
uv run pytest --collect-only tests/ 2>&1 | tail -40
uv run pytest --collect-only tests/ 2>&1 | grep -c "tests/bench"
# expect: 0
uv run pytest --collect-only tests/ 2>&1 | grep -c "test_clause_compact"
# expect: ≥ 1
```

Also confirm the wiring is clean:

```bash
grep -n "bench" retrieve/pyproject.toml
# expect: no `markers = [...bench...]` block, no bench_*.py glob,
#         no `tests/bench/*.py` per-file-ignore
```

## Step 3 — New correctness files (Phase A)

```bash
cd retrieve
uv run pytest tests/correctness/test_bloom_filter.py -v
uv run pytest tests/correctness/test_combine_filters.py -v
```

These exercise:

- `BloomFilter` shape / dtype / validation, no-FN invariant vs `ClauseIndex`,
  empirical FPR vs analytic bound, `evaluate_subset` parity with
  `mask.gather(1, ids)`, `evaluate_indices` default-path equivalence to
  `compact_mask(evaluate_mask)`, and edge cases (`p=0` subset,
  single-item index, CPU↔CUDA mask agreement).
- `combine_masks` None-tolerance + identity + 3-way AND + all-None.
- `combine_indices` equivalence to `compact_mask(combine_masks(...))`,
  filter-order independence as sets, empty-intermediate short-circuit,
  single-filter pass-through, `[]` / length-mismatch error paths,
  subsequent-filter-drains cascade.

## Step 4 — New correctness files (Phase B)

These are brand-new and test public-API surfaces that previously had no
direct test (only transitive coverage through other modules).

```bash
cd retrieve
uv run pytest tests/correctness/test_scorers.py -v
uv run pytest tests/correctness/test_retrieval_utils.py -v
uv run pytest tests/correctness/test_compact.py -v
```

Coverage:

- `test_scorers.py` — `DotProductScorer` shape / dtype, equivalence to
  `torch.einsum('bd,bpd->bp', q, embs[cand])`, B=1 path, P=0 path,
  `-1` candidate id semantics (Python negative-indexing, contract pin).
- `test_retrieval_utils.py` — `FullScanKNN` directly checked against a
  hand-rolled `(q @ x.T).topk(k)` reference (this oracle was previously
  used everywhere but never tested itself), mask path = post-filter
  semantics, candidate-ids path = gather + bmm, and `p < k` returns
  `actual_k = p` columns. `post_filter_topk` drop-failed / all-pass /
  all-fail.
- `test_compact.py` — `compact_mask` shape, counts = mask sum, per-row
  set equality, all-True / all-False / B=1 / mixed-row-pad-to-max
  edge cases.

## Step 5 — New parity test (Phase B)

```bash
cd retrieve
uv run pytest tests/parity/test_clause_compact.py -v
```

Direct kernel-vs-pure-torch parity for `clause_compact`, the production
kernel powering `ClauseIndex.evaluate_indices` on CUDA. Output id order
within a row is unspecified (atomics across tiles), so the assertion is
per-row **set** equality + counts equality. Cases: random parametrize on
`(N, C, A_max, pad_rate)`, all-reverse clauses, all-inactive query,
no-passing-items query, B=1 grid corner.

## Step 6 — Migrated + extended correctness files

These existed before either refactor; both phases touched them.

```bash
cd retrieve
uv run pytest tests/correctness/test_filters.py -v
uv run pytest tests/correctness/test_linr.py -v
uv run pytest tests/correctness/test_silvertorch.py -v
uv run pytest tests/correctness/test_quantize.py -v
uv run pytest tests/correctness/test_ivf.py -v
uv run pytest tests/parity/test_bloom_match.py -v
uv run pytest tests/parity/test_codesigned_probe_score.py -v
```

What to look for in each:

- **`test_filters.py`** — Phase A added the `LiNR_V3 + combine_masks`
  cross-compat smoke; Phase B added `all_reverse`, `single_item_index`,
  `evaluate_subset_p_zero` cases.
- **`test_linr.py`** — Phase B added a `TestEdgeCases` class covering:
  mask-all-True equals unmasked (V1, V3 both backends), mask-all-False
  produces no finite scores, V2 candidate_ids `p=0` returns full padding,
  B=1 single-query, K=N full-retrieval path.
- **`test_silvertorch.py`** — Phase A migrated `TestEquivalence` to use
  `BloomFilter`; Phase B added `query_clause_attrs=None ≡ IVF_INT8_ANN`
  (the documented fast path was previously unverified) and a
  mask-all-False-with-bloom case.
- **`test_quantize.py`** — Phase B added `int8_clamp_at_extremes`,
  `int8_zero_input_no_nan`, `oporp_seed_determinism`,
  `project_oporp_query_alone`, `popcount_int64_against_python_reference`
  (cross-checks the SWAR popcount against `bin(x).count('1')` — a true
  external oracle, not self-comparison), `popcount_int64_known_constants`,
  `popcount_int64_rejects_non_int64`.
- **`test_ivf.py`** — Phase B added `mask_all_false`, `n_lists_equals_n`
  (degenerate clustering, recall ≈ 1 at full probe),
  `candidate_ids_p_less_than_k`.
- **`test_bloom_match.py`** — Phase A rewrote it to compute the pure-torch
  reference on CPU (since `BloomFilter.evaluate_mask` itself dispatches to
  the kernel on CUDA — calling it would tautologically pass).
- **`test_codesigned_probe_score.py`** — Phase A migrated imports only.

## Step 7 — Full suite

```bash
cd retrieve
uv run pytest tests/ -x -q
```

The full suite is gated to CUDA via
`tests/conftest.py::pytest_collection_modifyitems`, so it will SKIP
rather than fail on a non-CUDA box. There is no `--bench` flag any more
— bench is gone.

Sanity counts:

```bash
uv run pytest --collect-only tests/correctness/ -q | tail -5
# Phase B raised the correctness count materially — should be well above
# the pre-refactor baseline.

uv run pytest --collect-only tests/parity/ -q | tail -5
# Should be the previous count + tests in test_clause_compact.py.
```

## Step 8 — Lint & docs sanity

```bash
cd /Users/rmnigm/work/retrieve
uv run --project retrieve ruff check retrieve/
# expect: clean. The `tests/bench/*.py = ["E731"]` per-file-ignore was
# dropped along with bench/, and lambdas in tests are not used elsewhere.

grep -rn "tests/bench\|--bench\|bench\.md" docs/system/ retrieve/tests/ retrieve/src/
# expect: no hits — every reference in the live tree was scrubbed.
# Hits in docs/plans/ are historical and intentional (this file, the
# refactor-plan, etc.).
```

## What to do if a test fails

1. **Import error on Step 1.** Stale path. Search
   `grep -rn "BloomIndex\|layers\.utils\.filters\|layers\.silvertorch\.bloom\b" retrieve/src retrieve/tests evaluation`.
2. **`test_bloom_filter.py::TestNoFalseNegatives` fails.** Bloom is
   producing fewer hits than the exact filter — bug in the new
   `BloomFilter.evaluate_mask` / `_build_signatures` move. Check the
   helper bodies in `layers/filters/bloom.py` are byte-identical to the
   pre-refactor source.
3. **`test_combine_filters.py::test_combine_indices_*` fails.** The
   cascade in `layers/filters/__init__.py` is the new code. Check the
   padding mask (`valid = arange(p) < counts.unsqueeze(1)`) and the
   re-compaction `argsort` — easy to get subtly wrong.
4. **`test_silvertorch.py::TestEquivalence` fails.** Likely the migrated
   `BloomFilter` doesn't match what SilverTorch builds internally.
   `silvertorch/main.py` imports `_build_signatures` / `_generate_seeds`
   from `retrieve.layers.filters.bloom` — confirm those helpers' bodies
   match the pre-refactor source. Determinism relies on the seed
   `0x515C0DE` literal.
5. **`test_silvertorch.py::TestEdgeCases::test_query_clause_attrs_none_equals_ivf` fails.**
   The SilverTorch `query_clause_attrs=None` fast path is supposed to be
   bit-equivalent to `IVF_INT8_ANN`. If it diverges, check
   `silvertorch/main.py` lines 137-148 (the `qb is None` branch wiring
   `query_bits=None, bloom_sigs=None` into `codesigned_probe_score`).
6. **`test_clause_compact.py::test_clause_compact_no_passing_items` fails.**
   `clause_compact` returns `P >= 1` even when nothing passes (the kernel
   wrapper does `max(int(counts.max().item()), 1)` to avoid a zero-sized
   alloc). The test asserts on `counts`, not on `out_indices` shape — if
   it fails, the kernel may be miscounting empty rows.
7. **`test_quantize.py::test_popcount_int64_against_python_reference` fails.**
   This is the only test that compares the SWAR popcount against a true
   external reference (Python `bin().count('1')`); a failure means the
   SWAR magic constants in `quantize.py:15-22` were edited incorrectly.
8. **Anything else.** Read the diff for the relevant file, fix in place,
   re-run that file's tests with `-v`.

## Files touched (for orientation)

### Phase A — filter API

Created:

- `retrieve/src/retrieve/layers/filters/__init__.py`
- `retrieve/src/retrieve/layers/filters/clause.py`
- `retrieve/src/retrieve/layers/filters/bloom.py`
- `retrieve/tests/correctness/test_bloom_filter.py`
- `retrieve/tests/correctness/test_combine_filters.py`

Deleted:

- `retrieve/src/retrieve/layers/utils/filters.py`
- `retrieve/src/retrieve/layers/silvertorch/bloom.py`
- `retrieve/tests/correctness/test_bloom.py`

Modified:

- `retrieve/src/retrieve/interfaces.py`
- `retrieve/src/retrieve/__init__.py`
- `retrieve/src/retrieve/layers/__init__.py`
- `retrieve/src/retrieve/layers/utils/__init__.py`
- `retrieve/src/retrieve/layers/silvertorch/__init__.py`
- `retrieve/src/retrieve/layers/silvertorch/main.py`
- `retrieve/tests/correctness/test_filters.py`
- `retrieve/tests/correctness/test_linr.py`
- `retrieve/tests/correctness/test_silvertorch.py`
- `retrieve/tests/parity/test_bloom_match.py`
- `retrieve/tests/parity/test_codesigned_probe_score.py`
- `evaluation/retrieval/eval_perf.py` (since deleted in eval-harness refactor)
- `docs/system/filtering.md`
- `docs/system/architecture.md`
- `docs/system/kernels.md`
- `docs/plans/filtering-api.md` (status line only)

### Phase B — test-suite refactor

Created:

- `retrieve/tests/correctness/test_scorers.py`
- `retrieve/tests/correctness/test_retrieval_utils.py`
- `retrieve/tests/correctness/test_compact.py`
- `retrieve/tests/parity/test_clause_compact.py`

Deleted:

- `retrieve/tests/bench/__init__.py`
- `retrieve/tests/bench/README.md`
- `retrieve/tests/bench/conftest.py`
- `retrieve/tests/bench/bench_kernels.py`
- `retrieve/tests/bench/bench_linr.py`
- `retrieve/tests/bench/bench_silvertorch.py`
- `retrieve/tests/bench/render.py`
- `retrieve/tests/bench/run.py`
- `retrieve/tests/bench/` (directory)

Modified:

- `retrieve/pyproject.toml` (drop `bench_*.py` glob, drop `bench` marker,
  drop `tests/bench/*.py` ruff ignore)
- `retrieve/tests/correctness/test_quantize.py` (edge cases)
- `retrieve/tests/correctness/test_bloom_filter.py` (edge cases)
- `retrieve/tests/correctness/test_filters.py` (edge cases)
- `retrieve/tests/correctness/test_combine_filters.py` (edge cases)
- `retrieve/tests/correctness/test_linr.py` (TestEdgeCases class)
- `retrieve/tests/correctness/test_ivf.py` (edge cases)
- `retrieve/tests/correctness/test_silvertorch.py` (edge cases)
- `docs/system/architecture.md` (Testing subsection, kernel table fix)
- `docs/system/kernels.md` (drop bench numerics references)
- `docs/plans/filtering-api.md` (drop Verification step 5 about benches)

## Why no GPU run on the implementing host

The implementing host was darwin/arm64 — `triton` has no wheel for that
platform, so `uv` cannot bootstrap the env. All Python files were
syntax-checked with `ast.parse` and the module structure is sound, but
runtime semantics for any CUDA-only path (Triton kernels, `cuda` device
ops in `_generate_seeds`, `clause_compact` atomics) were never executed.
