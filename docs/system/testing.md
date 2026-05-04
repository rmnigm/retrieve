<!-- claude code generated file -->

# `retrieve` testing

Live reference for the `retrieve` test suite. Covers layout, fixtures,
baseline strategy, what each module asserts, and how to add a new test.
For the package architecture see [architecture.md](architecture.md);
for kernel internals see [kernels.md](kernels.md).

## Scope

`retrieve/tests/` is **correctness only**. Every test asserts a
semantic invariant, a numerical equivalence, or an edge-case contract.
Performance characterization (latency, memory, recall sweeps) lives
separately in [`evaluation/`](../../evaluation/) and never gates CI.

The suite is **GPU-only**. The root
[`conftest.py`](../../retrieve/tests/conftest.py) installs a collection
hook that skips every test when CUDA is unavailable — there is no
torch-CPU fallback path in the library.

## Layout

```
retrieve/tests/
├── conftest.py            # global fixtures + CUDA skip gate
├── correctness/           # module-level semantics vs torch baselines
│   ├── test_bloom_filter.py
│   ├── test_combine_filters.py
│   ├── test_compact.py
│   ├── test_filters.py             (ExactAttributeFilter)
│   ├── test_linr.py                (SimilarityMasking, PrefilterKNN, OneBitKNN × torch / Triton)
│   ├── test_quantize.py            (int8, OPORP, popcount)
│   ├── test_retrieval_utils.py     (FullScanKNN, post_filter_topk)
│   ├── test_scorers.py             (DotProductScorer)
│   └── test_silvertorch.py         (SilverTorch, both bloom-on and bloom-off)
└── parity/                # Triton kernel vs pure-torch reference
    ├── conftest.py        # assert_topk_matches helper
    ├── test_bloom_match.py
    ├── test_clause_compact.py
    ├── test_codesigned_probe_score.py
    ├── test_fused_masked_knn_topk.py
    └── test_oporp_1bit_match_topk.py
```

The split is **by purpose**, not by module:

- `correctness/` answers *"does this layer compute the right thing?"*
  The oracle is a hand-rolled torch reference (matmul + topk, broadcast
  subset, brute-force compaction). One file per public class or helper
  group.
- `parity/` answers *"does the Triton kernel match the torch path on
  the same inputs?"* The oracle is a pure-torch implementation living
  inside the test file (`_ref`). One file per kernel.

A module that has both a torch and a Triton backend (like LiNR V1/V2/V3)
appears in both trees: `test_linr.py` covers semantics, the parity
files cover kernel agreement.

## Fixtures and helpers

### Root [`conftest.py`](../../retrieve/tests/conftest.py)

Six fixture functions — all return CUDA tensors with deterministic seeds:

- `make_index(n, d, *, normalized=True, seed=0, dtype=float32)` —
  random `[N, D]` item embeddings, unit-norm by default.
- `make_query(b, d, *, normalized=True, seed=1, dtype=float32)` —
  random `[B, D]` query embeddings, unit-norm by default.
- `make_mask(b, n, *, pass_rate, seed=2)` — random `[B, N]` bool with
  the given expected pass rate.
- `make_attrs(n, c, a_max, *, n_vocab=50, pad_rate=0.3, seed=3)` —
  random `[N, C, A_max]` int64 clause attributes with `-1` padding.
- `make_query_attrs(b, c, *, n_vocab=50, inactive_rate=0.2, seed=4)` —
  random `[B, C]` int64 query attributes with `-1` marking inactive
  clauses.

Three evaluation helpers:

- `valid_id_set(ids, scores, b)` — set of ids in row `b` with
  finite score and non-padded id (`>= 0`). The canonical "what came
  back from this retrieval call" check.
- `recall_at_k(approx_ids, exact_ids)` — mean per-row set overlap of
  approximate vs exact ids, divided by K.
- `assert_recall_monotone(recalls, *, slack=0.02)` — non-decreasing
  check with absolute slack; used to verify ANN recall grows with
  `n_probe`.
- `assert_topk_id_sets_match(out_ids, out_scores, ref_ids, ref_scores,
  b, *, atol=1e-3, rtol=1e-3)` — per-row id-set comparison with tie
  tolerance at the K-th boundary. Used from the **correctness** tree
  (e.g. `test_linr.py`) when comparing two retrieval calls (typically
  torch backend vs Triton backend) on a specific row; symmetric-
  difference ids must lie within `atol` of their side's finite-score
  minimum, which absorbs the tiebreak flip caused by `tl.dot` vs torch
  `@` accumulator-order drift. The parity tree's `assert_topk_matches`
  is the all-rows analogue.

The CUDA gate is implemented as
`pytest_collection_modifyitems` — every collected item gets a
`skip` marker if CUDA is missing. No marker is needed on individual
tests.

### [`tests/parity/conftest.py`](../../retrieve/tests/parity/conftest.py)

One helper:

- `assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores, *,
  atol=1e-3, rtol=1e-3)` — the standard kernel-vs-torch assertion.
  Compares **sets** of finite-score ids per row (tie-breaking on the id
  permutation differs between backends, so position-equality is not
  asserted) and **sorted descending scores** with `allclose` (with
  `-inf` replaced by 0 so padded rows still compare cleanly).

Use `assert_topk_matches` for any new parity test; do not write a
position-equality assertion for fp32 score paths.

## Conventions

- **Framework**: vanilla `pytest`. No `hypothesis`, no `unittest`.
- **Parametrize**: `@pytest.mark.parametrize` for shape and pass-rate
  matrices; class-level parametrize for backend cross-product
  (`@pytest.mark.parametrize("cls", [SimilarityMasking, SimilarityMaskingTriton])`).
- **No marks**: there is no `slow`, `gpu`, or `bench` marker. Every
  test is GPU-bound and runs in CI; the CUDA skip gate handles
  no-GPU hosts globally.
- **No mocking**: tests use real CUDA tensors and real kernels.
  Triton kernel paths run on every CUDA test; there is no
  `monkeypatch` of the kernel.
- **Determinism**: every random tensor is built with an explicit
  `torch.Generator(device="cuda").manual_seed(...)`; never use
  `torch.randn` without a generator.
- **Set-vs-position assertions**: when comparing two retrieval
  outputs, compare per-row `set(ids)` (or `valid_id_set`), not
  position-by-position. Tie-breaking on equal scores is implementation-
  specific and is not asserted.
- **Score tolerance**: `atol=1e-3, rtol=1e-3` for fp32 tile-blocked
  reductions (`SimilarityMasking` / `PrefilterKNN` paths); strict
  `torch.equal` only for `OneBitKNN` (popcount is exact by construction)
  and for path-isolated comparisons on identical reductions.
- **Module-scope fixtures**: heavy index builds (IVF, LiNR with N>1k)
  use `@pytest.fixture(scope="module")` to amortize across the file.
  Per-test mutation of these fixtures is forbidden.

## Baseline strategy

All baselines are **torch-only**. There are no `numpy`, `scipy`,
`sklearn`, or `faiss` references in the suite — the test deps are
`torch`, `triton`, `pytest`. Each module is checked against a hand-
rolled torch implementation that is structurally simpler than the
module under test.

| Component                  | Baseline / oracle |
|----------------------------|-------------------|
| `FullScanKNN`              | `(q @ x.T).topk(k)` written inline in `_topk_reference`. The single test that breaks the otherwise-circular use of `FullScanKNN` as suite oracle. |
| `DotProductScorer`         | `torch.einsum("bd,bpd->bp", q, embs[cand])` |
| `compact_mask`             | `mask.nonzero` per row + length comparison |
| `post_filter_topk`         | per-row `mask.gather` + `(~keep).fill(-1)` |
| `quantize_int8`            | known constants + reconstruction error bounded by `abs_max / 127` |
| `quantize_oporp_1bit`      | self-consistency (item-side bits == `project_oporp_1bit_query` of same emb) + Pearson correlation with cosine similarity |
| `popcount_int64`           | Python `bin(x & 0xFFFFFFFFFFFFFFFF).count("1")` — true external oracle |
| `ExactAttributeFilter.evaluate_mask`| dense broadcast `(item == query.unsqueeze).any(-1).all(-1)` written explicitly in `test_clause_index_all_reverse` |
| `ExactAttributeFilter.evaluate_indices` | `compact_mask(evaluate_mask(...))` (the ABC default path) |
| `BloomFilter.evaluate_mask`| `ExactAttributeFilter.evaluate_mask` ⊆ result (no-FN invariant) + analytic FPR `(1 - exp(-k/m))^k` |
| `BloomFilter.evaluate_subset` | `evaluate_mask(q).gather(1, ids)` |
| `combine_masks`            | iterated `&` of non-`None` inputs |
| `combine_indices`          | `compact_mask(combine_masks(*[f.evaluate_mask(q)]))` |
| `SilverTorch` (no bloom)   | `FullScanKNN` for recall (asserts ≥ 0.85 at full probe) + recall monotone in `n_probe` |
| `SilverTorch` (with bloom) | `SilverTorch (no bloom)(mask=BloomFilter.evaluate_mask(qa))` — component composition |
| `SilverTorch` (qa=None)    | `SilverTorch (no bloom)` directly — `query_clause_attrs=None` is a documented fast path |
| `SimilarityMasking` semantics | `(q @ x.T).masked_fill(~mask, -inf).topk(k)` |
| `PrefilterKNN` semantics      | gather + bmm + local topk + scatter |
| `OneBitKNN` semantics         | `FullScanKNN` recall (asserts ≥ 0.4 at K=200, N=2048) |
| `*Triton` siblings            | the corresponding torch reference class |
| `fused_masked_knn_topk`    | `compact_mask(mask)` → `bmm(q.unsqueeze(1), embs[ids].transpose(1,2)).squeeze(1)` → topk |
| `oporp_1bit_match_topk`    | `popcount_int64(xor) → D - 2*hamming` → topk (bit-exact) |
| `bloom_match`              | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to avoid tautology with the kernel-routed `BloomFilter.evaluate_mask`  |
| `clause_compact`           | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` |
| `codesigned_probe_score`   | `_ref_phase23` in the parity file: bloom subset + INT8 dequant + dot + topk |

## What each correctness file asserts

Concrete invariants by file. Use this as a map when adding new tests
or hunting a regression.

### [`test_compact.py`](../../retrieve/tests/correctness/test_compact.py)

`compact_mask([B, N] bool) → ([B, P] int64, [B] int64)`.

- `counts == mask.sum(1)`.
- `P == counts.max()`; rows with fewer passing items are right-padded
  with arbitrary item ids (callers must use `counts` to bound reads).
- Per-row valid-id set equals the set of `True` positions in `mask`.
- `pass_rate = 1.0` → `ids` covers all `[0, N)`, `counts == N`.
- `pass_rate = 0.0` → `ids.shape == [B, 0]`, `counts == 0`.
- `B = 1` and mixed-row-pad-to-max corners.

### [`test_scorers.py`](../../retrieve/tests/correctness/test_scorers.py)

`DotProductScorer`.

- Buffer shape / dtype / device after `register_index`.
- Output equals `torch.einsum("bd,bpd->bp", q, embs[cand])` (atol=1e-5).
- `B = 1` path.
- `P = 0` returns `[B, 0]`.
- `candidate_ids = -1` reads the last item via Python negative
  indexing (contract is "caller must mask post-hoc"; pinned so a
  stricter implementation is a deliberate change).

### [`test_retrieval_utils.py`](../../retrieve/tests/correctness/test_retrieval_utils.py)

`FullScanKNN` and `post_filter_topk`.

- `FullScanKNN()` matches `(q @ x.T).topk(k)` exactly (id positions and
  scores). This is the single non-circular check on the oracle the rest
  of the suite uses.
- Mask path: post-filter, not in-loop. Surviving ids retain their
  original positions; failed positions become `-1`.
- Candidate-ids path: gather + bmm reference; `p < k` returns
  `actual_k = p` columns (no padding, by current contract).
- `post_filter_topk` — drop-failed (`counts == mask.sum`), all-pass
  (`out_ids == topk_ids`), all-fail (`(out_ids == -1).all()`).

### [`test_quantize.py`](../../retrieve/tests/correctness/test_quantize.py)

INT8 + OPORP + popcount.

- INT8 reconstruction error bounded by `abs_max / 127`.
- INT8 codes in `[-127, 127]`; values at `±abs_max` round to exactly
  `±127`; all-zero input doesn't NaN (the `clamp_min(1e-8)` guard).
- OPORP shapes and dtypes; `signs ∈ {-1, +1}`; `perm` is a permutation.
- `quantize_oporp_1bit(embs, ...)` bits equal
  `project_oporp_1bit_query(embs, signs, perm)` (round-trip check).
- Hamming-similarity has Pearson correlation > 0.5 with true cosine.
- `D % 64 != 0` raises `ValueError`.
- Same `seed` → identical `(bits, signs, perm)`; different seed →
  different `(signs, perm)` (and almost certainly different bits).
- `popcount_int64` matches Python `bin(u).count("1")` after
  unsigned reinterpret. Known constants `[0, 1, 2, 3, -1, 0x5555...]
  → [0, 1, 1, 2, 64, 32]`. Non-int64 input raises `TypeError`.

### [`test_filters.py`](../../retrieve/tests/correctness/test_filters.py)

`ExactAttributeFilter` semantics.

- AND/OR semantics: per-clause OR over `A_max` slot, AND across `C`
  clauses. Verified against a hardcoded item layout and four hand-
  curated queries.
- `evaluate_indices` (Triton path) matches `compact_mask(evaluate_mask)`
  per-row as sets.
- Reverse clauses: `clause_is_reverse` flips per-clause result.
- Inactive query (`-1`) overrides reverse → clause always passes.
- `query == [-1, -1, ...]` returns all items, `counts == N`.
- All-reverse clauses against an explicit reference using
  `(~matched).all(dim=-1)`.
- `N = 1` corner.
- `evaluate_subset` with `[B, 0]` candidates returns `[B, 0]`.
- `OneBitKNNTriton(mask=combine_masks(clause_mask, bloom_mask))`
  cross-compat smoke (combined mask should equal clause mask, since
  bloom is a strict superset).

### [`test_bloom_filter.py`](../../retrieve/tests/correctness/test_bloom_filter.py)

`BloomFilter` semantics.

- Buffer shape (`bloom_sigs[N, m_bits/64]`, `hash_seeds[k_hash, 2]`).
- `evaluate_mask` shape and dtype.
- `m_bits` non-power-of-2 / non-multiple-of-64 raises `ValueError`;
  `k_hash <= 0` raises `ValueError`.
- **No false negatives**: `clause_mask <= bloom_mask` element-wise on
  random inputs. This is the load-bearing Bloom invariant.
- All-inactive query passes all items.
- Empirical FPR on a deliberately-disjoint query vocab is ≤ 4× the
  analytic bound `(1 - exp(-k/m))^k` (or `1e-6` floor).
- `evaluate_subset(q, ids)` equals `evaluate_mask(q).gather(1, ids)`.
- `evaluate_indices` (default path) equals
  `compact_mask(evaluate_mask)`.
- `evaluate_subset` with `[B, 0]` candidates returns `[B, 0]`.
- `N = 1` corner.
- CPU evaluation (pure-torch broadcast) agrees with CUDA evaluation
  (Triton `bloom_match`) on the same input.

### [`test_combine_filters.py`](../../retrieve/tests/correctness/test_combine_filters.py)

`combine_masks` and `combine_indices`.

- `combine_masks(None)` → `None`; `combine_masks(a, None)` returns `a`
  (identity); `combine_masks(a, b)` returns `a & b`.
- 3-way AND.
- All-`None` inputs → `None`.
- `combine_indices` matches `compact_mask(combine_masks(...))` as sets
  per row.
- Filter order independent for the resulting set
  (`[ci, bf]` ≡ `[bf, ci]`).
- Empty intermediate (first filter passes nothing) short-circuits to
  `[B, 0]`.
- Subsequent-filter-drains (first passes all, second rejects all) →
  `[B, 0]`.
- Single-filter pass-through equals direct `evaluate_indices`.
- `[]` filter list raises `ValueError`.
- Filter / query length mismatch raises `ValueError`.

### [`test_linr.py`](../../retrieve/tests/correctness/test_linr.py)

LiNR V1, V2, V3 in both backends.

- V1: top-K is sorted descending; mask path returns ids satisfying
  the mask.
- V2: full path (no candidates) returns sorted top-K; candidate path
  returns ids ⊆ candidates and respects `counts`; default `counts`
  (all P) treated as all-valid.
- V3: full-scan recall vs `FullScanKNN` ≥ 0.4; candidate path
  returns ids ⊆ candidates.
- Cross-backend: torch ↔ Triton return identical valid-id sets per
  row for `SimilarityMasking`, `PrefilterKNN`, `OneBitKNN` across
  `pass_rate ∈ {None, 0.01, 0.1, 0.8}`. Sorted scores `allclose`
  (atol=1e-3) for fp32 paths; bit-exact for `OneBitKNN`.
- `SimilarityMasking` (mask path) ≡ `PrefilterKNN` (compact_mask of same
  mask) as id sets.
- `ExactAttributeFilter` decoupled composition: `SimilarityMasking` with
  `mask = ef.evaluate_mask & extra`, `PrefilterKNN` with
  `compact_mask(combined)`, `PrefilterKNN` with `ef.evaluate_indices`.
- Edge cases (`TestEdgeCases`):
  - mask-all-True ≡ unmasked path (`SimilarityMasking`, `OneBitKNN` ×
    torch / Triton).
  - mask-all-False produces no finite scores.
  - `PrefilterKNN` `candidate_ids` shape `[B, 0]` returns full padding
    (`(-1, -inf)` × K).
  - `B = 1` single-query.
  - `K = N` returns every item (a permutation of `[0, N)`).

### [`test_silvertorch.py`](../../retrieve/tests/correctness/test_silvertorch.py)

`SilverTorch` — bloom-configured and bloom-disabled paths in one file.

- Output shape `(B, K)` with `query_clause_attrs`, without it, and on a
  bloom-disabled module; ids long, scores float32.
- INT8 buffers on the bloom-disabled module: `item_codes` int8,
  `item_scales` float32, centroid / cluster shapes; `bloom_sigs` and
  `hash_seeds` are *not* allocated.
- Param validation: `m_bits` non-power-of-2, `k_hash <= 0`, partial
  bloom config (only one of `m_bits`/`k_hash` set), `n_lists > N`, and
  `n_probe > n_lists` all raise `ValueError`. Passing
  `query_clause_attrs` or `item_clause_attrs` to a bloom-disabled
  module raises.
- Mask honored at `pass_rate ∈ {0.05, 0.5}` (no-bloom path); externally-
  composed `ExactAttributeFilter` mask gives ids ⊆ passing items.
- **Equivalence**: `SilverTorch(q, qa)` ≡
  `SilverTorch(no bloom)(q, mask=BloomFilter.evaluate_mask(qa))` —
  sorted ids match exactly, sorted scores `allclose` (atol=1e-3). Also,
  `query_clause_attrs=None` on a bloom-configured module ≡ a freshly
  built no-bloom module with the same kmeans seed.
- Recall ≥ 0.90 at `n_probe = n_lists`; recall ≥ 0.85 at full probe and
  monotone in `n_probe`.
- Candidate-ids path returns ids ⊆ candidates (with bloom, without
  bloom, and with `p < k`).
- Edge cases:
  - mask-all-False → no finite scores (with and without bloom).
  - `n_lists = N` (one item per cluster) → recall ≥ 0.85 at full probe.

## What each parity file asserts

Each parity file calls one Triton kernel directly and a hand-rolled
torch reference (`_ref(...)`) on the same inputs, then
`assert_topk_matches` (set + sorted-score tolerance) — except for
exact-by-construction kernels, which use stricter assertions.

| File | Kernel | Reference | Notes |
|------|--------|-----------|-------|
| [`test_fused_masked_knn_topk.py`](../../retrieve/tests/parity/test_fused_masked_knn_topk.py) | `fused_masked_knn_topk`     | `compact_mask` → gather + bmm + topk         | mirrors `PrefilterKNN._forward_prefilter` |
| [`test_oporp_1bit_match_topk.py`](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) | `oporp_1bit_match_topk`     | `popcount_int64(xor).sum(W)` → topk          | popcount is bit-exact by construction (SWAR matches between torch and Triton) |
| [`test_bloom_match.py`](../../retrieve/tests/parity/test_bloom_match.py)                     | `bloom_match`               | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to keep the test from tautologically routing through the kernel via `BloomFilter.evaluate_mask` | parametrize on `(n, m_bits, k_hash)` |
| [`test_clause_compact.py`](../../retrieve/tests/parity/test_clause_compact.py)               | `clause_compact`            | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` | row-set match (kernel order is unspecified — atomic stream compaction); cases for reverse clauses, all-inactive query, no-passing-items, `B=1` grid corner |
| [`test_codesigned_probe_score.py`](../../retrieve/tests/parity/test_codesigned_probe_score.py) | `codesigned_probe_score`  | `_ref_phase23`: bloom subset + INT8 dequant + dot + topk | the SilverTorch fused path; cases for `(qb, no qb) × (mask, no mask)` |

## Adding a new test

### New correctness test for a public class

1. One file per class or helper group. Match an existing file's shape
   (e.g. [`test_scorers.py`](../../retrieve/tests/correctness/test_scorers.py))
   for a small surface, [`test_linr.py`](../../retrieve/tests/correctness/test_linr.py)
   for cross-backend / cross-version coverage.
2. Use the fixtures from `tests.conftest` — `make_index`, `make_query`,
   `make_mask`, `make_attrs`, `make_query_attrs`. Don't roll your own
   tensor builder.
3. Build the torch oracle inline as a `_ref` helper. Keep it
   structurally simpler than the module under test so it's obviously
   correct on inspection.
4. Use `valid_id_set(ids, scores, b)` to compare retrieval outputs;
   compare `set(ids[b].tolist())` if the call returns no `-1` padding.
5. Pin the edge cases: `pass_rate ∈ {0.0, 1.0}`, `B = 1`, `p = 0`,
   `K = N`, `N = 1` where applicable.

### New parity test for a Triton kernel

1. One file per kernel, named `test_<kernel_name>.py`.
2. Import the kernel directly from
   `retrieve.kernels.triton.<subtree>.<kernel>`.
3. Write `_ref(...)` as a pure-torch implementation that does the
   same computation step by step.
4. Use `from tests.parity.conftest import assert_topk_matches` for the
   assertion; do not write a position-equality assertion for fp32 score
   paths.
5. Parametrize over shape (`(b, n, d, k)` or similar) and any kernel-
   internal flag (`HAS_MASK`, `HAS_INDICES`, etc.).
6. Cover degenerate cases: `B = 1`, `p = 0`, all-pass / all-fail
   masks, padding ids (`-1`).

### Determinism & performance constraints

- **Always** seed `torch.Generator(device="cuda").manual_seed(...)`.
  Tests must be reproducible; flaky retries are not acceptable.
- Keep N ≤ ~16k, B ≤ 32 for new tests unless there's a specific
  reason for larger sizes — the suite is interactive on a single GPU.
  Latency / memory at scale belongs in `evaluation/`.
- Use `@pytest.fixture(scope="module")` for any tensor that is built
  once and read many times (`make_index` outputs, IVF index builds).

## Running

CUDA is required. The whole suite is GPU-gated.

The repo is a uv workspace ([root pyproject](../../pyproject.toml)); `retrieve/` and `evaluation/` share a single `.venv` at the workspace root. Running `uv` from inside `retrieve/` discovers the workspace root automatically — no per-subdir sync needed.

```bash
cd retrieve   # or, from the root: uv run --directory retrieve <cmd>

# Full suite (correctness + parity)
uv run pytest tests/ -x -q

# Just one tree
uv run pytest tests/correctness/ -v
uv run pytest tests/parity/ -v

# One file
uv run pytest tests/correctness/test_linr.py -v

# One test
uv run pytest tests/correctness/test_linr.py::TestEdgeCases::test_k_equals_n -v

# Collect-only (no CUDA needed) — verifies file structure and imports
uv run pytest --collect-only tests/
```

There are no markers to enable / disable; there is no `--bench` flag.
The pytest config in [pyproject.toml](../../retrieve/pyproject.toml)
collects `test_*.py` only.
