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
│   ├── test_bit_knn_base.py        (_PackedBitsKNN base: ctor/buffers/k_bits sentinel/candidates semantics)
│   ├── test_bloom_filter.py
│   ├── test_bloom_hash.py          (bloom_hash builders: chunked vs loop-free equality, seed determinism)
│   ├── test_combine_filters.py
│   ├── test_compact.py
│   ├── test_filters.py             (ExactAttributeFilter)
│   ├── test_linr.py                (PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, OneBitKNN, SimHashKNN × torch / Triton)
│   ├── test_quantize.py            (int8, OPORP, popcount)
│   ├── test_retrieval_utils.py     (FullScanKNN, post_filter_topk)
│   ├── test_silvertorch.py         (SilverTorch, all three filter_modes: none / bloom / exact)
│   ├── test_topk_util.py           (masked_topk / counts_to_valid)
│   └── test_tune_smoke.py          (one tiny sweep point per tune-kernels spec; CUDA-gated)
├── parity/                # Triton kernel vs pure-torch reference
│   ├── conftest.py        # assert_topk_matches helper
│   ├── test_bloom_compact.py
│   ├── test_bloom_match.py
│   ├── test_clause_compact.py
│   ├── test_clause_mask.py
│   ├── test_codesigned_probe_score.py
│   ├── test_codesigned_probe_score_exact.py
│   ├── test_fused_masked_knn_topk.py
│   └── test_oporp_1bit_match_topk.py
└── compile/               # torch.compile / torch.export gates
    ├── test_silvertorch_compile.py # compiled == eager + zero graph breaks, all three filter_modes
    └── test_export_kernel_ref.py   # torch.export preserves the triton_op kernel reference
```

The split is **by purpose**, not by module:

- `correctness/` answers *"does this layer compute the right thing?"*
  The oracle is a hand-rolled torch reference (matmul + topk, broadcast
  subset, brute-force compaction). One file per public class or helper
  group.
- `parity/` answers *"does the Triton kernel match the torch path on
  the same inputs?"* The oracle is a pure-torch implementation living
  inside the test file (`_ref`). One file per kernel.
- `compile/` answers *"does the compile/export machinery still see the
  kernels?"* — `test_silvertorch_compile.py` asserts the compiled
  forward matches eager and captures with **zero graph breaks** on all
  three filter_modes; `test_export_kernel_ref.py` exports a module
  calling `codesigned_probe_score_exact` and asserts the exported graph
  keeps a live reference to the Triton kernel *and* replays
  bit-identically (the regression gate for the "`wrap_triton` stays
  textually inline" invariant).

A module that has both a torch and a Triton backend (like LiNR V1/V2/V3)
appears in both trees: `test_linr.py` covers semantics, the parity
files cover kernel agreement.

## Fixtures and helpers

### Root [`conftest.py`](../../retrieve/tests/conftest.py)

Five fixture functions — all return CUDA tensors with deterministic seeds:

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

Four evaluation helpers:

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
  (`@pytest.mark.parametrize("backend", ["torch", "triton"])`, then
  construct `Cls(..., backend=backend)`).
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
  reductions (`PostfilterKNN` / `PrefilterKNN` paths); strict
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
| `SilverTorch` (qa=None)    | `SilverTorch (no bloom)` directly — `query_clause_attrs=None` is a documented fast path |
| `PostfilterKNN` semantics | `(q @ x.T).masked_fill(~mask, -inf).topk(k)` |
| `PrefilterKNN` semantics      | gather + bmm + local topk + scatter |
| `OneBitKNN` semantics         | `FullScanKNN` recall (asserts ≥ 0.4 at K=200, N=2048) |
| `SimHashKNN` semantics        | `FullScanKNN` recall at parametrized `k_bits`, plus a quality-lift-over-OPORP check at higher `k_bits` |
| `masked_topk` / `counts_to_valid` | hand-built score/mask tensors per edge case (empty rows, `counts < k`, all-masked, `-inf` ties, `gather_ids` mapping, `pad_to_k` both ways) |
| `bloom_hash` builders         | chunked `build_signatures` vs loop-free `build_query_signatures` row-wise equality (incl. across the chunk boundary via a monkeypatched batch size) |
| `backend="torch"` vs `"triton"` | cross-backend agreement within each class (same module, both backends) |
| `tune.py` specs               | one tiny regime per `KernelTuneSpec` runs end-to-end; registry covers all seven kernels |
| `fused_masked_knn_topk`    | `compact_mask(mask)` → `bmm(q.unsqueeze(1), embs[ids].transpose(1,2)).squeeze(1)` → topk |
| `oporp_1bit_match_topk`    | `popcount_int64(xor) → D - 2*hamming` → topk (bit-exact) |
| `bloom_match`              | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to avoid tautology with the kernel-routed `BloomFilter.evaluate_mask`  |
| `bloom_compact`            | `compact_mask(bloom_match(qb, sigs))` — kernel called directly with a hand-built `qb` to keep the parity check honest |
| `clause_mask`              | Pure-torch `[B, N, C, A_max]` broadcast inlined as `_ref_mask` — intentionally materializes the intermediate this kernel exists to avoid; bit-exact via `torch.equal` since no compaction order ambiguity |
| `clause_compact`           | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` (the mask path itself routes to `clause_mask` on CUDA, so this transitively cross-checks both kernels) |
| `codesigned_probe_score`   | `_ref_phase23` in the parity file: bloom subset + INT8 dequant + dot + topk |
| `codesigned_probe_score_exact` | `_ref_phase23` in the parity file: exact AND-of-OR predicate + INT8 dequant + dot + topk |

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
- Cross-compat smoke: `combine_masks(clause_mask, bloom_mask)` →
  `compact_mask` → `OneBitKNN(backend="triton")` candidates path
  (the combined mask should equal the clause mask, since bloom is a
  strict superset; every returned id must satisfy the clause filter).

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

LiNR V1, V2, V3, V4 plus `SimHashKNN` in both backends.

- V1 (`PostfilterKNN`): top-K is sorted descending; mask path
  returns ids satisfying the mask.
- V2 (`PrefilterKNN`): full path (no candidates) returns sorted top-K;
  candidate path returns ids ⊆ candidates and respects `counts`;
  default `counts` (all P) treated as all-valid.
- V3 (`OneBitKNN`): full-scan recall vs `FullScanKNN` ≥ 0.4; candidate
  path returns ids ⊆ candidates; `torch.compile(fullgraph=True)`
  compiles without a break (also at `k_bits < D`).
- `SimHashKNN`: same shape of coverage as V3 (recall vs `FullScanKNN`,
  candidates ⊆, fullgraph compile) parametrized over `k_bits` incl.
  `k_bits > D`; plus a quality-lift assertion vs OPORP at higher
  `k_bits`.
- V4 (`PostfilterKNNInt8`): full-scan recall vs `FullScanKNN` ≥
  0.95 at K=10 on unit-norm random embeddings (the int8 quantization
  preserves topk ordering modulo per-element rounding); mask path
  returns ids satisfying the mask; small-batch `_int_mm` padding corner.
- Cross-backend: torch ↔ Triton return identical valid-id sets per
  row for `PostfilterKNN`, `PrefilterKNN`, `OneBitKNN`, `SimHashKNN`
  across `pass_rate ∈ {None, 0.01, 0.1, 0.8}`. Sorted scores `allclose`
  (atol=1e-3) for fp32 paths; **strict equality** for the bit-KNNs
  (popcount is exact by construction). (For `PostfilterKNNInt8` the
  `backend=` flag is a no-op — cuBLAS LtGemm runs the same code on
  both paths.)
- `PostfilterKNN` (mask path) ≡ `PrefilterKNN` (compact_mask of same
  mask) as id sets.
- `ExactAttributeFilter` decoupled composition: `PostfilterKNN` with
  `mask = ef.evaluate_mask & extra`, `PrefilterKNN` with
  `compact_mask(combined)`, `PrefilterKNN` with `ef.evaluate_indices`.
- Edge cases (`TestEdgeCases`):
  - mask-all-True ≡ unmasked path (`PostfilterKNN`,
    `PostfilterKNNInt8` × torch / Triton); all-candidates ≡ unmasked
    for `OneBitKNN`.
  - mask-all-False produces no finite scores.
  - `OneBitKNN` zero `counts` returns full `(-1, -inf)` sentinels.
  - `PrefilterKNN` `candidate_ids` shape `[B, 0]` returns full padding
    (`(-1, -inf)` × K).
  - `B = 1` single-query.
  - `K = N` returns every item (a permutation of `[0, N)`).
- `reduce-overhead` compile parity: one compiled forward per family
  matches eager with zero graph breaks
  (`test_reduce_overhead_compile_zero_graph_breaks_and_parity`).

### [`test_silvertorch.py`](../../retrieve/tests/correctness/test_silvertorch.py)

`SilverTorch` — all three filter modes (`"none"`, `"bloom"`, `"exact"`) in one file.

- Output shape `(B, K)` with `query_clause_attrs`, without it, on
  `filter_mode="none"`, and on `filter_mode="exact"`; ids long, scores
  float32.
- Buffer dtypes/shapes on the no-filter module: `item_codes` int8,
  centroid / cluster shapes; `bloom_sigs` / `hash_seeds` /
  `item_clause_attrs` are *not* allocated. The exact-mode module
  allocates the `item_clause_attrs` buffer (`[N, C, A_max]` long)
  instead of the bloom buffers.
- Param validation: unknown `filter_mode`, `m_bits` non-power-of-2,
  `k_hash <= 0`, partial bloom config (only one of `m_bits`/`k_hash`
  set), bloom params passed with `filter_mode="none"` or `"exact"`,
  `filter_mode="exact"` without `item_clause_attrs`,
  `clause_is_reverse` outside `filter_mode="exact"`, `n_lists > N`, and
  `n_probe > n_lists` all raise `ValueError`. Passing
  `query_clause_attrs` or `item_clause_attrs` to a no-filter module
  raises.
- **Equivalence**: `query_clause_attrs=None` on a bloom-configured module
  ≡ a freshly built no-filter module with the same kmeans seed; same
  invariant holds for `filter_mode="exact"`, plus an all-inactive query
  (`[-1, ...]`) on `filter_mode="exact"` matches the no-filter result.
- Recall ≥ 0.90 at `n_probe = n_lists`; recall ≥ 0.85 at full probe and
  monotone in `n_probe`.
- Candidate-ids path returns ids ⊆ candidates (with bloom, with exact,
  without filter, and with `p < k` — the short-`P` case returns
  `min(k, P)` columns, no pad). Passing `query_clause_attrs` together
  with `candidate_ids` raises `ValueError` (the candidates path would
  silently skip the fused filter otherwise).
- Cross-backend (`torch` vs `triton`) agreement on all three modes plus
  reverse clauses on `filter_mode="exact"`.
- `build_silvertorch` builder round-trips (bloom / no-filter / exact /
  torch backend).
- Edge case: `n_lists = N` (one item per cluster) → recall ≥ 0.85 at
  full probe.

### [`test_topk_util.py`](../../retrieve/tests/correctness/test_topk_util.py)

`masked_topk` / `counts_to_valid` — the shared torch-side top-K
epilogue every layer's torch path routes through.

- `counts_to_valid` prefix-mask shape/values.
- Empty rows (`counts = 0`), `counts < k`, all-masked rows → `-1` /
  `-inf` sentinels.
- Tie-at-`-inf` boundary with `valid`; no sentinel pass when `valid` is
  `None` and `P >= k`.
- `gather_ids` local→global mapping, with and without a mask.
- `pad_to_k` both ways at `P < k`; pad/mask interaction; output dtypes.

### [`test_bit_knn_base.py`](../../retrieve/tests/correctness/test_bit_knn_base.py)

`_PackedBitsKNN` base-class contract shared by `OneBitKNN` / `SimHashKNN`.

- Class hierarchy (both subclass the base and `RetrievalModule`) and
  constructor attrs.
- Buffer names and registration order per subclass (`item_bits` +
  `oporp_signs`/`oporp_perm` vs `item_bits` + `simhash_R`).
- `k > N` raises at `register_index`; `item_bits` shape and `d_total`.
- `k_bits=0` sentinel resolves to `D`; explicit `k_bits` kept;
  re-registering with a different-dim corpus re-resolves the sentinel
  (from the pristine constructor arg).
- Torch-eager candidates path: `P < k` returns `min(k, P)` columns (no
  pad — frozen behavior), zero `counts` returns sentinels, returned ids
  ⊆ candidates, full-scan shape.

### [`test_bloom_hash.py`](../../retrieve/tests/correctness/test_bloom_hash.py)

`bloom_hash` signature builders (the property the fused paths assume).

- `build_signatures(attrs)[i] == build_query_signatures(attrs[i:i+1])[0]`
  row-wise — chunked and loop-free paths agree exactly, including
  across the `_BUILD_SIGS_BATCH` chunk boundary (monkeypatched small).
- `generate_seeds` is deterministic and odd.

### [`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py)

CUDA-gated tuner smoke: `KERNELS` registry covers all seven kernel
specs, and each spec's `run(make_inputs(dev, smoke_regime), config)`
executes one tiny sweep point — schema drift between `tune.py` and the
kernel `_impl`s breaks CI instead of a tuning session.

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
| [`test_bloom_compact.py`](../../retrieve/tests/parity/test_bloom_compact.py)                 | `bloom_compact`             | `compact_mask(bloom_match(qb, sigs))` with hand-built `qb` | row-set match (kernel order is unspecified — atomic stream compaction); covers `B=1`, all-inactive query, `N < BLOCK_N`, and a routed-via-`BloomFilter.evaluate_indices` smoke check |
| [`test_clause_mask.py`](../../retrieve/tests/parity/test_clause_mask.py)                     | `clause_mask`               | Pure-torch `[B, N, C, A_max]` broadcast inlined as `_ref_mask` | bit-exact via `torch.equal`; covers reverse clauses, all-reverse, all-inactive query, `A_max=1`, `B=1`, `N < BLOCK_N` |
| [`test_clause_compact.py`](../../retrieve/tests/parity/test_clause_compact.py)               | `clause_compact`            | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` (transitively goes through `clause_mask` on CUDA) | row-set match (kernel order is unspecified — atomic stream compaction); cases for reverse clauses, all-inactive query, no-passing-items, `B=1` grid corner |
| [`test_codesigned_probe_score.py`](../../retrieve/tests/parity/test_codesigned_probe_score.py) | `codesigned_probe_score`  | `_ref_phase23`: bloom subset + INT8 dequant + dot + topk | the SilverTorch bloom-fused path; cases for `(qb, no qb)` |
| [`test_codesigned_probe_score_exact.py`](../../retrieve/tests/parity/test_codesigned_probe_score_exact.py) | `codesigned_probe_score_exact` | `_ref_phase23`: exact AND-of-OR predicate over narrow attrs + INT8 dequant + dot + topk | the SilverTorch exact-fused path; cases for active vs all-inactive query clauses |

## Adding a new test

### New correctness test for a public class

1. One file per class or helper group. Match an existing file's shape
   (e.g. [`test_retrieval_utils.py`](../../retrieve/tests/correctness/test_retrieval_utils.py))
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
   `retrieve.kernels.<subtree>.<kernel>`.
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

# Full suite (correctness + parity + compile)
uv run pytest tests/ -x -q

# Just one tree
uv run pytest tests/correctness/ -v
uv run pytest tests/parity/ -v
uv run pytest tests/compile/ -v

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
