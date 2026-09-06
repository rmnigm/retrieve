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
│   ├── test_silvertorch.py         (SilverTorch × filter_mode {none,bloom,exact} × backend {triton,torch,cuda,cute,official})
│   ├── test_topk_util.py           (masked_topk / counts_to_valid)
│   └── test_tune_smoke.py          (one tiny sweep point per tune-kernels spec, all 11; CUDA-gated)
├── parity/                # kernel vs pure-torch reference
│   ├── conftest.py        # assert_topk_matches + ref_cps_phase23 + the cuda/cute probe-family builders
│   ├── test_bloom_compact.py
│   ├── test_bloom_match.py
│   ├── test_clause_compact.py
│   ├── test_clause_mask.py
│   ├── test_codesigned_probe_score.py
│   ├── test_codesigned_probe_score_cuda.py  (CUDA backend vs ref *and* bit-exact vs Triton)
│   ├── test_codesigned_probe_score_cute.py  (CuTe DSL backend: same gates, plus bit-exact vs the CUDA backend)
│   ├── test_codesigned_probe_score_exact.py
│   ├── test_fused_masked_knn_topk.py
│   ├── test_official.py                     (Meta's official ops: T1–T7 of the integration plan; bit-exact vs Triton on the int32 path)
│   └── test_oporp_1bit_match_topk.py
└── compile/               # torch.compile / torch.export gates
    ├── test_silvertorch_compile.py # compiled == eager + zero graph breaks, filter_mode × backend
    └── test_export_kernel_ref.py   # torch.export preserves the kernel reference (triton_op + cuda/cute custom_op)
```

The split is **by purpose**, not by module:

- `correctness/` answers *"does this layer compute the right thing?"*
  The oracle is a hand-rolled torch reference (matmul + topk, broadcast
  subset, brute-force compaction). One file per public class or helper
  group.
- `parity/` answers *"does the kernel match the torch path on the same
  inputs?"* The oracle is a pure-torch implementation living inside the
  test file (`_ref`), or — where two backends share one — in
  `parity/conftest.py`. One file per kernel.
- `compile/` answers *"does the compile/export machinery still see the
  kernels?"* — `test_silvertorch_compile.py` asserts the compiled
  forward matches eager and captures with **zero graph breaks** across
  nine `(filter_mode, backend, D)` rows: all three modes on `"triton"`
  at `D=64`, and all three on `"cuda"` at both `D=64` and `D=128`
  (whose `custom_op`s carry the same opacity contract, and where `D`
  selects a different kernel instantiation);
  `test_export_kernel_ref.py` exports a module
  calling `codesigned_probe_score_exact` — on `"triton"` and on
  `"cuda"` — and asserts the exported graph keeps a live reference to
  the kernel *and* replays bit-identically. On triton that is the
  regression gate for the "`wrap_triton` stays textually inline"
  invariant; on cuda it is the gate that export keeps the C++ custom op
  as one opaque node (there is no decomposed form to fall back on).

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

A second, narrower gate covers the CUDA C++ backend:

- `require_cps_cuda()` — call it from a test (or an `autouse` fixture)
  that needs `SilverTorch(backend="cuda")`. It splits the two ways the
  extension can be absent, and the split is the point:
  - **no toolchain** (no CUDA device, or no `nvcc` on PATH / under
    `$CUDA_HOME`) → `skip`. A machine can have a GPU and still lack
    `nvcc`; the suite is expected to run there.
  - **a toolchain that exists and failed to build** → `pytest.fail`
    carrying the nvcc/ninja output. A compile error that silently skipped
    is exactly what a broken first GPU run would look like, and it would
    look green.

  The first call pays the one-time JIT compile; later calls hit the
  outcome memo in the wrapper (and the ninja cache). The wrapper also
  checks the `nvcc` major against `torch.version.cuda` before building,
  because `cpp_extension.load`'s JIT path never runs torch's own
  `_check_cuda_version`.

Note the import inside it is a full module path
(`retrieve.kernels.silvertorch.codesigned_probe_score_cuda`) on purpose:
the package `__init__` re-exports the *op* under the same name as the
module, so a package-level import would bind the op and shadow the
module.

- `require_cps_cute()` — the same gate for `SilverTorch(backend="cute")`,
  the CuTe DSL port of the C++ backend. It skips on `CuteMissing`
  (`nvidia-cutlass-dsl` — the `cute` extra — not installed, or no CUDA
  device) and fails on any other `ImportError` (the DSL is present but
  the kernels failed to import or compile), carrying the DSL error text.
  The first call pays the import plus one ~0.1 s kernel compile.

- `require_official()` — the same gate for `SilverTorch(backend=
  "official")`, Meta's `torch.ops.st.*` ops (the `official` extra). It
  skips on `OfficialMissing` (`silvertorch` not installed, or no CUDA
  device — so the Mac collects and skips the whole official surface) and
  fails on any other `ImportError` (`silvertorch` imports but its
  `_C` extension did not build / load, or the pinned sha lacks an op the
  adapter calls). Memoized in the adapter after the first call.

### [`tests/parity/conftest.py`](../../retrieve/tests/parity/conftest.py)

Six helpers — three fixtures that build synthetic probe layouts and
three assertions / references:

- `make_probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1,
  seed=7)` — a synthetic padded IVF layout (`padded [n_lists, max_size]`
  with `-1` pads, each id at most once), the probed cluster ids and the
  flattened `[B, P]` probe pool the layer's phase 1 would produce. The
  cuda / cute parity files and `test_official.py` build both the padded
  and the CSR view from it, so every backend scores the same candidates.
- `make_bloom(n, b, padded, *, m_bits=512, k_hash=5)` — row-wise item
  signatures, their transposed (cluster-major) index over `padded` and
  the query signatures (`sigs, sigs_t, qb`). The `sigs_t` half goes at
  roadmap B4 with the cuda / cute backends.
- `make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8)` — item
  and query clause attrs plus the reverse flags; `reverse="mixed"` flips
  clause 0 only so one call covers both branches of the XOR.
- `assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores, *,
  atol=1e-3, rtol=1e-3)` — the standard kernel-vs-torch assertion.
  Compares **sets** of finite-score ids per row (tie-breaking on the id
  permutation differs between backends, so position-equality is not
  asserted) and **sorted descending scores** with `allclose` (with
  `-inf` replaced by 0 so padded rows still compare cleanly).
- `assert_ids_equal_up_to_ties(ids_a, ids_b, scores)` — the id half of a
  bit-exact backend comparison. The caller asserts
  `torch.equal(scores_a, scores_b)` first and passes that tensor in; ids
  must then be equal, or differ only at positions inside a run of tied
  scores whose id multisets match. `torch.topk` documents its tie order
  as "not guaranteed stable across invocations", and ties do occur in
  this data (int32 dots of ~±1e5 over P≈768 → about one tied pair per
  row), so a tie permutation is a top-K property, not a kernel bug.
- `ref_cps_phase23(query, flat_items, item_codes, global_scale, k, *,
  qb=None, bloom_sigs=None)` — the shared pure-torch reference for
  SilverTorch phases 2+3: int8 × int8 → int32 dot with one global scale
  and a per-row query scale, plus an optional row-wise bloom subset
  test. Computed in fp32 because at these `D` the integer products fit
  the fp32 mantissa exactly, so it is bit-identical to an int32
  accumulator. Both the Triton and the CUDA parity suites score against
  it — the CUDA backend consumes a *transposed* bloom index and a
  cluster-major clause mask, but both predicates are boolean-identical to
  the row-wise forms here, which is what makes one reference legitimate
  for all of them.

Use `assert_topk_matches` for any new parity test; do not write a
position-equality assertion for fp32 score paths.

**The one exception** is
[`test_codesigned_probe_score_cuda.py`](../../retrieve/tests/parity/test_codesigned_probe_score_cuda.py)
(and its cute twin,
[`test_codesigned_probe_score_cute.py`](../../retrieve/tests/parity/test_codesigned_probe_score_cute.py)),
which asserts `torch.equal` on the **scores** between the CUDA and
Triton backends (and `assert_ids_equal_up_to_ties` on the ids). That is
deliberate and stronger than the tolerance-based comparison: the two
backends are provably bit-identical on scores (exact int32 dot,
boolean-identical predicate, same left-associated fp32 dequant — see
[kernels.md](kernels.md#numerics-bit-identical-to-triton-and-how-that-is-kept)),
so any score drift is a real regression rather than accumulator-order
noise.
It also covers the pieces that have no Triton analogue:
`test_build_transposed_sigs_bits` checks the transposed layout bit by
bit against the row-wise index, and `test_bloom_mask_matches_rowwise` /
`test_clause_mask_matches_rowwise` check each phase-2 mask alone —
unpacking every bit of the packed word against the torch predicate, at a
`max_size` that is not a multiple of 64 so the pad tail is exercised.

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
| `tune.py` specs               | one tiny regime per `KernelTuneSpec` runs end-to-end; registry covers all eleven kernel specs (incl. the CUDA and CuTe backends) |
| `fused_masked_knn_topk`    | `compact_mask(mask)` → `bmm(q.unsqueeze(1), embs[ids].transpose(1,2)).squeeze(1)` → topk |
| `oporp_1bit_match_topk`    | `popcount_int64(xor) → D - 2*hamming` → topk (bit-exact) |
| `bloom_match`              | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to avoid tautology with the kernel-routed `BloomFilter.evaluate_mask`  |
| `bloom_compact`            | `compact_mask(bloom_match(qb, sigs))` — kernel called directly with a hand-built `qb` to keep the parity check honest |
| `clause_mask`              | Pure-torch `[B, N, C, A_max]` broadcast inlined as `_ref_mask` — intentionally materializes the intermediate this kernel exists to avoid; bit-exact via `torch.equal` since no compaction order ambiguity |
| `clause_compact`           | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` (the mask path itself routes to `clause_mask` on CUDA, so this transitively cross-checks both kernels) |
| `codesigned_probe_score`   | `_ref_phase23` in the parity file: bloom subset + INT8 dequant + dot + topk |
| `codesigned_probe_score_exact` | `ref_cps_phase23` in `parity/conftest.py`, exact-predicate keywords: `clause_subset_match` over the gathered attrs + INT8 dequant + dot + topk. The cuda and cute twins (`codesigned_probe_score_exact_cuda` / `..._cute`) share it, and additionally assert `torch.equal` against the Triton op |

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
  `actual_k = p` columns (no padding, by current contract). A `-1`-tailed
  candidate tensor (row `b` keeps `2 + b` real ids of 24) never returns
  item `N-1` (where a raw `-1` would wrap), has exactly the real ids
  finite and the tail `-1` / `-inf`, and its finite part matches a
  pad-free re-rank of the same row.
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

- Buffer shape (`bloom_sigs[N, m_bits/64]`, `hash_seeds[k_hash, 2]`; the
  `clause_salt[C]` buffer is covered in `test_bloom_hash.py`).
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

`SilverTorch` — all three filter modes (`"none"`, `"bloom"`, `"exact"`)
across all five backends (`"torch"`, `"triton"`, `"cuda"`, `"cute"`,
`"official"`) in one file. Cuda cells call `require_cps_cuda()`, which
skips only when there is no toolchain at all (a build *failure* fails
the test); cute cells call `require_cps_cute()` likewise; official cells
call `require_official()`; every mode/backend combination is otherwise
exercised. On `"official"` the buffer checks read the CSR layout
(`cluster_offsets` / `sort_perm` / `inv_perm`, no
`padded_cluster_items`; `item_clause_attrs` in cluster-sorted order) and
the cross-backend rows cover `none` / `exact` / `exact`-reverse against
Triton — bloom is Meta's own hash there, so its gates are semantic and
live in `test_official.py`.

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
- **State-dict round trip** (`TestStateDict`, every backend × filter
  mode): a deep copy of a built module with every buffer zeroed and the
  cached scalars poisoned (`_global_scale_f = nan`, `_max_cluster_size =
  -1`) is loaded from the source's `state_dict()`; the load hook must
  restore both scalars, every buffer must be `torch.equal`, and the
  forwards must agree (`torch.equal` scores, ids up to ties). The copy
  is deliberate: a second `register_index` cannot promise the same IVF
  padding width (GPU k-means is not bit-deterministic) and
  `load_state_dict` checks shapes.
- Candidate-ids path returns ids ⊆ candidates (with bloom, with exact,
  without filter, and with `p < k` — the short-`P` case returns
  `min(k, P)` columns, no pad). A `-1`-tailed candidate tensor (row `b`
  keeps `2 + b` real ids of 24, `k = 8`) never returns item `N-1` (where
  a raw `-1` would wrap — through `inv_perm` on official), has `-1` ids
  exactly where scores are `-inf`, `min(k, 2 + b)` finite slots, and a
  finite part `torch.equal` to a pad-free re-rank of the same row (ids
  up to ties). Passing `query_clause_attrs` together
  with `candidate_ids` raises `ValueError` (the candidates path would
  silently skip the fused filter otherwise).
- Cross-backend agreement: `torch` vs `triton`, `cuda` vs `triton` and
  `cute` vs `triton`, each on all three modes plus reverse clauses on
  `filter_mode="exact"`. Layer-level cross-backend checks
  compare **id sets** (accumulator order may flip ties); the
  kernel-level parity suites separately prove cuda↔triton and
  cute↔triton scores bit-identical on a shared probe family (and ids
  equal up to tie permutation). `cute` vs `cuda` is held to
  `torch.equal` through the whole layer — state_dict (same buffers, same
  keys) and scores — on all three modes.
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
- `clause_salt` as a buffer (B5): `build_signatures(...,
  clause_salt=buffer)` ≡ the on-the-fly salt ≡ the pre-B5 inline
  computation (replicated in the test), bit for bit, on CUDA *and* on
  CPU (the hash core is plain tensor math); the salt is device-
  independent; a wrong-length salt raises; `BloomFilter` and
  `SilverTorch(filter_mode="bloom")` expose it in `state_dict`, it
  follows `.cpu()` / `.cuda()`, and an attribute-less `SilverTorch`
  bloom index carries an empty salt and derives it at query time.

### [`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py)

CUDA-gated tuner smoke: `KERNELS` registry covers all eleven kernel
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
| [`test_codesigned_probe_score_cuda.py`](../../retrieve/tests/parity/test_codesigned_probe_score_cuda.py) | `codesigned_probe_score_cuda`, `..._bloom_cuda`, `..._exact_cuda` | shared `ref_cps_phase23` **and** the Triton kernels themselves | two gates per filter mode: tolerance-based vs the reference, and vs Triton at `D ∈ {64, 128, 256}` × `(bloom, no bloom)` / × `(reverse, no reverse)` for exact — `torch.equal` on scores, `assert_ids_equal_up_to_ties` on ids. Also covers `build_transposed_sigs` bit layout, both phase-2 mask kernels alone vs their torch predicates (with a non-multiple-of-64 `max_size` for the pad tail), config plumbing (`block_p`/`num_warps`/`unroll` change nothing observable), the launcher's must-reject configs, the tiny-`max_size` carry-loop stress, and `opcheck` at `d ∈ {64, 128, 256, 96}`. Gated by `require_cps_cuda()` |
| [`test_official.py`](../../retrieve/tests/parity/test_official.py) | Meta's `torch.ops.st.fused_kmean_ann` / `_with_partial_masks`, `bloom_index_build`, `parse_expression_query_batch`, `bloom_index_search_batch` / `_return_partial_response` through [`official.py`](../../retrieve/src/retrieve/kernels/silvertorch/official.py) | `ref_cps_phase23`, the Triton `_impl`s (plain and exact), `clause_mask`, and the `SilverTorch` layer itself | Plan §5.2 gates. **T1** int32 path: scores `torch.equal` vs the reference and vs Triton (plain at `D ∈ {64, 128}`, `D=96` reference-only, a 90-wide cluster for the remainder path; exact via `clause_mask` → `pack_mask` → `filtering_bit_mask`, with reverse clauses), ids up to ties after normalising `-inf` slots to the `-1` sentinel; the raw op contract (int32/int32, width `round_up(·, 32)`, `INT32_MIN`/`-1` pads, returned positions = the probed set). **T2** fp16 path: `max_rel_err ≤ 2⁻¹⁰`, no overflow, `jaccard@32 ≥ 0.99`, and `default_divisor` is the overflow bound. **T3** bit order, pinned to roadmap A3's measurement (`OFFICIAL_BIT_ORDER = "high_first"`, a test-side constant the adapter's `MASK_BIT_ORDER` / `BLOOM_OUTPUT_BIT_ORDER` must equal): the packed bloom output round-trips `pack_mask` under the pinned order and not the other (README corpus, README hits reproduced); a one-doc `filtering_bit_mask` over a 70-doc cluster (docs 0 / 5 / 40 / 69: both mask words, warp and remainder paths) is honoured under the pinned order and, under the other, scores exactly the mirrored doc `63 − d % 64` of the same word (nothing when that lies past the cluster); `unpack_partial_mask` decodes under the pinned order only; pack / unpack / `reverse_bits64` round trip. **T4** official bloom ⊇ exact on the full mask and on the partial masks (which must equal the full mask on the probed docs), FPR at `b_multiplier=10` recorded and `< 5 %`; `NOT` terms have **no false positives** (⊆ exact — the guarantee flips for a complemented bloom; false-negative rate recorded); expression mapping and the LRU plan cache. **T5** `_with_partial_masks` scores `torch.equal` the full-mask scores and the unfiltered scores masked by the bloom (paper §4.3). **T6** the layer on the Triton module's transplanted index (GPU k-means is not bit-deterministic): int32 path `torch.equal` vs Triton on `none` / `exact` / `exact`-reverse (+ all-inactive queries), fp16 default `jaccard@K ≥ 0.99`, bloom (`m_bits=None`, both `bloom_path`s bit-identical to each other and to the `cache_plans=False` run) ⊆ Triton's exact top-K up to bloom false positives — both after normalising `-inf` slots to the `-1` sentinel, since the Triton epilogue leaves the padded item id there while the official one goes through `masked_topk`, state-dict key order, candidates path through `inv_perm` bit-equal to Triton, state-dict round trip, attribute-less bloom index raises on attribute queries, `k_hash > 10` rejected. **T7** `module.compile()` and a `fullgraph` `torch.compile` raise (`RuntimeError`), default-mode compile raises or matches eager; host syncs per op counted under `set_sync_debug_mode("warn")` — c10's `warn_or_error_on_sync` lines captured on fd 2 (a sync inside a C++ op never becomes a Python warning) plus Python-side warnings (aten syncs never reach fd 2), the two being disjoint — printed and recorded, asserted `> 0` for the scorer; and per `SilverTorch(backend="official")` forward on every filter path with `cache_plans` on and off (3 / 3 / 7 / 4 for none / exact / bloom-partial / bloom-full on the A100). Gated by `require_official()` |
| [`test_codesigned_probe_score_cute.py`](../../retrieve/tests/parity/test_codesigned_probe_score_cute.py) | `codesigned_probe_score_cute`, `..._bloom_cute`, `..._exact_cute` | the cuda file's gates (`ref_cps_phase23`, the Triton kernels, the row-wise mask predicates) **and** the cuda backend itself | every test of the cuda file with the cute `_impl`s (must-reject configs are `ValueError`s from the host module), plus the gate the port exists for: `torch.equal` against the **cuda** backend on the raw phase-2 mask words and on the full `[B, P]` phase-3 score buffer, at `d ∈ {64, 128, 256, 96}` × `{none, bloom, exact}` × `unroll ∈ {1, 2, 4}`. Gated by `require_cps_cute()`; the cuda-vs-cute tests also by `require_cps_cuda()` |

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
