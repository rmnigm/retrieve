---
title: testing
created: 2026-09-26
updated: 2026-09-26
type: concept
tags: [testing, library]
sources: [retrieve/tests/]
---

# `retrieve` testing

The `retrieve` test suite: its layout, fixtures,
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
├── conftest.py            # global fixtures + CUDA skip gate (+ the `cpu` marker exemption)
├── test_public_api.py     # CPU: retrieve.__all__ / retrieve.modules.__all__ pinned; `import retrieve` registers no kernel
├── correctness/           # module-level semantics vs torch baselines
│   ├── test_bit_knn_base.py        (_PackedBitsKNN base: ctor/buffers/k_bits sentinel/candidates semantics)
│   ├── test_bloom_filter.py
│   ├── test_bloom_hash.py          (bloom_hash builders: chunked vs loop-free equality, seed determinism)
│   ├── test_boundary.py            (the library side of the library / harness contract: SilverTorch + LiNRV1–V4;
│   │                                no wide intermediate in a forward)
│   ├── test_combine_filters.py
│   ├── test_compact.py
│   ├── test_filters.py             (ExactAttributeFilter)
│   ├── test_kmeans.py              (KMeans.fit bit-reproducible; kmeans++ init)
│   ├── test_large_offsets.py       (addressing past 2³¹ elements, one planted case per overflow class; skipped
│   │                                below 48 / 24 GiB free — kernels.md § Addressing)
│   ├── test_linr.py                (PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, OneBitKNN, SimHashKNN × torch / Triton;
│   │                                LiNRV1–V4 torch.equal to the hand-composed primitives; short candidate lists,
│   │                                missing filters)
│   ├── test_op_boundary.py         (every Triton op rejects a non-contiguous item table with `ValueError`)
│   ├── test_quantize.py            (int8, OPORP, popcount)
│   ├── test_retrieval_utils.py     (FullScanKNN, post_filter_topk)
│   ├── test_silvertorch.py         (SilverTorch × filter_mode {none,bloom,exact} × backend {triton,torch,official}; SilverTorchBuilder)
│   ├── test_topk_util.py           (masked_topk / counts_to_valid)
│   └── test_tune_smoke.py          (one tiny sweep point per tune-kernels spec, all 7; CUDA-gated)
├── parity/                # kernel vs pure-torch reference
│   ├── conftest.py        # assert_scores_match / assert_topk_matches / assert_topk_equal /
│   │                      # assert_ids_equal_up_to_ties, poison_empty, the probe-family builders
│   ├── test_accumulation.py                 (every scoring kernel against an fp64 oracle)
│   ├── test_bloom_compact.py
│   ├── test_bloom_match.py
│   ├── test_clause_compact.py
│   ├── test_clause_mask.py
│   ├── test_codesigned_probe_score.py
│   ├── test_codesigned_probe_score_exact.py
│   ├── test_compact_order.py                (ascending item order, run to run and process to process)
│   ├── test_fused_masked_knn_topk.py
│   ├── test_official.py                     (Meta's official ops: gates T1–T7; bit-exact vs Triton on the int32 path;
│   │                                         op schemas pinned)
│   └── test_oporp_1bit_match_topk.py
└── compile/               # torch.compile / torch.export gates
    ├── test_linr_compile.py        # compiled LiNR V1–V3 × clause/bloom: cudagraph_skips == 0, replays == eager
    ├── test_silvertorch_compile.py # compiled replays == eager + zero graph breaks, filter_mode × backend
    └── test_export_kernel_ref.py   # torch.export preserves the kernel reference (triton_op); the op-registry gates
```

### Oracle policy

The shared `retrieve.ops.reference` twin (the
`"torch"` backend) is the default parity oracle — a new parity test
reaches for it first. A private, test-local oracle (an inlined `_ref`)
is written only where independence from the shared twin is the point
(e.g. the test would otherwise route through the very kernel it is
checking, or it exists to catch a defect the shared twin could share),
and the test's docstring says why, the way
[`test_clause_mask.py`](../../retrieve/tests/parity/test_clause_mask.py)
does ("intentionally materializing the `[B, N, C, A_max]` intermediate
this kernel exists to avoid"). At the time of writing,
`test_codesigned_probe_score.py`, `test_codesigned_probe_score_exact.py`,
`test_compact_order.py` and `test_official.py` import `ops.reference`
directly; `test_bloom_match.py`, `test_bloom_compact.py`,
`test_clause_mask.py`, `test_clause_compact.py`,
`test_fused_masked_knn_topk.py` and `test_oporp_1bit_match_topk.py` carry
a private `_ref` — bringing the latter group's docstrings up to the
"say why" half of the policy is unscheduled work, not a claim this page
makes about them today.

The split is **by purpose**, not by module:

- `correctness/` answers *"does this layer compute the right thing?"*
  The oracle is a hand-rolled torch reference (matmul + topk, broadcast
  subset, brute-force compaction). One file per public class or helper
  group.
- `parity/` answers *"does the kernel match the torch path on the same
  inputs?"* The oracle is a pure-torch implementation living inside the
  test file (`_ref`), or — where two backends share one — in
  `parity/conftest.py`. One file per kernel, plus `test_accumulation.py`,
  whose oracle is fp64 rather than the torch path (a same-dtype reference
  shares the kernel's accumulation error and cannot see it). Every file
  whose kernel pads its width (kernels.md § Padding) also runs a
  non-power-of-two case against the oracle at the true width: D = 192 and
  768 (yfcc10m's and pubmed's), OPORP `W = 3` / `12`, bloom `W = 3` / `12`.
- `compile/` answers *"does the compile/export machinery still see the
  kernels?"* — `test_silvertorch_compile.py` warms the
  `reduce-overhead` module up on one query, then replays the captured
  graph on two queries it was not captured on, each **bit-exact** to eager
  (`assert_topk_equal`) with zero `cudagraph_skips`, and captures with
  **zero graph breaks** across
  three `(filter_mode, backend, D)` rows: all three modes on `"triton"`
  at `D=64` (the official backend is eager-only by contract;
  `test_official.py` T7 asserts that it refuses to compile);
  `test_linr_compile.py` compiles the filtered LiNR variants with the
  harness's `graph` settings, warms up five times, then replays on two new
  queries: `cudagraph_skips == 0`, each replay's scores `torch.equal` to
  eager on the same inputs, ids up to ties (the gate on the compaction
  kernels being `custom_op`s, [kernels](kernels.md));
  `test_export_kernel_ref.py` exports a module
  calling `codesigned_probe_score_exact` and asserts the exported graph
  keeps a live reference to the kernel *and* replays bit-identically —
  the regression gate for the "`wrap_triton` stays textually inline"
  invariant. The same file holds the **op-registry gates**, over every
  `retrieve::` schema registered (at least 10, asserted): each op has a
  same-named callable in `retrieve.ops.reference` whose parameter names
  and kinds equal the schema's; no schema declares a mutable argument;
  every tensor input is `torch.equal` to its pre-call clone after the op
  and after its twin; `torch.library.opcheck` passes on the two
  `custom_op`s with a `register_fake` (`clause_compact`,
  `bloom_compact`), every utility included, run under
  `torch.use_deterministic_algorithms(True)`: their tail past `counts` is
  unwritten, and deterministic mode's filled `torch.empty` is what makes
  the eager-vs-AOT comparison of the whole output bitwise.

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
  b, *, atol, rtol)` — per-row id-set comparison with tie
  tolerance at the K-th boundary. The tolerances have no default: a
  caller that omits them gets a `TypeError`. Used from the **correctness** tree
  (e.g. `test_linr.py`) when comparing two retrieval calls (typically
  torch backend vs Triton backend) on a specific row; symmetric-
  difference ids must lie within `atol` of their side's finite-score
  minimum, which absorbs the tiebreak flip caused by `tl.dot` vs torch
  `@` accumulator-order drift. The parity tree's `assert_topk_matches`
  is a loop over it plus a sorted-score check.

The CUDA gate is implemented as
`pytest_collection_modifyitems` — every collected item gets a
`skip` marker if CUDA is missing, except items marked `pytest.mark.cpu`
(registered in `pyproject.toml`; today only `test_public_api.py`).

A second, narrower gate covers the optional official backend:

- `require_official()` — call it from a test that needs
  `SilverTorch(backend="official")`, Meta's `torch.ops.st.*` ops (the
  `official` extra). It splits the two ways the extension can be absent,
  and the split is the point: it skips on `OfficialMissing`
  (`silvertorch` not installed, or no CUDA device — so a CPU-only box
  collects and skips the whole official surface) and fails on any other
  `ImportError` (`silvertorch` imports but its `_C` extension did not
  build / load, or the pinned sha lacks an op the adapter calls) — a
  build failure that silently skipped would look green on the first GPU
  run. Memoized in the adapter after the first call.

### [`tests/parity/conftest.py`](../../retrieve/tests/parity/conftest.py)

The probe-layout builders, four assertions and the poisoned allocator:

- `make_probe_family(b, n_lists, max_size, n_probe, *, empty_rate=0.1,
  seed=7)` — a synthetic CSR probe (`ProbeLayout`: distinct `probe_ids`
  per row, `cluster_offsets` with some empty clusters and one at
  `max_size`, a random `sort_perm`, the compact `width`). The three
  backends and the independent loop oracles of the probe parity files all
  score it, so every backend sees the same candidates.
- `make_bloom(n, b, *, m_bits=512, k_hash=5)` — the transposed scorer's
  inputs (`query_bit_positions`, `bloom_transposed`) and the row-wise
  item and query signatures (`sigs, qb`) they are checked against. No live caller; kept for
  the parity tests of the Triton transposed-bloom kernel (roadmap G-a,
  TF-1).
- `make_words(n, w, seed)` — raw `[N, W]` int64 signatures and `[8, W]`
  query words, half subsets of an item's signature and half random sparse
  words, for the bloom ops at a `W` no `BloomFilter` produces.
- `make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8)` — item
  and query clause attrs plus the reverse flags; `reverse="mixed"` flips
  clause 0 only so one call covers both branches of the XOR.
- `assert_scores_match(out, ref, *, atol, rtol)` — slot-wise: the
  non-finite pattern must be identical (`isfinite`, `isnan`, the sign of
  each infinity), finite slots within `atol + rtol·|ref|`; a failure
  prints the mismatch count, the first slots and both values.
- `assert_topk_matches(out_ids, out_scores, ref_ids, ref_scores, *,
  atol, rtol)` — the tolerance-based kernel-vs-torch assertion, for
  paths that reduce in a different order:
  `assert_topk_id_sets_match` over every row (**sets** of finite-score
  ids per row) and the **sorted descending scores** through
  `assert_scores_match`. No default tolerance.
- `assert_topk_equal(out_ids, out_scores, ref_ids, ref_scores)` — the
  bit-exact top-K gate: `assert_scores_match` at zero tolerance slot by
  slot, then `assert_ids_equal_up_to_ties`.
- `assert_ids_equal_up_to_ties(ids_a, ids_b, scores)` — the id half of a
  bit-exact backend comparison. The caller asserts
  `torch.equal(scores_a, scores_b)` first and passes that tensor in; ids
  must then be equal, or differ only at positions inside a run of tied
  scores whose id multisets match. `torch.topk` documents its tie order
  as "not guaranteed stable across invocations", and ties do occur in
  this data (int32 dots of ~±1e5 over P≈768 → about one tied pair per
  row), so a tie permutation is a top-K property, not a kernel bug. A tie
  run cut by the K boundary is not covered (which tied items made it in
  is arbitrary); a test whose inputs reorder items — the permutation
  identities — therefore runs at `k = P`.
- `poison_empty(monkeypatch, shape)` — patches `torch.empty` so every fp32
  allocation of `shape` is filled with `POISON = 1e30` (outranks every
  real score) and returns the list of hits; the poisoned-output tests
  assert it is non-empty, so a score buffer that moved to another
  allocator fails instead of passing vacuously.

A path that is bit-exact by construction — an integer score (popcount),
the int8 probe scorers (exact int32 dot, the same two fp32 multiplies on
every backend, [kernels.md](kernels.md#numerics)), a kernel against
itself (config override, row alone vs batch, permutation, compiled vs
eager) — uses `assert_topk_equal`: any score drift there is a real
regression, not accumulator-order noise.
[`test_official.py`](../../retrieve/tests/parity/test_official.py)'s T1
does the same between the official int32 path and the Triton kernels.
Only a path that reduces floats in a different order
(`fused_masked_knn_topk`'s `tl.sum` vs `bmm`, an fp16 backend vs an fp32
one) uses `assert_topk_matches`, with its tolerance and reason at the
call site.

## Conventions

- **Framework**: vanilla `pytest`. No `hypothesis`, no `unittest`.
- **Parametrize**: `@pytest.mark.parametrize` for shape and pass-rate
  matrices; class-level parametrize for backend cross-product
  (`@pytest.mark.parametrize("backend", ["torch", "triton"])`, then
  construct `Cls(..., backend=backend)`).
- **One mark**: `cpu`, for the tests that run without CUDA. There is no
  `slow`, `gpu` or `bench` marker; every other test is GPU-bound, and the
  CUDA skip gate handles no-GPU hosts globally.
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
- **Score tolerance**: none by default. Exact paths (above) compare
  with `torch.equal` / `assert_topk_equal`: the bit-KNNs torch vs Triton,
  `SilverTorch` torch vs Triton, every probe-scorer and OPORP parity test,
  mask-all-True ≡ unmasked. A stated tolerance remains only where the two
  sides round differently: `fused_masked_knn_topk` vs `bmm` at
  `atol=1e-6` (measured ≤ 6e-8); `atol=2e-6` where cuBLAS's truncating
  tensor-core accumulator meets an fp32 sum (`PrefilterKNN` torch vs
  Triton, `PostfilterKNN` vs `PrefilterKNN`; measured ≤ 1.1e-6); and
  `atol=rtol=1e-3` where one side scores the fp32 table and the other its
  fp16 copy (the `B = 1` edge case) or comes from the official fp16 path
  (T2 bounds it at 2⁻¹⁰ relative).
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
| `LiNRV1`–`LiNRV4`          | the primitives composed by hand on the same inputs (`torch.equal` scores, ids up to ties) |
| `SilverTorchBuilder` / `LiNRBuilder` `set_state_dict` | a fresh `set_item_embeddings` build of the same seed (buffers `torch.equal`, forwards equal) |
| `KMeans(init="kmeans++")`  | the seeds are distinct rows of the index, one per blob on separable blobs; inertia ≤ `init="random"` there |
| `PostfilterKNN` semantics | `(q @ x.T).masked_fill(~mask, -inf).topk(k)` |
| `PrefilterKNN` semantics      | gather + bmm + local topk + scatter |
| `OneBitKNN` semantics         | `FullScanKNN` recall (asserts ≥ 0.4 at K=200, N=2048) |
| `SimHashKNN` semantics        | `FullScanKNN` recall at parametrized `k_bits`, plus a quality-lift-over-OPORP check at higher `k_bits` |
| `masked_topk` / `counts_to_valid` | hand-built score/mask tensors per edge case (empty rows, `counts < k`, all-masked, `-inf` ties, `gather_ids` mapping, `pad_to_k` both ways) |
| `bloom_hash` builders         | chunked `build_signatures` vs loop-free `build_query_signatures` row-wise equality (incl. across the chunk boundary via a monkeypatched batch size) |
| `backend="torch"` vs `"triton"` | cross-backend agreement within each class (same module, both backends) |
| `ops/tune.py` specs           | one tiny regime per `KernelTuneSpec` runs end-to-end; registry covers all seven kernel specs |
| `fused_masked_knn_topk`    | `compact_mask(mask)` → `bmm(q.unsqueeze(1), embs[ids].transpose(1,2)).squeeze(1)` → topk |
| `oporp_1bit_match_topk`    | `popcount_int64(xor) → D - 2*hamming` → topk (bit-exact) |
| `bloom_match`              | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to avoid tautology with the kernel-routed `BloomFilter.evaluate_mask`  |
| `bloom_compact`            | `compact_mask(bloom_match(qb, sigs))` — kernel called directly with a hand-built `qb` to keep the parity check honest |
| `clause_mask`              | Pure-torch `[B, N, C, A_max]` broadcast inlined as `_ref_mask` — intentionally materializes the intermediate this kernel exists to avoid; bit-exact via `torch.equal` since no compaction order ambiguity |
| `clause_compact`           | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` (the mask path itself routes to `clause_mask` on CUDA, so this transitively cross-checks both kernels) |
| `codesigned_probe_score`   | `retrieve.ops.reference.codesigned_probe_score` / `_bloom`: bloom subset + INT8 dequant + dot + `masked_topk` |
| `codesigned_probe_score_exact` | `retrieve.ops.reference.codesigned_probe_score_exact`: `clause_subset_match` over the gathered attrs + INT8 dequant + dot + `masked_topk`. `test_official.py` shares it, and additionally asserts `torch.equal` against the Triton op |

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
- Cross-backend: torch ↔ Triton across
  `pass_rate ∈ {None, 0.01, 0.1, 0.8}`. The bit-KNNs (`OneBitKNN` incl.
  `k_bits < D`, `SimHashKNN`) are `torch.equal` on scores, ids up to
  ties; `PrefilterKNN` goes through `assert_topk_id_sets_match` at
  `atol=2e-6`, `rtol=0` (both fp32 sums of the same fp16 products, different order). (For `PostfilterKNNInt8` the
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
    for `OneBitKNN` — both `torch.equal` on scores, ids up to ties.
  - mask-all-False produces no finite scores.
  - `OneBitKNN` zero `counts` returns full `(-1, -inf)` sentinels.
  - `PrefilterKNN` `candidate_ids` shape `[B, 0]` returns full padding
    (`(-1, -inf)` × K).
  - `B = 1` single-query.
  - `K = N` returns every item (a permutation of `[0, N)`).
- `reduce-overhead` compile parity: zero graph breaks, then after the
  warm-up two replays on queries the graph was not captured on, each
  `assert_topk_equal` to eager
  (`test_reduce_overhead_compile_zero_graph_breaks_and_parity`); the
  `fullgraph` compile tests likewise `assert_topk_equal`.

- `backend` validation: every LiNR class and `ExactAttributeFilter`
  raise `ValueError("unknown backend …")` for `"official"`, `"cuda"` and a
  typo — `LinrBackend` is `torch | triton`, no silent torch fallback.
- The composites (`TestComposites`, the composites' bit-exactness gate):
  `LiNRV1` ≡ `PostfilterKNN` + `filter.evaluate_mask`, `LiNRV2` ≡
  `PrefilterKNN` over `filter.evaluate_indices`, `LiNRV3` ≡ `OneBitKNN` →
  `PrefilterKNN` bounded by the survivors' count, `LiNRV4` ≡
  `PostfilterKNNInt8` + mask — on `torch` and `triton` × filter kind
  `{none, clause, bloom}` (V2: the two filter kinds), scores `torch.equal`,
  ids `assert_ids_equal_up_to_ties`. V3's filter is the `torch` one on
  every row; both backends emit the same ascending candidate order, so the
  choice is immaterial. `register_index(embs, attrs,
  reverse)` registers the attached filter, under the `filter.` prefix.

### [`test_silvertorch.py`](../../retrieve/tests/correctness/test_silvertorch.py)

`SilverTorch` — all three filter modes (`"none"`, `"bloom"`, `"exact"`)
across all three backends (`"torch"`, `"triton"`, `"official"`) in one
file. Official cells call `require_official()`, which skips only when the
extra is not installed (a broken build *fails* the test); every
mode/backend combination is otherwise exercised. The buffer checks read the CSR layout every
backend registers (`cluster_offsets` / `sort_perm` / `inv_perm`, the
cluster-sorted `item_codes` and `item_clause_attrs`, `_probe_width`) and
the cross-backend rows cover `none` / `exact` / `exact`-reverse against
Triton — bloom is Meta's own hash there, so its gates are semantic and
live in `test_official.py`.

- Output shape `(B, K)` with `query_clause_attrs`, without it, on
  `filter_mode="none"`, and on `filter_mode="exact"`; ids long, scores
  float32.
- Buffer dtypes/shapes on the no-filter module: `item_codes` int8,
  centroid / cluster shapes; `bloom_transposed` / `hash_seeds` /
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
  cached scalars poisoned (`_global_scale_f = nan`, `_probe_width =
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
- Cross-backend agreement: `torch` vs `triton` on all three modes plus
  reverse clauses on `filter_mode="exact"`, and `official` vs `triton` on
  `none` / `exact` / `exact`-reverse. torch vs Triton is `torch.equal`
  on scores, ids up to ties (the same int32 dot and fp32 dequant);
  official vs Triton compares **id sets** at `atol=rtol=1e-3` (the
  official default is the fp16 path); the
  kernel-level parity suite (`test_official.py` T1/T6) separately proves
  official↔triton int32 scores bit-identical on a shared index (and ids
  equal up to tie permutation).
- `SilverTorchBuilder` (`TestBuilder`, every backend × filter mode):
  `set_state_dict(src.state_dict()).build()` equals a *fresh*
  `set_item_embeddings(x).build()` of the same seed — key order, every
  buffer `torch.equal`, the two cached scalars, forwards (`torch.equal`
  scores, ids up to ties); `build_timings` is `{}` on the prebuilt module;
  the full chain (`set_item_attributes` with reverse flags, `set_backend`,
  `set_device`); `build()` with neither or both sources raises.
- `kmeans_init` (`TestKMeansInit`): the default is `"random"`
  ([decisions](../decisions.md#library): the recorded numbers depend on it); a `"kmeans++"` module builds on
  `torch` / `triton`, its centroids differ from random init's, are
  reproducible, and `build_timings["kmeans_s"] > 0`.
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
- `clause_salt` as a buffer: `build_signatures(...,
  clause_salt=buffer)` ≡ the on-the-fly salt ≡ an inline computation
  replicated in the test, bit for bit, on CUDA *and* on
  CPU (the hash core is plain tensor math); the salt is device-
  independent; a wrong-length salt raises; `BloomFilter` and
  `SilverTorch(filter_mode="bloom")` expose it in `state_dict`, it
  follows `.cpu()` / `.cuda()`, and an attribute-less `SilverTorch`
  bloom index carries an empty salt and derives it at query time.
- `build_transposed_sigs`: the transposed index round-trips bit by bit
  against the row-wise signatures, with `N` not a multiple of 64 (a
  zero pad tail); `build_query_bit_positions` scattered is
  `build_query_signatures` bit for bit, `-1` exactly at inactive
  clauses' slots.

### [`test_kmeans.py`](../../retrieve/tests/correctness/test_kmeans.py)

`KMeans.fit` is bit-reproducible: the order-fixed reduction (integer
`bincount` counts, a float64 one-hot GEMM for the sums) against a kept
copy of the float-atomic `index_add_` reduction, whose summation order
follows block scheduling, at the 200k/128/1024 gate size and a
chunk-tail size, on CUDA and CPU. And `init="kmeans++"`:

- `n_iter=0` exposes the seeds: each is a row of the index, all distinct
  (a point at zero D² mass is never redrawn — `searchsorted(right=True)`),
  one per blob on 16 separable blobs, and the returned assignments are
  `KMeans.assign` of the returned centroids.
- Deterministic per seed (two seed-0 fits `torch.equal`; seed 1 differs).
- Inertia ≤ random init on the separable blobs at `n_iter ∈ {0, 5}`
  (random init lands two seeds in one blob almost surely and Lloyd's cannot
  recover).
- An unknown `init` raises.

### [`test_boundary.py`](../../retrieve/tests/correctness/test_boundary.py)

The library side of the library-harness boundary
([decisions](../decisions.md#harness)), for
each of `SilverTorch` (`filter_mode="exact"`), `LiNRV1`–`LiNRV4` (with an
`ExactAttributeFilter`) on a tiny index, on `triton` and `torch`:

- `module.k = k'` changes the output width without re-registration and
  leaves every buffer untouched.
- `buffers()` covers every tensor attribute: no `torch.Tensor` in any
  submodule's `__dict__` outside `_buffers`, and the state dict is exactly
  the named buffers.
- A forward under `torch.cuda.set_sync_debug_mode("error")` raises nothing
  (after one warm-up call for the Triton JIT). torch itself warns that the
  mode "does not yet detect all synchronizing operations"; the harness's
  `cudagraph_skips == 0` is the stronger evidence.
- `set_state_dict(...).build()` through the class's builder equals a fresh
  build: key order, buffers, forward.
- `capturable` is defined on the class (a `bool` or a property), never
  stamped on the instance; `True` on both backends, `False` on
  `SilverTorch(backend="official")` (gated by `require_official`).
- `DISPATCH` names every algo-level class × the three backends, every key
  is a `retrieve.modules` class, `None` cells raise `ValueError("unknown
  backend")` at construction.
- `set_query_params`: `SilverTorch(n_probe=…)` re-runs the two
  `register_index` validations (`<= n_lists`, probe pool `>= k`);
  `LiNRV3(candidate_pool=…)` re-validates against `N`.
- `build_timings`: the four keys in phase order, non-negative floats,
  `kmeans_s > 0`; `{}` before registration.
- No wide intermediate: a `TorchDispatchMode` records the output shape of
  every op a second forward dispatches; on Triton no ≥ 3-d tensor exceeds
  `B·N` elements (a `[B, P, D]` gather or a `[B, N, C, A_max]` broadcast
  would); the torch backend, which materializes `[B, ·, C, A_max]`, must
  trip the same check — the control that the recorder sees anything.
  Custom ops are opaque to the recorder (it sees the `retrieve::` call and
  its outputs, not the kernel's own buffers), so this gates module-level
  code only.

### [`test_tune_smoke.py`](../../retrieve/tests/correctness/test_tune_smoke.py)

CUDA-gated tuner smoke: `KERNELS` registry covers all seven kernel
specs, and each spec's `run(make_inputs(dev, smoke_regime), config)`
executes one tiny sweep point — schema drift between `ops/tune.py` and the
kernel `_impl`s breaks CI instead of a tuning session.

## What each parity file asserts

Each parity file calls one Triton kernel directly and a reference (the
shared `ops.reference` twin or a private `_ref(...)`, per the oracle
policy above) on the same inputs, then `assert_topk_equal` (bit-exact) —
except `fused_masked_knn_topk`, which goes through `assert_topk_matches`
at `atol=1e-6`, and the mask / compaction kernels, which compare with
plain `torch.equal`.

The top-K kernel files (`fused_masked_knn_topk`, both probe scorers,
OPORP) also carry, bit-exact: a **poisoned-output** test
(`poison_empty` on the score buffer's shape, hit asserted, no `POISON` in
the output, parity with the reference); **cutoff** runs on both sides of
the code's own constants (`_P_BUCKETS[0]` / `_N_BUCKETS[0]` ± 1 and
`P % DEFAULT_CONFIG.block_*` ∈ {0, 1}, the regime asserted inside the
test); **degenerate rows** (`count = 0` → `(-1, -inf)` in every slot,
`count = 1` → the one candidate then `(-1, -inf)`; the fused file's
older test covers its own); **row alone ≡ row in batch**, and — on
`fused_masked_knn_topk`, `codesigned_probe_score` and OPORP full —
**item-table permutation** permutes the ids and leaves the scores
equal (at `k = P`). Variant identities: OPORP indirect over `arange(N)`
≡ full scan; `codesigned_probe_score_bloom` with an all-zero query
signature ≡ `codesigned_probe_score`; `clause_compact` ≡
`compact_mask(clause_mask(...))` on counts and the `[:counts]` prefix
(`N % block_n` ∈ {0, 1}, one row passing nothing). The two compaction
files' poisoned-output tests assert the opposite of the scorers': the
`[:counts]` prefix equals the reference and every slot past it **still
holds the poison** — the scatter writes nothing there
([kernels](kernels.md#clause_compact--fused-clause-eval--stream-compaction)).

| File | Kernel | Reference | Notes |
|------|--------|-----------|-------|
| [`test_fused_masked_knn_topk.py`](../../retrieve/tests/parity/test_fused_masked_knn_topk.py) | `fused_masked_knn_topk`     | `compact_mask` → gather + bmm + topk         | mirrors `PrefilterKNN._forward_prefilter` |
| [`test_oporp_1bit_match_topk.py`](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py) | `oporp_1bit_match_topk`     | `popcount_int64(xor).sum(W)` → topk          | popcount is bit-exact by construction (SWAR matches between torch and Triton): `assert_topk_equal`; the private oracle writes `-1` at every `-inf` slot, the op's sentinel contract |
| [`test_bloom_match.py`](../../retrieve/tests/parity/test_bloom_match.py)                     | `bloom_match`               | `(qb & sigs) == qb` per word, AND-reduced — computed on CPU to keep the test from tautologically routing through the kernel via `BloomFilter.evaluate_mask` | parametrize on `(n, m_bits, k_hash)` |
| [`test_bloom_compact.py`](../../retrieve/tests/parity/test_bloom_compact.py)                 | `bloom_compact`             | `compact_mask(bloom_match(qb, sigs))` with hand-built `qb` | rows equal in order (the kernel's contract is ascending item order); covers `B=1`, all-inactive query, `N < BLOCK_N`, and a routed-via-`BloomFilter.evaluate_indices` smoke check |
| [`test_compact_order.py`](../../retrieve/tests/parity/test_compact_order.py)                 | `clause_compact`, `bloom_compact` | `ops.reference.clause_compact` / `bloom_compact` | the determinism gate: ids **and** counts `torch.equal` in order (ascending item order) over `make_exact` × reverse ∈ {none, mixed} × `(C, A_max)` and `make_bloom` × `m_bits` ∈ {256, 512, 1024}, at one tile / many tiles / a ragged last tile; the order holds under three non-default tile configs; the same call is `torch.equal` on counts and the `[:counts]` prefix ten launches running and in a fresh interpreter; and **every consumer ignores the tail** — `LiNRV2` (Triton / torch rescoring, clause / bloom), `LiNRV3` (Triton / torch) and `combine_indices`, run in a subprocess with the compaction's `torch.empty` poisoned to `-1` and then to `2**40` (past every table), return `torch.equal` results; a consumer that used a tail id would differ, one that gathered through it would fault (hence the subprocess). Goes red with the reference op's `counts` bound removed, or with `LiNRV2` dropping `counts` |
| [`test_clause_mask.py`](../../retrieve/tests/parity/test_clause_mask.py)                     | `clause_mask`               | Pure-torch `[B, N, C, A_max]` broadcast inlined as `_ref_mask` | bit-exact via `torch.equal`; covers reverse clauses, all-reverse, all-inactive query, `A_max=1`, `B=1`, `N < BLOCK_N` |
| [`test_clause_compact.py`](../../retrieve/tests/parity/test_clause_compact.py)               | `clause_compact`            | `ExactAttributeFilter.evaluate_mask(...)` + `compact_mask` (transitively goes through `clause_mask` on CUDA) | rows equal in order (ascending item order); cases for reverse clauses, all-inactive query, no-passing-items, `B=1` grid corner |
| [`test_codesigned_probe_score.py`](../../retrieve/tests/parity/test_codesigned_probe_score.py) | `codesigned_probe_score` (+ bloom) | the shared `ops.reference` twin, plus two private oracles: a per-row loop concatenating the probed clusters (no shared slot arithmetic) and the row-wise bloom subset test (the transposed index must answer as the row-wise signatures do) | the compact CSR probe layout; cluster-size cutoffs at `block_p`; bit-exact |
| [`test_codesigned_probe_score_exact.py`](../../retrieve/tests/parity/test_codesigned_probe_score_exact.py) | `codesigned_probe_score_exact` | `_ref_phase23`: exact AND-of-OR predicate over narrow attrs + INT8 dequant + dot + topk | the SilverTorch exact-fused path; cases for active vs all-inactive query clauses; bit-exact |
| [`test_accumulation.py`](../../retrieve/tests/parity/test_accumulation.py) | `fused_masked_knn_topk`, `codesigned_probe_score`, `codesigned_probe_score_exact`, `oporp_1bit_match_topk_indirect`; the exact LiNR scorers `PostfilterKNN` (± mask) and `PrefilterKNN` (dense, torch and Triton candidates) | **fp64** computation of the same operation on the same inputs, every candidate returned (`k = P`) and mapped back by id | the accumulation-width gate the other parity files cannot be (they compare both sides at the same input dtype, so a width defect cancels). Goodreads-like unnormalised fp16 inputs; the fp32 dot within `1e-4` abs of fp64 where an fp16 tree reduction of the same products is asserted to miss it; the int8 kernels' fp32 dequant within `1e-6` rel of fp64 where fp16 cannot hold the int32 dot; the popcount path `torch.equal` to the int64 truth; the LiNR scorers on YFCC-shaped near ties (every row's top-100 inside ~20 fp16 quanta): fp32 scores within `2e-6` of fp64 and the fp64 top-k selected up to ties inside that bound, where an fp16 score of the same dots is asserted to miss (recall < 0.95; measured 0.88 / 0.92) |
| [`test_official.py`](../../retrieve/tests/parity/test_official.py) | Meta's `torch.ops.st.fused_kmean_ann` / `_with_partial_masks`, `bloom_index_build`, `parse_expression_query_batch`, `bloom_index_search_batch` / `_return_partial_response` through [`ops/official/`](../../retrieve/src/retrieve/ops/official/adapter.py) | `retrieve.ops.reference`, the Triton `_impl`s (plain and exact), `clause_mask`, and the `SilverTorch` layer itself | The official parity gates T1-T7. **T1** int32 path: scores `torch.equal` vs the reference and vs Triton (plain at `D ∈ {64, 128}`, `D=96` reference-only, a 90-wide cluster for the remainder path; exact via `clause_mask` → `pack_mask` → `filtering_bit_mask`, with reverse clauses), ids up to ties after normalising `-inf` slots to the `-1` sentinel; the raw op contract (int32/int32, width `round_up(·, 32)`, `INT32_MIN`/`-1` pads, returned positions = the probed set). **T2** fp16 path: `max_rel_err ≤ 2⁻¹⁰`, no overflow, `jaccard@32 ≥ 0.99`, and `default_divisor` is the overflow bound. **T3** bit order, pinned to the measured order ([kernels](kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend); `OFFICIAL_BIT_ORDER = "high_first"`, a test-side constant the adapter's `MASK_BIT_ORDER` / `BLOOM_OUTPUT_BIT_ORDER` must equal): the packed bloom output round-trips `pack_mask` under the pinned order and not the other (README corpus, README hits reproduced); a one-doc `filtering_bit_mask` over a 70-doc cluster (docs 0 / 5 / 40 / 69: both mask words, warp and remainder paths) is honoured under the pinned order and, under the other, scores exactly the mirrored doc `63 − d % 64` of the same word (nothing when that lies past the cluster); `unpack_partial_mask` decodes under the pinned order only; pack / unpack / `reverse_bits64` round trip. **T4** official bloom ⊇ exact on the full mask and on the partial masks (which must equal the full mask on the probed docs), FPR at `b_multiplier=10` recorded and `< 5 %`; `NOT` terms have **no false positives** (⊆ exact — the guarantee flips for a complemented bloom; false-negative rate recorded); expression mapping and the LRU plan cache. **T5** `_with_partial_masks` scores `torch.equal` the full-mask scores and the unfiltered scores masked by the bloom (paper §4.3). **T6** the layer on the Triton module's transplanted index (GPU k-means is not bit-deterministic): int32 path `torch.equal` vs Triton on `none` / `exact` / `exact`-reverse (+ all-inactive queries), fp16 default `jaccard@K ≥ 0.99`, bloom (`m_bits=None`, both `bloom_path`s bit-identical to each other and to the `cache_plans=False` run) ⊆ Triton's exact top-K up to bloom false positives — both after normalising `-inf` slots to the `-1` sentinel, since the Triton epilogue leaves the padded item id there while the official one goes through `masked_topk`, state-dict key order, candidates path through `inv_perm` bit-equal to Triton, state-dict round trip, attribute-less bloom index raises on attribute queries, `k_hash > 10` rejected. **Schemas**: `str(torch.ops.st.<op>.default._schema)` equals the string pinned for 21aa35e, for exactly the adapter's `REQUIRED_OPS`. **T7** `module.compile()` and a `fullgraph` `torch.compile` raise (`RuntimeError`), default-mode compile raises or matches eager; host syncs per op counted under `set_sync_debug_mode("warn")` — c10's `warn_or_error_on_sync` lines captured on fd 2 (a sync inside a C++ op never becomes a Python warning) plus Python-side warnings (aten syncs never reach fd 2), the two being disjoint — printed and recorded, asserted `> 0` for the scorer; and per `SilverTorch(backend="official")` forward on every filter path with `cache_plans` on and off (3 / 3 / 7 / 4 for none / exact / bloom-partial / bloom-full on the A100). Gated by `require_official()` |

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
   `retrieve.ops.triton.<kernel>` (the file; the package attribute of the
   same name is the op). Score it against the same-named op in
   `retrieve.ops.reference` — the default oracle (see the oracle policy
   above). Write a private `_ref(...)` pure-torch implementation instead
   only where independence from the shared twin is the point, and say why
   in the docstring.
4. Use `assert_topk_equal` from `tests.parity.conftest` when the path is
   exact by construction, else `assert_topk_matches` with the tolerance
   and a one-line reason at the call site.
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

There is no marker to select and no `--bench` flag.

**After editing a `triton_op` body, run the compile gates on a fresh inductor cache**
(`TORCHINDUCTOR_CACHE_DIR=$(mktemp -d)`). The on-disk FX-graph / AOT-autograd caches key a graph
on the custom op, not on the Python source of its `@triton_op` body. With a warm default
cache (`/tmp/torchinductor_<user>`), a compiled call can replay the *old* body. Measured:
after the OPORP candidate path changed its output width, four `test_linr.py` compile gates
failed with the pre-change width on the warm cache and passed on a cold one. The same
hazard applies to any compiled (`graph`-mode) harness run on a box whose cache predates a
library change.
The pytest config in [pyproject.toml](../../retrieve/pyproject.toml)
collects `test_*.py` only.
