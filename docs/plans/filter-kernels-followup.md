# Deferred filter kernels — `bloom_compact` and `clause_mask`

> **Status: deferred.** Both kernels are pure performance swaps over a
> stable API contract. Ship only when profiling justifies.

The unified filter API ([filtering-api.md](filtering-api.md), shipped) left
two kernels on the table by design:

1. **`bloom_compact`** — a fused subset-test + stream-compaction kernel
   for `BloomFilter.evaluate_indices`. Today the default falls through to
   `compact_mask(bloom_match(.))`, which materializes `[B, N]` bool and
   then argsorts. The fused kernel would eliminate the `[B, N]`
   intermediate the same way `clause_compact` already does for
   `ClauseIndex`.
2. **`clause_mask`** — a fused clause-evaluation kernel for
   `ClauseIndex.evaluate_mask`. Today the path is
   `(q == ic).any(-1).all(-1)` pure-torch broadcast, which materializes a
   `[B, N, C, A_max]` bool intermediate (`C·A_max`× the output mask).
   The fused kernel would do the equality + reduction in registers and
   emit `[B, N]` directly.

Neither is necessary today; both are structurally simple. This plan
captures the design so a future agent can implement them on demand
without re-deriving the rationale.

## When to actually do this

Don't ship either kernel speculatively. The trigger is profile data,
specifically:

- **`bloom_compact`** — when the V2 / V3 candidate-id path with
  `BloomFilter` is in production use AND a filter-bench profile shows
  `compact_mask` (host-side argsort + slice) takes a
  meaningful share of `BloomFilter.evaluate_indices` wall time. The
  compact step's cost grows with N and the pass-rate; bloom is approximate
  with looser pass-rates than clause, so this is more likely to bite at
  large N than `ClauseIndex` does.
- **`clause_mask`** — when V1 with `ClauseIndex` is on the critical path
  AND a profile shows the `[B, N, C, A_max]` intermediate is causing
  HBM pressure (transient peak shown by the `evaluation/` benchmark
  harness) or graph breaks are preventing inductor from fusing the
  broadcast.

If neither trigger fires, leave the defaults in place. The whole point
of the API design is that the kernels are drop-in: callers don't change.

## `bloom_compact` — design

**Goal.** Replace
[`BloomFilter.evaluate_indices`](../../retrieve/src/retrieve/layers/filters/bloom.py)'s
ABC-default fallthrough with a fused kernel:

```python
def evaluate_indices(self, query_clause_attrs):
    qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
    if qb.is_cuda:
        from retrieve.kernels.triton.filters.bloom_compact import bloom_compact
        return bloom_compact(qb, self.bloom_sigs)
    return compact_mask(self.evaluate_mask(query_clause_attrs))
```

**Model.** Lift the structure from
[`clause_compact`](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py).
Replace the per-clause inner loop with the bloom subset test from
[`bloom_match`](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py):

```
per program (b, n_tile):
    n_offsets = n_tile * BLOCK_N + arange(BLOCK_N)
    n_valid   = n_offsets < N

    qb   = load(qb_ptr  + b*stride_qb_b + arange(W)*stride_qb_w)         # [W]
    sigs = load(sig_ptr + n_offsets[:, None]*stride_s_n
                        + arange(W)[None, :]*stride_s_w,
                mask=n_valid[:, None])                                   # [BLOCK_N, W]

    masked     = qb[None, :] & sigs                                      # [BLOCK_N, W]
    pass_word  = (masked == qb[None, :]).to(int32)                       # [BLOCK_N, W]
    pass_mask  = (tl.min(pass_word, axis=1) != 0) & n_valid              # [BLOCK_N]

    pass_int  = where(pass_mask, 1, 0).to(int32)
    intra     = cumsum(pass_int, axis=0) - 1
    tile_sum  = sum(pass_int)

    base      = atomic_add(counts_ptr + b, tile_sum.to(int64))
    write_pos = base + intra.to(int64)

    store(out_ptr + b*stride_ob + write_pos*stride_on,
          n_offsets.to(int64),
          mask=pass_mask)
```

**Output.** `(positive_indices [B, P] int64, counts [B] int64)` where
`P = max(counts.max(), 1)`. Order within a row is unspecified (atomics
across tiles) — same convention as `clause_compact`. V2's
`fused_masked_knn_topk` consumes this set-not-order, so callers don't
care.

**Hyperparams.** Single fixed config like `clause_compact` —
`_BLOCK_N = 256`, `_NUM_WARPS = 4`. Don't autotune: `tl.atomic_add`
accumulates across autotune trials and corrupts `counts`. Same gotcha
that bit `clause_compact`.

**Where to put it.** New file
`retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py`. Mirrors
`clause_compact.py` so the two are easy to diff. Update
[`retrieve/src/retrieve/kernels/triton/filters/__init__.py`](../../retrieve/src/retrieve/kernels/triton/filters/__init__.py)
to export it.

**Wiring.** Override `evaluate_indices` in `BloomFilter`. CPU branch
keeps the ABC default (`compact_mask(evaluate_mask)`).

**Verification.** New parity test
`retrieve/tests/parity/test_bloom_compact.py`, modelled on
[`test_bloom_match.py`](../../retrieve/tests/parity/test_bloom_match.py) +
[`test_filters.py::test_evaluate_indices_matches_compact_evaluate_mask`](../../retrieve/tests/correctness/test_filters.py):

- `(positive_indices, counts)` from `bloom_compact` matches
  `compact_mask(bloom_match(qb, sigs))` row-by-row as sets, counts
  byte-equal.
- Sweep `(N, m_bits, k_hash)` cells matching the existing
  `test_bloom_match.py` matrix.
- A bench cell `bloom_compact_triton` next to the existing
  `bloom_match_triton` cell in the `evaluation/` benchmark harness so
  the win-vs-fallback is measurable.

**Risk.** Atomic-add ordering on the `counts_ptr` buffer means writes to
`out_indices` from different tiles must use the atomic's returned base
offset — same hazard `clause_compact` already documents. If you copy the
launch boilerplate, you copy the right behaviour.

## `clause_mask` — design

**Goal.** Replace
[`ClauseIndex.evaluate_mask`](../../retrieve/src/retrieve/layers/filters/clause.py)'s
pure-torch broadcast with a fused kernel that emits `[B, N]` bool
directly, with no `[B, N, C, A_max]` intermediate.

```python
def evaluate_mask(self, query_clause_attrs):
    if query_clause_attrs.is_cuda:
        from retrieve.kernels.triton.filters.clause_mask import clause_mask
        return clause_mask(
            self.item_clause_attrs,
            self.clause_is_reverse,
            query_clause_attrs,
        )
    # CPU fallback: existing pure-torch broadcast.
    ...
```

**Model.** Strip the cumsum + atomic_add tail off `clause_compact` and
emit a bool tile instead of compact indices:

```
per program (b, n_tile):
    n_offsets = n_tile * BLOCK_N + arange(BLOCK_N)
    n_valid   = n_offsets < N

    pass_mask = full([BLOCK_N], 1, int1)
    for c in static_range(C):
        q_c   = load(query_attrs_ptr + b*stride_qb + c*stride_qc)
        rev_c = load(is_reverse_ptr + c).to(int1)
        clause_match = full([BLOCK_N], 0, int1)
        for a in static_range(A_MAX):
            ia = load(item_attrs_ptr + n_offsets*stride_in
                                     + c*stride_ic + a*stride_ia,
                      mask=n_valid, other=-1)
            clause_match = clause_match | (ia == q_c)
        clause_match = clause_match ^ rev_c
        inactive     = q_c == -1
        clause_match = clause_match | inactive
        pass_mask    = pass_mask & clause_match
    pass_mask = pass_mask & n_valid

    store(out_ptr + b*stride_ob + n_offsets*stride_on, pass_mask, mask=n_valid)
```

**Output.** `[B, N]` bool. No `counts`, no compaction.

**Where to put it.** New file
`retrieve/src/retrieve/kernels/triton/filters/clause_mask.py`. Companion
to `clause_compact.py` — the two kernels share inner-loop structure;
diff is purely in the output epilogue.

**Hyperparams.** Same `BLOCK_N = 256` family. Autotune is *safe* here
(no atomics) but probably not worth the compile-time cost — start with
fixed config and only autotune if a follow-up benchmark shows it
matters.

**Wiring.** Override `evaluate_mask` in `ClauseIndex` with a CUDA-routed
path; keep the existing pure-torch broadcast as the CPU fallback. Note
that `ClauseIndex.evaluate_subset` currently uses the pure-torch
broadcast indirectly via gather — leave that untouched, since
`evaluate_subset` already operates on `[B, P]` (small) rather than
`[B, N]` (large).

**Verification.** Add a parity test
`retrieve/tests/parity/test_clause_mask.py`:

- For random `(attrs, q, is_reverse)`, `clause_mask(...)` equals the
  pure-torch broadcast bit-for-bit.
- Edge cases: all-`-1` query (every item passes), empty `A_max=1`,
  reverse-only-clause, mixed reverse + active + inactive.
- Sweep `(N, C, A_max)` to mirror the `clause_compact` parity matrix.

Plus a bench cell `clause_mask_triton` next to `clause_compact_triton`
in the `evaluation/` benchmark harness. The interesting comparison is
*transient peak bytes* (the broadcast's `[B, N, C, A_max]` intermediate)
— not just wall time.

**Risk.** Lower than `bloom_compact`. No atomics, no cumsum. The kernel
is essentially the existing `clause_compact` with the tail amputated.

## Touch list when implementing

For `bloom_compact`:

- Create: `retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py`
- Modify: `retrieve/src/retrieve/kernels/triton/filters/__init__.py` (export)
- Modify: `retrieve/src/retrieve/layers/filters/bloom.py` (override `evaluate_indices`)
- Create: `retrieve/tests/parity/test_bloom_compact.py`
- Modify: bench cell in `evaluation/` benchmark harness
- Modify: `docs/system/filtering.md` and `docs/system/kernels.md`
  (update the kernel table; remove the "uses ABC default" note from the
  bloom row in filtering.md)

For `clause_mask`:

- Create: `retrieve/src/retrieve/kernels/triton/filters/clause_mask.py`
- Modify: `retrieve/src/retrieve/kernels/triton/filters/__init__.py` (export)
- Modify: `retrieve/src/retrieve/layers/filters/clause.py` (CUDA-route `evaluate_mask`)
- Create: `retrieve/tests/parity/test_clause_mask.py`
- Modify: bench cell in `evaluation/` benchmark harness
- Modify: `docs/system/architecture.md` and `docs/system/kernels.md`
  (kernel table)

Both follow the existing kernel-PR shape — kernel + parity test + bench
cell + doc table refresh. No API changes, no callers touched.

## Out of scope (don't bundle)

- Multi-value query (`[B, C, Q_max]`) for `clause_mask` — same multi-value
  deferral as the rest of the filter API.
- A fused `bloom_compact` that takes `query_clause_attrs` directly and
  builds `qb` inside the kernel. The host-side `_build_signatures`
  already runs once per call and is cheap; folding it in adds register
  pressure with no obvious win. Keep `qb` host-built.
- Touching `codesigned_probe_score` (silvertorch's in-cluster bloom). It
  has its own bloom path and is not affected by either kernel here.
