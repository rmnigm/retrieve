# `mask_compact` — Triton stream-compaction over a pre-built bool mask

> **Status: deferred.** Drop-in performance swap behind a stable API
> (`compact_mask` in [layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py)).
> Ship only when profiling justifies on a real workload — see "When to
> actually do this" below.

The current host-side [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py#L7-L22)
takes a `[B, N]` bool and returns `(positive_indices[B, P], counts[B])`
via:

```python
counts = mask.sum(dim=1)
p = int(counts.max().item())          # host sync
sorted_idx = mask.float().argsort(    # fp32 [B, N] alloc + O(N log N) stable sort
    dim=1, descending=True, stable=True
)[:, :p]
```

Two things are wrong with this for large `N` on CUDA:

1. **`mask.float()` allocates a fresh `[B, N]` fp32**, 4× the original
   bool. At `B=64, N=10M` that's a transient ~2.5 GB just to feed the
   sort.
2. **`argsort(stable=True)` is O(N log N)** with the giant fp32 keys
   buffer driving the bandwidth cost. We don't actually need a sort —
   we need stream compaction (which is `O(N)`).

We already have the right kernel template in the codebase:
[`bloom_compact`](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py)
and [`clause_compact`](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py)
both use `tl.cumsum` (intra-tile) + `tl.atomic_add` (inter-tile base
offset) + `tl.store` to compact predicate hits in one launch. Their
*tail* is exactly what we want; their *head* is fused predicate
evaluation we don't want here. `mask_compact` is the same epilogue with
the inner predicate replaced by a single `tl.load` of the pre-built bool.

## Applicability

Three call sites end up here today.

### 1. `LiNR_V3_Triton` masked path — *the only hot CUDA caller*

[v3_triton.py:51](../../retrieve/src/retrieve/layers/linr/v3_triton.py#L51):

```python
positive_indices, counts = compact_mask(mask)
return oporp_1bit_match_topk(
    query_bits, self.item_bits, self.k,
    positive_indices=positive_indices, counts=counts,
)
```

V3 always compacts on the masked path because popcount is cheap enough
that the gather penalty never crosses the dense-fallback break-even
([kernels.md §"Layer dispatch"](../system/kernels.md#layer-dispatch)).
So every V3-with-external-mask query pays the `mask.float().argsort()`
cost before any kernel even launches. At production scale (`N` in the
1M–1B range) the compaction is plausibly larger than the popcount kernel
itself.

This is the call site that justifies the kernel.

### 2. `combine_indices` cascade re-compaction

[layers/filters/\_\_init\_\_.py:66](../../retrieve/src/retrieve/layers/filters/__init__.py#L66)
in the sparse-cascade helper:

```python
sub_mask = f.evaluate_subset(q, safe_ids)        # [B, P] bool
sub_mask = sub_mask & valid
new_counts = sub_mask.sum(dim=1)
new_p = int(new_counts.max().item())
...
sorted_idx = sub_mask.float().argsort(dim=1, descending=True, stable=True)[:, :new_p]
ids = ids.gather(1, sorted_idx)
```

Same `mask.float().argsort(stable)` anti-pattern, but on `[B, P]` rather
than `[B, N]`. After the first filter, `P` is typically small ("caller
orders filters most-selective first"), so the per-step cost is bounded —
but the cascade runs this once per additional filter. Not a bottleneck
on its own; comes for free if `mask_compact` exists. Reuse via
`mask_compact(sub_mask)` after the inline `sub_mask & valid`, then
`ids = ids.gather(1, mask_compact_result_ids)`.

### 3. `FilterModule.evaluate_indices` ABC default + per-filter CUDA fallbacks

The [ABC default in interfaces.py:43](../../retrieve/src/retrieve/interfaces.py#L43)
is `compact_mask(self.evaluate_mask(qa))`. Both shipping filters
(`ClauseIndex`, `BloomFilter`) override this on CUDA with their own
fused compact kernels (`clause_compact`, `bloom_compact`), so the ABC
fallback only fires:

- on CPU (where this whole plan is irrelevant — `compact_mask` stays
  pure-torch),
- for any *future* `FilterModule` subclass that hasn't shipped a fused
  compact kernel yet.

For (b), `mask_compact` becomes the "good-enough on CUDA without writing
a fused predicate-and-compact kernel" intermediate option. It strictly
beats the current torch path; it's strictly worse than a
predicate-fused kernel like `clause_compact`. The right discipline
remains: ship a fused kernel (à la `bloom_compact`) when a new filter is
hot. `mask_compact` is the floor, not the ceiling.

### Where it does **not** apply

- **CPU paths.** Keep the current torch implementation as the CPU
  branch. The fp32-argsort cost on CPU is small relative to the
  surrounding torch code, and we don't want to add a CPU compaction
  primitive just for symmetry.
- **`FullScanKNN` / `LiNR_V1` post-mask.** [`post_filter_topk`](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9-L21)
  applies a `[B, K]` mask (already-small) to top-K ids — no compaction
  needed.
- **`combine_masks`** ([layers/filters/\_\_init\_\_.py:13](../../retrieve/src/retrieve/layers/filters/__init__.py#L13)).
  Returns the AND'd mask as-is; the caller decides whether to compact
  downstream. No change.

## When to actually do this

Ship `mask_compact` when **one** of these triggers fires:

- A V3 production / benchmark profile shows `compact_mask` taking
  ≥ ~10% of `LiNR_V3_Triton` masked-path wall time. Easiest signal: a
  spike in `mask.float()` allocation in
  [`evaluation/`](../../evaluation/) memory traces, or a measurable
  drop in throughput when the mask is dense (large `P`).
- Adding a new `FilterModule` subclass that wants `evaluate_indices` on
  CUDA without (yet) a fused compact kernel. `mask_compact` lets the
  ABC default beat the current floor without a kernel-PR.

Don't ship speculatively. The whole point of the API design is that
`compact_mask` is a one-line helper — a kernel that does the same thing
faster is a drop-in swap, not an architectural commitment.

## Design

### Inputs / output

```
inputs:    mask        [B, N]  bool
output:    out_indices [B, N]  int64  (worst-case scratch)
           counts      [B]     int64
return:    positive_indices [B, P] int64   (P = max(counts.max(), 1))
           counts           [B]    int64
```

Output ordering within a row is **unspecified** — atomics across tiles
race with each other. Same convention as
[`clause_compact`](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py)
and `bloom_compact`. Downstream consumers
(`oporp_1bit_match_topk` HAS_INDICES path, `fused_masked_knn_topk`)
only care about the *set* of passing ids; callers needing a stable
order must sort.

### Kernel model

Strip `bloom_compact`'s subset-test inner block, replace with a bool
load:

```
per program (b, n_tile):
    n_offsets = n_tile * BLOCK_N + arange(BLOCK_N)
    n_valid   = n_offsets < N

    pass_mask = tl.load(
        mask_ptr + b * stride_mb + n_offsets * stride_mn,
        mask=n_valid,
        other=False,
    ).to(tl.int1)
    pass_mask = pass_mask & n_valid

    pass_int  = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra     = tl.cumsum(pass_int, axis=0) - 1     # 0-indexed inclusive offset
    tile_sum  = tl.sum(pass_int)

    base      = tl.atomic_add(counts_ptr + b, tile_sum.to(tl.int64))
    write_pos = base + intra.to(tl.int64)

    tl.store(
        out_indices_ptr + b * stride_ob + write_pos * stride_on,
        n_offsets.to(tl.int64),
        mask=pass_mask,
    )
```

Diff vs `bloom_compact`: deletes the `qb` / `sigs` loads, the `&`
subset test, and the `tl.min` AND-reduce. Replaces with a single
elementwise bool load. Tail (cumsum + atomic_add + store) is identical.

### Hyperparams

Single fixed config — same gotcha as `clause_compact` /
`bloom_compact`:

```python
_BLOCK_N = 256
_NUM_WARPS = 4
```

**Do not autotune.** `tl.atomic_add` accumulates across autotune trials
and corrupts `counts`. If a later sweep is needed, re-enable autotune
with `reset_to_zero` covering both `counts_ptr` and `out_indices_ptr`,
and verify behaviour through both the autotune phase and steady-state
calls.

### Host wrapper

Mirror `bloom_compact`'s host wrapper — same shape, fewer args:

```python
def mask_compact(mask: Tensor) -> tuple[Tensor, Tensor]:
    if mask.dim() != 2:
        raise ValueError("mask must be [B, N]")
    if mask.dtype != torch.bool:
        raise TypeError("mask must be bool")
    b, n = mask.shape
    mask = mask.contiguous()

    out_indices = torch.empty((b, n), dtype=torch.int64, device=mask.device)
    counts = torch.zeros((b,), dtype=torch.int64, device=mask.device)

    grid = (b, triton.cdiv(n, _BLOCK_N))
    _mask_compact_kernel[grid](
        mask, out_indices, counts,
        N=n,
        stride_mb=mask.stride(0), stride_mn=mask.stride(1),
        stride_ob=out_indices.stride(0), stride_on=out_indices.stride(1),
        BLOCK_N=_BLOCK_N, num_warps=_NUM_WARPS,
    )

    p = max(int(counts.max().item()), 1)
    return out_indices[:, :p].contiguous(), counts
```

The host sync on `counts.max().item()` stays. Removing it (returning
`out_indices` at full width and letting downstream kernels bound by
`counts`) would be the next optimization, but downstream kernels expect
a tight `P` — that's a separate refactor (touches `oporp_1bit_match_topk`
and `fused_masked_knn_topk` allocation paths). Out of scope for this
plan.

### Where to put it

- New file: `retrieve/src/retrieve/kernels/triton/filters/mask_compact.py`.
  Lives next to its structural twins `clause_compact.py` and
  `bloom_compact.py`; the three should be diffable.
- Export: add to
  [`retrieve/src/retrieve/kernels/triton/filters/__init__.py`](../../retrieve/src/retrieve/kernels/triton/filters/__init__.py).

### Wiring

Update [`compact_mask`](../../retrieve/src/retrieve/layers/utils/compact.py)
to route by device. Keep the function name and signature so no caller
changes:

```python
def compact_mask(mask: Tensor) -> tuple[Tensor, Tensor]:
    if mask.is_cuda:
        from retrieve.kernels.triton.filters.mask_compact import mask_compact
        return mask_compact(mask)
    # CPU fallback: existing torch implementation.
    counts = mask.sum(dim=1)
    p = int(counts.max().item())
    if p == 0:
        b = mask.shape[0]
        return (
            torch.zeros(b, 0, dtype=torch.long, device=mask.device),
            counts,
        )
    sorted_idx = mask.float().argsort(dim=1, descending=True, stable=True)
    return sorted_idx[:, :p], counts
```

`combine_indices` ([layers/filters/\_\_init\_\_.py:66](../../retrieve/src/retrieve/layers/filters/__init__.py#L66))
should also switch to `compact_mask(sub_mask)` instead of its inline
`sub_mask.float().argsort(...)` once the CUDA path is faster — same
function, same API, free win in the cascade.

## Verification

New parity test
`retrieve/tests/parity/test_mask_compact.py`, modelled on
[`test_bloom_compact.py`](../../retrieve/tests/parity/test_bloom_compact.py):

- For random `[B, N]` bool masks across `(B, N)` cells matching the
  existing parity-test matrix, `mask_compact(mask)` matches
  `compact_mask`'s torch behaviour:
  - `counts` byte-equal,
  - `positive_indices` row-wise equal as a *set* (not a sequence —
    atomics produce arbitrary order, just like `clause_compact` /
    `bloom_compact`).
- Edge cases: all-True mask (every item passes), all-False mask
  (`counts == 0` everywhere; `P == 1` after the `max(., 1)` clamp),
  ragged rows (counts varying widely across `b`).

A bench cell `mask_compact_triton` next to the existing compact bench
cells in the [`evaluation/`](../../evaluation/) harness, comparing
against the current `compact_mask` torch path. The interesting metric
is *transient peak bytes* (the eliminated `[B, N]` fp32 alloc) as much
as wall time.

## Risk

Lowest-risk in the kernel family — it's `bloom_compact` minus the
predicate. The atomic-add hazard is the only thing to watch:

- Writes to `out_indices` from different tiles must use the atomic's
  returned base offset. Copying `bloom_compact`'s launch boilerplate
  copies the right behaviour.
- Don't autotune. (Stated above; restated because it's the one way to
  silently corrupt `counts`.)

No new numerical surface — it's bool in, int64 out.

## Out of scope (don't bundle)

- **A "hide the host sync" rewrite.** Returning `out_indices` at full
  width and pushing `counts`-bounded reads into downstream kernels is a
  larger refactor across `oporp_1bit_match_topk` and
  `fused_masked_knn_topk` host wrappers. Worth doing, but separately.
- **CPU compaction primitive.** Keep the torch fallback. CPU users are
  not the target; cluttering with a CPU `mask_compact` for symmetry
  isn't worth the maintenance.
- **Cross-query batching** (the larger optimization for the actually-
  hot full-scan kernels — `oporp_1bit_match_topk` no-indices,
  `bloom_match`, `bloom_compact`, `clause_mask`, `clause_compact`).
  Different kernel-shape change, different plan.
- **Subsuming `clause_compact` / `bloom_compact` by routing through
  `evaluate_mask` + `mask_compact`.** Strictly worse — the existing
  fused predicate-and-compact kernels avoid the `[B, N]` bool
  intermediate that this composition would re-introduce.

## Touch list when implementing

- Create: `retrieve/src/retrieve/kernels/triton/filters/mask_compact.py`
- Modify: `retrieve/src/retrieve/kernels/triton/filters/__init__.py` (export)
- Modify: `retrieve/src/retrieve/layers/utils/compact.py` (CUDA-route `compact_mask`)
- Modify: `retrieve/src/retrieve/layers/filters/__init__.py` (`combine_indices` reuses `compact_mask`)
- Create: `retrieve/tests/parity/test_mask_compact.py`
- Modify: bench cell in [`evaluation/`](../../evaluation/) harness
- Modify: [`docs/system/kernels.md`](../system/kernels.md) (kernel table; add a
  `mask_compact` row next to `clause_compact` / `bloom_compact`)
- Modify: [`docs/system/filtering.md`](../system/filtering.md) (note the new
  CUDA path in the `compact_mask` description)