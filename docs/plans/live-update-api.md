# Live-update (Upsert / Delete) API — eager-only, no torch.export dependency

## Context

The previous version of this plan was gated on **all 6 phases of the torch-export refactor landing first** ([torch-export-refactor.md](torch-export-refactor.md)). That refactor has not landed: modules under [retrieve/src/retrieve/layers/](../../retrieve/src/retrieve/layers/) still carry `Optional[Tensor]` args, `is_cuda` device-routing branches, and one `.item()` call ([one_bit_knn_triton.py:53](../../retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py#L53)); there is no `evaluation/retrieval/build_export.py` and no `torch.export` usage anywhere in the repo.

This replan delivers the **same eager-mode behavior** (capacity-aware buffers, in-place upsert / delete, valid-mask folded into forward) **without requiring the export refactor as a prerequisite**. Per the LiNR paper's deployment-lessons section, enabling live updates was worth +6% in their production A/B — the value driver is freshness, not the `.pt2` packaging step. The export refactor remains a separate, parallel effort; this plan is designed to compose cleanly with it later (no Optionals or `.item()` introduced on the new upsert / delete code paths) but does not depend on it.

**What changes vs. the previous version:**

- Targets the **current module set**: `OneBitKNN` + `OneBitKNNTriton`, `PrefilterKNN` + `PrefilterKNNTriton`, `SimilarityMasking` + `SimilarityMaskingTriton`, `FullScanKNN`, `BloomFilter`, `ExactAttributeFilter`. (`LiNR_V2 / V3 / ClauseIndex` no longer exist.)
- **Drops the `build_export.py` extension phase** — no export entry point exists today, so there is nothing to extend. Re-add later once the export refactor lands.
- **`SilverTorch` is still deferred** — same reasoning as before: per-cluster overcommit + route-by-centroid is its own substantial effort, and the current `register_index` does kmeans + padded-cluster construction which doesn't live-update cleanly.
- Forward integration adapts to today's `mask: Tensor | None = None` signatures instead of a post-refactor "mask is required" signature. We fold `valid_mask` in regardless of whether the caller passed a mask.

## Approach

### Phase A — Shared scaffolding (`LiveIndexMixin`, interface tweaks)

**Files:**
- `retrieve/src/retrieve/layers/utils/live_update.py` — new
- [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py)

**Steps:**

1. Add `LiveIndexMixin` in a new `live_update.py`:

   ```python
   class LiveIndexMixin:
       """Provides n_active + valid_mask buffers and a live_enabled flag.
       Subclasses call _alloc_live_buffers(capacity, n_initial) inside
       register_index after computing the initial item count N."""

       n_active: Tensor
       valid_mask: Tensor

       def _alloc_live_buffers(self, capacity: int, n_initial: int, device) -> None:
           self.register_buffer("n_active", torch.tensor(n_initial, dtype=torch.int64, device=device))
           valid = torch.zeros(capacity, dtype=torch.bool, device=device)
           valid[:n_initial] = True
           self.register_buffer("valid_mask", valid)
           self._capacity = int(capacity)
           self._live_enabled = True

       def _bump_watermark(self, rows: Tensor) -> None:
           # In-place; no .item(). Keeps upsert export-clean.
           self.n_active.copy_(torch.maximum(self.n_active, rows.max() + 1))

       def _effective_mask(self, B: int, mask: Tensor | None) -> Tensor | None:
           # Layer-side helper: AND caller mask with valid_mask. Returns None
           # when live-update is off AND caller did not pass a mask, preserving
           # today's fast path (no mask materialized).
           if not getattr(self, "_live_enabled", False):
               return mask
           vm = self.valid_mask.unsqueeze(0).expand(B, -1)
           return vm if mask is None else (mask & vm)
   ```

2. **Add non-abstract `upsert` / `delete` defaults** to `RetrievalModule` and `FilterModule` in [interfaces.py](../../retrieve/src/retrieve/interfaces.py): both raise `NotImplementedError`. Subclasses opt in. Do **not** add `mode=` or other export-refactor signature changes here — those belong to the export refactor.

3. **Extend the abstract `register_index` signature** with an optional `capacity: int | None = None` kwarg. Default `capacity=None` means "no overcommit, no live updates" → buffers sized at exactly `N` (today's behavior, byte-for-byte).

### Phase B — Retrieval modules: `OneBitKNN(Triton)`, `PrefilterKNN(Triton)`, `SimilarityMasking(Triton)`, `FullScanKNN`

**Files:**
- [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py), [one_bit_knn_triton.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py)
- [retrieve/src/retrieve/layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py), [prefilter_knn_triton.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn_triton.py)
- [retrieve/src/retrieve/layers/linr/similarity_masking.py](../../retrieve/src/retrieve/layers/linr/similarity_masking.py), [similarity_masking_triton.py](../../retrieve/src/retrieve/layers/linr/similarity_masking_triton.py)
- [retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) — `FullScanKNN`
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py), [test_retrieval_utils.py](../../retrieve/tests/correctness/test_retrieval_utils.py), new `test_live_update.py`

**Steps:**

1. **Capacity-aware `register_index`**. For each retrieval module, accept `capacity: int | None = None`:
   - `OneBitKNN`: `quantize_oporp_1bit(item_embs)` returns the initial bits/signs/perm. Allocate `item_bits` at `[capacity, W]`, copy initial `[:N]`, leave the rest zero (masked by `valid_mask` anyway). Reuse [quantize_oporp_1bit](../../retrieve/src/retrieve/layers/utils/quantize.py) — `signs` / `perm` are frozen at `register_index` time, exactly what upsert needs.
   - `PrefilterKNN`, `FullScanKNN`: allocate `item_embs` at `[capacity, D]`, copy initial slice.
   - `SimilarityMasking`: layout is pre-transposed `[D, capacity]` — keep the layout, upsert writes a column slice.
   - All call `self._alloc_live_buffers(capacity, N, device)` after the initial copy. When `capacity is None`, skip both the over-allocation and the mixin call (legacy path stays byte-identical).

2. **`upsert(rows: Tensor[K], embs: Tensor[K, D]) -> None`** — eager, in-place:
   - `OneBitKNN` / `OneBitKNNTriton`: `bits = _pack_signs_to_int64((embs * signs.to(embs.dtype)).index_select(1, perm))` — same op `quantize_oporp_1bit` does for the initial fill, just over `[K, D]`. Then `self.item_bits.index_copy_(0, rows, bits)`, `self.valid_mask.index_fill_(0, rows, True)`, `self._bump_watermark(rows)`. Triton subclass inherits — no kernel work, just buffer mutation.
   - `PrefilterKNN` / `PrefilterKNNTriton`, `FullScanKNN`: `self.item_embs.index_copy_(0, rows, embs)` + valid_mask + watermark.
   - `SimilarityMasking` / `SimilarityMaskingTriton`: `self.item_embs_t.index_copy_(1, rows, embs.t().contiguous())` + valid_mask + watermark. (Triton variant inherits unchanged.)

3. **`delete(rows: Tensor[K]) -> None`**: `self.valid_mask.index_fill_(0, rows, False)`. Tombstone only — embedding row stays; reuse happens on next upsert to that row. One op.

4. **Forward integration — fold `valid_mask` into the existing mask path** without breaking today's signatures or the no-live-update fast path:

   - `OneBitKNN._forward_full`: replace `if mask is not None: scores.masked_fill_(~mask, -inf)` with `effective = self._effective_mask(query.shape[0], mask); if effective is not None: scores.masked_fill_(~effective, -inf)`. Keep the `where(isfinite, topk_ids, -1)` wrap-up — works because tombstoned slots get `-inf` scores.
   - `OneBitKNN._forward_candidates`: when live-update is on, do `valid = self.valid_mask[candidate_ids]; scores.masked_fill_(~valid, -inf)` before the topk; the post-topk `gather` already uses these scores.
   - `OneBitKNNTriton.forward`: today routes to the unmasked Triton kernel when `mask is None`. With live-update on, treat valid_mask as the mask: `effective = self._effective_mask(B, mask)`; if `effective is None`, take the existing unmasked fast path; otherwise `compact_mask(effective)` → masked kernel. **Performance note**: the masked path is slower than the unmasked one (gather + compact), so live-update is opt-in by capacity and pays its own cost. Document in the docstring.
     - The pre-existing `int(counts.max().item()) == 0` short-circuit at [one_bit_knn_triton.py:53](../../retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py#L53) is **not regressed** by this plan — it's already on the eager path today. The export refactor will remove it later.
   - `PrefilterKNN._forward_full`: when live-update is on, `effective = self.valid_mask[None, :].expand(B, -1)`; apply `masked_fill_(~effective, -inf)` on `scores`. Otherwise unchanged.
   - `PrefilterKNN._forward_prefilter`: gather `valid_in_cand = self.valid_mask[safe_ids]` (`[B, P]`), `scores.masked_fill_(~valid_in_cand, -inf)` before the topk. Tombstoned candidates fall out naturally.
   - `PrefilterKNNTriton.forward`: same treatment in the prefilter Triton path — gather `valid_in_cand` host-side, AND it row-wise into `counts` (or trim `candidate_ids` so only live rows reach the kernel). Pass `(candidate_ids, counts)` to `fused_masked_knn_topk` unchanged.
   - `SimilarityMasking.forward`: same `_effective_mask` substitution as `OneBitKNN._forward_full`.
   - `FullScanKNN.forward`: today applies mask post-topk via [post_filter_topk](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9). Extend that: build effective mask from caller's `mask` AND `valid_mask`, post-filter as today.
   - `FullScanKNN._forward_candidates`: gather `valid_mask[candidate_ids]`, `masked_fill_` before the local topk.

5. **Watermark accuracy**: `n_active` is a soft watermark (per the article — "high-water mark to track the working set"). It is monotonic-increasing; `delete` does not lower it. That's intentional — the watermark bounds the effective row range for compaction-time scans, not for forward correctness (forward correctness is `valid_mask`-only).

6. **Tests** in new `retrieve/tests/correctness/test_live_update.py`:
   - **Rebuild-vs-incremental parity**: insert N items via N `upsert` calls and assert forward output exactly matches `register_index([all N items])`-then-forward.
   - **Delete tombstones**: `register_index(N)` → `delete(some_rows)` → assert deleted ids never appear in topk.
   - **Reuse on upsert**: delete row r, upsert different embedding into row r, assert forward returns the new embedding's neighbors.
   - **Capacity exhausted**: `next_free_rows(too_many)` raises.
   - **No-live-update legacy path**: `register_index(item_embs)` with `capacity=None` produces buffers, forward, and shapes byte-for-byte identical to today (regression guard against accidentally turning live-update on).

### Phase C — Filters: `BloomFilter`, `ExactAttributeFilter`

**Files:**
- [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [retrieve/src/retrieve/layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py)
- [retrieve/tests/correctness/test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py), [test_filters.py](../../retrieve/tests/correctness/test_filters.py)

**Steps:**

1. Filters do **not** maintain their own `valid_mask`. The retrieval module's `valid_mask` is the single source of truth for liveness; the filter only mutates its per-row state to match the retrieval module's row indexing.

2. **`upsert(rows, item_clause_attrs)`**:
   - `ExactAttributeFilter`: `self.item_clause_attrs.index_copy_(0, rows, item_clause_attrs)`. `A_max` is fixed at `register_index` time; caller pads / truncates to fit. `clause_is_reverse` is per-clause, not row-dependent — left untouched.
   - `BloomFilter`: recompute signatures for the K rows on-device by calling [_build_signatures](../../retrieve/src/retrieve/layers/filters/bloom.py#L130) on a `[K, C, A_max]` slice — the function is shape-generic and chunked. Then `self.bloom_sigs.index_copy_(0, rows, new_sigs)`. Per-row independent; no global rebuild.

3. **`delete(rows)`**: no-op for both filters. Documented in docstring; tombstoning lives on the retrieval-side `valid_mask`.

4. Both filters' `register_index` accepts `capacity: int | None = None`. When set, `item_clause_attrs` / `bloom_sigs` allocate at `[capacity, ...]`. Initial slice copied; rest zero. Filters do not call `_alloc_live_buffers` (they have no `valid_mask` of their own); they just over-allocate the row dimension to match the retrieval module's capacity.

### Phase D — Host-only helpers (not in forward)

**File:** `retrieve/src/retrieve/layers/utils/live_update.py` (same file as Phase A).

**Add to `LiveIndexMixin`:**

1. **`next_free_rows(self, n_new: int) -> Tensor[n_new]`** — host-side row allocator. Walks `valid_mask` (CPU copy via `.cpu()`) to find tombstoned slots first, then extends past `n_active` up to `_capacity`. Raises `RuntimeError` if `_capacity` exhausted. Caller passes the returned rows into `upsert`. Uses `.tolist()` / Python loops freely — **off the export path**, so the constraints there don't apply.

2. **`compact(self) -> Tensor[capacity]`** — optional host-side helper. Rebuilds the module's row-indexed buffers so live rows occupy `[0:n_active)` densely; returns an old→new row map so the application can rewrite its ID→row table. Implemented per-module (each one knows its own buffer set); the mixin provides the `valid_mask` walk + remap logic. Reuse [compact_mask](../../retrieve/src/retrieve/layers/utils/compact.py) — it derives the dense old→new permutation from the `[capacity]` valid bool.

### Deferred — `SilverTorch` IVF live updates

Out of scope. The current [silvertorch/main.py register_index](../../retrieve/src/retrieve/layers/silvertorch/main.py#L68) runs kmeans + builds `padded_cluster_items` with a per-cluster `max_size`; live update there needs per-cluster overcommit, route-by-centroid on upsert, and periodic rebalance. Standalone follow-up plan.

### Deferred — Export entries (`.pt2` per algo × {upsert, delete})

Drop from this plan. Re-add as a follow-up once the torch-export refactor lands and `evaluation/retrieval/build_export.py` exists. The upsert / delete code introduced here is **deliberately written to not block future export**: no `.item()` in `upsert` / `delete` bodies, no `Optional[Tensor]` in their signatures (`embs` and `rows` are required, K-row tensors), the watermark update is on-device. The `_effective_mask` helper is layer-side host code — that's fine; the kernel-facing tensors stay clean.

### Phase E — Documentation + concurrency contract

**Files:**
- [docs/system/architecture.md](../system/architecture.md)
- [docs/system/checkpoints.md](../system/checkpoints.md) — capacity restoration on load
- New: `docs/system/live-updates.md`

**Document:**

- The `capacity=` parameter and how to pick it (paper: "pre-allocating larger tensors").
- The application layer owns the ID→row map; module APIs take rows.
- Forward + `upsert` + `delete` must share a CUDA stream OR be serialized via stream sync. Inside the module there is no locking. (Matches the article's "minimal data access serialization.")
- `delete` is a tombstone, not a row free; reuse happens at next `upsert(row, ...)` to that row.
- `SilverTorch` does not support live updates yet; deferred.
- **Note**: this plan's API is eager-only; `torch.export` compatibility is preserved at the per-method level (no `.item()` / no `Optional` in `upsert` / `delete` themselves) but no `.pt2` is produced today. That ships with the future export refactor.

## Critical files to modify

| File | Reason |
|---|---|
| [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) | Add `capacity=None` to abstract `register_index`; non-abstract `upsert` / `delete` defaults |
| `retrieve/src/retrieve/layers/utils/live_update.py` (new) | `LiveIndexMixin` (`n_active`, `valid_mask`, `_bump_watermark`, `_effective_mask`, `next_free_rows`, `compact`) |
| [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py), [one_bit_knn_triton.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py) | Capacity-aware register_index, `upsert` / `delete`, valid_mask in forward |
| [retrieve/src/retrieve/layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py), [prefilter_knn_triton.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn_triton.py) | Same |
| [retrieve/src/retrieve/layers/linr/similarity_masking.py](../../retrieve/src/retrieve/layers/linr/similarity_masking.py), [similarity_masking_triton.py](../../retrieve/src/retrieve/layers/linr/similarity_masking_triton.py) | Same; column-slice upsert into `[D, capacity]` |
| [retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | Same for `FullScanKNN` |
| [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py), [exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | Capacity-aware register_index, `upsert` (no delete) |
| `retrieve/tests/correctness/test_live_update.py` (new) | Rebuild-vs-incremental parity, delete tombstone, reuse, capacity-exhausted, no-live-update regression |
| [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py), [test_filters.py](../../retrieve/tests/correctness/test_filters.py), [test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py) | Add `capacity=` smoke cases; existing default-path coverage stays green |

## Reused existing utilities

- **[quantize_oporp_1bit](../../retrieve/src/retrieve/layers/utils/quantize.py#L59)** + **[_pack_signs_to_int64](../../retrieve/src/retrieve/layers/utils/quantize.py#L44)** — `OneBitKNN.upsert` re-uses the pack-signs routine over `[K, D]` with the existing per-module `oporp_signs` / `oporp_perm`.
- **[_build_signatures](../../retrieve/src/retrieve/layers/filters/bloom.py#L130)** — `BloomFilter.upsert` re-uses it for per-row signature recompute (already chunked + shape-generic).
- **[compact_mask](../../retrieve/src/retrieve/layers/utils/compact.py)** — `LiveIndexMixin.compact` host helper uses it to derive the dense old→new permutation from `valid_mask`. Also already used by `OneBitKNNTriton.forward` for the masked path — `_effective_mask` produces an input shape that works with it.
- **[post_filter_topk](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9)** — `FullScanKNN` already uses it; we just feed it the AND-folded effective mask.

No new factories needed — current modules are instantiated directly via constructor + `register_index`. (No `build_*` factory exists for these layers today; the previous plan's "thread `capacity` through factories" step is moot.)

## Verification

End-to-end correctness:

```bash
# 1. Functional: rebuild-vs-incremental parity for each retrieval module + filter.
cd retrieve && uv run pytest tests/correctness/test_live_update.py -v

# 2. No regressions in eager paths: full suite green with capacity=None default.
cd retrieve && uv run pytest tests/ -v

# 3. Recall@k unchanged for the registered algorithms (no upsert traffic):
cd evaluation && uv run evaluate --config conf/<yaml> \
    --algorithms linr_v3_then_v2 silvertorch torch_fullscan triton_knn

# 4. Manual smoke (Python REPL):
#   idx = OneBitKNNTriton(k=K); idx.register_index(item_embs, capacity=2*N)
#   ids0, _ = idx(query)
#   new_rows = idx.next_free_rows(100)
#   idx.upsert(new_rows, fresh_embs)
#   idx.delete(rows_to_remove)
#   ids1, _ = idx(query)
#   ids1 should differ from ids0 only in upserted/deleted/displaced rows.

# 5. (Forward-looking) the upsert/delete bodies stay export-clean:
grep -n "\.item()" retrieve/src/retrieve/layers/utils/live_update.py \
    retrieve/src/retrieve/layers/linr/one_bit_knn*.py \
    retrieve/src/retrieve/layers/linr/prefilter_knn*.py \
    retrieve/src/retrieve/layers/linr/similarity_masking*.py
# Only host-helper (next_free_rows / compact) hits permitted; upsert / delete must be clean.
```

## What's explicitly NOT in this plan

- `evaluation/retrieval/build_export.py`, `_upsert.pt2`, `_delete.pt2` — no export entry exists today; revisit after the torch-export refactor lands.
- `SilverTorch` IVF live updates — separate plan.
- Mode-flag refactor of `forward` signatures (`Optional[Tensor]` → required + `mode=`) — that's the export refactor's job, not this one.
- Removing the existing `int(counts.max().item()) == 0` short-circuit at [one_bit_knn_triton.py:53](../../retrieve/src/retrieve/layers/linr/one_bit_knn_triton.py#L53) — already in eager; not regressed by this plan; export refactor will address.
- A GPU-side ID→row hash table — application-layer concern; module APIs take row indices.
