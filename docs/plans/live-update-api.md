# Live-update (upsert / delete) API — refresh for V1, V2, and the surrounding LiNR family

## Context

The LiNR paper ([articles/linr.md §4.3](../../articles/linr.md), lines 177-181) describes a live-update path: Venice CDC stream → `Live Update Ingestor` → classify upserts/deletes → in-place GPU buffer mutation. Techniques are **pre-allocating larger tensors**, a **high-water mark** to track the working set, and **minimal data-access serialization**. The §5.6.1 ablation (line 317) reports **+6% production lift** from enabling live updates — the value driver is freshness for newly created items, not export packaging. A separate benchmark (line 313) shows no measurable inference-latency impact at production update rates.

Scope:
- Each LiNR module (`PostfilterKNN`, `PrefilterKNN`, `OneBitKNN`, `FullScanKNN`) is one class per file dispatching on `backend: Backend = "triton"`; `upsert` / `delete` land directly on that class.
- The filter side (`BloomFilter`, `ExactAttributeFilter`) is included so V2 end-to-end can be live-updated.
- Stays export-clean (no `.item()`, no `.cpu()`, no `Optional[Tensor]` in upsert/delete bodies) so the in-flight export refactor ([torch-export-refactor.md](torch-export-refactor.md)) composes cleanly.
- Defers `.pt2` export entries (no `build_export.py` exists today; see torch-export-refactor.md for the in-flight scaffold).

> **SilverTorch (2026-05-23):** silvertorch is back in scope at the roadmap level, but live updates are a separate follow-up — its IVF clustering means upsert requires cluster reassignment / rebalancing, not just a row-slice `index_copy_`. The original deferral notes survive in [../plans-silvertorch-backup/live-update-api.md](../plans-silvertorch-backup/live-update-api.md); reuse the `LiveIndexMixin` shape established here when that lift is scheduled.

## Approach

### Design invariants

- **Single source of truth for liveness** = retrieval module's `valid_mask: [capacity] bool`. Filters do **not** maintain their own valid_mask; they just over-allocate the row dim to match.
- `register_index(item_embs, capacity: int | None = None)`. `capacity is None` ⇒ byte-for-byte identical to today (legacy path, no overcommit, no live updates, no mixin attrs set on the module).
- `upsert(rows: Tensor[K], embs: Tensor[K, D])` — eager, in-place. Mutates buffers at `rows`, sets `valid_mask[rows]=True`, bumps watermark.
- `delete(rows: Tensor[K])` — tombstone-only: `valid_mask[rows]=False`. Rows stay; reuse on next upsert to that row.
- `n_active: Tensor` (int64 scalar) — soft, monotonic-increasing watermark; bounds compaction-time scans, not forward correctness.
- **All paths are CUDA-only.** No `.cpu()`, no `.item()`, no Python loops over tensor entries in any new code.
- Concurrency: forward + upsert + delete must share a CUDA stream OR be serialized externally; **no internal locking** (matches §4.3 line 181 "minimal data access serialization").

### Phase A — Shared scaffolding

**New:** [retrieve/src/retrieve/layers/utils/live_update.py](../../retrieve/src/retrieve/layers/utils/live_update.py)

`LiveIndexMixin` (pure mixin, no `nn.Module` base — concrete classes already subclass `RetrievalModule(nn.Module)`):

- `_alloc_live_buffers(capacity, n_initial, device)` → registers `n_active` scalar buffer, `valid_mask: [capacity] bool` (initial `[:n_initial]=True`), sets `_capacity: int` and `_live_enabled = True`.
- `_bump_watermark(rows)` → `self.n_active.copy_(torch.maximum(self.n_active, rows.max().to(int64) + 1))`. Stays on-device; no `.item()`.
- `_effective_mask(B, mask)` → folds `valid_mask` into the caller's mask. **Returns the caller's mask unchanged when `_live_enabled` is False** — this is what guarantees `capacity=None` is bit-identical to today.
- `next_free_rows(n_new) -> Tensor` → fully on-device. `free = (~self.valid_mask).nonzero(as_tuple=False).flatten()`. Naturally returns tombstoned slots (below the watermark) followed by extension slots (above it) in row-index order. `if free.numel() < n_new: raise RuntimeError("capacity exhausted: ...")` — `numel()` is tensor metadata (a Python int), no device sync. Return `free[:n_new]`.

**Edit:** [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py)

- Add `capacity: int | None = None` to abstract `RetrievalModule.register_index` and `FilterModule.register_index`.
- Add non-abstract defaults on `RetrievalModule` for `upsert(rows, embs)` and `delete(rows)` that raise `NotImplementedError(f"{type(self).__name__} does not support upsert/delete")`. Subclasses opt in by overriding.
- Add non-abstract `FilterModule.upsert(rows, item_clause_attrs)` default (no `delete` on `FilterModule` — no per-row liveness).
- Do **not** add `mode=` or any other signature reshuffle. That belongs to the parallel torch-export refactor.

### Phase B — Retrieval modules (one class each, in-place edits)

Every module follows the same skeleton:

```python
def register_index(self, item_embs, capacity=None):
    if capacity is None:
        # body identical to today; no _alloc_live_buffers call
        return
    # over-allocate, copy [:N], call self._alloc_live_buffers(...)

def upsert(self, rows, embs):
    self.<buffer>.index_copy_(<dim>, rows, <encoded embs>)
    self.valid_mask.index_fill_(0, rows, True)
    self._bump_watermark(rows)

def delete(self, rows):
    self.valid_mask.index_fill_(0, rows, False)
```

#### B1 — `PostfilterKNN` (V1) — [postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py)

**Per-module quirk:** buffer is **pre-transposed** `item_embs_t: [D, N]` (line 35). Upsert is a **column-slice**: `self.item_embs_t.index_copy_(1, rows, embs.t().contiguous())`. The pre-transpose is deliberate (lines 31-34 comment about cuBLAS operand alignment) — preserve it through overcommit by allocating `[D, capacity]` and indexing along dim 1.

Forward ([line 37](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L37)): swap `if mask is not None: scores = scores.masked_fill(~mask, -inf)` for `effective = self._effective_mask(query.shape[0], mask); if effective is not None: scores = scores.masked_fill(~effective, -inf)`. The existing isfinite → `-1` wrap (lines 46-51) already handles tombstoned scores (`-inf`) correctly.

#### B2 — `PrefilterKNN` (V2) — [prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)

**Per-module quirk:** forward has **no `mask=` parameter** (line 40 signature is `(query, candidate_ids, counts)`). Fold `valid_mask` inside each sub-path; do not widen the public signature.

Buffer is `item_embs: [N, D]`. `upsert` is the straight row-slice `index_copy_(0, rows, embs)`.

Forward sub-paths:

- **`_forward_full`** ([line 52](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L52)) — full matmul + topk. When `_live_enabled`, apply `masked_fill(~valid_mask, -inf)` after the matmul and before topk; wrap top-K ids with the same isfinite → `-1` pattern V1 uses.

- **`_forward_prefilter`** ([line 57](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L57), torch backend) — gather `valid_in_cand = self.valid_mask[safe_ids]` (`[B, P]`) and AND it into the existing `counts`-derived validity mask (lines 76-78) before topk.

- **`_forward_prefilter_triton`** ([line 96](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L96)) — the `fused_masked_knn_topk` kernel (line 111) requires the first `counts[i]` entries to be valid. Strategy: gather `valid_in_cand`, AND with the arange-from-counts mask, **stable-sort per row** so live entries pack contiguously at the front, gather `candidate_ids` along that permutation, set `counts = valid_in_cand.sum(dim=1).long()`. Pass to the kernel unchanged. Cost: one per-row stable sort over P — small vs the kernel cost.

#### B3 — `OneBitKNN` (V3 stage 1) — [one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py)

**Per-module quirk:** `oporp_signs` and `oporp_perm` are **frozen at register_index time** — exactly what upsert needs. Forward signature is `(query, candidate_ids=None, counts=None)` — there is no `mask` parameter; the masked path was retired with Stage 2b, so `valid_mask` folds into the existing sub-paths.

`upsert` reuses the same op chain `quantize_oporp_1bit` runs initially ([quantize.py:81](../../retrieve/src/retrieve/layers/utils/quantize.py#L81)) but over `[K, D]`:

```python
proj = (embs * self.oporp_signs.to(embs.dtype)).index_select(1, self.oporp_perm)
bits = _pack_signs_to_int64(proj)       # quantize.py:66
self.item_bits.index_copy_(0, rows, bits)
```

Forward ([line 93](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L93)) has four sub-paths — torch full / torch candidates / triton full / triton candidates. Fold `valid_mask` into each:

- `_forward_torch_eager` full-scan sub-path (`candidate_ids is None`): apply `effective = self._effective_mask(B, None); if effective is not None: scores = scores.masked_fill(~effective, -inf)` before topk.
- `_forward_torch_eager` candidate sub-path (`candidate_ids is not None`): gather `valid_in_cand = self.valid_mask[candidate_ids.clamp_min(0)]`, AND with the existing counts-derived validity mask, then `masked_fill(~valid, -inf)` before topk. The existing isfinite → `-1` wrap handles tombstones automatically.
- `_forward_triton` full-scan sub-path (`candidate_ids is None` — calls `oporp_1bit_match_topk_full`): when `_live_enabled`, synthesize `candidate_ids = arange(N).expand(B, N)` + `counts = valid_mask.sum() * ones(B)` and route through `oporp_1bit_match_topk_indirect` so the kernel respects the live set. Cost: one `arange` + a sum per call; acceptable for an opt-in path.
- `_forward_triton` candidate sub-path: same stable-sort permutation strategy as V2 triton path. Gather `valid_in_cand`, AND with the arange-from-counts validity, stable-sort to pack live entries at the front of each row, gather `candidate_ids` along that permutation, recompute `counts = valid.sum(dim=1).long()`, then dispatch to the kernel unchanged.

Docstring note: the synthetic-candidates full-scan route is slower than the contiguous-load full path; live-update is opt-in via `capacity=`, so the cost is paid only when freshness is enabled.

#### B4 — `FullScanKNN` — [utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py)

**Per-module quirk:** mask is **post-topk** via `post_filter_topk` ([line 9](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9)), distinct from V1/V3. For tombstones to never appear in the K slots (not just blank to `-1` after the fact), apply `valid_mask` **pre-topk** in `forward` ([line 36](../../retrieve/src/retrieve/layers/utils/retrieval.py#L36)):

```python
if getattr(self, "_live_enabled", False):
    vm = self.valid_mask.unsqueeze(0).expand(query.shape[0], -1)
    scores = scores.masked_fill(~vm, float("-inf"))
topk_scores, topk_ids = torch.topk(scores, self.k, dim=1)
if mask is not None:
    topk_ids, _ = post_filter_topk(topk_ids, mask)
```

Caller's `mask` keeps its post-topk → `-1` semantics. Live-mask is separate (pre-topk, never appears). `_forward_candidates` ([line 50](../../retrieve/src/retrieve/layers/utils/retrieval.py#L50)): gather `valid_mask[candidate_ids]`, `masked_fill` before the local topk.

### Phase C — Filters

#### C1 — `ExactAttributeFilter` — [exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py)

Buffer `item_clause_attrs: [N, C, A_max]` int64 ([line 28](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L28)). `clause_is_reverse: [C]` ([line 29](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L29)) is per-clause, not per-row — untouched.

Capacity-aware register_index over-allocates as `[capacity, C, A_max]` filled with `-1` (padding sentinel — treated as "no match" by the `clause_pass.any(dim=-1)` semantics at line 64). No `valid_mask` allocated; the filter does not call `_alloc_live_buffers`.

`upsert(rows, item_clause_attrs)` → `self.item_clause_attrs.index_copy_(0, rows, item_clause_attrs)`. Caller pads/truncates to `A_max`. No delete.

#### C2 — `BloomFilter` — [bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)

Buffer `bloom_sigs: [N, W]` int64 ([line 29](../../retrieve/src/retrieve/layers/filters/bloom.py#L29)). `hash_seeds` ([line 30](../../retrieve/src/retrieve/layers/filters/bloom.py#L30)) frozen at register_index.

Capacity-aware register_index over-allocates as `[capacity, W]` filled with zero (all-zero sig fails the `(qb & sigs) == qb` test at line 80 for any non-zero query).

`upsert(rows, item_clause_attrs)` reuses [`_build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom.py#L134) on the `[K, C, A_max]` slice — the function is **shape-generic in leading dims** (line 141) and chunked by 131k rows (line 131), so it works on any K without rewriting. Then `self.bloom_sigs.index_copy_(0, rows, new_sigs)`. No delete.

### Phase D — Docs

- New: `docs/system/live-updates.md` — concurrency contract, capacity sizing guidance, tombstone semantics, what does and does not support live updates.
- Edit: `docs/system/architecture.md` (mention the `capacity=` knob and the opt-in mask-path cost).
- Edit: `docs/system/checkpoints.md` (capacity is implicit in buffer shapes; checkpoint round-trip must preserve it).

### Out of scope

- **SilverTorch IVF live updates** — separate follow-up; IVF clustering means upsert is a cluster-reassignment problem, not a row-slice `index_copy_`. Reuse `LiveIndexMixin` when scheduled.
- **`torch.export` / `.pt2` packaging.** No `build_export.py` exists today. The upsert/delete bodies stay export-clean (no `.item()`, no `.cpu()`, no `Optional[Tensor]`) so the in-flight export refactor ([torch-export-refactor.md](torch-export-refactor.md)) composes cleanly.
- **GPU-side ID→row hash table.** Application-layer concern; module APIs take row indices.

## Critical files to modify

| File | Touch |
|---|---|
| [retrieve/src/retrieve/layers/utils/live_update.py](../../retrieve/src/retrieve/layers/utils/live_update.py) (new) | `LiveIndexMixin`: `_alloc_live_buffers`, `_bump_watermark`, `_effective_mask`, `next_free_rows`. All on-device — no `.cpu()`, no `.item()`, no Python loops. |
| [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) | `capacity=None` on abstract `register_index` for both `RetrievalModule` and `FilterModule`; non-abstract `upsert` / `delete` defaults; `FilterModule.upsert` default (no `delete`). |
| [retrieve/src/retrieve/layers/linr/postfilter_knn.py](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) | Mix in `LiveIndexMixin`. Capacity-aware register_index allocating `[D, capacity]`. `upsert` does **column-slice** `index_copy_(1, rows, embs.t().contiguous())`. Forward folds `_effective_mask`. |
| [retrieve/src/retrieve/layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py) | Mix in `LiveIndexMixin`. Capacity-aware register_index. `upsert` row-slice. Forward has **no `mask=` arg**: fold `valid_mask` inside `_forward_full` (masked_fill pre-topk), `_forward_prefilter` (`valid_mask[safe_ids]` gather), `_forward_prefilter_triton` (per-row stable-sort permutation of `candidate_ids` + recompute `counts`). |
| [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) | Mix in `LiveIndexMixin`. Capacity-aware register_index (signs/perm frozen, item_bits over-allocated). `upsert` reuses `_pack_signs_to_int64` over `[K, D]`. Forward folds `valid_mask` across all four sub-paths (torch full/cand, triton full/cand); triton full-scan synthesizes `(candidate_ids, counts)` and routes through `_indirect` when live. |
| [retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | `FullScanKNN` mix in `LiveIndexMixin`. Capacity-aware register_index. `upsert`/`delete`. Forward applies `valid_mask` **pre-topk** (so tombstones never appear); caller's mask stays post-topk via existing `post_filter_topk`. |
| [retrieve/src/retrieve/layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | Capacity-aware register_index (`-1`-padded over-alloc). `upsert(rows, item_clause_attrs)` = `index_copy_(0, ...)`. No `valid_mask`, no `delete`. |
| [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | Capacity-aware register_index (zero-padded over-alloc). `upsert` reuses [`_build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom.py#L134) on the `[K, C, A_max]` slice + `index_copy_`. No delete. |
| `retrieve/tests/correctness/test_live_update.py` (new) | Test plan below. |
| `docs/system/live-updates.md` (new), `architecture.md`, `checkpoints.md` (edits) | Concurrency contract; capacity sizing; what supports live updates. |

## Reused existing utilities (no new factories)

- [`_pack_signs_to_int64`](../../retrieve/src/retrieve/layers/utils/quantize.py#L44) — `OneBitKNN.upsert` reuses it over `[K, D]` with the stored `oporp_signs` / `oporp_perm`.
- [`_build_signatures`](../../retrieve/src/retrieve/layers/filters/bloom.py#L134) — `BloomFilter.upsert` re-calls it on a K-row slice. Already chunked & shape-generic.
- [`post_filter_topk`](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9) — `FullScanKNN` keeps using it for caller-mask post-filtering; live-mask is pre-topk and separate.

## Verification

New test file: `retrieve/tests/correctness/test_live_update.py`. Uses the existing [conftest helpers](../../retrieve/tests/conftest.py) (`make_index`, `make_query`, `make_attrs`) and shape conventions from [test_linr.py](../../retrieve/tests/correctness/test_linr.py) (N=2048, D=128, B=16, K=200).

Parametrize over `(PostfilterKNN, "torch"|"triton"), (PrefilterKNN, "torch"|"triton"), (OneBitKNN, "torch"|"triton"), (FullScanKNN, None)`. Test cases:

1. **Rebuild-vs-incremental parity** — `register_index(embs[:N], capacity=2N)` then upsert-the-rest equals `register_index(embs)`. Bit-exact for V1/V2 full path; id-set equality for V3 (Hamming ties).
2. **Delete tombstones excluded** — delete rows that DO appear in the baseline top-K, assert they vanish.
3. **Reuse after delete** — `delete([r]) → upsert([r], near_query)` → r appears in top-K with the new embedding's neighbors.
4. **Capacity exhausted raises** — `next_free_rows(n_new > available)` → `RuntimeError("capacity exhausted")`.
5. **Legacy path byte-identical** (regression guard) — `register_index(embs)` with no capacity produces identical forward output to today AND `not hasattr(m, "valid_mask")`.
6. **V2 end-to-end with filter** — `PrefilterKNN` + `ExactAttributeFilter` both with `capacity=2N`; compute candidate_ids; delete some rows from `PrefilterKNN`; re-evaluate; deleted rows absent. Exercises the V2 triton stable-sort path.
7. **Filter upsert parity** — `BloomFilter` / `ExactAttributeFilter`: incremental upsert produces the same `evaluate_mask` output as a single-shot `register_index` over the equivalent attrs.
8. **Filter legacy path** — `capacity=None` produces today's `bloom_sigs` / `item_clause_attrs` shape and dtype; `upsert` raises `NotImplementedError` (interface default).
9. **Watermark monotonic** — `delete` does not lower `n_active`; `upsert(rows=[r])` raises it to `max(r)+1`.
10. **V3 cascade** — `OneBitKNN` → `PrefilterKNN` (mirrors [linr_v3.py:50-52](../../evaluation/retrieval/algos/linr_v3.py#L50)). Both stages capacity-aware; delete on both with same rows; deleted absent from final ids.

Commands:

```bash
# 1. New live-update suite.
cd /workspace/retrieve/retrieve && uv run pytest tests/correctness/test_live_update.py -v

# 2. Regression — full correctness suite must stay green; capacity=None default
#    means no observable change.
cd /workspace/retrieve/retrieve && uv run pytest tests/ -v

# 3. Export-cleanliness check on the new code paths.
grep -nE "\.item\(\)|\.cpu\(\)|Optional\[Tensor\]" \
    /workspace/retrieve/retrieve/src/retrieve/layers/utils/live_update.py \
    /workspace/retrieve/retrieve/src/retrieve/layers/linr/{one_bit_knn,prefilter_knn,postfilter_knn}.py \
    /workspace/retrieve/retrieve/src/retrieve/layers/utils/retrieval.py \
    /workspace/retrieve/retrieve/src/retrieve/layers/filters/{bloom,exact_attribute}.py
# Expected: zero hits in the new code. The pre-existing one_bit_knn.py:156
# .item() in the masked-kernel short-circuit is orthogonal to this plan.

# 4. End-to-end recall regression (no upsert traffic).
cd /workspace/retrieve/evaluation && uv run evaluate --config conf/<yaml> \
    --algorithms linr_v3_then_v2 torch_fullscan triton_knn

# 5. Manual smoke:
#   m = OneBitKNN(k=200, backend="triton")
#   m.register_index(item_embs, capacity=2 * N)
#   ids0, _ = m(query)
#   rows = m.next_free_rows(100)
#   m.upsert(rows, fresh_embs)
#   m.delete(rows_to_kill)
#   ids1, _ = m(query)
#   assert set(rows_to_kill.tolist()).isdisjoint(set(ids1.flatten().tolist()))
```
