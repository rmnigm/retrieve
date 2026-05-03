# Live-update (Upsert / Delete) API — LiNR-style, post-torch.export

## Context

The LiNR paper at [articles/linr.md](../../articles/linr.md) describes a model-based GPU retriever whose key freshness mechanism is a pair of `Upsert` / `Delete` APIs **on the model itself**. The Updator subscribes to a Venice CDC stream, classifies events as upserts or deletes, and calls those methods concurrently with `forward`. Implementation tricks the paper highlights: pre-allocated larger tensors, a high-water mark for the working set, and minimal serialization between updaters and the inference path. Enabling live updates was worth **+6%** in their A/B (deployment-lessons section).

Our modules today do **not** support this. Every retrieval / filter module re-allocates its buffers from scratch in `register_index`. Buffer sizes are exact (`[N, D]`, `[N, C, A_max]`, etc.), there is no over-allocation, no active-row count, and no tombstone/valid mask. To get live updates, we need:

1. Capacity-aware buffer allocation (`capacity ≥ N`).
2. A scalar `n_active` watermark + a `[capacity] bool valid_mask` per retrieval module.
3. `upsert(rows, embs, ...)` and `delete(rows)` methods that mutate buffers in-place.
4. Forward paths that respect `valid_mask`.

The paper points at this exact stack: the article's "Inference on Native Stack" section says they started on TorchScript and are moving to `torch.export`. Their model has multiple traceable methods (`forward`, `upsert`, `delete`) — under torch.export, each becomes its own `.pt2`. That maps directly onto the convention established by [torch-export-refactor.md](torch-export-refactor.md): **one signature → one trace → one `.pt2`**. So this work is built on top of the export refactor, not in tension with it.

This plan assumes **all 6 phases of the torch-export refactor have landed**.

## Is it hard to combine with torch.export?

**Medium difficulty, mostly already paid for by the export refactor.** The disciplines the export refactor enforces — no `.item()` on the export path, no `Optional[Tensor]` in launch / forward signatures, mode-pinned methods, post-launch host code in the layer, build-factory device routing — apply identically to `upsert` / `delete`. Once those are in place, adding the live-update methods is mostly:

- A `capacity` kwarg on `register_index` (off the export path).
- Two new buffers per retrieval module: `n_active: Tensor[()] int64` and `valid_mask: Tensor[capacity] bool`.
- One in-place AND of `valid_mask` into every forward's existing mask path.
- Method bodies that take row-index inputs and mutate buffers via `index_copy_` / `scatter_` (both export-traceable).

Genuinely new constraints (none individually hard):

- **`n_active` is a scalar tensor, not a Python int.** Updating it as `n_active.fill_(torch.maximum(n_active, rows.max() + 1))` keeps it on-device, no `.item()`.
- **`valid_mask` threads into forward.** Today only `LiNR_V2` / `V3` / `FullScanKNN` accept an external `mask`; they need to AND it with `self.valid_mask[None, :]` before topk. That's one additional line per `_forward_full`.
- **`LiNR_V3`'s OPORP must quantize new rows on-device** at upsert time using the existing `oporp_signs` / `oporp_perm` buffers. Same `quantize_oporp_1bit` op as `register_index`, just on a `[K, D]` slice.
- **`BloomFilter` recomputes signatures per upsert row** using existing `hash_seeds`. Per-row independent — just a smaller call to `_build_signatures`.

Genuinely deferred (out of scope):

- **`SilverTorch` (IVF) live update.** Per the article, this is where most of LiNR's engineering effort went: per-cluster overcommit, route-by-centroid on upsert, periodic cluster rebalancing. Substantial enough to be its own plan; deferring.
- **GPU-side ID→row hash table.** The paper's Updator runs in the host (Apache Beam → Updator → GPU); the application layer owns the ID map. Module APIs take row indices.
- **Concurrent updates from multiple streams.** We promise correctness only when forward and upsert/delete are issued on the same CUDA stream (or serialized via a stream sync). The article's "minimal data access serialization" matches this; the inside of the module does no locking.

## Approach

### Phase A — Shared scaffolding (interfaces, base buffers)

**Files:**
- [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py)
- `retrieve/src/retrieve/layers/utils/live_update.py` — new
- [docs/system/architecture.md](../system/architecture.md)

**Steps:**

1. Add a small `LiveIndexMixin` in a new `live_update.py`:

   ```python
   class LiveIndexMixin:
       """Provides n_active + valid_mask buffers and a free-row helper.
       Modules that mix this in must call _alloc_live_buffers(capacity) inside
       their register_index, after computing the initial item count N."""

       def _alloc_live_buffers(self, capacity: int, n_initial: int) -> None:
           self.register_buffer("n_active", torch.tensor(n_initial, dtype=torch.int64))
           valid = torch.zeros(capacity, dtype=torch.bool)
           valid[:n_initial] = True
           self.register_buffer("valid_mask", valid)
           self._capacity = int(capacity)  # python attr, frozen at export time

       def _bump_watermark(self, rows: Tensor) -> None:
           """In-place watermark update; no .item(). Export-safe."""
           self.n_active.copy_(torch.maximum(self.n_active, rows.max() + 1))
   ```

2. Extend `RetrievalModule` and `FilterModule` (in [interfaces.py](../../retrieve/src/retrieve/interfaces.py)) with optional declarations: `register_index(item_embs, *, capacity: int | None = None)`. Default `capacity=None` means `capacity=N` (no overcommit, identical to today).

3. Add `upsert` and `delete` to `RetrievalModule` as **non-abstract** methods that raise `NotImplementedError`. Subclasses opt in. Filters get the same on `FilterModule` where applicable.

### Phase B — `LiNR_V2` / `LiNR_V2_Triton`, `LiNR_V3` / `LiNR_V3_Triton`, `FullScanKNN`

**Files:**
- [retrieve/src/retrieve/layers/linr/v2.py](../../retrieve/src/retrieve/layers/linr/v2.py), [v2_triton.py](../../retrieve/src/retrieve/layers/linr/v2_triton.py)
- [retrieve/src/retrieve/layers/linr/v3.py](../../retrieve/src/retrieve/layers/linr/v3.py), [v3_triton.py](../../retrieve/src/retrieve/layers/linr/v3_triton.py)
- [retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py)
- [retrieve/src/retrieve/layers/linr/builder.py](../../retrieve/src/retrieve/layers/linr/builder.py)
- [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py), `test_full_scan_knn.py` *(or wherever FullScan is tested)*

**Steps:**

1. **`register_index` capacity-aware.** Each module's `register_index(item_embs, *, capacity=None)` allocates `item_embs` at `[capacity, D]`, copies the initial `[:N]` rows, leaves the rest zero, then calls `_alloc_live_buffers(capacity, N)`. For `LiNR_V3`, `item_bits` is `[capacity, W]` — quantize the initial slice, leave the rest zero (they're masked by `valid_mask` anyway).

2. **`upsert(rows: Tensor[K], embs: Tensor[K, D]) -> None`** (eager + traceable):
   - **V2 / FullScan**: `self.item_embs.index_copy_(0, rows, embs)`, `self.valid_mask.index_fill_(0, rows, True)`, `self._bump_watermark(rows)`.
   - **V3**: `bits = quantize_oporp_1bit(embs, self.oporp_signs, self.oporp_perm)`, then `self.item_bits.index_copy_(0, rows, bits)` + same valid_mask + watermark.
   - The Triton subclasses inherit `upsert` from their torch base — no kernel work; the operation is pure tensor mutation.

3. **`delete(rows: Tensor[K]) -> None`**: `self.valid_mask.index_fill_(0, rows, False)`. We do **not** clear the embedding row — it's masked, and reclaim happens on the next `upsert` to that row. This avoids a redundant write and keeps `delete` a single op.

4. **Forward integration.** Every `_forward_full` / `_forward_filtered` AND-folds `valid_mask` into the score path:
   - V2 (`_forward_full`): apply `mask & self.valid_mask[None, :]`. The post-export-refactor signature already requires `mask` (Phase 3 of the export refactor), so the AND becomes unconditional.
   - V3 (`_forward_full` and `_forward_masked`): same pattern. The kernel-side mask path (`oporp_1bit_match_topk`) already accepts a mask tensor.
   - V2 / V3 candidates path: gather `valid_mask[candidate_ids]` and zero-out tombstoned candidate scores before topk.
   - FullScanKNN: same as V2.

5. **Builder updates.** [builder.py](../../retrieve/src/retrieve/layers/linr/builder.py) `build_linr_v*` factories pass `capacity` through. Default behavior unchanged when `capacity` is omitted.

### Phase C — `ClauseIndex`, `BloomFilter`

**Files:**
- [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py)
- [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py)
- [retrieve/tests/correctness/test_filters.py](../../retrieve/tests/correctness/test_filters.py), [test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py)

**Steps:**

1. Filters do **not** carry their own `valid_mask` — they trust the retrieval module's mask. They only need to keep their per-row state (`item_clause_attrs`, `bloom_sigs`) consistent with the retrieval module's row indexing.

2. **`upsert(rows: Tensor[K], item_clause_attrs: Tensor[K, C, A_max]) -> None`**:
   - `ClauseIndexTorch` / `ClauseIndexTriton`: `self.item_clause_attrs.index_copy_(0, rows, item_clause_attrs)`. `A_max` is fixed at `register_index` time; caller pads / truncates to it.
   - `BloomFilterTorch` / `BloomFilterTriton`: recompute signatures for the K rows on-device via the same `_build_signatures` op already used at register time, then `self.bloom_sigs.index_copy_(0, rows, new_sigs)`. Per-row independent — no global rebuild.

3. **`delete(rows)`**: no-op for filters. Documented in the docstring; the retrieval-side `valid_mask` is the source of truth for liveness.

4. Both filters expose `register_index(item_clause_attrs, *, capacity=None)`.

### Phase D — Host-only helpers (not exported)

**Files:**
- `retrieve/src/retrieve/layers/utils/live_update.py` (same file as Phase A)

**Steps:**

1. **`next_free_rows(self, n_new: int) -> Tensor[n_new]`** — Python-side helper on `LiveIndexMixin`. Walks `valid_mask` (CPU copy) to find tombstoned slots first, then extends past `n_active` up to `_capacity`. Raises if `_capacity` exhausted. Caller uses the returned rows as input to `upsert`. **Not on the export path** — it's the application's row-allocator and uses `.item()` / `.tolist()` freely.

2. **`compact(self) -> Tensor[capacity]`** — optional, host-side. Rebuilds buffers to put live rows in `[0:n_active)` order; returns an old→new row map so the caller can rewrite their ID→row table. Implemented per-module (each knows its own buffers) but follows a shared interface.

### Phase E — Export entries (`build_export.py` extension)

**Files:**
- `evaluation/retrieval/build_export.py` (created in Phase 3 of the export refactor; extended by Phases 5–6)

**Steps:**

For each retrieval algo with live-update support, add two entries alongside its existing forward `.pt2`:

```python
def export_linr_v3_upsert(idx, out_dir):
    rows_ex = torch.zeros(8, dtype=torch.int64)
    embs_ex = torch.zeros(8, D, dtype=torch.float32)
    ep = torch.export.export(
        idx.upsert,
        (rows_ex, embs_ex),
        dynamic_shapes=({0: Dim("k", min=1, max=idx._capacity)}, {0: Dim("k", ...)}),
    )
    torch.export.save(ep, out_dir / "linr_v3_upsert.pt2")

def export_linr_v3_delete(idx, out_dir):
    rows_ex = torch.zeros(8, dtype=torch.int64)
    ep = torch.export.export(idx.delete, (rows_ex,), dynamic_shapes=({0: Dim("k", ...)},))
    torch.export.save(ep, out_dir / "linr_v3_delete.pt2")
```

Final `.pt2` zoo (concatenating with what the export refactor produces):

- `linr_v2_full.pt2`, `linr_v2_candidates.pt2`, **`linr_v2_upsert.pt2`**, **`linr_v2_delete.pt2`**
- `linr_v3_full.pt2`, `linr_v3_masked.pt2`, `linr_v3_candidates.pt2`, **`linr_v3_upsert.pt2`**, **`linr_v3_delete.pt2`**
- `full_scan_full.pt2`, **`full_scan_upsert.pt2`**, **`full_scan_delete.pt2`** (only if FullScanKNN gets exported per the export refactor's "out of scope" note)
- `clause_filter_upsert.pt2`, `bloom_filter_upsert.pt2` (delete is a no-op for filters; no entry)

The native serving stack loads all entries for an algo as parallel AOTI artifacts and dispatches per the article's classification of CDC events into upserts vs deletes.

### Phase F — Documentation + concurrency contract

**Files:**
- [docs/system/architecture.md](../system/architecture.md)
- [docs/system/checkpoints.md](../system/checkpoints.md) (capacity restoration on load)
- New: `docs/system/live-updates.md` — short doc on the contract.

**Document:**

- The `capacity` parameter and how to pick it (article: "pre-allocating larger tensors").
- The application layer owns the ID→row map; module APIs take rows.
- Forward + upsert + delete must share a CUDA stream OR be serialized via stream sync. Inside the module there is no locking. (Matches the article's "minimal data access serialization.")
- `delete` is a tombstone, not a row free; reuse happens at next `upsert(row, ...)` to that row.
- SilverTorch (IVF) does **not** support live updates yet; deferred to a follow-up plan.

## Critical files to modify

| File | Reason |
|---|---|
| [retrieve/src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) | Add `capacity=` to `register_index`; non-abstract `upsert` / `delete` declarations |
| `retrieve/src/retrieve/layers/utils/live_update.py` (new) | `LiveIndexMixin` with `n_active` / `valid_mask` / `_bump_watermark` / `next_free_rows` / `compact` |
| [retrieve/src/retrieve/layers/linr/v2.py](../../retrieve/src/retrieve/layers/linr/v2.py), [v3.py](../../retrieve/src/retrieve/layers/linr/v3.py), [v2_triton.py](../../retrieve/src/retrieve/layers/linr/v2_triton.py), [v3_triton.py](../../retrieve/src/retrieve/layers/linr/v3_triton.py) | Capacity-aware register_index, upsert / delete impls, valid_mask in forward |
| [retrieve/src/retrieve/layers/utils/retrieval.py](../../retrieve/src/retrieve/layers/utils/retrieval.py) | Same for `FullScanKNN` |
| [retrieve/src/retrieve/layers/filters/clause.py](../../retrieve/src/retrieve/layers/filters/clause.py), [bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | Capacity-aware register_index, upsert (no delete) on filters |
| [retrieve/src/retrieve/layers/linr/builder.py](../../retrieve/src/retrieve/layers/linr/builder.py) | Thread `capacity` through factories |
| `evaluation/retrieval/build_export.py` | Add `_upsert.pt2` and `_delete.pt2` entries per algo |
| [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py), [test_filters.py](../../retrieve/tests/correctness/test_filters.py), [test_bloom_filter.py](../../retrieve/tests/correctness/test_bloom_filter.py) + new `test_live_update.py` | Coverage |

## Reused existing utilities

- **`quantize_oporp_1bit`** at [retrieve/src/retrieve/layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py) — reused inside `LiNR_V3.upsert` to quantize new rows with the existing per-module `oporp_signs` / `oporp_perm`.
- **`_build_signatures`** at [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) — reused inside `BloomFilter.upsert` for per-row signature recompute.
- **`compact_mask`** at [retrieve/src/retrieve/layers/utils/compact.py](../../retrieve/src/retrieve/layers/utils/compact.py) — reused inside `compact()` host helper to rebuild dense indices.
- **`build_*` factories** in each layer file — extended in place; same signature shape established by the export refactor's Phase 1 conventions.

## Verification

End-to-end correctness:

```bash
# 1. Functional: rebuild-vs-incremental parity. Insert N items via N upsert calls
#    and assert forward output matches a single register_index of the same N items.
cd retrieve && uv run pytest tests/correctness/test_live_update.py -v

# 2. Eager regressions: full suite still green.
cd retrieve && uv run pytest tests/ -v

# 3. Recall@k unchanged for the registered algorithms (no upsert traffic):
cd evaluation && uv run python -m retrieval.benchmark --config conf/<yaml> \
    --algorithms linr_v3_then_v2 silvertorch torch_fullscan triton_knn

# 4. Export round-trip per (algo, op):
cd evaluation && uv run python -m retrieval.build_export \
    --algo linr_v3 --mode upsert --checkpoint-dir <path> --out-dir <path>
cd evaluation && uv run python -m retrieval.build_export \
    --algo linr_v3 --mode delete --checkpoint-dir <path> --out-dir <path>
# Then load each .pt2, run a synthetic upsert+delete+forward sequence,
# and assert the live-updated forward result matches an eager-mode reference.

# 5. No-.item() / no-Optional sweeps stay clean (continues the export refactor's bar):
grep -rn "\.item()" retrieve/src/retrieve/layers/  # only register_index / host-helper hits
grep -rn "Optional\[Tensor\]\|Tensor | None" retrieve/src/retrieve/layers/  # zero on upsert / delete
```

End-to-end live-update demo (manual smoke):

```python
idx = build_linr_v3_triton(item_embs, k=K, capacity=2 * len(item_embs))
ids0, _ = idx(query)                          # baseline
new_rows = idx.next_free_rows(100)            # host-side allocator
idx.upsert(new_rows, fresh_embs)              # inserts
idx.delete(rows_to_remove)                    # tombstones
ids1, _ = idx(query)                          # reflects the live update
```

The live-updated `ids1` should differ from `ids0` only in rows whose embeddings were upserted, deleted, or pushed off the top-K.
