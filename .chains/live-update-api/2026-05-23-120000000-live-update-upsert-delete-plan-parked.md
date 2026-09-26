---
chain: "live-update-api"
branch: "main"
nextStep: "Re-scope against the L layout before executing (roadmap G-e): LiveIndexMixin lands on retrieve.modules; rebind every anchor (bloom builders are indexing.bloom_hash.build_signatures, OneBitKNN is a _PackedBitsKNN subclass)."
created: "2026-05-23T12:00:00Z"
---

# Live-update (upsert / delete) API for the LiNR family (parked plan)

Source: `docs/plans/live-update-api.md`, written 2026-05-23, never started. Banner at archive time: every line reference stale (`_build_signatures` moved to public `build_signatures` in `indexing/bloom_hash.py` at K5; `one_bit_knn.py` became a `_PackedBitsKNN` subclass and its `.item()` short-circuit is gone; commands used `/workspace/...` paths and algo names that no longer exist). Its Phase A dependency is satisfied: `RetrievalModule` exists (K6.2 added it for this plan). Parked as roadmap G-e.

## Motivation
LiNR paper §4.3: CDC stream -> Live Update Ingestor -> in-place GPU buffer mutation, via pre-allocated larger tensors, a high-water mark and minimal serialization. §5.6.1: +6 % production lift from live updates (freshness); no measurable latency impact at production update rates.

## Design invariants
- One liveness source: the retrieval module's `valid_mask: [capacity] bool`; filters over-allocate rows but keep no mask.
- `register_index(item_embs, capacity: int | None = None)`; `capacity=None` byte-identical to today (no mixin attrs set).
- `upsert(rows, embs)` eager, in place, sets `valid_mask`, bumps the watermark; `delete(rows)` tombstone only.
- `n_active` int64 soft monotonic watermark; CUDA-only, no `.cpu()` / `.item()` / Python loops over tensor entries; no internal locking (share a stream or serialize externally).

## Phases
- A: `indexing/live_update.py` `LiveIndexMixin` (`_alloc_live_buffers`, `_bump_watermark`, `_effective_mask` returning the caller's mask unchanged when not live, `next_free_rows` fully on device and raising "capacity exhausted"); `capacity=` on both ABCs; `upsert` / `delete` defaults raising `NotImplementedError`.
- B: `PostfilterKNN` (pre-transposed `[D, capacity]`, column-slice `index_copy_(1, ...)`); `PrefilterKNN` (no mask param: fold `valid_mask` into full / torch-prefilter / triton-prefilter paths; on Triton a per-row stable sort packs live candidates first and recomputes `counts`); `OneBitKNN` (signs / perm frozen; upsert re-projects `[K, D]`; Triton full scan routes through `_indirect` with synthetic candidates when live); `FullScanKNN` (live mask applied pre-topk, the caller's mask stays post-topk).
- C: `ExactAttributeFilter` over-allocated with `-1`, `BloomFilter` with zero signatures (fail any non-zero query); `upsert` only.
- D: a `docs/system/live-updates.md` page and edits to architecture / checkpoints.

Out of scope: SilverTorch IVF live updates (cluster reassignment, not a row slice), `.pt2` packaging, an id -> row hash table.

## Verification as written
`tests/correctness/test_live_update.py` (N=2048, D=128, B=16, K=200): rebuild-vs-incremental parity (bit-exact V1 / V2, id sets for V3), deleted rows vanish, reuse after delete, capacity exhausted raises, legacy path byte-identical and without `valid_mask`, V2 end to end with a filter, filter upsert parity, filter legacy path, watermark monotonic, V3 cascade.
