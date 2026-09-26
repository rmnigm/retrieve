---
chain: "live-update-api"
branch: "main"
parent: "2026-05-23-120000000-live-update-upsert-delete-plan-parked.md"
nextStep: "Queued for roadmap phase G (G-e), after F5 (the paper). Not before the campaign or the current kernel-optimization pass (G-a/G-d/Q4). When picked up: start from Phase A (LiveIndexMixin in a new indexing/live_update.py) exactly as scoped below — the anchors below are current as of 2026-09-26."
created: "2026-09-26T00:13:33Z"
---

# Re-scoped: live upsert/delete API for the LiNR family

## Size/risk estimate
**Medium-large** — comparable in scope to G-a+Q4 combined, as flagged before starting.
This is a genuinely new subsystem (in-place GPU buffer mutation across five module
classes plus both filters), not a bugfix pass. Unlike the torch-export plan, this one
did not shrink: nothing about the design was already half-built by other work.

## What changed since 2026-05-23 (verified against current code, 2026-09-26)
Every anchor the original plan worried about ("re-scope against the L layout... rebind
every anchor") checks out as **already correct, no further rebinding needed**:
- `RetrievalModule` exists at `retrieve/src/retrieve/interfaces.py:77` (`class
  RetrievalModule(nn.Module, abc.ABC)`) — the Phase A dependency the note said was
  "satisfied" still is.
- `OneBitKNN` is confirmed a `_PackedBitsKNN` subclass
  (`retrieve/src/retrieve/modules/bit_knn.py:86` / `:20`) — matches the note's K5 update.
- Bloom signature building is `build_signatures` / `build_query_signatures` in
  `retrieve/src/retrieve/indexing/bloom_hash.py` (lines 124, 159) — the public names the
  note anticipated, not the old private `_build_signatures`.
- The five target classes are all present at the locations the plan implies:
  `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`, `FullScanKNN`
  (`modules/knn.py`), `OneBitKNN` (`modules/bit_knn.py`), `BloomFilter` /
  `ExactAttributeFilter` (`modules/filters.py`).
- No live-update scaffolding has been added anywhere since (`grep -r "live" modules/
  indexing/` turns up nothing relevant) — this is a clean, unstarted slate, not a partial
  implementation to reconcile with.

Net effect: the plan's substance is unchanged and fully current. Nothing shrank, nothing
grew, nothing needs re-derivation beyond what's written below.

## Updated plan (identical to the 2026-05-23 note's design; anchors confirmed, not changed)
- **Phase A**: `retrieve/src/retrieve/indexing/live_update.py`, a `LiveIndexMixin` with
  `_alloc_live_buffers`, `_bump_watermark`, `_effective_mask` (returns the caller's mask
  unchanged when not live), `next_free_rows` (fully on-device, raises "capacity exhausted"
  rather than syncing to check); `capacity=` added to both retrieval/filter ABCs;
  `upsert()` / `delete()` defaults raising `NotImplementedError`.
- **Phase B** (the retrieval side): `PostfilterKNN` pre-transposes item embeddings to
  `[D, capacity]` for a column-slice `index_copy_(1, ...)` upsert; `PrefilterKNN` drops its
  mask param and folds `valid_mask` into the full / torch-prefilter / triton-prefilter
  paths (Triton needs a per-row stable sort to pack live candidates first and recompute
  `counts`); `OneBitKNN` freezes `signs`/`perm` and re-projects `[K, D]` on upsert, routing
  the Triton full scan through the existing `_indirect` path with synthetic candidates when
  live; `FullScanKNN` applies the live mask pre-topk while the caller's mask stays
  post-topk.
- **Phase C** (the filter side): `ExactAttributeFilter` over-allocated with `-1` padding;
  `BloomFilter` with zero signatures on unused rows (so an unset row fails any non-zero
  query); `upsert`-only (delete is a filter no-op, since a deleted row's retrieval-side
  tombstone already excludes it).
- **Phase D** (docs): a new `docs/system/live-updates.md` page, plus edits to
  `architecture.md` and `checkpoints.md` — deferred to the implementing step, not written
  now.

## Verification, unchanged from the original plan
`tests/correctness/test_live_update.py` (N=2048, D=128, B=16, K=200): rebuild-vs-incremental
parity (bit-exact V1/V2, id-set match for V3), deleted rows vanish, row reuse after delete,
capacity-exhausted raises, the legacy (non-live) path stays byte-identical and carries no
`valid_mask`, V2 end-to-end with a filter, filter upsert parity, filter legacy path, watermark
monotonicity, V3's cascade behavior under live updates.

## Out of scope (per the original plan and unchanged)
SilverTorch IVF live updates (cluster reassignment is not a row slice — a different, harder
problem), `.pt2` packaging, an id→row hash table. Composes independently with the
torch-export plan (see [torch-export-refactor](../torch-export-refactor/)) — doing one first
does not block or reshape the other.
