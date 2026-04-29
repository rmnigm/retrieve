<!-- claude code generated file -->

# Sharding `retrieve` modules across GPUs

> Previously: `retrieve/docs/sharding.md` (originally `retrieve/docs/SHARDING.md`).

Notes on what the SilverTorch paper does, what's easy to implement against
the current codebase, and the trade-offs of each design. Not implemented
yet — captured here so the next pass doesn't re-derive it.

## What the paper actually does

Reference: SilverTorch §5.3 ("Scale Out") + §6 evaluation setup.

> "The ANN index and Bloom index are sharded across GPU cards, with each
> GPU processing a partition of items… each GPU independently computes
> local pre-filtered results through ANN search and feature filtering.
> Item embeddings are then gathered to a single GPU to compute the final
> retrieval results."

Concretely:

1. **Item-partition sharding.** Items are split along `N`. Each GPU owns
   a slice (e.g. round-robin or contiguous). The index on each GPU is a
   **fully independent SilverTorch instance** — its own kmeans run, its
   own centroids, its own `padded_cluster_items`, its own `bloom_sigs`.
   Centroids are *not* shared across shards.
2. **Standalone forward per shard.** At query time each GPU runs the
   whole pipeline (Phase 1 centroid scoring → Phase 2/3 probe+score →
   Phase 4 local top-K) on its slice. **No cross-GPU communication
   during search.**
3. **Single-GPU aggregation.** Partial `(ids, scores)` top-Ks gather on
   one GPU, ids are remapped to global, and one merge `topk(2k → k)`
   produces the final result. OverArch (replicated on every GPU) then
   re-ranks the merged set.
4. **Batch-DP on top of index sharding.** Because OverArch and Value
   Model are replicated, the *batch* is also distributed across GPUs in
   parallel — index sharding is orthogonal to request-batch DP.
5. **Sharding is memory-driven, not throughput-driven.** §5.3: "GPU
   memory is the primary factor determining the number of GPUs required
   for serving." 80M dataset → 2 shards because the INT8 ANN index
   doesn't fit in one 40 GB card; 10M runs un-sharded.

What the paper does **not** do:

- Tensor-parallel-within-a-kernel (no NCCL inside `forward`).
- Sharding by clusters or within clusters (would require a global
  kmeans, which they don't build).
- Cross-shard centroid replication.

## Implementation against the current codebase

This is dramatically simpler than tensor-parallel. **Zero kernel changes
required** for any path.

### `ShardedSilverTorch` (and analogous wrappers for `IVF_INT8_ANN`, `FullScanKNN`)

A thin wrapper that owns one underlying `SilverTorch` per device:

- **`register_index(item_embs, item_clause_attrs)`** — partition along
  `N` (default: contiguous slices of size `ceil(N / world_size)`), then
  for each device construct a [`SilverTorch`](../../retrieve/src/retrieve/layers/silvertorch/main.py)
  pointed at its slice. Store per-shard `id_offset = shard_idx *
  shard_size` so local ids can be mapped back to global.
- **`forward(query, query_clause_attrs=None, mask=None)`** —
  1. Replicate `query` (and `query_clause_attrs` / `mask` if present) to
     every device. Mask needs a per-shard slice along the `N` axis so
     each shard sees only its own items.
  2. Dispatch each shard's `forward` on its own
     [`torch.cuda.Stream`](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html)
     — they run concurrently because there is no cross-shard data
     dependency.
  3. Gather `(local_ids[B, k], local_scores[B, k])` to a designated
     output device. Add `id_offset` to local ids before gather (cheap
     elementwise add) so the merged set has globally unique ids.
  4. Concatenate scores along dim 1 (`[B, world_size * k]`), run one
     final `torch.topk(merged, k, dim=1)`, gather merged ids the same
     way the existing kernel wrappers do (`gather(1, topk_local)`).

Scope: ~80–120 lines for `ShardedSilverTorch`, similar for the other
two. Only depends on existing kernels — `codesigned_probe_score`,
`int8_ann_fused`, `bloom_match` all run unchanged on each shard.

### Cost of the merge step

For typical bench cells (B=16, k=1024, fp32 scores + int64 ids):

- Per-shard payload: `B * k * (4 + 8) = 192 KiB`.
- Two shards on the same node: ~10 µs over NVLink. Negligible vs the
  single-shard kernel times in [bench.md](../system/bench.md).

The merge `topk(2k → k)` runs on `[B, 2k]` = 32 KiB and is essentially
free.

## Trade-offs

| design | recall | comms | code | when to pick |
|---|---|---|---|---|
| **paper / scatter-gather** (above) | slightly lower — per-shard kmeans can't form clusters that span shards | none during search; one gather of `[B, k]` per shard | trivial wrapper, no kernel changes | matches paper; right default |
| **within-cluster sharding** (each shard holds half of every cluster's items, centroids replicated) | matches single-GPU baseline (one global kmeans) | one shard-id-routing comm in Phase 2 | wrapper + a build-time helper to slice `padded_cluster_items` consistently | only if recall regression of scatter-gather is shown to matter |
| **shard-by-cluster** (each shard owns disjoint clusters) | matches single-GPU | per-query routing of probed clusters across shards is annoying | medium; needs dynamic dispatch on `probe_ids` | not recommended — strictly more complex than the other two with no real upside |
| **TP-within-kernel** (NCCL inside `forward`) | matches single-GPU | NCCL on the inner loop | hard; would require kernel-level reductions | not what the paper does; only for situations where one shard's slice doesn't fit a single kernel launch |

The paper accepts the recall hit from per-shard kmeans because their
headline numbers (the 80M-dataset results) are reported with sharded
indexes — they don't quantify the loss vs a global kmeans, but it's
clearly tolerable for production. Default to the paper's design.

## Open questions for when this gets built

- **Shard partitioning policy.** Contiguous slices vs. round-robin vs.
  hash-based. Round-robin gives more uniform per-shard kmeans
  convergence on datasets with item-id clustering; contiguous is simpler
  and matches the paper's "partition of items" wording. Probably
  contiguous + a flag.
- **Mask shape on the API.** External `mask` is `[B, N]` global. The
  wrapper needs to slice it per shard; cheap, but worth keeping in mind
  for tests.
- **`candidate_ids` path.** If a caller passes global candidate ids,
  route each id to the owning shard (`shard = id // shard_size`) and
  gather. The current `_forward_candidates` works on local ids; the
  wrapper needs a tiny scatter step.
- **Per-shard recall metric in benches.** `recall@K` should be measured
  vs an exact full-scan on the *unsharded* index, so the per-shard
  kmeans loss is visible in the bench output. Otherwise the regression
  is invisible.
- **Stream synchronization.** `torch.cuda.Event` per shard, wait on
  device 0 before the merge `topk`. Standard pattern; not a research
  question.
