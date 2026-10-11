# Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
# rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C4, C5).

"""Shape functions of the ops a SilverTorch forward calls, so ``torch.compile`` traces it whole
(the harness ``graph`` mode): every output shape follows from the input shapes and the int
arguments alone, which is what C3 / C4 made true."""

import torch


def _scores(cluster_ids, embeddings, max_tensor_size_per_row, divisor_for_int8, scale):
    width = (max_tensor_size_per_row + 31) // 32 * 32
    if embeddings.dtype != torch.int8:
        dtype = embeddings.dtype
    else:
        dtype = torch.half if (divisor_for_int8 != -1 or scale is not None) else torch.int32
    rows = cluster_ids.shape[0]
    return (
        embeddings.new_empty((rows, width), dtype=dtype),
        embeddings.new_empty((rows, width), dtype=torch.int32),
    )


@torch.library.register_fake("stfork::fused_kmean_ann")
def _(cluster_offsets, cluster_ids, cluster_length, embeddings, queries, max_tensor_size_per_row,
      filtering_bit_mask=None, invalid_index_value=-1, divisor_for_int8=-1,
      filtering_bit_index=None, per_embedding_scale=None):  # fmt: skip
    return _scores(
        cluster_ids, embeddings, max_tensor_size_per_row, divisor_for_int8, per_embedding_scale
    )


@torch.library.register_fake("stfork::fused_kmean_ann_with_partial_masks")
def _(cluster_offsets, cluster_ids, cluster_length, embeddings, queries, max_tensor_size_per_row,
      partial_mask_column_counts_cumsum, partial_mask_first_item_offset_in_column,
      partial_mask_column_results, invalid_index_value=-1, divisor_for_int8=-1,
      filtering_bit_index=None, per_embedding_scale=None, cluster_warp_size=None,
      cluster_warp_rounded_length_cumsum=None, cluster_remaining_length_cumsum=None,
      cluster_warp_size_cumsum=None, total_cluster_rounded_warps=0,
      total_cluster_remaining_warps=0):  # fmt: skip
    return _scores(
        cluster_ids, embeddings, max_tensor_size_per_row, divisor_for_int8, per_embedding_scale
    )


@torch.library.register_fake("stfork::bloom_index_search_batch_prepared")
def _(bloom_index, bloom_bundle_b_offsets, plans, plan_count, max_stack_size, k, hash_k,
      return_bool_mask=False):  # fmt: skip
    columns = (bloom_bundle_b_offsets.shape[0] - 1) * 32
    if return_bool_mask:
        return bloom_index.new_empty((plan_count, columns * 64), dtype=torch.bool)
    return bloom_index.new_empty((plan_count, columns), dtype=torch.int64)


@torch.library.register_fake("stfork::bloom_index_search_batch_return_partial_response_prepared")
def _(bloom_index, bloom_bundle_b_offsets, plans, plan_count, max_stack_size,
      selected_cluster_offsets, selected_cluster_lengths, k, hash_k, query_plan_index,
      max_columns):  # fmt: skip
    clusters = selected_cluster_lengths.numel()
    return (
        bloom_index.new_empty((clusters,), dtype=torch.int32),
        bloom_index.new_empty((clusters,), dtype=torch.int8),
        bloom_index.new_empty((max_columns,), dtype=torch.int64),
    )


@torch.library.register_fake("stfork::search")
def _(query, probe_ids, cluster_offsets, cluster_sizes, item_codes, sort_perm, global_scale, k,
      max_tensor_size_per_row, divisor_for_int8=-1, filtering_bit_mask=None, bloom_index=None,
      bloom_bundle_b_offsets=None, plans=None, plan_count=0, max_stack_size=0,
      query_plan_index=None, bloom_k=0, hash_k=0, bloom_partial=True, item_clause_attrs=None,
      clause_is_reverse=None, query_clause_attrs=None):  # fmt: skip
    rows = query.shape[0]
    return (
        query.new_empty((rows, k), dtype=torch.int64),
        query.new_empty((rows, k), dtype=torch.float32),
    )
