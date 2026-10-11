# Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
# rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C4, C5).

"""Shape functions of every ``stfork`` op, so ``torch.compile`` traces a forward whole (the
harness ``graph`` mode) and ``torch.library.opcheck`` passes on each (the compile gate). The
forward's ops have outputs whose shapes follow from the input shapes and int arguments alone
(what C3 / C4 made true); the build / parse / upstream-shaped search ops, outside any compiled
forward, give their data-dependent sizes as unbacked sizes (``new_dynamic_size``)."""

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


def _bundles(feature_ids, feature_offsets):
    docs = (feature_offsets.shape[0] - 1) // feature_ids.shape[0]
    return (docs + 2047) // 2048


@torch.library.register_fake("stfork::bloom_index_build")
def _(feature_ids, feature_offsets, feature_values, b_multiplier, k, fast_build=False):
    words = torch.library.get_ctx().new_dynamic_size()
    return (
        feature_values.new_empty((words,), dtype=torch.int64),
        feature_values.new_empty((_bundles(feature_ids, feature_offsets) + 1,), dtype=torch.int64),
    )


@torch.library.register_fake("stfork::bloom_index_build_fixed")
def _(feature_ids, feature_offsets, feature_values, bits_per_doc, k, fast_build=False):
    bundles = _bundles(feature_ids, feature_offsets)
    return (
        feature_values.new_empty((bundles * bits_per_doc * 32,), dtype=torch.int64),
        feature_values.new_empty((bundles + 1,), dtype=torch.int64),
    )


@torch.library.register_fake("stfork::bloom_index_search_batch")
def _(bloom_index, bloom_bundle_b_offsets, bloom_query_plans_data, bloom_query_plans_offsets, k,
      hash_k, return_bool_mask=True):  # fmt: skip
    plans = bloom_query_plans_offsets.shape[0] // 8
    columns = (bloom_bundle_b_offsets.shape[0] - 1) * 32
    if return_bool_mask:
        return bloom_index.new_empty((plans, columns * 64), dtype=torch.bool)
    return bloom_index.new_empty((plans, columns), dtype=torch.int64)


def _partial(bloom_index, selected_cluster_lengths):
    clusters = selected_cluster_lengths.numel()
    columns = torch.library.get_ctx().new_dynamic_size()
    return (
        bloom_index.new_empty((clusters,), dtype=torch.int32),
        bloom_index.new_empty((clusters,), dtype=torch.int8),
        bloom_index.new_empty((columns,), dtype=torch.int64),
    )


@torch.library.register_fake("stfork::bloom_index_search_batch_return_partial_response")
def _(bloom_index, bloom_bundle_b_offsets, bloom_query_plans_data, bloom_query_plans_offsets,
      selected_cluster_offsets, selected_cluster_lengths, k, hash_k, query_plan_index=None):  # fmt: skip
    return _partial(bloom_index, selected_cluster_lengths)


@torch.library.register_fake("stfork::bloom_index_search_batch_return_partial_response_multiple")
def _(list_bloom_index, list_bloom_bundle_b_offsets, bloom_query_plans_data,
      bloom_query_plans_offsets, list_selected_cluster_offsets, list_selected_cluster_lengths, k,
      hash_k, query_plan_index=None, fuse=False):  # fmt: skip
    parts = [_partial(b, n) for b, n in zip(list_bloom_index, list_selected_cluster_lengths)]
    return [p[0] for p in parts], [p[1] for p in parts], [p[2] for p in parts]


@torch.library.register_fake("stfork::fused_kmean_ann_with_partial_masks_multiple")
def _(queries, list_cluster_offsets, list_selected_cluster_ids, list_selected_cluster_lengths,
      list_embeddings, list_max_tensor_size_per_row, list_partial_mask_column_counts_cumsum,
      list_partial_mask_first_item_offset_in_column, list_partial_mask_column_results,
      invalid_index_value=-1, divisor_for_int8=-1, filtering_bit_index=None,
      list_per_embedding_scale=None, fuse=False):  # fmt: skip
    scores, indices = [], []
    for i, (ids, emb, width) in enumerate(
        zip(list_selected_cluster_ids, list_embeddings, list_max_tensor_size_per_row)
    ):
        scale = None if list_per_embedding_scale is None else list_per_embedding_scale[i]
        sc, ix = _scores(ids, emb, width, divisor_for_int8, scale)
        scores.append(sc)
        indices.append(ix)
    return scores, indices


@torch.library.register_fake("stfork::generate_column_info_for_clusters")
def _(selected_cluster_offsets, selected_cluster_lengths):
    n = selected_cluster_lengths.numel()
    return (
        selected_cluster_lengths.new_empty((n,), dtype=torch.int32),
        selected_cluster_lengths.new_empty((n,), dtype=torch.int32),
        selected_cluster_lengths.new_empty((n,), dtype=torch.int8),
    )


@torch.library.register_fake("stfork::is_topk")
def _(scores, ks, is_topk_output, largest=True):
    return None


@torch.library.register_fake("stfork::parse_expression_query_batch")
def _(expressions, silvertorch_ks, bloom_hash_k, return_query_plan=True, max_sub_queries=5):
    # The parse depends only on the strings and ints (``silvertorch_ks`` is ignored upstream), so
    # its shapes are the real CPU parse's, run outside fake mode.
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        ks = torch.ones(len(expressions), dtype=torch.int64)
        stack, (data, offsets) = torch.ops.stfork.parse_expression_query_batch(
            list(expressions), ks, bloom_hash_k, return_query_plan, max_sub_queries
        )
    return stack, [
        silvertorch_ks.new_empty(data.shape, dtype=data.dtype),
        silvertorch_ks.new_empty(offsets.shape, dtype=offsets.dtype),
    ]


@torch.library.register_fake("stfork::probe_prep")
def _(query, probe_ids, cluster_offsets, cluster_sizes):
    rows, n_probe = probe_ids.shape
    i64 = probe_ids.new_empty((rows, n_probe), dtype=torch.int64)
    return (
        query.new_empty(query.shape, dtype=torch.int8),
        query.new_empty((rows,), dtype=torch.float32),
        i64,
        torch.empty_like(i64),
        torch.empty_like(i64),
        probe_ids.new_empty((rows * n_probe,), dtype=torch.int32),
        probe_ids.new_empty((rows, n_probe), dtype=torch.int32),
        probe_ids.new_empty((rows * n_probe,), dtype=torch.int32),
        probe_ids.new_empty((rows * n_probe,), dtype=torch.int32),
    )


@torch.library.register_fake("stfork::take_top_k_and_gather_from_main_and_fresh")
def _(main_scores, fresh_scores, main_indices, fresh_indices, k, keys, list_main_data,
      list_fresh_data, kernel_version=2):  # fmt: skip
    rows = main_scores.shape[0]
    kk = min(k, main_scores.shape[1] + fresh_scores.shape[1])
    gathered = {
        key: data.new_empty((rows * kk, data.shape[1]) if data.dim() == 2 else (rows * kk,))
        for key, data in zip(keys, list_main_data)
    }
    return (
        main_scores.new_empty((rows, kk)),
        main_scores.new_empty((rows, kk), dtype=torch.int64),
        main_scores.new_empty((rows, kk), dtype=torch.bool),
        gathered,
    )
