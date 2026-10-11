// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C5).

#pragma once

#include <ATen/ATen.h>

#include <optional>
#include <tuple>

// The upstream-file entry points stfork::search (fork_search_cuda.cu) composes, defined in
// fused_kmean_ann_cuda.cu and bloom_index_search_cuda.cu.
namespace st::ops::fused_kmean_ann {

std::tuple<at::Tensor, at::Tensor> fused_kmean_ann_run_payloads(
    const at::Tensor& selected_cluster_ids,
    const at::Tensor& embeddings,
    const at::Tensor& queries,
    int64_t max_tensor_size_per_row,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    bool has_filtering_bitmask,
    const at::Tensor& warp_payloads,
    int64_t total_needed_warps,
    const int32_t* total_needed_warps_ptr,
    const at::Tensor& remaining_payloads,
    int64_t remaining_docs,
    const int32_t* remaining_docs_ptr,
    const std::optional<at::Tensor>& per_embedding_scale,
    bool query_on_shared_mem);

std::tuple<at::Tensor, at::Tensor> fused_kmean_ann_cuda_precomputed(
    const at::Tensor& cluster_offsets,
    const at::Tensor& cluster_ids,
    const at::Tensor& cluster_length,
    const at::Tensor& embeddings,
    const at::Tensor& queries,
    int64_t max_tensor_size_per_row,
    const std::optional<at::Tensor>& filtering_bit_mask,
    int64_t divisor_for_int8,
    const at::Tensor& cluster_warp_rounded_length_cumsum,
    const at::Tensor& cluster_remaining_length_cumsum,
    const at::Tensor& cluster_warp_size_cumsum);

std::tuple<at::Tensor, at::Tensor> fused_kmean_ann_cuda_with_partial_masks(
    const at::Tensor& cluster_offsets,
    const at::Tensor& cluster_ids,
    const at::Tensor& cluster_length,
    const at::Tensor& embeddings,
    const at::Tensor& queries,
    int64_t max_tensor_size_per_row,
    const at::Tensor& partial_mask_column_counts_cumsum,
    const at::Tensor& partial_mask_first_item_offset_in_column,
    const at::Tensor& partial_mask_column_results,
    int64_t invalid_index_value,
    int64_t divisor_for_int8,
    const std::optional<at::Tensor>& filtering_bit_index,
    const std::optional<at::Tensor>& per_embedding_scale,
    const std::optional<at::Tensor>& cluster_warp_size_opt,
    const std::optional<at::Tensor>& cluster_warp_rounded_length_cumsum_opt,
    const std::optional<at::Tensor>& cluster_remaining_length_cumsum_opt,
    const std::optional<at::Tensor>& cluster_warp_size_cumsum_opt,
    int64_t total_cluster_rounded_warps,
    int64_t total_cluster_remaining_warps);

} // namespace st::ops::fused_kmean_ann

namespace st::ops::bloom_search {

at::Tensor bloom_index_search_batch_prepared_cuda(
    const at::Tensor& bloom_index,
    const at::Tensor& bloom_bundle_b_offsets,
    const at::Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    int64_t k,
    int64_t onebit_hash_k,
    bool return_bool_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor>
bloom_index_search_batch_return_partial_response_prepared_cuda(
    const at::Tensor& bloom_index,
    const at::Tensor& bloom_bundle_b_offsets,
    const at::Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    const at::Tensor& selected_cluster_offsets,
    const at::Tensor& selected_cluster_lengths,
    int64_t k,
    int64_t onebit_hash_k,
    const std::optional<at::Tensor>& query_plan_index,
    int64_t max_columns);

std::tuple<at::Tensor, int64_t, at::Tensor, int64_t> bloom_partial_payloads_prepared_cuda(
    const at::Tensor& bloom_index,
    const at::Tensor& bloom_bundle_b_offsets,
    const at::Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    const std::optional<at::Tensor>& query_plan_index,
    int64_t k,
    int64_t onebit_hash_k,
    const at::Tensor& cluster_offsets,
    const at::Tensor& cluster_ids,
    const at::Tensor& cluster_length,
    const at::Tensor& warp_rounded_length_cumsum,
    const at::Tensor& remaining_length_cumsum,
    const at::Tensor& warp_size_cumsum,
    int64_t max_tensor_size_per_row,
    const std::optional<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& recheck);

} // namespace st::ops::bloom_search
