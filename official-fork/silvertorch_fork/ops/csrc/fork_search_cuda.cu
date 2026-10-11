// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C5).

// stfork::search: the whole SilverTorch phase 2 + 3 forward in one op call, composed of the
// (forked) upstream kernels: query int8, the probed clusters' lengths, the filter's masks (bloom
// over prepared plans, partial or full; or a given full-N mask), payloads + scorer, top-k on the
// raw scores, and the [B, k] epilogue. No host sync, no data-dependent allocation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <limits>
#include <optional>
#include <tuple>

#include "fork_search.h"
#include "fused_kmean_ann.cuh"

namespace st::ops::fork_search {

using at::Tensor;

namespace {

// retrieve.indexing.quantize.quantize_int8, op for op (so bit for bit): per-row abs max clamped
// at 1e-8, scale = abs_max / 127, codes = round(fp32(fp64(x) / abs_max) * 127) clamped to int8.
std::tuple<Tensor, Tensor> quantize_rows(const Tensor& query) {
  Tensor abs_max = query.abs().amax({1}, /*keepdim=*/true).clamp_min(1e-8);
  Tensor scales = (abs_max / 127.0).squeeze(1);
  Tensor codes = (query.to(at::kDouble) / abs_max)
                     .to(at::kFloat)
                     .mul(127.0)
                     .round()
                     .clamp(-128, 127)
                     .to(at::kChar);
  return {codes, scales};
}

} // namespace

std::tuple<Tensor, Tensor> search_cuda(
    const Tensor& query,
    const Tensor& probe_ids,
    const Tensor& cluster_offsets,
    const Tensor& cluster_sizes,
    const Tensor& item_codes,
    const Tensor& sort_perm,
    const Tensor& global_scale,
    int64_t k,
    int64_t max_tensor_size_per_row,
    int64_t divisor_for_int8,
    const std::optional<Tensor>& filtering_bit_mask,
    const std::optional<Tensor>& bloom_index,
    const std::optional<Tensor>& bloom_bundle_b_offsets,
    const std::optional<Tensor>& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    const std::optional<Tensor>& query_plan_index,
    int64_t bloom_k,
    int64_t hash_k,
    bool bloom_partial) {
  TORCH_CHECK(query.is_cuda() && query.dim() == 2 && query.scalar_type() == at::kFloat);
  TORCH_CHECK(probe_ids.dim() == 2 && probe_ids.size(0) == query.size(0));
  TORCH_CHECK(item_codes.scalar_type() == at::kChar && item_codes.size(1) == query.size(1));
  TORCH_CHECK(!(plans.has_value() && filtering_bit_mask.has_value()));
  at::cuda::CUDAGuard device_guard(query.device());

  auto [q_codes, q_scales] = quantize_rows(query.contiguous());
  Tensor cluster_ids = probe_ids.to(at::kLong).contiguous();
  Tensor cluster_length = cluster_sizes.index_select(0, cluster_ids.view(-1))
                              .to(at::kLong)
                              .view(cluster_ids.sizes())
                              .contiguous();
  Tensor offsets = cluster_offsets.to(at::kLong).contiguous();

  Tensor raw;
  Tensor idx;
  if (plans.has_value() && bloom_partial) {
    TORCH_CHECK(bloom_index.has_value() && bloom_bundle_b_offsets.has_value());
    const int64_t max_columns = cluster_ids.size(0) *
        (max_tensor_size_per_row / 64 + 2 * cluster_ids.size(1));
    auto [cumsum, first, columns] = st::ops::bloom_search::
        bloom_index_search_batch_return_partial_response_prepared_cuda(
            *bloom_index,
            *bloom_bundle_b_offsets,
            *plans,
            plan_count,
            max_stack_size,
            offsets.index_select(0, cluster_ids.view(-1)).view(cluster_ids.sizes()),
            cluster_length,
            bloom_k,
            hash_k,
            query_plan_index,
            max_columns);
    std::tie(raw, idx) =
        st::ops::fused_kmean_ann::fused_kmean_ann_cuda_with_partial_masks(
            offsets,
            cluster_ids,
            cluster_length,
            item_codes,
            q_codes,
            max_tensor_size_per_row,
            cumsum,
            first,
            columns,
            /*invalid_index_value=*/-1,
            divisor_for_int8,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            0,
            0);
  } else {
    std::optional<Tensor> mask = filtering_bit_mask;
    if (plans.has_value()) {
      TORCH_CHECK(bloom_index.has_value() && bloom_bundle_b_offsets.has_value());
      // the packed search output is high-first words, the scorer's mask order
      mask = st::ops::bloom_search::bloom_index_search_batch_prepared_cuda(
          *bloom_index,
          *bloom_bundle_b_offsets,
          *plans,
          plan_count,
          max_stack_size,
          bloom_k,
          hash_k,
          /*return_bool_mask=*/false);
    }
    std::tie(raw, idx) = st::ops::fused_kmean_ann::fused_kmean_ann_cuda(
        offsets,
        cluster_ids,
        cluster_length,
        item_codes,
        q_codes,
        max_tensor_size_per_row,
        mask,
        /*invalid_index_value=*/-1,
        divisor_for_int8,
        std::nullopt,
        std::nullopt);
  }

  // The epilogue of retrieve.ops.official.official_probe_score, op for op: top-k on the raw
  // scores (pads at the dtype's minimum), then dequantize only the [B, k] winners.
  Tensor valid = idx.ge(0);
  const bool int32_path = raw.scalar_type() == at::kInt;
  Tensor floored = int32_path
      ? at::where(valid, raw, at::Scalar(std::numeric_limits<int32_t>::min()))
      : at::where(valid, raw, at::Scalar(-std::numeric_limits<double>::infinity()));
  const int64_t actual_k = std::min<int64_t>(k, raw.size(1));
  auto [top_raw, top_slot] = at::topk(floored, actual_k, /*dim=*/1);
  Tensor top_valid = valid.gather(1, top_slot);
  Tensor pos = idx.gather(1, top_slot).to(at::kLong).clamp_min(0);
  Tensor ids = at::where(
      top_valid, sort_perm.index_select(0, pos.view(-1)).view(pos.sizes()), -1);
  Tensor scores = int32_path
      ? top_raw.to(at::kFloat).mul(q_scales.unsqueeze(1)).mul(global_scale)
      : top_raw.to(at::kFloat).mul(
            q_scales.mul(static_cast<double>(divisor_for_int8))
                .mul(global_scale)
                .unsqueeze(1));
  scores = at::where(
      top_valid, scores, at::Scalar(-std::numeric_limits<double>::infinity()));
  if (actual_k < k) {
    ids = at::cat({ids, ids.new_full({ids.size(0), k - actual_k}, -1)}, 1);
    scores = at::cat(
        {scores,
         scores.new_full(
             {scores.size(0), k - actual_k},
             -std::numeric_limits<float>::infinity())},
        1);
  }
  return {ids, scores};
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl("search", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(search_cuda)));
}

} // namespace st::ops::fork_search
