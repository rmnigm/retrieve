// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C5).

// stfork::search: the whole SilverTorch phase 2 + 3 forward in one op call, composed of the
// (forked) upstream kernels: query int8, the probed clusters' lengths, the filter's masks (bloom
// over prepared plans, partial or full; or a given full-N mask), payloads + scorer, top-k on the
// raw scores, and the [B, k] epilogue. No host sync, no data-dependent allocation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include <limits>
#include <optional>
#include <tuple>
#include <type_traits>

#include "fork_search.h"
#include "fused_kmean_ann.cuh"

namespace st::ops::fork_search {

using at::Tensor;

namespace {

constexpr int kPrepThreads = 256;

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

// One launch for everything before the masks (C6). Blocks [0, B): row b's int8 codes and scale,
// bit for bit retrieve.indexing.quantize.quantize_int8 as eager torch computes it: abs max
// clamped at fp32(1e-8); scale = abs_max * fp32(1 / 127) (torch divides by a CPU scalar as a
// multiply by its fp32 reciprocal); code = rint(fp32(fp64(x) / fp64(abs_max)) * 127) clamped to
// [-128, 127]. Block B: the probed clusters (ids, lengths, offsets) and the scorer's warp
// rounding with its three cumsums (warp-rounded lengths per row, remainders and warp counts
// over all B * P), as round_cluster_to_warp + cumsum would, scanned in int64.
__global__ void __launch_bounds__(kPrepThreads) probe_prep_kernel(
    const float* __restrict__ query,
    int64_t rows,
    int64_t dim,
    const int64_t* __restrict__ probe_ids,
    int64_t n_probe,
    const int64_t* __restrict__ cluster_offsets,
    const int64_t* __restrict__ cluster_sizes,
    int8_t* __restrict__ codes,
    float* __restrict__ scales,
    int64_t* __restrict__ cluster_ids,
    int64_t* __restrict__ cluster_length,
    int64_t* __restrict__ selected_offsets,
    int32_t* __restrict__ warp_size,
    int32_t* __restrict__ warp_rounded_length_cumsum,
    int32_t* __restrict__ remaining_length_cumsum,
    int32_t* __restrict__ warp_size_cumsum) {
  using Reduce = cub::BlockReduce<float, kPrepThreads>;
  using Scan = cub::BlockScan<int64_t, kPrepThreads>;
  __shared__ union {
    typename Reduce::TempStorage reduce;
    typename Scan::TempStorage scan[3];
  } tmp;
  __shared__ float row_abs_max;
  __shared__ int64_t carry[3];

  if (blockIdx.x < rows) {
    const float* x = query + blockIdx.x * dim;
    float m = 0.0f;
    for (int64_t d = threadIdx.x; d < dim; d += kPrepThreads) {
      m = fmaxf(m, fabsf(x[d]));
    }
    m = Reduce(tmp.reduce).Reduce(m, MaxOp());
    if (threadIdx.x == 0) {
      row_abs_max = fmaxf(m, static_cast<float>(1e-8));
      scales[blockIdx.x] = row_abs_max * (1.0f / 127.0f);
    }
    __syncthreads();
    const double a = static_cast<double>(row_abs_max);
    int8_t* out = codes + blockIdx.x * dim;
    for (int64_t d = threadIdx.x; d < dim; d += kPrepThreads) {
      float v = static_cast<float>(static_cast<double>(x[d]) / a) * 127.0f;
      v = fminf(fmaxf(rintf(v), -128.0f), 127.0f);
      out[d] = static_cast<int8_t>(v);
    }
    return;
  }

  if (threadIdx.x < 3) {
    carry[threadIdx.x] = 0;
  }
  __syncthreads();
  const int64_t total = rows * n_probe;
  for (int64_t base = 0; base < total; base += kPrepThreads) {
    const int64_t i = base + threadIdx.x;
    int64_t rounded = 0;
    int64_t remainder = 0;
    int64_t warps = 0;
    if (i < total) {
      const int64_t c = probe_ids[i];
      const int64_t len = cluster_sizes[c];
      cluster_ids[i] = c;
      cluster_length[i] = len;
      selected_offsets[i] = cluster_offsets[c];
      warps = len / 32;
      rounded = warps * 32;
      remainder = len % 32;
      warp_size[i] = static_cast<int32_t>(warps);
    }
    int64_t s_rounded;
    int64_t s_remainder;
    int64_t s_warps;
    int64_t t_rounded;
    int64_t t_remainder;
    int64_t t_warps;
    Scan(tmp.scan[0]).InclusiveSum(rounded, s_rounded, t_rounded);
    Scan(tmp.scan[1]).InclusiveSum(remainder, s_remainder, t_remainder);
    Scan(tmp.scan[2]).InclusiveSum(warps, s_warps, t_warps);
    if (i < total) {
      // the warp-rounded lengths' cumsum over all B * P first; made per row below
      warp_rounded_length_cumsum[i] = static_cast<int32_t>(carry[0] + s_rounded);
      remaining_length_cumsum[i] = static_cast<int32_t>(carry[1] + s_remainder);
      warp_size_cumsum[i] = static_cast<int32_t>(carry[2] + s_warps);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      carry[0] += t_rounded;
      carry[1] += t_remainder;
      carry[2] += t_warps;
    }
    __syncthreads();
  }
  // Per row: subtract the global sum before the row's first cluster. Chunks run back to front,
  // reads before writes, so every base an element reads is still the global value.
  for (int64_t base = (total - 1) / kPrepThreads * kPrepThreads; base >= 0;
       base -= kPrepThreads) {
    const int64_t i = base + threadIdx.x;
    int32_t value = 0;
    if (i < total) {
      const int64_t row_first = i / n_probe * n_probe;
      value = warp_rounded_length_cumsum[i] -
          (row_first == 0 ? 0 : warp_rounded_length_cumsum[row_first - 1]);
    }
    __syncthreads();
    if (i < total) {
      warp_rounded_length_cumsum[i] = value;
    }
    __syncthreads();
  }
}

// The two-level top-k (C7; our Triton epilogue's, kernels.md § SilverTorch kernels "Top-k"):
// from kTopkMinSlots slots of [B, W], where a row has at least kTopkMinRatio whole kTopkBlock-slot
// blocks per top-k slot, the top-k runs on the slots of the row's k best blocks. Exact: every
// block holding a slot above the k-th score has a larger max than any block without one, and
// there are fewer than k of them.
constexpr int64_t kTopkBlock = 256;
constexpr int64_t kTopkMinSlots = int64_t{1} << 23;
constexpr int64_t kTopkMinRatio = 8;

template <typename T>
using TopkAcc = std::conditional_t<std::is_same_v<T, int32_t>, int32_t, float>;

// One warp per block of kTopkBlock slots: the max of the slots' floored scores (a slot the
// scorer never wrote, index -1, counts as `floor`, as the single-level path's where()).
template <typename T>
__global__ void topk_block_max_kernel(
    const T* __restrict__ raw,
    const int32_t* __restrict__ idx,
    int64_t width,
    int64_t n_blocks,
    int64_t total_blocks,
    TopkAcc<T> floor,
    T* __restrict__ block_max) {
  const int lane = threadIdx.x % 32;
  for (int64_t w = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       w < total_blocks;
       w += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
    const int64_t row = w / n_blocks;
    const int64_t first = (w % n_blocks) * kTopkBlock;
    TopkAcc<T> m = floor;
    for (int64_t j = lane; j < kTopkBlock && first + j < width; j += 32) {
      const int64_t slot = row * width + first + j;
      if (idx[slot] >= 0) {
        const TopkAcc<T> v = static_cast<TopkAcc<T>>(raw[slot]);
        m = v > m ? v : m;
      }
    }
#pragma unroll
    for (int off = 16; off > 0; off /= 2) {
      const TopkAcc<T> o = __shfl_xor_sync(0xffffffff, m, off);
      m = o > m ? o : m;
    }
    if (lane == 0) {
      block_max[w] = static_cast<T>(m);
    }
  }
}

// The chosen blocks' slots as candidates [B, k * kTopkBlock]: floored scores and slot indices
// (-1 past the row's width).
template <typename T>
__global__ void topk_gather_kernel(
    const T* __restrict__ raw,
    const int32_t* __restrict__ idx,
    int64_t width,
    const int64_t* __restrict__ blocks,
    int64_t k,
    int64_t total,
    TopkAcc<T> floor,
    T* __restrict__ cand_val,
    int64_t* __restrict__ cand_slot) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / (k * kTopkBlock);
    const int64_t slot =
        blocks[row * k + (i / kTopkBlock) % k] * kTopkBlock + i % kTopkBlock;
    const bool in_row = slot < width;
    T v = static_cast<T>(floor);
    if (in_row && idx[row * width + slot] >= 0) {
      v = raw[row * width + slot];
    }
    cand_val[i] = v;
    cand_slot[i] = in_row ? slot : -1;
  }
}

bool use_two_level_topk(int64_t rows, int64_t width, int64_t k) {
  return rows * width >= kTopkMinSlots && width / kTopkBlock >= kTopkMinRatio * k;
}

// (top raw scores [B, k], their slots [B, k], -1 for a candidate past the row's width).
std::tuple<Tensor, Tensor> raw_topk(const Tensor& raw, const Tensor& idx, int64_t k) {
  const bool int32_path = raw.scalar_type() == at::kInt;
  const int64_t rows = raw.size(0);
  const int64_t width = raw.size(1);
  if (!use_two_level_topk(rows, width, k)) {
    Tensor floored = int32_path
        ? at::where(idx.ge(0), raw, at::Scalar(std::numeric_limits<int32_t>::min()))
        : at::where(idx.ge(0), raw, at::Scalar(-std::numeric_limits<double>::infinity()));
    return at::topk(floored, k, /*dim=*/1);
  }
  const int64_t n_blocks = (width + kTopkBlock - 1) / kTopkBlock;
  const int64_t candidates = k * kTopkBlock;
  Tensor block_max = at::empty({rows, n_blocks}, raw.options());
  Tensor cand_val = at::empty({rows, candidates}, raw.options());
  Tensor cand_slot = at::empty({rows, candidates}, raw.options().dtype(at::kLong));
  const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream();
  const int64_t max_grid = 32 * sms;
  auto launch = [&](auto zero) {
    using T = decltype(zero);
    const TopkAcc<T> floor = int32_path
        ? static_cast<TopkAcc<T>>(std::numeric_limits<int32_t>::min())
        : static_cast<TopkAcc<T>>(-std::numeric_limits<float>::infinity());
    const T* raw_ptr = reinterpret_cast<const T*>(raw.const_data_ptr());
    topk_block_max_kernel<T><<<
        std::min(max_grid, (rows * n_blocks * 32 + 255) / 256),
        256,
        0,
        stream>>>(
        raw_ptr,
        idx.const_data_ptr<int32_t>(),
        width,
        n_blocks,
        rows * n_blocks,
        floor,
        reinterpret_cast<T*>(block_max.mutable_data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    Tensor blocks = std::get<1>(at::topk(block_max, k, 1, true, /*sorted=*/false)).contiguous();
    topk_gather_kernel<T><<<
        std::min(max_grid, (rows * candidates + 255) / 256),
        256,
        0,
        stream>>>(
        raw_ptr,
        idx.const_data_ptr<int32_t>(),
        width,
        blocks.const_data_ptr<int64_t>(),
        k,
        rows * candidates,
        floor,
        reinterpret_cast<T*>(cand_val.mutable_data_ptr()),
        cand_slot.mutable_data_ptr<int64_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  if (int32_path) {
    launch(int32_t{0});
  } else {
    launch(c10::Half{0.0f});
  }
  auto [top_raw, cand_pos] = at::topk(cand_val, k, /*dim=*/1);
  return {top_raw, cand_slot.gather(1, cand_pos)};
}

struct Prep {
  Tensor codes;
  Tensor scales;
  Tensor cluster_ids;
  Tensor cluster_length;
  Tensor selected_offsets;
  Tensor warp_size;
  Tensor warp_rounded_length_cumsum;
  Tensor remaining_length_cumsum;
  Tensor warp_size_cumsum;
};

Prep probe_prep(
    const Tensor& query,
    const Tensor& probe_ids,
    const Tensor& cluster_offsets,
    const Tensor& cluster_sizes) {
  TORCH_CHECK(query.is_cuda() && query.dim() == 2 && query.scalar_type() == at::kFloat);
  TORCH_CHECK(probe_ids.dim() == 2 && probe_ids.size(0) == query.size(0));
  TORCH_CHECK(cluster_offsets.scalar_type() == at::kLong);
  TORCH_CHECK(cluster_sizes.scalar_type() == at::kLong);
  const int64_t rows = query.size(0);
  const int64_t n_probe = probe_ids.size(1);
  Tensor q = query.contiguous();
  Tensor probes = probe_ids.to(at::kLong).contiguous();
  auto i64 = probes.options();
  auto i32 = probes.options().dtype(at::kInt);
  Prep p{
      at::empty(q.sizes(), q.options().dtype(at::kChar)),
      at::empty({rows}, q.options()),
      at::empty({rows, n_probe}, i64),
      at::empty({rows, n_probe}, i64),
      at::empty({rows, n_probe}, i64),
      at::empty({rows * n_probe}, i32),
      at::empty({rows, n_probe}, i32),
      at::empty({rows * n_probe}, i32),
      at::empty({rows * n_probe}, i32)};
  probe_prep_kernel<<<
      rows + 1,
      kPrepThreads,
      0,
      at::cuda::getCurrentCUDAStream()>>>(
      q.const_data_ptr<float>(),
      rows,
      q.size(1),
      probes.const_data_ptr<int64_t>(),
      n_probe,
      cluster_offsets.contiguous().const_data_ptr<int64_t>(),
      cluster_sizes.contiguous().const_data_ptr<int64_t>(),
      p.codes.mutable_data_ptr<int8_t>(),
      p.scales.mutable_data_ptr<float>(),
      p.cluster_ids.mutable_data_ptr<int64_t>(),
      p.cluster_length.mutable_data_ptr<int64_t>(),
      p.selected_offsets.mutable_data_ptr<int64_t>(),
      p.warp_size.mutable_data_ptr<int32_t>(),
      p.warp_rounded_length_cumsum.mutable_data_ptr<int32_t>(),
      p.remaining_length_cumsum.mutable_data_ptr<int32_t>(),
      p.warp_size_cumsum.mutable_data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return p;
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
  TORCH_CHECK(item_codes.scalar_type() == at::kChar && item_codes.size(1) == query.size(1));
  TORCH_CHECK(!(plans.has_value() && filtering_bit_mask.has_value()));
  at::cuda::CUDAGuard device_guard(query.device());

  Tensor offsets = cluster_offsets.to(at::kLong).contiguous();
  Prep prep = probe_prep(query, probe_ids, offsets, cluster_sizes.to(at::kLong));
  const Tensor& q_codes = prep.codes;
  const Tensor& q_scales = prep.scales;
  const Tensor& cluster_ids = prep.cluster_ids;
  const Tensor& cluster_length = prep.cluster_length;

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
            prep.selected_offsets,
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
            prep.warp_size,
            prep.warp_rounded_length_cumsum,
            prep.remaining_length_cumsum,
            prep.warp_size_cumsum,
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
    std::tie(raw, idx) =
        st::ops::fused_kmean_ann::fused_kmean_ann_cuda_precomputed(
            offsets,
            cluster_ids,
            cluster_length,
            item_codes,
            q_codes,
            max_tensor_size_per_row,
            mask,
            divisor_for_int8,
            prep.warp_rounded_length_cumsum,
            prep.remaining_length_cumsum,
            prep.warp_size_cumsum);
  }

  // The epilogue of retrieve.ops.official.official_probe_score: top-k on the raw scores (pads at
  // the dtype's minimum; two-level from kTopkMinSlots), then dequantize only the [B, k] winners.
  const bool int32_path = raw.scalar_type() == at::kInt;
  const int64_t actual_k = std::min<int64_t>(k, raw.size(1));
  auto [top_raw, top_slot] = raw_topk(raw, idx, actual_k);
  Tensor top_idx = idx.gather(1, top_slot.clamp_min(0));
  Tensor top_valid = top_slot.ge(0).logical_and(top_idx.ge(0));
  Tensor pos = top_idx.to(at::kLong).clamp_min(0);
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

// The prep kernel's outputs as an op, for its gate (codes / scales against quantize_int8, the
// rest against round_cluster_to_warp + cumsum).
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>
probe_prep_cuda(
    const Tensor& query,
    const Tensor& probe_ids,
    const Tensor& cluster_offsets,
    const Tensor& cluster_sizes) {
  at::cuda::CUDAGuard device_guard(query.device());
  Prep p = probe_prep(query, probe_ids, cluster_offsets, cluster_sizes);
  return {
      p.codes,
      p.scales,
      p.cluster_ids,
      p.cluster_length,
      p.selected_offsets,
      p.warp_size,
      p.warp_rounded_length_cumsum,
      p.remaining_length_cumsum,
      p.warp_size_cumsum};
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl("search", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(search_cuda)));
  m.impl(
      "probe_prep", torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(probe_prep_cuda)));
}

} // namespace st::ops::fork_search
