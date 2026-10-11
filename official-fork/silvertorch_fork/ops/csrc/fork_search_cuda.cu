// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C5-C9).

// stfork::search: the whole SilverTorch phase 2 + 3 forward in one op call: one prep launch
// (query int8, probed clusters, warp cumsums), the filter (bloom over prepared plans or the exact
// clauses, evaluated into the scorer's payloads over the probed docs only; or a full-N mask),
// upstream's scorer, top-k on the raw scores and the [B, k] epilogue. No host sync, no
// data-dependent allocation.

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

#include "fork_filters.cuh"
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

// official_probe_score's [B, k] epilogue in one launch: a winner is valid when its slot is in the
// row and the scorer wrote it (index >= 0); its id is sort_perm[index], its score
// (fp32(dot) * q_scale) * global_scale, or fp32(dot / divisor) * ((q_scale * divisor) *
// global_scale) on the fp16 path, the same fp32 multiplies in the same order; else -1 / -inf,
// as for the k - actual_k pad columns.
template <typename T>
__global__ void epilogue_kernel(
    const T* __restrict__ top_raw,
    const int64_t* __restrict__ top_slot,
    int64_t actual_k,
    const int32_t* __restrict__ idx,
    int64_t width,
    const int64_t* __restrict__ sort_perm,
    const float* __restrict__ q_scales,
    const float* __restrict__ global_scale,
    int64_t divisor,
    int64_t rows,
    int64_t k,
    int64_t* __restrict__ ids,
    float* __restrict__ scores) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * k) {
    return;
  }
  const int64_t row = i / k;
  const int64_t j = i % k;
  int64_t id = -1;
  float score = -INFINITY;
  if (j < actual_k) {
    const int64_t slot = top_slot[row * actual_k + j];
    const int32_t pos = slot >= 0 ? idx[row * width + slot] : -1;
    if (pos >= 0) {
      id = sort_perm[pos];
      const float dot = static_cast<float>(top_raw[row * actual_k + j]);
      score = divisor == -1
          ? __fmul_rn(__fmul_rn(dot, q_scales[row]), *global_scale)
          : __fmul_rn(
                dot,
                __fmul_rn(
                    __fmul_rn(q_scales[row], static_cast<float>(divisor)),
                    *global_scale));
    }
  }
  ids[i] = id;
  scores[i] = score;
}

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
    if (int32_path) {
      // the scorer pre-fills every slot it does not write with INT_MIN, the floor itself
      return at::topk(raw, k, /*dim=*/1);
    }
    // fp16 pre-fills -65504, a finite score; pads rank below it at -inf
    return at::topk(
        at::where(idx.ge(0), raw, at::Scalar(-std::numeric_limits<double>::infinity())),
        k,
        /*dim=*/1);
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

// C9: the exact filter over the probed docs only (our clause_subset_match: a clause passes when
// any of the doc's values equals the query's, negated for a reverse clause; an inactive clause,
// query value -1, always passes), evaluated into the scorer's payloads like C8's bloom.
using st::ops::fork_filters::exact_pass;
using st::ops::fused_kmean_ann::RemainingPayload;
using st::ops::fused_kmean_ann::WarpPayload;

// One warp per full-warp payload: lane l tests doc start + l; the ballot, bit-reversed, is the
// payload's high-first 32-bit mask (doc start at bit 31, as get_bit_32_bit_mask reads it). With a
// full-N `pre_mask` (C13: the bloom's, high-first words) only its passing docs are tested.
__device__ __forceinline__ bool pre_mask_bit(
    const uint64_t* __restrict__ pre_mask,
    int64_t pre_mask_columns,
    int64_t row,
    int64_t doc) {
  return pre_mask == nullptr ||
      ((pre_mask[row * pre_mask_columns + doc / 64] >> (63 - doc % 64)) & 1);
}

__global__ void exact_warp_payload_kernel(
    const int64_t* __restrict__ attrs,
    int64_t n_clauses,
    int64_t a_max,
    const bool* __restrict__ reverse,
    const int64_t* __restrict__ query_attrs,
    const uint64_t* __restrict__ pre_mask,
    int64_t pre_mask_columns,
    const int64_t* __restrict__ cluster_offsets,
    const int64_t* __restrict__ cluster_ids,
    int64_t clusters_per_row,
    int32_t n_clusters,
    const int32_t* __restrict__ warp_rounded_length_cumsum,
    const int32_t* __restrict__ warp_size_cumsum,
    int64_t max_tensor_size_per_row,
    WarpPayload* __restrict__ payloads) {
  const int64_t total = warp_size_cumsum[n_clusters - 1];
  const int lane = threadIdx.x % 32;
  for (int64_t w = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       w < total;
       w += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
    int32_t lo = 0;
    int32_t hi = n_clusters - 1;
    while (lo < hi) {
      int32_t mid = (lo + hi) >> 1;
      if (warp_size_cumsum[mid] > w) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    const int32_t ci = lo;
    const int32_t row = static_cast<int32_t>(ci / clusters_per_row);
    const int64_t col = ci % clusters_per_row;
    const int32_t doc_offset = static_cast<int32_t>(
        (w - (ci == 0 ? 0 : warp_size_cumsum[ci - 1])) * 32);
    const int32_t doc_start =
        static_cast<int32_t>(cluster_offsets[cluster_ids[ci]]) + doc_offset;
    const int64_t doc = doc_start + lane;
    const bool pass = pre_mask_bit(pre_mask, pre_mask_columns, row, doc) &&
        exact_pass(attrs, n_clauses, a_max, reverse, query_attrs + row * n_clauses, doc);
    const uint32_t mask = __brev(__ballot_sync(0xffffffff, pass));
    if (lane == 0) {
      WarpPayload p;
      p.doc_start_index = doc_start;
      p.write_index = static_cast<int32_t>(
          row * max_tensor_size_per_row +
          (col == 0 ? 0 : warp_rounded_length_cumsum[ci - 1]) + doc_offset);
      p.cluster_ids_row = row;
      p.filtering_mask = mask;
      reinterpret_cast<int4*>(payloads)[w] = p.payload;
    }
  }
}

// One warp per probed cluster: lane j writes the payload of the cluster's tail doc j (< 32).
__global__ void exact_remaining_payload_kernel(
    const int64_t* __restrict__ attrs,
    int64_t n_clauses,
    int64_t a_max,
    const bool* __restrict__ reverse,
    const int64_t* __restrict__ query_attrs,
    const uint64_t* __restrict__ pre_mask,
    int64_t pre_mask_columns,
    const int64_t* __restrict__ cluster_offsets,
    const int64_t* __restrict__ cluster_ids,
    const int64_t* __restrict__ cluster_length,
    int64_t clusters_per_row,
    int64_t n_clusters,
    const int32_t* __restrict__ warp_rounded_length_cumsum,
    const int32_t* __restrict__ remaining_length_cumsum,
    int64_t max_tensor_size_per_row,
    RemainingPayload* __restrict__ payloads) {
  const int lane = threadIdx.x % 32;
  for (int64_t i = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
       i < n_clusters;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
    const int32_t row = static_cast<int32_t>(i / clusters_per_row);
    const int32_t length = static_cast<int32_t>(cluster_length[i]);
    const int32_t handled = length / 32 * 32;
    if (lane >= length - handled) {
      continue;
    }
    const int32_t doc = static_cast<int32_t>(cluster_offsets[cluster_ids[i]]) + handled + lane;
    const int32_t out_offset = i == 0 ? 0 : remaining_length_cumsum[i - 1];
    const int32_t row_first =
        row == 0 ? 0 : remaining_length_cumsum[row * clusters_per_row - 1];
    RemainingPayload p;
    p.doc_index = doc;
    p.cluster_ids_row = row;
    p.write_index = static_cast<int32_t>(row * max_tensor_size_per_row) +
        warp_rounded_length_cumsum[row * clusters_per_row + clusters_per_row - 1] +
        (out_offset - row_first) + lane;
    p.filtering_mask = pre_mask_bit(pre_mask, pre_mask_columns, row, doc) &&
        exact_pass(attrs, n_clauses, a_max, reverse, query_attrs + row * n_clauses, doc);
    reinterpret_cast<int4*>(payloads)[out_offset + lane] = p.payload;
  }
}

std::tuple<Tensor, int64_t, Tensor, int64_t> exact_payloads(
    const Tensor& item_clause_attrs,
    const Tensor& clause_is_reverse,
    const Tensor& query_clause_attrs,
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_length,
    const Tensor& warp_rounded_length_cumsum,
    const Tensor& remaining_length_cumsum,
    const Tensor& warp_size_cumsum,
    int64_t max_tensor_size_per_row,
    const std::optional<Tensor>& pre_mask) {
  TORCH_CHECK(item_clause_attrs.dim() == 3 && item_clause_attrs.scalar_type() == at::kLong);
  TORCH_CHECK(query_clause_attrs.dim() == 2 && query_clause_attrs.scalar_type() == at::kLong);
  TORCH_CHECK(clause_is_reverse.scalar_type() == at::kBool);
  TORCH_CHECK(query_clause_attrs.size(1) == item_clause_attrs.size(1));
  const int64_t rows = cluster_ids.size(0);
  const int64_t per_row = cluster_ids.size(1);
  const int64_t n_clusters = rows * per_row;
  const int64_t warps = n_clusters == 0 ? 0 : rows * (max_tensor_size_per_row / 32);
  const int64_t remainders =
      n_clusters == 0 ? 0 : rows * std::min(31 * per_row, max_tensor_size_per_row);
  auto bytes = cluster_ids.options().dtype(at::kChar);
  Tensor warp_payloads =
      at::empty({warps * static_cast<int64_t>(sizeof(WarpPayload))}, bytes);
  Tensor remaining_payloads =
      at::empty({remainders * static_cast<int64_t>(sizeof(RemainingPayload))}, bytes);
  if (n_clusters == 0) {
    return {warp_payloads, warps, remaining_payloads, remainders};
  }
  Tensor attrs = item_clause_attrs.contiguous();
  Tensor rev = clause_is_reverse.contiguous();
  Tensor q = query_clause_attrs.contiguous();
  const uint64_t* pre = pre_mask
      ? reinterpret_cast<const uint64_t*>(pre_mask->const_data_ptr<int64_t>())
      : nullptr;
  const int64_t pre_columns = pre_mask ? pre_mask->size(1) : 0;
  const int64_t max_grid =
      128L * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream();
  exact_warp_payload_kernel<<<std::min(max_grid, (warps * 32 + 255) / 256), 256, 0, stream>>>(
      attrs.const_data_ptr<int64_t>(),
      attrs.size(1),
      attrs.size(2),
      rev.const_data_ptr<bool>(),
      q.const_data_ptr<int64_t>(),
      pre,
      pre_columns,
      cluster_offsets.const_data_ptr<int64_t>(),
      cluster_ids.const_data_ptr<int64_t>(),
      per_row,
      static_cast<int32_t>(n_clusters),
      warp_rounded_length_cumsum.const_data_ptr<int32_t>(),
      warp_size_cumsum.const_data_ptr<int32_t>(),
      max_tensor_size_per_row,
      reinterpret_cast<WarpPayload*>(warp_payloads.mutable_data_ptr<int8_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  exact_remaining_payload_kernel<<<
      std::min(max_grid, (n_clusters * 32 + 255) / 256),
      256,
      0,
      stream>>>(
      attrs.const_data_ptr<int64_t>(),
      attrs.size(1),
      attrs.size(2),
      rev.const_data_ptr<bool>(),
      q.const_data_ptr<int64_t>(),
      pre,
      pre_columns,
      cluster_offsets.const_data_ptr<int64_t>(),
      cluster_ids.const_data_ptr<int64_t>(),
      cluster_length.const_data_ptr<int64_t>(),
      per_row,
      n_clusters,
      warp_rounded_length_cumsum.const_data_ptr<int32_t>(),
      remaining_length_cumsum.const_data_ptr<int32_t>(),
      max_tensor_size_per_row,
      reinterpret_cast<RemainingPayload*>(remaining_payloads.mutable_data_ptr<int8_t>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {warp_payloads, warps, remaining_payloads, remainders};
}

const int32_t* last_element(const Tensor& inclusive_cumsum) {
  return inclusive_cumsum.numel() == 0
      ? nullptr
      : inclusive_cumsum.const_data_ptr<int32_t>() + inclusive_cumsum.numel() - 1;
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
    bool bloom_partial,
    const std::optional<Tensor>& item_clause_attrs,
    const std::optional<Tensor>& clause_is_reverse,
    const std::optional<Tensor>& query_clause_attrs) {
  TORCH_CHECK(item_codes.scalar_type() == at::kChar && item_codes.size(1) == query.size(1));
  // query_clause_attrs alone is the exact filter (C9); with plans, the bloom's exact recheck (C13)
  const bool exact = query_clause_attrs.has_value();
  TORCH_CHECK(
      !(filtering_bit_mask.has_value() && (plans.has_value() || exact)),
      "search takes one filter: plans (+ query_clause_attrs to recheck), query_clause_attrs or "
      "filtering_bit_mask");
  TORCH_CHECK(!exact || (item_clause_attrs.has_value() && clause_is_reverse.has_value()));
  at::cuda::CUDAGuard device_guard(query.device());

  Tensor offsets = cluster_offsets.to(at::kLong).contiguous();
  Prep prep = probe_prep(query, probe_ids, offsets, cluster_sizes.to(at::kLong));
  const Tensor& q_codes = prep.codes;
  const Tensor& q_scales = prep.scales;
  const Tensor& cluster_ids = prep.cluster_ids;
  const Tensor& cluster_length = prep.cluster_length;
  const int64_t width = (max_tensor_size_per_row + 31) / 32 * 32;

  // The filter, into the scorer's payloads (bloom partial, exact) or as a full-N mask.
  std::optional<std::tuple<Tensor, int64_t, Tensor, int64_t>> payloads;
  std::optional<Tensor> mask = filtering_bit_mask;
  if (plans.has_value()) {
    TORCH_CHECK(bloom_index.has_value() && bloom_bundle_b_offsets.has_value());
    if (bloom_partial) {
      std::optional<std::tuple<Tensor, Tensor, Tensor>> recheck;
      if (exact) {
        recheck = std::make_tuple(
            item_clause_attrs->contiguous(),
            clause_is_reverse->contiguous(),
            query_clause_attrs->contiguous());
      }
      payloads = st::ops::bloom_search::bloom_partial_payloads_prepared_cuda(
          *bloom_index,
          *bloom_bundle_b_offsets,
          *plans,
          plan_count,
          max_stack_size,
          query_plan_index,
          bloom_k,
          hash_k,
          offsets,
          cluster_ids,
          cluster_length,
          prep.warp_rounded_length_cumsum,
          prep.remaining_length_cumsum,
          prep.warp_size_cumsum,
          width,
          recheck);
    } else {
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
  }
  if (!payloads && exact) {
    // the exact test; on the full bloom path its mask pre-filters (C13's recheck)
    payloads = exact_payloads(
        *item_clause_attrs,
        *clause_is_reverse,
        *query_clause_attrs,
        offsets,
        cluster_ids,
        cluster_length,
        prep.warp_rounded_length_cumsum,
        prep.remaining_length_cumsum,
        prep.warp_size_cumsum,
        width,
        mask);
  }

  Tensor raw;
  Tensor idx;
  if (payloads) {
    const auto& [warp_payloads, warps, remaining_payloads, remainders] = *payloads;
    std::tie(raw, idx) = st::ops::fused_kmean_ann::fused_kmean_ann_run_payloads(
        cluster_ids,
        item_codes,
        q_codes,
        width,
        /*invalid_index_value=*/-1,
        divisor_for_int8,
        /*has_filtering_bitmask=*/true,
        warp_payloads,
        warps,
        last_element(prep.warp_size_cumsum),
        remaining_payloads,
        remainders,
        last_element(prep.remaining_length_cumsum),
        std::nullopt,
        /*query_on_shared_mem=*/false);
  } else {
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
  const int64_t rows = raw.size(0);
  const int64_t actual_k = std::min<int64_t>(k, raw.size(1));
  auto [top_raw, top_slot] = raw_topk(raw, idx, actual_k);
  top_raw = top_raw.contiguous();
  top_slot = top_slot.contiguous();
  Tensor ids = at::empty({rows, k}, sort_perm.options());
  Tensor scores = at::empty({rows, k}, query.options());
  Tensor gs = global_scale.to(at::kFloat).contiguous();
  if (rows * k > 0) {
    const int64_t blocks = (rows * k + 255) / 256;
    auto launch = [&](auto zero) {
      using T = decltype(zero);
      epilogue_kernel<T><<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
          reinterpret_cast<const T*>(top_raw.const_data_ptr()),
          top_slot.const_data_ptr<int64_t>(),
          actual_k,
          idx.const_data_ptr<int32_t>(),
          idx.size(1),
          sort_perm.const_data_ptr<int64_t>(),
          q_scales.const_data_ptr<float>(),
          gs.const_data_ptr<float>(),
          divisor_for_int8,
          rows,
          k,
          ids.mutable_data_ptr<int64_t>(),
          scores.mutable_data_ptr<float>());
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if (int32_path) {
      launch(int32_t{0});
    } else {
      launch(c10::Half{0.0f});
    }
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
