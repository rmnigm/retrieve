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

#include <cstdlib>
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

constexpr int kPrepThreads = 1024;
// Average full warps per probed cluster from which the payload kernels run a warp per cluster.
constexpr int64_t kPerClusterWarps = 40;
// ... and only with many clusters (K fix 14: per-warp won at 1536 and 4096 clusters, per-cluster at 16384).
constexpr int64_t kPerClusterMinClusters = 8192;


struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

// K fix 2: the top-k as a radix select per row chunk in one CTA, its [B, k] epilogue fused. A
// score's key is a uint32 that orders like the score (int32: sign bit flipped; fp16: the IEEE
// order trick on the bits; a slot the scorer never wrote ranks lowest, INT_MIN on int32, -inf
// on fp16). Four 8-bit histogram passes find the k-th largest key T; keys above T are taken,
// and keys equal to T in index order until k (as torch's top-k), so the choice is deterministic
// (a graph replay picks the ids eager picks). The k winners are sorted by (key desc, slot asc) with a bitonic
// sort. A row wider than kSelectChunk slots is selected per chunk first (stage 1, its k winners
// per chunk in index order, unsorted), then once over the chunks' winners (stage 2): the row's
// top-k lies in the union of its chunks' top-ks.
constexpr int kSelectThreads = 512;
constexpr int kSelectItems = 16; // keys a thread holds in registers
constexpr int64_t kSelectChunk = kSelectThreads * kSelectItems;
constexpr int64_t kSelectMaxK = 1024;
// the two-level block size and thresholds (C7, K fix 9)
constexpr int64_t kTopkBlock = 256;
constexpr int64_t kTopkMinSlots = int64_t{1} << 23;
constexpr int64_t kTopkMinRatio = 8;

__device__ __forceinline__ uint32_t score_key(int32_t v) {
  return static_cast<uint32_t>(v) ^ 0x80000000u;
}

__device__ __forceinline__ uint32_t score_key(c10::Half v) {
  const uint16_t h = v.x;
  return static_cast<uint32_t>((h & 0x8000) ? static_cast<uint16_t>(~h) : (h | 0x8000)) << 16;
}

// (key desc, slot asc) in one uint64 that sorts descending; 0 is below every real entry.
__device__ __forceinline__ uint64_t composite(uint32_t key, uint32_t slot) {
  return (static_cast<uint64_t>(key) << 32) | (0xffffffffu - slot);
}

__device__ __forceinline__ uint32_t composite_slot(uint64_t c) {
  return 0xffffffffu - static_cast<uint32_t>(c);
}

// Raw scores of row `row`, slots [first, first + n): the key, -inf's on fp16 for unwritten slots.
// `limit[row]` (K fix 10): the row's written width; slots past it are empty (lowest key).
template <typename T>
struct RawKeys {
  const T* raw;
  const int32_t* idx;
  int64_t width;
  const int32_t* limit;
  __device__ __forceinline__ uint32_t key(int64_t row, int64_t slot) const {
    if (slot >= limit[row]) {
      return 0;
    }
    if constexpr (std::is_same_v<T, int32_t>) {
      return score_key(raw[row * width + slot]);
    } else {
      return idx[row * width + slot] >= 0 ? score_key(raw[row * width + slot])
                                          : 0x03ffu << 16;
    }
  }
  __device__ __forceinline__ uint32_t slot(int64_t /*row*/, int64_t s) const {
    return static_cast<uint32_t>(s);
  }
  __device__ __forceinline__ int64_t row_end(int64_t row) const {
    return limit[row];
  }
};

// Stage-1 winners [B, chunks * k] (composites) as the stage-2 input.
struct CompositeKeys {
  const uint64_t* cand;
  int64_t per_row;
  __device__ __forceinline__ uint32_t key(int64_t row, int64_t i) const {
    return static_cast<uint32_t>(cand[row * per_row + i] >> 32);
  }
  __device__ __forceinline__ uint32_t slot(int64_t row, int64_t i) const {
    return composite_slot(cand[row * per_row + i]);
  }
  __device__ __forceinline__ int64_t row_end(int64_t /*row*/) const {
    return per_row;
  }
};

struct SelectShared {
  uint32_t hist[256];
  uint64_t best[kSelectMaxK];
  uint32_t digit;
  int64_t need;
  int64_t taken;
};

// The k (<= kSelectMaxK, <= n) largest of entries [first, first + n) of `row`, sorted into
// sh.best[0, k) by (key desc, slot asc); sh.best[k, pow2) holds 0.
// Entry i of the CTA's range is read as key(i); thread t's it-th entry is it * kSelectThreads + t
// (coalesced). With n <= kSelectChunk (CACHED) a thread holds its kSelectItems keys in registers
// after one load, else every pass reloads them.
// SORT: best[0, k) sorted by (key desc, slot asc). Without it (stage 1, CACHED only) best[0, k)
// holds the keys above the threshold in index order, then the tied ones in index order: a
// deterministic layout whose index order, chunk after chunk, keeps the row's index order for any
// stage-2 tie (a chunk's entries equal to a stage-2 threshold are all in one of its two parts).
template <bool CACHED, bool SORT, typename KEYS, int ITEMS = kSelectItems>
__device__ void select_topk(
    const KEYS& keys,
    int64_t row,
    int64_t first,
    int64_t n,
    int64_t k,
    SelectShared& sh) {
  using Scan = cub::BlockScan<int32_t, kSelectThreads>;
  __shared__ typename Scan::TempStorage scan;
  const int tid = threadIdx.x;
  const int64_t iters = (n + kSelectThreads - 1) / kSelectThreads;
  uint32_t cache[CACHED ? ITEMS : 1];
  // entry index of a thread's it-th key
  auto index_of = [&](int64_t it) -> int64_t {
    return it * kSelectThreads + tid;
  };
  if constexpr (CACHED) {
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      const int64_t i = index_of(it);
      cache[it] = i < n ? keys.key(row, first + i) : 0;
    }
  }
  auto key_at = [&](int64_t it) -> uint32_t {
    if constexpr (CACHED) {
      return cache[it];
    } else {
      const int64_t i = index_of(it);
      return i < n ? keys.key(row, first + i) : 0;
    }
  };
  uint32_t prefix = 0;
  uint32_t mask = 0;
  if (tid == 0) {
    sh.need = k;
  }
  for (int shift = 24; shift >= 0; shift -= 8) {
    for (int d = tid; d < 256; d += kSelectThreads) {
      sh.hist[d] = 0;
    }
    __syncthreads();
    // scores share their top digits, so lanes with one digit add once (warp-aggregated)
#pragma unroll
    for (int64_t it = 0; it < (CACHED ? ITEMS : iters); ++it) {
      if (it >= iters) {
        break; // uniform: iters is the CTA's
      }
      const int64_t i = index_of(it);
      const uint32_t key = key_at(it);
      const bool in = i < n && (key & mask) == prefix;
      const uint32_t digit = in ? (key >> shift) & 0xff : 256;
      const uint32_t peers = __match_any_sync(0xffffffff, digit);
      if (in && (__ffs(peers) - 1) == static_cast<int>(threadIdx.x % 32)) {
        atomicAdd(&sh.hist[digit], static_cast<uint32_t>(__popc(peers)));
      }
    }
    __syncthreads();
    if (tid < 32) {
      // lane l owns bins [255 - 8l - 7, 255 - 8l]; the first lane (from the top) whose
      // cumulative count reaches `need` scans its own bins
      uint32_t own = 0;
      for (int b = 0; b < 8; ++b) {
        own += sh.hist[255 - 8 * tid - b];
      }
      uint32_t cum = own;
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t up = __shfl_up_sync(0xffffffff, cum, off);
        if (tid >= off) {
          cum += up;
        }
      }
      const int64_t need = sh.need;
      const uint32_t hit = __ballot_sync(0xffffffff, cum >= need);
      const int lane = hit ? __ffs(hit) - 1 : 31;
      if (tid == lane) {
        int64_t above = cum - own;
        int d = 255 - 8 * tid;
        for (int b = 0; b < 7; ++b, --d) {
          if (above + sh.hist[d] >= need) {
            break;
          }
          above += sh.hist[d];
        }
        sh.digit = d;
        sh.need = need - above;
      }
    }
    __syncthreads();
    prefix |= sh.digit << shift;
    mask |= 0xffu << shift;
  }
  const uint32_t threshold = prefix;
  const int64_t ties = sh.need; // how many entries equal to the threshold to take
  int64_t pow2 = 1;
  while (pow2 < k) {
    pow2 <<= 1;
  }
  for (int64_t j = tid; j < pow2; j += kSelectThreads) {
    sh.best[j] = 0;
  }
  if (tid == 0) {
    sh.taken = 0;
  }
  __syncthreads();
  // Collect: keys above the threshold anywhere in best[0, k - ties) (the sort orders them);
  // keys equal to it, `ties` of them, in index order (what torch's top-k takes; deterministic,
  // so graph and eager agree). Index order is slab-major (slab it = entries [it * T, (it + 1) * T)):
  // when CACHED each warp's per-slab counts go to shared memory, one warp prefix-sums them in
  // slab-major order, and a lane's rank adds its rank in the warp's ballot; otherwise one scan
  // per slab.
  auto take = [&](int64_t it, int32_t& eq_rank) {
    const int64_t i = index_of(it);
    if (i >= n) {
      return;
    }
    const uint32_t key = key_at(it);
    if (key > threshold) {
      const int64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(&sh.taken), 1ull);
      sh.best[pos] = composite(key, keys.slot(row, first + i));
    } else if (key == threshold) {
      if (eq_rank < ties) {
        sh.best[(k - ties) + eq_rank] = composite(key, keys.slot(row, first + i));
      }
      ++eq_rank;
    }
  };
  if constexpr (CACHED) {
    constexpr int kWarps = kSelectThreads / 32;
    __shared__ int32_t slab_warp[ITEMS * kWarps];
    const int lane = tid % 32;
    const int warp = tid / 32;
    const uint32_t below = (1u << lane) - 1;
    __shared__ int32_t slab_warp_gt[ITEMS * kWarps];
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      const uint32_t eq = __ballot_sync(0xffffffff, index_of(it) < n && cache[it] == threshold);
      const uint32_t gt = __ballot_sync(0xffffffff, index_of(it) < n && cache[it] > threshold);
      if (lane == 0) {
        slab_warp[it * kWarps + warp] = __popc(eq);
        slab_warp_gt[it * kWarps + warp] = __popc(gt);
      }
    }
    __syncthreads();
    if (tid < 32) {
      constexpr int kPer = ITEMS * kWarps / 32;
      int32_t local[kPer];
      int32_t sum = 0;
#pragma unroll
      for (int e = 0; e < kPer; ++e) {
        local[e] = slab_warp_gt[tid * kPer + e];
        sum += local[e];
      }
      int32_t inc = sum;
      for (int off = 1; off < 32; off <<= 1) {
        const int32_t up = __shfl_up_sync(0xffffffff, inc, off);
        if (tid >= off) {
          inc += up;
        }
      }
      int32_t run = inc - sum;
#pragma unroll
      for (int e = 0; e < kPer; ++e) {
        slab_warp_gt[tid * kPer + e] = run;
        run += local[e];
      }
    }
    if (tid >= 32 && tid < 64) {
      const int t = tid - 32;
      // exclusive prefix over the ITEMS * kWarps counts, slab-major
      constexpr int kPer = ITEMS * kWarps / 32;
      int32_t local[kPer];
      int32_t sum = 0;
#pragma unroll
      for (int e = 0; e < kPer; ++e) {
        local[e] = slab_warp[t * kPer + e];
        sum += local[e];
      }
      int32_t inc = sum;
      for (int off = 1; off < 32; off <<= 1) {
        const int32_t up = __shfl_up_sync(0xffffffff, inc, off);
        if (t >= off) {
          inc += up;
        }
      }
      int32_t run = inc - sum;
#pragma unroll
      for (int e = 0; e < kPer; ++e) {
        slab_warp[t * kPer + e] = run;
        run += local[e];
      }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      const int64_t i = index_of(it);
      const uint32_t eq = __ballot_sync(0xffffffff, i < n && cache[it] == threshold);
      const uint32_t gt = __ballot_sync(0xffffffff, i < n && cache[it] > threshold);
      if (i >= n) {
        continue;
      }
      if ((gt >> lane) & 1) {
        const int32_t r = slab_warp_gt[it * kWarps + warp] + __popc(gt & below);
        sh.best[r] = composite(cache[it], keys.slot(row, first + i));
      } else if ((eq >> lane) & 1) {
        const int32_t r = slab_warp[it * kWarps + warp] + __popc(eq & below);
        if (r < ties) {
          sh.best[(k - ties) + r] = composite(cache[it], keys.slot(row, first + i));
        }
      }
    }
  } else {
    int64_t eq_seen = 0;
    for (int64_t it = 0; it < iters; ++it) {
      const int32_t is_eq = (index_of(it) < n && key_at(it) == threshold) ? 1 : 0;
      int32_t eq_rank;
      int32_t eq_total;
      Scan(scan).ExclusiveSum(is_eq, eq_rank, eq_total);
      eq_rank += static_cast<int32_t>(eq_seen);
      take(it, eq_rank);
      eq_seen += eq_total;
      __syncthreads();
    }
  }
  __syncthreads();
  if constexpr (!SORT) {
    return;
  }
  // bitonic sort of best[0, pow2), descending. Thread t holds entries j = r * T + t: a stride
  // below 32 pairs lanes of one warp (shuffles), one of T or more pairs a thread's own registers,
  // and only strides in [32, T) go through shared memory.
  constexpr int kMaxPer = kSelectMaxK / kSelectThreads;
  const int per = pow2 >= kSelectThreads ? static_cast<int>(pow2 / kSelectThreads) : 1;
  uint64_t v[kMaxPer];
#pragma unroll
  for (int r = 0; r < kMaxPer; ++r) {
    const int64_t j = r * kSelectThreads + tid;
    v[r] = (r < per && j < pow2) ? sh.best[j] : 0;
  }
  for (int64_t size = 2; size <= pow2; size <<= 1) {
    for (int64_t stride = size >> 1; stride > 0; stride >>= 1) {
      if (stride >= kSelectThreads) {
        const int rs = static_cast<int>(stride / kSelectThreads);
#pragma unroll
        for (int r = 0; r < kMaxPer; ++r) {
          const int64_t j = r * kSelectThreads + tid;
          if (r < per && (r & rs) == 0) {
            const bool descending = (j & size) == 0;
            const uint64_t a = v[r];
            const uint64_t b = v[r + rs];
            if ((a < b) == descending) {
              v[r] = b;
              v[r + rs] = a;
            }
          }
        }
      } else if (stride >= 32) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kMaxPer; ++r) {
          if (r < per) {
            sh.best[r * kSelectThreads + tid] = v[r];
          }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kMaxPer; ++r) {
          const int64_t j = r * kSelectThreads + tid;
          if (r < per && j < pow2) {
            const uint64_t b = sh.best[j ^ stride];
            const bool descending = (j & size) == 0;
            const bool low = (j & stride) == 0;
            v[r] = (low == descending) == (v[r] < b) ? b : v[r];
          }
        }
      } else {
#pragma unroll
        for (int r = 0; r < kMaxPer; ++r) {
          const int64_t j = r * kSelectThreads + tid;
          const uint64_t b = __shfl_xor_sync(0xffffffff, v[r], static_cast<int>(stride));
          if (r < per && j < pow2) {
            const bool descending = (j & size) == 0;
            const bool low = (j & stride) == 0;
            v[r] = (low == descending) == (v[r] < b) ? b : v[r];
          }
        }
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int r = 0; r < kMaxPer; ++r) {
    const int64_t j = r * kSelectThreads + tid;
    if (r < per && j < pow2) {
      sh.best[j] = v[r];
    }
  }
  __syncthreads();
}

// Stage 1: one CTA per (row, chunk) of the row's n_row entries, its k winners to
// cand[row, chunk * k + j]; repeated over the winners while they exceed one chunk.
template <typename T>
__global__ void __launch_bounds__(kSelectThreads, 2) select_chunk_kernel(
    T keys,
    int64_t n_row,
    int64_t chunks,
    int64_t k,
    uint64_t* __restrict__ cand) {
  __shared__ SelectShared sh;
  const int64_t row = blockIdx.x / chunks;
  const int64_t chunk = blockIdx.x % chunks;
  const int64_t first = chunk * kSelectChunk;
  const int64_t rest = n_row - first;
  const int64_t n = rest < kSelectChunk ? rest : kSelectChunk;
  const int64_t kk = std::min(k, n);
  if (first >= keys.row_end(row)) { // all past the row's written width: no winners
    for (int64_t j = threadIdx.x; j < k; j += kSelectThreads) {
      cand[row * chunks * k + chunk * k + j] = 0;
    }
    return;
  }
  select_topk<true, false>(keys, row, first, n, kk, sh);
  for (int64_t j = threadIdx.x; j < k; j += kSelectThreads) {
    cand[row * chunks * k + chunk * k + j] = j < kk ? sh.best[j] : 0;
  }
}

// K fix 9: very wide rows (>= kTopkMinRatio blocks of kTopkBlock slots per top-k slot) are cut
// to the slots of their k best blocks first (the block maxima folded by the scorer, K fix 12) (C7's two-level argument: a block holding a slot above
// the k-th score has a larger max than any block without one). Blocks tied at the k-th max are
// taken in index order and the chosen blocks gathered in ascending order, so the earliest tied
// slots stay candidates and the tie choice is still torch's.
struct ArrayKeys {
  const uint32_t* keys;
  int64_t per_row;
  __device__ __forceinline__ uint32_t key(int64_t row, int64_t i) const {
    return keys[row * per_row + i];
  }
  __device__ __forceinline__ uint32_t slot(int64_t /*row*/, int64_t i) const {
    return static_cast<uint32_t>(i);
  }
};

// One CTA per row: the k best blocks (ties in index order), written in ascending block order.
__global__ void __launch_bounds__(kSelectThreads) select_blocks_kernel(
    ArrayKeys keys,
    int64_t n_blocks,
    int64_t k,
    int64_t* __restrict__ blocks) {
  __shared__ SelectShared sh;
  extern __shared__ uint32_t chosen[]; // one bit per block
  const int64_t row = blockIdx.x;
  const int64_t words = (n_blocks + 31) / 32;
  for (int64_t i = threadIdx.x; i < words; i += kSelectThreads) {
    chosen[i] = 0;
  }
  if (n_blocks <= kSelectChunk) {
    select_topk<true, false>(keys, row, 0, n_blocks, k, sh);
  } else {
    select_topk<false, true>(keys, row, 0, n_blocks, k, sh);
  }
  for (int64_t j = threadIdx.x; j < k; j += kSelectThreads) {
    const uint32_t b = composite_slot(sh.best[j]);
    atomicOr(&chosen[b / 32], 1u << (b % 32));
  }
  __syncthreads();
  // compaction in index order: thread t owns words [t * per, (t + 1) * per)
  using Scan = cub::BlockScan<int32_t, kSelectThreads>;
  __shared__ typename Scan::TempStorage scan;
  const int64_t per = (words + kSelectThreads - 1) / kSelectThreads;
  int32_t mine = 0;
  for (int64_t i = threadIdx.x * per; i < (threadIdx.x + 1) * per && i < words; ++i) {
    mine += __popc(chosen[i]);
  }
  int32_t pos;
  Scan(scan).ExclusiveSum(mine, pos);
  for (int64_t i = threadIdx.x * per; i < (threadIdx.x + 1) * per && i < words; ++i) {
    uint32_t bits = chosen[i];
    while (bits) {
      const int b = __ffs(bits) - 1;
      bits &= bits - 1;
      blocks[row * k + pos++] = i * 32 + b;
    }
  }
}

// The chosen blocks' slots as composites [rows, k * kTopkBlock], ascending slot order (0 past the
// row's width).
template <typename T>
__global__ void gather_blocks_kernel(
    RawKeys<T> keys,
    const int64_t* __restrict__ blocks,
    int64_t rows,
    int64_t k,
    uint64_t* __restrict__ cand) {
  const int64_t total = rows * k * kTopkBlock;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = i / (k * kTopkBlock);
    const int64_t slot = blocks[row * k + (i / kTopkBlock) % k] * kTopkBlock + i % kTopkBlock;
    cand[i] = slot < keys.width
        ? composite(keys.key(row, slot), static_cast<uint32_t>(slot))
        : 0;
  }
}

// Final stage: one CTA per row, select over KEYS, then official_probe_score's epilogue on the
// sorted winners (see epilogue_kernel).
template <typename T, typename KEYS>
__global__ void __launch_bounds__(kSelectThreads) select_epilogue_kernel(
    KEYS keys,
    int64_t n,
    const T* __restrict__ raw,
    const int32_t* __restrict__ idx,
    int64_t width,
    const int32_t* __restrict__ limit,
    const int64_t* __restrict__ sort_perm,
    const float* __restrict__ q_scales,
    const float* __restrict__ global_scale,
    int64_t divisor,
    int64_t k,
    int64_t* __restrict__ ids,
    float* __restrict__ scores) {
  __shared__ SelectShared sh;
  const int64_t row = blockIdx.x;
  const int64_t kk = std::min(k, n);
  if (n <= kSelectChunk) {
    select_topk<true, true>(keys, row, 0, n, kk, sh);
  } else if (n <= 2 * kSelectChunk) {
    select_topk<true, true, KEYS, 2 * kSelectItems>(keys, row, 0, n, kk, sh);
  } else {
    select_topk<false, true>(keys, row, 0, n, kk, sh);
  }
  for (int64_t j = threadIdx.x; j < k; j += kSelectThreads) {
    int64_t id = -1;
    float score = -INFINITY;
    if (j < kk && sh.best[j] != 0) {
      const uint32_t slot = composite_slot(sh.best[j]);
      const int32_t pos = slot < limit[row] ? idx[row * width + slot] : -1;
      if (pos >= 0) {
        id = sort_perm[pos];
        const float dot = static_cast<float>(raw[row * width + slot]);
        score = divisor == -1
            ? __fmul_rn(__fmul_rn(dot, q_scales[row]), *global_scale)
            : __fmul_rn(
                  dot,
                  __fmul_rn(
                      __fmul_rn(q_scales[row], static_cast<float>(divisor)),
                      *global_scale));
      }
    }
    ids[row * k + j] = id;
    scores[row * k + j] = score;
  }
}

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
// [-128, 127]. Blocks [B, 2B): one row each, its probed clusters (ids, lengths, offsets) and the
// scorer's warp rounding with its three cumsums (warp-rounded lengths per row, remainders and
// warp counts over all B * P), as round_cluster_to_warp + cumsum would, scanned in int64.
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
    int32_t* __restrict__ warp_size_cumsum,
    int32_t* __restrict__ row_written) {
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

  // Blocks [rows, 2 * rows): block rows + r scans row r. The row's prefix over rows < r (warp-
  // rounded lengths are per row; remainders and warp counts run over all B * P) is summed from
  // those rows' cluster lengths by this block itself, so the rows scan in parallel.
  using Sum = cub::BlockReduce<int64_t, kPrepThreads>;
  __shared__ typename Sum::TempStorage sum_tmp;
  const int64_t r = blockIdx.x - rows;
  int64_t before_rem = 0;
  int64_t before_warps = 0;
  for (int64_t i = threadIdx.x; i < r * n_probe; i += kPrepThreads) {
    const int64_t len = cluster_sizes[probe_ids[i]];
    before_rem += len % 32;
    before_warps += len / 32;
  }
  before_rem = Sum(sum_tmp).Sum(before_rem);
  __syncthreads();
  before_warps = Sum(sum_tmp).Sum(before_warps);
  if (threadIdx.x == 0) {
    carry[0] = 0;
    carry[1] = before_rem;
    carry[2] = before_warps;
  }
  __syncthreads();
  const int64_t first = r * n_probe;
  for (int64_t base = 0; base < n_probe; base += kPrepThreads) {
    const int64_t j = base + threadIdx.x;
    const int64_t i = first + j;
    int64_t rounded = 0;
    int64_t remainder = 0;
    int64_t warps = 0;
    if (j < n_probe) {
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
    if (j < n_probe) {
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
  // the slots the scorer writes in the row: its full warps' docs, then its remainders
  if (threadIdx.x == 0) {
    row_written[r] = static_cast<int32_t>(carry[0] + (carry[1] - before_rem));
  }
}

// The two-level top-k (C7; our Triton epilogue's, kernels.md § SilverTorch kernels "Top-k"):
// from kTopkMinSlots slots of [B, W], where a row has at least kTopkMinRatio whole kTopkBlock-slot
// blocks per top-k slot, the top-k runs on the slots of the row's k best blocks. Exact: every
// block holding a slot above the k-th score has a larger max than any block without one, and
// there are fewer than k of them.

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

// fork (K fix 7): the cluster of warp slot w (first i with inclusive_cumsum[i] > w), searched
// by the 32 lanes of a warp together: each step every lane tests one of 32 evenly spaced
// positions, so ~log32(n) dependent loads instead of log2(n). Every lane gets the result.
__device__ __forceinline__ int32_t warp_find_cluster(
    const int32_t* __restrict__ inclusive_cumsum,
    int32_t n,
    int64_t w,
    int lane) {
  int32_t lo = 0;
  int32_t hi = n - 1;
  while (hi - lo >= 32) {
    const int32_t step = (hi - lo + 30) / 31; // lane 31 reaches hi
    const int32_t p = min(lo + lane * step, hi);
    const uint32_t above = __ballot_sync(0xffffffff, inclusive_cumsum[p] > w);
    const int f = __ffs(above) - 1; // the last lane's position hi is always above
    hi = min(lo + f * step, hi);
    lo = f == 0 ? lo : lo + (f - 1) * step + 1;
  }
  const int32_t p = min(lo + lane, hi);
  const uint32_t above = __ballot_sync(0xffffffff, inclusive_cumsum[p] > w);
  return lo + __ffs(above) - 1;
}

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
template <bool PER_CLUSTER>
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
  // PER_CLUSTER: one warp per probed cluster looping its full warps (wide probes, many warps a
  // cluster); else one warp per full warp, its cluster found by warp_find_cluster.
  auto payload = [&](int32_t ci, int64_t w, int64_t w_first) {
      const int32_t row = static_cast<int32_t>(ci / clusters_per_row);
      const int64_t col = ci % clusters_per_row;
      const int32_t doc_offset = static_cast<int32_t>((w - w_first) * 32);
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
  };
  if constexpr (PER_CLUSTER) {
    for (int64_t c = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         c < n_clusters;
         c += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
      const int32_t ci = static_cast<int32_t>(c);
      const int64_t w_first = ci == 0 ? 0 : warp_size_cumsum[ci - 1];
      for (int64_t w = w_first; w < warp_size_cumsum[ci]; ++w) {
        payload(ci, w, w_first);
      }
    }
  } else {
    for (int64_t w = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
         w < total;
         w += static_cast<int64_t>(gridDim.x) * blockDim.x / 32) {
      const int32_t ci = warp_find_cluster(warp_size_cumsum, n_clusters, w, lane);
      payload(ci, w, ci == 0 ? 0 : warp_size_cumsum[ci - 1]);
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
  // wide probes (capacity >= kPerClusterWarps full warps a cluster): a warp per cluster
  auto launch_warps = [&](auto per_cluster) {
    constexpr bool kPerCluster = decltype(per_cluster)::value;
    const int64_t threads = (kPerCluster ? n_clusters : warps) * 32;
    exact_warp_payload_kernel<kPerCluster>
        <<<std::min(max_grid, (threads + 255) / 256), 256, 0, stream>>>(
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

  };
  if (warps >= kPerClusterWarps * n_clusters && n_clusters >= kPerClusterMinClusters) {
    launch_warps(std::true_type{});
  } else {
    launch_warps(std::false_type{});
  }
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

// C11: the scorer's variant per (D, rows). STFORK_QUERY_SHARED=0/1 overrides it (the sweep).
bool scorer_query_on_shared_mem(int64_t dim, int64_t rows) {
  static const int forced = [] {
    const char* v = std::getenv("STFORK_QUERY_SHARED");
    return v == nullptr ? -1 : std::atoi(v);
  }();
  if (forced >= 0) {
    return forced == 1;
  }
  (void)dim;
  (void)rows;
  return false;
}

// C11: the cooperative-load scorer per D (process_cluster_coop). STFORK_COOP=0/1 forces it.
bool scorer_coop(int64_t dim) {
  static const int forced = [] {
    const char* v = std::getenv("STFORK_COOP");
    return v == nullptr ? -1 : std::atoi(v);
  }();
  if (forced >= 0) {
    return forced == 1;
  }
  return dim >= 128;
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
  Tensor row_written;
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
      at::empty({rows * n_probe}, i32),
      at::empty({rows}, i32)};
  if (rows > 0) {
    probe_prep_kernel<<<
        2 * rows,
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
        p.warp_size_cumsum.mutable_data_ptr<int32_t>(),
        p.row_written.mutable_data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
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
    // the bloom over the probed clusters' columns only (C8): what the scorer reads of Meta's
    // full-N mask, so the full path's result too, without its scan of all N (K fix 17)
    TORCH_CHECK(bloom_index.has_value() && bloom_bundle_b_offsets.has_value());
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
  }
  if (!payloads && exact) {
    // the exact test; a given full-N mask pre-filters it
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

  const bool query_shared = scorer_query_on_shared_mem(item_codes.size(1), cluster_ids.size(0));
  const bool coop = scorer_coop(item_codes.size(1));
  // our select reads only each row's written slots, so the scorer need not pre-fill (K fix 10)
  const bool write_failed = k <= kSelectMaxK;
  // very wide rows go through the block prefilter (K fix 9), whose block maxima the scorer
  // folds as it writes (K fix 12)
  const int64_t rows = cluster_ids.size(0);
  const int64_t n_blocks = (width + kTopkBlock - 1) / kTopkBlock;
  const bool prefilter = write_failed && width > kSelectChunk &&
      width / kTopkBlock >= kTopkMinRatio * std::min<int64_t>(k, width);
  Tensor block_keys;
  st::ops::fused_kmean_ann::BlockKeys bk;
  if (prefilter) {
    block_keys = at::zeros({rows, n_blocks}, query.options().dtype(at::kInt));
    bk = {reinterpret_cast<uint32_t*>(block_keys.mutable_data_ptr<int32_t>()), width, n_blocks};
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
        query_shared,
        write_failed,
        bk,
        coop);
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
            prep.warp_size_cumsum,
            query_shared,
            write_failed,
            bk,
            coop);
  }

  // The epilogue of retrieve.ops.official.official_probe_score: top-k on the raw scores (pads at
  // the dtype's minimum; two-level from kTopkMinSlots), then dequantize only the [B, k] winners.
  const bool int32_path = raw.scalar_type() == at::kInt;
  const int64_t actual_k = std::min<int64_t>(k, raw.size(1));
  Tensor ids = at::empty({rows, k}, sort_perm.options());
  Tensor scores = at::empty({rows, k}, query.options());
  Tensor gs = global_scale.to(at::kFloat).contiguous();
  if (rows > 0 && k <= kSelectMaxK) {
    const int64_t w = raw.size(1);
    const int64_t chunks = (w + kSelectChunk - 1) / kSelectChunk;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto launch = [&](auto zero) {
      using T = decltype(zero);
      const T* raw_ptr = reinterpret_cast<const T*>(raw.const_data_ptr());
      const RawKeys<T> keys{
          raw_ptr, idx.const_data_ptr<int32_t>(), w, prep.row_written.const_data_ptr<int32_t>()};
      auto finish = [&](auto final_keys, int64_t n) {
        select_epilogue_kernel<T, decltype(final_keys)>
            <<<rows, kSelectThreads, 0, stream>>>(
                final_keys,
                n,
                raw_ptr,
                idx.const_data_ptr<int32_t>(),
                w,
                prep.row_written.const_data_ptr<int32_t>(),
                sort_perm.const_data_ptr<int64_t>(),
                q_scales.const_data_ptr<float>(),
                gs.const_data_ptr<float>(),
                divisor_for_int8,
                k,
                ids.mutable_data_ptr<int64_t>(),
                scores.mutable_data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      };
      if (chunks <= 1) {
        finish(keys, w);
        return;
      }
      Tensor cand;
      uint64_t* cand_ptr;
      int64_t per_row;
      if (prefilter) {
        TORCH_CHECK(w == width && block_keys.defined());
        const uint32_t* bkeys = reinterpret_cast<const uint32_t*>(block_keys.const_data_ptr<int32_t>());
        const int64_t sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
        Tensor blocks = at::empty({rows, k}, raw.options().dtype(at::kLong));
        select_blocks_kernel<<<
            rows,
            kSelectThreads,
            ((n_blocks + 31) / 32) * sizeof(uint32_t),
            stream>>>(ArrayKeys{bkeys, n_blocks}, n_blocks, k, blocks.mutable_data_ptr<int64_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        per_row = k * kTopkBlock;
        cand = at::empty({rows, per_row}, raw.options().dtype(at::kLong));
        cand_ptr = reinterpret_cast<uint64_t*>(cand.mutable_data_ptr<int64_t>());
        gather_blocks_kernel<T>
            <<<std::min(64 * sms, (rows * per_row + 255) / 256), 256, 0, stream>>>(
                keys, blocks.const_data_ptr<int64_t>(), rows, k, cand_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      } else {
        cand = at::empty({rows, chunks * k}, raw.options().dtype(at::kLong));
        cand_ptr = reinterpret_cast<uint64_t*>(cand.mutable_data_ptr<int64_t>());
        select_chunk_kernel<RawKeys<T>><<<rows * chunks, kSelectThreads, 0, stream>>>(
            keys, w, chunks, k, cand_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        per_row = chunks * k;
      }
      while (per_row > 2 * kSelectChunk) { // the final stage holds up to 2 chunks
        const int64_t next_chunks = (per_row + kSelectChunk - 1) / kSelectChunk;
        Tensor next = at::empty({rows, next_chunks * k}, cand.options());
        uint64_t* next_ptr = reinterpret_cast<uint64_t*>(next.mutable_data_ptr<int64_t>());
        select_chunk_kernel<CompositeKeys><<<rows * next_chunks, kSelectThreads, 0, stream>>>(
            CompositeKeys{cand_ptr, per_row}, per_row, next_chunks, k, next_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cand = next;
        cand_ptr = next_ptr;
        per_row = next_chunks * k;
      }
      finish(CompositeKeys{cand_ptr, per_row}, per_row);
    };
    if (int32_path) {
      launch(int32_t{0});
    } else {
      launch(c10::Half{0.0f});
    }
    return {ids, scores};
  }
  Tensor top_raw;
  Tensor top_slot;
  std::tie(top_raw, top_slot) = raw_topk(raw, idx, actual_k);
  top_raw = top_raw.contiguous();
  top_slot = top_slot.contiguous();
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
