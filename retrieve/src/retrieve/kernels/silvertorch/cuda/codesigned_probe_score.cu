// CUDA C++ backend for SilverTorch, Algorithm 1 (partial_bloom) phases 2+3.
// Three kernels: two phase-2 filters that write the SAME 1-bit-per-item mask
// layout — cps_bloom_mask_kernel evaluates bloom against a transposed,
// cluster-major bit matrix, cps_clause_mask_kernel evaluates the exact
// AND-of-OR clause predicate over the probed ids — and cps_score_kernel, which
// scores int8 code rows through __dp4a gated by whichever mask it is handed.
// The scorer is filter-agnostic on purpose: a filter is a bit-vector.
//
// Design, layout contract, perf model, and the full constraint list:
//   docs/system/kernels.md § codesigned_probe_score_cuda
//
// BIT-EXACTNESS: this kernel must return results identical to the Triton
// kernel in ../codesigned_probe_score.py, and tests/parity asserts it with
// torch.equal. Do NOT add --use_fast_math to the build flags and do not
// reassociate the fp32 epilogue. The UNROLL knob on cps_score_kernel changes
// only how many items a segment keeps in flight, never the arithmetic: the
// dp4a order, the segment reduction and the two fp32 multiplies are the same
// expressions at every UNROLL, so parity is independent of the config.

#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <cstdint>

// __dp4a is an sm_61 (Pascal, CUDA 8) intrinsic — see sm_61_intrinsics.h. There
// is no portable fallback here on purpose: the whole scoring design is the
// 4-MAC dot instruction, so an older target must fail loudly at compile time
// rather than silently miscompile or fall back.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
#error "retrieve CUDA SilverTorch backend requires sm_61 or newer (__dp4a). \
Set TORCH_CUDA_ARCH_LIST (e.g. TORCH_CUDA_ARCH_LIST=8.0) before building."
#endif

namespace {

constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr int kMaskKernelThreads = 256;

// Exact integer sum over the SEG lanes named in `mask`. __reduce_add_sync is
// hardware-accelerated on sm_80+; the butterfly fallback keeps compute_XX PTX
// JIT targets below 80 correct. Both orders are exact — integer addition —
// so the choice never affects results.
template <int SEG>
__device__ __forceinline__ int seg_reduce_add(unsigned mask, int v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  return __reduce_add_sync(mask, v);
#else
#pragma unroll
  for (int off = SEG / 2; off > 0; off >>= 1) v += __shfl_xor_sync(mask, v, off);
  return v;
#endif
}

// Phase 2. One thread produces one output mask word (= the verdict for 64
// items); consecutive threads walk consecutive words of one cluster span, so
// the sigs_t row reads coalesce.
//
// NOTE, unlike cps_clause_mask_kernel: the pad tail of a span's last word is NOT
// forced to 0 here. build_transposed_sigs zero-fills those columns, so an AND
// over any set query bit clears them — but an *empty* QB short-circuits to the
// ~0 identity and leaves the tail at 1. Harmless, because phase 3 only ever
// addresses bit slot for slot < max_size, and the parity test checks the same
// range. Do not start relying on a zero tail from this kernel.
__global__ void cps_bloom_mask_kernel(
    const long long* __restrict__ qb,         // [B, W] query signature
    const long long* __restrict__ sigs_t,     // [W*64, n_lists*wpc] transposed index
    const long long* __restrict__ probe_ids,  // [B, n_probe] probed cluster ids
    long long* __restrict__ mask,             // [B, n_probe*wpc] out
    long long sig_row_words,                  // n_lists * wpc
    long long mask_words,                     // n_probe * wpc
    long long n_probe,
    int wpc,
    int W) {
  const long long b = blockIdx.y;
  const long long idx =
      static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= mask_words) return;

  const long long pi = idx / wpc;              // probe ordinal
  const long long w_in_c = idx - pi * wpc;     // word within the cluster span
  const long long cluster = __ldg(probe_ids + b * n_probe + pi);
  const long long col = cluster * wpc + w_in_c;

  long long result = ~0LL;  // AND identity: no set query bits => all pass
  for (int qw = 0; qw < W; ++qw) {
    unsigned long long bits =
        static_cast<unsigned long long>(__ldg(qb + b * W + qw));
    while (bits) {  // iterate ONLY the set bits of QB (paper Fig. 3(b))
      const int j = __ffsll(static_cast<long long>(bits)) - 1;
      bits &= bits - 1;
      const long long m = static_cast<long long>(qw) * 64 + j;
      result &= __ldg(sigs_t + m * sig_row_words + col);  // one and.b64 = 64 items
    }
  }
  mask[b * mask_words + idx] = result;
}

// Phase 2, exact-clause variant. Same output layout as cps_bloom_mask_kernel
// (bit s%64 of word pi*wpc + s/64 = verdict for slot s of the pi-th probed
// cluster), so cps_score_kernel<..., HAS_MASK=true> consumes it unchanged.
//
// One WARP owns one output word: lane l evaluates slot l of each 32-slot half
// and two __ballot_sync calls pack the halves into the low/high 32 bits. Ids
// come from flat_items — the very tensor phase 3 reads — so exact mode needs no
// cluster-major attribute copy and no extra registered buffer; the 32 lanes of
// a half read 32 consecutive int64 ids, one coalesced 256-byte run.
//
// The predicate is bit-identical to retrieve.kernels.common.clause_pass:
// keep = (id >= 0) AND over clauses of [ (OR over A_MAX values of attr == q_c)
// XOR rev_c OR (q_c == -1) ]. Padding slots (id < 0) and the pad tail of the
// last word of a span get bit 0.
__global__ void cps_clause_mask_kernel(
    const long long* __restrict__ flat_items,      // [B, P] int64, -1 = pad
    const long long* __restrict__ item_attrs,      // [N, C, A_MAX] int64
    const unsigned char* __restrict__ is_reverse,  // [C] torch.bool storage
    const long long* __restrict__ query_attrs,     // [B, C] int64, -1 = inactive
    long long* __restrict__ mask,                  // [B, n_probe*wpc] out
    long long P,
    long long max_size,  // items per cluster span (P = n_probe * max_size)
    long long mask_words,
    int wpc,
    int n_clauses,
    int a_max) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  // Warp-uniform (depends only on blockIdx.x and the warp ordinal), so this
  // early return never leaves a warp partially populated at the ballots below.
  const long long word_idx =
      static_cast<long long>(blockIdx.x) * (blockDim.x >> 5) + warp;
  if (word_idx >= mask_words) return;

  const long long b = blockIdx.y;
  const long long pi = word_idx / wpc;          // probe ordinal
  const long long w_in_c = word_idx - pi * wpc; // word within the cluster span
  const long long slot0 = w_in_c * 64;          // first slot this word covers
  const long long span = b * P + pi * max_size; // flat_items base of the span
  const long long* q_row = query_attrs + b * n_clauses;

  unsigned packed[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const long long slot = slot0 + h * 32 + lane;
    int bit = 0;
    if (slot < max_size) {  // pad tail of the last word stays 0
      const long long id = __ldg(flat_items + span + slot);
      if (id >= 0) {        // cluster padding stays 0
        bool keep = true;
        for (int c = 0; c < n_clauses; ++c) {
          // q_c / rev are warp-uniform: 32 lanes hit one address, broadcast.
          const long long q_c = __ldg(q_row + c);
          const bool rev = __ldg(is_reverse + c) != 0;
          const long long* attr =
              item_attrs + (id * n_clauses + c) * a_max;
          bool match = false;
          for (int a = 0; a < a_max; ++a) match |= (__ldg(attr + a) == q_c);
          // XOR the reverse flag first, then let the inactive sentinel
          // override it — the order clause_pass uses.
          keep &= (match != rev) || (q_c == -1);
        }
        bit = keep ? 1 : 0;
      }
    }
    // Outside every lane-divergent branch: all 32 lanes must reach both ballots.
    packed[h] = __ballot_sync(kFullMask, bit);
  }
  if (lane == 0) {
    mask[b * mask_words + word_idx] = static_cast<long long>(
        static_cast<unsigned long long>(packed[0]) |
        (static_cast<unsigned long long>(packed[1]) << 32));
  }
}

// Phase 3, specialized. D = SEG * WPL * 4 is a compile-time constant, so each
// lane's query words stay in registers for the whole block. The dispatch
// switch below depends on this table:
//   D=64  → SEG=16, WPL=1 (two items in flight per warp)
//   D=128 → SEG=32, WPL=1 (one 128B line per row)
//   D=256 → SEG=32, WPL=2
//
// UNROLL (1, 2 or 4; config-gated, 1 = the original one-item-per-iteration
// loop) is how many items one segment keeps in flight per iteration. The body
// runs in three phases — ids+verdicts, then all UNROLL row loads, then the
// dots — so UNROLL independent row gathers are issued before the first __dp4a
// stalls on one. Purely a memory-level-parallelism knob: see
// docs/system/kernels.md § Phase 3 for the Little's-law argument.
template <int SEG, int WPL, bool HAS_MASK, int UNROLL>
__global__ void cps_score_kernel(
    const int8_t* __restrict__ q_codes,        // [B, D] int8
    const float* __restrict__ q_scales,        // [B] fp32
    const long long* __restrict__ mask,        // [B, mask_words] (HAS_MASK only)
    const long long* __restrict__ flat_items,  // [B, P] int64, -1 = pad
    const int8_t* __restrict__ item_codes,     // [N, D] int8
    float* __restrict__ out,                   // [B, P] fp32
    float global_scale,
    long long P,
    long long max_size,   // items per cluster span (P = n_probe * max_size)
    long long mask_words,
    int wpc,
    int items_per_warp) {
  constexpr int SPW = 32 / SEG;      // segments per warp
  constexpr int STEP = SPW * UNROLL; // items one segment covers per iteration

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int seg = lane / SEG;  // segment index within the warp
  const int sl = lane % SEG;   // lane index within the segment
  // `if constexpr`, not a ternary: at SEG == 32 the shift arm reads `1u << 32`,
  // which is undefined even sitting in a dead arm, so it must not be
  // instantiated at all.
  unsigned seg_mask = kFullMask;
  if constexpr (SEG < 32) seg_mask = ((1u << SEG) - 1u) << (seg * SEG);

  const long long b = blockIdx.y;

  // Block-lifetime registers: this lane's query words + the row scale.
  int qw[WPL];
  const int* q_row = reinterpret_cast<const int*>(q_codes + b * (SEG * WPL * 4));
#pragma unroll
  for (int w = 0; w < WPL; ++w) qw[w] = __ldg(q_row + sl + w * SEG);
  const float q_scale = __ldg(q_scales + b);
  // `if constexpr` again: without a mask `mask` is a 1x1 dummy, so even forming
  // (never mind reading) mask + b*mask_words would be out-of-range.
  const long long* mask_row = nullptr;
  if constexpr (HAS_MASK) mask_row = mask + b * mask_words;

  // The warp owns a contiguous item range; segments interleave inside it so
  // the SPW ids/stores in flight per iteration stay adjacent (one sector).
  const long long warp_start =
      static_cast<long long>(blockIdx.x) * (items_per_warp * (blockDim.x >> 5)) +
      static_cast<long long>(warp) * items_per_warp;
  const long long warp_end_raw = warp_start + items_per_warp;
  const long long warp_end = warp_end_raw < P ? warp_end_raw : P;
  const long long p_first = warp_start + seg;

  // (cluster, slot) of the segment's base item, tracked incrementally so the
  // hot loop carries no 64-bit division: one divide here, then bounded carry
  // loops of at most STEP (<= 8) subtractions. Correct for any max_size >= 1,
  // including max_size < STEP, where one step crosses several cluster spans.
  long long cl = 0;
  long long slot = 0;
  if constexpr (HAS_MASK) {
    cl = p_first / max_size;
    slot = p_first - cl * max_size;
  }

  for (long long p0 = p_first; p0 < warp_end; p0 += STEP) {
    // (a) ids and keep verdicts for the UNROLL items in flight. Both are
    // segment-uniform: every lane of a segment reads the same addresses.
    long long ids[UNROLL];
    bool keep[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const long long p_u = p0 + u * SPW;
      const bool live = p_u < warp_end;  // tile tail: no load, no store
      // p0 is always a real slot, so the tail's dead items never so much as
      // form an out-of-range address; the load itself is predicated off.
      const long long p_ld = live ? p_u : p0;
      ids[u] = live ? __ldg(flat_items + b * P + p_ld) : -1;
      keep[u] = live && ids[u] >= 0;
    }
    if constexpr (HAS_MASK) {
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        if (keep[u]) {  // segment-uniform
          // Derive this item's (cluster, slot) from the segment's base by the
          // same carry loop; u * SPW <= 6, so it is a handful of subtractions.
          long long slot_u = slot + u * SPW;
          long long cl_u = cl;
          while (slot_u >= max_size) {
            slot_u -= max_size;
            ++cl_u;
          }
          // The mask word covers 64 consecutive slots, so it stays L1-resident
          // across a segment's iterations.
          const long long word = __ldg(mask_row + cl_u * wpc + (slot_u >> 6));
          keep[u] = (word >> (slot_u & 63)) & 1;
        }
      }
    }
    // (b) all UNROLL row gathers, issued back to back before the first dot so
    // their latencies overlap. Predicated loads rather than serialized `if`
    // blocks: keep[u] is segment-uniform, so this is branch-free and a
    // filtered item still costs no row traffic. A rejected ids[u] is clamped to
    // row 0: the load never happens, and the address stays inside the table.
    int rw[UNROLL][WPL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const int* row = reinterpret_cast<const int*>(
          item_codes + (keep[u] ? ids[u] : 0) * (SEG * WPL * 4));
#pragma unroll
      for (int w = 0; w < WPL; ++w)
        rw[u][w] = keep[u] ? __ldg(row + sl + w * SEG) : 0;
    }
    // (c) one dot + epilogue + store per item, in item order. Identical
    // arithmetic at every UNROLL, which is what keeps parity config-free.
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const long long p_u = p0 + u * SPW;
      if (p_u >= warp_end) continue;  // segment-uniform: the whole segment skips
      float score = -INFINITY;
      if (keep[u]) {
        int acc = 0;
#pragma unroll
        for (int w = 0; w < WPL; ++w) acc = __dp4a(rw[u][w], qw[w], acc);
        acc = seg_reduce_add<SEG>(seg_mask, acc);
        // Two fp32 multiplies, left-associated — matches the Triton epilogue.
        score = static_cast<float>(acc) * q_scale * global_scale;
      }
      if (sl == 0) out[b * P + p_u] = score;
    }
    if constexpr (HAS_MASK) {
      slot += STEP;
      while (slot >= max_size) {
        slot -= max_size;
        ++cl;
      }
    }
  }
}

// Phase 3, generic fallback for any D % 4 == 0 outside {64, 128, 256}:
// full-warp segments with a runtime word loop; query words reload through the
// read-only cache (L1-resident — the row gathers dominate traffic anyway).
// This one has no UNROLL parameter and the launcher ignores the config's
// value: the fallback exists for correctness on off-table D, and a runtime
// word loop already has WPL-worth of loads in flight without help.
template <bool HAS_MASK>
__global__ void cps_score_kernel_generic(
    const int8_t* __restrict__ q_codes,
    const float* __restrict__ q_scales,
    const long long* __restrict__ mask,
    const long long* __restrict__ flat_items,
    const int8_t* __restrict__ item_codes,
    float* __restrict__ out,
    float global_scale,
    long long P,
    long long max_size,
    long long mask_words,
    int wpc,
    int items_per_warp,
    int d_words) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const long long D = static_cast<long long>(d_words) * 4;

  const long long b = blockIdx.y;
  const int* q_row = reinterpret_cast<const int*>(q_codes + b * D);
  const float q_scale = __ldg(q_scales + b);
  // `if constexpr`, not a ternary: without a mask `mask` is a 1x1 dummy, so even
  // forming (never mind reading) mask + b*mask_words would be out of range.
  const long long* mask_row = nullptr;
  if constexpr (HAS_MASK) mask_row = mask + b * mask_words;

  const long long warp_start =
      static_cast<long long>(blockIdx.x) * (items_per_warp * (blockDim.x >> 5)) +
      static_cast<long long>(warp) * items_per_warp;
  const long long warp_end_raw = warp_start + items_per_warp;
  const long long warp_end = warp_end_raw < P ? warp_end_raw : P;

  for (long long p = warp_start; p < warp_end; ++p) {
    const long long id = __ldg(flat_items + b * P + p);
    bool keep = id >= 0;
    // `if constexpr` so the mask_row arithmetic and the `p / max_size` divide are
    // not even instantiated in the no-filter specialization (where max_size is 0).
    if constexpr (HAS_MASK) {
      if (keep) {
        const long long cl = p / max_size;
        const long long slot = p - cl * max_size;
        const long long word = __ldg(mask_row + cl * wpc + (slot >> 6));
        keep = (word >> (slot & 63)) & 1;
      }
    }
    float score = -INFINITY;
    if (keep) {
      const int* row = reinterpret_cast<const int*>(item_codes + id * D);
      int acc = 0;
      for (int w = lane; w < d_words; w += 32)
        acc = __dp4a(__ldg(row + w), __ldg(q_row + w), acc);
      acc = seg_reduce_add<32>(kFullMask, acc);
      score = static_cast<float>(acc) * q_scale * global_scale;
    }
    if (lane == 0) out[b * P + p] = score;
  }
}

long long cdiv(long long a, long long b) { return (a + b - 1) / b; }

const long long* i64_ptr(const at::Tensor& t) {
  return reinterpret_cast<const long long*>(t.const_data_ptr<int64_t>());
}

// Phase-2 host launcher. The TORCH_CHECKs below validate *shapes, dtypes and
// contiguity* — the Python wrapper checks the same things first, so these fire
// only on a direct extension call. What no layer checks, here or in Python, is
// the *values*: `probe_ids` entries and `flat_items` ids index the sigs / code /
// attribute tables without bounds tests, on purpose. Both come from the layer's
// own registered buffers (`padded_cluster_items` gathered by a `topk` over
// `n_lists` centroids), so they are in range by construction, and a device-side
// check per item would cost a branch in the hot loop. A caller who hand-builds
// them out of range gets an out-of-bounds read, not an exception.
void bloom_partial_mask(
    const at::Tensor& query_bits,   // [B, W] int64
    const at::Tensor& bloom_sigs_t, // [W*64, n_lists*wpc] int64
    const at::Tensor& probe_ids,    // [B, n_probe] int64
    at::Tensor& mask_out,           // [B, n_probe*wpc] int64
    int64_t wpc) {
  TORCH_CHECK(query_bits.is_cuda(), "query_bits must be a CUDA tensor");
  TORCH_CHECK(query_bits.is_contiguous() && bloom_sigs_t.is_contiguous() &&
                  probe_ids.is_contiguous() && mask_out.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(query_bits.scalar_type() == at::kLong &&
                  bloom_sigs_t.scalar_type() == at::kLong &&
                  probe_ids.scalar_type() == at::kLong &&
                  mask_out.scalar_type() == at::kLong,
              "all tensors must be int64");
  const int64_t b = query_bits.size(0);
  const int64_t w = query_bits.size(1);
  const int64_t n_probe = probe_ids.size(1);
  const int64_t mask_words = mask_out.size(1);
  TORCH_CHECK(bloom_sigs_t.size(0) == w * 64, "bloom_sigs_t rows must equal m_bits = W*64");
  TORCH_CHECK(bloom_sigs_t.size(1) % wpc == 0, "bloom_sigs_t width must be n_lists*wpc");
  TORCH_CHECK(mask_words == n_probe * wpc, "mask_out width must be n_probe*wpc");
  TORCH_CHECK(probe_ids.size(0) == b && mask_out.size(0) == b, "batch mismatch");
  TORCH_CHECK(b <= 65535, "B exceeds grid.y limit, got ", b);

  const c10::cuda::OptionalCUDAGuard device_guard(query_bits.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(cdiv(mask_words, kMaskKernelThreads)),
                  static_cast<unsigned>(b));
  cps_bloom_mask_kernel<<<grid, kMaskKernelThreads, 0, stream>>>(
      i64_ptr(query_bits), i64_ptr(bloom_sigs_t), i64_ptr(probe_ids),
      reinterpret_cast<long long*>(mask_out.mutable_data_ptr<int64_t>()),
      bloom_sigs_t.size(1), mask_words, n_probe, static_cast<int>(wpc),
      static_cast<int>(w));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Phase-2 host launcher, exact-clause variant. Same contract as
// bloom_partial_mask: shape/dtype checks here and in Python, ids trusted as
// in-range layer buffers with no device-side bounds test. `max_size` is passed
// in (not derived from a probe_ids width) because this path reads ids from
// flat_items and never sees the cluster ids.
void clause_partial_mask(
    const at::Tensor& flat_items,   // [B, P] int64
    const at::Tensor& item_attrs,   // [N, C, A_max] int64
    const at::Tensor& is_reverse,   // [C] bool
    const at::Tensor& query_attrs,  // [B, C] int64
    at::Tensor& mask_out,           // [B, n_probe*wpc] int64
    int64_t max_size) {
  TORCH_CHECK(flat_items.is_cuda(), "flat_items must be a CUDA tensor");
  TORCH_CHECK(flat_items.is_contiguous() && item_attrs.is_contiguous() &&
                  is_reverse.is_contiguous() && query_attrs.is_contiguous() &&
                  mask_out.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(flat_items.scalar_type() == at::kLong &&
                  item_attrs.scalar_type() == at::kLong &&
                  query_attrs.scalar_type() == at::kLong &&
                  mask_out.scalar_type() == at::kLong,
              "flat_items, item_attrs, query_attrs and mask_out must be int64");
  // torch.bool storage is one byte per element; the kernel reads it as
  // unsigned char because __ldg has no bool overload.
  TORCH_CHECK(is_reverse.scalar_type() == at::kBool, "is_reverse must be bool");
  TORCH_CHECK(item_attrs.dim() == 3, "item_attrs must be [N, C, A_max]");
  TORCH_CHECK(query_attrs.dim() == 2, "query_attrs must be [B, C]");

  const int64_t b = flat_items.size(0);
  const int64_t p = flat_items.size(1);
  const int64_t c = item_attrs.size(1);
  const int64_t a_max = item_attrs.size(2);
  TORCH_CHECK(max_size > 0 && p % max_size == 0, "P must be n_probe * max_size");
  const int64_t wpc = (max_size + 63) / 64;
  const int64_t mask_words = mask_out.size(1);
  TORCH_CHECK(mask_words == (p / max_size) * wpc, "mask_out must be [B, n_probe*wpc]");
  TORCH_CHECK(query_attrs.size(1) == c, "clause-count mismatch: items C != query C");
  TORCH_CHECK(is_reverse.size(0) == c, "is_reverse must be [C]");
  TORCH_CHECK(query_attrs.size(0) == b && mask_out.size(0) == b, "batch mismatch");
  TORCH_CHECK(b <= 65535, "B exceeds grid.y limit, got ", b);

  const c10::cuda::OptionalCUDAGuard device_guard(flat_items.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int kWarpsPerBlock = kMaskKernelThreads / 32;  // one warp = one word
  const dim3 grid(static_cast<unsigned>(cdiv(mask_words, kWarpsPerBlock)),
                  static_cast<unsigned>(b));
  cps_clause_mask_kernel<<<grid, kMaskKernelThreads, 0, stream>>>(
      i64_ptr(flat_items), i64_ptr(item_attrs),
      reinterpret_cast<const unsigned char*>(is_reverse.const_data_ptr<bool>()),
      i64_ptr(query_attrs),
      reinterpret_cast<long long*>(mask_out.mutable_data_ptr<int64_t>()),
      p, max_size, mask_words, static_cast<int>(wpc), static_cast<int>(c),
      static_cast<int>(a_max));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Phase-3 host launcher.
void cps_scores(
    const at::Tensor& q_codes,     // [B, D] int8
    const at::Tensor& q_scales,    // [B] fp32
    const at::Tensor& mask,        // [B, n_probe*wpc] int64 (1x1 dummy if !has_mask)
    const at::Tensor& flat_items,  // [B, P] int64
    const at::Tensor& item_codes,  // [N, D] int8
    at::Tensor& out_scores,        // [B, P] fp32
    double global_scale,
    bool has_mask,
    int64_t max_size,
    int64_t block_p,
    int64_t num_warps,
    int64_t unroll) {
  TORCH_CHECK(q_codes.is_cuda(), "q_codes must be a CUDA tensor");
  TORCH_CHECK(q_codes.is_contiguous() && q_scales.is_contiguous() && mask.is_contiguous() &&
                  flat_items.is_contiguous() && item_codes.is_contiguous() &&
                  out_scores.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(q_codes.scalar_type() == at::kChar, "q_codes must be int8");
  TORCH_CHECK(item_codes.scalar_type() == at::kChar, "item_codes must be int8");
  TORCH_CHECK(q_scales.scalar_type() == at::kFloat, "q_scales must be float32");
  TORCH_CHECK(flat_items.scalar_type() == at::kLong, "flat_items must be int64");
  TORCH_CHECK(mask.scalar_type() == at::kLong, "mask must be int64");
  TORCH_CHECK(out_scores.scalar_type() == at::kFloat, "out_scores must be float32");

  const int64_t b = q_codes.size(0);
  const int64_t d = q_codes.size(1);
  const int64_t p = flat_items.size(1);
  TORCH_CHECK(d % 4 == 0, "D must be a multiple of 4 for dp4a packing, got ", d);
  TORCH_CHECK(item_codes.size(1) == d, "item_codes D mismatch");
  TORCH_CHECK(flat_items.size(0) == b && out_scores.size(0) == b, "batch mismatch");
  TORCH_CHECK(out_scores.size(1) == p, "out_scores P mismatch");
  TORCH_CHECK(b <= 65535, "B exceeds grid.y limit, got ", b);
  TORCH_CHECK(num_warps >= 1 && num_warps <= 32, "num_warps must be in [1, 32]");
  TORCH_CHECK(block_p > 0 && block_p % num_warps == 0,
              "block_p must be a positive multiple of num_warps");
  // Template parameter of the specialized scorer, so the dispatch table below
  // enumerates it; the generic fallback ignores it.
  TORCH_CHECK(unroll == 1 || unroll == 2 || unroll == 4,
              "unroll must be 1, 2 or 4, got ", unroll);
  int64_t wpc = 1;
  if (has_mask) {
    TORCH_CHECK(max_size > 0 && p % max_size == 0, "P must be n_probe * max_size");
    wpc = (max_size + 63) / 64;
    TORCH_CHECK(mask.size(0) == b && mask.size(1) == (p / max_size) * wpc,
                "mask must be [B, n_probe*wpc]");
  }
  const int64_t mask_words = mask.size(1);

  const c10::cuda::OptionalCUDAGuard device_guard(q_codes.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const dim3 grid(static_cast<unsigned>(cdiv(p, block_p)), static_cast<unsigned>(b));
  const dim3 block(static_cast<unsigned>(num_warps * 32));
  const int items_per_warp = static_cast<int>(block_p / num_warps);
  const float gs = static_cast<float>(global_scale);

  const int8_t* q_codes_p = q_codes.const_data_ptr<int8_t>();
  const float* q_scales_p = q_scales.const_data_ptr<float>();
  const long long* mask_p = i64_ptr(mask);
  const long long* flat_p = i64_ptr(flat_items);
  const int8_t* codes_p = item_codes.const_data_ptr<int8_t>();
  float* out_p = out_scores.mutable_data_ptr<float>();

  // Dispatch is {64, 128, 256} x HAS_MASK x {1, 2, 4}: three nested macros, one
  // per axis, so the table reads top-down as D -> unroll -> launch.
#define CPS_LAUNCH(SEG, WPL, HASM, UNR)                                               \
  cps_score_kernel<SEG, WPL, HASM, UNR><<<grid, block, 0, stream>>>(                  \
      q_codes_p, q_scales_p, mask_p, flat_p, codes_p, out_p, gs, p, max_size,         \
      mask_words, static_cast<int>(wpc), items_per_warp)
#define CPS_LAUNCH_GENERIC(HASM)                                                      \
  cps_score_kernel_generic<HASM><<<grid, block, 0, stream>>>(                         \
      q_codes_p, q_scales_p, mask_p, flat_p, codes_p, out_p, gs, p, max_size,         \
      mask_words, static_cast<int>(wpc), items_per_warp, static_cast<int>(d / 4))
#define CPS_LAUNCH_UNROLL(SEG, WPL, HASM)                                             \
  switch (unroll) {                                                                   \
    case 1: CPS_LAUNCH(SEG, WPL, HASM, 1); break;                                     \
    case 2: CPS_LAUNCH(SEG, WPL, HASM, 2); break;                                     \
    case 4: CPS_LAUNCH(SEG, WPL, HASM, 4); break;                                     \
    default: TORCH_CHECK(false, "unroll must be 1, 2 or 4, got ", unroll);            \
  }
#define CPS_LAUNCH_D(HASM)                                                            \
  switch (d) {                                                                        \
    case 64: CPS_LAUNCH_UNROLL(16, 1, HASM); break;                                   \
    case 128: CPS_LAUNCH_UNROLL(32, 1, HASM); break;                                  \
    case 256: CPS_LAUNCH_UNROLL(32, 2, HASM); break;                                  \
    default: CPS_LAUNCH_GENERIC(HASM); break;  /* ignores unroll */                   \
  }

  if (has_mask) {
    CPS_LAUNCH_D(true);
  } else {
    CPS_LAUNCH_D(false);
  }

#undef CPS_LAUNCH
#undef CPS_LAUNCH_GENERIC
#undef CPS_LAUNCH_UNROLL
#undef CPS_LAUNCH_D

  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bloom_partial_mask", &bloom_partial_mask,
        "SilverTorch partial bloom over probed clusters (transposed index, "
        "and.b64 per 64 items) into a preallocated [B, n_probe*wpc] int64 mask",
        py::arg("query_bits"), py::arg("bloom_sigs_t"), py::arg("probe_ids"),
        py::arg("mask_out"), py::arg("wpc"));
  m.def("clause_partial_mask", &clause_partial_mask,
        "SilverTorch exact AND-of-OR clause predicate over the probed items "
        "(one warp per output word, two ballots) into a preallocated "
        "[B, n_probe*wpc] int64 mask, same layout as bloom_partial_mask",
        py::arg("flat_items"), py::arg("item_attrs"), py::arg("is_reverse"),
        py::arg("query_attrs"), py::arg("mask_out"), py::arg("max_size"));
  m.def("cps_scores", &cps_scores,
        "SilverTorch fused index-matmul int8 dp4a scoring with an optional "
        "1-bit-per-item mask into a preallocated [B, P] fp32 score buffer",
        py::arg("q_codes"), py::arg("q_scales"), py::arg("mask"),
        py::arg("flat_items"), py::arg("item_codes"), py::arg("out_scores"),
        py::arg("global_scale"), py::arg("has_mask"), py::arg("max_size"),
        py::arg("block_p"), py::arg("num_warps"), py::arg("unroll"));
}
