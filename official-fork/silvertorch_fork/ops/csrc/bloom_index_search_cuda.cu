// Copyright (c) Meta Platforms, Inc. and affiliates.
// Modified by the retrieve authors, 2026-10-10: package silvertorch -> silvertorch_fork, ops torch.ops.st -> torch.ops.stfork; C4: prepared-plan search ops, no host sync; C8: bloom-evaluating payload kernels; C13: their exact recheck; K fix 4: signature_match loads after resolving (official-fork/CHANGES.md).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h> // @manual
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/macros/Macros.h>
#include <cuda.h> // @manual
#include <torch/torch.h> // @manual
#include <algorithm>
#include <cstring>
#include <tuple>
#include <type_traits>
#include <vector>

#define BLOOM_INDEX_CUDA
#include "bloom_index_util.cuh"
#include "bloom_index_util.h"
#include "fork_filters.cuh"
#include "fused_kmean_ann.cuh"

namespace st {
namespace ops {
namespace bloom_search {

using at::PackedTensorAccessor32;
using at::PackedTensorAccessor64;
using at::Tensor;
using namespace st::ops::bloom_index;

namespace {

static constexpr int32_t C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL = 16;
static constexpr int32_t C_NUM_OFFSETS_PER_PLAN = 8;

// A read-only 1-d view the kernels index like the PackedTensorAccessor32 it replaces, but
// constructible on the device (the prepared-plan path builds plans per thread).
template <typename T>
struct DeviceSpan {
  T* ptr;
  int32_t n;
  C10_HOST_DEVICE T& operator[](int64_t i) const {
    return ptr[i];
  }
  C10_HOST_DEVICE int32_t size(int /*dim*/) const {
    return n;
  }
  C10_HOST_DEVICE T* data() const {
    return ptr;
  }
};

template <typename T>
DeviceSpan<T> span_of(PackedTensorAccessor32<T, 1> a) {
  return DeviceSpan<T>{a.data(), static_cast<int32_t>(a.size(0))};
}

template <bool is_bloom_index_v2>
struct QueryPlanCuda {
 public:
  using OneBits = typename QueryPlanOneBitsPType<is_bloom_index_v2>::type;
  QueryPlanCuda(
      PackedTensorAccessor32<uint8_t, 1> operators,
      PackedTensorAccessor32<int32_t, 1> offsets,
      PackedTensorAccessor32<int32_t, 1> parameters,
      PackedTensorAccessor32<OneBits, 1> oneBitsPosition)
      : operators(span_of(operators)),
        offsets(span_of(offsets)),
        parameters(span_of(parameters)),
        oneBitsPosition(span_of(oneBitsPosition)) {}
  __device__ QueryPlanCuda(
      DeviceSpan<uint8_t> operators,
      DeviceSpan<int32_t> offsets,
      DeviceSpan<int32_t> parameters,
      DeviceSpan<OneBits> oneBitsPosition)
      : operators(operators),
        offsets(offsets),
        parameters(parameters),
        oneBitsPosition(oneBitsPosition) {}

  DeviceSpan<uint8_t> operators;
  DeviceSpan<int32_t> offsets;
  DeviceSpan<int32_t> parameters;
  DeviceSpan<OneBits> oneBitsPosition;
};

// Plans in the parser's own encoding (expression_query_parser: per plan 8 byte offsets
// [ops, offsets, parameters, oneBitsPosition] x [start, end) into the data, each segment
// 8-aligned), read in place on the device: plan i is built per thread, with no host decode.
// The prepared blob is int64 [8 * plan_count + data words]: the offsets, then the data.
struct EncodedQueryPlans {
  const int64_t* __restrict__ offsets;
  const int8_t* __restrict__ data;

  __device__ __forceinline__ QueryPlanCuda<true> operator[](int64_t i) const {
    const int64_t* o = offsets + 8 * i;
    int8_t* base = const_cast<int8_t*>(data);
    return QueryPlanCuda<true>(
        DeviceSpan<uint8_t>{
            reinterpret_cast<uint8_t*>(base + o[0]),
            static_cast<int32_t>(o[1] - o[0])},
        DeviceSpan<int32_t>{
            reinterpret_cast<int32_t*>(base + o[2]),
            static_cast<int32_t>((o[3] - o[2]) / sizeof(int32_t))},
        DeviceSpan<int32_t>{
            reinterpret_cast<int32_t*>(base + o[4]),
            static_cast<int32_t>((o[5] - o[4]) / sizeof(int32_t))},
        DeviceSpan<uint64_t>{
            reinterpret_cast<uint64_t*>(base + o[6]),
            static_cast<int32_t>((o[7] - o[6]) / sizeof(uint64_t))});
  }
};

EncodedQueryPlans encoded_plans(const Tensor& plans, int64_t plan_count) {
  TORCH_CHECK(plans.is_cuda() && plans.scalar_type() == at::kLong);
  TORCH_CHECK(plans.dim() == 1 && plans.is_contiguous());
  TORCH_CHECK(plans.numel() >= 8 * plan_count);
  const int64_t* base = plans.const_data_ptr<int64_t>();
  return EncodedQueryPlans{
      base, reinterpret_cast<const int8_t*>(base + 8 * plan_count)};
}

// This class is created to allocate cuda memory once and copy once to improve
// performance. This class is using visitor pattern. usage:
//   1. call add_size_and_data for all the potential GPU memory allocations.
//   2. call visit to create GPU tensor for all tensors.
//   3. add_size_without_data must be called after all add_size_and_data.
template <bool is_bloom_index_v2>
class TensorCreationVisitor {
 public:
  TensorCreationVisitor(const at::Device& device)
      : m_offset(0), m_size_to_copy(0), m_device(device) {}

  template <typename T>
  void add_size_and_data(const std::vector<T>& vec) {
    TORCH_CHECK(m_cudaVec.numel() == 0);
    using ValueType =
        std::conditional_t<std::is_same_v<T, Operator>, uint8_t, T>;
    auto offset = m_hostVec.size();
    m_hostVec.resize(offset + align_size(vec.size() * sizeof(ValueType)));
    // @lint-ignore CLANGSECURITY facebook-security-vulnerable-memcpy
    std::memcpy(
        m_hostVec.data() + offset, vec.data(), vec.size() * sizeof(ValueType));
    m_size_to_copy = m_hostVec.size();
  }

  void add_size_and_data(const QueryPlan<is_bloom_index_v2>& query_plan) {
    TORCH_CHECK(m_cudaVec.numel() == 0);
    // Sequence here need to be exact same as visit.
    add_size_and_data(query_plan.operators);
    add_size_and_data(query_plan.offsets);
    add_size_and_data(query_plan.parameters);
    add_size_and_data(query_plan.oneBitsPosition);
  }

  void add_size_without_data(size_t size) {
    m_hostVec.resize(m_hostVec.size() + align_size(size));
  }

  template <typename T>
  Tensor visit(const std::vector<T>& vec) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(m_device.index());

    if (m_cudaVec.numel() == 0) {
      TORCH_CHECK(!m_hostVec.empty());
      m_cudaVec = at::empty(
          {static_cast<int64_t>(m_hostVec.size())},
          c10::TensorOptions().dtype(at::kChar).device(m_device),
          at::MemoryFormat::Contiguous);

      AT_CUDA_CHECK(cudaMemcpyAsync(
          reinterpret_cast<void*>(m_cudaVec.mutable_data_ptr<int8_t>()),
          reinterpret_cast<void*>(m_hostVec.data()),
          m_size_to_copy,
          cudaMemcpyHostToDevice,
          at::cuda::getCurrentCUDAStream()));
    }

    return vector_to_tensor<T>(vec);
  }

  // Convert query plan to a variant that uses cuda-compatible data
  // structures.
  QueryPlanCuda<is_bloom_index_v2> visit(
      const QueryPlan<is_bloom_index_v2>& query_plan) {
    // Sequence here need to be exact same as add_size_and_data.
    auto operators = visit(query_plan.operators);
    auto offsets = visit(query_plan.offsets);
    auto parameters = visit(query_plan.parameters);
    auto oneBitsPosition = visit(query_plan.oneBitsPosition);
    return QueryPlanCuda<is_bloom_index_v2>(
        operators.template packed_accessor32<uint8_t, 1>(),
        offsets.template packed_accessor32<int32_t, 1>(),
        parameters.template packed_accessor32<int32_t, 1>(),
        oneBitsPosition.template packed_accessor32<
            typename QueryPlanOneBitsPType<is_bloom_index_v2>::type,
            1>());
  }

  const QueryPlanCuda<is_bloom_index_v2>* visit_with_data(
      const std::vector<QueryPlanCuda<is_bloom_index_v2>>&
          query_plan_cuda_host) {
    at::cuda::OptionalCUDAGuard device_guard;
    device_guard.set_index(m_device.index());

    TORCH_CHECK(
        m_offset == m_size_to_copy,
        "add_size_without_data must call only after all add_size_and_data, same as visit_with_data.");

    size_t size_of_obj =
        query_plan_cuda_host.size() * sizeof(QueryPlanCuda<is_bloom_index_v2>);
    size_t size_to_add = align_size(size_of_obj);
    TORCH_CHECK(m_offset + size_to_add <= m_hostVec.size());

    QueryPlanCuda<is_bloom_index_v2>* query_plan_cuda_ptr =
        reinterpret_cast<QueryPlanCuda<is_bloom_index_v2>*>(
            m_cudaVec.data_ptr<int8_t>() + m_offset);
    AT_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(query_plan_cuda_ptr),
        reinterpret_cast<const void*>(query_plan_cuda_host.data()),
        size_of_obj,
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));

    m_offset += size_to_add;
    return query_plan_cuda_ptr;
  }

 private:
  // Populate a tensor with data from a vector.
  template <typename T>
  Tensor vector_to_tensor(const std::vector<T>& vec) {
    using ValueType =
        std::conditional_t<std::is_same_v<T, Operator>, uint8_t, T>;

    size_t size_to_add = align_size(vec.size() * sizeof(ValueType));
    TORCH_CHECK(m_offset + size_to_add <= m_hostVec.size());

    Tensor ret = torch::from_blob(
        m_cudaVec.data_ptr<int8_t>() + m_offset,
        {static_cast<int64_t>(vec.size())}, // size
        c10::TensorOptions().dtype(c10::CppTypeToScalarType<ValueType>::value));

    m_offset += size_to_add;
    return ret;
  }

  size_t m_offset;
  size_t m_size_to_copy;
  std::vector<char> m_hostVec;
  Tensor m_cudaVec;
  at::Device m_device;
};

// Prepare query plans from the query plans data.
template <bool is_bloom_index_v2>
__inline__ __device__ uint64_t signature_match(
    const int64_t* __restrict__ bloom_index,
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    typename QueryPlanOneBitsPType<is_bloom_index_v2>::type oneBitsPosition[]);

template <>
__inline__ __device__ uint64_t signature_match<false /*is_bloom_v2*/>(
    const int64_t* __restrict__ bloom_index,
    const int64_t total_column_count,
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t /*p_k*/,
    typename QueryPlanOneBitsPType<false /*is_bloom_v2*/>::type
        oneBitsPosition[]) {
  uint64_t result = ~(static_cast<uint64_t>(0));
  for (int64_t i = 0; i < k; ++i) {
    result &= bloom_index
        [static_cast<int64_t>(oneBitsPosition[i]) * total_column_count +
         column];
  }
  return result;
}

template <>
__inline__ __device__ uint64_t signature_match<true>(
    const int64_t* __restrict__ bloom_index,
    const int64_t /*total_column_count*/,
    const int64_t b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k,
    typename QueryPlanOneBitsPType<true>::type oneBitsPosition[]) {
  uint64_t result = ~(static_cast<uint64_t>(0));
  // TODO(zhenwang): get rid of duplicate check here.
  // we use MAX_K_V2 here because I observed that if we use large number
  // here (e.g. MAX_K), the process_op will sometimes get illegal memory
  // access error, meanwhile, the faster_process_op is working fine.
  // My guess is that it may hit stack size limit.
  std::array<typename BIndexType<true>::type, MAX_K_V2> used_b_p = {0};
  int32_t p = 0;
  int32_t resolved = 0;
  int64_t column_in_bundle = column % C_BLOOM_V2_COL_BUNDLE_SIZE;
  while (p < p_k && resolved < k) {
    auto b_p =
        static_cast<typename BIndexType<true>::type>(oneBitsPosition[p++] % b);
    bool duplicate = false;
    for (int32_t i = 0; i < resolved; ++i) {
      if (used_b_p[i] == b_p) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      used_b_p[resolved++] = b_p;
    }
  }
  // fork (K fix 4): the k index words loaded after the positions are resolved, so the loads
  // are independent and overlap (upstream loads each one inside the resolve loop); same bits.
#pragma unroll
  for (int32_t i = 0; i < MAX_K_V2; ++i) {
    if (i < resolved) {
      result &= __ldg(
          bloom_index + (bundle_b_start + used_b_p[i]) * C_BLOOM_V2_COL_BUNDLE_SIZE +
          column_in_bundle);
    }
  }
  return result;
}

template <bool is_bloom_index_v2>
__inline__ __device__ int64_t
get_one_bits_position_step(int64_t k, int64_t p_k);

template <>
__inline__ __device__ int64_t
get_one_bits_position_step<false /*is_bloom_v2*/>(int64_t k, int64_t p_k) {
  return k;
}

template <>
__inline__ __device__ int64_t
get_one_bits_position_step<true>(int64_t k, int64_t p_k) {
  return p_k;
}

template <bool is_bloom_index_v2>
__device__ uint64_t process_op(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int32_t op_index,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  int32_t param_start = query_plan.offsets[op_index];
  int32_t param_end = query_plan.offsets[op_index + 1];
  if (query_plan.operators[op_index] == Operator::TERM) {
    return signature_match<is_bloom_index_v2>(
        bloom_index,
        total_column_count,
        column_b,
        bundle_b_start,
        column,
        k,
        p_k,
        query_plan.oneBitsPosition.data() +
            (query_plan.parameters[param_start] *
             get_one_bits_position_step<is_bloom_index_v2>(k, p_k)));
  }

  if (query_plan.operators[op_index] == Operator::AND) {
    uint64_t result = ~(static_cast<uint64_t>(0));
    for (; param_start < param_end; ++param_start) {
      result &= process_op<is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          column_b,
          bundle_b_start,
          column,
          query_plan.parameters[param_start],
          k,
          p_k,
          query_plan);
    }
    return result;
  }

  if (query_plan.operators[op_index] == Operator::OR) {
    uint64_t result = 0;
    for (; param_start < param_end; ++param_start) {
      result |= process_op<is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          column_b,
          bundle_b_start,
          column,
          query_plan.parameters[param_start],
          k,
          p_k,
          query_plan);
    }
    return result;
  }

  if (query_plan.operators[op_index] == Operator::NOT) {
    return ~process_op<is_bloom_index_v2>(
        bloom_index,
        total_column_count,
        column_b,
        bundle_b_start,
        column,
        query_plan.parameters[param_start],
        k,
        p_k,
        query_plan);
  }

  if (query_plan.operators[op_index] == Operator::EMPTY) {
    return UINT64_MAX;
  }

  // Shouldn't go here.
  return 0;
}

template <int32_t STACK_SIZE>
class Stack {
 public:
  __device__ Stack() : m_cur(0) {}

  __inline__ __device__ void push(uint64_t value) {
    m_value[m_cur++] = value;
  }

  __inline__ __device__ uint64_t pop() {
    // m_cur can't never be less than 0, since we run the check at
    // get_max_stack_size method.
    return m_value[--m_cur];
  }

 private:
  int32_t m_cur;
  uint64_t m_value[STACK_SIZE];
};

// Use fix size stack as a fast route.
template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__inline__ __device__ uint64_t fast_process_op(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t column_b,
    const int64_t bundle_b_start,
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  Stack<STACK_SIZE> stack;
  int32_t one_bits_position_cur = 0;
  for (size_t j = 0; j < query_plan.operators.size(0); ++j) {
    switch (query_plan.operators[j]) {
      case Operator::TERM: {
        // push to stack.
        stack.push(
            signature_match<is_bloom_index_v2>(
                bloom_index,
                total_column_count,
                column_b,
                bundle_b_start,
                column,
                k,
                p_k,
                query_plan.oneBitsPosition.data() +
                    (one_bits_position_cur++ *
                     get_one_bits_position_step<is_bloom_index_v2>(k, p_k))));
        break;
      }

      case Operator::AND: {
        uint64_t result = ~(static_cast<uint64_t>(0));
        int32_t children = query_plan.offsets[j + 1] - query_plan.offsets[j];
        for (int32_t i = 0; i < children; ++i) {
          result &= stack.pop();
        }
        stack.push(result);
        break;
      }

      case Operator::OR: {
        uint64_t result = 0;
        int32_t children = query_plan.offsets[j + 1] - query_plan.offsets[j];
        for (int32_t i = 0; i < children; ++i) {
          result |= stack.pop();
        }
        stack.push(result);
        break;
      }

      case Operator::NOT: {
        stack.push(~stack.pop());
        break;
      }

      case Operator::EMPTY: {
        stack.push(UINT64_MAX);
        break;
      }
    }
  }

  return stack.pop();
}

template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__device__ uint64_t run_query_plan(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* __restrict__ bundle_b_offsets, // useful for bloom index v2
                                                  // only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<is_bloom_index_v2>& query_plan) {
  int64_t b = -1;
  int64_t bundle_b_start = -1;
  if constexpr (is_bloom_index_v2) {
    int64_t bundle = column / C_BLOOM_V2_COL_BUNDLE_SIZE;
    bundle_b_start = __ldg(bundle_b_offsets + bundle);
    b = __ldg(bundle_b_offsets + bundle + 1) - bundle_b_start;
  }
  return fast_process_op<STACK_SIZE, is_bloom_index_v2>(
      bloom_index,
      total_column_count,
      b,
      bundle_b_start,
      column,
      k,
      p_k,
      query_plan);
}

// fall back to recursive if requires stack size is too large.
// We use 0 to indicate this.
template <>
__device__ uint64_t run_query_plan<0, true>(
    const int64_t* __restrict__ bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* __restrict__ bundle_b_offsets, // useful for bloom index v2
                                                  // only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<true>& query_plan) {
  // Start from last one, and then recursive from backward.
  int32_t root_op_index = query_plan.operators.size(0) - 1;
  int64_t bundle = column / C_BLOOM_V2_COL_BUNDLE_SIZE;
  int64_t bundle_b_start = __ldg(bundle_b_offsets + bundle);
  int64_t b = __ldg(bundle_b_offsets + bundle + 1) - bundle_b_start;
  return process_op<true>(
      bloom_index,
      total_column_count,
      b,
      bundle_b_start,
      column,
      root_op_index,
      k,
      p_k,
      query_plan);
}
template <>
__device__ uint64_t run_query_plan<0, false /*is_bloom_v2*/>(
    const int64_t* bloom_index, // if v1, its 2d, if v2, its 1d.
    const int64_t total_column_count, // useful for bloom index v1 only.
    const int64_t* bundle_b_offsets, // useful for bloom index v2 only.
    int64_t column,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    const QueryPlanCuda<false /*is_bloom_v2*/>& query_plan) {
  // Start from last one, and then recursive from backward.
  int32_t root_op_index = query_plan.operators.size(0) - 1;
  return process_op<false /*is_bloom_v2*/>(
      bloom_index,
      total_column_count,
      -1,
      -1,
      column,
      root_op_index,
      k,
      p_k,
      query_plan);
}

union MaskBlock {
  struct {
    float4 data1;
    float4 data2;
    float4 data3;
    float4 data4;
  };
  bool bits[C_BITS_IN_UINT64];
};

template <typename RETURN_T>
__device__ __inline__ void
store(RETURN_T* dest, uint64_t result, int64_t column) {
  reinterpret_cast<uint64_t*>(dest)[column] = result;
}

template <>
__device__ __inline__ void
store<bool>(bool* dest, uint64_t result, int64_t column) {
  //     lower doc id put at higher bits in unit64_t, for example:
  //     |< high                                       low >|
  //      Doc0, Doc1, Doc2 ...... Doc 60, Doc61, Doc62, Doc63
  MaskBlock mask;
#pragma unroll
  for (int i = 0; i < C_BITS_IN_UINT64; i++) {
    static constexpr uint64_t C_MASK = 1ULL << 63;
    mask.bits[i] = ((result & C_MASK) != 0);
    result <<= 1;
  }

  reinterpret_cast<MaskBlock*>(dest)[column] = mask;
}

template <
    typename RETURN_T,
    int32_t STACK_SIZE,
    bool is_bloom_index_v2,
    typename PLANS = const QueryPlanCuda<is_bloom_index_v2>*>
__global__ void process_documents(
    const int64_t* bloom_index, // if v1, its 2d, if v2, its 1d.
    const bool* column_mask_ptr,
    // total_column_count is
    //   bloom_index.size(1) for v1
    //   (bundle_b_offsets.numel()-1)*C_BLOOM_V2_COL_BUNDLE_SIZE for v2.
    const int64_t total_column_count,
    const int64_t* bundle_b_offsets, // useful for bloom index v2 only.
    PackedTensorAccessor64<RETURN_T, 2> document_mask,
    PLANS query_plan,
    int64_t k,
    int64_t p_k, // useful for bloom index v2 only.
    TORCH_DSA_KERNEL_ARGS) {
  int64_t row_count = document_mask.size(0);
  int64_t total_document_mask_block = row_count * total_column_count;

  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int64_t document_index =
           static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       document_index < total_document_mask_block;
       document_index += static_cast<size_t>(blockDim.x) * gridDim.x) {
    int64_t row = document_index / total_column_count;
    int64_t column = document_index % total_column_count;

    if (column_mask_ptr == nullptr || column_mask_ptr[column]) {
      uint64_t result = run_query_plan<STACK_SIZE, is_bloom_index_v2>(
          bloom_index,
          total_column_count,
          bundle_b_offsets,
          column,
          k,
          p_k,
          query_plan[row]);

      store<RETURN_T>(
          document_mask.data() + row * document_mask.size(1), result, column);
    }
  }
}

__global__ void generate_column_info_for_clusters_kernel(
    const int64_t* __restrict__ selected_cluster_offsets_ptr,
    const int64_t* __restrict__ selected_cluster_lengths_ptr,
    int64_t total_cluster_count,
    int32_t* column_counts,
    int32_t* start_column_ids,
    int8_t* first_item_offsets_in_column,
    TORCH_DSA_KERNEL_ARGS) {
  // Stride loop:
  // https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
  for (int64_t cluster_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cluster_index < total_cluster_count;
       cluster_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t local_cluster_len = selected_cluster_lengths_ptr[cluster_index];
    int64_t local_cluster_start = selected_cluster_offsets_ptr[cluster_index];
    int32_t start_col_id =
        static_cast<int32_t>(local_cluster_start / C_BITS_IN_UINT64);
    int8_t offset_in_col =
        static_cast<int8_t>(local_cluster_start % C_BITS_IN_UINT64);
    start_column_ids[cluster_index] = start_col_id;
    first_item_offsets_in_column[cluster_index] = offset_in_col;
    column_counts[cluster_index] = local_cluster_len > 0
        ? static_cast<int32_t>(
              (local_cluster_len + offset_in_col + C_BITS_IN_UINT64 - 1) /
              C_BITS_IN_UINT64)
        : 0;
  }
}

template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__global__ void process_documents_on_assigned_cluster_columns_kernel(
    const int64_t* __restrict__ bloom_index,
    const int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    const PackedTensorAccessor64<int32_t, 1> column_assigned_cluster_index,
    const PackedTensorAccessor64<int32_t, 1> column_counts_cumsum,
    const PackedTensorAccessor64<int32_t, 1> start_column_ids,
    int64_t cluster_count_per_query,
    PackedTensorAccessor64<int64_t, 1> column_mask_response,
    const QueryPlanCuda<is_bloom_index_v2>* __restrict__ query_plan,
    int64_t k,
    int64_t p_k,
    const uint64_t* __restrict__ query_plan_index_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  int64_t total_column_count = column_assigned_cluster_index.size(0);
  for (int64_t column_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       column_index < total_column_count;
       column_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t assigned_cluster_index =
        column_assigned_cluster_index[column_index];
    int64_t query_id = assigned_cluster_index / cluster_count_per_query;
    query_id = query_plan_index_ptr == nullptr
        ? query_id
        : static_cast<int64_t>(query_plan_index_ptr[query_id]);
    int64_t col_idx_in_local_cluster = column_index -
        (assigned_cluster_index == 0
             ? 0
             : column_counts_cumsum[assigned_cluster_index - 1]);
    int64_t column_id =
        start_column_ids[assigned_cluster_index] + col_idx_in_local_cluster;
    uint64_t result = run_query_plan<STACK_SIZE, is_bloom_index_v2>(
        bloom_index,
        index_total_column_count,
        bundle_b_offsets,
        column_id,
        k,
        p_k,
        query_plan[query_id]);
    store<int64_t>(column_mask_response.data() + column_index, result, 0);
  }
}

// The prepared-plan partial search: one thread per probed column, its cluster found by a
// binary search over the device column cumsum (no host repeat_interleave), the total read on
// the device; the grid comes from the caller's static column capacity.
template <int32_t STACK_SIZE>
__global__ void process_documents_on_probed_columns_kernel(
    const int64_t* __restrict__ bloom_index,
    const int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    const int32_t* __restrict__ column_counts_cumsum,
    const int32_t* __restrict__ start_column_ids,
    int32_t cluster_count,
    int64_t cluster_count_per_query,
    int64_t* __restrict__ column_mask_response,
    EncodedQueryPlans query_plans,
    int64_t k,
    int64_t p_k,
    const int64_t* __restrict__ query_plan_index_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  const int64_t total_column_count = column_counts_cumsum[cluster_count - 1];
  for (int64_t column_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       column_index < total_column_count;
       column_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int32_t lo = 0;
    int32_t hi = cluster_count - 1;
    while (lo < hi) {
      int32_t mid = (lo + hi) >> 1;
      if (column_counts_cumsum[mid] > column_index) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    const int64_t assigned_cluster_index = lo;
    int64_t query_id = assigned_cluster_index / cluster_count_per_query;
    query_id = query_plan_index_ptr == nullptr ? query_id
                                               : query_plan_index_ptr[query_id];
    int64_t col_idx_in_local_cluster = column_index -
        (assigned_cluster_index == 0
             ? 0
             : column_counts_cumsum[assigned_cluster_index - 1]);
    int64_t column_id =
        start_column_ids[assigned_cluster_index] + col_idx_in_local_cluster;
    uint64_t result = run_query_plan<STACK_SIZE, true>(
        bloom_index,
        index_total_column_count,
        bundle_b_offsets,
        column_id,
        k,
        p_k,
        query_plans[query_id]);
    store<int64_t>(column_mask_response + column_index, result, 0);
  }
}

// C8: the scorer's payloads with the bloom evaluated in place, for the partial path. A full
// warp's 32 docs (and a cluster's < 32 remainder docs, one warp per cluster) lie in at most two
// 64-doc bloom columns, so the payload kernel evaluates those columns' plan results itself and cuts its mask out of
// them as upstream's payload kernels cut it out of column_mask_response: the same bits, without
// the column assignment, its cumsum, the response buffer or its launch. Index math as
// fused_kmean_ann_cuda.cu's partial-mask payload kernels; warp slots found as C3 finds them.
using st::ops::fused_kmean_ann::RemainingPayload;
using st::ops::fused_kmean_ann::WarpPayload;

template <int32_t STACK_SIZE>
__device__ __forceinline__ uint64_t column_result(
    const int64_t* __restrict__ bloom_index,
    int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    int64_t column,
    int64_t k,
    int64_t p_k,
    const QueryPlanCuda<true>& plan) {
  return run_query_plan<STACK_SIZE, true>(
      bloom_index, index_total_column_count, bundle_b_offsets, column, k, p_k, plan);
}

template <int32_t STACK_SIZE>
__global__ void bloom_warp_payload_kernel(
    const int64_t* __restrict__ bloom_index,
    int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    EncodedQueryPlans plans,
    const int64_t* __restrict__ query_plan_index,
    int64_t k,
    int64_t p_k,
    const int64_t* __restrict__ cluster_offsets,
    const int64_t* __restrict__ cluster_ids,
    int64_t clusters_per_row,
    int32_t n_clusters,
    const int32_t* __restrict__ warp_rounded_length_cumsum,
    const int32_t* __restrict__ warp_size_cumsum,
    int64_t max_tensor_size_per_row,
    WarpPayload* __restrict__ payloads) {
  const int64_t total = warp_size_cumsum[n_clusters - 1];
  for (int64_t w = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < total;
       w += static_cast<int64_t>(blockDim.x) * gridDim.x) {
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
    WarpPayload p;
    p.doc_start_index = static_cast<int32_t>(cluster_offsets[cluster_ids[ci]]) + doc_offset;
    p.write_index = static_cast<int32_t>(
        row * max_tensor_size_per_row +
        (col == 0 ? 0 : warp_rounded_length_cumsum[ci - 1]) + doc_offset);
    p.cluster_ids_row = row;
    const auto plan = plans[query_plan_index == nullptr ? row : query_plan_index[row]];
    const int64_t first_column = p.doc_start_index / 64;
    const int64_t pos = p.doc_start_index % 64;
    uint64_t columns[2];
    columns[0] = column_result<STACK_SIZE>(
        bloom_index, index_total_column_count, bundle_b_offsets, first_column, k, p_k, plan);
    columns[1] = pos > 32
        ? column_result<STACK_SIZE>(
              bloom_index,
              index_total_column_count,
              bundle_b_offsets,
              first_column + 1,
              k,
              p_k,
              plan)
        : 0;
    p.filtering_mask = get_next_32_bit_mask(columns, pos);
    reinterpret_cast<int4*>(payloads)[w] = p.payload;
  }
}

template <int32_t STACK_SIZE>
__global__ void bloom_remaining_payload_kernel(
    const int64_t* __restrict__ bloom_index,
    int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    EncodedQueryPlans plans,
    const int64_t* __restrict__ query_plan_index,
    int64_t k,
    int64_t p_k,
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
    const int32_t remaining = length - handled;
    if (remaining == 0) {
      continue;
    }
    const int32_t doc_start =
        static_cast<int32_t>(cluster_offsets[cluster_ids[i]]) + handled;
    const int64_t first_column = doc_start / 64;
    const bool second = (doc_start + remaining - 1) / 64 != first_column;
    uint64_t mine = 0;
    if (lane == 0 || (lane == 1 && second)) {
      const auto plan = plans[query_plan_index == nullptr ? row : query_plan_index[row]];
      mine = column_result<STACK_SIZE>(
          bloom_index,
          index_total_column_count,
          bundle_b_offsets,
          first_column + lane,
          k,
          p_k,
          plan);
    }
    uint64_t columns[2];
    columns[0] = __shfl_sync(0xffffffff, mine, 0);
    columns[1] = __shfl_sync(0xffffffff, mine, 1);
    if (lane >= remaining) {
      continue;
    }
    const int32_t doc = doc_start + lane;
    const int32_t out_offset = i == 0 ? 0 : remaining_length_cumsum[i - 1];
    const int32_t row_first =
        row == 0 ? 0 : remaining_length_cumsum[row * clusters_per_row - 1];
    RemainingPayload p;
    p.doc_index = doc;
    p.cluster_ids_row = row;
    p.write_index = static_cast<int32_t>(row * max_tensor_size_per_row) +
        warp_rounded_length_cumsum[row * clusters_per_row + clusters_per_row - 1] +
        (out_offset - row_first) + lane;
    p.filtering_mask = get_bit_64_bit_mask(columns, doc - first_column * 64);
    reinterpret_cast<int4*>(payloads)[out_offset + lane] = p.payload;
  }
}

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

// C13: the same payloads with the bloom's survivors rechecked exactly (bloom ∧ exact = exact,
// the bloom having no false negatives). One
// warp per full-warp payload: lanes 0 / 1 evaluate the ≤ 2 columns, every lane tests its own
// bloom-passing doc, the bit-reversed ballot is the mask.
using st::ops::fork_filters::ExactClauses;

template <int32_t STACK_SIZE, bool PER_CLUSTER>
__global__ void bloom_recheck_warp_payload_kernel(
    const int64_t* __restrict__ bloom_index,
    int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    EncodedQueryPlans plans,
    const int64_t* __restrict__ query_plan_index,
    int64_t k,
    int64_t p_k,
    ExactClauses clauses,
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
      const int64_t first_column = doc_start / 64;
      const int64_t pos = doc_start % 64;
      uint64_t mine = 0;
      if (lane == 0 || (lane == 1 && pos > 32)) {
        const auto plan = plans[query_plan_index == nullptr ? row : query_plan_index[row]];
        mine = column_result<STACK_SIZE>(
            bloom_index,
            index_total_column_count,
            bundle_b_offsets,
            first_column + lane,
            k,
            p_k,
            plan);
      }
      uint64_t columns[2];
      columns[0] = __shfl_sync(0xffffffff, mine, 0);
      columns[1] = __shfl_sync(0xffffffff, mine, 1);
      const uint32_t bloom = get_next_32_bit_mask(columns, pos);
      const bool pass =
          ((bloom >> (31 - lane)) & 1) && clauses.pass(row, doc_start + lane);
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

template <int32_t STACK_SIZE>
__global__ void bloom_recheck_remaining_payload_kernel(
    const int64_t* __restrict__ bloom_index,
    int64_t index_total_column_count,
    const int64_t* __restrict__ bundle_b_offsets,
    EncodedQueryPlans plans,
    const int64_t* __restrict__ query_plan_index,
    int64_t k,
    int64_t p_k,
    ExactClauses clauses,
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
    const int32_t remaining = length - handled;
    if (remaining == 0) {
      continue;
    }
    const int32_t doc_start =
        static_cast<int32_t>(cluster_offsets[cluster_ids[i]]) + handled;
    const int64_t first_column = doc_start / 64;
    const bool second = (doc_start + remaining - 1) / 64 != first_column;
    uint64_t mine = 0;
    if (lane == 0 || (lane == 1 && second)) {
      const auto plan = plans[query_plan_index == nullptr ? row : query_plan_index[row]];
      mine = column_result<STACK_SIZE>(
          bloom_index,
          index_total_column_count,
          bundle_b_offsets,
          first_column + lane,
          k,
          p_k,
          plan);
    }
    uint64_t columns[2];
    columns[0] = __shfl_sync(0xffffffff, mine, 0);
    columns[1] = __shfl_sync(0xffffffff, mine, 1);
    if (lane >= remaining) {
      continue;
    }
    const int32_t doc = doc_start + lane;
    const int32_t out_offset = i == 0 ? 0 : remaining_length_cumsum[i - 1];
    const int32_t row_first =
        row == 0 ? 0 : remaining_length_cumsum[row * clusters_per_row - 1];
    RemainingPayload p;
    p.doc_index = doc;
    p.cluster_ids_row = row;
    p.write_index = static_cast<int32_t>(row * max_tensor_size_per_row) +
        warp_rounded_length_cumsum[row * clusters_per_row + clusters_per_row - 1] +
        (out_offset - row_first) + lane;
    p.filtering_mask = get_bit_64_bit_mask(columns, doc - first_column * 64) &&
        clauses.pass(row, doc);
    reinterpret_cast<int4*>(payloads)[out_offset + lane] = p.payload;
  }
}

// Copies the device data pointers of a list of tensors into a single device
// buffer with one HostToDevice copy, exposing them to kernels as a T**.
template <typename T>
class CudaTensorPointerListCooker {
 public:
  using pointer = const T*;
  CudaTensorPointerListCooker(
      const std::vector<Tensor>& tensor_list,
      const at::Device& device) {
    std::vector<pointer> pointerList;
    pointerList.reserve(tensor_list.size());
    for (const auto& tensor : tensor_list) {
      pointerList.push_back(tensor.data_ptr<T>());
    }
    size_t size_to_copy = sizeof(pointer) * pointerList.size();
    m_cudaVector = at::empty(
        {static_cast<int64_t>(size_to_copy)},
        c10::TensorOptions().dtype(at::kChar).device(device),
        at::MemoryFormat::Contiguous);
    m_pointerListPtr =
        reinterpret_cast<pointer*>(m_cudaVector.data_ptr<int8_t>());
    AT_CUDA_CHECK(cudaMemcpyAsync(
        reinterpret_cast<void*>(m_pointerListPtr),
        reinterpret_cast<void*>(pointerList.data()),
        size_to_copy,
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
  }
  const pointer* get() {
    return m_pointerListPtr;
  }

 private:
  Tensor m_cudaVector;
  pointer* m_pointerListPtr;
};

// Multi-chunk variant of the column-info kernel: computes per-cluster column
// info for clusters spread across a list of chunks (indexed by chunk).
__global__ void generate_column_info_for_cluster_list_kernel(
    int64_t num_chunk,
    int64_t num_clusters,
    const int64_t* const* __restrict__ selected_cluster_offsets_list,
    const int64_t* const* __restrict__ selected_cluster_lengths_list,
    const int32_t* __restrict__ per_chunk_cluster_counts_cumsum,
    const int32_t* __restrict__ cluster_assigned_chunk_indices,
    PackedTensorAccessor64<int32_t, 1> column_counts,
    PackedTensorAccessor64<int32_t, 1> start_column_ids,
    PackedTensorAccessor64<int8_t, 1> first_item_offsets_in_column,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t cluster_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cluster_index < num_clusters;
       cluster_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int32_t chunk_idx = cluster_assigned_chunk_indices[cluster_index];
    int32_t local_cluster_idx = cluster_index -
        (chunk_idx == 0
             ? 0
             : __ldg(per_chunk_cluster_counts_cumsum + chunk_idx - 1));
    int64_t cluster_len =
        __ldg((selected_cluster_lengths_list[chunk_idx]) + local_cluster_idx);
    int64_t cluster_start =
        __ldg((selected_cluster_offsets_list[chunk_idx]) + local_cluster_idx);
    int8_t offset_in_col =
        static_cast<int8_t>(cluster_start % C_BITS_IN_UINT64);
    start_column_ids[cluster_index] =
        static_cast<int32_t>(cluster_start / C_BITS_IN_UINT64);
    first_item_offsets_in_column[cluster_index] = offset_in_col;
    column_counts[cluster_index] = cluster_len > 0
        ? static_cast<int32_t>(
              (cluster_len + offset_in_col + C_BITS_IN_UINT64 - 1) /
              C_BITS_IN_UINT64)
        : 0;
  }
}

// Multi-chunk variant of process_documents: one thread per output column across
// all chunks; resolves the owning chunk/query and evaluates its query plan.
template <int32_t STACK_SIZE, bool is_bloom_index_v2>
__global__ void
process_documents_on_assigned_cluster_columns_on_multiple_chunk_kernel(
    int64_t num_queries,
    const int64_t* const* __restrict__ bloom_index_list,
    const int64_t* __restrict__ index_total_column_count_list,
    const int64_t* const* __restrict__ bundle_b_offsets_list,
    const int32_t* __restrict__ cluster_assigned_chunk_index,
    const int32_t* __restrict__ per_chunk_cluster_counts_cumsum,
    int64_t total_column_count,
    const int32_t* __restrict__ column_assigned_cluster_index,
    const int32_t* __restrict__ column_counts_cumsum,
    const int32_t* __restrict__ start_column_ids,
    PackedTensorAccessor64<int64_t, 1> column_mask_response,
    const QueryPlanCuda<is_bloom_index_v2>* __restrict__ query_plan,
    int64_t k,
    int64_t p_k,
    const uint64_t* __restrict__ query_plan_index_ptr,
    TORCH_DSA_KERNEL_ARGS) {
  for (int64_t column_index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       column_index < total_column_count;
       column_index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int32_t assigned_cluster_index =
        column_assigned_cluster_index[column_index];
    int32_t assigned_chunk_index =
        cluster_assigned_chunk_index[assigned_cluster_index];
    int32_t chunk_start_cluster_idx =
        (assigned_chunk_index == 0
             ? 0
             : per_chunk_cluster_counts_cumsum[assigned_chunk_index - 1]);
    int32_t cluster_idx_in_chunk =
        assigned_cluster_index - chunk_start_cluster_idx;
    int32_t chunk_cluster_count =
        per_chunk_cluster_counts_cumsum[assigned_chunk_index] -
        chunk_start_cluster_idx;
    int64_t probe = chunk_cluster_count / num_queries;
    int64_t query_id = cluster_idx_in_chunk / probe;
    query_id = query_plan_index_ptr == nullptr
        ? query_id
        : static_cast<int64_t>(query_plan_index_ptr[query_id]);
    int64_t col_idx_in_local_cluster = column_index -
        (assigned_cluster_index == 0
             ? 0
             : column_counts_cumsum[assigned_cluster_index - 1]);
    int64_t column_id =
        start_column_ids[assigned_cluster_index] + col_idx_in_local_cluster;
    uint64_t result = run_query_plan<STACK_SIZE, is_bloom_index_v2>(
        bloom_index_list[assigned_chunk_index],
        index_total_column_count_list[assigned_chunk_index],
        bundle_b_offsets_list == nullptr
            ? nullptr
            : bundle_b_offsets_list[assigned_chunk_index],
        column_id,
        k,
        p_k,
        query_plan[query_id]);
    store<int64_t>(column_mask_response.data() + column_index, result, 0);
  }
}

} // namespace

template <bool is_bloom_index_v2>
Tensor process_documents_on_assigned_cluster_columns(
    const Tensor& bloom_index,
    const Tensor* bloom_bundle_b_offsets,
    const Tensor& column_assigned_cluster_index,
    const Tensor& column_counts_cumsum,
    const Tensor& start_column_ids,
    int64_t cluster_count_per_query,
    const QueryPlanCuda<is_bloom_index_v2>* query_plan,
    int64_t k,
    int64_t onebit_position_k,
    int32_t query_plans_max_stack_size,
    const uint64_t* query_plan_index_ptr) {
  int64_t* bundle_b_offsets_ptr = nullptr;
  int64_t index_total_column_count;
  if constexpr (is_bloom_index_v2) {
    TORCH_CHECK(bloom_bundle_b_offsets != nullptr);
    bundle_b_offsets_ptr = bloom_bundle_b_offsets->data_ptr<int64_t>();
    index_total_column_count =
        (bloom_bundle_b_offsets->numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  } else {
    index_total_column_count = bloom_index.size(1);
  }
  constexpr int64_t kThreads = 256L;
  int64_t kBlocks = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
          ->multiProcessorCount;
  int64_t total_column_count = column_assigned_cluster_index.numel();
  int64_t block_size = kThreads;
  int64_t grid_size =
      std::min(kBlocks, (total_column_count + block_size - 1) / block_size);
  Tensor column_mask_response = at::zeros(
      {total_column_count},
      c10::TensorOptions().dtype(at::kLong).device(bloom_index.device()));
#define LAUNCH_KERNEL_WITH_STACK_SIZE(STACK_SIZE)                    \
  TORCH_DSA_KERNEL_LAUNCH(                                           \
      (process_documents_on_assigned_cluster_columns_kernel<         \
          STACK_SIZE,                                                \
          is_bloom_index_v2>),                                       \
      grid_size,                                                     \
      block_size,                                                    \
      0,                                                             \
      at::cuda::getCurrentCUDAStream(),                              \
      bloom_index.data_ptr<int64_t>(),                               \
      index_total_column_count,                                      \
      bundle_b_offsets_ptr,                                          \
      column_assigned_cluster_index.packed_accessor64<int32_t, 1>(), \
      column_counts_cumsum.packed_accessor64<int32_t, 1>(),          \
      start_column_ids.packed_accessor64<int32_t, 1>(),              \
      cluster_count_per_query,                                       \
      column_mask_response.packed_accessor64<int64_t, 1>(),          \
      query_plan,                                                    \
      k,                                                             \
      onebit_position_k,                                             \
      query_plan_index_ptr);
  if (query_plans_max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) {
    LAUNCH_KERNEL_WITH_STACK_SIZE(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);
  } else {
    LAUNCH_KERNEL_WITH_STACK_SIZE(0);
  }
#undef LAUNCH_KERNEL_WITH_STACK_SIZE
  return column_mask_response;
}

std::tuple<Tensor, Tensor, Tensor> generate_column_info_for_clusters(
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths) {
  // this function will be just one step of a kernel, so I didn't add
  // device guard in this function, the guard will be the caller
  // function's responsibility.
  constexpr int64_t kThreads = 256L;
  int64_t kBlocks = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
          ->multiProcessorCount;
  int64_t total_cluster_count = selected_cluster_lengths.numel();
  int64_t block_size = total_cluster_count <= 128 ? 128L : kThreads;
  int64_t grid_size =
      std::min(kBlocks, (total_cluster_count + block_size - 1) / block_size);
  auto device = selected_cluster_lengths.device();
  Tensor column_counts = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor start_column_ids = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor first_item_offsets_in_column = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kChar).device(device));
  at::cuda::CUDAGuard device_guard(device);
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_DSA_KERNEL_LAUNCH(
      generate_column_info_for_clusters_kernel,
      grid_size,
      block_size,
      0,
      stream,
      selected_cluster_offsets.data_ptr<int64_t>(),
      selected_cluster_lengths.data_ptr<int64_t>(),
      total_cluster_count,
      column_counts.data_ptr<int32_t>(),
      start_column_ids.data_ptr<int32_t>(),
      first_item_offsets_in_column.data_ptr<int8_t>());
  return std::make_tuple(
      std::move(column_counts),
      std::move(start_column_ids),
      std::move(first_item_offsets_in_column));
}

template <
    bool is_bloom_index_v2,
    typename PLANS = const QueryPlanCuda<is_bloom_index_v2>*>
Tensor bloom_index_search_common(
    const Tensor& bloom_index,
    const Tensor*
        bloom_bundle_b_offsets, // only used when is_bloom_index_v2 is true
    PLANS query_plan_cuda,
    size_t query_plans_size,
    int32_t query_plans_max_stack_size,
    int64_t k,
    int64_t onebit_position_k, // only used when is_bloom_index_v2 is true
    bool return_bool_mask) {
  int64_t* bundle_b_offsets_ptr = nullptr;
  int64_t total_column_count;
  if constexpr (is_bloom_index_v2) {
    TORCH_CHECK(bloom_bundle_b_offsets != nullptr);
    bundle_b_offsets_ptr = bloom_bundle_b_offsets->data_ptr<int64_t>();
    total_column_count =
        (bloom_bundle_b_offsets->numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  } else {
    total_column_count = bloom_index.size(1);
  }
  constexpr int64_t kThreads = 256L;
  auto block_size = kThreads;
  auto grid_size = std::min(
      (total_column_count * static_cast<int64_t>(query_plans_size) +
       block_size - 1) /
          block_size,
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount);

  int64_t document_mask_alloc_size;
  at::ScalarType document_mask_alloc_type;
  if (return_bool_mask) {
    document_mask_alloc_size =
        static_cast<int64_t>(total_column_count * C_BITS_IN_UINT64);
    document_mask_alloc_type = at::kBool;
  } else {
    document_mask_alloc_size = static_cast<int64_t>(total_column_count);
    document_mask_alloc_type = at::kLong;
  }

  Tensor document_mask;
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(bloom_index.device().index());
  document_mask = at::empty(
      {static_cast<int64_t>(query_plans_size), document_mask_alloc_size},
      c10::TensorOptions()
          .dtype(document_mask_alloc_type)
          .device(bloom_index.device()));

#define LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(RETURN_TYPE, STACK_SIZE) \
  {                                                                        \
    TORCH_DSA_KERNEL_LAUNCH(                                               \
        (process_documents<                                                \
            RETURN_TYPE,                                                   \
            STACK_SIZE,                                                    \
            is_bloom_index_v2,                                             \
            PLANS>),                                                       \
        grid_size,                                                         \
        block_size,                                                        \
        0,                                                                 \
        at::cuda::getCurrentCUDAStream(),                                  \
        bloom_index.data_ptr<int64_t>(),                                   \
        nullptr,                                                           \
        total_column_count,                                                \
        bundle_b_offsets_ptr,                                              \
        document_mask.packed_accessor64<RETURN_TYPE, 2>(),                 \
        query_plan_cuda,                                                   \
        k,                                                                 \
        onebit_position_k);                                                \
  }

#define LAUNCH_KERNEL_WITH_RETURN_TYPE(RETURN_TYPE)                          \
  {                                                                          \
    if (query_plans_max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) { \
      LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(                             \
          RETURN_TYPE, C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);                 \
    } else {                                                                 \
      LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE(RETURN_TYPE, 0);             \
    }                                                                        \
  }

  if (return_bool_mask) {
    LAUNCH_KERNEL_WITH_RETURN_TYPE(bool);
  } else {
    LAUNCH_KERNEL_WITH_RETURN_TYPE(int64_t);
  }

#undef LAUNCH_KERNEL_WITH_RETURN_TYPE_STACK_SIZE
#undef LAUNCH_KERNEL_WITH_RETURN_TYPE

  return document_mask;
}

Tensor bloom_index_search_batch_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    int64_t k,
    int64_t onebit_hash_k,
    bool return_bool_mask) {
  TORCH_CHECK(bloom_index.is_cuda());
  TORCH_CHECK(bloom_index.dim() == 1);
  TORCH_CHECK(bloom_index.is_contiguous());

  // Move plans to CPU for decoding (decode_query_plan uses from_blob +
  // data_ptr)
  auto plans_data_cpu = bloom_query_plans_data.cpu();
  auto plans_offsets_cpu = bloom_query_plans_offsets.cpu();

  int64_t query_plans_size = plans_offsets_cpu.numel() / 8;
  TensorCreationVisitor<true> tensor_creation_visitor(bloom_index.device());

  std::vector<QueryPlan<true>> query_plans;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    Tensor operators_t, offsets_t, parameters_t, oneBitsPosition_t;
    decode_query_plan<true>(
        plans_data_cpu,
        plans_offsets_cpu,
        i * 8,
        operators_t,
        offsets_t,
        parameters_t,
        oneBitsPosition_t);
    QueryPlan<true> plan;
    plan.operators.clear();
    plan.offsets.clear(); // remove default {0} from constructor
    auto* ops_ptr = operators_t.data_ptr<uint8_t>();
    for (int64_t j = 0; j < operators_t.numel(); ++j) {
      plan.operators.push_back(static_cast<Operator>(ops_ptr[j]));
    }
    auto* offsets_ptr = offsets_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < offsets_t.numel(); ++j) {
      plan.offsets.push_back(offsets_ptr[j]);
    }
    auto* params_ptr = parameters_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < parameters_t.numel(); ++j) {
      plan.parameters.push_back(params_ptr[j]);
    }
    auto* onebits_ptr =
        reinterpret_cast<const uint64_t*>(oneBitsPosition_t.data_ptr());
    for (int64_t j = 0; j < oneBitsPosition_t.numel(); ++j) {
      plan.oneBitsPosition.push_back(onebits_ptr[j]);
    }
    query_plans.push_back(std::move(plan));
  }

  int32_t query_plans_max_stack_size = get_max_stack_size<true>(query_plans);
  for (int64_t i = 0; i < query_plans_size; ++i) {
    tensor_creation_visitor.add_size_and_data(query_plans[i]);
  }
  tensor_creation_visitor.add_size_without_data(
      query_plans_size * sizeof(QueryPlanCuda<true>));

  std::vector<QueryPlanCuda<true>> query_plan_cudas_host;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    query_plan_cudas_host.push_back(
        tensor_creation_visitor.visit(query_plans[i]));
  }
  const QueryPlanCuda<true>* query_plan_cuda =
      tensor_creation_visitor.visit_with_data(query_plan_cudas_host);

  return bloom_index_search_common<true>(
      bloom_index,
      &bloom_bundle_b_offsets,
      query_plan_cuda,
      query_plans_size,
      query_plans_max_stack_size,
      k,
      onebit_hash_k,
      return_bool_mask);
}

// Prepare device-side query plans from encoded query-plan data. The returned
// pointer is owned by `visitor`, which must outlive every kernel that uses it.
static const QueryPlanCuda<true>* prepare_query_plan_cuda(
    TensorCreationVisitor<true>& visitor,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    int32_t& query_plans_max_stack_size) {
  auto plans_data_cpu = bloom_query_plans_data.cpu();
  auto plans_offsets_cpu = bloom_query_plans_offsets.cpu();

  int64_t query_plans_size = plans_offsets_cpu.numel() / 8;
  std::vector<QueryPlan<true>> query_plans;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    Tensor operators_t, offsets_t, parameters_t, oneBitsPosition_t;
    decode_query_plan<true>(
        plans_data_cpu,
        plans_offsets_cpu,
        i * 8,
        operators_t,
        offsets_t,
        parameters_t,
        oneBitsPosition_t);
    QueryPlan<true> plan;
    plan.operators.clear();
    plan.offsets.clear();
    auto* ops_ptr = operators_t.data_ptr<uint8_t>();
    for (int64_t j = 0; j < operators_t.numel(); ++j) {
      plan.operators.push_back(static_cast<Operator>(ops_ptr[j]));
    }
    auto* offsets_ptr = offsets_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < offsets_t.numel(); ++j) {
      plan.offsets.push_back(offsets_ptr[j]);
    }
    auto* params_ptr = parameters_t.data_ptr<int32_t>();
    for (int64_t j = 0; j < parameters_t.numel(); ++j) {
      plan.parameters.push_back(params_ptr[j]);
    }
    auto* onebits_ptr =
        reinterpret_cast<const uint64_t*>(oneBitsPosition_t.data_ptr());
    for (int64_t j = 0; j < oneBitsPosition_t.numel(); ++j) {
      plan.oneBitsPosition.push_back(onebits_ptr[j]);
    }
    query_plans.push_back(std::move(plan));
  }

  query_plans_max_stack_size = get_max_stack_size<true>(query_plans);
  for (int64_t i = 0; i < query_plans_size; ++i) {
    visitor.add_size_and_data(query_plans[i]);
  }
  visitor.add_size_without_data(query_plans_size * sizeof(QueryPlanCuda<true>));

  std::vector<QueryPlanCuda<true>> query_plan_cudas_host;
  for (int64_t i = 0; i < query_plans_size; ++i) {
    query_plan_cudas_host.push_back(visitor.visit(query_plans[i]));
  }
  return visitor.visit_with_data(query_plan_cudas_host);
}

// Run the partial-response search for one chunk given already-prepared device
// query plans. Shared by the single- and multiple-index CUDA ops.
static std::tuple<Tensor, Tensor, Tensor>
bloom_index_search_return_partial_response_one_chunk_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const QueryPlanCuda<true>* query_plan_cuda,
    int32_t query_plans_max_stack_size,
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths,
    int64_t k,
    int64_t onebit_hash_k,
    const uint64_t* query_plan_index_ptr) {
  auto [column_counts, start_column_ids, first_item_offsets_in_column] =
      generate_column_info_for_clusters(
          selected_cluster_offsets, selected_cluster_lengths);

  Tensor assigned_cluster_index_tensor =
      torch::arange(
          selected_cluster_lengths.numel(),
          selected_cluster_lengths.options().dtype(at::kInt))
          .repeat_interleave(column_counts);
  Tensor column_counts_cumsum_tensor =
      column_counts.cumsum(/*dim=*/0, /*dtype=*/at::kInt);

  Tensor column_mask_response =
      process_documents_on_assigned_cluster_columns<true>(
          bloom_index,
          &bloom_bundle_b_offsets,
          assigned_cluster_index_tensor,
          column_counts_cumsum_tensor,
          start_column_ids,
          selected_cluster_lengths.size(1),
          query_plan_cuda,
          k,
          onebit_hash_k,
          query_plans_max_stack_size,
          query_plan_index_ptr);
  return std::make_tuple(
      std::move(column_counts_cumsum_tensor),
      std::move(first_item_offsets_in_column),
      std::move(column_mask_response));
}

std::tuple<Tensor, Tensor, Tensor>
bloom_index_search_batch_return_partial_response_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths,
    int64_t k,
    int64_t onebit_hash_k,
    const std::optional<Tensor>& query_plan_index) {
  TORCH_CHECK(bloom_index.is_cuda());
  TORCH_CHECK(bloom_index.dim() == 1);
  TORCH_CHECK(bloom_index.is_contiguous());
  TORCH_CHECK(selected_cluster_offsets.is_cuda());
  TORCH_CHECK(selected_cluster_lengths.is_cuda());

  TensorCreationVisitor<true> tensor_creation_visitor(bloom_index.device());
  int32_t query_plans_max_stack_size = 0;
  const QueryPlanCuda<true>* query_plan_cuda = prepare_query_plan_cuda(
      tensor_creation_visitor,
      bloom_query_plans_data,
      bloom_query_plans_offsets,
      query_plans_max_stack_size);

  const uint64_t* query_plan_index_ptr = nullptr;
  if (query_plan_index) {
    TORCH_CHECK(query_plan_index->size(0) == selected_cluster_lengths.size(0));
    TORCH_CHECK(query_plan_index->is_cuda());
    TORCH_CHECK(query_plan_index->dim() == 1);
    query_plan_index_ptr = reinterpret_cast<const uint64_t*>(
        query_plan_index->data_ptr<int64_t>());
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(bloom_index.get_device());

  return bloom_index_search_return_partial_response_one_chunk_cuda(
      bloom_index,
      bloom_bundle_b_offsets,
      query_plan_cuda,
      query_plans_max_stack_size,
      selected_cluster_offsets,
      selected_cluster_lengths,
      k,
      onebit_hash_k,
      query_plan_index_ptr);
}

// bloom_index_search_batch over prepared plans (SilverTorch.prepare_queries uploads the
// parser's encoding once, outside the timed call): one plan per row, no host decode, no
// copy, no sync.
Tensor bloom_index_search_batch_prepared_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    int64_t k,
    int64_t onebit_hash_k,
    bool return_bool_mask) {
  TORCH_CHECK(bloom_index.is_cuda());
  TORCH_CHECK(bloom_index.dim() == 1);
  TORCH_CHECK(bloom_index.is_contiguous());
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(bloom_index.get_device());
  return bloom_index_search_common<true, EncodedQueryPlans>(
      bloom_index,
      &bloom_bundle_b_offsets,
      encoded_plans(plans, plan_count),
      static_cast<size_t>(plan_count),
      static_cast<int32_t>(max_stack_size),
      k,
      onebit_hash_k,
      return_bool_mask);
}

// bloom_index_search_batch_return_partial_response over prepared plans, with no host sync:
// column_mask_response has the caller's static capacity max_columns (a row's probed clusters
// span at most max_tensor_size_per_row / 64 + 2 * P columns); its tail past the device total
// stays zero and is never read (the scorer addresses columns through the cumsum).
std::tuple<Tensor, Tensor, Tensor>
bloom_index_search_batch_return_partial_response_prepared_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths,
    int64_t k,
    int64_t onebit_hash_k,
    const std::optional<Tensor>& query_plan_index,
    int64_t max_columns) {
  TORCH_CHECK(bloom_index.is_cuda());
  TORCH_CHECK(bloom_index.dim() == 1);
  TORCH_CHECK(bloom_index.is_contiguous());
  TORCH_CHECK(selected_cluster_offsets.is_cuda());
  TORCH_CHECK(selected_cluster_lengths.is_cuda());
  TORCH_CHECK(selected_cluster_lengths.dim() == 2);
  const int64_t* query_plan_index_ptr = nullptr;
  if (query_plan_index) {
    TORCH_CHECK(query_plan_index->size(0) == selected_cluster_lengths.size(0));
    TORCH_CHECK(query_plan_index->is_cuda());
    TORCH_CHECK(query_plan_index->dim() == 1);
    TORCH_CHECK(query_plan_index->scalar_type() == at::kLong);
    query_plan_index_ptr = query_plan_index->const_data_ptr<int64_t>();
  }
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(bloom_index.get_device());

  auto [column_counts, start_column_ids, first_item_offsets_in_column] =
      generate_column_info_for_clusters(
          selected_cluster_offsets, selected_cluster_lengths);
  Tensor column_counts_cumsum =
      column_counts.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
  Tensor column_mask_response = at::zeros(
      {max_columns},
      c10::TensorOptions().dtype(at::kLong).device(bloom_index.device()));

  const int64_t cluster_count = selected_cluster_lengths.numel();
  constexpr int64_t kThreads = 256L;
  const int64_t grid_size = std::min(
      128L *
          CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
              ->multiProcessorCount,
      (max_columns + kThreads - 1) / kThreads);
  const int64_t index_total_column_count =
      (bloom_bundle_b_offsets.numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  if (cluster_count > 0 && grid_size > 0) {
#define LAUNCH_PROBED_COLUMNS(STACK_SIZE)                          \
  TORCH_DSA_KERNEL_LAUNCH(                                         \
      process_documents_on_probed_columns_kernel<STACK_SIZE>,      \
      grid_size,                                                   \
      kThreads,                                                    \
      0,                                                           \
      at::cuda::getCurrentCUDAStream(),                            \
      bloom_index.const_data_ptr<int64_t>(),                       \
      index_total_column_count,                                    \
      bloom_bundle_b_offsets.const_data_ptr<int64_t>(),            \
      column_counts_cumsum.const_data_ptr<int32_t>(),              \
      start_column_ids.const_data_ptr<int32_t>(),                  \
      static_cast<int32_t>(cluster_count),                         \
      selected_cluster_lengths.size(1),                            \
      column_mask_response.mutable_data_ptr<int64_t>(),            \
      encoded_plans(plans, plan_count),                            \
      k,                                                           \
      onebit_hash_k,                                               \
      query_plan_index_ptr);
    if (max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) {
      LAUNCH_PROBED_COLUMNS(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);
    } else {
      LAUNCH_PROBED_COLUMNS(0);
    }
#undef LAUNCH_PROBED_COLUMNS
  }
  return std::make_tuple(
      std::move(column_counts_cumsum),
      std::move(first_item_offsets_in_column),
      std::move(column_mask_response));
}

// C8 host side: (warp payloads, warp capacity, remainder payloads, remainder capacity) for the
// partial bloom path, the capacities as C3's (rows · max / 32 warps; rows · min(31 · P, max)
// remainder docs); the warp and remainder totals stay on the device (the cumsums' last elements).
std::tuple<Tensor, int64_t, Tensor, int64_t> bloom_partial_payloads_prepared_cuda(
    const Tensor& bloom_index,
    const Tensor& bloom_bundle_b_offsets,
    const Tensor& plans,
    int64_t plan_count,
    int64_t max_stack_size,
    const std::optional<Tensor>& query_plan_index,
    int64_t k,
    int64_t onebit_hash_k,
    const Tensor& cluster_offsets,
    const Tensor& cluster_ids,
    const Tensor& cluster_length,
    const Tensor& warp_rounded_length_cumsum,
    const Tensor& remaining_length_cumsum,
    const Tensor& warp_size_cumsum,
    int64_t max_tensor_size_per_row,
    const std::optional<std::tuple<Tensor, Tensor, Tensor>>& recheck) {
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
  const int64_t* plan_index_ptr =
      query_plan_index ? query_plan_index->const_data_ptr<int64_t>() : nullptr;
  const int64_t index_total_column_count =
      (bloom_bundle_b_offsets.numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE;
  const int64_t max_grid = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream();
  const EncodedQueryPlans encoded = encoded_plans(plans, plan_count);
  if (recheck) {
    const auto& [attrs, reverse, query_attrs] = *recheck;
    TORCH_CHECK(attrs.dim() == 3 && attrs.scalar_type() == at::kLong && attrs.is_contiguous());
    TORCH_CHECK(reverse.scalar_type() == at::kBool && reverse.is_contiguous());
    TORCH_CHECK(query_attrs.scalar_type() == at::kLong && query_attrs.is_contiguous());
    const ExactClauses clauses{
        attrs.const_data_ptr<int64_t>(),
        attrs.size(1),
        attrs.size(2),
        reverse.const_data_ptr<bool>(),
        query_attrs.const_data_ptr<int64_t>()};
#define LAUNCH_RECHECK_PAYLOADS(STACK_SIZE, PER_CLUSTER)                          \
  {                                                                               \
    bloom_recheck_warp_payload_kernel<STACK_SIZE, PER_CLUSTER><<<                 \
        std::min(max_grid, ((PER_CLUSTER ? n_clusters : warps) * 32 + 255) / 256), \
        256,                                                                      \
        0,                                                                        \
        stream>>>(                                                                \
        bloom_index.const_data_ptr<int64_t>(),                                    \
        index_total_column_count,                                                 \
        bloom_bundle_b_offsets.const_data_ptr<int64_t>(),                         \
        encoded,                                                                  \
        plan_index_ptr,                                                           \
        k,                                                                        \
        onebit_hash_k,                                                            \
        clauses,                                                                  \
        cluster_offsets.const_data_ptr<int64_t>(),                                \
        cluster_ids.const_data_ptr<int64_t>(),                                    \
        per_row,                                                                  \
        static_cast<int32_t>(n_clusters),                                         \
        warp_rounded_length_cumsum.const_data_ptr<int32_t>(),                     \
        warp_size_cumsum.const_data_ptr<int32_t>(),                               \
        max_tensor_size_per_row,                                                  \
        reinterpret_cast<WarpPayload*>(warp_payloads.mutable_data_ptr<int8_t>())); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
    bloom_recheck_remaining_payload_kernel<STACK_SIZE><<<                         \
        std::min(max_grid, (n_clusters * 32 + 255) / 256),                             \
        256,                                                                      \
        0,                                                                        \
        stream>>>(                                                                \
        bloom_index.const_data_ptr<int64_t>(),                                    \
        index_total_column_count,                                                 \
        bloom_bundle_b_offsets.const_data_ptr<int64_t>(),                         \
        encoded,                                                                  \
        plan_index_ptr,                                                           \
        k,                                                                        \
        onebit_hash_k,                                                            \
        clauses,                                                                  \
        cluster_offsets.const_data_ptr<int64_t>(),                                \
        cluster_ids.const_data_ptr<int64_t>(),                                    \
        cluster_length.const_data_ptr<int64_t>(),                                 \
        per_row,                                                                  \
        n_clusters,                                                               \
        warp_rounded_length_cumsum.const_data_ptr<int32_t>(),                     \
        remaining_length_cumsum.const_data_ptr<int32_t>(),                        \
        max_tensor_size_per_row,                                                  \
        reinterpret_cast<RemainingPayload*>(                                      \
            remaining_payloads.mutable_data_ptr<int8_t>()));                      \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
  }
    // wide probes (capacity >= 40 full warps a cluster): a warp per cluster
    const bool per_cluster = warps >= 40 * n_clusters && n_clusters >= 1024;
    if (max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) {
      if (per_cluster) {
        LAUNCH_RECHECK_PAYLOADS(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL, true);
      } else {
        LAUNCH_RECHECK_PAYLOADS(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL, false);
      }
    } else if (per_cluster) {
      LAUNCH_RECHECK_PAYLOADS(0, true);
    } else {
      LAUNCH_RECHECK_PAYLOADS(0, false);
    }
#undef LAUNCH_RECHECK_PAYLOADS
    return {warp_payloads, warps, remaining_payloads, remainders};
  }
#define LAUNCH_BLOOM_PAYLOADS(STACK_SIZE)                                         \
  {                                                                               \
    bloom_warp_payload_kernel<STACK_SIZE><<<                                      \
        std::min(max_grid, (warps + 255) / 256),                                  \
        256,                                                                      \
        0,                                                                        \
        stream>>>(                                                                \
        bloom_index.const_data_ptr<int64_t>(),                                    \
        index_total_column_count,                                                 \
        bloom_bundle_b_offsets.const_data_ptr<int64_t>(),                         \
        encoded,                                                                  \
        plan_index_ptr,                                                           \
        k,                                                                        \
        onebit_hash_k,                                                            \
        cluster_offsets.const_data_ptr<int64_t>(),                                \
        cluster_ids.const_data_ptr<int64_t>(),                                    \
        per_row,                                                                  \
        static_cast<int32_t>(n_clusters),                                         \
        warp_rounded_length_cumsum.const_data_ptr<int32_t>(),                     \
        warp_size_cumsum.const_data_ptr<int32_t>(),                               \
        max_tensor_size_per_row,                                                  \
        reinterpret_cast<WarpPayload*>(warp_payloads.mutable_data_ptr<int8_t>())); \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
    bloom_remaining_payload_kernel<STACK_SIZE><<<                                 \
        std::min(max_grid, (n_clusters * 32 + 255) / 256),                             \
        256,                                                                      \
        0,                                                                        \
        stream>>>(                                                                \
        bloom_index.const_data_ptr<int64_t>(),                                    \
        index_total_column_count,                                                 \
        bloom_bundle_b_offsets.const_data_ptr<int64_t>(),                         \
        encoded,                                                                  \
        plan_index_ptr,                                                           \
        k,                                                                        \
        onebit_hash_k,                                                            \
        cluster_offsets.const_data_ptr<int64_t>(),                                \
        cluster_ids.const_data_ptr<int64_t>(),                                    \
        cluster_length.const_data_ptr<int64_t>(),                                 \
        per_row,                                                                  \
        n_clusters,                                                               \
        warp_rounded_length_cumsum.const_data_ptr<int32_t>(),                     \
        remaining_length_cumsum.const_data_ptr<int32_t>(),                        \
        max_tensor_size_per_row,                                                  \
        reinterpret_cast<RemainingPayload*>(                                      \
            remaining_payloads.mutable_data_ptr<int8_t>()));                      \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
  }
  if (max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) {
    LAUNCH_BLOOM_PAYLOADS(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);
  } else {
    LAUNCH_BLOOM_PAYLOADS(0);
  }
#undef LAUNCH_BLOOM_PAYLOADS
  return {warp_payloads, warps, remaining_payloads, remainders};
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, std::vector<int64_t>>
generate_column_info_for_cluster_list(
    const std::vector<Tensor>& selected_cluster_offsets_list,
    const std::vector<Tensor>& selected_cluster_lengths_list,
    const at::Device& device) {
  // this function will be just one step of a kernel, so I didn't add
  // device guard in this function, the guard will be the caller
  // function's responsibility.
  CudaTensorPointerListCooker<int64_t> selected_cluster_offsets_list_cooker(
      selected_cluster_offsets_list, device);
  CudaTensorPointerListCooker<int64_t> selected_cluster_lengths_list_cooker(
      selected_cluster_lengths_list, device);
  std::vector<int64_t> per_chunk_cluster_count_list;
  per_chunk_cluster_count_list.reserve(selected_cluster_lengths_list.size());
  for (auto selected_cluster_offsets : selected_cluster_lengths_list) {
    per_chunk_cluster_count_list.push_back(
        static_cast<int64_t>(selected_cluster_offsets.numel()));
  }
  Tensor per_chunk_cluster_counts = at::tensor(
      per_chunk_cluster_count_list,
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor per_chunk_cluster_counts_cumsum =
      per_chunk_cluster_counts.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
  Tensor cluster_assigned_chunk_indices =
      torch::arange(
          static_cast<int64_t>(selected_cluster_lengths_list.size()),
          c10::TensorOptions().dtype(at::kInt).device(device))
          .repeat_interleave(per_chunk_cluster_counts);
  int64_t total_cluster_count = cluster_assigned_chunk_indices.numel();
  constexpr int64_t kThreads = 256L;
  int64_t kBlocks = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
          ->multiProcessorCount;
  int64_t block_size = total_cluster_count <= 128 ? 128L : kThreads;
  int64_t grid_size =
      std::min(kBlocks, (total_cluster_count + block_size - 1) / block_size);
  Tensor column_counts = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor start_column_ids = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kInt).device(device));
  Tensor first_item_offsets_in_column = at::empty(
      {total_cluster_count},
      c10::TensorOptions().dtype(at::kChar).device(device));
  TORCH_DSA_KERNEL_LAUNCH(
      generate_column_info_for_cluster_list_kernel,
      grid_size,
      block_size,
      0,
      at::cuda::getCurrentCUDAStream(),
      static_cast<int64_t>(selected_cluster_offsets_list.size()),
      total_cluster_count,
      selected_cluster_offsets_list_cooker.get(),
      selected_cluster_lengths_list_cooker.get(),
      per_chunk_cluster_counts_cumsum.data_ptr<int32_t>(),
      cluster_assigned_chunk_indices.data_ptr<int32_t>(),
      column_counts.packed_accessor64<int32_t, 1>(),
      start_column_ids.packed_accessor64<int32_t, 1>(),
      first_item_offsets_in_column.packed_accessor64<int8_t, 1>());
  return std::make_tuple(
      std::move(column_counts),
      std::move(start_column_ids),
      std::move(first_item_offsets_in_column),
      std::move(per_chunk_cluster_counts_cumsum),
      std::move(cluster_assigned_chunk_indices),
      std::move(per_chunk_cluster_count_list));
}

template <bool is_bloom_index_v2>
Tensor process_documents_on_assigned_cluster_columns_multiple_chunks(
    int64_t num_queries,
    const std::vector<Tensor>& bloom_index_list,
    const int64_t* const*
        bloom_bundle_b_offsets_list, // only used when is_bloom_index_v2 is true
    const Tensor& index_total_column_counts, // how many columns in each index.
    const Tensor& cluster_assigned_chunk_index,
    const Tensor& chunk_cluster_counts_cumsum,
    const Tensor& column_assigned_cluster_index,
    const Tensor& column_counts_cumsum,
    const Tensor& start_column_ids,
    const QueryPlanCuda<is_bloom_index_v2>* query_plan,
    int64_t k,
    int64_t onebit_position_k, // only used when is_bloom_index_v2 is true
    int32_t query_plans_max_stack_size,
    const uint64_t* query_plan_index_ptr,
    const at::Device& device) {
  constexpr int64_t kThreads = 256L;
  int64_t kBlocks = 128L *
      CHECK_NOTNULL(at::cuda::getCurrentDeviceProperties())
          ->multiProcessorCount;
  int64_t total_column_count = column_assigned_cluster_index.numel();
  int64_t block_size = kThreads;
  int64_t grid_size =
      std::min(kBlocks, (total_column_count + block_size - 1) / block_size);
  Tensor column_mask_response = at::zeros(
      {total_column_count},
      c10::TensorOptions().dtype(at::kLong).device(device));
  CudaTensorPointerListCooker<int64_t> bloom_index_list_cooker(
      bloom_index_list, device);
#define LAUNCH_KERNEL_WITH_STACK_SIZE(STACK_SIZE)                              \
  TORCH_DSA_KERNEL_LAUNCH(                                                     \
      (process_documents_on_assigned_cluster_columns_on_multiple_chunk_kernel< \
          STACK_SIZE,                                                          \
          is_bloom_index_v2>),                                                 \
      grid_size,                                                               \
      block_size,                                                              \
      0,                                                                       \
      at::cuda::getCurrentCUDAStream(),                                        \
      num_queries,                                                             \
      bloom_index_list_cooker.get(),                                           \
      index_total_column_counts.data_ptr<int64_t>(),                           \
      bloom_bundle_b_offsets_list,                                             \
      cluster_assigned_chunk_index.data_ptr<int32_t>(),                        \
      chunk_cluster_counts_cumsum.data_ptr<int32_t>(),                         \
      total_column_count,                                                      \
      column_assigned_cluster_index.data_ptr<int32_t>(),                       \
      column_counts_cumsum.data_ptr<int32_t>(),                                \
      start_column_ids.data_ptr<int32_t>(),                                    \
      column_mask_response.packed_accessor64<int64_t, 1>(),                    \
      query_plan,                                                              \
      k,                                                                       \
      onebit_position_k,                                                       \
      query_plan_index_ptr);
  if (query_plans_max_stack_size <= C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL) {
    LAUNCH_KERNEL_WITH_STACK_SIZE(C_FAST_ROUTE_STACK_SIZE_FIRST_LEVEL);
  } else {
    LAUNCH_KERNEL_WITH_STACK_SIZE(0);
  }
#undef LAUNCH_KERNEL_WITH_STACK_SIZE
  return column_mask_response;
}

template <bool is_bloom_index_v2>
std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
bloom_index_search_common_return_partial_response_multiple_chunks(
    int64_t num_queries,
    const std::vector<Tensor>& bloom_index_list,
    const int64_t* const*
        bloom_bundle_b_offsets_list, // only used when is_bloom_index_v2 is true
    const Tensor& index_total_column_counts, // how many columns in each index.
    const QueryPlanCuda<is_bloom_index_v2>* query_plan_cuda,
    int32_t query_plans_max_stack_size,
    const std::vector<Tensor>& selected_cluster_offsets_list,
    const std::vector<Tensor>& selected_cluster_lengths_list,
    int64_t k,
    int64_t onebit_position_k, // only used when is_bloom_index_v2 is true)
    const std::optional<Tensor>& query_plan_index,
    const at::Device& device) {
  size_t num_chunk = bloom_index_list.size();
  TORCH_CHECK(selected_cluster_offsets_list.size() == num_chunk);
  TORCH_CHECK(selected_cluster_lengths_list.size() == num_chunk);
  for (size_t i = 0; i < num_chunk; ++i) {
    TORCH_CHECK(bloom_index_list[i].device() == device);
    TORCH_CHECK(bloom_index_list[i].is_contiguous())
    TORCH_CHECK(selected_cluster_offsets_list[i].is_contiguous());
    TORCH_CHECK(selected_cluster_lengths_list[i].is_contiguous());
    TORCH_CHECK(selected_cluster_offsets_list[i].device() == device);
    TORCH_CHECK(selected_cluster_lengths_list[i].device() == device);
    TORCH_CHECK(selected_cluster_offsets_list[i].dim() == 2);
    TORCH_CHECK(selected_cluster_lengths_list[i].dim() == 2);
    TORCH_CHECK(selected_cluster_offsets_list[i].size(0) == num_queries);
    TORCH_CHECK(selected_cluster_lengths_list[i].size(0) == num_queries);
    TORCH_CHECK(
        selected_cluster_offsets_list[i].size(1) ==
        selected_cluster_lengths_list[i].size(1));
  }
  const uint64_t* query_plan_index_ptr = nullptr;
  if (query_plan_index) {
    TORCH_CHECK(query_plan_index->size(0) == num_queries);
    TORCH_CHECK(query_plan_index->device() == device);
    TORCH_CHECK(query_plan_index->dim() == 1);
    TORCH_CHECK(query_plan_index->is_contiguous());
    query_plan_index_ptr = reinterpret_cast<const uint64_t*>(
        query_plan_index->data_ptr<int64_t>());
  }
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(device.index());

  auto
      [column_counts_cumsum,
       start_column_indices,
       first_item_offsets_in_column,
       chunk_cluster_counts_cumsum,
       cluster_assigned_chunk_index,
       cluster_counts_per_chunk_vec] =
          generate_column_info_for_cluster_list(
              selected_cluster_offsets_list,
              selected_cluster_lengths_list,
              device);
  // column_counts_cumsum currently holds the raw per-cluster column counts.
  // Split them per chunk (clusters are grouped by chunk) so we can build each
  // chunk's own column_counts_cumsum and total column count.
  std::vector<Tensor> column_counts_pieces = at::split_with_sizes(
      column_counts_cumsum, cluster_counts_per_chunk_vec, 0);

  // Global column->cluster assignment for the single fused process kernel.
  Tensor column_counts_cumsum_global =
      column_counts_cumsum.cumsum(/*dim=*/0, /*dtype=*/at::kInt);
  Tensor column_assigned_cluster_index =
      torch::arange(
          column_counts_cumsum.numel(),
          c10::TensorOptions().dtype(at::kInt).device(device))
          .repeat_interleave(column_counts_cumsum);

  Tensor column_mask_response =
      process_documents_on_assigned_cluster_columns_multiple_chunks<
          is_bloom_index_v2>(
          num_queries,
          bloom_index_list,
          bloom_bundle_b_offsets_list,
          index_total_column_counts,
          cluster_assigned_chunk_index,
          chunk_cluster_counts_cumsum,
          column_assigned_cluster_index,
          column_counts_cumsum_global,
          start_column_indices,
          query_plan_cuda,
          k,
          onebit_position_k,
          query_plans_max_stack_size,
          query_plan_index_ptr,
          device);

  std::vector<Tensor> column_counts_cumsum_tensor_list;
  column_counts_cumsum_tensor_list.reserve(num_chunk);
  std::vector<int64_t> column_counts_per_chunk_vec;
  column_counts_per_chunk_vec.reserve(num_chunk);
  for (size_t i = 0; i < num_chunk; ++i) {
    Tensor per_chunk_cumsum =
        column_counts_pieces[i].cumsum(/*dim=*/0, /*dtype=*/at::kInt);
    column_counts_per_chunk_vec.push_back(
        per_chunk_cumsum.numel() == 0 ? 0
                                      : per_chunk_cumsum[-1].item<int64_t>());
    column_counts_cumsum_tensor_list.push_back(std::move(per_chunk_cumsum));
  }
  std::vector<Tensor> first_item_offsets_in_column_tensor_list =
      at::split_with_sizes(
          first_item_offsets_in_column, cluster_counts_per_chunk_vec, 0);
  std::vector<Tensor> column_mask_response_tensor_list = at::split_with_sizes(
      column_mask_response, column_counts_per_chunk_vec, 0);
  return std::make_tuple(
      std::move(column_counts_cumsum_tensor_list),
      std::move(first_item_offsets_in_column_tensor_list),
      std::move(column_mask_response_tensor_list));
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
bloom_index_search_batch_return_partial_response_multiple_cuda(
    const std::vector<Tensor>& list_bloom_index,
    const std::vector<Tensor>& list_bloom_bundle_b_offsets,
    const Tensor& bloom_query_plans_data,
    const Tensor& bloom_query_plans_offsets,
    const std::vector<Tensor>& list_selected_cluster_offsets,
    const std::vector<Tensor>& list_selected_cluster_lengths,
    int64_t k,
    int64_t onebit_hash_k,
    const std::optional<Tensor>& query_plan_index,
    bool fuse) {
  const auto n = list_bloom_index.size();
  TORCH_CHECK(
      n > 0,
      "bloom_index_search_batch_return_partial_response_multiple: "
      "list_bloom_index should not be empty.");
  TORCH_CHECK(
      list_bloom_bundle_b_offsets.size() == n,
      "list_bloom_index and list_bloom_bundle_b_offsets should have the same size, got ",
      n,
      " vs ",
      list_bloom_bundle_b_offsets.size());
  TORCH_CHECK(
      list_selected_cluster_offsets.size() == n,
      "list_bloom_index and list_selected_cluster_offsets should have the same size, got ",
      n,
      " vs ",
      list_selected_cluster_offsets.size());
  TORCH_CHECK(
      list_selected_cluster_lengths.size() == n,
      "list_bloom_index and list_selected_cluster_lengths should have the same size, got ",
      n,
      " vs ",
      list_selected_cluster_lengths.size());
  for (size_t i = 0; i < n; ++i) {
    TORCH_CHECK(list_bloom_index[i].is_cuda());
    TORCH_CHECK(list_bloom_index[i].dim() == 1);
    TORCH_CHECK(list_bloom_index[i].is_contiguous());
  }

  const auto& first_bloom_index = list_bloom_index.front();
  TensorCreationVisitor<true> tensor_creation_visitor(
      first_bloom_index.device());
  int32_t query_plans_max_stack_size = 0;
  const QueryPlanCuda<true>* query_plan_cuda = prepare_query_plan_cuda(
      tensor_creation_visitor,
      bloom_query_plans_data,
      bloom_query_plans_offsets,
      query_plans_max_stack_size);

  const uint64_t* query_plan_index_ptr = nullptr;
  if (query_plan_index) {
    TORCH_CHECK(
        query_plan_index->size(0) == list_selected_cluster_lengths[0].size(0));
    TORCH_CHECK(query_plan_index->is_cuda());
    TORCH_CHECK(query_plan_index->dim() == 1);
    query_plan_index_ptr = reinterpret_cast<const uint64_t*>(
        query_plan_index->data_ptr<int64_t>());
  }

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(first_bloom_index.get_device());

  if (fuse) {
    int64_t num_queries = query_plan_index
        ? query_plan_index->size(0)
        : list_selected_cluster_lengths[0].size(0);
    std::vector<int64_t> index_total_column_counts_vec;
    index_total_column_counts_vec.reserve(n);
    for (const auto& bundle_b_offsets : list_bloom_bundle_b_offsets) {
      index_total_column_counts_vec.push_back(
          (bundle_b_offsets.numel() - 1) * C_BLOOM_V2_COL_BUNDLE_SIZE);
    }
    Tensor index_total_column_counts = at::tensor(
        index_total_column_counts_vec,
        c10::TensorOptions().dtype(at::kLong).device(
            first_bloom_index.device()));
    CudaTensorPointerListCooker<int64_t> bloom_bundle_b_offsets_list_cooker(
        list_bloom_bundle_b_offsets, first_bloom_index.device());
    return bloom_index_search_common_return_partial_response_multiple_chunks<
        true>(
        num_queries,
        list_bloom_index,
        bloom_bundle_b_offsets_list_cooker.get(),
        index_total_column_counts,
        query_plan_cuda,
        query_plans_max_stack_size,
        list_selected_cluster_offsets,
        list_selected_cluster_lengths,
        k,
        onebit_hash_k,
        query_plan_index,
        first_bloom_index.device());
  }

  std::vector<Tensor> list_column_counts_cumsum;
  list_column_counts_cumsum.reserve(n);
  std::vector<Tensor> list_first_item_offsets_in_column;
  list_first_item_offsets_in_column.reserve(n);
  std::vector<Tensor> list_column_mask_response;
  list_column_mask_response.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    auto
        [column_counts_cumsum,
         first_item_offsets_in_column,
         column_mask_response] =
            bloom_index_search_return_partial_response_one_chunk_cuda(
                list_bloom_index[i],
                list_bloom_bundle_b_offsets[i],
                query_plan_cuda,
                query_plans_max_stack_size,
                list_selected_cluster_offsets[i],
                list_selected_cluster_lengths[i],
                k,
                onebit_hash_k,
                query_plan_index_ptr);
    list_column_counts_cumsum.push_back(std::move(column_counts_cumsum));
    list_first_item_offsets_in_column.push_back(
        std::move(first_item_offsets_in_column));
    list_column_mask_response.push_back(std::move(column_mask_response));
  }
  return std::make_tuple(
      std::move(list_column_counts_cumsum),
      std::move(list_first_item_offsets_in_column),
      std::move(list_column_mask_response));
}

static std::tuple<Tensor, Tensor, Tensor>
generate_column_info_for_clusters_cuda(
    const Tensor& selected_cluster_offsets,
    const Tensor& selected_cluster_lengths) {
  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(selected_cluster_offsets.get_device());
  auto [column_counts, start_column_ids, first_item_offsets_in_column] =
      generate_column_info_for_clusters(
          selected_cluster_offsets, selected_cluster_lengths);
  return std::make_tuple(
      column_counts, start_column_ids, first_item_offsets_in_column);
}

// Register operators
TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl(
      "generate_column_info_for_clusters",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(generate_column_info_for_clusters_cuda)));
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl(
      "bloom_index_search_batch",
      torch::dispatch(
          c10::DispatchKey::CUDA, TORCH_FN(bloom_index_search_batch_cuda)));
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl(
      "bloom_index_search_batch_return_partial_response",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(bloom_index_search_batch_return_partial_response_cuda)));
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl(
      "bloom_index_search_batch_prepared",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(bloom_index_search_batch_prepared_cuda)));
  m.impl(
      "bloom_index_search_batch_return_partial_response_prepared",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(
              bloom_index_search_batch_return_partial_response_prepared_cuda)));
}

TORCH_LIBRARY_IMPL(stfork, CUDA, m) {
  m.impl(
      "bloom_index_search_batch_return_partial_response_multiple",
      torch::dispatch(
          c10::DispatchKey::CUDA,
          TORCH_FN(
              bloom_index_search_batch_return_partial_response_multiple_cuda)));
}
} // namespace bloom_search
} // namespace ops
} // namespace st
