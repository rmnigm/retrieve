// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C5).

#include <torch/library.h>

// stfork::search (fork_search_cuda.cu, CUDA only): the whole forward after phase 1 in one call.
// Filter (at most one): none; bloom over prepared plans (`plans`, evaluated over the probed
// clusters' columns only: Meta's full-N mask gives the same result, K fix 17); exact (`query_clause_attrs` against the
// cluster-sorted `item_clause_attrs` / `clause_slots` / `clause_is_reverse`, C14's int32 layout,
// over the probed docs only); or a
// given full-N `filtering_bit_mask`. Returns
// (ids int64 [B, k], scores fp32 [B, k]) with -1 / -inf pads, like every SilverTorch backend.
TORCH_LIBRARY_FRAGMENT(stfork, m) {
  m.def(
      "search("
      "Tensor query, "
      "Tensor probe_ids, "
      "Tensor cluster_offsets, "
      "Tensor cluster_sizes, "
      "Tensor item_codes, "
      "Tensor sort_perm, "
      "Tensor global_scale, "
      "int k, "
      "int max_tensor_size_per_row, "
      "int divisor_for_int8=-1, "
      "Tensor? filtering_bit_mask=None, "
      "Tensor? bloom_index=None, "
      "Tensor? bloom_bundle_b_offsets=None, "
      "Tensor? plans=None, "
      "int plan_count=0, "
      "int max_stack_size=0, "
      "Tensor? query_plan_index=None, "
      "int bloom_k=0, "
      "int hash_k=0, "
      "Tensor? item_clause_attrs=None, "
      "Tensor? clause_slots=None, "
      "Tensor? clause_is_reverse=None, "
      "Tensor? query_clause_attrs=None)"
      "-> (Tensor, Tensor)");

  // search's first launch (C6), exposed for its gate: int8 codes, scales, the probed clusters'
  // ids / lengths / offsets, warp sizes and the three cumsums the scorer reads.
  m.def(
      "probe_prep("
      "Tensor query, "
      "Tensor probe_ids, "
      "Tensor cluster_offsets, "
      "Tensor cluster_sizes)"
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
}
