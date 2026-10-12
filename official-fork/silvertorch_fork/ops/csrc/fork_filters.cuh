// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (official-fork/LICENSE); added in the fork (official-fork/CHANGES.md, C13).

#pragma once

#include <cstdint>

namespace st::ops::fork_filters {

// Our exact clause predicate (retrieve.functional.clause_subset_match): a clause passes when any
// of the doc's values equals the query's, negated for a reverse clause; an inactive clause
// (query value -1) passes. attrs is the cluster-sorted [N, C, A] table, query the row's [C].
__device__ __forceinline__ bool exact_pass(
    const int64_t* __restrict__ attrs,
    int64_t n_clauses,
    int64_t a_max,
    const bool* __restrict__ reverse,
    const int64_t* __restrict__ query,
    int64_t doc) {
  const int64_t* doc_attrs = attrs + doc * n_clauses * a_max;
  for (int64_t c = 0; c < n_clauses; ++c) {
    const int64_t q = query[c];
    if (q == -1) {
      continue;
    }
    bool match = false;
    for (int64_t a = 0; a < a_max; ++a) {
      match |= doc_attrs[c * a_max + a] == q;
    }
    if (match == reverse[c]) {
      return false;
    }
  }
  return true;
}

// The exact test's inputs as a kernel argument.
struct ExactClauses {
  const int64_t* attrs;
  int64_t n_clauses;
  int64_t a_max;
  const bool* reverse;
  const int64_t* query_attrs;

  __device__ __forceinline__ bool pass(int64_t row, int64_t doc) const {
    return exact_pass(attrs, n_clauses, a_max, reverse, query_attrs + row * n_clauses, doc);
  }
};

} // namespace st::ops::fork_filters
