// Copyright (c) the retrieve authors. Licensed under the Apache License, Version 2.0, as the
// rest of this package (silvertorch/LICENSE); added in the fork (silvertorch/CHANGES.md, C13).

#pragma once

#include <cstdint>

namespace st::ops::fork_filters {

// Our exact clause predicate (retrieve.functional.clause_subset_match): a clause passes when any
// of the doc's values equals the query's, negated for a reverse clause; an inactive clause
// (query value -1) passes. C14: attrs is the clause-major int32 [R, n_docs] slot table of
// fork_clause_attrs, clause c in slots [clause_slots[c], clause_slots[c + 1]), so a warp's
// docs read coalesced; each doc's values are sorted descending, so a -1 ends its values.
__device__ __forceinline__ bool exact_pass(
    const int32_t* __restrict__ attrs,
    const int64_t* __restrict__ clause_slots,
    int64_t n_clauses,
    int64_t n_docs,
    const bool* __restrict__ reverse,
    const int32_t* __restrict__ query,
    int64_t doc) {
  for (int64_t c = 0; c < n_clauses; ++c) {
    const int32_t q = query[c];
    if (q == -1) {
      continue;
    }
    bool match = false;
    for (int64_t s = clause_slots[c]; s < clause_slots[c + 1]; ++s) {
      const int32_t v = attrs[s * n_docs + doc];
      if (v == q || v == -1) {
        match = v == q;
        break;
      }
    }
    if (match == reverse[c]) {
      return false;
    }
  }
  return true;
}

// The exact test's inputs as a kernel argument.
struct ExactClauses {
  const int32_t* attrs;
  const int64_t* clause_slots;
  int64_t n_clauses;
  int64_t n_docs;
  const bool* reverse;
  const int32_t* query_attrs;

  __device__ __forceinline__ bool pass(int64_t row, int64_t doc) const {
    return exact_pass(
        attrs, clause_slots, n_clauses, n_docs, reverse, query_attrs + row * n_clauses, doc);
  }
};

} // namespace st::ops::fork_filters
