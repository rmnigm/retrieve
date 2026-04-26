<!-- claude code generated file -->

# `retrieve` architecture

## Clause / attribute data layout

Items have `C` clauses. Each clause holds up to `A_max` int64 attribute IDs,
padded with `-1`. Storage lives in two buffers on `ClauseIndex`:

- `item_clause_attrs` — `[N, C, A_max]` int64.
- `clause_is_reverse` — `[C]` bool, `True` marks a *reverse* clause.

At query time a query supplies one attribute ID per clause in
`query_clause_attrs[B, C]`.

Semantics:

- A clause **passes** for an item if **any** of the item's attribute IDs for
  that clause equals the query attribute (OR within a clause). Padded `-1`
  slots never equal a valid query attribute, so they are inert.
- For a **reverse** clause the outcome is inverted: the item passes iff it
  does *not* match.
- A query attribute of `-1` marks the clause **inactive** — it always passes.
- An item passes overall when **all** clauses pass (AND between clauses).

`ClauseIndex.evaluate` produces `[B, N]` bool.

## Mask composition

`combine_masks(clause_index, query_clause_attrs, external_mask)` returns the
effective pre-filter mask for a LiNR forward:

- If both the clause mask and the external mask are provided, the result is
  their bitwise AND.
- If only one is provided, that one is returned.
- If neither is provided, the result is `None` (the caller should skip
  masking entirely).

## LiNR variants

All three combine clause evaluation and retrieval in a single module and
return `(ids[B, K], scores[B, K])`.

- **V1 — similarity masking.** Full `query @ item_embs.T` matmul, masked
  scores set to `-inf`, then top-K. Works well when the mask is dense.
  With Triton, uses `fused_matmul_topk` when no mask is active and
  `fused_masked_knn_topk` over compact positive indices when one is.
- **V2 — explicit pre-filter.** Argsort the mask, gather the passing items,
  run a reduced `bmm` on the subset, top-K, then map back to global IDs.
  Pays off at low pass rates. Falls back to a V1-style exhaustive matmul
  when no mask is active. The Triton path reuses `fused_masked_knn_topk`.
- **V3 — INT8 quantized.** Stores embeddings as int8 codes plus per-item
  float32 scales (4× memory vs FP32). Three forward paths, in priority
  order:
  1. `candidate_ids` provided — gather codes, dequantize on the fly, bmm,
     top-K, map back. Clause/mask inputs are ignored on this path.
  2. `mask` or `query_clause_attrs` provided — full INT8 matmul with the
     combined mask applied to scores.
  3. Neither provided — full exhaustive INT8 scan.

## Triton kernels

Both kernels live in [retrieve/kernels/triton/](../kernels/triton/) and
launch one program per query in the batch, holding the query vector in
SRAM to cut HBM traffic.

### `fused_matmul_topk`

Computes dot products between the query and every item in tiled
`BLOCK_N`-wide chunks, applying the boolean mask inline (one `tl.where`
in the same pass as the dot product), and writes scores to a `[B, N]`
buffer. `torch.topk` selects the final top-K from that buffer.

Versus `query @ item_embs.T` + `masked_fill` + `topk`, it:

- Loads each item embedding block once.
- Applies the mask without a separate pass.
- Keeps the query in SRAM across all tiles.

When `has_mask=False` the wrapper allocates a 1×1 dummy tensor so the
kernel's strides stay well-defined — the pointer is never dereferenced,
because the `HAS_MASK` constexpr guards the mask load.

### `fused_masked_knn_topk`

Takes compact `positive_indices[B, P]` (produced by argsort-then-slice
over a mask) plus per-row `counts[B]`, and reads **only** the passing
items' rows via indirect loads. Scores land in a `[B, P]` buffer and
`torch.topk` selects top-K locally; the wrapper gathers global IDs from
`positive_indices`, and pads to `K` with `-1` / `-inf` when `P < k`.

Compared with the PyTorch paths:

- V1 loads all `N` embeddings regardless of mask — wasteful at low pass
  rates.
- V2 materializes a `[B, P, D]` gathered tensor in HBM before matmul.
- This kernel reads the `P` passing rows directly and fuses the dot
  product, so there is no intermediate gather tensor.

## Builder layout

- `quantize_int8` lives in [`retrieve.layers.quantize`](../src/retrieve/layers/quantize.py)
  and is consumed by `LiNR_V3.register_index`.
- Per-version builders live next to their classes:
  `build_linr_v1` in `linr_v1.py`, `build_linr_v2` in `linr_v2.py`,
  `build_linr_v3` in `linr_v3.py`.
- [`retrieve.layers.builder`](../src/retrieve/layers/builder.py) is a thin
  dispatcher: `build_linr_index(version, …)` looks up the right per-version
  builder in a small table and forwards kwargs.
