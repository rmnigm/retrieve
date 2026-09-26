---
chain: "architecture-review-2026-09-06"
branch: "main"
nextStep: "Library side: A3(b) (pack_mask byte packing; still int64 [B, N] at HEAD and the cause of the 826 MiB official-exact peak B3 measured) and A7 (per-class score dtype at the layer boundary) remain open; the rest landed at B2, B4, L1."
created: "2026-09-06T10:00:00Z"
---

# `retrieve` library architecture and code-design review, 2026-09-06

Source: `docs/plans/architecture-review-2026-09-06-library.md`. Review only, on `dev/integration` at `5cf009d` (development + B1 adapter + B5 salt buffer + A2 pin), CPU-only, every claim checked against the code. Reviewed against kernels-layers-design K1-K9 and plan O §5.

## A. The five highest-value changes
| # | change | when | fate |
|---|---|---|---|
| 1 | official plan cache opt-in for timing: `parse_plans` was `lru_cache`d on the expression tuple, so a harness timing one fixed closure never paid the 58.7 µs/call parse (10-20 % of an eager bloom forward). Add `OfficialConfig.cache_plans`; timing runs pass False | before B3 | applied `55ebb41`; the harness sets `cache_plans=False` in `run.perf` |
| 2 | stale-record sweep + pin the bit order (`official.py` "not yet measured", kernels.md "12 launches + 2 syncs", a three-valued `Backend`, "eight" tune subcommands, modules.md promising fp32 scores) | now | applied `1ee2f0a`; `OFFICIAL_BIT_ORDER = "high_first"` pinned |
| 3 | close the `-1` trap on the candidates paths: `SilverTorch._forward_candidates` and `FullScanKNN._forward_candidates` indexed with raw ids; a `-1` wrapped to the last item (on official through `inv_perm`) and returned as a real id | before D1 | applied `37cef19`: `masked_topk(valid=ids >= 0, gather_ids=ids, pad_to_k=False)` with a `clamp_min(0)` gather |
| 4 | a loaded state dict must be usable: `_global_scale_f`, `_max_cluster_size` were Python caches set only in `register_index` (T6 patched them by hand) | now | applied `ee9c339`: load post-hook; `TestStateDict` |
| 5 | give `Backend` its real shape at B4: one five-valued literal accepted everywhere, unvalidated (`OneBitKNN(k, backend="foo")` ran) -> `LinrBackend` / `SilverTorchBackend`, validated, table dispatch | B4 + C1 | applied at B4 (`010681d`) |

## B. Findings
- A1 dispatch vacuous outside SilverTorch (see #5).
- A2 SilverTorch after B4: table dispatch; merge `_register_filter_buffers` / `_register_official_filter_buffers` (their exact branches were copies up to `[sort_perm]`). Applied at B4.
- A3 the official adapter fits §5.1 with three seams: (a) dead `csr_from_assignments` disagreeing with `_build_ivf` on `argsort(stable=True)`: deleted; stable sort applied at B2 (`0521a67`) after measuring it bit-identical on nine regimes; (b) `pack_mask` materialises a `[B, N]` int64 (8x the mask, 1.3 GB at N = 10 M, B = 16) on the exact-on-official path; proposed a uint8 byte-packing (bit-identical, T3 gates it): NOT applied; (c) `queries_to_expressions(..., clause_is_reverse)` only exercised by T4: documented.
- A4 `k_hash` vs `hash_k` naming: `OfficialConfig.hash_k` renamed `n_stored_hashes`.
- A5 duplicated `_CpsLaunch`/`_cps_finish` and `_CpseLaunch`/`_cpse_finish`, the 3-D grid split pasted in three filter kernels: planned for G-a, done at L1 as `ops/triton/_host.py`.
- A6 `@triton_op` pattern and the tune registry correct; `bloom_match` without config / `_impl` / spec by K2.2's choice.
- A7 score dtype differs by path (`PostfilterKNN` fp16, `PostfilterKNNInt8` fp16 of `dots >> 5`, `PrefilterKNN` fp16 on torch, fp32 on Triton); modules.md fixed, the boundary cast left open.
- A8 `ExactAttributeFilter.register_index` did not `.long()` its attrs (an int32 tensor compiled a second `clause_mask` variant): fixed (L1).
- A9 `__all__` gaps (`OfficialConfig`, `compact_mask`), a private `_impl` exported, op / submodule name collisions: fixed at L1 / B4.
- D1 the B5 salt still copied H2D when the index had no attributes: `generate_clause_salt` uses Python-int constants.
- D2 host syncs all register-time or documented (the `.tolist()` in `queries_to_expressions` is the adapter's one intrinsic sync).
- D3 per-call dummy tensors in `_cps_prep` / `_oporp_prep`: fixed at L1 (existing tensors as dummy pointers).
- D4 compile friendliness fine; `_max_cluster_size` as a Python int avoids a SymInt.
- D5 the dequant `dot.float() * q_scale * global_scale` appears in six places and the bit-exact contract rests on all of them: documented in kernels.md § Numerics.
- D6 error handling fine; D7 `_forward_two_kernel` went with B4; D8 no `QuantizedIVF` residue in the library.
- Tests: T2 gaps (-1 candidates, fresh-module state dict, backend rejection, plan-cache fairness) closed by the items above; T3 `assert_topk_id_sets_match` / `assert_topk_matches` fold done at L1; T4 bit-order pin done.

## C. Keep as is (looks odd, is right)
`P`, `N`, `D`, `W` as `tl.constexpr` and the bucketing ladders; `bloom_match` without scaffolding; `fused_masked_knn_topk`'s public op skipping bucketing and the pad tail; the Python-scalar caches; full-width `evaluate_indices`; `PostfilterKNNInt8`'s `>> 5` and `_PAD_M = 17` (cuBLAS constraints); `FullScanKNN`'s post-filter semantics; `combine_indices`' per-stage `.item()`; k-means Lloyd's as a deviation for F1's table; `SilverTorch.compile()` plus the `is_compiling()` check (two entry points); `OfficialConfig` frozen with `__post_init__`; plans kept on CPU and the static `[B, P]` width.

## E. Applied 2026-09-06 on `dev/integration` (five commits on `12e3927`)
Gates: ruff clean, collect 683 -> 699, links 0; GPU claims unverified until B2 (which then passed them).
