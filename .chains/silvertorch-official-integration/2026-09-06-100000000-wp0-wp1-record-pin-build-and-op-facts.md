---
chain: "silvertorch-official-integration"
branch: "main"
parent: "2026-09-05-120000000-plan-official-ops-as-reference-backend.md"
nextStep: "WP-2 / WP-3: run the adapter's T1-T7 on the A100 with the HIGH-first bit order pinned."
created: "2026-09-06T10:00:00Z"
---

# §13 record: WP-0 and WP-1 (roadmap A2 / A3), 2026-09-06

A100-SXM4-80GB, nvcc 12.4, torch 2.10.0+cu128, triton 3.6.0, branch `dev/a0-a3-deps-official`. Artifacts: `docs/artifacts/official-silvertorch/` (`official_facts.py`, `official_facts.json`, `official_facts.txt`, `wp0_build.txt`, `wp0_environment.txt`, `wp0_upstream_pytest.txt`). Everything measured; nothing read off the source.

## 13.1 WP-0: pin and build
`silvertorch @ 21aa35e28b6dd9a91e9ee35efb0857715e86bda7`. `uv sync --extra official` built `silvertorch._C` in 2 m 17 s (ninja, 10 TUs, `TORCH_CUDA_ARCH_LIST="8.0"`, `MAX_JOBS=32`); `torch.ops.st.fused_kmean_ann` exists. Gate passed.

Corrections:
- 12.4 is fine: torch warns on a minor mismatch and builds; raises only on major.
- Nine `st::` ops, not eleven: `is_topk` and `take_top_k_and_gather_from_main_and_fresh` are registered in sources absent from `setup.py`; dead source at this sha. `faster_repeat_interleave.cu` compiles but registers no op.
- The README's `pytest silvertorch/` fails at collection (3 errors): Meta's line-based `@oss-disable` stripping mangled three test files that load Buck targets (`test_fresh_index_post_processing.py`, `test_bloom_search_integration.py`: a commented-out closing paren; `test_is_topk.py:24`: an uncommented `load_library`). Excluding them: 99 passed, 3 subtests passed in 11.6 s, every CUDA test included. With the bogus lines removed `test_bloom_search_integration.py` passes (9 tests). Green for everything the OSS build ships.
- "7 open issues" are 7 PRs (#5-#10, #16); zero issues ever. #16 is CCCL-3 / CUDA-13 compatibility.

Recipe additions: the virtual workspace root re-exports the extra (`official = ["retrieve[official]"]`); `setuptools` / `wheel` / `ninja` go in a root `[dependency-groups] dev`; `uv lock` takes 1 s. The `cute` extra stays until B4.

## 13.2 WP-1: op facts
Config: N = 16,384 (64 x 256), D = 128, `n_probe = 8` (P = 2048), B = 16, k = 5, `hash_k = 7`, `b_multiplier = 8.0`, plans on CPU.

Bit order HIGH-first, confirmed three ways: (A) the packed search for a doc-0 predicate gives word 0 = `0x8000000000000000`; (B) a hand-built `filtering_bit_mask` with bit 63 set keeps doc 0, bit 0 keeps doc 63; (C) A's packed word round-trips as B's mask.

Host syncs (instrument matters: `warnings.catch_warnings` around `set_sync_debug_mode("warn")` reports zero for every C++ op, because a `TORCH_WARN` in a C++ op goes to fd 2; counting `warn_or_error_on_sync` lines on fd 2, validated against a `t.item()` control):

| op | predicted | measured |
|---|---|---|
| `fused_kmean_ann` | 2 | 3 |
| `fused_kmean_ann_with_partial_masks` | 1-3 | 4 |
| `bloom_index_search_batch` (plans on CPU) | 0 | 0 |
| `..._return_partial_response` | >= 1 | 2 |
| `bloom_index_build` (CUDA) | 2 | 2 |
| `generate_column_info_for_clusters` | 0 | 0 |

Launches (torch.profiler, one call after 3 warm-ups):

| op | launches | distinct | D2H | H2D | aten ops |
|---|---|---|---|---|---|
| `fused_kmean_ann` (no filter / full mask) | 19 | 16 | 3 | 0 | 58 |
| `fused_kmean_ann_with_partial_masks` | 19 | 16 | 4 | 0 | 62 |
| `bloom_index_search_batch` (packed, full N) | 1 | 1 | 0 | 2 | 4 |
| `..._return_partial_response` | 13 | 13 | 2 | 2 | 43 |
| `generate_column_info_for_clusters` | 1 | 1 | 0 | 0 | 3 |
| `bloom_index_build` (CUDA) | 16 | 10 | 2 | 3 | 47 |

Only 2 of the scorer's 19 launches are `process_cluster` / `process_cluster_remaining`; the rest is payload prep. A bloom forward is ~32 launches + >= 5 syncs vs Triton's 1 launch.

CUDA-graph capture: `generate_column_info_for_clusters` captures; `bloom_index_search_batch` captures and its replay raises `CUDA error: an illegal memory access` (the host plan decode and pageable upload are not in the graph); the other five fail with `cudaErrorStreamCaptureInvalidated`. D7 stands, and `bloom_index_search_batch` must be on the not-capturable list explicitly. A failed capture poisons the CUDA context (next `synchronize()` raises), so each probe runs in its own subprocess.

Parse cost (CPU, plans returned to CPU):

| batch | µs / call | µs / query |
|---|---|---|
| B=16, `c:v AND c:v` | 58.7 | 3.67 |
| B=16, `c:v` | 41.7 | 2.61 |
| B=16, `c:v AND NOT c:v` | 66.8 | 4.18 |
| B=1, `c:v AND c:v` | 12.6 | 12.62 |

Scale paths confirmed with saturated codes (+-127, D=128, dot 2,064,512): (i) int32 returns exactly 2,064,512; (ii) divisor 64 returns fp16 32,256.0 vs exact 32,258.0 (spacing 16 at 2^15, rel. err 6.2e-5); (iii) `per_embedding_scale = ones` returns `inf` in all 256 slots.

## 13.3 What changed
Confirmed: bit order, scale paths, sync counts for build and column info, "keep plans on CPU", D5, D7. Corrected: scorer syncs 3 / 4 (not 2 / 1-3), launches 19 (not ~12), capturability (search captures but faults on replay), `is_topk` / fresh-index ops never compiled, 7 PRs not issues, CUDA 12.x is fine. No timing claim; nothing here is a citable performance result.
