---
chain: "cuda-silvertorch"
branch: "main"
nextStep: "Run the handoff runbook on an A100: build sanity, the parity gates of test_codesigned_probe_score_cuda.py, the kernel tune sweeps and the shared-input head-to-heads."
created: "2026-07-06T12:00:00Z"
---

# CUDA C++ SilverTorch backend (`backend="cuda"`): design and runbook

Source: `docs/plans/archive/cuda-silvertorch-handoff.md` §1-§12, written 2026-07-06, extended 2026-09-01. The backend was deleted at roadmap B4 (2026-09-06); the last commit holding it is tagged `cuda-cute-backends-final`.

## 1. Intent
Implement SilverTorch Algorithm 1 "partial_bloom" in CUDA C++ as a third `SilverTorch` backend, following the paper rather than the Triton kernel. The Triton `codesigned_probe_score` reads row-wise signatures: `8·W` bytes of bloom per probed item (128 B at `m_bits=1024`, as much as the D=128 code row).

| paper claim | implemented as |
|---|---|
| transposed bloom index, iterate set bits of QB, one `and.b64` tests 64 items (Fig. 3b) | `cps_bloom_mask_kernel` + `build_transposed_sigs` |
| partial bloom over probed clusters, per-cluster masks `M_c` | mask kernel over `probe_ids` spans |
| ANN kernel takes a 1-bit-per-item mask | `cps_score_kernel` reads one bit per item |
| fused index-matmul, no gather tensor | scorer gathers `item_codes[id]` rows directly |
| int8 with dp4a | `__dp4a` chain per lane |
| one warp per contiguous item tile | warp-owned ranges, SEG-lane segments |

## 2. Design
- Phase 2 bloom: `bloom_sigs_t[m]` is a bit-vector over padded IVF slots, cluster-major, `wpc = ceil(max_cluster_size/64)` words per cluster. One thread = one output word = 64 items; `__ffsll` over the set bits of QB. QB with no set bits yields `~0`. Filter read traffic `popcount(QB)/8` B/item (~1-3 B) vs Triton's 64-128 B: ~40-100x less (paper: 12.6-42.7x over the forward index).
- Phase 2 exact (added by phase 2, WP-B): `cps_clause_mask_kernel`, one warp per output word, two `__ballot_sync` halves; ids read from `flat_items`, so no new buffer and cuda+exact state dicts equal triton+exact.
- Phase 3 `cps_score_kernel`: grid `(cdiv(P, block_p), B)`; `SEG = min(32, D/4)` lanes per item, query words in registers, no shared memory; filtered items skip the row load (segment-uniform branch); `__reduce_add_sync` (REDUX, sm_80+).
- `UNROLL` knob in {1, 2, 4}: Little's law, A100 needs ~0.9 MB in flight (~67 rows of 128 B per SM); ~50 warps x 1 row is borderline short.
- Rejected: IMMA / tensor cores (the dot is M=1; int8 `mma` has M=16 granularity, Triton pads to it: `min_dot_size = (1, 1, 32)`), `cp.async` (targets smem; the data-dependent gather defeats prefetch), a smem tile GEMM (reintroduces everything the co-design removes).

## 3. Numerics: why CUDA equals Triton bit for bit
1. int8 x int8 -> int32 dot is exact (`|dot| <= 256·128·128 = 2^22`), order-independent without overflow. 2. int32 -> fp32 exact below 2^24. 3. Dequant is the same left-associated pair of fp32 multiplies, no `--use_fast_math`. 4. transposed-AND predicate boolean-identical to the row-wise subset test; 5. clause predicate identical to `common.clause_pass`; 6. same host `torch.topk` + `gather`. Gate: scores `torch.equal`, ids equal up to permutation within tied scores (`torch.topk` tie order is not stable; int32 dots of ~+-1e5 over P~768 give about one tied pair per row).

## 4. Traffic model
Per probed item at bloom pass rate `s`: Triton `8 + 8·W + s·D + 4`; CUDA `8 + popcount(QB)/8 + ~0.25 + s·D + 4`. Arxiv-like (D=128, W=16, popcount 25, s=0.2): 165 vs 41 B/item, ~4x. HBM-bound everywhere (~1 MAC/B vs a ~25 MAC/B knee). `bloom_sigs_t` bytes = `bloom_sigs` bytes x (`n_lists · max_cluster_size / N`), the IVF padding slack; a `RuntimeWarning` above 2x.

## 5. Acceptance criteria (§9)
(1) correctness hard: parity bit-exact, e2e quality byte-identical. (2) bloom: cuda >= 1.5x Triton at large P expected, gate >= 1.0x everywhere. (3) no filter within +-10 %. (3b) exact within +-10 %. (4) e2e median <= triton x 1.10. (5) otherwise merge as experimental.

## 6. Fallbacks
F1 build (nvcc major-version check, `CUDA_HOME`, ninja, nuke `TORCH_EXTENSIONS_DIR`); F2 opcheck / fake metadata, never silenced; F3 graph breaks (only `torch.empty`, current stream, no `.item()`; last resort `cudagraph_unsafe`); F4 bit drift = a bug, split mask vs scorer with `test_bloom_mask_matches_rowwise`; F5 perf short = re-tune, UNROLL, else experimental; F6 e2e quality drift never acceptable.
