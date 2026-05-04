# SilverTorch native CUDA impl — shelved (tried, failed, not worth it)

> **Status: dead.** A native-CUDA `codesigned_probe_score` kernel was attempted as a Triton replacement and **does not justify the complexity**. The directory `retrieve/src/retrieve/kernels/cuda/silvertorch/` has been removed from the runtime; the source tree is archived at `/workspace/silvertorch_cuda_rnd.zip` for reference. Do not revive without a fundamentally new approach.

## What was tried

1. **v1 — warp-per-item strip-mine, dp4a + and.b64 fused bloom.** Each warp scored 16 items strip-mined, lanes 0-7 issuing coalesced cp.async loads, lane-cooperative dp4a. Tuned `N_STAGES` from 2 → 16 (max in-flight prefetches per warp).

2. **v2 — block-cooperative tile loader + multi-stage cp.async.** All 4 warps cooperatively loaded a `[BLOCK_P, D]` tile, then split scoring across warps; multi-stage pipeline kept N_STAGES tiles in flight. Intended to attack the bandwidth-bound regime via larger transactions.

3. **Fused per-block top-K via `cub::BlockRadixSort`.** Each block emitted a per-block top-K of partial candidates; final `torch.topk` reduced across blocks. Intended to skip the `[B, P]` HBM write and cut the dominant post-kernel topk cost.

## Why each failed

- **v2** lost ~3× occupancy (5 blocks/SM vs 16) chasing larger transactions that didn't actually exist — Triton's win wasn't bigger transactions, it was deeper per-warp pipelining and **lane-per-item layout** giving 32 in-flight loads per warp vs our 8. v2 measured 0.58× of Triton on paper-like (B=16, P=30k, K=2048); v1 was 0.82×. Reverted.

- **N_STAGES tuning.** 2→3 saved 3µs, 3→8 saved 6µs, 8→16 saved 9µs on paper-like — diminishing returns. Final v1 kernel was 90µs vs Triton 73µs at paper-like (0.82×) but **3× faster on small/medium P** at the kernel level. Net full-path (kernel + topk) on most shapes was ~1.5-2× over Triton, paper-like was 0.82×.

- **Fused top-K.** `cub::BlockRadixSort` temp_storage cratered occupancy (3-5 blocks/SM vs 16). Going to `ITEMS_PER_BLOCK_F=128` helped marginally but **every fused shape regressed full-path 10-17%** vs unfused + `torch.topk`. The HBM write savings didn't compensate for lower occupancy + sort overhead.

## The structural ceiling

Triton's kernel is **compute-light** (it casts int8 → fp32 and does plain multiply-add — no dp4a, no tensor cores) but **memory-pattern-heavy**: each warp issues 32 simultaneous loads, one per lane, scattered across items. Our v1 dp4a path coalesces 8 lanes onto a single item — fewer in-flight loads per warp.

To beat Triton you'd need a **lane-per-item** memory layout AND keep dp4a's compute edge. Those constraints contradict each other for this op (dp4a needs lanes to share an item). The clean win available is **either** Triton's pattern in CUDA (matches Triton, doesn't beat it) **or** a tensor-core (`mma.sync m16n8k32` int8) rewrite — large effort, high risk, unproven.

## Verdict

**Not worth it.** The CUDA path won 1.5-2× on small/medium P at the kernel level but lost on paper-like (the workload that actually matters at scale), and the post-kernel torch.topk dominated full-path time anyway. Triton is the runtime. The C++/CUDA build chain (JIT extension load, two extra files, additional surface to maintain) is overhead the project doesn't need to carry.

If a future attempt tries again, it should:
1. **Profile first** with `ncu` (was unavailable in our session — we flew blind on attribution).
2. **Target the topk**, not the scoring. Topk is 50-90% of full-path time; scoring is already tight.
3. **Skip the v2 / cub paths** — they were the failures here.
4. **Consider `mma.sync` int8** if compute-bound after step 2.
