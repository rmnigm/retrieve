---
chain: "cute-dsl-scorer"
branch: "main"
nextStep: "Port the four kernels (WP-1), wire and test them (WP-2), review against the .cu line by line (WP-3), then tune and benchmark triton / cuda / cute (WP-4)."
created: "2026-09-02T10:00:00Z"
---

# CuTe DSL port of the CUDA SilverTorch backend: plan and spike

Source: `docs/plans/archive/cute-dsl-scorer.md` §1-§4. Planned 2026-09-02 on `feat/cute-dsl-scorer` (from `refactor/kernels-eval` at `0f7792c`). Box: A100-SXM4-80GB, driver 580 / CUDA 13.0 driver, torch 2.10.0+cu128, triton 3.6.0, `nvidia-cutlass-dsl` 4.7.x, Python 3.11. Raw outputs: `docs/artifacts/cute-dsl-scorer/`.

Question: how big is the CUDA SilverTorch kernel rewritten in the CuTe DSL, and does it keep the C++ backend's speed and its margin over Triton?

Starting point: `.cu` 672 lines + 611-line host module; kernels `cps_bloom_mask_kernel`, `cps_clause_mask_kernel<C, A>`, `cps_score_kernel<SEG, HAS_MASK, UNROLL>`, `cps_score_kernel_generic<HAS_MASK>`. Targets (handoff §13, D=128, P=58,368, kernel-only µs): no filter B=16 87.8 (Triton 85.5); bloom B=16 58.6 (Triton 124.6); exact B=16 95.3 + 56.6 (Triton 121.7).

## Decisions
- D1: a fourth backend `backend="cute"`, not a replacement; the C++ backend is the bit-exact reference and baseline.
- D2: same op contract, same buffers and layout (transposed `bloom_sigs_t` imported from the cuda module): cute checkpoints byte-identical to cuda ones.
- D3: port all three kernels plus the ~60-line generic fallback (opcheck uses D=96).
- D4: templates become `cutlass.Constexpr` args with a compiled-callable cache keyed on them; runtime shapes are Int scalars so a new shape never recompiles.
- D5: bit-exactness vs cuda and vs Triton (`torch.equal` on `[B, P]` scores and on mask words); no fast-math.
- D6: `CodesignedProbeScoreCuteConfig(block_p, num_warps, unroll)`, tune specs on the cuda grid.
- D7: optional extra `cute`, lazy import; `CuteMissing` (skip) vs compile failure (fail).
- D8: no new algorithms: a one-to-one port so the comparison is of languages, not designs.

## Spike (WP-0) findings
`nvidia-cutlass-dsl 4.7.1` (cu12 libs, cuda-bindings 12.9.4) as extra `cute`; sync with `uv sync --all-packages --extra cute` (a plain `--extra cute` from `retrieve/` prunes the evaluation member). No nvcc needed; the DSL ships ptxas. Every primitive verified bit-exact on the A100: `dp4a` via `inline_ptx` (`IDP.4A.S8.S8`), `warp_redux_sync` (`REDUX.SUM.S32`), shuffle / ballot, `int4` `.cs` loads (`LDG.E.EF.128`), `cttz` + dynamic `while`, `Constexpr` specialization, int32 half stores into int64 masks via `recast_ptr`, fp32 epilogue as exactly two `mul.f32` (no fma), torch interop via `make_ptr` + `Int64` scalars on torch's stream (eager, in a `custom_op`, under `torch.compile(fullgraph=True)`).

Costs: `cute.compile` ~60-100 ms per specialization, no disk cache on this path (the `@cute.jit` direct-call cache costs ~4.7 ms per call, so never used hot). Per-launch host overhead ~9-12 µs (`make_ptr` route) vs ~5 µs for a torch op; `from_dlpack` + `mark_layout_dynamic` ~7 µs per tensor, and an unmarked `from_dlpack` tensor silently bakes its shape in.

Taken from the spike: pointers + scalars 1:1 with the C++ launchers; compiled callables cached per `(SEG, HAS_MASK, UNROLL)` / `(C, A)` (Constexpr params stripped from the callable's signature); `cuda.CUstream` cached per torch stream (4.5 µs to build); `Int64 // x`, `% x` are floor semantics, fine since all operands are non-negative; sm_80+ only (the sub-sm_80 butterfly is not ported).
