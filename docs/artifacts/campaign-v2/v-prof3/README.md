# V-PROF3: three one-cell profiles from EXHIBITS run 1

Roadmap V-PROF3, pod 1, campaign-v2.1. Each cell is built as `bench run` builds it (`load_matrix`,
`sweep_assets`, `build_module`, `graph_callable`, the rotating pool); k 100, seed 0. NOT CITABLE.

| item | cell | variants |
|---|---|---|
| a | goodreads `codesign`, official bloom `c0_genre`, `n_lists` 1024, `n_probe` 32, bs 16 | `bloom_path` partial vs full (C5) |
| b | goodreads-synth `synth`, V1 clause `p01`, bs 1 | Triton vs `torch.compile(max-autotune)` (C3) |
| c0001, c1 | goodreads-synth `synth`, V1 Triton clause `p0001` / `p1`, bs 16 | eager vs graph replay |
| ac0001, ac1 | arxiv-synth `synth` (3 M), the same | eager vs graph replay |

| file | what |
|---|---|
| [`prof3.py`](prof3.py) | `profile ITEM VARIANT OUT`: 20 calls in one sentinel-bracketed profiler session (padded retry as `measure.profile_once`), writes per-call kernel table, CUDA API counts (launches, graph launches, memcpy, malloc, syncs), peak scratch and a chrome trace; `time ITEM OUT`: both variants in one process, interleaved (`measure.latency_group`; eager vs graph alternates whole windows) |
| [`v-prof3.sh`](v-prof3.sh) | the driver (on [`../v-pod1-run/common.sh`](../v-pod1-run/common.sh)): oracles, twelve profile processes (one profiler session per process), six timing processes |

## Outcome (2026-10-09, staging `0627961`, library `f01255f1`, A100 GPU 0, 0.10 GPU-h)

Hub `artifacts/v-prof3` ([hub-index](../../hub-index.md)); state in [validation](../../../validation.md). Device µs
exclude the `## Call CompiledFxGraph` range annotation, which the profiler lists as device time on compiled calls.

- **(a) co-design.** Interleaved: partial 1.440 ms, full 1.223 ms (full / partial 0.85). Device 418 vs 383 µs (+35 µs);
  partial runs 101 vs 84 kernels, 90 vs 76 launch calls, 9 vs 6 memcpys and **7.1 vs 4.1 host syncs** per call. Partial's extra
  kernels are mask gathers / copies (`index_elementwise` ×4 32 µs, `direct_copy` ×4 19 µs). Most of the 0.22 ms is host-side
  (three more syncs, more launches): Meta's partial path is slower for reasons outside the scorer.
- **(b) V1 bs 1, Triton vs `torch.compile(max-autotune)`.** Interleaved 0.488 vs 0.250 ms. Device 533 vs 265 µs. Triton (our V1 =
  cuBLAS mat-vec + Triton clause mask): `gemv2N` 284 µs, a separate `_clause_mask_kernel` 36 µs, fp32 radix top-k (4 digit passes,
  `computeDigitCumSum` 93 µs). Compiled: one fused masked mat-vec reduction (`triton_red_fused_..._mm_..._where`, 152 µs), top-k over
  fp16 scores (2 passes, 46 µs), replayed as one CUDA graph (1 launch vs 32).
- **(c) V1 bs 16 graph vs eager.** goodreads-synth: graph/eager 1.022 (p 0.001), 0.988 (p 1); arxiv-synth: 1.077 (p 0.001),
  1.036 (p 1). Mechanism: eager masks scores with one in-place elementwise kernel whose cost rises with p (72 → 109 µs at 0.8 M,
  223 → 393 µs at 3 M); in the graph, inductor functionalizes it into a `direct_copy` of the score matrix plus a fused `where`
  (77 + 61 µs at 0.8 M, 264 + 241 µs at 3 M), constant in p. Every other kernel is the same. A fix (a masking form inductor fuses
  without the copy) is a library step of its own.
