---
chain: "cute-dsl-scorer"
branch: "main"
parent: "2026-09-02-100000000-cute-dsl-port-plan-and-spike.md"
nextStep: "None. Deleted at roadmap B4 with the CUDA backend; last commit holding it is tagged cuda-cute-backends-final."
created: "2026-09-02T18:00:00Z"
---

# Validation record (WP-4, WP-6), 2026-09-02

A100-SXM4-80GB, driver 580.159.03, torch 2.10.0+cu128 (CUDA 12.8 runtime), triton 3.6.0, nvidia-cutlass-dsl 4.7.1. Raw scripts and outputs: `docs/artifacts/cute-dsl-scorer/` (`wp4/`, `wp5/`, `wp6/`, `spike/`).

## Gates
Review fixes R1 (executor per `(specialization, device)` via `JitCompiledFunction.to(device)`) and R2 (capability < (8, 0) -> `CuteMissing`) first. Parity 82/82 after every change; `tests/parity tests/compile` 286 passed; ruff clean.

## Tune, best vs best (do_bench ms incl. topk; cute0 = as ported, cute = after host trims)

| regime | triton | cuda | cute0 | cute | tri/cute | cuda/cute |
|---|---|---|---|---|---|---|
| P=1024, HAS_QB=0 | 0.132 | 0.075 | 0.153 | 0.104 | 1.27x | 0.72x |
| P=1024, HAS_QB=1 | 0.128 | 0.091 | 0.216 | 0.129 | 0.99x | 0.71x |
| P=8192, HAS_QB=0 | 0.139 | 0.137 | 0.157 | 0.137 | 1.01x | 1.00x |
| P=8192, HAS_QB=1 | 0.133 | 0.152 | 0.227 | 0.152 | 0.87x | 1.00x |
| P=65536, HAS_QB=0 | 0.222 | 0.233 | 0.233 | 0.232 | 0.96x | 1.00x |
| P=65536, HAS_QB=1 | 0.209 | 0.229 | 0.297 | 0.227 | 0.92x | 1.01x |
| exact N=797085, B=1, C=4, A=4 | 0.130 | 0.123 | 0.225 | 0.131 | 0.99x | 0.93x |
| exact N=797085, B=16, C=4, A=4 | 0.157 | 0.161 | 0.193 | 0.160 | 0.98x | 1.01x |
| exact N=2988997, B=1, C=5, A=4 | 0.134 | 0.124 | 0.220 | 0.134 | 1.00x | 0.93x |
| exact N=2988997, B=16, C=5, A=4 | 0.168 | 0.173 | 0.225 | 0.172 | 0.98x | 1.01x |
| exact N=15000001, B=16, C=5, A=4 | 0.169 | 0.173 | 0.219 | 0.173 | 0.98x | 1.00x |

Winners: triton (64, 4) / (256, 8); cuda (128, 8, 4) / (256, 8, 4) (shipped (128, 8, 1) is 1.6 % off in geomean, not re-pasted); cute flat, reconciled by geomean ratio-to-best -> `DEFAULT_CONFIG = (block_p=256, num_warps=8, unroll=4)` (1.056x per-regime best, worst 1.144x). `unroll=4` costs 48 registers (34 at unroll=1), no occupancy cliff.

## Kernel-only, D=128, layout A (P=58,368), torch.profiler µs

| regime | triton | cuda (128,8,1) | cute (128,8,1) | cute (256,8,4) |
|---|---|---|---|---|
| no filter, B=1 | 9.9 | 10.6 | 10.1 | 12.3 |
| no filter, B=16 | 85.6 | 89.9 | 90.0 | 91.2 |
| bloom (pass 0.002), B=1, scorer + mask | 9.9 | 6.0 + 6.2 | 6.1 + 5.9 | 5.9 + 5.9 |
| bloom (pass 0.002), B=16 | 116.8 | 50.5 + 6.7 | 51.2 + 6.3 | 39.8 + 6.3 |
| exact (pass 0.96), B=1 | 10.3 | 10.2 + 5.7 | 10.0 + 5.7 | 11.3 + 5.7 |
| exact (pass 0.96), B=16 | 129.1 | 88.8 + 54.8 | 101.6 + 54.7 | 87.7 + 54.7 |

## Shared-input head-to-head incl. topk (do_bench ms, final code)

| mode | B | P | triton | cuda | cute | tri/cute | cuda/cute |
|---|---|---|---|---|---|---|---|
| none | 1 | 1024 | 0.131 | 0.066 | 0.098 | 1.34x | 0.68x |
| none | 1 | 58368 | 0.201 | 0.140 | 0.172 | 1.17x | 0.81x |
| none | 16 | 1024 | 0.126 | 0.082 | 0.098 | 1.29x | 0.84x |
| none | 16 | 58368 | 0.250 | 0.250 | 0.250 | 1.00x | 1.00x |
| bloom | 1 | 1024 | 0.121 | 0.079 | 0.135 | 0.90x | 0.58x |
| bloom | 1 | 58368 | 0.197 | 0.155 | 0.202 | 0.98x | 0.77x |
| bloom | 16 | 1024 | 0.122 | 0.089 | 0.125 | 0.98x | 0.72x |
| bloom | 16 | 58368 | 0.281 | 0.213 | 0.203 | 1.38x | 1.05x |
| bloom | 16 | 46720 | 0.256 | 0.201 | 0.209 | 1.23x | 0.96x |
| exact | 1 | 1024 | 0.135 | 0.078 | 0.129 | 1.05x | 0.61x |
| exact | 1 | 58368 | 0.214 | 0.156 | 0.209 | 1.02x | 0.75x |
| exact | 16 | 1024 | 0.138 | 0.087 | 0.131 | 1.05x | 0.66x |
| exact | 16 | 58368 | 0.299 | 0.306 | 0.305 | 0.98x | 1.00x |
| exact | 16 | 46720 | 0.270 | 0.278 | 0.277 | 0.98x | 1.00x |

Before host trims cute was 0.35-0.67x of cuda everywhere except the GPU-bound none B=16 P=58,368 row.

## Host overhead (enqueue-only µs, B=16)

| launcher | cuda | cute as ported | cute trimmed |
|---|---|---|---|
| phase 3 scorer | 4.0 | 63.2 | 16.2 |
| bloom mask | 9.9-11.0 | 63.2 | 20.5 |
| clause mask | 10.6-10.9 | 72.2 | 24.0 |
| compiled callable alone | - | 27.4 | 9.0 |

Trims: (a) stream handle from `torch._C._cuda_getCurrentRawStream` (0.2 µs vs 4.7); (b) skip the device guard when current (2.2 -> 0.3 µs); (c) bypass the DSL's per-call argument adaptation (`generate_execution_args`, 27 µs for 15 args) with `_Launch`: per-thread ctypes cells written in place, `run_compiled_program` called directly, packing verified once against the DSL's own path with a fallback. Floor: ~9 µs launch + ~7 µs validation and `torch.empty`, 3-4x the C++ launcher (from 15x). Host-bound residue at layout A went 102/209/153/209/62 -> 46/99/38/100/40 µs vs cuda 15/50/38/49/40.

P3: pinning the no-mask UNROLL=1 loop to one copy (`cutlass.range(..., unroll=1)`) is bit-exact but marginally slower (90.6 vs 89.3 µs B=16); left unpinned (D8).

Compile facts: cold `cute.compile` 116-173 ms per specialization, no disk cache; fresh process: imports 12-13 s, `ensure_built` 3.2-3.5 s (DSL import ~3 s); first `_impl` call 229 / 349 / 432 ms (none / bloom / exact), second 0.5-0.6 ms. SASS: 4x `IDP.4A.S8.S8` per item, one `LDG.E.EF.128`, `REDUX.SUM.S32`, two `FMUL`, no `FFMA`, no `BAR.SYNC`, no spills; registers 34 / 48 (C++ 30 / 40).

Reading: kernel for kernel the port matches C++ at the same config within +-1 µs, except the `HAS_MASK, UNROLL=1` scorer at high pass rate (exact B=16 101.6 vs 88.8 µs, +14 %: a divergent branch with a `REDUX.OR` test (P4) and lost read-only-cache hints (P2)); at UNROLL=4 it is 87.7 µs. The port's real cost is the launch. After trims cuda and cute are wall-clock equal at B=16 in every mode; cute stays 0.6-0.8x of cuda at B=1 and P=1024 (host-bound). Against Triton it stands where C++ stands: bloom 1.23-1.38x at large layouts (kernel-only 2.5x), none and exact within +-3 % at B=16.

## 5.1 Compiled / CUDA-graph replay (WP-6)
Full `SilverTorch.forward`, D=128, k=64, synthetic balanced IVF (P exactly 58,368 or 1,024). `torch.equal` across variants and backends asserted, 36/36 cells.

| mode | B | P | eager tri / cuda / cute | compile tri / cuda / cute | graph tri / cuda / cute |
|---|---|---|---|---|---|
| none | 1 | 1024 | 0.306 / 0.300 / 0.329 | 0.144 / 0.379 / 0.406 | 0.056 / 0.071 / 0.072 |
| none | 1 | 58368 | 0.528 / 0.445 / 0.487 | 0.336 / 0.522 / 0.561 | 0.118 / 0.130 / 0.133 |
| none | 16 | 1024 | 0.368 / 0.295 / 0.332 | 0.180 / 0.362 / 0.408 | 0.059 / 0.082 / 0.082 |
| none | 16 | 58368 | 0.537 / 0.452 / 0.486 | 0.343 / 0.531 / 0.568 | 0.237 / 0.256 / 0.255 |
| bloom | 1 | 1024 | 1.018 / 0.948 / 1.054 | 0.283 / 0.493 / 0.596 | 0.079 / 0.094 / 0.091 |
| bloom | 1 | 58368 | 1.220 / 1.167 / 1.250 | 0.439 / 0.654 / 0.714 | 0.132 / 0.173 / 0.142 |
| bloom | 16 | 1024 | 1.043 / 1.013 / 1.178 | 0.308 / 0.531 / 0.579 | 0.079 / 0.102 / 0.103 |
| bloom | 16 | 58368 | 1.286 / 1.232 / 1.290 | 0.473 / 0.714 / 0.742 | 0.278 / 0.232 / 0.222 |
| exact | 1 | 1024 | 0.428 / 0.346 / 0.405 | 0.223 / 0.420 / 0.481 | 0.059 / 0.084 / 0.087 |
| exact | 1 | 58368 | 0.594 / 0.518 / 0.614 | 0.365 / 0.623 / 0.651 | 0.126 / 0.142 / 0.143 |
| exact | 16 | 1024 | 0.422 / 0.351 / 0.412 | 0.219 / 0.421 / 0.486 | 0.066 / 0.109 / 0.098 |
| exact | 16 | 58368 | 0.609 / 0.535 / 0.582 | 0.395 / 0.615 / 0.661 | 0.287 / 0.315 / 0.316 |

Every cell `cudagraph_skips == 0`, one `cudaGraphLaunch` per forward. A manual `torch.cuda.CUDAGraph` of the eager forward fails to capture bloom on every backend: `build_query_signatures` built its salt with `torch.tensor(_SALT, device=cuda)`, a pageable H2D copy (fixed later by roadmap B5, the `clause_salt` buffer). The eager gap is host cost; default-mode compile does not remove it (opaque `custom_op`s); under cudagraph replay cuda and cute are equal within noise.
