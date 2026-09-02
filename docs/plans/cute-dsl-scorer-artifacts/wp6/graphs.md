# WP-6 Task A — eager vs torch.compile vs CUDA-graph replay (A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, nvidia-cutlass-dsl 4.7.1)

Full `SilverTorch.forward` (phase-1 centroid matmul + topk + gather, the backend's custom op incl. its `quantize_int8`
prep and the host `topk` epilogue), `D=128`, `k=64`. Modules built with a synthetic balanced IVF (every cluster exactly
`max_size` wide; registered through the module's own steps, no k-means) so `P` is exactly layout A `(1664, 1824, 32)`
→ 58 368 (N = 3 035 136) and the small layout `(64, 32, 32)` → 1024. Attributes / query attributes as in the WP-4
`bench_common.py` (bloom pass rate 0.0015 at layout A, exact 0.961).
`triton.testing.do_bench(rep=300, warmup=50)` median ms. Variants: **eager** `module(q, attrs)`; **compile**
`torch.compile(module, fullgraph=True)` default mode; **graph** `torch.compile(module, fullgraph=True,
mode="reduce-overhead")` (cudagraph trees). `torch.equal` on `(ids, scores)` across the three variants per backend and
on `scores` across backends was asserted before every timing (36/36 cells). Raw: `graphs.json`, `run_graphs.log`.

| mode | B | P | eager tri | cuda | cute | compile tri | cuda | cute | graph tri | cuda | cute | cuda/cute eager | compile | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1 | 1024 | 0.306 | 0.300 | 0.329 | 0.144 | 0.379 | 0.406 | 0.056 | 0.071 | 0.072 | 0.91× | 0.93× | 0.98× |
| none | 1 | 58368 | 0.528 | 0.445 | 0.487 | 0.336 | 0.522 | 0.561 | 0.118 | 0.130 | 0.133 | 0.91× | 0.93× | 0.98× |
| none | 16 | 1024 | 0.368 | 0.295 | 0.332 | 0.180 | 0.362 | 0.408 | 0.059 | 0.082 | 0.082 | 0.89× | 0.89× | 1.00× |
| none | 16 | 58368 | 0.537 | 0.452 | 0.486 | 0.343 | 0.531 | 0.568 | 0.237 | 0.256 | 0.255 | 0.93× | 0.93× | 1.00× |
| bloom | 1 | 1024 | 1.018 | 0.948 | 1.054 | 0.283 | 0.493 | 0.596 | 0.079 | 0.094 | 0.091 | 0.90× | 0.83× | 1.03× |
| bloom | 1 | 58368 | 1.220 | 1.167 | 1.250 | 0.439 | 0.654 | 0.714 | 0.132 | 0.173 | 0.142 | 0.93× | 0.92× | 1.22× |
| bloom | 16 | 1024 | 1.043 | 1.013 | 1.178 | 0.308 | 0.531 | 0.579 | 0.079 | 0.102 | 0.103 | 0.86× | 0.92× | 0.99× |
| bloom | 16 | 58368 | 1.286 | 1.232 | 1.290 | 0.473 | 0.714 | 0.742 | 0.278 | 0.232 | 0.222 | 0.95× | 0.96× | 1.05× |
| exact | 1 | 1024 | 0.428 | 0.346 | 0.405 | 0.223 | 0.420 | 0.481 | 0.059 | 0.084 | 0.087 | 0.85× | 0.87× | 0.96× |
| exact | 1 | 58368 | 0.594 | 0.518 | 0.614 | 0.365 | 0.623 | 0.651 | 0.126 | 0.142 | 0.143 | 0.84× | 0.96× | 0.99× |
| exact | 16 | 1024 | 0.422 | 0.351 | 0.412 | 0.219 | 0.421 | 0.486 | 0.066 | 0.109 | 0.098 | 0.85× | 0.86× | 1.11× |
| exact | 16 | 58368 | 0.609 | 0.535 | 0.582 | 0.395 | 0.615 | 0.661 | 0.287 | 0.315 | 0.316 | 0.92× | 0.93× | 1.00× |

## Manual `torch.cuda.CUDAGraph` of the eager forward (pure replay, inputs not re-copied)

| mode | B | P | manual tri | cuda | cute | cuda/cute |
|---|---|---|---|---|---|---|
| none | 1 | 1024 | 0.064 | 0.067 | 0.068 | 0.98× |
| none | 1 | 58368 | 0.125 | 0.127 | 0.129 | 0.99× |
| none | 16 | 1024 | 0.075 | 0.078 | 0.078 | 0.99× |
| none | 16 | 58368 | 0.253 | 0.251 | 0.250 | 1.00× |
| bloom | 1 | 1024 | — | — | — | — |
| bloom | 1 | 58368 | — | — | — | — |
| bloom | 16 | 1024 | — | — | — | — |
| bloom | 16 | 58368 | — | — | — | — |
| exact | 1 | 1024 | 0.068 | 0.071 | 0.073 | 0.97× |
| exact | 1 | 58368 | 0.133 | 0.136 | 0.137 | 0.99× |
| exact | 16 | 1024 | 0.080 | 0.084 | 0.086 | 0.99× |
| exact | 16 | 58368 | 0.304 | 0.310 | 0.311 | 1.00× |

Bloom rows: capture failed identically for all three backends —
`AcceleratorError: CUDA error: operation failed due to a previous error during capture` — because the eager forward's `build_query_signatures` (`layers/filters/bloom_hash.py`)
materialises its salt constants with `torch.tensor(_SALT, device=cuda)`, a pageable host→device copy that invalidates
a raw stream capture. Under `torch.compile` those constants are folded into the graph, which is why cudagraph trees
captured the same forward. A layer-level fact, unrelated to the scoring backend.

## What captured (profiler, one forward under reduce-overhead)

Every cell: `cudagraph_skips == 0`, no "skipping cudagraphs" perf-hint, exactly one `cudaGraphLaunch` per forward; the
only other host-side CUDA calls are the input copies into cudagraph trees' static buffers (`cudaLaunchKernel×1` for
the query in no-filter mode, `cudaMemcpyAsync×2` for query + attrs otherwise). The cute op's driver-API launch
(`cuLaunchKernel` from the DSL host stub on `torch._C._cuda_getCurrentRawStream`) landed on the capture stream every time.

| mode | B | P | backend | reduce-overhead host calls (one forward) | kernels in graph | manual graph kernels |
|---|---|---|---|---|---|---|
| none | 1 | 1024 | triton | cudaLaunchKernel×1, cudaGraphLaunch×1 | 12 | 18 |
| none | 1 | 1024 | cuda | cudaLaunchKernel×1, cudaGraphLaunch×1 | 20 | 18 |
| none | 1 | 1024 | cute | cudaLaunchKernel×1, cudaGraphLaunch×1 | 20 | 18 |
| none | 1 | 58368 | triton | cudaLaunchKernel×1, cudaGraphLaunch×1 | 26 | 32 |
| none | 1 | 58368 | cuda | cudaLaunchKernel×1, cudaGraphLaunch×1 | 34 | 32 |
| none | 1 | 58368 | cute | cudaLaunchKernel×1, cudaGraphLaunch×1 | 34 | 32 |
| none | 16 | 1024 | triton | cudaLaunchKernel×1, cudaGraphLaunch×1 | 10 | 16 |
| none | 16 | 1024 | cuda | cudaLaunchKernel×1, cudaGraphLaunch×1 | 18 | 16 |
| none | 16 | 1024 | cute | cudaLaunchKernel×1, cudaGraphLaunch×1 | 18 | 16 |
| none | 16 | 58368 | triton | cudaLaunchKernel×1, cudaGraphLaunch×1 | 26 | 32 |
| none | 16 | 58368 | cuda | cudaLaunchKernel×1, cudaGraphLaunch×1 | 34 | 32 |
| none | 16 | 58368 | cute | cudaLaunchKernel×1, cudaGraphLaunch×1 | 34 | 32 |
| bloom | 1 | 1024 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 16 | — |
| bloom | 1 | 1024 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 24 | — |
| bloom | 1 | 1024 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 24 | — |
| bloom | 1 | 58368 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 31 | — |
| bloom | 1 | 58368 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 40 | — |
| bloom | 1 | 58368 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 40 | — |
| bloom | 16 | 1024 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 13 | — |
| bloom | 16 | 1024 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 22 | — |
| bloom | 16 | 1024 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 22 | — |
| bloom | 16 | 58368 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 31 | — |
| bloom | 16 | 58368 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 39 | — |
| bloom | 16 | 58368 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 39 | — |
| exact | 1 | 1024 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 9 | 15 |
| exact | 1 | 1024 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 17 | 15 |
| exact | 1 | 1024 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 17 | 15 |
| exact | 1 | 58368 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 26 | 31 |
| exact | 1 | 58368 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 34 | 31 |
| exact | 1 | 58368 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 34 | 31 |
| exact | 16 | 1024 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 8 | 14 |
| exact | 16 | 1024 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 16 | 14 |
| exact | 16 | 1024 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 16 | 14 |
| exact | 16 | 58368 | triton | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 26 | 31 |
| exact | 16 | 58368 | cuda | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 34 | 31 |
| exact | 16 | 58368 | cute | cudaMemcpyAsync×2, cudaGraphLaunch×1 | 34 | 31 |
