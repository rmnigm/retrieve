import json
rows = json.load(open("graphs.json"))
cells = {(r["mode"], r["B"], r["P"], r["backend"]): r for r in rows}
keys = sorted({(r["mode"], r["B"], r["P"]) for r in rows}, key=lambda k: (["none","bloom","exact"].index(k[0]), k[1], k[2]))
B = ("triton", "cuda", "cute")
f = lambda x: "—" if x is None else f"{x:.3f}"
lines = []
lines.append("| mode | B | P | eager tri | cuda | cute | compile tri | cuda | cute | graph tri | cuda | cute | cuda/cute eager | compile | graph |")
lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for mode, b, p in keys:
    c = {bk: cells[(mode, b, p, bk)] for bk in B}
    e = [c[bk]["eager"] for bk in B]; co = [c[bk]["compile"] for bk in B]; g = [c[bk]["cudagraph"] for bk in B]
    lines.append(f"| {mode} | {b} | {p} | " + " | ".join(map(f, e + co + g)) +
                 f" | {e[1]/e[2]:.2f}× | {co[1]/co[2]:.2f}× | {g[1]/g[2]:.2f}× |")
main = "\n".join(lines)
lines2 = ["| mode | B | P | manual tri | cuda | cute | cuda/cute |", "|---|---|---|---|---|---|---|"]
for mode, b, p in keys:
    c = {bk: cells[(mode, b, p, bk)] for bk in B}
    m = [c[bk]["manual_graph"] for bk in B]
    ratio = f"{m[1]/m[2]:.2f}×" if m[1] and m[2] else "—"
    lines2.append(f"| {mode} | {b} | {p} | " + " | ".join(map(f, m)) + f" | {ratio} |")
manual = "\n".join(lines2)
lines3 = ["| mode | B | P | backend | reduce-overhead host calls (one forward) | kernels in graph | manual graph kernels |", "|---|---|---|---|---|---|---|"]
for mode, b, p in keys:
    for bk in B:
        r = cells[(mode, b, p, bk)]
        ro = r["ro_launches"]; ml = r.get("manual_launches") or {}
        host = ", ".join(f"{k}×{v}" for k, v in ro.items() if k != "cuda_kernels")
        lines3.append(f"| {mode} | {b} | {p} | {bk} | {host} | {ro['cuda_kernels']} | {ml.get('cuda_kernels', '—')} |")
launches = "\n".join(lines3)
pr = {(r["mode"], r["P"], r["B"]): r["pass_rate"] for r in rows if "pass_rate" in r}
err = next(r["manual_error"] for r in rows if r.get("manual_error"))
md = f"""# WP-6 Task A — eager vs torch.compile vs CUDA-graph replay (A100-SXM4-80GB, torch 2.10.0+cu128, triton 3.6.0, nvidia-cutlass-dsl 4.7.1)

Full `SilverTorch.forward` (phase-1 centroid matmul + topk + gather, the backend's custom op incl. its `quantize_int8`
prep and the host `topk` epilogue), `D=128`, `k=64`. Modules built with a synthetic balanced IVF (every cluster exactly
`max_size` wide; registered through the module's own steps, no k-means) so `P` is exactly layout A `(1664, 1824, 32)`
→ 58 368 (N = 3 035 136) and the small layout `(64, 32, 32)` → 1024. Attributes / query attributes as in the WP-4
`bench_common.py` (bloom pass rate {pr[('bloom',58368,16)]:.4f} at layout A, exact {pr[('exact',58368,16)]:.3f}).
`triton.testing.do_bench(rep=300, warmup=50)` median ms. Variants: **eager** `module(q, attrs)`; **compile**
`torch.compile(module, fullgraph=True)` default mode; **graph** `torch.compile(module, fullgraph=True,
mode="reduce-overhead")` (cudagraph trees). `torch.equal` on `(ids, scores)` across the three variants per backend and
on `scores` across backends was asserted before every timing (36/36 cells). Raw: `graphs.json`, `run_graphs.log`.

{main}

## Manual `torch.cuda.CUDAGraph` of the eager forward (pure replay, inputs not re-copied)

{manual}

Bloom rows: capture failed identically for all three backends —
`{err.splitlines()[0]}` — because the eager forward's `build_query_signatures` (`layers/filters/bloom_hash.py`)
materialises its salt constants with `torch.tensor(_SALT, device=cuda)`, a pageable host→device copy that invalidates
a raw stream capture. Under `torch.compile` those constants are folded into the graph, which is why cudagraph trees
captured the same forward. A layer-level fact, unrelated to the scoring backend.

## What captured (profiler, one forward under reduce-overhead)

Every cell: `cudagraph_skips == 0`, no "skipping cudagraphs" perf-hint, exactly one `cudaGraphLaunch` per forward; the
only other host-side CUDA calls are the input copies into cudagraph trees' static buffers (`cudaLaunchKernel×1` for
the query in no-filter mode, `cudaMemcpyAsync×2` for query + attrs otherwise). The cute op's driver-API launch
(`cuLaunchKernel` from the DSL host stub on `torch._C._cuda_getCurrentRawStream`) landed on the capture stream every time.

{launches}
"""
open("graphs.md", "w").write(md)
print(main); print(); print(manual)
