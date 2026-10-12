"""WP-6 Task A: does the cute host-overhead gap survive torch.compile and CUDA-graph replay?

Full SilverTorch forward (phase 1 + custom op + topk) for backend x filter_mode x layout x B,
four ways: eager, torch.compile(fullgraph=True) default, torch.compile(mode="reduce-overhead")
(cudagraph trees), and a manual torch.cuda.CUDAGraph of the eager forward (pure replay floor).
Modules are built with a synthetic *balanced* IVF (every cluster exactly max_size wide) through
the module's own registration steps, skipping k-means, so P is exactly the plan's layout A
(1664, 1824, 32) -> 58368 and S (64, 32, 32) -> 1024. Bit-exactness across variants and
backends is asserted before timing. Usage: graphs.py OUT.json [modes] [layouts] [Bs]."""

from __future__ import annotations

import json
import logging
import sys
import time

import torch
import torch._dynamo
import torch._inductor.config
import triton.testing as tt
from torch._dynamo.utils import counters

import importlib

cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.layers.filters.bloom_hash import build_query_signatures
from retrieve.layers.silvertorch.main import SilverTorch

LAYOUTS = {"S": (64, 32, 32), "A": (1664, 1824, 32)}
D, K, M_BITS, K_HASH, C, A_MAX, N_VOCAB = 128, 64, 1024, 5, 2, 2, 50
BACKENDS = ("triton", "cuda", "cute")
REP, WARM = 300, 50

out_path = sys.argv[1]
MODES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["none", "bloom", "exact"]
LAYS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["S", "A"]
BS = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1, 16]

torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch._logging.set_logs(perf_hints=True)
_skip_msgs: list[str] = []


class _H(logging.Handler):
    def emit(self, rec):
        m = rec.getMessage()
        if "cudagraph" in m.lower():
            _skip_msgs.append(m)


logging.getLogger("torch._inductor").addHandler(_H())
cu.ensure_built()
ct.ensure_built()
print("torch", torch.__version__, "| cudagraph_trees", torch._inductor.config.triton.cudagraph_trees,
      "| default cudagraphs", torch._inductor.config.triton.cudagraphs, flush=True)


def make_data(layout, seed=0):
    n_lists, max_size, n_probe = LAYOUTS[layout]
    n = n_lists * max_size
    g = torch.Generator(device="cuda").manual_seed(seed)
    embs = torch.randn(n, D, generator=g, device="cuda")
    embs = embs / embs.norm(dim=1, keepdim=True)
    cents = torch.randn(n_lists, D, generator=g, device="cuda")
    cents = cents / cents.norm(dim=1, keepdim=True)
    padded = torch.randperm(n, generator=g, device="cuda").reshape(n_lists, max_size)
    attrs = torch.randint(0, N_VOCAB, (n, C, A_MAX), generator=g, device="cuda")  # bench_common's sparsity
    rev = torch.zeros(C, dtype=torch.bool, device="cuda")
    rev[0] = True
    return dict(n_lists=n_lists, max_size=max_size, n_probe=n_probe, n=n, embs=embs, cents=cents,
                padded=padded, attrs=attrs, rev=rev)


def build_module(data, mode, backend):
    m = SilverTorch(k=K, n_lists=data["n_lists"], n_probe=data["n_probe"], filter_mode=mode,
                    m_bits=M_BITS if mode == "bloom" else None, k_hash=K_HASH if mode == "bloom" else None,
                    backend=backend)
    # register_index's order, minus k-means: centroids, item_codes/global_scale, padded, sizes, filter.
    m.register_buffer("centroids", data["cents"].clone())
    m._quantize_items(data["embs"])
    m.register_buffer("padded_cluster_items", data["padded"].clone())
    m.register_buffer("cluster_sizes", torch.full((data["n_lists"],), data["max_size"], dtype=torch.long, device="cuda"))
    m._max_cluster_size = data["max_size"]
    m._register_filter_buffers(data["n"], data["attrs"] if mode != "none" else None,
                               data["rev"] if mode == "exact" else None)
    return m.eval()


def make_query(b, mode, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(b, D, generator=g, device="cuda")
    q = q / q.norm(dim=1, keepdim=True)
    if mode == "none":
        return q, None
    qa = torch.randint(0, N_VOCAB, (b, C), generator=g, device="cuda")
    if mode == "exact":
        qa[:, -1] = -1  # one inactive clause (bench_common)
    return q, qa


def pass_rate(m, q, qa, mode):
    """Fraction of the [B, P] slots that pass phase 2 (from the cuda mask kernels)."""
    if mode == "none":
        return 1.0
    probe_ids, flat = m._phase1_probe_with_ids(q)
    if mode == "bloom":
        qb = build_query_signatures(qa.long().unsqueeze(-1), m.hash_seeds, m.m_bits, m.k_hash, m.word_count)
        mask = cu._bloom_partial_mask_cuda_impl(qb, m.bloom_sigs_t, probe_ids, m._max_cluster_size)
    else:
        mask = cu._clause_partial_mask_cuda_impl(flat, m.item_clause_attrs, m.clause_is_reverse, qa.long(), m._max_cluster_size)
    bits = sum(int(bin(w & ((1 << 64) - 1)).count("1")) for w in mask.flatten().tolist())
    return bits / flat.numel()


def bench(fn):
    return tt.do_bench(fn, rep=REP, warmup=WARM, return_mode="median")


def clone_out(o):
    return tuple(t.clone() for t in o)


def same(a, b):
    return torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def profile_launches(fn):
    """Count cudaGraphLaunch vs individual kernel launches in one call."""
    from torch.profiler import ProfilerActivity, profile
    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    ev = {}
    for e in prof.key_averages():
        if e.key in ("cudaGraphLaunch", "cuLaunchKernel", "cudaLaunchKernel", "cuGraphLaunch", "cudaMemcpyAsync"):
            ev[e.key] = e.count
    n_kernels = sum(e.count for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA)
    ev["cuda_kernels"] = n_kernels
    return ev


def manual_graph(m, q, qa):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            m(q, qa)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        out = m(q, qa)
    torch.cuda.synchronize()
    return g, out


rows = []
t_start = time.perf_counter()
for layout in LAYS:
    data = make_data(layout)
    n_lists, max_size, n_probe = LAYOUTS[layout]
    P = n_probe * max_size
    for mode in MODES:
        for b in BS:
            q, qa = make_query(b, mode)
            ref_scores = None
            for backend in BACKENDS:
                cell = dict(mode=mode, B=b, P=P, layout=layout, backend=backend)
                t0 = time.perf_counter()
                m = build_module(data, mode, backend)
                if backend == "cuda":
                    cell["pass_rate"] = pass_rate(m, q, qa, mode)
                eager = lambda: m(q, qa)  # noqa: E731
                for _ in range(3):
                    eager()
                torch.cuda.synchronize()
                o_eager = clone_out(eager())
                if ref_scores is None:
                    ref_scores = o_eager[1]
                assert torch.equal(o_eager[1], ref_scores), f"{cell}: scores differ from the first backend"
                cell["eager"] = bench(eager)

                torch._dynamo.reset()
                cm = torch.compile(m, fullgraph=True)
                comp = lambda: cm(q, qa)  # noqa: E731
                for _ in range(3):
                    comp()
                torch.cuda.synchronize()
                o = clone_out(comp())
                assert same(o, o_eager), f"{cell}: compile output != eager"
                cell["compile"] = bench(comp)

                torch._dynamo.reset()
                counters.clear()
                _skip_msgs.clear()
                rm = torch.compile(m, fullgraph=True, mode="reduce-overhead")
                ro = lambda: rm(q, qa)  # noqa: E731
                for _ in range(5):
                    ro()
                torch.cuda.synchronize()
                o = clone_out(ro())
                assert same(o, o_eager), f"{cell}: reduce-overhead output != eager"
                cell["cudagraph_skips"] = int(counters["inductor"].get("cudagraph_skips", 0))
                cell["skip_msgs"] = list(_skip_msgs)
                cell["ro_launches"] = profile_launches(ro)
                cell["cudagraph"] = bench(ro)

                torch._dynamo.reset()
                torch.cuda.synchronize()
                try:
                    g, og = manual_graph(m, q, qa)
                    g.replay()
                    torch.cuda.synchronize()
                    assert same(clone_out(og), o_eager), f"{cell}: manual graph output != eager"
                    cell["manual_launches"] = profile_launches(g.replay)
                    cell["manual_graph"] = bench(g.replay)
                    del g
                except Exception as e:  # capture failure is a result, not an abort
                    cell["manual_graph"] = None
                    cell["manual_error"] = f"{type(e).__name__}: {e}"[:400]
                cell["cell_seconds"] = time.perf_counter() - t0
                rows.append(cell)
                print(f"{mode:<6} B={b:>2} P={P:>6} {backend:<7} eager {cell['eager']:.4f} compile {cell['compile']:.4f} "
                      f"ro {cell['cudagraph']:.4f} manual {cell.get('manual_graph')} skips={cell['cudagraph_skips']} "
                      f"ro_launch={cell['ro_launches']} man_launch={cell.get('manual_launches')} "
                      f"[{cell['cell_seconds']:.0f}s]", flush=True)
                del m
                torch.cuda.empty_cache()
                json.dump(rows, open(out_path, "w"), indent=1)
    del data
    torch.cuda.empty_cache()
print(f"total {time.perf_counter() - t_start:.0f}s", flush=True)
