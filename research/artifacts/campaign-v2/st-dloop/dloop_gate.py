"""ST-DLOOP gates on synthetic pass rates, one width per process: `before` = the campaign-v2.1 library
(package `retrieve_v21`, make_v21_pkg.sh), `after` = the working tree's `retrieve`.

  exact  ids + scores `torch.equal` between the two `_impl`s: modes none / bloom / exact, p in
         {0, 0.001, 0.01, 0.1, 1.0} plus an all-inactive query (edge set), bs {1, 16}, k {100, 1000},
         n_probe {24, 1024}.
  time   the scorer kernel alone, before vs after, as tile-skip/tile_skip.py (#16): a CUDA graph of 64
         launches over 8 query batches per arm, windows ABAB, `--pairs` pairs, ratio = after / before
         per pair, 95 % t-interval of the mean log ratio; SM clock sampled after every window; plus Meta's
         scorer and phase-2 mask device time on the same cells (torch.profiler; reference only).
         n_probe 24, k 100, bs {1, 16}, p {0.001, 0.01, 0.1, 1.0}.

Index: N Gaussian items (seed 0), n_lists 1024, n_iter 5; one clause, A_max 1, item value 1 with
probability p (else 0), every query asks 1 (#16's fixture).

    PYTHONPATH=<v21 pkg>:retrieve/src python dloop_gate.py {exact|time} D out.json [--n 2000000]
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import statistics
import subprocess
from pathlib import Path

import torch
import triton

import retrieve
from retrieve import OfficialConfig, SilverTorch
from retrieve.ops import official as off
from retrieve.ops.triton._host import tile_for_width, width_tiles

N_LISTS, N_ITER = 1024, 5
M_BITS, K_HASH = 1024, 5
POOL, LAUNCHES = 8, 64
FORCE_SKIP = False  # --force-skip: the after arm skips at every width (an experiment, not shipped)
DEV = torch.device("cuda")
T975 = {7: 2.365, 11: 2.201, 15: 2.131, 19: 2.093, 23: 2.069}
FILES = {"bloom": "codesigned_probe_score", "none": "codesigned_probe_score",
         "exact": "codesigned_probe_score_exact"}  # fmt: skip
V21_HOST = importlib.import_module("retrieve_v21.ops.triton._host")
ARMS = {
    arm: {m: importlib.import_module(f"{pkg}.ops.triton.{f}") for m, f in FILES.items()}
    for arm, pkg in (("before", "retrieve_v21"), ("after", "retrieve"))
}


def sm_mhz() -> int:
    q = [
        "nvidia-smi",
        "--query-gpu=clocks.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    return int(subprocess.check_output(q, text=True).strip().splitlines()[0])


def build_index(n: int, d: int) -> SilverTorch:
    g = torch.Generator(device=DEV).manual_seed(0)
    m = SilverTorch(k=100, n_lists=N_LISTS, n_probe=24, n_iter=N_ITER, seed=0)
    m.register_index(torch.randn(n, d, device=DEV, generator=g))
    return m


def filtered(
    base: SilverTorch, mode: str, attrs, backend="triton", official=None
) -> SilverTorch:
    """A module in ``mode`` sharing ``base``'s IVF layout, with the filter buffers for ``attrs``."""
    if mode == "none" and backend == "triton":
        return base
    kw = {"filter_mode": mode}
    if mode == "bloom":
        kw |= {"m_bits": M_BITS if backend != "official" else None, "k_hash": K_HASH}
    m = SilverTorch(
        k=100, n_lists=N_LISTS, n_probe=24, backend=backend, official=official, **kw
    )
    for name in ("centroids", "item_codes", "global_scale", "cluster_offsets",
                 "cluster_sizes", "sort_perm", "inv_perm"):  # fmt: skip
        m.register_buffer(name, getattr(base, name))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    if mode != "none":
        m._register_filter_buffers(attrs.shape[0], attrs, None, base.sort_perm)
    return m


def prep(mod, mode, m, q, probe, qa, width, cfg=None):
    """``(kernel, launch)`` of one arm's scorer on one batch: its own prep, the shipped tile or ``cfg``."""
    lay = (
        q,
        probe,
        m.cluster_offsets,
        m.item_codes,
        m.sort_perm,
        m._global_scale_f,
        width,
    )
    if cfg is None and mod.__name__.startswith("retrieve_v21."):
        cfg = V21_HOST.tile_for_width(mod.CONFIGS, q.shape[1])
    elif cfg is None:
        cfg = tile_for_width(mod.CONFIGS, q.shape[1], q.shape[0], width)
    if FORCE_SKIP and mod.__name__.startswith("retrieve."):
        cfg = dataclasses.replace(cfg, skip=True)
    if mode == "exact":
        la = mod._cpse_prep(*lay, item_clause_attrs=m.item_clause_attrs,
                            clause_is_reverse=m.clause_is_reverse, query_clause_attrs=qa, cfg=cfg)  # fmt: skip
        return mod._codesigned_probe_score_exact_kernel, la
    qb = None if mode == "none" else m._query_bit_positions(qa)
    bt = None if mode == "none" else m.bloom_transposed
    la = mod._cps_prep(*lay, query_bit_positions=qb, bloom_transposed=bt, cfg=cfg)
    return mod._codesigned_probe_score_kernel, la


def impl(mod, mode, m, q, probe, qa, k, width):
    lay = (
        q,
        probe,
        m.cluster_offsets,
        m.item_codes,
        m.sort_perm,
        m._global_scale_f,
        k,
        width,
    )
    if mode == "exact":
        return mod._codesigned_probe_score_exact_impl(
            *lay, item_clause_attrs=m.item_clause_attrs, clause_is_reverse=m.clause_is_reverse,
            query_clause_attrs=qa)  # fmt: skip
    if mode == "bloom":
        return mod._codesigned_probe_score_impl(
            *lay, query_bit_positions=m._query_bit_positions(qa), bloom_transposed=m.bloom_transposed)  # fmt: skip
    return mod._codesigned_probe_score_impl(*lay)


def probe_width(m: SilverTorch, n_probe: int) -> int:
    return int(m.cluster_sizes.sort(descending=True).values[:n_probe].sum())


def graph_of(ls):
    def run():
        for i in range(LAUNCHES):
            kern, la = ls[i % len(ls)]
            kern[la.grid](**la.kwargs)

    run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            run()
    torch.cuda.synchronize()
    return g


def window_us(g) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1e3 / LAUNCHES


def ratio_ci(before: list[float], after: list[float]) -> tuple[float, float, float]:
    logs = [math.log(a / b) for a, b in zip(after, before, strict=True)]
    mu, sd = statistics.mean(logs), statistics.stdev(logs)
    half = T975[len(logs) - 1] * sd / math.sqrt(len(logs))
    return math.exp(mu), math.exp(mu - half), math.exp(mu + half)


def official_us(m_off, mode, pool_q, probes, qa, cfg) -> dict:
    """Meta's scorer (+ its phase-2 mask) device µs per call by kernel class, profiler-timed."""

    def call(i):
        q, p = pool_q[i], probes[i]
        mask = partial = None
        a = qa[: q.shape[0]]
        if mode == "exact":
            mk = retrieve.ops.triton.clause_mask(
                m_off.item_clause_attrs, m_off.clause_is_reverse, a
            )
            mask = off.pack_mask(mk, off.MASK_BIT_ORDER)
        elif mode == "bloom":
            plans = off.parse_plans(off.queries_to_expressions(a), cfg.n_stored_hashes,
                                    cfg.max_sub_queries, cache=False)  # fmt: skip
            partial = off.bloom_partial_masks(
                m_off.bloom_index, m_off.bundle_b_offsets, plans, m_off.cluster_offsets[p],
                m_off.cluster_sizes[p], m_off.k_hash, cfg.n_stored_hashes)  # fmt: skip
        return off.official_scores_full(
            q, p, m_off.cluster_offsets, m_off.cluster_sizes, m_off.item_codes, m_off.sort_perm,
            m_off.global_scale, m_off._probe_width, score_path=cfg.score_path, divisor=cfg.divisor,
            filtering_bit_mask=mask, partial=partial)  # fmt: skip

    for i in range(POOL):
        call(i)
    torch.cuda.synchronize()
    reps = 3
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(reps):
            for i in range(POOL):
                call(i)
        torch.cuda.synchronize()
    by = {"scorer": 0.0, "mask": 0.0}
    for e in prof.key_averages():
        name = e.key.lower()
        if "process_cluster" in name or "fused_kmean" in name:
            by["scorer"] += e.device_time_total
        elif "process_documents" in name or "bloom" in name or "clause_mask" in name:
            by["mask"] += e.device_time_total
    return {k: v / (reps * POOL) for k, v in by.items()}


def run_exact(base, d, n, res):
    g = torch.Generator(device=DEV).manual_seed(1)
    pool = [torch.randn(16, d, device=DEV, generator=g) for _ in range(2)]
    for p in (0.0, 0.001, 0.01, 0.1, 1.0):
        ga = torch.Generator(device=DEV).manual_seed(2)
        attrs = (torch.rand(n, 1, 1, device=DEV, generator=ga) < p).long()
        for mode in ("none", "bloom", "exact"):
            if mode == "none" and p != 1.0:
                continue
            m = filtered(base, mode, attrs)
            qas = {"asks1": torch.ones(16, 1, dtype=torch.long, device=DEV)}
            if p == 0.001 and mode != "none":
                qas["inactive"] = torch.full((16, 1), -1, dtype=torch.long, device=DEV)
            for n_probe in (24, N_LISTS):
                width = probe_width(base, n_probe)
                for bs in (1, 16):
                    for k in (100, 1000):
                        for qname, qa_all in qas.items():
                            same, n_inf = True, 0
                            for q16 in pool:
                                q, qa = q16[:bs].contiguous(), qa_all[:bs]
                                probe = torch.topk(
                                    q @ base.centroids.t(), n_probe, dim=1
                                ).indices
                                ib, sb = impl(
                                    ARMS["before"][mode],
                                    mode,
                                    m,
                                    q,
                                    probe,
                                    qa,
                                    k,
                                    width,
                                )
                                ia, sa = impl(
                                    ARMS["after"][mode], mode, m, q, probe, qa, k, width
                                )
                                same &= torch.equal(ib, ia) and torch.equal(sb, sa)
                                n_inf += int(torch.isinf(sb).sum())
                            row = {"d": d, "mode": mode, "p": p, "query": qname, "n_probe": n_probe,
                                   "bs": bs, "k": k, "equal": same, "inf_slots": n_inf}  # fmt: skip
                            res["rows"].append(row)
                            print(f"  {row}", flush=True)
            if m is not base:
                del m
            torch.cuda.empty_cache()


def run_time(base, d, n, pairs, res):
    g = torch.Generator(device=DEV).manual_seed(1)
    pool16 = [torch.randn(16, d, device=DEV, generator=g) for _ in range(POOL)]
    qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
    ocfg = OfficialConfig(score_path="fp16", cache_plans=False)
    width = base._probe_width
    for p in (0.001, 0.01, 0.1, 1.0):
        ga = torch.Generator(device=DEV).manual_seed(2)
        attrs = (torch.rand(n, 1, 1, device=DEV, generator=ga) < p).long()
        for mode in ("bloom", "exact", "none"):
            if mode == "none" and p != 1.0:
                continue
            m = filtered(base, mode, attrs)
            m_off = filtered(base, mode, attrs, backend="official", official=ocfg)
            for bs in (1, 16):
                pool_q = [q[:bs].contiguous() for q in pool16]
                probes = [m._phase1_probe_ids(q) for q in pool_q]
                ls = {arm: [prep(ARMS[arm][mode], mode, m, q, pr, qa[:bs], width)
                            for q, pr in zip(pool_q, probes, strict=True)]
                      for arm in ARMS}  # fmt: skip
                same = True
                for (kb, lb), (ka, la) in zip(ls["before"], ls["after"], strict=True):
                    kb[lb.grid](**lb.kwargs)
                    ka[la.grid](**la.kwargs)
                    same &= torch.equal(lb.all_scores, la.all_scores)
                gb, gaf = graph_of(ls["before"]), graph_of(ls["after"])
                for _ in range(3):
                    window_us(gb), window_us(gaf)
                tb, ta, mhz = [], [], []
                for _ in range(pairs):
                    tb.append(window_us(gb))
                    mhz.append(sm_mhz())
                    ta.append(window_us(gaf))
                    mhz.append(sm_mhz())
                r, lo, hi = ratio_ci(tb, ta)
                o = official_us(m_off, mode, pool_q, probes, qa, ocfg)
                row = {"d": d, "p": p if mode != "none" else None, "mode": mode, "bs": bs,
                       "before_us": statistics.median(tb), "after_us": statistics.median(ta),
                       "ratio": r, "ci_lo": lo, "ci_hi": hi,
                       "official_scorer_us": o["scorer"], "official_mask_us": o["mask"],
                       "scores_equal": same, "sm_mhz": [min(mhz), max(mhz)],
                       "unstable": max(mhz) - min(mhz) > 50}  # fmt: skip
                res["rows"].append(row)
                print(f"  d={d} p={row['p']} {mode} bs={bs}: before {row['before_us']:.1f} after "
                      f"{row['after_us']:.1f} us ratio {r:.3f} [{lo:.3f}, {hi:.3f}] official "
                      f"scorer {o['scorer']:.1f} mask {o['mask']:.1f} equal {same} "
                      f"sm {row['sm_mhz']}{' UNSTABLE' if row['unstable'] else ''}", flush=True)  # fmt: skip
                del gb, gaf, ls
            del m, m_off
            torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exact", "time"])
    ap.add_argument("d", type=int)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--force-skip", action="store_true")
    args = ap.parse_args()
    global FORCE_SKIP
    FORCE_SKIP = args.force_skip
    torch.manual_seed(0)
    base = build_index(args.n, args.d)
    res = {"what": args.what, "force_skip": args.force_skip, "d": args.d, "n": args.n, "n_lists": N_LISTS,
           "width_n_probe_24": base._probe_width, "device": torch.cuda.get_device_name(),
           "torch": torch.__version__, "triton": triton.__version__,
           "after_configs": {m: repr(width_tiles(ARMS["after"][m].CONFIGS, args.d)) for m in FILES},
           "rows": []}  # fmt: skip
    print(f"[d={args.d}] index built, width {base._probe_width}", flush=True)
    (run_exact if args.what == "exact" else run_time)(
        base, args.d, args.n, *(() if args.what == "exact" else (args.pairs,)), res)  # fmt: skip
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
