"""ST-LANE gates, one width per process: before = campaign-v2.4 (library d67d6263, package `retrieve_v24` from
st-ids/make_pkg.sh), after = this tree, whose bloom scorer masks its bloom word loads by `valid` instead of the running
`keep`. Both arms get the bloom pass-rate table (v2.4 ships ST-SKIP128's gate). Index, filters and timing helpers:
ST-DLOOP's dloop_gate.py (N Gaussian items, n_lists 1024; one clause, item value 1 with probability p, every query asks 1).

  exact  ids + scores `torch.equal`, before `_impl` vs after `_impl`: none / bloom / exact ×
         p {0, 0.001, 0.003, 0.01, 0.1, 1} (+ an all-inactive query) × bs {1, 16} × k {100, 1000} × n_probe {24, 1024}.
  time   the scorer kernel alone, before vs after, CUDA graphs ABAB, `--pairs` pairs, 95 % t-interval:
         bloom / exact × p {0.001, 0.003, 0.01, 0.1, 1} × bs {1, 16}, plus none.

    PYTHONPATH=<pkgs>:<v21 pkg>:retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop \
        python lane_gate.py {exact|time} D out.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
from pathlib import Path

import torch
import triton
from dloop_gate import (
    DEV,
    N_LISTS,
    POOL,
    build_index,
    filtered,
    graph_of,
    probe_width,
    ratio_ci,
    sm_mhz,
    window_us,
)

FILES = {"bloom": "codesigned_probe_score", "none": "codesigned_probe_score",
         "exact": "codesigned_probe_score_exact"}  # fmt: skip
PKGS = {"before": "retrieve_v24", "after": "retrieve"}
ARMS = {
    a: {m: importlib.import_module(f"{p}.ops.triton.{f}") for m, f in FILES.items()}
    for a, p in PKGS.items()
}
HOSTS = {a: importlib.import_module(f"{p}.ops.triton._host") for a, p in PKGS.items()}


def tables(arm, mode, m):
    """The bloom pass-rate table, on both arms (v2.4 ships the gate)."""
    return {"bloom_bit_freq": m.bloom_bit_freq} if mode == "bloom" else {}


def impl(arm, mode, m, q, probe, qa, k, width):
    mod = ARMS[arm][mode]
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
            query_clause_attrs=qa, **tables(arm, mode, m))  # fmt: skip
    if mode == "bloom":
        return mod._codesigned_probe_score_impl(
            *lay, query_bit_positions=m._query_bit_positions(qa), bloom_transposed=m.bloom_transposed,
            **tables(arm, mode, m))  # fmt: skip
    return mod._codesigned_probe_score_impl(*lay)


def prep(arm, mode, m, q, probe, qa, width):
    """``(kernel, launch)`` of one arm's scorer on one batch, at its shipped tile."""
    mod = ARMS[arm][mode]
    cfg = HOSTS[arm].tile_for_width(mod.CONFIGS, q.shape[1], q.shape[0], width)
    lay = (
        q,
        probe,
        m.cluster_offsets,
        m.item_codes,
        m.sort_perm,
        m._global_scale_f,
        width,
    )
    t = tables(arm, mode, m)
    if mode == "exact":
        extra = {}
        la = mod._cpse_prep(*lay, item_clause_attrs=m.item_clause_attrs, clause_is_reverse=m.clause_is_reverse,
                            query_clause_attrs=qa, cfg=cfg, **extra)  # fmt: skip
        return mod._codesigned_probe_score_exact_kernel, la
    qb = None if mode == "none" else m._query_bit_positions(qa)
    bt = None if mode == "none" else m.bloom_transposed
    extra = {"bit_freq": t["bloom_bit_freq"]} if t else {}
    la = mod._cps_prep(
        *lay, query_bit_positions=qb, bloom_transposed=bt, cfg=cfg, **extra
    )
    return mod._codesigned_probe_score_kernel, la


def attrs_at(n, p):
    ga = torch.Generator(device=DEV).manual_seed(2)
    return (torch.rand(n, 1, 1, device=DEV, generator=ga) < p).long()


def run_exact(d, n, res):
    base = build_index(n, d)
    g = torch.Generator(device=DEV).manual_seed(1)
    pool = [torch.randn(16, d, device=DEV, generator=g) for _ in range(2)]
    for p in (0.0, 0.001, 0.003, 0.01, 0.1, 1.0):
        attrs = attrs_at(n, p)
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
                            same = True
                            for q16 in pool:
                                q, qa = q16[:bs].contiguous(), qa_all[:bs]
                                probe = torch.topk(
                                    q @ base.centroids.t(), n_probe, dim=1
                                ).indices
                                ib, sb = impl("before", mode, m, q, probe, qa, k, width)
                                ia, sa = impl("after", mode, m, q, probe, qa, k, width)
                                same &= torch.equal(ib, ia) and torch.equal(sb, sa)
                            row = {"d": d, "mode": mode, "p": p, "query": qname, "n_probe": n_probe,
                                   "bs": bs, "k": k, "equal": same}  # fmt: skip
                            res["rows"].append(row)
                            print(f"  {row}", flush=True)


def run_time(d, n, pairs, res):
    base = build_index(n, d)
    g = torch.Generator(device=DEV).manual_seed(1)
    pool16 = [torch.randn(16, d, device=DEV, generator=g) for _ in range(POOL)]
    qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
    width = base._probe_width
    cells = [
        (mode, p) for p in (0.001, 0.003, 0.01, 0.1, 1.0) for mode in ("bloom", "exact")
    ]
    for mode, p in [*cells, ("none", None)]:
        m = filtered(base, mode, attrs_at(n, p if p is not None else 1.0))
        for bs in (1, 16):
            pool_q = [q[:bs].contiguous() for q in pool16]
            probes = [m._phase1_probe_ids(q) for q in pool_q]
            ls = {arm: [prep(arm, mode, m, q, pr, qa[:bs], width) for q, pr in zip(pool_q, probes, strict=True)]
                  for arm in ARMS}  # fmt: skip
            same = True
            for (kb, lb), (ka, la) in zip(ls["before"], ls["after"], strict=True):
                kb[lb.grid](**lb.kwargs)
                ka[la.grid](**la.kwargs)
                same &= torch.equal(lb.all_scores, la.all_scores)
            gb, ga = graph_of(ls["before"]), graph_of(ls["after"])
            for _ in range(3):
                window_us(gb), window_us(ga)
            tb, ta, mhz = [], [], []
            for _ in range(pairs):
                tb.append(window_us(gb))
                mhz.append(sm_mhz())
                ta.append(window_us(ga))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            gated = bool(ls["after"][0][1].kwargs.get("GATED"))
            row = {"d": d, "mode": mode, "p": p, "bs": bs, "gated_kernel": gated,
                   "before_us": statistics.median(tb), "after_us": statistics.median(ta),
                   "ratio": r, "ci_lo": lo, "ci_hi": hi, "scores_equal": same,
                   "sm_mhz": [min(mhz), max(mhz)], "unstable": max(mhz) - min(mhz) > 50}  # fmt: skip
            res["rows"].append(row)
            print(f"  d={d} {mode} p={p} bs={bs}: before {row['before_us']:.1f} after {row['after_us']:.1f} us "
                  f"ratio {r:.3f} [{lo:.3f}, {hi:.3f}] equal {same} sm {row['sm_mhz']}"
                  f"{' UNSTABLE' if row['unstable'] else ''}", flush=True)  # fmt: skip
            del gb, ga, ls
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["exact", "time"])
    ap.add_argument("d", type=int)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    res = {"what": args.what, "d": args.d, "n": args.n, "device": torch.cuda.get_device_name(),
           "torch": torch.__version__, "triton": triton.__version__, "rows": []}  # fmt: skip
    if args.what == "exact":
        run_exact(args.d, args.n, res)
    else:
        run_time(args.d, args.n, args.pairs, res)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
