"""ST-IDS gates, one width per process; before = the campaign-v2.2 library (package `retrieve_v22`,
make_pkg.sh), after = the working tree's `retrieve`. Index, filters and helpers: ST-DLOOP's
dloop_gate.py (N Gaussian items, one clause at item pass rate p, every query asks 1).

  exact  (1) ids + scores `torch.equal` between the two `_impl`s on ST-DLOOP's synthetic grid: none / bloom /
         exact × p {0, 0.001, 0.01, 0.1, 1} (+ an all-inactive query) × bs {1, 16} × k {100, 1000} ×
         n_probe {24, 1024}. (2) On a second index with n_lists 4096: k 1000 × n_probe {2048, 4096}, which
         v2.2 cannot compile: the after epilogue's ids `torch.equal` to a torch oracle of the slot → id map
         (`searchsorted` over the probes' cumulative ends) on the same top-k slots, and the whole op against
         the torch reference op at bs 1 (scores `torch.equal`, ids up to ties); D 128 only (the epilogue
         does not depend on D, and the reference materializes [B, width, D]).
  time   the id epilogue alone, before vs after, on the after scorer's real top-k slots: CUDA graphs of 64
         launches over 8 batches, ABAB windows, `--pairs` pairs, 95 % t-interval of the mean log ratio.
         bs {1, 16} × (k, n_probe) cells whose v2.2 variants are compiled on pod b already.

    PYTHONPATH=<pkgs>:<v21 pkg>:retrieve/src:retrieve:docs/artifacts/campaign-v2/st-dloop \
        python ids_gate.py {exact|time} D out.json
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
    N_ITER,
    POOL,
    filtered,
    graph_of,
    impl,
    probe_width,
    ratio_ci,
    sm_mhz,
    window_us,
)

from retrieve import SilverTorch
from retrieve.ops import reference
from tests.parity.conftest import assert_topk_equal

FILES = {"bloom": "codesigned_probe_score", "none": "codesigned_probe_score",
         "exact": "codesigned_probe_score_exact"}  # fmt: skip
PKGS = {"before": "retrieve_v22", "after": "retrieve"}
ARMS = {
    arm: {m: importlib.import_module(f"{pkg}.ops.triton.{f}") for m, f in FILES.items()}
    for arm, pkg in PKGS.items()
}
HOSTS = {
    arm: importlib.import_module(f"{pkg}.ops.triton._host") for arm, pkg in PKGS.items()
}
COMMON = {
    arm: importlib.import_module(f"{pkg}.ops.triton.common")
    for arm, pkg in PKGS.items()
}
TIME_CELLS = ((100, 24), (100, 256), (100, 1024), (1000, 24), (1000, 64), (1000, 1024))


def build_index(n: int, d: int, n_lists: int) -> SilverTorch:
    g = torch.Generator(device=DEV).manual_seed(0)
    m = SilverTorch(k=100, n_lists=n_lists, n_probe=24, n_iter=N_ITER, seed=0)
    m.register_index(torch.randn(n, d, device=DEV, generator=g))
    return m


def scores_and_slots(m, mode, q, probe, qa, k, width):
    """The after scorer's ``[B, width]`` scores and its ``torch.topk`` (the epilogue's input)."""
    mod = ARMS["after"][mode]
    lay = (
        q,
        probe,
        m.cluster_offsets,
        m.item_codes,
        m.sort_perm,
        m._global_scale_f,
        width,
    )
    cfg = HOSTS["after"].tile_for_width(mod.CONFIGS, q.shape[1], q.shape[0], width)
    if mode == "exact":
        la = mod._cpse_prep(*lay, item_clause_attrs=m.item_clause_attrs,
                            clause_is_reverse=m.clause_is_reverse, query_clause_attrs=qa, cfg=cfg)  # fmt: skip
        mod._codesigned_probe_score_exact_kernel[la.grid](**la.kwargs)
    else:
        qb = None if mode == "none" else m._query_bit_positions(qa)
        bt = None if mode == "none" else m.bloom_transposed
        la = mod._cps_prep(*lay, query_bit_positions=qb, bloom_transposed=bt, cfg=cfg)
        mod._codesigned_probe_score_kernel[la.grid](**la.kwargs)
    return la


def ids_oracle(slots, scores, probe, m):
    """Slot → original id in torch: the row's probed clusters back to back, searchsorted over their
    cumulative ends; -1 at -inf."""
    lo = m.cluster_offsets[probe]
    size = m.cluster_offsets[probe + 1] - lo
    end = size.cumsum(1)
    j = torch.searchsorted(end, slots, right=True).clamp_max(probe.shape[1] - 1)
    pos = slots - (end - size).gather(1, j) + lo.gather(1, j)
    ids = m.sort_perm[pos.clamp(0, m.sort_perm.numel() - 1)]
    return torch.where(scores > float("-inf"), ids, torch.full_like(ids, -1))


def run_exact(d, n, res):
    from dloop_gate import build_index as gate_index

    base = gate_index(n, d)
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
                            same = True
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
                            row = {"d": d, "mode": mode, "p": p, "query": qname, "n_probe": n_probe,
                                   "bs": bs, "k": k, "check": "v22", "equal": same}  # fmt: skip
                            res["rows"].append(row)
                            print(f"  {row}", flush=True)
    del base, m
    torch.cuda.empty_cache()
    if (
        d != 128
    ):  # the epilogue does not depend on D; the reference op materializes [B, width, D]
        return
    wide = build_index(n, d, 4096)
    ga = torch.Generator(device=DEV).manual_seed(2)
    attrs = (torch.rand(n, 1, 1, device=DEV, generator=ga) < 0.01).long()
    for mode in ("none", "bloom", "exact"):
        m = filtered(wide, mode, attrs)
        for n_probe in (2048, 4096):
            width = probe_width(wide, n_probe)
            for bs in (1, 16):
                q = pool[0][:bs].contiguous()
                qa = torch.ones(bs, 1, dtype=torch.long, device=DEV)
                probe = torch.topk(q @ wide.centroids.t(), n_probe, dim=1).indices
                la = scores_and_slots(m, mode, q, probe, qa, 1000, width)
                fin = HOSTS["after"].probe_topk(
                    la, 1000, probe, m.cluster_offsets, m.sort_perm
                )
                COMMON["after"].probe_ids_kernel[fin.grid](**fin.kwargs)
                slots = torch.topk(la.all_scores, 1000, dim=1).indices
                oracle_ok = torch.equal(
                    fin.ids, ids_oracle(slots, fin.scores, probe, m)
                )
                if bs == 1:
                    ia, sa = impl(
                        ARMS["after"][mode], mode, m, q, probe, qa, 1000, width
                    )
                    lay = (
                        q,
                        probe,
                        m.cluster_offsets,
                        m.item_codes,
                        m.sort_perm,
                        m._global_scale_f,
                        1000,
                        width,
                    )
                    if mode == "exact":
                        ref = reference.codesigned_probe_score_exact(
                            *lay[:5], m.item_clause_attrs, m.clause_is_reverse, qa, *lay[5:])  # fmt: skip
                    elif mode == "bloom":
                        ref = reference.codesigned_probe_score_bloom(
                            *lay[:5], m._query_bit_positions(qa), m.bloom_transposed, *lay[5:])  # fmt: skip
                    else:
                        ref = reference.codesigned_probe_score(*lay)
                    assert_topk_equal(ia, sa, *ref)
                row = {"d": d, "mode": mode, "p": 0.01, "n_probe": n_probe, "bs": bs, "k": 1000,
                       "check": "oracle" + ("+reference" if bs == 1 else ""), "equal": oracle_ok}  # fmt: skip
                res["rows"].append(row)
                print(f"  {row}", flush=True)


def run_time(d, n, pairs, res):
    from dloop_gate import build_index as gate_index

    base = gate_index(n, d)
    g = torch.Generator(device=DEV).manual_seed(1)
    pool16 = [torch.randn(16, d, device=DEV, generator=g) for _ in range(POOL)]
    for k, n_probe in TIME_CELLS:
        width = probe_width(base, n_probe)
        for bs in (1, 16):
            launches = {arm: [] for arm in ARMS}
            for q16 in pool16:
                q = q16[:bs].contiguous()
                probe = torch.topk(q @ base.centroids.t(), n_probe, dim=1).indices
                la = scores_and_slots(base, "none", q, probe, None, k, width)
                for arm in ARMS:
                    fin = HOSTS[arm].probe_topk(
                        la, k, probe, base.cluster_offsets, base.sort_perm
                    )
                    launches[arm].append((COMMON[arm].probe_ids_kernel, fin))
            same = True
            for (kb, fb), (ka, fa) in zip(
                launches["before"], launches["after"], strict=True
            ):
                kb[fb.grid](**fb.kwargs)
                ka[fa.grid](**fa.kwargs)
                same &= torch.equal(fb.ids, fa.ids)
            graphs = {arm: graph_of(ls) for arm, ls in launches.items()}
            for _ in range(3):
                window_us(graphs["before"]), window_us(graphs["after"])
            tb, ta, mhz = [], [], []
            for _ in range(pairs):
                tb.append(window_us(graphs["before"]))
                mhz.append(sm_mhz())
                ta.append(window_us(graphs["after"]))
                mhz.append(sm_mhz())
            r, lo, hi = ratio_ci(tb, ta)
            row = {"d": d, "k": k, "n_probe": n_probe, "bs": bs, "before_us": statistics.median(tb),
                   "after_us": statistics.median(ta), "ratio": r, "ci_lo": lo, "ci_hi": hi,
                   "ids_equal": same, "sm_mhz": [min(mhz), max(mhz)],
                   "unstable": max(mhz) - min(mhz) > 50}  # fmt: skip
            res["rows"].append(row)
            print(f"  k={k} n_probe={n_probe} bs={bs}: before {row['before_us']:.2f} after "
                  f"{row['after_us']:.2f} us ratio {r:.3f} [{lo:.3f}, {hi:.3f}] equal {same} "
                  f"sm {row['sm_mhz']}{' UNSTABLE' if row['unstable'] else ''}", flush=True)  # fmt: skip
            del graphs, launches


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
