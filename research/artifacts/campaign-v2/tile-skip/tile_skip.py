"""Probe-scorer tile skipping (#16): before/after/official kernel timings at synthetic pass rates.
Method and prediction: README.md next to this file.

    python tile_skip.py {arxiv|goodreads} out.json [--before-ref b96e1f2] [--pairs 12]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton

import retrieve
from retrieve import OfficialConfig, SilverTorch
from retrieve.ops import official as off
from retrieve.ops.triton._host import tile_for_width

SHAPES = {
    "arxiv": {"n": 3_000_000, "n_lists": 1664},
    "goodreads": {"n": 800_000, "n_lists": 1024},
}
D, N_PROBE, K, N_ITER = 128, 24, 100, 5
M_BITS, K_HASH = 1024, 5
PASS_RATES = (0.001, 0.01, 0.1, 1.0)
BATCH_SIZES = (1, 16)
POOL, LAUNCHES = 8, 64
DEV = torch.device("cuda")
REPO = Path(__file__).resolve().parents[4]
T975 = {11: 2.201, 15: 2.131, 19: 2.093, 23: 2.069}


def sm_mhz() -> int:
    q = ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"]
    return int(subprocess.check_output(q, text=True).strip().splitlines()[0])


def load_before(ref: str, rel: str, name: str):
    """The kernel file at ``ref``, under a private op namespace so it can live beside ours."""
    src = subprocess.check_output(["git", "-C", str(REPO), "show", f"{ref}:{rel}"], text=True)
    path = Path("/scratch/cv2-lib") / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(src.replace('"retrieve::', f'"{name}::'))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_index(shape):
    g = torch.Generator(device=DEV).manual_seed(0)
    embs = torch.randn(shape["n"], D, device=DEV, generator=g)
    m = SilverTorch(k=K, n_lists=shape["n_lists"], n_probe=N_PROBE, n_iter=N_ITER, seed=0)
    m.register_index(embs)
    return m, embs


def filtered(base: SilverTorch, mode: str, attrs, backend="triton", official=None):
    """A module in ``mode`` sharing ``base``'s IVF layout, with the filter buffers for ``attrs``."""
    kw = {"filter_mode": mode}
    if mode == "bloom":
        kw |= {"m_bits": M_BITS if backend != "official" else None, "k_hash": K_HASH}
    m = SilverTorch(k=K, n_lists=base.n_lists, n_probe=N_PROBE, backend=backend,
                    official=official, **kw)  # fmt: skip
    for name in ("centroids", "item_codes", "global_scale", "cluster_offsets",
                 "cluster_sizes", "sort_perm", "inv_perm"):  # fmt: skip
        m.register_buffer(name, getattr(base, name))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    m._register_filter_buffers(attrs.shape[0], attrs, None, base.sort_perm)
    return m


def launches(mod, kind, m, pool_q, probes, qa):
    """Per pool entry, the scorer kernel and its launch kwargs (prep once, outside timing)."""
    out = []
    lay = (m.cluster_offsets, m.item_codes, m.sort_perm, m._global_scale_f, m._probe_width)
    for q, p in zip(pool_q, probes, strict=True):
        if kind == "bloom":
            cfg = tile_for_width(mod.CONFIGS, D)
            la = mod._cps_prep(q, p, *lay, query_bit_positions=m._query_bit_positions(qa[: q.shape[0]]),
                               bloom_transposed=m.bloom_transposed, cfg=cfg)  # fmt: skip
            out.append((mod._codesigned_probe_score_kernel, la))
        else:
            cfg = tile_for_width(mod.CONFIGS, D)
            la = mod._cpse_prep(q, p, *lay, item_clause_attrs=m.item_clause_attrs,
                                clause_is_reverse=m.clause_is_reverse,
                                query_clause_attrs=qa[: q.shape[0]], cfg=cfg)  # fmt: skip
            out.append((mod._codesigned_probe_score_exact_kernel, la))
    return out


def graph_of(ls):
    def run():
        for i in range(LAUNCHES):
            kern, la = ls[i % len(ls)]
            kern[la.grid](**la.kwargs)

    run()  # compile + warm
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


def outputs_equal(la_before, la_after, kb, ka) -> bool:
    kb[la_before.grid](**la_before.kwargs)
    ka[la_after.grid](**la_after.kwargs)
    torch.cuda.synchronize()
    return torch.equal(la_before.all_scores, la_after.all_scores)


def official_us(m_off, kind, pool_q, probes, qa, cfg) -> dict:
    """Meta's scorer (+ its bloom phase 2) device µs per call, by kernel class, profiler-timed."""
    width = m_off._probe_width

    def call(i):
        q, p = pool_q[i], probes[i]
        mask = partial = None
        a = qa[: q.shape[0]]
        if kind == "exact":
            mk = retrieve.ops.triton.clause_mask(m_off.item_clause_attrs, m_off.clause_is_reverse, a)
            mask = off.pack_mask(mk, off.MASK_BIT_ORDER)
        elif kind == "bloom":
            plans = off.parse_plans(off.queries_to_expressions(a), cfg.n_stored_hashes,
                                    cfg.max_sub_queries, cache=False)  # fmt: skip
            partial = off.bloom_partial_masks(
                m_off.bloom_index, m_off.bundle_b_offsets, plans, m_off.cluster_offsets[p],
                m_off.cluster_sizes[p], m_off.k_hash, cfg.n_stored_hashes)  # fmt: skip
        return off.official_scores_full(
            q, p, m_off.cluster_offsets, m_off.cluster_sizes, m_off.item_codes, m_off.sort_perm,
            m_off.global_scale, width, score_path=cfg.score_path, divisor=cfg.divisor,
            filtering_bit_mask=mask, partial=partial)  # fmt: skip

    for i in range(POOL):
        call(i)
    torch.cuda.synchronize()
    reps = 3
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            for i in range(POOL):
                call(i)
        torch.cuda.synchronize()
    by = {"scorer": 0.0, "mask": 0.0}
    for e in prof.key_averages():
        name = e.key.lower()
        us = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if "process_cluster" in name or "fused_kmean" in name:
            by["scorer"] += us
        elif "process_documents" in name or "bloom" in name or "clause_mask" in name:
            by["mask"] += us
    return {k: v / (reps * POOL) for k, v in by.items()}


def ratio_ci(before: list[float], after: list[float]) -> tuple[float, float, float]:
    logs = [math.log(a / b) for a, b in zip(after, before, strict=True)]
    mu, sd = statistics.mean(logs), statistics.stdev(logs)
    half = T975[len(logs) - 1] * sd / math.sqrt(len(logs))
    return math.exp(mu), math.exp(mu - half), math.exp(mu + half)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shape", choices=sorted(SHAPES))
    ap.add_argument("out")
    ap.add_argument("--before-ref", default="b96e1f2")
    ap.add_argument("--pairs", type=int, default=12)
    args = ap.parse_args()
    torch.manual_seed(0)
    before = {
        "bloom": load_before(args.before_ref, "retrieve/src/retrieve/ops/triton/codesigned_probe_score.py", "cv2before_cps"),
        "exact": load_before(args.before_ref, "retrieve/src/retrieve/ops/triton/codesigned_probe_score_exact.py", "cv2before_cpse"),
    }  # fmt: skip
    # The package attribute is the op of the same name; the kernel file is the module.
    after = {k: importlib.import_module(f"retrieve.ops.triton.{m}") for k, m in
             (("bloom", "codesigned_probe_score"), ("exact", "codesigned_probe_score_exact"))}  # fmt: skip
    shape = SHAPES[args.shape]
    base, _ = build_index(shape)
    n = shape["n"]
    g = torch.Generator(device=DEV).manual_seed(1)
    pool16 = [torch.randn(16, D, device=DEV, generator=g) for _ in range(POOL)]
    qa = torch.ones(16, 1, dtype=torch.long, device=DEV)
    ocfg = OfficialConfig(score_path="fp16", cache_plans=False)
    res = {"shape": args.shape, **shape, "d": D, "n_probe": N_PROBE, "width": base._probe_width,
           "before_ref": args.before_ref, "device": torch.cuda.get_device_name(),
           "torch": torch.__version__, "triton": triton.__version__, "rows": []}  # fmt: skip
    print(f"[{args.shape}] index built, width {base._probe_width}", flush=True)
    for p in PASS_RATES:
        ga = torch.Generator(device=DEV).manual_seed(2)
        attrs = (torch.rand(n, 1, 1, device=DEV, generator=ga) < p).long()
        for kind in ("bloom", "exact"):
            m = filtered(base, kind, attrs)
            m_off = filtered(base, kind, attrs, backend="official", official=ocfg)
            for bs in BATCH_SIZES:
                pool_q = [q[:bs].contiguous() for q in pool16]
                probes = [m._phase1_probe_ids(q) for q in pool_q]
                lb = launches(before[kind], kind, m, pool_q, probes, qa)
                la = launches(after[kind], kind, m, pool_q, probes, qa)
                same = all(outputs_equal(b[1], a[1], b[0], a[0]) for b, a in zip(lb, la, strict=True))
                gb, gaft = graph_of(lb), graph_of(la)
                for _ in range(3):
                    window_us(gb), window_us(gaft)
                tb, ta, mhz = [], [], []
                for _ in range(args.pairs):
                    tb.append(window_us(gb))
                    mhz.append(sm_mhz())
                    ta.append(window_us(gaft))
                    mhz.append(sm_mhz())
                r, lo, hi = ratio_ci(tb, ta)
                o = official_us(m_off, kind, pool_q, probes, qa, ocfg)
                row = {"p": p, "mode": kind, "bs": bs, "before_us": statistics.median(tb),
                       "after_us": statistics.median(ta), "ratio": r, "ci_lo": lo, "ci_hi": hi,
                       "official_scorer_us": o["scorer"], "official_mask_us": o["mask"],
                       "scores_equal": same, "sm_mhz": [min(mhz), max(mhz)],
                       "unstable": max(mhz) - min(mhz) > 50}  # fmt: skip
                res["rows"].append(row)
                print(f"  p={p} {kind} bs={bs}: before {row['before_us']:.1f} after "
                      f"{row['after_us']:.1f} us ratio {r:.3f} [{lo:.3f}, {hi:.3f}] official "
                      f"scorer {o['scorer']:.1f} mask {o['mask']:.1f} equal {same} "
                      f"sm {row['sm_mhz']} {'UNSTABLE' if row['unstable'] else ''}", flush=True)
                del gb, gaft, lb, la
            del m, m_off
            torch.cuda.empty_cache()
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
