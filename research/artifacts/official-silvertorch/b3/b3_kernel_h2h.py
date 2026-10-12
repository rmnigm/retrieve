"""B3 (roadmap) — Triton vs the official SilverTorch kernels, kernel-only (plan O §9a/§9b).

One process per dataset. Phase 1 (centroid matmul + probe top-k) is hoisted out of every
timed region: it is our code in both arms and identical by construction (same k-means seed,
same assignments), so what is timed is Algorithm 1 phases 2+3 only — the part where the two
implementations differ.

Three tiers, all on real data (the harness's own inputs, sweeps and query pool):

  A. phases 2+3 wall, per (filter mode, batch size, arm), with the harness estimator
     (``bench.measure.latency``: 50 warm-ups, 3 windows, median of window medians, spread,
     the SM clock sampled under load) and a ``torch.profiler`` per-kernel split classified
     into scorer / prep / mask / topk / quantize / other — the kernel-only row.
  B. phase 2 alone: our row-wise ``bloom_match`` and ``clause_mask`` (full N) against the
     official ``bloom_index_search_batch`` (full N, packed) and
     ``…_return_partial_response`` (probed clusters only), plus the CPU expression parse.
  C. parity and selectivity alongside speed: jaccard@100 and ``score_max_abs_diff`` of each
     official arm against the Triton arm on the same batches, on both the int32 (bit-exact)
     and fp16 (shipped) score paths; bloom false-positive rate and index bytes of both
     blooms, and the official bloom's FPR/bytes against ``b_multiplier``.

Usage (from ``evaluation/``):  python b3_kernel_h2h.py goodreads c0_genre out.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from bench import algos, config, inputs, measure, metrics
from retrieve.ops import official as off
from retrieve.ops import triton as T

DIM = 128
SEED = 0
N_PROBE = 24
K = 100
BATCH_SIZES = (1, 16)
N_POOL = 32  # pool batches rotated inside a timed window
PARITY_BATCHES = 32  # batches compared for jaccard / score_max_abs_diff (bs=16 only)
DEVICE = torch.device("cuda")

# torch.profiler kernel-name → class. First match wins; order matters.
KERNEL_CLASSES = (
    ("scorer", ("process_cluster", "codesigned_probe_score")),
    ("mask", ("process_documents", "bloom_match", "clause_mask", "bloom_hash")),
    ("topk", ("topk", "sort", "RadixSelect", "gatherTopK", "Kernel_computeBlockwiseWithinKCounts",
              "Kernel_computeBlockwiseKthCounts", "Kernel_computeDigitCumSum", "radix")),
    ("quantize", ("quantize", "abs_max", "reduce_kernel", "amax")),
    ("prep", ("generate_", "cumsum", "repeat_interleave", "arange", "fill", "scan", "Scan",
              "index_select", "gather", "scatter", "IndexKernel", "copy", "elementwise",
              "vectorized_elementwise", "reduce")),
)


def classify(name: str) -> str:
    for cls, needles in KERNEL_CLASSES:
        if any(n.lower() in name.lower() for n in needles):
            return cls
    return "other"


def kernel_split(fn) -> dict[str, Any]:
    """One profiled call: per-class device µs and the full kernel list."""
    ks = measure.profile_once(fn, top=64)
    by_class: dict[str, float] = {}
    for e in ks:
        by_class[classify(e["kernel"])] = by_class.get(classify(e["kernel"]), 0.0) + e["us"]
    return {
        "device_us_total": sum(e["us"] for e in ks),
        "device_us_by_class": by_class,
        "n_kernel_launches": sum(e["calls"] for e in ks),
        "kernels": ks[:12],
    }


def cpu_latency(fn, n: int = 200) -> dict[str, Any]:
    """Wall-clock ms per call for a host-side op: CUDA events would time an empty GPU
    timeline. 20 warm-ups, then ``n`` calls; median / p99 / min of the per-call walls."""
    for _ in range(20):
        fn()
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t) * 1e3)
    t = torch.tensor(ms, dtype=torch.float64)
    q = torch.quantile(t, torch.tensor([0.5, 0.99], dtype=torch.float64)).tolist()
    return {"median_ms": q[0], "p99_ms": q[1], "min_ms": float(t.min()), "n": n,
            "timer": "perf_counter (host-side op)"}


def build_modules(item_embs, item_attrs, reverse, mode: str, backend: str, official_cfg=None):
    kw = dict(k=K, backend=backend, filter_kind={"none": "none", "bloom": "bloom",
                                                 "exact": "clause"}[mode],
              item_attrs=item_attrs, clause_is_reverse=reverse, seed=SEED,
              params={"n_probe": N_PROBE})
    if backend == "official" and official_cfg is not None:
        # algos.build has no OfficialConfig hook; construct the same module by hand.
        from retrieve import SilverTorch

        bloom = algos.BLOOM_DEFAULTS if mode == "bloom" else {}
        m = SilverTorch(k=K, filter_mode=mode, seed=SEED, backend="official",
                        official=official_cfg,
                        **{**algos.SILVERTORCH_DEFAULTS, "n_probe": N_PROBE, **bloom})
        if mode == "none":
            m.register_index(item_embs)
        else:
            m.register_index(item_embs, item_clause_attrs=item_attrs,
                             clause_is_reverse=reverse if mode == "exact" else None)
        return m
    return algos.build("silvertorch", item_embs, **kw)


def arms_for(mode: str, tri, offi, pool_q, pool_a, cfg_int32, cfg_fp16):
    """Zero-arg callables, one per arm, each rotating the same pool of batches.

    Every arm covers exactly Algorithm 1 phases 2+3 as its backend implements them: the
    Triton arm is one fused launch plus the shared ``masked_topk``; the official arm is the
    mask op(s) its filter mode needs, ``fused_kmean_ann*``, the host epilogue and the same
    ``masked_topk``. Phase 1 (``flat_items`` / ``probe_ids``) is precomputed per pool batch.
    """
    n = len(pool_q)
    tri_flat = [tri._phase1_probe(q) for q in pool_q]
    off_probe = [offi._phase1_probe_ids(q) for q in pool_q]
    state = {"i": 0}

    def nxt() -> int:
        i = state["i"]
        state["i"] = (i + 1) % n
        return i

    def triton_arm():
        i = nxt()
        q, flat = pool_q[i], tri_flat[i]
        if mode == "none":
            return T.codesigned_probe_score(q, flat, tri.item_codes, tri._global_scale_f, tri.k)
        if mode == "bloom":
            qb = tri._query_bits(pool_a[i])
            return T.codesigned_probe_score_bloom(q, flat, tri.item_codes, qb, tri.bloom_sigs,
                                                  tri._global_scale_f, tri.k)
        return T.codesigned_probe_score_exact(q, flat, tri.item_codes, tri.item_clause_attrs,
                                              tri.clause_is_reverse, pool_a[i].long(),
                                              tri._global_scale_f, tri.k)

    def official_arm_factory(cfg):
        def arm():
            i = nxt()
            q, probe_ids = pool_q[i], off_probe[i]
            mask = partial = None
            if mode == "exact":
                m = T.clause_mask(offi.item_clause_attrs, offi.clause_is_reverse,
                                  pool_a[i].long())
                mask = off.pack_mask(m, off.MASK_BIT_ORDER)
            elif mode == "bloom":
                expressions = off.queries_to_expressions(pool_a[i])
                plans = off.parse_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries,
                                        cache=cfg.cache_plans)
                partial = off.bloom_partial_masks(
                    offi.bloom_index, offi.bundle_b_offsets, plans,
                    offi.cluster_offsets[probe_ids], offi.cluster_sizes[probe_ids],
                    offi.k_hash, cfg.n_stored_hashes)
            return off.official_probe_score(
                q, probe_ids, offi.cluster_offsets, offi.cluster_sizes, offi.item_codes,
                offi.sort_perm, offi.global_scale, offi.k, offi.n_probe * offi._max_cluster_size,
                score_path=cfg.score_path, divisor=cfg.divisor, filtering_bit_mask=mask,
                partial=partial)

        return arm

    return {
        "triton": triton_arm,
        "official-fp16": official_arm_factory(cfg_fp16),
        "official-int32": official_arm_factory(cfg_int32),
    }, tri_flat, off_probe


def parity(mode, tri, offi, pool_q, pool_a, tri_flat, off_probe, cfg):
    """jaccard@K and score_max_abs_diff of one official arm against the Triton arm."""
    jac, dmax, n = 0.0, 0.0, 0
    for i in range(min(PARITY_BATCHES, len(pool_q))):
        q, flat, probe_ids = pool_q[i], tri_flat[i], off_probe[i]
        if mode == "none":
            t_ids, t_sc = T.codesigned_probe_score(q, flat, tri.item_codes,
                                                   tri._global_scale_f, tri.k)
            mask = partial = None
        elif mode == "bloom":
            qb = tri._query_bits(pool_a[i])
            t_ids, t_sc = T.codesigned_probe_score_bloom(q, flat, tri.item_codes, qb,
                                                         tri.bloom_sigs, tri._global_scale_f,
                                                         tri.k)
            expressions = off.queries_to_expressions(pool_a[i])
            plans = off.parse_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries,
                                    cache=True)
            partial = off.bloom_partial_masks(offi.bloom_index, offi.bundle_b_offsets, plans,
                                              offi.cluster_offsets[probe_ids],
                                              offi.cluster_sizes[probe_ids], offi.k_hash,
                                              cfg.n_stored_hashes)
            mask = None
        else:
            t_ids, t_sc = T.codesigned_probe_score_exact(q, flat, tri.item_codes,
                                                         tri.item_clause_attrs,
                                                         tri.clause_is_reverse,
                                                         pool_a[i].long(),
                                                         tri._global_scale_f, tri.k)
            m = T.clause_mask(offi.item_clause_attrs, offi.clause_is_reverse, pool_a[i].long())
            mask, partial = off.pack_mask(m, off.MASK_BIT_ORDER), None
        o_ids, o_sc = off.official_probe_score(
            q, probe_ids, offi.cluster_offsets, offi.cluster_sizes, offi.item_codes,
            offi.sort_perm, offi.global_scale, offi.k, offi.n_probe * offi._max_cluster_size,
            score_path=cfg.score_path, divisor=cfg.divisor, filtering_bit_mask=mask,
            partial=partial)
        jac += metrics.jaccard_at_k(t_ids, o_ids, tri.k) * t_ids.shape[0]
        finite = torch.isfinite(t_sc) & torch.isfinite(o_sc)
        if finite.any():
            dmax = max(dmax, float((t_sc[finite] - o_sc[finite]).abs().max()))
        n += t_ids.shape[0]
    return {"jaccard_at_k": jac / n, "score_max_abs_diff": dmax, "n_queries": n,
            "k": tri.k, "score_path": cfg.score_path}


def boundary_stats(mode, tri, pool_q, pool_a, tri_flat, k_wide: int = 200):
    """How closely packed the scores are at the top-k boundary — the quantity that decides
    how much an fp16 score path costs in rank agreement. Per row: the gap between rank K and
    rank K+1 of the exact (Triton, int32-accumulated) scores."""
    gaps = []
    scale = []
    for i in range(min(PARITY_BATCHES, len(pool_q))):
        q, flat = pool_q[i], tri_flat[i]
        if mode == "none":
            _, sc = T.codesigned_probe_score(q, flat, tri.item_codes, tri._global_scale_f, k_wide)
        elif mode == "bloom":
            qb = tri._query_bits(pool_a[i])
            _, sc = T.codesigned_probe_score_bloom(q, flat, tri.item_codes, qb, tri.bloom_sigs,
                                                   tri._global_scale_f, k_wide)
        else:
            _, sc = T.codesigned_probe_score_exact(q, flat, tri.item_codes, tri.item_clause_attrs,
                                                   tri.clause_is_reverse, pool_a[i].long(),
                                                   tri._global_scale_f, k_wide)
        ok = torch.isfinite(sc[:, K - 1]) & torch.isfinite(sc[:, K])
        if ok.any():
            gaps.append((sc[ok, K - 1] - sc[ok, K]).double())
            scale.append(sc[ok, K - 1].abs().double())
    if not gaps:
        return None
    g, v = torch.cat(gaps), torch.cat(scale)
    qs = torch.quantile(
        g, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=g.device)
    ).tolist()
    return {"mode": mode, "k": K, "n_rows": int(g.numel()),
            "gap_p10": qs[0], "gap_median": qs[1], "gap_p90": qs[2],
            "score_at_k_median": float(v.median()),
            "gap_over_score_median": float((g / v.clamp_min(1e-12)).median()),
            "fp16_ulp_of_score_median": float((v * 2 ** -11).median()),
            "frac_gap_below_fp16_ulp": float((g < v * 2 ** -11).double().mean())}


def phase2_rows(tri, offi, pool_q, pool_a, bs_idx, cfg):
    """Tier B: phase 2 alone, ours (row-wise, full N) vs official (transposed)."""
    rows = []
    qa = pool_a[bs_idx]
    probe_ids = offi._phase1_probe_ids(pool_q[bs_idx])
    sel_off, sel_len = offi.cluster_offsets[probe_ids], offi.cluster_sizes[probe_ids]
    expressions = off.queries_to_expressions(qa)
    plans_cached = off.parse_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries,
                                   cache=True)

    def ours_bloom():
        qb = tri._query_bits(qa)
        return T.bloom_match(qb, tri.bloom_sigs)

    def ours_query_bits():
        return tri._query_bits(qa)

    def off_full():
        return off.bloom_filtering_mask(offi.bloom_index, offi.bundle_b_offsets, plans_cached,
                                        offi.k_hash, cfg.n_stored_hashes)

    def off_partial():
        return off.bloom_partial_masks(offi.bloom_index, offi.bundle_b_offsets, plans_cached,
                                       sel_off, sel_len, offi.k_hash, cfg.n_stored_hashes)

    def parse_only():
        return off.parse_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries,
                               cache=False)

    def expr_only():
        return off.queries_to_expressions(qa)

    for name, fn, scope, host in (
        ("ours_query_bits (bloom hash, [B,W])", ours_query_bits, "ours", False),
        ("ours_bloom_match (row-wise, full N)", ours_bloom, "ours", False),
        ("official_bloom_index_search_batch (full N, packed)", off_full, "official", False),
        ("official_return_partial_response (probed only)", off_partial, "official", False),
        ("official_queries_to_expressions (host, incl. D2H)", expr_only, "official", True),
        ("official_parse_expression_query_batch (host)", parse_only, "official", True),
    ):
        if host:
            row = {"op": name, "arm": scope, "bs": qa.shape[0], **cpu_latency(fn)}
        else:
            perf, _ = measure.latency(fn, bs=qa.shape[0], mode="eager", n_min=200, n_max=2000)
            row = {"op": name, "arm": scope, "bs": qa.shape[0],
                   "median_ms": perf["median_ms"], "p99_ms": perf["p99_ms"],
                   "min_ms": perf["min_ms"], "spread": perf["spread"],
                   "unstable": perf["unstable"], "sm_mhz": perf["sm_mhz"], "n": perf["n"],
                   "timer": "cuda_events", **kernel_split(fn)}
        rows.append(row)
    return rows


def bloom_selectivity(tri, offi, item_attrs, reverse, pool_a, cfg, n_items):
    """FPR and bytes of both blooms at the shipped settings, on real attributes."""
    exact_filter = algos.build_filter("clause", item_attrs, clause_is_reverse=reverse,
                                      backend="triton")
    tot_ours = tot_off = tot_exact = 0.0
    rows = 0
    for i in range(8):
        qa = pool_a[i]
        ex_cnt = exact_filter(qa.long()).sum(1).double()
        qb = tri._query_bits(qa)
        ours = T.bloom_match(qb, tri.bloom_sigs).sum(1).double()
        expressions = off.queries_to_expressions(qa)
        plans = off.parse_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries, cache=True)
        packed = off.bloom_full_mask(offi.bloom_index, offi.bundle_b_offsets, plans, offi.k_hash,
                                     cfg.n_stored_hashes, return_bool_mask=True)
        offc = packed[:, :n_items].sum(1).double()
        denom = (n_items - ex_cnt).clamp_min(1)
        tot_ours += float(((ours - ex_cnt).clamp_min(0) / denom).sum())
        tot_off += float(((offc - ex_cnt).clamp_min(0) / denom).sum())
        tot_exact += float((ex_cnt / n_items).sum())
        rows += qa.shape[0]
    return {
        "n_queries": rows,
        "exact_pass_rate": tot_exact / rows,
        "ours_bloom_fp_rate": tot_ours / rows,
        "official_bloom_fp_rate": tot_off / rows,
        "ours_index_mib": tri.bloom_sigs.numel() * tri.bloom_sigs.element_size() / 2**20,
        "official_index_mib": (offi.bloom_index.numel() * offi.bloom_index.element_size()
                               + offi.bundle_b_offsets.numel() * 8) / 2**20,
        "ours_bits_per_doc": tri.m_bits,
        "official_b_multiplier": cfg.b_multiplier,
    }


def main() -> None:
    ds_name, sweep, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    measure.setup(SEED)
    measure.warm_gpu_once()
    ds = config.load_dataset(Path(f"config/{ds_name}.yaml"), DIM)
    inp = inputs.load_inputs(ds, DEVICE, with_filters=True)
    clauses = ds.clauses["clause"].get(sweep) or ds.clauses["bloom"][sweep]
    qa_sweep, skip_mask = inputs.sweep_qa(inp["qa"], tuple(clauses))
    item_embs, item_attrs, reverse = inp["item_embs"], inp["item_attrs"], inp["clause_is_reverse"]
    n_items = int(item_embs.shape[0])

    cfg_fp16 = off.OfficialConfig(score_path="fp16", cache_plans=False)
    cfg_int32 = off.OfficialConfig(score_path="int32", cache_plans=False)
    out: dict[str, Any] = {
        "dataset": ds_name, "dim": DIM, "sweep": sweep, "clauses": list(clauses),
        "n_items": n_items, "k": K, "n_probe": N_PROBE, "n_lists": algos.SILVERTORCH_DEFAULTS[
            "n_lists"], "seed": SEED,
        "env": {**measure.provenance(), **{f"idle_{k}": v for k, v in measure.clocks().items()}},
        "official_config": {"score_path_timed": ["fp16", "int32"], "bloom_path": "partial",
                            "b_multiplier": cfg_fp16.b_multiplier,
                            "n_stored_hashes": cfg_fp16.n_stored_hashes,
                            "cache_plans": False},
        "bloom_defaults": algos.BLOOM_DEFAULTS,
        "tiers": {"A_phase23": [], "B_phase2": [], "C_parity": [], "C_bloom": None},
    }

    for mode in ("none", "bloom", "exact"):
        t0 = time.perf_counter()
        tri = build_modules(item_embs, item_attrs, reverse, mode, "triton")
        offi = build_modules(item_embs, item_attrs, reverse, mode, "official",
                             official_cfg=cfg_fp16)
        same = {
            "centroids_equal": bool(torch.equal(tri.centroids, offi.centroids)),
            "codes_equal_through_sort_perm": bool(
                torch.equal(offi.item_codes, tri.item_codes[offi.sort_perm])),
            "global_scale_equal": bool(torch.equal(tri.global_scale, offi.global_scale)),
            "max_tensor_size_per_row": int(offi.n_probe * offi._max_cluster_size),
            "p_slots": int(off.padded_rows(offi.n_probe * offi._max_cluster_size)),
            "mode": mode,
        }
        out.setdefault("shared_index_check", []).append(same)
        print(f"[{ds_name}] built {mode} in {time.perf_counter() - t0:.1f}s {same}", flush=True)
        for bs in BATCH_SIZES:
            pool_q, pool_a = inputs.query_pool(
                inp, qa_sweep if mode != "none" else None,
                skip_mask if mode != "none" else None,
                bs=bs, seed=SEED, n_pool=N_POOL, device=DEVICE)
            pool_q = list(pool_q)
            pool_a = list(pool_a) if pool_a is not None else [None] * len(pool_q)
            arms, tri_flat, off_probe = arms_for(mode, tri, offi, pool_q, pool_a,
                                                 cfg_int32, cfg_fp16)
            # The harness times every forward under inference_mode (run.perf); match it, so
            # the kernel-only and end-to-end numbers differ only in what is inside the call.
            ctx = torch.inference_mode()
            ctx.__enter__()
            for arm_name, fn in arms.items():
                perf, _ = measure.latency(fn, bs=bs, mode="eager")
                row = {"tier": "A", "mode": mode, "bs": bs, "arm": arm_name,
                       "median_ms": perf["median_ms"], "mean_ms": perf["mean_ms"],
                       "p99_ms": perf["p99_ms"], "min_ms": perf["min_ms"],
                       "iqr_ms": perf["iqr_ms"], "qps": perf["qps"],
                       "host_gap_ms": perf["host_gap_ms"], "spread": perf["spread"],
                       "unstable": perf["unstable"], "sm_mhz": perf["sm_mhz"],
                       "peak_fwd_mib": perf["peak_fwd_mib"], "n": perf["n"],
                       "window_medians_ms": perf["window_medians_ms"],
                       **kernel_split(fn)}
                out["tiers"]["A_phase23"].append(row)
                print(f"  A {mode} bs={bs} {arm_name}: {row['median_ms']:.4f} ms "
                      f"(spread {row['spread']:.3f}, sm {row['sm_mhz']})", flush=True)
            for cfg in (cfg_int32, cfg_fp16) if bs == 16 else ():
                p = parity(mode, tri, offi, pool_q, pool_a, tri_flat, off_probe, cfg)
                p.update(tier="C", mode=mode, bs=bs)
                out["tiers"]["C_parity"].append(p)
                print(f"  C {mode} bs={bs} {cfg.score_path}: jaccard "
                      f"{p['jaccard_at_k']:.6f} dmax {p['score_max_abs_diff']:.3e}", flush=True)
            if bs == 16:
                b = boundary_stats(mode, tri, pool_q, pool_a, tri_flat)
                out["tiers"].setdefault("C_boundary", []).append(b)
                print(f"  C {mode} boundary: {b}", flush=True)
            if mode == "bloom" and bs == 16:
                out["tiers"]["B_phase2"] += phase2_rows(tri, offi, pool_q, pool_a, 0, cfg_fp16)
                out["tiers"]["C_bloom"] = bloom_selectivity(tri, offi, item_attrs, reverse,
                                                            pool_a, cfg_fp16, n_items)
            ctx.__exit__(None, None, None)
            del pool_q, pool_a, tri_flat, off_probe
            torch.cuda.empty_cache()
        del tri, offi
        torch.cuda.empty_cache()

    out["env"]["load_sm_mhz_after"] = measure.clocks()["sm_mhz"]
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
