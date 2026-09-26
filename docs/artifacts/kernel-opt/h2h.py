"""Kernel-only head-to-head of SilverTorch phases 2+3, Triton against Meta's official kernels,
re-run for TF-9 / TF-1 with the b3 methodology (docs/artifacts/official-silvertorch/b3/
b3_kernel_h2h.py, tier A and the int32 parity of tier C): phase 1 hoisted out, the harness's
`measure.latency` (eager, inference_mode) plus a one-call `torch.profiler` split into scorer /
mask / prep / topk / quantize / other device µs.

It runs against either library tree, so `interleave`-style alternation can time the padded
(pre-TF-9) and compact layouts in separate processes with the official arm as a control in
both. The tree is chosen by PYTHONPATH; the API is detected from the module's buffers.

    RETRIEVE_SRC=/path/to/retrieve/src python h2h.py goodreads c0_genre out.json   (from evaluation/)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if os.environ.get("RETRIEVE_SRC"):
    sys.path.insert(0, os.environ["RETRIEVE_SRC"])

import torch  # noqa: E402

import retrieve  # noqa: E402
from bench import algos, config, inputs, measure, metrics  # noqa: E402
from retrieve import SilverTorch  # noqa: E402
from retrieve.ops import official as off, triton as T  # noqa: E402

DIM, SEED, N_PROBE, K = 128, 0, 24, 100
BATCH_SIZES = (1, 16)
N_POOL = 32
PARITY_BATCHES = 16
DEVICE = torch.device("cuda")

KERNEL_CLASSES = (
    ("scorer", ("process_cluster", "codesigned_probe_score")),
    ("mask", ("process_documents", "bloom_match", "clause_mask", "bloom_hash")),
    ("topk", ("topk", "sort", "RadixSelect", "gatherTopK", "Kernel_computeBlockwiseWithinKCounts",
              "Kernel_computeBlockwiseKthCounts", "Kernel_computeDigitCumSum", "radix")),
    ("quantize", ("quantize", "abs_max", "reduce_kernel", "amax")),
    ("prep", ("generate_", "cumsum", "repeat_interleave", "arange", "fill", "scan", "Scan",
              "index_select", "gather", "scatter", "IndexKernel", "copy", "elementwise",
              "vectorized_elementwise", "reduce", "searchsorted")),
)  # fmt: skip


def classify(name: str) -> str:
    for cls, needles in KERNEL_CLASSES:
        if any(n.lower() in name.lower() for n in needles):
            return cls
    return "other"


def kernel_split(fn) -> dict[str, Any]:
    ks = measure.profile_once(fn, top=64)
    by_class: dict[str, float] = {}
    for e in ks:
        by_class[classify(e["kernel"])] = by_class.get(classify(e["kernel"]), 0.0) + e["us"]
    return {"device_us_total": sum(e["us"] for e in ks), "device_us_by_class": by_class,
            "n_kernel_launches": sum(e["calls"] for e in ks), "kernels": ks[:12]}  # fmt: skip


def build(item_embs, item_attrs, reverse, mode, backend, official_cfg=None):
    bloom = algos.BLOOM_DEFAULTS if mode == "bloom" else {}
    m = SilverTorch(k=K, filter_mode=mode, seed=SEED, backend=backend, official=official_cfg,
                    **{**algos.SILVERTORCH_DEFAULTS, "n_probe": N_PROBE, **bloom})  # fmt: skip
    if mode == "none":
        m.register_index(item_embs)
    else:
        m.register_index(item_embs, item_attrs, reverse if mode == "exact" else None)
    return m


def triton_arm_factory(mode, tri, pool_q, pool_a):
    """The Triton phases 2+3 on the padded layout (pre-TF-9 tree) or the compact one."""
    padded = hasattr(tri, "padded_cluster_items")
    phase1 = [tri._phase1_probe(q) if padded else tri._phase1_probe_ids(q) for q in pool_q]
    state = {"i": 0}

    def arm(i=None):
        if i is None:
            i = state["i"] = (state["i"] + 1) % len(pool_q)
        q, p1, qa = pool_q[i], phase1[i], pool_a[i]
        gs = tri._global_scale_f
        if padded:
            if mode == "none":
                return T.codesigned_probe_score(q, p1, tri.item_codes, gs, tri.k)
            if mode == "bloom":
                return T.codesigned_probe_score_bloom(q, p1, tri.item_codes, tri._query_bits(qa),
                                                      tri.bloom_sigs, gs, tri.k)  # fmt: skip
            return T.codesigned_probe_score_exact(q, p1, tri.item_codes, tri.item_clause_attrs,
                                                  tri.clause_is_reverse, qa.long(), gs, tri.k)  # fmt: skip
        lay = (p1, tri.cluster_offsets, tri.item_codes, tri.sort_perm)
        tail = (gs, tri.k, tri._probe_width)
        if mode == "none":
            return T.codesigned_probe_score(q, *lay, *tail)
        if mode == "bloom":
            return T.codesigned_probe_score_bloom(q, *lay, tri._query_bit_positions(qa),
                                                  tri.bloom_transposed, *tail)  # fmt: skip
        return T.codesigned_probe_score_exact(q, *lay, tri.item_clause_attrs,
                                              tri.clause_is_reverse, qa.long(), *tail)  # fmt: skip

    return arm


def official_arm_factory(mode, offi, pool_q, pool_a, cfg):
    probes = [offi._phase1_probe_ids(q) for q in pool_q]
    width = getattr(offi, "_probe_width", None) or offi.n_probe * getattr(offi, "_max_cluster_size", 0)
    state = {"i": 0}

    def arm(i=None):
        if i is None:
            i = state["i"] = (state["i"] + 1) % len(pool_q)
        q, probe_ids = pool_q[i], probes[i]
        mask = partial = None
        if mode == "exact":
            m = T.clause_mask(offi.item_clause_attrs, offi.clause_is_reverse, pool_a[i].long())
            mask = off.pack_mask(m, off.MASK_BIT_ORDER)
        elif mode == "bloom":
            plans = off.parse_plans(off.queries_to_expressions(pool_a[i]), cfg.n_stored_hashes,
                                    cfg.max_sub_queries, cache=cfg.cache_plans)  # fmt: skip
            partial = off.bloom_partial_masks(
                offi.bloom_index, offi.bundle_b_offsets, plans, offi.cluster_offsets[probe_ids],
                offi.cluster_sizes[probe_ids], offi.k_hash, cfg.n_stored_hashes)  # fmt: skip
        return off.official_probe_score(
            q, probe_ids, offi.cluster_offsets, offi.cluster_sizes, offi.item_codes,
            offi.sort_perm, offi.global_scale, offi.k, width, score_path=cfg.score_path,
            divisor=cfg.divisor, filtering_bit_mask=mask, partial=partial)  # fmt: skip

    return arm, width


def main() -> None:
    ds_name, sweep, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    measure.setup(SEED)
    measure.warm_gpu_once()
    ds = config.load_dataset(Path(f"config/{ds_name}.yaml"), DIM)
    inp = inputs.load_inputs(ds, DEVICE, with_filters=True)
    clauses = ds.clauses["clause"].get(sweep) or ds.clauses["bloom"][sweep]
    qa_sweep, skip_mask = inputs.sweep_qa(inp["qa"], tuple(clauses))
    item_embs, item_attrs, reverse = inp["item_embs"], inp["item_attrs"], inp["clause_is_reverse"]
    cfg16 = off.OfficialConfig(score_path="fp16", cache_plans=False)
    cfg32 = off.OfficialConfig(score_path="int32", cache_plans=False)
    out: dict[str, Any] = {"dataset": ds_name, "sweep": sweep, "n_items": int(item_embs.shape[0]),
                           "k": K, "n_probe": N_PROBE, "retrieve": retrieve.__file__,
                           "env": measure.provenance(), "rows": [], "parity": []}  # fmt: skip
    for mode in ("none", "bloom", "exact"):
        t0 = time.perf_counter()
        tri = build(item_embs, item_attrs, reverse, mode, "triton")
        offi = build(item_embs, item_attrs, reverse, mode, "official", cfg16)
        print(f"[{ds_name}] built {mode} in {time.perf_counter() - t0:.1f}s", flush=True)
        for bs in BATCH_SIZES:
            pool_q, pool_a = inputs.query_pool(
                inp, qa_sweep if mode != "none" else None, skip_mask if mode != "none" else None,
                bs=bs, seed=SEED, n_pool=N_POOL, device=DEVICE)  # fmt: skip
            pool_q = list(pool_q)
            pool_a = list(pool_a) if pool_a is not None else [None] * len(pool_q)
            tri_arm = triton_arm_factory(mode, tri, pool_q, pool_a)
            o16, width = official_arm_factory(mode, offi, pool_q, pool_a, cfg16)
            o32, _ = official_arm_factory(mode, offi, pool_q, pool_a, cfg32)
            with torch.inference_mode():
                for name, fn in (("triton", tri_arm), ("official-fp16", o16),
                                 ("official-int32", o32)):  # fmt: skip
                    perf, _ = measure.latency(fn, bs=bs, mode="eager")
                    row = {"mode": mode, "bs": bs, "arm": name, "width": width,
                           **{k: perf[k] for k in ("median_ms", "p99_ms", "spread", "unstable",
                                                   "sm_mhz", "peak_fwd_mib")},
                           **kernel_split(fn)}  # fmt: skip
                    out["rows"].append(row)
                    c = row["device_us_by_class"]
                    print(f"  {mode} bs={bs} {name}: wall {row['median_ms']:.4f} ms, scorer "
                          f"{c.get('scorer', 0):.1f} mask {c.get('mask', 0):.1f} topk "
                          f"{c.get('topk', 0):.1f} total {row['device_us_total']:.1f} us "
                          f"(sm {row['sm_mhz']}, unstable {row['unstable']})", flush=True)  # fmt: skip
                if bs == 16:
                    jac, dmax, n = 0.0, 0.0, 0
                    for i in range(PARITY_BATCHES):
                        ti, ts = tri_arm(i)
                        oi, os_ = o32(i)
                        jac += metrics.jaccard_at_k(ti, oi, K) * ti.shape[0]
                        fin = torch.isfinite(ts) & torch.isfinite(os_)
                        if fin.any():
                            dmax = max(dmax, float((ts[fin] - os_[fin]).abs().max()))
                        n += ti.shape[0]
                    out["parity"].append({"mode": mode, "jaccard_at_k": jac / n,
                                          "score_max_abs_diff": dmax, "n_queries": n})  # fmt: skip
                    print(f"  {mode} parity vs official-int32: jaccard {jac / n:.6f} "
                          f"dmax {dmax:.3e}", flush=True)  # fmt: skip
            del pool_q, pool_a
            torch.cuda.empty_cache()
        del tri, offi
        torch.cuda.empty_cache()
    Path(out_path).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
