"""Check 2: LiNR V3 at LiNR's 512-bit operating point (LiNR §5.3, Fig. 7) against 128 bits.

Sign-OPORP bins D coordinates into k_bits groups, so k_bits must divide D (quantize.py
``_oporp_k_bits``): 512 bits at D = 128 does not exist for OPORP. The library's k_bits > D stage 1
is ``SimHashKNN`` (Gaussian projection), so the arms are
  v3_oporp128   the study's V3 (OneBitKNN, 128 bits)
  v3_simhash128 SimHash at 128 bits (separates the projection from the bit count)
  v3_simhash512 SimHash at 512 bits, LiNR's bit count
  v3_oporp768   pubmed only: the study's V3 there (k_bits = D)
each with stage 2 the exact rescoring (shared PrefilterKNN). Pools: LiNR's 1 % of the catalog
(``linrN``) and the harness's fractions of the mean pass count with its 2000 floor.
Quality: recall@100 = overlap with the exact filtered top-100 (LiNR V2), first QUALITY_Q kept
queries. Timing (graph, harness windows) on TIMED sweeps and TIMED_POOLS only, with V1 / V2.

    cd evaluation && python ../research/artifacts/gpu-checks/check2_v3bits.py [datasets...]
"""

from __future__ import annotations

import copy
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import Records, env, load, log, synth_attrs  # noqa: E402

from bench import algos, measure  # noqa: E402
from bench import inputs as binputs  # noqa: E402
from bench.run import POOL_MIN, _rotate, prepare_pool  # noqa: E402
from retrieve import LiNRV3, OneBitKNN, SimHashKNN  # noqa: E402

K = 100
BATCHES = [1, 16]
SYNTH = [0.11, 0.2, 0.5]
FRACS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
QUALITY_Q = 2000
CHUNK = 16
TIMED = {
    "goodreads": ["c0_genre", "c1_lang_reverse", "all4", "p0.11", "p0.2", "p0.5"],
    "arxiv": ["c0_maincat", "c2_year", "all4", "p0.11", "p0.2", "p0.5"],
    "pubmed": ["c0_mesh", "c2_year", "all5", "p0.11", "p0.2", "p0.5"],
}
TIMED_ARMS = ["v3_oporp128", "v3_simhash512"]
TIMED_POOLS = ["linrN", "f0.01", "f0.05"]


def stage1s(dim: int, item_embs: torch.Tensor) -> dict[str, torch.nn.Module]:
    arms = {
        "v3_oporp128": OneBitKNN(k=POOL_MIN, seed=0, k_bits=128),
        "v3_simhash128": SimHashKNN(k=POOL_MIN, k_bits=128, seed=0),
        "v3_simhash512": SimHashKNN(k=POOL_MIN, k_bits=512, seed=0),
    }
    if dim != 128:
        arms["v3_oporp768"] = OneBitKNN(k=POOL_MIN, seed=0, k_bits=dim)
    for m in arms.values():
        m.to(item_embs.device).register_index(item_embs)
    return arms


def v3_of(stage1: torch.nn.Module, stage2: torch.nn.Module, filt) -> LiNRV3:
    m = LiNRV3(K, candidate_pool=POOL_MIN, filter=filt)
    m.stage1, m.stage2 = stage1, stage2
    return m


def sweeps(ds, inp, device):
    """``(name, filter, qa_sweep, skip)``: the dataset's real clause sweeps, then synthetic."""
    real = algos.build_filter(
        "clause", inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"]
    )
    for name, clauses in ds.clauses["clause"].items():
        qa_s, skip = binputs.sweep_qa(inp["qa"], clauses)
        yield name, real, qa_s, skip
    flags = synth_attrs(inp["n_items"], SYNTH, device)
    for j, rate in enumerate(SYNTH):
        attrs = flags[:, j : j + 1, :].contiguous()
        f = algos.build_filter(
            "clause",
            attrs,
            clause_is_reverse=torch.zeros(1, dtype=torch.bool, device=device),
        )
        yield f"p{rate}", f, torch.ones(inp["n_queries"], 1, dtype=torch.int64), None


@torch.inference_mode()
def topk_all(module, queries, prepared) -> torch.Tensor:
    return torch.cat([module(q, p)[0] for q, p in zip(queries, prepared, strict=True)])


def recall(ids: torch.Tensor, ref: torch.Tensor, counts: torch.Tensor) -> float:
    hit = ((ids[:, :, None] == ref[:, None, :]) & (ref[:, None, :] >= 0)).any(1).sum(1)
    denom = counts.clamp(max=K)
    keep = denom > 0
    return (hit[keep].double() / denom[keep].double()).mean().item()


def main(names: list[str]) -> None:
    warnings.filterwarnings("ignore")
    device = torch.device("cuda")
    measure.setup(0)
    measure.warm_gpu_once()
    recs = Records("check2", "v3bits")
    done = recs.done()
    provenance = env()
    for name in names:
        ds, inp = load(name, device)
        n = inp["n_items"]
        v1 = algos.build("linr_v1_filter_mask", inp["item_embs"], k=K, backend="triton",
                         filter_kind="clause", filter_mod=None)  # fmt: skip
        v2 = algos.build("linr_v2", inp["item_embs"], k=K, backend="triton", filter_kind="clause",
                         filter_mod=algos.build_filter("clause", inp["item_attrs"][:1]))  # fmt: skip
        s1 = stage1s(ds.dim, inp["item_embs"])
        del inp["item_embs"]
        torch.cuda.empty_cache()
        log(name, "items", n, "stage1", {a: m.item_bits.shape for a, m in s1.items()})
        for sweep, filt, qa_s, skip in sweeps(ds, inp, device):
            cell = f"{name}/{sweep}"
            if cell in done:
                continue
            v1.filter = filt
            v2.filter = filt
            arms = {a: v3_of(m, v2.idx, filt) for a, m in s1.items()}
            keep = (
                torch.arange(inp["n_queries"])
                if skip is None
                else (~skip).nonzero()[:, 0]
            )
            rows = keep[:QUALITY_Q]
            qs = inp["queries"][rows].to(device).split(CHUNK)
            qa_rows = qa_s[rows].to(device).split(CHUNK)
            prepared = [filt.prepare_queries(a) for a in qa_rows]
            with torch.inference_mode():
                counts = torch.cat(
                    [filt.evaluate_indices(p)[1] for p in prepared]
                ).long()
            ref = topk_all(v2, qs, prepared)
            mean_pass = counts.double().mean().item()
            pools = {"linrN": max(K, round(0.01 * n))} | {
                f"f{f}": max(POOL_MIN, round(f * mean_pass)) for f in FRACS
            }
            rec = {"cell": cell, "dataset": name, "dim": ds.dim, "n_items": n, "sweep": sweep,
                   "n_quality_queries": int(rows.numel()), "mean_pass_count": mean_pass,
                   "pass_rate": mean_pass / n, "pools": pools, "k": K, "env": provenance,
                   "recall": {}, "perf": {}}  # fmt: skip
            rec["recall"]["v1"] = recall(topk_all(v1, qs, prepared), ref, counts)
            for a, m in arms.items():
                for p, pool in pools.items():
                    m.set_query_params(candidate_pool=min(pool, n))
                    rec["recall"][f"{a}/{p}"] = recall(
                        topk_all(m, qs, prepared), ref, counts
                    )
            log(
                cell,
                f"pass {mean_pass / n:.4f}",
                {k: round(v, 4) for k, v in rec["recall"].items()},
            )
            if sweep in TIMED[name]:
                for bs in BATCHES:
                    pool, qa_pool = binputs.query_pool(
                        inp, qa_s, skip, bs=bs, seed=0, device=device
                    )
                    variants = {"v1": v1, "v2": v2}
                    for a in TIMED_ARMS:
                        for p in TIMED_POOLS:
                            # own k per variant: a shallow copy shares the bit buffers
                            m = v3_of(copy.copy(s1[a]), v2.idx, filt)
                            m.set_query_params(candidate_pool=min(pools[p], n))
                            variants[f"{a}/{p}"] = m
                    preps = {
                        v: prepare_pool(m, qa_pool)[0] for v, m in variants.items()
                    }
                    torch._dynamo.reset()
                    with torch.inference_mode():
                        graphs = {}
                        for v, m in variants.items():
                            try:
                                graphs[v] = measure.graph_callable(
                                    m, pool[0], preps[v][0]
                                )
                            except measure.NotCapturable as exc:
                                rec["perf"][f"{v}/bs{bs}"] = {"reason": str(exc)}
                        fns = [_rotate(c, pool, preps[v]) for v, c in graphs.items()]
                        for v, (d, _) in zip(
                            graphs,
                            measure.latency_group(fns, bs=bs, mode="graph"),
                            strict=True,
                        ):
                            rec["perf"][f"{v}/bs{bs}"] = d
                    torch._dynamo.reset()
                    med = {
                        v: rec["perf"][f"{v}/bs{bs}"].get("median_ms") for v in variants
                    }
                    best = min(med["v1"], med["v2"])
                    rec.setdefault("ratio_to_best_exact", {}).update(
                        {
                            f"{v}/bs{bs}": med[v] / best
                            for v in variants
                            if v.startswith("v3")
                        }
                    )
                    log(cell, f"bs{bs}", {v: round(x, 4) for v, x in med.items()})
                    del pool, qa_pool, preps, graphs, fns, variants
            rec["unstable_any"] = any(p.get("unstable") for p in rec["perf"].values())
            recs.append(rec)
            del arms, qs, qa_rows, prepared, ref
        del v1, v2, s1, inp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["goodreads", "arxiv", "pubmed"])
