"""Run d: SilverTorch with Meta's bloom NOT on the four reverse-clause real filters, in the head-to-head protocol.

The harness's bloom mode refuses reverse clauses, so, as gpu-checks check 3, the bloom index is built
forward-only and the query side hands the reverse flags to the adapter's ``queries_to_expressions``
(``NOT c:v`` is Meta's complement of the bloom's ``c:v`` mask). Bloom 1,024 bits, 5 hashes, configured width,
no recheck: the head-to-head's SilverTorch. k-means: the current inner-product one, random init, 10 iterations.

Per dataset and ``n_lists`` (the final run's grids): quality over an ``n_probe`` grid up to ``n_lists / 4``
against the harness's exhaustive fp32 oracle (``bench.oracle.compute``), recall@k at k 100 / 1000 / 2048 over
every kept query; per (k, target 0.90 / 0.95) the smallest ``n_probe`` reaching the target. Those points are
timed, graph mode, in one round-robin group with LiNR V1 and V2 (exact filter) at the same k and batch
(1 / 16 / 64), the harness's pool, windows, clocks and ``unstable`` flag. V3 is not timed (head-to-head V3 was
k 100 only and its pools on these filters were not picked).

    cd evaluation && python ../research/artifacts/gpu-followups/run_d.py [goodreads] [pubmed]
"""

from __future__ import annotations

import copy
import functools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import torch

from bench import algos, measure, metrics, oracle
from bench import inputs as binputs
from bench.config import load_dataset
from bench.run import _rotate, prepare_pool
from retrieve import OfficialConfig, SilverTorch
from retrieve.ops import official as official_mod
from retrieve.ops.official import adapter

KS = [100, 1000, 2048]
TARGETS = [0.90, 0.95]
BATCHES = [1, 16, 64]
CHUNK = 64
FILTERS = {
    "goodreads": ["c1_lang_reverse", "all4"],
    "pubmed": ["c3_journal_reverse", "all5"],
}
CONFIG = {
    "goodreads": ("config/goodreads.yaml", 128),
    "pubmed": ("config/pubmed.yaml", 768),
}
N_LISTS = {"goodreads": [2048, 4096, 8192], "pubmed": [4096, 16384, 65536]}
PROBES = [24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
          12288, 16384]  # fmt: skip
OUT = Path(os.environ.get("GF_D_OUT", "/scratch/gpu-followups/d/out"))


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def append(name: str, rec: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{name}.jsonl").open("a") as f:
        f.write(
            json.dumps(
                {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **rec},
                default=float,
            )
            + "\n"
        )


def set_k(module: torch.nn.Module, k: int) -> None:
    for m in module.modules():
        if isinstance(getattr(m, "k", None), int):
            m.k = k


def at(st: SilverTorch, k: int, n_probe: int) -> SilverTorch | None:
    """A shallow copy of ``st`` (same index tensors) at its own k and n_probe; None where the probed
    clusters cannot hold k candidates (``set_query_params`` refuses)."""
    c = copy.copy(st)
    c.k = k
    try:
        c.set_query_params(n_probe=n_probe)
    except ValueError:
        return None
    return c


def build_st(item_embs, attrs, n_lists: int) -> SilverTorch:
    st = SilverTorch(k=max(KS), n_lists=n_lists, n_probe=24, filter_mode="bloom", seed=0, backend="cuda",
                     official=OfficialConfig(recheck=False), n_iter=10, **algos.BLOOM_DEFAULTS)  # fmt: skip
    st.register_index(item_embs, item_clause_attrs=attrs)
    return st


def not_prepare(st: SilverTorch, rev_cpu: torch.Tensor, qa: torch.Tensor):
    official_mod.queries_to_expressions = functools.partial(
        adapter.queries_to_expressions, clause_is_reverse=rev_cpu
    )
    try:
        return st.prepare_queries(qa)
    finally:
        official_mod.queries_to_expressions = adapter.queries_to_expressions


@torch.inference_mode()
def main(names: list[str]) -> None:
    warnings.filterwarnings("ignore")
    device = torch.device("cuda")
    measure.setup(0)
    measure.warm_gpu_once()
    prov = measure.provenance()
    for name in names:
        path, dim = CONFIG[name]
        ds = load_dataset(Path(path), dim)
        inp = binputs.load_inputs(ds, device)
        attrs, rev = inp["item_attrs"], inp["clause_is_reverse"]
        rev_cpu = rev.cpu()
        filt = algos.build_filter("clause", attrs, clause_is_reverse=rev)
        sweeps = {s: tuple(ds.clauses["clause"][s]) for s in FILTERS[name]}
        qas, gts = {}, {}
        for s, clauses in sweeps.items():
            qa_s, skip = binputs.sweep_qa(inp["qa"], clauses)
            blob = oracle.compute(
                inp["item_embs"],
                inp["queries"],
                qa_s,
                skip,
                filt,
                max(KS),
                device=device,
            )
            rows = (~skip).nonzero()[:, 0]
            qas[s] = (qa_s, skip, rows)
            gts[s] = blob["topk"][rows]
            log(name, s, "kept", rows.numel(), "pass rate", blob["pass_rate"])
        picks: dict[tuple, set] = {}
        for n_lists in N_LISTS[name]:
            st = build_st(inp["item_embs"], attrs, n_lists)
            for s in sweeps:
                qa_s, _, rows = qas[s]
                q_all, qa_all = inp["queries"][rows], qa_s[rows]
                preps = [not_prepare(st, rev_cpu, qa_all[i:i + CHUNK].to(device))
                         for i in range(0, rows.numel(), CHUNK)]  # fmt: skip
                for npb in [p for p in PROBES if p <= n_lists // 4]:
                    recall = {}
                    for k in KS:
                        c = at(st, k, npb)
                        if c is None:
                            continue
                        acc = metrics.accumulator([k], device)
                        for j, i in enumerate(range(0, rows.numel(), CHUNK)):
                            ids, _ = c(q_all[i : i + CHUNK].to(device), preps[j])
                            metrics.accumulate(
                                acc, ids, gts[s][i : i + CHUNK, :k].to(device)
                            )
                        recall[k] = metrics.finalize(acc)[f"recall@{k}"]
                    append("curve", {"dataset": name, "dim": dim, "sweep": s, "n_lists": n_lists, "n_probe": npb,
                                     "n_queries": rows.numel(), "recall": recall, "env": prov})  # fmt: skip
                    log(
                        name,
                        s,
                        n_lists,
                        npb,
                        {k: round(v, 4) for k, v in recall.items()},
                    )
                    for k in recall:
                        for t in TARGETS:
                            if recall[k] >= t and not any(p[0] == n_lists and p[1] <= npb
                                                          for p in picks.get((s, k, t), ())):  # fmt: skip
                                picks.setdefault((s, k, t), set()).add((n_lists, npb))
                    if len(recall) == len(KS) and all(
                        v >= max(TARGETS) for v in recall.values()
                    ):
                        break
            del st
            torch.cuda.empty_cache()
        append(
            "picks",
            {
                "dataset": name,
                "picks": [[s, k, t, sorted(v)] for (s, k, t), v in picks.items()],
            },
        )
        log(name, "picks", picks)

        items_cpu = inp.pop("item_embs").cpu()
        torch.cuda.empty_cache()
        emb = items_cpu.to(device)
        linr = {a: algos.build(a, emb, k=max(KS), backend="triton", filter_kind="clause", filter_mod=filt)
                for a in ("linr_v1_filter_mask", "linr_v2")}  # fmt: skip
        del emb
        torch.cuda.empty_cache()
        for n_lists in N_LISTS[name]:
            todo = {(s, k): sorted({p[1] for t in TARGETS for p in picks.get((s, k, t), ()) if p[0] == n_lists})
                    for s in sweeps for k in KS}  # fmt: skip
            if not any(todo.values()):
                continue
            emb = items_cpu.to(device)
            st = build_st(emb, attrs, n_lists)
            del emb
            torch.cuda.empty_cache()
            for (s, k), probes in todo.items():
                if not probes:
                    continue
                qa_s, skip, _ = qas[s]
                for bs in BATCHES:
                    pool, qa_pool = binputs.query_pool(
                        inp, qa_s, skip, bs=bs, seed=0, device=device
                    )
                    arms, preps = {}, {}
                    for a, m in linr.items():
                        set_k(m, k)
                        arms[a] = m
                        preps[a] = prepare_pool(m, qa_pool)[0]
                    st_prep = [
                        not_prepare(st, rev_cpu, qa_pool[i])
                        for i in range(qa_pool.shape[0])
                    ]
                    torch._dynamo.reset()
                    graphs, rec_perf = {}, {}
                    for a, m in arms.items():
                        graphs[a] = measure.graph_callable(m, pool[0], preps[a][0])
                    for npb in probes:
                        mod = at(st, k, npb)
                        if mod is None:
                            continue
                        try:
                            graphs[f"st/np{npb}"] = measure.graph_callable(
                                mod, pool[0], st_prep[0]
                            )
                            preps[f"st/np{npb}"] = st_prep
                        except measure.NotCapturable as exc:
                            rec_perf[f"st/np{npb}"] = {"reason": str(exc)}
                    fns = [_rotate(c, pool, preps[a]) for a, c in graphs.items()]
                    for a, (d, _) in zip(
                        graphs,
                        measure.latency_group(fns, bs=bs, mode="graph"),
                        strict=True,
                    ):
                        rec_perf[a] = d
                    torch._dynamo.reset()
                    best = min(rec_perf[a]["median_ms"] for a in linr)
                    rec = {"dataset": name, "dim": dim, "sweep": s, "n_lists": n_lists, "k": k, "bs": bs,
                           "perf": rec_perf, "best_linr_ms": best, "env": prov}  # fmt: skip
                    append("timed", rec)
                    log(
                        name,
                        s,
                        n_lists,
                        "k",
                        k,
                        "bs",
                        bs,
                        {
                            a: round(p.get("median_ms", float("nan")), 3)
                            for a, p in rec_perf.items()
                        },
                    )
                    del pool, qa_pool, preps, graphs, fns, st_prep
            del st
            torch.cuda.empty_cache()
        del linr, inp, filt, items_cpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["goodreads", "pubmed"])
