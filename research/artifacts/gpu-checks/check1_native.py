"""Check 1: LiNR's native-PyTorch baseline as LiNR §6 states it, against our Triton V1 / V2.

Native arms (eager, session code, not the library):
  native_index  boolean clause mask in broadcast compares, then per query (LiNR §5.3 splits the
                batch per query) ``mask.nonzero()`` (host sync, data-dependent shape),
                ``items[idx]`` gather, fp16 matmul on the gathered rows, ``topk``, map back.
  native_mask   the same boolean mask, a dense fp16 matmul, failing scores set to -inf
                (LiNR V1's similarity masking in native ops), ``topk``.
Custom arms: the library's LiNRV1 / LiNRV2 on the triton backend, graph replay and eager, timed
with the harness's windows. One synthetic clause per cell (draw 0, ``u_i < p``).

    cd evaluation && python ../research/artifacts/gpu-checks/check1_native.py [datasets...]
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import Records, env, load, log, synth_attrs  # noqa: E402

from bench import algos, measure  # noqa: E402
from bench import inputs as binputs  # noqa: E402
from bench.run import _rotate, prepare_pool  # noqa: E402

RATES = [
    0.001,
    0.01,
    0.1,
    0.11,
    0.5,
]  # 0.11: LiNR's high-pass-rate set, 1.7 M of 15.5 M
BATCHES = [1, 16]
K = 100
NATIVE_LATENCY = {"warmup": 3, "n_min": 10, "n_max": 200, "target_s": 1.0}
PROBE = 4  # batches compared for agreement


def native_mask(
    attrs: torch.Tensor, reverse: list[bool], qa: torch.Tensor
) -> torch.Tensor:
    m = torch.ones(qa.shape[0], attrs.shape[0], dtype=torch.bool, device=attrs.device)
    for c, rev in enumerate(reverse):
        hit = (attrs[None, :, c, :] == qa[:, c, None, None]).any(-1)
        if rev:
            hit = ~hit
        m &= hit | (qa[:, c] == -1)[:, None]
    return m


def make_native(items: torch.Tensor, attrs: torch.Tensor, reverse: list[bool], k: int):
    def masked(q, qa):
        mask = native_mask(attrs, reverse, qa)
        s = (q.to(items.dtype) @ items.T).float().masked_fill(~mask, float("-inf"))
        v, i = s.topk(k, dim=1)
        return torch.where(v == float("-inf"), -1, i), v

    def index(q, qa):
        mask = native_mask(attrs, reverse, qa)
        q = q.to(items.dtype)
        ids = torch.full((q.shape[0], k), -1, dtype=torch.int64, device=q.device)
        scores = torch.full((q.shape[0], k), float("-inf"), device=q.device)
        for b in range(q.shape[0]):
            idx = mask[b].nonzero().squeeze(1)
            s = (items[idx] @ q[b]).float()
            kk = min(k, idx.numel())
            v, j = s.topk(kk)
            ids[b, :kk] = idx[j]
            scores[b, :kk] = v
        return ids, scores

    return {"native_mask": masked, "native_index": index}


def overlap(a: tuple, b: tuple) -> float:
    ia, ib = a[0].cpu(), b[0].cpu()
    tot = hit = 0
    for ra, rb in zip(ia, ib, strict=True):
        sa, sb = set(ra[ra >= 0].tolist()), set(rb[rb >= 0].tolist())
        tot += max(len(sb), 1)
        hit += len(sa & sb)
    return hit / tot


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k != "samples"}


def main(names: list[str]) -> None:
    warnings.filterwarnings("ignore")
    device = torch.device("cuda")
    measure.setup(0)
    measure.warm_gpu_once()
    recs = Records("check1", "native")
    done = recs.done()
    provenance = env()
    for name in names:
        ds, inp = load(name, device)
        n = inp["n_items"]
        flags = synth_attrs(n, RATES, device)
        log(
            name,
            "items",
            n,
            "dim",
            ds.dim,
            "pass counts",
            flags[:, :, 0].sum(0).tolist(),
        )
        # one fp16 table per layout, shared by every rate: the modules' own buffers
        v1 = v2 = None
        for j, rate in enumerate(RATES):
            attrs = flags[:, j : j + 1, :].contiguous()
            reverse = torch.zeros(1, dtype=torch.bool, device=device)
            filt = algos.build_filter("clause", attrs, clause_is_reverse=reverse)
            if v1 is None:
                v1 = algos.build("linr_v1_filter_mask", inp["item_embs"], k=K, backend="triton",
                                 filter_kind="clause", filter_mod=filt)  # fmt: skip
                v2 = algos.build("linr_v2", inp["item_embs"], k=K, backend="triton",
                                 filter_kind="clause", filter_mod=filt)  # fmt: skip
                del inp["item_embs"]
                torch.cuda.empty_cache()
            v1.filter = filt
            v2.filter = filt
            native = make_native(v2.idx.item_embs, attrs, [False], K)
            qa = torch.ones(inp["n_queries"], 1, dtype=torch.int64)
            for bs in BATCHES:
                cell = f"{name}/p{rate}/bs{bs}/k{K}"
                if cell in done:
                    continue
                log("cell", cell)
                pool, qa_pool = binputs.query_pool(
                    inp, qa, None, bs=bs, seed=0, device=device
                )
                arms = {"triton_v1": v1, "triton_v2": v2}
                preps = {a: prepare_pool(m, qa_pool)[0] for a, m in arms.items()}
                rec: dict = {"cell": cell, "dataset": name, "dim": ds.dim, "n_items": n,
                             "rate": rate, "pass_count": int(attrs.sum()), "bs": bs, "k": K,
                             "env": provenance, "perf": {}}  # fmt: skip
                with torch.inference_mode():
                    ref = {
                        a: [m(pool[i], preps[a][i]) for i in range(PROBE)]
                        for a, m in arms.items()
                    }
                    for a, fn in native.items():
                        out = [fn(pool[i], qa_pool[i]) for i in range(PROBE)]
                        rec[f"overlap_{a}_vs_triton_v2"] = (
                            sum(
                                overlap(o, r)
                                for o, r in zip(out, ref["triton_v2"], strict=True)
                            )
                            / PROBE
                        )
                    rec["overlap_triton_v1_vs_triton_v2"] = (
                        sum(
                            overlap(o, r)
                            for o, r in zip(
                                ref["triton_v1"], ref["triton_v2"], strict=True
                            )
                        )
                        / PROBE
                    )
                    torch._dynamo.reset()
                    graphs = {}
                    for a, m in arms.items():
                        try:
                            graphs[a] = measure.graph_callable(m, pool[0], preps[a][0])
                        except measure.NotCapturable as exc:
                            rec["perf"][f"{a}/graph"] = {"reason": str(exc)}
                    fns = [_rotate(c, pool, preps[a]) for a, c in graphs.items()]
                    for a, (d, _) in zip(
                        graphs,
                        measure.latency_group(fns, bs=bs, mode="graph"),
                        strict=True,
                    ):
                        rec["perf"][f"{a}/graph"] = d
                    fns = [_rotate(m, pool, preps[a]) for a, m in arms.items()]
                    for a, (d, _) in zip(
                        arms,
                        measure.latency_group(fns, bs=bs, mode="eager"),
                        strict=True,
                    ):
                        rec["perf"][f"{a}/eager"] = d
                    fns = [_rotate(fn, pool, qa_pool) for fn in native.values()]
                    timed = measure.latency_group(
                        fns, bs=bs, mode="eager", **NATIVE_LATENCY
                    )
                    for a, (d, _) in zip(native, timed, strict=True):
                        rec["perf"][f"{a}/eager"] = d
                    torch._dynamo.reset()
                med = {a: p.get("median_ms") for a, p in rec["perf"].items()}
                custom = min(med["triton_v1/graph"], med["triton_v2/graph"])
                rec["ratio"] = {
                    f"{a}/vs_best_graph": med[f"{a}/eager"] / custom for a in native
                } | {
                    f"{a}/vs_{c}": med[f"{a}/eager"] / med[c]
                    for a in native for c in ("triton_v1/graph", "triton_v2/graph",
                                              "triton_v1/eager", "triton_v2/eager")
                }  # fmt: skip
                rec["unstable_any"] = any(
                    p.get("unstable") for p in rec["perf"].values()
                )
                recs.append(rec)
                log(cell, {a: round(v, 4) for a, v in med.items()},
                    {a: round(v, 1) for a, v in rec["ratio"].items() if "best" in a})  # fmt: skip
                del pool, qa_pool, preps, graphs, fns, ref
        del v1, v2, inp, flags
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["goodreads", "arxiv", "pubmed"])
