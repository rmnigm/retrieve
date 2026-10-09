"""V1-FUSE GPU checks, one process per dataset (run from evaluation/, the v1-fuse venv, GPU 0).

  python v1fuse_gpu.py gate DATASET RECORDS OUT.json
      Per (filter kind, sweep, bs): the new LiNRV1 (filter folded into the score write) against
      (a) the campaign-v2.1 records' ``ids_sha256`` / ``ids_sha256_canon`` (same seeded pool,
      seed 0), eager and graph, k 100 and 1000; (b) the old composition (cuBLAS scores, the
      filter's [B, N] bool, ``masked_topk(valid=)``) in this process, ``torch.equal`` ids and
      scores on the probe batches; (c) one interleaved timing group at k 100: new eager, old
      eager, new graph, old graph (``measure.latency_group``), so new/old and graph/eager are
      paired ratios with a 95 % CI over windows. Graph callables come from
      ``measure.graph_callable`` (0 cudagraph skips, one cudaGraphLaunch per call, or it raises).

  python v1fuse_gpu.py fused DATASET OUT.json
      Item 1 as an experiment, not shipped: a Triton fp32-accumulate mat-vec at bs 1, its
      latency against cuBLAS (``torch.mm(out_dtype=fp32)``), and how many of 256 pool queries
      per cell return other ids than V1 once the same predicate and topk follow.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from torch import nn

from bench import algos, inputs, measure
from bench.config import load_dataset
from bench.run import IDS_PROBE_BATCHES, ids_sha256
from retrieve.functional import masked_topk

SWEEPS = ("p0001", "p001", "p01", "p1")
KINDS = ("clause", "bloom")
BSS = (1, 16)
DEV = torch.device("cuda")
LAT = {"warmup": 20, "windows": 7, "target_s": 0.4, "n_min": 200, "n_max": 3000}


class OldV1(nn.Module):
    """The campaign-v2.1 V1 op sequence: cuBLAS scores, the filter's [B, N] bool, masked_topk."""

    capturable = True

    def __init__(self, new: nn.Module) -> None:
        super().__init__()
        self.idx, self.filter = new.idx, new.filter

    @property
    def k(self) -> int:
        return self.idx.k

    @k.setter
    def k(self, k: int) -> None:
        self.idx.k = int(k)

    def forward(self, q, qa):
        return masked_topk(self.idx.score(q), self.idx.k, valid=self.filter.evaluate_mask(qa))


def _records(path: str, dataset: str) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r["algo"] != "linr_v1_filter_mask" or r["backend"] != "triton" or r["seed"] != 0:
            continue
        for e in r.get("perf") or []:
            if e.get("ids_sha256"):
                key = (r["filter_kind"], r["sweep"], e["bs"], e["k"], e["mode"])
                out[key] = (e["ids_sha256"], e["ids_sha256_canon"])
    assert out, f"no V1 triton seed-0 perf entries for {dataset} in {path}"
    return out


def _ks(sweep: str) -> tuple[int, ...]:
    return (100,) if sweep == "p0001" else (100, 1000)  # suites.yaml ks_by_sweep, both datasets


@torch.inference_mode()
def _probe_equal(a, b, pool, qa_pool) -> bool:
    for i in range(IDS_PROBE_BATCHES):
        ia, sa = a(pool[i], qa_pool[i])
        ib, sb = b(pool[i], qa_pool[i])
        if not (torch.equal(ia, ib) and torch.equal(sa, sb)):
            return False
    return True


def _rotate(fn, pool, qa_pool):
    it = itertools.count()

    def call():
        i = next(it) % pool.shape[0]
        return fn(pool[i], qa_pool[i])

    return call


def _paired(num: list[float], den: list[float]) -> dict:
    """Median of per-round ratios and a 95 % t-interval on their logs (rounds are interleaved)."""
    logs = [math.log(a / b) for a, b in zip(num, den, strict=True)]
    m, n = statistics.mean(logs), len(logs)
    half = 2.447 * statistics.stdev(logs) / math.sqrt(n) if n > 1 else float("nan")  # t(6), 7 rounds
    return {
        "ratio": statistics.median(a / b for a, b in zip(num, den, strict=True)),
        "lo": math.exp(m - half),
        "hi": math.exp(m + half),
    }


def gate(dataset: str, records: str, out_path: str) -> None:
    rec = _records(records, dataset)
    ds = load_dataset(Path("config") / f"{dataset}.yaml", 128)
    inp = inputs.load_inputs(ds, DEV)
    rows = []
    for fk in KINDS:
        filters = inputs.build_filters(fk, inp, ["triton"], bloom=algos.BLOOM_DEFAULTS)
        new = algos.build(
            "linr_v1_filter_mask", inp["item_embs"], k=100, backend="triton",
            filter_kind=fk, filter_mod=filters["triton"],
        )  # fmt: skip
        old = OldV1(new)
        for sw in SWEEPS:
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk][sw])
            for bs in BSS:
                pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=bs, seed=0, device=DEV)
                for k in _ks(sw):
                    new.k = k
                    row = {"filter_kind": fk, "sweep": sw, "bs": bs, "k": k}
                    row["eager_vs_old_equal"] = _probe_equal(new, old, pool, qa_pool)
                    torch._dynamo.reset()
                    g_new = measure.graph_callable(new, pool[0], qa_pool[0])
                    timed_k = k == 100 and not os.environ.get("V1FUSE_NO_TIMING")
                    g_old = measure.graph_callable(old, pool[0], qa_pool[0]) if timed_k else None
                    for mode, fn in (("eager", new), ("graph", g_new)):
                        got = ids_sha256(fn, pool, qa_pool)
                        want = rec.get((fk, sw, bs, k, mode))
                        row[f"{mode}_ids_equal_v21"] = None if want is None else got[0] == want[0]
                        row[f"{mode}_canon_equal_v21"] = None if want is None else got[1] == want[1]
                    row["graph_vs_old_equal"] = _probe_equal(g_new, old, pool, qa_pool)
                    if g_old is not None:
                        arms = [new, old, g_new, g_old]
                        with torch.inference_mode():
                            timed = measure.latency_group(
                                [_rotate(a, pool, qa_pool) for a in arms], bs=bs, mode="eager", **LAT
                            )
                        wm = [d["window_medians_ms"] for d, _ in timed]
                        row["p50_ms"] = dict(
                            zip(("new_eager", "old_eager", "new_graph", "old_graph"),
                                [d["median_ms"] for d, _ in timed], strict=True)
                        )  # fmt: skip
                        row["sm_mhz"] = [d.get("sm_mhz") for d, _ in timed]
                        row["new_over_old_eager"] = _paired(wm[0], wm[1])
                        row["new_over_old_graph"] = _paired(wm[2], wm[3])
                        row["graph_over_eager_new"] = _paired(wm[2], wm[0])
                        row["graph_over_eager_old"] = _paired(wm[3], wm[1])
                    print(json.dumps(row), flush=True)
                    rows.append(row)
                    del g_new, g_old
                    torch._dynamo.reset()
    env = measure.provenance() | measure.clocks()
    Path(out_path).write_text(json.dumps({"dataset": dataset, "env": env, "rows": rows}, indent=1))


@triton.jit
def _matvec_kernel(q_ptr, x_ptr, out_ptr, N, D: tl.constexpr, BLOCK_D: tl.constexpr,
                   BLOCK_N: tl.constexpr):  # fmt: skip
    n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + d).to(tl.float32)
        x = tl.load(x_ptr + d[:, None] * N + n[None, :], mask=n[None, :] < N, other=0.0)
        acc += tl.sum(q[:, None] * x.to(tl.float32), axis=0)
    tl.store(out_ptr + n, acc, mask=n < N)


def _matvec(q, x_t, block_n=256):
    d, n = x_t.shape
    out = torch.empty(1, n, dtype=torch.float32, device=q.device)
    _matvec_kernel[(triton.cdiv(n, block_n),)](q.to(torch.float16).reshape(-1), x_t, out, n,
                                               D=d, BLOCK_D=32, BLOCK_N=block_n)  # fmt: skip
    return out


def fused(dataset: str, out_path: str) -> None:
    ds = load_dataset(Path("config") / f"{dataset}.yaml", 128)
    inp = inputs.load_inputs(ds, DEV)
    rows = []
    for fk in KINDS:
        filters = inputs.build_filters(fk, inp, ["triton"], bloom=algos.BLOOM_DEFAULTS)
        v1 = algos.build(
            "linr_v1_filter_mask", inp["item_embs"], k=100, backend="triton",
            filter_kind=fk, filter_mod=filters["triton"],
        )  # fmt: skip
        x_t = v1.idx.item_embs_t

        def exp(q, qa, k):
            return masked_topk(v1.filter.mask_scores(_matvec(q, x_t), qa), k, masked=True)

        for sw in SWEEPS:
            qa_s, skip = inputs.sweep_qa(inp["qa"], ds.clauses[fk][sw])
            pool, qa_pool = inputs.query_pool(inp, qa_s, skip, bs=1, seed=0, device=DEV, n_pool=256)
            row = {"filter_kind": fk, "sweep": sw}
            with torch.inference_mode():
                for k in _ks(sw):
                    v1.k = k
                    diff = 0
                    for i in range(pool.shape[0]):
                        a = v1(pool[i], qa_pool[i])[0]
                        b = exp(pool[i], qa_pool[i], k)[0]
                        diff += int(not torch.equal(a, b))
                    row[f"k{k}_queries_with_other_ids_of_256"] = diff
                timed = measure.latency_group(
                    [_rotate(lambda q, qa: v1.idx.score(q), pool, qa_pool),
                     _rotate(lambda q, qa: _matvec(q, x_t), pool, qa_pool)],
                    bs=1, mode="eager", **LAT,
                )  # fmt: skip
            row["cublas_ms"], row["triton_matvec_ms"] = (d["median_ms"] for d, _ in timed)
            row["matvec_over_cublas"] = _paired(
                timed[1][0]["window_medians_ms"], timed[0][0]["window_medians_ms"]
            )
            print(json.dumps(row), flush=True)
            rows.append(row)
    Path(out_path).write_text(json.dumps({"dataset": dataset, "rows": rows}, indent=1))


if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    {"gate": gate, "fused": fused}[cmd](*args)
