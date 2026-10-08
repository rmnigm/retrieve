"""G-oracle: the item-chunked oracle (``bench.oracle.compute``) against the staged blob v4 of
the same fingerprint, on the GPU. Per sweep: ``topk``, ``pass_counts`` and
``targets_in_filter`` compared; every ``topk`` row that differs is classified by recomputing
its query batch's fp32 scores both ways (the one-shot ``[64, N]`` matmul the staged blob
used, and the new chunked one): a *tie flip* has the same score at every position of the two
lists under both, anything else is a red gate. ``SWEEP:bloom`` also compares the chunked bloom
pass counts with the one-shot ``evaluate_mask``. Peak GPU memory of the chunked build is
reported above the resident inputs. Writes nothing to the staged ``gt_dir``.

    cd evaluation && python ../docs/artifacts/campaign-v2/harness-core/oracle_gate.py \\
        arxiv 128 /scratch/cv2-harness-core/oracle-arxiv.json c0_maincat all4
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from bench import algos, config, inputs, measure, oracle

KS = (1000, 400, 200, 100)


def staged(inp, ds, sweep, clauses, qa_s):
    for k in KS:
        fp = oracle.fingerprint(
            inp["item_embs"],
            inp["queries"],
            inp["targets"],
            qa_s,
            clauses,
            k,
            attrs_digest=inp["attrs_digest"],
        )
        path = oracle.blob_path(ds.gt_dir, sweep, fp)
        if path.exists():
            return k, path
    raise SystemExit(f"no staged blob for {sweep} at k in {KS}")


def batch_scores(inp, et, exact, qa_s, rows, chunk):
    """The fp32 scores of ``rows`` (one oracle batch) the one-shot way (``et``: the staged
    blob's ``item_embs.t().contiguous()``) and the chunked way."""
    q = inp["queries"][rows].cuda()
    qa = qa_s[rows].cuda()
    full = (q @ et).masked_fill(~exact.evaluate_mask(qa), -torch.inf)
    parts = []
    for s in range(0, inp["n_items"], chunk):
        e = min(s + chunk, inp["n_items"])
        parts.append(
            (q @ inp["item_embs"][s:e].t()).masked_fill(~exact.evaluate_mask(qa, s, e), -torch.inf)
        )
    return full, torch.cat(parts, dim=1)


def classify(old, new, inp, exact, qa_s, skip):
    keep_idx = (~skip).nonzero().reshape(-1)
    pos = {int(u): i for i, u in enumerate(keep_idx)}
    rows = (old != new).any(dim=1).nonzero().reshape(-1).tolist()
    out = {
        "rows": len(rows),
        "positions": int((old != new).sum()),
        "tie": 0,
        "one_way": 0,
        "other": 0,
        "examples": [],
    }
    by_batch: dict[int, list[int]] = {}
    for u in rows:
        by_batch.setdefault(pos[u] // 64, []).append(u)
    et = inp["item_embs"].t().contiguous() if by_batch else None
    for b, us in by_batch.items():
        batch = keep_idx[b * 64 : (b + 1) * 64]
        full, chunked = batch_scores(inp, et, exact, qa_s, batch, oracle.ITEM_CHUNK)
        for u in us:
            i = (batch == u).nonzero().item()
            o, n = old[u], new[u]
            same = []
            for sc in (full[i].cpu(), chunked[i].cpu()):
                so = torch.where(o >= 0, sc[o.clamp(min=0)], -torch.inf)
                sn = torch.where(n >= 0, sc[n.clamp(min=0)], -torch.inf)
                same.append(bool(torch.equal(so, sn)))
            kind = "tie" if all(same) else "one_way" if any(same) else "other"
            out[kind] += 1
            if kind != "tie" and len(out["examples"]) < 5:
                d = (o != n).nonzero().reshape(-1)[:4].tolist()
                out["examples"].append(
                    {"row": u, "pos": d, "old": o[d].tolist(), "new": n[d].tolist()}
                )
    del et
    torch.cuda.empty_cache()
    return out


def main() -> None:
    name, dim, out_path, specs = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
    measure.setup(0)
    ds = config.load_dataset(Path(f"config/{name}.yaml"), dim)
    inp = inputs.load_inputs(ds, torch.device("cuda"), with_filters=True)
    exact = algos.build_filter(
        "clause", inp["item_attrs"], clause_is_reverse=inp["clause_is_reverse"], backend="triton"
    )
    res = {
        "dataset": name,
        "dim": dim,
        "n_items": inp["n_items"],
        "item_chunk": oracle.ITEM_CHUNK,
        "env": measure.provenance() | measure.clocks(),
        "sweeps": [],
    }
    for spec in specs:
        sweep, kind = spec.split(":") if ":" in spec else (spec, "clause")
        clauses = ds.clauses[kind][sweep]
        qa_s, skip = inputs.sweep_qa(inp["qa"], clauses)
        k, path = staged(inp, ds, sweep, clauses, qa_s)
        old = torch.load(path, map_location="cpu", weights_only=True)
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        new = oracle.compute(
            inp["item_embs"],
            inp["queries"],
            qa_s,
            skip,
            exact,
            k,
            targets=inp["targets"],
            device=torch.device("cuda"),
        )
        secs = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated() - base
        row = {
            "sweep": sweep,
            "kind": kind,
            "k_gt": k,
            "staged": path.name,
            "seconds": secs,
            "peak_mib_above_inputs": peak / 2**20,
            "bn_fp32_mib": 64 * inp["n_items"] * 4 / 2**20,
            "pass_counts_equal": torch.equal(new["pass_counts"], old["pass_counts"]),
            "targets_in_filter_equal": torch.equal(
                new["targets_in_filter"], old["targets_in_filter"]
            ),
            "topk_equal": torch.equal(new["topk"], old["topk"]),
        }
        if not row["topk_equal"]:
            row["topk_diff"] = classify(old["topk"], new["topk"], inp, exact, qa_s, skip)
        if kind == "bloom":
            bloom = algos.build_filter(
                "bloom", inp["item_attrs"], backend="triton", **algos.BLOOM_DEFAULTS
            )
            chunked = oracle.pass_counts(
                bloom, qa_s, skip, inp["n_items"], device=torch.device("cuda")
            )
            one = torch.full_like(chunked, -1)
            for s in range(0, qa_s.shape[0], 64):
                idx = torch.arange(s, min(s + 64, qa_s.shape[0]))
                idx = idx[~skip[idx]]
                if idx.numel():
                    one[idx] = bloom.evaluate_mask(qa_s[idx].cuda()).sum(dim=1).cpu()
            row["bloom_pass_counts_equal"] = torch.equal(chunked, one)
        print(json.dumps(row), flush=True)
        res["sweeps"].append(row)
    Path(out_path).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
