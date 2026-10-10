"""Parity at 30 M: official int32 vs our triton on laion30m d256 bloom c0_domain, n_probe 128 (n_lists 16384, k 100), bloom_path partial + full.
One 30 M module per process (a process that builds several runs out of memory):
    run KEY OUT.pt          build `KEY` in {triton-partial, official-partial, triton-full, official-full}; ids + scores of every kept query (bs 16)
    diff DIR OUT.json       CPU: differing queries per path pair; (query row, item, side) for every item one side returns and the other does not
                            at bloom_path partial (exhibits' rows 2570 / 4406 / 4606 / 4845 / 5845 always listed)
    items KEY DIFF OUT.json per pair, from KEY's module: triton-partial -> probed (our phase 1), our bloom (build_signatures, LSB-first words), exact
                            filter, exact int8 rescoring, in the filtered oracle top-100, pass count; official-partial -> Meta's probes equal ours,
                            Meta's bloom (bloom_index_search_batch bool mask)
    report DIR OUT.json     CPU: merge and count categories
All from evaluation/ of the campaign-v2.10 tree with CONFIG_DIR in $PARITY_CFG (suite `parity30m`)."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

TARGETS = [2570, 4406, 4606, 4845, 5845]
KEYS = {"triton-partial": ("triton", "partial"), "official-partial": ("official", "partial"),
        "triton-full": ("triton", "full"), "official-full": ("official", "full")}  # fmt: skip


def _setup(key):
    from bench import config, inputs, run

    dev = torch.device("cuda")
    cfg = Path(os.environ["PARITY_CFG"])
    jobs = {
        (j.backend, j.build.get("bloom_path")): j
        for j in config.load_matrix(
            cfg / "laion30m.yaml", cfg / "suites.yaml", "parity30m"
        )
    }
    j = jobs[KEYS[key]]
    inp = inputs.load_inputs(j.data, dev)
    assets = run.sweep_assets(j, inp, 100, dev)
    m = run.build_module(j, inp, assets, 100, {**j.build, **j.query[0]})
    return dev, j, inp, assets, m


def cmd_run(key, out):
    dev, j, inp, assets, m = _setup(key)
    rows = assets["keep"].nonzero().reshape(-1)
    ids, sc = [], []
    with torch.inference_mode():
        for s in range(0, rows.numel(), 16):
            r = rows[s : s + 16]
            i, c = m(
                inp["queries"][r].to(dev), m.prepare_queries(assets["qa_s"][r].to(dev))
            )
            ids.append(i.cpu())
            sc.append(c.float().cpu())
    torch.save({"rows": rows.cpu(), "ids": torch.cat(ids), "scores": torch.cat(sc), "centroids_sum": float(m.centroids.double().sum()),
                "sort_perm_sum": int(m.sort_perm.sum()), "pass_count": assets["blob"]["pass_counts"].cpu() if "pass_counts" in assets["blob"] else None}, out)  # fmt: skip


def _sets(d, i):
    x = d["ids"][i]
    return set(x[x >= 0].tolist())


def cmd_diff(dirp, out):
    d = {k: torch.load(Path(dirp) / f"{k}.pt") for k in KEYS}
    rows = d["triton-partial"]["rows"].tolist()
    res = {
        "n_queries": len(rows),
        "same_index_sums": {
            k: [v["centroids_sum"], v["sort_perm_sum"]] for k, v in d.items()
        },
    }
    for a, b in (
        ("triton-partial", "official-partial"),
        ("triton-full", "official-full"),
        ("triton-partial", "triton-full"),
        ("official-partial", "official-full"),
    ):
        res[f"{a} vs {b}"] = [
            rows[i] for i in range(len(rows)) if _sets(d[a], i) != _sets(d[b], i)
        ]
    focus = sorted(
        set(res["triton-partial vs official-partial"])
        | {t for t in TARGETS if t in rows}
    )
    pairs = []
    for row in focus:
        i = rows.index(row)
        t, o = _sets(d["triton-partial"], i), _sets(d["official-partial"], i)
        tf, of = _sets(d["triton-full"], i), _sets(d["official-full"], i)
        for side, mine, other, src in (
            ("ours_only", t, o, "triton-partial"),
            ("meta_only", o, t, "official-partial"),
        ):
            for item in sorted(mine - other):
                ids = d[src]["ids"][i].tolist()
                pairs.append({"row": row, "item": item, "side": side, "reported": float(d[src]["scores"][i][ids.index(item)]),
                              "in_triton_full": item in tf, "in_official_full": item in of})  # fmt: skip
    res["pairs"] = pairs
    Path(out).write_text(json.dumps(res, indent=1))
    print({k: (len(v) if isinstance(v, list) else v) for k, v in res.items()})


def cmd_items(key, diffp, out):
    from bench import inputs
    from retrieve.indexing.bloom_hash import build_signatures
    from retrieve.ops.official import adapter as official_adapter

    dev, j, inp, assets, m = _setup(key)
    pairs = json.loads(Path(diffp).read_text())["pairs"]
    perm = m.sort_perm
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=dev)
    oracle = assets["blob"]["topk"]
    if key == "triton-partial":
        filters = inputs.build_filters("bloom", inp, ["triton"], bloom=j.bloom)
        exact = inputs.exact_filter("bloom", filters, inp, "triton")
        attrs = inp["item_attrs"].long()
    res = []
    for row in sorted({p["row"] for p in pairs}):
        r = torch.tensor([row], device=dev)
        q = inp["queries"][r].to(dev)
        qa = assets["qa_s"][r].to(dev)
        probes = m._phase1_probe_ids(q)[0]
        offs = m.cluster_offsets

        def cluster_of(item, offs=offs):
            return int(torch.searchsorted(offs, inv[item], right=True)) - 1

        if key == "triton-partial":
            qbits = m._query_bit_positions(qa)[0]
            qbits = qbits[qbits >= 0].tolist()
            ex = exact.evaluate_mask(qa)[0]
            orc = set(oracle[row, :100][oracle[row, :100] >= 0].tolist())
        else:
            plans = m.prepare_queries(qa)
            mb = official_adapter.bloom_full_mask(m.bloom_index, m.bundle_b_offsets, (plans.plans_data, plans.plans_offsets), m.k_hash,
                                                  m.official.n_stored_hashes, return_bool_mask=True)[0]  # fmt: skip
        for p in (p for p in pairs if p["row"] == row):
            it = p["item"]
            c = cluster_of(it)
            rec = {
                "row": row,
                "item": it,
                "cluster": c,
                "probed": c in set(probes.tolist()),
            }
            if key == "triton-partial":
                sig = build_signatures(
                    attrs[it : it + 1],
                    m.hash_seeds,
                    m.m_bits,
                    m.k_hash,
                    m.word_count,
                    clause_salt=m.clause_salt,
                )[0]
                rec.update({"ours_bloom": all(((int(sig[b // 64]) >> (b % 64)) & 1) for b in qbits), "exact": bool(ex[it]),
                            "in_oracle": it in orc, "pass_count": int(ex.sum()),
                            "int8_rescore": float(m._forward_candidates(q, torch.tensor([[it]], device=dev))[1][0, 0])})  # fmt: skip
            else:
                rec.update(
                    {"meta_bloom": bool(mb[int(inv[it])]), "probe_ids": probes.tolist()}
                )
            res.append(rec)
    Path(out).write_text(json.dumps(res))


def cmd_report(dirp, out):
    d = Path(dirp)
    diff = json.loads((d / "diff.json").read_text())
    t = {
        (r["row"], r["item"]): r
        for r in json.loads((d / "items-triton-partial.json").read_text())
    }
    o = {
        (r["row"], r["item"]): r
        for r in json.loads((d / "items-official-partial.json").read_text())
    }
    cats, rows = Counter(), []
    for p in diff["pairs"]:
        a, b = t[p["row"], p["item"]], o[p["row"], p["item"]]
        c = (p["side"], "probed-ours" if a["probed"] else "unprobed-ours", "probed-meta" if b["probed"] else "unprobed-meta",
             "exact" if a["exact"] else "not-exact", "ours-bloom" if a["ours_bloom"] else "not-ours-bloom",
             "meta-bloom" if b["meta_bloom"] else "not-meta-bloom", "in-oracle" if a["in_oracle"] else "not-oracle",
             "in-official-full" if p["in_official_full"] else "not-official-full", "in-triton-full" if p["in_triton_full"] else "not-triton-full")  # fmt: skip
        cats[c] += 1
        rows.append({**p, **{k: a[k] for k in ("cluster", "probed", "ours_bloom", "exact", "in_oracle", "pass_count", "int8_rescore")},
                     "meta_probed": b["probed"], "meta_bloom": b["meta_bloom"]})  # fmt: skip
    rep = {k: v for k, v in diff.items() if k != "pairs"}
    rep["differing_counts"] = {k: len(v) for k, v in diff.items() if " vs " in k}
    rep["categories"] = {" / ".join(k): v for k, v in cats.most_common()}
    rep["pairs"] = rows
    Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs"}, indent=1))


if __name__ == "__main__":
    c, *a = sys.argv[1:]
    {"run": cmd_run, "diff": cmd_diff, "items": cmd_items, "report": cmd_report}[c](*a)
