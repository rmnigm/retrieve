"""Parity at 30 M (controller 2026-10-12): why official int32 and our triton disagree on laion30m d256 bloom c0_domain at n_probe 128.
Builds, one at a time, SilverTorch triton and official int32 at bloom_path partial and full (n_lists 16384, k 100) through the harness, and
compares their top-100 id sets on the first N kept queries (bs 16). For up to M differing queries it classifies every item one side returns and
the other does not: (a) in a probed cluster (phase 1 is the same function on the same centroids, checked), (b) passes our bloom (recomputed from
build_signatures, LSB-first words; sanity: every item ours returns must pass), Meta's bloom (bloom_index_search_batch, bool mask), the exact filter,
(c) its exact int8 dot vs the score each side reports, (d) whether it is in the exact filtered oracle top-100.
    python diagnose.py CONFIG_DIR OUT.json [N] [M]    (N = all kept queries by default; the five exhibits rows are always analysed)          (from evaluation/ of the campaign-v2.10 tree)"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch

from bench import config, inputs, run
from retrieve.indexing.bloom_hash import build_signatures
from retrieve.ops.official import adapter as official_adapter

dev = torch.device("cuda")
cfg, out = Path(sys.argv[1]), Path(sys.argv[2])
NQ = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
MQ = int(sys.argv[4]) if len(sys.argv) > 4 else 25
jobs = {
    (j.backend, j.build.get("bloom_path")): j
    for j in config.load_matrix(cfg / "laion30m.yaml", cfg / "suites.yaml", "parity30m")
}
inp = inputs.load_inputs(next(iter(jobs.values())).data, dev)
j0 = jobs["triton", "partial"]
assets = run.sweep_assets(j0, inp, 100, dev)
rows = assets["keep"].nonzero().reshape(-1)[:NQ]
TARGETS = [
    2570,
    4406,
    4606,
    4845,
    5845,
]  # exhibits 150000000: the c0_domain rows whose sets differ (pass 88 / 1 / 57,907 / 4,404 / 5,841)
qa_all = assets["qa_s"]
filters = inputs.build_filters("bloom", inp, ["triton"], bloom=j0.bloom)
exact = inputs.exact_filter("bloom", filters, inp, "triton")
oracle = assets["blob"]["topk"]


def run_module(key):
    j = jobs[key]
    m = run.build_module(
        j, inp, run.sweep_assets(j, inp, 100, dev), 100, {**j.build, **j.query[0]}
    )
    ids, sc = [], []
    with torch.inference_mode():
        for s in range(0, rows.numel(), 16):
            r = rows[s : s + 16]
            i, c = m(inp["queries"][r].to(dev), m.prepare_queries(qa_all[r].to(dev)))
            ids.append(i.cpu())
            sc.append(c.float().cpu())
    return m, torch.cat(ids), torch.cat(sc)


res, keep = {}, {}
for key in [
    ("triton", "partial"),
    ("official", "partial"),
    ("triton", "full"),
    ("official", "full"),
]:
    m, ids, sc = run_module(key)
    keep[key] = (ids, sc)
    if key == ("triton", "partial"):
        T = {"centroids": m.centroids.cpu(), "sort_perm": m.sort_perm.cpu(), "offsets": m.cluster_offsets.cpu(), "seeds": m.hash_seeds, "salt": m.clause_salt, "m_bits": m.m_bits, "k_hash": m.k_hash,
             "words": m.word_count, "module": m}  # fmt: skip
        continue
    if key == ("official", "partial"):
        res["same_index"] = {"centroids": bool(torch.equal(T["centroids"], m.centroids.cpu())), "sort_perm": bool(torch.equal(T["sort_perm"], m.sort_perm.cpu())),
                             "item_codes": bool(torch.equal(T["module"].item_codes, m.item_codes))}  # fmt: skip
        om = m
        continue
    del m
    torch.cuda.empty_cache()


def diff_rate(a, b):
    sa, sb = keep[a][0], keep[b][0]
    d = [
        i
        for i in range(sa.shape[0])
        if set(sa[i][sa[i] >= 0].tolist()) != set(sb[i][sb[i] >= 0].tolist())
    ]
    return d


res["n_queries"] = int(rows.numel())
dp = diff_rate(("triton", "partial"), ("official", "partial"))
df = diff_rate(("triton", "full"), ("official", "full"))
res["differing_queries"] = {"partial": len(dp), "full": len(df), "both": len(set(dp) & set(df)),
                            "triton partial vs full": len(diff_rate(("triton", "partial"), ("triton", "full"))),
                            "official partial vs full": len(diff_rate(("official", "partial"), ("official", "full")))}  # fmt: skip
print(json.dumps(res), flush=True)

tm, perm = T["module"], T["sort_perm"].to(dev)
inv = torch.empty_like(perm)
inv[perm] = torch.arange(perm.numel(), device=dev)
attrs_sorted = inp["item_attrs"].long()[perm]
cats = Counter()
examples = []
sane = Counter()
row_list = rows.tolist()
focus = [row_list.index(t) for t in TARGETS if t in row_list]
res["targets_kept"] = [row_list[i] for i in focus]
res["targets_differ"] = {str(row_list[i]): i in dp for i in focus}
for qi in list(dict.fromkeys(focus + dp))[:MQ]:
    r = rows[qi : qi + 1]
    q = inp["queries"][r].to(dev)
    qa = qa_all[r].to(dev)
    probes = tm._phase1_probe_ids(q)[0]
    probes_o = om._phase1_probe_ids(q)[0]
    co, cs = T["offsets"].to(dev), tm.cluster_sizes
    probed_pos = torch.cat(
        [
            torch.arange(int(co[c]), int(co[c]) + int(cs[c]), device=dev)
            for c in probes.tolist()
        ]
    )
    probed = torch.zeros(perm.numel(), dtype=torch.bool, device=dev)
    probed[perm[probed_pos]] = True
    qbits = tm._query_bit_positions(qa)[0]
    qbits = qbits[qbits >= 0]
    ex = exact.evaluate_mask(qa)[0]
    plans = om.prepare_queries(qa)
    mb = official_adapter.bloom_full_mask(om.bloom_index, om.bundle_b_offsets, (plans.plans_data, plans.plans_offsets), om.k_hash,
                                          om.official.n_stored_hashes, return_bool_mask=True)[0]  # fmt: skip
    meta_pass = torch.zeros(perm.numel(), dtype=torch.bool, device=dev)
    meta_pass[perm] = mb[: perm.numel()].bool()
    it, ot = keep["triton", "partial"][0][qi], keep["official", "partial"][0][qi]
    st, so = keep["triton", "partial"][1][qi], keep["official", "partial"][1][qi]
    orc = set(oracle[r][0, :100][oracle[r][0, :100] >= 0].tolist())
    for side, mine, other, scores in (
        ("ours_only", it, ot, st),
        ("meta_only", ot, it, so),
    ):
        extra = sorted(set(mine[mine >= 0].tolist()) - set(other[other >= 0].tolist()))
        for item in extra:
            pos = int(inv[item])
            sig = build_signatures(
                attrs_sorted[pos : pos + 1],
                T["seeds"],
                T["m_bits"],
                T["k_hash"],
                T["words"],
                clause_salt=T["salt"],
            )[0]
            ours_pass = bool(
                all(((int(sig[p // 64]) >> (p % 64)) & 1) for p in qbits.tolist())
            )
            rescore = float(
                tm._forward_candidates(q, torch.tensor([[item]], device=dev))[1][0, 0]
            )
            reported = float(scores[(mine == item).nonzero()[0, 0]])
            c = (side, "probed" if bool(probed[item]) else "unprobed", "exact" if bool(ex[item]) else "not-exact",
                 "ours-bloom" if ours_pass else "not-ours-bloom", "meta-bloom" if bool(meta_pass[item]) else "not-meta-bloom",
                 "in-oracle" if item in orc else "not-oracle")  # fmt: skip
            cats[c] += 1
            if len(examples) < 40:
                examples.append(
                    {
                        "query_row": int(r),
                        "item": item,
                        "cat": c,
                        "reported": reported,
                        "int8_rescore": rescore,
                    }
                )
    # sanity: everything ours returns passes our recomputed bloom
    for item in it[it >= 0].tolist()[:20]:
        pos = int(inv[item])
        sig = build_signatures(
            attrs_sorted[pos : pos + 1],
            T["seeds"],
            T["m_bits"],
            T["k_hash"],
            T["words"],
            clause_salt=T["salt"],
        )[0]
        sane[all(((int(sig[p // 64]) >> (p % 64)) & 1) for p in qbits.tolist())] += 1
    res.setdefault("probe_sets_equal", []).append(bool(torch.equal(probes, probes_o)))
res["sanity_ours_returned_pass_ours_bloom"] = {str(k): v for k, v in sane.items()}
res["item_categories"] = {" / ".join(k): v for k, v in cats.most_common()}
res["examples"] = examples
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "examples"}, indent=1))
