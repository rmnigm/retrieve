"""OFFICIAL-REWORK M1 gate: ids + scores of the SilverTorch forward with the filter prepared outside it (this tree)
against the pre-M1 library (staging 5618bb6, package `retrieve_pre` from campaign-v2/st-ids/make_pkg.sh), on real
data: one dataset per process, every filter mode, both bloom paths, triton and official (int32 and fp16 score paths),
bs {1, 16}, k {100, 1000}. Each pair shares one index: built in both packages from the same seed, then the
new module's state dict loaded into the old one. Scores `torch.equal`; ids equal up to ties
(tests/parity/conftest.py's `assert_topk_equal`).

    cd evaluation && PYTHONPATH=<pkgs>:.:../retrieve/src:../retrieve python m1_gate.py {goodreads|arxiv} out.json
"""

import importlib
import json
import sys
from pathlib import Path

import torch
import yaml
from bench import config, inputs
from tests.parity.conftest import assert_topk_equal

DEV = torch.device("cuda")
SWEEP = {"goodreads": "c0_genre", "arxiv": "c0_maincat"}
PKG = {"before": "retrieve_pre", "after": "retrieve"}
ST = {a: importlib.import_module(f"{p}.modules.silvertorch") for a, p in PKG.items()}
OC = {a: ST[a].OfficialConfig for a in PKG}

ds, out = sys.argv[1], Path(sys.argv[2])
cfg_path = Path(
    f"config/{ds}.yaml"
)  # run from evaluation/: the yaml's data_dir is relative to it
data = config.load_dataset(cfg_path, 128)
inp = inputs.load_inputs(data, DEV, with_filters=True)
clauses = tuple(yaml.safe_load(cfg_path.read_text())["filters"]["clause"][SWEEP[ds]])
qa_s, skip = inputs.sweep_qa(inp["qa"], clauses)
keep = (
    (~skip).nonzero().reshape(-1) if skip is not None else torch.arange(qa_s.shape[0])
)
rows = keep[:48]
queries, qa = inp["queries"][rows].to(DEV), qa_s[rows].to(DEV)
item_embs = inp["item_embs"].to(DEV)
attrs = inp["item_attrs"].to(DEV)
rev = inp["clause_is_reverse"]
rev = None if rev is None else rev.to(DEV)

ARMS = [("triton", "none", None, None), ("triton", "exact", None, None),
        ("triton", "bloom", "partial", None), ("triton", "bloom", "full", None)]  # fmt: skip
ARMS += [("official", m, p, sp) for sp in ("int32", "fp16")
         for m, p in (("none", None), ("exact", None), ("bloom", "partial"), ("bloom", "full"))]  # fmt: skip


def build(arm, backend, mode, path, score_path):
    kw = {
        "k": 1000,
        "n_lists": 1024,
        "n_probe": 32,
        "n_iter": 10,
        "seed": 0,
        "backend": backend,
    }
    if mode != "none":
        kw["filter_mode"] = mode
    if mode == "bloom":
        kw.update(m_bits=1024 if backend != "official" else None, k_hash=5)
    if backend == "official":
        kw["official"] = OC[arm](
            score_path=score_path, **({"bloom_path": path} if path else {})
        )
    elif path == "full":
        kw["bloom_path"] = "full"
    m = ST[arm].SilverTorch(**kw)
    if mode == "none":
        m.register_index(item_embs)
    else:
        m.register_index(
            item_embs,
            item_clause_attrs=attrs,
            clause_is_reverse=rev if mode == "exact" else None,
        )
    return m


rows_out = []
with torch.inference_mode():
    for backend, mode, path, sp in ARMS:
        after = build("after", backend, mode, path, sp)
        before = build("before", backend, mode, path, sp)
        before.load_state_dict(after.state_dict())
        for k in (100, 1000):
            after.k = before.k = k
            for bs in (1, 16):
                ok = True
                for i in range(0, 48 if bs == 16 else 4, bs):
                    q, a = queries[i : i + bs], qa[i : i + bs]
                    want = before(q, None if mode == "none" else a)
                    got = after(q, None if mode == "none" else after.prepare_queries(a))
                    try:
                        assert_topk_equal(*got, *want)
                    except AssertionError:
                        ok = False
                row = {"dataset": ds, "backend": backend, "mode": mode, "bloom_path": path, "score_path": sp,
                       "k": k, "bs": bs, "equal": ok}  # fmt: skip
                rows_out.append(row)
                print(row, flush=True)
        del after, before
        torch.cuda.empty_cache()
out.write_text(
    json.dumps(
        {"dataset": ds, "n_items": int(item_embs.shape[0]), "rows": rows_out}, indent=1
    )
)
print("ALL EQUAL" if all(r["equal"] for r in rows_out) else "MISMATCH", flush=True)
