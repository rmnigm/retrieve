"""C5-OURS bit-exact gates on the SilverTorch layer, one (N, D) per process. Arms: `before` = the tree before the
option (dev/v2-fill 394d3e9, package `retrieve_pre` from ../st-ids/make_pkg.sh; `C5_BEFORE=retrieve_v24` for the
campaign-v2.5 bundle gate), `after` = this tree.

Inputs: Gaussian unit-norm items, four single-value clauses with vocabularies {1000, 100, 7, 1} (pass rates ≈ 0.001,
0.01, 0.14, 1 plus bloom false positives), m_bits 1024, k_hash 5, n_lists 1024, k 100. A query batch activates one clause
per row (16 rows: four of each clause), or one clause for the whole batch at bs 1.

  default  after's default (`partial`) forward torch.equal to before's (ids and scores), triton
  full     after's `full` against after's `partial`: scores torch.equal, ids equal up to ties; triton
  torch    after's triton `full` against the torch backend's `partial` (bit-exact, the int32 path), at N ≤ 200 000

for n_probe {8, 32, 128} × bs {1, 16} × four query seeds.

    PYTHONPATH=<pkgs>:retrieve/src:retrieve python c5_gate.py N D out.json
"""

import argparse
import importlib
import json
import os

import torch
from tests.parity.conftest import assert_topk_equal

BEFORE = os.environ.get(
    "C5_BEFORE", "retrieve_pre"
)  # the bundle gate sets retrieve_v24
ST = {a: importlib.import_module(f"{p}.modules.silvertorch").SilverTorch
      for a, p in (("before", BEFORE), ("after", "retrieve"))}  # fmt: skip
VOCAB = (1000, 100, 7, 1)
DEV = torch.device("cuda")

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("d", type=int)
ap.add_argument("out")
args = ap.parse_args()
g = torch.Generator(device=DEV).manual_seed(0)
embs = torch.randn(args.n, args.d, device=DEV, generator=g)
embs = embs / embs.norm(dim=1, keepdim=True)
attrs = torch.stack(
    [torch.randint(0, v, (args.n,), device=DEV, generator=g) for v in VOCAB], 1
).unsqueeze(2)


def layer(arm, backend="triton", **kw):
    m = ST[arm](k=100, n_lists=1024, n_probe=8, filter_mode="bloom", m_bits=1024, k_hash=5, n_iter=5, seed=0,
                backend=backend, **kw)  # fmt: skip
    m.register_index(embs, attrs)
    return m


arms = {
    "before": layer("before"),
    "partial": layer("after"),
    "full": layer("after", bloom_path="full"),
}
if args.n <= 200_000:
    arms["torch"] = layer("after", backend="torch")
rows = []
for n_probe in (8, 32, 128):
    for m in arms.values():
        m.set_query_params(n_probe=n_probe)
    for bs in (1, 16):
        for qs in range(4):
            gq = torch.Generator(device=DEV).manual_seed(100 + qs)
            q = torch.randn(bs, args.d, device=DEV, generator=gq)
            clause = (
                torch.arange(bs, device=DEV) % 4
                if bs > 1
                else torch.tensor([qs], device=DEV)
            )
            qa = torch.full((bs, 4), -1, dtype=torch.long, device=DEV)
            qa[torch.arange(bs, device=DEV), clause] = 0
            out = {a: m(q, qa) for a, m in arms.items()}
            row = {"n_probe": n_probe, "bs": bs, "query_seed": qs,
                   "default_equal": all(torch.equal(x, y) for x, y in zip(out["before"], out["partial"], strict=True)),
                   "passing": int((out["partial"][0] >= 0).sum())}  # fmt: skip
            for other in ("full", "torch"):
                if other in out:
                    ref = out["partial"] if other == "full" else out["torch"]
                    try:
                        assert_topk_equal(*out["full"], *ref)
                        row[f"full_vs_{other if other == 'torch' else 'partial'}"] = (
                            True
                        )
                    except AssertionError:
                        row[f"full_vs_{other if other == 'torch' else 'partial'}"] = (
                            False
                        )
            rows.append(row)
            print(f"n={args.n} d={args.d} {row}", flush=True)
ok = all(
    all(v for k, v in r.items() if k.startswith(("default", "full_vs"))) for r in rows
)
json.dump({"n": args.n, "d": args.d, "before": BEFORE, "device": torch.cuda.get_device_name(), "all_equal": ok, "rows": rows},
          open(args.out, "w"), indent=1)  # fmt: skip
print("ALL EQUAL" if ok else "MISMATCH", flush=True)
