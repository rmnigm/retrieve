"""Oracle blobs across code versions (PubMed by default; GT_DIR for another dataset) (gt_dir is shared and keyed by content, not code_version): `aside CV` moves every
blob built at CV to /scratch/oracle-<CV[:8]>/pubmed/ (never deletes); `compare CV` checks each set-aside blob against the
blob of the same name now in gt_dir (rebuilt at the current version), tensor by tensor with torch.equal, and prints EQUAL
or the differing keys; `restore CV` sets the current blobs aside under their own code_version and moves CV's back.

    python oracle_swap.py aside|compare|restore CODE_VERSION [GT_DIR]
"""

import shutil
import sys
from pathlib import Path

import torch

GT = Path(sys.argv[3] if len(sys.argv) > 3 else "/data/pubmed-medcpt/gt_d768")
CONTENT = ("topk", "pass_counts", "pass_rate", "targets_in_filter", "target_in_filter", "n_items", "n_queries", "n_kept",
           "k_gt", "sweep", "clauses", "fingerprint")  # fmt: skip


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


def side(cv):
    d = Path(f"/scratch/oracle-{cv[:8]}") / GT.parent.name
    d.mkdir(parents=True, exist_ok=True)
    return d


op, cv = sys.argv[1], sys.argv[2]
if op == "aside":
    for p in sorted(GT.glob("oracle_v4_*.pt")):
        if load(p)["code_version"] == cv:
            shutil.move(p, side(cv) / p.name)
            print("moved", p.name, "->", side(cv))
elif op == "compare":
    bad = 0
    for old in sorted(side(cv).glob("oracle_v4_*.pt")):
        new = GT / old.name
        if not new.exists():
            print("not rebuilt", old.name)
            continue
        a, b = load(old), load(new)
        diff = [k for k in CONTENT
                if not (torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k])]  # fmt: skip
        bad += bool(diff)
        print(
            old.name,
            a["code_version"][:8],
            "vs",
            b["code_version"][:8],
            "EQUAL" if not diff else f"DIFFER {diff}",
        )
    sys.exit(1 if bad else 0)
elif op == "restore":
    for p in sorted(GT.glob("oracle_v4_*.pt")):
        cur = load(p)["code_version"]
        if cur != cv:
            shutil.move(p, side(cur) / p.name)
            print("set aside", p.name, cur[:8])
    for p in sorted(side(cv).glob("oracle_v4_*.pt")):
        shutil.move(p, GT / p.name)
        print("restored", p.name)
