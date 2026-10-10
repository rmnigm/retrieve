"""Compare each oracle blob in NEW_DIR with the blob of the same name in OLD_DIR (read-only on both), tensor by tensor with torch.equal;
prints EQUAL / DIFFER per blob and `no counterpart` for names OLD_DIR lacks; exit 1 on any DIFFER.

    python oracle_compare.py OLD_DIR NEW_DIR
"""

import sys
from pathlib import Path

import torch

CONTENT = ("topk", "pass_counts", "pass_rate", "targets_in_filter", "target_in_filter", "n_items", "n_queries", "n_kept",
           "k_gt", "sweep", "clauses", "fingerprint")  # fmt: skip
old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
bad = 0
for new in sorted(new_dir.glob("oracle_v4_*.pt")):
    old = old_dir / new.name
    if not old.exists():
        print("no counterpart", new.name)
        continue
    a, b = (torch.load(p, map_location="cpu", weights_only=False) for p in (old, new))
    diff = [k for k in CONTENT if not (torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k])]
    bad += bool(diff)
    print(new.name, a["code_version"][:8], "vs", b["code_version"][:8], "EQUAL" if not diff else f"DIFFER {diff}")
sys.exit(1 if bad else 0)
