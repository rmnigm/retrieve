"""Exit 0 when the second record file re-records cells of the first only as `partial` (a profile-only
pass such as H-KSUM's: run.sh gives it its own tree), else 1 (the leg joins the tree)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "evaluation"))
from bench import records  # noqa: E402

old, new = (list(records.read_records(Path(p))) for p in sys.argv[1:3])
keys = {records.record_key(r) for r in old}
shared = [r for r in new if records.record_key(r) in keys]
sys.exit(0 if shared and all(r["status"] == "partial" for r in shared) else 1)
