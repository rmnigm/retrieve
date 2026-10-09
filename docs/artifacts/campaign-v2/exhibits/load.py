"""Records of every fetched tree, one tree per code_version (README.md), as flat dicts."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "evaluation"))
from bench import records  # noqa: E402

CV_LABEL = {
    "408b1188": "v2",
    "f01255f1": "v2.1",
    "0d23c615": "v2.2",
    "1258a63e": "v2.3",
    "d67d6263": "v2.4",
    "72e5a90c": "d1",
    "c0e42d1a": "d1-c0e4",
}
ALGO = {
    "linr_v1_filter_mask": "V1",
    "linr_v2": "V2",
    "linr_v3": "V3",
    "silvertorch": "ST",
    "postfilter": "PF",
}
EXACT = {"linr_v1_filter_mask", "linr_v2"}


def cv(r):
    c = r["env"]["code_version"]
    return CV_LABEL.get(c[:8], c[:8])


def load(trees):
    """Latest record per resume key in each tree; tree name in ``_tree``."""
    out = []
    for t in trees:
        for r in records.latest(Path(t)):
            r["_tree"] = Path(t).name
            out.append(r)
    return out


def arm(r):
    p = {k: v for k, v in r["params"].items()}
    s = f"{ALGO.get(r['algo'], r['algo'])}/{r['backend']}"
    return s + ("" if not p else " " + ",".join(f"{k}={v}" for k, v in sorted(p.items())))


def key(r, drop=()):
    return (
        r["dataset"],
        r["dim"],
        r["suite"],
        r["filter_kind"],
        r["sweep"],
        r["algo"],
        r["backend"],
        json.dumps({k: v for k, v in r["params"].items() if k not in drop}, sort_keys=True),
        r["seed"],
    )


def short(r):
    return (
        f"{cv(r)}:{r['dataset']}/{r['suite']}/{r['filter_kind']}/{r['sweep']}/{arm(r)}/s{r['seed']}"
    )


def recall(r, k=100):
    q = (r.get("quality") or {}).get("oracle") or {}
    return q.get(f"recall@{k}")


def perf(r, bs, k, mode):
    for e in r.get("perf") or []:
        if e["bs"] == bs and e["k"] == k and e["mode"] == mode and e.get("median_ms") is not None:
            return e
    return None


def pass_p(r):
    """The synth sweep's target pass rate from its name (p0001 -> 0.001, p1 -> 1.0)."""
    s = r["sweep"][1:]
    return float(s) if s == "1" else float("0." + s[1:])
