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
    "472f2fc6": "v2.5",
    "20e83bfc": "v2.6",
    "641ec3b8": "v2.7",
    "78cfbc72": "v2.8",
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


def box(r):
    """The box a record ran on (`env.host`, first 6 chars). Timings compare only within one box
    (decisions, 2026-10-10)."""
    return (r["env"].get("host") or "?")[:6]


# Records timed off physical GPU 0 before the clock-device fix: their clock fields read GPU 0, so a
# clock-driven `unstable` means clock-unknown; latencies and quality stand (controller, 2026-10-10;
# pod d GPU 1 legs). Records carry no device field, so the legs are named here: (box, dataset prefix).
# The fix (clock-device, d1d24d8) is in every harness from the campaign-v2.7 tag on (b0fbd11), so only
# the legs before it qualify.
CLOCK_UNKNOWN = (("38f5e1", "laion30m", ("v2.5", "v2.6")),)


def clock_unknown(r):
    return any(
        box(r) == b and r["dataset"].startswith(d) and cv(r) in vs for b, d, vs in CLOCK_UNKNOWN
    )


# SYNTH-TRIM widened the uniform synth attrs 7 -> 10 clauses, and until CLAUSE-SKIP (v2.7) the clause
# kernels load every clause per item whatever the query uses (+~0.7 ms at bs 16 on 3 M, flat in p;
# controller 2026-10-10). Records carry no width field: uniform synth records at v2.5 / v2.6 are the
# 10-clause, pre-CLAUSE-SKIP ones; their clause timings carry the width cost, their recall does not.
UNIFORM_SYNTH = ("arxiv-synth", "goodreads-synth", "yfcc10m-synth", "laion30m-synth")


def pre_clause_skip(r):
    return (
        r["dataset"] in UNIFORM_SYNTH and r["filter_kind"] == "clause" and cv(r) in ("v2.5", "v2.6")
    )


def official_build(r):
    """How Meta's extension was built for an official record: the recorded flags once the env field
    exists (campaign-v2.8 adds the -O3 build, OF-11), else "unrecorded" (shipped -O0 host code before
    v2.8 unless the leg's validation row says otherwise)."""
    if r["backend"] != "official":
        return ""
    b = r["env"].get("official_build")
    if not b:
        return "unrecorded"
    return f"{b.get('nvcc_append_flags') or 'shipped flags'} so:{(b.get('so_sha256') or '?')[:8]}"


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
    """The synth sweep's target pass rate from its name (p0001 -> 0.001, p1 -> 1.0; c001 -> 0.01)."""
    s = r["sweep"][1:]
    return float(s) if s == "1" else float("0." + s[1:])
