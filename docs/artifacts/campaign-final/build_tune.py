"""Writes the `tune-q` and `tune-timed` blocks of evaluation/config/suites.yaml (README.md): the campaign's per-bench
SilverTorch tuning, folded from its scratch configs (campaign-v2/v-pod1-run/tune_*config.py, campaign-v2.10/pubmed-tune).

- `tune-q` (quality only): n_lists per bench x n_probe {8 ... 4096} with n_probe <= n_lists / 4 (the user's 25 % cap,
  now also on YFCC, whose v2.9 grid went to n_lists), k 100, seed 0; clause on the kept / synth sweeps, bloom where
  the dataset config has a bloom block for the sweep.
- `tune-timed`: the timed Pareto candidates (`tune-timed-candidates.json`: [dataset, sweep, kind, n_lists, n_probe];
  goodreads / arXiv / synth from the v2.10 quality grid, YFCC the v2.9 frontier, PubMed the v2.10 candidates),
  one arm per (dataset, sweep, kind, n_lists) so each index is built once, candidates past the cap dropped;
  bs {1, 16, 64}.

usage: build_tune.py > blocks.yaml (pasted into suites.yaml; the grid tests pin the counts)
"""

import json
import sys
from pathlib import Path

import yaml

N_PROBE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
SMALL = [256, 512, 1024, 2048, 4096, 8192, 16384]
BENCH = {  # dataset: (n_lists, clause sweeps, bloom sweeps, dim)
    "goodreads": (SMALL, ["c0_genre", "c1_lang_reverse", "all4"], ["c0_genre"], 128),
    "goodreads-synth": (SMALL, ["p001", "p01", "p1"], ["p001", "p01", "p1"], 128),
    "arxiv": (SMALL, ["c3_nversions", "c0_maincat", "all4"], ["c3_nversions", "c0_maincat", "all4"], 128),
    "arxiv-synth": (SMALL, ["p001", "p01", "p1"], ["p001", "p01", "p1"], 128),
    "arxiv-corr-synth": (SMALL, ["c001", "c003", "c01"], ["c001", "c003", "c01"], 128),
    "yfcc10m": ([2048, 4096, 8192, 16384], ["tags_and"], [], 192),
    "yfcc10m-synth": ([2048, 4096, 8192, 16384], ["p001", "p01", "p1"], ["p001", "p01", "p1"], 192),
    "pubmed": ([1024, 4096, 16384, 65536], ["c0_mesh", "c3_journal_reverse", "all5"], ["c0_mesh"], 768),
}  # fmt: skip


def flow(d):
    return yaml.safe_dump(d, default_flow_style=True, width=1000).replace("\n...", "").strip()


def tune_q():
    arms = []
    for nl in sorted({n for v in BENCH.values() for n in v[0]}):
        probes = [p for p in N_PROBE if p <= nl // 4]
        for fk, idx in (("clause", 1), ("bloom", 2)):
            ds = {d: {"sweeps": v[idx]} for d, v in BENCH.items() if nl in v[0] and v[idx]}
            if ds:
                arms.append({"algo": "silvertorch", "backends": ["triton"], "filter_kinds": [fk],
                             "build": {"n_lists": [nl]}, "query": {"n_probe": probes}, "datasets": ds})  # fmt: skip
    sweeps = {d: sorted(set(v[1] + v[2])) for d, v in BENCH.items()}
    return arms, sweeps


def tune_timed(cands):
    groups: dict[tuple, list[int]] = {}
    for ds, sw, fk, nl, npb in cands:
        if npb <= nl // 4:  # the cap; YFCC's v2.9 frontier went past it
            groups.setdefault((ds, sw, fk, nl), []).append(npb)
    arms = [{"algo": "silvertorch", "backends": ["triton"], "filter_kinds": [fk], "build": {"n_lists": [nl]},
             "query": {"n_probe": sorted(set(nps))}, "datasets": {ds: {"sweeps": [sw]}}}
            for (ds, sw, fk, nl), nps in sorted(groups.items())]  # fmt: skip
    sweeps: dict[str, set] = {}
    for ds, sw, *_ in cands:
        sweeps.setdefault(ds, set()).add(sw)
    return arms, {d: sorted(s) for d, s in sorted(sweeps.items())}


def block(name, comment, head, arms, sweeps):
    out = [comment, f"{name}:"]
    out += [f"  {k}: {flow(v)}" for k, v in head.items()]
    out.append("  sweeps:")
    out += [f"    {d}: {flow(s)}" for d, s in sweeps.items()]
    out.append("  arms:")
    out += [f"    - {flow(a)}" for a in arms]
    return "\n".join(out)


def main():
    cands = json.loads((Path(__file__).parent / "tune-timed-candidates.json").read_text())
    dims = sorted({v[3] for v in BENCH.values()})
    qa, qs = tune_q()
    ta, ts = tune_timed(cands)
    print(block("tune-q", "# Per-bench SilverTorch tuning, quality only (generated: docs/artifacts/campaign-final/build_tune.py).",
                {"perf": False, "datasets": list(BENCH), "dims": dims, "filter_kinds": ["clause", "bloom"], "ks": [100],
                 "batch_sizes": [16], "seeds": [0]}, qa, qs))  # fmt: skip
    print()
    print(block("tune-timed", "# The timed Pareto points of the tuning (generated: docs/artifacts/campaign-final/build_tune.py).",
                {"datasets": list(ts), "dims": dims, "filter_kinds": ["clause", "bloom"], "ks": [100],
                 "batch_sizes": [1, 16, 64], "seeds": [0]}, ta, ts))  # fmt: skip
    print(f"# tune-q arms {len(qa)}, tune-timed arms {len(ta)} from {len(cands)} candidates", file=sys.stderr)


if __name__ == "__main__":
    main()
