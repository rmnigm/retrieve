"""`bench merge` against `build_results.py` (results-draft): the same rows, row for row.

Usage: uv run --no-sync --with pandas python check_merge.py FR TABLES SUMMARY

FR holds the fetched `fr/*` subtrees, TABLES is `bench merge FR/* --out TABLES`, SUMMARY is the results-draft run's
`summary.json` (Hub `artifacts/results-draft/20261015e-fr`). Three checks: (1) the set of (subtree, resume key, k, bs,
mode) rows `build_results.load()` keeps equals the accepted `ok` non-YFCC rows of `timings.parquet` plus the accepted
quality-only records; (2) its per-run counts and total equal SUMMARY's; (3) the manifest's cross-host duplicates are
printed (expected none in the FR legs).
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

FR, TABLES, SUMMARY = (Path(a) for a in sys.argv[1:4])
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results-draft"))
sys.argv = ["build_results.py", str(FR), "/nonexistent"]
import build_results as br  # noqa: E402

KEY = ["dataset", "dim", "inputs", "suite", "filter_kind", "sweep", "algo", "backend", "params", "seed"]
VARIANT = ["perf_k", "perf_bs", "perf_mode"]


def theirs() -> pd.DataFrame:
    """`build_results.load()` with the subtree name kept: its body, with `subtree=d.name` added."""
    parts = []
    for d in sorted(FR.iterdir()):
        m = br.RUN.match(d.name)
        if not m or not (d / "results.parquet").is_file():
            continue
        x = pd.read_parquet(d / "results.parquet")
        owner = [re.search(r"-(pod1|pode)-", s.name) for s in d.iterdir() if s.is_dir()]
        owner = next((o[1] for o in owner if o), m["pod"])
        if m["on"]:
            owner = m["pod"]
        elif m["pod"] == "podf" and m["seed"] == "2":
            owner = "pod1"
        parts.append(x.assign(subtree=d.name, leg=m["leg"], pod=owner, run_seed=int(m["seed"]), rerun=bool(m["rerun"])))
    df = pd.concat(parts, ignore_index=True)
    df = df[(df.status == "ok") & ~df.dataset.str.startswith("yfcc")]
    df["unstable"] = df.unstable.fillna(False).astype(bool)
    key = ["leg", "pod", "suite", "dataset", "dim", "filter_kind", "sweep", "algo", "backend", "params", "seed",
           "perf_mode", "perf_bs", "perf_k"]
    df = df.sort_values(["unstable", "rerun"], ascending=[True, False])
    return df.drop_duplicates(key, keep="first").reset_index(drop=True)


def rows(df: pd.DataFrame, cv: str) -> set:
    """(subtree, key block, code_version, k, bs, mode) per row; a quality-only row's variant is None."""
    def v(x):
        return None if isinstance(x, float) and pd.isna(x) else x

    return {
        (r.subtree, *(getattr(r, k) for k in KEY), getattr(r, cv), *(v(getattr(r, c)) for c in VARIANT))
        for r in df.itertuples(index=False)
    }


def main() -> None:
    t = theirs()
    real = br.load()
    assert len(real) == len(t), (len(real), len(t))
    summary = json.loads(SUMMARY.read_text())
    runs = t.groupby(["leg", "pod", "run_seed", "rerun"]).size()
    runs = {f"{lg}-{p}-s{s}{'-rerun' if r else ''}": int(n) for (lg, p, s, r), n in runs.items()}
    assert runs == summary["runs"], {k: (runs.get(k), summary["runs"].get(k)) for k in runs.keys() | summary["runs"].keys() if runs.get(k) != summary["runs"].get(k)}
    assert len(t) == summary["records"] == len(real), (len(t), summary["records"])
    print(f"build_results.load(): {len(t)} rows, per-run counts equal summary.json")

    tm = pd.read_parquet(TABLES / "timings.parquet")
    rc = pd.read_parquet(TABLES / "records.parquet")
    keep = lambda d: d[d.accepted & (d.status == "ok") & ~d.dataset.str.startswith("yfcc")]  # noqa: E731
    tm, rc = keep(tm), keep(rc)
    quality_only = rc[~rc.cell.isin(tm.cell)].assign(perf_k=None, perf_bs=None, perf_mode=None)
    mine = rows(tm, "env_code_version") | rows(quality_only, "env_code_version")
    t = t.rename(columns={"env_code_version": "cv"})
    ours = rows(t, "cv")
    only_theirs, only_mine = ours - mine, mine - ours
    print(f"merge tables: {len(mine)} rows ({len(tm)} timed + {len(quality_only)} quality-only)")
    if only_theirs or only_mine:
        for r in sorted(only_theirs, key=str)[:10]:
            print("  only build_results:", r)
        for r in sorted(only_mine, key=str)[:10]:
            print("  only merge:", r)
        sys.exit(1)
    print("row for row equal")
    man = yaml.safe_load((TABLES / "manifest.yaml").read_text())
    print(f"cross-host duplicate cells: {len(man['cross_host_duplicates'])}")
    for d in man["cross_host_duplicates"]:
        print("  ", d)
    print(f"subtrees {len(man['subtrees'])}, cell groups {len(man['cells'])}, rows {man['rows']}")


if __name__ == "__main__":
    main()
