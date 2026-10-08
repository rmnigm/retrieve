"""Record inventory for campaign v2 (roadmap P-INV): one row per (record, perf entry) of every
bench record under a tree of fetched Hub subtrees, plus the summaries the roadmap needs.

    python -I inventory.py <fetched root> <out dir>

<fetched root> holds one directory per hub-index subtree (`bench fetch --results <root>/<subtree>`).
Writes inventory.csv, pivot.csv (dataset x algo x backend x code_version: records, unstable),
pass_rates.csv, wall_time.csv and reuse.csv into <out dir>. CPU only, standard library only.
"""

import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GOODREADS_FINAL = "sasrec-ssm-logq-d128"
# Subtrees whose arXiv records postdate the license-clause attrs fix (validation, Campaign).
ARXIV_FINAL_SUBTREES = ("d1/arxiv", "d1/arxiv-deep", "d1/arxiv-codesign", "artifacts/h2")
V2_BS, V2_K = {1, 16}, {100, 1000}
V2_SWEEPS = {
    "goodreads": {"c0_genre", "c1_lang_reverse", "all4"},
    "arxiv": {"c3_nversions", "c0_maincat", "all4"},
    "pubmed": {"c0_mesh", "c3_journal_reverse", "all5"},
    "yfcc10m": {"tags_and"},
}
CLOCK_FRACTION_MAX = 0.10
# Timed on a kernel the frozen code no longer runs (validation: Probe-scorer tile per width).
TIMING_SUPERSEDED = {("pubmed", "silvertorch", "triton", "72e5a90")}  # the reuse rule: < 10 % of windows below the run's max clock


def records(root: Path):
    for path in sorted(root.rglob("*.jsonl")):
        if path.name.endswith(".samples.jsonl"):
            continue
        subtree = path.relative_to(root).parts
        # <subtree...>/<suite>/<dataset>-d<dim>.jsonl; the subtree is everything above the suite dir
        sub = "/".join(subtree[:-2])
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and {"algo", "suite", "dataset"} <= rec.keys():
                yield sub, path.relative_to(root).as_posix(), i, rec


def inputs_of(rec):
    if rec.get("inputs"):
        return rec["inputs"]
    if rec["dataset"] == "goodreads":
        return f"gsasrec-d{rec['dim']}-drop0.5-id"
    return f"content_d{rec['dim']}"


def split_params(rec):
    p = rec.get("params") or {}
    q = {k: v for k, v in p.items() if k in ("n_probe", "candidate_pool", "alpha")}
    b = {k: v for k, v in p.items() if k not in q}
    return b, q


def main(root: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    recs = list(records(root))
    # The run's max sampled clock: per record file (one leg / one process family).
    file_max: dict[str, float] = defaultdict(float)
    for _, f, _, rec in recs:
        for e in rec.get("perf") or []:
            for s in e.get("window_sm_mhz") or []:
                if s:
                    file_max[f] = max(file_max[f], s)
    rows = []
    for sub, f, line, rec in recs:
        env = rec.get("env") or {}
        b, q = split_params(rec)
        o = ((rec.get("quality") or {}).get("oracle") or {})
        base = {
            "subtree": sub, "file": f, "line": line,
            "dataset": rec["dataset"], "dim": rec.get("dim"), "suite": rec["suite"],
            "inputs": inputs_of(rec), "algo": rec["algo"], "backend": rec.get("backend"),
            "filter_kind": rec.get("filter_kind"), "sweep": rec.get("sweep"),
            "build_params": json.dumps(b, sort_keys=True), "query_params": json.dumps(q, sort_keys=True),
            "seed": rec.get("seed"), "status": rec.get("status", "ok"),
            "schema_version": rec.get("schema_version"),
            "code_version": env.get("code_version"), "git_branch": env.get("git_branch"),
            "gpu": env.get("gpu"), "pass_rate": rec.get("pass_rate"),
            "recall_oracle@100": o.get("recall@100"), "recall_oracle@1000": o.get("recall@1000"),
            "elapsed_s": rec.get("elapsed_s"), "build_s": rec.get("build_s"),
            "record_unstable": rec.get("unstable"),
        }
        entries = rec.get("perf") or [None]
        for e in entries:
            row = dict(base)
            if e is None:
                row.update(k=None, bs=None, mode=None, median_ms=None, unstable=None,
                           frac_windows_below_max=None, n_windows=None)
            else:
                w = [s for s in (e.get("window_sm_mhz") or []) if s]
                mx = file_max.get(f) or 0
                row.update(
                    k=e.get("k"), bs=e.get("bs"), mode=e.get("mode"), median_ms=e.get("median_ms"),
                    unstable=e.get("unstable"), n_windows=len(w),
                    frac_windows_below_max=(sum(s < mx for s in w) / len(w)) if w and mx else None,
                )
            rows.append(row)
    cols = list(rows[0]) if rows else []
    with open(out / "inventory.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        wr.writerows(rows)

    # pivot: dataset x algo x backend x code_version -> records, records with any unstable entry
    piv: dict[tuple, list[int]] = defaultdict(lambda: [0, 0, 0])
    for sub, f, _, rec in recs:
        key = (rec["dataset"], inputs_of(rec), rec["algo"], rec.get("backend"),
               (rec.get("env") or {}).get("code_version"), sub)
        piv[key][0] += 1
        piv[key][1] += bool(rec.get("unstable"))
        piv[key][2] += rec.get("status", "ok") != "ok"
    with open(out / "pivot.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "inputs", "algo", "backend", "code_version", "subtree",
                     "records", "unstable", "not_ok"])
        for k in sorted(piv, key=lambda t: tuple(str(x) for x in t)):
            wr.writerow([*k, *piv[k]])

    # (a) per-sweep pass rate (the exact oracle's mean, identical across algos)
    pr: dict[tuple, set] = defaultdict(set)
    for _, _, _, rec in recs:
        if rec.get("pass_rate") is not None and rec.get("filter_kind") in ("clause", "bloom"):
            pr[(rec["dataset"], inputs_of(rec), rec["sweep"])].add(round(rec["pass_rate"], 6))
    with open(out / "pass_rates.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "inputs", "sweep", "pass_rate", "kept_in_v2"])
        for (d, i, s), v in sorted(pr.items()):
            wr.writerow([d, i, s, ";".join(map(str, sorted(v))), s in V2_SWEEPS.get(d, set())])

    # (b) per-cell wall time: elapsed_s per ok record, by dataset/suite/algo/backend/filter_kind;
    # elapsed_s excludes the job's build (build_s, once per job); old grids timed 3 bs x 3 k.
    wt: dict[tuple, list[float]] = defaultdict(list)
    bt: dict[tuple, list[float]] = defaultdict(list)
    shape: dict[tuple, set] = defaultdict(set)
    for _, _, _, rec in recs:
        if rec.get("status", "ok") in ("ok", "partial") and rec.get("elapsed_s"):
            k = (rec["dataset"], rec["suite"], rec["algo"], rec.get("backend"), rec.get("filter_kind"))
            wt[k].append(rec["elapsed_s"])
            bt[k].append(rec.get("build_s") or 0.0)
            shape[k].add((len(rec.get("batch_sizes") or []), len(rec.get("ks") or [])))
    with open(out / "wall_time.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["dataset", "suite", "algo", "backend", "filter_kind", "records",
                     "median_s_per_cell", "p90_s_per_cell", "sum_h", "median_build_s", "grid_bs_x_k"])
        for k in sorted(wt, key=lambda t: tuple(str(x) for x in t)):
            v = sorted(wt[k])
            wr.writerow([*k, len(v), round(statistics.median(v), 1),
                         round(v[int(0.9 * (len(v) - 1))], 1), round(sum(v) / 3600, 2),
                         round(statistics.median(bt[k]), 1),
                         ";".join(f"{a}x{b}" for a, b in sorted(shape[k]))])

    # (c) the reuse rule, per record: inputs final; kernels identical (quality: pending the freeze
    # gate's golden equality; perf: also pending CV2-LIB #16 for silvertorch/triton); timing from an
    # interleaved run (none before v2) or < 10 % of windows below the leg's max clock.
    with open(out / "reuse.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["subtree", "dataset", "suite", "algo", "backend", "filter_kind", "sweep",
                     "params", "seed", "code_version", "status", "inputs_final", "in_v2_grid",
                     "quality_reusable", "timing_clock_ok", "timing_reusable"])
        for sub, f, _, rec in recs:
            d = rec["dataset"]
            inp = inputs_of(rec)
            if d == "goodreads":
                inputs_final = inp == GOODREADS_FINAL
            elif d == "arxiv":
                inputs_final = sub in ARXIV_FINAL_SUBTREES
            else:
                inputs_final = d in ("yfcc10m", "pubmed")
            p = rec.get("params") or {}
            in_grid = (rec["algo"] != "linr_v4"
                       and not (rec.get("backend") == "official" and rec.get("filter_kind") == "clause")
                       and p.get("n_probe") not in (4, 256)
                       and rec.get("sweep") in V2_SWEEPS.get(d, set()) | {"full_scan"})
            ok = rec.get("status", "ok") == "ok"
            q_reuse = ok and inputs_final and rec.get("quality") is not None
            ents = [e for e in rec.get("perf") or [] if e.get("bs") in V2_BS and e.get("k") in V2_K]
            w = [s for e in ents for s in (e.get("window_sm_mhz") or []) if s]
            mx = file_max.get(f) or 0
            frac = (sum(s < mx for s in w) / len(w)) if w and mx else None
            clock_ok = frac is not None and frac < CLOCK_FRACTION_MAX
            cv = (rec.get("env") or {}).get("code_version")
            t_reuse = ok and inputs_final and clock_ok and (
                (d, rec["algo"], rec.get("backend"), (cv or "")[:7]) not in TIMING_SUPERSEDED)
            if t_reuse and rec["algo"] == "silvertorch" and rec.get("backend") == "triton":
                t_reuse = "pending-cv2-lib-16"  # the tile-skip change may move this arm's timing
            wr.writerow([sub, d, rec["suite"], rec["algo"], rec.get("backend"), rec.get("filter_kind"),
                         rec.get("sweep"), json.dumps(p, sort_keys=True), rec.get("seed"),
                         (rec.get("env") or {}).get("code_version"), rec.get("status", "ok"),
                         inputs_final, in_grid, q_reuse, clock_ok, t_reuse])
    print(f"{len(recs)} records, {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
