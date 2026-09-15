"""``bench report`` — every thesis and paper table and figure, from the records only
(roadmap D4, H §6 WP-6, V §5.3).

``records.flatten`` writes ``flat.csv`` first and every table and figure is built from it
(H §8.2 G); ``records.read_records`` is read a second time for the provenance block alone,
because ``schema_version``, ``partial_reasons``, ``stage`` and ``error`` are not columns of
``flat.csv``.

**Nothing emitted here is citable unless ``--gate`` names a green roadmap gate** (CLAUDE.md
rule 2). Without it — or with a `failed` cell, a `partial` record, a `dirty` library subtree
or a record taken on a `dev/*` branch — every artifact carries a visible NOT CITABLE marker
in its own banner and in its caption. ``--gate`` cannot override the evidence, only the
default.

Failed cells never enter a number and are listed in ``report.md``; `partial` records are
marked ``*`` and `unstable` perf entries ``†``, both counted in the caption. Latency
artifacts state which clock estimator they used: the per-variant under-load
``perf[].sm_mhz``, never an idle or whole-run median (H §12.4).
"""

from __future__ import annotations

import csv
import datetime as dt
import inspect
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click
import matplotlib

from bench import records

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — after use("Agg")

ALGO_LABEL = {
    "linr_v1_filter_mask": "LiNR V1",
    "linr_v2": "LiNR V2",
    "linr_v4": "LiNR V4",
    "linr_v3": "LiNR V3",
    "silvertorch": "SilverTorch",
}
DATASET_LABEL = {"goodreads": "Goodreads", "arxiv": "arXiv"}
SPEEDUP_BASE = "linr_v1_filter_mask"  # the thesis's 1.00x column

# H §2.7, verbatim: the differences that make our numbers not theirs.
COMPARABILITY = (
    "in-process measurement: no RPC, no user tower, no 5\\,000-request replay, "
    "closed-loop single client rather than open-loop client-side QPS",
    "0.8--5.4\\,M items against SilverTorch's 10/80\\,M and LiNR's 15.5\\,M; "
    "one A100-SXM4-80GB, never sharded",
    "$k \\leq 1000$ against SilverTorch's top-2048 and LiNR's label recall@2000",
    "no OverArch, no Value Model, no multi-embedding queries, no live updates",
    "query predicates synthesised from item attributes on public datasets, "
    "not replayed production traffic",
)

# Reported by the papers, for the comparison table. Constants, cited per row; never derived.
PAPER_REPORTED = (
    ("SilverTorch", "80M items, 24 probes, 2$\\times$A100", "--", "$\\leq$200 (budget)",
     "1210", "--", "silvertorch.md \\S6.2"),
    ("SilverTorch", "10M items, 1 A100", "--", "$\\leq$200 (budget)",
     "3802", "--", "silvertorch.md \\S6.2"),
    ("LiNR V1 (PyTorch)", "high pass, 15.5M, $B=1$", "4.8", "4.9 (p95)",
     "--", "0.688", "linr.md \\S5.3 tab.3"),
    ("LiNR V1 (PyTorch)", "high pass, 15.5M, $B=16$", "22.8", "23.1 (p95)",
     "--", "0.688", "linr.md \\S5.3 tab.3"),
    ("LiNR V2 (PyTorch)", "high pass, 15.5M, $B=1$", "14.6", "47.8 (p95)",
     "--", "0.688", "linr.md \\S5.3 tab.3"),
    ("LiNR V2 (PyTorch)", "low pass, 15.5M, $B=1$", "1.9", "2.1 (p95)",
     "--", "--", "linr.md \\S5.3 tab.5"),
    ("LiNR V2 (PyTorch)", "low pass, 15.5M, $B=16$", "21.4", "21.9 (p95)",
     "--", "--", "linr.md \\S5.3 tab.5"),
)  # fmt: skip


# ----- loading ----------------------------------------------------------------


def _cast(v: str) -> Any:
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _load(results_dir: Path, out: Path) -> tuple[Path, list[dict[str, Any]]]:
    out.mkdir(parents=True, exist_ok=True)  # records.flatten writes, it does not create
    flat = records.flatten(results_dir, out / "flat.csv")
    with open(flat, newline="") as f:
        return flat, [{k: _cast(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _latest(results_dir: Path) -> list[dict[str, Any]]:
    """The same selection ``flatten`` makes — last record per resume key — but nested, so the
    provenance block can see the fields ``flat.csv`` does not carry."""
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(results_dir).glob("*/*.jsonl")):
        if not path.name.endswith(".samples.jsonl"):
            for rec in records.read_records(path):
                latest[records.record_key(rec)] = rec
    return list(latest.values())


def _samples(results_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*/*.samples.jsonl")):
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        for i, line in enumerate(lines):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1:
                    raise
    return out


# ----- provenance and citability ----------------------------------------------


MAIN_BRANCHES = ("development", "main")  # rule 2: numbers from a branch are not paper material


def provenance(recs: list[dict[str, Any]], gate: str | None) -> dict[str, Any]:
    """The rule-2 verdict over a set of records. Shared with ``bench upload``, which puts
    it in every published ``MANIFEST.json`` so a Hub copy cannot claim more than the
    tables would."""
    env = [r.get("env") or {} for r in recs]
    status = Counter(r.get("status", "ok") for r in recs)
    dirty = [r for r, e in zip(recs, env, strict=True) if e.get("dirty")]
    blockers = []
    if not gate:
        blockers.append("no --gate given: these records come from no green roadmap gate")
    if status["failed"]:
        blockers.append(f"{status['failed']} record(s) with status=failed")
    if status["partial"]:
        blockers.append(f"{status['partial']} record(s) with status=partial")
    if dirty:
        blockers.append(f"{len(dirty)} record(s) with env.dirty (library subtree was dirty)")
    off = sorted({e["git_branch"] for e in env
                  if e.get("git_branch") and e["git_branch"] not in MAIN_BRANCHES})
    if off:
        blockers.append(f"record(s) produced on {', '.join(off)} — CLAUDE.md rule 2: "
                        "harness numbers from a branch are not paper material")
    if not recs:
        blockers.append("no records")
    started = sorted(e.get("started") or "" for e in env)
    return {
        "n_records": len(recs),
        "status": dict(sorted(status.items())),
        "unstable": sum(1 for r in recs if r.get("unstable")),
        "schema_versions": sorted({r.get("schema_version") for r in recs}, key=str),
        "code_versions": sorted({e.get("code_version") for e in env if e.get("code_version")}),
        "commits": sorted({e.get("commit") for e in env if e.get("commit")}),
        "branches": sorted({e.get("git_branch") for e in env if e.get("git_branch")}),
        "gpus": sorted({e.get("gpu") for e in env if e.get("gpu")}),
        "hosts": sorted({e.get("host") for e in env if e.get("host")}),
        "runs": sorted({(e.get("commit"), e.get("git_branch")) for e in env}),
        "started": (started[0], started[-1]) if started else ("", ""),
        "gate": gate,
        "citable": gate is not None and not blockers,
        "blockers": blockers,
        "generated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _banner(prov: dict[str, Any], results_dir: Path) -> str:
    v = prov["citable"]
    lines = [
        "% GENERATED by `bench report` — DO NOT EDIT BY HAND; regenerate instead.",
        f"% source: {results_dir}   generated: {prov['generated']}",
        f"% records: {prov['n_records']} {prov['status']}"
        f"   cells flagged unstable: {prov['unstable']}",
        f"% code_version: {','.join(prov['code_versions']) or '?'}"
        f"   commit: {','.join(prov['commits']) or '?'}"
        f"   branch: {','.join(prov['branches']) or '?'}",
        f"% schema_version: {','.join(map(str, prov['schema_versions'])) or '?'}"
        f"   gpu: {','.join(prov['gpus']) or '?'}   run window: {prov['started'][0]}"
        f" .. {prov['started'][1]}",
    ]
    if v:
        lines.append(f"% PROVENANCE: citable — gate {prov['gate']} declared green by the caller.")
    else:
        lines.append("% PROVENANCE: *** NOT CITABLE *** (CLAUDE.md rule 2). Reasons:")
        lines += [f"%   - {b}" for b in prov["blockers"]]
        lines.append("%   Built from pre-campaign records: roadmap D1 has not produced these.")
    return "\n".join(lines) + "\n"


def _caption(prov: dict[str, Any], text: str) -> str:
    mark = "" if prov["citable"] else "\\textbf{[PRE-CAMPAIGN RECORDS — NOT CITABLE]}~"
    return mark + text


# ----- selection and reduction -------------------------------------------------


def _sel(rows: list[dict[str, Any]], **eq: Any) -> list[dict[str, Any]]:
    keep = [r for r in rows if r.get("status") != "failed"]
    for field, want in eq.items():
        if want is not None:
            keep = [r for r in keep if r.get(field) == want]
    return keep


def _reduce(rows: list[dict[str, Any]], field: str, notes: list[str], what: str) -> dict | None:
    """Median across seeds of one value, with the seed min--max, the marks the caller must
    print, and a note when several parameter sets collide onto one table cell."""
    rows = [r for r in rows if r.get(field) is not None]
    if not rows:
        return None
    by_params = defaultdict(list)
    for r in rows:
        by_params[r.get("params") or "{}"].append(r)
    params = sorted(by_params)
    if len(params) > 1:
        notes.append(f"{_esc(what)}: {len(params)} parameter sets present "
                     f"{_esc(', '.join(params))}; showing {_esc(params[0])}")
    chosen = by_params[params[0]]
    by_seed: dict[Any, list[float]] = defaultdict(list)
    for r in chosen:
        by_seed[r.get("seed")].append(float(r[field]))
    per_seed = [statistics.median(v) for v in by_seed.values()]
    return {
        "value": statistics.median(per_seed),
        "lo": min(per_seed),
        "hi": max(per_seed),
        "n_seeds": len(per_seed),
        "unstable": field.startswith("perf_") and any(r.get("perf_unstable")
                                                        for r in chosen),
        "partial": any(r.get("status") == "partial" for r in chosen),
        "spread": max((r.get("perf_spread") or 0.0) for r in chosen),
        "params": params[0],
    }


def _cells(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per record: the perf columns repeat a cell's quality and memory columns."""
    seen, out = set(), []
    for r in rows:
        key = tuple(r.get(f) for f in records.KEY_FIELDS)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _esc(text: Any) -> str:
    """LaTeX-escape a string that came out of the records (sweep names, parameter JSON)."""
    out = str(text)
    for ch, rep in (("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("#", "\\#"),
                    ("%", "\\%"), ("&", "\\&"), ("$", "\\$")):
        out = out.replace(ch, rep)
    return out


def _f(x: float | None, places: int = 4) -> str:
    """A bare number in the thesis's decimal-comma convention."""
    return "---" if x is None else f"${format(x, f'.{places}f').replace('.', '{,}')}$"


def _sci(x: float | None) -> str:
    if x is None:
        return "---"
    if x == 0:
        return "$0$"
    e = int(f"{x:e}".split("e")[1])
    return f"${x / 10 ** e:.2f}{{\\times}}10^{{{e}}}$".replace(".", "{,}", 1)


def _mark(v: dict[str, Any]) -> str:
    return ("^{\\dagger}" if v["unstable"] else "") + ("^{*}" if v["partial"] else "")


def _num(v: dict[str, Any] | None, places: int = 4) -> str:
    if v is None:
        return "---"
    body = format(v["value"], f".{places}f").replace(".", "{,}")
    return f"${body}{_mark(v)}$"


def _clock_note(rows: list[dict[str, Any]]) -> str:
    mhz = sorted({r["perf_sm_mhz"] for r in rows if r.get("perf_sm_mhz") is not None})
    seen = f"{mhz[0]:.0f}--{mhz[-1]:.0f}\\,MHz" if mhz else "no sample"
    return (
        "Clock estimator: the per-variant \\emph{under-load} \\texttt{perf[].sm\\_mhz} "
        f"({seen} over the {len(rows)} rows behind this table). Idle and whole-run medians "
        "(\\texttt{env.sm\\_mhz\\_idle}, schema-1 \\texttt{env.sm\\_mhz}) are not used and no "
        "clock normalisation is applied; SM clocks cannot be locked on this box."
    )


# ----- LaTeX -------------------------------------------------------------------


def _table(prov, results_dir, *, caption, label, colspec, header, body, notes) -> str:
    ncol = len(colspec.replace("|", ""))
    rows = body or [[f"\\multicolumn{{{ncol}}}{{c}}{{--- no matching cells ---}}"]]
    note_tex = ""
    if notes:
        inner = "\n".join(f"    \\footnotesize {n} \\\\" for n in notes)
        note_tex = ("\n    \\vspace{2pt}\\par\n    \\begin{minipage}{\\linewidth}\n"
                    + inner + "\n    \\end{minipage}")
    return (
        _banner(prov, results_dir)
        + "\\begin{table}[htbp]\n"
        f"    \\caption{{{_caption(prov, caption)}}}\n"
        f"    \\label{{{label}}}\n"
        "    \\centering\n"
        + ("    \\small\n" if ncol > 6 else "")
        + f"    \\begin{{tabular}}{{{colspec}}}\n"
        "        \\toprule\n"
        + ("        " + " & ".join(header) + " \\\\\n        \\midrule\n" if header else "")
        + "".join("        " + " & ".join(r) + " \\\\\n" for r in rows)
        + "        \\bottomrule\n    \\end{tabular}"
        + note_tex
        + "\n\\end{table}\n"
    )


def _legend(notes: list[str]) -> list[str]:
    return [
        "$^{\\dagger}$ the cell's timing windows spread more than 5\\,\\% "
        "(\\texttt{unstable}); $^{*}$ built from a \\texttt{partial} record. "
        "Cells with \\texttt{status: failed} are excluded and listed in \\texttt{report.md}.",
        *dict.fromkeys(notes),
    ]


# ----- artifacts ---------------------------------------------------------------


def tab_recall_nofilter(c) -> list[Path]:
    rows = _sel(c.rows, filter_kind="none", dim=c.dim, backend=c.backend)
    datasets = sorted({r["dataset"] for r in rows})
    algos = [a for a in ALGO_LABEL if any(r["algo"] == a for r in rows)]
    notes: list[str] = []
    body = [
        [DATASET_LABEL.get(d, d)]
        + [
            _num(_reduce(_cells(_sel(rows, dataset=d, algo=a)), f"heldout_recall@{c.k}",
                         notes, f"{d}/{a}"))
            for a in algos
        ]
        for d in datasets
    ]  # fmt: skip
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Recall@{c.k} относительно пользовательских действий (без фильтрации), "
                f"$d={c.dim}$, backend \\texttt{{{c.backend}}}",
        label="tab:recall_nofilter",
        colspec="l" + "c" * len(algos),
        header=["Набор данных"] + [ALGO_LABEL[a] for a in algos],
        body=body, notes=_legend(notes),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-recall_nofilter.tex", tex)]


def tab_pareto(c) -> list[Path]:
    written = []
    for ds in sorted({r["dataset"] for r in c.rows}):
        rows = _sel(c.rows, dataset=ds, dim=c.dim, backend=c.backend)
        perf = _sel(rows, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode)
        sweeps = sorted({r["sweep"] for r in rows if r["filter_kind"] != "none"})
        notes, body = [], []
        for sw in sweeps:
            base = _reduce(_sel(perf, sweep=sw, algo=SPEEDUP_BASE), "perf_median_ms",
                           [], f"{ds}/{sw}/base")
            for i, a in enumerate([x for x in ALGO_LABEL
                                   if any(r["algo"] == x and r["sweep"] == sw for r in rows)]):
                q = _reduce(_cells(_sel(rows, sweep=sw, algo=a)), f"oracle_recall@{c.k}",
                            notes, f"{ds}/{sw}/{a}")
                t = _reduce(_sel(perf, sweep=sw, algo=a), "perf_median_ms", [], f"{ds}/{sw}/{a}")
                m = _reduce(_cells(_sel(rows, sweep=sw, algo=a)), "index_mib", [], f"{ds}/{sw}/{a}")
                sp = ("---" if not (base and t)
                      else _f(base["value"] / t["value"], 2)[:-1] + "\\times$")
                body.append([
                    f"\\textbf{{{_esc(sw)}}}" if i == 0 else "",
                    ALGO_LABEL[a], _num(q), _num(t, 3), sp, _num(m, 1),
                ])
        tex = _table(
            c.prov, c.results_dir,
            caption=f"Качество, время работы и память на {DATASET_LABEL.get(ds, ds)} "
                    f"($d={c.dim}$, $K={c.k}$, $B={c.bs}$, режим \\texttt{{{c.mode}}}, "
                    f"backend \\texttt{{{c.backend}}}).",
            label=f"tab:pareto_{ds}",
            colspec="llcccc",
            header=["Условие", "Алгоритм", "Полнота", "$t_\\mathrm{med}$ (мс)",
                    "Ускорение", "Память (МиБ)"],
            body=body,
            notes=_legend(notes + [
                f"Ускорение относительно {ALGO_LABEL.get(SPEEDUP_BASE, SPEEDUP_BASE)} "
                "в том же условии ('---' when that row is absent).",
                _clock_note(perf),
            ]),
        )  # fmt: skip
        written.append(_write(c.out / "tables" / f"tab-pareto_{ds}.tex", tex))
    return written


def _batch_dataset(c) -> str | None:
    """The dataset with the most (algo, bs) coverage; ties alphabetically. Printed in the
    banner and in report.md so the choice is never invisible."""
    counts = Counter()
    for r in _sel(c.rows, dim=c.dim, perf_k=c.k, perf_mode=c.mode, backend=c.backend):
        counts[r["dataset"]] = counts[r["dataset"]] + 1
    return min(sorted(counts), key=lambda d: -counts[d]) if counts else None


def tab_batch_scaling(c) -> list[Path]:
    ds = c.batch_dataset or _batch_dataset(c)
    rows = _sel(c.rows, dataset=ds, dim=c.dim, perf_k=c.k, perf_mode=c.mode, backend=c.backend)
    rows = [r for r in rows if r.get("sweep") == c.sweep] if c.sweep else rows
    bss = sorted({r["perf_bs"] for r in rows if r.get("perf_bs") is not None})
    algos = [a for a in ALGO_LABEL if any(r["algo"] == a for r in rows)]
    notes, body = [], []
    for a in algos:
        cells = []
        for bs in bss:
            v = _reduce(_sel(rows, algo=a, perf_bs=bs), "perf_median_ms", notes, f"{a}/bs{bs}")
            if v is None:
                cells.append("---")
                continue
            amortised = {**v, "value": v["value"] / bs}
            cells.append(_num(amortised, 3) + f"\\,{{\\tiny$\\pm${v['spread'] * 100:.1f}\\%}}")
        body.append([ALGO_LABEL[a]] + cells)
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Время работы (мс/запрос) при различных размерах батча $B$ "
                f"({DATASET_LABEL.get(ds, ds)}, $d={c.dim}$, $K={c.k}$, "
                f"режим \\texttt{{{c.mode}}}).",
        label="tab:batch_scaling",
        colspec="l" + "c" * len(bss),
        header=["Алгоритм"] + [f"$B={b}$" for b in bss],
        body=body,
        notes=_legend(notes + [
            f"Dataset chosen by coverage: \\texttt{{{ds}}}. $\\pm$ is the window spread "
            "$(\\max-\\min)/\\mathrm{med}$ of the three timing windows. "
            "\\textbf{$B=1$ is noise-dominated on this box} — the pre-v2 harness's own "
            "repeats spread up to 21.1\\,\\% at $B=1$ against $\\leq$0.4\\,\\% at $B=8$ — "
            "so no speedup claim is made from a $B=1$ column.",
            _clock_note(rows),
        ]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-batch_scaling.tex", tex)]


def tab_memory(c) -> list[Path]:
    rows = _cells(_sel(c.rows, dim=c.dim, backend=c.backend))
    datasets = sorted({r["dataset"] for r in rows})
    algos = [a for a in ALGO_LABEL if any(r["algo"] == a for r in rows)]
    notes: list[str] = []
    body = [
        [DATASET_LABEL.get(d, d)]
        + [_num(_reduce(_sel(rows, dataset=d, algo=a), "index_mib", notes, f"{d}/{a}"), 1)
           for a in algos]
        for d in datasets
    ]  # fmt: skip
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Размер индекса (МиБ) при $d={c.dim}$.",
        label="tab:memory",
        colspec="l" + "c" * len(algos),
        header=["Датасет"] + [ALGO_LABEL[a] for a in algos],
        body=body,
        notes=_legend(notes + [
            "\\texttt{index\\_mib} is $\\sum$ buffers of the algorithm module, its filter "
            "submodule included; \\texttt{silvertorch} carries its attribute tensors inside "
            "this number.",
        ]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-memory.tex", tex)]


def tab_backend_parity(c) -> list[Path]:
    rows = _sel(c.rows, dim=c.dim, perf_k=c.k, perf_bs=c.bs)
    body = []
    for key in sorted({(r["dataset"], r["sweep"], r["algo"], r["backend"]) for r in rows}):
        ds, sw, algo, be = key
        sub = _sel(rows, dataset=ds, sweep=sw, algo=algo, backend=be)
        cell = _cells(sub)[0]
        eager = _reduce(_sel(sub, perf_mode="eager"), "perf_median_ms", [], "")
        graph = _reduce(_sel(sub, perf_mode="graph"), "perf_median_ms", [], "")
        ratio = (_f(eager["value"] / graph["value"], 2)[:-1] + "\\times$"
                 if eager and graph else "---")
        jac = cell.get(f"quality_jaccard_vs_first@{c.k}")
        diff = cell.get("quality_score_max_abs_diff")
        body.append([
            DATASET_LABEL.get(ds, ds), _esc(sw), ALGO_LABEL.get(algo, algo),
            f"\\texttt{{{_esc(be)}}}", f"\\texttt{{{_esc(cell.get('path'))}}}",
            _f(jac, 6), _sci(diff),
            _num(eager, 3), _num(graph, 3), ratio,
        ])
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Backend parity and eager-vs-graph latency ($d={c.dim}$, $K={c.k}$, "
                f"$B={c.bs}$).",
        label="tab:backend_parity",
        colspec="lllllccccc",
        header=["Dataset", "Sweep", "Algo", "Backend", "Path", f"jaccard@{c.k}",
                "$|\\Delta s|_{\\max}$", "eager (мс)", "graph (мс)", "eager/graph"],
        body=body,
        notes=_legend([
            "\\texttt{jaccard} and $|\\Delta s|_{\\max}$ are against the first backend of the "
            "same $(dataset, dim, algo, params, seed)$ group (the parity spill file); the "
            "reference row itself shows '---'. A \\texttt{graph} column of '---' is a "
            "non-capturable path (\\texttt{official} is eager-only).",
            _clock_note(rows),
        ]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-backend_parity.tex", tex)]


def tab_recall_at_budget(c) -> list[Path]:
    """H §8.2 H: the best recall reachable under a p99 latency budget."""
    rows = _sel(c.rows, dim=c.dim, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode)
    body = []
    for ds, sw in sorted({(r["dataset"], r["sweep"]) for r in rows}):
        sub = _sel(rows, dataset=ds, sweep=sw)
        cells = []
        for budget in c.budgets:
            best = None
            for r in sub:
                p99, rec = r.get("perf_p99_ms"), r.get(f"oracle_recall@{c.k}")
                if p99 is not None and rec is not None and p99 <= budget:
                    if best is None or rec > best[0]:
                        best = (rec, r["algo"], r["backend"])
            cells.append("---" if best is None
                         else _f(best[0]) + f" ({ALGO_LABEL.get(best[1], best[1])})")
        body.append([DATASET_LABEL.get(ds, ds), _esc(sw)] + cells)
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Лучшая полнота Recall@{c.k} при бюджете p99 (режим \\texttt{{{c.mode}}}, "
                f"$B={c.bs}$, $d={c.dim}$).",
        label="tab:recall_at_budget",
        colspec="ll" + "c" * len(c.budgets),
        header=["Датасет", "Условие"] + [f"p99 $\\leq$ {b:g} мс" for b in c.budgets],
        body=body,
        notes=_legend([_clock_note(rows)]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-recall_at_budget.tex", tex)]


def tab_paper_comparison(c) -> list[Path]:
    rows = _sel(c.rows, dim=c.dim, perf_bs=c.compare_bs, perf_k=c.k, perf_mode="eager",
                backend=c.backend)
    notes, body = [], []
    for ds, sw, algo in sorted({(r["dataset"], r["sweep"], r["algo"]) for r in rows}):
        sub = _sel(rows, dataset=ds, sweep=sw, algo=algo)
        cell = _cells(sub)[0]
        mean = _reduce(sub, "perf_mean_ms", notes, f"{ds}/{sw}/{algo}")
        p99 = _reduce(sub, "perf_p99_ms", [], "")
        qps = _reduce(sub, "perf_qps", [], "")
        pr = cell.get("pass_rate")
        body.append([
            f"ours: {ALGO_LABEL.get(algo, algo)}",
            f"{DATASET_LABEL.get(ds, ds)} {_esc(sw)}, "
            f"{cell.get('n_items')} items, $B={c.compare_bs}$",
            _num(mean, 3), _num(p99, 3), _num(qps, 0), _f(pr, 3),
            "this work",
        ])
    body += [list(r) for r in PAPER_REPORTED]
    tex = _table(
        c.prov, c.results_dir,
        caption="Наши числа рядом с опубликованными SilverTorch и LiNR. "
                "Строки не сопоставимы напрямую: см. сноски.",
        label="tab:paper_comparison",
        colspec="llccccl",
        header=["Система", "Условия", "mean (мс)", "p99 / p95 (мс)", "QPS",
                "pass rate / recall", "Источник"],
        body=body,
        notes=_legend(notes + [
            "\\textbf{Не сравнение при равных условиях.} Отличия протокола (H \\S2.7): "
            + "; ".join(COMPARABILITY) + ".",
            "Наши строки — \\texttt{eager} (число, сопоставимое с обеими статьями); "
            "\\texttt{graph} приводится отдельно и никогда вместо него.",
            _clock_note(rows),
        ]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-paper_comparison.tex", tex)]


# ----- figures -----------------------------------------------------------------


def _figure(path: Path, fig, prov: dict[str, Any]) -> Path:
    if not prov["citable"]:
        fig.text(0.5, 0.5, "PRE-CAMPAIGN\nNOT CITABLE", ha="center", va="center",
                 fontsize=34, color="red", alpha=0.18, rotation=25, zorder=10)
    fig.text(0.005, 0.005,
             f"bench report {prov['generated']} | code_version "
             f"{','.join(prov['code_versions'])[:12]} | commit {','.join(prov['commits'])}",
             fontsize=5, color="0.35")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty(path: Path, prov, msg: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, wrap=True)
    return _figure(path, fig, prov)


def fig_pareto(c) -> list[Path]:
    written = []
    for ds in sorted({r["dataset"] for r in c.rows}) or [None]:
        path = c.out / "figures" / f"fig-pareto-{ds}.png"
        rows = _sel(c.rows, dataset=ds, dim=c.dim, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode,
                    backend=c.backend)
        pts = [(r, r.get("perf_median_ms"), r.get(f"oracle_recall@{c.k}")) for r in rows]
        pts = [p for p in pts if p[1] is not None and p[2] is not None]
        if not pts:
            written.append(_empty(path, c.prov, f"no cells: {ds} d{c.dim} k{c.k} "
                                                f"bs{c.bs} {c.mode} {c.backend}"))
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for algo in [a for a in ALGO_LABEL if any(p[0]["algo"] == a for p in pts)]:
            sub = [p for p in pts if p[0]["algo"] == algo]
            ax.scatter([p[1] for p in sub], [p[2] for p in sub], s=38,
                       label=ALGO_LABEL.get(algo, algo))
            for r, x, y in sub:
                ax.annotate(f"{r['sweep']} {r['params'] if r['params'] != '{}' else ''}",
                            (x, y), fontsize=6, xytext=(3, 3),
                            textcoords="offset points", color="0.4")
        ax.set_xlabel(f"$t_{{med}}$ (ms), {c.mode}, $B={c.bs}$")
        ax.set_ylabel(f"Recall@{c.k} vs the exact filtered oracle")
        ax.set_title(f"Recall–latency Pareto, {DATASET_LABEL.get(ds, ds)} d{c.dim}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        written.append(_figure(path, fig, c.prov))
    return written


def fig_qps_recall(c) -> list[Path]:
    path = c.out / "figures" / "fig-qps-recall.png"
    rows = _sel(c.rows, dim=c.dim, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode, backend=c.backend)
    pts = [(r, r.get("perf_qps"), r.get(f"oracle_recall@{c.k}")) for r in rows]
    pts = [p for p in pts if p[1] is not None and p[2] is not None]
    if not pts:
        return [_empty(path, c.prov, f"no cells with qps and oracle_recall@{c.k}")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for algo in [a for a in ALGO_LABEL if any(p[0]["algo"] == a for p in pts)]:
        sub = [p for p in pts if p[0]["algo"] == algo]
        ax.scatter([p[2] for p in sub], [p[1] for p in sub], s=38,
                   label=ALGO_LABEL.get(algo, algo))
    ax.set_xlabel(f"Recall@{c.k}")
    ax.set_ylabel(f"QPS (closed loop, $B={c.bs}$, {c.mode})")
    ax.set_title(f"QPS vs recall, d{c.dim}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return [_figure(path, fig, c.prov)]


def fig_batch_scaling(c) -> list[Path]:
    ds = c.batch_dataset or _batch_dataset(c)
    path = c.out / "figures" / "fig-batch-scaling.png"
    rows = _sel(c.rows, dataset=ds, dim=c.dim, perf_k=c.k, perf_mode=c.mode, backend=c.backend)
    bss = sorted({r["perf_bs"] for r in rows if r.get("perf_bs") is not None})
    if not bss:
        return [_empty(path, c.prov, f"no perf rows for {ds} d{c.dim}")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for algo in [a for a in ALGO_LABEL if any(r["algo"] == a for r in rows)]:
        xs, ys, err = [], [], []
        for bs in bss:
            v = _reduce(_sel(rows, algo=algo, perf_bs=bs), "perf_median_ms", [], "")
            if v:
                xs.append(bs)
                ys.append(v["value"] / bs)
                err.append(v["value"] / bs * v["spread"])
        if xs:
            ax.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=ALGO_LABEL[algo])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("batch size $B$")
    ax.set_ylabel("amortised ms / query")
    ax.set_title(f"Batch scaling, {DATASET_LABEL.get(ds, ds)} d{c.dim}, K={c.k}, {c.mode}"
                 "\n(error bars = window spread; $B=1$ is noise-dominated on this box)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return [_figure(path, fig, c.prov)]


def fig_deep_sweep(c) -> list[Path]:
    """Quality and latency against the swept parameter, whiskers = seed min--max."""
    written = []
    rows = _sel(c.rows, dim=c.dim, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode, backend=c.backend)
    groups = defaultdict(list)
    for r in rows:
        for name, value in json.loads(r.get("params") or "{}").items():
            groups[(r["dataset"], r["sweep"], r["algo"], name)].append((value, r))
    swept = {g: v for g, v in groups.items() if len({x for x, _ in v}) > 1}
    if not swept:
        return [_empty(c.out / "figures" / "fig-deep-sweep.png", c.prov,
                       "no parameter is swept over more than one value in these records")]
    for (ds, sw, algo, name), items in sorted(swept.items()):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax2 = ax.twinx()
        xs = sorted({x for x, _ in items})
        handles = []
        for axis, field, colour, marker, style, lbl in (
            (ax, f"oracle_recall@{c.k}", "tab:blue", "o", "-", f"Recall@{c.k}"),
            (ax2, "perf_median_ms", "tab:red", "s", "--", f"$t_{{med}}$ (ms), {c.mode}"),
        ):
            ys, lo, hi = [], [], []
            for x in xs:
                v = _reduce([r for val, r in items if val == x], field, [], "")
                ys.append(None if v is None else v["value"])
                lo.append(None if v is None else v["value"] - v["lo"])
                hi.append(None if v is None else v["hi"] - v["value"])
            if any(y is not None for y in ys):
                handles.append(axis.errorbar(xs, ys, yerr=[lo, hi], marker=marker, capsize=3,
                                             color=colour, linestyle=style, label=lbl))
            axis.set_ylabel(lbl, color=colour)
            axis.tick_params(axis="y", labelcolor=colour)
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set_xlabel(name)
        n_seeds = len({r.get("seed") for _, r in items})
        ax.set_title(f"{ALGO_LABEL.get(algo, algo)} on {DATASET_LABEL.get(ds, ds)} {sw}: "
                     f"{name} sweep\n(whiskers = min--max over {n_seeds} seed(s))")
        ax.grid(alpha=0.3)
        written.append(_figure(
            c.out / "figures" / f"fig-deep-sweep-{ds}-{sw}-{algo}-{name}.png", fig, c.prov))
    return written


def fig_latency_violin(c) -> list[Path]:
    path = c.out / "figures" / "fig-latency-violin.png"
    sel = [s for s in c.samples
           if s.get("k") == c.k and s.get("bs") == c.bs and s.get("mode") == c.mode
           and (c.dim is None or s.get("dim") == c.dim) and s.get("ms")]
    if not sel:
        return [_empty(path, c.prov,
                       f"no samples sidecar rows at k={c.k} bs={c.bs} mode={c.mode}")]
    sel = sorted(sel, key=lambda s: (s["dataset"], s["algo"], s["backend"],
                                     json.dumps(s.get("params"), sort_keys=True)))
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(sel)), 4.4))
    ax.violinplot([s["ms"] for s in sel], showmedians=True, widths=0.85)
    ax.set_xticks(range(1, len(sel) + 1))
    ax.set_xticklabels(
        [f"{s['dataset'][:4]}\n{ALGO_LABEL.get(s['algo'], s['algo'])}\n{s['backend']}"
         + (f"\n{json.dumps(s['params'])}" if s.get("params") else "")
         for s in sel], fontsize=6)
    ax.set_ylabel("per-call latency (ms)")
    ax.set_yscale("log")
    ax.set_title(f"Per-call latency distribution, K={c.k}, $B={c.bs}$, {c.mode} "
                 "(the chosen timing window's per-call vector)")
    ax.grid(alpha=0.3, axis="y")
    return [_figure(path, fig, c.prov)]


# ----- methodology and coverage -------------------------------------------------


def methodology(c) -> list[Path]:
    """The §"Методология замеров" paragraph, with the constants read out of the code that
    produced the records — so text and code cannot drift apart again (H WP-6)."""
    from bench import inputs, measure, run  # noqa: PLC0415 — torch only when reporting

    lat = inspect.signature(measure.latency).parameters
    pool = inspect.signature(inputs.query_pool).parameters["n_pool"].default
    recs = c.recs
    env = recs[0]["env"] if recs else {}
    ks = sorted({k for r in recs for k in (r.get("ks") or [])})
    bss = sorted({b for r in recs for b in (r.get("batch_sizes") or [])})
    seeds = sorted({r.get("seed") for r in recs})
    body = [
        f"\\item \\textbf{{Стенд.}} {env.get('gpu', '?')}, драйвер {env.get('driver', '?')}, "
        f"CUDA {env.get('cuda', '?')}, PyTorch {env.get('torch', '?')}, "
        f"Triton {env.get('triton', '?')}, Python {env.get('python', '?')}. "
        "Частоты SM \\textbf{нельзя зафиксировать} в этом контейнере "
        "(\\texttt{nvidia-smi -lgc} запрещён), поэтому каждая запись хранит выборку "
        "частоты под нагрузкой (\\texttt{perf[].sm\\_mhz}), и все сравнения задержек "
        "делаются при одинаковом оценщике частоты.",
        "\\item \\textbf{Изоляция.} Один процесс на "
        "$(\\mathrm{dataset}, \\mathrm{dim}, \\mathrm{algo}, \\mathrm{backend})$: "
        "ни кэш \\texttt{torch.compile}, ни пул CUDA-графов, ни аллокатор не переживают "
        "смену бэкенда.",
        f"\\item \\textbf{{Прогрев и замер.}} {lat['warmup'].default} прогревочных вызовов, "
        f"затем {lat['windows'].default} окна по "
        f"$N = \\mathrm{{clamp}}({lat['target_s'].default:g}\\,\\text{{с}} / "
        f"\\tilde t, {lat['n_min'].default}, {lat['n_max'].default})$ вызовов; каждый вызов "
        "обрамлён CUDA-событиями, окно — одним \\texttt{synchronize} в конце. Запросы "
        f"берутся из фиксированного пула в {pool} батчей по кругу, без сброса L2. "
        "Отчёт ведётся по окну с медианным медианом: median / mean / p95 / p99 / min / IQR, "
        "QPS (closed loop), \\texttt{host\\_gap}, разброс окон "
        "$(\\max-\\min)/\\mathrm{med}$; разброс выше 5\\,\\% помечает ячейку "
        "\\texttt{unstable}.",
        f"\\item \\textbf{{Режимы.}} {', '.join(run.MODES)}: \\texttt{{eager}} — число, "
        "сопоставимое с обеими статьями, \\texttt{graph} "
        "(\\texttt{torch.compile(mode=\"reduce-overhead\", dynamic=False, fullgraph=True)}) "
        "— приводится рядом, никогда вместо. Захват графа проверяется: "
        "\\texttt{cudagraph\\_skips} обязан быть 0, иначе ячейка — ошибка, а не число.",
        f"\\item \\textbf{{Качество.}} Один проход в режиме eager при "
        f"$k_{{\\max}} = \\max(K)$, чанками по {run.QUALITY_CHUNK} запросов; метрики "
        f"recall / ndcg / precision / mrr при $K \\in \\{{{', '.join(map(str, ks)) or '?'}\\}}$ "
        "накапливаются на устройстве. Эталон: точный фильтрованный fp32-оракул на "
        "фильтрующих ячейках и отложенные взаимодействия всегда.",
        "\\item \\textbf{Память.} \\texttt{index\\_mib} — сумма всех буферов модуля поиска "
        "вместе с подмодулем фильтра; \\texttt{peak\\_fwd\\_mib} — "
        "\\texttt{max\\_memory\\_allocated} за первое eager-окно.",
        f"\\item \\textbf{{Повторы.}} Размеры батча "
        f"$B \\in \\{{{', '.join(map(str, bss)) or '?'}\\}}$, "
        f"сиды $\\{{{', '.join(str(s) for s in seeds) or '?'}\\}}$; по сидам приводится "
        "медиана с усами min--max.",
    ]
    tex = (
        _banner(c.prov, c.results_dir)
        + "% Для \\section{Методология замеров скорости и памяти} (docs/thesis/main.tex).\n"
        + "\\begin{itemize}\n" + "\n".join("    " + b for b in body) + "\n\\end{itemize}\n"
    )
    return [_write(c.out / "methodology.tex", tex)]


def coverage(c) -> list[Path]:
    """``report.md`` — the view a person reviewing a run reads."""
    p = c.prov
    lines = [
        "# `bench report`",
        "",
        f"- generated: `{p['generated']}`  from `{c.results_dir}`",
        f"- records: **{p['n_records']}** {p['status']}, cells flagged unstable "
        f"(window spread or clock drift): {p['unstable']}",
        f"- schema_version: {p['schema_versions']}  |  code_version: {p['code_versions']}",
        f"- runs (commit, branch): {p['runs']}",
        f"- gpu: {p['gpus']}  host: {p['hosts']}  window: {p['started'][0]} .. "
        f"{p['started'][1]}",
        "",
        "## Citability (CLAUDE.md rule 2)",
        "",
    ]
    if p["citable"]:
        lines += [f"**CITABLE** — the caller declared gate `{p['gate']}` green.", ""]
    else:
        lines += ["**NOT CITABLE.** Every artifact carries the marker. Reasons:", ""]
        lines += [f"- {b}" for b in p["blockers"]]
        lines += ["", "These records predate the D1 campaign; they come from C4's gate run "
                  "and C5's one-cell check and are evidence about the harness, not results.", ""]
    lines += [
        "## Selection used by the tables",
        "",
        f"`--dim {c.dim}` `--k {c.k}` `--bs {c.bs}` `--mode {c.mode}` `--backend {c.backend}` "
        f"`--compare-bs {c.compare_bs}` `--budget-ms {list(c.budgets)}`; "
        f"batch-scaling dataset: `{c.batch_dataset or _batch_dataset(c)}`"
        f"{'' if c.batch_dataset else ' (chosen by coverage)'}.",
        "",
        "## Coverage",
        "",
        "| dataset | dim | suite | filter | sweep | algo | backend | params | seeds | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = defaultdict(lambda: [set(), Counter()])
    for r in c.recs:
        key = (r["dataset"], r["dim"], r["suite"], r["filter_kind"], r["sweep"], r["algo"],
               r["backend"], json.dumps(r["params"], sort_keys=True))
        groups[key][0].add(r["seed"])
        groups[key][1][r.get("status", "ok")] += 1
    for key in sorted(groups, key=str):
        seeds, status = groups[key]
        lines.append("| " + " | ".join(
            [*(str(x) for x in key), str(sorted(seeds)), str(dict(status))]) + " |")
    failed = [r for r in c.recs if r.get("status") == "failed"]
    partial = [r for r in c.recs if r.get("status") == "partial"]
    lines += ["", "## Failed cells (excluded from every number)", ""]
    lines += [f"- `{ {k: r[k] for k in records.KEY_FIELDS} }` — stage `{r.get('stage')}`: "
              f"{(r.get('error') or '').splitlines()[-1][:160] if r.get('error') else '?'}"
              for r in failed] or ["none"]
    lines += ["", "## Partial records (marked `*`)", ""]
    lines += [f"- `{ {k: r[k] for k in records.KEY_FIELDS} }` — {r.get('partial_reasons')}"
              for r in partial] or ["none"]
    unstable = [(r, e) for r in c.recs for e in (r.get("perf") or []) if e.get("unstable")]
    lines += ["", f"## Unstable perf variants (marked `†`): {len(unstable)}", ""]
    lines += [f"- {r['dataset']} {r['algo']}/{r['backend']} k={e['k']} bs={e['bs']} "
              f"{e['mode']}: spread {e['spread'] * 100:.1f}%" for r, e in unstable[:40]] or ["none"]
    if len(unstable) > 40:
        lines.append(f"- ... {len(unstable) - 40} more")
    lines += ["", "## Artifacts", ""]
    lines += [f"- `{q.relative_to(c.out)}`" for q in sorted(c.written)]
    lines += ["", "## Clock estimator", "",
              "Latency artifacts use the per-variant under-load `perf[].sm_mhz`. "
              "Idle samples (`env.sm_mhz_idle`, and the schema-1 `env.sm_mhz`, which is a "
              "whole-run median dominated by idle) are provenance only and are never "
              "compared with an under-load sample — the error that made 92 of 99 of C4's "
              "latency rows appear to fail.", ""]
    return [_write(c.out / "report.md", "\n".join(lines) + "\n")]


ARTIFACTS = {
    "recall_nofilter": tab_recall_nofilter,
    "pareto": tab_pareto,
    "batch_scaling": tab_batch_scaling,
    "memory": tab_memory,
    "parity": tab_backend_parity,
    "recall_at_budget": tab_recall_at_budget,
    "paper_comparison": tab_paper_comparison,
    "fig_pareto": fig_pareto,
    "fig_qps_recall": fig_qps_recall,
    "fig_batch_scaling": fig_batch_scaling,
    "fig_deep_sweep": fig_deep_sweep,
    "fig_latency_violin": fig_latency_violin,
    "methodology": methodology,
}


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Ctx:
    """Everything every artifact needs: the flat rows, the nested records, the samples, the
    provenance verdict and the one set of selectors."""

    def __init__(self, results_dir: Path, out: Path, gate: str | None, **sel: Any) -> None:
        self.results_dir, self.out = Path(results_dir), Path(out)
        self.flat, self.rows = _load(self.results_dir, self.out)
        self.recs = _latest(self.results_dir)
        self.samples = _samples(self.results_dir)
        self.prov = provenance(self.recs, gate)
        self.written: list[Path] = [self.flat]
        for k, v in sel.items():
            setattr(self, k, v)


def generate(results_dir: Path, out: Path, *, gate: str | None = None,
             only: tuple[str, ...] = (), **sel: Any) -> Ctx:
    c = Ctx(results_dir, out, gate, **sel)
    for name in only or tuple(ARTIFACTS):
        c.written += ARTIFACTS[name](c)
    c.written += coverage(c)
    return c


@click.command()
@click.argument("results", default="results")
@click.option("--out", default=None, help="output directory [default: <results>/report]")
@click.option("--gate", default=None,
              help="the roadmap step whose gate is green for these records (e.g. D1). Without "
                   "it every artifact is marked NOT CITABLE; it cannot override evidence.")
@click.option("--only", multiple=True, type=click.Choice(sorted(ARTIFACTS)),
              help="emit only these artifacts (report.md is always written)")
@click.option("--dim", default=128, show_default=True, type=int)
@click.option("--k", default=100, show_default=True, type=int)
@click.option("--bs", default=1, show_default=True, type=int, help="batch size for the tables")
@click.option("--compare-bs", default=16, show_default=True, type=int,
              help="batch size of the paper comparison table (H §2.7 compares at B=16)")
@click.option("--mode", default="eager", type=click.Choice(("eager", "graph")), show_default=True)
@click.option("--backend", default="triton", show_default=True)
@click.option("--sweep", default=None, help="narrow the batch-scaling table to one condition")
@click.option("--batch-dataset", default=None,
              help="dataset of tab:batch_scaling [default: the best-covered one]")
@click.option("--budget-ms", "budgets", multiple=True, type=float,
              default=(0.5, 1.0, 2.0, 5.0), show_default=True)
def report(results, out, gate, only, dim, k, bs, compare_bs, mode, backend, sweep,
           batch_dataset, budgets) -> None:
    """Thesis and paper tables and figures from the records (roadmap D4, H §6 WP-6)."""
    results_dir = Path(results)
    if not results_dir.is_dir():
        raise click.ClickException(f"{results_dir}: not a directory")
    c = generate(
        results_dir, Path(out) if out else results_dir / "report", gate=gate, only=only,
        dim=dim, k=k, bs=bs, compare_bs=compare_bs, mode=mode, backend=backend, sweep=sweep,
        batch_dataset=batch_dataset, budgets=tuple(budgets),
    )
    for path in c.written:
        click.echo(str(path))
    click.echo(f"{len(c.written)} artifacts from {c.prov['n_records']} records — "
               + ("CITABLE" if c.prov["citable"] else "NOT CITABLE: " + "; ".join(
                   c.prov["blockers"])))


__all__ = ["ARTIFACTS", "generate", "provenance", "report"]
