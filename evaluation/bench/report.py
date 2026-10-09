"""``bench report``: every paper table and figure, from the records only
(docs/system/evaluation.md § Report).

``records.aggregate`` writes ``results.parquet`` first and every table and figure is built
from it; ``records.latest`` is read a second time for the provenance block alone,
which needs the nested ``env`` the table flattens to a few columns.

**Nothing emitted here is citable unless ``--gate`` names a green roadmap gate** (CLAUDE.md
rule 2). Without it — or with a `failed` cell, a `partial` record, a `dirty` library subtree
or a record taken on a `dev/*` branch — every artifact carries a visible NOT CITABLE marker
in its own banner and in its caption. ``--gate`` cannot override the evidence, only the
default.

Failed cells never enter a number and are listed in ``report.md``; `partial` records are
marked ``*`` and `unstable` perf entries ``†``, both counted in the caption. Latency
artifacts state which clock estimator they used: the per-variant under-load
``perf[].sm_mhz``, never an idle or whole-run median.
"""

from __future__ import annotations

import datetime as dt
import functools
import inspect
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np
import yaml

from bench import inputs, measure, records, run, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGO_LABEL = {
    "linr_v1_filter_mask": "LiNR V1",
    "linr_v2": "LiNR V2",
    "linr_v3": "LiNR V3",
    "silvertorch": "SilverTorch",
    "postfilter": "postfilter",
}
PARAM_LABEL = {"alpha": "$\\alpha$"}
# The baseline is torch by definition (decisions.md § Harness): it passes any --backend.
FIXED_BACKEND = {"postfilter": "torch"}
DATASET_LABEL = {
    "goodreads": "Goodreads",
    "arxiv": "arXiv",
    "yfcc10m": "YFCC-10M",
    "pubmed": "PubMed",
    "goodreads-synth": "Goodreads (synthetic filter)",
    "arxiv-synth": "arXiv (synthetic filter)",
    "yfcc10m-synth": "YFCC-10M (synthetic filter)",
}
SPEEDUP_BASE = "linr_v1_filter_mask"

# ----- loading ----------------------------------------------------------------


def _load(results_dir: Path, out: Path) -> tuple[Path, list[dict[str, Any]]]:
    out.mkdir(parents=True, exist_ok=True)  # records.aggregate writes, it does not create
    table = records.aggregate(results_dir, out / "results.parquet")
    return table, records.read_table(table)


# ----- campaign manifest (docs/system/evaluation.md § Campaign manifest) -------------------

MATCH_REQUIRED = ("dataset", "suite", "algo", "backend")
MATCH_FIELDS = (*MATCH_REQUIRED, "filter_kind", "sweep")
# Record fields that come from the quality pass, taken from the quality-accepted record.
QUALITY_FIELDS = (
    "quality", "per_query", "pass_rate", "bloom_fp_rate", "n_queries_oracle",
    "n_queries_heldout", "n_targets_in_filter",
)  # fmt: skip


def load_manifest(path: Path) -> dict[str, Any]:
    m = yaml.safe_load(Path(path).read_text())
    for e in m.get("entries") or []:
        bad = set(e["match"]) - set(MATCH_FIELDS)
        lacks = set(MATCH_REQUIRED) - set(e["match"])
        if bad or lacks or set(e) - {"match", "quality", "perf"}:
            raise click.ClickException(f"{path}: bad entry {e} (match on {MATCH_FIELDS})")
    for side in ("quality", "perf"):
        for e in [m.get("default"), *(m.get("entries") or [])]:
            if e is not None and "code_version" not in (e.get(side) or {}):
                raise click.ClickException(f"{path}: {side} needs a code_version in {e}")
    return m


def _entry(m: dict[str, Any], rec: dict[str, Any]) -> dict[str, Any] | None:
    """The manifest entry of one cell: the most specific matching entry, else ``default``."""
    hits = [e for e in m.get("entries") or [] if all(rec[f] == v for f, v in e["match"].items())]
    if not hits:
        return m.get("default")
    best = max(len(e["match"]) for e in hits)
    top = [e for e in hits if len(e["match"]) == best]
    if len(top) > 1:
        raise click.ClickException(f"manifest: {len(top)} entries match {records.key_block(rec)}")
    return top[0]


def select(
    recs: list[dict[str, Any]], m: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict]:
    """Per cell (key block), quality from the record at the entry's quality ``code_version`` and
    perf from the record at its perf ``code_version``; nothing from any other code_version. A
    cell with no entry is excluded, one with no record at an accepted version is missing.
    Returns the merged records, every source record accepted (what provenance judges), and
    the excluded / missing cells."""
    cells: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in recs:
        key = json.dumps(records.key_block(r), sort_keys=True)
        cells[key][r["env"]["code_version"]] = r
    out, used, why = [], [], {"no entry": [], "quality missing": [], "perf missing": []}
    for key, by_cv in sorted(cells.items()):
        e = _entry(m, next(iter(by_cv.values())))
        if e is None:
            why["no entry"].append(key)
            continue
        q, p = by_cv.get(e["quality"]["code_version"]), by_cv.get(e["perf"]["code_version"])
        if q is None:
            why["quality missing"].append(key)
        if p is None:
            why["perf missing"].append(key)
        if q is None and p is None:
            continue
        used += [q] if q is p else [r for r in (q, p) if r is not None]
        rec = dict(p or q)
        if p is None:
            rec["perf"] = None
        elif q is None:
            rec |= dict.fromkeys(QUALITY_FIELDS)
        elif q is not p:
            rec |= {f: q.get(f) for f in QUALITY_FIELDS}
            rec["quality_source"] = {"seed": q["seed"], "code_version": q["env"]["code_version"]}
        out.append(rec)
    return out, used, why


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


# rule 2: numbers from a branch are not paper material
MAIN_BRANCHES = ("staging", "development", "main")


def provenance(recs: list[dict[str, Any]], gate: str | None) -> dict[str, Any]:
    """The rule-2 verdict over a set of records. Shared with ``bench upload``, which puts
    it in every published ``MANIFEST.json`` so a Hub copy cannot claim more than the
    tables would."""
    env = [r.get("env") or {} for r in recs]
    status = Counter(r.get("status", "ok") for r in recs)
    dirty = [r for r, e in zip(recs, env, strict=True) if e.get("dirty")]
    why = Counter(
        x for r in recs if r.get("status") == "partial" for x in r.get("partial_reasons") or ["?"]
    )
    why_s = ", ".join(f"{x} {n}" for x, n in sorted(why.items()))
    found: list[tuple[str, str]] = []  # (blocker for report.md / the manifest, caption mark)

    def add(blocker: str, mark: str) -> None:
        found.append((blocker, mark))

    if not gate:
        add(
            "no --gate given: no roadmap gate is declared green for these records", "gate not green"
        )
    if status["failed"]:
        add(f"{status['failed']} record(s) with status=failed", f"{status['failed']} failed")
    if status["partial"]:
        add(
            f"{status['partial']} record(s) with status=partial (partial_reasons: {why_s})",
            f"{status['partial']} partial: {why_s}",
        )
    if dirty:
        add(
            f"{len(dirty)} record(s) with env.dirty (library subtree was dirty)",
            f"{len(dirty)} dirty",
        )
    off = sorted(
        {
            e["git_branch"]
            for e in env
            if e.get("git_branch") and e["git_branch"] not in MAIN_BRANCHES
        }
    )
    if off:
        add(
            f"record(s) produced on {', '.join(off)} — CLAUDE.md rule 2: "
            "harness numbers from a branch are not paper material",
            f"branch {', '.join(off)}",
        )
    if not recs:
        add("no records", "no records")
    blockers = [b for b, _ in found]
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
        "marks": [m for _, m in found],
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
    return "\n".join(lines) + "\n"


def _caption(prov: dict[str, Any], text: str) -> str:
    if prov["citable"]:
        return text
    return f"\\textbf{{[NOT CITABLE: {_esc('; '.join(prov['marks']))}]}}~{text}"


# ----- selection and reduction -------------------------------------------------


def _sel(rows: list[dict[str, Any]], **eq: Any) -> list[dict[str, Any]]:
    keep = [r for r in rows if r.get("status") != "failed"]
    for field, want in eq.items():
        if want is None:
            continue
        if field == "backend":
            keep = [r for r in keep if r[field] == want or r["algo"] in FIXED_BACKEND]
        else:
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
        notes.append(
            f"{_esc(what)}: {len(params)} parameter sets present "
            f"{_esc(', '.join(params))}; showing {_esc(params[0])}"
        )
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
        "unstable": field.startswith("perf_") and any(r.get("perf_unstable") for r in chosen),
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


def _attach(rows: list[dict[str, Any]], recs: list[dict[str, Any]]) -> None:
    """Join each flat row to its nested record (``_rec``) and perf entry (``_entry``), where
    the statistics read the windows, the sidecar path and the interleave block (the parquet
    carries the same values as ``perf_window_medians_ms``, ``per_query`` and
    ``interleave_*``)."""
    by_key = {records.record_key(r): r for r in recs}
    for row in rows:
        key = {f: json.loads(row[f]) if f == "params" else row[f] for f in records.KEY_FIELDS}
        row["_rec"] = rec = by_key[records.resume_key(key, row["env_code_version"])]
        row["_params"] = key["params"]
        want = (row.get("perf_k"), row.get("perf_bs"), row.get("perf_mode"))
        row["_entry"] = next(
            (e for e in rec.get("perf") or [] if (e["k"], e["bs"], e["mode"]) == want), None
        )


def _windows(rows: list[dict[str, Any]]) -> list[float]:
    return [
        w
        for r in rows
        if r["_entry"] and r["_entry"].get("median_ms") is not None
        for w in r["_entry"]["window_medians_ms"]
    ]


def _marks(rows: list[dict[str, Any]], perf: bool) -> dict[str, Any]:
    return {
        "n_seeds": len({r.get("seed") for r in rows}),
        "unstable": perf and any(r.get("perf_unstable") for r in rows),
        "partial": any(r.get("status") == "partial" for r in rows),
    }


def _lat(rows: list[dict[str, Any]]) -> dict | None:
    """One arm's latency at one ``(bs, k, mode)``: the median of its per-window medians over
    seed x window, with the bootstrap CI (§ Statistics)."""
    rows = [r for r in rows if r["_entry"] and r["_entry"].get("median_ms") is not None]
    ci = stats.median_ci(_windows(rows))
    if ci is None:
        return None
    return {"value": ci[0], "lo": ci[1], "hi": ci[2], **_marks(rows, True)}


@functools.cache
def _sidecar(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return dict(z)


def _recall(c, rows: list[dict[str, Any]], k: int) -> dict | None:
    """One arm's ``recall_oracle@k``: the mean over queries of the per-query recall (averaged
    across the arm's distinct sidecars, i.e. its seeds when quality depends on the seed), with
    the bootstrap CI over queries. Without a sidecar (schema <= 3): the median across seeds,
    no CI."""
    rows = [r for r in rows if r.get(f"oracle_recall@{k}") is not None]
    if not rows:
        return None
    paths = sorted({r["_rec"]["per_query"] for r in rows if r["_rec"].get("per_query")})
    if not paths:
        v = statistics.median(
            statistics.median(float(r[f"oracle_recall@{k}"]) for r in rows if r["seed"] == s)
            for s in {r["seed"] for r in rows}
        )
        return {"value": v, "lo": None, "hi": None, **_marks(rows, False)}
    est, lo, hi = _query_ci(c.results_dir, tuple(paths), k)
    return {"value": est, "lo": lo, "hi": hi, **_marks(rows, False)}


@functools.cache
def _query_ci(root: Path, paths: tuple[str, ...], k: int) -> tuple[float, float, float]:
    cars = [_sidecar(root / p) for p in paths]
    if any(not np.array_equal(cars[0]["rows"], z["rows"]) for z in cars[1:]):
        raise click.ClickException(f"sidecars of one arm cover different queries: {paths}")
    per_query = np.stack([z[f"recall_oracle@{k}"] for z in cars]).mean(axis=0)
    return stats.mean_ci(per_query[~np.isnan(per_query)])


def _ratio(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> tuple[tuple | None, bool]:
    """Latency of arm A over arm B at one ``(bs, k, mode)``. Paired per round when every
    common seed ran both arms in one interleave group, else unpaired."""
    ra = {r["seed"]: r for r in a if r["_entry"] and r["_entry"].get("median_ms") is not None}
    rb = {r["seed"]: r for r in b if r["_entry"] and r["_entry"].get("median_ms") is not None}
    seeds = sorted(ra.keys() & rb.keys())
    groups = [
        ((ra[s]["_rec"].get("interleave") or {}).get("group"),
         (rb[s]["_rec"].get("interleave") or {}).get("group"))
        for s in seeds
    ]  # fmt: skip
    if seeds and all(g is not None and g == h for g, h in groups):
        return stats.paired_ratio_ci(
            _windows([ra[s] for s in seeds]), _windows([rb[s] for s in seeds])
        ), True
    return stats.ratio_ci(_windows(list(ra.values())), _windows(list(rb.values()))), False


def _ratio_tex(ci: tuple | None, paired: bool) -> str:
    if ci is None:
        return "---"
    body = f"{ci[0]:.2f}\\times\\,[{ci[1]:.2f}, {ci[2]:.2f}]" + ("" if paired else "^{u}")
    return f"${body}$" if stats.differs(ci) else f"no difference (${body}$)"


def _ci_tex(v: dict | None, places: int) -> str:
    if v is None or v["lo"] is None:
        return ""
    return f"\\,{{\\tiny$[{v['lo']:.{places}f}, {v['hi']:.{places}f}]$}}"


def _arm(algo: str, params: str | dict, tex: bool = True) -> str:
    """One row label per parameter set: ``postfilter ($\\alpha$=1)``. Rows are never averaged
    across parameters (the alpha rule, docs/system/evaluation.md § Report)."""
    p = json.loads(params) if isinstance(params, str) else params
    bits = [
        f"{PARAM_LABEL.get(n, _esc(n) if tex else n)}={v}" for n, v in sorted((p or {}).items())
    ]
    return ALGO_LABEL.get(algo, algo) + (f" ({', '.join(bits)})" if bits else "")


def _esc(text: Any) -> str:
    """LaTeX-escape a string that came out of the records (sweep names, parameter JSON)."""
    out = str(text)
    for ch, rep in (
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("#", "\\#"),
        ("%", "\\%"),
        ("&", "\\&"),
        ("$", "\\$"),
        ("~", "$\\sim$"),
    ):
        out = out.replace(ch, rep)
    return out


def _f(x: float | None, places: int = 4) -> str:
    return "---" if x is None else f"${x:.{places}f}$"


def _sci(x: float | None) -> str:
    if x is None:
        return "---"
    if x == 0:
        return "$0$"
    e = int(f"{x:e}".split("e")[1])
    return f"${x / 10**e:.2f}{{\\times}}10^{{{e}}}$"


def _mark(v: dict[str, Any]) -> str:
    return ("^{\\dagger}" if v["unstable"] else "") + ("^{*}" if v["partial"] else "")


def _num(v: dict[str, Any] | None, places: int = 4) -> str:
    if v is None:
        return "---"
    return f"${v['value']:.{places}f}{_mark(v)}$"


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
        note_tex = (
            "\n    \\vspace{2pt}\\par\n    \\begin{minipage}{\\linewidth}\n"
            + inner
            + "\n    \\end{minipage}"
        )
    return (
        _banner(prov, results_dir) + "\\begin{table}[htbp]\n"
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


STATS_NOTE = (
    "Latency: median of the per-window medians over seed $\\times$ window; recall: mean "
    "over queries of the per-query \\texttt{recall\\_oracle} (the exact filtered oracle). "
    f"Brackets: 95\\,\\% percentile bootstrap CI, $B={stats.B}$. A ratio is paired per round "
    "when both arms ran interleaved ($^{u}$: unpaired); a ratio CI containing 1 prints "
    "``no difference''."
)


def _legend(notes: list[str]) -> list[str]:
    return [
        "$^{\\dagger}$ the cell's timing windows spread more than 5\\,\\% "
        "(\\texttt{unstable}); $^{*}$ built from a \\texttt{partial} record. "
        "Cells with \\texttt{status: failed} are excluded and listed in \\texttt{report.md}.",
        *dict.fromkeys(notes),
    ]


# ----- artifacts ---------------------------------------------------------------


def tab_pareto(c) -> list[Path]:
    written = []
    for ds in sorted({r["dataset"] for r in c.rows}):
        rows = _sel(c.rows, dataset=ds, dim=c.dim, backend=c.backend)
        perf = _sel(rows, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode)
        conds = sorted({(r["filter_kind"], r["sweep"]) for r in rows if r["filter_kind"] != "none"})
        notes, body = [], []
        for fk, sw in conds:
            base = _sel(perf, filter_kind=fk, sweep=sw, algo=SPEEDUP_BASE)
            arms = sorted(
                {
                    (r["algo"], r["params"])
                    for r in rows
                    if (r["filter_kind"], r["sweep"]) == (fk, sw) and r["algo"] in ALGO_LABEL
                },
                key=lambda ap: (list(ALGO_LABEL).index(ap[0]), ap[1]),
            )
            for i, (a, pj) in enumerate(arms):
                arm = _sel(rows, filter_kind=fk, sweep=sw, algo=a, params=pj)
                arm_perf = _sel(perf, filter_kind=fk, sweep=sw, algo=a, params=pj)
                q = _recall(c, _cells(arm), c.k)
                t = _lat(arm_perf)
                m = _reduce(_cells(arm), "index_mib", notes, f"{ds}/{sw}/{a}")
                body.append(
                    [
                        f"\\textbf{{{_esc(sw)}}} ({fk})" if i == 0 else "",
                        _arm(a, pj),
                        _num(q) + _ci_tex(q, 4),
                        _num(t, 3) + _ci_tex(t, 3),
                        "---" if a == SPEEDUP_BASE else _ratio_tex(*_ratio(base, arm_perf)),
                        _num(m, 1),
                    ]
                )
        tex = _table(
            c.prov, c.results_dir,
            caption=f"Quality, latency and memory on {DATASET_LABEL.get(ds, ds)} "
                    f"($d={c.dim}$, $K={c.k}$, $B={c.bs}$, mode \\texttt{{{c.mode}}}, "
                    f"backend \\texttt{{{c.backend}}}).",
            label=f"tab:pareto_{ds}",
            colspec="llcccc",
            header=["Sweep", "Arm", f"Recall@{c.k}", "$t_\\mathrm{med}$ (ms)",
                    "Speedup", "Memory (MiB)"],
            body=body,
            notes=_legend(notes + [
                f"Speedup = latency of {ALGO_LABEL[SPEEDUP_BASE]} on the same sweep over the "
                "arm's ('---' when either is absent). " + STATS_NOTE,
                _clock_note(perf),
            ]),
        )  # fmt: skip
        written.append(_write(c.out / "tables" / f"tab-pareto_{ds}.tex", tex))
    return written


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
        caption=f"Index size (MiB) at $d={c.dim}$.",
        label="tab:memory",
        colspec="l" + "c" * len(algos),
        header=["Dataset"] + [ALGO_LABEL[a] for a in algos],
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
        eager, graph = _sel(sub, perf_mode="eager"), _sel(sub, perf_mode="graph")
        jac = cell.get(f"quality_jaccard_vs_first@{c.k}")
        diff = cell.get("quality_score_max_abs_diff")
        body.append(
            [
                DATASET_LABEL.get(ds, ds),
                _esc(sw),
                ALGO_LABEL.get(algo, algo),
                f"\\texttt{{{_esc(be)}}}",
                f"\\texttt{{{_esc(cell.get('path'))}}}",
                _f(jac, 6),
                _sci(diff),
                _num(_lat(eager), 3),
                _num(_lat(graph), 3),
                _ratio_tex(*_ratio(eager, graph)),
            ]
        )
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Backend parity and eager-vs-graph latency ($d={c.dim}$, $K={c.k}$, "
                f"$B={c.bs}$).",
        label="tab:backend_parity",
        colspec="lllllccccc",
        header=["Dataset", "Sweep", "Algo", "Backend", "Path", f"jaccard@{c.k}",
                "$|\\Delta s|_{\\max}$", "eager (ms)", "graph (ms)", "eager/graph"],
        body=body,
        notes=_legend([
            "\\texttt{jaccard} and $|\\Delta s|_{\\max}$ are against the first backend of the "
            "same $(dataset, dim, algo, params, seed)$ group (the parity spill file); the "
            "reference row itself shows '---'. A \\texttt{graph} column of '---' is a "
            "non-capturable path (\\texttt{official} is eager-only). " + STATS_NOTE,
            _clock_note(rows),
        ]),
    )  # fmt: skip
    return [_write(c.out / "tables" / "tab-backend_parity.tex", tex)]


# ----- matched recall -----------------------------------------------------------

# The parameter a curve is swept along, per algo; every other param (n_lists, ...) splits curves.
CURVE_AXIS = {"silvertorch": ("n_probe",), "linr_v3": ("candidate_pool", "candidate_pool_frac")}
TARGETS = (0.90, 0.95)


def _timed(rows: list[dict[str, Any]], bs: int, k: int) -> tuple[list[dict[str, Any]], str]:
    """The rows of one ``(bs, k)``: ``graph`` (the headline mode) when it ran, else ``eager``
    (official is not capturable)."""
    for mode in ("graph", "eager"):
        sub = [
            r
            for r in rows
            if (r.get("perf_bs"), r.get("perf_k"), r.get("perf_mode")) == (bs, k, mode)
            and r["_entry"]
            and r["_entry"].get("median_ms") is not None
        ]
        if sub:
            return sub, mode
    return [], "-"


def curves(c, rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    """Every recall curve in ``rows``: one per (dataset, dim, suite, filter_kind, sweep, algo,
    backend, the params other than the axis), its points sorted by the axis value."""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        params = json.loads(r["params"])
        axis = next((a for a in CURVE_AXIS.get(r["algo"], ()) if a in params), None)
        if axis is None or r.get("status") == "failed":
            continue
        rest = {n: v for n, v in params.items() if n != axis}
        key = (r["dataset"], r["dim"], r["suite"], r["filter_kind"], r["sweep"], r["algo"],
               r["backend"], json.dumps(rest, sort_keys=True), axis)  # fmt: skip
        groups[key][params[axis]].append(r)
    out = []
    for key, by_x in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ds, dim, suite, fk, sw, algo, be, rest, axis = key
        pts = [(x, _recall(c, _cells(rs), k), rs) for x, rs in sorted(by_x.items())]
        pts = [(x, q, rs) for x, q, rs in pts if q is not None]
        reach = [x for x, q, _ in pts if q["value"] >= 0.95]
        out.append({
            "dataset": ds, "dim": dim, "suite": suite, "filter_kind": fk, "sweep": sw,
            "algo": algo, "backend": be, "rest": rest, "axis": axis, "points": pts,
            "n95": min(reach) if axis == "n_probe" and reach else None,
        })  # fmt: skip
    return out


def matched(curve: dict[str, Any], bs: int, k: int, target: float) -> dict[str, Any]:
    """Latency of one curve at ``recall_oracle@k == target`` at batch size ``bs``."""
    pts, modes = [], set()
    for x, q, rs in curve["points"]:
        sub, mode = _timed(rs, bs, k)
        t = _lat(sub)
        if t is not None:
            pts.append((q["value"], t["value"], f"{curve['axis']}={x}"))
            modes.add(mode)
    return {**stats.at_recall(pts, target), "mode": "/".join(sorted(modes)) or "-"}


def _curve_label(cv: dict[str, Any], tex: bool = True) -> str:
    be = "" if cv["algo"] in FIXED_BACKEND else f" [{cv['backend']}]"
    return (
        f"{_arm(cv['algo'], cv['rest'], tex)}{be} along {_esc(cv['axis']) if tex else cv['axis']}"
    )


def tab_matched(c) -> list[Path]:
    """Latency at matched recall (0.90, 0.95) per curve and batch size, the bracketing points
    named, and ``n95`` per curve (the value the campaign writes into suites.yaml)."""
    body, dump = [], []
    for cv in curves(c, c.rows, c.k):
        bss = sorted({r["perf_bs"] for _, _, rs in cv["points"] for r in rs if r.get("perf_bs")})
        for bs in bss:
            hits = {t: matched(cv, bs, c.k, t) for t in TARGETS}
            cells = []
            for t in TARGETS:
                h = hits[t]
                cells.append(
                    f"${h['latency']:.3f}$ ({_esc(' .. '.join(h['bracket']))})"
                    if h["latency"] is not None
                    else _esc(h["reason"])
                )
            body.append([
                f"{DATASET_LABEL.get(cv['dataset'], cv['dataset'])} d{cv['dim']}",
                f"{_esc(cv['sweep'])} ({cv['filter_kind']}, {_esc(cv['suite'])})",
                _curve_label(cv), f"${bs}$", hits[TARGETS[0]]["mode"], *cells,
                "---" if cv["n95"] is None else f"${cv['n95']}$",
            ])  # fmt: skip
            dump.append({
                **{f: cv[f] for f in ("dataset", "dim", "suite", "filter_kind", "sweep", "algo",
                                       "backend", "axis", "n95")},
                "params": json.loads(cv["rest"]), "bs": bs, "k": c.k,
                **{f"at_{t:.2f}": hits[t] for t in TARGETS},
            })  # fmt: skip
    tex = _table(
        c.prov, c.results_dir,
        caption=f"Latency (ms) at matched \\texttt{{recall\\_oracle@{c.k}}} and the smallest "
                "measured $n_\\mathrm{probe}$ reaching 0.95.",
        label="tab:matched_recall",
        colspec="lllcl" + "c" * len(TARGETS) + "c",
        header=["Dataset", "Sweep", "Curve", "$B$", "mode",
                *[f"@{t:.2f} (bracket)" for t in TARGETS], "$n_{95}$"],
        body=body,
        notes=_legend([
            "Piecewise-linear between the two adjacent measured points that bracket the "
            "target on the recall-sorted curve (named in brackets), never extrapolated. "
            "$n_{95}$: the smallest measured $n_\\mathrm{probe}$ with recall $\\geq 0.95$. "
            + STATS_NOTE,
        ]),
    )  # fmt: skip
    return [
        _write(c.out / "tables" / "tab-matched_recall.tex", tex),
        _write(c.out / "matched_recall.json", json.dumps(dump, indent=1) + "\n"),
    ]


# ----- figures -----------------------------------------------------------------


def _figure(path: Path, fig, prov: dict[str, Any]) -> Path:
    fig.tight_layout()
    if not prov["citable"]:
        fig.text(
            0.5,
            0.5,
            "NOT CITABLE",
            ha="center",
            va="center",
            fontsize=34,
            color="red",
            alpha=0.18,
            rotation=25,
            zorder=10,
        )
        fig.text(0.5, 0.3, "; ".join(prov["marks"]), ha="center", fontsize=9, color="red")
    fig.text(
        0.005,
        0.005,
        f"bench report {prov['generated']} | code_version "
        f"{','.join(prov['code_versions'])[:12]} | commit {','.join(prov['commits'])}",
        fontsize=5,
        color="0.35",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty(path: Path, prov, msg: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.axis("off")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, wrap=True)
    return _figure(path, fig, prov)


def fig_deep_sweep(c) -> list[Path]:
    """Quality and latency against each swept parameter, one curve per value of the others,
    whiskers = 95 % bootstrap CI."""
    written = []
    rows = _sel(c.rows, dim=c.dim, perf_k=c.k, perf_bs=c.bs, perf_mode=c.mode, backend=c.backend)
    groups = defaultdict(list)
    for r in rows:
        params = json.loads(r.get("params") or "{}")
        for name, value in params.items():
            rest = "".join(f"-{n}{v}" for n, v in sorted(params.items()) if n != name)
            key = (r["dataset"], r["filter_kind"], r["sweep"], r["algo"], name, rest)
            groups[key].append((value, r))
    swept = {g: v for g, v in groups.items() if len({x for x, _ in v}) > 1}
    if not swept:
        return [
            _empty(
                c.out / "figures" / "fig-deep-sweep.png",
                c.prov,
                "no parameter is swept over more than one value in these records",
            )
        ]
    for (ds, fk, sw, algo, name, rest), items in sorted(swept.items()):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        ax2 = ax.twinx()
        xs = sorted({x for x, _ in items})
        handles = []
        for axis, est, colour, marker, style, lbl in (
            (ax, lambda rs: _recall(c, _cells(rs), c.k), "tab:blue", "o", "-", f"Recall@{c.k}"),
            (ax2, _lat, "tab:red", "s", "--", f"$t_{{med}}$ (ms), {c.mode}"),
        ):
            vs = [est([r for val, r in items if val == x]) for x in xs]
            pts = [(x, v) for x, v in zip(xs, vs, strict=True) if v is not None]
            if pts:
                err = [
                    [0.0 if v["lo"] is None else v["value"] - v["lo"] for _, v in pts],
                    [0.0 if v["hi"] is None else v["hi"] - v["value"] for _, v in pts],
                ]
                handles.append(
                    axis.errorbar(
                        [x for x, _ in pts],
                        [v["value"] for _, v in pts],
                        yerr=err,
                        marker=marker,
                        capsize=3,
                        color=colour,
                        linestyle=style,
                        label=lbl,
                    )
                )
            axis.set_ylabel(lbl, color=colour)
            axis.tick_params(axis="y", labelcolor=colour)
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set_xlabel(name)
        n_seeds = len({r.get("seed") for _, r in items})
        ax.set_title(
            f"{ALGO_LABEL.get(algo, algo)} on {DATASET_LABEL.get(ds, ds)} {sw} ({fk}): "
            f"{name} sweep {rest.lstrip('-')}\n(whiskers = 95% bootstrap CI, {n_seeds} seed(s))"
        )
        ax.grid(alpha=0.3)
        written.append(
            _figure(
                c.out / "figures" / f"fig-deep-sweep-{ds}-{fk}-{sw}-{algo}-{name}{rest}.png",
                fig,
                c.prov,
            )
        )
    return written


def fig_latency_violin(c) -> list[Path]:
    path = c.out / "figures" / "fig-latency-violin.png"
    sel = [
        s
        for s in c.samples
        if s.get("k") == c.k
        and s.get("bs") == c.bs
        and s.get("mode") == c.mode
        and (c.dim is None or s.get("dim") == c.dim)
        and s.get("ms")
    ]
    if not sel:
        return [_empty(path, c.prov, f"no samples sidecar rows at k={c.k} bs={c.bs} mode={c.mode}")]
    sel = sorted(
        sel,
        key=lambda s: (
            s["dataset"],
            s["algo"],
            s["backend"],
            json.dumps(s.get("params"), sort_keys=True),
        ),
    )
    fig, ax = plt.subplots(figsize=(max(6.5, 0.8 * len(sel)), 4.4))
    ax.violinplot([s["ms"] for s in sel], showmedians=True, widths=0.85)
    ax.set_xticks(range(1, len(sel) + 1))
    ax.set_xticklabels(
        [
            f"{s['dataset'][:4]}\n{ALGO_LABEL.get(s['algo'], s['algo'])}\n{s['backend']}"
            + (f"\n{json.dumps(s['params'])}" if s.get("params") else "")
            for s in sel
        ],
        fontsize=6,
    )
    ax.set_ylabel("per-call latency (ms)")
    ax.set_yscale("log")
    ax.set_title(
        f"Per-call latency distribution, K={c.k}, $B={c.bs}$, {c.mode} "
        "(the chosen timing window's per-call vector)"
    )
    ax.grid(alpha=0.3, axis="y")
    return [_figure(path, fig, c.prov)]


# ----- paper exhibits (docs/system/evaluation.md § Paper exhibits) -----------------------

REAL = ("goodreads", "arxiv", "yfcc10m", "pubmed")
EXHIBIT_K = 100
EXHIBIT_BS = (1, 16)
CLAIMS = Path(__file__).resolve().parents[1] / "claims.yaml"


def _with(rows: list[dict[str, Any]], **params: Any) -> list[dict[str, Any]]:
    return [r for r in rows if all(r["_params"].get(n) == v for n, v in params.items())]


def _order(arm: tuple) -> tuple:
    return (list(ALGO_LABEL).index(arm[0]), arm[1:])


def _arms(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """The ``(algo, backend, params)`` arms of ``rows``, labelled algos only, in label order."""
    found = {(r["algo"], r["backend"], r["params"]) for r in rows if r["algo"] in ALGO_LABEL}
    return sorted(found, key=_order)


def _arm_be(algo: str, backend: str, params: str | dict, tex: bool = True) -> str:
    be = "" if algo in FIXED_BACKEND else f" [{backend}]"
    return _arm(algo, params, tex) + be


def _lat_tex(sub_mode: tuple[list[dict[str, Any]], str]) -> str:
    sub, mode = sub_mode
    v = _lat(sub)
    return _num(v, 3) + _ci_tex(v, 3) + ("$^{e}$" if v and mode == "eager" else "")


def _pass_rate(rows: list[dict[str, Any]]) -> float | None:
    v = [r["pass_rate"] for r in _cells(rows) if r.get("pass_rate") is not None]
    return statistics.median(v) if v else None


def _by_scale(datasets: set[str], rows: list[dict[str, Any]]) -> list[str]:
    n = {r["dataset"]: r.get("n_items") or 0 for r in rows}
    return sorted(datasets, key=lambda d: (n.get(d, 0), d))


EXHIBIT_NOTE = (
    "$^{e}$ eager: the arm has no \\texttt{graph} entry (the official ops are not "
    "capturable); every other latency is \\texttt{graph}, the headline mode."
)


def tab_t2(c) -> list[Path]:
    """T2: the real-filter headline, per dataset: every arm at the paper's operating point
    (SilverTorch at ``n_probe`` 24) and SilverTorch at matched recall 0.95, bs 1 and 16."""
    written = []
    for ds in [d for d in REAL if _sel(c.rows, dataset=d, suite="filter")]:
        rows = _sel(c.rows, dataset=ds, suite="filter")
        body = []
        for fk, sw in sorted({(r["filter_kind"], r["sweep"]) for r in rows}):
            cond = _sel(rows, filter_kind=fk, sweep=sw)
            lines = []
            for a, be, pj in _arms(cond):
                if json.loads(pj).get("n_probe", 24) != 24:
                    continue
                arm = _sel(cond, algo=a, backend=be, params=pj)
                q = _recall(c, _cells(arm), EXHIBIT_K)
                lat = [_lat_tex(_timed(arm, bs, EXHIBIT_K)) for bs in EXHIBIT_BS]
                lines.append([_arm_be(a, be, pj), _num(q) + _ci_tex(q, 4), *lat])
            for cv in curves(c, cond, EXHIBIT_K):
                if cv["algo"] != "silvertorch":
                    continue
                cells = []
                for bs in EXHIBIT_BS:
                    h = matched(cv, bs, EXHIBIT_K, 0.95)
                    cells.append(
                        f"${h['latency']:.3f}$ {{\\tiny({_esc(' .. '.join(h['bracket']))})}}"
                        if h["latency"] is not None
                        else _esc(h["reason"])
                    )
                label = f"{_arm_be('silvertorch', cv['backend'], cv['rest'])} @ 0.95"
                lines.append([label, "matched", *cells])
            body += [
                [f"\\textbf{{{_esc(sw)}}} ({fk})" if i == 0 else "", *ln]
                for i, ln in enumerate(lines)
            ]
        tex = _table(
            c.prov,
            c.results_dir,
            caption=f"Real filters on {DATASET_LABEL.get(ds, ds)}: recall and p50 latency "
            f"(ms) at the operating point ($n_\\mathrm{{probe}}=24$) and at matched "
            f"recall 0.95, $K={EXHIBIT_K}$.",
            label=f"tab:t2_{ds}",
            colspec="llccc",
            header=["Sweep", "Arm", f"Recall@{EXHIBIT_K}", "$B=1$", "$B=16$"],
            body=body,
            notes=_legend(
                [
                    "Matched rows interpolate latency at recall 0.95 along $n_\\mathrm{probe}$ "
                    "between the bracketing measured points (in brackets), never extrapolated.",
                    EXHIBIT_NOTE,
                    STATS_NOTE,
                    _clock_note(rows),
                ]
            ),
        )
        written.append(_write(c.out / "tables" / f"tab-t2_{ds}.tex", tex))
    if written:
        return written
    tex = _table(
        c.prov,
        c.results_dir,
        caption="Real filters: no \\texttt{filter} records.",
        label="tab:t2",
        colspec="llccc",
        header=[],
        body=[],
        notes=[],
    )
    return [_write(c.out / "tables" / "tab-t2.tex", tex)]


def _kernels(rows: list[dict[str, Any]]) -> tuple[float | None, int | None, bool]:
    """Kernel-only time (us) and launches of one eager call, median over seeds: the entry's
    ``kernels_us`` / ``kernels_calls`` (every kernel). Records profiled before those fields
    existed (H2H-FINAL at campaign-v2.1) fall back to the sum over the top-8 kernels, a lower
    bound, flagged by the third value so the table marks it."""
    es = [r["_entry"] for r in rows if r["_entry"]]
    full = [e for e in es if e.get("kernels_us") is not None]
    if full:
        return (
            statistics.median(e["kernels_us"] for e in full),
            int(statistics.median(e["kernels_calls"] for e in full)),
            False,
        )
    ks = [e["kernels"] for e in es if e.get("kernels")]
    if not ks:
        return None, None, False
    return (
        statistics.median(sum(x["us"] for x in k) for k in ks),
        int(statistics.median(sum(x["calls"] for x in k) for k in ks)),
        True,
    )


def _ids_identity(sub: list[dict[str, Any]], ref: list[dict[str, Any]]) -> str:
    """T3's ids column against the Triton eager arm, per common seed: ``=`` when the canonical
    ids hash is equal at every one; ``= (ties)`` when every seed that differs has bit-equal
    scores (the parity spill's ``|Δs|_max`` is 0 over every query at ``k_max``, so the sorted
    score lists agree at every k and only a tied id at the k-th cut can differ); else ``≠``."""
    ref_ids = {r["seed"]: r["_entry"].get("ids_sha256_canon") for r in ref if r["_entry"]}
    pairs = [
        (r["_entry"].get("ids_sha256_canon") == ref_ids[r["seed"]], r)
        for r in sub
        if r["_entry"] and r["_entry"].get("ids_sha256_canon") and ref_ids.get(r["seed"])
    ]
    if not pairs:
        return "---"
    if all(same for same, _ in pairs):
        return "$=$"
    if all(same or r.get("quality_score_max_abs_diff") == 0 for same, r in pairs):
        return "$=$ (ties)"
    return "$\\neq$"


def tab_t3(c) -> list[Path]:
    """T3: official against our Triton reimplementation, from the interleaved ``h2h`` suite."""
    rows = _sel(c.rows, suite="h2h")
    body = []
    for ds, fk, sw in sorted({(r["dataset"], r["filter_kind"], r["sweep"]) for r in rows}):
        cond = _sel(rows, dataset=ds, filter_kind=fk, sweep=sw)
        for k in sorted({r["perf_k"] for r in cond if r.get("perf_k")}):
            for bs in sorted({r["perf_bs"] for r in cond if r.get("perf_bs")}):
                ref = _sel(cond, backend="triton", perf_k=k, perf_bs=bs, perf_mode="eager")
                first = True
                for a, be, pj in _arms(cond):
                    arm = _sel(cond, algo=a, backend=be, params=pj, perf_k=k, perf_bs=bs)
                    for mode in ("eager", "graph"):
                        sub = _sel(arm, perf_mode=mode)
                        t = _lat(sub)
                        if t is None:
                            continue
                        us, calls, top8 = _kernels(sub)
                        cell = _cells(sub)[0]
                        body.append(
                            [
                                f"{DATASET_LABEL.get(ds, ds)} {_esc(sw)} ({fk}), $K={k}$, $B={bs}$"
                                if first
                                else "",
                                _arm_be(a, be, pj),
                                mode,
                                _num(t, 3) + _ci_tex(t, 3),
                                "---"
                                if be == "triton" and mode == "eager"
                                else _ratio_tex(*_ratio(sub, ref)),
                                "---" if us is None else f"${us:.1f}" + ("^{8}$" if top8 else "$"),
                                "---" if calls is None else f"${calls}$",
                                _f(cell.get("index_mib"), 1),
                                _f(cell.get("perf_peak_fwd_mib"), 1),
                                _ids_identity(sub, ref),
                                _f(cell.get(f"quality_jaccard_vs_first@{k}"), 4),
                                _sci(cell.get("quality_score_max_abs_diff")),
                            ]
                        )
                        first = False
    tex = _table(
        c.prov,
        c.results_dir,
        caption="Official SilverTorch against our Triton reimplementation (\\texttt{h2h}, arms "
        "timed interleaved in one process).",
        label="tab:t3",
        colspec="lllcccccccc" + "c",
        header=[
            "Cell",
            "Arm",
            "mode",
            "p50 (ms)",
            "/ triton eager",
            "kernel ($\\mu$s)",
            "launches",
            "index MiB",
            "peak MiB",
            "ids",
            "jaccard",
            "$|\\Delta s|_{\\max}$",
        ],
        body=body,
        notes=_legend(
            [
                "Kernel-only time and launches: one eager call under \\texttt{torch.profiler}, the "
                "sum over every device kernel, median over seeds; $^{8}$: a record profiled "
                "before every kernel was summed, the top-8 kernels only (a lower bound). ids: "
                "the sha256 of the returned ids on the fixed probe batches equals ($=$) the "
                "Triton eager arm's at every common seed; $=$ (ties): equal up to boundary ties "
                "(where it differs the scores are bit-equal, $|\\Delta s|_{\\max}=0$, so only a "
                "tied id at the $k$-th cut differs). jaccard and $|\\Delta s|_{\\max}$ "
                "against the first backend of the cell (the parity spill). Official is "
                "eager-only.",
                STATS_NOTE,
                _clock_note(rows),
            ]
        ),
    )
    return [_write(c.out / "tables" / "tab-t3.tex", tex)]


def fig_f1(c) -> list[Path]:
    """F1: p50 latency against the synthetic pass rate, one panel per scale and batch size;
    SilverTorch at matched recall 0.95."""
    path = c.out / "figures" / "fig-f1-latency-vs-pass-rate.png"
    rows = [r for r in c.rows if r["dataset"].endswith("-synth") and r["filter_kind"] == "clause"]
    synth = _by_scale({r["dataset"] for r in rows}, rows)
    if not synth:
        return [_empty(path, c.prov, "F1: no synth records")]
    fig, axes = plt.subplots(
        len(EXHIBIT_BS), len(synth), figsize=(4.4 * len(synth), 7.2), squeeze=False
    )
    for j, ds in enumerate(synth):
        dsr = _sel(rows, dataset=ds)
        sweeps = sorted({r["sweep"] for r in dsr})
        for i, bs in enumerate(EXHIBIT_BS):
            ax = axes[i][j]
            for a, be, pj in _arms(dsr):
                if a == "silvertorch" or (be != "triton" and a not in FIXED_BACKEND):
                    continue
                pts = []
                for sw in sweeps:
                    sub = _sel(dsr, algo=a, backend=be, params=pj, sweep=sw)
                    t, x = _lat(_timed(sub, bs, EXHIBIT_K)[0]), _pass_rate(sub)
                    if t and x:
                        pts.append((x, t))
                if pts:
                    ax.errorbar(
                        [x for x, _ in pts],
                        [t["value"] for _, t in pts],
                        yerr=[
                            [t["value"] - t["lo"] for _, t in pts],
                            [t["hi"] - t["value"] for _, t in pts],
                        ],
                        marker="o",
                        capsize=2,
                        label=_arm_be(a, be, pj, tex=False),
                    )
            lines = defaultdict(list)
            for cv in curves(c, _sel(dsr, algo="silvertorch", backend="triton"), EXHIBIT_K):
                h = matched(cv, bs, EXHIBIT_K, 0.95)
                x = _pass_rate([r for _, _, rs in cv["points"] for r in rs])
                if h["latency"] is not None and x:
                    lines[cv["rest"]].append((x, h["latency"]))
            for rest, pts in sorted(lines.items()):
                pts.sort()
                ax.plot(
                    [x for x, _ in pts],
                    [t for _, t in pts],
                    marker="D",
                    linestyle="--",
                    label=f"{_arm('silvertorch', rest, False)} [triton] @ recall 0.95",
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{DATASET_LABEL.get(ds, ds)}, B={bs}", fontsize=9)
            ax.set_xlabel("pass rate")
            ax.set_ylabel(f"p50 latency (ms), K={EXHIBIT_K}")
            ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("F1: latency vs pass rate (graph; whiskers = 95% bootstrap CI)", fontsize=10)
    return [_figure(path, fig, c.prov)]


BUCKETS = 10.0 ** np.arange(-4.0, 0.01, 0.5)
MIN_BUCKET = 20


def _buckets(c, rows: list[dict[str, Any]], k: int) -> list[tuple[float, float, int]]:
    """Per-query recall of real sweeps grouped by per-query pass rate (half-decade buckets of
    ``pass_count / n_items``): (geometric-mean pass rate, mean recall, n) per bucket with at
    least ``MIN_BUCKET`` queries."""
    rate, rec = [], []
    for r in _cells(rows):
        rel = r["_rec"].get("per_query")
        if not rel:
            continue
        z = _sidecar(c.results_dir / rel)
        ok = (z["pass_count"] > 0) & ~np.isnan(z[f"recall_oracle@{k}"])
        rate.append(z["pass_count"][ok] / r["n_items"])
        rec.append(z[f"recall_oracle@{k}"][ok])
    if not rate:
        return []
    rate, rec = np.concatenate(rate), np.concatenate(rec)
    idx = np.digitize(rate, BUCKETS)
    return [
        (
            float(np.exp(np.log(rate[idx == b]).mean())),
            float(rec[idx == b].mean()),
            int((idx == b).sum()),
        )
        for b in np.unique(idx)
        if (idx == b).sum() >= MIN_BUCKET
    ]


def fig_f2(c) -> list[Path]:
    """F2: recall_oracle@100 against the pass rate: synthetic curves (SilverTorch per fixed
    ``n_probe``, V3 per pool) with the real ``filter`` sweeps overlaid as per-query buckets."""
    path = c.out / "figures" / "fig-f2-recall-vs-pass-rate.png"
    algos = ("silvertorch", "linr_v3")
    synth = [
        r
        for r in c.rows
        if r["dataset"].endswith("-synth")
        and r["filter_kind"] == "clause"
        and r["algo"] in algos
        and r["backend"] == "triton"
    ]
    real = [
        r
        for r in c.rows
        if r["suite"] == "filter"
        and r["filter_kind"] == "clause"
        and r["algo"] in algos
        and r["backend"] == "triton"
        and r["_rec"].get("per_query")
    ]
    panels = _by_scale({r["dataset"].removesuffix("-synth") for r in synth + real}, c.rows)
    if not panels:
        return [_empty(path, c.prov, "F2: no synth or real silvertorch / V3 records")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], panels, strict=True):
        dsr = _sel(synth, dataset=f"{ds}-synth")
        for a, _, pj in _arms(dsr):
            pts = []
            for sw in sorted({r["sweep"] for r in dsr}):
                sub = _sel(dsr, algo=a, params=pj, sweep=sw)
                q, x = _recall(c, _cells(sub), EXHIBIT_K), _pass_rate(sub)
                if q and x:
                    pts.append((x, q))
            if pts:
                ax.errorbar(
                    [x for x, _ in pts],
                    [q["value"] for _, q in pts],
                    yerr=[
                        [0 if q["lo"] is None else q["value"] - q["lo"] for _, q in pts],
                        [0 if q["hi"] is None else q["hi"] - q["value"] for _, q in pts],
                    ],
                    marker="o",
                    capsize=2,
                    label=f"synth: {_arm(a, pj, False)}",
                )
        for a, _, pj in _arms(_sel(real, dataset=ds)):
            b = _buckets(c, _sel(real, dataset=ds, algo=a, params=pj), EXHIBIT_K)
            if b:
                ax.scatter(
                    [x for x, _, _ in b],
                    [y for _, y, _ in b],
                    marker="x",
                    s=30,
                    label=f"real (per-query buckets): {_arm(a, pj, False)}",
                )
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        ax.set_title(DATASET_LABEL.get(ds, ds), fontsize=9)
        ax.set_xlabel("pass rate")
        ax.set_ylabel(f"recall_oracle@{EXHIBIT_K}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=5)
    fig.suptitle(
        "F2: recall vs pass rate (uniform synthetic filter; real sweeps as "
        f"per-query buckets of >= {MIN_BUCKET} queries)",
        fontsize=10,
    )
    return [_figure(path, fig, c.prov)]


def fig_f3(c) -> list[Path]:
    """F3: the recall-latency Pareto curves of the ``deep`` suite, one panel per (dataset,
    sweep) and batch size, filter kinds as separate curves."""
    path = c.out / "figures" / "fig-f3-pareto.png"
    rows = _sel(c.rows, suite="deep")
    cvs = curves(c, rows, EXHIBIT_K)
    datasets = _by_scale({cv["dataset"] for cv in cvs}, rows)
    panels = [
        (ds, sw)
        for ds in datasets
        for sw in sorted({cv["sweep"] for cv in cvs if cv["dataset"] == ds})
    ]
    if not panels:
        return [_empty(path, c.prov, "F3: no deep-suite curves")]
    fig, axes = plt.subplots(
        len(panels),
        len(EXHIBIT_BS),
        figsize=(4.6 * len(EXHIBIT_BS), 3.7 * len(panels)),
        squeeze=False,
    )
    for i, (ds, sw) in enumerate(panels):
        for j, bs in enumerate(EXHIBIT_BS):
            ax = axes[i][j]
            for cv in [cv for cv in cvs if cv["dataset"] == ds and cv["sweep"] == sw]:
                pts = []
                for x, q, rs in cv["points"]:
                    t = _lat(_timed(rs, bs, EXHIBIT_K)[0])
                    if t:
                        pts.append((t["value"], q["value"], x))
                if pts:
                    ax.plot(
                        [p[0] for p in pts],
                        [p[1] for p in pts],
                        marker="o",
                        label=f"{_curve_label(cv, False)} {cv['filter_kind']}",
                    )
            for target in TARGETS:
                ax.axhline(target, color="0.5", linestyle=":", linewidth=0.8)
            ax.set_xscale("log")
            ax.set_title(f"{DATASET_LABEL.get(ds, ds)} {sw}, B={bs}", fontsize=9)
            ax.set_xlabel("p50 latency (ms)")
            ax.set_ylabel(f"recall_oracle@{EXHIBIT_K}")
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=5)
    fig.suptitle("F3: recall-latency Pareto (deep suite)", fontsize=10)
    return [_figure(path, fig, c.prov)]


def fig_f4a(c) -> list[Path]:
    """F4a: bloom false-positive rate and memory against ``m_bits`` (``bloomwidth*`` suites),
    one line per (dataset, backend, k_hash); the same numbers as a table."""
    rows = [r for r in c.rows if r["suite"].startswith("bloomwidth") and "m_bits" in r["_params"]]
    path = c.out / "figures" / "fig-f4a-bloomwidth.png"
    if not rows:
        return [_empty(path, c.prov, "F4a: no bloomwidth records")]
    series = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["dataset"], r["sweep"], r["backend"], r["_params"].get("k_hash"))
        series[key][r["_params"]["m_bits"]].append(r)
    fig, (ax_fp, ax_mem) = plt.subplots(1, 2, figsize=(10, 4))
    body = []
    for (ds, sw, be, kh), by_m in sorted(series.items(), key=lambda kv: str(kv[0])):
        ms = sorted(by_m)
        fp = [
            statistics.median(
                r["bloom_fp_rate"] for r in _cells(by_m[m]) if r.get("bloom_fp_rate") is not None
            )
            for m in ms
        ]
        mem = [statistics.median(r["index_mib"] for r in _cells(by_m[m])) for m in ms]
        label = f"{DATASET_LABEL.get(ds, ds)} {sw} [{be}] k_hash={kh}"
        ax_fp.plot(ms, fp, marker="o", label=label)
        ax_mem.plot(ms, mem, marker="o", label=label)
        for m, f, mib in zip(ms, fp, mem, strict=True):
            cells = _cells(by_m[m])
            q = _recall(c, cells, EXHIBIT_K)
            t = _lat(_timed(by_m[m], 16, EXHIBIT_K)[0])
            body.append(
                [
                    f"{DATASET_LABEL.get(ds, ds)} {_esc(sw)}",
                    f"\\texttt{{{be}}}",
                    f"${kh}$",
                    f"${m}$",
                    _sci(f),
                    _f(statistics.median(r["filter_mib"] for r in cells), 2),
                    _f(mib, 1),
                    _num(q),
                    _num(t, 3),
                ]
            )
    ax_fp.set_xscale("log", base=2)
    ax_fp.set_yscale("symlog", linthresh=1e-6)
    ax_fp.set_xlabel("m_bits")
    ax_fp.set_ylabel("bloom false-positive rate (symlog below 1e-6)")
    ax_mem.set_xscale("log", base=2)
    ax_mem.set_xlabel("m_bits")
    ax_mem.set_ylabel("index MiB (filter included)")
    for ax in (ax_fp, ax_mem):
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=5)
    fig.suptitle("F4a: bloom width", fontsize=10)
    tex = _table(
        c.prov,
        c.results_dir,
        caption="Bloom width: false-positive rate, memory and recall against $m_\\mathrm{bits}$.",
        label="tab:f4a_bloomwidth",
        colspec="lllcccccc",
        header=[
            "Sweep",
            "Backend",
            "$k_\\mathrm{hash}$",
            "$m_\\mathrm{bits}$",
            "FPR",
            "filter MiB",
            "index MiB",
            f"Recall@{EXHIBIT_K}",
            "p50 $B=16$ (ms)",
        ],
        body=body,
        notes=_legend(
            [
                "FPR: \\texttt{bloom\\_fp\\_rate}, the mean per-query $(\\mathrm{bloom} - "
                "\\mathrm{exact}) / (N - \\mathrm{exact})$. \\texttt{filter\\_mib} is the filter "
                "submodule alone (0 where the module carries its attributes inside "
                "\\texttt{index\\_mib}). Latency only on the timed widths.",
                EXHIBIT_NOTE,
                STATS_NOTE,
            ]
        ),
    )
    return [_figure(path, fig, c.prov), _write(c.out / "tables" / "tab-f4a_bloomwidth.tex", tex)]


def fig_f4b(c) -> list[Path]:
    """F4b: co-design, partial against full bloom path along ``n_probe`` with the paired
    full / partial ratio and its CI (``codesign`` suite)."""
    rows = [r for r in _sel(c.rows, suite="codesign") if "bloom_path" in r["_params"]]
    path = c.out / "figures" / "fig-f4b-codesign.png"
    if not rows:
        return [_empty(path, c.prov, "F4b: no codesign records")]
    panels = sorted({(r["dataset"], r["perf_bs"]) for r in rows if r.get("perf_bs")})
    fig, axes = plt.subplots(len(panels), 2, figsize=(10, 3.4 * len(panels)), squeeze=False)
    body = []
    for (ds, bs), (ax_t, ax_r) in zip(panels, axes, strict=True):
        for sw in sorted({r["sweep"] for r in rows if r["dataset"] == ds}):
            cond = _sel(rows, dataset=ds, sweep=sw, perf_bs=bs, perf_k=EXHIBIT_K)
            rest = sorted(
                {
                    json.dumps(
                        {
                            n: v
                            for n, v in r["_params"].items()
                            if n not in ("bloom_path", "n_probe")
                        },
                        sort_keys=True,
                    )
                    for r in cond
                }
            )
            for other in rest:
                probes = sorted({r["_params"]["n_probe"] for r in cond})
                ratios = []
                for path_name in ("partial", "full"):
                    pts = []
                    for n in probes:
                        sub, _ = _timed(
                            _with(cond, bloom_path=path_name, n_probe=n, **json.loads(other)),
                            bs,
                            EXHIBIT_K,
                        )
                        t = _lat(sub)
                        if t:
                            pts.append((n, t))
                    if pts:
                        ax_t.errorbar(
                            [n for n, _ in pts],
                            [t["value"] for _, t in pts],
                            yerr=[
                                [t["value"] - t["lo"] for _, t in pts],
                                [t["hi"] - t["value"] for _, t in pts],
                            ],
                            marker="o",
                            capsize=2,
                            label=f"{sw} {path_name}",
                        )
                for n in probes:
                    full = _timed(
                        _with(cond, bloom_path="full", n_probe=n, **json.loads(other)),
                        bs,
                        EXHIBIT_K,
                    )
                    part = _timed(
                        _with(cond, bloom_path="partial", n_probe=n, **json.loads(other)),
                        bs,
                        EXHIBIT_K,
                    )
                    ci, paired = _ratio(full[0], part[0])
                    if ci:
                        ratios.append((n, ci))
                    body.append(
                        [
                            f"{DATASET_LABEL.get(ds, ds)} {_esc(sw)}",
                            _esc(other),
                            f"${bs}$",
                            f"${n}$",
                            _lat_tex(part),
                            _lat_tex(full),
                            _ratio_tex(ci, paired),
                        ]
                    )
                if ratios:
                    ax_r.errorbar(
                        [n for n, _ in ratios],
                        [ci[0] for _, ci in ratios],
                        yerr=[
                            [ci[0] - ci[1] for _, ci in ratios],
                            [ci[2] - ci[0] for _, ci in ratios],
                        ],
                        marker="o",
                        capsize=2,
                        label=sw,
                    )
        ax_r.axhline(1.0, color="0.4", linestyle=":")
        for ax, ylabel in ((ax_t, "p50 latency (ms)"), (ax_r, "full / partial (paired)")):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("n_probe")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{DATASET_LABEL.get(ds, ds)}, B={bs}", fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=5)
    fig.suptitle("F4b: co-design, partial vs full bloom mask (whiskers = 95% CI)", fontsize=10)
    tex = _table(
        c.prov,
        c.results_dir,
        caption="Co-design: partial against full bloom mask, p50 latency (ms) and the paired "
        f"full / partial ratio, $K={EXHIBIT_K}$.",
        label="tab:f4b_codesign",
        colspec="llccccc",
        header=[
            "Sweep",
            "Build",
            "$B$",
            "$n_\\mathrm{probe}$",
            "partial",
            "full",
            "full / partial",
        ],
        body=body,
        notes=_legend([EXHIBIT_NOTE, STATS_NOTE, _clock_note(rows)]),
    )
    return [_figure(path, fig, c.prov), _write(c.out / "tables" / "tab-f4b_codesign.tex", tex)]


def _claim_value(c, item: dict[str, Any]) -> str:
    """One ``ours`` entry of ``claims.yaml``: ``metric`` (latency | recall | matched:<target> |
    field:<record column>) of the one arm ``where`` selects (``params`` is a subset match,
    ``null`` = absent), or its ratio over the arm ``vs`` selects."""
    k, bs = item.get("k", EXHIBIT_K), item.get("bs", 1)

    def pick(where: dict[str, Any]) -> list[dict[str, Any]]:
        where = dict(where)
        params = where.pop("params", {})
        rows = _with(_sel(c.rows, **where), **params)
        arms = {(r["algo"], r["backend"], r["params"]) for r in rows}
        if len(arms) > 1 and not item["metric"].startswith("matched:"):
            raise click.ClickException(f"claims.yaml {item['label']!r}: {len(arms)} arms {arms}")
        return rows

    def value(rows: list[dict[str, Any]]) -> float | None:
        metric = item["metric"]
        if metric == "latency":
            v = _lat(_timed(rows, bs, k)[0])
        elif metric == "recall":
            v = _recall(c, _cells(rows), k)
        elif metric.startswith("matched:"):
            hits = [matched(cv, bs, k, float(metric.split(":")[1])) for cv in curves(c, rows, k)]
            return hits[0]["latency"] if len(hits) == 1 else None
        else:
            vals = [
                r[metric.split(":")[1]]
                for r in _cells(rows)
                if r.get(metric.split(":")[1]) is not None
            ]
            return statistics.median(vals) if vals else None
        return None if v is None else v["value"]

    a = pick(item["where"])
    if "vs" not in item:
        v = value(a)
        return "---" if v is None else f"${v:.4g}$"
    b = pick(item["vs"])
    if item["metric"] == "latency":
        return _ratio_tex(*_ratio(_timed(a, bs, k)[0], _timed(b, bs, k)[0]))
    va, vb = value(a), value(b)
    return "---" if va is None or not vb else f"${va / vb:.3g}\\times$"


def tab_t1(c) -> list[Path]:
    """T1: the claims table. ``claims.yaml`` holds each claim's original number, its source and
    the record selectors of "ours"; this fills "ours" and prints the human-written verdict,
    never one of its own."""
    claims = yaml.safe_load(CLAIMS.read_text())["claims"]
    body = [
        [
            c_["id"],
            _esc(c_["claim"]),
            f"{_esc(c_['original'])} ({_esc(c_['source'])})",
            "; ".join(f"{_esc(i['label'])}: {_claim_value(c, i)}" for i in c_["ours"]),
            _esc(c_["verdict"]) if c_["verdict"] else "---",
        ]
        for c_ in claims
    ]
    tex = _table(
        c.prov,
        c.results_dir,
        caption="Claims of the original papers against this reproduction.",
        label="tab:t1_claims",
        colspec="lp{3.2cm}p{3.2cm}p{4.2cm}l",
        header=["", "Claim", "Original", "Ours", "Verdict"],
        body=body,
        notes=_legend(
            [
                "``Ours'' is computed from the records by the selectors in "
                "\\texttt{evaluation/claims.yaml}; verdicts are written there by hand after the "
                "campaign and never generated ('---' until then).",
                EXHIBIT_NOTE,
                STATS_NOTE,
            ]
        ),
    )
    return [_write(c.out / "tables" / "tab-t1_claims.tex", tex)]


# ----- methodology and coverage -------------------------------------------------


def methodology(c) -> list[Path]:
    """The measurement-methodology itemize, its constants read out of the code that produced
    the records, so text and code cannot drift apart."""
    lat = inspect.signature(measure.latency).parameters
    pool = inspect.signature(inputs.query_pool).parameters["n_pool"].default
    recs = c.recs
    env = recs[0]["env"] if recs else {}
    ks = sorted({k for r in recs for k in (r.get("ks") or [])})
    bss = sorted({b for r in recs for b in (r.get("batch_sizes") or [])})
    seeds = sorted({r.get("seed") for r in recs})
    body = [
        f"\\item \\textbf{{Testbed.}} {env.get('gpu', '?')}, driver {env.get('driver', '?')}, "
        f"CUDA {env.get('cuda', '?')}, PyTorch {env.get('torch', '?')}, "
        f"Triton {env.get('triton', '?')}, Python {env.get('python', '?')}. "
        "SM clocks \\textbf{cannot be locked} in this container "
        "(\\texttt{nvidia-smi -lgc} is denied), so every record keeps the under-load "
        "clock sample (\\texttt{perf[].sm\\_mhz}) and every latency comparison uses the "
        "same clock estimator.",
        "\\item \\textbf{Isolation.} One process per "
        "$(\\mathrm{dataset}, \\mathrm{dim}, \\mathrm{algo}, \\mathrm{backend})$: "
        "no \\texttt{torch.compile} cache, CUDA-graph pool or allocator state survives a "
        "change of backend.",
        f"\\item \\textbf{{Warm-up and timing.}} {lat['warmup'].default} warm-up calls, "
        f"then {lat['windows'].default} windows of "
        f"$N = \\mathrm{{clamp}}({lat['target_s'].default:g}\\,\\mathrm{{s}} / "
        f"\\tilde t, {lat['n_min'].default}, {lat['n_max'].default})$ calls; each call "
        "is bracketed by CUDA events, each window by one \\texttt{synchronize} at its end. "
        f"Queries cycle through a fixed pool of {pool} batches, without flushing L2. "
        "Each window yields its median; a window spread "
        "$(\\max-\\min)/\\mathrm{med}$ above 5\\,\\% marks the cell \\texttt{unstable}.",
        f"\\item \\textbf{{Modes.}} {', '.join(run.MODES)}: \\texttt{{graph}} "
        '(\\texttt{torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)}) '
        "is the headline, \\texttt{eager} the secondary; the official SilverTorch ops "
        "cannot be captured and are timed eager only. Capture is checked: "
        "\\texttt{cudagraph\\_skips} must be 0, otherwise the cell is an error, not a number.",
        "\\item \\textbf{Quality.} One eager pass at "
        f"$k_{{\\max}} = \\max(K)$, in chunks of {run.QUALITY_CHUNK} queries; "
        f"recall / ndcg / precision / mrr at $K \\in \\{{{', '.join(map(str, ks)) or '?'}\\}}$ "
        "accumulate on the device. References: the exact filtered fp32 oracle on filter "
        "cells, and the held-out interactions always.",
        "\\item \\textbf{Memory.} \\texttt{index\\_mib} is the sum of every buffer of the "
        "retrieval module, its filter submodule included; \\texttt{peak\\_fwd\\_mib} is "
        "\\texttt{max\\_memory\\_allocated} over the first eager window.",
        "\\item \\textbf{Repeats.} Batch sizes "
        f"$B \\in \\{{{', '.join(map(str, bss)) or '?'}\\}}$, "
        f"seeds $\\{{{', '.join(str(s) for s in seeds) or '?'}\\}}$.",
        "\\item \\textbf{Statistics.} A latency is the median of the per-window medians "
        "pooled over seed $\\times$ window; a recall is the mean over queries of the "
        "per-query \\texttt{recall\\_oracle}. Both carry a 95\\,\\% percentile bootstrap CI "
        f"($B = {stats.B}$, fixed generator seed) over those units. Arms timed interleaved "
        "(ABAB, one process) are compared by the per-round ratio, its median and its CI "
        "over seed $\\times$ round; other comparisons resample each arm independently. A "
        "ratio whose CI contains 1 is reported as no difference.",
    ]
    tex = (
        _banner(c.prov, c.results_dir)
        + "\\begin{itemize}\n"
        + "\n".join("    " + b for b in body)
        + "\n\\end{itemize}\n"
    )
    return [_write(c.out / "methodology.tex", tex)]


def _manifest_md(c) -> list[str]:
    if c.manifest is None:
        return ["## Selection", "", "Latest record per resume key (no `--manifest`).", ""]
    lines = ["## Manifest selection", ""]
    for name, keys in c.why.items():
        lines += [f"### {name}: {len(keys)}", ""] + [f"- `{k}`" for k in keys] + [""]
    lines += ["### Manifest log", ""]
    lines += [f"- {x}" for x in c.manifest.get("log") or []] or ["none"]
    return [*lines, ""]


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
        f"- gpu: {p['gpus']}  host: {p['hosts']}  window: {p['started'][0]} .. {p['started'][1]}",
        "",
        "## Citability (CLAUDE.md rule 2)",
        "",
    ]
    if p["citable"]:
        lines += [f"**CITABLE** — the caller declared gate `{p['gate']}` green.", ""]
    else:
        lines += ["**NOT CITABLE.** Every artifact carries the marker. Reasons:", ""]
        lines += [f"- {b}" for b in p["blockers"]]
        lines += [""]
    lines += [
        *_manifest_md(c),
        "## Selection used by the tables",
        "",
        f"`--dim {c.dim}` `--k {c.k}` `--bs {c.bs}` `--mode {c.mode}` `--backend {c.backend}`.",
        "",
        "## Coverage",
        "",
        "| dataset | dim | suite | filter | sweep | algo | backend | params | seeds | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    groups = defaultdict(lambda: [set(), Counter()])
    for r in c.recs:
        key = (
            r["dataset"],
            r["dim"],
            r["suite"],
            r["filter_kind"],
            r["sweep"],
            r["algo"],
            r["backend"],
            json.dumps(r["params"], sort_keys=True),
        )
        groups[key][0].add(r["seed"])
        groups[key][1][r.get("status", "ok")] += 1
    for key in sorted(groups, key=str):
        seeds, status = groups[key]
        lines.append(
            "| "
            + " | ".join([*(str(x) for x in key), str(sorted(seeds)), str(dict(status))])
            + " |"
        )
    failed = [r for r in c.recs if r.get("status") == "failed"]
    partial = [r for r in c.recs if r.get("status") == "partial"]
    lines += ["", "## Failed cells (excluded from every number)", ""]
    lines += [
        f"- `{records.key_block(r)}` — stage `{r.get('stage')}`: "
        f"{(r.get('error') or '').splitlines()[-1][:160] if r.get('error') else '?'}"
        for r in failed
    ] or ["none"]
    lines += ["", "## Partial records (marked `*`)", ""]
    lines += [f"- `{records.key_block(r)}` — {r.get('partial_reasons')}" for r in partial] or [
        "none"
    ]
    unstable = [(r, e) for r in c.recs for e in (r.get("perf") or []) if e.get("unstable")]
    lines += ["", f"## Unstable perf variants (marked `†`): {len(unstable)}", ""]
    lines += [
        f"- {r['dataset']} {r['algo']}/{r['backend']} k={e['k']} bs={e['bs']} "
        f"{e['mode']}: spread {e['spread'] * 100:.1f}%"
        for r, e in unstable[:40]
    ] or ["none"]
    if len(unstable) > 40:
        lines.append(f"- ... {len(unstable) - 40} more")
    lines += ["", "## Artifacts", ""]
    lines += [f"- `{q.relative_to(c.out)}`" for q in sorted(c.written)]
    lines += [
        "",
        "## Clock estimator",
        "",
        "Latency artifacts use the per-variant under-load `perf[].sm_mhz`. "
        "Idle samples (`env.sm_mhz_idle`, and the schema-1 `env.sm_mhz`, which is a "
        "whole-run median dominated by idle) are provenance only and are never "
        "compared with an under-load sample — the error that made 92 of 99 of C4's "
        "latency rows appear to fail.",
        "",
    ]
    return [_write(c.out / "report.md", "\n".join(lines) + "\n")]


ARTIFACTS = {
    "pareto": tab_pareto,
    "memory": tab_memory,
    "parity": tab_backend_parity,
    "matched": tab_matched,
    "t1": tab_t1,
    "t2": tab_t2,
    "t3": tab_t3,
    "f1": fig_f1,
    "f2": fig_f2,
    "f3": fig_f3,
    "f4a": fig_f4a,
    "f4b": fig_f4b,
    "fig_deep_sweep": fig_deep_sweep,
    "fig_latency_violin": fig_latency_violin,
    "methodology": methodology,
}


def _one_inputs_per_dataset(rows: list[dict[str, Any]]) -> None:
    """Tables select by dataset and dim, never by encoder: two encoders under one
    ``(dataset, dim)`` would land in one cell, so the report refuses them."""
    seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in rows:
        seen[r["dataset"], r["dim"]].add(r["inputs"])
    mixed = {f"{d} d{dim}": sorted(v) for (d, dim), v in seen.items() if len(v) > 1}
    if mixed:
        raise click.ClickException(
            f"records on more than one input identity: {mixed}; report one encoder per "
            "results directory"
        )


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


class Ctx:
    """Everything every artifact needs: the flat rows, the nested records, the samples, the
    provenance verdict and the one set of selectors."""

    def __init__(
        self, results_dir: Path, out: Path, gate: str | None, manifest: Path | None, **sel: Any
    ) -> None:
        self.results_dir, self.out = Path(results_dir), Path(out)
        self.manifest, self.why = None, None
        if manifest is None:
            self.flat, self.rows = _load(self.results_dir, self.out)
            self.recs = records.latest(self.results_dir)
            self.samples = _samples(self.results_dir)
        else:
            self.manifest = load_manifest(manifest)
            merged, used, self.why = select(records.latest(self.results_dir), self.manifest)
            tree = self.out / "selected"
            shutil.rmtree(tree, ignore_errors=True)
            for r in merged:
                records.append_record(tree / r["suite"] / f"{r['dataset']}-d{r['dim']}.jsonl", r)
            self.flat, self.rows = _load(tree, self.out)
            self.recs = records.latest(tree)
            self.samples = []  # the samples sidecar carries no code_version to select by
        _attach(self.rows, self.recs)
        _one_inputs_per_dataset(self.rows)
        self.prov = provenance(self.recs if manifest is None else used, gate)
        self.written: list[Path] = [self.flat]
        for k, v in sel.items():
            setattr(self, k, v)


def generate(
    results_dir: Path,
    out: Path,
    *,
    gate: str | None = None,
    only: tuple[str, ...] = (),
    manifest: Path | None = None,
    **sel: Any,
) -> Ctx:
    c = Ctx(results_dir, out, gate, manifest, **sel)
    for name in only or tuple(ARTIFACTS):
        c.written += ARTIFACTS[name](c)
    c.written += coverage(c)
    return c


@click.command()
@click.argument("root", default="results")
@click.option("--out", default=None, help="output directory [default: <root>/report]")
@click.option(
    "--gate",
    default=None,
    help="the roadmap step whose gate is green for these records (e.g. D1). Without "
    "it every artifact is marked NOT CITABLE; it cannot override evidence.",
)
@click.option(
    "--manifest",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="campaign manifest (evaluation/campaign.yaml): select each cell's quality and perf "
    "records by their accepted code_version instead of the latest record",
)
@click.option(
    "--only",
    multiple=True,
    type=click.Choice(sorted(ARTIFACTS)),
    help="emit only these artifacts (report.md is always written)",
)
@click.option("--dim", default=128, show_default=True, type=int)
@click.option("--k", default=100, show_default=True, type=int)
@click.option("--bs", default=1, show_default=True, type=int, help="batch size for the tables")
@click.option("--mode", default="eager", type=click.Choice(("eager", "graph")), show_default=True)
@click.option("--backend", default="triton", show_default=True)
def report(root, out, gate, manifest, only, dim, k, bs, mode, backend) -> None:
    """Paper tables and figures from the records (docs/system/evaluation.md § Report)."""
    results_dir = Path(root)
    if not results_dir.is_dir():
        raise click.ClickException(f"{results_dir}: not a directory")
    c = generate(
        results_dir,
        Path(out) if out else results_dir / "report",
        gate=gate,
        only=only,
        manifest=manifest,
        dim=dim,
        k=k,
        bs=bs,
        mode=mode,
        backend=backend,
    )
    for path in c.written:
        click.echo(str(path))
    if c.why:
        click.echo("manifest: " + ", ".join(f"{len(v)} {n}" for n, v in c.why.items()))
    click.echo(
        f"{len(c.written)} artifacts from {c.prov['n_records']} records — "
        + ("CITABLE" if c.prov["citable"] else "NOT CITABLE: " + "; ".join(c.prov["blockers"]))
    )


__all__ = ["ARTIFACTS", "generate", "provenance", "report"]
