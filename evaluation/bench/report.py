"""``bench report``: a results tree as ``results.parquet`` plus ``report.md`` (provenance, the
citability verdict, coverage, one line per cell, the failed / partial / unstable cells); nothing
is citable unless ``--gate`` names a green gate and the evidence agrees (docs/system/evaluation.md
§ Report)."""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from bench import records

MAIN_BRANCHES = ("staging", "development", "main")  # rule 2: a branch's numbers are not citable


def provenance(recs: list[dict[str, Any]], gate: str | None) -> dict[str, Any]:
    """The rule-2 verdict over a set of records, shared with ``bench upload``'s manifest."""
    env = [r.get("env") or {} for r in recs]
    status = Counter(r.get("status", "ok") for r in recs)
    dirty = [r for r, e in zip(recs, env, strict=True) if e.get("dirty")]
    why = Counter(
        x for r in recs if r.get("status") == "partial" for x in r.get("partial_reasons") or ["?"]
    )
    why_s = ", ".join(f"{x} {n}" for x, n in sorted(why.items()))
    found: list[tuple[str, str]] = []  # (blocker for report.md / the manifest, short mark)

    def add(blocker: str, mark: str) -> None:
        found.append((blocker, mark))

    if not gate:
        add(
            "no --gate given: no validation gate is declared passed for these records",
            "gate not green",
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
            f"record(s) produced on {', '.join(off)} — "
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


def one_inputs_per_dataset(recs: list[dict[str, Any]]) -> None:
    """A tree holds one encoder per ``(dataset, dim)``: two would land in one cell line."""
    seen: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in recs:
        seen[r["dataset"], r["dim"]].add(records.inputs_of(r))
    mixed = {f"{d} d{dim}": sorted(v) for (d, dim), v in seen.items() if len(v) > 1}
    if mixed:
        raise click.ClickException(
            f"records on more than one input identity: {mixed}; report one encoder per "
            "results directory"
        )


def _cell_line(rec: dict[str, Any]) -> str:
    q = rec.get("quality") or {}
    recall = (q.get("oracle") or {}).get(f"recall@{rec.get('k_max')}")
    timings = ", ".join(
        f"k{e['k']} bs{e['bs']} {e['mode']} {e['median_ms']:.4g}{'†' if e.get('unstable') else ''}"
        for e in rec.get("perf") or []
        if e.get("median_ms") is not None
    )
    cells = [
        rec["dataset"], str(rec["dim"]), rec["suite"], rec["filter_kind"], rec["sweep"],
        rec["algo"], rec["backend"], f"`{json.dumps(rec['params'], sort_keys=True)}`",
        str(rec["seed"]),
        rec.get("status", "ok") + ("*" if rec.get("status") == "partial" else ""),
        "" if recall is None else f"{recall:.4f}", timings or "—",
    ]  # fmt: skip
    return "| " + " | ".join(cells) + " |"


def report_md(results_dir: Path, recs: list[dict[str, Any]], prov: dict[str, Any]) -> str:
    """Provenance and verdict, coverage, one line per cell, then the failed / partial /
    unstable cells."""
    p = prov
    lines = [
        "# `bench report`",
        "",
        f"- generated: `{p['generated']}`  from `{results_dir}`",
        f"- records: **{p['n_records']}** {p['status']}, cells flagged unstable "
        f"(window spread or clock drift): {p['unstable']}",
        f"- schema_version: {p['schema_versions']}  |  code_version: {p['code_versions']}",
        f"- runs (commit, branch): {p['runs']}",
        f"- gpu: {p['gpus']}  host: {p['hosts']}  window: {p['started'][0]} .. {p['started'][1]}",
        "",
        "## Citability",
        "",
    ]
    if p["citable"]:
        lines += [f"**CITABLE** — the caller declared gate `{p['gate']}` green.", ""]
    else:
        lines += ["**NOT CITABLE.** Reasons:", ""] + [f"- {b}" for b in p["blockers"]] + [""]
    groups: dict[tuple, tuple[set, Counter]] = defaultdict(lambda: (set(), Counter()))
    for r in recs:
        key = (r["dataset"], r["dim"], r["suite"], r["filter_kind"], r["sweep"], r["algo"],
               r["backend"], json.dumps(r["params"], sort_keys=True))  # fmt: skip
        groups[key][0].add(r["seed"])
        groups[key][1][r.get("status", "ok")] += 1
    lines += [
        "## Coverage",
        "",
        "| dataset | dim | suite | filter | sweep | algo | backend | params | seeds | status |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(
            "| " + " | ".join([*map(str, key), str(sorted(seeds)), str(dict(status))]) + " |"
            for key, (seeds, status) in sorted(groups.items(), key=lambda kv: str(kv[0]))
        ),
        "",
        "## Cells",
        "",
        "One line per record: `recall` is `recall_oracle@k_max`, the timings are the median ms "
        "per `(k, bs, mode)` (`†` = unstable window spread, `*` = a partial record).",
        "",
        "| dataset | dim | suite | filter | sweep | algo | backend | params | seed | status | "
        "recall | median ms |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        *(_cell_line(r) for r in recs if r.get("status") != "failed"),
    ]
    failed = [r for r in recs if r.get("status") == "failed"]
    partial = [r for r in recs if r.get("status") == "partial"]
    lines += ["", "## Failed cells (in no number)", ""]
    lines += [
        f"- `{records.key_block(r)}` — stage `{r.get('stage')}`: "
        f"{(r.get('error') or '').splitlines()[-1][:160] if r.get('error') else '?'}"
        for r in failed
    ] or ["none"]
    lines += ["", "## Partial records (marked `*`)", ""]
    lines += [f"- `{records.key_block(r)}` — {r.get('partial_reasons')}" for r in partial] or [
        "none"
    ]
    unstable = [(r, e) for r in recs for e in (r.get("perf") or []) if e.get("unstable")]
    lines += ["", f"## Unstable perf variants (marked `†`): {len(unstable)}", ""]
    lines += [
        f"- {r['dataset']} {r['algo']}/{r['backend']} k={e['k']} bs={e['bs']} "
        f"{e['mode']}: spread {e['spread'] * 100:.1f}%"
        for r, e in unstable[:40]
    ] or ["none"]
    if len(unstable) > 40:
        lines.append(f"- ... {len(unstable) - 40} more")
    lines += [
        "",
        "## Clock estimator",
        "",
        "Latencies are read against the per-variant under-load `perf[].sm_mhz`; the idle "
        "`env.sm_mhz_idle` is provenance only.",
        "",
    ]
    return "\n".join(lines) + "\n"


def generate(results_dir: Path, out: Path, *, gate: str | None = None) -> dict[str, Any]:
    """``results.parquet`` and ``report.md`` under ``out``; returns the provenance verdict."""
    out.mkdir(parents=True, exist_ok=True)
    recs = records.latest(results_dir)
    one_inputs_per_dataset(recs)
    records.aggregate(results_dir, out / "results.parquet")
    prov = provenance(recs, gate)
    (out / "report.md").write_text(report_md(results_dir, recs, prov))
    return prov


@click.command()
@click.argument("root", default="results")
@click.option("--out", default=None, help="output directory [default: <root>/report]")
@click.option(
    "--gate",
    default=None,
    help="the validation gate passed for these records. Without it the report is "
    "marked NOT CITABLE; it cannot override evidence.",
)
def report(root, out, gate) -> None:
    """results.parquet and report.md for a results tree (docs/system/evaluation.md § Report)."""
    results_dir = Path(root)
    if not results_dir.is_dir():
        raise click.ClickException(f"{results_dir}: not a directory")
    out_dir = Path(out) if out else results_dir / "report"
    prov = generate(results_dir, out_dir, gate=gate)
    click.echo(f"{out_dir / 'results.parquet'}\n{out_dir / 'report.md'}")
    click.echo(
        f"{prov['n_records']} records — "
        + ("CITABLE" if prov["citable"] else "NOT CITABLE: " + "; ".join(prov["blockers"]))
    )
