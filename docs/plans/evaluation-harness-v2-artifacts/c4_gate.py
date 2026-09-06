#!/usr/bin/env python3
"""c4_gate.py — roadmap C4 / evaluation-harness-v2.md §6 WP-4: the golden comparison.

Compares one or more harness-v2 JSONL result files (``bench run`` output,
``results/<suite>/<dataset>-d<dim>.jsonl``) against roadmap A1's golden cells
(``evaluation/golden/*.json``, the old harness's per-row JSON) and prints a per-cell
PASS / FAIL table. Exit status 1 on any FAIL.

    python3 docs/plans/evaluation-harness-v2-artifacts/c4_gate.py \\
        evaluation/results/filter/goodreads-d128.jsonl evaluation/results/filter/arxiv-d128.jsonl \\
        [--golden evaluation/golden] [--golden-sm-mhz 1140] [--tol 1e-6] [--latency-tol 0.05] \\
        [--jaccard-official 0.99] [--seed 0]

The WP-4 gates, one row each:

1. **quality** — ``quality.oracle.recall@k`` / ``ndcg@k`` within ``--tol`` (1e-6) of the golden
   ``recall@k`` / ``ndcg@k`` for every golden ``(dataset, dim, sweep, algo, backend, k)``. The
   v2 cell that stands for a golden cell is the one at the algo's *default* params (the
   golden ran with ``extra.params = {}``): ``DEFAULT_PARAMS``. Other cells (``n_probe: 32``,
   the ``official`` backend) have no golden and are checked on 3–5 only.
2. **latency** — the v2 ``graph`` ``median_ms`` within ``--latency-tol`` (5 %) of the golden
   ``median_ms`` at the same ``(k, bs)``. Both SM clocks are printed: the golden's is
   ``--golden-sm-mhz`` (1140 MHz, from ``evaluation/golden/_logs/clocks.csv`` — A1 could not
   lock clocks), the run's is the per-variant ``perf[].sm_mhz`` (sampled under load right
   after the timing window; ``env.sm_mhz`` — the idle sample — is the fallback). When the two
   differ by more than 2 % the ratio is clock-normalised, ``median_ms × sm_mhz / golden_sm_mhz``,
   and the row says so: H §7's fallback for a box that cannot pin clocks. That normalisation
   is linear and approximate (memory-bound kernels do not scale with the SM clock), so a
   normalised PASS is weaker than a same-clock one — the table keeps both ratios visible.
   Remember also that the golden number is a *cold-L2* ``do_bench`` median (H §1) and v2
   does not flush, so v2 is expected at or below golden; a larger gap needs the ``--flush-l2``
   explanation WP-4 (2) asks for.
3. **graph** — every ``graph`` perf entry on a capturable backend was *measured*: a capture
   that recorded ``cudagraph_skips > 0`` (or more than one ``cudaGraphLaunch`` per call) is a
   null entry with ``reason``, and that is a FAIL. On ``official`` the only accepted reason is
   ``not_capturable`` (O D7).
4. **parity** — ``quality.jaccard_vs_first@100 == 1.0`` for the exact algos
   (``linr_v1_filter_mask``, ``linr_v2``) on the backend that was compared against the
   reference (``quality.parity == "vs_<backend>"``); reported, not gated, for the inexact
   algos; ``silvertorch``/``official`` needs ``≥ --jaccard-official`` (0.99, O WP-6's gate).
5. **stable** — no record with ``unstable: true`` (any perf entry's window spread > 5 %, or a
   clock drift between cells). At unlocked clocks this may fire; it is still a FAIL here and
   the reader decides with the clocks in view.

Also checked: every golden cell has exactly one matching v2 record (the *last* record per key
block wins, as ``report.py`` reads it), every v2 record is ``status: ok``, and the official
cell(s) ran end to end (``status: ok`` with quality and perf present, and every timed entry
at ``cache_plans: false``). Gate (6) — kill the child mid-run and ``--resume`` — is a manual
step and not checked here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# The golden rows carry ``extra.params = {}``: the algo defaults of the old harness, which are
# the library defaults ``algos.build`` still applies when a param is absent.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "silvertorch": {"n_lists": 1024, "n_probe": 24, "n_iter": 10},
    "linr_v3": {"candidate_pool": 5000},
}
EXACT_ALGOS = ("linr_v1_filter_mask", "linr_v2")
CAPTURABLE = {"triton": True, "torch": True, "official": False}
CLOCK_SAME = (
    0.02  # within 2 % the clocks count as equal (bench.clocks' clocks_locked band)
)
GOLDEN_SM_MHZ = (
    1140  # evaluation/golden/_logs/clocks.csv: the A100's load clock, unlocked
)


# ----- inputs ---------------------------------------------------------------------------


def load_golden(golden_dir: Path) -> dict[tuple, dict[str, Any]]:
    """``{(dataset, dim, filter_kind, sweep, algo, backend, seed): {"quality": {k: {...}},
    "latency": {(k, bs): median_ms}, "file": name}}`` from ``<golden_dir>/*.json``. The dataset
    and dim come from the file name (``<dataset>-d<dim>-<sweep>-<algo>-<backend>.json``); the
    rest from the rows. Quality is per ``k`` and must agree across ``bs`` rows."""
    out: dict[tuple, dict[str, Any]] = {}
    for path in sorted(golden_dir.glob("*.json")):
        dataset, d_dim = path.stem.split("-")[:2]
        dim = int(d_dim.lstrip("d"))
        rows = json.loads(path.read_text())
        if not rows:
            continue
        r0 = rows[0]
        key = (
            dataset,
            dim,
            r0["filter_kind"],
            r0["sweep"],
            r0["impl"],
            r0["backend"],
            r0["seed"],
        )
        cell: dict[str, Any] = {"quality": {}, "latency": {}, "file": path.name}
        for r in rows:
            k, bs = int(r["k"]), int(r["batch_size"])
            q = {"recall": float(r[f"recall@{k}"]), "ndcg": float(r[f"ndcg@{k}"])}
            if k in cell["quality"] and cell["quality"][k] != q:
                raise ValueError(
                    f"{path.name}: quality at k={k} differs across bs rows"
                )
            cell["quality"][k] = q
            cell["latency"][k, bs] = float(r["median_ms"])
        out[key] = cell
    if not out:
        raise FileNotFoundError(f"no golden JSON under {golden_dir}")
    return out


def load_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Last record per key block across the given JSONL files (a malformed line raises)."""
    last: dict[str, dict[str, Any]] = {}
    for path in paths:
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i + 1}: malformed record: {exc}") from exc
            last[cell_name(rec)] = rec
    return list(last.values())


def cell_name(rec: dict[str, Any]) -> str:
    params = json.dumps(rec.get("params") or {}, sort_keys=True, separators=(",", ":"))
    return (
        f"{rec['dataset']}-d{rec['dim']}-{rec['filter_kind']}/{rec['sweep']}-{rec['algo']}-"
        f"{rec['backend']}-s{rec['seed']}-{params}"
    )


def at_defaults(algo: str, params: dict[str, Any]) -> bool:
    """The cell that stands for the golden one: every param at the algo's default."""
    defaults = DEFAULT_PARAMS.get(algo, {})
    return all(k in defaults and defaults[k] == v for k, v in (params or {}).items())


# ----- the checks -----------------------------------------------------------------------


def _row(
    rows: list[dict[str, Any]], cell: str, check: str, ok: bool | None, detail: str
) -> None:
    rows.append(
        {"cell": cell, "check": check, "verdict": _verdict(ok), "detail": detail}
    )


def _verdict(ok: bool | None) -> str:
    return "INFO" if ok is None else ("PASS" if ok else "FAIL")


def _perf_index(rec: dict[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {(int(e["k"]), int(e["bs"]), e["mode"]): e for e in rec.get("perf") or []}


def check_quality(
    rows: list, name: str, rec: dict[str, Any], gold: dict[str, Any], tol: float
) -> None:
    q = (rec.get("quality") or {}).get("oracle")
    if q is None:
        _row(rows, name, "quality", False, "no quality.oracle block on the record")
        return
    for k, g in sorted(gold["quality"].items()):
        for metric in ("recall", "ndcg"):
            v = q.get(f"{metric}@{k}")
            if v is None:
                _row(
                    rows, name, f"quality {metric}@{k}", False, "missing on the record"
                )
                continue
            diff = abs(float(v) - g[metric])
            _row(
                rows,
                name,
                f"quality {metric}@{k}",
                diff <= tol,
                f"v2 {float(v):.6f} golden {g[metric]:.6f} |diff| {diff:.1e} (tol {tol:g})",
            )


def check_latency(
    rows: list,
    name: str,
    rec: dict[str, Any],
    gold: dict[str, Any],
    tol: float,
    golden_sm_mhz: float,
) -> None:
    perf = _perf_index(rec)
    env_sm = (rec.get("env") or {}).get("sm_mhz")
    for (k, bs), g_ms in sorted(gold["latency"].items()):
        e = perf.get((k, bs, "graph"))
        if e is None or e.get("median_ms") is None:
            reason = "no graph entry" if e is None else f"null ({e.get('reason')})"
            if not CAPTURABLE.get(rec["backend"], True):
                _row(
                    rows,
                    name,
                    f"latency k={k} bs={bs}",
                    None,
                    f"{reason}: not capturable",
                )
            else:
                _row(rows, name, f"latency k={k} bs={bs}", False, reason)
            continue
        ms = float(e["median_ms"])
        sm = e.get("sm_mhz") if e.get("sm_mhz") is not None else env_sm
        raw = ms / g_ms
        clock = "clock unknown"
        ratio = raw
        if sm is not None:
            same = abs(float(sm) - golden_sm_mhz) <= CLOCK_SAME * golden_sm_mhz
            clock = f"sm {float(sm):.0f}/{golden_sm_mhz:.0f} MHz"
            if not same:
                ratio = ms * float(sm) / golden_sm_mhz / g_ms
                clock += f" → normalised ratio {ratio:.3f}"
        eager = perf.get((k, bs, "eager")) or {}
        e_ms = eager.get("median_ms")
        detail = (
            f"v2 graph {ms:.4f} ms golden {g_ms:.4f} ms ratio {raw:.3f} [{clock}]"
            + (f"; v2 eager {float(e_ms):.4f} ms" if e_ms is not None else "")
        )
        _row(rows, name, f"latency k={k} bs={bs}", abs(ratio - 1.0) <= tol, detail)


def check_graph(rows: list, name: str, rec: dict[str, Any]) -> None:
    perf = [e for e in rec.get("perf") or [] if e["mode"] == "graph"]
    if not perf:
        _row(rows, name, "graph", None, "no graph entries (eager-only run)")
        return
    if not CAPTURABLE.get(rec["backend"], True):
        bad = [e for e in perf if e.get("reason") != "not_capturable"]
        _row(
            rows,
            name,
            "graph",
            not bad,
            f"{len(perf)} entries null with reason not_capturable"
            if not bad
            else f"unexpected on official: {sorted({str(e.get('reason')) for e in bad})}",
        )
        return
    bad = [e for e in perf if e.get("reason") or e.get("median_ms") is None]
    _row(
        rows,
        name,
        "graph",
        not bad,
        f"{len(perf)}/{len(perf)} graph entries measured (cudagraph_skips == 0)"
        if not bad
        else f"{len(bad)}/{len(perf)} graph entries not measured: "
        + ", ".join(f"k={e['k']} bs={e['bs']}: {e.get('reason')}" for e in bad),
    )


def check_parity(
    rows: list, name: str, rec: dict[str, Any], jaccard_official: float
) -> None:
    q = rec.get("quality") or {}
    parity, j = q.get("parity"), q.get("jaccard_vs_first@100")
    if parity == "reference" or parity is None:
        _row(
            rows,
            name,
            "parity",
            None,
            f"parity={parity} (wrote the reference or no quality)",
        )
        return
    if not str(parity).startswith("vs_"):
        _row(rows, name, "parity", False, f"parity={parity}")
        return
    if j is None:
        _row(rows, name, "parity", False, f"{parity}: jaccard_vs_first@100 missing")
        return
    j = float(j)
    diff = q.get("score_max_abs_diff")
    detail = f"{parity}: jaccard@100 {j:.6f}" + (
        f", score_max_abs_diff {float(diff):.3e}" if diff is not None else ""
    )
    if rec["algo"] in EXACT_ALGOS:
        _row(rows, name, "parity", j == 1.0, detail + " (exact algo: must be 1.0)")
    elif rec["backend"] == "official":
        _row(
            rows,
            name,
            "parity",
            j >= jaccard_official,
            detail + f" (≥ {jaccard_official})",
        )
    else:
        _row(rows, name, "parity", None, detail + " (inexact algo: reported)")


def check_stable(rows: list, name: str, rec: dict[str, Any]) -> None:
    unstable = [e for e in rec.get("perf") or [] if e.get("unstable")]
    drift = (rec.get("env") or {}).get("clocks_drift")
    ok = not rec.get("unstable")
    detail = "stable" if ok else "unstable: "
    if unstable:
        detail += ", ".join(
            f"k={e['k']} bs={e['bs']} {e['mode']} spread {float(e['spread']):.3f}"
            for e in unstable
        )
    if drift:
        detail += (" " if unstable else "") + "clocks_drift"
    _row(rows, name, "stable", ok, detail)


def check_official(rows: list, name: str, rec: dict[str, Any]) -> None:
    perf = rec.get("perf") or []
    timed = [e for e in perf if e.get("median_ms") is not None]
    cached = [e for e in timed if e.get("cache_plans") is not False]
    ok = (
        rec.get("status") == "ok"
        and rec.get("quality") is not None
        and bool(timed)
        and not cached
    )
    _row(
        rows,
        name,
        "official",
        ok,
        f"status {rec.get('status')}, quality {'present' if rec.get('quality') else 'missing'}, "
        f"{len(timed)} timed entries, {len(cached)} with the plan cache on",
    )


# ----- driver ---------------------------------------------------------------------------


def gate(
    jsonl: list[Path],
    golden_dir: Path,
    *,
    tol: float = 1e-6,
    latency_tol: float = 0.05,
    jaccard_official: float = 0.99,
    golden_sm_mhz: float = GOLDEN_SM_MHZ,
    seed: int | None = 0,
) -> tuple[list[dict[str, Any]], Counter]:
    """Run every check; returns the table rows and the verdict counts."""
    golden = load_golden(golden_dir)
    recs = load_records(jsonl)
    if seed is not None:
        recs = [r for r in recs if r.get("seed") == seed]
    rows: list[dict[str, Any]] = []
    matched: dict[tuple, list[str]] = {key: [] for key in golden}
    for rec in recs:
        name = cell_name(rec)
        if rec.get("status") != "ok":
            _row(
                rows,
                name,
                "status",
                False,
                f"status {rec.get('status')} ({rec.get('stage')})",
            )
            continue
        key = (
            rec["dataset"], rec["dim"], rec["filter_kind"], rec["sweep"], rec["algo"],
            rec["backend"], rec["seed"],
        )  # fmt: skip
        gold = (
            golden.get(key)
            if at_defaults(rec["algo"], rec.get("params") or {})
            else None
        )
        if gold is not None:
            matched[key].append(name)
            check_quality(rows, name, rec, gold, tol)
            check_latency(rows, name, rec, gold, latency_tol, golden_sm_mhz)
        else:
            _row(
                rows, name, "quality", None, "no golden cell for these params / backend"
            )
        check_graph(rows, name, rec)
        check_parity(rows, name, rec, jaccard_official)
        check_stable(rows, name, rec)
        if rec["backend"] == "official":
            check_official(rows, name, rec)
    for key, names in matched.items():
        gname = golden[key]["file"]
        if len(names) != 1:
            _row(
                rows,
                gname,
                "coverage",
                False,
                "no v2 record at the algo defaults"
                if not names
                else f"{len(names)} records",
            )
    counts = Counter(r["verdict"] for r in rows)
    return rows, counts


def render(rows: list[dict[str, Any]], counts: Counter) -> str:
    w = max((len(r["cell"]) for r in rows), default=4)
    wc = max((len(r["check"]) for r in rows), default=5)
    lines = [f"{'cell':<{w}}  {'check':<{wc}}  verdict  detail", "-" * (w + wc + 20)]
    for r in rows:
        lines.append(
            f"{r['cell']:<{w}}  {r['check']:<{wc}}  {r['verdict']:<7}  {r['detail']}"
        )
    lines.append("-" * (w + wc + 20))
    lines.append(
        f"PASS {counts.get('PASS', 0)}  FAIL {counts.get('FAIL', 0)}  INFO {counts.get('INFO', 0)}"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "jsonl", nargs="+", type=Path, help="harness-v2 JSONL result file(s)"
    )
    ap.add_argument("--golden", type=Path, default=Path("evaluation/golden"))
    ap.add_argument("--golden-sm-mhz", type=float, default=GOLDEN_SM_MHZ)
    ap.add_argument(
        "--tol", type=float, default=1e-6, help="quality tolerance (gate 1)"
    )
    ap.add_argument(
        "--latency-tol", type=float, default=0.05, help="graph median ratio (gate 2)"
    )
    ap.add_argument(
        "--jaccard-official", type=float, default=0.99, help="O WP-6's official gate"
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="only records at this seed; -1 = all"
    )
    a = ap.parse_args(argv)
    rows, counts = gate(
        a.jsonl,
        a.golden,
        tol=a.tol,
        latency_tol=a.latency_tol,
        jaccard_official=a.jaccard_official,
        golden_sm_mhz=a.golden_sm_mhz,
        seed=None if a.seed < 0 else a.seed,
    )
    print(render(rows, counts))
    return 1 if counts.get("FAIL", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
