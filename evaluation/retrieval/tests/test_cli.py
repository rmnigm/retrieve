"""CPU tests for ``retrieval.cli`` (harness v2 WP-3): ``bench run`` through ``CliRunner`` on
the ``conftest.py`` fixture, ``bench campaign`` spawning one real child (the ``e2e1`` suite has
one ``(dataset, dim, algo, backend)`` group — a second child proves nothing the first does not,
and each costs a cold torch + retrieve import; ``--skip-quality --skip-perf``: full-length
latency windows are minutes on CPU fp16 matmuls), the per-child log and the ``campaign.log``
summary, the parity-directory cleanup when the algo group closes, a faked timed-out child,
the zero-cells error, and the ``bench report`` D4 stub."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from retrieval import cli


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_cli_run_campaign_and_report(tiny_configs, tmp_path):
    ds, _ = tiny_configs
    out = tmp_path / "results"
    common = ["--config-dir", str(ds.parent), "--out", str(out)]
    r = CliRunner().invoke(
        cli.main,
        ["run", "--dataset", "tiny", "--suite", "e2e", "--algo", "linr_v1_filter_mask",
         "--sweep", "c0", "--mode", "eager", "--skip-perf", "--expected-sm-mhz", "0", *common],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert len(recs) == 1 and recs[0]["status"] == "partial" and recs[0]["perf"] is None
    assert len(recs[0]["env"]["config_sha"]) == 16 and recs[0]["env"]["expected_sm_mhz"] is None
    # campaign: one child per (dataset, dim, algo, backend) group — the e2e1 suite has one —
    # run for real in a subprocess, both modes (graph is the CPU null entry).
    r = CliRunner().invoke(
        cli.main,
        ["campaign", "--suite", "e2e1", "--dataset", "tiny", "--skip-quality", "--skip-perf",
         *common],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    logs = sorted(p.name for p in (out / "_logs").iterdir())
    assert logs == ["campaign.log", "e2e1_tiny-d8_linr_v1_filter_mask_torch.log"]
    assert (out / "_logs" / logs[1]).read_text().startswith("=== ")  # the command line first
    summary = (out / "_logs" / "campaign.log").read_text()
    assert summary.count(" rc=0 ") == 1 and "finished children=1 rc=0" in summary
    assert not (out / "_parity").exists()  # dropped when the algo group closed
    recs = _records(out / "e2e1" / "tiny-d8.jsonl")
    assert [(r["filter_kind"], r["sweep"]) for r in recs] == [
        ("none", "full_scan"), ("clause", "c0"), ("clause", "c0c1")
    ]  # fmt: skip
    assert all(r["partial_reasons"] == ["skip_quality", "skip_perf"] for r in recs)
    assert all(r["quality"] is None and r["perf"] is None and r["build_s"] > 0 for r in recs)
    r = CliRunner().invoke(cli.main, ["report", str(out)])
    assert r.exit_code == 2


def test_campaign_records_a_timed_out_child(tiny_configs, tmp_path, monkeypatch):
    """``--timeout`` kills a hung child: ``rc=timeout`` in the summary, exit code 124, the
    loop continues to the next group. The child is faked (no CPU-shaped way to hang one)."""
    ds, _ = tiny_configs
    out = tmp_path / "results"
    seen = []

    def hung(cmd, *, timeout, **kw):
        seen.append(timeout)
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(cli.subprocess, "call", hung)
    r = CliRunner().invoke(
        cli.main,
        ["campaign", "--suite", "e2e", "--timeout", "0.5", "--config-dir", str(ds.parent),
         "--out", str(out)],
    )  # fmt: skip
    assert r.exit_code == cli.RC_TIMEOUT == 124, r.output
    assert seen == [1800.0, 1800.0]  # two groups, both attempted
    summary = (out / "_logs" / "campaign.log").read_text()
    assert summary.count(" rc=timeout ") == 2 and "finished children=2 rc=124" in summary
    child_log = (out / "_logs" / "e2e_tiny-d8_linr_v4_torch.log").read_text()
    assert "killed after 0.5 h" in child_log


def test_zero_cells_is_an_error(tiny_configs, tmp_path):
    """A ``--sweep`` typo (or a ``--dataset`` that matches nothing) must not exit 0 with an
    empty summary; nothing is run or written."""
    ds, _ = tiny_configs
    out = tmp_path / "results"
    common = ["--config-dir", str(ds.parent), "--out", str(out)]
    r = CliRunner().invoke(
        cli.main, ["run", "--dataset", "tiny", "--suite", "e2e", "--sweep", "c0_typo", *common]
    )
    assert r.exit_code == 1 and "select no cells" in r.output
    assert not (out / "e2e").exists()
    r = CliRunner().invoke(cli.main, ["campaign", "--suite", "e2e", "--dataset", "nope", *common])
    assert r.exit_code == 1 and "no child was launched" in r.output
    r = CliRunner().invoke(cli.main, ["campaign", "--suite", "e2e", "--dim", "999", *common])
    assert r.exit_code == 1 and "no cells selected" in r.output
    assert list((out / "_logs").iterdir()) == [out / "_logs" / "campaign.log"]
