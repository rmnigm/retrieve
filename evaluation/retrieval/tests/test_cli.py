"""CPU tests for ``retrieval.cli`` (harness v2 WP-3): ``bench run`` through ``CliRunner`` on
the ``conftest.py`` fixture, ``bench campaign`` spawning one real child per ``(dataset, dim,
algo, backend)`` group (``--skip-perf``: full-length latency windows are minutes on CPU fp16
matmuls), the per-child logs and the ``campaign.log`` summary, the parity-directory cleanup
when the algo group closes, and the ``bench report`` D4 stub."""

from __future__ import annotations

import json
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
    # campaign: one child per (dataset, dim, algo, backend) group, run for real in a
    # subprocess (--skip-perf: full-length latency windows are minutes on CPU fp16 matmuls).
    r = CliRunner().invoke(
        cli.main,
        [
            "campaign",
            "--suite",
            "e2e",
            "--dataset",
            "tiny",
            "--mode",
            "eager",
            "--skip-perf",
            *common,
        ],
    )
    assert r.exit_code == 0, r.output
    logs = sorted(p.name for p in (out / "_logs").iterdir())
    assert logs == [
        "campaign.log", "e2e_tiny-d8_linr_v1_filter_mask_torch.log", "e2e_tiny-d8_linr_v4_torch.log"
    ]  # fmt: skip
    summary = (out / "_logs" / "campaign.log").read_text()
    assert summary.count(" rc=0 ") == 2 and "finished rc=0" in summary
    assert not (out / "_parity").exists()  # dropped when the last algo group closed
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert len(recs) == 1 + 6  # the partial cell re-ran; 3 + 3 cells (none, c0, c0c1) per algo
    assert {r["status"] for r in recs[1:]} == {"partial"}
    r = CliRunner().invoke(cli.main, ["report", str(out)])
    assert r.exit_code == 2
