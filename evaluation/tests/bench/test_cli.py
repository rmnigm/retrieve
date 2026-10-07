"""CPU tests for ``bench.cli``: ``bench run`` through ``CliRunner`` on
the ``conftest.py`` fixture, ``bench campaign`` spawning one real child (the ``e2e1`` suite has
one ``(dataset, dim, algo, backend)`` group — a second child proves nothing the first does not,
and each costs a cold torch + retrieve import; ``--skip-quality --skip-perf``: full-length
latency windows are minutes on CPU fp16 matmuls), the per-child log and the ``campaign.log``
summary, the parity-directory cleanup, a restart mid-group keeping its spill, a faked
timed-out child, the zero-cells error, and ``bench report`` over the campaign's own records."""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

from bench import cli, inputs, measure, records, run
from bench.config import load_matrix
from eval_datasets import layout


def _records(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def test_cli_run_campaign_and_report(tiny_configs, tmp_path):
    ds, _ = tiny_configs
    out = tmp_path / "results"
    common = ["--config-dir", str(ds.parent), "--out", str(out)]
    r = CliRunner().invoke(
        cli.main,
        ["run", "--dataset", "tiny", "--suite", "e2e", "--algo", "linr_v1_filter_mask",
         "--sweep", "c0", "--mode", "eager", "--skip-perf", *common],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    recs = _records(out / "e2e" / "tiny-d8.jsonl")
    assert len(recs) == 1 and recs[0]["status"] == "partial" and recs[0]["perf"] is None
    assert len(recs[0]["env"]["config_sha"]) == 16 and "sm_mhz_idle" in recs[0]["env"]
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
    assert len(records.read_table(out / "results.parquet")) == 4  # bench run's 1 + these 3
    recs = _records(out / "e2e1" / "tiny-d8.jsonl")
    assert [(r["filter_kind"], r["sweep"]) for r in recs] == [
        ("none", "full_scan"), ("clause", "c0"), ("clause", "c0c1")
    ]  # fmt: skip
    assert all(r["partial_reasons"] == ["skip_quality", "skip_perf"] for r in recs)
    assert all(r["quality"] is None and r["perf"] is None and r["build_s"] > 0 for r in recs)
    r = CliRunner().invoke(cli.main, ["report", str(out), "--gate", "D1"])
    assert r.exit_code == 0, r.output
    assert (out / "report" / "results.parquet").exists()
    assert "NOT CITABLE" in r.output  # three partial records veto --gate
    assert "NOT CITABLE" in (out / "report" / "tables" / "tab-memory.tex").read_text()


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


def test_campaign_restart_keeps_the_groups_parity_spill(tiny_configs, tmp_path, monkeypatch):
    """A kill mid-group, then a new campaign process: its first child still compares against
    the spill the killed process wrote; another group's leftover spill is dropped. Children
    run in-process (no second CPU backend exists; ``official`` is torch relabelled)."""
    ds, suites = tiny_configs
    out = tmp_path / "results"
    clause = [j for j in load_matrix(ds, suites, "e2e", algos=["linr_v1_filter_mask"])
              if j.filter_kind == "clause"]  # fmt: skip
    with monkeypatch.context() as m:  # the killed process: its official backend finished
        real_build = run.build_module
        m.setattr(run, "build_module", lambda job, *a, **kw: real_build(
            dataclasses.replace(job, backend="torch"), *a, **kw))  # fmt: skip
        m.setattr(run.algos, "filter_backend", lambda b: "torch")
        run.run([dataclasses.replace(j, backend="official") for j in clause], out_dir=out,
                skip_perf=True)  # fmt: skip
    stale = out / "_parity" / "other-d8_linr_v4" / "0.npz"
    stale.parent.mkdir()
    stale.write_bytes(b"")
    seen = []

    def child(cmd, **kw):
        seen.append(stale.exists())
        algo = cmd[cmd.index("--algo") + 1]
        run.run(load_matrix(ds, suites, "e2e", algos=[algo]), out_dir=out,
                skip_perf=True)  # fmt: skip
        return 0

    monkeypatch.setattr(cli.subprocess, "call", child)
    r = CliRunner().invoke(
        cli.main,
        ["campaign", "--suite", "e2e", "--config-dir", str(ds.parent), "--out", str(out)],
    )
    assert r.exit_code == 0, r.output
    assert seen == [False, False]
    recs = [r for r in _records(out / "e2e" / "tiny-d8.jsonl") if r["backend"] == "torch"]
    parity = {(r["algo"], r["sweep"]): r["quality"]["parity"] for r in recs}
    assert parity == {
        ("linr_v1_filter_mask", "full_scan"): "reference",
        ("linr_v1_filter_mask", "c0"): "vs_official",
        ("linr_v1_filter_mask", "c0c1"): "vs_official",
        ("linr_v4", "full_scan"): "reference",
        ("linr_v4", "c0"): "reference",
        ("linr_v4", "c0c1"): "reference",
    }
    assert not (out / "_parity").exists()


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


def test_env_prints_provenance_and_clocks_as_json(monkeypatch):
    monkeypatch.setattr(measure, "official_commit", lambda: "abc123")
    r = CliRunner().invoke(cli.main, ["env"])
    assert r.exit_code == 0, r.output
    env = json.loads(r.output)
    assert set(env) == set(measure.provenance()) | set(measure.clocks())
    assert env["official_commit"] == "abc123" and "sm_mhz" in env and "code_version" in env


def test_cli_run_checkpoint_override_lands_as_inputs(tiny_configs, tmp_path, monkeypatch):
    """``--checkpoint`` replaces the YAML's and the written key carries its directory name.
    The encoder is faked with the fixture's text embeddings (no SASRec on CPU)."""
    ds, _ = tiny_configs
    data_dir = Path(ds.read_text().split("\n")[0].removeprefix("data_dir: "))
    ds.write_text(
        ds.read_text().replace("content_dir: content", f"checkpoint: {data_dir}/ck/a/x.pt")
    )
    seen = []

    def fake_encode(ckpt, data_dir, **kw):
        seen.append(ckpt)
        items = layout.load_text_items(data_dir / "content", kw["device"])
        return (items, *layout.load_text_queries(data_dir, data_dir / "content", items.shape[1]))

    monkeypatch.setattr(inputs, "encode_split", fake_encode)
    out = tmp_path / "results"
    r = CliRunner().invoke(
        cli.main,
        ["run", "--dataset", "tiny", "--suite", "e2e", "--algo", "linr_v1_filter_mask",
         "--sweep", "c0", "--mode", "eager", "--skip-perf", "--config-dir", str(ds.parent),
         "--out", str(out), "--checkpoint", f"{data_dir}/ck/pinned-d{{dim}}/x.pt"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert seen == [data_dir / "ck" / "pinned-d8" / "x.pt"]
    (rec,) = _records(out / "e2e" / "tiny-d8.jsonl")
    assert rec["inputs"] == "pinned-d8"
