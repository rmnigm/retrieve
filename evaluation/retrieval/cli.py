"""``bench`` — the harness-v2 console script (H §3.4).

``bench run`` expands one ``(dataset, suite)`` through ``config.load_matrix`` (every option
below ``--suite`` is a narrow; ``--k`` / ``--bs`` / ``--mode`` replace the suite's lists and
make the records ``partial``) and runs the cells in *this* process via ``run.run``. Zero
cells — a ``--sweep`` typo, say — is an error (exit 1), never an empty success.
``bench campaign`` is the loop of H §3.4 / §8.2 K: one child process per ``(dataset, dim,
algo, backend)`` group, sequential, ``stdout+stderr`` to ``<out>/_logs/<group>.log``, a
one-line summary per child in ``<out>/_logs/campaign.log``, non-zero rc recorded and the
loop continues; ``--resume`` fills gaps. The child is ``python -m retrieval.cli run …`` on
the same interpreter (no second ``uv`` resolution). ``bench report`` lands in roadmap D4.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

import click
import yaml
from loguru import logger

from retrieval.algos import BACKENDS, FILTER_KINDS
from retrieval.config import load_matrix

EVAL_DIR = Path(__file__).resolve().parents[1]
SUITES = ("quality", "filter", "deep")


def _paths(config_dir: str, dataset: str) -> tuple[Path, Path]:
    cfg = Path(config_dir)
    cfg = cfg if cfg.is_absolute() else EVAL_DIR / cfg
    return cfg / f"{dataset}.yaml", cfg / "suites.yaml"


def _multi(v) -> list | None:
    return list(v) or None


@click.group()
def main() -> None:
    """Harness v2: `run` (one process), `campaign` (one child per group), `report` (D4)."""


@main.command()
@click.option("--dataset", required=True, help="config/<dataset>.yaml")
@click.option("--suite", required=True, help="a suite of suites.yaml (quality | filter | deep)")
@click.option("--dim", "dims", multiple=True, type=int)
@click.option("--algo", "algos", multiple=True)
@click.option("--backend", "backends", multiple=True, type=click.Choice(BACKENDS))
@click.option("--filter-kind", "filter_kinds", multiple=True, type=click.Choice(FILTER_KINDS))
@click.option("--sweep", "sweeps", multiple=True)
@click.option("--k", "ks", multiple=True, type=int)
@click.option("--bs", "batch_sizes", multiple=True, type=int)
@click.option("--seed", "seeds", multiple=True, type=int)
@click.option("--mode", "modes", multiple=True, type=click.Choice(("eager", "graph")))
@click.option("--skip-quality", is_flag=True)
@click.option("--skip-perf", is_flag=True)
@click.option("--profile", is_flag=True, help="torch.profiler top-8 kernels per eager variant")
@click.option("--out", default="results", show_default=True, help="results directory")
@click.option("--output", type=click.Path(), default=None, help="override the JSONL file")
@click.option("--resume/--force", default=True, help="skip cells already ok at this code_version")
@click.option("--expected-sm-mhz", default=1410, show_default=True, help="0 = no expectation")
@click.option("--config-dir", default="config", show_default=True)
def run(
    dataset, suite, dims, algos, backends, filter_kinds, sweeps, ks, batch_sizes, seeds, modes,
    skip_quality, skip_perf, profile, out, output, resume, expected_sm_mhz, config_dir,
) -> None:  # fmt: skip
    """Run the cells of one (dataset, suite) in this process."""
    from retrieval import run as run_mod  # noqa: PLC0415 — torch import only when running

    ds_yaml, suites_yaml = _paths(config_dir, dataset)
    jobs = load_matrix(
        ds_yaml,
        suites_yaml,
        suite,
        dims=_multi(dims),
        algos=_multi(algos),
        backends=_multi(backends),
        filter_kinds=_multi(filter_kinds),
        sweeps=_multi(sweeps),
        seeds=_multi(seeds),
        ks=_multi(ks),
        batch_sizes=_multi(batch_sizes),
    )
    if not jobs:
        raise click.ClickException(
            f"{dataset}/{suite}: the narrows select no cells (check --dim/--algo/--backend/"
            "--filter-kind/--sweep/--seed against the suite; see the log lines above)"
        )
    sha = hashlib.sha256(ds_yaml.read_bytes() + suites_yaml.read_bytes()).hexdigest()[:16]
    counts = run_mod.run(
        jobs,
        out_dir=Path(out),
        out_path=Path(output) if output else None,
        resume=resume,
        modes=_multi(modes) or run_mod.MODES,
        skip_quality=skip_quality,
        skip_perf=skip_perf,
        profile=profile,
        expected_sm_mhz=expected_sm_mhz or None,
        env_extra={"config_sha": sha},
    )
    click.echo(f"{dataset}/{suite}: {dict(counts)}")
    sys.exit(1 if counts["failed"] else 0)


@main.command()
@click.option("--suite", required=True, help="quality | filter | deep | all")
@click.option("--dataset", "datasets", multiple=True)
@click.option("--dim", "dims", multiple=True, type=int)
@click.option("--mode", "modes", multiple=True, type=click.Choice(("eager", "graph")))
@click.option("--skip-quality", is_flag=True, help="forwarded to every child")
@click.option("--skip-perf", is_flag=True, help="forwarded to every child")
@click.option("--profile", is_flag=True, help="forwarded to every child")
@click.option("--out", default="results", show_default=True)
@click.option("--resume/--force", default=True)
@click.option("--config-dir", default="config", show_default=True)
def campaign(
    suite, datasets, dims, modes, skip_quality, skip_perf, profile, out, resume, config_dir
) -> None:
    """One child process per (dataset, dim, algo, backend) group, in suite order."""
    out_dir = Path(out) if Path(out).is_absolute() else EVAL_DIR / out
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parity = out_dir / "_parity"
    worst = 0
    n_children = 0
    with open(log_dir / "campaign.log", "a") as summary:

        def say(line: str) -> None:
            click.echo(line)
            summary.write(line + "\n")
            summary.flush()

        say(f"=== campaign {suite} started {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
        for s in SUITES if suite == "all" else (suite,):
            _, suites_yaml = _paths(config_dir, "_")
            with open(suites_yaml) as f:
                listed = yaml.safe_load(f)[s]["datasets"]
            for ds in listed:
                if datasets and ds not in datasets:
                    continue
                ds_yaml, _ = _paths(config_dir, ds)
                jobs = load_matrix(ds_yaml, suites_yaml, s, dims=_multi(dims))
                groups = list(dict.fromkeys(j.group for j in jobs))
                if not groups:
                    say(f"{s} {ds}: no cells selected (dims {list(dims) or 'all'}) rc=1")
                    worst = max(worst, 1)
                last: tuple | None = None
                for d, dim, algo, backend in groups:
                    if (d, dim, algo) != last:  # the parity group closes: drop the spill file
                        shutil.rmtree(parity, ignore_errors=True)
                        last = (d, dim, algo)
                    cmd = [
                        sys.executable, "-m", "retrieval.cli", "run", "--dataset", d,
                        "--dim", str(dim), "--suite", s, "--algo", algo, "--backend", backend,
                        "--out", str(out_dir), "--config-dir", config_dir,
                        "--resume" if resume else "--force",
                    ]  # fmt: skip
                    for m in modes:
                        cmd += ["--mode", m]
                    flags = {
                        "skip-quality": skip_quality,
                        "skip-perf": skip_perf,
                        "profile": profile,
                    }
                    cmd += [f"--{name}" for name, on in flags.items() if on]
                    log = log_dir / f"{s}_{d}-d{dim}_{algo}_{backend}.log"
                    t0 = time.monotonic()
                    with open(log, "a") as lf:
                        lf.write(f"=== {' '.join(cmd)}\n")
                        lf.flush()
                        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=EVAL_DIR)
                    worst = max(worst, rc)
                    n_children += 1
                    say(
                        f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {s} {d} d{dim} {algo} "
                        f"{backend} rc={rc} {time.monotonic() - t0:.0f}s log={log.name}"
                    )
                shutil.rmtree(parity, ignore_errors=True)
        if n_children == 0:
            worst = max(worst, 1)
            say(f"no child was launched: --dataset {list(datasets)} / --dim {list(dims)} select "
                f"nothing in suite {suite!r}")  # fmt: skip
        say(f"=== campaign {suite} finished children={n_children} rc={worst}")
    sys.exit(worst)


@main.command()
@click.argument("results", required=False)
def report(results) -> None:
    """Tables and figures from the JSONL (roadmap D4, H §6 WP-6)."""
    logger.error("bench report is roadmap D4 (H §6 WP-6): flat.csv, tables, figures — not yet")
    sys.exit(2)


if __name__ == "__main__":
    main()

__all__ = ["main"]
