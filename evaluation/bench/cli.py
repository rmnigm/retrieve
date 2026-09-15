"""``bench`` — the harness console script (H §3.4): ``run`` (one process), ``campaign`` (one
child per group), ``check`` (the on-disk layout), ``upload`` (results → HF mirror),
``report`` (tables and figures from the records; roadmap D4).

``bench run`` expands one ``(dataset, suite)`` through ``config.load_matrix`` (every option
below ``--suite`` is a narrow; ``--k`` / ``--bs`` / ``--mode`` replace the suite's lists and
make the records ``partial``) and runs the cells in *this* process via ``run.run``. Zero
cells — a ``--sweep`` typo, say — is an error (exit 1), never an empty success.
``bench campaign`` is the loop of H §3.4 / §8.2 K: one child process per ``(dataset, dim,
algo, backend)`` group, sequential, ``stdout+stderr`` to ``<out>/_logs/<group>.log``, a
one-line summary per child in ``<out>/_logs/campaign.log``, non-zero rc recorded and the
loop continues; ``--resume`` fills gaps. A child that outlives ``--timeout`` hours is killed
and recorded as ``rc=timeout`` (exit code 124). The child is ``python -m bench.cli run …``
on the same interpreter (no second ``uv`` resolution).
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

from bench.algos import BACKENDS, FILTER_KINDS
from bench.config import load_dataset, load_matrix
from bench.report import report
from bench.upload import upload

EVAL_DIR = Path(__file__).resolve().parents[1]
SUITES = ("quality", "filter", "deep")
RC_TIMEOUT = 124  # the ``timeout(1)`` convention


def _paths(config_dir: str, dataset: str) -> tuple[Path, Path]:
    cfg = Path(config_dir)
    cfg = cfg if cfg.is_absolute() else EVAL_DIR / cfg
    return cfg / f"{dataset}.yaml", cfg / "suites.yaml"


def _multi(v) -> list | None:
    return list(v) or None


@click.group()
def main() -> None:
    """The retrieval benchmark: run, campaign, check, upload, report."""


main.add_command(upload)
main.add_command(report)


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
@click.option("--config-dir", default="config", show_default=True)
def run(
    dataset, suite, dims, algos, backends, filter_kinds, sweeps, ks, batch_sizes, seeds, modes,
    skip_quality, skip_perf, profile, out, output, resume, config_dir,
) -> None:  # fmt: skip
    """Run the cells of one (dataset, suite) in this process."""
    from bench import run as run_mod  # noqa: PLC0415 — torch import only when running

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
@click.option(
    "--timeout",
    "timeout_h",
    default=6.0,
    show_default=True,
    help="hours per child before it is killed and recorded as rc=timeout",
)
def campaign(
    suite, datasets, dims, modes, skip_quality, skip_perf, profile, out, resume, config_dir,
    timeout_h,
) -> None:  # fmt: skip
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
                        sys.executable, "-m", "bench.cli", "run", "--dataset", d,
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
                        try:
                            rc = subprocess.call(
                                cmd,
                                stdout=lf,
                                stderr=subprocess.STDOUT,
                                cwd=EVAL_DIR,
                                timeout=timeout_h * 3600,
                            )
                        except subprocess.TimeoutExpired:  # the child was killed
                            lf.write(f"=== killed after {timeout_h} h (--timeout)\n")
                            rc = RC_TIMEOUT
                    worst = max(worst, rc)
                    n_children += 1
                    say(
                        f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {s} {d} d{dim} {algo} "
                        f"{backend} rc={'timeout' if rc == RC_TIMEOUT else rc} "
                        f"{time.monotonic() - t0:.0f}s log={log.name}"
                    )
                shutil.rmtree(parity, ignore_errors=True)
        if n_children == 0:
            worst = max(worst, 1)
            say(f"no child was launched: --dataset {list(datasets)} / --dim {list(dims)} select "
                f"nothing in suite {suite!r}")  # fmt: skip
        say(f"=== campaign {suite} finished children={n_children} rc={worst}")
    sys.exit(worst)


@main.command()
@click.option("--dataset", required=True, help="config/<dataset>.yaml")
@click.option("--dim", "dims", multiple=True, type=int, help="default: every dim of the file")
@click.option("--config-dir", default="config", show_default=True)
def check(dataset, dims, config_dir) -> None:
    """Validate a dataset's on-disk layout (eval_datasets.layout.validate_layout) per dim."""
    from eval_datasets.layout import validate_layout  # noqa: PLC0415 — torch import only here

    ds_yaml, _ = _paths(config_dir, dataset)
    with open(ds_yaml) as f:
        all_dims = yaml.safe_load(f)["dims"]
    bad = 0
    for dim in _multi(dims) or all_dims:
        ds = load_dataset(ds_yaml, dim)
        problems = validate_layout(ds.data_dir, ds.content_dir)
        bad += len(problems)
        click.echo(f"{dataset} d{dim}: {'ok' if not problems else ''}")
        for p in problems:
            click.echo(f"  {p}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

__all__ = ["main"]
