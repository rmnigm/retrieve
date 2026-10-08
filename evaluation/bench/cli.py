"""``bench`` — the harness console script (H §3.4): ``run`` (one process), ``campaign`` (one
child per group), ``check`` (the on-disk layout), ``upload`` / ``fetch`` (results ↔ the HF Hub),
``report`` (tables and figures from the records; roadmap D4), ``env`` (the provenance and
clock block as JSON, for a validation record).

``bench run`` expands one ``(dataset, suite)`` through ``config.load_matrix`` (every option
below ``--suite`` is a narrow; ``--k`` / ``--bs`` / ``--mode`` replace the suite's lists and
make the records ``partial``) and runs the cells in *this* process via ``run.run``. Zero
cells — a ``--sweep`` typo, say — is an error (exit 1), never an empty success.
``bench campaign`` is the loop of H §3.4 / §8.2 K: one child process per ``(dataset, dim,
algo, backend)`` group, sequential, ``stdout+stderr`` to ``<out>/_logs/<group>.log``, a
one-line summary per child in ``<out>/_logs/campaign.log``, non-zero rc recorded and the
loop continues; ``--resume`` fills gaps; the records are aggregated into
``<out>/results.parquet`` at the end. A child that outlives ``--timeout`` hours is killed
and recorded as ``rc=timeout`` (exit code 124). The child is ``python -m bench.cli run …``
on the same interpreter (no second ``uv`` resolution).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import TextIO

import click
import yaml

import bench
from bench import measure, records
from bench import run as run_mod
from bench.algos import BACKENDS, FILTER_KINDS
from bench.config import interleave_units, load_dataset, load_matrix
from bench.report import report
from bench.upload import fetch, upload
from eval_datasets.layout import validate_layout

EVAL_DIR = Path(__file__).resolve().parents[1]
SUITES = ("filter", "deep", "codesign")
RC_TIMEOUT = 124  # the ``timeout(1)`` convention
# Sized for the largest group (arxiv `deep` silvertorch ~10 h; pubmed `filter` groups past 6 h):
# the timeout catches hangs, it does not budget work.
TIMEOUT_H = 48.0


def _say(summary: TextIO, line: str) -> None:
    click.echo(line)
    summary.write(line + "\n")
    summary.flush()


def _paths(config_dir: str, dataset: str) -> tuple[Path, Path]:
    cfg = Path(config_dir)
    cfg = cfg if cfg.is_absolute() else EVAL_DIR / cfg
    return cfg / f"{dataset}.yaml", cfg / "suites.yaml"


def _multi(v) -> list | None:
    return list(v) or None


def _children(jobs: list, interleave: bool) -> list[tuple[str, int, tuple, tuple]]:
    """``(dataset, dim, algos, backends)`` per campaign child, in suite order: one per
    ``Job.group``, or with ``interleave`` one per connected set of groups that an interleave
    unit spans (its ``--algo`` × ``--backend`` narrows must select exactly that set)."""
    if not interleave:
        return [(d, dim, (a,), (b,)) for d, dim, a, b in dict.fromkeys(j.group for j in jobs)]
    root: dict[tuple, tuple] = {j.group: j.group for j in jobs}

    def find(g: tuple) -> tuple:
        while root[g] != g:
            g = root[g]
        return g

    units = interleave_units(jobs)
    for _, unit in units:
        for j in unit[1:]:
            root[find(j.group)] = find(unit[0].group)
    comps: dict[tuple, list[tuple]] = {}
    for g in dict.fromkeys(j.group for j in jobs):
        comps.setdefault(find(g), []).append(g)
    out = []
    for gs in comps.values():
        algos = tuple(dict.fromkeys(g[2] for g in gs))
        backends = tuple(dict.fromkeys(g[3] for g in gs))
        if len(algos) * len(backends) != len(gs):
            raise click.ClickException(f"interleave groups {gs} are not one --algo x --backend")
        out.append((gs[0][0], gs[0][1], algos, backends))
    return out


@click.group()
def main() -> None:
    """The retrieval benchmark: run, campaign, check, upload, fetch, report, env."""


main.add_command(upload)
main.add_command(fetch)
main.add_command(report)


@main.command()
@click.option("--dataset", required=True, help="config/<dataset>.yaml")
@click.option("--suite", required=True, help="a suite of suites.yaml (filter | deep | h2h | ...)")
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
@click.option(
    "--interleave", is_flag=True, help="time each suite comparison group round-robin (ABAB)"
)
@click.option("--out", default="results", show_default=True, help="results directory")
@click.option("--output", type=click.Path(), default=None, help="override the JSONL file")
@click.option("--resume/--force", default=True, help="skip cells already ok at this code_version")
@click.option("--config-dir", default="config", show_default=True)
@click.option(
    "--checkpoint",
    default=None,
    help="replace the dataset's checkpoint ({dim} templated); recorded as the key's `inputs`",
)
def run(
    dataset, suite, dims, algos, backends, filter_kinds, sweeps, ks, batch_sizes, seeds, modes,
    skip_quality, skip_perf, profile, interleave, out, output, resume, config_dir, checkpoint,
) -> None:  # fmt: skip
    """Run the cells of one (dataset, suite) in this process."""
    cache = measure.inductor_cache_dir(measure.code_version(), bench.GIVEN_INDUCTOR_CACHE)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache
    click.echo(f"inductor cache: {cache}")
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
        checkpoint=checkpoint,
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
        interleave=interleave,
        env_extra={"config_sha": sha},
    )
    click.echo(f"{dataset}/{suite}: {dict(counts)}")
    sys.exit(1 if counts["failed"] else 0)


@main.command()
@click.option("--suite", required=True, help="filter | deep | codesign | all")
@click.option("--dataset", "datasets", multiple=True)
@click.option("--dim", "dims", multiple=True, type=int)
@click.option("--mode", "modes", multiple=True, type=click.Choice(("eager", "graph")))
@click.option("--skip-quality", is_flag=True, help="forwarded to every child")
@click.option("--skip-perf", is_flag=True, help="forwarded to every child")
@click.option("--profile", is_flag=True, help="forwarded to every child")
@click.option(
    "--interleave",
    is_flag=True,
    help="forwarded; the groups of a suite comparison share one child",
)
@click.option("--out", default="results", show_default=True)
@click.option("--resume/--force", default=True)
@click.option("--config-dir", default="config", show_default=True)
@click.option(
    "--timeout",
    "timeout_h",
    default=TIMEOUT_H,
    show_default=True,
    help="hours per child before it is killed and recorded as rc=timeout",
)
def campaign(
    suite, datasets, dims, modes, skip_quality, skip_perf, profile, interleave, out, resume,
    config_dir, timeout_h,
) -> None:  # fmt: skip
    """One child process per (dataset, dim, algo, backend) group, in suite order; with
    --interleave the groups an interleave unit spans share one child."""
    out_dir = Path(out) if Path(out).is_absolute() else EVAL_DIR / out
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parity = out_dir / "_parity"
    # The children key their own inductor cache; hand them the caller's value, not torch's default.
    child_env = {k: v for k, v in os.environ.items() if k != "TORCHINDUCTOR_CACHE_DIR"}
    if bench.GIVEN_INDUCTOR_CACHE:
        child_env["TORCHINDUCTOR_CACHE_DIR"] = bench.GIVEN_INDUCTOR_CACHE
    worst = 0
    n_children = 0
    with open(log_dir / "campaign.log", "a") as summary:
        say = partial(_say, summary)
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
                children = _children(jobs, interleave)
                if not children:
                    say(f"{s} {ds}: no cells selected (dims {list(dims) or 'all'}) rc=1")
                    worst = max(worst, 1)
                for d, dim, algos, backends in children:
                    # on disk, not in memory: a restart mid-group keeps the earlier process's spill
                    keep = {run_mod.parity_group(d, dim, a) for a in algos}
                    for stale in parity.glob("*"):
                        if stale.name not in keep:
                            shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
                    cmd = [
                        sys.executable, "-m", "bench.cli", "run", "--dataset", d,
                        "--dim", str(dim), "--suite", s,
                        *(x for a in algos for x in ("--algo", a)),
                        *(x for b in backends for x in ("--backend", b)),
                        "--out", str(out_dir), "--config-dir", config_dir,
                        "--resume" if resume else "--force",
                    ]  # fmt: skip
                    for m in modes:
                        cmd += ["--mode", m]
                    flags = {
                        "skip-quality": skip_quality,
                        "skip-perf": skip_perf,
                        "profile": profile,
                        "interleave": interleave,
                    }
                    cmd += [f"--{name}" for name, on in flags.items() if on]
                    algo, backend = "+".join(algos), "+".join(backends)
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
                                env=child_env,
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
        if records.record_files(out_dir):
            say(f"=== aggregated {records.aggregate(out_dir)}")
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


@main.command()
def env() -> None:
    """Print this box's provenance and one nvidia-smi clock sample as JSON."""
    click.echo(json.dumps(measure.provenance() | measure.clocks(), indent=2))


if __name__ == "__main__":
    main()

__all__ = ["main"]
