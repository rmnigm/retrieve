"""D1-a driver: `bench campaign`'s loop (cli.campaign) with the one narrow it has no flag for,
`--seed 0`. Same process boundary, same log layout, same parity-spill lifetime."""

from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[4] / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from bench.config import load_matrix  # noqa: E402

SUITE = "filter"
DIM = 128
SEED = 0
DATASETS = ("goodreads", "arxiv")
BACKENDS = ("triton", "torch", "official")


def main() -> int:
    out_dir = EVAL_DIR / "results"
    log_dir = out_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parity = out_dir / "_parity"
    worst = 0
    with open(log_dir / "campaign.log", "a") as summary:

        def say(line: str) -> None:
            print(line, flush=True)
            summary.write(line + "\n")
            summary.flush()

        say(f"=== d1a {SUITE} started {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%M:%SZ}")
        for ds in DATASETS:
            jobs = load_matrix(
                EVAL_DIR / "config" / f"{ds}.yaml",
                EVAL_DIR / "config" / "suites.yaml",
                SUITE,
                dims=[DIM],
                seeds=[SEED],
                backends=list(BACKENDS),
            )
            groups = list(dict.fromkeys(j.group for j in jobs))
            last: tuple | None = None
            for d, dim, algo, backend in groups:
                if (d, dim, algo) != last:
                    shutil.rmtree(parity, ignore_errors=True)
                    last = (d, dim, algo)
                cmd = [
                    sys.executable, "-m", "bench.cli", "run", "--dataset", d,
                    "--dim", str(dim), "--suite", SUITE, "--algo", algo, "--backend", backend,
                    "--seed", str(SEED), "--out", str(out_dir), "--config-dir", "config",
                    "--resume",
                ]  # fmt: skip
                log = log_dir / f"{SUITE}_{d}-d{dim}_{algo}_{backend}.log"
                t0 = time.monotonic()
                with open(log, "a") as lf:
                    lf.write(f"=== {' '.join(cmd)}\n")
                    lf.flush()
                    rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=EVAL_DIR)
                worst = max(worst, rc)
                say(
                    f"{dt.datetime.now(dt.timezone.utc):%H:%M:%S} {SUITE} {d} d{dim} {algo} "
                    f"{backend} rc={rc} {time.monotonic() - t0:.0f}s {log.name}"
                )
        shutil.rmtree(parity, ignore_errors=True)
        say(f"=== d1a {SUITE} done rc={worst}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
