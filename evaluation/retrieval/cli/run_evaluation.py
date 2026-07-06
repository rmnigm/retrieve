"""Orchestrate per-algo retrieval evals across one or more YAML configs.

For each (config, algo) pair, spawns `uv run evaluate` as a fresh subprocess
so torch.compile / Triton autotune / CUDA-graph state is cleared between
algos. The `quality` and `param-sweeps` eval-types additionally invoke
stage_results after each config to restructure outputs into the staged
layout.

Usage:
    uv run run-evaluation --eval-type {filter|quality|param-sweeps} [--resume|--force]
    uv run run-evaluation CONFIG [CONFIG ...] [--stage] [--resume|--force]
                          [-- extra args forwarded to `evaluate`]
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from retrieval.config import load_raw_config
from retrieval.results_io import load_rows

EVAL_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOGDIR = EVAL_DIR / "results" / "_runlogs"

EVAL_TYPES: dict[str, dict] = {
    "filter": {
        "configs": [f"config/{d}/d{n}-filter.yaml"
                    for d in ("arxiv", "goodreads") for n in (64, 128, 256)],
        "stage": False,
    },
    "quality": {
        "configs": [f"config/{d}/d{n}-quality.yaml"
                    for d in ("arxiv", "goodreads") for n in (64, 128, 256)],
        "stage": True,
    },
    "param-sweeps": {
        # linr_v4 is paramless — no curve to trace.
        "configs": [
            "config/deep_sweeps/arxiv-d128-silvertorch.yaml",
            "config/deep_sweeps/goodreads-d128-linr_v3.yaml",
        ],
        "stage": True,
    },
}


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except FileNotFoundError:
        return "(no nvidia-smi)"
    return out.splitlines()[0] if out else "(none)"


def _cfg_name(cfg: Path) -> str:
    """`config/arxiv/d64-filter.yaml` -> `arxiv_d64-filter`."""
    rel = str(cfg.resolve().relative_to(EVAL_DIR).with_suffix(""))
    if rel.startswith("config/"):
        rel = rel[len("config/"):]
    return rel.replace("/", "_")


def _load_cfg(path: Path) -> tuple[Path, list[str]]:
    data = load_raw_config(path)
    if data.get("output") is None:
        raise SystemExit(
            f"{path}: config must set output: (a directory) to be orchestrated"
        )
    out = Path(data["output"])
    return (out if out.is_absolute() else EVAL_DIR / out), list(data["algorithms"])


def _is_complete(json_path: Path) -> bool:
    """A non-empty list on disk counts as a finished run."""
    rows = load_rows(json_path)
    return rows is not None and len(rows) > 0


def _expand_configs(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        matches = sorted(glob.glob(a))
        if matches:
            out.extend(Path(m) for m in matches)
        else:
            out.append(Path(a))
    return out


def _format_dur(seconds: float) -> str:
    s = int(seconds)
    return f"{s}s ({s // 3600}h{(s % 3600) // 60}m)"


@dataclass
class LogSinks:
    """Tees orchestrator output to per-config log, full log, summary, and stdout.

    `tee(text)` writes to the active per-config log (if inside `per_config`),
    full.log, and stdout. With `summary=True` it also appends to SUMMARY.txt.
    """
    logdir: Path
    fresh: bool
    full_f: IO[str] = field(init=False)
    summary_f: IO[str] = field(init=False)
    per_f: IO[str] | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        mode = "w" if self.fresh else "a"
        self.full_f = open(self.logdir / "full.log", mode)
        self.summary_f = open(self.logdir / "SUMMARY.txt", mode)

    def close(self) -> None:
        self.full_f.close()
        self.summary_f.close()

    def tee(self, text: str, *, summary: bool = False) -> None:
        print(text)
        self._write(self.full_f, text + "\n")
        if summary:
            self._write(self.summary_f, text + "\n")
        if self.per_f is not None:
            self._write(self.per_f, text + "\n")

    def stream(self, line: str) -> None:
        """Write a raw line (already newline-terminated) to all active sinks."""
        sys.stdout.write(line)
        sys.stdout.flush()
        self._write(self.full_f, line)
        if self.per_f is not None:
            self._write(self.per_f, line)

    @staticmethod
    def _write(f: IO[str], s: str) -> None:
        f.write(s)
        f.flush()

    @contextmanager
    def per_config(self, cfg: Path) -> Iterator[Path]:
        """Open `<cfg_name>.log` and manage the `current.log` symlink for one config."""
        log_path = self.logdir / f"{_cfg_name(cfg)}.log"
        current = self.logdir / "current.log"
        current.unlink(missing_ok=True)
        current.symlink_to(log_path.name)
        with open(log_path, "w" if self.fresh else "a") as f:
            self.per_f = f
            try:
                yield log_path
            finally:
                self.per_f = None


def _stream_subprocess(cmd: list[str], name: str, sinks: LogSinks) -> int:
    """Banner, run cmd tee-ing its output, warn on non-zero rc."""
    sinks.tee(f"=== {_utcnow()} {name} ===")
    proc = subprocess.Popen(
        cmd, cwd=EVAL_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        sinks.stream(line)
    rc = proc.wait()
    if rc != 0:
        sinks.tee(f"warn: {name} exited rc={rc}")
    return rc


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    if "--" in argv:
        i = argv.index("--")
        own, extra = argv[:i], argv[i + 1:]
    else:
        own, extra = argv, []

    p = argparse.ArgumentParser(
        prog="run-evaluation",
        description="Orchestrate per-algo retrieval evals across YAML configs.",
    )
    p.add_argument("configs", nargs="*",
                   help="YAML config paths or globs (omit when using --eval-type)")
    p.add_argument("--eval-type", choices=sorted(EVAL_TYPES.keys()),
                   help="Named eval-type preset (mutually exclusive with positional configs)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true",
                      help="Skip (config, algo) whose output JSON exists and parses "
                           "as a non-empty list")
    mode.add_argument("--force", action="store_true",
                      help="Overwrite existing outputs without prompting")
    p.add_argument("--stage", action="store_true",
                   help="Run stage_results after each config "
                        "(off for ad-hoc; eval-types set their own default)")
    p.add_argument("--logdir", type=Path, default=DEFAULT_LOGDIR,
                   help=f"Log directory (default: {DEFAULT_LOGDIR.relative_to(EVAL_DIR)})")

    args = p.parse_args(own)
    if bool(args.eval_type) == bool(args.configs):
        p.error("must pass either --eval-type NAME or positional CONFIG(s), not both/neither")
    return args, extra


def _plan_work(
    configs: list[Path], resume: bool, force: bool,
) -> list[tuple[Path, Path, list[tuple[str, Path]]]]:
    """Resolve each config → (cfg, out_dir, items). Fail fast on collisions."""
    work = []
    collisions: list[Path] = []
    for cfg in configs:
        out_dir, algos = _load_cfg(cfg)
        items = [(a, out_dir / f"{a}.json") for a in algos]
        work.append((cfg, out_dir, items))
        if not resume and not force:
            collisions.extend(jp for _, jp in items if _is_complete(jp))
    if collisions:
        lines = [f"error: {len(collisions)} output(s) already exist. "
                 "Pass --resume to skip or --force to overwrite:"]
        lines.extend(f"  {jp}" for jp in collisions)
        sys.exit("\n".join(lines))
    return work


def _run_one_config(
    cfg: Path, out_dir: Path, items: list[tuple[str, Path]],
    *, stage: bool, resume: bool, extra: list[str], sinks: LogSinks,
) -> int:
    """Run all algos in one config, optionally stage, write summary line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sinks.tee(f"=== {_utcnow()} starting {cfg} ===", summary=True)
    start = time.monotonic()
    cfg_rc = 0

    with sinks.per_config(cfg) as per_log_path:
        try:
            for algo, json_path in items:
                if resume and _is_complete(json_path):
                    sinks.tee(f"resume: skipping {algo} ({json_path} already complete)")
                    continue
                cmd = ["uv", "run", "evaluate",
                       "--config", str(cfg), "--algo", algo, "--output", str(json_path),
                       *extra]
                cfg_rc = max(cfg_rc, _stream_subprocess(cmd, algo, sinks))

            if stage:
                cfg_rc = max(cfg_rc, _stream_subprocess(
                    ["uv", "run", "stage-results", str(cfg)], f"stage_results {cfg}", sinks,
                ))
        finally:
            dur = time.monotonic() - start
            sinks.tee(
                f"{cfg}  exit={cfg_rc}  duration={_format_dur(dur)}  log={per_log_path}",
                summary=True,
            )
    return cfg_rc


def main() -> int:
    args, extra = _parse_args(sys.argv[1:])
    if args.eval_type:
        spec = EVAL_TYPES[args.eval_type]
        configs = [EVAL_DIR / c for c in spec["configs"]]
        stage = spec["stage"]
    else:
        configs = _expand_configs(args.configs)
        stage = args.stage
    work = _plan_work(configs, args.resume, args.force)

    args.logdir.mkdir(parents=True, exist_ok=True)
    sinks = LogSinks(args.logdir, fresh=not args.resume)

    if args.resume:
        sinks.tee("", summary=True)
        sinks.tee(f"--- resume run @ {_utcnow()} ---", summary=True)
    else:
        sinks.tee(f"started at:  {_utcnow()}", summary=True)
        sinks.tee(f"host:        {socket.gethostname()}", summary=True)
        sinks.tee(f"gpu:         {_gpu_name()}", summary=True)
        sinks.tee(f"eval-type:   {args.eval_type or '(ad-hoc)'}", summary=True)
        sinks.tee(f"stage:       {stage}", summary=True)
        sinks.tee("", summary=True)

    rc = 0
    try:
        for cfg, out_dir, items in work:
            rc = max(rc, _run_one_config(
                cfg, out_dir, items, stage=stage, resume=args.resume, extra=extra, sinks=sinks,
            ))
    except KeyboardInterrupt:
        rc = 130

    (args.logdir / "current.log").unlink(missing_ok=True)
    sinks.tee("", summary=True)
    sinks.tee(f"finished at: {_utcnow()}", summary=True)
    sinks.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
