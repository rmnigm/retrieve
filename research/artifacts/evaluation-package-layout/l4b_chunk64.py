"""L4-b (linr-v2-backend-parity.md §4): one ``linr_v4`` cell with the quality chunk at 64
instead of ``run.QUALITY_CHUNK`` (16) — a measurement, not a change to the harness. Usage:
``python l4b_chunk64.py [bench run args...]``; the chunk is the only difference from ``bench run``."""

import sys

from bench import run
from bench.cli import main

run.QUALITY_CHUNK = 64
main(["run", *sys.argv[1:]])
