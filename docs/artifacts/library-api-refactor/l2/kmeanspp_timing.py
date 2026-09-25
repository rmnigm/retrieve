"""Plan L §11 risk check: greedy k-means++ seeding time at the sizes the plan names.

Times ``KMeans(init=...).fit`` with ``n_iter=0`` (the seeding alone) for both inits at
(N=200k, D=128, n_lists=1024) and (N=3M, D=128, n_lists=8192), plus one full ``n_iter=10``
fit at the 200k gate size, and writes ``kmeanspp_timing.json`` next to this file. Not a
recorded number (CLAUDE.md rule 1: clocks cannot be locked — ``sm_mhz`` is sampled).

    flock /workspace/gpu.lock -c 'uv run --no-sync python docs/artifacts/library-api-refactor/l2/kmeanspp_timing.py'
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch

from retrieve.indexing import KMeans


def sm_mhz() -> int | None:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    return int(out.stdout.split()[0]) if out.returncode == 0 else None


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    rows = []
    for n, d, n_lists, n_iter in [(200_000, 128, 1024, 0), (200_000, 128, 1024, 10), (3_000_000, 128, 8192, 0)]:
        g = torch.Generator(device="cuda").manual_seed(0)
        embs = torch.randn(n, d, generator=g, device="cuda")
        embs = embs / embs.norm(dim=1, keepdim=True)
        for init in ("random", "kmeans++"):
            km = KMeans(n_lists=n_lists, n_iter=n_iter, seed=0, init=init)
            km.fit(embs) if n <= 200_000 and n_iter == 0 else None  # warm the kernels once
            s = timed(lambda: km.fit(embs))
            rows.append({"n": n, "d": d, "n_lists": n_lists, "n_iter": n_iter, "init": init, "fit_s": s})
            print(rows[-1], flush=True)
        del embs
        torch.cuda.empty_cache()
    out = {"device": torch.cuda.get_device_name(), "sm_mhz": sm_mhz(), "torch": torch.__version__, "rows": rows}
    Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
