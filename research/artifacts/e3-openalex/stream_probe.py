"""Throughput of `openalex convert`'s per-row-group work (projected S3 read + `stage_table`) in
spawned worker processes, as the real run does it, on randomly chosen row groups of the
pinned manifest — the input to the full-stream wall-time estimate. Nothing is written.

    cd evaluation && RETRIEVE_DATA_ROOT=... PYTHONPATH=. \
        python ../docs/artifacts/e3-openalex/stream_probe.py <workers> <row_groups> <sample_rate>
"""

import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow.parquet as pq

from eval_datasets.etl import openalex


def job(key: str, rg: int, sample_rate: float) -> dict:
    pf = pq.ParquetFile(key, filesystem=openalex._s3())
    t0 = time.monotonic()
    tbl = pf.read_row_group(rg, columns=openalex.COLUMNS)
    t1 = time.monotonic()
    _, stats = openalex.stage_table(tbl, max_year=2026, sample_rate=sample_rate, seed=0)
    return {
        "bytes": openalex._projected_bytes(pf.metadata, range(rg, rg + 1)),
        "rows": stats["rows"],
        "kept": stats["after_abstract"],
        "t_read": t1 - t0,
        "t_stage": time.monotonic() - t1,
    }


if __name__ == "__main__":
    workers, n_jobs, rate = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
    files = openalex.works_files(openalex._load_manifest())
    rng = np.random.default_rng(1)
    w = np.array([f["records"] for f in files], dtype=np.float64)
    picks = rng.choice(len(files), size=n_jobs, replace=False, p=w / w.sum())
    jobs = []
    for i in picks:
        n_rg = pq.ParquetFile(files[i]["key"], filesystem=openalex._s3()).num_row_groups
        jobs.append((files[i]["key"], int(rng.integers(n_rg)), rate))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        list(ex.map(job, *zip(*jobs[:workers], strict=True)))  # warm the pool
        t0 = time.monotonic()
        res = list(ex.map(job, *zip(*jobs, strict=True)))
        wall = time.monotonic() - t0
    total = sum(r["bytes"] for r in res)
    print(json.dumps({
        "workers": workers, "row_groups": len(res), "sample_rate": rate,
        "projected_mb": round(total / 1e6, 1), "wall_s": round(wall, 1),
        "mb_per_s": round(total / 1e6 / wall, 1),
        "rows": sum(r["rows"] for r in res), "kept": sum(r["kept"] for r in res),
        "mean_t_read_s": round(sum(r["t_read"] for r in res) / len(res), 2),
        "mean_t_stage_s": round(sum(r["t_stage"] for r in res) / len(res), 2),
    }, indent=2))  # fmt: skip
