"""Fix 3: for a triton / official `_parity` pair of one bloom cell, check every returned id
against the exact clause predicate (the harness's own loaders and `sweep_qa`): per side, the
ids in the top-k that fail it (bloom false positives that reached the result), and for the
rows whose scores diverge by more than THRESH, how many hold a false positive on either side.

    cd evaluation && python bloom_row.py CONFIG_DIR DATASET DIM TRITON.npz,OFFICIAL.npz,CLAUSES ...

(CLAUSES as ``0`` or ``0+2``; several cells of one dataset share one load.)
"""

import sys
from pathlib import Path

import numpy as np
import torch

from bench import config, inputs

THRESH = 0.01
cfg_dir, dataset, dim = sys.argv[1:4]
ds = config.load_dataset(Path(cfg_dir) / f"{dataset}.yaml", int(dim))
inp = inputs.load_inputs(ds, torch.device("cpu"))
attrs = inp["item_attrs"]


def passes(qs: torch.Tensor, ids: np.ndarray) -> np.ndarray:
    """[rows, k] ids (-1 = empty) → [rows, k] bool: the item satisfies every live clause."""
    a = attrs[torch.as_tensor(ids, dtype=torch.long).clamp_min(0)]  # [rows, k, C, A]
    live = qs != -1  # [rows, C]
    hit = (a == qs[:, None, :, None]).any(-1)  # [rows, k, C]
    return (hit | ~live[:, None, :]).all(-1).numpy()


def census(t_npz: str, o_npz: str, clauses: tuple[int, ...]) -> None:
    qs, skip = inputs.sweep_qa(inp["qa"], clauses)
    qs = qs[~skip]  # the spill holds the kept rows only
    t, o = np.load(t_npz), np.load(o_npz)
    fin = {n: np.isfinite(z["scores"]) for n, z in (("triton", t), ("official", o))}
    ok = {n: passes(qs, z["ids"]) for n, z in (("triton", t), ("official", o))}
    fp = {n: int((fin[n] & ~ok[n]).sum()) for n in fin}
    fp_rows = {n: int((fin[n] & ~ok[n]).any(1).sum()) for n in fin}
    both = fin["triton"] & fin["official"]
    diff = np.where(both, np.abs(np.nan_to_num(t["scores"] - o["scores"], nan=0.0)), 0)
    div = np.where(diff.max(1) > THRESH)[0]
    fp_any = ((fin["triton"] & ~ok["triton"]) | (fin["official"] & ~ok["official"])).any(1)
    print(f"{dataset} {clauses} {Path(t_npz).stem}: rows {len(qs)}; false-positive ids in the "
          f"top-k {fp} (rows {fp_rows}); score_max_abs_diff {diff.max():.6f}, 99.9th pct "
          f"{np.quantile(diff[both], 0.999):.6f}; rows over {THRESH}: {len(div)}, of which "
          f"{int(fp_any[div].sum())} hold a false positive on either side", flush=True)
    for r in div[:3]:
        ti, oi = t["ids"][r][fin["triton"][r]], o["ids"][r][fin["official"][r]]
        extra_o, extra_t = np.setdiff1d(oi, ti), np.setdiff1d(ti, oi)
        n_o = int((~passes(qs[r : r + 1], extra_o[None])).sum())
        n_t = int((~passes(qs[r : r + 1], extra_t[None])).sum())
        print(f"  row {r} (query {qs[r].tolist()}): max diff {diff[r].max():.4f}; finite t/o "
              f"{len(ti)}/{len(oi)}; official-only {len(extra_o)} ({n_o} fail the clause), "
              f"triton-only {len(extra_t)} ({n_t} fail)", flush=True)


for cell in sys.argv[4:]:
    t_npz, o_npz, cl = cell.split(",")
    census(t_npz, o_npz, tuple(int(c) for c in cl.split("+")))
