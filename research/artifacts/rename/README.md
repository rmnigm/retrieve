# RENAME — the gate grid before / after the rename

Roadmap §5 (RENAME, 2026-10-16): the adapted SilverTorch code took Meta's names (`silvertorch/`, package
`silvertorch`, `torch.ops.st.*`, `SilverTorch(backend="cuda")`) and Meta's code as published left the workspace. A
rename changes no behaviour, so the gate is `torch.equal` on the META-FORK gate grid's int32 cells before and after.

| file | what |
|---|---|
| [`grid.txt`](grid.txt) | the META-FORK head grid ([`meta-fork/grids/head-32c0d61.txt`](../meta-fork/grids/head-32c0d61.txt)) minus its YFCC lines (the dataset is out of the study) and its timing / arm flags: 45 lines, none / exact / bloom (Meta's bloom semantics and the default), arXiv d128 / d256, goodreads d128, PubMed d768 |
| [`capture.py`](capture.py) | one tree, one backend name: per grid line the ids and scores of 16 pool batches per (sweep, n_probe, bs ∈ {1, 16, 64}, k ∈ {100, 1000}) at `score_path="int32"`, saved as `.pt` with a sha256 per cell in `manifest.json`; the index (k-means seed 0) built once per (dataset, dim, n_lists) and shared by the modes, as `gate_fork.py` did |
| [`compare.py`](compare.py) | two capture directories: hash per cell, `torch.equal` per batch where a hash differs |

Run (GPU 0 under `flock /scratch/gpu0.lock`, from each tree's `evaluation/`, the tree's own venv):

```bash
python capture.py grid.txt /scratch/rename/before --backend official-fork   # 9e86924e, the base
python capture.py grid.txt /scratch/rename/after  --backend cuda            # dev/rename
python compare.py /scratch/rename/before /scratch/rename/after
```

Result (pod e, A100-SXM4-80GB, 2026-10-16): 45 lines, 300 cells, **0 differ** (every cell's ids and scores
`torch.equal`). The captures (0.8 GB each) stay on the pod; nothing is uploaded (RENAME needs no Hub record). The row is
in [validation](../../validation.md#library-gates).
