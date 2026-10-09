# V2 cross-tree check: `campaign-v2` (408b1188) vs `campaign-v2.1` (f01255f1)

The surprise gate from pod b: V2 +43-45 % at p ≈ 1 on PubMed 10 M d768 at v2.1, against a Fix A gate that
covered goodreads 0.8 M d128 only. NOT CITABLE.

Two detached worktrees, `/scratch/wt/tree-408b` (tag `campaign-v2`) and `/scratch/wt/tree-v21` (tag
`campaign-v2.1`), one venv; each process imports one tree's `bench` and `retrieve` through `PYTHONPATH`
(logged per process) and builds the cells with that tree's own harness. V1 triton is the control.

| file | what |
|---|---|
| [`crosstree.py`](crosstree.py) | one process: group `gs` (goodreads-synth clause `p1`, `p0001`) or `pm` (PubMed d768 clause `c3_journal_reverse`, pass 0.9993), V1 + V2 triton timed round-robin per (bs {1, 16}, mode {eager, graph}), k 100, seed 0; `--profile`: one profiler session over V2's graph replays at bs 16 |
| [`crosstree.sh`](crosstree.sh) | the driver: tree checks, old oracle blobs moved aside and rebuilt by the new tree, then per group 4 ABAB pairs of processes (old/new, new/old, ...), one profile process per tree |
| [`crosstree_summary.py`](crosstree_summary.py) | per cell and arm: old / new ms per repeat, paired new / old geomean with a t 95 % CI, sm_mhz ranges; V2's top kernels per tree |
