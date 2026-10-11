# V2 cross-tree check: `campaign-v2` (408b1188) vs `campaign-v2.1` (f01255f1)

The surprise gate from pod b: V2 +43-45 % at p ≈ 1 on PubMed 10 M d768 at v2.1, against a Fix A gate that
covered goodreads 0.8 M d128 only. NOT CITABLE.

Two detached worktrees, `/scratch/wt/tree-408b` (tag `campaign-v2`) and `/scratch/wt/tree-v21` (tag
`campaign-v2.1`), one venv; each process imports one tree's `bench` and `retrieve` through `PYTHONPATH`
(logged per process) and builds the cells with that tree's own harness. V1 triton is the control.

| file | what |
|---|---|
| [`crosstree.py`](crosstree.py) | one process: group `gs` (goodreads-synth clause `p1`, `p0001`) or `pm` (PubMed d768 clause `c3_journal_reverse`, pass 0.9993), V1 + V2 triton timed round-robin per (bs {1, 16}, mode {eager, graph}), k 100, seed 0; `--profile`: one profiler session over V2's graph replays at bs 16 |
| [`crosstree.sh`](crosstree.sh) | the driver: tree checks, old oracle blobs moved aside and rebuilt by the new tree, then per group 4 ABAB pairs of processes (old/new, new/old, ...), one profile process per tree, then the `programs` sweep |
| [`crosstree3.sh`](crosstree3.sh), [`crosstree3_summary.py`](crosstree3_summary.py) | the three-tag V2 graph check (campaign-v2.1 / v2.4 / v2.5, fresh inductor dir per tree, 4 rotating rounds; group `ax3`: arxiv-synth `p0001` / `p01` / `p1`, V1 + V2, eager + graph, bs 1 / 16, ids hashed), pairwise ratios with CIs; [`crosstree3_group1.py`](crosstree3_group1.py) anchors today's v2.1 / v2.5 against pod 1's earlier v2.1 records. Outcome (Hub `artifacts/v2-crosstree3`): v2.4 = v2.1 on V2, v2.5 V2 0.84-0.996 × v2.4 (faster), ids equal, v2.1 reproduces group1 within 2 %: all on the 7-clause synth table; pod d's slower V2 is the 10-clause table's width, not the box |
| [`crosstree4.sh`](crosstree4.sh), [`crosstree4_summary.py`](crosstree4_summary.py) | the v2.6 slowdown split: v2.5 vs v2.6 with the pool's attrs and with `run.prepare_pool`'s (`crosstree.py --prep`), 3 rounds, one profile per variant (bs 16, V1 + V2, eager + graph), then the 8-cell `bench run` from each tree. Outcome (Hub `artifacts/v26-split`): prepared = raw; the `bench run` ratios match; only `_clause_mask_kernel` / `_clause_compact_kernel` grow (+414 / +669 µs at bs 16): the 10-clause synth table's width, not M1 or the harness |
| [`v2prof-yfcc.sh`](v2prof-yfcc.sh) | V2's YFCC profile, v2.5 vs v2.9 (`crosstree.py` group `yf --profile`, each tree its own venv): only `_clause_compact_kernel` moves (255.7 → 49.3 ms), CLAUSE-SKIP B (Hub `artifacts/v2prof-yfcc`) |
| [`sweep.py`](sweep.py) | the `fused_masked_knn_topk` `programs` sweep inside V2 (controller): v2.1 tree at programs {864, 1728, 3456, 6912, 13824} and `full` (one tile per program, the old grid's shape) via the op module's `DEFAULT_CONFIG` (read per call; no library edit), the old tree's kernel as reference; cells PubMed d768 10 M `c3_journal_reverse` (p 0.9993), arXiv d128 3 M `c3_nversions` (0.444, its highest real pass) and arXiv-synth `p1` (1.0); bs {1, 16}, k 100; eager and hand-captured CUDA graphs timed round-robin, fmkt kernel µs from one profiler session per variant |
| [`sweep_summary.py`](sweep_summary.py) | the sweep table against the old tree |
| [`crosstree_summary.py`](crosstree_summary.py) | per cell and arm: old / new ms per repeat, paired new / old geomean with a t 95 % CI, sm_mhz ranges; V2's top kernels per tree |
