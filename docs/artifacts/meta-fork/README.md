# META-FORK — our fork of Meta's SilverTorch: gate instruments and records

Chain `meta-fork` (plan C0-C13): the fork lives in [`official-fork/`](../../../official-fork/CHANGES.md) and runs as
`SilverTorch(backend="official-fork")`. Current state of every change: [validation](../../validation.md) section
*META-FORK*; how the fork's ops work: [kernels](../../system/kernels.md#official-fork--our-fork-of-metas-ops). Raw
outputs go to the Hub under `artifacts/meta-fork/<change>/` ([hub-index](../hub-index.md)). Reference GPU:
A100-SXM4-80GB, pod b GPU 0; SM clocks cannot be locked, every script samples them.

| script | what it measures | used by |
|---|---|---|
| [`syncs.py`](syncs.py) | per forward (none / exact / bloom partial / full × bs × k) of one backend: the forward under `set_sync_debug_mode("error")`, and per call over 10 profiled forwards the sync API calls, D2H copies, pageable H2D copies, launches and kernels | gate **S** (plan §1.1); the evidence for F1 / F2 / F4 |
| [`graph_launches.py`](graph_launches.py) | one `torch.cuda.graph` capture on static buffers replayed over 16 pool batches (`torch.equal` to eager), and the harness `graph` mode (`graph_callable`): `cudaGraphLaunch` per call, launches outside the graph, outputs against eager | gate **S** (plan §1.2) |
| [`gate_fork.py`](gate_fork.py) | fork / upstream `-O3` / Triton v2.12 (`retrieve_v212`) in one process on one shared IVF layout and a real dataset: equality (`equal`, `scores_equal`, `ids_up_to_ties`; per-score id sets above the k-th score), the two official bloom builds `torch.equal`, then 8 rounds of 30-call windows per arm, eager and `--graph`, ratios fork / upstream and fork / Triton; `--swap` (build and time order), `--aa` (fork against fork, the floor) | gates **X**, **K**, **T** (plan §5) |
| [`check_ties.py`](check_ties.py) | the vectorised `ids_up_to_ties` (one lexsort per arm on (row, score desc, id)) against the old per-row loop: 3,000 synthetic cases (permuted ties, a foreign id, a duplicated tied id, a swap across scores, `-inf` pads), then every `compare()` of one real gate block | `gate_fork.py`'s tie check (pod 1, 2026-10-11: 0 disagreements, 192 real pairs on arXiv d128 exact all4 np 1024, 498 s → 0.16 s) |
| [`c0_coexist.py`](c0_coexist.py) | C0: both packages in one process, both namespaces registered, the fork's upstream-shaped `fused_kmean_ann` `torch.equal` to upstream's (int32 / fp16, with and without a mask) | gate **X** at C0 |
| [`build.sh`](build.sh) | builds the fork at one commit (CPU): the build tree `/scratch/wt/meta-fork-build` detached at the commit, `silvertorch-fork` editable into `/venvs/meta-fork` (a copy of the final pass's `/venvs/final-ext-o3`, whose `silvertorch` is the `-O3` upstream arm), both `.so` sha256 recorded | every gate |
| [`gpu.sh`](gpu.sh) | one GPU block on pod b GPU 0: phase flag set and trap-removed, `flock /scratch/gpu0.lock`, `/scratch/gpu0.timed` around timed jobs, the build tree on `PYTHONPATH`, an inductor cache per commit | every gate |
| [`make_pkg.sh`](make_pkg.sh) | a tagged library as a renamed package (`make_pkg.sh <repo> campaign-v2.12 retrieve_v212 <dir>`) | the Triton arm |
| [`devtime.py`](devtime.py) | device time per kernel of one forward, fork vs Triton v2.12, under `torch.profiler` (`devtime.py BS NP MODE DATASET DIM N_LISTS SWEEP`; env `K`, `FORK_BLOOM=meta`, `BLOOM_PATH=full`) | every K fix's profile |
| [`run_grid.sh`](run_grid.sh) | one `gate_fork.py` block per spec line (`DATASET DIM MODE SWEEPS N_LISTS NPROBE BSS [args]`), each under the 900 s block limit; the specs of the pod b grids are in [`grids/`](grids/) | the timed grids |
| [`summ_grid.py`](summ_grid.py) | a grid directory's X failures, R cells below recall 1 or with false positives, and the cells where the fork is above Triton | the grid summaries |
| [`recheck_vs_exact.py`](recheck_vs_exact.py) | the fork's rechecked bloom against the fork's exact path on one real cell, `torch.equal` | gate R (bloom ∧ exact = exact) |
| [`triton_vs_fork_exact.py`](triton_vs_fork_exact.py) | the fork's exact path against Triton's: sorted scores equal, every differing id at the k-th score | gate R (recall below 1 = ties) |

Arms in one process: `silvertorch` (the pinned upstream, built `-O3` in the final pass's venv) and `silvertorch_fork`
register disjoint op namespaces (`torch.ops.st`, `torch.ops.stfork`).

## Running a gate

A change's gate runs on its own commit: `build.sh <commit> <logs>` (CPU, outside any timed leg), then the
GPU blocks through `gpu.sh`, each under 15 min. The scripts are run from this directory with the build tree on
`PYTHONPATH`, so one checkout of them serves every commit:

```bash
A=/scratch/wt/meta-fork/docs/artifacts/meta-fork; PY=/venvs/meta-fork/bin/python; T=/scratch/wt/meta-fork-build
$A/build.sh <commit> /scratch/meta-fork/logs
$A/gpu.sh 0 /scratch/meta-fork/logs/c0_coexist.log $PY $A/c0_coexist.py
$A/gpu.sh 0 /scratch/meta-fork/logs/pytest_fork.log $PY -m pytest $T/official-fork/silvertorch_fork -q -p no:cacheprovider
$A/gpu.sh 0 /scratch/meta-fork/logs/pytest_official.log $PY -m pytest $T/retrieve/tests/parity/test_official.py -q -p no:cacheprovider
$A/gpu.sh 0 /scratch/meta-fork/logs/syncs_fork.log $PY $A/syncs.py official-fork /scratch/meta-fork/logs/syncs_fork.json
(cd $T/evaluation && $A/gpu.sh 1 /scratch/meta-fork/logs/gate.log $PY $A/gate_fork.py arxiv 128 bloom c0_maincat 2048 24,256 1,16,64 /scratch/meta-fork/logs/gate.json --graph)
```

The Triton arm is `make_pkg.sh /workspace/retrieve campaign-v2.12 retrieve_v212 /scratch/meta-fork/pkgs`
(on `PYTHONPATH` through `gpu.sh`).
