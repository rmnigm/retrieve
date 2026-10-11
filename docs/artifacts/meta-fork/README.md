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
| [`make_pkg.sh`](make_pkg.sh) | a tagged library as a renamed package (`make_pkg.sh <repo> campaign-v2.12 retrieve_v212 <dir>`) | the Triton arm |

Arms in one process: `silvertorch` (the pinned upstream, built `-O3` in the final pass's venv) and `silvertorch_fork`
register disjoint op namespaces (`torch.ops.st`, `torch.ops.stfork`).
