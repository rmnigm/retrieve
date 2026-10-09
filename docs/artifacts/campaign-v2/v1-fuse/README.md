# V1-FUSE: LiNR V1's filter folded into the score write

`clause_mask_scores` / `bloom_match_scores` replace V1's mask kernel → `[B, N]` bool → `torch.where`
([kernels](../../../system/kernels.md#clause_mask--fused-clause-eval-emitting-b-n-bool)). Gate state and every number:
[validation](../../../validation.md#library-gates). Raw outputs: Hub `artifacts/v1-fuse`, `artifacts/v1-fuse-w2`, `artifacts/v1-fuse-w3`
([hub-index](../../hub-index.md)). Pod c A100-SXM4-80GB, GPU 0, `taskset -c 64-127`, under `flock /scratch/gpu0.lock`.

| file | what |
|---|---|
| [`v1fuse_gpu.py`](v1fuse_gpu.py) | `gate`: per (filter kind, p, bs, k) the new V1's `ids_sha256` vs the campaign-v2.1 records (eager and graph), ids + scores vs the old composition, `graph_callable` capture, and an interleaved keep-rule / graph-vs-eager timing group (new eager, old eager, new graph, old graph); `V1FUSE_KINDS`, `V1FUSE_NO_TIMING` narrow it. `fused`: the item-1 Triton mat-vec experiment |
| [`v1fuse_sweep.py`](v1fuse_sweep.py) | tile sweep of both score-masking kernels vs the old mask + `where`, interleaved, bitwise-checked |
| [`v1fuse_prof.py`](v1fuse_prof.py) | per-kernel device time of new vs old V1 (`torch.profiler`) |
| [`driver.sh`](driver.sh), [`followup.sh`](followup.sh) | window 1 (library `4c7119d0`, on v2.1) |
| [`driver2.sh`](driver2.sh) | window 2 (library `36387035`, on v2.2); its clause graph cells are void: one inductor cache across two code versions |
| [`driver3.sh`](driver3.sh) | window 3: library suite + clause gate on a fresh inductor cache keyed by code_version |

The "old" side of the ids gate is the campaign-v2.1 V1 records (arxiv-synth: V-AX-SYNTH group 1 at `f01255f1`; goodreads-synth: V-PILOT,
Hub `campaign-v2/goodreads-synth-synth`); V1's ops are the same at v2.1 and v2.2.
