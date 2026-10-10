# C6 recall-ceiling check, goodreads + arXiv (campaign-v2.10)

Modelled on [`../../campaign-v2.5/yfcc-int8/int8_check.py`](../../campaign-v2.5/yfcc-int8/int8_check.py): are SilverTorch's 0.95-0.99
ceilings on goodreads and arXiv the one global int8 item scale? Quality only. Hub `artifacts/ceiling-int8-gr-ax`.

| file | what |
|---|---|
| [`ceiling_check.py`](ceiling_check.py) | one SilverTorch triton index at a given (n_lists, n_probe), its own probed clusters per query; recall_oracle@100 with the shipped module, the same global-int8 arithmetic recomputed densely (method check), per-row int8 item scales, and fp16 |
| [`points.json`](points.json) | per sweep, the best clause point of the v2.10 tuning grid (Hub `campaign-v2.10/{goodreads,arxiv}-tune`) |
| [`ceiling-v210.sh`](ceiling-v210.sh) | the driver, one check per point, under `common.sh` (GPU lock, code_version, clocks) |

Outcome (pod 1, 2026-10-10, library `a3bec4a5`): fp16 on the same probes reaches 0.998-1.000 wherever p ≥ 0.1 (0.987-0.995 on the most selective sweeps), against 0.952-0.989 shipped; the global int8 recompute equals the shipped module, and per-row int8 recovers about a third of the gap. The ceilings are the global int8 item scale.
