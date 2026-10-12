# H-QLOOP: the quality pass off the host

Roadmap H-QLOOP (pod 1, 2026-10-09). Harness only; code_version unchanged. State in [validation](../../../validation.md).

| file | what |
|---|---|
| [`qprofile.py`](qprofile.py) | cProfile of one cell's `run.quality` on the GPU (py-spy cannot attach in the pod container) |
| [`qlines.py`](qlines.py) | host time per line of a copy of the chunk loop: which gather costs |
| [`gate.sh`](gate.sh), [`compare.py`](compare.py) | the byte-identical gate: the same cells with `--skip-perf` under the current harness and the H-QLOOP harness (same imported library), `quality` JSON and per-query sidecar bytes compared |

## Measured before the fix (goodreads-synth `v3bits`, V3 `p1`, `k_bits` 64, pool 1 %, library `0d23c615`)

`run.quality` 21.8 s for 625 chunks of 16 rows; 13.2 s self-time in its own lines (tensor indexing, which cProfile does not
attribute), 5.7 s in `metrics.accumulate`, 2.2 s in the V3 forward. Per line (`qlines.py`, 26.2 s wall): `blob["topk"][sel][m].to(dev)`
17.7 ms a chunk (11.1 s), `inp["targets"][sel][m].to(dev)` 8.5 ms (5.3 s), `targets_in_filter[sel][m].to(dev)` 5.7 ms (3.5 s), the
forward 3.5 ms, both accumulates 5.5 ms. All five per-row tensors (`queries`, `qa_s`, `targets`, oracle `topk`, `targets_in_filter`)
lived on the CPU; each chunk gathered them on the host (torch's 128 intra-op threads) and copied from pageable memory.
