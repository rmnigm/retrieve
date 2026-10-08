# V-GRAPH-IDS: Triton eager vs graph ids at arXiv bs 16 (roadmap Phase V)

H2H-FINAL's arXiv records (Hub `campaign-v2/arxiv-h2h`, code_version `408b1188`) have Triton
SilverTorch `graph` `ids_sha256` and `ids_sha256_canon` different from `eager` in 4 of 80 pairs:
**seed 1, bs 16**, both kinds (`none`, `bloom`), both k (100, 1000), `n_probe` 24. Every other
seed, bs 1, and all of goodreads are equal. In all 4, official int32's canonical hash equals
Triton **eager** (graph is the odd one out). The records store only hashes, so one GPU repro was
needed. Not citable until D1-G; the state is in
[validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated).

## Mechanism: a real score divergence from the compiled query quantization

Neither ties at the k-th score nor a harness artefact. Graph mode is
`torch.compile(mode="reduce-overhead")` of the whole forward (`measure.graph_callable`), so
inductor regenerates the torch-side prep of `codesigned_probe_score*`, including
`quantize_int8(query)` (`x / abs_max * 127.0`, then `round`). Inductor emits Triton's fp32 `/`,
which lowers to the approximate `div.full.f32` (≤ 2 ulp; the cached PTX of
`triton_per_fused__to_copy_abs_amax_clamp_div_mul_round_0` has it). Eager uses the correctly
rounded division. One query element lands on a rounding boundary:

| | value |
|---|---|
| element | pool batch 6, row 0, column 104 (flat row 96 of the 128 probed rows); same query in `none` and `bloom` |
| `x / abs_max * 127`, exact (fp64) | −7.49999996 |
| eager fp32 | −7.4999995 → `round` → **−7** |
| compiled fp32 | −7.5 → `round` (half to even) → **−8** |

One int8 query code differs, so every score in that row moves (up to 3.7e-4) and the ranking
near the k-th score changes:

| kind, k | rows differing | max \|Δs\| | ids across the cut |
|---|---|---|---|
| none, 100 | 1 / 128 | 2.4e-4 | same set, different order (scores differ, so canon differs too) |
| none, 1000 | 1 / 128 | 3.7e-4 | 2 ids swapped |
| bloom, 100 | 1 / 128 | 2.4e-4 | 1 id swapped |
| bloom, 1000 | 1 / 128 | 3.7e-4 | 3 ids swapped |

Phase-1 probe ids (`query @ centroids.t()` + `topk`, also compiled) are identical; `q_scales`
are identical; `q_codes` differ in that one row only. **Causal check:** the eager module with
`_host.quantize_int8` swapped in-process for the compiled `quantize_int8` reproduces the graph
ids and scores bit for bit (`torch.equal`) in all 4 cells. The Triton probe kernel is not
involved: its int8 dot is exact, and it gets the same codes in both modes.

How often: the trigger is a query element whose `x / abs_max * 127` lies within ~2 ulp of a
half-integer, so it depends on the pool's queries. That is 1 element in 8 × 16 × 128 at arXiv
seed 1 and none at the other 79 pairs. The quality pass runs eager only, so no recorded recall
moves, but `ids_sha256` is a bit-exact gate, and D1-G compares it for every capturable arm. Other callers of
`quantize_int8` under compile (the `torch` reference op, the candidate re-rank path) have the
same exposure; they were not run here.

## Fix (applied on `dev/v-graph-ids`, library tree `5d158f20`; the controller ruled it a correctness fix)

`quantize_int8` computes the code quotient as `(embs.double() / abs_max).float()`, in eager and
compiled alike ([kernels](../../../system/kernels.md#quantize_int8-retrieveindexing)). A correctly rounded fp64
quotient rounded to fp32 is the correctly rounded fp32 quotient (53 ≥ 2·24 + 2), and Triton's
fp64 `/` is correctly rounded (`div_probe.py`: compiled fp64 0 of 1.28 M quotients differ from
eager fp32, against 403,534 for compiled fp32). The scales stay `abs_max / 127.0`: eager evaluates a
division by a Python scalar as a multiply by fp32(1/127), and inductor emits the same multiply.
A first attempt (`fix-v1/` on the Hub) failed two ways, both measured: an
`is_compiling()` branch is never taken, because inductor traces a `triton_op`'s implementation
with make_fx, where the flag is false. Graph also kept the 36 diverging rows, and fp64 scales broke
eager's reciprocal multiply (549 rows off by 1 ulp).

Gates (`fix_gates.py`, A100 GPU 0, library tree `5d158f20` clean, worktree library on `PYTHONPATH`):

| gate | result |
|---|---|
| 1. eager codes + scales vs 408b1188 (`OLD`, verbatim), all 10,000 arXiv d128 queries | `torch.equal` ✓; compiled new = eager too (0 code, 0 scale diffs); compiled old: 45 codes in 36 rows |
| 2. graph == eager, ids and scores (`torch.equal`) | 4 H2H entries (seed 1, bs 16, 128 rows): new 0, old 1 row. All 10,000 queries in bs 16 chunks, `none`, clause `c0_maincat`, bloom `c0_maincat`, k 100 + 1000: new 0 rows, old 36 rows in every cell ✓ |
| 3. library suite on the GPU | 779 passed (`retrieve.__file__` the worktree's) ✓ |
| (c) latency new / old, interleaved, 10 rounds, clause `c0_maincat`, `n_probe` 24, k 100 | graph bs 1 0.997 [0.993, 1.004], bs 16 1.0007 [1.0006, 1.0009]; **eager bs 1 1.047 [1.044, 1.049], bs 16 1.047 [1.045, 1.052]** (0.615 → 0.644 ms, 0.622 → 0.654 ms: three more eager ops, ≈ 30 µs) |

Clocks in (c): 1410 MHz in every window except eager bs 1, where both arms ran all 10 windows at
1155 MHz (interleaved, so the ratio is paired); no window `unstable`.

**What goes stale.** (a) The official backend calls our `quantize_int8` eagerly in its timed and
scored path (`retrieve/src/retrieve/ops/official/adapter.py:294`, from `official_probe_score`,
`modules/silvertorch.py:490`): its ids and scores are unchanged (gate 1), and its eager time
gains the same ≈ 30 µs. (b) Quality runs eager (`evaluation/bench/run.py:283`, `module(q, qa)`), so
recorded recall does not move. The one exception is the `filter` suite's `torch` + `compile: max-autotune`
SilverTorch arm (`algos.compile_module`, `run.py:694`), whose quality goes through the compiled
reference op and can move by these rows. Every SilverTorch eager time (Triton, torch, official)
moves by ≈ +30 µs; graph times do not move.

The eager cost could be removed with a branch that detects the make_fx trace (an internal
proxy-mode check); not done, because it relies on private torch APIs. That choice is the controller's.

## Files

| file | what |
|---|---|
| [`repro.py`](repro.py) | builds the H2H-FINAL arXiv Triton cells as `bench run` does (seed, pool, k, bs), runs the first `IDS_PROBE_BATCHES` pool batches eager and through `measure.graph_callable`, dumps ids + scores, compares with `torch.equal`, isolates phase 1 and `quantize_int8` under the same compile mode; `--emulate-div` sets inductor's `emulate_divison_rounding` in-process |
| [`element.py`](element.py) | the differing element, the inductor code of `quantize_int8`, and the causal check (eager module + compiled `quantize_int8` = graph) |
| [`div_probe.py`](div_probe.py) | fp32, fp64, `true_divide` and a `tl.div_rn` Triton op, each compiled vs eager fp32 over every query, with the generated code |
| [`fix_gates.py`](fix_gates.py) | gates 1, 2 and the interleaved latency (c) of the fix, old (`OLD` patched into `_host` at compile / per call) and new arms side by side |

```bash
# from / so the config's relative data_dir resolves to /data/arxiv-papers
cd / && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-graph-ids flock /scratch/gpu0.lock \
  taskset -c 64-127 /venvs/retrieve/bin/python $REPO/docs/artifacts/campaign-v2/v-graph-ids/repro.py /scratch/v-graph-ids/repro
... element.py /scratch/v-graph-ids/repro
... repro.py /scratch/v-graph-ids/repro-divrn --emulate-div
# the fix: this worktree's library and harness ahead of the shared venv's
cd / && flock /scratch/gpu0.lock env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$WT/retrieve/src:$WT/evaluation \
  taskset -c 64-127 /venvs/retrieve/bin/python $WT/docs/artifacts/campaign-v2/v-graph-ids/fix_gates.py /scratch/v-graph-ids/fix
```

Raw outputs (the `.pt` dumps, `report.json`, `element.json`, `fix/fix_gates.json`, `fix-v1/`,
`divprobe/`, logs, the pytest log, the inductor code and PTX): Hub `artifacts/v-graph-ids`
([hub-index](../../hub-index.md)). A100-SXM4-80GB GPU 0 on pod a100-x1-c, torch 2.10.0+cu128,
triton 3.6.0, 2026-10-08; the repro at library tree `408b1188`, the fix gates at `5d158f20`.
