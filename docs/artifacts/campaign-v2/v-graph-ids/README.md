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

## Fix (proposed, not applied; the user decides)

**Counterfactual, measured:** the same repro with `torch._inductor.config.emulate_divison_rounding
= True` (inductor emits the correctly rounded `div_rn`, as eager) gives `torch.equal` ids **and**
scores eager vs graph in all 4 cells, `q_codes` equal in every row (`repro-divrn/report.json`).
Its latency cost was not measured.

Options, none applied:

1. **Harness:** `measure.graph_callable` compiles with `emulate_divison_rounding = True`.
   code_version unchanged. The graph number then times the correctly rounded division the eager
   path runs (one per query element, inside an already-launched fused kernel: expected
   negligible, unmeasured). It makes the gate hold for this mechanism without loosening it, but
   it does not change what a library user gets under `torch.compile`.
2. **Library:** quantize the query inside the Triton probe kernels (or a dedicated Triton
   quantize op that both modes call), so eager and compiled run the same instructions. Changes
   code_version and needs the Triton parity gates rerun.
3. **Gate:** compare eager vs graph with a score tolerance or `ids_sha256_canon`. That loosens a
   bit-exact gate (AGENTS.md rule 3), so it is only the user's to choose, and canon would not
   have hidden this case anyway (the scores differ).

## Files

| file | what |
|---|---|
| [`repro.py`](repro.py) | builds the H2H-FINAL arXiv Triton cells as `bench run` does (seed, pool, k, bs), runs the first `IDS_PROBE_BATCHES` pool batches eager and through `measure.graph_callable`, dumps ids + scores, compares with `torch.equal`, isolates phase 1 and `quantize_int8` under the same compile mode; `--emulate-div` sets inductor's `emulate_divison_rounding` in-process |
| [`element.py`](element.py) | the differing element, the inductor code of `quantize_int8`, and the causal check (eager module + compiled `quantize_int8` = graph) |

```bash
# from / so the config's relative data_dir resolves to /data/arxiv-papers
cd / && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-graph-ids flock /scratch/gpu0.lock \
  taskset -c 64-127 /venvs/retrieve/bin/python $REPO/docs/artifacts/campaign-v2/v-graph-ids/repro.py /scratch/v-graph-ids/repro
... element.py /scratch/v-graph-ids/repro
... repro.py /scratch/v-graph-ids/repro-divrn --emulate-div
```

Raw outputs (the `.pt` dumps, `report.json`, `element.json`, logs, the inductor code and PTX):
Hub `artifacts/v-graph-ids` ([hub-index](../../hub-index.md)). A100-SXM4-80GB GPU 0 on pod
a100-x1-c, library tree `408b1188`, torch 2.10.0+cu128, triton 3.6.0, 2026-10-08.
