# V-GR-FILTER: goodreads `filter` on the E1c encoder

Roadmap Phase V, V-GR-FILTER. Records on the Hub as `campaign-v2/goodreads-filter`
([hub-index](../../hub-index.md)); the state and its caveats are in
[validation](../../../validation.md#v-gr-filter-goodreads-filter-on-e1c). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | the sequential driver: code_version check, `bench oracle`, then one `bench run --interleave` per campaign child, stopping on the first non-zero one |
| [`summary.py`](summary.py) | counts, clocks, wall time and the per-arm table below, from the JSONL |
| [`tab-t2_goodreads.tex`](tab-t2_goodreads.tex) | T2's goodreads table as `bench report --manifest evaluation/campaign.yaml` generated it over `/scratch/campaign-v2/results` (V-PILOT's synth records in the same tree) |

The Hub subtree holds `filter/goodreads-d128.{jsonl,samples.jsonl,perquery/}`, `results.parquet`, and
under `logs/` the driver logs, every stream's log (command line, `nvidia-smi -q -d CLOCK` at start
and end), the 1 s `clocks.csv` sampler and `summary.txt`.

## How it ran

Two runs of the driver on GPU 0 of a one-GPU A100-SXM4-80GB pod (no neighbour load), main checkout
`staging` `f2be5c7`, code_version `408b1188`, `/venvs/retrieve`, `taskset -c 0-63,128-191`, results
tree `/scratch/campaign-v2/results`. The orchestrator held `linr_v3` back while the pilot's V3 seed
question was open, so the arms ran as explicit `bench run` streams rather than `bench campaign`:

| run | stream (`--algo` × `--backend`, `--interleave`) | records | s |
|---|---|---|---|
| 1 | `bench oracle --dataset goodreads --suite filter` (4 sweeps) | — | 60 |
| 1 | `linr_v1_filter_mask` + `linr_v2` × triton | 24 | 2,200 |
| 1 | `silvertorch` × triton + official | 15 | 1,564 |
| 1 | `silvertorch` × torch (plain and `compile=max-autotune`) | 24 | 2,550 |
| 1 | `postfilter` × torch (α 1, 8) | 24 | 1,783 |
| 2 | `linr_v3` × triton (oracle cached, 8 s) | 12 | 1,101 |

Driver wall time 2.27 h + 0.31 h = **2.58 GPU-h** against the roadmap's 5.

```bash
setsid nohup bash driver.sh > /scratch/v-gr-filter/driver.log 2>&1 &
setsid nohup bash driver.sh 'linr_v3|triton' > /scratch/v-gr-filter/driver-v3.log 2>&1 &
cd evaluation && python ../docs/artifacts/campaign-v2/v-gr-filter/summary.py /scratch/campaign-v2/results/filter/goodreads-d128.jsonl
```

## Per arm (k 100, median over seeds; NOT CITABLE)

`recall_oracle@100` mean (min over seeds); ms at bs 1, eager / graph; bs 16 graph QPS. A compiled
arm's eager row is the compiled module; it and official have no graph entry (`not_capturable`).

| arm | sweep | recall_oracle@100 | held-out recall@100 | bs 1 ms eager / graph | bs 16 QPS |
|---|---|---|---|---|---|
| V1 triton | c0_genre / c1_lang_reverse / all4 | 0.9997 / 0.9997 / 0.9998 | 0.2151 / 0.1390 / 0.2905 | 0.554 / 0.510 | 7,435 |
| V1 triton | bloom c0_genre | 0.9997 | 0.2151 | 0.761 / 0.478 | 13,708 |
| V2 triton | c0_genre / c1_lang_reverse / all4 | 0.9997 / 0.9997 / 0.9998 | as V1 | 0.685-0.705 / 0.319-0.357 | 4,576-5,838 |
| V2 triton | bloom c0_genre | 0.9997 | 0.2151 | 0.959 / 0.308 | 7,427 |
| V3 triton | c0_genre / c1_lang_reverse / all4 | 0.9166 / 0.9059 / 0.9799 | 0.1994 / 0.1293 / 0.2829 | 1.23-1.26 / 0.385-0.392 | 6,798-7,041 |
| V3 triton | bloom c0_genre | 0.9166 | 0.1994 | 1.737 / 0.363 | 11,696 |
| SilverTorch triton, `n_probe` 24 | c0_genre / c1_lang_reverse / all4 | 0.9461 (0.9446) / 0.8410 (0.8341) / 0.6272 (0.6239) | 0.2127 / 0.1352 / 0.2712 | 0.556-0.620 / 0.178-0.198 | 78,275-82,337 |
| SilverTorch triton | bloom c0_genre | 0.9461 | 0.2127 | 0.823 / 0.203 | 79,641 |
| SilverTorch official | bloom c0_genre | 0.9460 (0.9445) | 0.2128 | 1.597 / — | — |
| SilverTorch torch | clause (3 sweeps) / bloom | as triton | as triton | 0.937-1.028 / 0.183-0.202; bloom 1.329 / 0.205 | 17,800-19,185 |
| SilverTorch torch compiled | clause (3 sweeps) / bloom | as triton | as triton | 0.172-0.193 / — | — (bs 16 eager 0.40-0.45 ms) |
| postfilter α 1 | c0_genre / c1_lang_reverse / all4 | 0.6507 / 0.1750 / 0.0578 | 0.1552 / 0.0547 / 0.0634 | 0.504-0.508 / 0.417-0.418 | ~22,300 |
| postfilter α 8 | c0_genre / c1_lang_reverse / all4 | 0.9705 / 0.6478 / 0.2521 | 0.2118 / 0.1181 / 0.1791 | 0.563-0.576 / 0.444-0.445 | ~21,370 |

Pass rates (the records' `pass_rate`): `c0_genre` 0.331, `c1_lang_reverse` 0.621, `all4` 0.041; bloom
`c0_genre` false-positive rate 0. Bloom `c0_genre` is the only bloom sweep (the bloom map has no c1 and
no `all4`). SilverTorch runs `n_probe` 24 only: the n95 slots stay empty until V-GR-DEEP measures
goodreads' n95, so T2's "@ 0.95" rows read "not reached".
