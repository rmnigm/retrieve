# H2H-FINAL: official vs Triton on the release code (roadmap Phase V)

The `h2h` suite on goodreads (E1c, `c0_genre`) and arXiv (`c0_maincat`), d128, `none` + `bloom`,
SilverTorch `n_probe` 24: our Triton arm and Meta's official arm with `score_path` fp16 and int32,
one interleaved group of the three arms per cell, seeds 0-4 as the repeats, `--profile`.
code_version `408b1188` (main checkout at staging `f2be5c7`), A100-SXM4-80GB, GPU 0, cores pinned to
0-63,128-191, no neighbour GPU (single-GPU pod). Records: Hub `campaign-v2/goodreads-h2h` and
`campaign-v2/arxiv-h2h`; logs, the 1 Hz clock trace, the summaries and the `bench report --manifest`
output: Hub `artifacts/h2h-final` ([hub-index](../../hub-index.md)). Not citable until D1-G. The
state of the step is in [validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated).

| file | what |
|---|---|
| [`common.sh`](common.sh) | shared by the leg drivers: environment, code_version check (refuses on a mismatch), `nvidia-smi` at 1 Hz, `step` (rc and seconds per step, stops on a non-zero rc) |
| [`h2h-final.sh`](h2h-final.sh) | `bench oracle` per dataset, then `bench campaign --suite h2h --dataset goodreads --dataset arxiv --resume --interleave --profile` |
| [`stage-upload.sh`](stage-upload.sh) | copies one `(suite, dataset)` slice out of the shared campaign-v2 tree and runs `bench upload --verify` to `campaign-v2/<dataset>-<suite>` |
| [`h2h_summary.py`](h2h_summary.py) | per `(kind, k, bs, mode)` medians over seeds, paired official / Triton ratios, parity fields, clock windows (output on the Hub, `summary-<dataset>.txt`) |

```bash
setsid nohup bash docs/artifacts/campaign-v2/h2h-final/h2h-final.sh > /scratch/h2h-final/driver.log 2>&1 &
bash docs/artifacts/campaign-v2/h2h-final/stage-upload.sh h2h-final h2h goodreads 128
python3 docs/artifacts/campaign-v2/h2h-final/h2h_summary.py /scratch/campaign-v2/results/h2h/arxiv-d128.jsonl
```

## Outcome

Medians over the 5 seeds; ratio = official / Triton eager, paired per interleave group, range over
seeds in brackets. Official cannot be captured, so it has no `graph` entry.

| dataset, kind | official fp16 / Triton (eager) | official int32 / Triton (eager) | Triton eager ms | Triton graph ms |
|---|---|---|---|---|
| goodreads none | 1.77-1.84 | 1.71-1.79 | 0.54-0.59 | 0.17-0.21 |
| goodreads bloom | 1.91-2.18 | 1.90-2.15 | 0.81-0.83 | 0.20-0.21 |
| arXiv none | 1.77-1.85 | 1.73-1.82 | 0.53-0.60 | 0.17-0.32 |
| arXiv bloom | 1.91-2.18 | 1.87-2.14 | 0.82-0.84 | 0.20-0.33 |

(ranges over k {100, 1000} × bs {1, 16}). Every ratio's 95 % CI in `tab-t3` excludes 1.0 (noise
gate). The direction is D1's (official ~1.02 ms vs Triton 0.56 ms on goodreads bs 1).

- **Parity.** int32 vs Triton: `score_max_abs_diff` 0 and Jaccard ≥ 0.999986 on goodreads (both
  kinds) and on arXiv `none` at every seed. arXiv `bloom` int32 is 0 at seeds 0, 2, 4 and 3.5e-4 /
  5.9e-4 at seeds 1 / 3 (Jaccard ≥ 0.999994): Meta's bloom is a different hash from ours, so the
  two filters can pass different false positives; bloom has no bit-exact int32 claim
  ([validation](../../../validation.md), *Official vs Triton, int32 score path*). fp16 differs only
  at near-ties: max |Δs| 3.9e-4 (goodreads) / 4.8e-4 to 7.1e-4 (arXiv), Jaccard@100 ≥ 0.996 / ≥ 0.984.
  Held-out recall@100 is equal across the three arms (goodreads 0.1486 none, 0.2127 bloom; arXiv 0.9943).
- **Kernel-only split incomplete for Triton.** `--profile` stored an empty `kernels` list on every
  Triton bloom eager entry (40 / 40) and on 13 / 40 Triton `none` entries; official arms always have
  theirs. `tab-t3`'s kernel µs / launches column is therefore `---` for Triton bloom. A harness
  measurement defect (`measure.profile_once` captured no CUDA kernel for those calls), not fixed
  here; the timed numbers do not depend on it.
- **Eager vs graph ids.** Triton `graph` `ids_sha256_canon` differs from eager at arXiv bs 16 in one
  seed per `(kind, k)` (4 of 80 pairs); goodreads all equal. Left to D1-G's eager-vs-graph gate.
- **Clocks.** 499 of 960 timing windows below 1410 MHz (min 1140); 54 / 60 records `unstable`.
  The 1 Hz trace under > 50 % load held 1410 MHz (522 / 523 samples), max 39 °C, 325 W: the
  low-clock windows are the launch-bound calls, as in V-PILOT.
