# V-PILOT: goodreads-synth `synth` (roadmap Phase V)

The pilot of the synthetic selectivity axis: the `synth` suite on `goodreads-synth` (N 797,084,
d 128, E1c), every arm the suite lists for it, seeds {0, 1, 2}, code_version `408b1188`
(staging `f2be5c7`), A100-SXM4-80GB, GPU 0, cores pinned to 0-63,128-191, no neighbour GPU.
Records: Hub `campaign-v2/goodreads-synth-synth`; logs, the 1 Hz clock trace, the gate output and
the `bench report` output: Hub `artifacts/v-pilot` ([hub-index](../../hub-index.md)). Not
citable until D1-G. The gate verdict and the headline numbers are in
[validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated).

| file | what |
|---|---|
| [`PREDICTIONS.md`](PREDICTIONS.md) | the pre-registered predictions, committed before the first cell (`cc7d35e`) |
| [`driver.sh`](driver.sh) | the sequential driver: code_version check, `bench oracle`, `bench campaign --suite synth --dataset goodreads-synth --resume --interleave`, `nvidia-smi` at 1 Hz |
| [`gate.py`](gate.py) | the pilot gate and every table below, from the records and `synth_filter.json` |
| [`v3_seed_check.py`](v3_seed_check.py) | CPU: OPORP at `k_bits = D` is seed-invariant in its scores (output on the Hub, `v3_seed_check.txt`) |

```bash
setsid nohup bash docs/artifacts/campaign-v2/v-pilot/driver.sh > /scratch/v-pilot/driver.log 2>&1 &
python3 docs/artifacts/campaign-v2/v-pilot/gate.py \
  /scratch/campaign-v2/results/synth/goodreads-synth-d128.jsonl /data/goodreads-work-id/synth_filter.json
```

## Predictions against the outcome

Medians over seeds; ratios are medians of per-seed ratios of interleaved V1 / V2 records (V3 is
compared with V2 across processes). Clause unless named; bloom is the same within a few percent.

| prediction | outcome | held |
|---|---|---|
| P1 V1 flat in p, ±10 % | graph spread ≤ 3.1 % (bs 1 ≤ 0.4 %); eager ≤ 5.6 % except bloom bs 1 k 100 eager at 10.3 % (one low `p003` median) | held (one eager row at the edge) |
| P1 V2 falls with p | bs 16 graph 2.687 → 1.409 ms from p 1 to 0.001; bs 1 graph 0.325 → 0.253 | held |
| P1 V2/V1 at p 1, bs 16 graph, 1.1-1.6 | 2.85 (bloom 2.58) | **did not hold** |
| P1 crossover V2 = V1 at bs 16 between p 0.1 and 0.3 | no crossover: V2/V1 ≥ 1.50 at every p at bs 16, both modes, both kinds | **did not hold** |
| P1 V2/V1 at p ≤ 0.01: ≤ 0.35 at bs 16, ≤ 0.6 at bs 1 | bs 16 1.50-1.60; bs 1 graph 0.565-0.567 (bloom 0.602-0.605), bs 1 eager 1.33-1.39 | **did not hold** at bs 16; bs 1 graph only |
| P2 postfilter α 1, k 100, ±25 % of p (p ≤ 0.3) | 0.0023 / 0.0038 / 0.0098 / 0.0267 / 0.0892 / 0.2957 at p 0.001 … 0.3; 0.9996 at p 1 | held at p ≥ 0.01; **not** at 0.001 (2.3× p) and 0.003 (+27 %) |
| P2 postfilter α 8, k 100, ±25 % of 8p (p ≤ 0.1), ≥ 0.99 at p ≥ 0.3 | 0.0110 / 0.0261 / 0.0787 / 0.2187 / 0.7590; 0.9997 at p 0.3 | held except p 0.001 (+38 %) |
| P2 same shape at k 1000; monotone in p | α 1 0.0100 / 0.0275 / 0.0955 / 0.3053 / 0.9997, α 8 0.0843 / 0.2340 / 0.7923 / 0.9998 / 0.9997 (p 0.01 … 1); falls monotonically with p at both α and both k | held |
| P3 SilverTorch `n_probe` 24, recall@100 ranges | 0.143 / 0.348 / 0.616 / 0.784 / 0.904 / 0.949 / 0.965 (p 0.001 … 1), all seven inside their ranges | held |
| P3 SilverTorch recall@1000 ranges | 0.145 / 0.322 / 0.600 / 0.802 / 0.909 (p 0.01 … 1) | held at 0.01, 0.3, 1; **not** at 0.03 (0.322 < 0.35) and 0.1 (0.600 < 0.65) |
| P4 V3 ≥ 0.99 where N·p ≤ pool | p 0.001: 0.9998 (and p 0.003: 0.9993) | held |
| P4 V3 recall non-increasing in p | frac 0.01: 0.9998 … 0.869 at p 0.3, back up to 0.925 at p 1; frac 0.05: lowest 0.930 at p 0.03, up to 0.981 at p 1 | **did not hold** (U-shape) |
| P4 V3 at p 1: frac 0.05 ≥ 0.90, frac 0.01 0.70-0.95 | 0.981, 0.925 | held |
| P4 V3 faster than V2 at p 1, bs 16 graph | V3/V2 0.444 (0.01), 0.523 (0.05) | held |
| P4 V3/V2 ≥ 1.0 at p ≤ 0.01 | bs 16 graph 0.61-0.62 (V3 faster), bs 1 graph 1.01-1.03 | **did not hold** at bs 16; held at bs 1 |
| P5 exact arms ≥ 0.99 at every p; V1 / V2 quality seed-free | min 0.9981 (V2 torch compiled, k 100, p 1; eager 0.9996); quality identical across seeds | held |

Not predicted, found:

- **V2 has a floor at bs 16**: about 1.41 ms (graph) at every p ≤ 0.03, 1.5× V1, while at bs 1
  graph it is 0.56× V1. The floor does not move with the number of survivors.
- **V3 quality is identical across seeds 0-2** in all 84 records (each computed, not copied).
  This is by construction. At the default `k_bits = D` every OPORP bin holds one coordinate, so the
  projection is a signed permutation, and `popcount(q ^ x)` is invariant under it. The seed changes
  the bits, never the Hamming score, except on an exact-0 coordinate, which packs as bit 0 under
  either sign. The harness table has none: its only zero row is the pad row 0, which is dropped.
  [`v3_seed_check.py`](v3_seed_check.py) (CPU) checks this on a random table and on the first 20,000
  real E1c items. Bits differ across seeds 0-2, while the Hamming scores and LiNRV3 torch ids and
  scores are `torch.equal`. At `k_bits = D/2` the scores differ, and with two exact zeros kept they
  differ too. The `seed_scope: pool+build` label overstates the seed for V3; SilverTorch's quality
  does vary with the seed (k-means).
- The compiled V2 torch arm's recall is 0.0015 below eager at p 1 (0.9981 against 0.9996); V1
  compiled is equal to eager.
- **Clocks.** The 1 Hz trace never left 1410 MHz while the GPU was more than 50 % busy (max 44 °C,
  387 W). The windows below max (655 of 6,912; 398 at 1140 MHz) belong to the light, launch-bound
  arms: every V3 and SilverTorch record has ≥ 10 % of windows below max, as do 11 of the 42 V2
  Triton records, and 160 / 345 records are `unstable`. Under the reuse rule only the
  interleaved V1 / V2 timings are clean.
