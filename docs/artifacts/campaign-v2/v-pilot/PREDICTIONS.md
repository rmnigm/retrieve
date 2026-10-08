# V-PILOT: pre-registered predictions (goodreads-synth)

Written by the V-PILOT worker on 2026-10-08, before the first campaign cell of the step ran.
The commit timestamp is the evidence. This file is not edited after the run; the outcome is
compared against it in [validation](../../../validation.md).

Setting: `synth` suite on `goodreads-synth` (N = 797,084, d 128, E1c encoder), uniform per-item
pass flag `u_i < p`, p ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}, seeds {0, 1, 2}, code_version
`408b1188`, A100-SXM4-80GB. "Recall" is `recall_oracle@k` against the exact filtered top-k.

**Already known before the run** (not predictions): the achieved pass counts in
`synth_filter.json` (789 / 2,409 / 7,985 / 23,776 / 79,316 / 238,674 / 797,084). At p = 0.001
the achieved rate 0.000990 is 1.01 % below target, so the pilot gate's "within 1 % relative" is
missed at that point by construction of the pre-registered seed (binomial σ at N·p = 797 is
3.5 % relative); every other rate is within 0.75 %. The smoke at p = 0.01 (seed 0, quality only)
measured postfilter 0.0098 / 0.079 (α 1 / 8) and SilverTorch `n_probe` 24 0.609.

## P1. V2 vs V1 latency as p falls (triton, interleaved)

- V1 (`linr_v1_filter_mask`, mask then top-k over N) is flat in p: per (bs, k, mode) its median
  latency varies by no more than ±10 % across the seven rates.
- V2 (`linr_v2`, compact then score the survivors) falls monotonically with p.
- At p = 1.0, V2 is slower than V1 (V2/V1 between 1.1 and 1.6, bs 16, graph).
- The crossover V2 = V1 lies between p = 0.1 and p = 0.3 at bs 16.
- At p ≤ 0.01, V2/V1 ≤ 0.35 at bs 16 and ≤ 0.6 at bs 1 (bs 1 is launch-bound, so the gain is
  smaller).

## P2. Postfilter recall at α 1 (and 8) vs p

Expected recall ≈ min(1, p·α·c) with c ≈ 1 on a uniform, embedding-independent filter (the
number of passers in the top α·k is Binomial(α·k, p)):

- α 1, k 100: recall within ±25 % relative of p at p ≤ 0.3 (0.001, 0.003, 0.01, 0.03, 0.1, 0.3),
  and 1.0 at p = 1.0.
- α 8, k 100: within ±25 % relative of 8p at p ≤ 0.1 (0.008 … 0.8), ≥ 0.99 at p ≥ 0.3.
- Same shape at k 1000.
- Postfilter recall falls monotonically with p at both α.

## P3. SilverTorch recall at `n_probe` 24 vs p (triton, clause)

With 1024 lists and 24 probed, about 18.7 k items are scanned per query, about 18.7 k·p of them
passing, so recall@k is capped near 18.7 k·p / k at low p. Predicted `recall_oracle@100`:

| p | 1.0 | 0.3 | 0.1 | 0.03 | 0.01 | 0.003 | 0.001 |
|---|---|---|---|---|---|---|---|
| range | 0.93-0.98 | 0.90-0.97 | 0.85-0.95 | 0.72-0.88 | 0.55-0.68 | 0.30-0.50 | 0.08-0.25 |

`recall_oracle@1000` (k 1000 runs at p ≥ 0.01): p 1.0 0.85-0.95, 0.3 0.80-0.93, 0.1 0.65-0.85,
0.03 0.35-0.55, 0.01 0.10-0.20. Recall falls monotonically as p falls, roughly linear in log p
over 0.003-0.1, with the steepest fall below p = 0.01.

## P4. V3 vs V2 (triton)

- Quality: V3 `recall_oracle@100` ≥ 0.99 wherever N·p ≤ the pool (pool = max(2000, frac·N·p):
  p 0.001 at both fractions), because stage 2 then scores every passer exactly; V3 recall is
  non-increasing in p (the 2000 floor makes the pool a larger share of the passers at low p);
  at p = 1.0, frac 0.05 ≥ 0.90 and frac 0.01 between 0.70 and 0.95.
- Latency: V3 is faster than V2 at p = 1.0 (bs 16, graph) and not faster (V3/V2 ≥ 1.0) at
  p ≤ 0.01, where the pool floor keeps stage 2 the size of V2's whole job.

## P5. Exact arms

V1 and V2 (triton and torch, eager and compiled) reach `recall_oracle@k` ≥ 0.99 at every p, and
V1 / V2 quality is identical across seeds (seed-free quality cache).
