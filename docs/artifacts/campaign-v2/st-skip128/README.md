# ST-SKIP128 — a pass-rate-gated tile skip in the bloom probe scorer at D ≤ 256

Roadmap step ST-SKIP128 (campaign-v2.3 bundle; ST-DLOOP's improvement #3). #16's tile skip at `D_PAD ≤ 256` was about 30 %
faster at p 0.001 but cost 1-6 % at p ≥ 0.1, because every tile paid the cross-warp `tl.max(keep)` vote. Current state:
[validation](../../../validation.md) row *ST-SKIP128*; mechanism: [kernels](../../../system/kernels.md#silvertorch-kernels)
("Gated tile skip"). A100-SXM4-80GB (pod b, GPU 0). Before = the v2.3 candidate (`dev/v2-highp` f537c91) as package
`retrieve_v23` (`../st-ids/make_pkg.sh`). Raw outputs: Hub `artifacts/st-skip128/` ([hub-index](../../hub-index.md)).

## What the step can and cannot close (measured before building)

The step cites EXHIBITS run 4: Meta's scorer kernel alone at 0.60-0.80 of ours on d128 bloom. Those cells are the H2H cells
(goodreads `c0_genre`, arXiv `c0_maincat` p 0.136), far above this gate's threshold. Like-for-like they are not a gap: Meta's
bloom search and payload kernels run separately, while ours is fused (`real_cell.py profile`, bs 16):

| cell | our fused scorer | Meta scorer | Meta bloom search + payload | Meta total | ours / Meta |
|---|---|---|---|---|---|
| arXiv `c0_maincat` | 100.6 µs | 85.6 µs | 32.9 µs | 118.5 µs | 0.85 |
| goodreads `c0_genre` | 29.9 µs | 28.8 µs | 31.2 µs | 60.0 µs | 0.50 |

What is real is that Meta's per-document exit makes its scoring part scale with p, while ours has a floor. This gate cuts the
floor only below one expected passing lane per tile. A per-lane exit at moderate p (compacting passing lanes before the
dot, or not writing failing slots) is a different kernel design; it is on the improvement list.

## The change

`bloom_bit_freq [m_bits]` (fraction of items with bit m set, from `bloom_transposed`) is registered by `SilverTorch` and
passed to `codesigned_probe_score_bloom` (a new op argument; the reference twin takes and ignores it). The scorer loads the
query's bit frequencies at program start (one vector load). `bound` is the minimum, an upper bound on the row's pass rate,
and the program votes `tl.max(keep)` only when `bound · BLOCK_P < 1`. The host sets the constexpr `GATED` for the bloom
scorer at `D_PAD ≤ 256` when the grid has at least `MIN_PROGRAMS` programs (`_host.gate_pays`).

**Shipped on bloom only.** The exact scorer's version of the gate (a `[C, V]` value-frequency table, the minimum over active
clauses) delivered the same low-p gain (0.56-0.68 at p 0.001 bs 16) but cost 1.4-7.8 % at p ≥ 0.01. Mechanism: the d128 exact
tile went from 96 to 108 registers (cubins read with `cuobjdump -res-usage`), so 5 → 4 resident CTAs an SM. The bloom gate
stays at 96. The exact scorer is unchanged.

**Two fixes along the way** (each found by the timing gate, mechanism first): the first cut computed the bound inside the
bit loop, right before the branch, which exposed the loads' latency in every program, and it gated bs 1, where about 200
programs (under one wave) cannot gain from skipped tiles. bs 1 cells cost +2-9 %. Moving the loads to program start and
requiring `MIN_PROGRAMS` made bs 1 neutral.

## Gates (final tree)

- **Bit-exact** ([`skip_gate.py`](skip_gate.py) `exact`): **360 / 360** cells `torch.equal` (ids and scores) against v2.3 at D
  128 / 192 / 768 × none / bloom / exact × p {0, 0.001, 0.003, 0.01, 0.1, 1} (+ an all-inactive query) × bs × k {100, 1000} ×
  n_probe {24, 1024}. Plus two new parity tests that assert the launch is `GATED` and the gate regime is hit (rare values,
  8,192 bits) and compare against the reference and the ungated `_impl`.
- **Library suite** 790 passed on a fresh inductor cache.
- **SASS:** compiled without the table (`GATED` off), the scorers still differ from v2.3, because the new pointer argument
  shifts the parameter offsets ([`sass_ungated.py`](sass_ungated.py)). Timing shows the ungated kernels (`none`, d768) at
  1.000.
- **Keep rule, kernel-only** (ABAB, 12 pairs, 95 % CI):

| D | mode | p | bs | v2.3 µs | after µs | after / v2.3 [95 % CI] | gated launch | scores equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|---|
| 128 | bloom | 0.001 | 1 | 5.9 | 5.9 | 1.005 [0.999, 1.011] | no | yes | 1410-1410 |
| 128 | bloom | 0.001 | 16 | 36.6 | 24.5 | 0.669 [0.667, 0.670] | yes | yes | 1410-1410 |
| 128 | exact | 0.001 | 1 | 5.1 | 5.1 | 1.002 [0.995, 1.009] | no | yes | 1410-1410 |
| 128 | exact | 0.001 | 16 | 41.6 | 41.6 | 1.010 [0.988, 1.032] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.003 | 1 | 6.1 | 6.0 | 0.996 [0.989, 1.003] | no | yes | 1320-1410 (unstable) |
| 128 | bloom | 0.003 | 16 | 37.4 | 30.3 | 0.811 [0.809, 0.813] | yes | yes | 1410-1410 |
| 128 | exact | 0.003 | 1 | 5.1 | 5.1 | 1.004 [0.998, 1.010] | no | yes | 1410-1410 |
| 128 | exact | 0.003 | 16 | 41.7 | 41.7 | 1.017 [0.980, 1.055] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.01 | 1 | 6.1 | 6.1 | 0.982 [0.948, 1.018] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.01 | 16 | 38.3 | 38.5 | 1.007 [1.006, 1.007] | yes | yes | 1410-1410 |
| 128 | exact | 0.01 | 1 | 5.1 | 5.1 | 1.002 [0.997, 1.007] | no | yes | 1410-1410 |
| 128 | exact | 0.01 | 16 | 42.1 | 42.0 | 1.000 [0.999, 1.001] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.1 | 1 | 6.3 | 6.3 | 1.001 [0.997, 1.005] | no | yes | 1410-1410 |
| 128 | bloom | 0.1 | 16 | 43.2 | 43.2 | 0.998 [0.997, 0.999] | yes | yes | 1410-1410 |
| 128 | exact | 0.1 | 1 | 5.1 | 5.1 | 1.004 [0.998, 1.011] | no | yes | 1410-1410 |
| 128 | exact | 0.1 | 16 | 44.1 | 44.1 | 1.014 [0.986, 1.042] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 1.0 | 1 | 10.7 | 10.7 | 0.982 [0.957, 1.008] | no | yes | 1140-1410 (unstable) |
| 128 | bloom | 1.0 | 16 | 64.4 | 64.7 | 1.004 [1.002, 1.006] | yes | yes | 1410-1410 |
| 128 | exact | 1.0 | 1 | 8.8 | 8.9 | 1.014 [0.990, 1.038] | no | yes | 1140-1410 (unstable) |
| 128 | exact | 1.0 | 16 | 65.5 | 65.7 | 1.003 [1.002, 1.004] | no | yes | 1140-1140 |
| 128 | none | – | 1 | 9.2 | 9.1 | 0.996 [0.986, 1.007] | no | yes | 1140-1140 |
| 128 | none | – | 16 | 60.5 | 60.6 | 1.000 [0.999, 1.002] | no | yes | 1140-1140 |
| 192 | bloom | 0.001 | 1 | 7.6 | 7.6 | 1.003 [1.000, 1.007] | no | yes | 1410-1410 |
| 192 | bloom | 0.001 | 16 | 54.6 | 31.7 | 0.580 [0.579, 0.581] | yes | yes | 1410-1410 |
| 192 | exact | 0.001 | 1 | 6.6 | 6.6 | 1.010 [0.996, 1.025] | no | yes | 1410-1410 |
| 192 | exact | 0.001 | 16 | 62.1 | 62.1 | 1.017 [0.979, 1.056] | no | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.003 | 1 | 7.6 | 7.6 | 0.988 [0.955, 1.021] | no | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.003 | 16 | 55.7 | 41.8 | 0.751 [0.751, 0.752] | yes | yes | 1410-1410 |
| 192 | exact | 0.003 | 1 | 6.6 | 6.6 | 1.001 [0.995, 1.008] | no | yes | 1410-1410 |
| 192 | exact | 0.003 | 16 | 62.2 | 62.2 | 1.000 [1.000, 1.001] | no | yes | 1140-1140 |
| 192 | bloom | 0.01 | 1 | 7.7 | 7.7 | 1.005 [0.994, 1.016] | no | yes | 1410-1410 |
| 192 | bloom | 0.01 | 16 | 57.1 | 57.2 | 1.002 [1.001, 1.002] | yes | yes | 1410-1410 |
| 192 | exact | 0.01 | 1 | 6.6 | 6.6 | 1.002 [0.999, 1.005] | no | yes | 1410-1410 |
| 192 | exact | 0.01 | 16 | 62.5 | 62.6 | 1.017 [0.979, 1.057] | no | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.1 | 1 | 7.9 | 7.9 | 1.001 [0.997, 1.005] | no | yes | 1410-1410 |
| 192 | bloom | 0.1 | 16 | 63.3 | 62.9 | 0.994 [0.993, 0.995] | yes | yes | 1410-1410 |
| 192 | exact | 0.1 | 1 | 6.6 | 6.6 | 1.001 [0.997, 1.005] | no | yes | 1410-1410 |
| 192 | exact | 0.1 | 16 | 65.2 | 65.2 | 1.018 [0.982, 1.056] | no | yes | 1140-1410 (unstable) |
| 192 | bloom | 1.0 | 1 | 12.9 | 12.9 | 0.997 [0.990, 1.004] | no | yes | 1410-1410 |
| 192 | bloom | 1.0 | 16 | 84.7 | 84.5 | 0.997 [0.997, 0.998] | yes | yes | 1410-1410 |
| 192 | exact | 1.0 | 1 | 10.6 | 10.7 | 1.003 [0.999, 1.007] | no | yes | 1410-1410 |
| 192 | exact | 1.0 | 16 | 83.6 | 83.6 | 1.000 [0.998, 1.001] | no | yes | 1140-1410 (unstable) |
| 192 | none | – | 1 | 12.1 | 12.1 | 1.003 [0.995, 1.011] | no | yes | 1140-1140 |
| 192 | none | – | 16 | 80.9 | 80.9 | 1.000 [0.999, 1.001] | no | yes | 1140-1140 |
| 768 | bloom | 0.001 | 1 | 5.3 | 5.3 | 1.001 [0.996, 1.005] | no | yes | 1410-1410 |
| 768 | bloom | 0.001 | 16 | 17.4 | 17.4 | 1.002 [1.000, 1.004] | no | yes | 1410-1410 |
| 768 | exact | 0.001 | 1 | 5.7 | 5.7 | 1.015 [0.976, 1.056] | no | yes | 1140-1410 (unstable) |
| 768 | exact | 0.001 | 16 | 30.3 | 30.3 | 1.000 [0.999, 1.000] | no | yes | 1140-1140 |
| 768 | bloom | 0.003 | 1 | 6.1 | 6.1 | 1.003 [0.998, 1.008] | no | yes | 1410-1410 |
| 768 | bloom | 0.003 | 16 | 17.9 | 17.8 | 0.996 [0.994, 0.998] | no | yes | 1410-1410 |
| 768 | exact | 0.003 | 1 | 6.3 | 6.3 | 1.008 [0.999, 1.018] | no | yes | 1410-1410 |
| 768 | exact | 0.003 | 16 | 30.5 | 30.5 | 0.998 [0.996, 0.999] | no | yes | 1140-1140 |
| 768 | bloom | 0.01 | 1 | 6.7 | 6.7 | 1.011 [0.997, 1.026] | no | yes | 1410-1410 |
| 768 | bloom | 0.01 | 16 | 18.6 | 18.5 | 0.999 [0.997, 1.001] | no | yes | 1410-1410 |
| 768 | exact | 0.01 | 1 | 7.3 | 7.3 | 1.000 [0.998, 1.003] | no | yes | 1410-1410 |
| 768 | exact | 0.01 | 16 | 31.2 | 31.1 | 0.999 [0.997, 1.000] | no | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.1 | 1 | 7.3 | 7.3 | 1.002 [0.996, 1.009] | no | yes | 1410-1410 |
| 768 | bloom | 0.1 | 16 | 24.2 | 24.2 | 1.002 [0.997, 1.006] | no | yes | 1410-1410 |
| 768 | exact | 0.1 | 1 | 8.3 | 8.3 | 1.000 [0.997, 1.004] | no | yes | 1410-1410 |
| 768 | exact | 0.1 | 16 | 35.2 | 35.2 | 1.000 [1.000, 1.001] | no | yes | 1140-1140 |
| 768 | bloom | 1.0 | 1 | 7.7 | 7.7 | 1.001 [0.996, 1.005] | no | yes | 1410-1410 |
| 768 | bloom | 1.0 | 16 | 29.2 | 29.2 | 1.000 [0.999, 1.001] | no | yes | 1410-1410 |
| 768 | exact | 1.0 | 1 | 8.6 | 8.6 | 1.000 [0.997, 1.004] | no | yes | 1410-1410 |
| 768 | exact | 1.0 | 16 | 38.4 | 38.3 | 1.000 [0.999, 1.001] | no | yes | 1140-1140 |
| 768 | none | – | 1 | 7.2 | 7.3 | 1.003 [0.999, 1.006] | no | yes | 1140-1140 |
| 768 | none | – | 16 | 34.3 | 34.3 | 1.000 [0.999, 1.002] | no | yes | 1140-1140 |

Gated cells (bloom, bs 16, p ≤ 0.003): **0.58-0.81**. Elsewhere 0.98-1.02. Two bloom cells' CIs sit just above 1 (d128 p 0.01
+0.7 %, p 1 +0.4 %). The same run's exact rows are identical code and read up to 1.003 [1.002, 1.004], so +0.3-0.4 % is the
method's floor; +0.7 % is slightly above it. This is borderline, and stated as such.

## Re-gate against campaign-v2.3 (1258a63e, incl. V1-FUSE; the controller moved this step to campaign-v2.4)

`phase.sh` with `BEFORE_TREE` = a `campaign-v2.3` worktree and `PKGS` = its `retrieve_v23`, on `dev/st-skip128` merged with
staging ae7fcd0: bit-exact **360 / 360**, library suite 800 passed. Keep rule: bloom bs 16 0.579 / 0.669 at p 0.001,
0.752 / 0.810 at p 0.003 (d192 / d128); elsewhere 0.98-1.02, d128 bloom p 1 bs 16 1.006 [1.005, 1.007]. The controller
accepted the borderline cells as within the method floor (option (a)). Raw outputs on pod b, `/scratch/st-skip128/phase-v23/`.

## Bundle gate (campaign-v2.3: ST-IDS + V2-HIGHP + ST-SKIP128, on `dev/st-skip128` merged with staging incl. H-INDCACHE)

Library suite **791 passed** (fresh inductor dir per run, H-INDCACHE); `tests/parity` + `tests/compile` **350 passed**;
wide-D graph capture **7 / 7**: SilverTorch none / bloom / exact at D 768 (`reduce-overhead` replay `torch.equal` to eager,
0 cudagraph skips, 0 graph breaks; ST-DLOOP's kernels and ST-IDS's epilogue run in every forward) and LiNR V2 at D 768
(V2-HIGHP's SPLIT body).

## What goes stale on adoption

**SilverTorch Triton bloom perf at D ≤ 256, bs ≥ the program threshold** (goodreads / arXiv / YFCC bloom cells and the synth
bloom legs). It changes only where the gate engages (p < 1/256); elsewhere it is within noise. Quality is unchanged
(bit-exact).

## Improvements found, not applied

1. **A register-neutral exact gate:** the exact gate pays at low p (0.56-0.68) but adds 12 registers at d128. A per-row flag
   computed once (e.g. by the query-side prep, outside the scorer) would leave one scalar load in the kernel. Expected:
   the same low-p gain, no high-p cost; price: one small launch per call eagerly.
2. **A per-lane exit at moderate p** (the scoring part's floor that Meta's per-document exit avoids): compacting passing
   lanes before the dot, or writing only finite slots. A different kernel design; benefit unmeasured.

## Scripts

| script | what |
|---|---|
| [`skip_gate.py`](skip_gate.py) | `exact` (bit-exact against v2.3) and `time` (the keep-rule grid); uses ST-DLOOP's `dloop_gate.py` helpers |
| [`sass_ungated.py`](sass_ungated.py) | SASS hashes of the scorers without the table, per tree |
| [`phase.sh`](phase.sh) | the GPU phase as run |
