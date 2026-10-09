# V2-FILL — the wide `-inf` fill in `fused_masked_knn_topk` at D_PAD ≤ 256

Roadmap step V2-FILL (campaign-v2.5). At D_PAD ≤ 256, LiNR V2's scorer was found to be bound by its `-inf` fill.
Current state: [validation](../../../validation.md) row *V2-FILL*; mechanism: [kernels](../../../system/kernels.md#fused_masked_knn_topk--prefilterknn-sparse-path)
("Launch grid", "Count width").
A100-SXM4-80GB (pod b, GPU 0). Before = campaign-v2.4 (library d67d6263) as package `retrieve_v24` (`../st-ids/make_pkg.sh`).
Raw outputs: Hub `artifacts/v2-fill/` ([hub-index](../../hub-index.md)). Plan and prediction: `.chains/st-dloop/` (00:30).

## The change
v2.4's D_PAD ≤ 256 body was one grid-strided loop over all `cdiv(P, 32)` tiles of a row. It scored a tile below `count` and
stored 32 lanes of `-inf` above it. At low p almost every iteration was a 32-lane store. Now the body runs two loops:
- **Scoring loop:** strides over the row's counted tiles only, with q held across tiles.
- **Fill loop:** stores `-inf` past them in 1,024-lane chunks, strided over the same programs.

This is V2-HIGHP's SPLIT structure without the per-tile q reload. Each lane is still one `tl.sum` over D in the same layout,
so scores are bit-identical. The D_PAD > 256 (SPLIT) body is untouched.

The count is narrowed to int32 only at D_PAD ≤ 128, for the reason in the next section.

## Exploration ([`fill_variants.py`](fill_variants.py), [`sass_variants.py`](sass_variants.py))
Kernel-only, ABAB, against v2.4 with an A/A arm. Every variant's `[B, P]` scores were `torch.equal` to v2.4's.

- **Measured as predicted at low p.** The split bodies ran 0.22-0.38 at p ≤ 0.01 and 0.64-0.78 at p 0.1 (d64-d256, bs 1
  and 16).
- **The prediction missed p 1.** With the int64 count, d128 bs 16 ran 1.05-1.08, outside the A/A floor of 0.999-1.002.
  - num_stages 1 changed nothing.
  - Keeping v2.4's exact branched tile body inside the split loop changed nothing either (Hb, 1.06-1.08).
- **The difference is the loop bound.** The same body bounded by the constexpr `cdiv(P, BLOCK_N)` ran 0.93 (Hf). With a
  bound from the int64 `counts`, the induction variable and tile offsets are 64-bit: 248 vs 200 SASS lines at d128.
  - Narrowed to int32, d128 p 1 bs 16 went from 1.06 to 0.87.
- **At D_PAD 256 (d192, d256) narrowing reversed.**

  | body | registers | p 1 bs 16 |
  |---|---|---|
  | int32 count | 40 | 0.95-1.06 |
  | int64 count | 32 | 0.68 |
  | v2.4 | 40 | 1.00 |

  - At 8 warps, 40 registers allow 6 resident programs an SM, not 8. The 864 programs then need a second wave that runs
    one-third full.
  - So v2.4's d192 body was paying for that second wave, and the int64 body fits in one.
- **D_PAD 64:** int32 compiles to 25 registers and int64 to 32. int32 is better at p 1 (0.97 vs 1.04).

**Shipped:** q held, and the count narrowed at D_PAD ≤ 128 only (`INT32_COUNT`). This depends on the compiler's register
allocation. `fill_variants.py` records registers per body, and a toolchain change should re-check it.

## Gates ([`phase.sh`](phase.sh), [`fill_gate.py`](fill_gate.py), [`sass_fmkt.py`](sass_fmkt.py))
- **Bit-exact** ids and scores, public op and bucketed `_impl`, against campaign-v2.4: **120 / 120**.
  - Sizes: 0.8 M d128, 3 M d64 / d128 / d192, 10 M d768.
  - Cells: p {0, 0.001, 0.01, 0.1, 1} plus the skewed batch, × bs {1, 16} × k {100, 1000}.
- **SASS:** D 768 / 1024 (the SPLIT body) identical to v2.4, 5 / 5 cubins.
- **Tests:** LiNR compile + fmkt parity 29 passed; library suite **800 passed**. Each run used a fresh inductor dir.
- **Keep rule:** kernel-only, ABAB, 12 pairs, 95 % CI, after / before:

| width | p 0.001 | p 0.01 | p 0.1 | p 1 | skewed (2 rows at 0.12) |
|---|---|---|---|---|---|
| 0.8 M d128 bs 1 / 16 | 0.34 / 0.24 | 0.35 / 0.31 | 0.59 / 0.58 | 1.008 / 0.85 | 0.63 / 0.75 |
| 3 M d128 | 0.23 / 0.22 | 0.28 / 0.33 | 0.70 / 0.69 | 1.000 / 0.87 | 0.80 / 0.75 |
| 3 M d64 | 0.25 / 0.24 | 0.30 / 0.31 | 0.61 / 0.60 | 1.006 / 0.97 | 0.68 / 0.77 |
| 3 M d192 | 0.25 / 0.22 | 0.35 / 0.36 | 0.67 / 0.66 | 0.85 / 0.69 | 0.73 / 1.007 |
| 10 M d768 (unchanged) | 0.99 / 1.00 | 1.01 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

  Three cells above 1 were rechecked with 16 pairs and an A/A arm (v2.4 against itself) in the same run:
  - **p 1 bs 1:** the library read 1.011 at 3 M d64 and 1.013 at 0.8 M d128, against A/A 1.013 and 1.017.
  - **d192 skewed bs 16:** 1.000 [0.993, 1.008] against A/A 0.999.

  All are inside the identical-code floor, which reaches 1.013-1.017 at bs 1 p 1.
