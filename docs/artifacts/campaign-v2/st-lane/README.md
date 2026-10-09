# ST-LANE — a per-lane exit in the d128 probe scorer (measured not viable) and independent bloom loads (shipped)

Roadmap step ST-LANE (campaign-v2.5): make our d128 SilverTorch scorer scale with p as Meta's per-document exit does.
Current state: [validation](../../../validation.md) row *ST-LANE*; mechanism: [kernels](../../../system/kernels.md#codesigned_probe_score--ivf--int8--bloom).
A100-SXM4-80GB (pod b, GPU 0). Before = campaign-v2.4 (library d67d6263) as package `retrieve_v24` (`../st-ids/make_pkg.sh`).
Raw outputs: Hub `artifacts/st-lane/` ([hub-index](../../hub-index.md)). Plan, prediction and findings: `.chains/st-dloop/`
(22:30 plan, 23:15 findings).

## 1. Kernel split, measured first ([`kernel_split.py`](kernel_split.py))
d128 bloom, N 2 M, n_lists 1024, n_probe 24, bs 16, kernel-only µs, by ablation of a copy of the v2.4 body:

| p | full | no code load + dot | loads + dot | dot alone | store | empty | Meta scorer + mask |
|---|---|---|---|---|---|---|---|
| 0.001 | 45.4 | 19.2 | 26.2 | ≈ 0 | ≈ 0.6 | 13.7 | 27.3 + 13.9 |
| 0.01 | 38.3 | 15.6 | 22.7 | 0.2 | 1.0 | 11.1 | 26.7 + 11.9 |
| 0.136 | 44.4 | 15.8 | 28.6 | 3.9 | 1.8 | 11.1 | 37.6 + 12.2 |
| 1.0 | 64.6 | 15.9 | 48.7 | 2.2 | 3.6 | 11.2 | 90.1 + 11.8 |

## 2. The per-lane exit: built, measured, not viable ([`lane_proto.py`](lane_proto.py))
Compacted lanes: a scan of `keep` into a per-program scratch slot, a barrier, then chunks of 32 passing lanes gathered and
dotted (bit-exact). Against v2.4, bs 16: 1.22 (p 0.001, where v2.4 already skips), 0.91 (0.01), 0.94 (0.03), 1.03 (0.136),
1.00 (1.0). bs 1 up to 1.45 at p 1. The prediction (0.55-0.7 at p 0.01) was wrong. The split measured work per component;
the time is each program's latency chain (probe table → 10 bloom word loads → code gather → store). Compaction removed the
tile work and added a chain of its own (scan, barrier, scratch round trip, scatter).

Tiles do not move the floor either ([`tile_lowp.py`](tile_lowp.py)): 128 / 512 / 1024 lanes and 8 warps against the shipped
256 × 4, none better at bs 16 at any p (1.00-2.3×). At bs 1, 128 × 4 is about −5 %, a separate small-grid tuning question.

## 3. Shipped: independent bloom word loads
Each queried bit's word load was masked by the running `keep`, so the 10 loads waited on each other; masked by `valid`
they are independent (8-byte words shared by 64 lanes), and `keep` is the same. Gates on the final tree
([`lane_gate.py`](lane_gate.py), [`phase.sh`](phase.sh)):
- **Bit-exact 360 / 360** against v2.4 (D 128 / 192 / 768 × none / bloom / exact × p {0, 0.001, 0.003, 0.01, 0.1, 1} + an
  inactive query × bs × k × n_probe).
- **SASS** ([`../st-skip128/sass_ungated.py`](../st-skip128/sass_ungated.py)): exact 3 / 3 cubins and the id epilogue
  identical to v2.4. The bloom scorer's `none` variants (3) are identical; its 3 bloom variants changed by design.
- **Library suite** 800 passed (fresh inductor dir per run).
- **Keep rule**, kernel-only ABAB, 12 pairs, 95 % CI:

| D | mode | p | bs | v2.4 µs | after µs | after / v2.4 [95 % CI] | scores equal | sm_mhz |
|---|---|---|---|---|---|---|---|---|
| 128 | bloom | 0.001 | 1 | 5.9 | 5.8 | 0.989 [0.981, 0.998] | yes | 1410-1410 |
| 128 | bloom | 0.001 | 16 | 24.4 | 24.3 | 0.994 [0.993, 0.996] | yes | 1410-1410 |
| 128 | exact | 0.001 | 1 | 5.1 | 5.1 | 1.008 [0.989, 1.026] | yes | 1410-1410 |
| 128 | exact | 0.001 | 16 | 41.5 | 41.5 | 1.012 [0.986, 1.039] | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.003 | 1 | 6.1 | 5.8 | 0.962 [0.955, 0.970] | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.003 | 16 | 30.3 | 29.8 | 0.985 [0.981, 0.988] | yes | 1410-1410 |
| 128 | exact | 0.003 | 1 | 5.1 | 5.1 | 1.003 [0.997, 1.010] | yes | 1410-1410 |
| 128 | exact | 0.003 | 16 | 41.7 | 41.7 | 1.007 [0.990, 1.024] | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.01 | 1 | 6.1 | 5.8 | 0.956 [0.951, 0.961] | yes | 1410-1410 |
| 128 | bloom | 0.01 | 16 | 38.6 | 37.2 | 0.966 [0.964, 0.968] | yes | 1410-1410 |
| 128 | exact | 0.01 | 1 | 5.1 | 5.1 | 1.003 [0.995, 1.010] | yes | 1410-1410 |
| 128 | exact | 0.01 | 16 | 42.0 | 42.0 | 1.001 [1.000, 1.002] | yes | 1140-1410 (unstable) |
| 128 | bloom | 0.1 | 1 | 6.3 | 5.8 | 0.927 [0.922, 0.932] | yes | 1410-1410 |
| 128 | bloom | 0.1 | 16 | 43.1 | 39.9 | 0.927 [0.927, 0.928] | yes | 1410-1410 |
| 128 | exact | 0.1 | 1 | 5.1 | 5.1 | 1.004 [0.999, 1.009] | yes | 1410-1410 |
| 128 | exact | 0.1 | 16 | 44.0 | 44.1 | 1.001 [1.000, 1.002] | yes | 1140-1410 (unstable) |
| 128 | bloom | 1.0 | 1 | 10.7 | 9.8 | 0.905 [0.889, 0.922] | yes | 1200-1410 (unstable) |
| 128 | bloom | 1.0 | 16 | 64.7 | 62.1 | 0.961 [0.959, 0.963] | yes | 1410-1410 |
| 128 | exact | 1.0 | 1 | 8.9 | 8.9 | 1.013 [0.988, 1.038] | yes | 1410-1410 |
| 128 | exact | 1.0 | 16 | 65.5 | 65.8 | 1.003 [1.002, 1.005] | yes | 1140-1140 |
| 128 | none | – | 1 | 9.2 | 9.1 | 0.999 [0.990, 1.008] | yes | 1140-1140 |
| 128 | none | – | 16 | 60.6 | 60.6 | 0.999 [0.998, 1.001] | yes | 1140-1140 |
| 192 | bloom | 0.001 | 1 | 7.6 | 7.7 | 1.012 [1.007, 1.018] | yes | 1410-1410 |
| 192 | bloom | 0.001 | 16 | 31.7 | 31.7 | 1.000 [0.999, 1.001] | yes | 1410-1410 |
| 192 | exact | 0.001 | 1 | 6.6 | 6.6 | 1.001 [0.999, 1.004] | yes | 1410-1410 |
| 192 | exact | 0.001 | 16 | 62.1 | 62.0 | 1.017 [0.980, 1.057] | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.003 | 1 | 7.6 | 7.7 | 1.014 [1.011, 1.017] | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.003 | 16 | 41.8 | 40.9 | 0.978 [0.977, 0.979] | yes | 1410-1410 |
| 192 | exact | 0.003 | 1 | 6.6 | 6.6 | 0.996 [0.991, 1.000] | yes | 1410-1410 |
| 192 | exact | 0.003 | 16 | 62.2 | 62.2 | 1.017 [0.979, 1.057] | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.01 | 1 | 7.7 | 7.7 | 1.004 [0.998, 1.010] | yes | 1410-1410 |
| 192 | bloom | 0.01 | 16 | 57.2 | 55.0 | 0.961 [0.960, 0.962] | yes | 1410-1410 |
| 192 | exact | 0.01 | 1 | 6.6 | 6.6 | 1.002 [0.998, 1.005] | yes | 1410-1410 |
| 192 | exact | 0.01 | 16 | 62.6 | 62.6 | 1.017 [0.980, 1.056] | yes | 1140-1410 (unstable) |
| 192 | bloom | 0.1 | 1 | 7.9 | 7.8 | 0.985 [0.981, 0.989] | yes | 1410-1410 |
| 192 | bloom | 0.1 | 16 | 62.9 | 58.3 | 0.928 [0.927, 0.930] | yes | 1410-1410 |
| 192 | exact | 0.1 | 1 | 6.6 | 6.6 | 1.002 [0.996, 1.007] | yes | 1410-1410 |
| 192 | exact | 0.1 | 16 | 65.2 | 65.2 | 1.017 [0.980, 1.055] | yes | 1140-1410 (unstable) |
| 192 | bloom | 1.0 | 1 | 12.9 | 12.4 | 0.953 [0.932, 0.974] | yes | 1140-1410 (unstable) |
| 192 | bloom | 1.0 | 16 | 84.5 | 80.6 | 0.953 [0.952, 0.954] | yes | 1410-1410 |
| 192 | exact | 1.0 | 1 | 10.7 | 10.7 | 1.003 [0.999, 1.007] | yes | 1140-1410 (unstable) |
| 192 | exact | 1.0 | 16 | 83.6 | 83.5 | 0.999 [0.998, 1.000] | yes | 1140-1140 |
| 192 | none | – | 1 | 12.1 | 12.1 | 1.003 [1.000, 1.006] | yes | 1140-1140 |
| 192 | none | – | 16 | 80.9 | 80.8 | 1.000 [0.999, 1.001] | yes | 1140-1140 |
| 768 | bloom | 0.001 | 1 | 5.3 | 5.2 | 0.987 [0.969, 1.004] | yes | 1410-1410 |
| 768 | bloom | 0.001 | 16 | 17.4 | 17.4 | 1.002 [0.999, 1.005] | yes | 1410-1410 |
| 768 | exact | 0.001 | 1 | 5.7 | 5.7 | 1.001 [0.994, 1.008] | yes | 1140-1410 (unstable) |
| 768 | exact | 0.001 | 16 | 30.3 | 30.3 | 1.000 [0.999, 1.001] | yes | 1140-1140 |
| 768 | bloom | 0.003 | 1 | 6.1 | 5.9 | 0.954 [0.922, 0.987] | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.003 | 16 | 17.8 | 17.8 | 0.996 [0.994, 0.998] | yes | 1410-1410 |
| 768 | exact | 0.003 | 1 | 6.3 | 6.4 | 1.004 [0.999, 1.010] | yes | 1410-1410 |
| 768 | exact | 0.003 | 16 | 30.6 | 30.5 | 1.016 [0.977, 1.056] | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.01 | 1 | 6.7 | 6.4 | 0.948 [0.913, 0.984] | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.01 | 16 | 18.6 | 18.6 | 0.998 [0.996, 0.999] | yes | 1410-1410 |
| 768 | exact | 0.01 | 1 | 7.3 | 7.3 | 1.004 [1.000, 1.008] | yes | 1410-1410 |
| 768 | exact | 0.01 | 16 | 31.2 | 31.2 | 0.999 [0.998, 1.001] | yes | 1140-1410 (unstable) |
| 768 | bloom | 0.1 | 1 | 7.3 | 6.9 | 0.954 [0.951, 0.956] | yes | 1410-1410 |
| 768 | bloom | 0.1 | 16 | 24.2 | 24.0 | 0.996 [0.995, 0.996] | yes | 1410-1410 |
| 768 | exact | 0.1 | 1 | 8.3 | 8.3 | 1.000 [0.997, 1.004] | yes | 1140-1410 (unstable) |
| 768 | exact | 0.1 | 16 | 35.2 | 35.2 | 1.000 [0.999, 1.001] | yes | 1140-1140 |
| 768 | bloom | 1.0 | 1 | 7.7 | 7.2 | 0.929 [0.910, 0.950] | yes | 1410-1410 |
| 768 | bloom | 1.0 | 16 | 29.2 | 28.8 | 0.987 [0.984, 0.989] | yes | 1410-1410 |
| 768 | exact | 1.0 | 1 | 8.8 | 8.7 | 0.999 [0.990, 1.007] | yes | 1410-1410 |
| 768 | exact | 1.0 | 16 | 38.4 | 38.4 | 0.999 [0.996, 1.001] | yes | 1140-1410 (unstable) |
| 768 | none | – | 1 | 7.3 | 7.3 | 0.998 [0.990, 1.007] | yes | 1140-1140 |
| 768 | none | – | 16 | 34.3 | 34.3 | 1.001 [1.000, 1.002] | yes | 1140-1140 |

Bloom is faster or level everywhere except d192 bs 1 at p 0.001 / 0.003 (1.012 / 1.014). The same run's identical-code
exact rows read up to 1.013-1.017, so these are inside the method floor.

## What goes stale on adoption
SilverTorch Triton **bloom** perf at every width (−0 to −10 %). `none` and exact are unchanged (identical machine code).
Quality is unchanged (bit-exact).

## Scripts
| script | what |
|---|---|
| [`kernel_split.py`](kernel_split.py) | the ablation split, with Meta's scorer and mask on the same cells |
| [`lane_proto.py`](lane_proto.py) | the compacted-lane prototype against v2.4 |
| [`tile_lowp.py`](tile_lowp.py) | the shipped bloom tile against overrides across p |
| [`lane_gate.py`](lane_gate.py), [`phase.sh`](phase.sh) | the gates as run |
