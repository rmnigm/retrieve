# YFCC int8-precision mechanism check

Controller 2026-10-10 (from exhibits): is SilverTorch's recall ceiling on YFCC 10 M d192 the single global int8 item scale? Quality
only; ran at campaign-v2.6 (20e83bfc; the leg started after the tag, the directory keeps the v2.5 name it was written under). Driver [`../../campaign-v2/v-pod1-run/yfcc-int8.sh`](../../campaign-v2/v-pod1-run/yfcc-int8.sh); Hub
`artifacts/yfcc-int8`.

| file | what |
|---|---|
| [`int8_check.py`](int8_check.py) | one SilverTorch triton index (n_lists 4096, n_probe 256), its own probed clusters per query; recall_oracle@100 with the shipped module, the same arithmetic recomputed densely (method check), per-row int8 item scales on the int32 path, and fp16 scoring |

Outcome (pod 1 GPU 0, 2026-10-09, yfcc10m-synth d192, clause, bs 16 query pool, k 100, seed 0; Hub `artifacts/yfcc-int8`):

| sweep (pass) | shipped | dense global int8 | dense per-row int8 | dense fp16 |
|---|---|---|---|---|
| `p1` (1.0) | 0.544 | 0.544 | 0.753 | 0.972 |
| `p001` (0.010) | 0.761 | 0.761 | 0.874 | 0.976 |

The dense recompute matches the shipped module (≤ 3·10⁻⁵, ties), so the method holds. On the same probes fp16 reaches 0.97-0.98:
the probes are not the ceiling; the global int8 item scale costs 21-43 points and per-row scales win back about half of it.
