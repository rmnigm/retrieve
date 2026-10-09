# CLAUSE-SKIP — clause kernels skip the item attrs of clauses the query leaves inactive

Roadmap CLAUSE-SKIP (campaign-v2.7 bundle with BLOOM-BUILD-CHUNK). Current state: [validation](../../../validation.md)
row *CLAUSE-SKIP*; mechanism: [kernels § Shared kernel helpers](../../../system/kernels.md#shared-kernel-helpers-opstritoncommonpy)
("Inactive clauses"). A100-SXM4-80GB, pod b. Raw outputs: Hub `artifacts/clause-skip/` ([hub-index](../../hub-index.md)).

## Why
The v2.6 "V1 / V2 clause regression" on arxiv-synth was the table width, not M1. SYNTH-TRIM widened the synth attrs from
7 to 10 clauses, and `clause_pass` loaded every clause's item attrs per item and batch row, whichever clauses the query
used: +≈ 0.7 ms at bs 16 on 3 M, flat in p, growing with the batch (pod 1's profile: `_clause_mask_kernel` 658 → 1072 µs,
`_clause_compact_kernel` 535 → 1204 µs). The compact kernel also rose from 56 to 121 registers. A clause left at `-1`
passes whatever the item holds, so skipping its loads is bit-exact.

## Shipped: form B, a uniform branch per clause (`if q_c != -1`)
Forms measured, all bit-exact ([`cs_gate.py`](cs_gate.py) on goodreads and the 10-clause arxiv-synth; registers from
[`regs.py`](regs.py)):

| form | 10-clause synth, bs 16 | goodreads 1-clause sweeps, bs 16 | goodreads `all4`, bs 1, V2 graph | compact registers (goodreads / synth10) |
|---|---|---|---|---|
| staging | — | — | — | 56 / 121 |
| A: masked loads | V1 0.85-0.87, V2 0.79-0.91 | V1 0.53-0.54, V2 0.62-0.65 | 1.038 | 255 + spills / 118 |
| **B: branch (shipped)** | **V1 0.82, V2 0.65-0.86** | **V1 0.50-0.51, V2 0.55-0.61** | **1.095** | **96 / 48** |
| C / D: q values first (bitmask) | ≈ A / ≈ B | ≈ A / ≈ B | 1.040 / 1.093 | — |
| E: B with compact at 4 warps | — | ≈ B | 1.143 | 48 / 32 |

Elsewhere under B:
- **SilverTorch exact:** triton graph 0.88-0.98; official exact 0.55-0.56 at bs 16. Eager triton 0.98-0.99 (host-bound).
- **goodreads `all4`** (every clause active, nothing to skip): 0.96-0.99, except the cell above.

**Known cost, accepted by the controller (2026-10-11):** with every clause active, V2 at bs 1 in graph mode costs +9.5 %
(+26 µs of 0.27 ms). Every form fails that cell.
- **Ruled out by measurement:** the q-value dependency (C / D) and registers (E).
- **Kernel device time** in an eager profile is within noise (124 vs 118 µs, against a same-code 117.6 vs 109.5 µs).
- **The mechanism is open.**

## Gates
- **Bit-exact:** every row against staging c354bb5 (package `retrieve_stg`); V1 / V2 / SilverTorch exact triton + official
  × 3 sweeps × bs 1 / 16 on each dataset.
- **SASS** ([`sass_ops.py`](sass_ops.py), every Triton op once): the five clause kernels change (`_clause_compact`,
  `_clause_mask` × 2, `_clause_mask_packed`, `_codesigned_probe_score_exact`); every other cubin is identical.
- **Library suite** 820 passed on a fresh inductor dir.

arxiv-synth's 10-clause table was built on pod b with `eval-data synth-filter --dataset arxiv-synth` (seed 20261008,
rates `synth_filter.RATES`).
