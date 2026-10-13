# FULL-RERUN: every cited number at one tag

Roadmap step 7. The rerun regenerates every number the paper cites, at one frozen tag, into one `results.parquet`. It runs
under the user's rulings of 2026-10-12/13 ([decisions](../../decisions.md)):

- YFCC is out.
- LiNR runs the exact filter only.
- SilverTorch is Meta's code, adapted, on its bloom filter with the recheck off in timed code. The false positives are
  counted afterwards, untimed. The score path is fp16.
- PubMed runs on the dot-product build with the query's own document excluded.
- LiNR V3's effective pool is reported.
- Yambda-500m is the fifth dataset.

[`recipe.yaml`](recipe.yaml) lists every leg per pod, the data steps, the stages and the untimed steps. Nothing here is
citable until the gate in the recipe is green in [validation](../../validation.md).

| file | what |
|---|---|
| [`gen_suites.py`](gen_suites.py) → [`suites.yaml`](suites.yaml), [`suites-lean.yaml`](suites-lean.yaml) | stage 1: the adapted SilverTorch probe curves (clause, and bloom with the recheck off), the exact LiNR legs, V3 pools, the bloom width grid, Yambda |
| [`pick.py`](pick.py) → the timed suites (stage 2) | from seed 0's curves: bloom at k 100 (with V3 at matched recall), 1000 and 2048, per synth draw, and the clause reference row |
| [`check_suites.py`](check_suites.py) | CPU dry resolve of every suite plus the arm rules (recheck off, no LiNR bloom, no YFCC, no V2 bs 256 on PubMed) |
| [`sizing.py`](sizing.py) | GPU-h per leg and pod, the table below |

**Stages.**
1. Curves (quality only) and the legs that need no picks.
2. `pick.py` on seed 0's stage-1 records (controller C4).
3. The timed suites at every seed.
4. The final pass's suites (`evaluation/config/suites.yaml`: LiNR C1-C3, Meta's code as published for C5 and C7).
5. The untimed steps.

**Seeds** (controller C3):
- 0-2 where something is random: a SilverTorch build, a LiNR V3 pool, a synthetic draw.
- One seed for exact LiNR, except where it shares an interleave group with a seeded arm. There it is timed at every seed, and
  its quality is copied (`SEED_FREE_QUALITY`).
- Synthetic draw `d` is the config `<ds>-synth-s<d>` at job seed `d` ([datasets](../../../docs/system/datasets.md#synthetic-selectivity-attrs)).

**Not in the rerun:**
- LiNR V2 at bs 256 on PubMed (controller C2): the earlier TIMEOUT and the 2.55 s record stand.
- NUMBERS' own suites (`../numbers/`): these legs replace them. Their clause-mode bloom arms ran with the recheck on, which
  no longer matches the study.

## Sizing

From `sizing.py`. The basis is one seed's cost from the measured walls of the 2026-10-12/13 legs, the final pass's
`elapsed_s` (merged-4016, out-of-study arms dropped) and the NUMBERS estimates, times 1.2. Rough figures.

| profile | pod 1 GPU-h | pod e GPU-h | balanced wall | at 3.71 USD / h for both pods |
|---|---|---|---|---|
| full (curves, V3 pools at 3 seeds; clause reference row) | 106 | 88 | ≈ 97 h | ≈ 360 USD |
| lean (curves, V3 pools at seed 0; no clause row; final pass trimmed) | 58 | 48 | ≈ 53 h | ≈ 197 USD |
| credit | | | ≈ 31 h | 115 USD |

Neither profile fits the credit. Balancing the two pods means copying `pubmed-medcpt-dot` (~17 GB) and the LAION subsets to
pod e.

Trims, in order. Each line gives what it removes and the GPU-h it saves:
1. Curves and V3 pools at seed 0 (`--lean`): the picks read seed 0 anyway. Saves ~60.
2. The clause reference row. Saves ~18.
3. Final pass: `filter` at seed 0, no `deep` / `h2h` (not cited). Saves ~11.
4. k 1000 and k 2048 bloom timed at seed 0. Saves ~7.
5. Synth timed at draw 0 only. Saves ~7.
6. META-FORK grid: X / R from the accepted gate if the fork tree is unchanged at the tag; only the speed column is rerun. Saves ~2.
7. Bloom timed (k 100) at 2 seeds. Saves ~5.

After trims 1-7 the total is about 83 GPU-h, about 42 h on two balanced pods: still over the credit.

Beyond that, the only large lever is to reuse the exact LiNR and postfilter records (the final pass's, library `e16512f5`,
and NUMBERS' L legs) instead of rerunning them. That saves ~30 more, to about 50 GPU-h, which fits. It breaks "one tag for
everything" and is the user's call.
