# FULL-RERUN: every cited number at one tag, every random thing seeded

Roadmap step 7. It regenerates every number the paper cites, at the tag `final-freeze`, into one `results.parquet`. It runs
under the user's rulings of 2026-10-12/13 ([decisions](../../decisions.md)):

- YFCC is out.
- LiNR runs the exact filter only.
- SilverTorch is Meta's code, adapted, on its bloom filter with the recheck off in timed code. False positives are counted
  afterwards, untimed. The score path is fp16.
- PubMed runs on the dot-product build with the query's own document excluded.
- LiNR V3's effective pool is reported.
- Yambda-500m is the fifth dataset.

The user's GO (2026-10-13) sets the profile to **full**: three seeds wherever randomness matters, one for exact LiNR and the
oracle, no reuse of earlier records. The run is ordered **seed-major**: seed 0 of every leg first, then seed 1, then seed 2, so a
stop leaves a complete set at fewer seeds. Nothing here is citable until the gate in [`recipe.yaml`](recipe.yaml) is green in
[validation](../../validation.md).

| file | what |
|---|---|
| [`recipe.yaml`](recipe.yaml) | the plan: pods, seeds, the per-seed order, the gate |
| [`runbook-pod1.md`](runbook-pod1.md), [`runbook-pode.md`](runbook-pode.md) | the exact commands per pod: venv, data, launch, monitoring, kill and retry |
| [`queue.sh`](queue.sh) | one sequential driver per GPU, seed-major; calls the scripts below |
| [`run_leg.sh`](run_leg.sh) | one leg at one seed: resolved-arm check, `bench check` / `oracle` / `run` under the GPU lock, retry, upload to `fr/<leg>-<pod>-s<seed>` |
| [`gen_suites.py`](gen_suites.py) → [`suites.yaml`](suites.yaml) | stage 1: probe curves (clause, and bloom with the recheck off), the exact LiNR legs, V3 pools, the bloom width grid, Yambda, `fr-ceil-adapted` (the ceiling script's inputs). `--lean` → [`suites-lean.yaml`](suites-lean.yaml) is not the run's |
| [`pick.py`](pick.py), [`make_picks.sh`](make_picks.sh) | stage 2's timed suites from seed 0's curves: bloom at k 100 (with V3 at matched recall), 1000 and 2048, per synth draw, the clause reference row |
| [`prep_config.py`](prep_config.py) | a config dir: the dataset yamls, and one suites file merged from the final pass's, stage 1's and the picks; final-pass suites with a SilverTorch or V3 arm are raised to seeds 0-2 |
| [`check_suites.py`](check_suites.py) | CPU dry resolve plus the arm rules (recheck off, no LiNR bloom, no YFCC, no V2 bs 256 on PubMed) |
| [`untimed.sh`](untimed.sh), [`points.py`](points.py), [`metafork_grid.sh`](metafork_grid.sh) | the untimed steps: grade_bloom at the timed points, the C6 ceiling split, fp_adapted, score_gaps, and the META-FORK grid (int32 and fp16) |
| [`sizing.py`](sizing.py) | GPU-h per leg and pod |

**Seeds.**
- A SilverTorch build (k-means), a LiNR V3 pool (OPORP), a synthetic draw and the timing run at seeds 0, 1 and 2.
- Synthetic draw `d` is the config `<ds>-synth-s<d>`, run at job seed `d` ([datasets](../../../docs/system/datasets.md#synthetic-selectivity-attrs)).
- Exact LiNR and the oracle run once.
- An exact arm inside a seeded interleave group is timed at every seed, and its quality is copied (`SEED_FREE_QUALITY`).
- The untimed grade_bloom and ceiling steps take the seed (`SEED=`).
- fp_adapted (the bloom index has no k-means) and score_gaps are seed-free.
- The META-FORK grid is seed 0.

**Not in the rerun:** LiNR V2 at bs 256 on PubMed (controller C2: the earlier TIMEOUT and the 2.55 s record stand), and NUMBERS'
own suites (`../numbers/`).

## Sizing

From `sizing.py`. The basis is one seed's cost from the measured walls of the 2026-10-12/13 legs, the final pass's
`elapsed_s` (merged-4016, out-of-study arms dropped) and the NUMBERS estimates, times 1.2. The figures are rough. The new legs
(PubMed synth, LAION synth bloom, k 2048 bloom) and the META-FORK grid have no measured basis.

| profile | pod 1 GPU-h | pod e GPU-h | wall (both pods at once) | at 3.71 USD / h for both pods |
|---|---|---|---|---|
| **full, the run** | 119 | 125 | ≈ 125 h | ≈ 463 USD |
| its seed-0 pass | 58 | 56 | ≈ 58 h | ≈ 215 USD |
| credit | | | ≈ 31 h | 115 USD |

The credit covers about half of the seed-0 pass. Each pod reaches about this far:
1. Stage 1: ~20 h.
2. The picks.
3. The timed bloom suites: ~6 h.
4. The untimed steps.
5. Part of the final pass's suites.

Not reached on the credit: the rest of the final pass at seed 0, the META-FORK grid, the clause reference row, and seeds 1-2.
The user is told the top-up: ≈ 100 USD for the rest of seed 0, and ≈ 350 USD for the full run.
