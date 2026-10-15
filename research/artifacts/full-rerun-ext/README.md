# TUNE-EXT: extra curve points for the cells that miss recall 0.90

User, 2026-10-14 15:30Z (`.chains/control/2026-10-14-090000000-user-fr-scope-seed0-plus-variance.md`): the cells whose seed-0 SilverTorch curve
never reaches oracle recall@100 0.90 within n_probe <= n_lists / 4 get two more curve points, at n_lists / 2 and 3 n_lists / 4, at the largest
n_lists the cell used; then picks, timed bloom (LiNR re-timed in the same run), grade_bloom and clause-row points for those cells only.
Same tag (`final-freeze`), seed 0 only, existing picks and records unchanged. It runs after seed 0 and the clause row, before the seed variance.
Results and their state go in [validation](../../validation.md); raw outputs on the Hub as `fr/ext-<leg>-<pod>-s0`.

| file | what |
|---|---|
| [`gen_ext.py`](gen_ext.py) → [`suites-ext-curve.yaml`](suites-ext-curve.yaml) | the cells (from the `fr/curve*-s0` legs) and their quality-only curve suites, one per (pod that timed the dataset at seed 0, dim, real / synth): 13 cells, 13 jobs |
| [`pick_ext.py`](pick_ext.py) | after the ext curves: pick.py's rule over each cell's whole curve, only points not already in the seed-0 picks; writes the timed bloom suites, the clause-row suites (SilverTorch exact only) and the grade_bloom point files |

The cells (best recall@100 within n_probe <= n_lists / 4): goodreads-synth p0001 (pod e, bloom + clause, 4096 lists, 0.835); on pod 1 laion1m
c0_domain and tags4 (bloom + clause, 8192, 0.846 / 0.833), laion3m c0_domain (16384, 0.871), laion10m c0_domain (65536, 0.897), laion1m-synth
p0001 / p001 (clause, 512, 0.681 / 0.872), laion3m-synth p0001 (clause, 2048, 0.829). The LAION 1M / 3M synth cells are clause only: they get
curve points but no timed point, since pick.py times bloom, and its clause row covers the real datasets only.

Run (pod 1, from the FR tree, after seed 0 and the clause row):
```bash
export TREE=/scratch/wt/fr VENV=/venvs/fr CFG=/scratch/fr/ext/config FR=/scratch/fr LOCK=/scratch/gpu0.lock
mkdir -p $CFG && cp /scratch/fr/config/*.yaml $CFG/ && cat suites-ext-curve.yaml >> $CFG/suites.yaml
python ../full-rerun/check_suites.py $CFG $CFG/suites.yaml --only fr-ext-curve-pod1-d256,fr-ext-curve-synth-pod1-d256
cd $TREE/evaluation
../research/artifacts/full-rerun/run_leg.sh ext-curve-pod1 0 quality fr-ext-curve-pod1-d256 laion1m laion3m laion10m
../research/artifacts/full-rerun/run_leg.sh ext-curve-synth-pod1 0 quality fr-ext-curve-synth-pod1-d256 laion1m-synth laion3m-synth
# pod e: run_leg.sh ext-curve-synth-pode 0 quality fr-ext-curve-synth-pode-d128 goodreads-synth
# then, with every ext curve leg's results.parquet fetched next to the stage-1 ones:
python pick_ext.py $FR/picks.yaml $FR/ext/picks fr-curve-and-v3pool-and-ext-curve-parquets...
```
