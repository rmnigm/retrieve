# st-rerun: the at-risk head-to-head settings on the current SilverTorch IVF

Brief `.chains/pod-h-rerun/2026-10-17-120000000-brief-pod-h-rerun.md`. Since the final run, SilverTorch's IVF assigns by
maximum inner product in k-means and probes by plain inner product (`staging` `61045809`); LiNR is unchanged. The question:
does any head-to-head verdict flip on the current code? One A100-SXM4-80GB (pod a100-x1-h, driver 570.172.08, torch
2.10.0+cu128, triton 3.6.0, CUDA 12.8), `staging` `c4598123` on branch `dev/st-rerun`, library `code_version` `fed664d6`,
clean tree, `silvertorch` extension `so_sha256` `90fc7d06…`. Session scripts with no gate: **not citable**
([validation](../../validation.md#silvertorch-rerun-on-the-current-ivf-not-citable)). Records and logs are on the Hub under
`artifacts/st-rerun/<dataset>/` ([hub-index](../hub-index.md)).

| file | what |
|---|---|
| [`head_to_head_probe_budgets.csv`](head_to_head_probe_budgets.csv) | the final run's 312 settings (`results/tables/` on `origin/export`), the input |
| [`select.py`](select.py) | the at-risk settings ([`at_risk.csv`](at_risk.csv): ratio above 0.7 or never reached, Yambda out; 112) and each workload's timed SilverTorch grid from the final run's records ([`plan.json`](plan.json), 77 groups) |
| [`gen_suites.py`](gen_suites.py) | `plan.json` → one harness suite per dataset ([`suites.yaml`](suites.yaml)) |
| [`gen_followup.py`](gen_followup.py) | deeper points for the workloads whose setting the final run reached and the rerun's grid does not |
| [`run.sh`](run.sh) | the driver: per dataset `bench oracle`, then one `bench run --mode graph --interleave` (GPU 0, own inductor cache) |
| [`check_subset.sh`](check_subset.sh), [`check_laion.py`](check_laion.py) | the restaged LAION inputs against pod g's subset manifests and against the final run's records |
| [`reselect.py`](reselect.py) | the final run's selection rule on the rerun's records → [`reselect.csv`](reselect.csv) (with the follow-up points) and [`reselect_same_grid.csv`](reselect_same_grid.csv) (without) |
| [`points.py`](points.py) | every SilverTorch and LiNR point against the same point in the final run ([`points.csv`](points.csv)) |
| [`summary.py`](summary.py) | the counts and the named settings below |
| [`upload.py`](upload.py), [`upload.sh`](upload.sh) | one Hub commit per dataset with `bench`'s manifest, read back and checked |

## Method

- **Settings.** `head_to_head_probe_budgets.csv` from `origin/export` (`dev/export-r4`, which the brief names, is not on
  `origin`; `export` carries the same table and it reproduces `NUMBERS.md`'s counts exactly). 112 at risk (70 above 0.7×,
  42 not reached), Yambda excluded.
- **Grid.** Per (dataset, filter, batch, k): every (`n_lists`, `n_probe`) the final run timed for that workload at seed 0
  (its records in `results/records/records.parquet`), plus `n_lists` / 4 wherever the grid had points deeper than that, so a
  within-quarter pick can appear. Bloom 1,024 bits / 5 hashes, recheck off, graph mode only (every selection reads graph),
  seed 0 only (no table reads seeds 1-2), the harness's timing protocol. One process per dataset; the arms interleaved as the
  final run's suites were (`by: [algo, n_lists]`), so SilverTorch and the LiNR arms are timed in the same job.
- **LiNR.** The final run's pick per setting (V1 / V2, or V3 at its pool), timed again in the same job.
- **Selection.** `results/build_tables.py`'s rule: the fastest timed SilverTorch point reaching the target (oracle
  recall@k) with `n_probe` ≤ `n_lists` / 4, else the fastest at any depth (`deeper`), else `not_reached`; SilverTorch /
  LiNR against the final run's tie band per dataset.
- **Follow-up.** Where a setting the final run reached is not reached on the grid, a second job per dataset times the
  doubling ladder up to `n_lists` / 4 and `n_lists` / 2, 3 `n_lists` / 4 (the final run's follow-up depths), with LiNR again
  in the same job; such a point's ratio is to the LiNR of its own job. Both views are reported.
- **Safe settings.** The grid runs every batch × k of a filter, so some settings outside the 112 were timed too; one counts
  only where its own old pick point and its LiNR variant were both timed.
- **LAION inputs.** Pod g's rebuilt archive, rsynced to `/data/incoming/eval-laion30m` (24 GB), staged by hardlink as
  `data/laion30m`; 1 M / 3 M / 10 M rebuilt with `eval-data subset laion30m laion{1,3,10}m --n-items … --seed 20261013`.
  Before any LAION timing: the subsets against `subsets/*.MANIFEST.sha256` and every LAION filter the rerun reads against the
  final run's seed-0 records (`fr/timed-bloom-d256-*`, `fr/timed-bloom-synth-d256-*`, `fr/synth-mid-timed-pode-s0`).

RESULTS
