# FR-GATE: the gate on the final run's records

Roadmap step 7's gate ([recipe](../full-rerun/recipe.yaml) `gate`): "every planned cell ok (OOM fallbacks superseded),
D1-G's four checks, 0 manifest gaps, check_suites.py 0 problems". Its inputs are the 105 `fr/` subtrees on the Hub
([hub-index](../hub-index.md)): the 102 FULL-RERUN legs (with their `-rerun` and `-onpod` substitutes, the untimed and
TUNE-EXT outputs and the picks) and SYNTH-MID's three. State: [validation § FULL-RERUN](../../validation.md#full-rerun).
Run on pod e, 2026-10-16, from `dev/fr-gate`; the tables and the gate's outputs are on the Hub as `artifacts/fr-gate`.

| file | what |
|---|---|
| [`verify_trees.py`](verify_trees.py) | step 0: every fetched tree against the Hub: its `MANIFEST.json` downloaded and hashed against the hub-index row, every listed file checked by size and sha256, no unlisted local file; the verified trees hard-linked with their manifests into the merge's input dir |
| [`legs.yaml`](legs.yaml) | the planned legs at the final scope: per Hub subtree its seed, suites, datasets, config dir and tag tree, the `-rerun` supplements, the record-free legs' expected files, and the `dropped` list with each ruling |
| [`plan.py`](plan.py) | step 1: the planned cells, resolved with the tag's own `bench.config.load_matrix` (run under `/scratch/wt/fr` + `/venvs/fr` for `final-freeze`, `/scratch/wt/fr-clause` + `/venvs/fr-clause` for `final-freeze-clause`), one JSON line per cell with its key block; `--check` runs `check_suites.py` on every planned suite of every config dir |
| [`gate.py`](gate.py) | steps 2, 3a, 3c, 3d on `bench merge`'s tables: subtree coverage, planned cells against accepted records (status, subtree, tag), the orphan check, the unstable and superseded counts, D1-G (a) eager-vs-graph `ids_sha256`, (c) `median_ms(bs 16) < 16 x median_ms(bs 1)`, (d) the manifest; writes `gate.json` / `gate.md` |
| [`quality_legs.py`](quality_legs.py), [`compare_legs.py`](compare_legs.py) | step 4, D1-G (b) on the GPU: per leg a few quality-only cells rerun at the tag (eager, `--skip-perf`, same seed; one SilverTorch job at the smallest `n_lists`, one job per other arm, first query combo), with an oracle guard so no oracle this pod lacks is built; every rerun record against every FR record with its key block (`quality` and the per-query sidecars equal) |

## How the plan is built

The run's config dirs on pod e are the plan's: `/scratch/fr/config` (the FR suites with the seed-0 picks merged by
`prep_config.py`), `/scratch/fr/config-clause` (plus the lite clause-row suites, tag `final-freeze-clause`),
`/scratch/fr/synth-mid/config` (plus SYNTH-MID's suites and picks). The TUNE-EXT timed suites that pod 1 / pod f ran
were picked on pod 1 and are not on this box: `pick_ext.py` is deterministic, so they are regenerated from the fetched
curve, V3-pool and ext-curve legs into `/scratch/fr-gate/cfg-ext` (the pod e half equals the run's copy byte for byte,
the pod 1 half is checked against the records by the gate). The regeneration also writes a clause-row suite for the
goodreads-synth TUNE-EXT cell (`fr-ext-timed-clause-synth-pode-d128`) that no leg ran: `legs.yaml` carries it as
`never_run`, and the gate reports it.

A cell is one record: `(dataset, dim, inputs, suite, filter_kind, sweep, algo, backend, params, seed[, self_excluded])`,
`records.key_block`. The plan expands every leg's `(suite, dataset)` at the leg's seed and keeps the `in_suite` skip of
`run_leg.sh`. The accepted record per cell is `bench merge`'s (`ok` over `partial` over `failed`, stable over unstable,
the latest `env.started`): a `-rerun` subtree therefore replaces every cell it reran, the stable ones included (the
results-draft's `build_results.py` replaced only the unstable ones; the quality is the same, the timing is the rerun's).

## Running it

```bash
# step 0 (CPU, Hub token in the environment): the trees, verified and hard-linked under /scratch/fr-gate/trees
cd evaluation && PYTHONPATH=.:../retrieve/src python ../research/artifacts/fr-gate/verify_trees.py ../research/artifacts/hub-index.md /scratch/results-draft/fr /scratch/fr-gate/trees
# step 1 (CPU, once per tag tree): the plan
cd /scratch/wt/fr/evaluation && RETRIEVE_DATA_ROOT=/data PYTHONPATH=.:../retrieve/src /venvs/fr/bin/python <here>/plan.py <here>/legs.yaml --tree fr --out /scratch/fr-gate/plan-fr.jsonl --check
cd /scratch/wt/fr-clause/evaluation && RETRIEVE_DATA_ROOT=/data PYTHONPATH=.:../retrieve/src /venvs/fr-clause/bin/python <here>/plan.py <here>/legs.yaml --tree clause --out /scratch/fr-gate/plan-clause.jsonl --check
# step 2 (CPU): the tables
bench merge /scratch/fr-gate/trees/* --out /scratch/fr-gate/tables
# steps 2-3 (CPU): the gate
python <here>/gate.py <here>/legs.yaml /scratch/fr-gate/tables /scratch/fr-gate/trees /scratch/fr-gate/gate /scratch/fr-gate/plan-fr.jsonl /scratch/fr-gate/plan-clause.jsonl
# step 4 (GPU 0 under flock, once per tag tree, then the comparison)
cd /scratch/wt/fr/evaluation && CUDA_VISIBLE_DEVICES=0 RETRIEVE_DATA_ROOT=/data PYTHONPATH=.:../retrieve/src flock /scratch/gpu0.lock /venvs/fr/bin/python <here>/quality_legs.py <here>/legs.yaml --tree fr --out /scratch/fr-gate/rerun
python <here>/compare_legs.py /scratch/fr-gate/trees /scratch/fr-gate/rerun /scratch/fr-gate/rerun/chosen-fr.json /scratch/fr-gate/rerun/chosen-clause.json --out /scratch/fr-gate/gate/gate_b.json
```

## Result (pod e, 2026-10-16)

**GREEN.** [validation § FULL-RERUN](../../validation.md#full-rerun) has the row; in short: 106 / 106 trees verified; 3,577
planned cells in 79 legs, 3,577 accepted records, all `ok`, 0 gaps; 250 accepted cells unstable (kept, flagged), 87 OOM
`failed` lines and 163 rerun-replaced `ok` lines superseded; D1-G (a) 4,189 / 4,189 ids pairs equal, (b) 162 rerun cells
equal to 170 / 170 FR records (107 chosen cells skipped: their oracle caches are pod 1's), (c) 3,282 pairs, 0 failures and
1 listed known exception (user ruling), (d) 0 manifest problems; `check_suites.py` 0 problems on every config dir. The one
cell the plan found never run (`fr-ext-timed-clause-synth-pode-d128`) was run at final-freeze on the user's ruling
(`fr/ext-clause-synth-pode-s0`) and is in the 106. Step 4 took 41 GPU-min of its 60-min cap, the new cell 2.5.

Not covered by (b) on this pod (no oracle cache here; about 20-25 GPU-min to build the 13 oracles and cover them):
the pod 1 legs on laion1m / laion3m / laion10m and their synth draws, the pubmed-synth and laion10m-synth s1 / s2 draws,
arxiv-synth at d256, and the goodreads-synth / arxiv-synth s2 draws (`gate_b.json` has the per-leg counts).
