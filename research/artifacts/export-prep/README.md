# export-prep: the harness made ready for export

The user's ask of 2026-10-15 (`.chains/export-prep/`): result storage as a first-class harness feature, the roadmap §6
harness deletions, comment cleanup and a review pass, on `dev/export-harness`, merged only after SYNTH-MID and the FR gate.

| file | what |
|---|---|
| [`check_merge.py`](check_merge.py) | `bench merge` over the fetched `fr/*` legs against `build_results.py` (results-draft): the accepted `ok` non-YFCC rows equal its deduplicated frame row for row, its per-run counts equal the results-draft `summary.json` |

```bash
cd evaluation
python -m bench.cli merge <fetched>/fr/* --out <tables>
uv run --no-sync --with pandas python ../research/artifacts/export-prep/check_merge.py <fetched>/fr <tables> <summary.json>
```

State: [validation § Harness gates](../../validation.md#harness-gates).

## The records-don't-move GPU check (after SYNTH-MID, controller's grant)

| file | what |
|---|---|
| [`quality_subset.sh`](quality_subset.sh) | a quality-only rerun of FR cells on the merged tree, seed 0, eager, `--skip-perf`, from the FR config dir (`/scratch/fr/config` on pod e): adapted SilverTorch exact + bloom (`fr-curve-d128`, goodreads + arXiv, 84 cells), adapted SilverTorch unfiltered + exact LiNR on Yambda (`fr-yambda` d128), exact LiNR V1 / V2 / V3 on the real filters (`filter`, goodreads + arXiv d128, quality only) |
| [`compare_quality.py`](compare_quality.py) | every rerun record against the FR record with the same key block: `quality` equal (the parity note aside) and the per-query sidecar arrays equal; exit 1 on any difference |

```bash
TREE=/scratch/wt/export-harness CFG=/scratch/fr/config OUT=/scratch/export-prep/quality-subset \
    research/artifacts/export-prep/quality_subset.sh
```
