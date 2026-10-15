# export-prep: the harness made ready for export

The user's ask of 2026-10-15 (`.chains/export-prep/`): result storage as a first-class harness feature, the roadmap §6
harness deletions, comment cleanup and a review pass, on `dev/export-harness`, merged only after SYNTH-MID and the FR gate.

| file | what |
|---|---|
| [`check_merge.py`](check_merge.py) | `bench merge` over the fetched `fr/*` legs against `build_results.py` (results-draft): the accepted `ok` non-YFCC rows equal its deduplicated frame row for row, its per-run counts equal the results-draft `summary.json`, the manifest's cross-host duplicates printed |

```bash
cd evaluation
python -m bench.cli merge <fetched>/fr/* --out <tables>
uv run --no-sync --with pandas python ../research/artifacts/export-prep/check_merge.py <fetched>/fr <tables> <summary.json>
```

State: [validation § Harness gates](../../validation.md#harness-gates).
