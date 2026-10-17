# laion-inputs: the LAION 1M-30M inputs, rebuilt

The final run's staged LAION inputs were lost with the old pods and were never on the Hub. They were rebuilt on
2026-10-17 on one A100-SXM4-80GB at `staging` (dataset code unchanged since `final-freeze` but for comments; same
locked dependencies), checked against the final run's records, and added to the anonymous archive. They are not
on the private Hub: its quota is nearly full (user, 2026-10-17). State and results:
[validation](../../validation.md#datasets); the recipe: [datasets](../../../docs/system/datasets.md#laion30m).

| file | what |
|---|---|
| [`rebuild.sh`](rebuild.sh) | the whole build: download, prep, encode, ingest, synthetic draws 0-2, the three subsets, the config diff, `bench check` |
| [`records.py`](records.py) | the final run's LAION records (fetched `fr/` legs) flattened: counts, pass rates, quality, result-id sha256 |
| [`check_filters.py`](check_filters.py) | CPU: per-query pass counts of the real filters and the synthetic pass rates against the records |
| [`suites-exact.yaml`](suites-exact.yaml), [`exact_ids.sh`](exact_ids.sh) | GPU: exact LiNR V1 on both real filters at 1M-30M, compared with the records' recall, held-out counts and result-id sha256 |
| [`add_to_archive.py`](add_to_archive.py) | `eval-laion30m/` in the archive: hardlinks of the staged files, sidecars without local paths, the subsets' manifests |
