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
| [`suites-exact.yaml`](suites-exact.yaml), [`exact_ids.sh`](exact_ids.sh), [`compare_exact.py`](compare_exact.py) | GPU: exact LiNR V1 on both real filters at 1M-30M, compared with the records' recall, held-out counts and result-id sha256 |
| [`add_to_archive.py`](add_to_archive.py) | `eval-laion30m/` in the archive: hardlinks of the staged files, sidecars without local paths, the subsets' manifests |
| [`archive-text/`](archive-text/) | the archive's texts as written here: `eval-laion30m/README.md`, the archive README and LICENSES with the LAION parts (the archive README's other parts are the hf-export worker's and the controller's) |

Archive state after the add: 4,473 files, 50.50 GB (`eval-laion30m/` 53 files, 24.76 GB), `verify.py` 0 problems,
`MANIFEST.json` sha256 `0e46a14afd0e8f924cf88b237ae89965b75f23703a3f4dc48712ac0fa18f8ba7`. Identifier scan
(hf-export's `scrub_grep.py`): 4,474 files, 101 hits, every one a registered-domain string in
`eval-laion30m/vocab.json` (`claudemonetgallery.org`, `retroroadmap.com`, `fr-ski.com`, …): third-party data, as the
Goodreads "Roman" allowlisted there. Not uploaded from this session.
