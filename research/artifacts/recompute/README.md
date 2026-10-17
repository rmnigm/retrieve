# recompute: scripts behind the export's results/ extension (branch dev/export-r4)

Inputs: the Hub build `artifacts/results-draft/20261016a-fr/results.parquet` (see
[hub-index](../hub-index.md)) and the export's `results/records/records.parquet` at `2b05c792`.

| script | what |
|---|---|
| `add_records.py BUILD OLD NEW` | keeps the export's 2,986 records byte-equal on their columns, adds `n_queries_oracle` / `heldout_recall@100,1000` from the matching build rows, appends the native-PyTorch, V3 code-length, recall-bridge and reverse-clause records (3,529) |
| `execcfg.py RECORDS OUT` | `final-run-config/executed_configurations.csv` from the records (reproduces the previous file's rows exactly, as a set) |
| `extract_suites.py` | the suites behind the new records, from `b3bddc88` (`evaluation/config/suites.yaml`, `research/artifacts/full-rerun/suites.yaml`) and `500284eb` (`clause-opt/clause-lite.yaml`), comments stripped, backends renamed |
| `loadcheck.py` | loads every suite of `final-run-config/suites.yaml` with the shipped harness's `load_matrix` |
| `captions.py` | the figure caption facts from the CSVs |

The table builder, the figure script and the NUMBERS generator ship in the export's `results/`.
