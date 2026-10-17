---
viewer: false
tags: [benchmark-results, torchretrieve]
---

# eval-results: the final run's records

`runs/<run>/` is one leg of the final run as `bench` writes it: `<suite>/<dataset>-d<dim>.jsonl` (one record per
cell), `*.samples.jsonl` (per-call latencies), `*.perquery/*.npz` (per-query sidecars), `results.parquet` and
`MANIFEST.json` (provenance and a sha256 per file). Fetch and check one with
`bench fetch --repo-id <this repo> --path-in-repo runs/<run>`. The archive's top-level `README.md` describes
the runs and what was changed from the records as written (names and machine-local strings only).
