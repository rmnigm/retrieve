# Artifacts — dataset ingestion

Raw material behind the PubMed streaming ETL ([datasets](../../../docs/system/datasets.md#pubmed)).

## `pubmed/` — roadmap E2, 2026-09-16 (streaming ETL dry run, nothing staged)

Produced on the A100 box, no GPU. The record is
../dataset-candidates.md §7.2.

| file | what it is |
|---|---|
| [plan-full-2026-09-16.json](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/dataset-candidates/pubmed/plan-full-2026-09-16.json) | `eval-data pubmed plan --medline --prefetch 1 --mbps 23.9`: per-shard sizes from `HEAD` (114 MedCPT files + 1334 MEDLINE files) and rows from each npy header — 35,920,666 articles, 217.9 GB to download, peak disk 68.6 GB, 2.54 h at the measured single-stream rate; 110.3 GB of fp32 items on the device. |
| [plan-10m-2026-09-16.json](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/dataset-candidates/pubmed/plan-10m-2026-09-16.json) | The same with `--keep-items 10000000`: peak disk 27.2 GB, the same download, 30.7 GB of items on the device. |
| [dryrun-chunk37.log](pubmed/dryrun-chunk37.log) | The one-shard dry run: `convert --shards 37 --fetch --prefetch 1 --delete-raw` (1.79 GB fetched, 380,761 rows folded in 222 s, raw deleted), `attrs`, `queries`, `encode_queries --device cpu` (2,000 held-out titles). |
| [dryrun-chunk37-prep_log.json](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/dataset-candidates/pubmed/dryrun-chunk37-prep_log.json), [dryrun-chunk37-shard_index.json](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/dataset-candidates/pubmed/dryrun-chunk37-shard_index.json), [dryrun-chunk37-text_emb.meta.json](https://huggingface.co/datasets/pinkmeme/eval-results/blob/main/artifacts/dataset-candidates/pubmed/dryrun-chunk37-text_emb.meta.json) | What the dry run wrote: `pmid_ranges_disjoint: true`, one output shard `[0, 380761)`, `prefix: null`. |
| [dryrun-chunk37-config.yaml](pubmed/dryrun-chunk37-config.yaml) | `config/pubmed.yaml` with `data_dir` pointed at the dry-run directory; `bench check --dataset pubmed --config-dir …` → `pubmed d768: ok`. |
