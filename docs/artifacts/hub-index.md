# Hub index — what `pinkmeme/eval-results` holds

Results and raw outputs are kept on the private Hub dataset repo
[`pinkmeme/eval-results`](https://huggingface.co/datasets/pinkmeme/eval-results),
not in git ([evaluation](../system/evaluation.md#results-storage)). This page is
git's half: one row per published subtree, with the sha256 of its
`MANIFEST.json`, which in turn carries the sha256 of every file in the subtree.
A raw file that used to sit at `docs/artifacts/<plan>/<path>` is at
`artifacts/<plan>/<path>` on the Hub; `evaluation/golden/_logs/<path>` is at
`artifacts/golden-logs/<path>`. Fetch any subtree with
`bench fetch --path-in-repo <subtree> --results <dir>`.

Add a row with every `bench upload` (it prints the manifest sha256); a
re-upload of a subtree replaces its row.

| subtree | what | files | bytes | `MANIFEST.json` sha256 |
|---|---|---|---|---|
| `b3` | official-vs-ours end to end (roadmap B3): filter + quality records and samples, arXiv and Goodreads d128 | 8 | 16,918,821 | `372be6bdd3c68c03611679dcd368f309a72af35cba33810af1c02dae43e05654` |
| `c4` | harness v2 gate run (C4): filter records and samples, arXiv and Goodreads d128 | 4 | 12,983,727 | `20cc70458b4fd790889fb808bd1009ae0f5354f883f644ce21b469d4c88a8964` |
| `c5` | package-split gate run (C5): Goodreads d128 records, samples, `flat.csv` | 3 | 7,563,738 | `a4e60720e2960bfe503119997dd4786dcfff443aa479e8e71b92e394e3f140f4` |
| `d1-a` | D1 campaign stage a, Goodreads d128 filter suite: 126 records, samples, `results.parquet` | 3 | 78,261,847 | pending upload |
| `artifacts/cute-dsl-scorer` | timing and graph dumps of the deleted CuTe backend (tag `cuda-cute-backends-final`) | 19 | 540,746 | pending upload |
| `artifacts/dataset-candidates` | YFCC and PubMed ETL dry-run logs, plans, GT checks, rehearsal records | 15 | 40,346 | pending upload |
| `artifacts/deterministic-compaction` | golden re-derive cells (run1/run2), epilogue and timing dumps | 10 | 106,051 | pending upload |
| `artifacts/e2-pubmed` | PubMed d768 cell record, 10M-slice plan | 2 | 19,364 | pending upload |
| `artifacts/e3-openalex` | OpenAlex prep/convert logs, staging params, stream probes, plan, cell record | 10 | 11,923 | pending upload |
| `artifacts/e4-kuairand` | KuaiRand attribute vocabulary, convert and prep logs | 3 | 17,756 | pending upload |
| `artifacts/evaluation-harness-v2` | C4 resume and kmax-diag records, D1-a rerun records, A1 re-derive and step-4 cells | 21 | 1,597,720 | pending upload |
| `artifacts/evaluation-package-layout` | C5 L4-b chunk-64 record | 1 | 6,224 | pending upload |
| `artifacts/golden-logs` | run logs behind `evaluation/golden/*.json`: driver, per-cell, clocks | 13 | 43,586 | pending upload |
| `artifacts/kernel-opt` | head-to-head, phase A/B timing dumps, PubMed d768 record | 48 | 685,562 | pending upload |
| `artifacts/library-api-refactor` | L1 tensor capture, L2 k-means++ timing | 3 | 8,137 | pending upload |
| `artifacts/linr-v2-backend-parity` | V2 parity probes | 4 | 37,287 | pending upload |
| `artifacts/official-silvertorch` | B3 kernel-only dumps, official facts, WP3 parity probes | 5 | 775,632 | pending upload |

The `artifacts/*` and `d1-a` file counts and bytes are from the
[H1 staging dry run](h1-results-storage/README.md) and include the regenerated
`results.parquet` where a subtree holds records.
