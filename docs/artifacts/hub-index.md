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
| `d1-a` | D1 campaign stage a, Goodreads d128 filter suite: 126 records, samples, `results.parquet` | 3 | 78,261,847 | `72fa58381475321c33d40d66fb0d0f7203d87d9eae79526760670a7d4a171966` |
| `artifacts/cute-dsl-scorer` | timing and graph dumps of the deleted CuTe backend (tag `cuda-cute-backends-final`) | 19 | 540,746 | `785ae6b63e77f8d91f1df8d96ac5edeccf5da2f8c99873584fccbb875355b70e` |
| `artifacts/dataset-candidates` | YFCC and PubMed ETL dry-run logs, plans, GT checks, rehearsal records | 15 | 40,346 | `e06a33b267d6ffaa0e06c95412f3fc6766f40ace1f51bbb55c881bdfa6b41656` |
| `artifacts/deterministic-compaction` | golden re-derive cells (run1/run2), epilogue and timing dumps | 10 | 106,051 | `9758c1961f40443d170d7f4755a45264bacfd59db5c4d451bc5136573ad02911` |
| `artifacts/e2-pubmed` | PubMed d768 cell record, 10M-slice plan | 2 | 19,364 | `570d51246688e0ab3c6ccde80e9617aa70e930825cd1c8462268c368fa3a71c0` |
| `artifacts/e3-openalex` | OpenAlex prep/convert logs, staging params, stream probes, plan, cell record | 10 | 11,923 | `f1bbd3638f1bf7ca1f227437ab5df868a2bf91181863a561201ac3a2a0c0b3ef` |
| `artifacts/e4-kuairand` | KuaiRand attribute vocabulary, convert/prep logs, gSASRec training/probe logs, publish log, the filter-cell record, and the val/test cold-start check | 13 | 37,702 | `bfd1c9fecaabf9dc5a7346b0b203f37d53f56757b6830c1ac550ba2c5be81f14` |
| `artifacts/evaluation-harness-v2` | C4 resume and kmax-diag records, D1-a rerun records, A1 re-derive and step-4 cells | 21 | 1,597,720 | `01977159cb57cc935f7e5a127a9a65b98d750751f2e81ce0c35795c3af78041f` |
| `artifacts/evaluation-package-layout` | C5 L4-b chunk-64 record | 1 | 6,224 | `d69e60f9f723eb660dfa873871b7b7cdd6b5c2e665d4144c6a7fdc0393ba7b41` |
| `artifacts/golden-logs` | run logs behind `evaluation/golden/*.json`: driver, per-cell, clocks | 13 | 43,586 | `3000d733b07ec6ad3e1f6338bab5e020cc578f81b4b9f2e3b7843ea345363c53` |
| `artifacts/kernel-opt` | head-to-head, phase A/B timing dumps, PubMed d768 record | 48 | 685,562 | `0887e6bfcec8bf17847a0c74e7a8b0adbed302cb88cd9a4e13449af7fd249089` |
| `artifacts/l1-l2` | L1 + L2: YFCC precision and harness-gate records (before / after), index memory dumps, golden-cell records (base, after L1, final), timing JSONs and profile, the tail-poison audit | 22 | 108,639 | `844c3f1b74613c1cb0f1b2aeb1953f7bb9dd84b1a448e2eb984326513d5ed3d7` |
| `artifacts/library-api-refactor` | L1 tensor capture, L2 k-means++ timing | 3 | 8,137 | `2fe46baa992f152c6563f7e8da8d7aeb4fa5cb7328d86565d7c90fccb817d21c` |
| `artifacts/linr-v2-backend-parity` | V2 parity probes | 4 | 37,287 | `7066509248e9b21d3a6cc085c9c7d6673621ae616155c4e67883cfbfda4694ac` |
| `artifacts/official-silvertorch` | B3 kernel-only dumps, official facts, WP3 parity probes | 5 | 775,632 | `1d8d880391ce9d181d7e67c606838eb4fd2846f9e2c48da0f819937ceccb8b95` |

The `d1-a` and `artifacts/*` subtrees were published by
[H1](h1-results-storage/README.md) on 2026-09-26; their file counts and bytes
include the regenerated `results.parquet` where a subtree holds records.
