---
chain: "dataset-candidates"
branch: "main"
parent: "2026-09-16-120000000-e1-yfcc-staged-and-run-e2-pubmed-streaming-etl.md"
nextStep: "None from this note. The E2 budget measured from the server (previous note) supersedes the PubMed estimates below."
created: "2026-09-26T01:17:00Z"
---

# Fidelity restore: per-candidate disk and memory (survey §1, §4)

Written 2026-09-26 during the docs reorganisation. The survey note dropped the table's disk and fp16-memory columns and the §4 ingestion budgets. Restored from `docs/plans/dataset-candidates.md`. All values are the 2026-09-05 survey figures; *est.* marks back-of-envelope ones.

| dataset | disk (source) | fp16 items on device |
|---|---|---|
| Goodreads (797k) | small | 0.2 GB |
| arXiv (2.99M) | ~3 GB | 1.5 GB |
| Yambda-5B as used (5.37M) | ~100 GB (5b) | 1.4 GB |
| Amazon Reviews 2023 (48.19M) | ~100+ GB raw JSONL gz *est.* | 24.7 GB @256, 12.3 GB @128 |
| PubMed + MedCPT (~36M) | 102 GB npy + 44 GB JSON | 18.4 GB @256 (PCA), 9.2 GB @128, 55 GB @768 |
| KuaiRand-27K (32M) | 9.9 GB tar.gz (46 GB unpacked) + 6.9 GB supplement; captions 3.2 GB CSV | 8.2 GB @128 |
| Yambda-5B full (9.39M) | ~100 GB; audio embeddings float64 parquet 13.8 GB | 2.4 GB @128 |
| YFCC-10M | ~2 GB base | 1.9 GB uint8 / 3.8 GB fp16 |
| Cohere Wikipedia 2023-11 | en 97.86 GB, de 48.79 GB, fr 41.20 GB (fp32 parquet); it 24.49 GB, ceb 22.37 GB | 80M @128 = 20.5 GB, @256 = 41 GB (1024-d at 80M would be 164 GB) |
| MS MARCO Web Search 100M | 289.16 GB `vectors.bin` (~3,076 B/vector, consistent with 768-d fp32) | 100M @128 (PCA) = 25.8 GB |
| Semantic Scholar SPECTER2 | embeddings 30 x 8.5 GB listed, abstracts 30 x 1.8 GB | 50M @256 (PCA) = 25.6 GB |
| rejected: `wiki_dpr` | ~260 GB (DPR 768-d fp32) | - |
| rejected: OpenAlex works | 750-780 GB JSONL | - |

§4 budgets as planned (the PubMed one was later replaced by the measured E2 budget and native 768-d):
- PubMed: 146 GB raw, ~30 GB processed; A100 memory at D=128 9.2 GB fp16 + 4.6 GB int8, at D=256 18.4 + 9.2 GB; about 1 A100-hour for PCA and projection.
- Amazon '23: raw gz 60-120 GB, parquet ~40 GB, embeddings 24.7 + 12.3 + 6.2 GB, attrs 9.3 GB; A100 at D=128 12.3 GB fp16 + 6.2 GB int8.
- KuaiRand gSASRec: 16.4 GB fp32 per item table, plus two AdamW moments per table, about 100 GB without shared embeddings.
- Host RAM for `int64 [N, C, K]` attrs at 48-80M items: 9-15 GB, so the attrs step must stream.
- Cohere en: 187.85 GB of fp32 parquet, stream row group by row group.
