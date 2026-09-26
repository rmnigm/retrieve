---
chain: "dataset-candidates"
branch: "main"
nextStep: "E0: request the Semantic Scholar API key (long pole for E3); retry the YFCC-10M download with the exact URLs (E1)."
created: "2026-09-05T12:00:00Z"
---

# Dataset candidates for the filtered-retrieval benchmark (survey, plan "D")

Source: `docs/plans/dataset-candidates.md` §1-§5, survey dated 2026-09-05; every fact carried its source URL in the plan (kept below where load-bearing); UNVERIFIED items could not be confirmed that day; *est.* = back-of-envelope. Artifacts: `docs/artifacts/dataset-candidates/`.

## Requirements
The papers evaluate at 10M and 80M items (SilverTorch, D=128, int8 IVF + bloom, 6 filter features, ~7 conditions per query) and ~15.5M jobs with 25k queries plus a 240M x 128 fp16 stretch on one A100 (LiNR). Our pools topped out at 5.37M (Yambda-5B); only Goodreads (797k) and arXiv (2.99M) carried attributes. Needed: >= 10M items (5M floor, 50-100M stretch) fitting one A100-80GB; dense vectors or encodable content with a named encoder; per-item categorical attributes giving AND-of-OR filters from ~50 % to ~0.1 % pass rate; queries (user histories or text / held-out items; filtered GT always recomputed); redistributable without a registration wall.

## Candidates (fit: sem = text shape, rec = sequential shape, 0-3)
| dataset | items | embeddings | attributes | queries / GT | access | sem / rec |
|---|---|---|---|---|---|---|
| Amazon Reviews 2023 | 48.19M items, 54.51M users, 571.54M reviews | encode (nomic est. 10-14 A100-h) | main_category 33+Unknown, categories, store, price / rating buckets | Amazon-C4 21,223 text queries; 5-core histories | no license on card, ungated | 3 / 2 |
| PubMed + MedCPT | ~36M articles (38 chunks of ~1M) | shipped MedCPT 768-d (`embeds_chunk_*.npy`, 102 GB fp32 inferred) | MeSH (multi-valued, ~30k), year, journal / language via MEDLINE join | open MedCPT query encoder; NFCorpus 323, BioASQ 500 (not public), item-as-query | public domain, NCBI FTP | 3 / 0 |
| KuaiRand-27K | 32,038,725 videos, 27,285 users, 322,278,385 interactions | none; gSASRec or captions | author, video_type, upload_type, music, tags, 4-level categories, duration / upload buckets | 27,285 histories | CC BY 4.0 (Zenodo; site says BY-SA, UNVERIFIED) | 1 / 3 |
| Yambda-5B full + attrs | 9,390,623 tracks, 1M users, 4.65B listens | 128-d audio CNN for 7,721,749 tracks | artist 1,293,394 (median 2 tracks), album 3,367,691, length bucket | 459k test users | Apache-2.0 | 1 / 3 |
| YFCC-10M (Big-ANN'23 filtered) | 10,000,000 images | CLIP 192-d uint8 | tag bag, vocab 200,386 | 100,000 queries with 1-2 tags, filtered GT shipped | public (fbaipublicfiles) | 3 / 0 |
| Cohere Wikipedia 2023-11 | 247M paragraphs; en 41,488,110, de 20,772,081, fr 17,813,768 | 1024-d fp32 (PCA needed) | language only | none native, API-only query encoder | ungated | 2 / 0 |
| MS MARCO Web Search 100M | 100,924,960 docs | SimANS vectors 289.16 GB | only via ClueWeb22 agreement | 9,374 test queries + GT | non-commercial | 2 / 0 |
| Semantic Scholar SPECTER2 + papers | 120M embeddings (30 x 28 GB JSONL, ~840 GB), 200M papers | SPECTER2 768-d, open model | year, venue, fields of study, publication type, OA, citation bucket | item-as-query, citations for relevance | ODC-BY / Apache-2.0; API key required | 2 / 0 |
Rejected or deferred (§2.1): WIKI-ANN (35M, content-derived word labels), LAION-25M / ACORN sets (partly synthetic, gated), synthetic-label sets (SIFT, GIST, GloVe, Paper), Big-ANN'21 billion-scale (no attributes; usable for unfiltered scale), Qdrant / VectorDBBench / vendor filtered benchmarks (random labels), OpenAlex (324M works, CC0, richest real attributes, no vectors: ~670 GB works JSONL, 50M works est. 9-14 A100-h with nomic: the E3 fallback), MS MARCO v2 + Cohere (no attributes), wiki_dpr, BEIR / LoTTE (small; BioASQ queries reusable on PubMed), MIRACL (gated), ESCI (query set only), LAION-400M (gated), Wikidata (heavy ETL), KuaiSAR (6.89M items; second recsys choice), Sber MegaMarket / Zvuk, Taobao, Ali-* / Tmall, Amazon-M2, Spotify MPD, LFM-2b (withdrawn), MicroLens / PixelRec / Tenrec and the <= 1.8M sets.

Per-candidate gotchas worth keeping: Amazon '23 has no license field and `store` is free text; KuaiRand has only 27k users (multi-cut-point evaluation needed) and a 32M-row gSASRec table needs `reuse_item_embeddings` (~49 GB vs ~100 GB with separate tables + AdamW); Yambda's artist / album filters are extremely selective (median 2e-7); YFCC averages ~11 tags per item (K cap needed), one mixed vocabulary, uint8 L2 192-d; Cohere Embed-v3 has no Matryoshka truncation (PCA) and its query encoder is API-only; PubMed's total count unstated (~36M est.), MeSH multi-valued (~10-15 headings), journal / language need a MEDLINE join; S2 bulk downloads need the partner-form key.

## Decision (2026-09-05, final)
The study uses arXiv and Goodreads (rerun), YFCC-10M, PubMed + MedCPT, Semantic Scholar SPECTER2 (request the free research API key first) with OpenAlex as fallback, and KuaiRand-27K. Dropped: Amazon Reviews 2023 (out in general), Yambda-full and Cohere Wikipedia (scale without meaningful filters; Cohere rows carry only `_id/url/title/text/emb` and its query encoder is API-only). Later decision 2026-09-06: no PCA anywhere, every dataset at its encoder's native dim.

## Planned ingestion (§4, as written)
- 4.1 PubMed: download 114 chunk files; convert to parquet + fp16 shards (PCA plan superseded); `item_id_map`, 10k held-out; `encode_queries` with `ncbi/MedCPT-Query-Encoder`; attrs `[N, 5, 4]`: C0 MeSH heading (top-V, K=4 OR), C1 MeSH tree-top category, C2 year bucket, C3 journal (top-5k, reverse), C4 has-abstract; predicted pass rates MeSH ~0.01-5 %, category ~5-40 %, year ~10 %, journal ~1e-4.
- 4.3 KuaiRand: positives `is_click == 1`, exclude `is_rand == 1` from training, temporal split, several evaluation rows per user; gSASRec D=128 with a shared item table (est. 6-12 A100-h); attrs `[N, 7, 4]` (category levels, upload / music type, duration, upload month, author as reverse); Yambda full catalog as the low-effort fallback.
- 4.5 YFCC as the cross-paper anchor, not the headline set.

## Open questions (§5)
Amazon license / dedup / title coverage; KuaiRand coverage, tag vocabulary, positive signal, 32M-row output table; Cohere PCA acceptability; encoder choice for new text pools (nomic vs bge-small / e5-small, measure on 100k docs); YFCC max tags per item (decides K); Yambda multi-artist tracks; S2 key / ClueWeb22 obtainability; host RAM for `int64 [N, C, K]` at 48-80M (the attrs step must stream).
