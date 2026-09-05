# Dataset candidates for the filtered-retrieval benchmark (scale-up)

> **Status:** survey, 2026-09-05, nothing ingested. **Ordering authority:**
> [00-roadmap.md](00-roadmap.md) §1 Phase E — E1 = §4.3's Yambda-full fallback, E2 = §4.1
> PubMed + MedCPT, E4 (SIGIR) = §4.2 Amazon, §4.3 KuaiRand, §4.4 Cohere Wikipedia, §4.5 YFCC-10M.
> Ingestion follows `docs/system/datasets.md` § "Adding a dataset"; record what was actually
> downloaded, encoded and built in a "Ingestion record" section appended here.

Survey date: 2026-09-05. Every dataset fact below carries the URL it was
taken from; items marked **UNVERIFIED** could not be confirmed from a
primary source on that date. Numbers marked *est.* are my own
back-of-envelope estimates (memory = N × D × 2 bytes for fp16; encode
throughput assumptions are stated in § 4), not measurements.

## 1. Requirements recap

The two reproduced papers evaluate at 10M and 80M items (SilverTorch,
D=128, int8 IVF + Bloom filtering; 6 filter features with ~7 conditions
per query, [`articles/silvertorch.md`](../../articles/silvertorch.md)
§ Experiments) and at ~15.5M jobs with 25k queries plus a stretch test to
240M × 128-d fp16 on one A100 (LiNR,
[`articles/linr.md`](../../articles/linr.md) § Benchmark). Our current
pools top out at 5.37M (Yambda-5B) and only Goodreads (797k) and arXiv
(2.99M) carry attributes ([`docs/system/datasets.md`](../system/datasets.md),
[`docs/thesis/main.tex`](../thesis/main.tex) § 5.1). A replacement set
must (1) hold ≥ 10M items (5M floor, 50–100M stretch) whose fp16
embeddings at D=64–256 plus IVF/Bloom indexes fit one A100-80GB; (2) ship
dense vectors or encodable content with a named encoder; (3) carry
per-item categorical (or bucketable numeric) attributes that yield
conjunctive AND-of-OR equality filters spanning pass rates from ~50 % to
~0.1 %; (4) provide queries — user histories for the SASRec path, or text
queries / held-out items for the text path (filtered ground truth is
always recomputed by exact filtered scan); (5) be redistributable without
a registration wall, ideally from the HF Hub, and land in the on-disk
layout of `docs/system/datasets.md` § "On-disk layout the harness
expects".

## 2. Candidate table

Fit scores: 0 = unusable, 3 = drop-in. "sem" = semantic-search benchmark
(text shape), "rec" = recsys benchmark (sequential shape). Memory column
is fp16 item embeddings only, at the dim in the embeddings column.

| # | dataset | items | embeddings | attributes (cardinality; selectivity) | queries / GT | license / access | disk | fp16 mem | sem | rec | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c1 | Goodreads (current) | 797k works | gSASRec D=128, trained | genre 10, lang 30, format, year bucket, author; ~50 %→0.1 % | 313k test users, temporal split | UCSD mirror, research | small | 0.2 GB | 1 | 2 | [datasets.md](../system/datasets.md) |
| c2 | arXiv (current) | 2.99M | nomic-embed-v1.5 D=256/128/64 | main cat, license, year, n_versions | 10k held-out papers | HF `open-index/open-arxiv` | ~3 GB | 1.5 GB | 2 | 0 | [datasets.md](../system/datasets.md) |
| c3 | Yambda-5B (current use) | 5.37M tracks (Listen+ n-core) | gSASRec D=128, trained | none used today | 459k test users | Apache-2.0, HF | ~100 GB (5b) | 1.4 GB | 0 | 2 | [HF yandex/yambda](https://huggingface.co/datasets/yandex/yambda) |
| 1 | **Amazon Reviews 2023** | 48.19M items, 54.51M users, 571.54M reviews | encode title+features+description with nomic-embed-v1.5 (D=256/128/64), *est.* 10–14 A100-h; or bge-small *est.* 3 h | main_category 33+Unknown, categories (hierarchical), store, price bucket, avg-rating bucket, rating-count bucket; ~15 %→<0.01 % | Amazon-C4 21,223 text queries → 1 item each (1.06M-item pool ⊂ catalog); 5-core histories for SASRec; held-out item-as-query | no license stated on card (HF, not gated) | ~100+ GB raw JSONL gz (est.) | 24.7 GB @256, 12.3 GB @128 | **3** | **2** | [HF card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), [site](https://amazon-reviews-2023.github.io/) |
| 1b | **PubMed + MedCPT embeddings** | ~36M articles *est.* (38 chunks of "about 1M"; chunk 18 = 940,707; chunk 37 partial) | shipped MedCPT-Article-Encoder 768-d (`embeds_chunk_*.npy`, 102 GB, fp32 inferred from 2.9 GB ≈ 944k × 768 × 4) | MeSH terms (`m`, multi-valued, ~30k descriptors), year from `d`, journal / language via MEDLINE baseline join (UNVERIFIED); ~50 %→<0.01 % | open MedCPT-Query-Encoder → any text query set; BioASQ 500 test q (not public in BEIR), NFCorpus 323, item-as-query | public domain (NIH), FTP, no registration | 102 GB npy + 44 GB json | 36M @256 (PCA) = 18.4 GB, @128 = 9.2 GB; @768 = 55 GB | **3** | 0 | [FTP dir](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/), [README](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/README.txt), [model card](https://huggingface.co/ncbi/MedCPT-Article-Encoder) |
| 2 | **KuaiRand-27K** | 32,038,725 videos, 27,285 users, 322,278,385 interactions (Apr 8–May 8 2022) | none shipped; gSASRec D=128 from logs, or encode Chinese captions (3.2 GB CSV) with multilingual-e5-small, *est.* 1–2 A100-h | author_id, video_type {NORMAL, AD}, upload_type, visible_status, music_id, music_type, tag (multi-valued), 4-level category hierarchy, upload_dt / duration buckets; cardinalities UNVERIFIED | 27,285 user histories (avg ~12k events) with ms timestamps; 12 feedback signals | CC BY 4.0 on the Zenodo record (site summary says CC BY-SA 4.0 — UNVERIFIED which), captions CC BY 4.0 | 9.9 GB tar.gz (46 GB unpacked) + 6.9 GB supp. | 8.2 GB @128 | 1 | **3** | [kuairand.com](https://kuairand.com/), [paper](https://arxiv.org/abs/2208.08696), [Zenodo supp.](https://zenodo.org/records/18159199) |
| 3 | **Yambda-5B, full catalog + attrs** | 9,390,623 tracks (all events), 1M users, 4.65B listens | shipped 128-d audio CNN embeddings for 7,721,749 tracks (float64 parquet, 13.8 GB); or gSASRec D=128 as today | artist_id 1,293,394 (median 2 tracks, p99 91, max 75,489), album_id 3,367,691 (median 1, max 1,535), track_length bucket, has-audio-emb; ~40 %→1e-6 | 459k test users (existing split); item-as-query on audio embs | Apache-2.0, HF, not gated | ~100 GB | 2.4 GB @128 | 1 | **3** | [HF yandex/yambda](https://huggingface.co/datasets/yandex/yambda), parquet probe (§ 3.3) |
| 4 | **YFCC-10M (Big-ANN'23 filtered)** — access: user's attempt failed, HEAD 200 verified 2026-09-05 (§3.4) | 10,000,000 images | CLIP 192-d uint8, shipped | bag of tags, vocab 200,386 (description words, camera model, year, country) | 100,000 queries, each 1–2 required tags, filtered GT shipped | public download (fbaipublicfiles), YFCC CC licenses | ~2 GB base | 1.9 GB (uint8) / 3.8 GB fp16 | **3** | 0 | [neurips23 README](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md) |
| 5 | **Cohere Wikipedia 2023-11 (Embed-v3)** | 247M paragraphs, 300+ langs; en 41,488,110, de 20,772,081, fr 17,813,768, es 12,905,284, it 10,462,162 | shipped 1024-d fp32; must PCA/truncate to ≤256 for 80 GB | language (300+; pick 3–10 → 5–50 %), title/article grouping, paragraph position, text-length bucket; no categories | none native; item-as-query; new text queries need Cohere API | Wikipedia CC BY-SA; card lists no license; HF, not gated | en 97.86 GB, de 48.79 GB, fr 41.20 GB (fp32 parquet) | 80M @128 = 20.5 GB, @256 = 41 GB | 2 | 0 | [HF card](https://huggingface.co/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3), [datasets-server info](https://datasets-server.huggingface.co/info?dataset=CohereLabs/wikipedia-2023-11-embed-multilingual-v3) |
| 6 | **MS MARCO Web Search 100M** | 100,924,960 docs (ClueWeb22 subset) | shipped SimANS vectors, 289.16 GB `vectors.bin` (≈768-d fp32 inferred, UNVERIFIED) | language tag (207 doc langs), topic tag, URL — all require ClueWeb22 access via Lemur | 9,374 test query vectors + brute-force `truth.txt`; 9.2M train queries | non-commercial research; vectors free, text/tags behind ClueWeb22 agreement | 290 GB vectors | 100M @128 (PCA) = 25.8 GB | 2 | 0 | [GitHub README](https://raw.githubusercontent.com/microsoft/MS-MARCO-Web-Search/main/README.md), [paper](https://arxiv.org/html/2405.07526) |
| 7 | **Semantic Scholar SPECTER2 + papers** | 120M embedding records; 200M paper records (release 2026-09-01) | shipped SPECTER2 768-d (model-compatible, so queries encodable locally with `allenai/specter2`) | from `papers`: year, venue, fields of study, publication types, open access, citation-count bucket | none native; item-as-query or SPECTER2-encoded text queries | ODC-BY (papers) / Apache-2.0 (embeddings); **API key required** for bulk files | 28 GB listed for embeddings (implausible for 120M×768 fp32; UNVERIFIED) | 50M subset @256 (PCA) = 25.6 GB | 2 | 0 | [datasets release listing](https://api.semanticscholar.org/datasets/v1/release/latest), [S2 platform paper](https://arxiv.org/html/2301.10140v2) |
| 8 | Amazon-C4 (query set for #1) | 1,058,417-item pool | — | category | 21,223 ChatGPT-rephrased review queries, 1 relevant item each | HF, not gated | small | — | (query set) | — | [HF card](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4) |
| 9 | Re-LAION-5B research(-safe) | 5.5B pairs | CLIP embeddings **not** re-released with Re-LAION | LANGUAGE, WIDTH/HEIGHT, punsafe, pwatermark, AESTHETIC_SCORE, similarity (all bucketable) | none | Apache-2.0, **gated** on HF (affiliation form) | TB-scale | n/a | 1 | 0 | [LAION blog](https://laion.ai/blog/relaion-5b/) |

Rows for the remaining families surveyed (Big-ANN billion-scale tracks,
ACORN/Filtered-DiskANN/CAPS sets, Qdrant and VectorDBBench filtered sets,
OpenAlex, PubMed, ESCI, Taobao, Spotify MPD, LFM, MicroLens, Tenrec,
EB-NeRD, etc.) are in § 2.1 below.

## 2.1 Surveyed and rejected or deferred

### ANN / filtered-search benchmark sets

| dataset | items | embeddings | attributes | queries / GT | access | verdict | source |
|---|---|---|---|---|---|---|---|
| **WIKI-ANN** (VecFlow) | 35M en passages (+5k simple, +1M slice) | Cohere 22-12 embeddings re-packed as `.bin` (768-d per VecFlow; dim not on the card) | labels = words among the 4,000 most common words per passage, avg 22.5 labels/item (VecFlow); comma-separated per line | queries with exactly 2 labels, AND semantics; GT UNVERIFIED (VecFlow computed its own) | MIT, HF, not gated, files need `combine_base_vecs.py` | **viable ≥ 10M semantic set with precomputed vectors**; labels are content-derived words, not catalogue attributes — 35M × 768 fp16 = 54 GB → PCA needed | [HF 2024annonymous/wiki-ann](https://huggingface.co/2024annonymous/wiki-ann), [VecFlow](https://arxiv.org/html/2506.00812) |
| LAION-25M / LAION-1M (ACORN) | 24,653,427 / 1,000,448 | CLIP 512-d | real captions + keyword lists synthesised from 30 adjectives/nouns; selectivity 0.056–0.13 | ACORN's own; release of the attributed version UNVERIFIED | LAION metadata now gated (§ 3.8) | defer: attributes partly synthetic, source gated | [ACORN](https://arxiv.org/html/2403.04871) |
| TripClick (ACORN) | 1,055,976 | DPR 768-d | real: 28 clinical areas + publication year; selectivity 0.17–0.36 | health-search click logs | Rekabsaz et al. 2021 | too small | [ACORN](https://arxiv.org/html/2403.04871) |
| Paper (NHQ/ACORN), SIFT1M, GIST, GloVe (CAPS, NHQ, WST, iRangeGraph, SeRF) | ≤ 2.03M | 128–960-d | **synthetic** random labels (ACORN: ints in [1,12]; CAPS: L=3/10/100 with p=1/0.3/0.03; NHQ: generated date/location/size) | — | public | reject: synthetic attributes, small | [ACORN](https://arxiv.org/html/2403.04871), [CAPS](https://arxiv.org/abs/2308.15014), [NHQ](https://github.com/YujianFu97/NHQ), [WST](https://arxiv.org/abs/2402.00943) |
| Filtered-DiskANN "DANN" | 28M | — | internal Microsoft set | — | not released | reject | [Filtered-DiskANN](https://dl.acm.org/doi/10.1145/3543507.3583552) |
| UNIFY "Paper" / "WIT-Image" | UNVERIFIED | — | real (publication/topic/affiliation; image size) | — | UNVERIFIED | check if sizes ≥ 5M | [UNIFY](https://arxiv.org/abs/2412.02448) |
| Big-ANN'21: BIGANN-1B 128-d u8, SSNPP-1B 256-d u8, MS Turing-1B 100-d f32, MSSPACEV-1B 100-d i8, DEEP-1B 96-d f32, Text2Image-1B 200-d f32 | 1B each | precomputed | **none** | 10k–100k queries, GT | public | reject for filtered work (no attributes); usable for unfiltered scale sweeps | [neurips21](https://big-ann-benchmarks.com/neurips21.html) |
| Big-ANN'23 sparse (MSMARCO-SPLADE 8.8M), OOD (Text2Image-10M), streaming (MSTuring-30M) | 8.8M–30M | precomputed | none | GT shipped | public | reject (no attributes) | [neurips23 README](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md) |
| OpenAI-ArXiv (big-ann repo) | 2,321,096 | ada-002 1536-d | none in the release | 20,000 queries | public Azure blob | reject (no attributes; arXiv already covered) | [datasets.py](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py) |
| Qdrant ann-filtering-benchmark-datasets | arxiv-titles 2,138,591 (384-d, real arXiv metadata), h-and-m 105,100 (2048-d, 24 real categorical fields), laion-small 100k, random-* 1M/100k | precomputed | real for arxiv/h-and-m; synthetic for random-* | `tests.jsonl` per set | GCS tgz, no registration | too small; arxiv-titles duplicates our arXiv track | [repo](https://github.com/qdrant/ann-filtering-benchmark-datasets) |
| VectorDBBench (Zilliz) | Cohere 500k/5M/10M (768/1536-d), OpenAI 500k/5M (1536-d), LAION 100M (768-d), SIFT 500k, GIST 100k | precomputed on S3 | Integer-Filter (`id >= x`) and Label-Filter with **randomly generated** labels at 50–99.9 % filter rates | GT shipped | S3, no registration | reject for real-attribute work; LAION-100M usable as an unfiltered 100M scale test | [repo](https://github.com/zilliztech/VectorDBBench) |
| Weaviate / Elastic / Vespa / pgvector filtered benchmarks | 1M–20M | various | random/synthetic filters (Elastic: 8M × 768 with a 5 % random filter; a 20M ecommerce set UNVERIFIED) | — | — | reject | [Elastic blog](https://www.elastic.co/search-labs/blog/filtered-hnsw-knn-search), [weaviate-benchmarking](https://github.com/weaviate/weaviate-benchmarking), [Vespa](https://blog.vespa.ai/additions-to-hnsw/) |

### Text / document corpora

| dataset | items | embeddings | attributes | queries / GT | access | verdict | source |
|---|---|---|---|---|---|---|---|
| OpenAlex works | 324,184,258 (API count on 2026-09-05) | none; `abstract_inverted_index` present for >60 % of 2022 works, ~45 % pre-2000 | `type`, `language`, `publication_year`, `topics` (topic/subfield/field/domain), `open_access.is_oa`/`oa_status`, `primary_location.source`, `cited_by_count` | none | CC0; `s3://openalex --no-sign-request`; ~750 GB JSONL / ~780 GB parquet | strong "encode-it-yourself" stretch: 50M titled+abstracted works ≈ 12 A100-h *est.* with nomic; defer behind precomputed options | [works API](https://api.openalex.org/works), [work object](https://github.com/ourresearch/openalex-docs/blob/main/api-entities/works/work-object/README.md), [snapshot](https://help.openalex.org/access/snapshot/), [license](https://github.com/ourresearch/openalex-docs/blob/main/license.md) |
| MS MARCO v2 passages + `Cohere/msmarco-v2-embed-english-v3` | 138,364,198 | Cohere Embed-v3 (1024-d UNVERIFIED), ~290 GB parquet; `CohereLabs/msmarco-v2.1-embed-english-v3` also exists | thin: parent `msmarco_document_id`, passage length; no categorical field | train 277k, Dev1 3.9k, Dev2 4.3k, TREC-DL 21/22/23 graded qrels | HF; MS MARCO non-commercial terms, Cohere dump terms UNVERIFIED | defer: largest precomputed set with real queries, but no attributes | [ir-datasets](https://ir-datasets.com/msmarco-passage-v2.html), [HF](https://huggingface.co/datasets/Cohere/msmarco-v2-embed-english-v3) |
| MS MARCO v1 passages | 8,841,823 | no full-corpus dump verified | none | dev 6,980 | public | reject (no attributes) | [ir-datasets](https://ir-datasets.com/msmarco-passage.html) |
| `facebook/wiki_dpr` | 21,015,300 (2018 dump) | DPR 768-d fp32, ~260 GB | title only | NQ/TriviaQA via DPR | CC-BY-NC-4.0 | reject: thin attributes, NC | [HF](https://huggingface.co/datasets/facebook/wiki_dpr) |
| `wikimedia/wikipedia` 20231101.en | ~6.4M articles | none | title, namespace | none | CC BY-SA | raw source only | [HF](https://huggingface.co/datasets/wikimedia/wikipedia) |
| BEIR: BioASQ 14.91M (500 q, not public), Climate-FEVER 5.42M, DBPedia 4.6M; LoTTE 2.8M | ≤ 14.9M | none | domain only (LoTTE) | yes | BioASQ needs registration | reject as pools; BioASQ queries reusable on PubMed | [BEIR](https://github.com/beir-cellar/beir) |
| MIRACL (Cohere 22-12 embeddings) / mMARCO | tens of M across 18 langs (UNVERIFIED) / 8.8M × 14 langs | Cohere repos **gated (401)**; `miracl/miracl-corpus` raw is open | language | MIRACL qrels | gated / open | defer | [MIRACL](https://github.com/project-miracl/miracl), [mMARCO](https://github.com/unicamp-dl/mMARCO) |
| Amazon ESCI | products UNVERIFIED (well < 5M); 130,652 queries / 2,621,738 judgments (large) | none | `product_brand`, `product_color`, `product_locale` | ESCI labels E/S/C/I | Apache-2.0 | query set only; ASIN overlap with Amazon '23 UNVERIFIED | [repo](https://github.com/amazon-science/esci-data) |
| LAION-400M / relaion400m | 400M | original CLIP ViT-B/32 512-d npy mirrors (the-eye) reported dead — UNVERIFIED | caption, NSFW, similarity, width/height (language only in 2B-multi) | none | **gated** on HF | reject | [HF laion/laion400m](https://huggingface.co/datasets/laion/laion400m), [blog](https://laion.ai/blog/laion-400-open-dataset/) |
| Wikidata | ~120–140M items (2026 figure UNVERIFIED) | none; short multilingual descriptions encodable | P31 instance-of, P17 country, P577 date … (rich, real) | none | CC0, 103 GB bz2 JSON | interesting stretch, heavy ETL; open question | [dumps](https://dumps.wikimedia.org/wikidatawiki/entities/) |
| Crossref public file 2026 / Google Patents / PatentsView / ClueWeb22-B | 180M / 87M / >10M / 200M | none / 64-d non-semantic / none / none | type, publisher, year / CPC / CPC / language, topic | none | torrent / BigQuery / bulk / **paid licence** | reject | [Crossref](https://www.crossref.org/blog/2026-public-data-file-now-available/), [Patents](https://github.com/google/patents-public-data/blob/master/tables/dataset_Google%20Patents%20Public%20Datasets.md), [ClueWeb22](https://lemurproject.org/clueweb22/) |

### Recommender-system datasets

| dataset | items / users / events | attributes | histories | access | verdict | source |
|---|---|---|---|---|---|---|
| **KuaiSAR** (Kuaishou search + rec) | 6,890,707 items (3,026,189 search + 4,046,367 rec; a secondary source says ~9M/42k users — versions disagree, UNVERIFIED) / 25,877 users / 19,664,885 actions, May 22–Jun 10 2023 | 18 item features, 5 user features; cardinalities not published | 19 days, second-level timestamps, 9 feedback types | CC BY-NC-SA 4.0, Zenodo, no registration | only other recsys set above the 5M floor; short span, few users; second choice after KuaiRand | [site](https://kuaisar.github.io/), [paper](https://arxiv.org/abs/2306.07705), [Zenodo](https://zenodo.org/record/8181109) |
| Sber MegaMarket (KDD'24) | 3,562,321 items / 2,730,776 users / 196,644,020 events, Jan 15–May 14 2023 | 10,001 category ids, z-scored price; view/favorite/cart/purchase | 5 months, per-event timestamps | Kaggle; license UNVERIFIED | below floor; best sub-5M e-commerce option with real categories | [paper](https://arxiv.org/abs/2402.09766), [Kaggle](https://www.kaggle.com/datasets/alexxl/megamarket) |
| Sber Zvuk (KDD'24) | 1,506,950 tracks / 382,790 users / 244,673,551 plays | track/artist ids only (genre UNVERIFIED) | 5 months, session timestamps | Kaggle; license UNVERIFIED | below floor; Yambda dominates | [paper](https://arxiv.org/abs/2402.09766), [Kaggle](https://www.kaggle.com/datasets/alexxl/zvuk-dataset) |
| Taobao UserBehavior (Tianchi) | ~4,162,024 items / ~987,994 users / ~100M events, Nov 25–Dec 3 2017 | category id (3,623–9,439, sources disagree) | 9 days | Tianchi registration, research only | below floor, one attribute, registration | [Tianchi](https://tianchi.aliyun.com/dataset/dataDetail?dataId=649) |
| Ali-CCP / Alimama display ad / Tmall IJCAI-15 / AliExpress / iFashion | ≤ 4.3M items | anonymised feature ids; category, brand, price (Alimama); 80 leaf categories (iFashion) | CTR logs, not sequences | Tianchi registration | reject | [Ali-CCP](https://tianchi.aliyun.com/dataset/408), [Alimama](https://tianchi.aliyun.com/dataset/dataDetail?dataId=56), [Tmall](https://tianchi.aliyun.com/dataset/42), [iFashion](https://arxiv.org/abs/1905.01866) |
| Amazon-M2 (KDD Cup'23) | 1,410,675 products (6 locales) / 3,967,908 sessions / 16,789,783 interactions | locale, brand, size, model, material, color, author + title/description text | 3 weeks, sessions | Apache-2.0 | too small; attribute-rich cousin of Amazon '23 | [paper](https://arxiv.org/abs/2307.09688), [site](https://kddcup23.github.io/) |
| Spotify MPD | > 2M tracks / 1M playlists (no users) | artist, album | playlist edit dates only | AIcrowd registration, non-commercial | reject | [Spotify](https://research.atspotify.com/2020/09/the-million-playlist-dataset-remastered) |
| LFM-2b / LFM-1b | 120k users, > 1–2B events; tracks ~? | artist, album, track | 2005–2020 | **LFM-2b withdrawn** (licence); LFM-1b availability UNVERIFIED | reject | [LFM-2b](https://dl.acm.org/doi/10.1145/3498366.3505791), [LFM-1b](https://dl.acm.org/doi/10.1145/2911996.2912004) |
| Music4All-Onion / Melon / Deezer / 30Music | ≤ 650k tracks | genre, tags, audio features | yes (Onion 253M events) | academic / non-commercial | too small | [Onion](https://dl.acm.org/doi/10.1145/3511808.3557656), [Melon](https://mtg.github.io/melon-playlist-dataset/) |
| MicroLens (full) / PixelRec / NineRec / Tenrec | ~1M / ~408k / ~400k / items UNVERIFIED | raw video/image/text content; categories unconfirmed | yes (up to 1B events) | public / application-gated (Tenrec) | too small; content-rich | [MicroLens](https://arxiv.org/abs/2309.15379), [PixelRec](https://github.com/westlake-repl/PixelRec), [NineRec](https://github.com/westlake-repl/NineRec), [Tenrec](https://arxiv.org/abs/2210.10629) |
| KuaiRec / MIND / EB-NeRD / MovieLens-32M / Netflix / Steam / Yelp / RetailRocket / Diginetica / YooChoose / OTTO / Xing / Douban / Twitch / WeChat'21 | ≤ 1.8M items | various real attributes (categories, genres, tags) | yes | mostly public | far below floor | [KuaiRec](https://kuairec.com/), [EB-NeRD](https://dl.acm.org/doi/10.1145/3687151.3687152), [OTTO](https://www.kaggle.com/competitions/otto-recommender-system), [Xing](https://dl.acm.org/doi/10.1145/2959100.2959207) |
| Twitter RecSys'20/'21, Bilibili/Douyin corpora, Pinterest | hundreds of M tweets; UNVERIFIED | — | — | **not downloadable today** (UNVERIFIED) | reject | [Twitter](https://arxiv.org/abs/2109.08245) |
| Goodreads full UCSD graph | 2.36M books / 229M interactions | genres, authors, shelves | yes | academic use | already used (797k works after n-core); relaxing the n-core recovers up to 2.36M | [UCSD](https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html) |

## 3. Per-candidate notes (top 8)

### 3.1 Amazon Reviews 2023 (McAuley Lab)

Verified on the HF card and project site: 571.54M reviews, 54.51M users,
48.19M items, May 1996–Sep 2023, 33 categories + Unknown; item metadata
fields `main_category, title, average_rating, rating_number, features,
description, price, images, videos, store, categories, details,
parent_asin, bought_together`
([card](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)).
Largest categories: Clothing_Shoes_and_Jewelry 7.2M items, Books 4.4M,
Home_and_Kitchen 3.7M, Electronics 1.6M
([site](https://amazon-reviews-2023.github.io/)). Splits shipped:
0core/5core, last_out and timestamp variants
([GitHub README](https://raw.githubusercontent.com/hyp1231/AmazonReviews2023/main/README.md)).
Gotchas: (a) **no license field** on the HF card or the site — the
repo is public and ungated, but redistribution terms are unstated
([HF API](https://huggingface.co/api/datasets/McAuley-Lab/Amazon-Reviews-2023));
(b) "some items lack metadata" — title coverage per category is
UNVERIFIED, so the usable text pool may be < 48.19M; (c) `price` is
sparse and string-typed; (d) `store` is a free-text brand proxy with very
high cardinality; (e) the item count under 5-core for the SASRec path is
UNVERIFIED (expect a large drop); (f) Amazon-C4's 1.06M-item pool is
sampled from this catalog, so its 21,223 queries carry over to the full
48M pool with the same single relevant item
([Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4)).
Overlap of ESCI product ASINs with the 2023 catalog: UNVERIFIED.

### 3.2 KuaiRand-27K (Kuaishou)

Verified in the CIKM'22 paper (Table 1 and Appendix A): 27,285 users,
32,038,725 videos, 322,278,385 normal interactions plus 1,186,059 random
interventions, 30 user features, 62 item features, timestamps; the
release is 23 GB logs + 23 GB features
([arXiv 2208.08696](https://arxiv.org/abs/2208.08696)). The site lists
the item columns `author_id, video_type, upload_dt, upload_type,
visible_status, video_duration, server_width, server_height, music_id,
music_type, tag` and the 2026-01 supplementary files
`kuairand_video_captions.csv` (Chinese caption + `show_cover_text`,
3.2 GB) and `kuairand_video_categories.csv` (4-level category ids with
probabilities, 3.7 GB), joined on `final_video_id`
([kuairand.com](https://kuairand.com/),
[Zenodo 18159199](https://zenodo.org/records/18159199)). Gotchas: only
27k users, so a SASRec item table for 32M items is trained from
~10 events per item on average — fine for a systems benchmark whose GT is
exact kNN in the same space, but reviewers may prefer the caption-encoded
content vectors; caption/category coverage over the 32M videos is
UNVERIFIED; attribute cardinalities (tags, upload types, authors) are
not published on the site — compute them at ingest.

### 3.3 Yambda-5B, full catalog with artist/album attributes

Probed the HF parquet files directly on 2026-09-05 (`pyarrow` over
`HfFileSystem`): `embeddings.parquet` = 7,721,749 rows, `embed` and
`normalized_embed` are 128-d float64 lists (13.81 GB);
`artist_item_mapping.parquet` = 9,271,906 rows over 9,270,506 distinct
tracks and 1,293,394 artists (a few tracks have >1 artist; tracks per
artist: median 2, mean 7.2, p99 91, max 75,489; top-100 artists cover
4.5 % of tracks); `album_item_mapping.parquet` = 9,651,644 rows over
8,653,783 tracks and 3,367,691 albums (median 1, p99 22, max 1,535). The
card confirms 9,390,623 tracks in the 5B variant, Apache-2.0, ungated
([card](https://huggingface.co/datasets/yandex/yambda),
[HF API](https://huggingface.co/api/datasets/yandex/yambda)). The paper
lists per-event `track_length_seconds` (5-s granularity) and no genre or
language field ([`articles/yambda.md`](../../articles/yambda.md) § Data
Availability). Gotchas: artist/album filters are extremely selective
(single artist ≤ 0.8 % of the pool, median 2e-7) — high pass rates must
come from OR-lists of artists (e.g. the query user's top-k artists) or
from the duration bucket; ~740k tracks have no album row and ~1.67M have
no audio embedding, so `-1` padding must be exercised; the current
Listen+ n-core prep keeps 5.37M tracks — reaching 9.39M means dropping
the n-core or indexing all tracks while training only on the core.

### 3.4 YFCC-10M (NeurIPS'23 Big-ANN filtered track)

> **Access (2026-09-05).** The user's earlier attempt to download YFCC-10M failed. A `HEAD`
> request from the dev Mac the same day returned `HTTP/2 200` with full sizes for the four
> files below (`base.10M.u8bin` 1,920,000,008 B; `query.public.100K.u8bin` 19,200,008 B;
> `GT.public.ibin` 8,000,008 B; `base.metadata.10M.spmat` 945,683,840 B), so the paths are
> served right now. Retry with exactly these URLs (`curl -L -O`) before treating it as blocked;
> `query.metadata.public.100K.spmat` and `unfiltered.GT.public.ibin` were not probed.

10M CLIP vectors, 192-d uint8, L2; each image carries a bag of tags from
a 200,386-word vocabulary (description words, camera model, year,
country); 100,000 queries each require 1–2 tags; GT shipped
([neurips23 README](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md)).
Files (no registration) under
`https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/`:
`base.10M.u8bin`, `query.public.100K.u8bin`, `GT.public.ibin`,
`unfiltered.GT.public.ibin`, and the tag bags as custom sparse CSR files
`base.metadata.10M.spmat` / `query.metadata.public.100K.spmat` read by
`read_sparse_matrix` in
[`benchmark/datasets.py`](https://raw.githubusercontent.com/harsha-simhadri/big-ann-benchmarks/main/benchmark/datasets.py).
Ground truth is the top-k among items whose bag contains **all** query
tags (conjunctive AND)
([results paper](https://arxiv.org/html/2409.17424)). VecFlow reports
10.8 tags per item on average with a long-tailed label frequency and
uses a specificity threshold of 2,000 items to split "common" from
"rare" labels ([VecFlow](https://arxiv.org/html/2506.00812)); the
official low/medium/high specificity boundaries are UNVERIFIED.
Gotchas: (a) an average of ~11 tags per item means the narrow tensor
needs K well above 4 — either cap at K=16 (`[N, 1, 16]` int64 = 1.3 GB)
and drop the tail, or keep the tag clause as a wide/Bloom-only clause;
(b) tags are one vocabulary mixing words, camera, year and country, so
the "AND of features" structure of SilverTorch has to be simulated by
treating the year/country/camera tags as separate clauses (they are
distinguishable by prefix only if the vocabulary file exposes it —
UNVERIFIED); (c) uint8 CLIP vectors at 192-d are lower-dimensional
than our D=256 setting and ℓ2 rather than inner product; (d) query
selectivities are set by the organisers' tag sampling, not tunable.

### 3.5 Cohere Wikipedia 2023-11 (Embed-multilingual-v3)

Verified via the HF datasets-server: en 41,488,110 rows / 97.86 GB, de
20,772,081 / 48.79 GB, fr 17,813,768 / 41.20 GB, es 12,905,284 / 30.43
GB, it 10,462,162 / 24.49 GB, ceb 9,818,657 / 22.37 GB; columns `_id,
url, title, text, emb` (float32 list)
([info](https://datasets-server.huggingface.co/info?dataset=CohereLabs/wikipedia-2023-11-embed-multilingual-v3)).
1024-d per the card's preview; ungated; no license field in cardData
([HF API](https://huggingface.co/api/datasets/CohereLabs/wikipedia-2023-11-embed-multilingual-v3)).
Gotchas: (a) 1024-d fp16 for 80M rows is 164 GB — a PCA to 128/256
(fit on a 1M-row sample) is mandatory, and GT must be recomputed in the
reduced space; (b) the only categorical attribute is the language config
— article-level attributes (Wikidata `instance_of`, page views) would
have to be joined via `title`/`url`, which is an ingest project of its
own; (c) new text queries require the Cohere API (paid, registration),
so the query protocol should be held-out-paragraph-as-query (like arXiv);
(d) the older `Cohere/wikipedia-22-12-en-embeddings` (35.2M rows, 768-d,
carries `views`/`langs` per the community mirror) returned HTTP 401 from
the HF API and datasets-server on 2026-09-05 — it appears gated or
restricted, UNVERIFIED
([search result](https://huggingface.co/datasets/Cohere/wikipedia-22-12-en-embeddings),
[rag-repo mirror notes](https://rag-repo.org/source/cohere-wikipedia-22-12/)).

### 3.6 MS MARCO Web Search, 100M ANN set

100,924,960 document vectors (`vectors.bin` 289.16 GB, SPTAG binary
format) and 9,374 test query vectors with brute-force `truth.txt`; train
9,206,475 queries, dev 9,253; 93 query languages, 207 document languages;
"non-commercial research purposes only"; the ClueWeb22 text, language and
topic tags must be obtained from the Lemur Project
([README](https://raw.githubusercontent.com/microsoft/MS-MARCO-Web-Search/main/README.md),
[paper](https://arxiv.org/html/2405.07526)). Dimension is not stated;
289.16 GiB / 100.9M ≈ 3,076 B/vector is consistent with 768-d fp32 plus
a small header (my inference, UNVERIFIED). Gotcha: without ClueWeb22
there are **no attributes at all**, so this is only a stretch option if
the Lemur agreement is acceptable; even then the vectors must be PCA'd
to ≤ 256-d to fit.

### 3.7 Semantic Scholar SPECTER2 embeddings + `papers`

The 2026-09-01 release lists `papers` (200M records), `abstracts` (100M),
`embeddings-specter_v2` (120M records, "compatible with embeddings
produced by the pretrained model"), `tldrs` (58M), `s2orc_v2` (16M);
ODC-BY for metadata, Apache-2.0 for embeddings; full downloads need an
API key from the partner form
([release listing](https://api.semanticscholar.org/datasets/v1/release/latest)).
SPECTER2 vectors are 768-d
([S2 platform paper](https://arxiv.org/html/2301.10140v2)). Gotchas:
the listing's "28 GB" for 120M embeddings cannot be raw fp32 (that is
~370 GB) — file sizes are UNVERIFIED until a key is issued; the
per-dataset endpoint returned 401 without a key. Attribute fields
(year, venue, fields of study, publication types, open access) are
documented for the API's paper object, not re-verified for the bulk
`papers` schema.

### 3.8 PubMed + MedCPT embeddings (NCBI)

Verified on the NCBI FTP directory: 38 × `embeds_chunk_{i}.npy` (1.1–2.9
GB each, ~102 GB), 38 × `pmids_chunk_{i}.json` (row-aligned PMID lists)
and 38 × `pubmed_chunk_{i}.json` (~44 GB; per PMID `d` date, `t` title,
`a` abstract, `m` MeSH terms), README dated 2023-11-01, chunk 37
refreshed 2024-03-24
([dir](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/),
[README](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/README.txt)).
Each chunk is "about 1 million articles" (chunk 18 = 940,707), arrays are
`(N, 768)`; dtype is not stated — 2.9 GB / (768 × 4 B) ≈ 944k rows
matches the stated chunk size, so fp32 (my inference). The article
encoder (BERT, [CLS], 512 tokens, trained on 255M PubMed click pairs) and
the query encoder are public-domain on HF, so text queries can be
embedded locally — unlike the Cohere dumps
([model card](https://huggingface.co/ncbi/MedCPT-Article-Encoder),
[paper](https://arxiv.org/abs/2307.00589)). Gotchas: the total article
count is not stated anywhere (≈ 36M *est.*, ~10 % below the 40.45M
citations in the 2026 baseline, [FTP baseline](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/));
journal and language are not in the chunk JSON and need a MEDLINE
baseline join keyed by PMID (UNVERIFIED effort); MeSH is multi-valued
(typically 10–15 headings per article — UNVERIFIED) so it should be a
K-wide OR clause over a top-V vocabulary, with year bucket, MeSH
tree-top (category letter) and journal as scalar clauses; BioASQ's 500
BEIR test queries are the natural labelled set but BEIR marks the
corpus as not publicly redistributable ([BEIR](https://github.com/beir-cellar/beir)),
so plan on item-as-query plus NFCorpus/BioASQ only if registration is
done; 768-d fp16 at 36M is 55 GB, so PCA to 256/128 is required (GT
recomputed in the reduced space).

### 3.9 WIKI-ANN, OpenAlex, LAION (deferred)

**OpenAlex — facts verified 2026-09-05** (help.openalex.org snapshot page and the work-object
docs): the public snapshot is `s3://openalex` (`aws s3 sync --no-sign-request`, free, AWS Open
Data covers transfer), ~750 GB compressed JSONL of which works alone are ~670 GB (~780 GB as
parquet), partitioned by `updated_date` in part files of up to 400,000 records, refreshed
quarterly, CC0. Work fields: `title`, `abstract_inverted_index` (inverted for legal reasons,
trivially re-assembled; "over 60 % of works in 2022 have abstract data, 45 % for works older than
2000"), `type` (article / preprint / review / letter / …), `language` (ISO 639-1, auto-detected),
`publication_year`, `primary_topic` + up to 3 `topics` with a domain → field → subfield → topic
hierarchy, `open_access.{is_oa, oa_status ∈ {diamond, gold, green, hybrid, bronze, closed}}`,
`primary_location.source` (venue), `cited_by_count`, `is_retracted`, `authorships → institutions →
country_code`, `keywords` (≤ 5), and `referenced_works` / `related_works`. The last two matter:
citation links give a **real relevance signal** (query = a work, relevant = the works it cites,
SciDocs-style) instead of item-as-query, at any scale. Encoding plan: select works with an
abstract, `language = en`, `type ∈ {article, preprint, review}`, year ≥ 2000, sample 50 M;
`nomic-embed-text-v1.5` (same encoder as the arXiv track, so D = 256 / 128 / 64 without PCA) at
256 tokens ≈ 1,000–1,500 docs/s per A100 *est.* → 50 M ≈ 9–14 A100-h, roughly half on an H100;
`bge-small` / `e5-small` at 128 tokens ≈ 3–4 A100-h. The real cost is upstream: pulling and
streaming ~670 GB of works JSONL through a filter needs ~1 TB of scratch disk and hours of CPU
before the GPU starts. Attributes: topics hierarchy (4 domains → 26 fields → ~250 subfields →
~4,500 topics, per the OpenAlex topics docs, counts UNVERIFIED here), year, type, oa_status,
language, venue, country, citation bucket — the richest real conjunctive-filter set in this survey.

**WIKI-ANN** (§ 2.1) repackages the gated Cohere 22-12 English dump as
35M vectors with content-derived word labels (2-label AND queries), MIT,
ungated on HF — the cheapest ≥ 10M precomputed option after YFCC, but
its "attributes" are words from the passage text, which reviewers may
not accept as catalogue metadata; dim/dtype and GT files are UNVERIFIED
([HF](https://huggingface.co/2024annonymous/wiki-ann)). **OpenAlex** has
the richest real attributes of anything surveyed and CC0 terms, but no
vectors and only partial abstracts; it is the right choice if a second
scholarly pool beyond arXiv is wanted and ~12 A100-h of encoding is
acceptable. **LAION-400M** and Re-LAION are gated on HF and the
original CLIP `.npy` mirrors are reported dead, so they are out.

## 4. Recommendation

> **Decision taken 2026-09-05 (supersedes the ranking below).** The study uses arXiv and
> Goodreads (rerun), **YFCC-10M** (§3.4/§4.5), **OpenAlex ~50 M** (§3.9) and **KuaiRand-27K**
> (§3.2/§4.3): two semantic-search and two recsys corpora beyond arXiv, all with real filters.
> Dropped: Amazon Reviews 2023 (§4.2, out in general), Yambda-full and Cohere Wikipedia (§4.3
> fallback, §4.4 — scale without meaningful filters), PubMed + MedCPT (§4.1, fallback only if the
> OpenAlex snapshot pass is too heavy). Order and gates: [00-roadmap.md](00-roadmap.md) Phase E.


Throughput assumptions used below (*est.*, not measured; the repo's own
arXiv encode of 2.99M abstracts at 512 tokens is only described as
"hours of GPU time" in [datasets.md](../system/datasets.md)):
`nomic-embed-text-v1.5` fp16, 256-token truncation, batch 256 on one
A100 ≈ 1,000–1,500 docs/s; `bge-small-en-v1.5` / `multilingual-e5-small`
at ≤ 128 tokens ≈ 4,000–8,000 docs/s.

GPU budget per item, for planning: fp16 vectors 2·D bytes, int8 IVF codes
D bytes, plus a Bloom/attribute mask that is independent of D. At D=128
that is ~0.4 KB/item (40 GB at 100M), at D=256 ~0.8 KB/item (40 GB at
50M); host-side `item_attrs_narrow.pt` is 8·C·K bytes/item (9 GB for
48M × 6 × 4) — `synth_arxiv` already documents a hard cap on N from this
tensor, so the attrs step must stream and consider K ≤ 4.

### 4.1 Primary semantic-search set (precomputed): PubMed + MedCPT, ~36M articles

Why: zero encode cost, real multi-valued and scalar attributes (MeSH,
year, journal), an open query encoder in the same space, no
registration, public domain — and at ~36M it sits between the papers'
10M and 80M pools. Plan (`eval_datasets/pubmed.py`):

1. `download`: the 114 chunk files from the NCBI FTP (~146 GB; resumable
   `Range` mirror like `goodreads download`).
2. `convert`: stream each `pubmed_chunk_*.json` into parquet (`pmid,
   date, title, mesh`; drop abstracts unless the sequential twin is
   wanted) and each `.npy` into fp16 `text_emb_shard_*.pt` after a PCA
   to 256 and 128 fitted on a 1M-row sample (55 GB → 18.4 / 9.2 GB),
   reusing the `synth_arxiv` shard loader; record the PCA in
   `text_emb.meta.json`. Optional MEDLINE baseline join for `journal`
   and `language` (UNVERIFIED cost).
3. `prep`: `item_id_map.json` (dense over the PMIDs present in the
   chunks), `heldout.parquet` (10k articles as queries), optional
   `queries.parquet` for BioASQ/NFCorpus if their PMIDs are available.
4. `encode_queries`: `ncbi/MedCPT-Query-Encoder` on held-out titles (and
   BioASQ questions), projected with the same PCA — asymmetric
   query/document encoders, so the cross-check sweep is meaningful as
   with nomic's prefixes. Cost *est.* minutes.
5. `attrs`: `[N, 5, 4]`: C0 MeSH heading (top-V vocab, K=4 multi-valued
   OR), C1 MeSH tree-top category (A–Z, from the descriptor's tree
   number, needs the MeSH descriptor file), C2 year bucket (6), C3
   journal (top-5k, reverse clause), C4 has-abstract flag; query clause
   values drawn from the target article. Pass rates: MeSH heading
   ~0.01–5 %, category ~5–40 %, year ~10 %, journal ~1e-4 — covering
   the full range.

Disk: 146 GB raw, ~30 GB processed. A100 memory at D=128: 9.2 GB fp16
+ 4.6 GB int8 — trivially fits; at D=256: 18.4 + 9.2 GB. A100 hours:
≈ 1 (PCA + projection).

### 4.2 Dual-shape product set (encode yourself): Amazon Reviews 2023, 48M items

Why: the only ungated ≥ 10M catalog with short encodable text *and* six
natural categorical/numeric attributes *and* a native text-query set
(Amazon-C4) *and* user histories, i.e. it can serve both shapes. Plan
(`eval_datasets/amazon23.py`, `download → convert → prep → encode_text →
encode_queries → attrs`, mirroring `arxiv.py`):

1. `download`: the 33 `meta_<cat>.jsonl.gz` files (+ `<cat>.jsonl.gz`
   reviews only if the sequential shape is built) from the project site;
   `convert` to ZSTD parquet keeping `parent_asin, title, features,
   description, main_category, categories, store, price, average_rating,
   rating_number`. Dedupe on `parent_asin` across category files
   (cross-file duplication is UNVERIFIED — assert it at ingest).
2. `prep`: `item_id_map.json` (1-indexed dense), `items.parquet`,
   `heldout.parquet` (10k sampled items as queries, kept in the index as
   arXiv does), and `queries.parquet` from Amazon-C4 (21,223 queries,
   `target_item_id` mapped through `parent_asin`; queries whose target is
   missing from the catalog are dropped and counted in `prep_log.json`).
3. `encode_text`: `nomic-embed-text-v1.5`, `"search_document: "` +
   title + features + description[:1500 chars] (the arXiv default),
   `--max-seq-length 256`, Matryoshka 256/128/64 into
   `content{,_d128,_d64}/text_emb.pt` with meta sidecars. Cost *est.*
   48.19M / 1,200 docs/s ≈ 11 A100-h (3 h with `bge-small-en-v1.5`, but
   that loses the Matryoshka dims and the encoder shared with arXiv).
4. `encode_queries`: C4 queries and held-out titles with
   `"search_query: "`, same model.
5. `attrs`: `item_attrs_narrow.pt` int64 `[N, 6, 4]`, `-1` padded:
   C0 `main_category` (34 values, K=1), C1 second-level `categories`
   (top-V vocab, K ≤ 4, multi-valued OR), C2 `store` (top-10k vocab,
   else -1) as the **reverse** clause, C3 log-spaced price bucket (8,
   -1 when missing), C4 `rating_number` popularity bucket (6), C5
   `average_rating` bucket (5); `clause_is_reverse_narrow.pt =
   [F,F,T,F,F,F]`; `eval_split.parquet` aligned with the query table,
   clause values drawn from the target item via `common.synthesize_qa_narrow`
   so pass rates follow the natural marginals: C0 ≈ 1–15 %, C1 ≈
   0.01–5 %, C2 ≈ 1e-5–1e-3, C3/C4/C5 ≈ 10–25 %, conjunctions down to
   < 0.01 % — the full SilverTorch range.
6. Sequential shape (optional): `prep_seq` over the shipped 5-core
   reviews with `timesplit.sequential_split_train_val_test`, then the
   existing gSASRec trainer at D=128; the 5-core item count is
   UNVERIFIED and will be far below 48M, so treat this as a second,
   smaller pool.

Disk *est.*: raw gz 60–120 GB, parquet ~40 GB, embeddings 24.7 + 12.3 +
6.2 GB, attrs 9.3 GB. A100 memory at D=128: 12.3 GB fp16 + 6.2 GB int8
codes + index — comfortable; at D=256: 24.7 + 12.3 GB — fits.

### 4.3 Primary recsys set: KuaiRand-27K, 32M items (sequential shape), with Yambda-full as the low-effort fallback

Why: 32M real items with the richest real attribute set of any public
recsys release (author, type, upload type, music, tags, 4-level
categories, duration, upload date) plus dense timestamped histories; the
only ≥ 10M recsys candidate that is not behind a wall. Plan
(`eval_datasets/kuairand.py`):

1. `download`: Zenodo 10439422 (`KuaiRand-27K.tar.gz` 9.9 GB
   compressed, CC BY 4.0 on the record; the site's "46 GB" is the
   unpacked size, [record](https://zenodo.org/records/10439422)) and
   18159199 (6.9 GB);
   `convert` the `log_standard_*` CSV parts, `video_features_basic_27k`,
   `kuairand_video_categories.csv`, `kuairand_video_captions.csv` to
   parquet.
2. `prep`: positives = `is_click == 1` (alternatives: `long_view`;
   decide from counts in `prep_log.json`), exclude `is_rand == 1` rows
   from training, global temporal split at the last days of the
   4-22→5-08 log; histories capped at 200 like Yambda. Only 27,285
   users exist, so emit several evaluation rows per user at different
   cut points (the harness reads one query per `test.parquet` row) to
   reach ≥ 100k queries.
3. Train gSASRec D=128 over the 32,038,725-item vocabulary. The
   trainer allocates `nn.Embedding(num_items + 1, D)` twice (input and
   output tables unless `reuse_item_embeddings`,
   [`training/model.py`](../../evaluation/training/model.py) lines
   25–47): 16.4 GB fp32 each, plus two AdamW moments per table — ~100 GB
   with separate tables, ~49 GB with `reuse_item_embeddings=True`, so
   the 32M run needs the shared table (or a sharded/sparse optimizer)
   to fit an 80 GB A100. Full-catalog eval already scores in
   `eval_score_chunk = 262,144`-item chunks
   ([`training/evaluate.py`](../../evaluation/training/evaluate.py)),
   so evaluation is fine. Cost *est.* 6–12 A100-h, UNVERIFIED until a
   first epoch is timed.
4. Text-shape twin (optional, cheap): encode the Chinese captions with
   `multilingual-e5-small` (384-d, truncate to 128 via PCA) — *est.*
   32M / 6,000 docs/s ≈ 1.5 A100-h — and run item-as-query; this also
   gives an embedding for the tail items the sequential model barely
   trains.
5. `attrs`: `[N, 7, 4]`: C0 first-level category (K=1), C1 second-level
   category, C2 `upload_type`, C3 `music_type`, C4 duration bucket
   (from `video_duration`), C5 upload-month bucket, C6 `author_id`
   (reverse clause: author ≠ x); tags as a K ≤ 4 multi-valued clause if
   the vocabulary is small enough. Query-side clause values come from
   the user's recent positives (category ∈ {last-5 categories}, author
   ≠ most-recent author), giving ~50 % → ~1e-4 pass rates once
   cardinalities are measured at ingest.

If 27,285 users prove too few, **KuaiSAR** (6.89M items, 25,877 users,
19 days, 18 item features, CC BY-NC-SA, Zenodo — § 2.1) is the only
other recsys release above the 5M floor, but it has fewer users and a
shorter span, so it does not fix the query-count problem.

Fallback with ~1 A100-h of new work: **Yambda-5B full catalog**
(9.39M, § 3.3). Add `--keep-all-items` to `yambda prep` so
`item_id_map.json` covers all 9,390,623 tracks while training stays on
the Listen+ core; add `yambda attrs` joining `artist_item_mapping`
(first artist, K=2 for multi-artist tracks), `album_item_mapping`,
a `track_length_seconds` bucket (mode per track, 6 buckets) and a
`has_audio_embedding` flag; `clause_is_reverse = [F,F,F,T]` on the flag
or on artist. Query clause values: artist ∈ {top-8 artists in the user's
history} (≈ 1e-4–1e-3), album of the target (≈ 1e-7, the LiNR
"low-pass-rate" regime), duration bucket (≈ 15–40 %). A text-shape twin
is free: `normalized_embed` (7.72M × 128, float64 → fp16, 2 GB) with
10k held-out tracks as queries.

### 4.4 Stretch ≥ 50M: Cohere Wikipedia 2023-11, en + de + fr = 80.07M paragraphs

Why: precomputed, ungated, HF-hosted, and the pool size matches
SilverTorch's 80M setting exactly; the trade-off is thin attributes and
a mandatory dimensionality reduction. Plan (`eval_datasets/wiki_cohere.py`):

1. `download` the `en`, `de`, `fr` configs (97.86 + 48.79 + 41.20 =
   187.85 GB of fp32 parquet); stream row-group by row-group.
2. `reduce`: fit PCA to 256 and 128 on a 1M-row sample (Embed-v3 has
   no Matryoshka truncation — only Embed-v4 does, at 256/512/1024/1536,
   [Cohere docs](https://docs.cohere.com/docs/embeddings)), project,
   L2-normalise, write fp16
   `text_emb_shard_*.pt` + `shard_index.json` (reusing the
   `synth_arxiv` sharded loader) — 41 GB at 256, 20.5 GB at 128. GT is
   recomputed in the reduced space, so the benchmark stays
   self-consistent; state the PCA in the paper.
3. Queries: 10k held-out paragraphs per language (item-as-query); text
   queries would require the paid Cohere API, so they are out of scope.
4. `attrs`: C0 language (3 values, 52 / 26 / 22 %), C1 article-size
   bucket (paragraphs per `title`), C2 paragraph-position bucket
   (from `_id` order within an article — format UNVERIFIED), C3 text
   length bucket; the reverse clause is language ≠ query language
   (cross-lingual neighbours, which Embed-v3 supports). Selectivity
   50 % → ~1 % only; the sub-1 % regime would need a Wikidata join on
   `title` (open question 3).

Cost: no encoding; PCA + projection ≈ 1–2 GPU-h; download-bound.
Alternatives if a walled dataset is acceptable: MS MARCO Web Search 100M
(vectors free, attributes require the ClueWeb22 agreement, § 3.6) or
S2 SPECTER2 (API key, § 3.7) — both bring real query sets or rich
attributes that the Wikipedia dump lacks.

### 4.5 Add for comparability: YFCC-10M (retry the download first, §3.4)

Cheapest ingest of all (precomputed uint8 vectors, shipped filtered GT
for 100k queries) and the set every filtered-ANN paper reports on; the
user's first download attempt failed but the files were served on
2026-09-05 (§3.4), so retry before writing it off. Original reasoning:
the multi-valued tag bag maps to a single K-wide OR clause. Use it as
the cross-paper anchor rather than as the headline set, because its
attributes are one tag vocabulary rather than the multi-feature
conjunctions SilverTorch and LiNR describe.

## 5. Open questions

1. Amazon Reviews 2023: license terms (none stated on the HF card or
   site), `parent_asin` duplication across category files, and the
   share of the 48.19M items with a non-empty `title` — all decide the
   effective pool size.
2. KuaiRand-27K: caption and category coverage of the 32M videos, tag
   vocabulary size, the right positive signal (`is_click` vs
   `long_view`), and whether the gSASRec trainer copes with a 32M-row
   output table; 27,285 users may be too few test queries without
   multi-cut-point evaluation.
3. Cohere Wikipedia: is a PCA'd Embed-v3 space acceptable to reviewers
   as a "real" embedding, and can a Wikidata `instance_of` /
   `country` join via `title` deliver sub-1 % attributes cheaply?
4. Encoder choice for new text pools: keep `nomic-embed-text-v1.5`
   (Matryoshka, consistent with arXiv, ~4× slower) or switch to
   `bge-small` / `multilingual-e5-small`? The throughput numbers in § 4
   are estimates and should be measured on 100k docs first.
5. YFCC-10M: maximum tags per item decides K for the narrow tensor;
   if the tail is long, cap at K and record the truncation in
   `prep_log.json`.
6. Yambda-full: multi-artist tracks and 740k tracks without an album —
   confirm `-1` padding is exercised by the Bloom path.
7. Walled stretch options: is a Semantic Scholar API key (partner form)
   or a ClueWeb22 agreement obtainable within the paper timeline?
8. Host RAM: `int64 [N, C, K]` at 48M–80M items (9–15 GB) plus the
   fp32 → fp16 conversion buffers; the attrs step must stream.
