---
title: data-intro-literature
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/paper/outline.md, docs/system/datasets.md, docs/system/checkpoints.md, evaluation/config/, research/paper/provenance-and-disclosure.md, research/paper/release-and-licenses.md, research/validation.md, arXiv:2407.13218v3, arXiv:2511.14881v5, arXiv:2507.21989, arXiv:2508.16263, arXiv:2506.00812, arXiv:2409.17424]
---

# Material C1: data, introduction, literature

Collected facts for the [outline](../outline.md)'s §1 (introduction), §2 (the
originals, related work), §3 T-data and §8 (data licences). Not prose. Each
line has its source. **Citability:** "citable" means a published paper, or a
gate green in [validation](../../validation.md); "NOT CITABLE" says why.

**Citability of the data facts** (user ruling 2026-10-16). Every per-sweep
number in Part 1 (N, queries kept, pass rates) is a property of the staged
datasets, read from the final-pass records and unchanged by the FULL-RERUN
(the same datasets): **final, citable**, with the dataset configs and
[validation § Datasets](../../validation.md#datasets) as source.

**How the pass-rate spread was computed.** `pass_rate` per record is the mean
over kept queries of `pass_count / N`. The spread columns come from one
`linr_v1_filter_mask` clause record per (dataset, sweep): its sidecar's
`pass_count` array (one value per kept query, [evaluation](../../../docs/system/evaluation.md#per-query-sidecar)),
divided by N. Those sidecars are under Hub `campaign-final/<dataset>-c3-real/`
(goodreads, arXiv, PubMed), `laion30m-laion30m-d0/`, `laion{1,3,10}m-nsweep/`
and `arxiv-corr-synth-synth-d0/` (`linr_v3` there, the only exact-mask arm on
that dataset). The pass set is the exact clause predicate, so the bloom
records of a sweep carry the same `pass_rate`.

## Part 1. Data (T-data, §3)

### 1.1 One row per dataset

| dataset (harness name) | N items | d used in the final pass | queries (used / available) | query source | target | encoder | raw source, version | citable |
|---|---|---|---|---|---|---|---|---|
| Goodreads, work-id catalog (`goodreads`) | 797,084 | 128 (d256 only on `goodreads-synth`, LiNR V1, C3) | 10,000 / 313,178 test users (`users_limit`); kept per sweep 9,859-9,971 (§1.2) | the user's last 200 interactions (time split), encoded at eval time by our sequence encoder | the held-out next item(s) of the test split | ours: `sasrec-ssm-logq-d128`, the E1c recipe (gSASRec body, 2 blocks, ffn 4×D, sampled softmax with 4,096 in-batch + 8,192 uniform negatives and logQ correction, L2-normalised); test ndcg@10 0.0410, R@100 0.1518 | UCSD Book Graph (Wan & McAuley, RecSys '18), `mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads`; `is_read`, editions collapsed to works, n-core, time split; staged copy pinned at Hub `pinkmeme/eval-goodreads-work-id@4714e9b5` | N, queries: final. Encoder quality: **NOT CITABLE** (the E1c rows in [validation](../../validation.md#final-models-the-e1c-recipe) say "not yet validated") |
| arXiv papers (`arxiv`) | 2,988,996 | 128 (real filters); 128 and 256 on `arxiv-synth` | 10,000 / 10,000 | 10,000 papers drawn with a seed; the query text is the paper's own `"{title}. {abstract[:1500]}"` with prefix `search_query: ` | the paper itself, which stays in the index | `nomic-embed-text-v1.5`, 768-d encode truncated to 256 and 128 (Matryoshka, re-L2-normalised after slicing, no `layer_norm`); items with prefix `search_document: ` | HF `open-index/open-arxiv` (parquet copy of the Cornell arXiv metadata), snapshot 2026-03-24, 2,989,022 papers → 2,988,996 after dropping empty abstracts; pinned at `pinkmeme/eval-arxiv-papers@354e6937` | final |
| PubMed, 10 M slice (`pubmed`) | 10,000,000 | 768 | 10,000 / 10,000 | held-out articles; the query is the article title, encoded by `ncbi/MedCPT-Query-Encoder` ([CLS]) | the article itself | NCBI's precomputed MedCPT article embeddings, float32 768-d, stored fp16 | NCBI FTP `pub/lu/MedCPT/pubmed_embeddings/` (35,920,666 articles, sizes as served 2026-09-16) + MEDLINE baseline `pubmed26n*` (journal, language) + MeSH `desc2026`; slice = the 10 M smallest seeded splitmix64 hashes of the PMID (`--keep-items 10000000 --seed 0`); not on the Hub, pinned by sha256 | final |
| Re-LAION 30 M (`laion30m`) | 30,000,000 | 256 | 10,000 / 10,000 | held-out captions (from 928,288 distinct captions left outside the catalog), drawn with a seed | each query's exact unfiltered top-1 item (a caption has no relevant item of its own) | `nomic-embed-text-v1.5`, model-card Matryoshka recipe: `layer_norm` over 768, keep 256, L2-normalise, fp16; `max_seq_length` 128 | HF `laion/relaion2B-en-research-safe` (gated, auto-approved), parts 0-1 (32,776,409 rows) → 30,928,288 distinct captions → the 30 M smallest seeded hash ranks; not on the Hub | final |
| Re-LAION 1 / 3 / 10 M (`laion1m`, `laion3m`, `laion10m`) | 1,000,000 / 3,000,000 / 10,000,000 | 256 | 10,000 / 10,000 (the parent's queries) | as `laion30m` | recomputed per subset: exact unfiltered top-1 over the subset | as `laion30m` (rows copied, not re-encoded) | `eval-data subset`, seed 20261013; nested (1 M ⊂ 3 M ⊂ 10 M) | final |

Sources, by column: N, d, queries kept and pass rates from the final-pass
parquet (`n_items`, `dim`, `n_queries`, `n_kept`, `n_queries_oracle`, `pass_rate`).
Query and target construction, encoders, raw sources:
[datasets](../../../docs/system/datasets.md) §§ goodreads, arxiv, pubmed,
laion30m, Subsets; [`config/*.yaml`](../../../evaluation/config/) (`users_limit:
10000` on every dataset); the arXiv sidecar `content_d128/query_emb.meta.json` and
the Hub card of `pinkmeme/eval-arxiv-papers` (snapshot date, 2,989,022 → 2,988,996).
Goodreads encoder: [checkpoints](../../../docs/system/checkpoints.md#what-the-harness-reads),
[validation](../../validation.md#final-models-the-e1c-recipe).

**Arms per dataset in the final pass** (`algo` values): every real-filter dataset
runs `linr_v1_filter_mask`, `linr_v2`, `linr_v3`, `postfilter`, `silvertorch`;
the LAION N-sweep subsets (1 / 3 / 10 M) run V1, V2 and SilverTorch only; the
`-synth` datasets run V1 / V2 (plus V3 and the postfilter on goodreads-synth). The
study's SilverTorch numbers come from the NUMBERS legs, not the final pass
([outline](../outline.md#numbers-and-citability)).

### 1.2 Real filters: the sweeps in the final pass

Clause semantics: an item passes a clause when any of its slot values equals the
query's value for that clause; clauses are ANDed; a **reverse** clause is negated
(passes items whose slots do not match) ([filtering](../../../docs/system/filtering.md)).
Query values are **target-derived** on goodreads, arXiv and PubMed (the first
non-pad value of the held-out target's own attribute, `common.synthesize_qa_narrow`),
and the held-out caption's own tags on LAION.
Bloom sweeps never include a reverse clause.

| dataset | sweep | clauses (meaning) | kind in the final pass | queries kept / with ≥ 1 passing item | mean pass rate | per-query pass rate: p10 / median / p90 (min-max) |
|---|---|---|---|---|---|---|
| goodreads | `c0_genre` | C0 genre (top-4 genres per work) | clause, bloom | 9,859 / 9,859 | 0.3307 | 0.147 / 0.260 / 0.556 (0.040-0.556) |
| goodreads | `c1_lang_reverse` | C1 language, **reverse** ("not the target's language") | clause | 9,192 / 9,192 | 0.6205 | 0.572 / 0.572 / 0.947 (0.572-1.0) |
| goodreads | `all4` | C0 ∧ C1 (reverse) ∧ C2 format ∧ C3 publication-year bucket | clause | 9,971 / 9,971 | 0.0409 | 0.0057 / 0.021 / 0.064 (2.0e-5-0.984) |
| arXiv | `c0_maincat` | C0 main category | clause, bloom | 10,000 / 10,000 | 0.1363 | 0.024 / 0.115 / 0.247 (3.7e-6-0.247) |
| arXiv | `c3_nversions` | C3 version-count bucket | clause, bloom | 10,000 / 10,000 | 0.4444 | 0.088 / 0.603 / 0.603 (0.041-0.603) |
| arXiv | `all4` | C0 ∧ C1 licence bucket (11) ∧ C2 update-year bucket ∧ C3 | clause, bloom | 10,000 / 10,000 | 0.00938 | 4.7e-4 / 0.0055 / 0.024 (6.7e-7-0.037) |
| PubMed | `c0_mesh` | C0 MeSH descriptor (the 4 globally rarest of the article's headings; query = the target's rarest) | clause, bloom | 8,428 / 8,428 | 0.000189 | 2.4e-5 / 1.2e-4 / 3.9e-4 (2.0e-6-0.049) |
| PubMed | `c3_journal_reverse` | C3 journal (`MedlineTA`, top 5k), **reverse** | clause | 8,600 / 8,600 | 0.99935 | 0.999 / 1.0 / 1.0 (0.992-1.0) |
| PubMed | `all5` | C0 ∧ C1 MeSH tree-top category ∧ C2 year bucket ∧ C3 (reverse) ∧ C4 has-abstract | clause | 10,000 / 9,985 | 0.0181 | 2.4e-6 / 1.4e-5 / 0.121 (0-0.689) |
| LAION 30 M | `c0_domain` | C0 registered domain of the image URL ("search within a site") | clause, bloom | 9,722 / 9,722 | 0.00951 | 8.7e-7 / 0.00178 / 0.041 (3.3e-8-0.059) |
| LAION 30 M | `tags4` | C1 image size ∧ C2 CLIP similarity ∧ C4 watermark ∧ C5 unsafe (5 buckets each) | clause, bloom | 10,000 / 10,000 | 0.00266 | 8.2e-4 / 0.0022 / 0.0043 (9.9e-5-0.072) |
| LAION 1 / 3 / 10 M | `c0_domain` | as at 30 M | clause, bloom | 9,722 / 8,839 · 9,236 · 9,527 | 0.00952 / 0.00950 / 0.00951 | median 0.00177-0.00178 at each N |
| LAION 1 / 3 / 10 M | `tags4` | as at 30 M | clause, bloom | 10,000 / 10,000 | 0.00266 / 0.00265 / 0.00266 | median 0.0022 at each N |

Sources: final-pass parquet and sidecars (above); clause meanings:
[datasets](../../../docs/system/datasets.md) §§ goodreads, arxiv, pubmed
("Attribute semantics", "The MeSH cap"),
laion30m ("Attribute semantics"); sweep definitions:
[`goodreads.yaml`](../../../evaluation/config/goodreads.yaml),
[`arxiv.yaml`](../../../evaluation/config/arxiv.yaml),
[`pubmed.yaml`](../../../evaluation/config/pubmed.yaml),
[`laion30m.yaml`](../../../evaluation/config/laion30m.yaml). Final.

Notes on the table:
- **Kept vs passing.** "Kept" is `n_kept`, "with ≥ 1 passing item" is
  `n_queries_oracle`. On the LAION subsets, queries whose domain has no item in
  the subset drop out of the oracle: 883 at 1 M, 486 at 3 M, 195 at 10 M, 0 at
  30 M. On PubMed `all5`, 15 queries have no passing item.
- **Configured but not in the final pass:** goodreads `c2_format`, `c3_year`, `c0c1`;
  arXiv `c2_year`, `c0c2`; PubMed `c2_year`, `c0c2`; LAION `c3_domain_reverse`,
  `all_fwd`. The yaml files list them; no record of the final pass has them.
- **The staged goodreads and arXiv attrs are the 4-clause legacy files**: on disk
  `item_attrs_narrow.pt` is `[797085, 4, 4]` (goodreads) and `[2988997, 4, 4]`
  (arXiv), pad row 0 dropped by the loader (read 2026-10-12 on `/data`). So
  goodreads `all4` has no author clause and arXiv has no C4 author clause, though
  the current `attrs` step writes `[N, 5, 4]` for both ([datasets](../../../docs/system/datasets.md#arxiv)).
- **Selectivity range.** Mean pass rates of the real sweeps span 1.9e-4 (PubMed
  `c0_mesh`) to 0.999 (PubMed `c3_journal_reverse`); per query, from a single item
  (LAION 30 M `c0_domain`, 3.3e-8) to all items.
- The goodreads `c0_genre` 141 dropped queries (10,000 → 9,859) agree with
  [provenance §7](../provenance-and-disclosure.md#7-data-provenance).

### 1.3 Synthetic filters

| variant | how it is built | rates in the final pass | datasets (N, d) | achieved mean pass rate | citable |
|---|---|---|---|---|---|
| **uniform** | one `u_i ~ U(0, 1)` (float64) per item from `torch.Generator().manual_seed(20261008)`, in item-row order; item passes rate `p` iff `u_i < p`, so pass sets nest across rates; `p = 1` is the unfiltered point; every query carries all-ones attrs, so the pass set is the same for every query | `layout.SYNTH_RATES` = 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0; the `synth` suite runs nine (no 0.3 on arXiv / LAION; goodreads-synth also has 0.3) | `goodreads-synth` (0.8 M, d128; C3 also d256), `arxiv-synth` (3 M, d128 and d256), `laion{1,3,10,30}m-synth` (d256) | e.g. arXiv 0.000992 / 0.003002 / 0.010002 / … / 0.499745 at p 0.001 / 0.003 / 0.01 / … / 0.5; goodreads 0.000990 at p 0.001 (789 of 797,084 items; the V-PILOT check misses its 1 % bound there, [validation](../../validation.md#campaign-v2-phase-v) row V-PILOT); LAION 1 M 0.001058 at p 0.001 | final |
| **cluster-correlated** (IVF's best case: the pass set sits around the query) | for each rate `p`, spherical k-means (seed 20261008, random init, ≤ 25 Lloyd iterations, CPU) with `round(1/p)` centroids over the d128 item embeddings; an item's value is its cluster id, a query's value its nearest centroid by inner product; pass rate per query = its cluster's share of N; not nested across rates | 0.01, 0.03, 0.1 (sweeps `c001`, `c003`, `c01`) | `arxiv-corr-synth` only (3 M, d128) | mean 0.0105 / 0.0325 / 0.1074; per query 0.0050-0.0149 / 0.018-0.054 / 0.055-0.147 (std 0.0021 at p 0.01, from `synth_corr.json`) | final |

Sources: [datasets](../../../docs/system/datasets.md#synthetic-selectivity-attrs);
`/data/arxiv-papers/synth_filter.json` and `synth_corr.json` (seed, rates,
`n_items`, k-means settings, per-rate query pass-rate stats; read 2026-10-12);
final-pass parquet. The synth attrs are not on the Hub; they rebuild from the
seed in seconds ([release §1](../release-and-licenses.md#1-datasets)).

### 1.4 Ground truth

- **Exact filtered oracle**, ours, for every dataset: fp32 full scan, TF32 off,
  filtered top-K by inner product (cosine, since items and queries are
  L2-normalised); cached as `oracle_v4_<sweep>_<fingerprint[:16]>.pt`, where the
  fingerprint is a sha256 over shapes, dtypes and a 64-row sample of items,
  queries, targets and query attrs, the full bytes of the item attrs and reverse
  flags, the clauses and `k_gt`; a changed input can never reuse a stale blob.
  Source: [evaluation](../../../docs/system/evaluation.md#oracle-blob-v4). Citable
  as a method description.
- **Oracle check**: no study dataset ships a filtered ground truth; the oracle
  is checked by the independent exact LiNR arms, which must reach
  `recall_oracle@k_max` ≥ 0.99 against it in every cell and do on every study
  dataset ([deviations HB-2](../reproduction-deviations.md#6-deviations-of-our-harness-against-the-papers-protocols--and-against-itself)). Citable.
- **Oracle reuse cross-check**: the one rebuilt oracle (arXiv `c0_maincat`)
  equals the blob the baseline used, `torch.equal` on both `[10000, 1000]`
  tensors ([provenance §7](../provenance-and-disclosure.md#7-data-provenance)).

### 1.5 Licences and redistribution (§8)

Summarised from [release and licenses §1](../release-and-licenses.md#1-datasets)
(read from each source's own page, 2026-10-10; not legal advice).

| dataset | licence / terms | derived data shareable? | attribution |
|---|---|---|---|
| Goodreads | none; README: "academic use only! Please do not redistribute" | **no: recipe only**; the trained checkpoints are an open user decision | Wan & McAuley RecSys '18; Wan et al. ACL '19 |
| arXiv | metadata CC0 1.0 | yes (embeddings, attrs, splits, oracles) | none required; name the encoder |
| PubMed / MedCPT | NCBI public-domain notice; NLM terms | vectors and attrs yes; **titles / abstracts no** (publisher copyright) | "Courtesy of the U.S. National Library of Medicine"; state the 2026 baseline is not current; cite MedCPT (Jin et al., arXiv:2307.00589) |
| Re-LAION-2B-en-research-safe | Apache-2.0; "released for research purposes", gated on consent | vectors and tag buckets yes; **never captions or URLs** | keep the licence, name LAION, cite the Re-LAION post |
| synth attrs | ours | recipe (seeded) | — |

Open user decisions ([release §5](../release-and-licenses.md#5-decisions-for-the-user)):
(1) goodreads checkpoints, (2) Re-LAION vectors gated / ungated / recipe-only,
(3) PubMed recipe vs vectors-only repo, (4) the public May result repos,
(5) the public `eval-results` shape, (6) GitHub visibility during review.

### 1.6 Disagreements between the dataset sources

Rows 1-4 and 7 of C1's list are fixed on `staging` (results-draft `85c4ec8`,
`e7119b3`): provenance §7 has one row per study dataset as staged (goodreads
E1c, arXiv `[N+1, 4, 4]`, PubMed, LAION 1-30 M); datasets.md names
the staged 4-clause goodreads file and lists laion in the synth section.
What remains:

| # | what disagrees | where | which is right now |
|---|---|---|---|
| 5 | arXiv Hub card: links `github.com/anthropics/retrieve`, lists files the repo lacks, no licence | `pinkmeme/eval-arxiv-papers` README | stale; rewritten before it goes public ([release §3](../release-and-licenses.md#3-what-is-on-the-hub-and-github-now)), part of the release decisions ([outline](../outline.md) open item 2). Its snapshot date and counts agree with the final pass |
| 6 | Hub mirror layout `[N + 1, …]` vs the ETL's `[N, …]` | provenance §7 note, datasets "Shared conventions" | both pages agree; a disclosure, not a conflict |
| 8 | `eval-laion30m`: "registered … not created yet" vs "(private) is to hold … not yet published" | release §3 vs datasets § HuggingFace I/O | consistent in substance (nothing published); wording differs |

## Part 2. Introduction (§1)

### 2.1 The originals (all citable: published papers)

| | LiNR | SilverTorch |
|---|---|---|
| full title | *LiNR: Model Based Neural Retrieval on GPUs at LinkedIn* | *SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs* |
| venue | CIKM '24 (33rd ACM CIKM, Boise, 21-25 Oct 2024), 8 pp., DOI 10.1145/3627673.3680091 | SIGIR '26 (49th ACM SIGIR, Melbourne, 20-24 Jul 2026), 12 pp., DOI 10.1145/3805712.3809755 |
| arXiv read | 2407.13218v3 (7 Aug 2024) | 2511.14881v5 (8 May 2026) |
| authors | Borisyuk, Song, Zhou, Parameswaran, Arun, Popuri, Bingol, Pei, Lee, Zheng, Shao, Naqvi, Zhou, Gupta (LinkedIn) | Xue, Wu, Chen, Yang, … Zhang, Liu (32 authors, Meta Platforms) |
| deployment | out-of-network post recommendations on the LinkedIn Feed (two-tower embeddings, EBR); "+3 % relative increase in professional daily active users" (abstract, §4.1, §5.2) | Meta recommendation serving; "our production deployment, which serves hundreds of millions of items" (§2.2) |
| code | none released | `github.com/meta-recsys/silvertorch`, Apache-2.0 (created 2025-10-07; not linked from arXiv v5's text, grep of the PDF) |

### 2.2 Evidence for "only proprietary data" and "never compared"

- **LiNR's data.** Model quality: "We evaluated LiNR model on our internal dataset"
  (§5.1, Hit Rate@400). Latency (§5.3): the job recommendation index, "around 15.5
  million jobs with 25 thousand queries", high-pass set (geo + company-reverse,
  ~1.7 M passing) and low-pass set (+ exact title clause), d128 fp16. A/B tests §5.2.
  No public dataset anywhere in the paper. **Baselines:** cosine exhaustive search
  (§5.1); in production, dot-product EBR and FAISS-IVFPQ (§4.1, "our baselines");
  TF vs PyTorch implementations of V1 / V2 (Tables 3-4). (arXiv:2407.13218v3)
- **SilverTorch's data.** "real-world datasets sampled from production: an
  80-million item candidate pool (80M) and a 10-million item pool (10M), with
  embedding dimension 128 on A100-40G GPUs", HSTU user tower, 5,000 replayed
  production requests (§6); breakdowns on a 20 M × 128 item set with User-Tower
  queries (§6.2.1, §6.2.3) and a 40 M-item bloom index with 5,000 real filtering
  queries (§6.2.2). No public dataset. **Baselines:** Faiss-CPU (IVF) + CPU inverted
  index; Faiss-GPU (IVF) + GPU forward index; HNSW; CAGRA excluded because it
  "limits top-k to 1024" (§6, §6.2.1). (arXiv:2511.14881v5)
- **Never compared — verified.** SilverTorch cites LiNR once, in §2.1 Related Work:
  "LiNR [5] proposes model-based retrieval on GPUs. Unlike SilverTorch, LiNR does
  not provide co-designed index and does not propose a GPU solution for filtering.
  It also does not discuss the cost efficiency of introducing GPUs." LiNR is not
  among the evaluation baselines (§6). One more mention: §7 cites [5] for the
  append-only fresh-index approach. LiNR (2024) predates SilverTorch and does not
  cite it. [claims](../claims.md) is right.
- **A tension to state neutrally** (useful for §2): SilverTorch §1 says "To the best
  of our knowledge, no existing work studies feature filtering on GPUs", and §2.1
  says LiNR "does not propose a GPU solution for filtering"; LiNR §3.1 and §6
  describe "a custom CUDA solution for pre-filtering on the GPU" (attribute matching
  as a custom operator, §5.3).

### 2.3 Adoption and impact (what can be cited)

- `meta-recsys/silvertorch`: 34 stars, 13 forks, Apache-2.0, created 2025-10-07,
  last push 2026-09-09 (GitHub API, read 2026-10-12). Citable as "as of"; whether
  to mention it is the user's call (brief).
- Citation counts, OpenAlex (2026-10-12): LiNR (DOI) **1**; SilverTorch (DOI) **0**.
  Semantic Scholar was not reachable (HTTP 429 on every try); Google Scholar not
  queried. OpenAlex indexing of 2026 work lags, so the counts are a floor: **NOT
  CITABLE as an impact claim**; the paper should argue impact from deployment scale
  (2.1) and code availability, not citations.
- Follow-up from LinkedIn found while searching: *Large Scale Retrieval for the
  LinkedIn Feed Using Causal Language Models* (AAAI 2026, DOI
  10.1609/aaai.v40i47.41445; OpenAlex title search). Not read; whether it builds on
  LiNR is unverified.

### 2.4 The papers' own "model-based retrieval" wording (for §1 and the Faiss / cuVS exclusion)

- LiNR, abstract: "In LiNR, both items and model weights are integrated into the
  model binary. Viewing index construction as a form of model training …"
- LiNR §1: "combining exhaustive search with pre-filtering in a differentiable GPU
  model … In LiNR, item vectors and model weights coexist within the same model
  binary, unlike traditional search indexing methods."
- LiNR §2: Faiss, ScaNN, SONG, RAFT (HNSW, IVFPQ, CAGRA) are "Termed model-free …
  In contrast, our approach employs deep neural networks for a model-based search
  index … We integrate item indexes with neural network weights within a PyTorch or
  TensorFlow model."
- SilverTorch abstract: "a model-based serving system that brings all components
  into one unified model. It unifies model serving by replacing standalone indexing
  and filtering services with model layers."
- SilverTorch §1 contributions: "re-defines the standalone ANN and feature filtering
  services with in-model tensor operators"; §2.1: "SilverTorch's model-based approach
  defines ANN and feature filtering within PyTorch layers running on GPUs."
- SilverTorch §3: "Both the ANN index and bloom index are represented as GPU tensors
  that serve as parameters of the served model."
- Both papers frame Faiss-style libraries as separate services: SilverTorch §1
  ("libraries like Faiss, RAFT or a dedicated vector database system like Milvus";
  "Retrieval therefore relies on the indexing and filtering services"), LiNR §2
  ("model-free"). This is the textual ground for excluding Faiss / cuVS / HNSW as
  arms, though SilverTorch itself benchmarks against them (§6.2.1).
- **Wording caution for the paper.** LiNR calls *both* V1 (similarity masking) and
  V2 (explicit pre-filtering) "KNN with ABM" pre-filtering and argues *against*
  post-filtering (§5.3, §6 "Pre-filtering"). The submitted abstract's "an exhaustive
  GPU scan with pre- or post-filtering" reads V1 as post-filtering; the paper body
  should say "masking vs gathering" (V1 vs V2) and keep "post-filtering" for our
  `postfilter` baseline (matmul → top-k → drop).

## Part 3. Related work (§2, ≤ 0.3 page)

| work | what it is | GPU? | in-graph op or service? | relation to us | read |
|---|---|---|---|---|---|
| **Iff et al., SIGIR '26** | benchmark of 11 filtered-ANN (FANNS) methods on *arxiv-for-fanns*: >2.7 M arXiv abstracts, **4,096-d** `stella_en_400M_v5` embeddings, 11 real attributes (categorical, numerical, set-valued); EM / R / EMIS filters, 10k queries each, k 20-100, three scales (1k / 100k / 2.7 M), a selectivity study; 8 observations (e.g. specialised methods up to 10× faster; recall < 0.2 for some methods at 2.7 M; index builds up to 6 h; most methods stable across selectivity, some degrade at low selectivity) | **no**: CPU only (i7-1165G7 for medium; 2× Xeon Gold 6154, 384 GB for large; §5.4) | standalone index libraries | our arXiv set is a different build (open-arxiv 2026-03-24 snapshot, 2.99 M, nomic 128/256-d, 4 attribute clauses). Positioning: they benchmark CPU FANNS libraries on one new dataset; we test two industrial **GPU in-graph** systems' claims, head to head, on five datasets up to 30 M, with exact recall | full text |
| **Li et al., SIGMOD '26** (*Attribute Filtering in ANN Search: An In-depth Experimental Study*) | unified interface and taxonomy; 10 algorithms / 12 methods on 4 datasets (SIFT 10 M d128, SpaceV 10 M d100 synthetic labels; Redcaps 1 M d512, YouTube-RGB 1 M d1024 real or synthetic labels), selectivity 0.1 %-100 %, L2, QPS at one query thread | **no** (Xeon Platinum 8358, 128 cores; queries single-threaded) | standalone graph / partition indexes | same selectivity axis as our synth filters (0.1 %-100 %); no GPU, no in-graph op, no claim reproduction | full text |
| **Big-ANN NeurIPS'23, filtered track** (Simhadri et al., NeurIPS '25 D&B) | YFCC-10M: 10 M CLIP uint8 192-d, tag bags (200,386-word vocab), 100,000 queries with 1-2 tags, conjunctive AND; scored as the highest QPS at 10-recall@10 ≥ 0.9 on a standardised 8-vCPU Azure D8lds_v5 VM; Faiss IVF baseline (vector-first or metadata-first); 10 entries, winner ParlayANN ~37.7k QPS, > 11× the baseline; private query set at the end | **no** (CPU VM) | standalone indexes | we use its dataset and verify its shipped GT (§1.4); we measure latency per batch on a GPU, k 100 / 1000 rather than 10 | full text |
| **VecFlow, SIGMOD '26** (Xi et al., PACMMOD 3(4)) | GPU filtered-ANN system: label-centric index on CAGRA graphs, small / large-batch and single / multi-label paths; on an A100, ~5 M QPS at recall 0.9, up to 135× Filtered-DiskANN, reaches 0.99 where GPU baselines plateau near 0.8; evaluates on YFCC-10M among others | **yes** (A100) | standalone vector data management system, not an op inside a model graph | the closest GPU filtered system; candidate for C1 ("GPU filtered search exists outside the model graph"). | abstract, §1-2, dataset list |
| **CAGRA** (Ootomo et al., ICDE 2024) / **cuVS** filtering | GPU graph ANN; cuVS CAGRA takes a pre-filter **bitset** (filtered vectors never visited). cuVS docs: CAGRA filtering works well when only a small share is filtered out and brute force with pre-filtering is preferred for heavy filtering; branch 25.02 improved multi-CTA filtered search to filtering rates ≥ 90 % | yes | library / service (not a model op); SilverTorch excludes CAGRA for its top-k ≤ 1024 limit (§6.2.1) | the "standalone GPU library" side of the scope line | cuVS docs pages (search excerpts); CAGRA paper not read |
| **Garfield** (Li, Yu, Xu, Zhu, Gao; arXiv:2604.20121, 2026; venue not found) | GPU framework for multi-attribute *range*-filtered ANN: cell-partitioned graph index (GMG), out-of-core pipeline; 4.4× smaller index, 119.8× throughput over prior RFANNS | yes | standalone | range filters; we have equality clauses only | abstract only |

Positioning in one line each (facts, for the drafter): filtered-ANN benchmarks to
date are CPU-side library comparisons (Iff, Li, Big-ANN); GPU filtered search
exists as standalone systems (VecFlow, CAGRA / cuVS, Garfield); the two papers we
reproduce are the published instances of filtered retrieval as ops inside the
model graph, and none of the above evaluates them.

### 3.1 Citations, ready for BibTeX

All checked against arXiv and Crossref (2026-10-12) unless marked.

- Borisyuk, F., Song, Q., Zhou, M., Parameswaran, G., Arun, M., Popuri, S., Bingol,
  T., Pei, Z., Lee, K.-H., Zheng, L., Shao, Q., Naqvi, A., Zhou, S., Gupta, A.:
  LiNR: Model Based Neural Retrieval on GPUs at LinkedIn. In: CIKM '24, pp. 4366-4373.
  DOI 10.1145/3627673.3680091; arXiv:2407.13218.
- Xue, B., Wu, H., Chen, L., Yang, C., et al. (32 authors): SilverTorch: A Unified
  Model-based System to Democratize Large-Scale Recommendation on GPUs. In: SIGIR '26,
  pp. 2139-2149. DOI 10.1145/3805712.3809755; arXiv:2511.14881. (Full author list: the PDF's first
  page.)
- Iff, P., Brügger, P., Chrapek, M., Kochergin, D., Besta, M., Hoefler, T.:
  Benchmarking Filtered Approximate Nearest Neighbor Search Algorithms on
  Transformer-based Embedding Vectors. In: SIGIR '26, pp. 610-622. DOI
  10.1145/3805712.3809731; arXiv:2507.21989.
- Li, M., Yan, X., Lu, B., Zhang, Y., Cheng, J., Ma, C.: Attribute Filtering in
  Approximate Nearest Neighbor Search: An In-depth Experimental Study. Proc. ACM
  Manag. Data 3(6), 1-26 (SIGMOD '26). DOI 10.1145/3769763; arXiv:2508.16263.
  (A corrigendum exists: DOI 10.1145/3803528.)
- Xi, J., Mo, C., Karsin, B., Chirkin, A., Li, M., Zhang, M.: VecFlow: A
  High-Performance Vector Data Management System for Filtered-Search on GPUs. Proc.
  ACM Manag. Data 3(4), 1-27 (SIGMOD '26). DOI 10.1145/3749189; arXiv:2506.00812.
- Simhadri, H.V., Aumüller, M., Ingber, A., Douze, M., Williams, G., Dobson Manohar,
  M., Baranchuk, D., Liberty, E., Liu, F., Landrum, B., Karjikar, M., Dhulipala, L.,
  et al.: Results of the Big ANN: NeurIPS'23 competition. In: NeurIPS 38, Datasets
  and Benchmarks Track (2025). DOI 10.52202/085713-2303; arXiv:2409.17424. (Author
  list continues; copy from the arXiv page.)
- Ootomo, H., Naruse, A., Nolet, C., Wang, R., Feher, T., Wang, Y.: CAGRA: Highly
  Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs. In:
  ICDE 2024. DOI 10.1109/ICDE60146.2024.00323.
- Li, Z., Yu, H., Xu, Z., Zhu, Y., Gao, Y.: A GPU-Accelerated Framework for
  Multi-Attribute Range Filtered Approximate Nearest Neighbor Search.
  arXiv:2604.20121 (2026). Venue not found.
- Data: Wan, M., McAuley, J.: Item recommendation on monotonic behavior chains.
  RecSys '18; Wan, M., Misra, R., Nakashole, N., McAuley, J.: Fine-grained spoiler
  detection from large-scale review corpora. ACL '19 (the Goodreads README asks for
  both; **not re-checked here**). Jin, Q., et al.:
  MedCPT, arXiv:2307.00589 (**not re-checked**). Nussbaum, Z., et al.: Nomic Embed,
  arXiv:2402.01613 (**not re-checked**). Re-LAION: LAION blog post
  `laion.ai/blog/relaion-5b/` (`@misc{relaion}`).

## Open items

All C1 open items are closed or moved: the validation wording and the
provenance / datasets fixes are on `staging` (§1.6); the abstract's "pre- or
post-filtering" has a wording note under the abstract, and the
[outline](../outline.md) §1-§2 use LiNR's terms (masking vs gathering);
whether to cite stars or citation counts is the outline's open item 5.
Two facts stay weak and are used as such: the Goodreads encoder quality
(E1c rows not validated; the paper reports N, queries and filters, not
encoder quality). Three data citations (Wan RecSys '18 /
ACL '19, MedCPT, Nomic Embed) are to be re-checked
against the publishers' pages when the bibliography is built.
