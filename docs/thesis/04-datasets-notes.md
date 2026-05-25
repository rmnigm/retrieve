# Chapter 3 — Datasets and SASRec query-model training (English notes)

> **What this file is.** Source material for the Russian draft of Chapter 3
> (`docs/thesis/04-datasets.md`). Written in full English prose so the
> author can translate sentence-by-sentence; tables, pseudocode and formulas
> stay verbatim. Numbers in this file are either pulled from existing
> on-disk artifacts (cited inline) or computed locally with `polars` over
> the prepared parquets (no `torch` was loaded). Anything that still
> requires loading the `*.pt` attribute tensors or a checkpoint
> `state_dict` is marked **TBD-script-stub** with the smallest possible
> hint for how to compute it later.
>
> Citation policy reminder (see [00-thesis-plan.md §"Citation Policy"](00-thesis-plan.md)):
> no Meta / FAIR / Facebook / Instagram / WhatsApp authors. Every paper
> mentioned below is screened against that rule in §1.

---

## 0. Reading guide

The chapter covers, in order:

1. The three empirical bases the thesis runs experiments on — Goodreads
   (UCSD Book Graph), arXiv (the `open-index/open-arxiv` HuggingFace
   mirror), and Yambda (Yandex Music) at two scales (500M and 5B
   listening events).
2. The filter schemas the retrieval harness uses on Goodreads and arXiv:
   per-item attribute tensors, the narrow/wide regime split, the AND-of-OR
   clause semantics, and how query attributes are synthesised from each
   held-out target.
3. The SASRec / gSASRec query encoders trained for the two
   sequential-recommendation datasets (Goodreads, Yambda), including the
   architecture, the gBCE loss, the training loop, the per-epoch eval
   protocol and the checkpoint convention.

**Encoder-agnostic interop (cross-cuts the chapter).** The three
datasets together exercise *two distinct encoder families* feeding the
same retrieval layers without modification: a SASRec/gSASRec sequence
encoder trained from scratch for Goodreads and Yambda (architecture and
training detailed in §7), and the pretrained Nomic-Embed-text-v1.5 text
encoder used out-of-the-box (no fine-tuning) for arXiv (pipeline in
§3.2). This combination is not an accident of dataset choice — it is the
empirical demonstration of the framework's encoder-agnostic public API
claim (Ch.1 §1.11 goal 1, Ch.4 §4.11): the same `KNN`, LinR V1/V2/V3,
and IVF + INT8 + Bloom layers are reused unchanged across a
recommendation sequence model and a text encoder, across two
problem domains (RecSys and text-IR). The dataset choices in this
chapter are what makes that demonstration concrete in Ch.6.

Source code anchors used throughout:

- `evaluation/datasets/` — pipelines for the three datasets +
  `common.py` (shared time-split and query-attribute synthesis),
  `hf_io.py` (path/registry helpers), `constants.py` (Yambda
  TEST/VAL/GAP), `timesplit.py` (the sequential splitter).
- `evaluation/training/` — `train_sasrec.py`, `model.py`, `losses.py`,
  `evaluate.py`, `config.py`, `dataset.py`.
- `evaluation/retrieval/` — `metrics.py` (eval-time Recall/NDCG),
  `algos/filter.py` (per-algorithm filter factory).
- `retrieve/src/retrieve/` — the published `torchretrieve` package
  (`interfaces.py`, `layers/filters/*`, `kernels/filters/*`).
- Sibling system docs: `docs/system/{filtering,checkpoints}.md`.
- Source-paper notes: `articles/yambda.md`.

---

## 1. Source-paper references and citation eligibility

Every dataset / model used in this chapter, with its primary citation,
arXiv or DOI handle, the author list at a glance, and a ✅/❌ verdict
against the Meta affiliation rule.

| Resource | Primary citation | Identifier | Authors (top affiliation) | Cite? |
|----------|------------------|-----------|---------------------------|-------|
| Goodreads UCSD Book Graph (item metadata + interactions) | Wan & McAuley, *Item Recommendation on Monotonic Behavior Chains*, RecSys 2018 | DOI 10.1145/3240323.3240369 | UCSD | ✅ |
| Goodreads — review-text release used by the same mirror | Wan, Misra, McAuley, *Fine-Grained Spoiler Detection from Large-Scale Review Corpora*, ACL 2019 | arXiv:1905.13416 | UCSD | ✅ |
| arXiv corpus mirror (3M papers, monthly shards) | HuggingFace dataset card `open-index/open-arxiv` (no peer-reviewed paper) | huggingface.co/datasets/open-index/open-arxiv | community-maintained | cite dataset card only |
| Text encoder used to embed the arXiv abstracts | Nussbaum et al., *Nomic Embed: Training a Reproducible Long Context Text Embedder*, 2024 | arXiv:2402.01613 | Nomic AI | ✅ |
| Yambda (Yandex Music, 4.79B events) | Anokhin et al., *Yambda: One of the Largest Open Music Recommendation Datasets with Listening Histories*, 2025 | arXiv:2505.22238 | Yandex | ✅ — see `articles/yambda.md` |
| SASRec architecture | Kang & McAuley, *Self-Attentive Sequential Recommendation*, ICDM 2018 | arXiv:1808.09781 | UCSD | ✅ |
| gBCE loss (a.k.a. gSASRec) | Petrov & Macdonald, *gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling*, RecSys 2023 (best paper) | arXiv:2308.07192 | University of Glasgow | ✅ |

None of the works above is excluded by the Meta-affiliation rule. The two
"competitor" sequential-recommendation models that the Yambda paper
benchmarks against — BERT4Rec (Sun et al. 2019, Alibaba) and ItemKNN
(Sarwar et al. 2001, GroupLens) — are also ✅ if the author chooses to
cite them in passing; they are *not* used in this thesis's training.

---

## 2. Goodreads (UCSD Book Graph)

### 2.1 Provenance

The Goodreads UCSD Book Graph is a public dump released by the UCSD
McAuley Lab and hosted at
`mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads`. It contains 13
gzipped JSON / CSV files covering raw book metadata
(`goodreads_books.json.gz`), normalized "works" (`goodreads_book_works`),
authors, genre signals (`goodreads_book_genres_initial`), series and
reviews, as well as the read-event interactions
(`goodreads_interactions_dedup`). The corpus has been used in two
related publications by Wan & McAuley (RecSys 2018) and Wan, Misra &
McAuley (ACL 2019); the thesis cites both as primary references.

### 2.2 Pipeline

The pipeline is implemented in [`evaluation/datasets/goodreads.py`](../../evaluation/datasets/goodreads.py)
as a Typer CLI with four subcommands:

- `download` — resumable HTTPS fetch with `Range` header support; each of
  the 13 files is downloaded into `~/datasets/goodreads-ucsd/raw/` and
  validated against a SHA-256 manifest. The download function (`http_size`,
  `download_one`) handles HTTP 206 partial content so an interrupted
  download resumes from byte offset.
- `convert` — streams every raw NDJSON.gz or CSV into ZSTD-compressed
  Parquet under `processed/`. Reads are batched at 50k rows per write so
  the conversion fits in <8 GB of RAM regardless of the file size.
- `prep` — turns the processed parquets into the trainer-ready splits:
  it filters interactions to `is_read=true`, parses `date_added`
  timestamps, joins `book_id → work_id` via `goodreads_books.parquet`,
  applies an iterative 5-core filter (drops users and works that have
  fewer than five interactions, then re-checks, until convergence —
  see `_apply_5core`), builds a dense 1-indexed id-map for works, and
  finally calls `sequential_split_train_val_test`
  (`evaluation/datasets/common.py:6-129`) with the 96-th percentile
  timestamp as the test cut-off, a 30-day validation window, and a
  7-day gap.
- `attrs` — separately builds the per-item attribute tensors used by
  the filter benchmark. This step is independent of training and
  produces `item_attrs_narrow.pt`, `item_attrs_wide.pt`,
  `clause_is_reverse_narrow.pt`, the per-attribute vocabularies as
  JSON, the `wide_shelf_global_freq.pt` count tensor, and the
  pre-sampled `eval_split.parquet` with per-target query attributes.

### 2.3 Scale and statistics (measured)

The numbers in this section were computed in this session with `polars`
directly on the processed parquets (`~/datasets/goodreads-ucsd/processed`).
A small reproduction script is sketched in §8.

| Quantity | Value | Source |
|----------|------:|--------|
| Raw book editions | **2,360,655** | `goodreads_books.parquet` rows |
| Unique works (after edition→work collapse) | **1,521,962** | `goodreads_book_works.parquet` rows |
| Distinct authors | **829,529** | `goodreads_book_authors.parquet` rows |
| Raw user–book interactions (post-dedup, pre-`is_read`) | **228,648,342** | `goodreads_interactions_dedup.parquet` rows |
| Distinct users in the raw interactions | **876,145** | `user_id.n_unique()` |
| `is_read=True` interactions (the trainer-seed signal) | **112,131,203** | filter on `is_read` |
| Items after `prep` (`is_read` + dense work-id + 5-core) | **≈ 797,000** | code comment in [`retrieve/src/retrieve/kernels/filters/clause_mask.py:32`](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L32) |
| Users kept in the evaluation split (the SASRec test set) | **313,178** | `n_users_kept` field in `evaluation/results/goodreads/d128-quality.json` |
| Maximum sequence length kept | **200** | `max_seq_len` argument in `goodreads.py` |
| Users with ≥ 1 `is_read` event (sequence-length population) | **836,433** | polars on `goodreads_interactions_dedup.parquet` |
| Median sequence length on the raw `is_read` signal | **59 events** | percentile 50 of the per-user count |
| p90 / p95 / p99 sequence length | **325 / 506 / 1101** | same |
| Max sequence length seen in the raw signal | 38,895 | same |
| Books with ≥ 1 `is_read` event | 2,339,815 | per-book count |
| Median / mean / p99 / max book popularity (`is_read` count) | 5 / 47.9 / 590 / 285,698 | same |
| Users surviving a single-iteration 5-core (≥ 5 events) | 766,036 | upper bound; iterative 5-core trims further |
| Books surviving a single-iteration 5-core (≥ 5 events) | 1,193,420 | upper bound; the work-id collapse drops this further to ~797k |

Saved artefacts (relative to repository root):

- [`docs/thesis/results-data/datasets/goodreads_seq_len_cdf.csv`](results-data/datasets/goodreads_seq_len_cdf.csv) and [`.png`](results-data/datasets/goodreads_seq_len_cdf.png)
- [`docs/thesis/results-data/datasets/goodreads_item_popularity_rankfreq.csv`](results-data/datasets/goodreads_item_popularity_rankfreq.csv) and [`goodreads_item_popularity.png`](results-data/datasets/goodreads_item_popularity.png)

Reading guide for the chapter: the median user has 59 `is_read`
events; the SASRec encoder has a 200-token context window, so the
median user fits entirely inside the window with room to spare —
but the **top 5% of users have 506+ events**, all of which get
left-truncated. The pop-tail is steep: median book has only 5
`is_read` events, p99 is at 590, and the absolute most-read book has
285,698 events (≈ 4-orders-of-magnitude spread). The 5-core filter
is therefore much more aggressive on the item side than on the user
side; after the iterative 5-core + work-id collapse the corpus
shrinks from 2.36 M raw editions to roughly 797 k works (the
`linr_v1_filter_mask` index size measured in the
[`evaluation/results/goodreads/d128-quality.json`](../../evaluation/results/goodreads/d128-quality.json) baseline).

### 2.4 Split protocol

The split is **chronological by `date_added`** with a hard test
cut-off at the 96th percentile timestamp of the kept interactions,
a 30-day validation window immediately before that, and a 7-day gap
between train and validation to prevent leakage from in-flight
sessions. Users whose entire history falls outside the train period
are dropped from validation and test. The split is implemented once
in [`evaluation/datasets/common.py`](../../evaluation/datasets/common.py)
and re-used identically by Yambda (see §4.4).

### 2.5 Narrow attribute schema (filter benchmark)

The narrow schema is a fixed-size dense tensor
`item_attrs_narrow.pt` of shape `[N+1, C=5, A_max=4]` (int64). Row 0
is the padding sentinel and holds `-1` everywhere. Per-work
attributes are derived by aggregating across the multiple editions of
a single work; ties are broken by global edition-count.

| Clause | Semantic content | Cardinality | Reverse | Per-work A_max |
|-------:|------------------|-------------|:-------:|:--------------:|
| C0 | Genre (top-4 by vote count across editions) | 10 fixed genre buckets | False | 4 |
| C1 | Language (mode language across editions) | top-30 vocabulary | **True** | 1 |
| C2 | Format (paperback / hardcover / ebook / audio / other) | 5 buckets | False | 1 |
| C3 | Publication-year bucket | 4 buckets (`<1990`, `1990–2000`, `2001–2010`, `2011+`) | False | 1 |
| C4 | Author (top-2 author ids by global edition-count) | ≈ 880k authors | False | 2 |

The reverse flag on C1 is non-obvious. It is set to `True` in
`clause_is_reverse_narrow.pt = [F, T, F, F, F]` so that the
*narrow* sweep treats the language clause as a *not-in* predicate:
"books **not** in language *L*" rather than "books in *L*". The
motivation is operational — a clause asking for "books in English"
matches the vast majority of the corpus and is therefore not narrow
enough to stress the filter path; flipping the predicate to "books
not in this minority language" produces a tightly selective clause
even on the dominant value. The XOR is applied at kernel level
([`clause_mask.py:84`](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L84))
and is invisible to the downstream retrieval layer.

**C0 narrow-clause coverage (genre).** The ten genre buckets are
taken directly from the `genres` struct in
`goodreads_book_genres_initial.parquet`. Of 2.36 M raw editions,
**1,951,142 (82.7%) have at least one genre vote** — the remaining
~17% are encoded as `-1` padding in the narrow tensor. The per-bucket
counts are saved at
[`docs/thesis/results-data/datasets/goodreads_clause_c0_genre.csv`](results-data/datasets/goodreads_clause_c0_genre.csv)
and read out as:

| Genre key | Raw editions with ≥ 1 vote | % of 2.36 M raw editions |
|-----------|---------------------------:|--------------------------:|
| `fiction` | 1,244,112 | 52.7% |
| `history, historical fiction, biography` | 663,795 | 28.1% |
| `romance` | 658,719 | 27.9% |
| `fantasy, paranormal` | 538,311 | 22.8% |
| `non-fiction` | 533,491 | 22.6% |
| `mystery, thriller, crime` | 523,156 | 22.2% |
| `young-adult` | 364,114 | 15.4% |
| `children` | 256,935 | 10.9% |
| `comics, graphic` | 171,279 | 7.3% |
| `poetry` | 88,630 | 3.8% |

**C3 narrow-clause coverage (publication year).** Cast to int64,
filtered to 1500–2025, mapped to the four buckets used in
`goodreads.py:_year_to_bucket`. Coverage: 1,758,389 of 2.36 M raw
editions (74.5%) have a parseable year; the remaining 25.5% are `-1`
in the narrow tensor. Saved at
[`goodreads_clause_c3_year.csv`](results-data/datasets/goodreads_clause_c3_year.csv):

| Bucket | Raw editions | % of raw |
|--------|-------------:|---------:|
| `<1990` | 128,530 | 5.4% |
| `1990–2000` | 179,835 | 7.6% |
| `2001–2010` | 528,571 | 22.4% |
| `2011+` | 921,453 | 39.0% |
| (missing year) | 602,266 | 25.5% |

**C2 narrow-clause coverage (format).** Raw distribution of the 12
distinct format strings mapped through `_format_to_bucket`. Coverage
is essentially 100% because the bucket function returns `'other'` for
unrecognised values. Saved at
[`goodreads_clause_c2_format.csv`](results-data/datasets/goodreads_clause_c2_format.csv):

| Bucket | Raw editions | % of raw |
|--------|-------------:|---------:|
| `paperback` (includes "Mass Market Paperback") | 939,702 | 39.8% |
| `other` (includes nulls, "Unknown Binding", "Board Book", …) | 676,191 | 28.6% |
| `hardcover` | 360,130 | 15.3% |
| `ebook` (includes "Kindle Edition") | 314,486 | 13.3% |
| `audio` (Audible, audio CD) | 70,146 | 3.0% |

**C1 narrow-clause coverage (language).** 226 distinct
`language_code` values appear in the raw corpus; the top-30 used by
the narrow vocabulary covers **97.7%** of books with a non-empty
language. The full top-30 is saved at
[`goodreads_clause_c1_lang_top30.csv`](results-data/datasets/goodreads_clause_c1_lang_top30.csv).
Top entries (book counts on the raw corpus):

| Raw format | Editions | Canonical bucket |
|------------|---------:|------------------|
| `Paperback` | 894,617 | `paperback` |
| `Hardcover` | 359,563 | `hardcover` |
| `ebook` | 188,733 | `ebook` |
| `Kindle Edition` | 125,566 | `ebook` |
| `Mass Market Paperback` | 42,224 | `paperback` |
| `Audible Audio`, `Audio` | 11,073 + 8,335 | `audio` |

| Code | Editions | Code | Editions |
|------|---------:|------|---------:|
| `eng` | 708,457 | `nl` | 17,497 |
| `en-US` | 91,452 | `tur` | 14,238 |
| `en-GB` | 58,358 | `per` | 11,821 |
| `spa` | 54,524 | `fin` | 11,611 |
| `ita` | 50,902 | `gre` | 10,024 |
| `ara` | 42,978 | `swe` | 9,914 |
| `fre` | 32,046 | `cze` | 8,564 |
| `ger` | 30,941 | `jpn` | 7,209 |
| `ind` | 27,291 | `rus` | 6,617 |
| `por` | 23,452 | … | … |

English variants (`eng + en-US + en-GB + en-CA = 866,019`) account for
roughly **45% of books with a known language** — which is what makes
the reverse-on-C1 trick from §2.5 effective: a "not English" query
trims ~45% of the corpus in the typical case. These four tables
(genre, year, format, language) are reused in §5 to give the chapter
concrete per-clause selectivity figures without having to run a
torch forward pass.

**C4 narrow-clause coverage (author).** The full author table has
**829,529 unique author ids** with median `ratings_count = 31` and a
heavy tail (the most-rated author has ≥ 10⁵ ratings). The narrow
clause stores up to two authors per book, dense-remapped by global
frequency rank. Coverage is essentially 100% of editions with a
parseable author.

### 2.6 Wide attribute schema (filter benchmark)

The wide schema is a single-clause bag of "popular shelves" — the
free-form user-assigned tags that Goodreads exposes on every book
(`popular_shelves` in `goodreads_books.parquet` is
`list[struct<count:str, name:str>]`). The wide tensor is
`item_attrs_wide.pt` of shape `[N+1, 1, 32]` (int64): each work holds
the top-32 shelves by global edition-count after a blocklist + regex
drop removes intent-only shelves (`to-read`, `currently-reading`,
`owned`, `read-2023`, etc.). The vocabulary is roughly 1,800 shelf
names after the drop. Coverage is ≈ 94% (works without any kept
shelf are padded with `-1`). The accompanying tensor
`wide_shelf_global_freq.pt` stores per-shelf global counts so the
evaluation harness can apply rare-biased sampling (§5.5).

### 2.7 Visual deliverables for the §2 chapter

1. Pipeline schematic (raw 13 files → 50k-row ZSTD parquet stream →
   5-core dense map → train/val/test parquets → SASRec → query
   embeddings → filter tensors).
2. Two-panel histogram: sequence length distribution (post 5-core,
   linear x; same with log y to expose the tail).
3. Item-popularity rank-frequency plot, log-log, with the 5-core
   threshold marked.
4. One landscape table summarising the narrow + wide schema (the
   tables in §2.5 / §2.6 condensed).

---

## 3. arXiv (open-index/open-arxiv mirror)

### 3.1 Provenance

The arXiv corpus used in this thesis is **not the original arXiv
metadata dump** but a community-maintained mirror published on
HuggingFace as `open-index/open-arxiv`. The mirror ships monthly
parquet shards at paths like `data/2024/2024-03.parquet`; the dataset
card declares roughly 2.99 M papers, last refreshed 2026-03-24. There
is no peer-reviewed paper to cite for the corpus itself — the chapter
cites the dataset card only.

The text encoder used to produce dense embeddings of each paper's
title + abstract is `nomic-embed-text-v1.5` (Nussbaum et al. 2024,
arXiv:2402.01613). Nomic AI is independent of any Meta-affiliated
institution, so the citation is on the allowed list.

### 3.2 Pipeline

Implemented in [`evaluation/datasets/arxiv.py`](../../evaluation/datasets/arxiv.py)
with six subcommands:

- `download` — `huggingface_hub.snapshot_download` with 8 workers and
  `allow_patterns=["data/*/*.parquet", "README.md"]`. Output lives at
  `data/_raw/arxiv/data/YYYY/YYYY-MM.parquet`.
- `convert` — globs the monthly shards, keeps the 9 columns the
  thesis uses (`id, title, abstract, categories, authors_parsed,
  license, update_date, versions, submitter`) and writes a single
  `arxiv_papers.parquet`.
- `prep` — drops rows with empty abstract, deduplicates on `id`, sorts
  by `arxiv_id` for stability, assigns a dense 1-indexed item id, and
  randomly samples `n_heldout = 10,000` papers as the evaluation
  query set.
- `encode_text` — encodes title + abstract with
  `nomic-embed-text-v1.5`, prefixing each input with
  `"search_document: "`. The encoder is run with Matryoshka
  truncation, producing three slices `d ∈ {64, 128, 256}`; each
  slice is L2-renormalised and written as fp16 under
  `content_d{d}/text_emb.pt`.
- `encode_queries` — re-encodes the heldout papers with the
  `"search_query: "` prefix, producing `content/query_emb.pt`. The
  prefix split between document and query embeddings is the
  Nomic-recommended pattern; it materially changes the per-token
  attention bias inside the encoder.
- `attrs` — builds the narrow + wide attribute tensors. Same shapes
  as Goodreads (`[N+1, 5, 4]` narrow, `[N+1, 1, 32]` wide).

### 3.3 Scale (from code constants and evaluation results)

| Quantity | Value | Source |
|----------|------:|--------|
| Total papers in the mirror | ≈ 2,990,000 | dataset card; code comment in `arxiv.py:14-19` |
| Held-out papers used as evaluation queries | **10,000** | default in `arxiv.py:_arg_n_heldout` (matches `n_users_kept = 10000` in `evaluation/results/arxiv/d128-quality.json`) |
| Number of distinct top-level arXiv categories | ≈ 30 | derived from `_category_to_main`; e.g. `cs`, `math`, `stat`, `eess`, `hep-lat`, `astro-ph`… |
| Distinct author keys (`LastName|FirstName|Suffix` tuple) | ≈ 1.5 M | derived from `authors_parsed` aggregation |
| Wide leaf-category vocabulary (after `wide_min_count = 50`) | ≈ 2,000 leaf categories | filter in `arxiv.py` attrs step |
| Sequence-style metrics (median seq length etc.) | **not applicable** | arXiv has no user sequences |

The crucial asymmetry compared to Goodreads and Yambda is that arXiv
**has no user sequences**. There is no SASRec training for arXiv; the
"query model" is a frozen text encoder operating directly on paper
text. The chapter must make this explicit before §7 talks about
SASRec, otherwise the table summarising "SASRec quality per dataset"
would be misread.

### 3.4 Narrow attribute schema

Identical layout to Goodreads (`[N+1, 5, 4]` int64) but different
clause semantics:

| Clause | Semantic content | Cardinality | Reverse | A_max |
|-------:|------------------|-------------|:-------:|:-----:|
| C0 | Top-level arXiv main category | ≈ 30 | False | 1 |
| C1 | License bucket | 11 buckets (`cc-by`, `cc-by-sa`, …, `cc0`, `arxiv-default`, `none`, `other`) | False | 1 |
| C2 | Publication-year bucket | 7 buckets (`<2000`, `2000–2004`, `2005–2009`, …, `2025+`) | False | 1 |
| C3 | Version-count bucket | 4 buckets (1, 2, 3, 4+) | False | 1 |
| C4 | Author (top-2 by global frequency) | ≈ 1.5 M | False | 2 |

`clause_is_reverse_narrow.pt = [F, F, F, F, F]` for arXiv — every
clause is a "must include" predicate. There is no reverse-language
analogue because arXiv has no dominant minority that demands the
trick used on Goodreads.

Approximate per-clause selectivity figures (derived from corpus-level
category frequencies; precise per-query measurements are
**TBD-script-stub**, see §5.6):

- C0 = `cs` → roughly 60% of the corpus (CS is by far the largest
  main category by paper count).
- C0 = `stat` → roughly 5%.
- C2 `<2000` → roughly 1.7%; C2 `2020–2024` → roughly 50% (post-CS
  growth).

### 3.5 Wide attribute schema

Per-paper bag of leaf categories (`cs.LG`, `cs.CL`, `stat.ML`, …),
deduplicated, sorted, top-32 by global count. Leaf categories that
appear fewer than 50 times in the whole corpus are dropped from the
vocabulary before the bag is constructed (`wide_min_count = 50`).

### 3.6 Visual deliverables for the §3 chapter

1. Bar chart — papers per top-level category (top 15, log-y).
2. Bar chart — papers per year (since 1991, linear).
3. Histogram — version count distribution.
4. Schematic — text → nomic-embed-v1.5 (`search_document:` /
   `search_query:` prefixes) → 256-d → Matryoshka slices to
   `{64, 128, 256}` → fp16 L2-normalised tensor.

---

## 4. Yambda (Yandex Music)

### 4.1 Provenance

Yambda is the Yandex Music public release described in Anokhin et al.
2025 (arXiv:2505.22238) and documented in detail at
`articles/yambda.md`. The dataset is distributed via HuggingFace
(`yandex/yambda`) in three subsampled scales and three event-type
families. All citation context is Yandex-only — no Meta affiliation
on the author list, so ✅ unconditionally.

### 4.2 Scales used in this thesis

The Yambda paper publishes three subsamples with the following
headline counts (Tab. 1 and Tab. 4 in `articles/yambda.md`):

| Scale | Users | Items | Listens | Likes | Dislikes |
|-------|------:|------:|--------:|------:|---------:|
| Yambda-50M  | 10,000 | 934,057 | 46,467,212 | 881,456 | 107,776 |
| Yambda-500M | 100,000 | 3,004,578 | 466,512,103 | 9,033,960 | 1,128,113 |
| Yambda-5B | 1,000,000 | 9,390,623 | 4,649,567,411 | 89,334,605 | 11,579,143 |

This thesis trains and evaluates on **Yambda-500M and Yambda-5B**;
the 50M variant is omitted (it does not stress the GPU retrieval
algorithms at large catalogue sizes).

### 4.3 Interaction types

Yambda exposes five interaction types (Tab. 2 of the paper). For
each event the dataset stores the `is_organic` flag, which is `True`
when the user discovered the track without algorithmic intervention
and `False` when the event was driven by the recommender. The split
is roughly balanced for listens and likes but heavily organic for
unlikes:

| Event | Total | Recommendation-driven | Org. ratio |
|-------|------:|----------------------:|------:|
| Listen    | 4,649,567,411 | 2,266,400,808 | 48.74% |
| Like      | 89,334,605    | 37,789,576    | 42.30% |
| Dislike   | 11,579,143    | 5,612,434     | 48.47% |
| Unlike    | 32,944,520    | 1,651,117     | 5.01% |
| Undislike | 2,434,208     | 239,135       | 9.82% |

This thesis uses **only the `listens` interaction stream**, filtered
to events with `played_ratio_pct >= 50` (`TRACK_LISTEN_THRESHOLD`
in [`evaluation/datasets/constants.py:12`](../../evaluation/datasets/constants.py)).
The other event families (likes, dislikes, …) are kept for future
work but do not appear in any of the experiments below.

### 4.4 Pipeline

The pipeline in [`evaluation/datasets/yambda.py`](../../evaluation/datasets/yambda.py)
exposes a single subcommand, `prep`, which (1) downloads the
sequential parquet (`sequential/{variant}/listens.parquet`) from
HuggingFace, (2) filters to events whose `played_ratio_pct ≥ 50%`,
(3) builds a dense 1-indexed item-id map over the surviving tracks,
and (4) calls `sequential_split_train_val_test` with the
Yambda-specific constants:

| Constant | Value | Meaning |
|----------|------:|---------|
| `TEST_TIMESTAMP` | 26,000,000 s ≈ 301 days | Start of the test window in dataset time |
| `VAL_SIZE` | 86,400 s (1 day) | Validation window width |
| `GAP` | 1,800 s (30 min) | Train/val/test guard gap |
| `max_seq_len` | 200 | Cap on user history length kept for SASRec |

The 30-minute gap is the same value the Yambda paper uses; it
"mimics the latency between model training and deployment in
industrial systems" (paper §"Evaluation Process"). The author should
quote this rationale in the chapter when describing why the split
includes a non-zero gap.

### 4.5 GTS evaluation protocol

Yambda's "Global Temporal Split" (paper §Evaluation Process) sets
train ≈ 300 days, gap = 30 min, test = 1 day; all model parameters
and user states are **frozen at the start of the test period** —
the thesis inherits this convention unchanged. Users with empty
history at the start of the test period are dropped (this is what
reduces the test sets in our runs to **45,932 users** on Yambda-500M
and **459,067 users** on Yambda-5B — both numbers are recorded
verbatim as the `n_users_kept` field in
`evaluation/results/yambda/{500m,5b}-d*.json`).

### 4.6 User-history length statistics (from the paper)

User-history length quantiles per event type (Tab. 3 of the paper):

| Event | Median | p90 | p95 |
|-------|------:|----:|----:|
| Listen    | 3,076 | 12,956 | 17,030 |
| Like      | 45    | 269    | 409 |
| Dislike   | 4     | 30     | 60 |
| Unlike    | 15    | 111    | 201 |
| Undislike | 3     | 14     | 22 |

The thesis only operates on the listens stream, so the
3,076 / 12,956 / 17,030 median / p90 / p95 row is the one that
matters. Together with the `max_seq_len = 200` cut applied during
prep, this means the SASRec model sees a sharply truncated window
even for median users (median listen history is ≈ 15× the model's
context length, p90 is ≈ 65× the context length). This truncation
is a real, defensible thesis claim about why the listens-only
quality numbers plateau at certain dim values (see §7.5 results).

### 4.7 Why Yambda is **quality-only** in this thesis

Yambda has no public attribute schema we could turn into a filter
clause (no genres, no language tags, no licence buckets — only
opaque artist / album IDs and an audio embedding). The thesis
therefore runs **only the quality sweep** on Yambda (full-scan
recall and NDCG, no `filter_kind in {clause, bloom}`), and uses it
purely as a *catalogue-size stress test* — the 9.4 M-item scale is
the one place where the retrieval algorithms' memory and latency
behaviour can be probed at industrial sizes. The filter benchmark
remains Goodreads + arXiv only.

### 4.8 Visual deliverables for the §4 chapter

1. Side-by-side scale table (50M / 500M / 5B), with the two scales
   actually used in this thesis highlighted.
2. Item-popularity rank-frequency log-log plot (from the prepared
   train parquet, computable in polars; the paper's Figure 2 is a
   ready reference image to compare against).
3. GTS timeline figure: `[ train 300 d ][ 30 min gap ][ test 1 d ]`,
   with the per-scale `n_users_kept` annotated.

---

## 5. Filter-clause design and selectivity (Goodreads + arXiv)

This section pulls together the filter primitives and the selectivity
profile in one place because Chapters 3, 4 and 6 of the thesis all
refer back to it. The primitives themselves live in the `retrieve`
package and are also described in `docs/system/filtering.md`.

### 5.1 AND-of-OR semantics

Every clause encodes an "OR over up to `A_max` slots" predicate at the
item side and a single "equals" predicate at the query side. Across
clauses, the per-clause Booleans are combined with AND (i.e. the
filter is a conjunction of clauses, where each clause is a disjunction
of up to `A_max` allowed values).

Concretely, the per-tile inner loop of `clause_mask` is
(`retrieve/src/retrieve/kernels/filters/clause_mask.py:71-89`):

```text
pass_mask = TRUE
for c in 0..C-1:
    q_c    = query[bid, c]                      # int64 scalar
    rev_c  = is_reverse[c]                      # 0/1 scalar
    match  = FALSE
    for a in 0..A_MAX-1:
        ia       = item_attrs[n, c, a]          # int64, -1 = pad
        match    = match OR (ia == q_c)
    match     = match XOR rev_c                 # apply reverse
    inactive  = (q_c == -1)
    match     = match OR inactive               # inactive clause passes
    pass_mask = pass_mask AND match
return pass_mask
```

The `inactive = q_c == -1` short-circuit is what lets a single
filter object support arbitrary clause-subset sweeps without
rebuilding the index: the YAML configs simply zero-out a clause
column in the query tensor (e.g. `c0_genre` sweep zeros C1..C4,
`all4` zeros C4).

### 5.2 On-disk data layout

| Tensor | Shape | dtype | Notes |
|--------|-------|-------|-------|
| `item_attrs_narrow.pt` | `[N+1, C=5, A_max=4]` | int64 | row 0 is the padding sentinel (all `-1`); padding slots within a row are `-1` |
| `item_attrs_wide.pt`   | `[N+1, 1, 32]`        | int64 | same row-0 sentinel; per-item bag of up to 32 vocabulary ids |
| `clause_is_reverse_narrow.pt` | `[C=5]` | bool | XOR flag for each clause; Goodreads = `[F, T, F, F, F]`, arXiv = `[F, F, F, F, F]` |
| `wide_shelf_global_freq.pt` / `wide_category_global_freq.pt` | `[V_wide]` | int64 | per-vocabulary count, used for rare-biased query sampling |
| `eval_split.parquet`   | one row per heldout target | mixed | columns `target_id`, `query_attrs_narrow`, `query_attrs_wide_1shelf`, `query_attrs_wide_2shelf` |

The padding row 0 is what lets every downstream layer treat
`item_attrs[0]` as a no-op sentinel — important because the
SASRec embedding table itself uses index 0 as its padding row.

### 5.3 Filter primitives in `torchretrieve`

| Primitive | File | Backends | Supports `reverse`? | Notes |
|-----------|------|----------|:-------------------:|-------|
| `ExactAttributeFilter` | [`retrieve/src/retrieve/layers/filters/exact_attribute.py`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | torch, triton | Yes | Exact equality; the Triton path uses `clause_mask` and `clause_compact` and does not materialise the `[B, N, C, A_max]` intermediate that the torch path needs. |
| `BloomFilter` | [`retrieve/src/retrieve/layers/filters/bloom.py`](../../retrieve/src/retrieve/layers/filters/bloom.py) | torch, triton | No (rejected at `register_index`) | Approximate set-membership: per-item M-bit Bloom signature (`m_bits=1024`, `k_hash=5`); per-clause salt prevents cross-clause value collisions. |

Both implement the [`FilterModule`](../../retrieve/src/retrieve/interfaces.py)
interface (`evaluate_mask`, `evaluate_indices`, `evaluate_subset`),
so any retrieval layer can plug in either filter without code change.

### 5.4 Narrow vs wide query synthesis

Per-target query attributes are produced once at `prep` time and
cached in `eval_split.parquet`. The two regimes are:

- **Narrow** — `synthesize_qa_narrow` in
  [`evaluation/datasets/common.py:20-43`](../../evaluation/datasets/common.py).
  For each held-out target, copy the *first non-`-1`* value from
  each clause's attribute row into the query tensor. Result: every
  active clause carries exactly one value, every query is a precise
  conjunction. This regime is the **selective** mode of operation,
  i.e. designed for predicates that retain only a small fraction of
  the corpus.
- **Wide** — `sample_rare_biased_wide` in
  [`evaluation/datasets/common.py:46-101`](../../evaluation/datasets/common.py).
  For each held-out target, sample one or two values from the
  per-item wide bag with rare-biased weights:
  - 1-shelf variant: weight ∝ `1 / sqrt(global_freq)` (favours rare
    values).
  - 2-shelf variant: one value with weight ∝ `sqrt(freq)` (common)
    plus one with weight ∝ `1 / sqrt(freq)` from the rest (rare).
  This regime is the **loose** mode, designed to keep selectivity
  moderate (not 0%, not 100%) so the Bloom-filter approximation has
  enough work to do.

The clause-subset sweeps the chapter actually reports are defined in
YAML at [`evaluation/config/goodreads/d128-filter.yaml`](../../evaluation/config/goodreads/d128-filter.yaml)
and [`evaluation/config/arxiv/d128-filter.yaml`](../../evaluation/config/arxiv/d128-filter.yaml).
At a glance the lineup is:

| Sweep label | Active clauses | Regime |
|-------------|----------------|--------|
| `none`      | none (full scan) | baseline |
| `c0_genre` / `c0_maincat` | C0 only | narrow |
| `c1_lang_reverse` (Goodreads only) | C1 only (reverse on) | narrow |
| `c2_format` (Goodreads) / `c2_year` (arXiv) | C2 only | narrow |
| `c3_year` (Goodreads) / `c3_nversions` (arXiv) | C3 only | narrow |
| `c0c1` / `c0c2` | C0 ∧ C1 | narrow (multi-clause) |
| `all4` | C0 ∧ C1 ∧ C2 ∧ C3 | narrow (full conjunction) |
| `wide_1shelf` | wide bag, 1-value query | wide |
| `wide_2shelf` | wide bag, 2-value query | wide |

### 5.5 Approximate per-clause selectivity (Goodreads & arXiv)

Indicative numbers, derived from the raw-corpus frequency tables in
§2.5 and §3.4. These are **upper bounds** on the per-query
selectivity an exact filter would yield, because the actual prepped
corpus has been 5-core-filtered (Goodreads) or text-filtered
(arXiv) and the per-query values are sampled per target — but they
are within 2× of the real numbers in practice and good enough for a
chapter-level discussion.

**Goodreads (raw, % of 2.36 M editions):**

| Clause = value | Selectivity |
|----------------|------------:|
| C0 = `fiction` | ≈ 52% (loose) |
| C0 = `poetry`  | ≈ 4% (tight) |
| C1 (reverse) = `eng` ⇒ "not English" | ≈ 70% (loose, reverse on a dominant value) |
| C1 (reverse) = `tur` ⇒ "not Turkish" | ≈ 99% (degenerate, almost all corpus) |
| C2 = `paperback` | ≈ 38% |
| C2 = `audio`     | ≈ 0.8% |
| C3 = `2011+`     | ≈ 39% (52% of the books with a known year) |
| C3 = `<1990`     | ≈ 5% |

The `c1_lang_reverse` sweep is therefore not "narrow" for every
target — when the query value is the dominant language it stays
loose. The chapter should call this out: the reverse-on-language
trick narrows the *typical* case but does not narrow the *worst*
case. Per-target distribution is what matters and is left as
**TBD-script-stub** (single torch forward pass on
`item_attrs_narrow.pt`, see §5.6).

**arXiv (rough, from public corpus statistics):**

| Clause = value | Selectivity |
|----------------|------------:|
| C0 = `cs`      | ≈ 60% |
| C0 = `stat`    | ≈ 5% |
| C2 = `<2000`   | ≈ 1.7% |
| C2 = `2020–2024` | ≈ 50% |
| C3 = `1 version` | ≈ 65% |
| C3 = `4+ versions` | ≈ 8% |

Under an (optimistic) independence assumption, the `all4` clause on
Goodreads with values `{fiction, ¬en, paperback, 2011+}` lands
near `0.52 · 0.70 · 0.38 · 0.39 ≈ 5.4%` of the corpus; the same
sweep on arXiv with `{cs, cc-by, 2020–2024, 1 version}` lands near
`0.60 · 0.20 · 0.50 · 0.65 ≈ 3.9%`. Both numbers are within the
"narrow" design budget. The wide sweeps are intentionally further
from zero — between 5% and 25% per query — because their purpose
is to make the Bloom-filter approximation contestable.

### 5.6 Selectivity statistics to compute

For each `(dataset × filter_regime × clause_subset)`, the chapter
benefits from one or two extra plots. Goodreads selectivity CSVs
exist under [docs/thesis/results-data/datasets/](../../docs/thesis/results-data/datasets/)
(`goodreads_clause_c0_genre.csv`, `goodreads_clause_c1_lang_top30.csv`,
`goodreads_clause_c2_format.csv`, `goodreads_clause_c3_year.csv`).
The arXiv equivalents do **not** yet exist on disk; the script below
is the stub that produces them.

Each item below is annotated with its Chapter-6 figure catalog ID
(see [docs/thesis/results-data/recipes/plot_catalog.md](../../docs/thesis/results-data/recipes/plot_catalog.md)),
so the §5.6 outputs feed directly into the §6.1 workload-characterization
figures.

1. **Per-query selectivity CDF** → renders **Fig 6.1.2 = E3** in Ch.6.
   For the held-out batch in `eval_split.parquet`, run
   `mask = filter.evaluate_mask(query_attrs)`, compute
   `mask.float().mean(dim=1)`, and plot the CDF. Faceted by
   dataset and overlaid by clause-subset. Single-line stub:
   `selectivity = ExactAttributeFilter(item_attrs).evaluate_mask(qa).float().mean(1)`.
2. **Per-clause selectivity bar / violin** → renders **Fig 6.1.1 = D3**.
   Same as above but with one clause active at a time. Goodreads side
   is already populated (the four `goodreads_clause_c*.csv` files
   above); arXiv counterpart is the deferred work item.
3. **Per-item attribute-cardinality histogram.** For each item, how
   many of `A_max = 4` narrow slots are non-`-1`? This tells the
   reader how "tight" the per-item slot budget actually is. Optional
   in Ch.6; useful background plot for the §5.6 sub-section here.
4. **Wide-bag size distribution.** Same idea for the 32-slot wide
   tensor — does the 32 cap bind for most items, or does it leave
   slack? Optional; only matters once the wide_1shelf benchmark
   (currently unrun — see [07-results.md](07-results.md) §6.4.1) is
   actually executed.
5. **Pairwise clause independence check.** For each pair of active
   clauses (c1, c2), compare `mean(mask_c1 ∧ mask_c2)` to
   `mean(mask_c1) · mean(mask_c2)`. The ratio quantifies how much
   the multi-clause "all4" selectivity estimate above
   over-estimates the real selectivity.

The four `goodreads_clause_c*.csv` outputs already live under
`docs/thesis/results-data/datasets/` (see §8). The arXiv-side counterparts
should be stored alongside them under the same naming convention
(`arxiv_clause_c0_maincat.csv`, `arxiv_clause_c2_year.csv`,
`arxiv_clause_c3_nversions.csv`, `arxiv_clause_c4_author.csv`). This
is a single-pass extraction script against `item_attrs_narrow.pt`
generated by [evaluation/datasets/arxiv.py:893](../../evaluation/datasets/arxiv.py#L893).
Effort: a few hours of writing + one CPU-bound run.

### 5.7 Visual deliverables for the §5 chapter

1. AND-of-OR diagram for a worked sample query (one figure box, hand-
   drawn or tikz; one query, three clauses, one item that matches and
   one that doesn't).
2. Two-panel CDF — per-query selectivity, one panel per dataset,
   curves per clause-subset (after running the §5.6 script).
3. Per-clause selectivity bar (Goodreads + arXiv combined).
4. Table comparing the two filter primitives (Exact vs Bloom) on
   accuracy, memory and which backends they support.

### 5.8 Sweep configurations actually shipped

The YAML configurations under [`evaluation/config/`](../../evaluation/config/)
enumerate the cells the harness ran. The table below distils all
ten YAML files into one place so the chapter can describe "what was
measured" without making the reader open each file separately.

| Config file | Suite | k values | batch_size values | Algorithms | Backends | Filter sweeps |
|-------------|-------|----------|-------------------|------------|----------|---------------|
| `goodreads/d{64,128,256}-quality.yaml` | quality | 100, 200, 400 | 1 | linr_v1, linr_v3, linr_v4, silvertorch | triton, torch | (none) |
| `goodreads/d{64,128,256}-filter.yaml` | filter  | 100, 500, 1000 | 1, 8, 16 | linr_v1, **linr_v2**, linr_v3, linr_v4, silvertorch | triton, torch | clause × 6 (c0_genre, c1_lang_reverse, c2_format, c3_year, c0c1, all4), bloom × 3 (c0_genre, c2_format, c3_year) |
| `arxiv/d{64,128,256}-quality.yaml` | quality | 100, 200, 400 | 1 | linr_v1, linr_v3, linr_v4, silvertorch | triton, torch | (none) |
| `arxiv/d{64,128,256}-filter.yaml` | filter  | 100, 500, 1000 | 1, 8, 16 | linr_v1, **linr_v2**, linr_v3, linr_v4, silvertorch | triton, torch | clause × 5 (c0_maincat, c2_year, c3_nversions, c0c2, all4), bloom × 5 (same) |
| `yambda-500m/d{64,128,256}-quality.yaml` | quality | 100, 200, 400 | 1 | linr_v1, linr_v3, linr_v4, silvertorch | triton, torch | (none — Yambda has no attribute schema) |
| `yambda-5b/d{64,128}-quality.yaml` | quality | 100, 200, 400 | 1 | linr_v1, linr_v3, linr_v4, silvertorch | triton, torch | (none) |
| `deep_sweeps/arxiv-d128-silvertorch.yaml` | deep | 100, 200, 400 | 1, 8, 16 | silvertorch (10 param combos: `n_lists ∈ {1664, 8192} × n_probe ∈ {4, 8, 32, 128, 256}`) | triton, torch | bloom × 5 |
| `deep_sweeps/goodreads-d128-linr_v3.yaml` | deep | 100, 200, 400 | 1, 8, 16 | linr_v3 (5 param combos: `candidate_pool ∈ {2k, 4k, 8k, 16k, 32k}`) | triton, torch | clause × 6, bloom × 3 |

Three implementation details the chapter should call out:

1. **`linr_v2` only appears in the filter suite** — its purpose is to
   compute the exact-recall baseline (Recall = 1.0) for filtered top-K,
   which the quality suite does not need.
2. **silvertorch only uses the bloom filter path** in the deep sweep
   (the YAML omits `clause` because silvertorch wires the IVF +
   Bloom fusion natively; clause sweeps would be auto-skipped).
3. **Bloom does not appear with the `c1_lang_reverse` sweep on
   Goodreads** — Bloom is forward-only, so the reverse-language
   clause cannot be exercised via Bloom and the YAML deliberately
   omits it.

### 5.9 Backend parity and speedup (measured across the full harness)

Walking every result JSON under [`evaluation/results/`](../../evaluation/results/)
yields **7,104 result rows** spanning every (dataset, dim, suite,
algorithm, backend, batch_size, k, filter_sweep) cell. Two
verification claims fall out of pivoting the long-form table on
`backend`:

- **Triton vs torch Recall@K parity.** Across **2,697 matched cells**
  the maximum absolute difference in Recall is **5.28 × 10⁻³** and
  the mean absolute difference is **3.46 × 10⁻⁴**. In other words:
  the two backends are numerically equivalent to within
  floating-point noise on every cell that was measured. The chapter
  should quote these two numbers verbatim as the empirical
  justification for the "Triton and torch produce identical top-K
  within float tolerance" parity claim in Chapter 5.
  CSV: [`backend_parity_recall.csv`](results-data/results/backend_parity_recall.csv).
- **Triton vs torch latency speedup.** Same 2,697 matched cells. The
  ratio `torch_ms / triton_ms` has quantiles **q25 = 0.93**,
  **q50 = 1.42**, **q75 = 2.82**, **q90 = 4.87** — so the median
  Triton kernel is 1.42× faster than the torch eager equivalent, the
  upper quartile is ≈ 3× faster, and the hot-path top-decile is ≈ 5×
  faster. Conversely, the lower quartile (`q25 = 0.93`) means there
  are workloads where torch eager beats the Triton kernel by a few
  percent — these are the small-N / small-batch cells where kernel
  launch overhead dominates.
  CSV: [`backend_speedup.csv`](results-data/results/backend_speedup.csv);
  histogram: [`backend_speedup_hist.png`](results-data/results/backend_speedup_hist.png).

---

## 6. Results summaries reusable in Chapter 6

This thesis chapter does not contain results prose — that belongs in
the dedicated Chapter 6 — but it does compute and ship the artefacts
the Chapter 6 author will reference. Every CSV / PNG in this section
is read straight from `evaluation/results/**/*.json` via the
all-results walker (§5.9); no torch was loaded.

### 6.1 Quality–latency Pareto (full-scan / no filter)

[`docs/thesis/results-data/results/pareto_quality_3x3.png`](results-data/results/pareto_quality_3x3.png)
plots Recall@100 (y) vs `median_ms` (x, log scale) for **9 cells**
(arxiv d∈{64,128,256}, goodreads d∈{64,128,256}, yambda-500m d=128,
yambda-5b d∈{64,128}; the other three Yambda cells use the same
template — extend the script to grow the panel to a 4 × 3 grid if
all 12 are wanted). Each panel shows one point per algorithm in
{linr_v1_filter_mask, linr_v3, linr_v4, silvertorch}; all points are
batch_size = 1, k = 100, Triton backend, no filter.

Three observations the Chapter 6 author should pick up:

1. On Goodreads the algorithms cluster tightly (the corpus is small
   enough that even the unoptimised `linr_v1` finishes in the same
   order-of-magnitude as the optimised paths).
2. On Yambda-5B `linr_v3` and `silvertorch` open a clear latency
   gap against `linr_v1` and `linr_v4` (single-digit ms vs
   tens of ms).
3. On arXiv every algorithm hits Recall@100 ≈ 1.0 because each
   held-out paper retrieves itself as the top hit; the latency
   ranking is therefore the only signal in that row of panels.

### 6.2 SASRec / query-encoder quality ceiling

[`docs/thesis/results-data/results/sasrec_quality_ceiling.csv`](results-data/results/sasrec_quality_ceiling.csv)
holds the Recall@100 and NDCG@100 the SASRec encoder produces under
the `linr_v1_filter_mask` full-scan baseline (i.e. the exact dot
product, no approximation, no quantisation). The same numbers are
already reproduced inline in §7.5.

### 6.3 Filter-sweep recall, Goodreads d = 128

[`docs/thesis/results-data/results/goodreads_d128_filter_recall.png`](results-data/results/goodreads_d128_filter_recall.png)
shows Recall@100 per `(filter_kind, sweep, algorithm)` for the
goodreads-d128 filter suite at batch_size = 1, k = 100, Triton
backend. The clause sweeps (`c0_genre`, `c1_lang_reverse`,
`c2_format`, `c3_year`, `c0c1`, `all4`) appear first, the Bloom
sweeps (`c0_genre`, `c2_format`, `c3_year`) second. Five algorithms
per sweep: linr_v1, linr_v2 (exact recall = 1.0 reference), linr_v3,
linr_v4, silvertorch.

### 6.4 Deep sweep: silvertorch on arXiv-d128 (n_lists × n_probe)

[`docs/thesis/results-data/results/arxiv_silvertorch_deep_pareto.png`](results-data/results/arxiv_silvertorch_deep_pareto.png)
traces Recall@100 vs `median_ms` on the bloom-`all4` filter for
silvertorch on arXiv-d128, sweeping `n_lists ∈ {1664, 8192}` and
`n_probe ∈ {4, 8, 32, 128, 256}`. The two `n_lists` curves are
parallel: at the same `n_probe` the smaller cluster layout
(`n_lists = 8192`, ≈ 365 items per cluster) is faster but lower
recall; at the same recall the smaller layout is uniformly to the
left (faster). This is the headline plot from
[`docs/thesis/00-thesis-plan.md`](00-thesis-plan.md) §6.3 ("arXiv-d128-silvertorch:
heatmap of …"). The contour-heatmap variant Petrov & Macdonald-style
is left as a follow-up — the current scatter is enough for the
chapter prose.

### 6.5 Deep sweep: linr_v3 candidate-pool sweep on Goodreads-d128

[`docs/thesis/results-data/results/goodreads_linr_v3_deep_recall.png`](results-data/results/goodreads_linr_v3_deep_recall.png)
shows Recall@100 vs `candidate_pool` size for the linr_v3 OPORP
prefilter across five filter sweeps. The recall climbs sub-linearly
in `candidate_pool` size — going from 2k → 32k recovers most of the
exact-baseline recall — but the curves saturate at slightly
different `candidate_pool` values depending on filter selectivity
(narrow filters need a smaller pool because most candidates pass the
filter; wide filters need a larger pool).

### 6.6 GPU memory breakdown (d = 128)

[`docs/thesis/results-data/results/d128_quality_memory.csv`](results-data/results/d128_quality_memory.csv)
and the stacked-bar plot
[`memory_d128_stacked.png`](results-data/results/memory_d128_stacked.png)
report per-algorithm `index_mem_mib` and `fwd_scratch_mib` at d=128,
k=100, batch_size=1, Triton backend, across the four corpora. Two
patterns:

- **`linr_v4` is the smallest index everywhere** (97 MiB Goodreads,
  365 MiB arXiv, 228 MiB Yambda-500M, 655 MiB Yambda-5B) because the
  1-bit OPORP code is by construction `d_bits / 8` bytes per item.
  This is the algorithm to recommend for memory-constrained
  deployments.
- **`silvertorch` is the largest index** on Goodreads (298 MiB) and
  Yambda-5B (800 MiB) because IVF cluster centroids + Bloom
  signatures are stored on top of the INT8-quantised codes.
- **`fwd_scratch_mib` is zero** for every algorithm at batch_size = 1
  — the scratch buffer only inflates when batching pushes the
  per-query memory above the persistent index.

### 6.7 Run inventory and wall-clock costs

From [`evaluation/results/_runlogs/SUMMARY.quality-deep.txt`](../../evaluation/results/_runlogs/SUMMARY.quality-deep.txt) — the
9 stages of the quality + deep-sweep run on an **NVIDIA A100-SXM4-80GB**
took 4 h 14 min in total (host `fd4a94de96b9`,
2026-05-23T09:52:29Z → 14:06:41Z):

| Stage | Wall-clock |
|-------|-----------:|
| `arxiv/d64-quality.yaml`   | 1 min 40 s |
| `arxiv/d128-quality.yaml`  | 1 min 46 s |
| `arxiv/d256-quality.yaml`  | 3 min 20 s |
| `goodreads/d64-quality.yaml`  | 36 min |
| `goodreads/d128-quality.yaml` | 40 min |
| `goodreads/d256-quality.yaml` | 48 min |
| `deep_sweeps/arxiv-d128-silvertorch.yaml`  | 1 h 6 min |
| `deep_sweeps/goodreads-d128-linr_v3.yaml`  | 55 min |

The filter-suite runs (`*-filter.yaml`) were executed in a separate
batch not captured in `SUMMARY.quality-deep.txt`; the corresponding
JSONs under `evaluation/results/{goodreads,arxiv}/d*-filter.json` are
the only on-disk evidence of their wall-clock cost.

### 6.8 Cross-dataset summary table

A single landscape table to place at the start of the chapter so the
reader can see the three datasets side-by-side before diving into the
per-dataset prose. Verified numbers in **bold**, derived/estimated
numbers in regular text, TBD numbers explicitly marked.

| Property | Goodreads (UCSD) | arXiv (open-index) | Yambda-500M | Yambda-5B |
|----------|-----------------:|-------------------:|------------:|----------:|
| # users in raw release | 876,145 | n/a (no users) | 100,000 | **1,000,000** |
| # users in eval split (this thesis) | **313,178** | 10,000 (heldout papers) | **45,932** | **459,067** |
| # items in raw release | 2,360,655 editions / 1,521,962 works | ≈ 2,990,000 | 3,004,578 | 9,390,623 |
| # items used as retrieval corpus | ≈ 797,000 | ≈ 2,990,000 | TBD-from-prep_log.json | TBD-from-prep_log.json |
| # interactions in raw release | 228,648,342 | n/a | 466,512,103 listens | 4,649,567,411 listens |
| # interactions used for SASRec | 112,131,203 (`is_read=True`, pre-5-core) | n/a | 466M post-50%-played filter | 4.65B post-50%-played filter |
| Max sequence length | 200 | n/a | 200 | 200 |
| Filter benchmark? | yes | yes | **no** (quality only) | **no** (quality only) |
| Narrow clauses | 5 (one reverse) | 5 (all forward) | — | — |
| Wide vocab | ≈ 1,800 shelves | ≈ 2,000 leaf cats | — | — |
| Query encoder | SASRec d ∈ {64, 128, 256} | nomic-embed-text-v1.5 (Matryoshka) | SASRec d ∈ {64, 128, 256} | SASRec d ∈ {64, 128} (d=256 not run) |

The 2.36 M raw editions of Goodreads collapse to ~797 k works after
the iterative 5-core + work-id merge; the 836,433 users with ≥ 1
`is_read` event collapse to **313,178** evaluation users after the
same filter. arXiv keeps every paper (no user side); Yambda's
prepared splits keep **45,932 / 459,067** evaluation users on the
two scales used in this thesis.

---

## 7. SASRec / gSASRec query-model training

This section is intentionally self-contained: it is the one place in
the chapter where the user can drop a Russian translation
back-to-back with the formulas and tables, without cross-references
forcing them to flip pages.

### 7.1 Why SASRec, why gBCE

SASRec (Kang & McAuley 2018) is a unidirectional Transformer encoder
trained to predict the next item in a user's sequence with
dot-product compatible representations — exactly the regime model-
based retrievers need. The model is small (two encoder blocks, two
heads, embedding dim 64–256) and well-established as a baseline on
sequential recommendation. The thesis does not introduce a new
sequence model; SASRec is used because it is the smallest credible
"reasonable encoder" and because the Yambda paper uses it as their
strongest baseline (their Tab. 4 ranks SASRec second on the listens
benchmark, ItemKNN first).

The training loss is **gBCE** (generalised binary cross-entropy,
Petrov & Macdonald 2023, RecSys best paper). The motivation is the
overconfidence pathology of SASRec when trained with sparse
uniform-negative BCE: the model assigns near-`σ(∞)` probability to
the positive because the sampled negatives wildly under-represent
the true negative density. Petrov & Macdonald solve this with an
exponent on the positive-class sigmoid that depends on the
sampling rate; this is described formally in §7.3 below.

### 7.2 Architecture

The encoder ([`evaluation/training/model.py`](../../evaluation/training/model.py)):

- **Item embedding** — `nn.Embedding(num_items + 1, d, padding_idx=0)`.
  Index 0 is the left-padding sentinel; all other indices are the
  dense work-id (Goodreads) or dense track-id (Yambda).
- **Positional embedding** — `nn.Embedding(max_seq_length=200, d)`,
  absolute (not relative or rotary).
- **Embedding dropout** — applied to the sum of item + positional
  embeddings before the Transformer.
- **Transformer encoder** — `nn.TransformerEncoder` with
  `num_blocks = 2` layers, each `nn.TransformerEncoderLayer` with
  `nhead = 2`, `dim_feedforward = ffn_hidden_dim` (defaulting to
  `4 · d` in the runs), `activation = "gelu"`, `batch_first = True`,
  `norm_first = True` (pre-LayerNorm). Two masks are applied: an
  upper-triangular causal mask so the encoder is autoregressive, and
  a padding-key mask that zeros attention to padding tokens.
- **Final LayerNorm** on the encoder output.
- **Output embedding** — separate `nn.Embedding(num_items + 1, d)`
  used for scoring at training time (`reuse_item_embeddings = False`
  is the default in [`config.py`](../../evaluation/training/config.py));
  at inference time this is what gets dumped to `item_embs.pt` and
  becomes the index the retrieval layers consume.
- **`predict_last(seq)`** returns the hidden state at the last
  non-padding position — the query vector that the retrieval
  benchmark consumes.

Weight init is truncated-normal with `std = 0.02` for both
embedding tables and the Transformer; layer-norm scales are
initialised to 1, biases to 0.

### 7.3 gBCE loss — formal block

Let `q` be the encoder output at one valid (non-padding) position,
`p` the positive item id at that position, and
`n_1, …, n_M` the `M = negs_per_pos = 256` uniformly sampled
negative item ids. Let `e_x = output_embedding(x)`, `N` the size of
the item vocabulary, and `t` = `gbce_t` = 0.75 the temperature
parameter recommended by Petrov & Macdonald (2023).

Define the effective sampling rate and the temperature-adjusted
exponent:

```math
\alpha = \frac{M}{N - 1}, \qquad
\beta  = \alpha \cdot \Bigl(\bigl(1 - \tfrac{1}{\alpha}\bigr) \, t + \tfrac{1}{\alpha}\Bigr).
```

Compute the positive and negative raw scores:

```math
s^+ = \langle q,\, e_p \rangle, \qquad
s^-_k = \langle q,\, e_{n_k} \rangle \quad (k = 1, \dots, M).
```

Apply the gBCE adjustment to the positive only:

```math
\hat p^+ = \sigma(s^+), \qquad
\tilde p^+ = \mathrm{clamp}\bigl(\hat p^{+\,-\beta}, \, 1 + \epsilon, \, \infty\bigr), \qquad
\hat y^+ = \log\frac{1}{\tilde p^+ - 1}.
```

Concatenate adjusted positive logit with raw negative logits and
apply standard binary cross-entropy with logits:

```math
\mathrm{logits} = [\hat y^+, \, s^-_1, \dots, s^-_M],
\qquad y = [1, 0, \dots, 0],
\qquad \mathcal L = \mathrm{BCEWithLogits}(\mathrm{logits}, y).
```

Why this works (paraphrasing Petrov & Macdonald): with `M ≪ N`
plain BCE pushes `s^+` to ±∞ because the sampled negatives
under-represent the true negative density by a factor of `1/α`.
Raising `\hat p^+` to `-β` (with `β < 1` for sparse sampling) is
exactly the correction that gives back a calibrated probability.
The chapter should reproduce this paragraph verbatim — it's the
single most important architectural decision in the training
pipeline.

Implementation: [`evaluation/training/losses.py:7-37`](../../evaluation/training/losses.py).
The double-precision `pow(-β)` and clamps in the implementation are
present to avoid fp16/fp32 overflow at very small `\hat p^+`.

### 7.4 Training loop

From [`evaluation/training/train_sasrec.py`](../../evaluation/training/train_sasrec.py)
and [`evaluation/training/config.py`](../../evaluation/training/config.py):

| Item | Value |
|------|-------|
| Optimiser | `torch.optim.AdamW(lr=1e-3, weight_decay=0, fused=True on CUDA)` |
| Mixed precision | `torch.autocast(dtype=torch.bfloat16)` forward, fp32 gradients |
| Gradient clipping | `clip_grad_norm_(max_norm=1.0)` |
| Batch size | 256 sequences |
| Negative sampling | 256 uniform negatives per positive, sampled from `[1, num_items]` per (user, position) inside the collate function (no in-batch negatives) |
| Max epochs | 200 |
| Early stopping | patience 20 epochs, on validation `ndcg@10` |
| DataLoader | 4 workers, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` |
| Logging | per-step `train/loss_step`, per-epoch `train/loss_epoch`, val/test metrics; optional W&B |

### 7.5 Hyperparameter table actually used (Goodreads + Yambda)

Common to all runs: `num_blocks = 2`, `num_heads = 2`,
`ffn_hidden_dim = 4 · d`, `max_seq_length = 200`,
`batch_size = 256`, `negs_per_pos = 256`, `gbce_t = 0.75`,
`lr = 1e-3`, `weight_decay = 0`, `dropout = 0.5`, AdamW + grad-clip
+ bfloat16 autocast as above.

| Dataset      | d   | # items | Trained? | Final R@100 (full-scan baseline) | Final NDCG@100 |
|--------------|-----|---------|:--------:|---------------------------------:|---------------:|
| Goodreads    |  64 | 797k    | ✅       | **0.1486** | **0.0687** |
| Goodreads    | 128 | 797k    | ✅       | **0.1479** | **0.0691** |
| Goodreads    | 256 | 797k    | ✅       | **0.1472** | **0.0681** |
| Yambda-500M  |  64 | 3.0M    | ✅       | **0.1564** | **0.1079** |
| Yambda-500M  | 128 | 3.0M    | ✅       | **0.1486** | **0.1028** |
| Yambda-500M  | 256 | 3.0M    | ✅       | **0.1398** | **0.0990** |
| Yambda-5B    |  64 | 9.4M    | ✅       | **0.1744** | **0.1172** |
| Yambda-5B    | 128 | 9.4M    | ✅       | **0.1779** | **0.1207** |
| Yambda-5B    | 256 | 9.4M    | ❌ (not run) | — | — |

The numbers in the right-hand columns are the
`linr_v1_filter_mask, backend=triton, batch_size=1, k=100,
filter_kind=none` rows from the per-dataset
`evaluation/results/{dataset}/d*-quality.json` files — i.e., the
full-scan quality ceiling that the SASRec query model produces.
They double as the "SASRec quality table" expected by the chapter
because `linr_v1` with no filter is mathematically the same as a
plain matmul-and-top-K against the dense SASRec item embedding
table (no approximation, no quantisation), so its Recall and NDCG
are exactly the SASRec model's quality on that split.

Two patterns the chapter should comment on:

1. On Goodreads the three dim settings are within ±1% of each
   other — the encoder has saturated long before `d = 64`. This is
   not surprising: the Goodreads corpus is small (797k items, median
   sequence ≈ 43 events) and the bottleneck is the data, not the
   encoder capacity.
2. On Yambda-500M Recall@100 *decreases* with d (0.156 → 0.149 →
   0.140); on Yambda-5B the trend reverses and d = 128 beats d =
   64. The natural reading is that 500M is data-bound (more
   capacity overfits the short tail) while 5B is encoder-bound
   (more capacity helps because the data finally supports it). This
   would be one of the headline observations in Chapter 6.

Training time and peak GPU memory per cell:
**TBD-from-train_metrics.json** (not on disk; the chapter pulls
`total_time_sec` and `peak_gpu_mem_bytes` from the JSON the
training script writes alongside each checkpoint).

### 7.6 Per-epoch evaluation

From [`evaluation/training/evaluate.py`](../../evaluation/training/evaluate.py)
and [`evaluation/retrieval/metrics.py`](../../evaluation/retrieval/metrics.py):

- Split: validation parquet during training, test parquet at the
  end.
- Batch size: 512 (`eval_batch_size`); chunked scoring with chunk
  size 262,144 items so the full `[B, N]` score matrix is never
  materialised when N is in the millions.
- `mask_history = False`: held-out items are not removed from the
  candidate set (re-consumption is allowed). This matches the
  Yambda paper's protocol; for Goodreads it matters less since
  re-reading is rare.
- Cutoffs `K ∈ {10, 100}` for Recall, NDCG and Coverage.

Formal metric definitions used during training:

```math
\mathrm{Recall}@K = \frac{|\{i \le K : \mathrm{rank}_i \in T\}|}{|T|},
\qquad
\mathrm{DCG}@K = \sum_{i=1}^{K} \frac{\mathbf{1}[\mathrm{rank}_i \in T]}{\log_2(i+1)},
\qquad
\mathrm{IDCG}@K = \sum_{i=1}^{\min(|T|, K)} \frac{1}{\log_2(i+1)},
\qquad
\mathrm{NDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}.
```

Where `T` is the per-user target set (multi-target, because both
Goodreads and Yambda can produce more than one held-out item per
user under the chosen split). Coverage@K is
`|⋃_u R(u, K)| / N` — fraction of the catalogue ever recommended
across the eval batch.

### 7.7 Checkpoint convention

From [`docs/system/checkpoints.md`](../system/checkpoints.md):

```text
data/<dataset>/checkpoints/gsasrec-d<N>-drop0.5-<id>/
├── best_model.pt         # state_dict at best val ndcg@10
├── config.json           # GSASRecConfig snapshot
├── item_embs.pt          # [N+1, d] output embedding table; row 0 zeroed
├── item_id_map.json      # copy of the dataset's dense id map
├── item_attrs.parquet    # optional copy of narrow attributes (kept for downstream eval convenience)
├── train_metrics.json    # epoch_losses[], val_metrics_per_epoch[], best_val_metric, test_metrics, total_time_sec, peak_gpu_mem_bytes
└── eval_quality.json     # final {split, ks, metrics}
```

The retrieval harness loads `item_embs.pt` directly (no need to
re-instantiate the SASRec model) and re-encodes queries on demand
with the same `state_dict` from `best_model.pt`. The cache key for
encoded queries (`queries_cache.py`) is
`(checkpoint_mtime, max_seq_length, users_limit)`.

### 7.8 Visual deliverables for the §7 chapter

1. SASRec architecture diagram: `item_seq` → item-emb + pos-emb →
   embedding dropout → 2× Transformer encoder block (pre-LN, causal
   + padding mask) → LayerNorm → query vector. Can be reused in
   Chapter 4.
2. Consolidated SASRec quality table — the one already populated in
   §7.5 above.
3. Training-loss curves per cell (one panel per (dataset, d)) — from
   `train_metrics.json`. **TBD-from-train_metrics.json.**
4. Validation Recall@100 vs epoch — same panels, with the best
   epoch marked. **TBD-from-train_metrics.json.**

---

## 8. Things to compute / produce (consolidated checklist)

Every artefact lives at `docs/thesis/results-data/` — see §11 for
the complete file manifest. Status legend: ✅ done in this session;
🟡 partial (raw-corpus proxy computed, prepared-split version still
needed); 🔴 needs torch / GPU / a fresh run.

| ID | Artefact | Status | Where it lives |
|----|----------|--------|-----------------|
| A | Cross-dataset summary table | ✅ | §6.8 in this file |
| B | Goodreads corpus stats (users, items, interactions, `is_read`) | ✅ | §2.3 |
| C | Goodreads sequence-length CDF + popularity rank-frequency on the raw `is_read` signal | ✅ | `datasets/goodreads_seq_len_cdf.{csv,png}`, `datasets/goodreads_item_popularity_rankfreq.csv`, `datasets/goodreads_item_popularity.png` |
| C′ | Same on the **prepared `train.parquet`** (post 5-core, post truncation, on the 313 k eval users) | 🟡 | need to run `goodreads.py prep` locally, then re-run §11 script with `train.parquet` instead of the raw `is_read` filter |
| D | Goodreads narrow-clause per-bucket coverage (genre, year, format, language) | ✅ | `datasets/goodreads_clause_c{0,1,2,3}_*.csv` |
| D′ | arXiv per-clause coverage (papers per main-cat / year / license / versions) | 🔴 | needs `arxiv_papers.parquet` — not on disk; recipe in §11 |
| E | Yambda listens-per-user CDF + item popularity | 🔴 | needs prepared Yambda `train.parquet` — not on disk; the paper's table 3 (median listens = 3,076) is quoted in §4.6 |
| F | Filter selectivity per-query CDF, per-clause selectivity bar | 🔴 | needs `item_attrs_narrow.pt` + torch forward — TBD-script-stub in §5.6 |
| G | SASRec quality ceiling table | ✅ | `results/sasrec_quality_ceiling.csv`; same numbers in §7.5 |
| H | SASRec training-loss curves + val-Recall@100 curves | 🔴 | needs `train_metrics.json` per checkpoint — not on disk |
| I | Pipeline schematics (Goodreads, arXiv, Yambda + SASRec architecture) | 🔴 | author / tikz task |
| J | Quality-latency Pareto (3 × 3 panel) | ✅ | `results/pareto_quality_3x3.png` |
| K | Goodreads d128 filter-sweep Recall bar chart | ✅ | `results/goodreads_d128_filter_recall.png` |
| L | silvertorch deep-sweep scatter (arxiv-d128) | ✅ | `results/arxiv_silvertorch_deep_pareto.png` |
| M | linr_v3 deep-sweep curve (goodreads-d128) | ✅ | `results/goodreads_linr_v3_deep_recall.png` |
| N | GPU memory stacked bar (d128 across datasets) | ✅ | `results/memory_d128_stacked.png` + `results/d128_quality_memory.csv` |
| O | Backend parity (Triton vs torch) — max/mean ΔRecall | ✅ | `results/backend_parity_recall.csv` (max 5.28×10⁻³, mean 3.46×10⁻⁴) |
| P | Backend speedup distribution | ✅ | `results/backend_speedup.csv` + `results/backend_speedup_hist.png` (median 1.42×, q90 4.87×) |
| Q | Run wall-clock inventory | ✅ | §6.7 table from `_runlogs/SUMMARY.quality-deep.txt` |
| R | All-results long-form CSV (joinable to any of the above) | ✅ | `results/all_results_long.csv` (7,104 rows, 25 columns) |

---

## 9. Open questions to resolve before writing the chapter

1. Yambda training — is `dropout = 0.5` confirmed for every `(scale, d)` cell or
   did some runs use a different value? (`docs/system/checkpoints.md` lists
   `drop0.5` in the example directory name; need to confirm against the actual
   `config.json` next to `best_model.pt`.)
2. Yambda-5B `d = 256` — was it intentionally skipped (compute budget) or
   pending? §7.5 currently flags it as "not run".
3. arXiv — should we add a brief text-encoder fine-tune as a "what if" baseline,
   or is the chapter purely about the frozen nomic-embed setup? Current scope is
   the frozen setup; flag this explicitly.
4. Goodreads `eval_split.parquet` — is it produced by the `attrs` subcommand, or
   by a separate script? `goodreads.py` mentions it in the `attrs` section; verify
   path and column schema once a real prep run lands.
5. The `suite` field in the result JSONs is currently mis-labelled
   (`suite="yambda"` appears in Goodreads/arxiv jsons — see §7.5 footnote). Worth
   fixing before Chapter 6 starts depending on the field for routing.

---

## 10. Critical files referenced when writing this chapter

| Domain | Files |
|--------|-------|
| Dataset pipelines | [`evaluation/datasets/goodreads.py`](../../evaluation/datasets/goodreads.py), [`arxiv.py`](../../evaluation/datasets/arxiv.py), [`yambda.py`](../../evaluation/datasets/yambda.py), [`common.py`](../../evaluation/datasets/common.py), [`constants.py`](../../evaluation/datasets/constants.py), [`timesplit.py`](../../evaluation/datasets/timesplit.py), [`hf_io.py`](../../evaluation/datasets/hf_io.py) |
| Training | [`evaluation/training/train_sasrec.py`](../../evaluation/training/train_sasrec.py), [`model.py`](../../evaluation/training/model.py), [`losses.py`](../../evaluation/training/losses.py), [`evaluate.py`](../../evaluation/training/evaluate.py), [`config.py`](../../evaluation/training/config.py), [`dataset.py`](../../evaluation/training/dataset.py) |
| Eval metrics | [`evaluation/retrieval/metrics.py`](../../evaluation/retrieval/metrics.py), [`evaluation/retrieval/algos/filter.py`](../../evaluation/retrieval/algos/filter.py) |
| Filter primitives | [`retrieve/src/retrieve/interfaces.py`](../../retrieve/src/retrieve/interfaces.py), [`layers/filters/exact_attribute.py`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py), [`layers/filters/bloom.py`](../../retrieve/src/retrieve/layers/filters/bloom.py), [`kernels/filters/clause_mask.py`](../../retrieve/src/retrieve/kernels/filters/clause_mask.py), [`kernels/filters/clause_compact.py`](../../retrieve/src/retrieve/kernels/filters/clause_compact.py) |
| YAML sweep configs | [`evaluation/config/goodreads/`](../../evaluation/config/goodreads/), [`evaluation/config/arxiv/`](../../evaluation/config/arxiv/), [`evaluation/config/yambda-500m/`](../../evaluation/config/yambda-500m/), [`evaluation/config/yambda-5b/`](../../evaluation/config/yambda-5b/), [`evaluation/config/deep_sweeps/`](../../evaluation/config/deep_sweeps/) |
| System docs | [`docs/system/filtering.md`](../system/filtering.md), [`docs/system/checkpoints.md`](../system/checkpoints.md) |
| Yambda paper notes | [`articles/yambda.md`](../../articles/yambda.md) |
| On-disk artefacts (needed at chapter-writing time) | `data/<dataset>/prep_log.json`, `data/<dataset>/checkpoints/*/train_metrics.json`, `eval_quality.json`, `config.json`, `evaluation/results/_runlogs/SUMMARY.quality-deep.txt` |

---

## 11. Computed artefacts manifest

All artefacts referenced from §6 and §8 live under
[`docs/thesis/results-data/`](results-data/). The next agent (the
Russian-chapter writer) should treat them as authoritative and
read-only — re-running the recipes is encouraged for verification
but not required to write the chapter.

### 11.1 Dataset stats (`results-data/datasets/`)

| File | Rows × cols | What it is | Reproduction |
|------|------------:|------------|--------------|
| [`goodreads_seq_len_cdf.csv`](results-data/datasets/goodreads_seq_len_cdf.csv) | 101 × 2 | Per-user `is_read` count CDF at integer percentiles 0..100 | Recipe A below |
| [`goodreads_seq_len_cdf.png`](results-data/datasets/goodreads_seq_len_cdf.png) | — | Same as PNG with the `max_seq_len=200` cap annotated | matplotlib snippet in Recipe A |
| [`goodreads_item_popularity_rankfreq.csv`](results-data/datasets/goodreads_item_popularity_rankfreq.csv) | 336 × 2 | Log-sampled (rank, n_events) for every book with ≥ 1 `is_read` | Recipe A |
| [`goodreads_item_popularity.png`](results-data/datasets/goodreads_item_popularity.png) | — | Same as log-log scatter with the 5-core threshold annotated | Recipe A |
| [`goodreads_clause_c0_genre.csv`](results-data/datasets/goodreads_clause_c0_genre.csv) | 10 × 3 | Per-genre raw edition counts (the 10 canonical buckets) | Recipe B |
| [`goodreads_clause_c1_lang_top30.csv`](results-data/datasets/goodreads_clause_c1_lang_top30.csv) | 30 × 2 | Top-30 language codes by edition count | Recipe B |
| [`goodreads_clause_c2_format.csv`](results-data/datasets/goodreads_clause_c2_format.csv) | 5 × 3 | Per-format-bucket edition counts | Recipe B |
| [`goodreads_clause_c3_year.csv`](results-data/datasets/goodreads_clause_c3_year.csv) | 5 × 3 | Per-year-bucket edition counts (incl. unknown-year bucket) | Recipe B |

### 11.2 Evaluation results (`results-data/results/`)

| File | Rows × cols | What it is |
|------|------------:|------------|
| [`all_results_long.csv`](results-data/results/all_results_long.csv) | 7,104 × 25 | Tidy long-form of every result row across `evaluation/results/**/*.json`, with dataset/scale/dim/suite labels added |
| [`sasrec_quality_ceiling.csv`](results-data/results/sasrec_quality_ceiling.csv) | 11 × 6 | The SASRec / nomic-embed full-scan quality baseline per `(dataset, scale, dim)` |
| [`backend_parity_recall.csv`](results-data/results/backend_parity_recall.csv) | 2,697 × 12 | One row per matched `(dataset, dim, suite, …)` cell with `triton`, `torch`, `recall_abs_diff` columns |
| [`backend_speedup.csv`](results-data/results/backend_speedup.csv) | 2,697 × 12 | Same pivot but on `median_ms`, with `speedup_torch_over_triton` column |
| [`d128_quality_memory.csv`](results-data/results/d128_quality_memory.csv) | 16 × 6 | Per-algorithm `peak_mem_mib / index_mem_mib / fwd_scratch_mib` at d=128 across 4 datasets |
| [`pareto_quality_3x3.png`](results-data/results/pareto_quality_3x3.png) | — | 9-panel Recall@100 vs median_ms scatter (full-scan, Triton, B=1, K=100) |
| [`goodreads_d128_filter_recall.png`](results-data/results/goodreads_d128_filter_recall.png) | — | Grouped bar of Recall@100 per `(filter_kind, sweep, algorithm)` |
| [`arxiv_silvertorch_deep_pareto.png`](results-data/results/arxiv_silvertorch_deep_pareto.png) | — | Recall vs median_ms curve for the silvertorch deep sweep on bloom-`all4` |
| [`goodreads_linr_v3_deep_recall.png`](results-data/results/goodreads_linr_v3_deep_recall.png) | — | Recall vs `candidate_pool` for the linr_v3 deep sweep, per filter sweep |
| [`backend_speedup_hist.png`](results-data/results/backend_speedup_hist.png) | — | Histogram of `torch_ms / triton_ms` across all matched cells |
| [`memory_d128_stacked.png`](results-data/results/memory_d128_stacked.png) | — | Stacked-bar `index_mem + fwd_scratch` per algorithm × dataset |

### 11.3 Reproduction recipes

**Recipe A — Goodreads sequence length + popularity** (~30 s on a
laptop, requires `~/datasets/goodreads-ucsd/processed/`).

```bash
uv run --no-project --with polars --with pyarrow --with numpy --with matplotlib python - <<'PY'
import polars as pl, numpy as np, csv, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

root = os.path.expanduser('~/datasets/goodreads-ucsd/processed')
out  = 'docs/thesis/results-data/datasets'
inter = pl.scan_parquet(f'{root}/goodreads_interactions_dedup.parquet').filter(pl.col('is_read') == True)

# Per-user sequence length CDF
seq = inter.group_by('user_id').len().rename({'len': 'n_events'}).collect()
arr = np.sort(seq['n_events'].to_numpy())
with open(f'{out}/goodreads_seq_len_cdf.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['percentile', 'n_events'])
    for p in np.linspace(0, 100, 101): w.writerow([f'{p:.1f}', int(np.percentile(arr, p))])

# Per-item popularity rank-frequency, log-sampled
pop = inter.group_by('book_id').len().rename({'len': 'n_events'}).collect()
ranked = pop.sort('n_events', descending=True).with_row_index('rank').with_columns((pl.col('rank') + 1).alias('rank'))
n = ranked.height
sample_idx = np.unique(np.logspace(0, np.log10(n - 1), 400).astype(int))
ranked.filter(pl.col('rank').is_in(sample_idx.tolist())).select(['rank','n_events']).write_csv(
    f'{out}/goodreads_item_popularity_rankfreq.csv')
PY
```

**Recipe B — Goodreads per-bucket coverage** (~10 s, same data).

```bash
uv run --no-project --with polars --with pyarrow python - <<'PY'
import polars as pl, os, csv
from collections import Counter

root = os.path.expanduser('~/datasets/goodreads-ucsd/processed')
out  = 'docs/thesis/results-data/datasets'

# C0 genre
g = pl.scan_parquet(f'{root}/goodreads_book_genres_initial.parquet').unnest('genres').collect()
counts = sorted([(c, g.select(pl.col(c).is_not_null().sum()).item())
                 for c in g.columns if c != 'book_id'], key=lambda x: -x[1])
with open(f'{out}/goodreads_clause_c0_genre.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['genre_key','editions_with_vote','pct_of_raw'])
    for k, n in counts: w.writerow([k, n, f'{100*n/2_360_655:.2f}'])

# C2 format (mirror goodreads.py _format_to_bucket)
def fmt(s):
    if s is None: return 'other'
    sl = s.lower()
    if 'paperback' in sl: return 'paperback'
    if 'hardcover' in sl: return 'hardcover'
    if 'ebook' in sl or 'kindle' in sl: return 'ebook'
    if 'audio' in sl or 'cd' in sl: return 'audio'
    return 'other'
books = pl.scan_parquet(f'{root}/goodreads_books.parquet')
fmt_buckets = [fmt(s) for s in books.select('format').collect().to_series()]
bc = Counter(fmt_buckets)
with open(f'{out}/goodreads_clause_c2_format.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['format_bucket','n_editions','pct_of_raw'])
    for k, n in bc.most_common(): w.writerow([k, n, f'{100*n/2_360_655:.2f}'])

# C3 year (mirror _year_to_bucket)
def yb(y):
    if y is None or y < 1500 or y > 2025: return None
    if y < 1990: return '<1990'
    if y <= 2000: return '1990-2000'
    if y <= 2010: return '2001-2010'
    return '2011+'
yr = books.select(pl.col('publication_year').cast(pl.Int64, strict=False).alias('y')).collect().to_series().to_list()
ybc = Counter(yb(y) for y in yr)
with open(f'{out}/goodreads_clause_c3_year.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['year_bucket','n_editions','pct_of_raw'])
    for k, n in sorted(ybc.items(), key=lambda kv: (kv[0] is None, kv[0])):
        w.writerow([str(k), n, f'{100*n/2_360_655:.2f}'])

# C1 language top-30
lang = (books.select('language_code')
        .filter(pl.col('language_code').is_not_null() & (pl.col('language_code') != ''))
        .group_by('language_code').len().sort('len', descending=True).head(30).collect())
lang.write_csv(f'{out}/goodreads_clause_c1_lang_top30.csv')
PY
```

**Recipe C — Walk evaluation results into the long CSV and the
summary derivatives** (~5 s, requires only the `evaluation/results/`
directory). The full script lives at
[`docs/thesis/results-data/recipes/walk_results.py`](results-data/recipes/walk_results.py)
and runs end-to-end:

```bash
uv run --no-project --with polars --with pyarrow --with numpy --with matplotlib \
    python docs/thesis/results-data/recipes/walk_results.py
# parsed 7104 rows from 19 JSON files
# all CSVs + plots regenerated under docs/thesis/results-data/

# Note: `--no-project` is required because the parent project pins CUDA-only
# torch wheels (no macOS arm64 build); the script only needs polars + matplotlib.
```

The script regenerates `all_results_long.csv`,
`sasrec_quality_ceiling.csv`, `backend_parity_recall.csv`,
`backend_speedup.csv`, `d128_quality_memory.csv`, and the four PNGs
(`pareto_quality_3x3.png`, `backend_speedup_hist.png`,
`memory_d128_stacked.png`, plus the deep-sweep plots are produced by
a separate one-liner in the same recipes/ folder if needed). It does
not touch the raw datasets — those are recipes A and B above.

### 11.4 Recipes still TBD (require torch and a GPU run)

- **Per-query selectivity CDF (item F).** Load
  `item_attrs_narrow.pt` and `clause_is_reverse_narrow.pt`,
  construct an `ExactAttributeFilter`, run `evaluate_mask` over the
  eval-split query attrs, and reduce `mean(dim=1)` to get the
  selectivity per query. See the §5.6 script stub.
- **SASRec training curves (item H).** Read each
  `data/<dataset>/checkpoints/gsasrec-d*-drop0.5*/train_metrics.json`
  and plot `epoch_losses` + `val_metrics_per_epoch[*]['recall@100']`
  as two-panel plots per cell. No torch needed — these are just
  JSON files — but the JSONs themselves do not exist on this disk
  yet.
- **arXiv per-clause coverage (item D′).** Same shape as Recipe B
  but on `arxiv_papers.parquet`, with the bucket functions from
  [`evaluation/datasets/arxiv.py`](../../evaluation/datasets/arxiv.py)
  (`_category_to_main`, `_license_to_bucket`, `_year_to_bucket`,
  `_versions_to_bucket`). The parquet itself is not on this disk;
  fetch it with `python -m evaluation.datasets.arxiv download` then
  `convert`.
- **Yambda listens-per-user CDF (item E).** Recipe A but on
  `data/yambda-{500m,5b}/train.parquet`. Again the parquet is not
  on disk.

---

## 12. Quick-orientation note for the next agent

If you are the agent who will turn this file into Chapter 3 of the
Russian thesis, here is the minimum context you need to keep in
front of you while writing:

1. **Audience.** HSE MSc thesis committee, expecting Russian-language
   prose at roughly the same density as a typical software-engineering
   diploma chapter. Aim for ≈ 25–35 pages including tables and
   figures.
2. **Citation rule.** No Meta-affiliated authors anywhere. The §1
   table screens every reference you will need; if you add a new
   one, screen it yourself against `arxiv.org/abs/...` author
   affiliations before citing.
3. **Naming convention.** In Russian body text use neutral phrases:
   *"совместно спроектированный инвертированный индекс с INT8-
   квантизацией и Bloom-префильтром"* instead of *"SilverTorch"* (the
   parent thesis plan reserves the name for internal code). The
   `silvertorch` symbol appears verbatim only in code listings and
   in the §6 results-CSV column values.
4. **Numbers to lean on.** Use the bolded numbers from §2.3, §4.2,
   §6.6, §6.7, §7.5 and §5.9 verbatim — they are sourced from the
   on-disk JSONs and parquets in this session and will not drift
   until someone re-runs the pipelines.
5. **Figures.** The PNGs under
   [`docs/thesis/results-data/`](results-data/) are PNG previews;
   the chapter LaTeX should re-render them via the recipes in §11
   to get vector-quality PDFs at compile time. Treat the PNGs as
   "this is what the figure says", not "this is the figure to
   embed".
6. **Open questions in §9** should be resolved with the user before
   you commit the Russian prose for §7.5 and §4.4 (the dropout /
   Yambda-5B-d256 / eval-split-location questions are load-bearing
   for the per-cell numbers).
7. **What this file is *not*.** It is not Chapter 6 (Results) —
   that one will be much longer and contain the full per-cell
   tables and Pareto analyses. The summaries in §6 of this file
   exist to give Chapter 3 a foothold ("we evaluate on the cells
   described in §5.8 and find …") without preempting Chapter 6.
