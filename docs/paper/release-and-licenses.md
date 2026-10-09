---
title: release-and-licenses
created: 2026-10-10
updated: 2026-10-10
type: summary
tags: [paper, release]
sources: [https://github.com/MengtingWan/goodreads, https://info.arxiv.org/help/license/index.html, https://huggingface.co/datasets/open-index/open-arxiv, https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/README.txt, https://www.nlm.nih.gov/databases/download/terms_and_conditions.html, https://www.ncbi.nlm.nih.gov/home/about/policies/, https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md, https://laion.ai/blog/relaion-5b/, https://github.com/meta-recsys/silvertorch, https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers]
---

# Licenses, redistribution, and the public-release plan

Roadmap REL-LIC; it feeds F4 (packaging) and the Availability rows of
[ecir-criteria.md](ecir-criteria.md). Part 1 covers each input we use: its
license or terms, read from the source's own page; whether the derived files
we stage may be shared, or only the recipe that builds them; and what
attribution we owe. Part 2 covers what the user currently has on the Hub and
GitHub, what each repo must not contain, and the double-blind plan.

**Making anything public is the user's decision.** This page is a plan:
nothing has been made public, deleted or changed. The Hub and GitHub state
below was read through the API on 2026-10-10. This is not legal advice. When
a source's wording leaves a question open, the page says so and lists it
under [decisions for the user](#5-decisions-for-the-user).

## 1. Datasets

"Derived" means the files our ETL writes: embeddings, attribute tensors,
vocabularies, held-out splits, oracles. "Recipe" means the `eval-data`
command plus the checksums of what it produces.

| dataset | source | license / terms (as stated by the source) | may we share the derived data? | attribution owed |
|---|---|---|---|---|
| **goodreads** (work-id, E1c encoder) | UCSD Book Graph ([repo README](https://github.com/MengtingWan/goodreads), [site](https://mengtingwan.github.io/data/goodreads.html)) | No license. The README says: "**We collected these datasets for academic use only! Please do not redistribute them or use for commercial purposes.**" | **No: recipe only.** Our trainer parquets hold the users' interaction sequences, and our attrs and vocabularies (genre, language, format, year, author names) come from the catalog. Both are redistribution. See the checkpoint row and [§5](#5-decisions-for-the-user) | Cite Wan & McAuley, RecSys '18 and Wan et al., ACL '19 (the README asks for both) |
| **arXiv** papers + attrs | HF [`open-index/open-arxiv`](https://huggingface.co/datasets/open-index/open-arxiv), a parquet copy of the Cornell Kaggle arXiv metadata dump | The card says CC0-1.0. arXiv's [license page](https://info.arxiv.org/help/license/index.html) says: "A Creative Commons CC0 1.0 Universal Public Domain Dedication will apply to all metadata." We use metadata only (title, abstract, categories, license field, dates, versions, authors). The per-paper licenses the card tabulates cover the papers, not the metadata | **Yes**: embeddings, attrs, vocabularies, held-out split, oracles | None is required (CC0). Courtesy: arXiv, the Cornell Kaggle dataset, open-index. The encoder is `nomic-embed-text-v1.5` (Apache-2.0). We ship its outputs, not its weights, so no notice is owed. We name it because the vectors depend on it |
| **PubMed / MedCPT** 10 M slice | NCBI FTP [MedCPT article embeddings](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/README.txt) + the MEDLINE baseline + MeSH `desc2026` | The embeddings directory has a README but no license file. The MedCPT encoders' LICENSE (HF `ncbi/MedCPT-*`) is NCBI's public-domain notice: a US Government Work, no restriction on use or reproduction. NCBI's [policy page](https://www.ncbi.nlm.nih.gov/home/about/policies/) says the same of NCBI-produced data and adds that "NLM does not claim the copyright on the abstracts in PubMed; however, journal publishers or authors may." The [NLM download terms](https://www.nlm.nih.gov/databases/download/terms_and_conditions.html) require the attribution "Courtesy of the U.S. National Library of Medicine", forbid implying NLM endorsement, and require redistributed data either to be kept current or to state that it is not | **Yes for vectors and attrs**: item and query embeddings, MeSH / journal / year / has-abstract attrs, PMID list, oracles. **Not the text**: titles and abstracts (`articles.parquet`, and `queries.parquet`, whose query text is a held-out title) may be copyright of the publisher; share PMIDs instead. The slice rebuilds deterministically (63 min, CPU; [datasets](../system/datasets.md#pubmed)), so recipe + checksums is enough (F4) | "Courtesy of the U.S. National Library of Medicine"; a statement that the data is the 2026 baseline and not current; cite MedCPT (Jin et al. 2023, arXiv:2307.00589; both the README and the LICENSE ask for it) |
| NFCorpus (optional PubMed query set) | BEIR | The HF card says CC-BY-SA-4.0; BEIR asks that the corpus not be redistributed ([datasets](../system/datasets.md#query-sets)) | **No**, and no campaign cell reads it | Only if it is used |
| **YFCC-10M** | Big-ANN NeurIPS'23 filtered track (`dl.fbaipublicfiles.com`) | **CC BY 4.0**, in the track table of the [neurips23 README](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/neurips23/README.md). The images and their own licenses stay with YFCC100M; we use only the organisers' CLIP descriptors, tags and GT | **Yes**, with attribution and a note of our changes (the tag-CSR layout, `eval_split`). The upstream files are public and need no registration, so the recipe is enough | CC BY 4.0: credit the Big-ANN NeurIPS'23 organisers (Simhadri et al.) and YFCC100M (Thomee et al., CACM 2016); link the license; state our changes |
| **Re-LAION-2B-en-research-safe** (V-LAION30, 30 M, d256) | HF `laion/relaion2B-en-research-safe`, gated, auto-approved | **Apache-2.0** ([LAION's release post](https://laion.ai/blog/relaion-5b/): "We release both Re-LAION-5B-research and Re-LAION-5B-research-safe under Apache 2.0 License"). The same post says the data is "released for research purposes", advises "strongly AGAINST" industrial use and end products, and gates access on "submission of affiliation information and consent". The captions and URLs point at third-party images and text whose copyright LAION does not hold | **Vectors and tag buckets: yes under Apache-2.0**, ideally behind the same kind of gate (see [§5](#5-decisions-for-the-user)). This covers our caption embeddings, the size / similarity / punsafe / pwatermark buckets, the domain vocabulary and the row `key`s. **Never the captions or URLs**: they live only in `items.parquet` / `queries.parquet` under `_raw/relaion/laion30m/`, outside the staged `data/laion30m/` that `eval-data publish` reads (the staged dir holds vectors, coded attrs, `vocab.json` with the domains and buckets, and the targets), and `EVAL_IGNORE_PATTERNS` excludes both file names as well | Apache-2.0: keep the license and name LAION; cite the Re-LAION post (its BibTeX `@misc{relaion}`); name `nomic-embed-text-v1.5` |
| **synth attrs** (`synth-filter`, uniform and cluster-correlated) | Ours: a seeded `u_i ~ U(0,1)` per item, and k-means cluster ids over the parent's embeddings | Ours. The files hold no third-party content. The correlated variant's centroids are computed from the parent's vectors, so they share the parent's terms (arXiv: CC0) | **Recipe**: they are not on the Hub and rebuild in seconds from the seed ([datasets](../system/datasets.md#synthetic-selectivity-attrs)). Share the `synth_filter.json` / `synth_corr.json` checksums | — |
| **trained checkpoints**: goodreads `sasrec-ssm-logq-d{64,128,256}` (E1c, read by the harness) and the older `gsasrec-d*-drop0.5-id` | Ours, trained on goodreads ([checkpoints](../system/checkpoints.md)) | Our code is Apache-2.0. The weights include a per-item embedding table learned from the users' sequences, so they are derived from data the source asks not to redistribute. The terms do not mention models | **Open question.** Publishing the weights is the only way to replicate the goodreads cells exactly: retraining is not bit-reproducible, because ties between one user's events at the same second reorder ([datasets](../system/datasets.md#goodreads)). The options are in [§5](#5-decisions-for-the-user) | As goodreads |
| out of the study: yambda-500m / -5b, KuaiRand, OpenAlex | HF `yandex/yambda`; Zenodo; OpenAlex S3 | Apache-2.0 (yambda card); CC BY 4.0 ([datasets](../system/datasets.md#kuairand)); CC0 | Allowed, but no paper exhibit needs them | Only if published |

## 2. Third-party code

| code | license | what we ship | obligations |
|---|---|---|---|
| Meta's `meta-recsys/silvertorch` at the pin `21aa35e` ([pyproject.toml](../../pyproject.toml)) | **Apache-2.0** (the repo's `LICENSE` at the pin, file headers "Copyright (c) Meta Platforms, Inc. and affiliates", README "License: Apache 2.0"; no `NOTICE` file) | Not vendored: `uv` installs it from git. F4 puts the pinned sdist in the Zenodo record | Ship its `LICENSE` with the sdist, keep the file headers, say it is unmodified, and state that Meta does not endorse the work (Apache-2.0 §6 grants no trademark use) |
| `evaluation/eval_datasets/etl/yambda.py` (and `timesplit.py`, which comes with it) | Adapted from Yandex's yambda benchmark code (`yambda.py` docstring). The yambda HF card says Apache-2.0 | In our repo | Apache-2.0 §4: keep the attribution and note the changes. The docstring names the source but not its license; one line at F4 fixes that |
| `nomic-embed-text-v1.5`, `ncbi/MedCPT-*` | Apache-2.0; public domain | Not redistributed (downloaded at encode time) | None for outputs; cite them |
| **ours**: `retrieve/` + `evaluation/` | Apache-2.0, root [LICENSE](../../LICENSE) ("Copyright 2026" + the author's name) | GitHub, and the `torchretrieve` sdist (F4) | Two F4 items. The `retrieve/` sdist has no `LICENSE` file and no `license` field in [retrieve/pyproject.toml](../../retrieve/pyproject.toml), so the sdist would ship unlicensed. `torchretrieve` is not on PyPI yet (the JSON API returned 404 on 2026-10-10); the first release is F4 |

## 3. What is on the Hub and GitHub now

Read with `HfApi().list_datasets(author="pinkmeme")` / `list_models` and
`repo_info`. GitHub was read through its API. Sizes are what the Hub reports.

| repo | visibility now | holds | at submission (plan) | must not hold when public |
|---|---|---|---|---|
| GitHub `rmnigm/retrieve` | **public**, Apache-2.0 | the code, docs and full git history | stays public; the anonymous mirror serves review ([§4](#4-double-blind-review)) | tokens (scan the history once before F4); goodreads data (none is in git) |
| `pinkmeme/eval-results` (dataset) | private | 5,323 files, 2.28 GB: campaign records, samples, `results.parquet`, `MANIFEST.json`s, `artifacts/<plan>/`, `legacy/` | **public**, minus `artifacts/chains/` and `artifacts/d1-campaign/chain/`. Alternative: a new repo holding only the subtrees the final manifest names (F-REPRO's `campaign-final/` + the reused ones) | `artifacts/chains/` (32 note branches) and `artifacts/d1-campaign/chain/`: orchestrator handoff notes, which the [contract](../contracts/agent-orchestration.md#where-briefs-instructions-and-reports-live) keeps out of the repository. They name the account, the thesis, `ssh` usage and `/root` paths |
| `pinkmeme/eval-arxiv-papers` (dataset) | private | 3.14 GB: d256/d128/d64 embeddings, attrs (legacy `[N+1, …]` layout), vocabularies, held-out split | **public** (CC0 source) | nothing licence-bound. Its README is stale: it links `github.com/anthropics/retrieve` and a `your-fork` placeholder, lists `papers.parquet` and wide files the repo does not hold, gives a `python -m data.arxiv` command that no longer exists, and states no license. Rewrite the card before it goes public (CC0, sources, encoder, layout) |
| `pinkmeme/eval-yfcc10m` (dataset) | private | 9.82 GB: `content_d192`, attrs, tag CSR, `gt_shipped.pt`, GT checks, held-out split | public under CC BY 4.0 with an attribution card, **or** left private with the recipe as the path (upstream is public) | no card yet: add CC BY 4.0, the credits and our changes. [datasets](../system/datasets.md#huggingface-io) said this repo was "not published"; it holds the staged files (fixed there in this commit) |
| `pinkmeme/eval-goodreads-work-id` (dataset) | private | 9.66 GB: `trainer/` (interaction sequences), `test.parquet`, attrs, vocabularies, `item_id_map.json`, 6 checkpoints | **stays private** ([§1](#1-datasets)); at most the checkpoints, if the user decides so | any interaction data, catalog attrs or vocabularies |
| `pinkmeme/eval-laion30m` (registered on `dev/v-laion30`, not created yet) | — | — | gated, if the user publishes LAION at all | captions, URLs (`items.parquet`, `queries.parquet`) |
| `pinkmeme/eval-pubmed` (registered, not created) | — | — | a vectors-only repo or none (recipe + checksums) | titles, abstracts |
| `pinkmeme/eval-yambda-500m`, `-5b`, `eval-kuairand` (datasets) | private | trainer inputs + checkpoints (19.6 / 8.9 / 0.5 GB) | stay private: out of the study | — |
| `pinkmeme/arxiv-retrieval` (dataset), `goodreads-gsasrec`, `yambda-500m-gsasrec`, `yambda-5b-gsasrec` (models) | private | only `.gitattributes`: empty shells | stay private or are deleted (user) | — |
| `pinkmeme/retrieval-filter-evals-2026-05-20`, `-05-23` (datasets) | **public**, MIT | the thesis-era result JSONs and configs, including LiNR V4 rows; the card names `/workspace/retrieve/evaluation/` | the user decides: they predate the harness rewrite, and none of their numbers is paper material ([provenance](provenance-and-disclosure.md#5-what-was-compared-with-what--and-what-is-not-claimed)). They identify the account during review | — |

**Leak scan of `eval-results`.** Every text file under 20 MB was downloaded
(311 MB: `.json`, `.jsonl`, `.md`, `.log`, `.yaml`, `.csv`, `.py`, `.sh`).
The scan printed only counts and file names:

- no Hugging Face, Anthropic or GitHub token, no RunPod key, no private-key
  block, no `ip:port`, no e-mail address;
- `/workspace/` in 208 files, `/scratch/` in 900 and `/data/` in 50. These
  are the records' `path` / `inputs` fields and the logs: container paths,
  not personal ones;
- `/root/` and `ssh ` only in `artifacts/chains/` notes;
- `pinkmeme` and "thesis" in 21 files, all in `artifacts/chains/`,
  `artifacts/d1-campaign/chain/` or `legacy/` READMEs;
- `env.host` is the container id.

Binary files (parquet, `.pt`, samples) were not scanned. Neither were the
other repos, apart from their cards. The scan script is a list of grep
patterns: rerun it on the final subtree set before F4.

## 4. Double-blind review

The ECIR 2026 call (the 2027 call was not online on 2026-10-10) reviews
reproducibility papers double-blind, discourages arXiv preprints, and says
software goes "to an anonymous repository that is linked to in the
submission" or into EasyChair
([call](https://ecir2026.dryfta.com/calls/call-for-reproducibility-papers)).
Three things identify the author today. They are not secret, so the plan
keeps the paper from pointing at them rather than hiding them:

1. GitHub `rmnigm/retrieve` is public, and its LICENSE, commits and README
   carry the name and "my master's thesis";
2. the Hub account `pinkmeme` has two public result repos (and unrelated
   public models);
3. the library name `torchretrieve` points at both.

**At review time (plan):**

- **Code**: an [anonymous.4open.science](https://anonymous.4open.science)
  mirror of the GitHub repo at the submission tag. Its term list replaces
  the author's name, `rmnigm`, `pinkmeme`, the e-mail address and "thesis".
  The paper links only the mirror and names the package neutrally (or as
  `torchretrieve`, if the user accepts that a search finds GitHub). The
  mirror rewrites the `huggingface.co/datasets/pinkmeme/…` links in docs, so
  they break. Reviewers get the data another way (next two items).
- **Results**: one archive with `results.parquet`, the `MANIFEST.json`s and
  `campaign.yaml` of the subtrees the paper cites. It goes into the mirror
  or EasyChair as supplementary material. It is small: the whole
  `eval-results` is 2.28 GB, and the cited subset is a fraction of that.
- **Data**: the recipes (`eval-data … all`) and checksums. The derived arXiv
  and YFCC files are shared through an unlinked, anonymous Hub account, or
  not at all, since both rebuild from public sources (arXiv needs the GPU
  encode, about hours). Goodreads stays recipe-only, plus the checkpoint
  question in §5.

**After acceptance:** the repos marked "public" in §3 flip; a tagged
`torchretrieve` release goes to PyPI; a Zenodo DOI holds the tag, the
results subset and Meta's pinned sdist, which answers ECIR's "permanent
repository" question; the paper's links switch to the real repos and the
DOI. All of this is F4.

## 5. Decisions for the user

1. **Goodreads checkpoints.** Choose one: (a) ask the dataset authors
   (McAuley lab) whether the trained weights may be shared, and publish them
   if they agree; (b) keep the weights private and ship the training recipe.
   Under (b) the goodreads cells are reproducible only up to retraining
   noise, which is an Availability gap in [ecir-criteria](ecir-criteria.md);
   (c) publish the weights, reading the README's "do not redistribute" as
   covering the data and not models trained on it. (a) is the clean one, and
   the replies take time, like the author contact on the
   [roadmap](../roadmap.md#needs-the-user).
2. **Re-LAION derived vectors.** Choose between a gated Hub repo with
   LAION's consent wording, an ungated repo (Apache-2.0 allows it), or
   recipe only.
3. **PubMed slice.** Choose between recipe + checksums and a vectors-only
   repo (17 GB, with the NLM statement). F4 currently says "or".
4. **The public May result repos** (`retrieval-filter-evals-2026-05-*`): keep
   them, make them private during review, or delete them.
5. **`eval-results` when public**: the existing repo with the note subtrees
   deleted, or a fresh repo with only the cited subtrees. A fresh repo also
   drops the history, which still holds the notes after a delete.
6. **GitHub during review**: the repo stays public (the mirror suffices), or
   becomes private until acceptance.
