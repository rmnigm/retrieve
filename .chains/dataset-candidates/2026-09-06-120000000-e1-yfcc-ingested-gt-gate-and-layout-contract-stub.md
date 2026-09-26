---
chain: "dataset-candidates"
branch: "main"
parent: "2026-09-05-120000000-survey-and-decision-phase-e-datasets.md"
nextStep: "Run E1's cells on the harness v2 (one none + one filter cell) when the GPU is free; E2 deferred (disk, and no PCA)."
created: "2026-09-06T12:00:00Z"
---

# E1: YFCC-10M downloaded, loader written, shipped-GT gate passed; layout-contract stub

## YFCC-10M ingest (roadmap E1, 2026-09-06)
All six URLs under `https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/` served HTTP/2 200 with matching sizes: `base.10M.u8bin` 1,920,000,008 B, `query.public.100K.u8bin` 19,200,008 B, `GT.public.ibin` 8,000,008 B, `base.metadata.10M.spmat` 945,683,840 B, `query.metadata.public.100K.spmat` 1,907,024 B, `unfiltered.GT.public.ibin` 80,000,008 B. (The user's earlier attempt had failed; a HEAD from the Mac on 2026-09-05 returned 200.) Loader `evaluation/eval_datasets/yfcc.py` + `yfcc_check_gt.py` (now `etl/`). Answers to the open questions: the vocabulary exposes no prefix, so the tag bag stays one clause duplicated across two slots (query tag j -> clause j); the tail is long (max 1,517 tags on one item), so K capped at 32 with the truncation in `prep_log.json`. Deviations carried into the paper: the capped clause tensor makes the harness predicate stricter than the shipped one on 25.8 % of queries (the shipped GT is validated against the uncapped CSR instead), and the harness scores cosine while the shipped GT is squared L2. Both documented in `docs/system/datasets.md#yfcc10m`.

GT gate record (A100, `development` @ `41d4479`): our exact filtered oracle (conjunctive AND over the uncapped tag CSR, squared L2 in fp32, TF32 off) reproduces the shipped `GT.public.ibin` on 100,000 / 100,000 queries, all id-exact, `max_abs_distance_error = 0.0`, 1276 s wall (`docs/artifacts/dataset-candidates/yfcc10m/gt_check-cuda-100k.json`); a 200-query CPU subset had reproduced it bit-exactly earlier. The "one none + one filter cell" half deferred to E5 (user: heavy evals later).

## E2 deferred (2026-09-06, user)
No PCA, native 768-d only, heavy ETL / evals later; loader skeleton on `dev/e2-pubmed` (download / verify / medline / convert / attrs / queries / encode_queries, CPU tests, `pinkmeme/eval-pubmed` registry entry unpublished). No data staged: the ~198 GB raw mirror did not fit the then-understood `/workspace` quota.

## §6 layout contract for the E-phase loaders (stub, from the harness review §1.11)
Plan text only at the time. The on-disk contract was implicit and re-derived by every loader; PubMed's `cmd_queries` dropped empty-title rows and could append NFCorpus rows (misaligning `query_emb.pt`), YFCC omitted the sidecars and relied on a warning. The missing shared thing is the contract, not a base class: `eval_datasets/layout.py` with `write_query_set`, `merge_prep_log`, `validate_layout(data_dir, content_dir) -> list[str]` behind `bench check --dataset X`; `retry_download` / `checksum` / `year_to_bucket` into `common.py`; `goodreads.ROOT` through `data_root()`; an optional multi-target `item_ids` column; an explicit prefix policy (`prefix: null` = no prefix, a missing sidecar an error); rebase E1 / E2 onto the C3 harness. Gate: `bench check` green on arXiv, Goodreads and every E-phase dataset before its first campaign cell; E2 also needs fp16 items + a chunked oracle (36 M x 768 fp32 does not fit). Landed at C5 (`layout.py`, `bench check`) and 2026-09-16 (prefix policy); `write_query_set`, `merge_prep_log`, the helper dedup and the multi-target column still open.
