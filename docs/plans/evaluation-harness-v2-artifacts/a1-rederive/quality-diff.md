# A1 golden re-derive — quality diff

old: A1's cells as committed on `development` (`git show development:evaluation/golden/<cell>.json`), 99 rows  
new: `evaluation/golden/` on this branch, 99 rows

## How to read this

Both sides are the **same harness** (`origin/dev/a1-golden`, `1ccdb27`) on the
same box, against the same staged datasets, with the same `encoded_queries`
blob and the same cached oracle (`gt_topk_v3_*`, fingerprint hit — see the
per-cell logs in `evaluation/golden/_logs/`). The **only** difference is the
library tree: A1 ran `70bafc4:retrieve` (`6dc72aac`), this run ran
`4f52972:retrieve` (`28fda5ae`). So every delta below is a library delta.

Roadmap A1's predicted delta was: *SilverTorch-triton on rows with fewer than
`k` survivors moves down (the `-1` sentinel), torch and official unchanged.*
**That prediction did not hold** — see `evaluation-harness-v2.md`'s re-derive
record for the analysis. Three facts from this diff:

1. Among the **9,859 kept** goodreads users every query has ≥ 1000 filter
   survivors (the 141 dropped users are exactly the zero-survivor ones), so
   the `-1` sentinel cannot move *any* goodreads number. On arxiv only
   4 / 6 / 11 of 10,000 rows have fewer than 100 / 500 / 1000 survivors.
2. `silvertorch-torch` moved, and moved **up** — the deterministic k-means
   is backend-independent, so it changes the index on both backends.
3. `linr_v2-triton` and `linr_v3-triton` moved; their `torch` counterparts
   did not. These are exactly the two algos C4 fix (i) touched
   (`clause_compact` / `bloom_compact` as opaque custom ops).

Nothing here was adjusted to make the numbers agree.

## Quality deltas (new - old), rows where any column moved

| cell | k | bs | column | old | new | delta |
|---|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | ndcg@100 | 0.913703888 | 0.913732553 | +2.867e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | precision@100 | 0.883854982 | 0.883889982 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | recall@100 | 0.884007068 | 0.884042068 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | ndcg@100 | 0.913703888 | 0.913732553 | +2.867e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | precision@100 | 0.883854982 | 0.883889982 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | recall@100 | 0.884007068 | 0.884042068 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | ndcg@100 | 0.913703888 | 0.913732553 | +2.867e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | precision@100 | 0.883854982 | 0.883889982 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | recall@100 | 0.884007068 | 0.884042068 | +3.500e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | ndcg@500 | 0.875400891 | 0.875417934 | +1.704e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | precision@500 | 0.847079040 | 0.847098041 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | recall@500 | 0.847281392 | 0.847300391 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | ndcg@500 | 0.875400891 | 0.875417934 | +1.704e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | precision@500 | 0.847079040 | 0.847098041 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | recall@500 | 0.847281392 | 0.847300391 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | ndcg@500 | 0.875400891 | 0.875417934 | +1.704e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | precision@500 | 0.847079040 | 0.847098041 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | recall@500 | 0.847281392 | 0.847300391 | +1.900e-05 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 1 | ndcg@1000 | 0.848329870 | 0.848329757 | -1.130e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 1 | precision@1000 | 0.819421339 | 0.819422139 | +8.002e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 1 | recall@1000 | 0.819665465 | 0.819666078 | +6.135e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 8 | ndcg@1000 | 0.848329870 | 0.848329757 | -1.130e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 8 | precision@1000 | 0.819421339 | 0.819422139 | +8.002e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 8 | recall@1000 | 0.819665465 | 0.819666078 | +6.135e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 16 | ndcg@1000 | 0.848329870 | 0.848329757 | -1.130e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 16 | precision@1000 | 0.819421339 | 0.819422139 | +8.002e-07 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 16 | recall@1000 | 0.819665465 | 0.819666078 | +6.135e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | ndcg@100 | 0.999482644 | 0.999479007 | -3.636e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | precision@100 | 0.999278828 | 0.999273756 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | recall@100 | 0.999278832 | 0.999273761 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | ndcg@100 | 0.999482644 | 0.999479007 | -3.636e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | precision@100 | 0.999278828 | 0.999273756 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | recall@100 | 0.999278832 | 0.999273761 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | ndcg@100 | 0.999482644 | 0.999479007 | -3.636e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | precision@100 | 0.999278828 | 0.999273756 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | recall@100 | 0.999278832 | 0.999273761 | -5.072e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | ndcg@500 | 0.999503385 | 0.999504025 | +6.409e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | precision@500 | 0.999371547 | 0.999372358 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | recall@500 | 0.999371546 | 0.999372357 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | ndcg@500 | 0.999503385 | 0.999504025 | +6.409e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | precision@500 | 0.999371547 | 0.999372358 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | recall@500 | 0.999371546 | 0.999372357 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | ndcg@500 | 0.999503385 | 0.999504025 | +6.409e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | precision@500 | 0.999371547 | 0.999372358 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | recall@500 | 0.999371546 | 0.999372357 | +8.114e-07 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | ndcg@1000 | 0.999501245 | 0.999503561 | +2.315e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | precision@1000 | 0.999388209 | 0.999391048 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | recall@1000 | 0.999388181 | 0.999391021 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | ndcg@1000 | 0.999501245 | 0.999503561 | +2.315e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | precision@1000 | 0.999388209 | 0.999391048 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | recall@1000 | 0.999388181 | 0.999391021 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | ndcg@1000 | 0.999501245 | 0.999503561 | +2.315e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | precision@1000 | 0.999388209 | 0.999391048 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | recall@1000 | 0.999388181 | 0.999391021 | +2.840e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | ndcg@100 | 0.909125182 | 0.909142778 | +1.760e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | precision@100 | 0.876978375 | 0.877002718 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | recall@100 | 0.876978397 | 0.877002741 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | ndcg@100 | 0.909125182 | 0.909142778 | +1.760e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | precision@100 | 0.876978375 | 0.877002718 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | recall@100 | 0.876978397 | 0.877002741 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | ndcg@100 | 0.909125182 | 0.909142778 | +1.760e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | precision@100 | 0.876978375 | 0.877002718 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | recall@100 | 0.876978397 | 0.877002741 | +2.434e-05 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | ndcg@500 | 0.774973322 | 0.774979390 | +6.068e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | precision@500 | 0.724367414 | 0.724373905 | +6.491e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | recall@500 | 0.724367380 | 0.724373872 | +6.492e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | ndcg@500 | 0.774973322 | 0.774979390 | +6.068e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | precision@500 | 0.724367414 | 0.724373905 | +6.491e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | recall@500 | 0.724367380 | 0.724373872 | +6.492e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | ndcg@500 | 0.774973322 | 0.774979390 | +6.068e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | precision@500 | 0.724367414 | 0.724373905 | +6.491e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | recall@500 | 0.724367380 | 0.724373872 | +6.492e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | ndcg@1000 | 0.676167415 | 0.676169331 | +1.916e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | precision@1000 | 0.617543695 | 0.617545825 | +2.130e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | recall@1000 | 0.617543665 | 0.617545795 | +2.130e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | ndcg@1000 | 0.676167415 | 0.676169331 | +1.916e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | precision@1000 | 0.617543695 | 0.617545825 | +2.130e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | recall@1000 | 0.617543665 | 0.617545795 | +2.130e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | ndcg@1000 | 0.676167415 | 0.676169331 | +1.916e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | precision@1000 | 0.617543695 | 0.617545825 | +2.130e-06 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | recall@1000 | 0.617543665 | 0.617545795 | +2.130e-06 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 1 | ndcg@100 | 0.935841322 | 0.935957277 | +1.160e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 1 | precision@100 | 0.912643246 | 0.912800463 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 1 | recall@100 | 0.912643274 | 0.912800491 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 8 | ndcg@100 | 0.935841322 | 0.935957277 | +1.160e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 8 | precision@100 | 0.912643246 | 0.912800463 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 8 | recall@100 | 0.912643274 | 0.912800491 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 16 | ndcg@100 | 0.935841322 | 0.935957277 | +1.160e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 16 | precision@100 | 0.912643246 | 0.912800463 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 16 | recall@100 | 0.912643274 | 0.912800491 | +1.572e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 1 | ndcg@500 | 0.876174591 | 0.876261928 | +8.734e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 1 | precision@500 | 0.847044365 | 0.847149853 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 1 | recall@500 | 0.847044325 | 0.847149812 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 8 | ndcg@500 | 0.876174591 | 0.876261928 | +8.734e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 8 | precision@500 | 0.847044365 | 0.847149853 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 8 | recall@500 | 0.847044325 | 0.847149812 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 16 | ndcg@500 | 0.876174591 | 0.876261928 | +8.734e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 16 | precision@500 | 0.847044365 | 0.847149853 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 16 | recall@500 | 0.847044325 | 0.847149812 | +1.055e-04 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 1 | ndcg@1000 | 0.823258731 | 0.823339080 | +8.035e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 1 | precision@1000 | 0.788911489 | 0.789005312 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 1 | recall@1000 | 0.788911451 | 0.789005274 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 8 | ndcg@1000 | 0.823258731 | 0.823339080 | +8.035e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 8 | precision@1000 | 0.788911489 | 0.789005312 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 8 | recall@1000 | 0.788911451 | 0.789005274 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 16 | ndcg@1000 | 0.823258731 | 0.823339080 | +8.035e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 16 | precision@1000 | 0.788911489 | 0.789005312 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 16 | recall@1000 | 0.788911451 | 0.789005274 | +9.382e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 1 | ndcg@100 | 0.935926551 | 0.935957277 | +3.073e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 1 | precision@100 | 0.912758877 | 0.912800463 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 1 | recall@100 | 0.912758904 | 0.912800491 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 8 | ndcg@100 | 0.935926551 | 0.935957277 | +3.073e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 8 | precision@100 | 0.912758877 | 0.912800463 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 8 | recall@100 | 0.912758904 | 0.912800491 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 16 | ndcg@100 | 0.935926551 | 0.935957277 | +3.073e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 16 | precision@100 | 0.912758877 | 0.912800463 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 16 | recall@100 | 0.912758904 | 0.912800491 | +4.159e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 1 | ndcg@500 | 0.876302516 | 0.876261928 | -4.059e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 1 | precision@500 | 0.847199350 | 0.847149853 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 1 | recall@500 | 0.847199310 | 0.847149812 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 8 | ndcg@500 | 0.876302516 | 0.876261928 | -4.059e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 8 | precision@500 | 0.847199350 | 0.847149853 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 8 | recall@500 | 0.847199310 | 0.847149812 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 16 | ndcg@500 | 0.876302516 | 0.876261928 | -4.059e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 16 | precision@500 | 0.847199350 | 0.847149853 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 16 | recall@500 | 0.847199310 | 0.847149812 | -4.950e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 1 | ndcg@1000 | 0.823403114 | 0.823339080 | -6.403e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 1 | precision@1000 | 0.789083210 | 0.789005312 | -7.790e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 1 | recall@1000 | 0.789083173 | 0.789005274 | -7.790e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 8 | ndcg@1000 | 0.823403114 | 0.823339080 | -6.403e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 8 | precision@1000 | 0.789083210 | 0.789005312 | -7.790e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 8 | recall@1000 | 0.789083173 | 0.789005274 | -7.790e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 16 | ndcg@1000 | 0.823403114 | 0.823339080 | -6.403e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 16 | precision@1000 | 0.789083210 | 0.789005312 | -7.790e-05 |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 16 | recall@1000 | 0.789083173 | 0.789005274 | -7.790e-05 |

## Per-cell summary

| cell | rows | quality cols moved | max |delta| | direction | n_users_kept old/new |
|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 9 | 27/36 | 3.500e-05 | mixed | [10000] / [10000] |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v2-torch | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v2-triton | 9 | 27/36 | 5.072e-06 | mixed | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v3-torch | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v3-triton | 9 | 27/36 | 2.434e-05 | up | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v4-torch | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-linr_v4-triton | 9 | 0/36 | 0.000e+00 | — | [9859] / [9859] |
| goodreads-d128-c0_genre-silvertorch-torch | 9 | 27/36 | 1.572e-04 | up | [9859] / [9859] |
| goodreads-d128-c0_genre-silvertorch-triton | 9 | 27/36 | 7.790e-05 | mixed | [9859] / [9859] |

## Latency, for context only (not clock-controlled, not a gate)

| cell | k | bs | median_ms old | new | ratio |
|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | 0.1868 | 0.1571 | 0.841× |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | 0.3097 | 0.3069 | 0.991× |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | 0.4701 | 0.4626 | 0.984× |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | 0.2012 | 0.2014 | 1.001× |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | 0.3197 | 0.3177 | 0.994× |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | 0.4790 | 0.4786 | 0.999× |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 1 | 0.2020 | 0.1717 | 0.850× |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 8 | 0.3198 | 0.3200 | 1.001× |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 16 | 0.4801 | 0.4776 | 0.995× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 100 | 1 | 0.4340 | 0.4812 | 1.109× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 100 | 8 | 0.9449 | 0.9435 | 0.998× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 100 | 16 | 1.6256 | 1.6229 | 0.998× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 1 | 0.4506 | 0.4477 | 0.994× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 8 | 0.9574 | 0.9560 | 0.999× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 16 | 1.6389 | 1.6335 | 0.997× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 1 | 0.4500 | 0.4509 | 1.002× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 8 | 0.9576 | 0.9551 | 0.997× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 16 | 1.6382 | 1.6344 | 0.998× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | 1 | 0.4600 | 0.5242 | 1.140× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | 8 | 1.0907 | 1.0887 | 0.998× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | 16 | 1.9004 | 1.8948 | 0.997× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | 1 | 0.4756 | 0.4741 | 0.997× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | 8 | 1.1021 | 1.1009 | 0.999× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | 16 | 1.9098 | 1.9046 | 0.997× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 1 | 0.5475 | 0.4781 | 0.873× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 8 | 1.1028 | 1.1010 | 0.998× |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 16 | 1.9136 | 1.9060 | 0.996× |
| goodreads-d128-c0_genre-linr_v2-torch | 100 | 1 | 0.9320 | 0.8218 | 0.882× |
| goodreads-d128-c0_genre-linr_v2-torch | 100 | 8 | 4.6478 | 4.6465 | 1.000× |
| goodreads-d128-c0_genre-linr_v2-torch | 100 | 16 | 9.1234 | 9.1240 | 1.000× |
| goodreads-d128-c0_genre-linr_v2-torch | 500 | 1 | 0.9088 | 0.8390 | 0.923× |
| goodreads-d128-c0_genre-linr_v2-torch | 500 | 8 | 4.6556 | 4.6600 | 1.001× |
| goodreads-d128-c0_genre-linr_v2-torch | 500 | 16 | 9.1410 | 9.1331 | 0.999× |
| goodreads-d128-c0_genre-linr_v2-torch | 1000 | 1 | 0.8409 | 0.8399 | 0.999× |
| goodreads-d128-c0_genre-linr_v2-torch | 1000 | 8 | 4.6580 | 4.6598 | 1.000× |
| goodreads-d128-c0_genre-linr_v2-torch | 1000 | 16 | 9.1373 | 9.1424 | 1.001× |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | 0.3679 | 0.3490 | 0.948× |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | 1.6625 | 1.6260 | 0.978× |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | 3.2321 | 3.1735 | 0.982× |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | 0.4577 | 0.3635 | 0.794× |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | 1.6842 | 1.6432 | 0.976× |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | 3.2532 | 3.1944 | 0.982× |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | 0.4602 | 0.3989 | 0.867× |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | 1.6953 | 1.6478 | 0.972× |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | 3.2561 | 3.1960 | 0.982× |
| goodreads-d128-c0_genre-linr_v3-torch | 100 | 1 | 0.4742 | 0.5488 | 1.157× |
| goodreads-d128-c0_genre-linr_v3-torch | 100 | 8 | 1.8755 | 1.8741 | 0.999× |
| goodreads-d128-c0_genre-linr_v3-torch | 100 | 16 | 3.4024 | 3.3966 | 0.998× |
| goodreads-d128-c0_genre-linr_v3-torch | 500 | 1 | 0.5834 | 0.4893 | 0.839× |
| goodreads-d128-c0_genre-linr_v3-torch | 500 | 8 | 1.8952 | 1.8888 | 0.997× |
| goodreads-d128-c0_genre-linr_v3-torch | 500 | 16 | 3.4184 | 3.4133 | 0.999× |
| goodreads-d128-c0_genre-linr_v3-torch | 1000 | 1 | 0.5791 | 0.5760 | 0.995× |
| goodreads-d128-c0_genre-linr_v3-torch | 1000 | 8 | 1.8919 | 1.8912 | 1.000× |
| goodreads-d128-c0_genre-linr_v3-torch | 1000 | 16 | 3.4193 | 3.4142 | 0.999× |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | 0.4347 | 0.4710 | 1.084× |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | 1.3807 | 1.3038 | 0.944× |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | 2.4714 | 2.3806 | 0.963× |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | 0.5496 | 0.4904 | 0.892× |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | 1.4048 | 1.3291 | 0.946× |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | 2.4867 | 2.3966 | 0.964× |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | 0.4720 | 0.4283 | 0.907× |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | 1.4036 | 1.3326 | 0.949× |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | 2.4864 | 2.3989 | 0.965× |
| goodreads-d128-c0_genre-linr_v4-torch | 100 | 1 | 0.5993 | 0.5969 | 0.996× |
| goodreads-d128-c0_genre-linr_v4-torch | 100 | 8 | 1.1424 | 1.1389 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-torch | 100 | 16 | 1.8444 | 1.8392 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-torch | 500 | 1 | 0.6142 | 0.7265 | 1.183× |
| goodreads-d128-c0_genre-linr_v4-torch | 500 | 8 | 1.1539 | 1.1506 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-torch | 500 | 16 | 1.8551 | 1.8523 | 0.999× |
| goodreads-d128-c0_genre-linr_v4-torch | 1000 | 1 | 0.7399 | 0.6123 | 0.827× |
| goodreads-d128-c0_genre-linr_v4-torch | 1000 | 8 | 1.1551 | 1.1513 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-torch | 1000 | 16 | 1.8556 | 1.8509 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-triton | 100 | 1 | 0.6273 | 0.6260 | 0.998× |
| goodreads-d128-c0_genre-linr_v4-triton | 100 | 8 | 1.3012 | 1.3024 | 1.001× |
| goodreads-d128-c0_genre-linr_v4-triton | 100 | 16 | 2.5555 | 2.0788 | 0.813× |
| goodreads-d128-c0_genre-linr_v4-triton | 500 | 1 | 0.6470 | 0.6381 | 0.986× |
| goodreads-d128-c0_genre-linr_v4-triton | 500 | 8 | 1.3123 | 1.3082 | 0.997× |
| goodreads-d128-c0_genre-linr_v4-triton | 500 | 16 | 2.1000 | 2.0923 | 0.996× |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 1 | 0.6462 | 0.7704 | 1.192× |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 8 | 1.3135 | 1.3082 | 0.996× |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 16 | 2.1002 | 2.0908 | 0.996× |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 1 | 0.6128 | 0.6266 | 1.023× |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 8 | 3.4215 | 3.4513 | 1.009× |
| goodreads-d128-c0_genre-silvertorch-torch | 100 | 16 | 6.8018 | 6.8599 | 1.009× |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 1 | 0.6074 | 0.6434 | 1.059× |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 8 | 3.4368 | 3.4701 | 1.010× |
| goodreads-d128-c0_genre-silvertorch-torch | 500 | 16 | 6.8182 | 6.8714 | 1.008× |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 1 | 0.6025 | 0.6495 | 1.078× |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 8 | 3.4635 | 3.4660 | 1.001× |
| goodreads-d128-c0_genre-silvertorch-torch | 1000 | 16 | 6.8771 | 6.8680 | 0.999× |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 1 | 0.2107 | 0.2099 | 0.996× |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 8 | 0.4509 | 0.4478 | 0.993× |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | 16 | 0.8570 | 0.8518 | 0.994× |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 1 | 0.2721 | 0.2711 | 0.996× |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 8 | 0.4686 | 0.4632 | 0.989× |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | 16 | 0.8727 | 0.8663 | 0.993× |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 1 | 0.2777 | 0.2580 | 0.929× |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 8 | 0.4767 | 0.4638 | 0.973× |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | 16 | 0.8859 | 0.8687 | 0.981× |

## Memory columns

| cell | k | bs | column | old | new |
|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | peak_mem_mib | 2248.23779296875 | 2248.24560546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 1 | index_mem_mib | 421.2431640625 | 421.2509765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | peak_mem_mib | 2263.11279296875 | 2263.12060546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 8 | index_mem_mib | 421.2431640625 | 421.2509765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | peak_mem_mib | 2280.11279296875 | 2280.12060546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | 16 | index_mem_mib | 421.2431640625 | 421.2509765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | peak_mem_mib | 2248.22998046875 | 2248.24560546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 1 | index_mem_mib | 429.3603515625 | 429.3759765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | peak_mem_mib | 2263.10498046875 | 2263.12060546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 8 | index_mem_mib | 429.3603515625 | 429.3759765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | peak_mem_mib | 2280.10498046875 | 2280.12060546875 |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | 16 | index_mem_mib | 429.3603515625 | 429.3759765625 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 1 | peak_mem_mib | 2669.48095703125 | 2669.49658203125 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 8 | peak_mem_mib | 2684.35595703125 | 2684.37158203125 |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | 16 | peak_mem_mib | 2701.35595703125 | 2701.37158203125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 1 | peak_mem_mib | 684.72705078125 | 879.32763671875 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 8 | peak_mem_mib | 699.60205078125 | 894.20263671875 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 500 | 16 | peak_mem_mib | 716.60205078125 | 911.20263671875 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 1 | peak_mem_mib | 879.32763671875 | 684.72705078125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 8 | peak_mem_mib | 894.20263671875 | 699.60205078125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | 1000 | 16 | peak_mem_mib | 911.20263671875 | 716.60205078125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 1 | peak_mem_mib | 1073.92822265625 | 684.72705078125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 8 | peak_mem_mib | 1088.80322265625 | 699.60205078125 |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | 16 | peak_mem_mib | 1105.80322265625 | 716.60205078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | peak_mem_mib | 703.0634765625 | 684.72705078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 1 | fwd_scratch_mib | 10.21142578125 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | peak_mem_mib | 781.18701171875 | 699.60205078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 8 | fwd_scratch_mib | 73.4599609375 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | peak_mem_mib | 872.08251953125 | 716.60205078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | 16 | fwd_scratch_mib | 147.35546875 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | peak_mem_mib | 897.734375 | 879.32763671875 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 1 | fwd_scratch_mib | 10.28173828125 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | peak_mem_mib | 976.37353515625 | 894.20263671875 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 8 | fwd_scratch_mib | 74.0458984375 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | peak_mem_mib | 1067.85498046875 | 911.20263671875 |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | 16 | fwd_scratch_mib | 148.52734375 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | peak_mem_mib | 703.2275390625 | 684.72705078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 1 | fwd_scratch_mib | 10.37548828125 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | peak_mem_mib | 782.49951171875 | 699.60205078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 8 | fwd_scratch_mib | 74.7724609375 | 0.0 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | peak_mem_mib | 874.015625 | 716.60205078125 |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | 16 | fwd_scratch_mib | 149.28857421875 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | peak_mem_mib | 715.47265625 | 696.89111328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 1 | fwd_scratch_mib | 10.45654296875 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | peak_mem_mib | 802.84716796875 | 711.76611328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 8 | fwd_scratch_mib | 82.9560546875 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | peak_mem_mib | 902.77685546875 | 728.76611328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | 16 | fwd_scratch_mib | 165.8857421875 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | peak_mem_mib | 923.10205078125 | 697.68994140625 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 1 | fwd_scratch_mib | 10.5224609375 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | peak_mem_mib | 1010.9599609375 | 712.56494140625 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 8 | fwd_scratch_mib | 83.50537109375 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | peak_mem_mib | 1111.43896484375 | 729.56494140625 |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | 16 | fwd_scratch_mib | 166.984375 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | peak_mem_mib | 715.62646484375 | 696.89111328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 1 | fwd_scratch_mib | 10.6103515625 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | peak_mem_mib | 804.07763671875 | 711.76611328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 8 | fwd_scratch_mib | 84.1865234375 | 0.0 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | peak_mem_mib | 905.94482421875 | 728.76611328125 |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | 16 | fwd_scratch_mib | 169.0537109375 | 0.0 |
| goodreads-d128-c0_genre-linr_v4-torch | 1000 | 1 | peak_mem_mib | 587.42724609375 | 685.525390625 |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 1 | peak_mem_mib | 685.525390625 | 587.42724609375 |
| goodreads-d128-c0_genre-linr_v4-triton | 1000 | 8 | peak_mem_mib | 700.400390625 | 602.30224609375 |

