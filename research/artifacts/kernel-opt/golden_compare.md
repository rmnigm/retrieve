## vs golden (quality.oracle, tol 1e-6)

| golden cell | k | metric | golden | new | |diff| | verdict |
|---|---|---|---|---|---|---|
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | recall@100 | 0.884042068 | 0.884044068 | 2.0e-06 | FAIL |
| arxiv-d128-c0_maincat-silvertorch-triton | 100 | ndcg@100 | 0.913732553 | 0.913733980 | 1.4e-06 | FAIL |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | recall@500 | 0.847300391 | 0.847300391 | 6.0e-12 | PASS |
| arxiv-d128-c0_maincat-silvertorch-triton | 500 | ndcg@500 | 0.875417934 | 0.875417941 | 7.5e-09 | PASS |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | recall@1000 | 0.819666078 | 0.819666078 | 6.0e-12 | PASS |
| arxiv-d128-c0_maincat-silvertorch-triton | 1000 | ndcg@1000 | 0.848329757 | 0.848329763 | 6.5e-09 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-torch | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | recall@100 | 0.999694695 | 0.999694695 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 100 | ndcg@100 | 0.999780996 | 0.999780996 | 6.0e-12 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | recall@500 | 0.999656967 | 0.999656967 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 500 | ndcg@500 | 0.999728945 | 0.999728945 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | recall@1000 | 0.999627452 | 0.999627452 | 1.1e-16 | PASS |
| goodreads-d128-c0_genre-linr_v1_filter_mask-triton | 1000 | ndcg@1000 | 0.999696314 | 0.999696313 | 1.0e-09 | PASS |
| goodreads-d128-c0_genre-linr_v2-torch | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | recall@100 | 0.999273761 | 0.999724110 | 4.5e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v2-triton | 100 | ndcg@100 | 0.999479009 | 0.999802098 | 3.2e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | recall@500 | 0.999372560 | 0.999765294 | 3.9e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v2-triton | 500 | ndcg@500 | 0.999504185 | 0.999814545 | 3.1e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | recall@1000 | 0.999390412 | 0.999781319 | 3.9e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v2-triton | 1000 | ndcg@1000 | 0.999503065 | 0.999821746 | 3.2e-04 | FAIL |
| goodreads-d128-c0_genre-linr_v3-torch | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | recall@100 | 0.876943911 | 0.876961154 | 1.7e-05 | FAIL |
| goodreads-d128-c0_genre-linr_v3-triton | 100 | ndcg@100 | 0.909096071 | 0.909108991 | 1.3e-05 | FAIL |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | recall@500 | 0.724321939 | 0.724321939 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-linr_v3-triton | 500 | ndcg@500 | 0.774933983 | 0.774934087 | 1.0e-07 | PASS |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | recall@1000 | 0.617541029 | 0.617541029 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-linr_v3-triton | 1000 | ndcg@1000 | 0.676164299 | 0.676164355 | 5.6e-08 | PASS |
| goodreads-d128-c0_genre-linr_v4-torch | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-linr_v4-triton | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-silvertorch-torch | | | | | | no counterpart in today's campaign config |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | recall@100 | 0.912800491 | 0.912800491 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-silvertorch-triton | 100 | ndcg@100 | 0.935957277 | 0.935957282 | 5.5e-09 | PASS |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | recall@500 | 0.847149812 | 0.847149812 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-silvertorch-triton | 500 | ndcg@500 | 0.876261928 | 0.876261928 | 4.1e-10 | PASS |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | recall@1000 | 0.789005274 | 0.789005274 | 0.0e+00 | PASS |
| goodreads-d128-c0_genre-silvertorch-triton | 1000 | ndcg@1000 | 0.823339080 | 0.823339080 | 1.7e-10 | PASS |

## vs committed D1 records (pre-change code, same harness)

| cell | metrics compared | max |diff| | max diff metric |
|---|---|---|---|
| linr_v1_filter_mask triton {} | 24 | 0.0e+00 | heldout.mrr@1000 |
| linr_v2 triton {} | 24 | 0.0e+00 | heldout.mrr@1000 |
| linr_v3 triton {} | 24 | 0.0e+00 | heldout.mrr@1000 |
| silvertorch official {"n_probe": 24} | 24 | 0.0e+00 | heldout.mrr@1000 |
| silvertorch official {"n_probe": 32} | 24 | 0.0e+00 | heldout.mrr@1000 |
| silvertorch triton {"n_probe": 24} | 24 | 0.0e+00 | heldout.mrr@1000 |
| silvertorch triton {"n_probe": 32} | 24 | 0.0e+00 | heldout.mrr@1000 |
