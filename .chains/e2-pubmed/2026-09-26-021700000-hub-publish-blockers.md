---
chain: "e2-pubmed"
branch: "main"
parent: "2026-09-26-021430988-e2-pubmed-10m-slice-staged.md"
nextStep: "Before `eval-data publish pubmed`, in evaluation/eval_datasets/hub.py (outside E2's boundary, next to e3's work): rename the EVAL_REPOS key \"pubmed\" to \"pubmed-medcpt\" (the keys are local dir names), and add \"medline/*\" and \"staging/*\" to EVAL_IGNORE_PATTERNS. Then dry-run it; it should list 56 files, about 16.5 GB."
created: "2026-09-26T02:17:00Z"
---

# E2: two hub.py defects block the pubmed publish

A dry run showed this; nothing was uploaded.

- `eval-data publish pubmed --dry-run` fails with `FileNotFoundError: /data/pubmed does not exist`. `EVAL_REPOS` keys resolve to `data_root()/<key>` (`arxiv-papers`, `goodreads-work-id`), but pubmed is keyed `"pubmed"` while the ETL and config/pubmed.yaml use `pubmed-medcpt`. The workaround is `--source /data/pubmed-medcpt`.
- With `--source`, the dry run lists 1428 files, 17,298 MB: 1334 `medline/pubmed26n*.parquet` and 38 `staging/articles_chunk_*.parquet` are ETL intermediates that `EVAL_IGNORE_PATTERNS` does not exclude. The real layout is the other 56 files: the 38 fp16 shards and their metas, the attrs and vocabs, the parquets, and `gt_d768/oracle_v4_c0_mesh_*.pt`.
- E2's worker brief put hub.py out of bounds, since e3 works in hub.py-adjacent files, so neither fix was made. The publish decision itself stays deferred (see the parent note: the SilverTorch OOM may resize the slice).
