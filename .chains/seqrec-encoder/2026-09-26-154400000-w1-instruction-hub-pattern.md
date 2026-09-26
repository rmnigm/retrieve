---
chain: "seqrec-encoder"
branch: "w1-encoder"
parent: "2026-09-26-150000000-w1-instruction-shared-k-collapse.md"
nextStep: "W1: fix hub.py, then idle."
created: "2026-09-26T12:44:00Z"
---

# W1 instruction: report accepted; finish hub.py

> Report accepted. One open item from the brief: make `EPOCH_SNAPSHOT_PATTERN` in `evaluation/eval_datasets/hub.py` match the new `{encoder}-ep*.pt` names, add `_resume.pt` and `encoded_queries_v2.pt` to `CKPT_ALWAYS_IGNORE`, fix every doc copy of the old pattern, run ruff, the evaluation suite and the link checker, commit, and stay idle. No GPU.

Outcome: commit 9020e22 (the last line of the report note).
