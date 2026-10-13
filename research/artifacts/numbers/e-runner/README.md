# Pod e drivers (e-runner, 2026-10-12/13)

The scripts behind the NUMBERS legs that ran on pod e (a100-x1-e, A100-SXM4-80GB GPU 0, `flock /scratch/gpu0.lock`), as they ran; pod paths
are hard-coded. Results and gates: [validation](../../../validation.md#numbers-adapted-silvertorch-over-the-study-grid); Hub subtrees:
[hub-index](../../hub-index.md).

| file | leg (Hub `numbers/<leg>`) |
|---|---|
| [`run_b1s.sh`](run_b1s.sh) | A1-bloom-synth: `n-bloom-curve-synth-d128` on goodreads-synth, arxiv-synth (plan dev/results-draft 853a107) |
| [`run_b2s.sh`](run_b2s.sh) | A2-bloom-synth: `n-bloom-timed-synth-d128`, interleaved (picks dev/numbers-bloom-picks c2b7cf0) |
| [`run_b2s_rerun.sh`](run_b2s_rerun.sh) | A2-bloom-synth-rerun: the one unstable cell's interleave group, seeded with the 17 stable records; uploads only the new lines |
| [`run_k1000.sh`](run_k1000.sh) | A2-bloom-k1000-e: `n-bloom-timed-k1000-d128` / `-d256` with pod 1's official-fork `.so` (picks 2b3f662); laion30m's out-of-memory LiNR cells rerun one algo per process. Its inline manifest check also read `*.samples.jsonl` and crashed (KeyError `status`); the check was rerun on the record files only |
| [`selfx_run_leg.sh`](selfx_run_leg.sh), [`selfx_queue.sh`](selfx_queue.sh), [`selfx_queue_a4.sh`](selfx_queue_a4.sh) | the arXiv-family `-selfx` legs (tree run/selfmatch 970206d, plan staging 1006804); on the pod they were named `run_leg.sh`, `queue.sh`, `queue-a4.sh`. A4 used a config copy with `recheck: [false]` on the `n-bloomwidth` arm |
| [`selfx_check_arms.py`](selfx_check_arms.py) | the resolved-arm check before the `-selfx` legs (no LiNR bloom; SilverTorch bloom only with `recheck: false`) |
