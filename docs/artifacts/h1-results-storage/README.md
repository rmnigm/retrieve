# H1 — results and raw outputs off git

The removal list and the script that moves the Hub-bound half. The storage rule
itself is in [evaluation](../../system/evaluation.md#results-storage); what the
Hub holds is in [hub-index.md](../hub-index.md).

- [`cleanup.tsv`](cleanup.tsv): every JSON / JSONL / `flat.csv` that was tracked
  under `docs/artifacts/`, `evaluation/results/` and `evaluation/golden/`, plus
  `evaluation/golden/_logs/`, with one action and the reason:
  - `keep` (13): the 11 golden cells (a gate fixture `test_c4_gate` and the C4
    gate read), `golden/_logs/provenance.txt` (the exact HEAD, host and UTC of
    that fixture, cited by its README), `q3/roofline.json` (linked from
    validation).
  - `rm-on-hub` (15): already on the Hub as `b3`, `c4`, `c5`, sha256-equal.
  - `hub` (154): moved to the path in `hub_path`.
  - `drop` (52): superseded, derived or never citable; still in git history.
- [`stage_hub.sh`](stage_hub.sh): extracts every `hub` row from `8c493f2` (the
  last commit that tracks them) into one directory per subtree and runs
  `bench upload` on each — `--dry-run` by default, `PUBLISH=1` for the real
  upload with `--verify`.

Dry run, 2026-09-26, all 14 subtrees: listing, manifest and citability verdict
computed without error; `d1-a` is 3 files / 78,261,847 bytes / 126 records,
NOT CITABLE (no gate; produced on `dev/d1a-campaign`). No real upload has been
made from this list yet.
