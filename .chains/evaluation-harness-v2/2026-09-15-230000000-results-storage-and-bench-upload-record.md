---
chain: "evaluation-harness-v2"
branch: "main"
parent: "2026-09-15-200000000-d4-report-py-record.md"
nextStep: "Run bench upload on a finished D1 stage (never on a campaign in flight); exercise the LFS path with the ~450 MB D1 sidecars; verify a download from a different machine."
created: "2026-09-15T23:00:00Z"
---

# §14 record: results storage and `bench upload`, 2026-09-15, CPU only

Branch `dev/results-storage` off `development` @ `5fd05a6`, worktree `/workspace/wt/results`, `/venvs/results`, `CUDA_VISIBLE_DEVICES=""`. Nothing made public. Why now: records are the only citable thing and lived on a rented box with a ~26 GB `/workspace` quota; `bench upload` had never run ("exercised by no test", V §11).

## Policy (now in `docs/system/evaluation.md#results-storage`)
| what | where | why |
|---|---|---|
| JSONL records, `flat.csv`, `report/`, the validation record | git | 79 records = 750 KB, line-diffable, the evidence |
| `*.samples.jsonl`, `.perkernel/`, figures | HF Hub `pinkmeme/eval-results`, private | sidecars behind those 750 KB are 45 MB (60x); regenerable only on the GPU |
| `results/_parity/*.npz` (600-680 MB/run), `results/_logs/` | deleted | rewritten every run; the verdict is inside the record |
At D1's ~700 cells sidecars extrapolate to ~450 MB. Manifests with a sha256 per file are committed at `docs/artifacts/evaluation-harness-v2/results-storage/`.

## Built
`evaluation/bench/upload.py` 54 -> 264 lines; `tests/bench/test_upload.py` 15 tests, no network. One `--path-in-repo` subtree per invocation in one `create_commit`; `<prefix>/MANIFEST.json` = `report.provenance` + `{path, bytes, sha256}` per file; a root README regenerated from every manifest (existing ones fetched first; `viewer: false` front matter); `--verify` downloads back and checks every sha256; `--private/--public` default private; `--repo-id` default `upload.RESULTS_REPO`. `report._provenance` renamed and exported as `report.provenance` (not copied), so the Hub verdict and the LaTeX banner cannot drift; `--gate` vetoed by failed / partial / dirty / branch outside development / main. `eval_datasets/hub.py` untouched.

## Published (private), 20 files / 37 MB
| subtree | source | records | status | schema | branch | commit | files | bytes |
|---|---|---|---|---|---|---|---|---|
| c4 | evaluation-harness-v2 artifacts c4/results | 20 | ok=20 | 1 | dev/c4-gate-rerun | afc2ab9 | 4 | 12,983,727 |
| c5 | evaluation-package-layout artifacts c5/results | 6 | ok=6 | 2 | dev/c5-harness-split | 71430d7 | 3 | 7,563,738 |
| b3 | official-silvertorch artifacts b3/e2e | 30 | ok=24, partial=6 | 2 | dev/b3-head-to-head | e23309c | 8 | 16,918,821 |
All `"citable": false` (no `--gate`; records from `dev/...` branches; b3 also 6 partial). Unstable counts: c4 18, c5 5, b3 14.

Round trip on the local disk: `--verify` all sha256 equal (c4 4, c5 3, b3 8 files); a separate full `snapshot_download` + `diff -r`: identical, 36 MB. `repo_info(...).private` True. Transcript `results-storage/roundtrip.txt`.

Gates: ruff clean; harness suite 185 passed, 4 skipped (+15); `bench upload --help` renders; links 0.

Surprises: the 60:1 sidecar ratio (C4's goodreads record file 156 KB beside an 8.4 MB samples file); B3's 6 partial records surfaced by the machinery; the existing 45 MB of sidecars in git left alone (policy binds D1 onward).

Unverified: numbers (probe runs); never run against a live `results/` tree (a growing JSONL would publish a mismatched manifest: upload finished stages only); no upload over 37 MB, no LFS path exercised; round trip only on this box; `--path-in-repo ""` untested; nothing deleted.
