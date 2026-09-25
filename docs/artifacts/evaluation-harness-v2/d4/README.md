# D4 sample output — `bench report`

Everything here was produced by one command, on the records that existed on
2026-09-15 **before the D1 campaign**, so that the *shape* of the output can be
reviewed without running it. **Nothing here is a result** (CLAUDE.md rule 2) —
every file says so in its own banner, caption or watermark.

```bash
cd /workspace/wt/d4/evaluation
mkdir -p /tmp/d4/filter
cp ../docs/artifacts/evaluation-harness-v2/c4/results/filter/*.jsonl /tmp/d4/filter/
cat ../docs/artifacts/evaluation-package-layout/c5/results/filter/goodreads-d128.jsonl \
    >> /tmp/d4/filter/goodreads-d128.jsonl
cat ../docs/artifacts/evaluation-package-layout/c5/results/filter/goodreads-d128.samples.jsonl \
    >> /tmp/d4/filter/goodreads-d128.samples.jsonl
CUDA_VISIBLE_DEVICES="" uv run bench report /tmp/d4 --out <here>
```

The input is C4's gate run (20 records, **schema 1**) plus C5's one-cell check
(6 records, **schema 2**, seeds 0/1/2); the two runs share a `code_version`, so
the four records they have in common dedupe by resume key and 24 records remain.
The mix is deliberate: it exercises both schema versions, three backends, a
two-value `n_probe` sweep, three seeds and 21 cells flagged `unstable`.

| file | note |
|---|---|
| `report.md` | read this first: provenance, the citability verdict and its reasons, the selection the tables used, coverage, the failed / partial / unstable accounting |
| `flat.csv` | `records.flatten` — 432 rows × 84 columns, one row per `(record, perf entry)`. Written first; every table and figure is built from it |
| `tables/*.tex` | the seven LaTeX fragments. **Not compiled** — no TeX toolchain on this box; checked structurally by `evaluation/tests/bench/test_report.py` |
| `figures/*.png` | seven figures, each watermarked and footed with `code_version` + commit |
| `methodology.tex` | the thesis's §"Методология замеров" itemize, its constants read live out of `measure.latency`, `inputs.query_pool` and `run` |

Three things worth looking at, because they are the point of the step rather
than decoration:

- `tables/tab-pareto_goodreads.tex` — the caption's `[PRE-CAMPAIGN RECORDS —
  NOT CITABLE]` prefix, the `†` on the cell whose timing windows spread 22.8 %,
  and the footnote naming both `n_probe` sets when the table can show only one.
- every latency table's last footnote — **which clock estimator** the numbers
  are read against (`perf[].sm_mhz`, under load, with its observed range), and
  that no normalisation was applied.
- `tables/tab-batch_scaling.tex` — the per-cell window spread printed beside
  every number, and the `B=1` caveat spelled out in the caption.
