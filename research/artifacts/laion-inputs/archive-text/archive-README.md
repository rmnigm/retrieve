# Inputs and records of the final run

The inputs the benchmark harness reads to rerun the paper's final run (embeddings, filter attributes, query sets,
synthetic filters, the Goodreads sequential encoder checkpoint) and the complete per-run output of that run (every record,
the per-query sidecars and the latency samples). The code is the anonymous code release that accompanies the
paper; every path below is relative to that repository's `evaluation/`.

Each top-level directory is laid out exactly as one Hugging Face dataset repo that the harness reads,
`<owner>/eval-<dataset>` and `<owner>/eval-results`, so the harness uses it with only the owner changed
(`RETRIEVE_HUB_OWNER=<owner>`), or from local disk with no Hub at all.

## Contents

| directory | Hub repo | harness `data_dir` | configs | size |
|---|---|---|---|---|
| `eval-goodreads-work-id/` | `<owner>/eval-goodreads-work-id` | `data/goodreads-work-id` | `goodreads`, `goodreads-synth*` | 1.72 GB |
| `eval-arxiv-papers/` | `<owner>/eval-arxiv-papers` | `data/arxiv-papers` | `arxiv`, `arxiv-synth*`, `arxiv-corr-synth*` | 3.62 GB |
| `eval-pubmed/` | `<owner>/eval-pubmed` | `data/pubmed-medcpt-dot` | `pubmed`, `pubmed-synth*` | 19.80 GB |
| `eval-results/` | `<owner>/eval-results` | (results trees) | | 0.60 GB |
| `eval-laion30m/` | `<owner>/eval-laion30m` | `data/laion30m` (the subsets are rebuilt from it) | `laion*` | 24.76 GB |

Each dataset directory has its own `README.md` (what every file is). `LICENSES.md` says, per dataset, what is
shipped and under which terms; `MANIFEST.json` lists every file with its size and sha256, and
`python3 verify.py [ROOT] [PREFIX ...]` checks a copy against it.

The checkpoint: `eval-goodreads-work-id/checkpoints/sasrec-ssm-logq-d128` (the run read Goodreads at d128 only).
arXiv, PubMed and LAION are pre-encoded and need no checkpoint.

## Using it

With the archive on the Hub under an owner `<owner>`:

```bash
export RETRIEVE_HUB_OWNER=<owner> RETRIEVE_DATA_ROOT=/data   # evaluation/data must point at the data root (docs/system/datasets.md)
uv run --directory evaluation eval-data fetch goodreads-work-id --include-checkpoints
uv run --directory evaluation eval-data fetch arxiv-papers
uv run --directory evaluation eval-data fetch pubmed --target /data/pubmed-medcpt-dot
uv run --directory evaluation eval-data fetch laion30m
for n in 1 3 10; do uv run --directory evaluation eval-data subset laion30m laion${n}m --n-items ${n}000000; done   # seed 20261013
uv run --directory evaluation bench check --dataset goodreads        # the layout contract, per dataset config
uv run --directory evaluation bench check --dataset arxiv --dim 128 --dim 256   # arXiv ships the dims the run used
```

From a local copy, link or copy each directory to its `data_dir` name under the data root
(`eval-goodreads-work-id/` → `$RETRIEVE_DATA_ROOT/goodreads-work-id/`, `eval-pubmed/` →
`$RETRIEVE_DATA_ROOT/pubmed-medcpt-dot/`, and so on, as in the table). The oracle (exact filtered top-k) is not
shipped: `bench oracle --dataset D --suite S` builds it on the GPU before a run, and `bench run` builds a missing
one itself. Goodreads' query encode cache is rebuilt from the checkpoint the same way.

`bench run` / `bench campaign` then run the suites of `results/final-run-config/suites.yaml` (the final run's
suites) or `config/suites.yaml`.

## The records (`eval-results/runs/`)

105 runs, one directory per leg of the final run (one driver invocation on one GPU), each a results tree as `bench`
writes it:
`<suite>/<dataset>-d<dim>.jsonl` (one record per cell, failed attempts included), `*.samples.jsonl` (the per-call
latency vectors), `*.perquery/*.npz` (per-query recall and pass counts), `results.parquet` (the flat table, rebuilt
from the records) and `MANIFEST.json` (provenance and a sha256 per file). 3,752 records and 9,543 sample lines in
all. One run is fetched and checked with

```bash
uv run --directory evaluation bench fetch --repo-id <owner>/eval-results --path-in-repo runs/<run> --results fetched/<run>
uv run --directory evaluation bench merge fetched/* --out tables/
```

Run names: the suite or stage, then `-s<seed>`. A letter (`-a`, `-b`, `-c`) marks the parts of one suite that ran
as separate processes on different GPUs (split by dataset, or by seed); `-rerun` is the single rerun of the cells
that were `unstable` in the run of the same name (the rerun's record replaces the first unless it is unstable too).

| runs | what |
|---|---|
| `curve-*` | seed 0, quality only: SilverTorch's recall over (`n_lists`, `n_probe`), LiNR V3 over its pool, per dataset and filter; the timed points are picked from these |
| `timed-point-picks` | the picked points, as the `suites-timed.yaml` the timed runs were generated from |
| `timed-bloom-*` | the head-to-head: SilverTorch (Meta's code, adapted) on its Bloom filter at the picked points, LiNR V1 / V2 (and V3 at k 100) on the exact filter, batch 1 / 16 / 64, seeds 0-2 |
| `timed-clause-lite-*` | SilverTorch with its exact clause filter, the filters with a reverse clause |
| `v3pool*`, `linr-*` | LiNR V3's pool curves; the exact LiNR variants at batch 64 / 256 and k 2048, the PubMed synthetic and the correlated filters, LAION 1M-30M |
| `final-*` | the final passes of the suites in `config/suites.yaml` of the same name (`filter`, `deep`, `synth`, `c3`, `c3-real`, `codesign`, `h2h`, `bloomwidth`, `v3bits`, `nsweep*`, `seeds-c1*`, `laion30m*`) |
| `followup-*` | the follow-up runs: deeper probe points (`n_probe` at one half and three quarters of `n_lists`) for workloads short of recall 0.90, and their timings |
| `synth-broad-*`, `probe-sweep-s0` | the 20 / 30 / 50 % synthetic pass rates at 10 M and 30 M (curves, V3 pools, timings) and the probe sweep on LAION 30 M |
| `untimed-*` | untimed diagnostics: recall ceilings (`ceiling/`), the Bloom filter's grading against the exact filter (`grade_bloom-*.csv`) |
| `bloomwidth-s0` | the Bloom width grid (quality only) |

`results/records/records.parquet` in the code release is the subset of these records the paper's tables read.

**What differs from the records as written.** Only identifying or machine-local strings, nothing measured:

- run directory names (the run's machine tags removed, as above) and suite names (a run-stage prefix and machine
  tags removed, internal stage names replaced by descriptive ones: `followup-*`, `synth-broad-*`,
  `probe-sweep-d256`), in the `suite` field, in `per_query` paths and in the directory names;
- `env.host` (a container id) → `gpu-host-1` … `gpu-host-3`;
- absolute paths of the run's checkout and environment → relative paths (`env.lib_dir`, `env.*_build.so_path`,
  tracebacks in `error`);
- run-driver logs, progress markers and helper scripts left out (the records carry the clocks, the environment and
  every cell's timing); `results.parquet` and `MANIFEST.json` regenerated by the release's `bench` from the records.

Backend names are as written: `official-fork` is Meta's code, adapted (`cuda` in the release; `bench merge` and
`bench report` map it), `official` Meta's code as published (`published` in the release's tables). Every other field
is unchanged; each record and sample line, and every sidecar byte for byte, was checked against the original.

## Checks done on this archive

- Every per-query pass count stored in the records' sidecars was recomputed on CPU from the shipped attributes,
  with the library's clause semantics: Goodreads, arXiv and PubMed real filters (14 filter × mode pairs; every
  query of the single-clause filters, 300 queries of each multi-clause one): all equal. The uniform synthetic filters
  of Goodreads, arXiv and PubMed (45 filter × draw pairs): pass counts equal to the records'.
- LAION, rebuilt for this archive (below), against the final run's LAION records: every per-query pass count of the
  real filters at 1 M, 3 M, 10 M and 30 M (`c0_domain` every query, `tags4` 500 queries each): all equal; the
  records' item counts, kept queries and pass rates equal; the uniform synthetic pass rates (50 dataset × filter
  pairs) equal. Rebuilding the 1 M subset from `eval-laion30m/` reproduces its manifest's tensors byte for byte. On the GPU, exact LiNR V1 over the rebuilt vectors reproduces the records' result-id sha256, oracle and held-out recall and targets in filter on both real filters at every size (8 / 8).
- `eval_datasets.layout.validate_layout` (what `bench check` runs) is clean on every dataset directory at every
  dim the run used.
- The sha256 of every file is in `MANIFEST.json`.

## Known gaps

- **LAION is a rebuild, not the run's copy.** The staged LAION 30 M the final run read was not kept;
  `eval-laion30m/` was rebuilt with the release's ETL (`eval-data laion`) from the same two Re-LAION parts:
  deduplication, domains, buckets and query split reproduce the run's counts exactly (30,928,288 distinct captions,
  1,430,979 domains), and the filters and subset rows equal the records' (checks above). The caption embeddings are
  a new GPU encode with the same model and recipe (exact search on them returns the run's result ids); the subsets ship as their sha256 only
  (`eval-laion30m/subsets/`) and are rebuilt with `eval-data subset` (above).
- **The cluster-correlated arXiv filter is regenerated, not the run's copy.** The run's `item_attrs_corr.pt` was not
  kept; `eval-data synth-filter --dataset arxiv --correlated --dim 128` rebuilt it here. At rate 0.03 the k-means
  converges and every query's pass count equals the record's. At 0.01 and 0.1 it stops at its 25-iteration cap, and
  CPU floating-point order moves a few items between clusters: the mean pass rate equals the records' to six digits
  (0.010496, 0.107411), while the per-query pass count differs by a median relative 9×10⁻⁵ at 0.01 (max 0.35) and
  6×10⁻⁶ at 0.1 (max 0.011). Records on `c001` and `c01` were measured with the run's copy.
- The oracle and the encode caches are not shipped (rebuilt on the GPU by `bench`, above).
