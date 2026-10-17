# Results

The data behind the paper's tables and figures, the records they are computed from, the script
that computes them, and the configuration of the run that produced the records.

Every number comes from one final run on an NVIDIA A100-SXM4-80GB (torch 2.10.0+cu128,
Triton 3.6.0, CUDA 12.8, Python 3.11), with the code in this repository. SM clocks could not be
locked in the container; each record carries its sampled clock and an `unstable` flag (the
relative range of its three timing-window medians above 5 %).

## Files

| path | what |
|---|---|
| [`tables/`](tables/) | one CSV per exhibit, written by `build_tables.py` (below) |
| [`build_tables.py`](build_tables.py) | records → `tables/`; every selection rule is a constant or a docstring in it |
| [`records/records.parquet`](records/records.parquet) | the 2,926 timed records the exhibits read (below) |
| [`records/score_path_diagnostic.csv`](records/score_path_diagnostic.csv) | the untimed score-path diagnostic: SilverTorch recall@100 at fixed probed clusters with the exact filter, scored as shipped (int8 inputs, fp16 output) and with fp16 inputs and fp32 accumulation |
| [`final-run-config/suites.yaml`](final-run-config/suites.yaml) | the suites that produced the records, in the harness's suite format |
| [`final-run-config/executed_configurations.csv`](final-run-config/executed_configurations.csv) | every configuration that has records: arm, parameters, and the seeds, batch sizes, ks and modes actually run |

## Exhibits

| exhibit | file | rows |
|---|---|---|
| Table 1: the original papers' claims and what the study found | [`table1_claims.csv`](tables/table1_claims.csv) | one per measured quantity behind a verdict: `n` points, `min` and `max` (equal when `n` is 1) |
| Table 2: head-to-head, real vs synthetic filters | [`table2_head_to_head.csv`](tables/table2_head_to_head.csv) | filter type × recall target: workloads timed, target reached by SilverTorch, SilverTorch faster, median SilverTorch / LiNR latency over the reached ones |
| Representative real-filter selections (absolute latencies) | [`table3_selected_points.csv`](tables/table3_selected_points.csv) | the six workloads named in the text |
| Fig. 1: per-workload head-to-head at recall 0.90 / 0.95 | [`fig1_head_to_head.csv`](tables/fig1_head_to_head.csv) | 288 workload × target selections, including those SilverTorch does not reach |
| Fig. 2: LiNR V2 / V1 against the synthetic pass rate | [`fig2_linr_v2_over_v1.csv`](tables/fig2_linr_v2_over_v1.csv) | per dataset, batch 1 and 16, pass rate |
| Fig. 3: LiNR V3 against the faster exact variant, per pool | [`fig3_linr_v3_vs_exact.csv`](tables/fig3_linr_v3_vs_exact.csv) | per workload and timed pool, k 100 |
| Fig. 4: SilverTorch's probe sweep on LAION 30M synthetic filters | [`fig4_probe_sweep.csv`](tables/fig4_probe_sweep.csv) | per filter, `n_lists` and `n_probe`, batch 16, k 100, with LiNR V1 as the anchor |

Units: latencies (`*_ms`) are the median GPU forward time per batch in milliseconds; recalls are
exact filtered top-k recall against the exhaustive fp32 oracle (recall@k at the row's `k`);
`pass_rate` is the mean fraction of the catalogue passing the filter; ratios are the first
latency over the second. `timing_unstable` is true when any timed arm a row combines carries the
`unstable` flag. `filter` names the predicate: `c*_…`, `all4`, `tags4` are real metadata filters;
on the `*-synth` datasets `pNNNN` is a uniform random filter at pass rate 0.NNNN (`p1` = 100 %),
and `cNNN` on `arxiv-corr-synth` a cluster-correlated one.

## Selection rules

All exhibits use seed 0 and CUDA-graph timing; the head-to-head runs also hold seeds 1 and 2,
which no exhibit reads.

- **Head-to-head** (Table 2, Table 3, Fig. 1). A workload is (dataset, dim, filter, batch, k).
  SilverTorch runs Meta's code, adapted (`backend` `cuda`), on its bloom filter (1,024 bits, 5
  hashes) with the exact recheck off; LiNR V1 and V2 run the exact filter in the same process.
  For each workload and target (0.90, 0.95), the SilverTorch point is the fastest timed
  (`n_lists`, `n_probe`) with `n_probe` ≤ `n_lists` / 4 whose recall reaches the target. The
  LiNR side is the faster of V1 and V2 timed in the same run, or V3 at its fastest timed pool
  reaching the target (k 100 only) when that is faster still. A few SilverTorch points were
  timed later in a follow-up run (`followup`); their ratio is taken against the exact LiNR of
  their own run and rescaled by the original run's best-LiNR / exact-LiNR ratio. Workloads with
  no timed point inside the probe budget (24 workload × target combinations) are not counted.
- **V2 / V1** (Fig. 2, Table 1). Both exact variants timed interleaved on the synthetic filters,
  k 100. "Near 10 %" is a pass rate in [0.09, 0.11] with no unstable flag; the crossing is the
  first pass rate where V2 / V1 reaches 1, interpolated in log-log, on the uniform synthetic
  sets at batch 1.
- **V3 / exact** (Fig. 3, Table 1). Every timed V3 pool against the faster exact variant of the
  same run, k 100. Table 1 reads the LAION 10M / 30M synthetic filters at 20, 30 and 50 % with
  no unstable flag.
- **Probe sweep** (Fig. 4, Table 1). At 4,096 lists, the fastest swept `n_probe` reaching
  recall 0.90 per filter, against a fixed 1,024 probes.
- **Co-design** (Table 1). Meta's code as published (`backend` `published`, commit `21aa35e`;
  not part of this repository, see [parity](../docs/system/parity.md)), eager: the bloom filter
  over the full index against probe-then-filter, same run and point; a pair with either side
  unstable is left out.
- **Score-path diagnostic** (Table 1). PubMed `c3_journal_reverse` and LAION 1M `c0_domain`
  from `score_path_diagnostic.csv`.

Over the 219 reached selections of Fig. 1, dropping any row with `timing_unstable` leaves 206,
of which SilverTorch is faster in 203 (median ratio 0.268).

## Rebuilding the tables

From this directory, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --with polars==2.0.0 python build_tables.py records/records.parquet tables
```

The CSVs in `tables/` are that command's output. Each was also checked value by value against
the final run's original results build (every row and column equal to 1e-12); no cell differs.

## The records

`records/records.parquet` holds `bench` records (one row per timed entry, the harness's column
names; [record format](../docs/system/evaluation.md#output-one-jsonl-record-per-cell)), reduced
to the rows the exhibits read and the columns `build_tables.py` uses, with three labels added:

- `experiment`: `head_to_head`, `linr_pass_rate`, `probe_sweep` or `codesign`;
- `run`: an opaque id; records with the same `run` and `seed` were timed in one process, and
  only those are compared with each other;
- `followup`: the record comes from a follow-up run that added deeper probe points
  (`n_probe` at half and three quarters of `n_lists`) for workloads short of recall 0.90.

`backend` uses this repository's names: `cuda` is Meta's code, adapted (the `silvertorch/`
member), `triton` the LiNR kernels, `published` Meta's code as published. Failed records and
records superseded by a rerun are not included: a cell that was unstable was rerun once, and
the rerun's record replaces it unless it was unstable too, in which case the record keeps its
flag.

The complete per-run output of the final run (every record, the per-query sidecars, logs; about
2 GB, one directory per run with its `results.parquet` as `bench report` writes it and a
`MANIFEST.json` of checksums) is in the archive at `<anonymous archive URL>`. Hosted as a
Hugging Face dataset repo, one run is fetched and checked with
`bench fetch --repo-id <owner>/<repo> --path-in-repo <run>`
([`bench fetch`](../docs/system/evaluation.md#bench-fetch)).

## Final-run configuration

The final run did not run [`evaluation/config/suites.yaml`](../evaluation/config/suites.yaml) as
shipped. It ran the suites in [`final-run-config/suites.yaml`](final-run-config/suites.yaml):

- `timed-bloom-*`: the head-to-head, its timed points picked from seed 0's quality-only probe
  curves (the points reaching recall 0.90 / 0.95), at batch 1 / 16 / 64, k 100 / 1000 / 2048
  where listed, `n_lists` 512 to 65,536, the recheck off, V3 at the pools reaching the same
  recall;
  `-synth-s<d>` is synthetic draw `d`, run at seed `d`;
- `probe-sweep-d256`: Fig. 4;
- `linr-synth-pubmed`, `linr-corr`, `synth`, `nsweep-synth`, `laion30m-synth`: the V2 / V1
  pass-rate runs;
- `codesign`, `codesign-laion30m`, `nsweep-codesign`: the co-design pairs.

The follow-up runs at deeper probes and the 20 / 30 / 50 % synthetic timings were generated the
same way; their configurations are the `followup_run` rows and the 20-50 % `*-synth` rows of
`executed_configurations.csv`. That file is the authority on what ran: a suite lists seeds 0-2
where only some were run, and the seeds, batch sizes and ks per configuration there are those
that have records. Seed coverage: seed 0 everywhere; seeds 1 and 2 on the head-to-head runs
only (arXiv, Goodreads, PubMed, LAION 1M-30M and their synthetic sets, except seed 1 on the
arXiv and Goodreads synthetic sets and seed 2 on LAION 30M synthetic). The `published` arms ran
Meta's code as published, which this repository no longer installs.
