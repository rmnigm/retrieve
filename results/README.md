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
| [`tables/`](tables/) | the CSVs below, written by `build_tables.py` |
| [`build_tables.py`](build_tables.py) | records → `tables/`; every selection rule is a constant or a docstring in it |
| [`records/records.parquet`](records/records.parquet) | the 2,986 records the tables read (below) |
| [`records/score_path_diagnostic.csv`](records/score_path_diagnostic.csv) | untimed: SilverTorch recall@100 at fixed probed clusters with the exact filter, scored as shipped (int8 inputs, fp16 output) and with fp16 inputs and fp32 accumulation |
| [`records/bloom_false_positives.csv`](records/bloom_false_positives.csv) | untimed: the bloom filter's false-positive rate over the whole catalogue per width and hash count (false admissions / items failing the exact filter, per query), and the queries with a false negative |
| [`final-run-config/suites.yaml`](final-run-config/suites.yaml) | the suites that produced the records, in the harness's suite format |
| [`final-run-config/executed_configurations.csv`](final-run-config/executed_configurations.csv) | every configuration that has records: arm, parameters, and the seeds, batch sizes, ks and modes actually run |

## Tables

| file | what it holds | paper exhibit |
|---|---|---|
| [`claims_summary.csv`](tables/claims_summary.csv) | the measured quantities behind each verdict on the original papers' claims (LiNR V1/V2 ordering, V3 shortlist, SilverTorch probes and precision, co-design): `n` points, `min`, `max` | — |
| [`head_to_head_summary.csv`](tables/head_to_head_summary.csv) | per filter type (real, synthetic) and recall target: workloads timed, target reached by SilverTorch, SilverTorch faster, median SilverTorch / LiNR latency over the reached ones | — |
| [`head_to_head_selections.csv`](tables/head_to_head_selections.csv) | the 288 workload × recall-target selections behind the summary, including those SilverTorch does not reach: SilverTorch's selected `n_lists` / `n_probe`, its recall and latency, the selected LiNR variant and its latency | — |
| [`selected_points.csv`](tables/selected_points.csv) | six real-filter selections at k 100 with absolute latencies | — |
| [`linr_v2_over_v1.csv`](tables/linr_v2_over_v1.csv) | LiNR V1 and V2 latency and their ratio against the synthetic pass rate, batch 1 and 16 | — |
| [`linr_v3_vs_exact.csv`](tables/linr_v3_vs_exact.csv) | LiNR V3 per timed pool: recall, latency, and its ratio to the faster exact variant of the same run, batch 1 / 16 / 64 | — |
| [`probe_sweep.csv`](tables/probe_sweep.csv) | SilverTorch's recall and latency over its `n_probe` range on LAION 30M with seven synthetic filters (4,096 and 16,384 lists, batch 16), with LiNR V1 as the anchor | — |
| [`codesign_pairs.csv`](tables/codesign_pairs.csv) | Meta's published code: the bloom filter over the whole index against filtering only the searched clusters, per catalogue size, filter, batch, `n_lists` and `n_probe` | — |
| [`bloom_width.csv`](tables/bloom_width.csv) | SilverTorch's bloom filter per width (`m_bits`) and hash count (`k_hash`) on Goodreads, arXiv and PubMed: recall@100, index size, whole-catalogue false-positive rate, false-negative queries | — |
| [`index_memory.csv`](tables/index_memory.csv) | resident index size per arm (SilverTorch per `n_lists`, LiNR V1 / V2 / V3) per dataset | — |

Units and precision: latencies (`*_ms`) are the median GPU forward time per batch in
milliseconds, to 3 decimals; recalls are exact filtered top-k recall against the exhaustive fp32
oracle (recall@k at the row's `k`), to 3 decimals; ratios (`*_over_*`) are the first latency
over the second, to 3 decimals; `pass_rate` (the mean fraction of the catalogue passing the
filter) and false-positive rates to 4 significant digits; `index_mib` in whole MiB. In
`claims_summary.csv`, `min` and `max` carry the precision of their quantity (`n_probe` as a
count). `timing_unstable` is true when any timed arm a row combines carries the `unstable` flag.
`filter` names the predicate: `c*_…`, `all4`, `tags4` are real metadata filters; on the
`*-synth` datasets `pNNNN` is a uniform random filter at pass rate 0.NNNN (`p1` = 100 %), and
`cNNN` on `arxiv-corr-synth` a cluster-correlated one.

## Selection rules

Every table uses seed 0, and every timed table CUDA-graph timing except the co-design pairs
(eager); the head-to-head runs also hold seeds 1 and 2, which no table reads.

- **Head-to-head** (`head_to_head_*`, `selected_points`). A workload is (dataset, dim, filter,
  batch, k). SilverTorch runs Meta's code, adapted (`backend` `cuda`), on its bloom filter (1,024
  bits, 5 hashes) with the exact recheck off; LiNR V1 and V2 run the exact filter in the same
  process. For each workload and target (0.90, 0.95), the SilverTorch point is the fastest timed
  (`n_lists`, `n_probe`) with `n_probe` ≤ `n_lists` / 4 whose recall reaches the target. The
  LiNR side is the faster of V1 and V2 timed in the same run, or V3 at its fastest timed pool
  reaching the target (k 100 only) when that is faster still. A few SilverTorch points were
  timed later in a follow-up run (`followup`); their ratio is taken against the exact LiNR of
  their own run and rescaled by the original run's best-LiNR / exact-LiNR ratio. Workloads with
  no timed point inside the probe budget (24 workload × target combinations) are not counted.
- **V2 / V1** (`linr_v2_over_v1`). Both exact variants timed interleaved on the synthetic
  filters, k 100. In `claims_summary`, "near 10 %" is a pass rate in [0.09, 0.11] with no
  unstable flag, and the crossing is the first pass rate where V2 / V1 reaches 1, interpolated
  in log-log, on the uniform synthetic sets at batch 1.
- **V3 / exact** (`linr_v3_vs_exact`). Every timed V3 pool against the faster exact variant of
  the same run, k 100. `claims_summary` reads the LAION 10M / 30M synthetic filters at 20, 30
  and 50 % with no unstable flag.
- **Probe sweep** (`probe_sweep`). `claims_summary` reads, at 4,096 lists, the fastest swept
  `n_probe` reaching recall 0.90 per filter against a fixed 1,024 probes.
- **Co-design** (`codesign_pairs`). Meta's code as published (`backend` `published`, commit
  `21aa35e`; not part of this repository, see [parity](../docs/system/parity.md)), eager, the two
  filter placements in the same run at the same point. LAION 30M was timed in two runs
  (`repeat` 1 and 2). `claims_summary` leaves out a pair with either side unstable.
- **Bloom width** (`bloom_width`). Quality only, batch 16, the width grid on the real filters;
  false positives from `records/bloom_false_positives.csv`.
- **Index memory** (`index_memory`). The median `index_mib` over the head-to-head records per
  dataset, arm and `n_lists`: resident index tensors, not allocator reservation or peak memory.
- **Score-path diagnostic** (`claims_summary`). PubMed `c3_journal_reverse` and LAION 1M
  `c0_domain` from `records/score_path_diagnostic.csv`.

Over the 219 reached selections of `head_to_head_selections.csv`, dropping any row with
`timing_unstable` leaves 206, of which SilverTorch is faster in 203 (median ratio 0.268).

## Rebuilding the tables

From this directory, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --with polars==2.0.0 python build_tables.py records/records.parquet tables
```

The CSVs in `tables/` are that command's output, byte for byte. Each was also checked against
the final run's original results build, rounded the same way: every value equal.

## The records

`records/records.parquet` holds `bench` records (one row per timed entry, the harness's column
names; [record format](../docs/system/evaluation.md#output-one-jsonl-record-per-cell)), reduced
to the rows the tables read and the columns `build_tables.py` uses, with three labels added:

- `experiment`: `head_to_head`, `linr_pass_rate`, `probe_sweep`, `codesign` or `bloom_width`;
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
- `probe-sweep-d256`: the probe sweep;
- `bloomwidth`: the bloom width grid (quality only);
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
