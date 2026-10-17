# Results

The data behind the paper's tables and figures, the records they are computed from, the scripts
that compute and draw them, and the configuration of the run that produced the records.

Every number comes from one final run on an NVIDIA A100-SXM4-80GB (torch 2.10.0+cu128,
Triton 3.6.0, CUDA 12.8, Python 3.11), with the code in this repository. SM clocks could not be
locked in the container; each record carries its sampled clock and an `unstable` flag (the
relative range of its three timing-window medians above 5 %).

## Files

| path | what |
|---|---|
| [`tables/`](tables/) | the CSVs below, written by `build_tables.py` |
| [`build_tables.py`](build_tables.py) | records → `tables/`; every selection rule is a constant or a docstring in it |
| [`build_figures.py`](build_figures.py) | `tables/` and `originals.csv` → `figures/` (PDF and PNG), one function per figure |
| [`NUMBERS.md`](NUMBERS.md) | the facts the paper cites, each with its value and the CSV line(s) it is read from, written by [`build_numbers.py`](build_numbers.py) |
| [`originals.csv`](originals.csv) | the original papers' numbers the figures draw beside ours, quoted with their source (not measured here) |
| [`records/records.parquet`](records/records.parquet) | the 3,529 records the tables read (below) |
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
| [`linr_v2_over_v1.csv`](tables/linr_v2_over_v1.csv) | LiNR V1 and V2 latency and their ratio against the synthetic pass rate, batch 1 and 16 | Fig. `linr_v2_over_v1` |
| [`linr_v3_vs_exact.csv`](tables/linr_v3_vs_exact.csv) | LiNR V3 per timed pool: recall, latency, and its ratio to the faster exact variant of the same run, batch 1 / 16 / 64 | — |
| [`probe_sweep.csv`](tables/probe_sweep.csv) | SilverTorch's recall and latency over its `n_probe` range on LAION 30M with seven synthetic filters (4,096 and 16,384 lists, batch 16), with LiNR V1 as the anchor | Fig. `probe_sweep` |
| [`codesign_pairs.csv`](tables/codesign_pairs.csv) | Meta's published code: the bloom filter over the whole index against filtering only the searched clusters, per catalogue size, filter, batch, `n_lists` and `n_probe` | Fig. `catalogue_size` (b) |
| [`bloom_width.csv`](tables/bloom_width.csv) | SilverTorch's bloom filter per width (`m_bits`) and hash count (`k_hash`) on Goodreads, arXiv and PubMed: recall@100, index size, whole-catalogue false-positive rate, false-negative queries | — |
| [`index_memory.csv`](tables/index_memory.csv) | resident index size per arm (SilverTorch per `n_lists`, LiNR V1 / V2 / V3) per dataset | — |
| [`head_to_head_probe_budgets.csv`](tables/head_to_head_probe_budgets.csv) | all 312 workload × recall-target settings, each with an `outcome`: `within_quarter` (the selection of `head_to_head_selections`), `deeper` (reached only at one half or three quarters of the lists, follow-up points) or `not_reached`; the pick, SilverTorch's best recall at any depth, the dataset's `tie_band` and the tie-aware `verdict` | Figs. `head_to_head`, `catalogue_size` (a) |
| [`head_to_head_counts.csv`](tables/head_to_head_counts.csv) | counts over the settings per probe budget (`quarter`: within a quarter; `deeper`: reached only beyond it; `any`), k, filter type and target (`all` pools): settings, reached, SilverTorch faster (strict), wins / ties / losses under the tie band, median ratio, unstable, and the same over the stable settings | — |
| [`timing_spread.csv`](tables/timing_spread.csv) | SilverTorch's seed spread per dataset over its points timed at two or three seeds: time range over median, recall range, and the tie band (half the time spread) | — |
| [`silvertorch_latency.csv`](tables/silvertorch_latency.csv) | SilverTorch's latency at its within-quarter picks per dataset and batch: min, median, max | — |
| [`linr_v2_over_v1_batch64.csv`](tables/linr_v2_over_v1_batch64.csv) | LiNR V1 and V2 latency and their ratio on the synthetic filters at batch 64, from the head-to-head runs | Fig. `linr_v2_over_v1` |
| [`linr_v3_wins.csv`](tables/linr_v3_wins.csv) | LiNR V3 against the faster exact variant per filter type, batch and pool: pools timed, V3 faster, V3 faster with no unstable flag, the smallest ratio | — |
| [`linr_v3_code_length.csv`](tables/linr_v3_code_length.csv) | LiNR V3 per code length (`k_bits`: 64 / 128 at d 128, 256 / 768 at d 768), pool, filter, batch and k: recall, latency, latency over the shortest code | — |
| [`linr_native_pytorch.csv`](tables/linr_native_pytorch.csv) | the LiNR kernels against native PyTorch: per variant, filter, batch and mode (eager, CUDA graph), native and compiled (`max-autotune`, eager only) latency over the Triton kernel of the same run | — |
| [`linr_native_v2_over_v1.csv`](tables/linr_native_v2_over_v1.csv) | un-fused LiNR: native V2 over native V1 per filter, batch and mode on Goodreads synthetic (the two timed in different runs) | Fig. `linr_v2_over_v1` (diamonds) |
| [`codesign_medians.csv`](tables/codesign_medians.csv) | the co-design pairs per catalogue size, filter (`all` pools the two) and batch: pairs, stable pairs, their median / min / max, and the share of lists searched | — |
| [`recall_bridge.csv`](tables/recall_bridge.csv) | per SilverTorch configuration on Yambda (no filter) and Goodreads genre (bloom): recall against the exact top-k and recall of the held-out interactions, k 100 and 1000 | — |
| [`recall_bridge_correlation.csv`](tables/recall_bridge_correlation.csv) | Pearson and Spearman correlation between the two recalls per dataset, dimension and k, with `n` and both ranges | — |
| [`reverse_clause_points.csv`](tables/reverse_clause_points.csv) | the four filters with a reverse clause: SilverTorch on the exact clause filter at each timed point against the faster exact LiNR at the same batch (separate runs) | — |
| [`reverse_clause_selections.csv`](tables/reverse_clause_selections.csv) | the same per filter, batch and target: the fastest clause-filter point within a quarter reaching the target | — |
| [`evaluated_queries.csv`](tables/evaluated_queries.csv) | per dataset and filter: the queries whose exact filtered result is non-empty, the population every recall averages over | — |

Units and precision: latencies (`*_ms`) are the median GPU forward time per batch in
milliseconds, to 3 decimals; recalls are exact filtered top-k recall against the exhaustive fp32
oracle (recall@k at the row's `k`), to 3 decimals; ratios (`*_over_*`) are the first latency
over the second, to 3 decimals; `pass_rate` (the mean fraction of the catalogue passing the
filter) and false-positive rates to 4 significant digits; `index_mib` in whole MiB. In
`claims_summary.csv`, `min` and `max` carry the precision of their quantity (`n_probe` as a
count). `timing_unstable` is true when any timed arm a row combines carries the `unstable` flag.
`tie_band`, `time_spread`, `recall_spread` and shares searched are to 4 significant digits;
`probe_fraction` is exact.
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

- **Probe budgets** (`head_to_head_probe_budgets`, `head_to_head_counts`). The same rule with no
  limit on `n_probe` gives a pick for every setting with a timed SilverTorch point (312); a
  setting's pick is the within-quarter one where it exists. The 24 settings at k 1000 with no
  timed point inside a quarter have points only from the follow-up runs. The tie band of a
  dataset is half its SilverTorch time spread over seeds (`timing_spread`); a ratio within it of
  1 is a `tie`, below it `silvertorch_faster`, above it `linr_faster`.
- **Seed spread** (`timing_spread`). Per point (run, workload, `n_lists`, `n_probe`) timed at two
  or more seeds, (max − min) / median of its latency and max − min of its recall, averaged per
  dataset; `*` averages the datasets.
- **V2 / V1 at batch 64** (`linr_v2_over_v1_batch64`). The exact variants as timed in the
  head-to-head runs (the pass-rate runs have no batch 64), k 100, original runs only.
- **Native PyTorch** (`linr_native_pytorch`, `linr_native_v2_over_v1`). The native arms are
  sync-free: V1 masks the scores, V2 compacts the passing ids with a stable argsort and gathers
  them, neither with a host synchronisation; host-synchronising boolean indexing (`x[mask]`) was
  not timed. `max-autotune` cannot be captured in a CUDA
  graph, so the compiled rows are eager only. Native V1 and native V2 were timed in different runs.
- **V3 code length** (`linr_v3_code_length`). Quality and timing of V3 at two code lengths per
  dataset (the 512 bits of the original paper at d 128 was not run).
- **Recall bridge** (`recall_bridge*`). Yambda: every SilverTorch probe setting without a filter;
  Goodreads: every distinct `n_lists` / `n_probe` of SilverTorch on the genre bloom filter (recheck
  off). Held-out recall is the share of held-out interactions in the returned top-k.
- **Reverse-clause filters** (`reverse_clause_*`). The bloom path refuses a reverse clause (a
  negated bloom test would turn false positives into misses), so the four filters with one ran on
  the exact clause filter of Meta's code, adapted, in their own runs; LiNR V1 / V2 are those of
  the exact-filter runs at the same batch.
- **Evaluated queries** (`evaluated_queries`). Recall averages over the queries whose exact
  filtered top-k is non-empty; its denominator per query is min(k, items passing).

Over the 219 reached selections of `head_to_head_selections.csv`, dropping any row with
`timing_unstable` leaves 206, of which SilverTorch is faster in 203 (median ratio 0.268).

## Figures

`build_figures.py` draws each figure from the CSVs only (and the quoted values of
`originals.csv`); a legend entry exists only for a drawn series. Hollow markers are a pick beyond
a quarter of the lists, or a timing flagged unstable (stated per figure in its docstring).

| figure | what it draws |
|---|---|
| `linr_v2_over_v1` | LiNR V2 / V1 against the pass rate on the uniform synthetic filters (the cluster-correlated `arxiv-corr-synth` left out), batch 1 / 16 / 64, one line per catalogue and dimension; LiNR's quoted PyTorch and TensorFlow ratios; our native V2 / native V1 at batch 1 |
| `head_to_head` | SilverTorch / best LiNR at recall 0.90 and 0.95, k 100, per real filter and per synthetic pass rate (catalogues pooled); within a quarter filled, deeper hollow, not reached × |
| `catalogue_size` | (a) speed-up over the best LiNR on LAION's two real filters at 0.90 against the catalogue size; (b) co-design pairs on LAION, stable pairs only, with the range the SilverTorch paper reports at 20 M |
| `probe_sweep` | SilverTorch's recall@100 against latency over `n_probe` on LAION 30M at 4,096 lists, four filters, with LiNR V1's latency |

## Rebuilding the tables and figures

From this directory, with [uv](https://docs.astral.sh/uv/):

```bash
uv run --no-project --with polars==2.0.0 python build_tables.py records/records.parquet tables
uv run --no-project --with polars==2.0.0 --with matplotlib==3.10.0 python build_figures.py tables figures
uv run --no-project --with polars==2.0.0 python build_numbers.py
```

The CSVs in `tables/` are that command's output, byte for byte. Each was also checked against
the final run's original results build, rounded the same way: every value equal.

## The records

`records/records.parquet` holds `bench` records (one row per timed entry, the harness's column
names; [record format](../docs/system/evaluation.md#output-one-jsonl-record-per-cell)), reduced
to the rows the tables read and the columns `build_tables.py` uses (among them
`n_queries_oracle` and `heldout_recall@100` / `@1000`), with three labels added:

- `experiment`: `head_to_head`, `linr_pass_rate`, `probe_sweep`, `codesign`, `bloom_width`,
  `native_pytorch`, `v3_code_length`, `recall_bridge` or `reverse_clause`;
- `run`: an opaque id; records with the same `run` and `seed` were timed in one process, and
  only those are compared with each other;
- `followup`: the record comes from a follow-up run that added deeper probe points
  (`n_probe` at half and three quarters of `n_lists`) for workloads short of recall 0.90.

`backend` uses this repository's names: `cuda` is Meta's code, adapted (the `silvertorch/`
member), `triton` the LiNR kernels, `published` Meta's code as published. Failed records and
records superseded by a rerun are not included: a cell that was unstable was rerun once, and
the rerun's record replaces it unless it was unstable too, in which case the record keeps its
flag.

The complete per-run output of the final run is in the `<anonymous archive>`, directory `eval-results/runs/`:
106 runs, one per leg (one driver invocation on one GPU), 3,827 records, 0.61 GB. Each is a results tree as
`bench` writes it: `<suite>/<dataset>-d<dim>.jsonl` (every record, failed attempts included),
`*.samples.jsonl` (per-call latencies), `*.perquery/*.npz` (per-query recall and pass counts),
`results.parquet` and a `MANIFEST.json` of checksums. Hosted as the Hugging Face dataset repo
`<owner>/eval-results`, one run is fetched and checked with
`bench fetch --repo-id <owner>/eval-results --path-in-repo runs/<run>`
([`bench fetch`](../docs/system/evaluation.md#bench-fetch)), and `bench merge` reads any set of them. Run
names are the suite or stage and the seed (`timed-bloom-d128-s0`); a letter marks the parts of one suite run on
different GPUs and `-rerun` the one rerun of a run's unstable cells. The run's logs are not included; record
fields that held a machine-local path or a container id hold a relative path or `gpu-host-<n>`.

The same archive holds every input the final run read, laid out as the harness's Hub repos
(`eval-goodreads-work-id/`, `eval-arxiv-papers/`, `eval-pubmed/`, `eval-yambda-500m/`, with the Goodreads and
Yambda encoder checkpoints and the synthetic filters; LAION 1M-30M not yet), so `eval-data fetch` and
`bench run` read it with `RETRIEVE_HUB_OWNER` set to its owner
([datasets](../docs/system/datasets.md#huggingface-io)).

## Measured revision

The records were measured at the commit tagged `measured-final-run` (every record's
`code_version` is `3c8198dfc35c299ea0f9d3a59a5eaf299494332e`), except the 39 SilverTorch records of
the reverse-clause runs (`reverse_clause`, `algo` `silvertorch`): those were taken on a branch
commit outside this history (`code_version` `08d54df3e5d71a2c33b8afccb455f61877d5f9fe`) whose
code differs from the commit tagged `measured-clause-filter-runs` by one source comment; it adds
the exact filter's int32 clause-slot layout (CHANGES C14) to the first revision.

This tree differs from the measured revision in ways that change names or picks, not the
measured kernels:

- **Names.** The adapted code was `backend="official-fork"` (package `silvertorch_fork`) and Meta's
  code as published `backend="official"`; the records here carry today's names (`cuda`,
  `published`). Meta's code as published is no longer installed.
- **Clustering metric.** The records' IVF assigned items to clusters by L2 in k-means and probed
  by inner product; this tree uses inner product for both, so rebuilt indexes, and the probe
  counts that reach a recall target, can differ.
- **Removed.** Our Triton SilverTorch reimplementation and our LiNR bloom filter (neither in the
  records here), and the harness's `bloom_path` arm (see the historical suites below).

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
- `codesign`, `codesign-laion30m`, `nsweep-codesign`: the co-design pairs;
- `c3`, `c3-real`: the kernels against native and compiled PyTorch (V1); the native V2 rows are
  in `synth`;
- `v3bits`: the V3 code-length sweep;
- `curve-d128` (the Goodreads genre points), `yambda`: the recall bridge (quality only);
- `filter`, `linr-bs64`, `linr-bs256`: the exact LiNR on the reverse-clause filters (batch 1 / 16,
  and 64); `timed-clause-lite-d128` / `-d256` / `-d768`: SilverTorch's exact clause filter on
  them, run on a later revision ([Measured revision](#measured-revision)).

The follow-up runs at deeper probes and the 20 / 30 / 50 % synthetic timings were generated the
same way; their configurations are the `followup_run` rows and the 20-50 % `*-synth` rows of
`executed_configurations.csv`. That file is the authority on what ran: a suite lists seeds 0-2
where only some were run, and the seeds, batch sizes and ks per configuration there are those
that have records. Seed coverage: seed 0 everywhere; seeds 1 and 2 on the head-to-head runs
only (arXiv, Goodreads, PubMed, LAION 1M-30M and their synthetic sets, except seed 1 on the
arXiv and Goodreads synthetic sets and seed 2 on LAION 30M synthetic).

**Historical suites.** Five suites hold an arm on Meta's code as published (`backend`
`published`, commit `21aa35e`), which this repository no longer installs; the shipped harness
rejects them (`unknown backends ['published']`, and `bloom_path` for the co-design):

| suite | the published arm | runnable here |
|---|---|---|
| `codesign`, `codesign-laion30m`, `nsweep-codesign` | the whole suite (whole-index against searched-cluster bloom filtering, `bloom_path`) | no: needs Meta's code at `21aa35e` and the harness adapter of the time ([parity](../docs/system/parity.md#rerunning-the-comparison-against-the-published-code)) |
| `synth`, `filter` | one SilverTorch side arm no table reads | yes: `synth-shipped`, `filter-shipped` are the same suites without it |

Every other suite in the file loads with the shipped harness (checked with `load_matrix` for each
of its datasets).
