# Numbers

Facts the paper cites, each read from a CSV in [`tables/`](tables/) by the script that wrote this
file; the source column gives the CSV line(s) (line 1 is the header). Flag definition: a setting
is `timing_unstable` when any timed record it combines (SilverTorch's pick, the exact LiNR of its
run, V3 at its fastest pool reaching the target) carries the `unstable` flag (relative range of
its three window medians above 5 %). Tie band: half the dataset's SilverTorch seed spread
(`timing_spread.csv`); a ratio within the band of 1 is a tie. Counts are seed 0.

## Head-to-head

| fact | value | source |
|---|---|---|
| settings with a timed SilverTorch point (k 100 / 1000 / 2048 pooled); of them with one within a quarter of the lists | 312; 288 | `head_to_head_counts.csv` line 2 (settings, timed_within_quarter) |
| within a quarter: reached; SilverTorch faster (strict); wins / ties / losses under the tie band; median SilverTorch / LiNR | 219; 216; 215 / 1 / 3; 0.282 | `head_to_head_counts.csv` line 2 |
| within a quarter, settings with no unstable flag: reached; SilverTorch faster; median | 206; 203; 0.268 | `head_to_head_counts.csv` line 2 |
| unstable flag (any timed arm the setting combines) among the settings reached within a quarter | 13 of 219 | `head_to_head_counts.csv` line 2 |
| unstable flag over all 288 within-quarter selections, reached or not | 28 of 288 | `head_to_head_selections.csv` 288 rows (sum of timing_unstable) |
| k 100: settings; timed within a quarter; reached; SilverTorch faster; W / T / L; median | 234; 234; 183; 180; 180 / 0 / 3; 0.26 | `head_to_head_counts.csv` line 11 |
| k 1000: settings; timed within a quarter; reached; SilverTorch faster; W / T / L; median | 60; 36; 27; 27; 26 / 1 / 0; 0.337 | `head_to_head_counts.csv` line 20 |
| k 2048: settings; timed within a quarter; reached; SilverTorch faster; W / T / L; median | 18; 18; 9; 9; 9 / 0 / 0; 0.434 | `head_to_head_counts.csv` line 29 |
| reached only beyond a quarter (one half or three quarters of the lists): settings; SilverTorch faster; W / T / L | 51; 13; 12 / 3 / 36 | `head_to_head_counts.csv` line 38 |
| any depth: reached; SilverTorch faster; W / T / L; not reached at any depth | 270; 229; 227 / 4 / 39; 42 | `head_to_head_counts.csv` line 74 |
| real filters, k 100, target 0.90: reached within a quarter of settings; SilverTorch faster; median | 27 of 39; 27; 0.237 | `head_to_head_counts.csv` line 15 |
| real filters, k 100, target 0.95: reached within a quarter of settings; SilverTorch faster; median | 21 of 39; 20; 0.415 | `head_to_head_counts.csv` line 16 |
| synthetic filters, k 100, target 0.90: reached within a quarter of settings; SilverTorch faster; median | 75 of 78; 75; 0.174 | `head_to_head_counts.csv` line 18 |
| synthetic filters, k 100, target 0.95: reached within a quarter of settings; SilverTorch faster; median | 60 of 78; 58; 0.347 | `head_to_head_counts.csv` line 19 |
| real filters, k all, target 0.90: reached within a quarter of settings; SilverTorch faster; median | 51 of 78; 51; 0.265 | `head_to_head_counts.csv` line 6 |
| real filters, k all, target 0.95: reached within a quarter of settings; SilverTorch faster; median | 33 of 78; 32; 0.46 | `head_to_head_counts.csv` line 7 |
| synthetic filters, k all, target 0.90: reached within a quarter of settings; SilverTorch faster; median | 75 of 78; 75; 0.174 | `head_to_head_counts.csv` line 9 |
| synthetic filters, k all, target 0.95: reached within a quarter of settings; SilverTorch faster; median | 60 of 78; 58; 0.347 | `head_to_head_counts.csv` line 10 |
| the strict losses within a quarter (dataset filter batch k target: ratio) | laion10m tags4 b64 k100 0.95: 1.159; arxiv-synth p1 b64 k100 0.95: 1.211; laion10m-synth p05 b64 k100 0.95: 1.014 | `head_to_head_probe_budgets.csv` lines 79, 181, 241 |
| ties under the dataset's tie band (dataset filter batch k target outcome: ratio, band) | arxiv c3_nversions b64 k1000 0.95 within_quarter: 0.991, 0.01249; laion10m c0_domain b1 k100 0.9 deeper: 1.005, 0.008712; laion3m c0_domain b16 k100 0.95 deeper: 0.993, 0.01897; laion3m c0_domain b16 k1000 0.95 deeper: 1.019, 0.01897 | `head_to_head_probe_budgets.csv` lines 43, 62, 127, 129 |
| synthetic filters at 0.90, any depth: settings; SilverTorch faster; speed-up range | 78; 78; 1.06-34.5x | `head_to_head_probe_budgets.csv` 78 rows (filter_type synthetic, target 0.90, reached) |
| Goodreads synthetic 0.1 %: outcome, probe fraction and ratio at batch 1 / 16 / 64 (both targets) | deeper 0.5 b1 0.941, deeper 0.5 b16 0.626, deeper 0.5 b64 0.724 | `head_to_head_probe_budgets.csv` lines 182-187 |
| tie band (half SilverTorch's seed time spread) per dataset | arxiv 1.25 %, arxiv-synth 1.76 %, goodreads 2.08 %, goodreads-synth 2.17 %, laion10m 0.87 %, laion10m-synth 0.51 %, laion1m 2.29 %, laion30m 0.42 %, laion30m-synth 0.17 %, laion3m 1.90 %, pubmed 1.66 %, pubmed-synth 0.99 %, * 1.34 % | `timing_spread.csv` 13 rows |
| SilverTorch at batch 1, within-quarter picks: latency range per dataset (ms) | arxiv 0.191-0.268, arxiv-synth 0.192-0.232, goodreads 0.19-0.203, goodreads-synth 0.193-0.194, laion10m 0.309-0.601, laion10m-synth 0.203-0.48, laion30m 0.505-0.804, laion30m-synth 0.309-0.914, laion3m 0.247-0.247, pubmed 0.246-0.583, pubmed-synth 0.237-0.584 | `silvertorch_latency.csv` lines 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32 |
| queries with a non-empty exact filtered result per real filter (recall's population, of 10,000) | arxiv all4 10000, arxiv c0_maincat 10000, arxiv c3_nversions 10000, goodreads all4 9971, goodreads c0_genre 9859, goodreads c1_lang_reverse 9192, laion10m c0_domain 9527, laion10m tags4 10000, laion1m c0_domain 8839, laion1m tags4 10000, laion30m c0_domain 9722, laion30m tags4 10000, laion3m c0_domain 9236, laion3m tags4 10000, pubmed all5 9984, pubmed c0_mesh 8428, pubmed c3_journal_reverse 8600 | `evaluated_queries.csv` 17 rows |

## LiNR

| fact | value | source |
|---|---|---|
| V2 / V1 near 10 % pass, batch 1, uniform sets, no unstable flag | 0.183-0.681 | `linr_v2_over_v1.csv` lines 13, 31, 49, 69, 87, 105, 123, 141 |
| V2 / V1 near 10 % pass, batch 16, uniform sets, no unstable flag | 1.114-1.552 | `linr_v2_over_v1.csv` lines 22, 40, 59, 78, 96, 114, 132, 151 |
| pass rate where V2 / V1 reaches 1 at batch 1 (uniform sets) | 0.8483-0.9725 (n 8) | `claims_summary.csv` line 10 |
| V2 / V1 at batch 64, every synthetic point (V2 slower everywhere) | 1.050-31.220 | `linr_v2_over_v1_batch64.csv` 26 rows |
| V2 / V1 at batch 64 at each set's lowest pass rate (about 0.1 % or 20 %) | 1.050-1.633 | `linr_v2_over_v1_batch64.csv` lines 2, 6, 10, 17, 24 |
| native PyTorch V1 / Triton V1, graph, batch 1 (goodreads-synth) | 1.379-1.571 | `linr_native_pytorch.csv` lines 30, 42, 54, 66, 72, 78 |
| native PyTorch V1 / Triton V1, graph, batch 16 (goodreads-synth) | 1.679-1.857 | `linr_native_pytorch.csv` lines 31, 43, 55, 67, 73, 79 |
| native PyTorch V1 / Triton V1, eager, batch 1 (goodreads-synth) | 1.407-1.860 | `linr_native_pytorch.csv` lines 28, 40, 52, 64, 70, 76 |
| native PyTorch V1 / Triton V1, eager, batch 16 (goodreads-synth) | 5.792-6.702 | `linr_native_pytorch.csv` lines 29, 41, 53, 65, 71, 77 |
| compiled PyTorch V1 / Triton V1, eager, batch 1 (arxiv, arxiv-synth, goodreads, goodreads-synth, laion30m-synth, pubmed) | 0.517-1.024 | `linr_native_pytorch.csv` 24 rows |
| compiled PyTorch V1 / Triton V1, eager, batch 16 (arxiv, arxiv-synth, goodreads, goodreads-synth, laion30m-synth, pubmed) | 1.482-2.722 | `linr_native_pytorch.csv` 24 rows |
| native PyTorch V2 / Triton V2, graph, batch 1 (goodreads-synth) | 2.393-3.288 | `linr_native_pytorch.csv` lines 36, 48, 60 |
| native PyTorch V2 / Triton V2, graph, batch 16 (goodreads-synth) | 5.947-12.634 | `linr_native_pytorch.csv` lines 37, 49, 61 |
| native PyTorch V2 / Triton V2, eager, batch 1 (goodreads-synth) | 2.100-2.101 | `linr_native_pytorch.csv` lines 34, 46, 58 |
| native PyTorch V2 / Triton V2, eager, batch 16 (goodreads-synth) | 11.888-25.765 | `linr_native_pytorch.csv` lines 35, 47, 59 |
| compiled PyTorch V2 / Triton V2, eager, batch 1 (goodreads-synth) | 0.415-0.541 | `linr_native_pytorch.csv` lines 32, 44, 56 |
| compiled PyTorch V2 / Triton V2, eager, batch 16 (goodreads-synth) | 3.011-5.100 | `linr_native_pytorch.csv` lines 33, 45, 57 |
| un-fused: native V2 / native V1, graph, batch 1, Goodreads synthetic 1 / 10 / 100 % | 1.535, 1.558, 1.803 | `linr_native_v2_over_v1.csv` lines 3, 7, 11 |
| un-fused: native V2 / native V1, graph, batch 16, Goodreads synthetic 1 / 10 / 100 % | 6.595, 6.745, 8.25 | `linr_native_v2_over_v1.csv` lines 5, 9, 13 |
| un-fused: native V2 / native V1, eager, batch 1, Goodreads synthetic 1 / 10 / 100 % | 2.095, 2.09, 2.087 | `linr_native_v2_over_v1.csv` lines 2, 6, 10 |
| un-fused: native V2 / native V1, eager, batch 16, Goodreads synthetic 1 / 10 / 100 % | 4.094, 4.074, 4.075 | `linr_native_v2_over_v1.csv` lines 4, 8, 12 |
| V3 faster than the faster exact variant (filter type, batch, pool: faster / faster with no flag / timed, smallest ratio) | real b1 0.05: 1/1/5, 0.842; real b1 0.1: 1/1/6, 0.9; real b16 0.005: 2/1/8, 0.985; real b16 0.02: 1/1/2, 0.933; real b16 0.05: 3/2/5, 0.941; real b16 0.1: 2/2/6, 0.976; synthetic b1 0.005: 1/1/1, 0.427; synthetic b1 0.01: 4/4/4, 0.434; synthetic b1 0.02: 5/5/5, 0.5; synthetic b1 0.05: 2/2/2, 0.665 | `linr_v3_wins.csv` lines 4, 5, 7, 8, 9, 10, 17, 18, 19, 20 |
| V3 wins on real filters with no unstable flag (dataset filter batch pool: ratio, recall) | arxiv c3_nversions b1 0.05: 0.842, 0.933; arxiv c3_nversions b1 0.1: 0.9, 0.972; laion10m c0_domain b16 0.05: 0.965, 0.938; laion10m c0_domain b16 0.1: 0.977, 0.969; laion10m c0_domain b16 0.1: 0.976, 0.969; laion30m c0_domain b16 0.02: 0.933, 0.922; laion30m c0_domain b16 0.05: 0.941, 0.965; laion3m c0_domain b16 0.005: 0.985, 0.913 | `linr_v3_vs_exact.csv` lines 11, 12, 27, 29, 30, 79, 80, 112 |
| V3 recall@100 at a 1 % pool, goodreads real filters, 64 bits | 0.637-0.780 | `linr_v3_code_length.csv` lines 2, 18, 34 (batch 1) |
| V3 recall@100 at a 1 % pool, goodreads real filters, 128 bits | 0.867-0.935 | `linr_v3_code_length.csv` lines 3, 19, 35 (batch 1) |
| V3 recall@100 at a 1 % pool, pubmed real filters, 256 bits | 0.884-0.993 | `linr_v3_code_length.csv` lines 146, 162, 178 (batch 1) |
| V3 recall@100 at a 1 % pool, pubmed real filters, 768 bits | 0.987-0.999 | `linr_v3_code_length.csv` lines 147, 163, 179 (batch 1) |
| V3 time, longer over shorter code at the same pool (1 % pool, k 100, real filters) | 0.995-1.637 | `linr_v3_code_length.csv` lines 3, 7, 19, 23, 35, 39, 151, 163, 167, 179, 183 |

## SilverTorch

| fact | value | source |
|---|---|---|
| co-design whole index / searched clusters, laion1m, batch 16: median (stable pairs) all / domain / tags | 0.824 / 0.819 / 0.825 (stable 4 of 4) | `codesign_medians.csv` lines 6-8 |
| co-design whole index / searched clusters, laion1m, batch 64: median (stable pairs) all / domain / tags | 0.887 / 0.851 / 0.916 (stable 4 of 4) | `codesign_medians.csv` lines 9-11 |
| co-design whole index / searched clusters, laion3m, batch 16: median (stable pairs) all / domain / tags | 0.825 / 0.825 / 0.826 (stable 4 of 4) | `codesign_medians.csv` lines 20-22 |
| co-design whole index / searched clusters, laion3m, batch 64: median (stable pairs) all / domain / tags | 1.089 / 0.889 / 1.269 (stable 4 of 4) | `codesign_medians.csv` lines 23-25 |
| co-design whole index / searched clusters, laion10m, batch 16: median (stable pairs) all / domain / tags | 1.012 / 0.829 / 1.184 (stable 4 of 4) | `codesign_medians.csv` lines 26-28 |
| co-design whole index / searched clusters, laion10m, batch 64: median (stable pairs) all / domain / tags | 1.566 / 1.16 / 2.169 (stable 4 of 4) | `codesign_medians.csv` lines 29-31 |
| co-design whole index / searched clusters, laion30m, batch 16: median (stable pairs) all / domain / tags | 1.648 / 1.11 / 2.177 (stable 6 of 8) | `codesign_medians.csv` lines 32-34 |
| co-design whole index / searched clusters, laion30m, batch 64: median (stable pairs) all / domain / tags | 3.166 / 1.73 / 4.243 (stable 6 of 8) | `codesign_medians.csv` lines 35-37 |
| co-design share of the lists searched per catalogue (32 and 128 probes) | laion1m 0.0625-0.25, laion3m 0.01563-0.0625, laion10m 0.007813-0.03125, laion30m 0.001953-0.007813 | `codesign_medians.csv` lines 6, 9, 20, 23, 26, 29, 32, 35 |
| bloom 1,024 bits / 5 hashes: false-positive rate, mean and per-query max (dataset filter) | arxiv all4: 0.0 / 0.0; arxiv c0_maincat: 0.0 / 0.0; arxiv c3_nversions: 0.0 / 0.0; goodreads c0_genre: 0.0 / 0.0; pubmed c0_mesh: 4.279e-08 / 1.22e-05 | `bloom_width.csv` lines 12, 24, 36, 48, 60 |
| PubMed MeSH, 5 hashes, per width: mean / max false-positive rate, recall@100 (batch 16, one fixed probe setting) | 64: 0.0167 / 0.1784, 0.497; 128: 0.0008968 / 0.08907, 0.67; 256: 3.684e-05 / 0.004516, 0.69; 512: 1.246e-06 / 0.000272, 0.69; 1024: 4.279e-08 / 1.22e-05, 0.69; 2048: 1.009e-09 / 1e-06, 0.69 | `bloom_width.csv` lines 56-61 |
| load-matched point: PubMed at 128 bits holds at most 8 hashed terms per item (4 MeSH + 4 single-valued clauses), at least 16 bits per term, against Meta's about 1,024 / 60 = 17; mean / max FP, recall | 0.0008968 / 0.08907, 0.67 | `bloom_width.csv` line 57 (terms per item: docs/system/datasets.md (PubMed clauses) and the index build over every non-pad (clause, value)) |
| recall bridge: Pearson / Spearman of oracle vs held-out recall (dataset dim k: n) | goodreads d128 k100: 0.993 / 0.845 (n 9); goodreads d128 k1000: 0.972 / 1.0 (n 9) | `recall_bridge_correlation.csv` lines 2-3 |
| recall bridge, Goodreads genre: oracle and held-out recall ranges at k 100 | 0.712-0.985; 0.188-0.215 | `recall_bridge_correlation.csv` line 2 |
| reverse-clause filters at 0.90, batch 1: LiNR / SilverTorch exact clause filter (speed-up) per filter (separate runs) | goodreads all4 1.66x, goodreads c1_lang_reverse 1.56x, pubmed all5 4.76x, pubmed c3_journal_reverse 23.26x | `reverse_clause_selections.csv` lines 2, 8, 14, 20 |
| reverse-clause filters at 0.90, batch 16: LiNR / SilverTorch exact clause filter (speed-up) per filter (separate runs) | goodreads all4 2.60x, goodreads c1_lang_reverse 3.12x, pubmed all5 5.46x, pubmed c3_journal_reverse 5.68x | `reverse_clause_selections.csv` lines 4, 10, 16, 22 |
| reverse-clause filters at 0.90, batch 64: LiNR / SilverTorch exact clause filter (speed-up) per filter (separate runs) | goodreads all4 2.57x, goodreads c1_lang_reverse 4.48x, pubmed all5 2.70x, pubmed c3_journal_reverse 2.51x | `reverse_clause_selections.csv` lines 6, 12, 18, 24 |
| reverse-clause filters not reaching 0.95 within a quarter (best recall) | pubmed all5 b1 0.915, pubmed all5 b16 0.915, pubmed all5 b64 0.915, pubmed c3_journal_reverse b1 0.914, pubmed c3_journal_reverse b16 0.914, pubmed c3_journal_reverse b64 0.914 | `reverse_clause_selections.csv` lines 15, 17, 19, 21, 23, 25 |
