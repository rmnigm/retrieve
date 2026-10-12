---
title: results
created: 2026-10-11
updated: 2026-10-11
type: summary
tags: [paper]
sources: [research/artifacts/results-draft/build_results.py, research/artifacts/hub-index.md, research/validation.md, research/paper/claims.md]
---

# Results (draft): SilverTorch, adapted from Meta's code, against LiNR

**NOT CITABLE.** The LiNR numbers wait on D1-G being declared green
([validation](../validation.md)). Every SilverTorch number on this page is
**provisional** until the adapted official code is accepted and measured (roadmap NUMBERS) over
every SilverTorch cell (the META-FORK chain in the [roadmap](../roadmap.md)).

The study compares two families of filtered retrieval on one A100:
SilverTorch, which is Meta's official published code adapted by us (the
changes are listed in `official-fork/CHANGES.md`), and LiNR V1 to V3, which
we implemented from the paper. Our Triton reimplementation of SilverTorch is
not part of the study.

## Where the numbers come from

- **LiNR V1-V3, and SilverTorch recall:** the final pass, one table at one
  library version (Hub `campaign-final/merged-4016/results.parquet`).
- **Adapted SilverTorch latency:** the adapted code's gate grid at ≤ 10 M
  items (Hub `artifacts/meta-fork/head-82fafb6`). That grid times the adapted
  code and our Triton SilverTorch in the same process, but it is not the
  campaign harness: on cells under 0.2 ms its timer reads up to 40 % lower
  than the harness (median agreement 1.00; range in `summary.json`). So the
  adapted latency is **bridged**: the campaign's Triton latency times the
  adapted ÷ Triton ratio from the gate, both timing orders averaged. LiNR is
  then compared on the same harness.
- **Adapted SilverTorch recall:** the same as the campaign's SilverTorch
  records. The gate shows the adapted code returning the same ids as Meta's
  code on all 1,536 cells, and, unfiltered and exactly filtered, the same
  scores and the same ids up to ties as the Triton arm.
- **The script:** [`build_results.py`](../artifacts/results-draft/build_results.py)
  writes every table and figure to Hub `artifacts/results-draft/20261011`
  ([hub-index](../artifacts/hub-index.md)). When the NUMBERS legs' records are
  passed as an extra argument, its `official-fork` records replace the
  bridge, and the same tables fill with measured numbers.
- **The story page:** a short, chart-led reading of these tables
  (https://claude.ai/artifact/WMfvv46CKNGKcWaoidNXCg, private), built by
  `page/pagedata.py` and `page/assemble.py` next to the script.

All latencies are per batch, CUDA-graph mode, median. "Recall" is recall@k
against the exact filtered top-k.

## Who wins where (head to head)

Tables: `h2h.csv` (every cell), `matched.csv` (recall targets 0.90 / 0.95 /
0.99). The fair comparison is latency at a recall target: LiNR V1 and V2 are
exact, while SilverTorch's recall depends on how many clusters it probes.

### At SilverTorch's production setting (24 probes)

SilverTorch is faster than the best exact LiNR variant in every gated cell,
from about 1.3× to 50×. The gap grows with batch size and with N: 1.5-2×
at bs 1 on Goodreads (0.8 M), about 10× at bs 16 on arXiv (3 M), 20-50× at
bs 16 on PubMed and YFCC (10 M). But its recall at 24 probes is often far from exact: about
0.89 on Goodreads `c0_genre`, 0.68-0.85 on arXiv, 0.39-0.55 on PubMed, 0.45
on YFCC `tags_and`, and as low as 0.05 on the most selective synthetic
filter.

### At matched recall (0.95, k 100)

| dataset (N, d) | filter | bs 1 | bs 16 |
|---|---|---|---|
| Goodreads (0.8 M, 128) | `c0_genre`, `all4` | SilverTorch (0.57-0.65 of LiNR) | SilverTorch (0.28-0.68) |
| goodreads-synth (0.8 M, 128) | pass rate 0.01 to 1 | SilverTorch (0.54-0.75) | SilverTorch (0.29-0.65) |
| goodreads-synth | pass rate 0.001 | LiNR V2 (SilverTorch tops out at 0.84) | LiNR V2 |
| arXiv (3 M, 128) | `c0_maincat`, `all4`, clause or bloom | SilverTorch (0.33-0.48) | SilverTorch (0.23-0.50) |
| arxiv-synth (3 M, 128) | pass rate 0.01 to 1 | SilverTorch (0.32-0.58) | SilverTorch (0.30-0.93) |
| arxiv-synth | pass rate 0.001 | LiNR V2 (SilverTorch tops out at 0.94) | LiNR V2 |
| PubMed (10 M, 768) | `c0_mesh`, `all5` | LiNR V2 (SilverTorch tops out at 0.94 / 0.87) | LiNR |
| YFCC (10 M, 192) | `tags_and` | LiNR V2 (SilverTorch tops out at 0.72) | LiNR V1 |
| LAION 1 / 3 / 10 / 30 M (256) | `c0_domain`, `tags4`, synth | adapted SilverTorch not gated yet | |

At target 0.90, PubMed `c0_mesh` moves to SilverTorch at k 100 (0.10-0.81
of LiNR at 1024 probes). At k 1000 every target is harder for SilverTorch:
on arXiv `all4`, Goodreads `all4`, and both synthetic sets below pass rate
0.01, it does not reach 0.95 within the gated probes.

"Tops out" means the highest gated probe count: 512 of 2048 clusters on
arXiv, 1024 of 4096 elsewhere. The campaign's Triton records reach higher
recall with more probes or more clusters: on PubMed and LAION 30 M,
16384 lists with 4096 probes gives 0.89-0.96 (`c6_ceiling.csv`). Those
settings are not gated for the adapted code, so the LiNR wins in the
10 M rows hold only up to the gated probe counts.

### Index memory and build time

From the final pass `filter` records (`memory.csv`): SilverTorch's int8
index (Meta's code as published) takes about half of LiNR's fp16 index
(arXiv 483 vs 1,095 MiB, PubMed 7,965 vs 15,869 MiB). It needs a k-means
build (2 s at 0.8 M, 85 s at 10 M d768); LiNR builds in under 1.1 s. The
adapted code's own index size and build time come with the NUMBERS legs.

## How the parameters move the crossover

Figures: `fig_h2h_synth` (latency against pass rate, LiNR V1 / V2 and
SilverTorch at each gated probe count) and `fig_c1_v2_over_v1`.

- **Probes (recall target).** This is the main lever. SilverTorch's latency
  grows with probes, slowly at bs 1 and steeply at bs 16. A 0.95 target
  costs 256-1024 probes, which keeps SilverTorch ahead at ≤ 3 M (0.23-0.93
  of LiNR) but takes most of its lead.
- **Filter pass rate.** At a fixed probe count, SilverTorch's recall falls as
  the filter becomes more selective, because the passing items are spread
  over clusters it does not probe. Below about 0.001 (synthetic) or 0.0002
  (PubMed `c0_mesh`) it cannot reach 0.95 within the gated probes, and LiNR
  V2 wins. Above about 0.01 SilverTorch wins at matched recall. LiNR's own
  best variant switches with pass rate (below).
- **Batch size.** A larger batch helps SilverTorch: at bs 16 the ratio at
  matched recall is lower than at bs 1 in most rows. LiNR V2's
  gather-then-score loses its advantage at bs 16 except at very low pass
  rates.
- **k.** k 1000 needs more probes for the same recall, so SilverTorch's
  matched-recall window shrinks.
- **N and d.** At 10 M (PubMed d768, YFCC d192), recall at a fixed probe
  fraction drops and LiNR's exact scan costs more. The gated probe counts
  are not enough to settle the 0.95 comparison. The LAION N-sweep at fixed
  d 256 (1 to 30 M) is the planned test for N alone; it has only LiNR and
  Triton records so far (`linr_vs_n.csv`).
- **The adapted code against Triton.** The bridge carries the adapted ÷
  Triton ratio, so the four regimes where the adapted code still loses to
  Triton (wide unfiltered scans at d ≥ 192, the full-N bloom path,
  partial bloom at bs 1 in graph mode, exact filtering at wide probes) are
  in these numbers.

## Claim verification

The reproduction (each paper's claims against our measurement) is kept
apart from the head-to-head above. Claims C1-C7 are the current list in
[claims](claims.md). Its rewording for this study is in progress and has
not been merged, so this section will be reconciled with it.

### LiNR (our implementation, final pass; NOT CITABLE until D1-G)

| claim | verdict | evidence |
|---|---|---|
| C1: V1 beats V2 at high pass rate, V2 wins at low | **reproduced at bs 1, weak at bs 16.** At bs 1, V2 wins below a pass rate of 0.79-0.97 at every N and d, and by 2-5× at 0.001. At bs 16, V2 wins only below about 0.02 (0.065 at arxiv-synth d256), by at most 25 %, and never on laion1m or yfcc10m-synth. Real filters at bs 16: V2 wins only on PubMed `c0_mesh` | `c1_crossover.csv`, `c1_curves.csv`, `c1_real.csv` |
| C2: V3 keeping 1 % gives about 10 % at near-parity recall | **partially.** At bs 16 on real filters, V3 runs at 0.75-0.92 of V2 (median); on goodreads-synth it is slower (1.09). At bs 1 it is slower everywhere (median 1.03-1.35). Recall is near parity only with enough bits (PubMed 768 bits: 0.99-1.00; goodreads 128 bits: 0.87-0.98; arXiv default: 0.61-0.92) | `c2_v3.csv` |
| C3: the custom filter kernel is about 100× faster than native masking | **direction reproduced, size not.** Eager PyTorch is 1.4-2.0× slower at bs 1 and 5.8-6.8× at bs 16. `torch.compile` max-autotune runs at 0.5-1.06× of Triton at bs 1 (median 0.85-0.96, so mostly faster) and 1.5-4.8× slower at bs 16 | `c3_torch.csv` |

### SilverTorch: the adapted code (provisional) and Meta's code as published

| claim | verdict | evidence |
|---|---|---|
| C4: bloom false-positive rate about 0.067 at 1024 bits | **ours is lower.** At 1024 bits, 0 false positives on arXiv and Goodreads and 1e-5 on PubMed `c0_mesh` (k_hash 3). False positives appear only at ≤ 128 bits (≤ 512 on PubMed). The adapted code's default bloom adds an exact recheck, so it returns no false positives by construction; its recall gate is still running | `c4_bloom.csv` |
| C5: co-design (probe then filter) 1.79-2.15× faster than filter then probe | **Meta's code as published:** holds at 30 M (1.07-2.20 at bs 16, 1.69-5.56 at bs 64) and at 10 M bs 64 (1.13-2.35); reversed at 0.8-3 M at bs 1-16 (0.78-0.89); mixed at 3 M bs 64 and 10 M bs 16. **Adapted code (provisional):** neutral at bs 1 (median 0.93-1.03); at bs 64 it gives 1.08-1.56 (median), up to 3.2 on multi-clause filters; PubMed 1.26 at bs 16 | `c5_meta.csv`, `c5_adapted.csv` |
| C6: int8 IVF cannot reach 0.95 recall; selective filters need more probes | **partially.** At 24 probes recall on real filters is 0.33-0.95 (depending on the list count) and falls with N and with selectivity (0.05 on goodreads-synth at pass rate 0.001). With 16384 lists and 4096 probes, int8 reaches 0.95-0.99 on Goodreads, arXiv and the synthetic sets, but stays at 0.89-0.95 on PubMed and 0.90 on LAION 30 M `c0_domain` (fp16: 0.90-0.98). So 0.95 is reachable, at a cost in probes | `c6_recall.csv`, `c6_ceiling.csv` |
| C7 (restated): the adaptation keeps Meta's results and removes its overheads | **ids and scores identical to Meta's code in 1,536 / 1,536 gate cells; eager time 0.12-0.82 of Meta's (median per dataset and filter), at most 1.03** (arXiv d256 unfiltered, within the timing floor) | `adaptation.csv`; the gate's own record in [validation](../validation.md) |

The four paper claims tested only by arithmetic or not testable here
(production pipelines, TCO, A/B tests, scale above our data) are listed in
[claims](claims.md) and are not repeated here.

## Open before this page is citable

- The adapted code's NUMBERS legs over every SilverTorch cell (replaces the
  bridge; brings LAION 1-30 M, index memory and build time, and the deep
  probe curves that decide the 0.95 rows at 10 M).
- The adapted code's LAION 30 M gate cells.
- The adapted code's bloom recall gate (default width with the exact recheck).
- D1-G declared green, which covers every LiNR number.
- Reconciling with the rewritten claims list once it is merged.
