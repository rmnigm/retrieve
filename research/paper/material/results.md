---
title: results (material C3)
created: 2026-10-12
updated: 2026-10-12
type: summary
tags: [paper]
sources: [research/paper/outline.md, research/paper/results.md, research/paper/claims.md, research/paper/reproduction-deviations.md, research/validation.md, research/decisions.md, research/artifacts/hub-index.md, research/artifacts/results-draft/build_results.py, research/artifacts/campaign-final/README.md]
---

# Material C3: results, verdicts, deviations that move them

Collected facts for the [outline](../outline.md)'s §2 (T1 verdict column),
§4 (C1-C6), §5 (F3, T2), §6 (T3 numbers) and §7 (deviations that move a
verdict). Facts with sources, not prose.

**How each number was checked.** Every table number was re-read from Hub
`artifacts/results-draft/20261012` (manifest sha256 `aef86983…`, fetched with
`bench fetch`, 28 files byte-identical to the build output) or
`artifacts/results-draft/20261012-checks`, and the seed and LAION-pool
numbers from the leg parquets (`campaign-final/merged-4016`,
`numbers/L2b-v3pool-laion`). "verified" means the source gives the same
value at the stated rounding. "MISMATCH" means the refreshed
[results draft](../results.md) or [validation](../../validation.md) says
something else, and the source value follows. "C3-derived" means a number we
computed from the records that no `build_results.py` table carries.

**Citability.** [Validation](../../validation.md#numbers-adapted-silvertorch-over-the-study-grid):
the NUMBERS legs, the final pass (D1-G) and the META-FORK gates are green, so
their numbers are **citable**, with these exclusions:
- the yfcc10m-synth LiNR cells at k 1000 (HB-6), which are flagged;
- LiNR V2 at bs 256 on PubMed `c3_journal_reverse` / `all5` and on YFCC,
  which is a TIMEOUT;
- the cells unstable in every run, which are flagged.

Records of our Triton SilverTorch feed no number. Where a fact exists only on
Triton records, this page says **NOT ON THE STUDY'S ARM**. All numbers are
seed 0 unless stated otherwise. No paired CI exists in any table yet (see
[gaps](#gaps-that-need-a-decision)).

## T1 verdict column (§2)

These verdicts use the outline's §3 vocabulary. Each holds only under the
condition given with it.

| claim | verdict | condition | alternative wording (survey pattern 6) |
|---|---|---|---|
| C1 (L-3…L-5) | **reproduced in shape at bs 1; the crossover location differs from LiNR's; weak at bs 16; gone at bs ≥ 64** | a V1 / V2 crossover in pass rate exists at bs 1 at every N and d. LiNR's own high-pass setting (p ≈ 0.11) has V1 faster by 3.0×; ours has V2 faster there (fused V2, LN-1 / L-12) | "partially confirmed: the crossover exists, at a far higher pass rate than in the original" |
| C2 (L-6) | **not comparable as stated** (end-task recall, unpublished protocol); under our metric **holds in direction, not in size** at bs 16 | 1 % pool: V3 at 0.83 of V2 at bs 16, recall 0.89 (median); slower at bs 1. Goodreads end-task bridge below | "cannot be fully assessed due to the recall definition; on our metric, the gain appears only at bs 16" |
| C3 (L-2) | **holds in direction, not in size** | eager torch 1.4-2.0× (bs 1), 5.8-6.8× (bs 16), never near 100×; `torch.compile` is often faster at bs 1 | "the gap is significantly narrower than in the original" |
| C4 (S-8…S-10, S-14) | **FPR: ours lower** (Meta as published); the width curve and memory are **not testable on the study's arm** (see C4); the transposed layout is **not in the released code** (OF-1) | 1024-bit point on Meta's code as published | "confirmed within the original evaluation scope (FPR); the layout claim cannot be assessed on the released code" |
| C5 (S-3, S-11) | **Meta as published: reproduced only at 30 M (and 10 M bs 64), reversed at 0.8-3 M.** The adapted column reads "refused by design (K17)" (outline `d80b262`) | N and batch size decide it; the reversal is host work in the published partial path (OF-9) | "holds at the original's scale, reverses at smaller N" |
| C6 (S-6, S-7, S-16) | **reproduced in shape** (recall falls with selectivity and N at fixed probes); the **0.95 ceiling is dataset-dependent**: reached on Goodreads, arXiv, LAION `tags4` at 10 / 30 M and 18 of 31 synthetic sweeps, and not reached on LAION `c0_domain`, PubMed or YFCC | adapted code, probe curves up to `n_lists` / 4 | "partially confirmed: holds on PubMed, YFCC and LAION `c0_domain`, not on Goodreads and arXiv" |
| C7 (ours) | **holds**: bit-exact to Meta's code as published, faster end to end | gate grid ≤ 10 M plus the 30 M bloom cells | — |

Pre-refresh → refreshed `results.md` (Q5):
- **C2**: "partially" → "not comparable as stated", with the pool curve.
- **C6**: was "0.95 is reachable, at a cost in probes". That rested on
  16384-list / 4096-probe ceiling runs, which are Triton. It is now
  "not reached on LAION `c0_domain`, PubMed, YFCC", read on the adapted
  curves.
- **C1**: adds "gone at bs 64 / 256", correlated filters and k 2048. It
  drops three earlier facts, all still true per `c1_crossover.csv` /
  `c1_real.csv`:
  - "Real filters at bs 16: V2 wins only on PubMed `c0_mesh`";
  - "never on laion1m or yfcc10m-synth";
  - the d256 crossover at 0.065.
- **C4**: drops "false positives appear only at ≤ 128 bits". That was the
  Triton width curve, so dropping it is right.
- **C5**: drops "mixed at 3 M bs 64 and 10 M bs 16".
- **C7**: adds 30 M and the 2,151 eager / graph id checks.

## §4 claim by claim

### C1: V1 beats V2 at high pass rate, V2 wins at low (L-3, L-4, L-5)

**Original** (arXiv v3 §5.3.1-5.3.2, Tables 3-4, [claims](../claims.md)
L-3 / L-5):
- High-pass set, about 11 % pass, d128 fp16. At B = 1, PyTorch V1 averages
  4.8 ms and V2 14.6 ms, so V2 / V1 = 3.0. At B = 16, V1 takes 22.8 ms.
- Low-pass set: V2 takes 1.9 ms at B = 1 and 21.4 ms at B = 16.
- Recall@2000 (end-task) is 0.688 for all variants.
- LiNR's latencies are un-fused ops (L-12).

**Ours** (`c1_crossover.csv`, `c1_curves.csv`, `c1_real.csv`,
`linr_numbers.csv`; synthetic uniform filters, graph mode, k 100):

| fact | value | source row | check |
|---|---|---|---|
| bs 1 crossover p | 0.79 (yfcc10m-synth d192) to 0.97 (arxiv-synth d256) at every N and d | `c1_crossover` bs 1 `crossover_p` | verified |
| bs 1, V2 / V1 at p ≈ 0.001 | 0.195 (laion30m) to 0.643 (goodreads) | `c1_crossover` `v2_over_v1_lowest_p` | **MISMATCH**: results.md says V2 wins "by 2-5×"; the source gives 1.6-5.1× |
| bs 1, V2 / V1 at p ≈ 0.1 (LiNR's high-pass point) | 0.29-0.68 (V2 faster); LiNR: 3.0 (V1 faster) | `c1_curves` p 0.08-0.12 | C3-derived |
| bs 16 crossover p | 0.017 (arxiv d128) to 0.024 (laion30m), but 0.065 at arxiv-synth d256; **none** on laion1m-synth or yfcc10m-synth | `c1_crossover` bs 16 | verified; results.md's "about 0.02" omits d256 and the two no-crossover sets |
| bs 16, V2's largest edge | V2 / V1 0.75 (arxiv d256; 25 %); 0.87-0.96 on the other sets with a crossover | `c1_crossover` bs 16 `v2_over_v1_lowest_p` | verified |
| bs 16 at p 1 | V2 / V1 2.6-6.8 | `c1_crossover` `v2_over_v1_p1` | verified |
| bs 16, V2 / V1 at p ≈ 0.1 | 1.12-1.58 | `c1_curves` | C3-derived |
| real filters, bs 1 | V2 faster on every sweep except PubMed `c3_journal_reverse` (p 0.999, 1.07) | `c1_real` | verified |
| real filters, bs 16 | V1 faster everywhere except PubMed `c0_mesh` (p 0.00019): V2 / V1 0.53 | `c1_real` | verified. Not on the refreshed page; the "at most 25 %" there holds for synthetic filters only |
| bs 64 | V2 1.33-35.4× slower on every real filter; LAION 1.95-2.23 | `c1_real` / `linr_numbers` L1, L1b | verified ("2.0-2.2") |
| bs 256 | V2 3.86-55.1× slower where it finished; TIMEOUT on PubMed `c3_journal_reverse` / `all5` and YFCC `tags_and` | `linr_numbers` `V2_status` | verified |
| correlated (arxiv-corr-synth, p 0.010-0.107) | bs 1: 0.351-0.415 (k 100), 0.366-0.427 (k 1000); bs 16: 0.92-1.26 | `linr_numbers` L3 | verified at k 100 ("0.35-0.41") |
| k 2048 | bs 1: V2 / V1 0.084-0.769; bs 16: V2 wins only on PubMed `c0_mesh` (0.52); arXiv 1.34, Goodreads 1.61 | `linr_numbers` L4 | verified |
| seeds 0-2 | at the bs 16 cells beside the crossover (arxiv-synth p 0.02 / 0.05, laion30m-synth p 0.01 / 0.02, yfcc10m-synth p 0.01 / 0.02; k 100 and 1000), V2 / V1 differs across seeds by ≤ 0.011; 0 / 48 records unstable | `campaign-final/merged-4016`, suites `seeds-c1`, `seeds-c1-laion30m`, graph | C3-derived |

**Mechanism.** Our V2 is a fused compaction plus scoring kernel (LN-1).
LiNR §6 says its numbers are un-fused and predicts that fusing changes the
V1 / V2 picture (L-12). This explains why the bs 1 crossover moves from
below 11 % to 0.79-0.97. Batching favours V1: V2 gathers per query, V1
scores one shared matrix.

### C2: V3 keeping ~1 % gives ~10 % at near-parity recall (L-6)

**Original.** §5.3.1, Figure 7: Sign-OPORP at 512 bits, keeping 1 % for
rescoring, gives about 10 % further latency gain at "nearly parity"
recall@2000. That recall is end-task; Figure 7 sweeps filter size 0.5-2.5 %.

**Ours** (`c2_v3.csv`, suite `n-v3pool` = NUMBERS `L2-v3pool`; real
filters on arXiv, Goodreads, PubMed, YFCC; 10 filter rows per pool):

| pool | median recall@100 | V3 / V2 bs 16 (median) | V3 / V2 bs 1 (median) | check |
|---|---|---|---|---|
| 0.5 % | 0.843 | 0.828 | 1.035 | verified ("0.84") |
| 1 % | 0.886 | 0.832 | 1.084 | verified ("0.89", "0.83", "1.08") |
| 2 % | 0.927 | 0.857 | 1.083 | — |
| 5 % | 0.950 | 0.900 | 1.084 | — |
| 10 % | 0.978 | 0.937 | 1.097 | — |
| 20 % | 0.992 | 1.012 | 1.141 | verified ("0.99", "1.01", "1.14") |

- **LAION pool curve (L2b) is missing from `c2_v3.csv`.** `build_results.c2`
  merges V3 with V2 on rows of the same suites, and L2b ran V3 only, so all
  48 LAION records drop out. The outline names L2b as a C2 source.
  C3-derived from `numbers/L2b-v3pool-laion` (graph, k 100), with V2 taken
  from the `h2h` legs. This is a **different process, so the ratio is not
  interleaved**.
  - Median recall across LAION 1-30 M `c0_domain` / `tags4`: 0.894 at
    0.5-1 %, 0.917 at 2 %, 0.943 at 5 %, 0.949 at 10 %, 0.978 at 20 %.
  - V3 / V2 is 0.64 at bs 16 and 1.11-1.14 at bs 1, at every pool.
  - Below 5 % the pool is flat: identical recall at 0.5-2 % on most
    filters. This suggests a floor on the pool size; not explained.
- **Recall vs bits** (`c2_v3` suite `v3bits`, bs 16). Our code length is
  capped at d (LN-8); LiNR used 512.
  - Goodreads d128: 1 % pool 0.64-0.78 at 64 bits, 0.87-0.94 at 128 bits.
  - goodreads-synth: 128 bits 0.87-1.00 against 0.64-1.00 at 64.
  - PubMed: 0.93-0.99 at 256 bits, 0.990-0.998 at 768.
  - So near parity (≥ 0.99) at a 1 % pool needs d ≥ 768 bits here
    (PubMed). At d128 it is not reached.
- **End-task bridge** (Goodreads d128, `product_recall.csv`, LiNR V3 rows;
  held-out recall over the exact scan's). At a 1 % pool:
  - k 100: 0.89 (`c0_genre`), 0.93 (`c1_lang_reverse`), 0.95 (`all4`);
  - k 1000: 0.73-0.87.
  - A 5 % pool gives 0.95-0.98 at k 100 and 0.87-0.95 at
    k 1000.
  - This is the closest we get to LiNR's "nearly parity" in its own terms,
    on one dataset, at k ≤ 1000 (LiNR: 2000). C3-derived from the
    published CSV.

### C3: custom filter kernel ~100× over native masking (L-2)

**Original.** §6: native TF / PyTorch masking and indexing costs about 100×
in latency.

**Ours** (`c3_torch.csv`; suite `c3`: synthetic p 0.01 / 0.1 / 1 on
arxiv-synth d128 / d256, goodreads-synth d128 / d256, laion30m-synth,
yfcc10m-synth; suite `c3-real`: 10 real filters):

| arm against Triton | bs 1 | bs 16 | check |
|---|---|---|---|
| eager torch (eager mode) | 1.37-2.01, median 1.76 | 5.78-6.76, median 6.26 | verified ("1.4-2.0", "5.8-6.8") |
| eager torch (graph mode) | 1.39-1.58 | 1.68-1.86 | not on the page; it narrows the gap further |
| `torch.compile` max-autotune, synthetic | 0.50-0.96, median 0.85 | 1.49-2.33 | verified |
| `torch.compile` max-autotune, real | 0.56-1.06, median 0.96 | 1.96-4.82, median 2.21 | verified ("0.5-1.06", "1.5-4.8") |

**Mechanism.** The page attributes the bs 1 compile win to reduction order
and fp16 top-k keys. That rests on the [validation](../../validation.md)
V1-FUSE row, which C2 collects. Not re-checked here.

### C4: bloom FPR ≈ 0.067 at 1024 bits, memory (S-8…S-10, S-14)

**Original.** §6.2.2, Figure 8b, 40 M items:
- FPR 6.98 at 512 bits and 0.067 at 1024 (unit not stated; read as %).
- Memory 2.56 / 3.84 / 5 / 10 GB at 512 / 768 / 1024 / 2048 bits.

§6.2.4 says 0.00173 % → 3.89 % from 1024 to 768 bits (S-10). That is
inconsistent with S-9 and is the paper's own inconsistency.

**Ours** (`c4_bloom.csv`, read from `campaign-final` suite `bloomwidth`):

| backend | what | value | check |
|---|---|---|---|
| `official` (Meta as published; ignores `m_bits`, OF-10) | FPR, k_hash 3 / 5 | arXiv `c0_maincat` / `c3_nversions` / `all4` 0 / 0; Goodreads `c0_genre` 0 / 0; PubMed `c0_mesh` 9.3e-6 / 5.6e-8 | verified ("1e-5") |
| `triton` (our reimplementation) | FPR and recall at `m_bits` 64-2048 | 52 of 70 rows | **NOT ON THE STUDY'S ARM**: the whole width curve in `c4_bloom.csv` is Triton |
| adapted (`official-fork`) | false positives with its default bloom (C12 width + C13 recheck) | 0 on every cell, `m_bits` 64-2048 on Goodreads / arXiv / PubMed (36 cells) | [validation](../../validation.md#meta-fork-backendofficial-fork) gate R; citable |

- **F4a as planned (FPR and memory vs bits on the adapted code) has no
  source.** No NUMBERS `bloomwidth` leg ran (the 24 `numbers/*` legs list
  none). The adapted code's pre-recheck FPR is not recorded, and
  `c4_bloom.csv` has no memory column. The adapted bloom's index memory
  appears once, in `memory.csv`: arXiv bloom 1,141 MiB against Meta's 483
  MiB at the default width; Goodreads 306 against 145. That index is the
  configured width (C12), so it is larger than Meta's.
- **S-14 / OF-1:** the transposed layout is not in Meta's released code (a
  per-2048-doc bundle bloom instead). It is a paper-vs-code finding, not a
  measurement.
- Units: our FPR is a fraction. At 1024 bits Meta's code gives ≤ 9.3e-6
  (0.00093 %), against the paper's 0.067 (%) or 0.00173 %. Under either
  reading ours is lower or equal.

### C5: co-design 1.79-2.15× over full mask then probe (S-3, S-11)

**Original.** §6.2.3, Figure 9, at 20 M × 128: latency 1.55 → 0.72 ms,
1.79-2.15× on average. §6.1.1 gives about 17-25 % QPS end to end.

**Meta as published, eager** (`c5_meta.csv`; final pass `codesign`,
`nsweep-codesign`, `codesign-laion30m`; full / partial, above 1 means
co-design wins):

| N | bs 1 | bs 16 | bs 64 | check |
|---|---|---|---|---|
| 0.8 M | 0.78-0.80 | 0.81-0.89 | — | verified |
| 1 M (LAION) | — | 0.80-0.81 | 0.83-0.94 | verified |
| 3 M | 0.79-0.82 | 0.78-0.85 | 0.87-1.25 (mixed) | verified; "mixed" was dropped from the page |
| 10 M | — | 0.83-1.20 (median 0.99, mixed) | 1.13-2.35 | verified; "mixed" was dropped |
| 30 M | — | 1.07-2.20 | 1.69-5.56 | verified |

**Adapted: not the study's arm.** The outline (`d80b262`) evaluates C5 on
Meta's code as published only. The adapted code has one bloom path over the
probed columns and refuses `bloom_path="full"` (K fix 17, `ac8f310`, and
B2). `c5_adapted.csv` is read from the `head-82fafb6` gate grid. `82fafb6`
does **not** contain `ac8f310`, so its "full" is a full-N path that the
released adapted code no longer has. It is a **record of an intermediate
version: do not cite it as the adapted code's C5.** Its values, for the
record (median per dataset × d):

| bs | graph median | eager median | max | check |
|---|---|---|---|---|
| 1 | 0.93-1.03 | 0.99-1.04 | 1.12 | verified ("0.93-1.03") |
| 16 | 1.03-1.26 | 1.01-1.26 | 1.69 | PubMed 1.26 (dropped from the page) |
| 64 | 1.08-1.56 | 1.08-1.55 | 3.19 (arXiv `all4`) | verified ("1.08-1.56", "up to 3.2") |

- **Mechanism:** in Meta's published partial path, host work grows with
  the sub-query count. That is OF-9: partial is 14-22 % slower at 0.8-3 M
  and 1.70-3.69× faster at 30 M
  ([deviations](../reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)).
  The adapted code builds the partial mask in its payload kernels, so it is
  not host-bound.
- **Gaps against the brief:** no CIs on Meta's ratios (seed 0, one run per
  cell). results.md's C5 row still quotes the `82fafb6` adapted values
  ("neutral at bs 1 … up to 3.2"); under the new outline they leave the
  page.

### C6: int8 IVF recall ceiling < 0.95; selective filters need more probes (S-6, S-7, S-16)

**Original.** §6.2.1, §4.2, §7:
- "cannot reach 0.95 recall";
- "no measurable recall loss at 64 probes and top-2048";
- selective filters may need more probes.

Figure 7 covers 20 M × 128, B = 16, recall 0.35-0.92.

**Ours, adapted code** (`c6_recall.csv` from NUMBERS A1 / A1b; `at24.csv`
from the timed legs).

At 24 probes (`at24`, bs 16, k 100, recall@100):
- goodreads-synth, 4096 lists: 0.930 / 0.775 / 0.365 / 0.047 at p 1 / 0.1 /
  0.01 / 0.001. Verified ("0.93 to 0.05").
- Real filters: Goodreads 0.42-0.89, arXiv 0.68-0.85, PubMed 0.39-0.81, YFCC
  0.44, LAION 0.33-0.57. Verified.

Highest recall@100 on the probe curve, with the smallest
(`n_lists` / `n_probe`) reaching 0.90 / 0.95 (k 100):

| dataset | filter (p) | max | 0.90 at | 0.95 at |
|---|---|---|---|---|
| Goodreads 0.8 M | `c0_genre` (0.33) | 0.986 | 4096/32 | 4096/64 |
| | `c1_lang_reverse` (0.62) | 0.984 | 4096/128 | 4096/256 |
| | `all4` (0.041) | 0.971 | 4096/512 | 4096/1024 |
| arXiv 3 M | `c0_maincat` (0.14) | 0.966 | 2048/64 | 2048/128 |
| | `c3_nversions` (0.44) | 0.966 | 2048/64 | 2048/128 |
| | `all4` (0.0094) | 0.963 | 2048/256 | 2048/512 |
| LAION 1 M | `c0_domain` (0.0095) / `tags4` (0.0027) | 0.846 / 0.833 | — | — |
| LAION 3 M | `c0_domain` / `tags4` | 0.871 / 0.909 | — / 8192/2048 | — |
| PubMed 10 M | `c0_mesh` (0.00019) | 0.9495 | 4096/512 | — |
| | `c3_journal_reverse` (0.999) | 0.922 | 4096/256 | — |
| | `all5` (0.018) | 0.898 | — | — |
| YFCC 10 M | `tags_and` (0.018) | 0.718 | — | — |
| LAION 10 M | `c0_domain` / `tags4` | 0.897 / 0.951 | — / 4096/1024 | — / 65536/16384 |
| LAION 30 M | `c0_domain` / `tags4` | 0.908 / 0.961 | 16384/4096 / 4096/1024 | — / 16384/4096 |

- **Synthetic filters.** 0.95 is reached on 18 of 31 synthetic sweeps
  (7 uniform sets × 4 pass rates + 3 correlated), including all three
  arxiv-corr-synth sweeps. It is not reached on:
  - p 0.001 at 0.8-10 M;
  - laion3m-synth p 0.01;
  - any laion1m-synth sweep (only 512 lists, probes ≤ 128: a short curve);
  - any yfcc10m-synth sweep (max 0.55-0.81).

  At 30 M p 0.001, 0.95 needs 65536 / 16384.
- **Uniform vs correlated** (arXiv, 2048 lists). Correlated p 0.01-0.1
  reaches 0.90 at 64 probes and 0.95 at 256. Uniform p 0.01 needs 256 /
  512, and uniform p 0.1 needs 64 / 256. Correlated filters are no harder
  than uniform ones at the same pass rate, and easier at p 0.01.
- **k 1000 is harder.** Max recall@1000 at p 0.001: 0.23 (goodreads-synth),
  0.61 (arxiv-synth). Goodreads `all4`: 0.90.
- **Page wording check.** The page says "not reached on LAION `c0_domain`
  at any size (top 0.85-0.91)", PubMed 0.9499 and 0.92, and YFCC 0.72; all
  verified.
- **MISMATCH (setting and arm).** The page's "YFCC (0.72; per-row int8
  0.75, fp16 0.97)" puts together:
  - the real `tags_and` maximum (0.72, adapted), and
  - per-row int8 / fp16 values from `c6_ceiling` yfcc10m-synth p 1 at
    4096 / 256, which come from a **Triton** index.
- **Ceiling decomposition (int8 vs fp16).** Every `c6_ceiling.csv` row was
  computed by `ceiling_check.py` on a `backend="triton"` index, both the
  `campaign-final/ceiling-int8-*` rows and `numbers/A1b-laion-small-lists/ceil`.
  So this is **NOT ON THE STUDY'S ARM**.
  - The "shipped" column agrees with the adapted probe curve at the same
    (`n_lists`, `n_probe`) within 0.0003 where both exist, and within
    0.011-0.018 on YFCC.
  - The per-row int8 and fp16 columns are counterfactuals with no
    adapted-code equivalent.
  - Affected statements: "fp16 0.996" (PubMed `c3_journal_reverse`), the
    YFCC per-row / fp16 values, and "fp16 would add at most 0.006" on LAION
    1 / 3 M.
  - **MISMATCH** on that last one: the source gives up to 0.009 (LAION 3 M
    `c0_domain`, 0.8716 → 0.8808), and validation's "≤ 0.006" has the
    same error.
- **Mechanism, YFCC** (`score_gaps_yfcc_pubmed.json`, CPU, citable per
  validation):
  - the items sit in a narrow cone (a random pair's cosine is 0.97);
  - the top-100 spans 0.0022 in score, and the gap from rank 100 to 101 is
    3.9e-6;
  - fp16 storage moves a score by 1.5e-5, and per-row int8 by 1.9e-4.
- **Local pass rate** (Spearman, Hub
  `artifacts/exhibits/20261009-1057-gls`): those records are **Triton**
  (pre-study exhibits). **NOT ON THE STUDY'S ARM**, not re-read here.

## §5 SilverTorch against LiNR

### F3 points (Pareto per dataset)

- **SilverTorch probe curve:** `h2h.csv` (graph) or `h2h_eager.csv`.
  Columns `dataset, filter_kind, sweep, perf_bs, perf_k, n_lists, n_probe,
  recall, st_ms`.
- **LiNR V1 / V2:** the same rows, columns `V1`, `V2` (ms, interleaved in
  the same process), with `linr_best` / `linr_ms` / `st_over_linr` /
  `winner` and the `flagged` and `leg` columns. 1,677 rows; 30 flagged
  (`summary.json`).
- **V3 pool curve:**
  - `c2_v3.csv` (suite `n-v3pool`, recall and V3 / V2 only, no ms column);
  - L2b LAION from `numbers/L2b-v3pool-laion` directly (absent from
    `c2_v3`, see C2).
- Use bs 16. bs 1 changes the picture where LiNR V2 is the best LiNR (real
  filters at bs 1, `results.md`) and on `c3_journal_reverse` (V1).

### T2: matched-recall winners

Source: `tuned.csv` (1,089 rows), `by_size.csv`. A cell = dataset × filter
× bs × k × target. Winner and ratio = SilverTorch over the faster of LiNR
V1 / V2. Clause filters, k 100:

| dataset | filter | 0.90, bs 1 / 16 / 64 | 0.95, bs 1 / 16 / 64 | check |
|---|---|---|---|---|
| Goodreads | `c0_genre` | 0.58 / 0.25 / 0.16 | 0.58 / 0.26 / 0.19 | verified |
| | `c1_lang_reverse` | 0.51 / 0.31 / 0.25 | 0.51 / 0.41 / 0.40 | verified |
| | `all4` | 0.56 / 0.46 / 0.52 | 0.56 / 0.73 / 0.82 | verified |
| arXiv | `c0_maincat` | 0.46 / 0.18 / 0.22 | 0.57 / 0.63 / 0.98 | verified |
| | `c3_nversions` | 0.27 / 0.19 / 0.23 | 0.31 / 0.47 / 0.65 | verified |
| | `all4` | 0.29 / 0.32 / 0.39 | 0.33 / 0.53 / 0.64 | verified |
| LAION 3 M | `tags4` | 0.50 / 0.43 / 0.66 | not reached | verified |
| PubMed | `c0_mesh` | 0.41 / 0.27 / 0.34 | not reached | verified |
| | `c3_journal_reverse` | 0.04 / 0.20 / 0.47 | not reached | verified (0.045) |
| LAION 10 M | `tags4` | 0.26 / 0.39 / 0.66 | 0.41 / 0.45 / 0.78 | verified |
| LAION 30 M | `tags4` | 0.15 / 0.20 / 0.33 | 0.22 / 0.38 / 0.63 | verified |
| | `c0_domain` | 0.42 / 0.40 / 0.71 | not reached | verified |

Not reached at 0.90, so LiNR wins by default: LAION 1 M (both filters),
LAION 3 M / 10 M `c0_domain`, PubMed `all5`, YFCC `tags_and`. All verified.

**Counts** (clause; reached / cells, then SilverTorch wins):

| k | target | bs 1 | bs 16 | bs 64 | check |
|---|---|---|---|---|---|
| 100 | 0.90 | 35/49, 35 | 35/49, 35 | 35/49, 33 | verified ("35 of 49") |
| 100 | 0.95 | 26/49, 26 | 26/49, 25 | 26/49, 23 | verified |
| 1000 | 0.90 | 24/45, 24 | 24/45, 24 | 24/45, 21 | verified |
| 1000 | 0.95 | 17/45, 17 | 17/45, 16 | 17/45, 10 | verified ("LiNR V1 wins 7 of 17 at bs 64") |

- Median ratio of reached cells, k 1000, bs 1, 0.90: 0.42. Verified.
- **Bloom filters** (bloom; reached cells): k 100 8 / 12 at 0.90 and 5 / 12
  at 0.95; k 1000 4 / 12 and 2 / 12. SilverTorch wins all of them, ratio
  0.09-0.85. Verified.
- **k 2048** (A3 + L4): arXiv `c0_maincat` 0.23-0.72 and Goodreads
  `c0_genre` 0.30-0.81 at 0.90 / 0.95, verified. PubMed `c0_mesh` tops out
  at 0.856, so LiNR wins.
- **Paired CI: none.** The tables carry point ratios only. The interleaved
  groups (`interleave_group` in the A2 / A3 parquets) allow a paired CI, but
  no script computes one yet.

**Regime boundaries (Q8).**
- **Pass rate.**
  - At p 0.001, SilverTorch does not reach 0.90 at ≤ 3 M except
    arxiv-synth (0.57 / 0.35 / 0.59). It reaches it at 10 M and 30 M with
    more lists (0.25-0.72).
  - At p ≥ 0.01, it wins wherever it reaches the target (0.03-0.94), except
    LAION 1 M synth at bs 64 and p ≥ 0.1 (1.11-1.29).
  - Real filters at p < 0.01 that SilverTorch does not reach at 0.90:
    LAION 1 M `c0_domain` / `tags4` and LAION 3 M / 10 M `c0_domain`.
    PubMed `c0_mesh` at p 0.00019 does reach 0.90.
  - The binding variable is the recall the curve can reach, not the pass
    rate alone.
- **N.** 0.95 is not reached within the probe cap on PubMed, YFCC or LAION
  `c0_domain` at 10-30 M. LAION `tags4` reaches it at 10 M and 30 M only
  with 16384-65536 lists.
- **k and batch.** All 17 cells that SilverTorch reaches and still loses
  are at p ≥ 0.1 and bs ≥ 16 (verified):
  - 15 at bs 64 and 2 at bs 16 (laion3m-synth p 1 at 0.95, k 100 and
    k 1000);
  - the worst: laion3m-synth p 1, bs 64, 0.95: 2.74 (k 100) and 2.70
    (k 1000).
- **At 24 probes** (`at24.csv`): SilverTorch is faster in every one of
  363 cells. bs 1: 0.016-0.986. bs 16: 0.017-0.395; the page says
  "0.02-0.39", and 0.395 rounds to 0.40.

### Index memory and build time (Q9)

Source: `memory.csv`, clause unless stated.

| dataset | SilverTorch adapted | LiNR V1 / V2 | ratio | build (adapted) | check |
|---|---|---|---|---|---|
| Goodreads | 209 MiB | 292 MiB | 0.72 | 1.8 s | — |
| arXiv | 776 MiB | 1,095 MiB | 0.71 | 3.7 s | verified |
| YFCC | 6,870 MiB | 8,545 MiB | 0.80 | 28.9 s | — |
| PubMed | 9,015 MiB (8.80 GiB) | 16,174 MiB (15.80 GiB) | 0.56 | 85 s (4096 lists) | **MISMATCH**: the page says "9.0 vs 16.2 GiB"; those are thousands of MiB. The ratio 0.56 is right |

- **Bloom indexes are larger than clause indexes.** Adapted bloom: arXiv
  1,141, Goodreads 306, PubMed 10,236 MiB. These exceed LiNR's on arXiv
  and Goodreads. "About 70 %" holds for clause filters only.
- LiNR V3 adds 2-6 % over V1 / V2. LiNR builds in ≤ 1.1 s.
- **Build times** (adapted, by list count):
  - 2-4 s at ≤ 3 M with the default lists. LAION 3 M takes 20-40 s at 8192
    / 16384.
  - 29-85 s at 10 M with 4096 lists. Verified.
  - With 65536 lists: PubMed 1,426 s (23.8 min), LAION 30 M 1,734-1,742 s
    (29 min), but **LAION 10 M 578 s (9.6 min)**. **MISMATCH**: the page
    says "24-29 min with 65536 lists at 10-30 M"; the source gives 10-29
    min.

### Goodreads end-task bridge (recall caveat)

`product_recall.json` / `.csv`. 17 configurations per filter: the
SilverTorch curve and tuned points, LiNR V1 / V2, and V3 at pools
0.5-20 %.

- **Pearson:** at k 100, 0.875 (`c0_genre`), 0.881 (`c1_lang_reverse`),
  0.905 (`all4`). At k 1000, 0.906-0.935. Pooled: 0.896 / 0.913. Verified.
- **Spearman:** 0.70-0.78 at k 100 and 0.90-0.96 at k 1000. Verified.
- **Where our recall ≥ 0.90:** end-task recall is 0.928-1.005 of the exact
  scan's, median 0.998. Validation says "median 0.99"; that is a rounding
  down, since the median is 1.00.
- **Where our recall < 0.70:** 0.347-0.966 of the exact scan's (median
  0.753; 27 rows). **MISMATCH**: validation says "0.35-0.83". 0.83 is the
  k 100 median, not the maximum. Four SilverTorch configurations at our
  recall 0.61-0.70 keep 0.93-0.97 of end-task recall.

## §6 T3 numbers (Meta as published vs adapted)

Source: `adaptation.csv`, `summary.json`, and
[validation](../../validation.md#meta-fork-backendofficial-fork).

| row | value | check |
|---|---|---|
| X gate, ≤ 10 M | 1,536 / 1,536 cells `torch.equal` (ids and scores), 18 groups by dataset × d × filter kind | verified (`summary.json` `gate_x_equal`) |
| X gate, 30 M LAION | bloom partial + full 64 / 64 at `32c0d61`; 168 / 168 at `82fafb6`; none / exact last at `8ff11e0` (72 / 72) | validation row; not in `adaptation.csv` |
| eager time, adapted / Meta | per-group medians 0.120 (YFCC clause) to 0.817 (PubMed none); min 0.009; max 1.030 (arXiv d256 none) | verified ("0.12-0.82", "1.03") |
| A/A floor | 1.001-1.035 | validation |
| 30 M speed | median 0.279, min 0.059, max 0.467 (64 records) | validation |
| eager vs graph ids, NUMBERS adapted cells | 2,151 / 2,151 equal | verified (`summary.json`) |
| S / capture / compile gates | 0 syncs, 1 graph launch a replay, 35 compile tests pass | validation |

Per-dataset ranges for T3 (median of adapted / Meta by filter kind):

| dataset | none | clause | bloom |
|---|---|---|---|
| arXiv d128 | 0.50 | 0.17 | 0.35 |
| arXiv d256 | 0.67 | 0.17 | 0.39 |
| Goodreads | 0.26 | 0.15 | 0.22 |
| PubMed | 0.82 | 0.22 | 0.32 |
| YFCC | 0.63 | 0.12 | — |
| arxiv-synth / goodreads-synth | — | 0.28 / 0.19 | 0.24 / 0.19 |

## §4 / §7: failures, negatives, cost

**Timeouts, gate misses, partial and failed cells the paper's numbers
touch** (sources: [validation](../../validation.md#numbers-adapted-silvertorch-over-the-study-grid),
the leg parquets):

| case | what | rerun? | paper must say |
|---|---|---|---|
| LiNR V2 bs 256 on PubMed `c3_journal_reverse` / `all5`, YFCC `tags_and` | TIMEOUT after 3.6-9.8 h with 80 GB full (`c3_journal_reverse` finished at 2.55 s a batch after 11 h); YFCC V1 bs 256 did not run (same process) | no | listed as TIMEOUT, not measured |
| yfcc10m-synth LiNR k 1000 | V1 recall 0.983-0.995 < 0.99 exact gate (fp16 storage, HB-6); p01 / p1 k 1000 did not run | no; included and flagged (user) | flagged; SilverTorch on YFCC synth at k 100 only |
| A2-yfcc | 72 of 168 rows `partial` (k 100 entries only; 1 failed = p01 V1 gate miss) | no | YFCC synth at k 100 |
| L1-linr-bs64 | 12 of 148 rows `partial` (bs 256 gaps above); 6 failed records (PubMed OOM) superseded by `ok` | yes | — |
| A2-laion30m-real / L1b | 3 / 4 failed, superseded by `ok` reruns | yes | — |
| unstable | A2-small 69 → 0 after rerun; A2b 15 → 7 (5 SilverTorch at ~0.2 ms on LAION 1 M, 2 LiNR V2); A2-pubmed 2, A3 1 not rerun; 6 tuned points rest on a flagged cell | partly | flagged cells named |
| pilot pass rate (V-PILOT, goodreads-synth) | 0.000990 at target 0.001: 1.01 % low (789 of 797,084), inside binomial noise at N·p 797 | no | achieved pass rates are reported, not targets |
| `bs16 < 16 × bs1` (EXHIBITS check on the final pass) | 25 fails: 24 YFCC `tags_and` V2 / V3 (≈ 18×, per-query spread), 1 `torch.compile` V2 at k 1000 (autotune pick) | no | one line in §3 or §8 |
| probe cap | curves stop at `n_lists` / 4; laion1m-synth only at 512 lists (≤ 128 probes); arXiv / corr-synth SilverTorch capped at 512 probes in the final pass | — | "not reached within the probe cap" |
| 0.99 | not reached by any timed SilverTorch point | — | stated |

**Negative results.**
- **Router** (IVF ↔ exact): dropped 2026-10-11. On PubMed 10 M it is
  slower than exact V2 at bs 1 and collapses onto IVF at p ≈ 1; its bs 16
  passes are within 2 % of the front.
  ([decisions](../../decisions.md#campaign-v2-user-2026-10-08), Hub
  `campaign-v2.5/pubmed-router`).
- **Our int8 LiNR V4:** removed from the library (LN-2). Not in the study.
- **No `l_q` router entry** was found in `validation.md` or `decisions.md`
  under that name. The router above is the only dropped method on record.

**GPU-hours (Q14).**
- Final pass (every LiNR number, C1-C3, C4 / C5 as published): **73.5
  GPU-h sized** ([campaign-final](../../artifacts/campaign-final/README.md#sizing)).
  This is a sizing, not a measured wall total.
- NUMBERS legs: **not derivable from the records.** `elapsed_s` is repeated
  on every perf row of a cell, so a naive sum gives 830 h of cells, which
  is impossible. Index builds total 344 h, also over-counted.
- The legs' wall times are in the drivers' logs (Hub `numbers/<leg>/`).
  C2 sums the total. Per exhibit: T2 / F3 = A2 + A2b + A3; C6 / F2 = A1 +
  A1b; C1-C3 = final pass + L1-L4.

## §7: deviations that move a verdict

| id | effect on a verdict |
|---|---|
| LN-1 / L-12 (fused V2) | moves C1's bs 1 crossover from below 11 % (LiNR) to 0.79-0.97: "reproduced in shape", not in location |
| LN-5, HB-2 (recall@k against the exact oracle; LiNR used end-task recall@2000) | makes C2 "not comparable as stated"; any recall-based verdict (C2, C6) is on our metric; the Goodreads bridge links the two |
| LN-8 (V3 bits capped at d; LiNR 512) | C2's recall at a 1 % pool is lower at d ≤ 256; near parity only at 768 bits (PubMed) |
| LN-6 (eager vs compiled torch as the "native" baseline) | C3's size depends on the baseline: eager 1.4-6.8×, compiled 0.5-4.8× |
| OF-9 (host-bound partial path in Meta's code) | C5 as published reverses below 30 M |
| OF-10 (Meta's bloom ignores `m_bits`), OF-1 (no transposed layout) | C4's width curve and S-14 are not testable on Meta's code; the adapted code's width is not measured (gap) |
| OF-12 (Meta's bloom admits non-matching items at 30 M) | does not move C4's FPR at ≤ 10 M; the adapted bloom has 0 false positives (C13) |
| HB-6 (fp16 item storage vs fp32 oracle) | YFCC LiNR k 1000 flagged; does not move a verdict |
| HB-4 (seed 0) | every ratio is unreplicated except the C1 seed cells (≤ 0.011 spread) and META-FORK `seeds-c7`; no CI on T2 |

## Gaps that need a decision

1. **C4 / F4a has no study-arm source.** The width curve in
   `c4_bloom.csv` is Triton, and no NUMBERS `bloomwidth` leg ran. Options:
   - run it on `official-fork`;
   - report C4 as the 1024-bit point on Meta's code as published plus
     "0 false positives with the recheck";
   - cut F4a to text (the outline's cut order already allows it).
2. **C6 int8 / fp16 ceiling split is Triton-only.** Either the user accepts
   it as a quantisation counterfactual (its "shipped" column matches the
   adapted curves within 0.0003, YFCC 0.011-0.018), or it is rerun on the
   adapted index, or results.md drops the fp16 and per-row figures.
3. **The C6 local-pass-rate Spearman** (`20261009-1057-gls`) is Triton.
   Same choice.
4. **The LAION V3 pool curve (L2b) is absent from `c2_v3.csv`.** Fix in
   `build_results.c2`: take V2 from the A2 legs.
5. **No paired CIs** on T2 or C5. The outline promises them; a CPU script
   over `interleave_group` would do it, with no rerun needed.
6. **results.md's C5 adapted half** quotes `c5_adapted.csv`, measured at
   `82fafb6` before K17. Drop it or label it, to match outline
   `d80b262`.
7. **Wording fixes in results.md / validation.md**: the six MISMATCH rows
   above (C1 "2-5×", PubMed GiB, 65536-list build time, fp16 "≤ 0.006",
   the bridge's "0.35-0.83", YFCC per-row / fp16 setting).
