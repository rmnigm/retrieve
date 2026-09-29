# pubmed-fixes — D-aware probe tiles, chunked 1-bit build, the bloom divergence

The three defects the `campaign-d1` investigations found at pubmed scale (chain notes
`official-vs-triton-crossover-report`, `finding-official-bloom-tail-score-divergence`). A100-SXM4-80GB,
torch 2.10.0+cu128, triton 3.6.0; SM clocks not lockable, sampled per window. Current state is in
[validation](../../validation.md); mechanism in [kernels](../../system/kernels.md). Raw outputs
(sweep / tuner / timing JSONs, census logs) are for the Hub under `artifacts/pubmed-fixes/`
([hub-index](../hub-index.md)).

## Fix 1: D-aware probe-scorer tiles

**Spill, confirmed on fresh cubins.** Every `codesigned_probe_score*` cubin the sweep compiled,
read with `cuobjdump -res-usage` / `-sass`: at `D_PAD = 1024` the old `256 × 4` tile is REG 32,
STACK 9,944-10,352 B, 2,317-2,984 `LDL`; `128 × 4` is REG 255, STACK 136-176, about 35 `LDL`;
`64 × 4` has no spill. At `D_PAD ≤ 256` no tile spills.

**Sweep** ([`tile_sweep.py`](tile_sweep.py)): kernel-only `_impl` time of the three variants over
`(BLOCK_P, num_warps)`, L4's layout (1024 skewed clusters of at most 5,000, about 2.3 M items,
`n_probe` 24, k 100), kernel-opt's `time_fn`. Every config's scores are `torch.equal` to `256 × 4`'s
at the same shape (all 240 cases). µs at B = 16:

| variant | D | 256×4 | 256×8 | 128×4 | 64×4 | 32×4 |
|---|---|---|---|---|---|---|
| none | 128 | 292 | 298 | 298 | 306 | 368 |
| none | 192 | 319 | 344 | 339 | 333 | 394 |
| none | 384 | 465 | 494 | 419 | 432 | 481 |
| none | 768 | **6555** | 793 | 682 | **629** | 694 |
| bloom | 128 | 301 | 300 | 303 | 329 | 415 |
| bloom | 192 | 303 | 336 | 311 | 332 | 428 |
| bloom | 384 | 399 | 488 | 390 | 388 | 526 |
| bloom | 768 | **7644** | 619 | 521 | **527** | 634 |
| exact | 128 | 320 | 314 | 318 | 368 | 453 |
| exact | 192 | 320 | 346 | 314 | 406 | 460 |
| exact | 384 | 401 | 464 | 383 | 676 | 575 |
| exact | 768 | **7070** | 576 | **512** | 1172 | 1059 |

At B = 1 every config sits at the ~290-320 µs host floor, except `256 × 4` at D = 768 (737-835 µs).

**Pick.** `tune-kernels`, now per width (`codesigned-probe-score --d D --b 16`,
`codesigned-probe-score-exact --regime N,B,C,A_MAX,D` at goodreads / arXiv clause shapes), its own
rule against the shipped entry: at D = 768 it **keeps** `64 × 4` (none / bloom; `128 × 4` 1.055,
`256 × 4` 6.49 geomean) and `128 × 4` (exact; `256 × 8` 1.045, `256 × 4` 4.37); at D = 384 it moves
`256 × 4` → `128 × 4` (geomean 0.95 worst 1.03; exact 0.92 / 0.94). At D ≤ 256 the sweep's rule
keeps `256 × 4` for all three variants, so D = 128 / 192 are unchanged.

**Parity.** `test_codesigned_probe_score{,_exact}.py` through the shipped tile at D = 64, 128,
192, 768 (bit-exact vs the reference and the loop oracle) and the tile-cutoff cases at the D = 64
and D = 768 tiles.

**Same machine code at D = 128 / 192.** [`sass_identity.py`](sass_identity.py), base tree
(staging `a4edf26`) against this branch, fresh caches: **7 of 7 cubins identical**
(the three scorer variants at both widths, plus the shared id epilogue). The host adds one
`tile_for_width` lookup per call.

**Timing gate.** [`bench_width.py`](bench_width.py) (public ops, L4's layout, B ∈ {1, 16}) via
[`interleave.sh`](interleave.sh), base / new alternating process by process, summarised with
kernel-opt's [`compare.py`](../kernel-opt/compare.py):

5 rounds per side at D = 128 and 192, 3 at D = 768; N ≈ 2.3 M, k 100.

| D | case | base µs | new µs | new/base | noise band |
|---|---|---|---|---|---|
| 128 | none b1 / b16 | 400.4 / 372.9 | 374.0 / 393.1 | 0.934 / 1.054 | 0.163 / 0.096 |
| 128 | bloom b1 / b16 | 372.9 / 379.6 | 389.4 / 398.1 | 1.044 / 1.049 | 0.109 / 0.174 |
| 128 | exact b1 / b16 | 386.7 / 382.7 | 396.9 / 390.6 | 1.026 / 1.021 | 0.140 / 0.166 |
| 192 | none b1 / b16 | 389.1 / 373.5 | 376.8 / 380.7 | 0.968 / 1.019 | 0.120 / 0.096 |
| 192 | bloom b1 / b16 | 370.7 / 375.2 | 385.2 / 380.0 | 1.039 / 1.013 | 0.123 / 0.105 |
| 192 | exact b1 / b16 | 382.9 / 384.5 | 387.4 / 403.8 | 1.012 / 1.050 | 0.138 / 0.148 |
| 768 | none b1 / b16 | 730.7 / 6624.8 | 391.4 / 652.0 | 0.536 / 0.098 | 0.129 / 0.014 |
| 768 | bloom b1 / b16 | 809.6 / 7677.5 | 382.5 / 589.5 | 0.472 / 0.077 | 0.121 / 0.006 |
| 768 | exact b1 / b16 | 762.7 / 7070.5 | 407.3 / 572.3 | 0.534 / 0.081 | 0.124 / 0.007 |

D = 128 / 192: every ratio inside its noise band; every D ≤ 192 window is `unstable` on both sides
(spread > 5 %, the sub-0.5 ms host-bound behaviour L4 recorded), and the kernels are
SASS-identical. D = 768: 10-13× at B = 16, about 2× at B = 1. SM 1275-1410 MHz, per row in the
JSONs. A first attempt ran beside a CPU-heavy census and was discarded (spreads up to 3.6).

## Fix 2: chunked 1-bit build

[`oporp_build.py`](oporp_build.py) on the tables the harness hands LiNR V3 (fp32), the chunked
build against the one-shot chain (`project_*_query` over the whole table, the pre-fix body):

| table | method | chunked s / peak GiB | one-shot s / peak GiB | bits |
|---|---|---|---|---|
| arXiv 3.0M × 128 | OPORP | 0.10 / +0.26 | 0.02 / +10.02 | `torch.equal` |
| arXiv 3.0M × 128 | SimHash (k_bits = D) | 0.10 / +0.21 | 0.02 / +7.17 | `torch.equal` |
| yfcc10m 10M × 192 | OPORP | 0.20 / +0.55 | 0.09 / +50.29 | `torch.equal` |
| yfcc10m 10M × 192 | SimHash | 0.19 / +0.47 | 0.91 / +35.99 | `torch.equal` |
| pubmed 10M × 768 | OPORP | 0.46 / +2.21 | OOM ("Tried to allocate 28.61 GiB") | — |
| pubmed 10M × 768 | SimHash | 0.93 / +1.85 | OOM ("Tried to allocate 57.22 GiB") | — |

Output 0.04 / 0.22 / 0.89 GiB. Goodreads (encoded from a checkpoint, not a stored table) was not
run; it is smaller than arXiv. Each chunked build ran first, cold, in its process.

**LiNR V3 at pubmed**, `bench run --dataset pubmed --suite filter --dim 768 --algo linr_v3
--backend triton --filter-kind clause --sweep c0_mesh --seed 0 --skip-perf`: builds and
serves, `recall_oracle@100` 0.99853, `@1000` 0.99954, held-out `recall@100` 0.99893 (V1 on the
same cell in D1: 0.99853 / 0.99967 / 0.99893), n = 8,428, `index_mib` 17,090, 50.8 GB reserved.

## Fix 3: official vs triton bloom divergence

**Method.** `bench run` (not `campaign`) of `silvertorch` triton, then official, one bloom sweep,
seed 0, `--skip-perf`, both `n_probe`; the triton `_parity/` spill moved aside before the official
run so both sides' top-1000 are on disk. [`bloom_divergence.py`](bloom_divergence.py): rank-wise
diff, finite counts, finite scores on `id == -1`, `-inf` on real ids.
[`bloom_row.py`](bloom_row.py): every returned id of both sides checked against the exact clause
(the harness's `sweep_qa` and item attrs).

| cell | `score_max_abs_diff` (D1 / here) | false-positive ids in the top-1000s: official / triton (rows) | rows over 0.01, holding a false positive |
|---|---|---|---|
| arXiv bloom `c0_maincat`, pre-fix attrs, np 24 | 0.0951 / 0.0951 | 315 / 0 (1 / 0) | 1 of 1 |
| arXiv bloom `c0_maincat`, today's attrs, np 24 and 32 | — / 0.00048 | 0 / 0 | 0 |
| pubmed bloom `c0_mesh`, both np | 0.1087 / 0.1087 | 8,812-11,321 / 244-303 (2,331-2,695 / 75-95) | 1,570-1,836 of 1,570-1,836 |
| pubmed bloom `c2_year`, both np | 0.0107 / 0.0107 | 430-442 / 3-5 (326-337 / 3-5) | 1 of 1 |

No side has a finite score on an `id == -1` slot or `-inf` on a real id in any cell. The arXiv
query is row 9786, `c0 = 29`: triton's 49 candidates all pass, official's 315 extra all fail
(their `c0` is 16 / 18 / 24). The pre-fix attrs are the backups on the box,
`item_attrs_narrow.pt.pre-license-fix.bak` and `eval_split.parquet.pre-license-fix.bak`, through a
scratch config whose `data_dir` symlinks them (D1's arXiv bloom cells other than `all4` ran on
them, 2026-09-26).
