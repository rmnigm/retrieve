# ST-WIDE: the probe scorers at wide probes (campaign-v2.9 candidate)

C7 PubMed at v2.8, with Meta built `-O3`, at `n_probe` 1024, bs 16, d768, eager: official 2.605 ms, ours 8.837 ms. Our
`_codesigned_probe_score_kernel` was most of ours. Raw outputs are on the Hub at `artifacts/st-wide`
([hub-index](../../hub-index.md)). Pod b, A100-SXM4-80GB, 2026-10-09. **NOT CITABLE.**

## Mechanism
- **Each tile rebuilt its row's probe layout** (`common.probe_tile`). Every program ran three `NPP`-wide gathers and two
  `tl.cumsum`s, about 262k programs at C7's cell. Fixed by the per-row probe table (`probe_table_kernel`): each tile
  then finds its cluster with a two-level `FAN`-ary count over the tile ends. A binary search, 11 dependent scalar loads,
  cost 1.07-1.09 at bs 1, `n_probe` 256 (`ab_*_bsearch`). Scorer 7.24 → 2.78 ms.
- **The bloom test paid one memory latency per query-bit slot in every tile, inactive `-1` slots included.** Fewer than
  4 % of tiles pass, and the program runs at a 112-register kernel's occupancy. Fixed by the bloom two-pass: a light
  filter pass with a word-level tile vote, then a persistent dot pass over the passing tiles. Scorer 2.78 → about
  0.8 ms (micro runs `i`).

Filter-only ablations, d768, `n_probe` 1024, bs 16 (micro `c`, `f`):

| configuration | µs |
|---|---|
| all 25 slots | 1966 |
| first 5 slots | 746 |
| first slot | 569 |
| no bloom | 365 |

One-pass variants of the bloom loop (micro `d`, `e`, `g`, `h`, against 2768 µs):

| variant | µs |
|---|---|
| fold | 2608 |
| branch per slot | 3263 (slower: waits on each slot's load) |
| `[U, BLOCK_P]` gathers | 3392-43219 (138-255 registers, spills from U 16) |
| int32 words | 2256-2976 (161 registers) |
| groups of 4, int64 | 2328 |
| word-level vote in the one-pass kernel | 2574 |

The vote pays only once it is split from the dot kernel. Two-pass tiles of 64 / 128 items made the filter costlier than
the dot pass saved (micro `j`).

## Gate (`gate.py`, against staging 1d9abd15 as the renamed package `rv_before`, one process)
- **Bit-exact:** ids + scores `torch.equal` on 16 pool batches in every cell. That covers 4 eager runs, 3 graph runs and
  2 swapped-order runs, all × `n_probe` {24, 256, 1024} × bs {1, 16}.
- **Keep rule, after / before medians, 8 interleaved windows:**

  | cell | narrow | wide |
  |---|---|---|
  | PubMed d768 bloom, graph | 0.993-1.009 | 0.232-0.506 (1024 bs 16: 10.24 → 2.38 ms; swapped order 8.82 → 2.39, 0.271) |
  | PubMed d768 bloom, eager | 1.006-1.016 | 0.238-0.548 (swapped order 0.276 at 1024 bs 16) |
  | PubMed d768 exact, table only, graph | 0.998-1.001 | 0.437-0.714 |
  | arXiv d128 bloom, graph | 0.995-1.013 | 0.646-0.872 |
  | arXiv d128 bloom / exact, eager | 1.004-1.013 | 0.65-0.91 |

  - The arXiv A/A floor (staging vs staging, graph) is 1.010 at `n_probe` 24, bs 1.
  - The narrow kernels (`TABLE=False`) have opcode sequences identical to staging for all 9 cubins (`sass_narrow.py`,
    d128 / 192 / 768 × none / bloom / exact). Only constant-bank offsets move, from the added arguments. The eager
    narrow excess is host-side: graph mode reads 0.993-1.009.
  - **Build-order effect:** on PubMed 10 M, an A/A run with identical code timed the first-built index 0.86-0.89 at the
    wide cells. The swapped runs, with the before arm built first, bound it: 0.271-0.276 at 1024 bs 16.
- **d192** is not measured on real data (YFCC is not staged on pod b). The wide d192 parity cases and the narrow d192
  SASS check pass.

## Suites
- Library suite: 851 passed on fresh inductor and Triton caches.
- `test_silvertorch_compile` with the wide modes (4 rows × 128 probes, d768 bloom / bloom-full / exact): compiled
  replay `torch.equal` to eager, 0 cudagraph skips, 0 graph breaks.
- That compile test caught a cudagraph skip in the first two-pass cut. Mechanism and fix are in
  [kernels](../../../system/kernels.md#silvertorch-kernels), "Bloom two-pass".

## Files
- `gate.py`: the gate (`graph`, `aa` and `swap` modes).
- `sass_narrow.py`: narrow-config SASS hashes per tree.
- `prof.py`: per-cell wall time and top kernels.
- `micro/micro.py`: scorer-only timing.
- `micro/twopass.py`, `micro/tp2.py`: two-pass prototypes; `micro/frozen.py`: the reference arm.
- `micro/patches/k_*.patch`: the one-pass variants as diffs. Apply them to `git show
  9dad1f2:retrieve/src/retrieve/ops/triton/codesigned_probe_score.py`, saved as `frozen_cps.py`.

The micro runs used the WIP tree at 9dad1f2, where `_cps_prep` returned the launch alone.
