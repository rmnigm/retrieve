# ST-TOPK: a two-level top-k over 256-slot blocks (campaign-v2.11 candidate)

Roadmap AFTER-QUEUE 3, the top-k epilogue. Raw outputs are on the Hub at `artifacts/st-topk`
([hub-index](../../hub-index.md)). Pod b, A100-SXM4-80GB, 2026-10-10. **NOT CITABLE.**

## 1. Profile first (`prof.py`, `prof/p_*`; laion30m is not on pod b)
On PubMed 10 M d768 and arXiv 3 M d128 / d256, our triton arm and Meta's official arm (int32 score path) run the same
`torch.topk` radix select: four 8-bit digit passes plus a gather.
- **Ours costs 2-12 % more** (fp32 keys against Meta's int32).
- **Top-k is 35-75 % of our bloom forwards,** for example 1.7 of 2.7 ms at C7's cell (PubMed `n_probe` 1024, bs 16),
  where 99.97 % of the slots are `-inf`.
- **The roadmap's "1.18 vs 0.79 ms at 30 M"** is `artifacts/c7-laion30m-interleaved` (`n_probe` 128, bs 64). That suite
  sets no `score_path`, so Meta ran its default fp16 keys, which a bit-exact path cannot use.

## 2. The change (`_host.probe_prep` / `_two_level_topk`; the scorer kernels are unchanged)
- From `TOPK_MIN_SLOTS` = 2²³ slots of `[B, width]`, the score buffer is padded to whole 256-slot blocks (the scorer's
  `-inf` tail covers the pad).
- Where a row has at least 8 blocks per top-k slot:
  - block maxima (`amax`);
  - the k best blocks;
  - a gather of their slots;
  - the final top-k.
- **Exactness:** a top-k item's block has a max at least as large, so it is among the k best blocks; otherwise k other
  blocks each hold an item at least as large. So scores are `torch.equal`, and ids are equal up to ties.

**Rejected first cut: tile maxima written by the scorers** (`TILE_MAX`; gate outputs `p_ax256_after`, and the first
`g_*` gate, replaced). The top-k fell as much, but the d256 one-pass scorer slowed 37 % and one narrow cubin's SASS
changed.

## 3. Gate (`gate.py` against staging a05890e, library a3bec4a5 = campaign-v2.10; one process, swapped build order, 8 windows)
310 cells:

| cells | after / before |
|---|---|
| PubMed d768 bloom `c0_mesh` / `c2_year` | 0.500-1.008 (C7 0.555 graph: 2.36 → 1.31 ms) |
| PubMed exact | 0.739-1.006 |
| PubMed none | 0.946-1.009 |
| arXiv d128 bloom / exact / none | 0.776-1.018 |
| arXiv d256 | 0.834-1.013 |
| goodreads 0.8 M (path off) | 0.978-1.013 |
| k 1000 cells | 0.500-1.001 |
| arxiv-synth (tie-heavy) | 0.693-1.000 |

- **Exactness:** scores `torch.equal` in every cell, and ids `torch.equal` too (no tie broke differently). In the first
  gate's tile-max cut, one arxiv-synth cell differed in tie order only.
- **The threshold** was 2²² at first. One host-bound eager cell (arxiv-synth d256 `n_probe` 128 bs 16, 5.2 M slots) read
  1.119. At 2²³ it reads 0.941 eager / 1.000 graph (`h_*`); every real engaged cell sits at ≥ 9.2 M slots.
- **SASS:** all narrow scorer / id cubins are identical to staging (empty diff).
- **Suites:**
  - library: 879 passed;
  - harness: 572 passed, on the staging-merged tree;
  - new: `test_two_level_topk_ties_and_finite_count_boundary` (four distinct code rows, exactly k − 1 / k / k + 1
    passing items, d64 / d768, path asserted on).

## Files
- `prof.py`: per-kernel-family breakdown, ours and official.
- `gate.py`: the keep-rule gate. It reports strict / scores / ids-up-to-ties equality and takes modes `graph`, `swap`,
  `aa`, `slice=D` and `k=K`.
- `sass_narrow.py`: narrow-config SASS hashes.
