---
chain: "deterministic-compaction"
branch: "main"
nextStep: "None; L3 merged (4837a6c). Open follow-ups: retune bloom_compact's block_n for the two-phase shape (Phase G); fold the scan into the scatter launch if an eager B=1 number ever becomes paper material."
created: "2026-09-15T11:00:00Z"
---

# L3: deterministic stream compaction, plan and validation record

Source: `docs/plans/deterministic-compaction.md`. Planned 2026-09-15 on `development` at `dc58734` by the orchestrator after the golden re-derive found two cells that cannot reproduce themselves. User decision the same day: fix the kernel before the C4 gate rather than widen the gate. Executed on `dev/l3-deterministic-compaction` off `0930097`; merged `4837a6c`. Artifacts: `docs/artifacts/deterministic-compaction/`.

## 1. Evidence
| recall@k, bs=1 | golden | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|---|
| linr_v3 k=100 | 0.877002740643 | 0.877019983864 | 0.876952025615 | 0.876981440444 | 6.80e-5 |
| linr_v3 k=500 | 0.724373871613 | 0.724341413849 | 0.724360888599 | 0.724337356573 | 2.35e-5 |
| linr_v2 k=100 | 0.999273760709 | 0.999275789310 | 0.999273760709 | - | 2.03e-6 |
Controls: `linr_v1` / `linr_v4` (dense post-mask) and every torch backend bit-identical; SilverTorch (fused predicate) deterministic.

## 2. Cause
`clause_compact` / `bloom_compact` claimed each tile's slice of the output row with an `atomic_add` on the row counter, so surviving ids landed in tile-completion order. `linr_v2` (exact) moves only equal scores (2e-6, the tie floor); `linr_v3`'s 1-bit stage ties massively on integer Hamming distance, so candidate order changes which items enter the pool (7e-5, 34x). The last non-determinism of its kind (the atomic k-means was the other).

## 3. Decisions
- D1 the invariant is ascending item order (what `ops.reference` emits), so triton and torch agree by construction.
- D2 two-phase deterministic offsets, no atomics on the row base: per-tile counts `[B, T]`, an exclusive `torch.cumsum` scan (deterministic, microseconds), writes at fixed offsets.
- D3 (superseded by measurement): recompute the predicate in phase 3 rather than stash. Both premises false: `clause_compact` is 34 % of a `linr_v2` graph forward at B=1 and 47 % at B=16, and instruction-bound on the `C · A_max` int64 loads, so re-evaluation cost x1.9-3.1 on the kernel and +43-58 % end to end; the bitmask fallback x1.9 at B=1 (+28 %). Shipped instead: the tile stash (predicate launch keeps the one-pass `tl.cumsum` + masked-store epilogue, writing into the tile's own fixed slot range of an int32 scratch; a scan; a predicate-free scatter). Cost x1.02-1.05 on V2/V3 forwards under CUDA graph, x1.13-1.14 eager at B=1, scratch `4 · B · T · BLOCK_N` bytes (51 MB at goodreads B=16; ~2.3 GB at 36 M items B=16). Lesson: "already in cache" and "a small share of the forward" are hypotheses with units; a profiler settles them in minutes.
- D4 no change to op schemas, names, or the opaque `custom_op` registration (C4 fix i); tune keys stay valid.
- D5 a correctness fix, not Phase G kernel work.

## 5. Gate (A100)
1 order parity bit-exact vs `ops.reference` (ids and counts, in order); 2 launch-to-launch identity (10x in one process and once in a fresh process); 3 full suite green, tolerances untouched; 4 cost measured (report > ~2x on the kernel; come back if end to end regresses beyond a few percent); 5 the two golden cells reproduce bit-identically to each other (not to the old golden, which was one sample of the noise band).

## 7.1 Record, 2026-09-15
Driver 580.159.04, nvcc 12.4, Python 3.11, torch 2.10.0+cu128, triton 3.6.0, ruff 0.15.6 (uvx), `/venvs/l3`, worktree `/workspace/wt/l3`, inductor cache `/tmp/inductor-l3`, `flock`; clock sampled (1410 under load, first row of each run from 1155 idle); `ncu` blocked, attribution by `torch.profiler`.

Commits: `b9bd82a` D2/D3 as written (second launch re-evaluates; new `tests/parity/test_compact_order.py`; the two compaction parity files tightened from row-set to row-order equality); `2b74979` the tile stash (`common.compact_stash`, `common.compact_scatter_kernel`, `_host.compact_finish`); docs state the ordering guarantee.

Defects found and pinned: (i) `counts` as `tile_ends[:, -1].contiguous()` was a view at element offset `T - 1` when B=1, tripping inductor's 16-byte `assert_alignment` on custom-op outputs (compiled `LiNRV2(BloomFilter)` on goodreads shapes, only where `T - 1` is odd); now `.clone()`, pinned by `test_compiled_batch_of_one_bloom_v2` (N=512, `block_n` 256, B=1). (ii) Triton lays the tile out for the first reduction it meets: storing the tile count (`tl.sum`) before the scan made the predicate launch 1.7-1.8x slower at B=1 (count-only epilogue 0.217 ms vs the old 0.121); with `compact_store`'s scan first the launch keeps one-pass speed (`epilogue_variants.json`).

CPU gates: ruff clean (76 files); 11 pre-existing E501s in untouched evaluation files; links 0.
GPU gates: (1) `test_compact_order.py` 31 passed (clause on `make_exact` x reverse {none, mixed} x `(C, A_max)` in {(1,1), (2,2), (4,4)}; bloom on `make_bloom` x `m_bits` {256, 512, 1024}; N in {64, 4096, 100,003}; three non-default tile configs `(128, 2)`, `(256, 8)`, `(1024, 4)`); (2) `test_launch_to_launch_identity` 10 launches + a fresh interpreter at N=200,003, B=8 all `torch.equal` (fails on the pre-L3 kernel); (3) full suite 645 passed, 0 failed, 0 skipped in 74 s (613 + 32).

Gate 4 cost (do_bench medians, 500 reps; real goodreads `item_attrs_narrow` with `c0_genre`-shaped queries, pass rates 0.32-0.56):
| measurement | before ms | D3 recompute | bitmask (not shipped) | shipped stash |
|---|---|---|---|---|
| clause_compact synth (797,085, B=1, C=4, A=4) | 0.1229 | 0.3346 (x2.72) | 0.2374 (x1.93) | 0.1363 (x1.11) |
| clause_compact synth (797,085, B=16) | 1.4884 | 2.8695 (x1.93) | 1.5487 (x1.04) | 1.5167 (x1.02) |
| clause_compact synth (2,988,997, B=1, C=5) | 0.3441 | 1.0784 (x3.13) | 0.7994 (x2.32) | 0.3554 (x1.03) |
| clause_compact synth (2,988,997, B=16) | 3.0760 | 5.9098 (x1.92) | 3.4466 (x1.12) | 3.1111 (x1.01) |
| bloom_compact synth (797,085, B=1, W=16) | 0.0905 | 0.1598 (x1.77) | 0.1071 (x1.18) | 0.1055 (x1.17) |
| bloom_compact synth (797,085, B=16) | 0.4925 | 0.8666 (x1.76) | 0.6357 (x1.29) | 0.5439 (x1.10) |
| bloom_compact synth (2,988,997, B=1) | 0.2528 | 0.4714 (x1.86) | 0.2800 (x1.11) | 0.2712 (x1.07) |
| bloom_compact synth (2,988,997, B=16) | 1.7935 | 3.1265 (x1.74) | 2.2848 (x1.27) | 1.9343 (x1.08) |
| clause_compact goodreads c0_genre B=1 | 0.1226 | 0.3281 (x2.68) | 0.2328 (x1.90) | 0.1361 (x1.11) |
| clause_compact goodreads c0_genre B=16 | 1.4952 | 2.8745 (x1.92) | 1.5523 (x1.04) | 1.5364 (x1.03) |
| bloom_compact goodreads c0_genre B=1 | 0.0929 | 0.1631 (x1.76) | 0.1084 (x1.17) | 0.1087 (x1.17) |
| bloom_compact goodreads c0_genre B=16 | 0.5084 | 0.8702 (x1.71) | 0.6410 (x1.26) | 0.5884 (x1.16) |
| linr_v2 clause graph B=1 | 0.3639 | 0.5621 (x1.54) | 0.4698 (x1.29) | 0.3725 (x1.02) |
| linr_v2 clause graph B=16 | 3.1768 | 4.5550 (x1.43) | 3.2428 (x1.02) | 3.2269 (x1.02) |
| linr_v2 clause eager B=1 / 16 | 0.4022 / 3.2294 | x1.50 / x1.43 | x1.26 / x1.02 | 0.4596 (x1.14) / 3.2862 (x1.02) |
| linr_v2 bloom graph B=1 / 16 | 0.3464 / 2.2014 | x1.20 / x1.16 | x1.03 / x1.06 | 0.3588 (x1.04) / 2.2847 (x1.04) |
| linr_v2 bloom eager B=1 / 16 | 0.8577 / 2.3126 | x1.15 / x1.16 | x1.15 / x1.06 | 0.9730 (x1.13) / 2.4079 (x1.04) |
| linr_v3 clause graph B=1 | 0.3990 | 0.6089 (x1.53) | 0.5089 (x1.28) | 0.4174 (x1.05) |
| linr_v3 clause graph B=16 | 2.3832 | 3.7559 (x1.58) | 2.4389 (x1.02) | 2.4192 (x1.02) |
| linr_v3 clause eager B=1 / 16 | 1.1941 / 2.5539 | x1.10 / x1.54 | x1.08 / x1.02 | 1.2733 (x1.07) / 2.6030 (x1.02) |
Notes: every x1.1x row at B=1 is ~13 µs of scan + scatter + launches on a 90-140 µs kernel; `bloom_compact` at B=16 is x1.10-1.16 because its default `block_n = 256` runs twice as many stash / scatter programs as the clause kernel's 512 (a Phase G retune may recover it); no defaults retuned; the scratch is 960 MB at 15 M items B=16. Skipped: tune JSONs not regenerated; the bitmask exists only in `epilogue_variants.py`.

## 7.2 Gate 5: the two cells reproduce
Frozen golden worktree `/workspace/wt/golden` with this branch's `retrieve/` committed there as `6c70e74` (tree `75383be`); runbook `golden_two_cells_twice.sh`; shared inductor cache `/tmp/inductor-l3-golden`; same data, queries, oracle (9,859 of 10,000 kept).
| cell (goodreads d128 c0_genre triton) | run 1 vs run 2 | vs the 2026-09-15 re-derived golden |
|---|---|---|
| linr_v2 | byte-identical, all 9 rows | k=100 identical; k=500 moved 2.0e-7; k=1000 6.1e-7 (ndcg 5.0e-7) |
| linr_v3 | byte-identical, all 9 rows | k=100 moved 5.9e-5 (0.877002740643 -> 0.876943911216); k=500 5.2e-5; k=1000 4.8e-6 |
Moves inside the old noise band; within a run the three batch sizes now give identical quality. The two committed golden cells replaced by run 1 (`compare_cells.py`). Procedural slip disclosed: the runner started twice concurrently (a masked exit code); every cell ran two or three times, quality byte-identical across all writes. Latency context: linr_v2 bs=1 k=100 0.357 ms (golden 0.349), bs=16 3.23 (3.19); linr_v3 bs=1 0.403 (0.471), bs=16 2.41 (2.38); sampled SM median 1155 MHz.
