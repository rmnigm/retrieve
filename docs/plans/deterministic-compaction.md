# Deterministic stream compaction — making LiNR V2/V3 reproduce themselves

> **Status:** **executed 2026-09-15** on `dev/l3-deterministic-compaction`
> off `development` @ `0930097` — validation record in §7 (all five gates
> ran on the A100; the shipped kernel is the tile-stash shape of §7.1, not
> D3's re-evaluation, by measurement). Awaiting the orchestrator's review,
> the roadmap flip and the merge. Planned 2026-09-15 on `development` at
> `dc58734`; written by the orchestrator after the golden re-derive
> (eval-queue item 1) found that two cells cannot reproduce *themselves*
> across processes, which makes roadmap C4's `1e-6` gate unmeetable on them by
> any harness. User decision the same day: **fix the kernel before the C4
> gate runs**, rather than widening the gate or documenting the noise.
>
> Question this plan answers: *why do `linr_v2` and `linr_v3` on `triton` move
> between identical runs, and what is the smallest change that makes every
> library path reproducible bit-for-bit?*
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 Phase L, step
> **L3**. It blocks the C4 gate rerun (queue item 2) and therefore D1.

## 1. The evidence

Three runs of the same cell, same commit, same data, same `seed: 0`, same
cached queries and oracle, one process each
([evaluation-harness-v2.md](evaluation-harness-v2.md) §11.8):

| `recall@k`, bs=1 | golden | run 1 | run 2 | run 3 | spread |
|---|---|---|---|---|---|
| `linr_v3` k=100 | 0.877002740643 | 0.877019983864 | 0.876952025615 | 0.876981440444 | **6.80e-5** |
| `linr_v3` k=500 | 0.724373871613 | 0.724341413849 | 0.724360888599 | 0.724337356573 | 2.35e-5 |
| `linr_v2` k=100 | 0.999273760709 | 0.999275789310 | 0.999273760709 | — | **2.03e-6** |

And, from the re-derive itself: `linr_v1`, `linr_v4` (dense post-mask, no
compaction) and every `torch` backend are bit-identical across library
versions; `silvertorch` (fused predicate, no compaction) is deterministic
across three runs on two different library trees.

## 2. The cause, in our own words

[`ops/triton/clause_compact.py`](../../retrieve/src/retrieve/ops/triton/clause_compact.py)
says it in its module docstring:

> Output ordering within a row is unspecified (atomics across tiles); callers
> that need order must sort.

Each tile claims its slice of `out_indices[b]` with an `atomic_add` on the
per-row counter, so the surviving ids land in **tile-completion order**, which
the scheduler decides afresh every launch. `bloom_compact` has the same shape.

The order then reaches a tie-breaker:

- **`linr_v2`** = `ExactAttributeFilter` → `PrefilterKNN` over the compacted
  ids. The algorithm is exact, so only *equal scores* can move, and they move
  by position: 2e-6, the tie floor.
- **`linr_v3`** = compaction → `OneBitKNN(k=candidate_pool)` → `PrefilterKNN`.
  The 1-bit stage ranks by integer Hamming distance, which ties *massively*;
  a different candidate order changes **which items enter the pool**, not just
  their order inside it, and the second stage inherits that. Hence 7e-5 — 34×
  `linr_v2`'s, and a real quality difference rather than a presentation one.
- **`linr_v1` / `linr_v4`** never compact (dense mask), and the **reference**
  compaction emits ascending ids. Both are reproducible. That is the control.

The non-determinism is therefore **one property of two kernels**, not a
distributed problem, and it is the last one of its kind we know of: the
atomic k-means was the other, and C4 fix (ii) removed it.

## 3. Decisions

- **D1 — The invariant is ascending item order, not merely "some fixed
  order".** `ops.reference.clause_compact` already emits ascending ids
  (`nonzero` over a dense mask). Matching it makes the Triton and torch
  backends agree *by construction* instead of agreeing up to a tie-break, and
  turns the existing parity tests into real checks of the property. A merely
  stable-but-arbitrary order would leave `triton` ≠ `torch` on V2/V3 forever.
- **D2 — Two-phase, deterministic offsets; no atomics on the row base.**
  Phase 1: each tile counts its survivors into `tile_counts [B, T]`
  (`T = ceil(N / BLOCK_N)`). Phase 2: exclusive scan over `T` (a `torch.cumsum`
  on a `[B, T]` int32 tensor — T is in the thousands, this is microseconds and
  it is deterministic). Phase 3: each tile re-evaluates its predicate and
  writes at `tile_offset[b, t] + intra-tile cumsum`, the intra-tile cumsum
  being the one already in `compact_store`. `counts[b]` is the scan's total.
- **D3 — ~~Recompute the predicate rather than stash the mask.~~
  **Superseded by measurement, 2026-09-15 (orchestrator, on WP-1's numbers).**
  As written, D3 said: phase 3 re-runs the `clause_pass` / bloom test instead
  of reading a `[B, N]` bool that phase 1 wrote, because "the predicate is a
  handful of integer ops against values already in L2" and "cost is ≈ 2× the
  compaction kernel, which is a small share of a V2/V3 forward".

  **Both halves of that premise are false**, and WP-1 measured it rather than
  assuming it, as the decision's own last clause demanded. `clause_compact` is
  **34 % of a `linr_v2` graph forward at `B = 1` and 47 % at `B = 16`** — not a
  small share — and it is *instruction*-bound on the `C · A_max` int64 loads
  per item, not DRAM-bound, so a second pass costs a second full kernel rather
  than a cache replay: ×1.9–3.1 on the kernel and **+43–58 % end to end**, past
  both thresholds in §5 gate 4. The §6 bitmask fallback fared little better
  (×1.9 at `B = 1`, +28 % end to end) because of the Triton tile-layout defect
  in §7.1 (ii).

  **What ships instead: the tile stash.** The predicate launch keeps the
  pre-L3 epilogue (`tl.cumsum` ranks + masked store) and writes each tile's
  survivors into *its own fixed slot range* of an int32 scratch, so no atomic
  decides a base; the scan then gives each tile its row offset and a
  predicate-free scatter moves the runs into place. The invariant of D1 is
  unchanged and so is D4. Cost: **×1.02–1.05 on the V2/V3 forwards under CUDA
  graph** (what the harness measures), ×1.13–1.14 eager at `B = 1` from the two
  extra launches, and a `4 · B · T · BLOCK_N`-byte scratch — 51 MB at goodreads
  `B = 16`, alongside the `[B, N]` int64 result buffer that already exists, so
  it is peak memory rather than traffic. At E-phase scale (36 M items, `B = 16`)
  that scratch is ≈ 2.3 GB next to the 4.6 GB result, which fits this box but
  belongs in the scale-ladder notes.

  The general lesson, for the next kernel decision in this repository: "it is
  already in cache" and "it is a small share of the forward" are hypotheses
  with units, and a profiler settles them in minutes.
- **D4 — No change to the op schemas, names, or the `@custom_op` opacity.**
  `clause_compact` and `bloom_compact` stay opaque custom ops (C4 fix (i),
  which is what lets compiled V2/V3 capture a CUDA graph); only their bodies
  change. `KernelTuneSpec` keys stay valid, so the tune JSONs survive.
  `retrieve::` schemas are untouched, so `code_version` changes but nothing
  downstream reinterprets a recorded field.
- **D5 — This is a correctness fix, not the Phase G kernel work.** It is not
  TF-1 (the transposed bloom) and it is not a retune. It buys no speed and may
  cost some; it buys the ability to state a number twice. Phase G stays where
  it is, after the paper.

## 4. What it touches

| file | change |
|---|---|
| [`ops/triton/clause_compact.py`](../../retrieve/src/retrieve/ops/triton/clause_compact.py) | two-phase kernel; the docstring's "unspecified ordering" sentence becomes the opposite guarantee |
| [`ops/triton/bloom_compact.py`](../../retrieve/src/retrieve/ops/triton/bloom_compact.py) | same shape, bloom predicate |
| [`ops/triton/common.py`](../../retrieve/src/retrieve/ops/triton/common.py) | `compact_store` gains the deterministic-base variant, or is split into `count_tile` / `store_tile` |
| [`ops/reference/*`](../../retrieve/src/retrieve/ops/reference/) | unchanged — it is the definition of the target order |
| `docs/system/kernels.md`, `filtering.md` | the ordering guarantee is now part of the contract, stated where the kernels are described |

Nothing in `modules/` changes: the composites already treat the compacted
buffer as an id list.

## 5. Work package and gate

**WP-1 — deterministic compaction (GPU; one library-suite run).** One worker,
`fable` (kernel work, [agent-orchestration.md](agent-orchestration.md) §3),
branch `dev/l3-deterministic-compaction` off `development`.

Gate, all of it on the A100:

1. **Order parity, bit-exact.** `ops.triton.clause_compact` and
   `ops.triton.bloom_compact` outputs `torch.equal` to
   `ops.reference.*` (ids **and** counts, in order) on the parity suite's
   existing input families, every filter width and both `clause_is_reverse`
   shapes. This is a new parity file; it is the property, stated as a test.
2. **Launch-to-launch identity.** The same call repeated 10× in one process,
   and once in a fresh process, returns `torch.equal` ids — the check that
   would have caught this.
3. **Full library suite green** (613 on `development` at the merge; any count
   change explained), every existing parity file still bit-exact, tolerances
   untouched (CLAUDE.md rule 3).
4. **The cost is measured, not assumed.** `tune-kernels`-style timing of both
   compaction kernels before and after, and one end-to-end `linr_v2` and
   `linr_v3` forward, reported as a table in the record. A regression beyond
   ~2× *on the compaction kernel* is a finding to report; an end-to-end
   regression beyond a few percent is a reason to come back to the
   orchestrator, not to abandon the invariant.
5. **The cells reproduce.** Re-run the two golden cells
   (`goodreads-d128 c0_genre linr_v2 triton` and `linr_v3 triton`) twice in
   the frozen golden worktree (`/workspace/wt/golden`, `/venvs/golden`) with
   the fixed library swapped in: the two runs must be **bit-identical to each
   other**. They will *not* match the existing golden JSONs — the golden was
   taken at an arbitrary point in the old noise band — so those two cells are
   re-derived as part of this step and the golden README says why.

## 6. Risks

- **The re-derived V2/V3 golden cells move by up to the old noise band**
  (≈7e-5). That is expected and is not a regression: the new number is
  reproducible and the old one was not. C4 then gates all 14 cells at 1e-6.
- **Phase 3 re-reads item attributes.** On goodreads `item_attrs_narrow` is
  ~100 MB and does not fit L2, so phase 3 is a second pass over memory, not a
  cache hit. If the measured cost is bad, the fallback is a `[B, T]` *count*
  buffer plus a per-tile bitmask in shared memory — more code, same invariant.
- **`bloom_compact`'s predicate is a hash test**, cheaper to recompute than
  the clause test; the two kernels may end up with different phase-3 shapes.
  That is acceptable; they are separate files.
- **It changes V2/V3 latency before D1.** Deliberate: doing it after D1 would
  invalidate the campaign through the tree-hash resume key (H §8.2 B).

## 7. Validation record

### 7.1 WP-1 (roadmap L3) — 2026-09-15, A100-SXM4-80GB, `dev/l3-deterministic-compaction`

**Environment.** The A100 box (driver 580.159.04, nvcc 12.4 at
`/usr/local/cuda`), Python 3.11, torch 2.10.0+cu128, triton 3.6.0, ruff
0.15.6 via `uvx`, Meta's `silvertorch` built into `/venvs/l3` with
`uv sync --extra official --all-packages`. Branch point `development` @
`0930097`; worktree `/workspace/wt/l3`; private inductor cache
`/tmp/inductor-l3`. Every CUDA job ran under `flock /workspace/gpu.lock`, one
at a time. Clocks cannot be locked here: the SM clock is sampled before and
after every timing below (`sm_mhz` in the JSONs; **1410 MHz** under load on
every row but the first of each run, which starts from the 1155 MHz idle
state), so no number here is clock-controlled (H §7). `ncu` is blocked; the
per-kernel attribution came from `torch.profiler`. Artifacts under
[deterministic-compaction-artifacts/](deterministic-compaction-artifacts/):
`time_compaction.py` and its four JSON/log pairs, `epilogue_variants.py`
(+ JSON), `repro_bloom_align.py`, `golden_two_cells_twice.sh`,
`compare_cells.py`, `library-suite.log`, `golden-rederive/`.

**What was done.** Two commits on the branch, both keeping the invariant of
D1 (ascending item order, `torch.equal` to `ops.reference`) and D4 (schemas,
names and the opaque `custom_op` registration untouched):

1. `b9bd82a` — **D2/D3 as written**: each op is two launches of its kernel
   around a `torch.cumsum`; the second launch re-evaluates the predicate and
   writes at `tile_offset + intra-tile rank`. `common.compact_store` takes the
   caller's base. The new parity file
   [`tests/parity/test_compact_order.py`](../../retrieve/tests/parity/test_compact_order.py)
   (gates 1–2), the two existing compaction parity files tightened from
   row-*set* to row-*order* equality, the docs, and the first defect (below).
2. `2b74979` — **the second launch no longer re-evaluates the predicate.** The
   predicate launch keeps the pre-L3 epilogue (`tl.cumsum` ranks + masked
   store) with the `atomic_add` row base replaced by the tile's own fixed slot
   range in an int32 scratch (`common.compact_stash`), and a predicate-free
   `common.compact_scatter_kernel` (launched by `_host.compact_finish` after
   the scan) moves each tile's run to its row offset. Chosen by measurement —
   §7.1 gate 4 — over D3 and over the §6 bitmask, both of which were built and
   timed; the reasons are in the table's notes. Docs (`kernels.md`,
   `filtering.md`, `testing.md`, `architecture.md`, the sdist guide,
   `filters.py`'s docstrings) state the ordering guarantee; every "order is
   unspecified" sentence is gone.

**Two defects found on the way, both fixed and pinned.**
(i) `counts` was first returned as `tile_ends[:, -1].contiguous()`; at `B = 1`
that column is already contiguous, so the op handed back a *view* at element
offset `T − 1` of the scan buffer, and inductor's `assert_alignment` on
custom-op outputs (16 bytes) killed the compiled `LiNRV2(BloomFilter)` forward
on goodreads shapes — only at `B = 1`, only where `T − 1` is not a multiple of
two, and not in the compile suite's `N = 512, B = 4`. Now a `.clone()`;
pinned by `test_compiled_batch_of_one_bloom_v2` (`N = 512`, `block_n = 256`,
`B = 1`). (ii) Triton lays the tile out for the *first* reduction it meets: an
epilogue that stores the tile count (`tl.sum`) before the scan made the
predicate launch itself 1.7–1.8× slower at `B = 1` (under one wave, so
per-program latency *is* the kernel time), with no change at `B = 16`; the
bare count-only epilogue measured 0.217 ms against the old kernel's 0.121.
With `compact_store`'s scan first the launch keeps its one-pass speed
(`epilogue_variants.json`; the note in `compact_stash` says why the order
matters).

**Gates — CPU.**

| gate | result |
|---|---|
| `uvx ruff@0.15.6 check retrieve` and `format --check retrieve` | clean (76 files). `ruff check evaluation` has **11 pre-existing E501s** in `evaluation/eval_datasets/*` and `evaluation/training/train_sasrec.py`, identical on `development` @ `0930097`; not touched here |
| `python3 scripts/check_doc_links.py` | 0 broken links |

**Gates — GPU (under the lock).**

| gate | result |
|---|---|
| 1. order parity, bit-exact | `test_compact_order.py`: **31 passed** — `clause_compact` vs `ops.reference.clause_compact` on `make_exact` × `reverse ∈ {none, mixed}` × `(C, A_max) ∈ {(1,1), (2,2), (4,4)}`, `bloom_compact` vs `ops.reference.bloom_compact` on `make_bloom` × `m_bits ∈ {256, 512, 1024}` (`W = 4, 8, 16`), each at `N ∈ {64, 4096, 100 003}` (one tile, many tiles, a ragged last tile), ids **and** counts `torch.equal` with both tails normalised to `-1`; the order holds under three non-default tile configs `(128, 2), (256, 8), (1024, 4)` for both kernels |
| 2. launch-to-launch identity | `test_launch_to_launch_identity`: ten launches in one process and one in a fresh interpreter (`subprocess`, same seeded inputs at `N = 200 003, B = 8`) all `torch.equal` on ids and counts, both kernels. The same check was run by hand on the pre-L3 kernel while diagnosing: it fails there |
| 3. full library suite | **645 passed, 0 failed, 0 skipped** in 74 s on the shipped code (`library-suite.log`): 613 on `development` + 32 new — 31 in `test_compact_order.py` and `test_compiled_batch_of_one_bloom_v2`. No existing test changed its outcome; `test_clause_compact.py` / `test_bloom_compact.py` now assert rows in order (tightened, not loosened); every other parity file untouched, tolerances untouched |
| 4. cost measured | the table below |
| 5. the cells reproduce | §7.2 |

**Gate 4 — cost.** `do_bench` medians (500 reps) from `time_compaction.py`:
the two kernels' `_impl`s on the tuner's synthetic regimes and on the real
goodreads `item_attrs_narrow` with `c0_genre`-shaped queries (clause 0 active,
values drawn by item frequency: pass rates 0.32–0.56); `LiNRV2` / `LiNRV3`
(`k = 100`, `candidate_pool = 5000`) on the real attrs with random unit fp16
embeddings, eager and under the harness's compile settings
(`reduce-overhead`, `dynamic=False`, `fullgraph=True`). Three implementations
against the pre-L3 kernel; the shipped one is bold.

| measurement | before ms | D3 recompute (`b9bd82a`) | §6 bitmask (measured, not shipped) | **shipped: tile stash (`2b74979`)** |
|---|---|---|---|---|
| `clause_compact` synth (797 085, B=1, C=4, A=4) | 0.1229 | 0.3346 (×2.72) | 0.2374 (×1.93) | **0.1363 (×1.11)** |
| `clause_compact` synth (797 085, B=16) | 1.4884 | 2.8695 (×1.93) | 1.5487 (×1.04) | **1.5167 (×1.02)** |
| `clause_compact` synth (2 988 997, B=1, C=5) | 0.3441 | 1.0784 (×3.13) | 0.7994 (×2.32) | **0.3554 (×1.03)** |
| `clause_compact` synth (2 988 997, B=16) | 3.0760 | 5.9098 (×1.92) | 3.4466 (×1.12) | **3.1111 (×1.01)** |
| `bloom_compact` synth (797 085, B=1, W=16) | 0.0905 | 0.1598 (×1.77) | 0.1071 (×1.18) | **0.1055 (×1.17)** |
| `bloom_compact` synth (797 085, B=16) | 0.4925 | 0.8666 (×1.76) | 0.6357 (×1.29) | **0.5439 (×1.10)** |
| `bloom_compact` synth (2 988 997, B=1) | 0.2528 | 0.4714 (×1.86) | 0.2800 (×1.11) | **0.2712 (×1.07)** |
| `bloom_compact` synth (2 988 997, B=16) | 1.7935 | 3.1265 (×1.74) | 2.2848 (×1.27) | **1.9343 (×1.08)** |
| `clause_compact` goodreads `c0_genre` B=1 | 0.1226 | 0.3281 (×2.68) | 0.2328 (×1.90) | **0.1361 (×1.11)** |
| `clause_compact` goodreads `c0_genre` B=16 | 1.4952 | 2.8745 (×1.92) | 1.5523 (×1.04) | **1.5364 (×1.03)** |
| `bloom_compact` goodreads `c0_genre` B=1 | 0.0929 | 0.1631 (×1.76) | 0.1084 (×1.17) | **0.1087 (×1.17)** |
| `bloom_compact` goodreads `c0_genre` B=16 | 0.5084 | 0.8702 (×1.71) | 0.6410 (×1.26) | **0.5884 (×1.16)** |
| `linr_v2` clause, graph, B=1 | 0.3639 | 0.5621 (×1.54) | 0.4698 (×1.29) | **0.3725 (×1.02)** |
| `linr_v2` clause, graph, B=16 | 3.1768 | 4.5550 (×1.43) | 3.2428 (×1.02) | **3.2269 (×1.02)** |
| `linr_v2` clause, eager, B=1 / B=16 | 0.4022 / 3.2294 | ×1.50 / ×1.43 | ×1.26 / ×1.02 | **0.4596 (×1.14) / 3.2862 (×1.02)** |
| `linr_v2` bloom, graph, B=1 / B=16 | 0.3464 / 2.2014 | ×1.20 / ×1.16 | ×1.03 / ×1.06 | **0.3588 (×1.04) / 2.2847 (×1.04)** |
| `linr_v2` bloom, eager, B=1 / B=16 | 0.8577 / 2.3126 | ×1.15 / ×1.16 | ×1.15 / ×1.06 | **0.9730 (×1.13) / 2.4079 (×1.04)** |
| `linr_v3` clause, graph, B=1 | 0.3990 | 0.6089 (×1.53) | 0.5089 (×1.28) | **0.4174 (×1.05)** |
| `linr_v3` clause, graph, B=16 | 2.3832 | 3.7559 (×1.58) | 2.4389 (×1.02) | **2.4192 (×1.02)** |
| `linr_v3` clause, eager, B=1 / B=16 | 1.1941 / 2.5539 | ×1.10 / ×1.54 | ×1.08 / ×1.02 | **1.2733 (×1.07) / 2.6030 (×1.02)** |

Notes. (a) The premise of D3 — "the compaction kernel is a small share of a
V2/V3 forward" — is false on goodreads: `clause_compact` is 34 % of the
`linr_v2` graph forward at `B = 1` and 47 % at `B = 16`, and it scales
linearly with `B` (instruction-bound on the `C · A_max = 16` int64 loads per
item, not DRAM-bound), so re-evaluating the predicate costs a second full pass
— ×1.9–3.1 on the kernel, +43–58 % end to end. Beyond both of gate 4's
thresholds; hence the second commit. (b) The bitmask (§6's fallback) is
cheap in traffic (100 KB per row) but its epilogue triggers defect (ii), so it
is ×1.9 at `B = 1` — +28–29 % on the graph forwards. (c) The shipped stash
costs the survivors' traffic only (4 B in, 8 B out per id) plus the scan,
the `-1` prefill and two extra launches; its scratch allocation is
`4 · B · T · BLOCK_N` bytes (51 MB at goodreads `B = 16`; 960 MB at the
15 M-item synth regime with `B = 16`, half the `[B, N]` int64 result that
already exists there) and shows up as peak memory, not time. Under a CUDA
graph the forwards move by +2–5 %; eager `B = 1` pays the launches
(+13–14 %). Every ×1.1x row on the kernels at `B = 1` is the same ~13 µs of
scan + scatter + launches on a 90–140 µs kernel. (d) `bloom_compact` at
`B = 16` is ×1.10–1.16 rather than ×1.02: its default tile is `block_n = 256`,
so the stash and scatter run twice as many programs as the clause kernel's
512; a retune (Phase G) may recover it. (e) No defaults were retuned (D5).

**Skipped / unverified.** The tune JSONs were not regenerated (the `KernelTuneSpec`
keys are unchanged; `test_tune_smoke.py` passes in the suite). The bitmask
implementation exists only in the artifacts (`epilogue_variants.py`) and in
this record, not on the branch. Eager `B = 1` latency is +13 % from launch
count, which the harness's `graph` mode does not see; if an eager `B = 1`
number is ever paper material, folding the scan into the scatter launch is the
obvious next cut. The roadmap checkbox and the merge into `development` are
the orchestrator's; the two re-derived golden cells (§7.2) are replaced in
`evaluation/golden/` on this branch and wait for that review.

### 7.2 Gate 5 — the two cells reproduce (2026-09-15)

Run in the frozen golden worktree `/workspace/wt/golden` (`tmp/golden-rederive`,
the old harness at `1ccdb27` + the `Backend` literal), `/venvs/golden`, with
this branch's `retrieve/` swapped in and committed there as `6c70e74` (library
tree `75383be`, = `2b74979:retrieve`); the old harness imports the library
through the L shim and ran unchanged. Runbook:
[`golden_two_cells_twice.sh`](deterministic-compaction-artifacts/golden_two_cells_twice.sh)
(the a1-rederive recipe: `uv run --no-sync evaluate` from the worktree's
`evaluation/`, the GPU lock per cell, a private inductor cache
`/tmp/inductor-l3-golden` shared by the four processes, clocks sampled every
30 s). Same data, queries and oracle as the re-derive (cache hits, 9,859 of
10,000 users kept). Outputs, logs, `clocks.csv` and `provenance.txt` under
[`deterministic-compaction-artifacts/golden-rederive/`](deterministic-compaction-artifacts/golden-rederive/).

| cell (goodreads-d128, `c0_genre`, triton) | run 1 vs run 2 | vs the committed golden (2026-09-15 re-derive) |
|---|---|---|
| `linr_v2` | **byte-identical** on every `recall@` / `ndcg@` / `precision@` / `mrr@` / `n_users_kept` column, all 9 rows | k=100 identical; k=500 moved 2.0e-7; k=1000 moved 6.1e-7 (`recall@k`; `ndcg` 5.0e-7) |
| `linr_v3` | **byte-identical**, all 9 rows | k=100 moved 5.9e-5 (0.877002740643 → 0.876943911216); k=500 5.2e-5; k=1000 4.8e-6 |

Both moves are inside the band H §11.8 measured on the old kernel (2.0e-6 /
6.8e-5), as §6 expected: the old numbers were one arbitrary sample of that
band, the new ones are the same number twice. Within a run the three batch
sizes now give identical quality on every k (they did not in the old cells).
The two committed cells are replaced by run 1 (`compare_cells.py` is the
check); the golden README says so. One procedural slip, disclosed: the runner
was started twice by mistake (a masked exit code in the launch chain), so two
instances ran concurrently, each cell serialized by the per-cell GPU lock, and
the second instance re-ran cells the first was still producing — every cell
therefore ran two or three times, with the same result: the quality columns
are byte-identical across all writes, and only the latency columns of the
last writer differ from the earlier ones (`run2/…linr_v3…json` was rewritten
after the first commit of this record; the committed file is the final one).
`clocks.csv` holds both samplers' rows. Latency columns (context only): `linr_v2`
bs=1 k=100 0.357 ms (golden 0.349), bs=16 3.23 ms (3.19); `linr_v3` bs=1
0.403 ms (0.471 — the golden's own row is the slower one), bs=16 2.41 ms
(2.38); sampled SM clock median 1155 MHz over the four cells, 26 °C, not
clock-controlled.

