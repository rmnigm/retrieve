# Deterministic stream compaction — making LiNR V2/V3 reproduce themselves

> **Status:** planned 2026-09-15 on `development` at `dc58734` (nothing
> implemented). Written by the orchestrator after the golden re-derive
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
- **D3 — Recompute the predicate rather than stash the mask.** Phase 3 runs
  the same `clause_pass` / bloom test again instead of reading a `[B, N]`
  bool that phase 1 wrote. The predicate is a handful of integer ops against
  values already in L2; a dense `[B, N]` scratch is exactly what these kernels
  exist to avoid (that is the whole point of the fused path over the torch
  one). Cost is ≈ 2× the compaction kernel, which is a small share of a V2/V3
  forward — **measure it, do not assume it** (§5 gate 4).
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

*(appended by WP-1 when it runs; model:
[archive/cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360).)*
