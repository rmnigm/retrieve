# CUDA SilverTorch backend — phase 2 plan (exact filter, perf knobs, pre-GPU review)

> **Status:** planned 2026-09-01 on `refactor/kernels-eval` (working tree). Authored on
> the CUDA-less dev Mac — the venv cannot even install the cu128 torch wheel here, so
> nothing in this plan compiles or runs locally. Every item is "implement + static
> review now, validate on the A100 box via the runbook". The runbook is
> [cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md); this plan adds gates to it.

## 1. Where the backend stands

Landed (uncommitted, 2026-07-06 — see the handoff §1 for the file table). This is a
snapshot of the starting point; §3 records which work packages have since landed on
top of it (WP-B, for one, removed the cuda×exact `ValueError` this table lists):

| piece | state |
|---|---|
| `cps_bloom_mask_kernel` (phase 2, transposed index, set-bit iteration) | written, never compiled |
| `cps_score_kernel<SEG, WPL, HAS_MASK>` + generic fallback (phase 3, dp4a) | written, never compiled |
| host launchers + pybind, lazy `cpp_extension.load` | written |
| `build_transposed_sigs`, `_impl`, two custom ops + fakes | written |
| layer `_forward_cuda`, `bloom_sigs_t` registration, cuda×exact `ValueError` | written |
| tune spec `codesigned-probe-score-cuda` | written |
| parity (bit-exact vs Triton), correctness, compile, tune-smoke tests | written, collect on GPU only |
| eval `--backend cuda`, deep-sweep config | written |

Deferred by the handoff §11: `exact` filter port, paper stack machine, v2 kernel
knobs (UNROLL, id prefetch, `int2` rows at D=256, mask-kernel block size), multi-GPU,
`torch.export` gate for the cuda ops.

## 2. Decisions

**D1 — `filter_mode="exact"` on cuda is a second phase-2 mask kernel, not a third
scoring-kernel flavor.** The scorer stays filter-agnostic: *any* filter is a
1-bit-per-item mask laid out cluster-major over probed spans (paper Alg. 1 `M_c`),
and `cps_score_kernel<…, HAS_MASK=true>` is reused untouched. Rationale: (a) the
already-written scoring path's risk is unchanged; (b) traffic is a wash either way
(the `[C, A_max]` int64 attr row is one 32-byte sector at 2×2, same order as the id
read); (c) it is the handoff's own suggested follow-up and keeps the door open to the
paper's stack machine (masks are bit-vectors, AND/OR/NOT are word ops).

**D2 — the clause mask kernel reads ids from `flat_items`, not from a cluster-major
attribute copy.** No new buffer, no `register_index` work; a cuda+exact module's
`state_dict` is identical to triton+exact (unlike bloom, where cuda swaps `bloom_sigs`
for `bloom_sigs_t`). The op takes `max_size` as a Python int so it can rebuild the
cluster-span mask layout the scorer expects.

**D3 — predicate is bit-identical to `common.clause_pass`.** `keep = (id >= 0) ∧
∧_c [ (∨_a attr[c,a] == q_c) ⊕ rev_c ∨ (q_c == -1) ]`; padding slots (`id < 0`) and
the pad tail of the last word get bit 0. Gate: `torch.equal` vs the Triton exact op on
ids and scores (same argument as the handoff §3.2 — exact int32 dot, same fp32
epilogue, boolean-identical predicate, shared host topk).

**D4 — v2 knob = `UNROLL` only, config-gated, default 1 (today's behavior).** Little's
law: A100 needs ≈ 1555 GB/s × ~600 ns ≈ 0.9 MB in flight ≈ 8.6 KB/SM ≈ 67 rows of
128 B; ~50 resident warps × 1 row (D=128) ≈ 6.4 KB is borderline short, so 2–4 items
in flight per segment is the one v2 item with a first-principles case. `int2` rows and
mask-kernel block size stay deferred (bandwidth-bound kernel; instruction count is not
the limiter). Id prefetch is subsumed by UNROLL.

**D5 — no paper stack machine, no multi-GPU.** The repo's query model is conjunctive
`[B, C]` with reverse flags; there is no DSL to compile. Recorded as deferred.

## 3. Work packages

| WP | what | agent | depends on |
|---|---|---|---|
| A | **done (2026-09-01)** — static "compile-in-your-head" review of the landed backend (kernels, launchers, pybind, wrapper, fakes, tests); 14 findings, triaged to C (#2/#3/#4/#5/#11/#13) and D (#1/#6/#7/#8/#9/#10/#12/#14) | Opus | — |
| A′ | **done (2026-09-01)** — web-grounded check of the toolchain/API facts the backend relies on (`cpp_extension.load` arch defaults, `__reduce_add_sync`/`__ballot_sync`, `__ldg` overloads, pybind `at::Tensor&`, `custom_op` tuple returns under `from __future__ import annotations`, `opcheck` on `device_types="cuda"`, header locations); actionable: #6e (no JIT-path CUDA version check), #12 (topk tie order), #11 (do *not* tag `cudagraph_unsafe`), #4/#14 (dp4a floor attribution) | Sonnet | — |
| B | **implemented (2026-09-01), untested** — `exact` on cuda: `cps_clause_mask_kernel` + launcher, `_codesigned_probe_score_exact_cuda_impl`, `retrieve::codesigned_probe_score_exact_cuda` op + fake, layer routing (construction `ValueError` dropped), tune spec twin, tests (parity bit-exact vs Triton exact incl. reverse + inactive clauses, mask-vs-`clause_subset_match`, opcheck; correctness cross-backend; compile `("exact","cuda")`), eval guard note, docs | Opus | — |
| C | **implemented (2026-09-01), untested** — `UNROLL` template knob on `cps_score_kernel` (three-phase body, division-free `(cluster, slot)` tracking), `CodesignedProbeScoreCudaConfig.unroll`, tune grid × {1,2,4} (both cuda specs; `_sweep`/`_print` generalized to variable-arity grid entries), reviewer fixes #2/#3/#5/#13, `torch.equal` config-override parity test + must-reject configs + tiny-`max_size` carry stress, docs. Reviewer #4 (shfl-coalesced store) skipped: segments in a warp can have different trip counts at a tile tail, so a warp-wide `__shfl_sync` there would not be convergent | Opus | B (same files) |
| D | **done (2026-09-01), untested** — reviewed the B+C diffs and applied the A/A′ triage: `if constexpr` in `cps_score_kernel_generic` (dead-ternary OOB pointer arithmetic + a `p / max_size` divide instantiated at `max_size == 0`), explicit build-outcome memo + `nvcc`-major check + `ToolchainMissing` split from a build failure (so `require_cps_cuda` skips on the first and *fails* on the second), `assert_ids_equal_up_to_ties` for the topk-tie gate, `max_size` cached as `self._max_cluster_size` at index build (SymInt risk under `dynamic=True`), `bloom_sigs_t` padding-slack formula + `RuntimeWarning` above 2×, `test_opcheck` over `d ∈ {64,128,256,96}`, a `D=128` leg in the compile test, a cuda leg in the export test, and the doc/comment corrections (#7 block_p claim, #8 dispatch-not-check, #9 trusted ids, A′#11 why not `cudagraph_unsafe`, A′#4 dp4a attribution). Runbook: §5 gate table, §6c exact head-to-head, §7 memory gate, §8 phase-2/phase-3 split at B=1, §9 exact criterion + sign-off template | Opus | A, A′, B, C |

Not applied, with reasons: A#4 (shfl-coalesced store) — declined by WP-C, segments in
a warp can have different trip counts at a tile tail so a warp-wide shuffle there
would not be convergent. A′#11 (`cudagraph_unsafe` tag) — the reviewer's own
recommendation was *do not apply*; recorded as a kernels.md paragraph instead.

## 4. Runbook additions (landed in the handoff by WP-D)

- §5 2a: a per-test gate table for `test_codesigned_probe_score_cuda.py` — what each
  of the eleven tests protects, including the exact twins, the must-reject configs
  (`test_invalid_config_rejected`) and the tiny-`max_size` carry stress
  (`test_unrolled_mask_indexing_tiny_clusters`). Plus: a cuda test that *skips* is now
  itself a finding, because a build failure fails instead.
- §6a: `tune-kernels codesigned-probe-score-cuda` sweeps `unroll ∈ {1,2,4}`; paste the
  winner into `DEFAULT_CONFIG` including `unroll`. New `codesigned-probe-score-exact-cuda`
  spec mirrors the Triton exact spec's regimes.
- §6c (new): the exact head-to-head — same attrs through
  `_codesigned_probe_score_exact_impl` and `_codesigned_probe_score_exact_cuda_impl`,
  bit-exactness asserted before timing, mirroring the bloom snippet including
  `max_size`, with the pass rate printed since it is what the ±10% gate assumes.
- §7: clause sweeps on cuda cells are now legal; `arxiv-d128-silvertorch.yaml` stays
  bloom-only — add cuda to an exact config only after §5 and §6c pass. The memory gate
  is now quantitative: the bloom index delta must match the IVF padding slack, and an
  exact cell must show no index-memory delta at all.
- §8: time phase 2 and phase 3 **separately at `B=1`** before accepting the "mask
  kernels are not worth sweeping" decision — that decision is a traffic argument and
  the small-batch cost is a launch, not traffic.
- §9: exact-mode gate = cuda within ±10% of Triton exact (the predicate is one sector
  per item either way; expect parity, the win is in the code-row skip which both have).
  Sign-off template gains the exact head-to-head row, `unroll` + regs/thread, the IVF
  slack, and the phase-2/phase-3 split.
- §12 F1: the wrapper's own nvcc-major message is now the first thing to read.
- §3.2: "scores bit-identical; ids identical up to permutation within tied scores".

## 5. Deferred (unchanged from handoff §11 unless noted)

- Paper stack machine over per-feature masks (D5). Multi-GPU scale-out.
- `int2` row loads at D=256; swept mask-kernel block size (D4).
- ~~`torch.export` gating for the cuda ops.~~ **Done (WP-D).**
  `tests/compile/test_export_kernel_ref.py` is parametrized over
  `backend ∈ {triton, cuda}`; the cuda leg asserts export preserves
  `retrieve::codesigned_probe_score_exact_cuda` as one opaque node and that the
  exported module replays eager bit for bit.

## 6. Consolidated GPU-only uncertainties

Everything below was reasoned about on a CUDA-less box and **cannot** be settled
there. It is the list to work through with the runbook in hand; §-references are to
[cuda-silvertorch-handoff.md](cuda-silvertorch-handoff.md).

| # | uncertainty | why it is open here | first thing that would settle it |
|---|---|---|---|
| 1 | `__ldg(const unsigned char*)` on `torch.bool` storage | `__ldg` is a fixed overload set in `sm_*_intrinsics.h`, not a template; nothing local can tell us nvcc resolves this one | it compiles at all (§4 step 1). If it does not: read the byte through a `const uint8_t*` cast, or drop `__ldg` for that one load |
| 2 | Predicated `LDG` emission in the `UNROLL` body | `keep[u] ? __ldg(...) : 0` is *written* branch-free, but whether ptxas emits a predicated load or a branch is codegen | `cuobjdump -sass`: look for `@!P LDG` vs a `BRA` around the row gather (§8) |
| 3 | `#pragma unroll` on the three `UNROLL` loops | the trip count is a template parameter, so it *should* fully unroll, but nothing verifies it | SASS: `UNROLL` copies of the dp4a chain, no loop back-edge |
| 4 | Registers/thread at `UNROLL=4, WPL=2` | the ~55 estimate is arithmetic on live values, not a measurement; past ~64 A100 occupancy drops below 50% and a swept "win" turns into a loss | `ptxas -v` / ncu launch statistics before pasting any `unroll > 1` (§6a, §9) |
| 5 | Build time for 18 scoring-kernel instantiations | 3 `D` × 2 `HAS_MASK` × 3 `UNROLL`, up from 6; the "~30–90 s" figure in §4 predates the `UNROLL` axis | time step 1's cold build. If it is painful, drop `UNROLL=2` from the dispatch and sweep `{1, 4}` |
| 6 | `max_size` under `torch.compile(dynamic=True)` | it is now a Python int cached in `_build_ivf`, which *should* keep dynamo from ever seeing a SymInt in the op's `int` slot — but the failure mode (silent specialization, or a graph break) only appears when compiled | `tests/compile/test_silvertorch_compile.py` cuda+exact rows: zero graph breaks (§5 2c) |
| 7 | Top-K tie order | the ids gate is deliberately weakened to "equal up to permutation within tied scores"; whether ties actually appear, and how often, is data-dependent | count them at the parity shapes: `(scores[:, :-1] == scores[:, 1:]).sum()`. If it is 0, the weaker gate is costing nothing and the stronger one could come back |
| 8 | Phase-2 launch overhead at `B=1` | the "mask kernels are not worth sweeping" decision is a *traffic* argument; at `B=1, P≈1024` the second launch is a few µs that phase 3 may not amortize | the per-kernel split in §8 |
| 9 | `bloom_sigs_t` slack on real data | `n_lists · max_cluster_size / N` is 1.0–1.1× only if k-means produced balanced clusters; arxiv/goodreads have never been measured | the one-liner in §3.4, per dataset, before the §7 memory gate |
| 10 | `n_vocab` pass rate in the exact parity tests | `_make_exact` uses `n_vocab=8` to keep top-K from being all `-inf`, but the resulting pass rate is unmeasured — a near-1.0 rate would make the tests blind to a predicate that under-rejects | print `torch.isfinite(scores).float().mean()` in `test_cuda_exact_matches_ref` on the first run; tune `n_vocab` if it is above ~0.9 |
| 11 | `__reduce_add_sync` vs the butterfly | sm_80+ takes the hardware REDUX path, everything below the `__shfl_xor_sync` fallback. Both are exact integer sums, so this is a coverage question, not a correctness risk | build with `TORCH_CUDA_ARCH_LIST=8.0` (REDUX) and once with `7.5` (butterfly); parity must hold either way |
| 12 | Warp convergence in `cps_clause_mask_kernel` | the early return is warp-uniform *by construction* (one word per warp), which is the argument that both `__ballot_sync(0xFFFFFFFF, …)` calls see all 32 lanes; independent-thread-scheduling reality is not observable here | `compute-sanitizer --tool synccheck` on the exact parity test |
