---
chain: "cuda-silvertorch"
branch: "main"
parent: "2026-07-06-120000000-cuda-c-backend-design-and-runbook.md"
nextStep: "Validate on the A100 via the handoff runbook; work the twelve GPU-only uncertainties of §6 with it in hand."
created: "2026-09-01T12:00:00Z"
---

# Phase 2: exact mode on cuda, the UNROLL knob, two static reviews

Source: `docs/plans/archive/cuda-silvertorch-phase2.md`, planned 2026-09-01 on `refactor/kernels-eval`, authored on the CUDA-less Mac (nothing compiled locally).

## Decisions
- D1: exact on cuda is a second phase-2 mask kernel, not a third scorer flavour. The scorer stays filter-agnostic (any filter is a 1-bit cluster-major mask, paper `M_c`); traffic is a wash (a `[C, A_max]` row is one 32-byte sector at 2x2); keeps the stack machine possible.
- D2: the clause mask reads ids from `flat_items`, no cluster-major attr copy: no new buffer, portable exact state dicts. `max_size` passed as a Python int.
- D3: predicate bit-identical to `common.clause_pass`: `keep = (id >= 0) and AND_c [ (OR_a attr[c,a] == q_c) XOR rev_c or (q_c == -1) ]`; pads and the last word's tail get 0.
- D4: the only v2 knob is `UNROLL`, default 1; `int2` rows and the mask block size stay deferred (bandwidth-bound).
- D5: no paper stack machine (the query model is conjunctive `[B, C]` with reverse flags), no multi-GPU.

## Work packages (all 2026-09-01)
- A: static review of the landed backend, 14 findings triaged to C (#2/#3/#4/#5/#11/#13) and D (#1/#6/#7/#8/#9/#10/#12/#14). Opus.
- A': web-grounded check of toolchain facts (Sonnet): no JIT-path CUDA version check (#6e), topk tie order (#12), do not tag `cudagraph_unsafe` (#11), dp4a floor attribution to `sm_61_intrinsics.h` (#4/#14).
- B: exact on cuda: `cps_clause_mask_kernel`, `_codesigned_probe_score_exact_cuda_impl`, the op + fake, construction `ValueError` removed, tune spec twin, tests.
- C: `UNROLL` template knob (three-phase body, division-free `(cluster, slot)` carry), config field, tune grid x {1,2,4}. Reviewer #4 (shfl-coalesced store) declined: segments in a warp can have different trip counts at a tile tail, so a warp-wide shuffle would not be convergent.
- D: applied triage: `if constexpr` in the generic scorer (dead-ternary OOB pointer arithmetic, a divide at `max_size == 0`), build-outcome memo + nvcc-major check + `ToolchainMissing` split (skip on missing toolchain, fail on build error), `assert_ids_equal_up_to_ties`, `max_size` cached at index build (SymInt risk under `dynamic=True`), padding-slack warning, opcheck at d in {64,128,256,96}, D=128 compile leg, cuda export leg (one opaque node). Runbook gained the gate table, the exact head-to-head, the quantitative memory gate, and a phase-2/phase-3 split at B=1.

## GPU-only uncertainties listed for the A100 session
`__ldg` on `torch.bool` storage; predicated `LDG` vs branch in the UNROLL body; full unroll of the three loops; registers at UNROLL=4, WPL=2 (~55 est.); build time for 18 scorer instantiations; `max_size` under `dynamic=True`; tie frequency; phase-2 launch cost at B=1; real-data padding slack; exact-test pass rate at `n_vocab=8`; REDUX vs butterfly (arch 8.0 vs 7.5); warp convergence of the clause mask (`compute-sanitizer --tool synccheck`).
