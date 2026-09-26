---
chain: "silvertorch-official-integration"
branch: "main"
parent: "2026-09-06-100000000-wp0-wp1-record-pin-build-and-op-facts.md"
nextStep: "WP-5 (roadmap B4): delete the CUDA C++ and CuTe backends; the parity gate that rule 5 requires is green."
created: "2026-09-06T14:00:00Z"
---

# §14 record: WP-2 GPU gate + WP-3 parity gate (roadmap B1, B2, B5), 2026-09-06

## WP-2 authoring (B1, Mac, `dev/b1-official-adapter`)
Shipped `kernels/silvertorch/official.py` (availability probe splitting missing from broken, `OfficialConfig`, CSR + feature + expression mapping, `pack_mask` / `unpack_partial_mask` / `reverse_bits64`, raw and dequantised scorers, bloom partial / full), `Backend` + `"official"`, `SilverTorch(backend="official", official=OfficialConfig(...))`, compile / traced-forward refusal, `require_official()`, T1-T7 parametrised over both bit orders until A3 pinned it. Mac gate: ruff clean, 683 tests collected, official surface skips on `OfficialMissing`.

Deviations from the plan found against upstream source: (1) T4's "partial mask ⊇ exact for AND and NOT" cannot hold for NOT: a bloom NOT is the complement of a bloom term, so it has no false positives and may have false negatives; the test asserts ⊆ for NOT and records the FN rate. (2) the build `k` is a knob, `OfficialConfig.build_k`, default the search `k` (the upstream module builder passes `hash_k`). (3) `m_bits` optional on official (width is `b_multiplier`); `k_hash <= 10`. (4) the layer applies the `-1` id sentinel at every non-finite slot (`masked_topk`); the Triton `_impl`s did not, so tests normalise both sides.

## Environment
A100-SXM4-80GB (sm_80), driver 570.195.03, Python 3.11.10, torch 2.10.0+cu128, triton 3.6.0; `silvertorch._C` rebuilt with nvcc 12.8 (V12.8.93, 169 s, no warnings) at `21aa35e`. Branch `dev/integration`, commits `2b09f8e` (three pre-existing red cells), `0521a67` (stable argsort), `aadc380` (test fixes + per-forward sync test). SM clock unlocked (samples 210-1140 MHz); nothing here is a timing. Artifacts: `docs/artifacts/official-silvertorch/wp3/`.

## 14.1 Upstream suite on the 12.8 build
99 passed, 3 subtests passed in 2.58 s (same 99 as 12.4).

## 14.2 Library suite
| run | state | result |
|---|---|---|
| 1, `-x` | tip `563a0f3` | 337 passed, 42 skipped, stopped at `test_n_lists_equals_n[cute]` |
| 2 | + `2b09f8e`, `0521a67` | 561 passed, 11 failed (all `test_official.py`), 127 skipped |
| 3 | + `aadc380` | 574 passed, 0 failed, 127 skipped, 52 s |
All skips are `cute` cells (extra not installed). Pre-existing red cells fixed: two `test_topk_util.py` tests compared fp32 scores to Python literals (`0.8999999761581421 != 0.9`, could never pass), fixed as `torch.equal` against the fp32 inputs, no tolerance introduced; `test_n_lists_equals_n[cute]` gated.

## 14.3 The parity gate, 43 tests, run 3 all green
First run 32/43; the 11 red were test-side, none an adapter or kernel mismatch; no tolerance loosened.
- T1 int32 path `torch.equal` vs `ref_cps_phase23` and vs Triton: 4 layouts `(16,64,4,D64)`, `(64,96,8,D128)`, `(32,64,8,D96)` ref-only, `(32,90,8,D128)` remainder path x B in {1, 16}; exact via `clause_mask` -> `pack_mask`, 4 cells; raw op contract. Bit-exact on every regime. Fix: exact cells asserted `mask.shape == (b, N)` but `make_probe_family` pads ~10 % of slots, so the CSR doc space is `sort_perm.numel()` (5507 of 6144).
- T2 fp16: `max_rel_err` 4.76e-4 / 4.85e-4 / 4.88e-4 / 4.88e-4 (bound 2^-10 = 9.77e-4), `jaccard@32` 1.0 / 1.0 / 1.0 / 0.9924 (D=128, B=16).
- T3 HIGH-first confirmed. Fix: the negative control claimed a low-first mask scores nothing; it scores the mirrored doc `63 - d % 64` (0 -> 63, 5 -> 58, 40 -> 23; 69 -> 122 past the cluster -> nothing).
- T4 AND: no FN on full or partial masks (partial equals full on probed docs), FPR 0.0000 at `b_multiplier=10` on a 0.09 % pass-rate predicate; NOT: no FP, FN rate 0.0000.
- T5 `_with_partial_masks` ≡ full mask ≡ unfiltered ∘ mask, `torch.equal`.
- T6 layer: int32 `torch.equal` on none / exact / exact-reverse (+ all-inactive); fp16 `jaccard@64` 1.0000 on none and exact; bloom 0 FP among 1024 returned slots, partial ≡ full ≡ uncached bit for bit; state-dict order and round trip; candidates through `inv_perm` bit-equal to Triton; `k_hash > 10` rejected. Fix: the Triton epilogue left the padded item id at `-inf` slots while official writes `-1`; fp16 and bloom cells now normalise.
- T7: `compile()` and `fullgraph` raise; syncs per op parse 0, `fused_kmean_ann` 3, partial response 2, `_with_partial_masks` 4, full search 0. Fix: `_count_syncs` now counts fd-2 `warn_or_error_on_sync` lines plus Python warnings (disjoint instruments). New `test_t7_layer_forward_sync_count`.

## 14.4 T4 FPR at matched memory (both blooms k=5, official hash_k=7)
Matched memory is exact: at `b_multiplier = m_bits / (max_terms · 5)` the official index has our byte count.

T4 corpus (N=4096, C=2, A_max=2, vocab 50, pad 0.3; 32 two-term AND queries, exact pass rate 0.0009):
| arm | width | bytes/doc | index B | FPR | FN |
|---|---|---|---|---|---|
| official | b_mult 1.5 (30 bits) | 3.8 | 15,360 | 0.0010 | 0 |
| official | b_mult 2.0 (40) | 5.0 | 20,480 | 0.0002 | 0 |
| official | b_mult 3.0 .. 10.0 (60 .. 200) | 7.5 .. 25 | 30,720 .. 102,400 | 0.0000 | 0 |
| official | b_mult 12.8 / 25.6 / 51.2 (matched to m_bits 256 / 512 / 1024) | 32 / 64 / 128 | 131,072 / 262,144 / 524,288 | 0.0000 | 0 |
| ours | m_bits 256 / 512 / 1024 | 32 / 64 / 128 | 131,072 / 262,144 / 524,288 | 0.0000 | 0 |

Dense corpus (A_max=4, vocab 8, pad 0.1, up to 8 terms/doc; 32 one-term queries, exact pass rate 0.3787):
| arm | width | bytes/doc | index B | FPR | FN |
|---|---|---|---|---|---|
| official | b_mult 1.5 (60) | 7.5 | 30,720 | 0.0043 | 0 |
| official | b_mult 2.0 (80) | 10.0 | 40,960 | 0.0067 | 0 |
| official | b_mult 3.0 .. 5.0 | 15 .. 25 | 61,440 .. 102,400 | 0.0000 | 0 |
| official | b_mult 6.4 / 12.8 / 25.6 (matched) | 32 / 64 / 128 | 131,072 / 262,144 / 524,288 | 0.0000 | 0 |
| official | b_mult 40.0, 51.2 | 200, 256 | 819,200, 1,048,576 | 0.0000 | 0 |
| ours | 256 / 512 / 1024 | 32 / 64 / 128 | same | 0.0000 | 0 |
Reading: both blooms FPR 0 at matched memory on synthetic attributes; the informative S8 comparison (paper 6.98 % -> 0.067 % from 512 to 1024 bits) needs real attributes: WP-4 / D3.

## 14.5 T7 per layer forward (N=4096, D=128, B=16, K=64, n_lists 64, n_probe 8)
| forward | launches | distinct | D2H | H2D | syncs |
|---|---|---|---|---|---|
| triton none / exact / bloom | 17 / 18 / 39 | 15 / 15 / 31 | 0 | 0 | 0 |
| torch none / exact / bloom | 32 / 41 / 59 | 27 / 34 / 46 | 0 | 0 | 0 |
| official none (fp16 / int32) | 55 / 54 | 42 / 41 | 3 | 0 | 3 (3 + 0) |
| official exact | 63 / 62 | 48 / 47 | 3 | 0 | 3 (3 + 0) |
| official bloom partial | 70 / 69 | 44 / 43 | 7 | 2 | 7 (6 + 1) |
| official bloom full | 56 / 55 | 43 / 42 | 4 | 2 | 4 (3 + 1) |
`cache_plans` on and off give identical rows (the cache moves only CPU parse time). The +1 Python sync on bloom is the `.tolist()` in `queries_to_expressions`.

## 14.6 `argsort(stable=True)` in `_build_ivf` (`0521a67`)
Nine regimes (N = 256 .. 131,072, `n_lists` 8 .. 1024, both sides of torch's 4096-element small-sort threshold): permutation, padded layout and forwards bit-identical on torch / triton / official. Applied so slot order is a property of the assignment.

## 14.7 Findings carried
- Triton's epilogue left the padded item id at `-inf` slots; `torch` / `official` return `-1` through `masked_topk`. Landed later on `dev/c4-library-fixes` as `8df7e9a` (one `torch.where(isfinite)` in both `_finish` helpers, gated by `TestFewSurvivorsSentinel`).
- B4 unblocked: T1 `torch.equal` on every regime, bloom ⊇ / ⊆ hold, FPR at matched memory recorded.
- B5 (TF-2 salt buffer) GPU gate passed in run 3 (`test_bloom_hash.py` buffer-vs-inline bit equality on CUDA and CPU; bloom rows of `test_silvertorch.py`). Raw-capture latency claim unmeasured.
- No timing taken; none citable.
