# ST-IDS — the probe scorers' id epilogue without a dense `[k, n_probe]` tile

Roadmap step ST-IDS (campaign-v2.3 bundle). `common.probe_ids_kernel` (the id epilogue after `torch.topk`, launched by
both `codesigned_probe_score*`) built one dense `[next_pow2(k), next_pow2(n_probe)]` tile per row. At k 1000 every new
`n_probe` cost minutes of ptxas (pod c's diagnosis, `.chains/v-ax-corr/2026-10-09-024500000-diagnosis.md`), the variants
spilled, and k 1000 × `n_probe` > 1024 did not compile. Current state: [validation](../../../validation.md) row *ST-IDS*;
mechanism: [kernels](../../../system/kernels.md#silvertorch-kernels) ("Ids after the top-k").
A100-SXM4-80GB (pod b, GPU 0), torch 2.10.0+cu128, triton 3.6.0. Before = the campaign-v2.2 library as package
`retrieve_v22` ([`make_pkg.sh`](make_pkg.sh)). Raw outputs: Hub `artifacts/st-ids/` ([hub-index](../../hub-index.md)).

## The change

One program per `(row, BLOCK_K slots)` (grid `(B, cdiv(k, BLOCK_K))`, `BLOCK_K = min(next_pow2(k), 16)`). Each program walks
the row's probes in chunks of `BLOCK_N = min(next_pow2(n_probe), 512)` and carries the running cluster end across chunks.
A slot's position is the same `lo − start + slot` sum as before (one hit per finite slot), so the result is integer-exact.
`KP` / `NPP` are gone, and every k ≥ 16 × `n_probe` ≥ 512 is one compiled variant.

**A first cut failed the keep rule**, as the plan predicted it might not: one program per row looping over k chunks
(64) and probe chunks (128). It was 1.55× slower than v2.2 at k 100 × `n_probe` 1024. Mechanism: with one program per
row (1-16 programs) the row's work is a serial chain (2 k-chunks, each re-walking 8 dependent probe chunks), against
v2.2's single 1024-wide pass. [`ids_sweep.py`](ids_sweep.py) (36 configs × 10 cells, kernel-only, all bit-exact) put
the k chunks on a second grid axis instead: `BLOCK_K` 16 / `BLOCK_N` 512 / 4 warps is faster than v2.2 in every cell
(worst upper bound 0.495). Shipped.

## Gates (final tree, `phase.sh` into one output directory)

**Bit-exact, green.** [`ids_gate.py`](ids_gate.py) `exact`, **324 / 324**:
- 312 cells `torch.equal` (ids and scores) against v2.2's `_impl` on ST-DLOOP's synthetic grid. That is D 128 / 192 / 768
  × none / bloom / exact × p {0, 0.001, 0.01, 0.1, 1} (+ an all-inactive query) × bs {1, 16} × k {100, 1000} ×
  `n_probe` {24, 1024}, on N 2 M Gaussian items with n_lists 1024.
- 12 cells at k 1000 × `n_probe` {2048, 4096}, which v2.2 cannot compile (n_lists 4096, p 0.01, D 128). The epilogue's ids
  are `torch.equal` to a torch oracle of the slot → id map on the same top-k slots (`searchsorted` over the probes'
  cumulative ends), at bs 1 and 16. The whole op agrees with the torch reference op at bs 1 (scores `torch.equal`, ids up
  to ties). The reference materializes `[B, width, D]`, hence bs 1 and D 128.

**Library suite** on pod b: 787 passed (parity files 55 / 55, the d64 / d768 graph-capture tests included), rerun on a
fresh inductor cache. The first run reused a cache the first cut had compiled into, so its compile tests may have replayed
the old epilogue (testing.md § Running).

**Compile time, cold** ([`compile_time.py`](compile_time.py), [`compile_sweep.sh`](compile_sweep.sh)). CPU only, an
explicit sm_80 target, a fresh `TRITON_CACHE_DIR` and one core (96-191, away from the timed jobs) per variant;
registers and stack from `cuobjdump -res-usage`:

| k | n_probe | v2.2 compile s | v2.2 regs / stack B | ST-IDS compile s | ST-IDS regs / stack B |
|---|---|---|---|---|---|
| 100 | 24 | 1.8 | 80 / 0 | 0.76 | 40 / 0 |
| 100 | 64 | 3.5 | 174 / 0 | 0.82 | 72 / 0 |
| 100 | 128 | 9.5 | 255 / 160 | 0.95 | 56 / 0 |
| 100 | 256 | 10.7 | 32 / 4,528 | 1.05 | 89 / 0 |
| 100 | 512 | 12.4 | 255 / 304 | 1.15 | 91 / 0 |
| 100 | 1024 | 16.0 | 255 / 1,584 | 1.15 | 91 / 0 |
| 100 | 2048 | 28.6 | 255 / 616 | 1.15 | 91 / 0 |
| 100 | 4096 | 84.2 | 255 / 2,464 | 1.16 | 91 / 0 |
| 1000 | 24 | 27.3 | 255 / 1,216 | 0.76 | 40 / 0 |
| 1000 | 64 | 89.5 | 32 / 9,904 | 0.82 | 72 / 0 |
| 1000 | 128 | 599.1 | 32 / 18,680 | 0.94 | 56 / 0 |
| 1000 | 256 | 624.4 | 32 / 18,944 | 1.04 | 89 / 0 |
| 1000 | 512 | 812.8 | 72 / 15,688 | 1.16 | 91 / 0 |
| 1000 | 1024 | 998.0 | 64 / 21,440 | 1.16 | 91 / 0 |
| 1000 | 2048 | fails (numel > 2^20) | – | 1.14 | 91 / 0 |
| 1000 | 4096 | fails (numel > 2^20) | – | 1.14 | 91 / 0 |

These are ahead-of-time compiles without the JIT's argument specializations. Through the JIT, v2.2's k 1000 × 1024
variant compiled to 32 registers and a 122,376-byte stack (ST-DLOOP's observation), and the driver held a ~25 GB
local-memory pool for it. ST-IDS has no stack at any (k, `n_probe`).

**Epilogue time, interleaved** ([`ids_gate.py`](ids_gate.py) `time`). The epilogue alone on the after scorer's real top-k
slots (D 128, `n_probe` cells whose v2.2 variants compile in ≤ 90 s): CUDA graphs of 64 launches over 8 batches, ABAB, 12
pairs, 95 % t-interval:

| k | n_probe | bs | v2.2 µs | ST-IDS µs | after / before [95 % CI] | ids equal | sm_mhz |
|---|---|---|---|---|---|---|---|
| 100 | 24 | 1 | 5.38 | 2.41 | 0.452 [0.446, 0.457] | yes | 1410-1410 |
| 100 | 24 | 16 | 5.55 | 2.79 | 0.502 [0.499, 0.505] | yes | 1410-1410 |
| 100 | 256 | 1 | 178.48 | 5.69 | 0.032 [0.032, 0.033] | yes | 1140-1140 |
| 100 | 256 | 16 | 171.04 | 6.92 | 0.040 [0.039, 0.041] | yes | 1140-1275 (unstable) |
| 100 | 1024 | 1 | 58.08 | 9.29 | 0.160 [0.160, 0.160] | yes | 1275-1275 |
| 100 | 1024 | 16 | 57.95 | 12.62 | 0.218 [0.217, 0.218] | yes | 1140-1275 (unstable) |
| 1000 | 24 | 1 | 47.93 | 3.23 | 0.068 [0.067, 0.068] | yes | 1140-1140 |
| 1000 | 24 | 16 | 49.69 | 6.99 | 0.141 [0.140, 0.142] | yes | 1140-1140 |
| 1000 | 64 | 1 | 337.06 | 4.47 | 0.013 [0.013, 0.014] | yes | 1140-1275 (unstable) |
| 1000 | 64 | 16 | 338.02 | 9.88 | 0.029 [0.028, 0.030] | yes | 1275-1410 (unstable) |
| 1000 | 1024 | 1 | 1987.24 | 9.36 | 0.005 [0.005, 0.005] | yes | 1410-1410 |
| 1000 | 1024 | 16 | 4422.47 | 39.64 | 0.009 [0.009, 0.009] | yes | 1410-1410 |


Every cell is faster, ids equal everywhere. At the campaign's usual cell (k 100, `n_probe` 24) the epilogue goes from about
5.5 to 2.4-2.8 µs. v2.2's k 1000 × `n_probe` 128-512 variants were not timed (10-14 min of ptxas each, GPU held).

**Prediction against result** (`.chains/st-dloop/2026-10-09-050000000-st-ids-plan.md`): compile under 2 s everywhere, as
predicted (0.8-1.2 s); no stack, as predicted. The epilogue at k 100 × `n_probe` 24 was predicted within ±10 % and
measured at 0.45-0.50×. At k 1000 it was predicted ≤ before and measured at 0.005-0.14×. The first cut's k 100 × 1024
slowdown was not predicted; it is fixed above.

## What goes stale on adoption

The epilogue runs in every SilverTorch Triton call, so **every SilverTorch Triton perf record** moves. The move is about
−3 µs at k 100 × `n_probe` 24, more at k 1000 (up to −4.4 ms at k 1000 × 1024 bs 16). Quality does not move (bit-exact).
The `n_probe` ≤ 1024 at k 1000 limit is gone.

## Scripts

| script | what |
|---|---|
| [`make_pkg.sh`](make_pkg.sh) | a tagged library as a renamed package (`retrieve_v22`) for before / after in one process |
| [`ids_gate.py`](ids_gate.py) | `exact` (the bit-exact gate) and `time` (the interleaved epilogue timing); uses ST-DLOOP's `dloop_gate.py` helpers |
| [`ids_sweep.py`](ids_sweep.py) | the epilogue tile sweep: the first cut (`loop`, defined there) against the shipped grid form |
| [`compile_time.py`](compile_time.py), [`compile_sweep.sh`](compile_sweep.sh) | cold compile time per variant, CPU only |
| [`phase.sh`](phase.sh) | the GPU phase as run |
