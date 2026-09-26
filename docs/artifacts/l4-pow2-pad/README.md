# L4 — non-power-of-two D / W pad to the next power of two

Roadmap L4. Mechanism and correctness argument: [kernels § Padding](../../system/kernels.md#padding).
Raw timing JSONs are on the Hub under `artifacts/l4-pow2-pad/` ([hub-index](../hub-index.md)).

## Same code at a power-of-two width

[`ptx_identity.py`](ptx_identity.py) compiles every padded kernel (the three probe scorers,
`fused_masked_knn_topk` at D ∈ {64, 128, 256}; OPORP full / indirect and both bloom ops at
W ∈ {1, 2, 4, 16}) into a fresh `TRITON_CACHE_DIR`, once on `staging` (`bcee67c`) and once on
L4; the SASS of each cubin (`cuobjdump -sass`, addresses stripped) is compared. **30 of 30
identical.** PTX differs only in `.loc` line info and, for the kernels where Triton folds the
all-true mask late, one `mov.pred %p, -1` that ptxas removes. A first version with plain
masks in the probe scorers left 3 of 6 of their variants' SASS different (register
allocation); the constexpr-conditional masks fixed it.

## Timing, D = 128 (A100-SXM4-80GB, clocks not lockable)

[`bench_d128.py`](bench_d128.py) (kernel-opt's timing method, imported from
[`bench_kernels.py`](../kernel-opt/bench_kernels.py)), interleaved staging / L4 process by
process, 5 rounds each ([`interleave.sh`](interleave.sh), summarised with kernel-opt's
[`compare.py`](../kernel-opt/compare.py)). N ≈ 2.5 M (1024 skewed clusters), B = 16, k = 100.

| case | base µs | new µs | new/base | noise band | base sm MHz | new sm MHz | unstable (base/new) |
|---|---|---|---|---|---|---|---|
| cps_none_b16 | 390.6 | 365.4 | 0.936 | 0.181 | 1410-1410 | 1410-1410 | 5/5 , 5/5 |
| cps_bloom_b16 | 402.1 | 370.2 | 0.921 | 0.178 | 1410-1410 | 1275-1410 | 5/5 , 4/5 |
| cps_exact_b16 | 391.1 | 385.1 | 0.985 | 0.200 | 1275-1410 | 1275-1410 | 5/5 , 5/5 |
| fmkt_b16_k100 | 3870.3 | 3913.1 | 1.011 | 0.070 | 1275-1410 | 1275-1410 | 5/5 , 4/5 |
| oporp_full_b16_k5000 | 1496.7 | 1509.9 | 1.009 | 0.033 | 1410-1410 | 1290-1410 | 0/5 , 1/5 |
| oporp_indirect_b16_k5000 | 6977.0 | 6973.5 | 0.999 | 0.012 | 1410-1410 | 1320-1410 | 0/5 , 1/5 |
| bloom_match_b16 | 1142.5 | 1145.6 | 1.003 | 0.013 | 1410-1410 | 1410-1410 | 0/5 , 0/5 |
| bloom_compact_b16 | 1420.7 | 1418.3 | 0.998 | 0.013 | 1410-1410 | 1410-1410 | 1/5 , 0/5 |
| bloom_compact_b1 | 211.0 | 204.8 | 0.971 | 0.212 | 1275-1410 | 1410-1410 | 5/5 , 5/5 |

Every ratio is inside its noise band. The probe scorers and `bloom_compact_b1` are
`unstable` in every round on both sides (window spread > 5 %, the known sub-0.5 ms
behaviour); with identical SASS the kernels cannot differ, and the host-side addition is
one `triton.next_power_of_2` per call.

## Non-power-of-two layers end to end

[`layers_e2e.py`](layers_e2e.py): SilverTorch (none / bloom / exact), LiNR V2 and V3 (exact
and bloom filters), `backend="triton"` against `backend="torch"` at D = 192 and 768,
N = 20,000, B = 16, k = 100, eager and `torch.compile`: SilverTorch `torch.equal` scores
(ids up to ties); V2 / V3 within `fused_masked_knn_topk`'s parity tolerance (atol 1e-6).
All 28 cases pass. Not run on the real yfcc10m / pubmed tables (that is D1).
