# R-RES: the three golden residuals

The question: on the frozen campaign-v2 tree, why do three golden cells miss their golden JSONs
([validation](../../../validation.md#harness-gates))? The three residuals are `linr_v2` 4.5e-4,
`linr_v3` 1.7e-5 (goodreads `c0_genre`, gSASRec pinned) and arXiv SilverTorch triton 2.0e-6
(`c0_maincat`, `n_probe` 24). Inputs: the freeze's gate-3 records (code_version `408b1188`,
seed 0, eager, quality only) and `evaluation/golden/`. Box: A100-SXM4-80GB, one GPU, 2026-10-08.

## Where each residual sits

From the frozen records against the golden JSONs, in hits (residual × rows × k):

| cell | @100 | @500 | @1000 | direction |
|---|---|---|---|---|
| `linr_v2` | recall +4.5e-4 (444 hits) | +3.9e-4 | +3.9e-4 | up at every k; new V2 = V1 to the last digit at @100 and @500 |
| `linr_v3` | recall +1.7e-5 (17 hits) | 0 (ndcg +1.1e-7) | 0 (ndcg +5.6e-8) | up |
| arXiv SilverTorch | recall +2.0e-6 (2 hits), ndcg +1.4e-6 | 6e-12 (ndcg +7.5e-9) | 6e-12 (ndcg +6.5e-9) | up |

The golden JSONs carry no per-query data, so the rows that differ cannot be named from the
CPU side. [`boundary.py`](boundary.py) pairs each freeze spill with the cached v4 oracle that
reproduces its per-query sidecar exactly, and counts boundary ties. The exact LiNR arms have no
score tie at any k boundary. SilverTorch's int8 scores tie at the k = 100 boundary on 583 rows.

## LiNR V2 and V3: explained and reproduced

**Mechanism.** Both golden cells were produced at roadmap L3 (library tree `75383be`), before
L5 (`5fa1c99`). At that point `fused_masked_knn_topk`, the exact rescoring step of V2 and of
V3's stage 2, multiplied and reduced in **fp16**: Triton's `tl.sum` reduces in its operand
dtype ([kernels](../../../system/kernels.md#score-conventions); paper D-1 / LN-4). L5 widens
both operands to fp32 first. The frozen tree scores exactly. V2 now equals V1 (cuBLAS fp32) to
the last digit at @100 and @500, and its old fp16 error cost hits at every k. V3 rescores only
5,000 OPORP candidates, and its rank-k item is usually already outside the oracle top-k. So a
reordered near-tie changes V3's membership only at @100 (17 hits); at @500 and @1000 it changes
only the order, which ndcg sees at ~1e-7.

**Confirmation.** [`confirm.sh`](confirm.sh) B runs the frozen tree with the two fp32 widenings
reverted ([`pre_l5_fp16_sum.diff`](pre_l5_fp16_sum.diff), experiment only, undone on exit). It
runs once per k, as the old harness did. Both cells then land **on** their golden JSONs:

| cell | k | recall − golden | precision − golden | ndcg − golden |
|---|---|---|---|---|
| `linr_v2` | 100 / 500 / 1000 | 0.0 / −1.1e-16 / −1.1e-16 | 0.0 / 0.0 / −1.1e-16 | 3.6e-11 / 0.0 / −1.6e-9 |
| `linr_v3` | 100 / 500 / 1000 | 0.0 / 0.0 / 0.0 | 0.0 / 0.0 / 0.0 | 1.6e-9 / 1.3e-9 / −1.2e-9 |

The ndcg remainders are ≤ 1.6e-9, inside the gap between the two harnesses' metric code: V1,
which matches its golden, shows up to 4.8e-10. The residual is therefore **the L5 fix, measured
against a pre-L5 golden**, not a defect of the frozen tree.

**Re-derived.** [`golden_rederive.sh`](golden_rederive.sh) re-derives the two cells with the
golden README's method: the old harness (`tmp/golden-rederive-l1l2`) with `retrieve/` replaced
by tag `campaign-v2`'s, unpatched, on the throwaway branch `tmp/golden-rederive-cv2`, never
merged. The one port, [`golden_harness_port.diff`](golden_harness_port.diff), drops LiNR V4.
Each cell ran twice, and the two runs are identical on every quality column.
[`golden_check.py`](golden_check.py) shows that V2's new cell equals V1's golden to the last
digit at @100 and @500 (@1000: −1.0e-7). The freeze's `golden_vs_h2.py --golden` then passes
V2 and V3 at 0 on recall and ≤ 1.6e-9 on ndcg. The arXiv cell is the only golden FAIL left.
[Golden README](../../../../evaluation/golden/README.md#provenance) has the provenance.

## arXiv SilverTorch: unexplained after the 2 h box

Ruled out, by measurement:

- **The `k_max` slice.** `confirm.sh` A runs the cell at `--k 100` alone, as the old harness
  did, and lands on the frozen tree's k_max-prefix number to the last digit (+2.0e-6 against
  the golden). This repeats the earlier test behind R-2.
- **The oracle build.** [`oracle_oneshot.py`](oracle_oneshot.py) rebuilds the oracle the old
  harness's way (one `q @ E^T` over all N per 64-query batch, one `torch.topk`). It equals the
  item-chunked v4 blob on every id of all 10,000 rows. SilverTorch's recall@100 against either
  is 0.8840440675.
- **Exact ground-truth ties.** [`oracle_ties.py`](oracle_ties.py) rescores the oracle in fp64.
  One row has an exact tie at oracle rank 100 (row 3702: three items with equal scores). The
  frozen SilverTorch top-100 holds all three, so no oracle choice moves a hit.
- **The index and the scorer at large.** recall@500 and @1000 agree to 6e-12. Goodreads
  SilverTorch triton, from the same library era as this golden (`4f52972`), meets its golden
  exactly.

Not tested; these are the candidates left. Both are confined to near-ties, because recall moves
by 2 hits at @100 only. On 36 rows the oracle's rank-100/101 pair is within 1e-6 and
SilverTorch's top-100 holds exactly one of the two.

1. The old harness's arXiv **inputs**. The golden cell's queries and items came from its own
   loaders and `encoded_queries_test.pt`, which are no longer on any box. A difference at
   fp32-rounding level would flip such near-ties.
2. The SilverTorch **scorer** at `4f52972`, whose `codesigned_probe_score` kernels have been
   rewritten since. An fp32 rounding difference in the final score flips the same near-ties.

Either can be separated only by running the old harness at `87a9b38` with a per-query dump.

## Files

| file | what |
|---|---|
| [`boundary.py`](boundary.py) | CPU: spill ↔ oracle pairing, per-k boundary ties and the recall they can swing |
| [`oracle_ties.py`](oracle_ties.py) | CPU: arXiv oracle rank-k ties and near-ties in fp64, rows where SilverTorch holds part of the class |
| [`confirm.sh`](confirm.sh) | GPU: A (arXiv at `--k 100`), B (V2/V3 with the pre-L5 fp16 sum, per k) |
| [`pre_l5_fp16_sum.diff`](pre_l5_fp16_sum.diff) | the experiment's kernel revert; never committed to the library |
| [`compare_confirm.py`](compare_confirm.py) | `confirm.sh`'s records against the golden JSONs |
| [`oracle_oneshot.py`](oracle_oneshot.py) | GPU: the old-style one-shot oracle against the v4 blob |
| [`golden_rederive.sh`](golden_rederive.sh) | GPU: the two golden cells re-derived at the frozen library, twice |
| [`golden_harness_port.diff`](golden_harness_port.diff) | the throwaway branch's one harness port (LiNR V4 out) |
| [`golden_check.py`](golden_check.py) | run 1 vs run 2, new vs old cells, V2 vs V1's golden |

Raw outputs (records, sidecars, logs, the re-derive) are on the Hub under `artifacts/r-res`
([hub index](../../hub-index.md)); the `_parity` spills and compile caches are not kept.
