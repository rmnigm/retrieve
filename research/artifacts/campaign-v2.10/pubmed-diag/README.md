# PUBMED-DIAG: the exact arms' recall shortfall and the bloom recall gap (PubMed 10 M d768)

Not citable (single seed, diagnostic). Pod `a100-x1-b`, A100-SXM4-80GB GPU 0, 2026-10-10. (a) ran on the `campaign-v2.10` tag
(library `a3bec4a5`, detached worktree), quality only, 0.16 GPU-h. (b) is CPU only, from the records. Raw outputs: Hub
`artifacts/pubmed-diag` ([hub-index](../../hub-index.md)).

## (a) V1 / V2 recall_oracle@100 0.996-0.999: fp16 input rounding, not a defect

[`exact_miss.py`](exact_miss.py) runs the real `filter` suite's V1 (`linr_v1_filter_mask`) and V2 cells (triton, clause, seed 0,
k_max 1000) through the harness's quality pass. It reproduces F3's records (v2.9) to 1e-7.

Each missed item `m` is an item of the oracle top-100 that is absent from the arm's top-100. It is paired with `a`, the arm's
lowest-scored intruder: an item in the arm's top-100 that is not in the oracle's. [`v2_rescore.py`](v2_rescore.py) scores each pair
with the arm's own kernel: V1's cuBLAS fp16 × fp16 → fp32, and V2's Triton `fused_masked_knn_topk`. Exact fp32 is the oracle's
precision.

| sweep (pass) | arm | missed | fp16 order (arm(m) < arm(a), fp32(m) > fp32(a)) | tie at rank k in the arm's scores | fp32 tie (≤ 2 ulps) | other |
|---|---|---|---|---|---|---|
| `c0_mesh` (0.0002) | V1 / V2 | 1,235 / 1,236 | 1,225 / 1,227 (99 %) | 5 / 3 | 4 / 5 | 1 / 1 |
| `all5` (0.018) | V1 / V2 | 1,177 / 1,165 | 1,008 / 1,007 (86 %) | 92 / 93 | 77 / 65 | 0 / 0 |
| `c3_journal_reverse` (0.999) | V1 / V2 | 3,464 / 3,460 | 3,245 / 3,259 (94 %) | 111 / 130 | 103 / 62 | 5 / 9 |

- **No arm ever scores a missed item above the intruder it kept: 0 of 11,737 pairs.** So no passing item is dropped.
- Most of the shortfall is the fp16 catalog: its rounding orders pairs differently from fp32.
- Ties at rank k are a small share. They concentrate on `all5`, which has many exactly equal scores.
- "Other" (≤ 9 per cell) are fp32 near-ties of 3-8 ulps. There the arm's order agrees with this script's single-vector fp32, and
  the oracle's chunked fp32 matmul rounds the pair the other way.
- V1 and V2 return the same id sets on 99.2-99.9 % of rows. The rest differ only by these tie and rounding orders.
- The v2.10 k 100 / 1000 oracle blobs are `torch.equal` to the v2.9 blobs, or to their k 1000 prefix
  ([`oracle_compare.py`](oracle_compare.py)).

## (b) Saturated bloom recall 0.673 (v2.9) vs 0.6699 (408b1188): median vs mean, no change

[`bloom_seeds.py`](bloom_seeds.py) compares the 42 `bloomwidth` cells of `408b1188` and `e8958bd2`. All 42 per-query recall arrays
are identical, as is every seed's recall@100. The saturated cells read 0.6640 / 0.6843 / 0.6699 (seeds 0 / 1 / 2) at both versions.
Seed 0 at the default width is 0.6640 at every version run (408b1188, v2.1, v2.3, v2.9).

The 408b1188 and v2.1 artifacts tabulate the **median** over seeds: 0.6699 is seed 2. The v2.9 row gave the **mean**, 0.6727. No
oracle, bloom hash, index build or tie order changed between the two. The validation rows now name their statistic.
