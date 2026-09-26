---
chain: "silvertorch-official-integration"
branch: "main"
parent: "2026-09-15-120000000-wp4-record-b3-triton-vs-official-head-to-head.md"
nextStep: "After D1 (the resume key is the library tree hash): Phase G work, TF-9 probe layout and TF-1 transposed bloom, then rerun §9a/§9b; file the upstream capture issue (TF-10.1). WP-6..WP-8 not implemented."
created: "2026-09-16T12:00:00Z"
---

# TF-9 and TF-10 added to §8; §17 record: WP-9 paper section (roadmap F2)

## Standing conclusion from C4, answered by B3
C4 (2026-09-15) found `official` vs `triton` `jaccard_vs_first@100` 0.999849 on goodreads (plan cache off) but 0.985 / 0.9849 on arxiv, with `score_max_abs_diff` ~10x smaller on arxiv. B3 settled it: not the filter, the fp16 score path meeting arxiv's score distribution (95.3 % of arxiv queries have the rank-100/101 gap inside one fp16 ulp vs 3.1 % on goodreads); recall@100 cost 3.1e-4 arxiv, 4e-6 goodreads. Report it as a property of fp16 resolution against a corpus's score distribution.

## TF-9, the probe layout, not the scorer (added 2026-09-16 by the orchestrator)
B3 measured Meta's scorer 10.9-17.8x ours on goodreads and 1.15-3.2x on arxiv; cause: the padded probe layout (611,520 slots for ~18.7k real items, 97 % pad; 59 % on arxiv). A CSR or capped-pad layout (the official backend already uses CSR via `indexing.csr_layout`) removes that tax without touching the scorer's inner loop. Largest single kernel-side win measured, and larger on the more skewed corpus (the recsys direction of Phase E). Not before D1: a layout change alters `code_version` and invalidates every recorded cell. Phase G, alongside TF-1 (official transposed bloom beats ours 2.0x at 0.8 M items and 6.1x at 3.0 M; ours grows 3.3x with N vs their 1.07x).

## TF-10, upstream issue and a labelled patched-official ablation (added 2026-09-16)
`official`'s non-capturability is not intrinsic: upstream's `faster_repeat_interleave._with_cumsum_raw` takes an explicit host output size and `fused_kmean_ann_cuda.cu` never calls it; with it plus precomputed `total_cluster_*` ints the scorer's shapes would be static. Neither before D1: (1) file the upstream issue citing `faster_repeat_interleave.cuh:24-73` and the two sync sites; (2) optionally measure a patched build labelled "not the official release" in every table (B3 already measured their prep directly: 268-834 µs over 73-93 launches vs our 34-58; run only if a reviewer asks). The main comparison stays unpatched: the official arm's value is that it is Meta's released code; D7 stands.

## §17 WP-9 record (F2), 2026-09-16
Branch `dev/f2-official-section` off `development` @ `2e18d8c`, worktree `/scratch/wt/f2`. No measurement (D1-a held the GPU); every number from §16. Delivered `docs/paper/official-vs-reimplementation.md` (§1 question and two-sided answer, §2 fairness, §3 end to end, §4 kernel-only and pad tax, §5 phase 2 and S13, §6 parity and the fp16 / arxiv mechanism, §7 memory, §8 caveats: `pack_mask` makes official-exact numbers upper bounds, no claim on a bs=1 margin; §9 what it cannot say; §10 the D1 gap table; §11 falsified expectations). Updated companion rows: OF-7 and R-3 of `reproduction-deviations.md` (the "looser filter" reading replaced by the fp16 explanation), OF-4 (now cites B3 kernel-only), `provenance-and-disclosure.md` §5/§6 (B3 row closed, a D1-b row for CIs and seeds). Waits on: CIs, paired tests and a second seed (D1-b); report.py tables over the campaign (D1 -> D4); p95/p99 and open-loop QPS (D1); k in {500, 1000} ratios (D1); the full unfiltered sweep (D1-c); the controlled S9 ablation (D1-e / G-b); FPR vs width (D3); external baselines (D2); Triton transposed bloom (G-a); any post-fix padded-layout or `pack_mask` number (Phase G). Gates: links 0; no code, no tests, no GPU; checkbox left for the orchestrator.

Discrepancy found, not corrected: §16.2's prose quotes official prep "268-896 µs across 73-95 launches" and total device time "1.2-2.4x ours", while the tabulated rows span 268-834 µs, 73-93 launches, 1.20-1.93x. The paper quotes the tabulated ranges and attributes the wider ones to the prose (presumably JSON-only rows); whoever re-derives the tables should reconcile.
