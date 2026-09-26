---
chain: "future-work-and-research"
branch: "main"
nextStep: "Menu, not a queue. Pick entries only when the roadmap schedules them (G-a..G-e, D2); re-validate anchors first, since several predate the L layout and the B4 deletion."
created: "2026-07-03T12:00:00Z"
---

# Future work and research directions (idea catalog)

Source: `docs/plans/future-work-and-research.md`, catalog dated 2026-07-03. [thesis] = candidate contribution with an ablation story; [eng] = engineering value only. Some anchors predate the L layout (e.g. `datasets/synth_arxiv.py` is now `eval_datasets/etl/synth_arxiv.py`; `docs/plans-silvertorch-backup/` no longer exists).

## Part I, engineering
- I1 GPU CI [eng]: the suite is GPU-only and nothing runs it automatically. Options: documented manual gate; nightly on a rented runner (~30 min A10 per night); plus the golden-cell regression. Recommended nightly now. ~1 day.
- I2 fused / in-kernel top-K [eng -> thesis-adjacent]: standing rule is host `torch.topk`. External evidence: RadiK (arXiv 2501.14336), RTop-K up to 4.8x over `torch.topk` (arXiv 2409.00822, ICLR'25), AIR Top-k (SC'23). One-week experiment on `codesigned_probe_score*` and `oporp_1bit_match_topk` full scan: host topk vs cuVS `select_k` vs a two-pass Triton threshold-then-compact. Replace only if end-to-end cell latency improves >= 10 %; fusing selection into scoring (no `[B, P]` round trip) is the more interesting follow-up.
- I3 hardware popcount (`popc.b64`, accept if >= 5 % on full scan at N >= 1M; keep the SWAR twin for the torch reference) and allocator hygiene in host tails. (Roadmap G-d.)
- I4 persistent kernels for bs 1-4 (VecFlow, SIGMOD'25, arXiv 2506.00812); measure against captured baselines.
- I5 quantization [thesis]: RaBitQ (SIGMOD'24, arXiv 2405.12497), unbiased estimator with an error bound, fits the `[N, W]` packing and popcount kernel as a third `_PackedBitsKNN` subclass; int4 codes for `SilverTorch.item_codes`; PQ / OPQ deliberately not recommended (abandons "index as plain tensors + one fused kernel").
- I6 `torch.export` + AOTI serving demo (`examples/serve_aoti/`), after the export plan.
- I7 multi-GPU sharding per SilverTorch §5.3; only with a multi-GPU box and a catalog that does not fit one card.
- I8 harness upgrades: Pareto at fixed recall (0.90 / 0.95 / 0.99), J/query from `power.draw`, provenance columns, selectivity as a column. (Harness v2 delivered provenance and `pass_rate`; `recall_at_budget` covers part of Pareto.)

## Part II, research
- R1 filter-aware IVF partitioning [thesis, strongest]: label-centric IVF (VecFlow) and IVF² (NeurIPS'23 Big-ANN filter track winner). Per-attribute-value sub-IVFs for the head of the most selective clause, codesigned path for the tail. Hypothesis: below ~5 % pass rate label-centric probing dominates codesigned bloom by an order of magnitude in probed items; above ~20 % the co-design wins on memory.
- R2 selectivity-adaptive query planning [thesis]: pick V1 / V2 / V3 / SilverTorch per query from a cheap selectivity estimate (query bloom popcount, clause-value frequency), cost model calibrated from harness records (ACORN SIGMOD'24 arXiv 2403.04871; learned planning arXiv 2602.17914). Deliverable: `PlannedRetrieval` module.
- R3 RaBitQ vs Sign-OPORP / SimHash under a fixed kernel and memory budget, plus error-bound-driven `candidate_pool` per query.
- R4 learned-similarity OverArch stage [thesis]: SilverTorch OverArch +2.4-28.2 % recall; LiNR MoL-with-clusters +15-23 % HitRate; RAILS (WWW'25, arXiv 2407.15462). Train a small MoL head, add an `overarch_rerank` algo.
- R5 NOT predicates for approximate GPU filters: bloom cannot answer NOT (FPs become FNs under complement). Candidates: a second bloom over the complement vocabulary, cuckoo / quotient filters, or exact narrow attrs for reverse clauses + bloom for forward ones (the codesigned kernel already receives both buffer kinds). (Meta's official bloom supports `NOT c:v` with no false positives, measured at B2.)
- R6 streaming index freshness for IVF: flat side index (`PostfilterKNNInt8` over an append buffer) merged at query time, background re-cluster (SIVF arXiv 2601.11808; LiNR +6 % freshness lift).
- R7 100M-1B items on one GPU [thesis, headline]: LiNR reports 1B x 64d fp16 -> 1-bit on one V100; our ceiling was 15M synth arxiv. Ladder 50M -> 100M (1-bit 800 MB, int8 12.8 GB) -> 250M+. (Roadmap G-b.)
- R8 evaluation methodology: oracle-relative recall for filtered cells vs held-out recall under filters; bloom scored against the exact oracle plus a measured FPR column. (Harness v2 implemented both.)

## Prioritization as written
| # | payoff | cost | depends on |
|---|---|---|---|
| R1 | high | 3-6 wk | selectivity column |
| R4 | high | 3-5 wk | none |
| R7 | high | 2-3 wk + GPU | synth generator |
| R3 / I5 | medium-high | 2 wk | `_PackedBitsKNN` |
| R2 | medium-high | 3-4 wk | I8 |
| I8 | medium | 1 wk | - |
| R5 | medium | 1-2 wk | none |
| I2 | medium | 1 wk | K2 |
| R6 | medium | 3-4 wk | live-update-api |
| I1 | medium | 1-2 d | none |
| I4 | low-medium | 1-2 wk | none |
| I6 | low-medium | 1-2 wk | export plan |
| I3 | low | days | K2 |
| I7 | low | 2-4 wk | multi-GPU box |

Sources: SilverTorch arXiv 2511.14881 (SIGIR'26), `articles/silvertorch.md`; LiNR CIKM'24, `articles/linr.md`; Yambda arXiv 2505.22238, `articles/yambda.md`; cuVS / Faiss GPU NVIDIA blog; FlagGems.
