---
chain: "e4-kuairand"
branch: "main"
nextStep: "No external baseline exists to target. E4's training uses patience-20 early stopping as the empirical stopping criterion; the orchestrator separately instructed E4 to cut training short once NDCG@10 improvement clearly diminishes, since the goal is usable embeddings for the retrieval benchmark, not a best-in-class recommender."
created: "2026-09-26T10:00:00Z"
---

# SASRec/gSASRec baseline research for KuaiRand-27K: no comparable number exists

Research task (sonnet, web search + direct paper reads), dispatched to
sanity-check E4's gSASRec training trajectory (NDCG@10 0.0133 at epoch 2,
0.0267 at epoch 7, still climbing) against any published baseline.

## Findings

1. **The original KuaiRand paper** (CIKM'22, arXiv:2208.08696) is a pure
   dataset/resource paper — no SASRec/GRU4Rec/BERT4Rec baselines, no
   NDCG/Recall/HR tables. Sequential models (DIN, DIEN) are mentioned only
   as future work.
2. **No other paper reports a SASRec-family baseline on KuaiRand-27K**
   with full-ranking evaluation. Nearly all sequential-rec work on
   KuaiRand uses the smaller **1K** or **Pure** variants; papers found on
   27K use unrelated debiasing metrics (SNIPS-weighted Recall@20), not
   comparable.
3. **gSASRec's own paper** (Petrov & Macdonald, RecSys'23, arXiv:2308.07192)
   — the closest architectural match — validates full-ranking NDCG@10
   only up to **1,271,638 items** (Gowalla: SASRec 0.1097, gSASRec 0.1616;
   MovieLens-1M, 3,416 items: SASRec 0.131, gSASRec 0.176; Steam, 13,044
   items: SASRec 0.0581, gSASRec 0.0735). Catalog size isn't the only
   driver — Gowalla (1.27M items, repetitive POI check-ins) scores higher
   than Steam (13K items). **KuaiRand's 32M-item catalog is ~25x beyond
   anything gSASRec's own paper tests.**
4. **No full-ranking sequential-rec benchmark exists anywhere near 32M
   items** to calibrate against — this scale is unusual even in the
   broader literature (large-catalog papers typically sample negatives or
   cap around 1-2M items).

## Assessment

No directly comparable number exists (same dataset, same 27K variant,
full-ranking, this catalog scale). Given the catalog is far outside
gSASRec's own validated range, a lower absolute NDCG@10 than the paper's
best (0.16, on a 1.27M-item catalog) is expected for a much harder
ranking problem. The observed trajectory (0.0267 at epoch 7, climbing
from 0.0133 at epoch 2) is not anomalous — it looks like a model still
converging on a harder task than anything in the literature, not a sign
of a bug. Whether it plateaus near 0.03 or climbs further isn't
answerable from these papers; the training run's own patience-based early
stopping (further shortened per the user's "usable embeddings, not best
model" instruction) is the only way to find out empirically.
