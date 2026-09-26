---
chain: "seqrec-encoder"
branch: "research-pinterest"
parent: "2026-09-26-135000000-research-hstu-argus-losses-efficiency.md"
nextStep: "Pod orchestrator: add a training-objective ablation - widen the sampled-softmax positive target from 'strict next item' to 'uniform draw from the next K positions' (K in {1,3,5,10}), same loss/negatives/logQ, measured on strict next-item NDCG@10/Recall@100 on goodreads first (cheapest). This is PinnerFormer's dense all-action loss adapted to position-window instead of day-window, corroborated by PinFM's L_mtl/L_ftl and PinRec beating an HSTU baseline with multi-token targets. Second, cheap add: additive action-type embedding on the input tower (PinFM/TransAct-style) wherever action type exists (KuaiRand, yambda)."
created: "2026-09-26T16:29:47Z"
---

# Research: Pinterest sequential / user-representation models (PinnerFormer, TransAct family, PinFM, PinRec, item-table side)

Web research by a sonnet subagent, primary sources read directly (arXiv
abstracts/HTML, not secondary blogs, except where noted). No prompt-injection
attempts were found in the fetched pages this session. Numbers below are the
papers' own, not ours — not citable as our results, only as motivation.

## PinnerFormer (arXiv 2205.04507, KDD'22) — retrieval, closest match to our setup

- **Architecture**: 6-layer pre-norm transformer, d=768, 8 heads, causal
  self-attention, max sequence M=256 (tried to 512). Per-action input =
  PinSage pin embedding (256-d) + action-type/surface embeddings + log(duration)
  + modified Time2Vec (periods 0.25h–365d) + log-time, concatenated then
  projected. Output L2-normalized, final user embedding D=256.
- **Objective — dense all-action loss**: for randomly chosen sequence
  positions, predict *one positive drawn uniformly from all engagements in
  the next K-day window* (train K=28d, eval K=14d), not the literal next
  action. Sampled softmax with logQ correction, dot/τ score, τ learnable
  (init constrained ≥0.01). Negatives: in-batch (all other positives in
  batch, capped 5,000) + 8,192 uniform random from corpus; sampling
  probability estimated via count-min sketch.
- **Why not next-item**: explicitly framed against next-action baselines
  (SASRec-style); the multi-day window is argued to make the embedding
  robust to daily-batch staleness — realtime→batch recall drop is 8.3% for
  PinnerFormer vs 13.9% for a SASRec-style next-item baseline (their own
  ablation).
- **Offline**: Recall@10 on a random 1M-pin index — PinnerFormer 0.229 vs
  PinnerSage(5 clusters) 0.026, PinnerSage(20) 0.046.
- **Online A/B**: Homefeed (replacing PinnerSage): +7.5% repins, +6%
  close-ups, +1% CTR, +1% time spent, +0.4% DAU. Ads/Search/Homefeed CTR
  when *adding* PinnerFormer as a ranking feature: +7–10% CTR.
- **Serving**: daily incremental batch job, embeddings recomputed only for
  users active in the last 24h, HNSW ANN index. No code release found.

## TransAct (arXiv 2306.00248, KDD'23) — **ranking**, not retrieval

- 2-layer, 1-head transformer, ffn=32 (384 tested, +30% latency, dropped),
  **no positional encoding** (found "ineffective" offline), sequence length
  100. Per-action = 32-d action-type embedding + 32-d PinSage pin embedding.
  **Early fusion**: the *candidate* pin embedding is concatenated into every
  position of the sequence before self-attention — this is target-aware
  attention over a known candidate, output compressed to first-K columns +
  max-pool (11×96-d), then fed into a DCN-v2 crossing layer next to a
  PinnerFormer batch embedding. Multi-label weighted BCE over click/repin/hide.
- Scale: 3B examples, 177M users, 720M pins, 60M→92M params, GPU-optimized
  latency 8ms.
- Offline HIT@3/repin: WDL+seq +0.21%, BST +4.41%, **TransAct +9.40%**
  (Table 1); ablation shows TransAct and PinnerFormer are complementary
  (removing either costs 2.5–8.6% HIT@3/repin, Table 2).
- Online (1.5% Homefeed A/B, Table 5): +11.0% repin volume, +2.0% time
  spent, −10.0% hides. Code: `github.com/pinterest/transformer_user_action`.
- **Key structural fact**: because the candidate pin is fused into the
  sequence before attention, this is a per-candidate ranking computation,
  not a bi-encoder — it cannot produce a single `[B,D]` query vector scored
  against a full item table without re-running the transformer once per
  candidate.

## TransAct V2 (arXiv 2506.02267, 2025) — **ranking**, lifelong sequences

- Adds a lifelong sequence (~10⁴ actions, 2 years) and an impression
  (viewed-not-engaged) sequence to TransAct's realtime one. Compresses the
  lifelong sequence via **candidate-aware nearest-neighbor search**:
  `NN(S,c) = top-K by dot(PinSage(S), e_c)` — again requires the candidate
  at encode time. Concatenated sequence (~10² tokens) → 2-layer, 1-head,
  d=64 transformer, additive action/surface/position embeddings.
- **Next Action Loss (NAL)**, Eq. 5–7: sampled softmax
  `-log(e^<u(t),p(t+1)> / (e^<u(t),p(t+1)> + Σ e^<u(t),n>))`, added to the
  main CE loss with weight 0.01. Negatives from the **impression sequence**
  beat in-batch random (+1.10% vs +0.63% HIT@3/repin) — i.e. hard negatives
  from real, logged non-clicks.
- Offline (Table 2): TransAct(RT only) +7.74% HIT@3/repin → **TransAct V2
  +13.31%** (with lifelong seq + NAL). Online (Table 3): +6.35% repin
  volume, +1.41% time spent. Serving: request dedup + fused
  dequant + a fused kernel give 250–338× p99 latency reduction. No code
  release; paper notes no public CTR dataset has lifelong-sequence features.

## PinFM (arXiv 2507.12704, RecSys'25) — foundation model, pretrain=retrieval-shaped, finetune=ranking

- GPT-2-style pre-LN decoder, **20B+ params** (mostly embedding tables),
  pretrain sequence length 16,000, input = (timestamp, action_type,
  surface_type, item_id) additively combined + MLP + L2-norm.
- **Pretraining** (retrieval-shaped): InfoNCE with in-batch negatives, three
  target strategies — next-token, **multi-token** (predict any of the next
  L′ tokens, "interests stay consistent short-term"), and **future-token**
  (predict a window starting L_d steps ahead, "instruction-tuning-like"
  alignment to the downstream task). Table 3: adding L_mtl+L_ftl to
  next-token-only improved Save +0.95%, Hide +2.43% offline.
- **Finetuning** (ranking, not retrieval): early-fusion cross-attention of
  the candidate against cached user K/V — the **Deduplicated Cross-Attention
  Transformer (DCAT)** exploits the fact that in serving, unique user
  sequences : candidates ≈ 1:1000, caching K/V once per user and
  cross-attending per candidate (+600% serving throughput, +200% training
  throughput vs plain self-attention). This is a ranking-stage trick, not
  applicable to full-catalog retrieval (would mean one cross-attention pass
  per catalog item).
- int4 embedding quantization: 512→160 bit/vector, 0.06% offline metric
  drop, online-neutral. Online (Table 7): Home Feed sitewide saves +1.20%,
  Related Items +0.72%. No code/checkpoint release; CC BY 4.0 paper license.

## PinRec (arXiv 2504.10507, 2025) — generative retrieval, directly benchmarked against HSTU

- Transformer-decoder generative retrieval using **dense pretrained item
  embeddings** (OmniSage for pins, OmniSearchSage for queries) instead of
  semantic IDs — paper reports semantic IDs "frequently suffered from
  representational collapse" at their scale. Surface-agnostic
  pretrain→finetune.
- Sampled softmax with two negative pools: in-batch impression negatives
  (shown-not-engaged, +0.4% from an ablation) and 1M random negatives, with
  count-min-sketch bias correction (same recipe family as PinnerFormer).
  **Outcome-conditioned generation**: conditions the decoder on a desired
  action (save/click) to steer surface-specific optimization; +2–3% from
  conditioning alone.
- **Directly compares to HSTU as a baseline** (unordered recall@10): Home
  Feed PinRec-OC 0.625 vs **HSTU 0.596**; Search 0.352 vs **HSTU 0.179**.
  This is the one place in Pinterest's published work where an HSTU-style
  single-embedding retrieval baseline is beaten by a *multi-target*
  training/decoding scheme, corroborating PinnerFormer's and PinFM's
  window/multi-token findings independent of architecture.
- Online: Search saves +3.88%, Home Feed grid clicks +4.01%. Targets
  full-catalog retrieval (paper states ~1M items). Code: "publicly
  available upon publication" — release status unconfirmed from the text.

## Item-table side: PinSage, ItemSage, OmniSage, OmniSearchSage — not ID-only

- **PinSage**: GraphSAGE-style GNN over the pin-board graph + visual/text
  content, producing the 256-d pin embeddings PinnerFormer/TransAct consume.
- **ItemSage** (arXiv 2205.11728): multi-task product embeddings combining
  PinSage image embeddings + SearchSage query-string embeddings for shopping
  retrieval.
- **OmniSage** (arXiv 2504.17811, KDD'25 industry track): unifies GNN +
  content model + user-sequence model via multiple contrastive tasks into
  one universal entity embedding, billions of nodes, +2.5% sitewide repins
  across 5 surfaces; code stated public, CC BY 4.0.
- **OmniSearchSage** (arXiv 2404.16260): joint query/pin/product embeddings
  for search, +8% relevance / +7% engagement / +5% ads CTR.
- All four require content (images, text, graph structure, or search query
  logs) that our project explicitly excludes ("no per-dataset side features
  or content towers"). They matter only insofar as they explain *why*
  Pinterest's user-sequence papers use a 256-d dense "pin embedding" per
  action instead of a bare item id — a content input we don't have and
  aren't adding.

## Transfers to our setup (ranked, ID/timestamp/action-type only, generic across datasets)

1. **Multi-positive / future-window training objective** (PinnerFormer's
   dense all-action loss; corroborated by PinFM's `L_mtl`/`L_ftl` and by
   PinRec beating an HSTU baseline with multi-token targets). Adapt as: for
   an anchor at position `i`, sample the sampled-softmax positive uniformly
   from positions `i+1 .. i+K` instead of always `i+1`; keep our loss,
   temperature 0.05, negatives and logQ unchanged. ID+position(+timestamp
   if a day-window is preferred over a position-window)-only, no action
   type or content needed. Effort: a data-loader change plus a small K
   sweep (K∈{1,3,5,10}), ~1 GPU-hour per config on goodreads (cheapest
   dataset), then confirm on yambda.
   - **Evidence**: PinnerFormer's own motivation is batch-vs-realtime
     staleness, not next-item accuracy — its offline metric is Recall@10
     against a random 1M index using its own window-consistent eval, *not*
     a strict-next-item protocol like ours. PinFM's ablation (Table 3) is
     the more directly comparable one: a small multi-token addition on top
     of next-token improved both Save and Hide.
   - **Risk — assess carefully, do not assume it transfers**: our harness
     metric is NDCG@10/Recall@100 on the literal held-out next item.
     Widening the training target dilutes the next-item-specific signal;
     if K is too large relative to how fast a user's interest drifts
     (goodreads genres over months, yambda listens over weeks), this could
     *reduce* our reported next-item numbers even though it may be "more
     correct" by Pinterest's own multi-day-window standard. This is a real
     objective/eval mismatch, not a free win — treat K=1 as the control and
     require the sweep to show a strict-next-item gain before adopting.

2. **Additive action-type (+ surface, where present) embedding on the input
   tower**, PinFM/TransAct-style: `item_emb + action_type_emb + position_emb
   (+ surface_emb)`, all additive, no content. Compatible with KuaiRand
   (click/like/follow/…) and yambda (listen/like/dislike/unlike); Goodreads
   ratings could map to coarse action buckets. Cheap (one embedding table,
   few hundred params), low risk, matches an input we already listed as
   available ("possibly an action type") but aren't yet using structurally
   the way Pinterest does (as an additive input feature rather than only a
   softmax label/weight).

3. **Negative-sampling recipe cross-check**, not a new idea but a
   confirmation: PinnerFormer, TransAct V2 and PinFM all converge on the
   same recipe we already implemented (`dev/hstu-logq`, merged) — sampled
   softmax + logQ correction + mixed in-batch/uniform negatives. Worth one
   cheap check: our in-batch cap and uniform-negative count against
   PinnerFormer's found operating point (in-batch capped at 5,000, 8,192
   uniform) to see if our current counts are far off that ratio at our
   batch sizes. Low effort, low expected gain since we already have the
   qualitative recipe right; this is a tuning check, not an architecture
   change.

## Not applicable / needs data we lack

- **TransAct, TransAct V2, PinFM's finetune/DCAT**: all target-aware —
  the candidate is fused into the sequence (or cross-attended via cached
  K/V) *before* scoring. This is a ranking-stage architecture over a
  shortlist of candidates; it does not produce a `[B,D]` query vector
  usable against a full item table, and running it once per catalog item
  is infeasible at our N (1.9M–32M). Not portable to our bi-encoder
  retrieval setup regardless of GPU budget.
- **TransAct V2's impression-based hard negatives (NAL_imp)**: needs logged
  non-click impressions. Not generically available — KuaiRand-27K does log
  unclicked exposures, but yambda and goodreads do not, and the project
  requires the method be generic across datasets, so this is dataset-specific
  at best, not a portfolio-wide change.
- **PinFM's DCAT, 20B+ param scale, int4 quantized 8-sub-table embeddings,
  cold-start ID randomization**: production-infra scale and ranking-stage
  techniques (BCE finetune on click/hide labels), out of scope for a
  single-H100 few-GPU-hour experiment, and BCE ranking labels aren't part
  of our retrieval-only task.
- **PinSage / ItemSage / OmniSage / OmniSearchSage**: all require content
  (images, text) or a user-item-board graph — explicitly excluded by our
  "no per-dataset side features or content towers" constraint. They only
  explain where Pinterest's 256-d "pin embedding" input feature comes from;
  we use a bare learned item-id embedding instead, which is the setup those
  papers assume as a *given* input, not a design choice we can copy without
  the content pipeline behind it.
- **PinRec's dense pretrained item embeddings replacing semantic IDs /
  learned ids**: same content dependency (OmniSage embeddings) as above;
  our id-only design is closer to their "semantic IDs" alternative that
  PinRec's authors say underperformed *for them*, but that comparison was
  made against content-derived dense embeddings we don't have access to, so
  it doesn't transfer as a reason to change our id embedding table.
- **TransAct's serving-latency work (GPU kernel optimization, SKUT, request
  dedup)**: infra/latency engineering for production serving, not a model-
  quality lever for offline retrieval eval.

## Sources

- PinnerFormer: https://arxiv.org/abs/2205.04507 (https://ar5iv.labs.arxiv.org/html/2205.04507)
- TransAct: https://arxiv.org/abs/2306.00248 (https://ar5iv.labs.arxiv.org/html/2306.00248), code: https://github.com/pinterest/transformer_user_action
- TransAct V2: https://arxiv.org/abs/2506.02267 (https://arxiv.org/html/2506.02267v1)
- PinFM: https://arxiv.org/abs/2507.12704 (https://arxiv.org/html/2507.12704v3)
- PinRec: https://arxiv.org/html/2504.10507v5
- OmniSage: https://arxiv.org/abs/2504.17811
- OmniSearchSage: https://arxiv.org/abs/2404.16260
- ItemSage: https://arxiv.org/abs/2205.11728
- Rethinking Personalized Ranking at Pinterest (E2E ranking, context only, not fetched in full): https://arxiv.org/abs/2209.08435
- Multi-Embedding Retrieval at Pinterest (2025, context only, not fetched in full): https://arxiv.org/pdf/2506.23060
