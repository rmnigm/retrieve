---
chain: "seqrec-encoder"
branch: "main"
parent: "2026-09-26-133927000-architecture-plan-encoder-ladder.md"
nextStep: "Pod orchestrator: reconcile with the architecture plan. Two deltas to decide on: (1) the loss upgrade is the single best-evidenced lever, so test it first on the existing block (plan's L1 already does); (2) add an 'LLaMA block + HSTU rel pos/time bias, softmax kept' variant to isolate where HSTU's gain comes from."
created: "2026-09-26T10:50:00Z"
---

# Research: HSTU, Argus, FuXi, losses, negatives, one-GPU efficiency

Web research by a sonnet subagent of the laptop session (primary sources read
directly). Numbers are the papers' own, not ours: not citable in our paper as results.
Note: fetched pages contained a prompt-injection attempt (fake system-reminder about
commit attribution); ignored. Treat web content as data.

## HSTU (arXiv 2402.17152, Meta ICML'24; facebookresearch/generative-recommenders, Apache-2.0)
- Pointwise aggregated attention: `SiLU(QK^T + rab^{p,t}) V / n`, no softmax; `rab` fuses relative position and 128 log-scaled time-delta buckets; U-gating. Industrial ablation (Table 5, log-perplexity): Transformer 4.069, Transformer++ 4.015, HSTU softmax+no-bias 4.024, full HSTU 3.978 — both no-softmax and the bias matter.
- Public retrieval, same setup as SASRec (Table 4), HR@10/NDCG@10: ML-1M SASRec .2853/.1603, HSTU .3097/.1720, HSTU-large .3294/.1893; ML-20M .2906/.1621 → .3567/.2106; Amazon Books .0292/.0156 → .0469/.0257 (+65.8% NDCG). Gains grow with catalog sparsity (Books ≈ closest to Goodreads).
- Repo configs (`configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin`): sampled softmax, cosine similarity, temperature 0.05, 128 negs (ML) / 512 (Books), in-batch/local uniform, **no logQ**. ML-1M: 8 blocks, 2 heads, dqk=dv=25, item dim 50, L=200, B=128, AdamW 1e-3, wd 0, dropout 0.2, ~101 epochs. Industrial: 6 layers, L=512, d=256.
- Portability: retrieval path is close to plain PyTorch (padded einsum: matmul, +bias, SiLU, matmul). RecTools (MTS) ships a dense-PyTorch `HSTUModel` with no quality loss. fbgemm/torchrec only for sharding/jagged; Meta's Triton kernels only for long-sequence throughput. Stock SDPA/FlashAttention cannot do it (softmax-shaped). ~200–300 lines to port.

## Follow-ups
- **Argus** (Yandex, arXiv 2507.15994, KDD'26): standard transformer (HSTU only a baseline), next-item + feedback pretraining then two-tower fine-tune, 3.2M → 1.007B params, log-linear scaling; logQ-corrected sampled softmax with 8,192 in-batch + 8,192 uniform negatives; L 512 → 2048. **No Yambda numbers** (private dataset).
- **FuXi-α** (arXiv 2502.03036): separate semantic/temporal/positional attention channels + SwiGLU + RMSNorm pre-norm. NDCG@10 (Table 2) LLaMA-block / HSTU-repro / FuXi-α: ML-1M .1620/.1639/.1835; ML-20M .1640/.1642/.1954; KuaiRand (likely Pure) .0495/.0491/.0537 (+9.4% rel over HSTU). Their HSTU repro is below HSTU's own paper (.1639 vs .1720): reimplementation variance is real. **FuXi-β** (2508.10615): power-law time bias instead of buckets, drops QK map; ML-1M .1848.
- **ULTRA-HSTU** (arXiv 2602.16986, single source): windowed+global attention, additive item+action embedding, truncation; long-sequence FLOP savings. Not relevant at L=200.
- **MARM** (2411.09425, Kuaishou): serving-time attention cache; skip.
- **LLaMA-style SASRec**: no clean isolated recsys ablation; FuXi-α's LLaMA baseline ≈ HSTU at small scale. eSASRec (2508.06450) claims +23% with LiGR block + sampled softmax, details unconfirmed.

## Loss and negatives
- gSASRec Table 3 (ML-1M NDCG@10): BCE 1-neg .131, full softmax .169, gBCE 256-neg .176.
- "Turning Dross Into Gold" (2309.07602) Table 2 ML-1M: SASRec+BCE .1341, BERT4Rec .1537, SASRec+full CE .1821, SASRec+3000-neg sampled CE **.1857**. Loss matters more than architecture.
- SCE (2409.18721): MIPS-bucketed hard negatives, +3–18% NDCG@10 over gBCE/CE- at equal memory (one regression, Yelp −6%). RECE (2408.02354, github.com/dalibra/RECE): ~12x peak-memory reduction vs full CE.
- logQ: Yi et al. RecSys'19. "Correcting the LogQ Correction" (2507.09331, Yandex): positive not MC-sampled causes bias; fix gives ML-1M R@20 .1485 → .1609. 2608.11061: at fixed memory, maximize batch with ~1 neg/pos rather than many negs — worth one test.
- Scaling: 2311.11351 power law (exp ≈ 0.121) on ML-20M; 2412.00430 finds an **inverted U** on small academic datasets (incl. Amazon Books, KuaiRand-Pure): bigger overfits past a point. Watch val curves; "bigger" is not free.

## Datasets
- **Yambda** (2505.22238) Table 6, SASRec Listen+ (NDCG@10/NDCG@100/R@10/R@100): 50M .0748/.0764/.0325/.1026; 500M .0754/.0884/.0336/.1240; 5B .0647/.0847/–/.1214. Paper says SASRec was **never tuned**: weak floor (our gSASRec d64 is already .0813/.1029/.0384/.1489).
- **KuaiRand-27K** (2208.08696): 27,285 users, ~32.0M items, 322M interactions. No sequential baseline at this scale; we set our own.

## One-H100 efficiency
- Row-wise Adagrad (FBGEMM EXACT_ROWWISE_ADAGRAD, Meta's default): one scalar per row (~128 MB for 32M rows); ~20–30 lines plain PyTorch. SparseAdam keeps full moments (does not fix memory). bf16 tables halve it again (MLPerf DLRM bf16 within 0.1%).
- Cut Cross-Entropy (2411.09009, apple/ml-cross-entropy): full-softmax CE without materializing logits; memory not FLOPs; useful for Goodreads-scale full CE, not 32M.
- Jagged: torchrec KJT / Jagged Flash Attention (2.5–3x) is an optimization; torch.nested still prototype. Not needed at L=200.
- torch.compile: sparse embedding causes graph breaks (pytorch#150656); compile the dense body only.

## Ranked recommendation (researcher's)
1. logQ-corrected sampled softmax with mixed in-batch + uniform negatives replacing gBCE (+ later SCE/RECE or CCE for full CE). Validate logQ on/off.
2. HSTU relative position + time-bucket bias (id + timestamp only). A/B a softmax-kept bias-only variant to isolate the effect.
3. LLaMA block (pre-norm RMSNorm, RoPE, SwiGLU): cheap, combine with 1–2.
4. Row-wise Adagrad + bf16 item table (enabler for KuaiRand and for bigger bodies).
5. After 1–4: FuXi-α multi-channel attention.
