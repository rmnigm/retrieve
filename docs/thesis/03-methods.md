# Chapter 2. Methods of model-based GPU retrieval

> **Working document.** English-language content notes per subsection, prepared as raw material for the final Russian academic prose pass. Section structure 2.1–2.5 follows the contract in [`00-thesis-plan.md`](00-thesis-plan.md) verbatim. Citations are restricted to non-Meta-affiliated work (Citation Policy, `00-thesis-plan.md` §Citation Policy). The IVF + INT8 + Bloom retriever in §2.3 is presented as a second bundled retriever in the `retrieve` framework, assembled from classical primitives — it demonstrates that the framework's contracts (`FilterModule`, `Backend`) accommodate retrievers outside the LinR family. It is not described or cited under its industrial-shorthand name in body prose, though the in-repo module path `layers/silvertorch/` is preserved as a code-internal symbol (allowed per `00-thesis-plan.md` Citation Policy) and may appear in `path/file.py:LINE` references.

## Chapter introduction

This chapter formalizes the retrieval algorithms reimplemented in the `torchretrieve` package and studied empirically in subsequent chapters. The chapter is deliberately algorithmic: it gives the mathematical and pseudocode-level descriptions decoupled from Triton-specific implementation details (which belong in Ch.4) and from evaluation results (Ch.6). The reader is assumed to be familiar with standard GPU compute primitives (matmul on tensor cores, top-K via radix-select, popcount) but no familiarity with the LinR paper or with `torchretrieve` is required — every algorithm is described self-contained.

The chapter has five sections. §2.1 fixes the retrieval task formulation (top-K dot-product with attribute filtering) and the notation used throughout the thesis. §2.2 describes the LinR family (V1 post-filter mask, V2 pre-filter reduction, V3 1-bit OPORP) as published in Borisyuk et al. 2024 (CIKM, LinkedIn); each variant is given pseudocode, complexity analysis, and a region of applicability. §2.3 describes the co-designed IVF + INT8 + Bloom retriever as a second bundled retriever in the framework, assembled from three classical primitives (IVF clustering — Jégou et al. 2011; INT8 quantization with dp4a hardware path — NVIDIA CUDA Programming Guide; Bloom-style signature subset tests — Bloom 1970 and Goodwin et al. 2017 BitFunnel); the section demonstrates that the framework's `FilterModule` and `Backend` contracts compose retrievers outside the LinR line. §2.4 formalizes the unified `FilterModule` contract and its two composition operators (dense AND of masks; sparse cascade of indices). §2.5 gives the formal definitions of the two quantization schemes (per-tensor symmetric INT8; Sign-OPORP 1-bit) that the preceding sections rely on.

**Framing for the final prose.** The chapter's job is to be the formal reference the rest of the thesis points back to. Keep prose declarative and dense; mathematical formulas should be set in display mode where they carry the reasoning. Pseudocode blocks (Algorithms 1–4) are the load-bearing exposition of each retrieval algorithm — the surrounding prose explains *why* the algorithm has the form it does and *when* to use it, not what each line does. Wherever an algorithmic choice is motivated by a hardware constraint (dp4a alignment, cuBLAS launch path, popcount via bit-twiddle), say so explicitly — this is software-engineering-heavy work and the reader will be looking for those signals.

---

## 2.1 Retrieval task formulation

**Sources:** `articles/linr.md` (problem statement), `evaluation/retrieval/metrics.py` (for the connection to evaluation), `docs/system/filtering.md` (clause formalism).

**What to cover.** Introduce the retrieval task as top-K dot-product search over a fixed item catalogue, then layer attribute filtering on top of it. Roughly half a page. No figures.

### Setup

- Catalogue: $N$ items, each represented by an embedding $x_i \in \mathbb{R}^D$; matrix form $X \in \mathbb{R}^{N \times D}$.
- Query batch: $B$ queries $q_b \in \mathbb{R}^D$; matrix form $Q \in \mathbb{R}^{B \times D}$.
- Similarity: dot product $\langle q_b, x_i \rangle$. For $\ell_2$-normalized embeddings this equals cosine similarity up to monotone transform, so we can use either interchangeably without affecting top-K.
- Unfiltered top-K: $\mathcal{T}_b = \operatorname{top-}K_{i \in [N]} \langle q_b, x_i \rangle$.
- Filter predicate: $\phi_b: [N] \to \{0, 1\}$ depending on the query's clause attributes; passes items in some legal subset.
- Filtered top-K: $\mathcal{T}_b^{\phi} = \operatorname{top-}K_{i: \phi_b(i) = 1} \langle q_b, x_i \rangle$.
- Selectivity / pass-rate: $\rho_b = |\{i: \phi_b(i) = 1\}| / N$.

### Clause-based filter formalism (forward-pointer to §2.4)

A query carries $C$ clauses. Each clause is either an *exact-attribute* equality test (e.g., language = "ru") or a *Bloom* signature subset test. Clauses combine conjunctively: $\phi_b(i) = 1$ iff item $i$ satisfies every active clause. Reverse clauses (logical NOT) are supported on exact filters only — see §2.4.2 for why Bloom can't support negation. Full formal definition is in §2.4; here it is enough to give the reader the high-level picture.

### Why selectivity matters

Selectivity drives the algorithm choice. At $\rho \to 1$ the cost of computing the full $Q X^\top$ matrix is unavoidable anyway, and adding a mask is essentially free — V1 wins. At $\rho \ll 1$ the cost of scoring items that fail the filter is the dominant overhead — V2 (sparse pre-filter) wins, but only up to the cross-over point where index-time gather overhead overtakes the FLOP savings. Beyond a certain catalogue size (roughly $N \gtrsim 10^6$), even V2 stops scaling because the candidate list grows with $\rho N$; this is the regime that motivates the IVF + INT8 + Bloom retriever in §2.3. The whole point of having multiple retrieval algorithms in the library is that no single one dominates across $(\rho, N, D, \text{filter shape})$ — Ch.6 maps out the regions empirically.

### Connection to evaluation

Quality (Recall@K, NDCG@K, MRR@K) and latency (median, p20, p80) metrics are defined formally in Ch.5; here just mention that retrieval quality is measured against held-out future interactions (per-dataset protocol in Ch.3), not against the full ranking-stage quality, which is out of scope.

**Framing for the final prose.** Open with the canonical two-stage architecture (retrieval → ranking) to anchor the reader; the lit review will have introduced this already in §1.1, so a single sentence is enough. The bulk of §2.1 should set up the notation that survives the whole chapter. Make sure the symbols $B, N, D, K, P, L, L_p, M, H, W, W_b, C, A_{\max}$ are all introduced here so later sections do not need to re-introduce them.

### Notation summary (introduce here, reused throughout)

| Symbol | Meaning |
|--------|---------|
| $B$ | batch size (queries per call) |
| $N$ | catalogue size (items) |
| $D$ | embedding dimension |
| $K$ | top-K request size |
| $P$ | candidate pool width (sparse path) |
| $W = D / 64$ | int64 word count for 1-bit OPORP |
| $L$ | number of IVF lists (clusters) |
| $L_p$ | n_probe (clusters probed per query) |
| $\bar{|C_l|}$ | average cluster size = $N / L$ |
| $M$ | Bloom filter size in bits; $W_b = M / 64$ words |
| $H$ | number of hash functions in Bloom filter |
| $C$ | number of clauses; $A_{\max}$ = max attrs per clause per item |

---

## 2.2 The LinR family

**Sources:** `articles/linr.md`; `retrieve/src/retrieve/layers/linr/*.py`.

**Source-of-truth note.** Borisyuk et al. 2024 (CIKM, LinkedIn) is the only LinR citation — the LinR paper introduces all three variants together. The reimplementation in this thesis preserves the published behaviour exactly (parity tests in `retrieve/tests/layers/`, see Ch.4 §4.9). The pseudocode below is the published algorithm, not a thesis contribution.

### 2.2.1 LinR V1 — FullScan with post-filter mask

**Source files:** [`retrieve/src/retrieve/layers/linr/postfilter_knn.py:9-59`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) (fp16 variant), [`retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py:28-122`](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) (INT8 variant).

**Algorithm.** Compute the full $B \times N$ similarity matrix $S = Q X^\top$, mask out items failing $\phi$ by overwriting their scores with $-\infty$, take top-K per row.

```
Algorithm 1. LinR V1 — FullScan with post-filter mask

Input:
  Q ∈ R^{B×D}            — query batch
  X ∈ R^{N×D}            — catalogue (fp16 or INT8 codes + scale)
  mask ∈ {0,1}^{B×N}     — predicate φ as boolean mask (optional)
  K                       — top-K size
Output:
  topk_ids ∈ [N]^{B×K}, topk_scores ∈ R^{B×K}

 1: S ← Q · X^⊤                            // fp16 matmul or INT8×INT8 → INT32 (dp4a)
 2: if mask is given:
 3:     S ← masked_fill(S, ¬mask, −∞)      // remove filtered items from the running
 4: (topk_scores, topk_ids) ← top-K(S, dim=1, K)
 5: return (topk_ids, topk_scores)
```

**Implementation notes.** Catalogue is stored pre-transposed as $X^\top \in \mathbb{R}^{D \times N}$, contiguous, in fp16 — this hits the aligned cuBLAS path and fixes accumulator order, eliminating tiebreak jitter on the top-K boundary at the noise floor (see [`postfilter_knn.py:36-42`](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py)). The `backend=` flag exists for API symmetry but is a no-op here — there is no Triton kernel that beats cuBLAS + CUB on this workload (the original `fused_matmul_topk` was removed, see `docs/system/kernels.md`).

**INT8 variant.** Catalogue stored as INT8 codes with a single global scalar scale (per-tensor symmetric, §2.5.1); query is per-batch quantized at forward time; dot product computed in INT32 via dp4a, single dequantization with $s_q \cdot s$. Epilogue includes an arithmetic right-shift by 5 to fit into fp16: at $D \lesssim 2^{14}$ a worst-case INT8 dot is bounded by $D \cdot 127^2 \lesssim 2^{21}$, which does not fit fp16, but shifting by 5 bits brings the accumulator into $2^{16}$ range (the shift is monotone on the positive side, so top-K order is preserved). Padding details: items padded to multiple of 8 for cuBLAS alignment, small batches zero-padded to $M \geq 17$ then sliced post-matmul; quantization happens before padding so the scale is not perturbed.

**Complexity.**

- Time: $O(BND)$ FLOP. Dense matmul on tensor cores is the dominant cost; mask + top-K are negligible.
- Index memory: $O(2ND)$ bytes for fp16, or $O(ND)$ bytes for INT8 (4× reduction).
- Per-call scratch: $O(BN)$ fp16 for the score matrix.

**When to use.** High selectivity ($\rho \gtrsim 0.5$) or no filter; small to medium $N$ where the $BN$ scratch fits in HBM. For large $N$ at low selectivity, V2 or the co-designed retriever wins.

**Framing for the final prose.** Emphasize that V1's strength is its simplicity: it has the lowest constant factor per scored item (one tensor-core FMA) because there is no indirection in the memory access pattern. The mask + masked_fill is essentially a free operation in this regime. The visual ([FIGURE: ...]) should be the simplest of the four flow diagrams.

[FIGURE: LinR V1 dataflow — Q → full matmul Q · X^⊤ → masked_fill (−∞) → top-K → (ids, scores)]

### 2.2.2 LinR V2 — Pre-filter with reduced matmul

**Source file:** [`retrieve/src/retrieve/layers/linr/prefilter_knn.py:10-119`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py).

**Algorithm.** Materialize the predicate as `(candidate_ids: [B, P] int64, counts: [B] int64)`; gather only the passing rows of $X$ and score them.

```
Algorithm 2. LinR V2 — Prefilter with reduced catalogue matrix

Input:
  Q ∈ R^{B×D}                                  — queries
  X ∈ R^{N×D}                                  — catalogue
  candidate_ids ∈ [N]^{B×P}                    — passing-item indices per query
  counts ∈ ℕ^{B}                               — number of valid slots per row
  K
Output:
  topk_ids ∈ [N]^{B×K}, topk_scores ∈ R^{B×K}

 1: for each b ∈ [B] in parallel:
 2:     X_b ← X[candidate_ids[b, :counts[b]]]   // gather [P_b, D]
 3:     s_b ← Q[b] · X_b^⊤                      // batched dot, [P_b]
 4:     slots j ≥ counts[b] masked to −∞
 5: (topk_scores_local, topk_local) ← top-K(s, K)
 6: topk_ids ← candidate_ids.gather(1, topk_local)
 7: return (topk_ids, topk_scores_local)
```

**Two backends.**

- **Torch path** ([`prefilter_knn.py:65-102`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)): fancy-index gather → materialize `[B, P, D]` → `torch.bmm` → top-K → gather. Simple, but `[B, P, D]` lives in HBM.
- **Triton path** ([`prefilter_knn.py:104-119`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py)): fused `fused_masked_knn_topk` kernel — no `[B, P, D]` intermediate; the kernel gathers, scores in fp32 accumulator, and produces top-K in one pass.

**Complexity.**

- Time: $O(B \bar{P} D)$ where $\bar{P}$ is mean per-row pass count (~ $\rho N$).
- Index memory: $O(2ND)$ — V2 does not shrink the index; it only saves FLOPs on the scoring path.
- Extra: $O(BP)$ for the candidate list buffer.

**When to use.** Low selectivity ($\rho \ll 1$) where the gather overhead is amortized by the FLOP savings. There is an empirical cross-over with V1 that depends on hardware — at high pass rates the gather cost dominates and V1 wins. The Triton path widens the cross-over region in V2's favour because it removes the HBM round-trip.

**Framing for the final prose.** The substantive trade-off here is "FLOPs vs memory traffic" — V2 saves on FLOPs but pays in indirection. Make sure the prose calls this out, because the same trade-off recurs in §2.3.4 (the fused probe-filter-score kernel) but under a different guise (avoiding `[B, P, D]` materialization entirely).

[FIGURE: LinR V2 dataflow — Q + filter predicate → (candidate_ids, counts) → gather X[candidate_ids] → bmm → top-K → (ids, scores)]

### 2.2.3 LinR V3 — Sign-OPORP 1-bit + Hamming popcount

**Source file:** [`retrieve/src/retrieve/layers/linr/one_bit_knn.py:34-167`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py).

**Algorithm.** Quantize every embedding to 1 bit per dimension via Sign-OPORP (formally defined in §2.5.2), pack into 64-bit words, score via popcount(XOR).

The connection between Hamming and dot product on $\pm 1$ vectors of length $D$:

$$
\langle u, v \rangle \;=\; D - 2 \, d_H(u, v), \quad u, v \in \{-1, +1\}^D .
$$

Mapping $\{-1 \to 0, +1 \to 1\}$ at the bit-packing step preserves this: Hamming distance on the sign vectors equals popcount of XOR on the packed int64 words. The score becomes

$$
\hat\sigma(q, x) \;=\; D - 2 \cdot \operatorname{popcount}(\tilde q \oplus \tilde x) .
$$

```
Algorithm 3. LinR V3 — Sign-OPORP 1-bit + popcount-XOR

Input:
  q ∈ R^D                                      — query
  q_bits ∈ ℤ_{2^{64}}^{W}                      — packed query bits
  X_bits ∈ ℤ_{2^{64}}^{N×W}                    — packed catalogue
  D, W = D / 64
  K
Output:
  topk_ids ∈ [N]^K, topk_scores ∈ R^K

 0: (signs, perm) ← OPORP_params(seed)         // fixed offline, see §2.5.2
 1: q_bits ← pack_signs((q ⊙ signs)[perm])
 2: for each i ∈ [N] in parallel:
 3:     hamming_i ← Σ_{w=1..W} popcount(q_bits[w] ⊕ X_bits[i, w])
 4:     score_i  ← D − 2 · hamming_i
 5: (topk_scores, topk_ids) ← top-K(score, K)
 6: return (topk_ids, topk_scores)
```

**Two paths.** Full-scan path (above) and `candidate_ids`-driven path (replace the loop over $i$ with a loop over `candidate_ids[b]`); the latter is used in cascades (e.g., V3 as a first-stage filter with fp16/INT8 re-rank on the survivors).

**popcount implementation.** Bit-twiddle: five operations over a 64-bit word using masks $M_1, M_2, M_4, H_{01}$ (see [`quantize.py:7-22`](../../retrieve/src/retrieve/layers/utils/quantize.py)). The Triton-side `_popcount_int64` uses the exact same masks, so the two backends are bit-exact (this is a parity-test invariant, see Ch.4 §4.9).

**Complexity.**

- Time: $O(BNW)$ bit operations. At $W = D/64$ this is asymptotically the same as $O(BND)$, but the constant factor is ~64× smaller (one int64 XOR + popcount replaces 64 fp16 multiplies).
- Index memory: $O(ND/64)$ bits — 16× reduction vs fp16.
- Per-call scratch: $O(BN)$ int32 (Hamming counts) or fused into the kernel.

**When to use.** Memory budget is the binding constraint (very large $N$, modest GPU); or as a first-stage filter in a cascade with a precise re-rank on the top-N candidates. Quality cost relative to fp16 grows mildly with $D$ and depends on the embedding distribution; the empirical numbers are in Ch.6.

**Framing for the final prose.** Two things to emphasize: (i) V3's memory win is real and 16× is a step change, not a marginal improvement — it changes which datasets fit on a single GPU; (ii) the Hamming-to-dot conversion is exact in expectation but noisy on a per-pair basis, and the noise distribution depends on $D$ (concentration improves as $D$ grows). The OPORP construction itself is in §2.5.2; here just state that the projection is deterministic by seed and shared between query and catalogue.

[FIGURE: LinR V3 dataflow — X → OPORP(signs, perm) → pack 64-bit → X_bits; q → same projection → q_bits → popcount(q_bits ⊕ X_bits) → D − 2h → top-K]

### Summary table for LinR variants (inline)

**Table 2.1.** LinR family — complexity, memory, applicability.

| Variant | Score time | Index memory | Key params | When to use |
|---------|------------|--------------|------------|-------------|
| V1 (fp16) | $O(BND)$ | $O(2ND)$ B | $K$ | $\rho \gtrsim 0.5$ or no filter |
| V1 (INT8) | $O(BND)$ via dp4a | $O(ND)$ B | $K$, scale $s$ | Same as V1, memory-constrained |
| V2 | $O(B \bar{P} D)$ | $O(2ND)$ B | $K$, $P$ | $\rho \ll 1$ |
| V3 | $O(BNW)$ bit ops, $W = D/64$ | $O(ND/64)$ B | $K$, seed | Memory priority; $D \geq 64$ |

---

## 2.3 Co-designed IVF + INT8 + Bloom retriever

**Sources:** `articles/silvertorch.md` (internal reference, NOT to be cited or named in body prose — see Citation Policy); [`retrieve/src/retrieve/layers/silvertorch/main.py:27-241`](../../retrieve/src/retrieve/layers/silvertorch/main.py); [`retrieve/src/retrieve/layers/filters/bloom.py`](../../retrieve/src/retrieve/layers/filters/bloom.py); [`retrieve/src/retrieve/layers/utils/kmeans.py`](../../retrieve/src/retrieve/layers/utils/kmeans.py).

**Citation Policy reminder.** Open the section with the lineage statement from `00-thesis-plan.md` Citation Policy → "Lineage / framing statement template" (reproduce verbatim, translated). The body of the section never names the industrial-shorthand "SilverTorch" system; the design is presented as the author's composition of classical primitives, bundled as a second reference retriever in the framework alongside the LinR family. The file-path symbol `silvertorch/main.py` is allowed in code references but never appears in body prose.

**Lineage statement (verbatim, to be reproduced as a blockquote in §2.3 intro):**

> The `retrieve` framework ships two reference retriever families. The LinR family (V1 post-filter masking, V2 pre-filter reduction, V3 1-bit Sign-OPORP) is paper-faithful to Borisyuk et al. (2024, CIKM). The composite IVF + INT8 + Bloom retriever is the author's own design, assembled from classical primitives (Jégou 2011 IVF, INT8 quantization, BitFunnel-style bloom signatures (Goodwin 2017)) under the unified `FilterModule` contract introduced by the framework; it demonstrates that the framework extends naturally beyond the LinR line and accommodates retrievers with different indexing and filtering strategies.

### Motivation (before getting into §2.3.1)

LinR V3 solves the memory problem (16× reduction via 1-bit OPORP) but still scans the entire catalogue at every query — the FLOP count is $O(BNW)$ and at $N \gtrsim 10^9$ (Yambda 5b scale) even bit operations become the bottleneck. LinR V2 reduces the scanned set via the pre-filter pass but materializes a `[B, P]` candidate list in HBM, which grows with selectivity. The co-designed retriever combines three classical primitives to avoid both pathologies simultaneously: IVF clustering shrinks the scanned set, INT8 quantization shrinks the per-item data, and Bloom-signature subset tests are fused into the score kernel so filtered items never trigger an INT8 read. All three are well-known individually; the contribution is the fused GPU kernel that combines them without materializing any intermediate `[B, P, *]` tensor in HBM.

**Frame for the final prose.** The motivation should make clear that this is not a new algorithm in the theoretical sense — every component is decades old (IVF: 2011; Bloom: 1970). The contribution at the algorithm level is the integration under the framework's `FilterModule` contract and the fused kernel that makes the combination GPU-friendly. At the chapter level, the contribution this section carries is to demonstrate that the framework's `FilterModule` and `Backend` contracts accommodate retrievers outside the LinR family — the second bundled retriever exists precisely to make that demonstration concrete. Position it next to LinR V3 specifically (not next to V1/V2), since the natural framing is "what if we needed to push V3 further on large filtered catalogues."

### 2.3.1 IVF clustering and probing

**Source file:** [`silvertorch/main.py:128-220`](../../retrieve/src/retrieve/layers/silvertorch/main.py) (index registration), [`retrieve/src/retrieve/layers/utils/kmeans.py`](../../retrieve/src/retrieve/layers/utils/kmeans.py) (k-means routine).

- Catalogue partitioned into $L$ clusters by $k$-means on the fp32 embeddings. Each cluster $C_l \subseteq [N]$ is the set of items closest to centroid $\mu_l \in \mathbb{R}^D$.
- Initialization: $k$-means++ (Arthur & Vassilvitskii 2007, SODA) — provides a well-conditioned starting point and converges quickly on the empirical distributions encountered in this thesis. Seeded by the module's `seed` parameter for determinism.
- Probing at query time: $\operatorname{top-}L_p \langle q_b, \mu_l \rangle$ — cheap, $O(BLD)$ FLOP, runs on the host side before the fused kernel.
- The visited set per query is $\bigcup_{l \in \text{top-}L_p} C_l$, on average $L_p \cdot N / L$ items.
- Trade-off: $L_p \uparrow$ → recall $\uparrow$, latency $\uparrow$. Typical configurations have $L \in [10^3, 10^4]$ and $L_p \in [16, 128]$ — concrete values from `evaluation/config/*.yaml` go into Ch.5/Ch.6.

**Framing for the final prose.** Cite Jégou et al. 2011 here as the classical IVF reference and Arthur & Vassilvitskii 2007 for k-means++. Note that the IVF idea on GPU is not novel — what is contributed in §2.3.4 is the *fusion* with filtering and INT8 scoring.

### 2.3.2 INT8 global-scale dot product

**Source file:** [`quantize.py:36-55`](../../retrieve/src/retrieve/layers/utils/quantize.py).

- Per-tensor symmetric quantization (formal definition in §2.5.1): single global scalar $s = \max_{i, d} |X_{i, d}| / 127$, codes $\widetilde X_{i, d} = \operatorname{round}(X_{i, d} / s) \in [-128, 127]$.
- Query quantized per-batch on the forward path: $s_{q, b} = \max_d |q_{b, d}| / 127$, $\widetilde q_{b, d} = \operatorname{round}(q_{b, d} / s_{q, b})$.
- Dot product: $\langle q_b, x_i \rangle \approx s_{q, b} \cdot s \cdot \langle \widetilde q_b, \widetilde x_i \rangle_{\mathrm{INT32}}$.
- Hardware path: dp4a (Pascal+, 4× INT8×INT8 → INT32 per instruction) or IMMA tensor cores (Ampere+). Documentation reference: NVIDIA CUDA Programming Guide (manual reference; no academic citation needed). Triton lowers the kernel onto the appropriate path automatically.
- Choice of per-tensor over per-row: avoids gathering a per-item scale at score time, keeping the kernel's memory access pattern regular. Cost: long-tailed row-norm distributions get coarser representations for low-norm rows. Empirical impact measured in Ch.6; the library exposes both schemes (`quantize_int8` for per-row, `quantize_int8_global` for per-tensor) so a future user can swap if needed (see comments in [`silvertorch/main.py:55-62`](../../retrieve/src/retrieve/layers/silvertorch/main.py)).

**Framing for the final prose.** This subsection should explain the per-tensor choice as a deliberate trade-off — not a quality-of-implementation issue. A reader familiar with quantization literature will assume per-row by default; flag the deviation and explain why.

### 2.3.3 Bloom-signature subset test

**Source file:** [`retrieve/src/retrieve/layers/filters/bloom.py:12-119`](../../retrieve/src/retrieve/layers/filters/bloom.py); hash function `_mix64` at [`bloom.py:122-128`](../../retrieve/src/retrieve/layers/filters/bloom.py).

- Per-item signature $\sigma_i \in \{0, 1\}^M$: $M$ bits, $H$ hash functions. For each clause $c$ and attribute value $v$ that item $i$ has, set bits $\sigma_i[h_h(c, v)] := 1$ for $h \in [H]$.
- Hash key is the pair `(clause_idx, value)`, not the raw value — this is what prevents cross-clause collisions (same value appearing in two clauses hashes to different bit sets).
- Hash family: `_mix64` is XORshift–multiply–XORshift with the multipliers forced odd and a fixed salt (`0x515C0DE`). Cheap, GPU-friendly, no division.
- Per-query signature $\sigma_{q, b}$ built the same way from the query's clause attributes.
- Subset test (predicate evaluation): $(\sigma_{q, b} \wedge \sigma_i) = \sigma_{q, b}$, evaluated as a word-wise AND followed by AND-reduce over $W_b = M / 64$ int64 words.
- False-positive rate $\approx (1 - e^{-H K_a / M})^H$ where $K_a$ is the number of distinct (clause, value) keys per item. False-negative rate is zero by construction.
- **Reverse clauses are NOT supported.** Subset tests are asymmetric and cannot express negation; the BloomFilter constructor raises on registration if `clause_is_reverse` is non-empty. This is a hard constraint, not a TODO.
- Forward-index `item_clause_attrs[N, C, A_max]` is stored alongside the signature for the case where exact-clause checks are needed downstream (e.g., to filter out Bloom false positives in a precision-critical sweep).

**Framing for the final prose.** Cite Bloom 1970 (CACM) as the original Bloom filter and Goodwin et al. 2017 (SIGIR, Microsoft, BitFunnel) as the intellectual antecedent for using signature-based subset tests as a retrieval primitive. The contribution here is not the Bloom filter (obviously) but the specialization to recommendation-scale attribute filtering and the fusion with the score kernel — emphasize the fusion as the load-bearing engineering point.

### 2.3.4 Fused probe → filter → score kernel

**Source files:** [`silvertorch/main.py:27-241`](../../retrieve/src/retrieve/layers/silvertorch/main.py) (module); kernels in `retrieve/src/retrieve/kernels/silvertorch/` — `codesigned_probe_score.py`, `codesigned_probe_score_bloom`, `codesigned_probe_score_exact.py`.

The defining feature: phases 2 (filter predicate) and 3 (INT8 score) execute in a single Triton kernel. Phase 1 (probing) runs on the host side because it is small ($O(BLD)$, mostly meant to feed the inner kernel a flat list of candidate item indices).

```
Algorithm 4. Co-designed IVF + INT8 + Bloom retriever (fused kernel)

Input:
  Q ∈ R^{B×D}                                  — queries
  μ ∈ R^{L×D}                                  — IVF centroids
  ~X ∈ ℤ_8^{N×D}                               — INT8 catalogue codes
  s ∈ R                                        — global catalogue scale
  padded_cluster_items, cluster_sizes          — IVF layout
  Σ ∈ ℤ_{2^{64}}^{N×W_b}                       — Bloom signatures of items
  q_clause_attrs ∈ ℤ^{B×C}                     — query clauses
  K, L_p
Output:
  topk_ids ∈ [N]^{B×K}, topk_scores ∈ R^{B×K}

// Phase 1: probing on the host
 1: cent_scores ← Q · μ^⊤                       // [B, L]
 2: top_probes  ← top-K(cent_scores, dim=1, L_p)// [B, L_p]
 3: flat_items  ← padded_cluster_items[top_probes]
                                                // [B, P], P = L_p · max|C_l|

// Phases 2–3 inside one Triton kernel
 4: ~q_b ← quantize_int8_per_row(Q[b])
 5: σ_q,b ← build_query_signature(q_clause_attrs[b])
 6: for each (b, j) ∈ [B] × [P] in parallel:
 7:     i ← flat_items[b, j]
 8:     if i is padding or j ≥ counts[b]:
 9:         continue
10:     // Bloom subset test:
11:     pass ← AND_{w=1..W_b}[(σ_q,b[w] ∧ Σ[i, w]) = σ_q,b[w]]
12:     if ¬pass:
13:         continue
14:     // INT8 score (dp4a / IMMA):
15:     dot ← Σ_{d=1..D} ~q_b[d] · ~X[i, d]     // INT32 accumulator
16:     score_{b, j} ← s_{q, b} · s · dot       // dequant in epilogue
17: (topk_scores, topk_ids) ← top-K per query
18: return (topk_ids, topk_scores)
```

**Key co-design points (these are the load-bearing observations).**

1. **Filter-before-read.** If Bloom rejects an item (step 12), the kernel never reads its INT8 codes from HBM. At low selectivity this saves the dominant memory-bandwidth cost.
2. **No `[B, P, *]` in HBM.** The Bloom mask `[B, P, W_b]`, the gathered codes `[B, P, D]`, and the per-pair scores `[B, P]` all live in shared memory / registers within a single kernel launch. Only top-K results write to HBM.
3. **Regular INT8 access.** Codes are stored contiguous along $D$; dp4a's 4-way packed load matches the data layout; dequant is a single scalar multiply in the epilogue.
4. **Offline autotuning.** Launch parameters (`BLOCK_P`, `num_warps`, `num_stages`) come from `DEFAULT_CONFIG` per kernel, tuned offline via `uv run tune-kernels` (details in Ch.4 §4.6). Not `@triton.autotune` — that interacts badly with `cudagraph_trees` (see Ch.4).

**Exact-mode variant.** Setting `filter="exact"` swaps the Bloom check in steps 5, 10–13 for an exact-clause AND-of-OR predicate against `item_clause_attrs[N, C, A_max]`. Same kernel structure, different predicate. The Bloom variant has lower bandwidth per item ($W_b$ words vs $C \cdot A_{\max}$ ints) but admits false positives; exact has zero false positives at higher bandwidth cost. Bench: `_agent_scratch/bench_results/silvertorch_int8mm_quality.json` (not for direct citation; for the writer's awareness).

**No-filter variant.** `filter="none"`: plain IVF + INT8 ANN, no predicate. Used as the baseline within this family in Ch.6.

**Complexity.**

- Phase 1: $O(BLD)$ FLOP (host).
- Phases 2–3: $O(B \cdot L_p \cdot \bar{|C_l|} \cdot (W_b + D/4))$ operations per kernel invocation. The $D/4$ factor reflects 4-way INT8 packing through dp4a. $W_b$ is the Bloom check cost per item.
- Index memory: $O(ND/4)$ B for INT8 codes + $O(N W_b)$ B for Bloom signatures (first term usually dominates).
- Per-call scratch: top-K accumulator $O(BK)$ in registers + small.

**When to use.** Large catalogues ($N \gtrsim 10^6$), arbitrary selectivity, especially when the filter has many clauses or rich combinatorial structure that makes a dense `[B, N]` mask impractical. This is the only one of the four algorithms in this chapter that maintains sub-linear-in-$N$ scoring cost in the average case.

**Framing for the final prose.** Two emphases. First, the *fusion* is the engineering contribution; the components themselves are well-known. Second, the kernel design follows from the GPU memory hierarchy — once you decide "do not materialize `[B, P, D]` in HBM" the rest of the structure is mostly forced (filter goes inside the kernel; INT8 access has to be the inner loop because that is where the bandwidth is; top-K accumulator stays in registers because writing it out negates the savings). Walk the reader through that derivation; do not just present the kernel as a black box.

[FIGURE: Co-designed dataflow — query → IVF probe (L_p clusters) → bloom subset-test inside the kernel → INT8 dp4a score for survivors → top-K → (ids, scores); annotate which arrows stay in registers/SMEM vs which write to HBM]

---

## 2.4 Filter primitives

**Sources:** `docs/system/filtering.md`; [`retrieve/src/retrieve/interfaces.py`](../../retrieve/src/retrieve/interfaces.py); [`retrieve/src/retrieve/layers/filters/`](../../retrieve/src/retrieve/layers/filters/).

### 2.4.0 The `FilterModule` contract (introduce here, reuse throughout)

All filter primitives implement a single abstract base class `FilterModule` (defined in [`interfaces.py`](../../retrieve/src/retrieve/interfaces.py)) with three evaluation methods. Each method serves a different downstream consumer:

| Method | Output shape | Consumed by |
|--------|--------------|-------------|
| `evaluate_mask(query)` | `[B, N]` bool | V1 (post-filter mask) |
| `evaluate_indices(query)` | `([B, P] int64, [B] int64)` — `(candidate_ids, counts)` | V2 (pre-filter); first stage of cascade |
| `evaluate_subset(query, candidate_ids)` | `[B, P]` bool | Subsequent stages of cascade |

The point of supporting all three is that the cost of materializing each intermediate (dense mask vs sparse candidates vs subset mask) varies wildly with $N$ and $\rho$; the consumer picks. The contract is the unifying interface for §2.4.3's composition operators.

### 2.4.1 ExactAttributeFilter

**Source file:** [`exact_attribute.py:12-107`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py).

**Data layout.**

- `item_clause_attrs: [N, C, A_max] int64` — for each item, $C$ clauses, up to $A_{\max}$ allowed values per clause; unused slots filled with $-1$.
- `query_clause_attrs: [B, C] int64` — one value per clause per query; $-1$ means inactive (clause is auto-satisfied).
- `clause_is_reverse: [C] bool` — per-clause NOT flag.

**Semantics.**

$$
\mathrm{clause\_pass}(b, n, c) = \bigvee_{a \in [A_{\max}]} \big[ \mathrm{item}[n, c, a] = \mathrm{query}[b, c] \big]
$$

(OR over allowed values within a clause). If `clause_is_reverse[c]` is set, the result is inverted. Inactive clauses (query value $= -1$) pass unconditionally. Item passes overall iff all clauses pass:

$$
\phi_b(n) = \bigwedge_{c \in [C]} \mathrm{clause\_pass}(b, n, c) .
$$

**Backends.**

- Torch: materialize `[B, N, C, A_max]` comparison tensor, any along $a$, AND along $c$.
- Triton (`clause_mask` / `clause_compact` kernels): fused, keeps the four-axis intermediate in registers; produces the `[B, N]` mask or `(candidate_ids, counts)` pair directly.

**Complexity.**

- Time: $O(B N C A_{\max})$ dense, $O(B P C A_{\max})$ on `evaluate_subset`.
- Index memory: $O(N C A_{\max})$.

**Note for prose.** ExactAttributeFilter is the only filter that supports reverse clauses; this is its main differentiator from Bloom. It is also the only one with zero false positives — no probability of incorrectly admitting an item — at the cost of higher per-item bandwidth on the predicate check.

### 2.4.2 BloomFilter

**Source file:** [`bloom.py:12-119`](../../retrieve/src/retrieve/layers/filters/bloom.py).

Formal definition is in §2.3.3 (it is the same primitive — described there in the context of the co-designed kernel). Recap as a standalone module:

- Constructor params: $M$ (signature bits), $H$ (hash count), seed.
- `register_index(item_clause_attrs, ...)`: build per-item signatures into `bloom_sigs: [N, W_b]` int64.
- `evaluate_mask(query)`: compute query signature, word-wise AND with item sigs, equality check with query sig, AND-reduce over words.
- `evaluate_indices` / `evaluate_subset`: same predicate, different output materialization. Triton backend uses `bloom_compact` to avoid `[B, N]` masks.
- Hard constraint: raises on registration if `clause_is_reverse` is non-empty. Document this prominently — it is the principal API limitation of this filter.

**Complexity.**

- Time: $O(B N W_b)$ dense, $O(B P W_b)$ on subset.
- Index memory: $O(N W_b) = O(N M / 64)$ bits.

**Note for prose.** Bloom's appeal in this thesis is twofold: (i) constant per-item bandwidth in the predicate check regardless of how many clause values the item has, and (ii) GPU-friendly word-aligned AND-reduce. Calibrating $M$ and $H$ for a target false-positive rate is the only tuning knob; the false-positives are tolerated because the downstream score-then-top-K step naturally filters out items with too-low scores. If precision must be exact, an exact-clause check on the returned top-K candidates restores zero false positives.

### 2.4.3 Composition: `combine_masks` vs `combine_indices`

**Source file:** [`filters/__init__.py:13-70`](../../retrieve/src/retrieve/layers/filters/__init__.py).

When a query carries multiple independent predicates (e.g., language + tag-set), evaluate each separately and combine. Two strategies:

**`combine_masks` (dense AND).** Each filter returns a `[B, N]` mask via `evaluate_mask`; combine pointwise:

$$
m_{\text{joint}}(b, n) = \bigwedge_k m^{(k)}(b, n) .
$$

- Time: $O(BN \sum_k T_k)$.
- Memory: $O(BN)$ per mask in flight.
- Natural consumer: V1.

**`combine_indices` (sparse cascade).** First filter via `evaluate_indices` → `(ids, counts)`. Subsequent filters via `evaluate_subset` on the running candidate list. After each, re-compact (sort valid entries to front, update counts).

- Time: $O(B(T_1 + \sum_{k \geq 2} P_{k-1} T_k))$. If filter 1 is selective, $P_{k-1}$ shrinks fast and the cascade is dominated by $T_1$.
- Memory: $O(BP)$ for the running candidate list; no `[B, N]` ever materialized.
- Natural consumer: V2 and the co-designed retriever.

**Ordering heuristic.** Most-selective-first — minimizes $P_k$ at each subsequent step. Current implementation requires the caller to pick the order; there is no automatic planner based on per-filter selectivity estimates. (Worth a one-sentence "future work" note here, but not a TODO.)

**Note for prose.** This subsection is short but load-bearing for §2.3.4: the co-designed retriever consumes `combine_indices` output as its starting candidate list, and the cascade's `evaluate_subset` interface is what makes the fused kernel's filter phase modular (Bloom or exact, swappable). Connect the abstraction back to the algorithm so the reader sees why the contract has three methods and not two.

---

## 2.5 Quantization schemes

**Source file:** [`quantize.py`](../../retrieve/src/retrieve/layers/utils/quantize.py).

Two quantization schemes are used in this thesis: per-tensor symmetric INT8 (used in V1-INT8 and the co-designed retriever) and Sign-OPORP 1-bit (used in V3). Both are standard; the formal definitions are given here so subsequent sections (Ch.4 implementation, Ch.6 quality discussion) can refer back without re-introducing the math.

### 2.5.1 Per-tensor symmetric INT8

**Source:** [`quantize.py:36-55`](../../retrieve/src/retrieve/layers/utils/quantize.py) (`quantize_int8_global`). Per-row variant `quantize_int8` at [`quantize.py:25-33`](../../retrieve/src/retrieve/layers/utils/quantize.py) is mentioned as an alternative but not used in the deployed retrievers.

**Definition.** Global scale and per-element codes:

$$
s = \frac{1}{127} \max_{i, d} |X_{i, d}|, \qquad \widetilde X_{i, d} = \mathrm{clamp}\!\Big( \operatorname{round}\big( X_{i, d} / s \big), -128, 127 \Big) \in \mathbb{Z}_8 .
$$

**Reconstruction.** $\widehat X_{i, d} = s \cdot \widetilde X_{i, d}$; element-wise error $|\widehat X_{i, d} - X_{i, d}| \leq s / 2$ (rounding error; no clipping by construction of $s$).

**Dot product reconstruction.** With per-query scale $s_{q, b}$:

$$
\langle q_b, x_i \rangle \approx s_{q, b} \cdot s \cdot \langle \widetilde q_b, \widetilde x_i \rangle_{\mathrm{INT32}} .
$$

The INT32 accumulator is computed on hardware via dp4a (4× INT8 MAC per cycle) or IMMA (Ampere+ tensor cores). The dequantization is one scalar multiply in the kernel's epilogue, which does not break access regularity.

**Why per-tensor and not per-row.** Per-row would gather a different scale per item, which breaks the regular access pattern and adds an extra HBM read per scored item. Per-tensor pays a quality cost on long-tailed row norms (low-norm rows get coarser representation) for the bandwidth win. The library exposes both via `quantize_int8` and `quantize_int8_global`; the deployed retrievers use the latter.

**Note for prose.** A reader who knows quantization will assume per-row by default. Flag the per-tensor choice as an explicit decision, with the bandwidth motivation. Reserve quality-cost numbers for Ch.6.

### 2.5.2 Sign-OPORP 1-bit

**Source:** [`quantize.py:58-128`](../../retrieve/src/retrieve/layers/utils/quantize.py); construction at `:58-63`, packing at `:66-78`, index quantization at `:81-108`, query projection at `:111-128`.

**Citation.** Li & Li 2023 (arXiv:2302.03505, "OPORP: One Permutation + One Random Projection").

**Definition.** Two seeded random objects shared between catalogue and query:

- Sign vector $s \in \{-1, +1\}^D$ (Rademacher, sampled per-coord uniform).
- Permutation $\pi: [D] \to [D]$ (uniform random).

Sign-OPORP projection of a vector $x \in \mathbb{R}^D$:

$$
\tilde x = \operatorname{sign}\!\big( (x \odot s)[\pi] \big) \in \{-1, +1\}^D .
$$

Bit-packing: bit $b$ of word $w$ is set iff $\tilde x_{64 w + b} > 0$. Result: $\widetilde X^{\mathrm{pack}} \in (\mathbb{Z}_{2^{64}})^{N \times W}$ where $W = D / 64$. Total memory: $D / 64$ int64 words per item, i.e. 1 bit per dim, i.e. 16× reduction vs fp16.

**Determinism.** $(s, \pi)$ generated by `torch.Generator(device).manual_seed(seed)`. The same `(s, \pi)` must be used for catalogue (at `OneBitKNN.register_index`) and queries (at forward time via `project_oporp_1bit_query`); the module persists the buffers as part of its state.

**Why this works for retrieval.** For random Rademacher $s$ and random $\pi$, the Hamming distance $d_H(\tilde q, \tilde x)$ concentrates around a value that is monotone decreasing in $\cos(q, x)$ — so pairwise ranking by dot product is preserved in expectation. Full analysis in Li & Li 2023; for our purposes it is enough to state the property. The variance shrinks as $D$ grows, so larger embedding dimensions give tighter rank preservation.

**Scoring rule.** From §2.2.3: $\hat\sigma(q, x) = D - 2 \cdot \operatorname{popcount}(\tilde q \oplus \tilde x)$, which is the dot product of $\{-1, +1\}$ vectors expressed via packed-bit Hamming weight.

**popcount.** Both Torch and Triton backends use the same five-operation bit-twiddle (masks $M_1, M_2, M_4, H_{01}$ — see [`quantize.py:7-22`](../../retrieve/src/retrieve/layers/utils/quantize.py)). This is what makes the two backends bit-exact on V3.

**Note for prose.** OPORP itself is a published technique — the contribution in this thesis is not the projection but its reproduction in the open-source Triton path. Cite Li & Li 2023 here. Make sure the prose distinguishes between the random-projection step (which is the OPORP contribution) and the sign-quantization step (which is the cheap step that makes the whole thing 1-bit and packable).

---

## Summary table for all retrieval algorithms (inline)

**Table 2.2.** All retrieval algorithms — time, memory, parameters, applicability.

| Algorithm | Score time | Index memory | Key params | When to use |
|-----------|------------|--------------|------------|-------------|
| LinR V1 (fp16) | $O(BND)$ | $O(2ND)$ B | $K$ | $\rho \gtrsim 0.5$ or no filter; small-mid $N$ |
| LinR V1 (INT8) | $O(BND)$ via dp4a | $O(ND)$ B | $K$, scale $s$ | Same regime, memory-constrained |
| LinR V2 | $O(B \bar{P} D)$ | $O(2ND)$ B | $K$, $P$ | $\rho \ll 1$ |
| LinR V3 (1-bit OPORP) | $O(BNW)$ bit ops, $W = D/64$ | $O(ND/64)$ B | $K$, seed | Memory priority; $D \geq 64$ |
| IVF + INT8 + Bloom | $O(B L_p \bar{|C_l|} (W_b + D/4))$ | $O(ND/4 + N W_b)$ B | $L$, $L_p$, $M$, $H$ | Large $N$ at arbitrary $\rho$ |

No algorithm dominates across the $(N, D, \rho)$ space — Ch.6 maps the regions of optimality empirically.

---

## Chapter summary (notes for the closing paragraph)

This chapter has formalized five retrieval algorithms bundled in `torchretrieve`. The LinR family (V1, V2, V3) reproduces Borisyuk et al. 2024 (CIKM) — three variants spanning the post-filter / pre-filter / quantization axis — as the framework's first reference retriever family. The co-designed IVF + INT8 + Bloom retriever (§2.3) is the framework's second bundled retriever, composing three classical primitives (Jégou 2011 IVF, INT8 quantization with dp4a, Bloom 1970 / Goodwin 2017 BitFunnel-style signatures) under the unified `FilterModule` contract and the fused probe-filter-score kernel; it demonstrates that the framework's contracts accommodate retrievers outside the LinR family. §2.4 made the filter contract explicit (three evaluation modes, two composition operators); §2.5 gave the formal quantization definitions used throughout.

All algorithms are tested for parity between Torch and Triton backends (Ch.4 §4.9) and evaluated on Goodreads, arXiv, and Yambda (Ch.3, Ch.6). The empirical study in Ch.6 maps where each algorithm dominates and where it loses, and the limitations section (Ch.7) discusses what is left open — most notably, no multi-GPU sharding, no live index updates, and no direct head-to-head with hand-tuned CUDA implementations of the same algorithms.

---

## Citation checklist for the Russian prose pass

When converting this document to Russian academic prose, ensure:

1. **Lineage statement** appears verbatim (translated) at the start of §2.3.
2. **Only these citations appear** in the body prose:
   - Borisyuk et al. 2024 (CIKM, LinkedIn) — LinR
   - Jégou, Douze & Schmid 2011 (TPAMI) — IVF/PQ (Jégou was at INRIA; the paper predates Meta affiliation per `00-thesis-plan.md` §1.3)
   - Bloom 1970 (CACM) — Bloom filters
   - Goodwin et al. 2017 (SIGIR, Microsoft) — BitFunnel
   - Li & Li 2023 (arXiv:2302.03505) — OPORP
   - Arthur & Vassilvitskii 2007 (SODA) — k-means++
   - NVIDIA CUDA Programming Guide — dp4a (manual reference)
3. **Forbidden in body prose:**
   - The word "SilverTorch" / "силвер торч" anywhere (allowed only in markdown code-ref links like `silvertorch/main.py`).
   - Any reference to Meta / Facebook / FAIR / Instagram / WhatsApp papers (the audit table in `00-thesis-plan.md` lists the specific dropped citations).
   - FAISS or FAISS-GPU.
4. **Render the lineage statement as a blockquote** in the Russian prose, since it is a structurally important paragraph the reader should recognize on sight.

## Open notes / for the writer to decide

- Concrete $L_p$ and $L$ defaults: not stated here because they vary per dataset/dim — pull from `evaluation/config/*.yaml` if the prose needs a concrete example; otherwise reference Ch.6 for the empirical sweep.
- Exact cross-over point between V1 and V2 as a function of $\rho$: empirical, varies per hardware — leave to Ch.6, do not assert a number in Ch.2.
- INT8 quality cost relative to fp16: empirical, in Ch.6.
- Bloom $M$, $H$ defaults used in evaluation: pull from `evaluation/config/*.yaml` if naming a concrete value; otherwise leave abstract.

The chapter should not assert any of these numbers; it states the algorithms and the trade-offs. Empirical quantification is Ch.6's job.
