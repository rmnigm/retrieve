# OPORP fix + SimHash addition

> Companion to [00-roadmap.md](00-roadmap.md). Standalone correctness + new-feature work; orthogonal to live-update and torch-export refactors.

## A. Summary

**Change 1.** [`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py#L81) skips the bin-sum step that defines Sign-OPORP. At the lib's only operating point (k_bits = D), the Rademacher `signs[D]` cancels in XOR and `perm[D]` is a bijection on the popcount axis — the 1-bit code is algebraically `sign(x)` regardless of seed. Empirical confirmation in [experiments/oporp_quality/README.md](../../experiments/oporp_quality/README.md). Fix: add a `k_bits` parameter (default = D, byte-identical to today) and insert the paper's full sketch — `.view(N, k_bits, D//k_bits).sum(-1)` then per-row L2 normalize — between `index_select` and `_pack_signs_to_int64`. Output bits are scale-invariant (`sign(x/|x|) = sign(x)`), so the L2 step is paper-faithful **and** leaves today's bits unchanged at k_bits = D. Exposes the bit-budget knob LiNR §5.6 uses (512-bit codes at D=128).

**Change 2.** Add `SimHashKNN`, parallel to `OneBitKNN`. Each output bit = `sign(r_j · x)` for independent `r_j ∈ ℝ^D` — mixes coordinates and admits `k_bits > D`, where the table in `experiments/oporp_quality/README.md` shows 2.6–3.0× recall@10 lift at 4–8× memory. Reuses the existing Triton kernel unchanged: it is shape-generic on `W = k_bits / 64`.

## B. Change 1: OPORP fix

### B1. [retrieve/src/retrieve/layers/utils/quantize.py:81-128](../../retrieve/src/retrieve/layers/utils/quantize.py#L81)

```python
def quantize_oporp_1bit(
    embs: Tensor,
    seed: int = 0,
    k_bits: int | None = None,                                 # NEW
) -> tuple[Tensor, Tensor, Tensor]:
    if embs.dim() != 2: raise ValueError(...)
    n, d = embs.shape
    if k_bits is None: k_bits = d                              # NEW (default = D → identity bin-sum)
    if d % k_bits != 0:
        raise ValueError(f"k_bits must divide D; got k_bits={k_bits}, D={d}")
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    b = d // k_bits                                            # NEW
    signs, perm = _build_oporp(d, seed, embs.device)
    proj = (embs * signs.to(embs.dtype)).index_select(1, perm)
    binned = proj.view(n, k_bits, b).sum(dim=-1)               # NEW bin-sum (identity at b=1)
    sketch = binned / binned.norm(dim=-1, keepdim=True).clamp_min(1e-8)  # NEW L2 (paper-faithful; bit-invariant)
    return _pack_signs_to_int64(sketch), signs, perm
```

At `b=1`, `view(n, d, 1).sum(-1)` is the identity reshape. The L2 normalize divides each row by a positive scalar — `sign(x/|x|) = sign(x)` per element — so output bits are bit-for-bit equal to today's both with and without it at any k_bits. `clamp_min(1e-8)` guards the all-zero row edge (matches the int8 path's [line 30](../../retrieve/src/retrieve/layers/utils/quantize.py#L30) idiom).

`project_oporp_1bit_query` ([line 111](../../retrieve/src/retrieve/layers/utils/quantize.py#L111)): add `k_bits: int | None = None` and mirror both new steps (bin-sum + L2). `k_bits` must be plumbed from the caller — `signs`/`perm` are both `[D]` and don't encode it.

### B2. [retrieve/src/retrieve/layers/linr/one_bit_knn.py:64-91](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L64)

```python
def __init__(self, k: int, seed: int = 0, backend: Backend = "triton",
             k_bits: int | None = None) -> None:              # NEW
    super().__init__()
    self.k, self.seed, self.backend = k, seed, backend
    self.k_bits = k_bits                                       # NEW; resolved in register_index

def register_index(self, item_embs: Tensor) -> None:
    n = item_embs.shape[0]
    if self.k > n: raise ValueError(...)
    bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed, k_bits=self.k_bits)
    self.k_bits = self.k_bits or item_embs.shape[1]            # NEW; materialize for _project_query
    self.register_buffer("item_bits", bits)
    self.register_buffer("oporp_signs", signs)
    self.register_buffer("oporp_perm", perm)

def _project_query(self, query: Tensor) -> Tensor:
    return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm, self.k_bits)  # NEW
```

`self.k_bits` is a Python int — constant from Dynamo's POV, no extra recompiles. `d_total` property is unchanged: `64 * item_bits.shape[1]` returns the post-binning width naturally.

## C. Change 2: SimHash addition

### C1. [retrieve/src/retrieve/layers/utils/quantize.py](../../retrieve/src/retrieve/layers/utils/quantize.py) — append

```python
def _build_simhash_R(d: int, k_bits: int, seed: int, device) -> Tensor:
    g = torch.Generator(device=device).manual_seed(int(seed))
    return torch.randn(k_bits, d, generator=g, device=device)

def quantize_simhash_1bit(
    embs: Tensor, k_bits: int, seed: int = 0,
) -> tuple[Tensor, Tensor]:
    """SimHash 1-bit quantization (Charikar 2002 / Manku 2007).

    Each output bit = sign(r_j · x) for r_j ~ N(0, I_D). Cost: one fp32
    matmul + pack. Unlike Sign-OPORP at k=D, admits k_bits > D for
    recall-vs-memory trades the OPORP family cannot reach.

    Returns (bits[N, k_bits//64] int64, R[k_bits, D] fp32).
    """
    if embs.dim() != 2: raise ValueError(...)
    if k_bits % 64 != 0:
        raise ValueError(f"k_bits must be a multiple of 64, got {k_bits}")
    r = _build_simhash_R(embs.shape[1], k_bits, seed, embs.device)
    return _pack_signs_to_int64(embs @ r.t()), r

def project_simhash_1bit_query(query: Tensor, r: Tensor) -> Tensor:
    if query.dim() != 2: raise ValueError(...)
    return _pack_signs_to_int64(query @ r.t())
```

### C2. [retrieve/src/retrieve/layers/linr/simhash_knn.py](../../retrieve/src/retrieve/layers/linr/simhash_knn.py) — NEW

Sketch (full body mirrors [`OneBitKNN`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L34) line-for-line; split point is `_project_query`):

```python
class SimHashKNN(nn.Module):
    """SimHash 1-bit Hamming scoring, selectable backend.

    Item embeddings are projected via a fixed Gaussian R ∈ R^{k_bits × D}
    and sign-quantized. Score = D_TOTAL - 2 * popcount(query_bits ^
    item_bits), D_TOTAL = k_bits (NOT input D). Unlike OneBitKNN, k_bits
    can exceed D for a recall-vs-memory trade — see
    experiments/oporp_quality/README.md (2.6-3.0× recall@10 lift at 4-8×
    memory on anisotropic data).

    Reuses the OPORP Triton kernel — kernel reads only W =
    item_bits.shape[1]; the algorithm that produced the bits is opaque.
    """
    item_bits: Tensor
    simhash_R: Tensor

    def __init__(self, k: int, k_bits: int, seed: int = 0, backend: Backend = "triton"):
        super().__init__()
        self.k, self.k_bits, self.seed, self.backend = k, k_bits, seed, backend

    def register_index(self, item_embs: Tensor) -> None:
        if self.k > item_embs.shape[0]: raise ValueError(...)
        bits, r = quantize_simhash_1bit(item_embs, self.k_bits, self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("simhash_R", r)

    @property
    def d_total(self) -> int: return 64 * self.item_bits.shape[1]

    def _project_query(self, query: Tensor) -> Tensor:
        return project_simhash_1bit_query(query, self.simhash_R)

    def forward(self, query, candidate_ids=None, counts=None):
        # body byte-for-byte from OneBitKNN.forward / _forward_torch_eager /
        # _forward_triton (lines 93-166). Duplicate rather than refactor —
        # cudagraph captures benefit from a flat method; the ~70 lines
        # duplicated are cheaper than a base class that risks an extra
        # super() call inside the compiled region.
        ...
```

### C3. [retrieve/src/retrieve/layers/linr/__init__.py](../../retrieve/src/retrieve/layers/linr/__init__.py)

Add `from retrieve.layers.linr.simhash_knn import SimHashKNN`; append to `__all__`.

### C4. [retrieve/src/retrieve/__init__.py](../../retrieve/src/retrieve/__init__.py)

Add `SimHashKNN` to the imports + `__all__` (alongside `OneBitKNN`). Also export `quantize_simhash_1bit` next to `quantize_oporp_1bit` (line 17).

### C5. Kernel reuse trace — confirms no kernel change needed

[oporp_1bit_match_topk.py:55-124](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L55):
- Line 64: `D_TOTAL: tl.constexpr` — passed by wrapper, only place output-bit-count enters.
- Lines 82, 113: `w_off = tl.arange(0, W)`; `xor_words = qb[None, :] ^ item_rows` — W is the only output-bit-count input the kernel body sees.
- Wrappers ([full:280](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L280), [indirect:344](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L344)) derive `d_total = 64 * w` from `w = query_bits.shape[1]`.

`D_TOTAL = 64 * W` is the **only** place output-bit-count enters; bit-words width tracks `query_bits.shape[1]` and `item_bits.shape[1]` naturally. SimHash at k_bits=512 → W=8; OPORP at k_bits=64 with D=128 → W=1; same kernel both cases. No kernel edits.

## D. Tests

### [retrieve/tests/correctness/test_quantize.py](../../retrieve/tests/correctness/test_quantize.py)

- **Regression** `test_oporp_default_kbits_matches_today`: capture today's `quantize_oporp_1bit(embs, seed=0)` bits in a fixture (or run alongside a frozen reference call into a local copy of the old impl); after the change, `torch.equal` the new default output.
- **NEW** `test_oporp_kbits_lt_d_matches_torch_reference`: k_bits=64, D=128 — pure-torch reference computes the full paper pipeline `(x*signs)[perm].view(N, 64, 2).sum(-1)`, L2-normalize, `> 0`, pack; assert bit-equal to lib output (also covers query path).
- **NEW** `test_oporp_l2_normalize_is_bit_invariant`: at k_bits ∈ {D, D/2}, the lib output equals a reference that drops the L2 step. Guards the sign-scale-invariance claim and pins down the "paper-faithful + bit-stable" property.
- **NEW** `test_oporp_kbits_validation`: raises on `d % k_bits != 0` and `k_bits % 64 != 0`.
- **NEW** `test_simhash_shapes_and_dtypes`: k_bits=256, D=128 → `bits.shape=(N, 4)` int64, `R.shape=(256, 128)` fp32.
- **NEW** `test_simhash_query_consistency`: parallels [test_oporp_query_consistency:51](../../retrieve/tests/correctness/test_quantize.py#L51) — `project_simhash_1bit_query(embs, R)` equals `bits`.
- **NEW** `test_simhash_dot_product_proxy`: analog of [test_oporp_dot_product_proxy:65](../../retrieve/tests/correctness/test_quantize.py#L65) — per-row Pearson > 0.5 at k_bits = 4D on isotropic data.
- **NEW** `test_simhash_kbits_validation`: raises on `k_bits % 64 != 0`.

### [retrieve/tests/parity/test_oporp_1bit_match_topk.py](../../retrieve/tests/parity/test_oporp_1bit_match_topk.py)

No new file. Add `@pytest.mark.parametrize("quant", ["oporp", "simhash"])` to `test_oporp_1bit_full_matches_torch` and `test_oporp_1bit_indices_matches_torch`, branching at fixture build to use `quantize_simhash_1bit` / `project_simhash_1bit_query` with k_bits=128. Justification: the kernel input is `[N, W]` int64; the algorithm that produced the bits is opaque to it. One added parametrize covers both quantizers cleanly.

### [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py)

- **Regression**: existing `TestOneBitKNN` class (line 165) runs unchanged with default `k_bits=None`; the `recall ≥ 0.4` floor at K=200 (line 176) and the cross-backend agreement test (line 127) stay unchanged.
- **NEW** `TestSimHashKNN` (parametrized on backend and k_bits ∈ {D, 4*D}): full-scan recall vs `FullScanKNN` exact; candidate_ids subset; `test_torch_compile_fullgraph_no_break` (parallels [line 191](../../retrieve/tests/correctness/test_linr.py#L191)). At k_bits=4*D expect recall > 0.6 on this synthetic fixture.
- **NEW** `test_one_bit_knn_kbits_lt_d_cross_backend`: `OneBitKNN(k=K, k_bits=64)` torch and triton backends produce identical top-K (extends the existing [test_one_bit_torch_matches_one_bit_triton:127](../../retrieve/tests/correctness/test_linr.py#L127) coverage).

## E. Compatibility & risk

1. **torch.compile / cudagraph_trees**. `@triton_op` wrapper signatures unchanged — `oporp_1bit_match_topk_full(query_bits, item_bits, k)` and `_indirect(..., positive_indices, counts)`. Only the W dim of input tensors varies. `self.k_bits` is a Python int (constant under Dynamo), not a tensor. SimHash's `simhash_R` is set once in `register_index` and stored as `register_buffer` — stable address, no reallocation in forward. `query @ R.t()` is a standard matmul Inductor traces cleanly under `dynamic=True`.
2. **Layer contract**. `forward(query, candidate_ids=None, counts=None) -> (ids, scores)` preserved on `OneBitKNN`; `SimHashKNN` adopts the same signature. `k_bits` lives in `__init__`. [live-update-api.md §B3 (line 85)](live-update-api.md) treats `oporp_signs` / `oporp_perm` as frozen-at-register — `simhash_R` is the same shape contract and drops in identically.
3. **Kernel contract**. Traced in §C5. The kernel reads `D_TOTAL = 64 * W` from a single source ([line 64, 280, 344](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L64)).
4. **Backward compat**. `OneBitKNN(k=K)` with no `k_bits` → `register_index` resolves `k_bits = D` → `view(N, D, 1).sum(-1)` is identity → today's bits. [evaluation/retrieval/algos/linr_v3.py:59](../../evaluation/retrieval/algos/linr_v3.py#L59) and every test call site are unchanged.
5. **Tests**. See §D — every existing assertion preserved; new tests are additive.
6. **Plan-doc layout**. File at `docs/plans/oporp-fix-and-simhash.md`, follows [03-kernel-optimizations.md](03-kernel-optimizations.md) tone.

**Cudagraph capture, new tensors**: `OneBitKNN` registers no new buffers (`item_bits` / `oporp_signs` / `oporp_perm` unchanged, just shorter in the k_bits<D case). `SimHashKNN` registers `item_bits[N, k_bits//64]` and `simhash_R[k_bits, D]` once in `register_index`; lifetimes match the module. No forward-side allocations beyond the standard matmul output → pack → topk chain.

## F. Validation

```bash
# 1. Correctness + parity.
uv run pytest retrieve/tests/correctness/test_quantize.py -v
uv run pytest retrieve/tests/parity/test_oporp_1bit_match_topk.py -v
uv run pytest retrieve/tests/correctness/test_linr.py -v
# Expected: green, including new SimHash and k_bits<D tests.

# 2. Diagnostic re-run with lib impl substituted.
# Edit experiments/oporp_quality/compare.py: swap the imports from
# oporp_impls.{quantize_correct, project_correct, quantize_simhash,
# project_simhash} to retrieve.layers.utils.quantize.{quantize_oporp_1bit,
# project_oporp_1bit_query, quantize_simhash_1bit,
# project_simhash_1bit_query}. Then:
cd experiments/oporp_quality
python compare.py synthetic --d 128 --n-items 10000 --n-queries 500 \
    --seeds 0 1 2 --device cuda --out /tmp/lib_impl.json
# Compare cell-by-cell against committed results/synth_anisotropic.json.
# Expect exact match on correct (k_bits=128) and seed-invariant zero-var
# on the k_bits=D row; SimHash within MC noise.

# 3. Eval-harness pass on real data, confirms the k_bits knob moves recall.
cd evaluation
uv run evaluate --config conf/<dataset>.yaml --algorithms linr_v3
# Baseline (k_bits=None → D).
uv run evaluate --config conf/<dataset>.yaml --algorithms linr_v3 \
    --linr_v3.k_bits=$((D/2))
# Expect recall@10 drop matching the table (e.g. D=128 → k_bits=64 →
# recall@10 drops from ~0.20 to ~0.08 on anisotropic synthetic).
```

## G. Open questions

1. **`Backend` enum value vs separate `SimHashKNN` layer** — **recommend separate layer**. `Backend = Literal["torch", "triton"]` ([interfaces.py:8](../../retrieve/src/retrieve/interfaces.py#L8)) is the execution backend, orthogonal to quantizer choice; conflating forces `OneBitKNN` to dispatch on algorithm at every method and either compute both projections at register time or branch in the cudagraph-captured forward. Two small layers (~70 lines duplicated) keep `register_index` / `_project_query` linear.

2. **Drop dead `signs`/`perm` buffers at k_bits = D** — **recommend keep**. They are O(D), a few hundred bytes — negligible vs `item_bits[N, W]` int64. Removing branches `register_index`, breaks the V3 live-update upsert path ([live-update-api.md:91](live-update-api.md) reuses them on the `[K, D]` slice), and forks the projection helper. Minimize the diff.

3. **L2-normalize bin-sum before sign** — **recommend include** (paper-correct). Sign-quantization is scale-invariant, so the L2 step does not change the output bits (`sign(x/|x|) = sign(x)`); a `test_oporp_l2_normalize_is_bit_invariant` test pins this down. Cost is a per-row norm + divide — `O(N · k_bits)` once at register_index, `O(B · k_bits)` per forward call — trivial vs the matmul/popcount work. Including it makes the implementation match the paper algorithm letter-for-letter and avoids a footgun if the lib later exposes the real-valued sketch.
