# `retrieve` kernels + layers design improvements

> **Status: IMPLEMENTED (2026-07-06); library test gates passed on A100 2026-09-02, harness gates pending** (see the roadmap, Open work 1). All phases K1-K9 are
> committed on branch `refactor/kernels-eval`. This document is now a **record of intent**, not
> a work queue — read it to understand *why* the current shape is what it is, not to execute
> anything. The as-built shape is documented in
> [../system/architecture.md](../system/architecture.md) and
> [../system/kernels.md](../system/kernels.md), which are the maintained references.
>
> It stays here (rather than in [archive/](archive/)) only until
> [refactor-validation-handoff.md](refactor-validation-handoff.md) signs off — the fallback
> recipes there refer back to these sections. Archive it once validation passes.
>
> Line references are against commit `2b1ff80` and are **stale**; functions are also named, so
> drifted references are recoverable by name.
>
> Companion plans: [evaluation-refactor.md](archive/evaluation-refactor.md),
> [future-work-and-research.md](future-work-and-research.md).
>
> **Relationship to in-flight plans:** this plan is *structure-preserving* — it deduplicates
> and re-homes code without changing any layer's observable outputs or any kernel's launch
> shape. It deliberately does **not** touch the `mode: Literal[...]` / sibling-`forward_*`
> signature work owned by [torch-export-refactor.md](torch-export-refactor.md), and it keeps
> `register_index` signatures compatible with the `capacity=` extension planned in
> [live-update-api.md](live-update-api.md). Land this first — both of those plans get smaller
> on the deduplicated base.

## Context

The library is in good shape where it counts: kernels are individually well-documented, and
the offline-tuning + `triton_op`/`custom_op` architecture (roadmap Stages 1–2b) matches
current PyTorch guidance — [`triton_op` + `wrap_triton`](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
is the documented path for traceable, exportable user Triton kernels, and avoiding runtime
`@triton.autotune` in latency-sensitive paths is recognized best practice. The debt is almost
entirely **duplication**, repeated at three levels, plus a few API-consistency gaps and one
broken tool:

- **Host-wrapper level**: every kernel file carries an `_impl` *and* one or two `@triton_op`
  functions whose bodies repeat prep + launch-args + epilogue verbatim (~40–70 lines per
  copy). Input validation exists **only** in `_impl` — the production ops are the unvalidated
  path.
- **Kernel level**: the exact-clause predicate is written out three times, the bloom subset
  test three times (in two different algebraic forms), the compaction epilogue twice.
- **Layer level**: `SimHashKNN` is a ~95% copy of `OneBitKNN`; the masked top-K epilogue is
  written out six times; the bloom signature builder exists in two near-identical variants and
  is consumed by `SilverTorch` through private `_`-imports.
- **Tooling**: `tune.py` is six copies of one sweep skeleton, and its
  `codesigned-probe-score` subcommand calls the public op with kwargs that were removed from
  its schema in Stage 2b — **currently broken** (K1).

Design references: PyTorch [user-defined Triton kernels tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html)
(traceability + export guarantees for `triton_op`); FlagGems'
[`LibEntry`](https://github.com/FlagOpen/FlagGems) (prior art for "declare once, generate the
plumbing" around Triton kernels — we keep our simpler offline `DEFAULT_CONFIG` scheme, which
is the right call for a serving-oriented library, and borrow the registry idea for `tune.py`
in K7).

## Out of scope

- `mode=` flags / sibling `forward_*` methods / Optional-arg collapse —
  [torch-export-refactor.md](torch-export-refactor.md) owns those. This plan's base-class and
  helper extractions must not add Optional-typed attrs or new data-dependent branches that
  would complicate that work.
- `upsert`/`delete`/`capacity` — [live-update-api.md](live-update-api.md).
- New kernel *optimizations* (hardware popcount, allocator hygiene, in-kernel top-k) —
  roadmap Stage 3 and [future-work-and-research.md](future-work-and-research.md).
- Any change to tuned `DEFAULT_CONFIG` values, launch grids, or score conventions.
- IVF-side algorithmic work (label-centric IVF, filter-aware clustering) — research track.

## Invariants (apply to every phase)

1. **Bit-exactness**: the parity suite (`retrieve/tests/parity/`, 8 files) passes unmodified,
   with no tolerance relaxation. OPORP/SimHash parity asserts strict equality — any
   popcount/packing refactor that changes bits is a bug by definition
   ([kernels.md → Numerics](../system/kernels.md)).
2. **`wrap_triton` stays textually inline** in every `@triton_op` body — torch.export's kernel
   registry walks the decorated function's *source* to preserve the kernel reference
   ([kernels.md → Autotune separation §4](../system/kernels.md#autotune-separation)). Dedup
   must target the code *around* the `wrap_triton(_kernel)[grid](...)` call, never the call
   itself.
3. **Op schemas are frozen**: no signature change to any registered `retrieve::*` op — the
   eval harness's compiled graphs and any exported artifacts reference them by schema.
4. **Buffer names and registration order are frozen** (state_dict compatibility): `item_embs`,
   `item_embs_t`, `item_codes_t`, `item_bits`, `oporp_signs`, `oporp_perm`, `simhash_R`,
   `centroids`, `item_codes`, `global_scale`, `padded_cluster_items`, `cluster_sizes`,
   `bloom_sigs`, `hash_seeds`, `item_clause_attrs`, `clause_is_reverse`.
5. GPU suite green at every phase boundary: `cd retrieve && uv run pytest tests/`.
6. Perf gate for kernel-adjacent phases (K2/K3): `uv run tune-kernels <kernel>` medians on the
   shipped regimes within ±5% of pre-refactor.

---

## Phase K1 — Fix the broken tuner subcommand (do first, tiny)

**Evidence.** [tune.py:38-41](../../retrieve/src/retrieve/tune.py#L38-L41) imports the
**public op**:

```python
from retrieve.kernels.silvertorch.codesigned_probe_score import (
    CodesignedProbeScoreConfig,
    codesigned_probe_score,
)
```

and `_tune_cps` calls it with `query_bits=`, `bloom_sigs=`, `config=` kwargs
([tune.py:280-303](../../retrieve/src/retrieve/tune.py#L280-L303)). Those kwargs exist only on
`_codesigned_probe_score_impl`
([codesigned_probe_score.py:123-133](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L123-L133));
the `@triton_op` schema is `(query, flat_probed_items, item_codes, global_scale, k)`. The
subcommand has been broken since Stage 2b moved the kwargs off the public op —
`uv run tune-kernels codesigned-probe-score` raises TypeError on the first sweep point.

**Edits.**

1. Change the import to `_codesigned_probe_score_impl` and both call sites (warmup loop at
   280-290, `_bench` lambda at 292-303) to call `_impl` — mirroring how every other subcommand
   already calls its `_impl` (`_fused_masked_knn_topk_impl`, `_oporp_1bit_match_topk_impl`,
   `_clause_mask_impl`, `_clause_compact_impl`, `_bloom_compact_impl`).
2. Add a CUDA-gated smoke test, `retrieve/tests/correctness/test_tune_smoke.py`:

   ```python
   """One tiny invocation per tuner benchmark body, so schema drift between
   tune.py and the kernel _impls breaks CI instead of a tuning session."""
   import pytest, torch
   from retrieve import tune

   pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

   def test_tune_bodies_run_one_point(monkeypatch):
       # shrink every grid/regime list to a single cheap point
       monkeypatch.setattr(tune, "_FMKT_GRID", [(32, 4)])
       monkeypatch.setattr(tune, "_P_BUCKETS", (256,), raising=False)  # via module under test
       ...  # analogous one-point monkeypatches per kernel
       dev = torch.device("cuda:0")
       tune._tune_fmkt(dev, d=64, b=2)
       tune._tune_oporp(dev, w=2, b=2)
       tune._tune_cps(dev, d=64, b=2, w=2)
       tune._tune_clause_mask(dev, regimes=((4096, 2, 2, 2),))
       tune._tune_clause_compact(dev, regimes=((4096, 2, 2, 2),))
       tune._tune_bloom_compact(dev, regimes=((4096, 2, 4),))
   ```

   (K7's registry rework later collapses this to one loop over specs; write it against the
   current six functions now — the test is the point, not its shape.)
3. `codesigned_probe_score_exact` has **no** subcommand; [kernels.md](../system/kernels.md)
   says the regular subcommand "mirrors" it, but the exact kernel's regime axes are
   `(C, A_MAX)` clause shapes, not bloom-word `W`. Do **not** hand-write a seventh copy now —
   K7's registry adds it declaratively. Leave a TODO in tune.py referencing K7.

**Acceptance**: smoke test green; `uv run tune-kernels codesigned-probe-score` completes on a
dev GPU and prints a `DEFAULT_CONFIG = CodesignedProbeScoreConfig(...)` line.

## Phase K2 — Host-wrapper dedup: shared prep/epilogue per kernel file

**The pattern**, quantified on the worst case
([codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py)):
`_impl` (lines 123-218), `codesigned_probe_score` (221-282), `codesigned_probe_score_bloom`
(285-348) each repeat: `quantize_int8(query)` + `.contiguous()` chain, dummy
`query_bits`/`bloom_sigs` allocation, `torch.empty` score buffer, the 25-line stride kwarg
list, and the `topk → gather` epilogue — three copies ≈ 190 lines of launch plumbing for one
kernel. Same shape at smaller scale in
[codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py)
(2 copies), [clause_compact.py](../../retrieve/src/retrieve/kernels/filters/clause_compact.py) (2),
[clause_mask.py](../../retrieve/src/retrieve/kernels/filters/clause_mask.py) (2),
[bloom_compact.py](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py) (2),
[fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/linr/fused_masked_knn_topk.py) (2),
[oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py) (3).

Invariant 2 permits sharing everything **except the launch line itself**. The recipe, worked
in full for `codesigned_probe_score.py` (the other six files follow mechanically):

### K2.1 Worked example — `codesigned_probe_score.py`

```python
@dataclass(frozen=True)
class _CpsLaunch:
    grid: tuple[int, int]
    kwargs: dict[str, object]        # every kernel arg: tensors, strides, constexprs, cfg
    all_scores: Tensor
    flat_probed_items: Tensor        # post-contiguous, for the epilogue gather


def _cps_prep(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    global_scale: float,
    *,
    query_bits: Tensor | None,
    bloom_sigs: Tensor | None,
    cfg: CodesignedProbeScoreConfig,
) -> _CpsLaunch:
    """Validation + contiguity + buffers + the full launch-arg dict.
    THE single place input checking happens — shared by _impl and both ops."""
    if query.dim() != 2 or flat_probed_items.dim() != 2:
        raise ValueError("query must be [B, D] and flat_probed_items [B, P]")
    if item_codes.dtype != torch.int8:
        raise TypeError(f"item_codes must be int8, got {item_codes.dtype}")
    has_qb = query_bits is not None
    if has_qb and bloom_sigs is None:
        raise ValueError("bloom_sigs is required when query_bits is provided")

    b, d = query.shape
    p = flat_probed_items.shape[1]
    q_codes, q_scales = quantize_int8(query)
    q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
    flat_probed_items = flat_probed_items.contiguous()
    item_codes = item_codes.contiguous()
    if has_qb:
        query_bits, bloom_sigs = query_bits.contiguous(), bloom_sigs.contiguous()
        w = query_bits.shape[1]
    else:
        # 1×1 int64 dummies: HAS_QB=False gates every load; never dereferenced.
        query_bits = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        bloom_sigs = torch.empty(1, 1, dtype=torch.int64, device=query.device)
        w = 1
    # torch.empty is safe: every in-bounds lane is overwritten (dot or -inf).
    all_scores = torch.empty((b, p), dtype=torch.float32, device=query.device)
    kwargs = dict(
        q_codes_ptr=q_codes, q_scales_ptr=q_scales, qb_ptr=query_bits,
        flat_items_ptr=flat_probed_items, item_codes_ptr=item_codes,
        bloom_sigs_ptr=bloom_sigs, out_scores_ptr=all_scores,
        global_scale=float(global_scale),
        P=p, D=d, W=w,
        stride_qcb=q_codes.stride(0), stride_qcd=q_codes.stride(1),
        stride_qs=q_scales.stride(0),
        stride_qbb=query_bits.stride(0), stride_qbw=query_bits.stride(1),
        stride_fb=flat_probed_items.stride(0), stride_fp=flat_probed_items.stride(1),
        stride_cn=item_codes.stride(0), stride_cd=item_codes.stride(1),
        stride_bn=bloom_sigs.stride(0), stride_bw=bloom_sigs.stride(1),
        stride_ob=all_scores.stride(0), stride_op=all_scores.stride(1),
        HAS_QB=has_qb,
        BLOCK_P=cfg.block_p, num_warps=cfg.num_warps, num_stages=cfg.num_stages,
    )
    grid = (triton.cdiv(p, cfg.block_p), b)
    return _CpsLaunch(grid, kwargs, all_scores, flat_probed_items)


def _cps_finish(launch: _CpsLaunch, k: int) -> tuple[Tensor, Tensor]:
    # P >= k by SilverTorch.register_index assert; no pad tail.
    topk_scores, topk_local = torch.topk(launch.all_scores, k, dim=1)
    return launch.flat_probed_items.gather(1, topk_local), topk_scores
```

The three entry points collapse to:

```python
def _codesigned_probe_score_impl(query, flat_probed_items, item_codes, global_scale, k, *,
                                 query_bits=None, bloom_sigs=None, config=None):
    cfg = config if config is not None else DEFAULT_CONFIG
    launch = _cps_prep(query, flat_probed_items, item_codes, global_scale,
                       query_bits=query_bits, bloom_sigs=bloom_sigs, cfg=cfg)
    _codesigned_probe_score_kernel[launch.grid](**launch.kwargs)
    return _cps_finish(launch, k)


@triton_op("retrieve::codesigned_probe_score", mutates_args=())
def codesigned_probe_score(query, flat_probed_items, item_codes, global_scale, k):
    launch = _cps_prep(query, flat_probed_items, item_codes, global_scale,
                       query_bits=None, bloom_sigs=None, cfg=DEFAULT_CONFIG)
    wrap_triton(_codesigned_probe_score_kernel)[launch.grid](**launch.kwargs)   # inline, K2 invariant
    return _cps_finish(launch, k)


@triton_op("retrieve::codesigned_probe_score_bloom", mutates_args=())
def codesigned_probe_score_bloom(query, flat_probed_items, item_codes,
                                 query_bits, bloom_sigs, global_scale, k):
    launch = _cps_prep(query, flat_probed_items, item_codes, global_scale,
                       query_bits=query_bits, bloom_sigs=bloom_sigs, cfg=DEFAULT_CONFIG)
    wrap_triton(_codesigned_probe_score_kernel)[launch.grid](**launch.kwargs)
    return _cps_finish(launch, k)
```

~348 lines → ~210, one validation site, both ops now validated.

**Tracing caveat to verify on the first file** (this is the one genuine risk in K2):
`triton_op` requires its body to be traceable by `make_fx` for the fake path. `_cps_prep`
raises on bad dims (fine — shape checks on fake tensors work), builds tensors (fine), and
returns a dataclass carrying a plain-dict of launch kwargs (opaque to tracing until the
`wrap_triton` call consumes it — fine, it's just Python structure). If `wrap_triton(...)`
rejects `**kwargs` splat or the dataclass indirection under tracing, fall back to `_prep`
returning a plain tuple and splatting positionally — same dedup, less sugar. Test with
`tests/compile/test_silvertorch_compile.py` plus a `torch.export.export` spot check (see
Verification 4) **before** rolling the pattern to the other six files.

### K2.2 Per-file specifics for the remaining six

| file | prep/finish quirks to preserve |
|---|---|
| `codesigned_probe_score_exact.py` | prep validates `C`/`A_max`/batch agreement (lines 148-166) and does the `clause_is_reverse.to(torch.int8)` cast (Triton can't load torch.bool — keep the comment); finish identical to `_cps_finish` |
| `fused_masked_knn_topk.py` | `_impl` buckets P via `_bucket_p` + `p == 0` early-return + `actual_k < k` pad tail (lines 110-184); the public op does none of those (documented: "catalog size is fixed per deployment", `p >= k > 0` guaranteed by `PrefilterKNN`). Express as `_fmkt_prep(..., bucket: bool)` and `_fmkt_finish(..., pad_to_k: bool)` flags so the intentional difference is explicit rather than implicit in divergent copies |
| `oporp_1bit_match_topk.py` | prep handles the `HAS_INDICES` dummy-tensor branch (lines 156-178); finish handles the `clamp_max(n_loop - 1)` + `where(isfinite, …, -1)` tail. The indirect op *keeps* its `max(_bucket_n(P), _bucket_n(k))` widening (guarantees ≥ k lanes, [oporp:293-295](../../retrieve/src/retrieve/kernels/linr/oporp_1bit_match_topk.py#L293-L295)) — that one is not vestigial |
| `clause_compact.py` / `bloom_compact.py` | prep allocates the `-1`-filled `out_indices` + zeroed `counts` (the atomic_add-safety allocation notes at [clause_compact.py:124-129](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L124-L129) move onto `_prep`'s docstring); finish is identity (`return out_indices, counts`) — skip a finish helper |
| `clause_mask.py` | prep only; output buffer is the return value |
| `bloom_match.py` | leave as-is (single 80-line file, one op, no `_impl`, no Config — see K9 for the doc note) |

**Bucketing-policy documentation** (fold into this phase): `fused_masked_knn_topk`'s public op
skips bucketing while `oporp_1bit_match_topk_indirect` buckets. Both are correct today —
candidate widths are static per deployment (the compact family returns full-width `[B, N]`;
linr_v3's stage-2 width is the fixed `candidate_pool`) and oporp's bucket doubles as the
≥ k-lanes guarantee. Write this down once, in a module docstring shared paragraph, so the next
reader doesn't re-derive it.

**Acceptance**: parity suite green per file; `−350…450` lines net across the seven files;
grep gate: every `@triton_op` body contains exactly one `wrap_triton(` and zero `.stride(`
calls (strides all come from `_prep`).

## Phase K3 — Kernel-body dedup: shared `@triton.jit` helpers

Triton supports calling `@triton.jit` functions from kernels (already used:
`_popcount_int64`, `_or_combine`). Duplicated bodies:

1. **Exact-clause predicate** (inner OR over `A_MAX`, outer AND over `C`, reverse XOR,
   `q_c == -1` inactive override) — three copies:
   [clause_mask.py:59-79](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L59-L79),
   [clause_compact.py:62-83](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L62-L83),
   [codesigned_probe_score_exact.py:81-99](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py#L81-L99)
   (same math; the codesigned copy addresses items indirectly via `safe_ids` and gates loads
   with `valid` instead of `n_valid`).
2. **Bloom subset test** — three copies in **two algebraic forms**:
   equality + min-reduce ([bloom_match.py:40-43](../../retrieve/src/retrieve/kernels/silvertorch/bloom_match.py#L40-L43),
   [bloom_compact.py:66-69](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L66-L69))
   vs `qb & ~sig` + OR-reduce
   ([codesigned_probe_score.py:93-97](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L93-L97)).
3. **Compaction epilogue** (`cumsum → atomic_add → masked store`) — two copies:
   [clause_compact.py:85-97](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L85-L97),
   [bloom_compact.py:71-83](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L71-L83).

**New file** `retrieve/src/retrieve/kernels/common.py`:

```python
"""Shared @triton.jit building blocks. Every helper is a pure function of
already-loaded tiles or takes fully-resolved addressing from the caller —
no helper decides its own tile shape or masking policy."""
import triton
import triton.language as tl


@triton.jit
def or_combine(a, b):
    return a | b


@triton.jit
def popcount_int64(x):
    """SWAR popcount; torch twin: retrieve.layers.utils.quantize.popcount_int64
    (bit-exact pairing is load-bearing — see kernels.md → Numerics)."""
    M1 = 0x5555555555555555
    M2 = 0x3333333333333333
    M4 = 0x0F0F0F0F0F0F0F0F
    H01 = 0x0101010101010101
    x = x - ((x >> 1) & M1)
    x = (x & M2) + ((x >> 2) & M2)
    x = (x + (x >> 4)) & M4
    return ((x * H01) >> 56).to(tl.int32)


@triton.jit
def bloom_subset_pass(qb, sigs):
    """(qb & sig) == qb per word ⇔ qb & ~sig == 0; OR-reduce over W.
    Operates on already-loaded tiles: qb [W], sigs [BLOCK, W] → [BLOCK] int1."""
    diff = qb[None, :] & ~sigs
    return tl.reduce(diff, axis=1, combine_fn=or_combine) == 0


@triton.jit
def clause_pass(item_attrs_ptr, is_reverse_ptr, query_attrs_ptr,
                ids, load_mask, bid,
                stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
                C: tl.constexpr, A_MAX: tl.constexpr):
    """AND-of-OR exact clause predicate over a tile of item rows.
    `ids` = item row indices ([BLOCK], contiguous n_offsets or gathered
    safe_ids); `load_mask` gates the attr loads. Returns [BLOCK] int1."""
    keep = load_mask
    for c in tl.static_range(C):
        q_c = tl.load(query_attrs_ptr + bid * stride_qb + c * stride_qc)
        rev_c = tl.load(is_reverse_ptr + c).to(tl.int1)
        clause_match = tl.zeros(ids.shape, tl.int1)
        for a in tl.static_range(A_MAX):
            ia = tl.load(item_attrs_ptr + ids * stride_in + c * stride_ic + a * stride_ia,
                         mask=load_mask, other=-1)
            clause_match = clause_match | (ia == q_c)
        clause_match = clause_match ^ rev_c
        clause_match = clause_match | (q_c == -1)   # inactive clause always passes
        keep = keep & clause_match
    return keep


@triton.jit
def compact_store(pass_mask, ids, counts_ptr, out_ptr, bid, stride_ob, stride_on):
    """Stream compaction: cumsum intra-tile offsets + atomic_add row base."""
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1
    tile_sum = tl.sum(pass_int)
    base = tl.atomic_add(counts_ptr + bid, tile_sum.to(tl.int64))
    tl.store(out_ptr + bid * stride_ob + (base + intra.to(tl.int64)) * stride_on,
             ids.to(tl.int64), mask=pass_mask)
```

**Call-site rewrites** (each kernel keeps its own launch grid, loads, and epilogue policy):

- `clause_mask` kernel body: `pass_mask = clause_pass(...) & n_valid` + existing store.
- `clause_compact`: `pass_mask = clause_pass(...) & n_valid` + `compact_store(...)`.
- `codesigned_probe_score_exact`: `keep = clause_pass(..., ids=safe_ids, load_mask=valid, ...)`
  — note `keep` starts from `valid` inside the helper, matching the current inlined logic.
- `bloom_match` / `bloom_compact`: switch to `bloom_subset_pass` — **this standardizes on the
  `qb & ~sig` OR-reduce form** (the cheaper one per [kernels.md](../system/kernels.md), which
  documents it saves the int32 cast + min reduce). Boolean-identical output; parity tests
  (`test_bloom_match.py`, `test_bloom_compact.py`) confirm.
- `codesigned_probe_score`: already the OR-reduce form; swap the inline lines for the helper.
- `oporp_1bit_match_topk`: swap `_popcount_int64` for `common.popcount_int64`; delete the
  local copy. The torch twin stays in `layers/utils/quantize.py` (Invariant 1's bit-exact
  pairing note moves to `common.py`'s docstring).

**Perf gate + fallback**: helper calls are inlined by Triton, but a helper with this many
pointer params can perturb register allocation. Run `uv run tune-kernels clause-mask
clause-compact bloom-compact codesigned-probe-score` (post-K1 fix) on the shipped regimes
before/after. If any kernel regresses > 5%, revert **that kernel** to the inlined form with a
`# keep in sync with kernels/common.py::clause_pass` breadcrumb — measured perf beats textual
purity. `bloom_subset_pass`/`popcount_int64`/`compact_store` are small enough that regression
is implausible; `clause_pass` is the one to watch.

**Explicit non-goal**: do **not** merge `codesigned_probe_score` and
`codesigned_probe_score_exact` into one kernel with a `HAS_QB`/`HAS_EXACT` constexpr matrix.
Two ops with clean schemas beat one op with dummy `W`/`C`/`A_MAX` params and coupled register
pressure; the shared helpers remove the duplication without coupling the specializations. (The
bloom/none pair already shares one kernel via `HAS_QB` — that stays.)

## Phase K4 — Layer-level dedup

### K4.1 `masked_topk` utility (six copies → one)

The epilogue "mask invalid scores to `-inf` → `topk` → map local→global ids → replace
non-finite winners with `-1` → optionally pad to k" appears six times with drift-prone
variations: [prefilter_knn.py:66-84](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L66-L84)
(cat-style pad), [one_bit_knn.py:104-119](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L104-L119),
[simhash_knn.py:90-105](../../retrieve/src/retrieve/layers/linr/simhash_knn.py#L90-L105),
[silvertorch/main.py:302-316](../../retrieve/src/retrieve/layers/silvertorch/main.py#L302-L316)
(full+slice-assign pad), [silvertorch/main.py:330-333](../../retrieve/src/retrieve/layers/silvertorch/main.py#L330-L333)
(no pad, no sentinel), [postfilter_knn.py:32-41](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L32-L41)
+ [postfilter_knn_int8.py:68-78](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L68-L78)
(mask-optional dense form).

**New file** `retrieve/src/retrieve/layers/utils/topk.py`:

```python
"""Shared masked top-K epilogue for the torch-side layer paths.

Pure tensor-flow (no .item(), no data-dependent Python branches beyond
min(k, P) over static shapes) — traces cleanly under the algo-level
torch.compile(dynamic=True, mode='reduce-overhead') exactly like the
previously-inlined originals."""
from __future__ import annotations

import torch
from torch import Tensor


def counts_to_valid(counts: Tensor, p: int) -> Tensor:
    """[B] counts → [B, p] bool prefix mask."""
    return torch.arange(p, device=counts.device).unsqueeze(0) < counts.unsqueeze(1)


def masked_topk(
    scores: Tensor,                      # [B, P] fp
    k: int,
    *,
    valid: Tensor | None = None,         # [B, P] bool; None = all valid
    gather_ids: Tensor | None = None,    # [B, P] local→global map; None = identity
    pad_to_k: bool = True,               # rows with P < k or few survivors pad -1/-inf
) -> tuple[Tensor, Tensor]:
    b, p = scores.shape
    if valid is not None:
        scores = scores.masked_fill(~valid, float("-inf"))
    actual_k = min(k, p)
    topk_scores, topk_local = torch.topk(scores, actual_k, dim=1)
    topk_ids = gather_ids.gather(1, topk_local) if gather_ids is not None else topk_local
    if valid is not None or actual_k < k:
        topk_ids = torch.where(torch.isfinite(topk_scores), topk_ids,
                               topk_ids.new_full((), -1))
    if pad_to_k and actual_k < k:
        pad = k - actual_k
        topk_ids = torch.cat([topk_ids, topk_ids.new_full((b, pad), -1)], dim=1)
        topk_scores = torch.cat(
            [topk_scores, topk_scores.new_full((b, pad), float("-inf"))], dim=1)
    return topk_ids, topk_scores
```

**Per-call-site replacements** (each must reproduce current behavior exactly — the table is
the review checklist):

| site | replacement | behavioral notes |
|---|---|---|
| `PrefilterKNN._forward_prefilter` | `masked_topk(scores, self.k, valid=counts_to_valid(counts, p) if counts is not None else None, gather_ids=candidate_ids)` | keeps the cat-pad + isfinite sentinel |
| `OneBitKNN._forward_torch_eager` (candidates branch) | same shape | current code applies the `-1` sentinel only `if counts is not None` — `masked_topk` applies it whenever `valid` was given; identical observable behavior since `valid is None ⇒` all scores finite. Current code also does **not** pad to k on this path (`actual_k = min(...)` then returns) — pass `pad_to_k=False` to preserve; **verify against `tests/correctness/test_linr.py` expectations before flipping to True** |
| `SimHashKNN._forward_torch_eager` | identical to OneBitKNN's | disappears entirely under K4.2 |
| `SilverTorch._forward_torch_eager` epilogue | `masked_topk(scores, self.k, valid=keep, gather_ids=flat_items)` | replaces lines 302-316 incl. the full+slice pad |
| `SilverTorch._forward_candidates` | `masked_topk(scores, self.k, gather_ids=candidate_ids, pad_to_k=False)` | current code has **no** `-inf`/`-1` handling here (no mask ⇒ none needed) |
| `PostfilterKNN.forward` / `PostfilterKNNInt8.forward` | `masked_topk(scores, self.k, valid=mask)` — but the mask is Optional here | keep the existing `if mask is not None` outer branch; the [torch-export-refactor](torch-export-refactor.md) mode-split will dissolve it. Inside each branch, call `masked_topk` |

Add `tests/correctness/test_topk_util.py`: empty rows (`counts=0`), `counts < k`, all-masked
row, tie-at-`-inf` boundary, `gather_ids` mapping, `pad_to_k` both ways.

**Kernel host tails are separate**: `_fmkt_finish`/oporp's finish (K2) have clamp-before-gather
logic tied to bucketed buffer widths; where semantics coincide they *may* call `masked_topk`,
but do not force unification — the kernel-side tails' subtleties (uninitialized lanes past
`counts[b]`) are documented in kernel files and should stay there.

### K4.2 Fold `SimHashKNN` into a shared bit-KNN base (~130 duplicated lines → ~30)

[simhash_knn.py](../../retrieve/src/retrieve/layers/linr/simhash_knn.py) duplicates
[one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) almost line-for-line:
identical module-level `_score_full_*_eager` (18-27 both files), `_forward_torch_eager` (78-109
vs 90-123), `_forward_triton` (111-134 vs 125-148), `d_total` property, `k > n` guard. Real
differences: the index-time quantizer, the query projection, and OneBitKNN's `k_bits=0`
sentinel.

**New file** `retrieve/src/retrieve/layers/linr/_bit_knn.py`:

```python
class _PackedBitsKNN(nn.Module):
    """Hamming-similarity KNN over packed int64 sign bits: score =
    64·W − 2·popcount(q ^ item). Subclasses own bit production; scoring,
    dispatch, and the candidates/full paths are common.

    Both backends produce byte-identical bits (projection is shared), so
    torch and Triton paths score identically — the parity property the
    tests assert."""

    item_bits: Tensor

    def __init__(self, k: int, backend: Backend = "triton") -> None:
        super().__init__()
        self.k = k
        self.backend = backend

    # -- subclass hooks -------------------------------------------------
    def _quantize_index(self, item_embs: Tensor) -> None:
        """Build + register item_bits and projection buffers."""
        raise NotImplementedError

    def _project_query(self, query: Tensor) -> Tensor:
        """[B, D] → [B, W] int64, same bit space as item_bits."""
        raise NotImplementedError

    # -- common ----------------------------------------------------------
    def register_index(self, item_embs: Tensor) -> None:
        if self.k > item_embs.shape[0]:
            raise ValueError(f"k={self.k} exceeds corpus size N={item_embs.shape[0]}")
        self._quantize_index(item_embs)

    @property
    def d_total(self) -> int:
        return 64 * self.item_bits.shape[1]

    def forward(self, query, candidate_ids=None, counts=None): ...        # current dispatch
    def _forward_torch_eager(self, query, candidate_ids, counts): ...     # uses masked_topk (K4.1)
    def _forward_triton(self, query, candidate_ids, counts): ...          # current body verbatim
```

Subclasses:

```python
class OneBitKNN(_PackedBitsKNN):
    item_bits: Tensor; oporp_signs: Tensor; oporp_perm: Tensor
    def __init__(self, k, seed=0, backend="triton", k_bits=0):
        super().__init__(k, backend); self.seed = seed; self.k_bits = k_bits
    def _quantize_index(self, item_embs):
        bits, signs, perm = quantize_oporp_1bit(item_embs, seed=self.seed, k_bits=self.k_bits)
        if self.k_bits == 0:                       # sentinel resolution — keep the
            self.k_bits = item_embs.shape[1]       # Dynamo-stability comment with it
        self.register_buffer("item_bits", bits)
        self.register_buffer("oporp_signs", signs)
        self.register_buffer("oporp_perm", perm)
    def _project_query(self, query):
        return project_oporp_1bit_query(query, self.oporp_signs, self.oporp_perm, self.k_bits)


class SimHashKNN(_PackedBitsKNN):
    item_bits: Tensor; simhash_R: Tensor
    def __init__(self, k, k_bits, seed=0, backend="triton"): ...
    def _quantize_index(self, item_embs):
        bits, r = quantize_simhash_1bit(item_embs, self.k_bits, self.seed)
        self.register_buffer("item_bits", bits)
        self.register_buffer("simhash_R", r)
    def _project_query(self, query):
        return project_simhash_1bit_query(query, self.simhash_R)
```

Public names, constructor signatures, buffer names, and class docstrings (the paper
references — Li et al. Sign-OPORP; Charikar/Manku SimHash) are **unchanged**; existing
cross-backend equality tests (`TestSimHashKNNCrossBackend`,
`test_simhash_torch_matches_simhash_triton`) pin bit-exactness through the move. Torch-export
note: the base introduces no Optional attrs and no new branches — the export plan's
`mode: Literal["full", "candidates"]` flag later lands **once** on `_PackedBitsKNN` instead of
twice.

**Micro-dedup in `utils/quantize.py`** (same PR):
`quantize_oporp_1bit` ([quantize.py:68-94](../../retrieve/src/retrieve/layers/utils/quantize.py#L68-L94))
and `project_oporp_1bit_query` (97-121) repeat validation + the
sign-flip → permute → bin → normalize → pack chain. Extract:

```python
def _oporp_project(x: Tensor, signs: Tensor, perm: Tensor, k_bits: int) -> Tensor:
    b_or_n, d = x.shape
    # (shared k_bits validation here)
    bin_w = d // k_bits
    proj = (x * signs.to(x.dtype)).index_select(1, perm)
    binned = proj.view(b_or_n, k_bits, bin_w).sum(dim=-1)
    sketch = binned / binned.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return _pack_signs_to_int64(sketch)
```

Both public functions become thin wrappers (the quantizer adds `_build_oporp`). Bit-exactness
gate: `tests/correctness/test_quantize.py` + strict-equality parity.

### K4.3 SilverTorch structural cleanups (no semantics change)

1. **Split `register_index`**
   ([main.py:101-188](../../retrieve/src/retrieve/layers/silvertorch/main.py#L101-L188), ~90
   lines, four phases in one method):

   ```python
   def register_index(self, item_embs, item_clause_attrs=None, clause_is_reverse=None):
       self._validate_register_args(item_embs, item_clause_attrs, clause_is_reverse)
       self._build_ivf(item_embs)                     # k-means, padded_cluster_items, sizes (122-148)
       self._quantize_items(item_embs)                # codes + global_scale + _global_scale_f (150-164)
       self._register_filter_buffers(item_embs.shape[0], item_clause_attrs, clause_is_reverse)  # (166-188)
   ```

   Buffer names and **registration order** stay identical (Invariant 4).
2. **Guard the silent filter-skip on the candidates path**: `forward` returns
   `_forward_candidates` *before* validating `query_clause_attrs`
   ([main.py:199-200](../../retrieve/src/retrieve/layers/silvertorch/main.py#L199-L200)), so
   `forward(q, query_clause_attrs=qa, candidate_ids=ids)` silently ignores the predicate:

   ```python
   if candidate_ids is not None:
       if query_clause_attrs is not None:
           raise ValueError(
               "candidate_ids path scores the given candidates without the fused "
               "attribute filter; pass query_clause_attrs OR candidate_ids, not both")
       return self._forward_candidates(query, candidate_ids)
   ```

   Add a correctness test asserting the raise.
3. **Share the eager predicate math with the filters**: SilverTorch's torch-eager exact block
   ([main.py:288-296](../../retrieve/src/retrieve/layers/silvertorch/main.py#L288-L296)) is a
   copy of `ExactAttributeFilter.evaluate_subset`
   ([exact_attribute.py:74-82](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L74-L82));
   its bloom block (277-287) copies `BloomFilter.evaluate_subset`
   ([bloom.py:87-90](../../retrieve/src/retrieve/layers/filters/bloom.py#L87-L90)). Extract two
   free functions into the filters package (natural home: `exact_attribute.py` and the K5
   `bloom_hash.py`):

   ```python
   def clause_subset_match(gathered_attrs, query_attrs, clause_is_reverse) -> Tensor:
       """[B, P, C, A] attrs vs [B, C] query → [B, P] bool. Reverse-XOR and
       q==-1 inactive semantics — the single torch-side definition."""

   def bloom_subset_match(qb, sigs) -> Tensor:
       """[B, W] query sigs vs [B, P, W] gathered sigs → [B, P] bool."""
   ```

   consumed by the filter classes' `evaluate_subset` **and** SilverTorch's eager path.
   SilverTorch deliberately does not hold `FilterModule` instances (the in-model fused filter
   is the paper's point) — it shares the ten-line predicate math, not the module.
4. **`filter` shadows the builtin** ([main.py:53](../../retrieve/src/retrieve/layers/silvertorch/main.py#L53)):
   rename param + attr to `filter_mode` (keep the `FilterMode` type alias). Call sites:
   `build_silvertorch` (main.py:342), `SilvertorchAlgo`
   (evaluation/retrieval/algos/silvertorch.py:69, 87),
   `tests/correctness/test_silvertorch.py`, `tests/compile/test_silvertorch_compile.py`,
   `tests/parity/test_codesigned_probe_score*.py` (grep `filter=` under `retrieve/`). No
   deprecation shim (pre-1.0, repo convention). Skip only if it churns in-flight thesis text.

### K4.4 `FullScanKNN` mask semantics — document loudly

[retrieval.py:33-45](../../retrieve/src/retrieve/layers/utils/retrieval.py#L33-L45) applies the
mask **post-topk** (`post_filter_topk`): masked winners become `-1` but their scores remain in
`topk_scores`, and rows with fewer than k survivors are *not* backfilled from the remaining
corpus — the opposite of every other layer, and the docstring ("Optional post-filter mask or
candidate_ids path") doesn't say so. This asymmetry is *intentional* — it is the LiNR paper's
post-filtering baseline, and [torch-export-refactor.md](torch-export-refactor.md) already
models it as `mode="masked"` keeping `post_filter_topk`. Fix = documentation:

```python
"""Exhaustive matmul + top-K.

`mask` implements POST-filter semantics (LiNR baseline): top-K is selected
over the full corpus first, then masked hits are tombstoned to id=-1 — they
are NOT replaced by the next-best passing items, and their scores remain in
the returned score tensor. Recall against a pre-filter oracle is therefore
expected to be < 1 by design. For pre-filter semantics use PostfilterKNN
(mask before top-K) or PrefilterKNN (candidates path)."""
```

Also note `post_filter_topk` returns `(ids, counts)` but `forward` drops `counts` — either
document ("count of survivors is discarded; callers needing it call `post_filter_topk`
directly") or return it; keep-and-document is the non-breaking choice.

## Phase K5 — Bloom signature builder: one core, public API

1. **New module** `retrieve/src/retrieve/layers/filters/bloom_hash.py` receives (from
   `bloom.py`): `_generate_seeds` → `generate_seeds`, `_mix64` (stays private),
   `_BUILD_SIGS_BATCH`, and the two builders re-cored:

   ```python
   _SALT_C1 = 0x9E3779B97F4A7C15 - (1 << 64)     # named once; today inlined twice
   _SALT_C2 = 0xBF58476D1CE4E5B9 - (1 << 64)

   def _signature_batch(flat, valid, seeds, m_bits, k_hash, word_count, clause_salt) -> Tensor:
       """Hash + salt + scatter + word-pack for one [b, C*A] slab → [b, W].
       The shared core of the two builders below; extracted verbatim from
       the chunk-loop body at bloom.py:159-170."""

   def build_signatures(attrs, seeds, m_bits, k_hash, word_count) -> Tensor:
       """Index-side: chunked over _BUILD_SIGS_BATCH rows (peak-alloc bound)."""

   def build_query_signatures(attrs, seeds, m_bits, k_hash, word_count) -> Tensor:
       """Query-side: single loop-free call — a Python range() loop would make
       dynamo specialize on trip count under torch.compile(dynamic=True)."""
   ```

   The bodies are ~85% identical today
   ([bloom.py:119-173](../../retrieve/src/retrieve/layers/filters/bloom.py#L119-L173) vs
   [176-224](../../retrieve/src/retrieve/layers/filters/bloom.py#L176-L224)); after extraction
   each is ~10 lines around `_signature_batch`. The `(clause_idx, value)` keying invariant
   comment (the false-positive-leak fix documented in
   [kernels.md → bloom_match](../system/kernels.md)) moves onto `_signature_batch`.
2. **Kill the private cross-module imports**:
   [silvertorch/main.py:16-20](../../retrieve/src/retrieve/layers/silvertorch/main.py#L16-L20)
   imports `_build_query_signatures`, `_build_signatures`, `_generate_seeds` from `bloom.py`.
   Both `bloom.py` (the `BloomFilter` class keeps only class logic) and `silvertorch/main.py`
   import the public names from `bloom_hash.py`.
3. **Bit-stability gate**: persisted `bloom_sigs` buffers must stay valid — the hash math is
   untouched, only re-homed. `tests/correctness/test_bloom_filter.py` (hash-invariant cases)
   plus one new test: `build_signatures(attrs)[i] == build_query_signatures(attrs[i:i+1])[0]`
   for a random slab — the chunked and loop-free paths must agree exactly (this property is
   assumed today but never asserted).

## Phase K6 — Interface consistency

### K6.1 `FilterModule.register_index` signatures must agree

Three incompatible orderings today:

| | signature after `item_clause_attrs` |
|---|---|
| ABC ([interfaces.py:19-23](../../retrieve/src/retrieve/interfaces.py#L19-L23)) | `item_embs=None` |
| `ExactAttributeFilter` ([exact_attribute.py:25-30](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L25-L30)) | `clause_is_reverse=None, item_embs=None` |
| `BloomFilter` ([bloom.py:36-41](../../retrieve/src/retrieve/layers/filters/bloom.py#L36-L41)) | `item_embs=None, clause_is_reverse=None` |

A positional `clause_is_reverse` passed to `BloomFilter` lands in `item_embs` and is silently
ignored — the same bug class the silvertorch-reverse plan fixed at the eval layer.

**Fix**: `grep -rn "item_embs" retrieve/src/retrieve/layers/filters/ evaluation/` confirms
nothing ever passes `item_embs` to a filter (it was speculative). New unified signature on the
ABC and both classes:

```python
def register_index(self, item_clause_attrs: Tensor, *,
                   clause_is_reverse: Tensor | None = None) -> None: ...
```

— keyword-only after the first arg, `item_embs` dropped (a future embedding-aware filter can
extend its own signature; the ABC shouldn't carry speculative params). `BloomFilter` keeps its
"paper-strict: reverse must be all-False" raise. Callers to touch:
evaluation/retrieval/algos/filter.py:44-51
(already keyword-style — only the bloom branch's `bf.register_index(item_attrs_narrow)` is
signature-compatible as-is), library tests. Add the K8 TypeError test locking kw-only.

### K6.2 Reintroduce a minimal `RetrievalModule` ABC

`interfaces.py` contains only `Backend` + `FilterModule`;
[docs/system/architecture.md](../system/architecture.md) still describes `RetrievalModule` and
`ScorerModule`. Every retrieval layer subclasses raw `nn.Module` with a *conventional*
`register_index → forward → (ids, scores)` shape nothing enforces — and
[live-update-api.md](live-update-api.md) Phase A plans to add `capacity: int | None` "to
abstract `RetrievalModule.register_index`", i.e. it assumes the ABC exists.

```python
class RetrievalModule(nn.Module, abc.ABC):
    """A top-K retriever over a registered item index.

    Contract: construct with k (+ knobs) → register_index(item_embs, ...)
    exactly once → forward(query, ...) → (ids [B, k] int64, scores [B, k]).
    -1 / -inf are the "no item" sentinels. Forward signatures vary by family
    (mask vs candidates vs fused-filter) and are being unified per-mode by
    the torch-export plan; this ABC intentionally constrains only the
    lifecycle, not forward.
    """
    k: int

    @abc.abstractmethod
    def register_index(self, item_embs: Tensor, **kwargs) -> None: ...
```

Subclass it in: `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN`, `_PackedBitsKNN` (covers
`OneBitKNN`/`SimHashKNN`), `SilverTorch`, `FullScanKNN`. Not `KMeansTorch` (not a retriever).
`ScorerModule` stays dead — do not resurrect; K9 removes it from the docs.

### K6.3 Docstring corrections (one-liners)

- [postfilter_knn_int8.py:20-24](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L20-L24)
  claims "topk order is exact"; the `>> 5` range shift (line 67) floor-divides int32 dots —
  order-preserving but tie-*introducing* within 32-unit buckets. Qualify: "topk ordering is
  exact up to ties introduced by the >>5 range compression (boundary ties are
  quality-equivalent)."
- [compact.py:6-14](../../retrieve/src/retrieve/layers/utils/compact.py#L6-L14) says it
  "matches the triton bloom_compact/clause_compact contract" — align the wording: all three
  return **full-width** `[B, N]` indices; entries past `counts[b]` are arbitrary (argsort
  tail) for `compact_mask` vs `-1` (prefilled) for the kernels; consumers must bound by
  `counts` either way. State both facts in both places.

## Phase K7 — `tune.py`: one sweep engine + registry (~640 → ~300 lines)

Six `_tune_*` functions ([tune.py:100-499](../../retrieve/src/retrieve/tune.py#L100-L499))
share one skeleton — iterate regimes → build inputs → for each `(block, warps)`: warm 3×,
`do_bench`, track best → plurality-vote (ties → lower warps) → print paste line — and six
`_print_*` functions differ only in path + class name.

```python
@dataclass(frozen=True)
class KernelTuneSpec:
    name: str                                  # subcommand, e.g. "clause-mask"
    config_cls: type                           # ClauseMaskConfig, ...
    grid: tuple[tuple[int, int], ...]          # (block, warps) candidates
    default_regimes: tuple[tuple[int, ...], ...]
    regime_arity: int | None                   # 4 = N,B,C,A_MAX; 3 = N,B,W; None = no --regime
    regime_fmt: str                            # metavar for --help
    make_inputs: Callable[[torch.device, tuple], tuple]   # regime → kernel args
    run: Callable[..., object]                 # (inputs, config) → one _impl call
    paste_path: str                            # file the DEFAULT_CONFIG line goes into


KERNELS: tuple[KernelTuneSpec, ...] = (
    KernelTuneSpec("fused-masked-knn-topk", FusedMaskedKnnTopkConfig, _FMKT_GRID, ...),
    KernelTuneSpec("oporp-1bit-match-topk", ...),
    KernelTuneSpec("codesigned-probe-score", ...),          # calls _impl (K1)
    KernelTuneSpec("codesigned-probe-score-exact", ...),    # NEW — (N, B, C, A_MAX) regimes
    KernelTuneSpec("clause-mask", ...),
    KernelTuneSpec("clause-compact", ...),
    KernelTuneSpec("bloom-compact", ...),
)
```

One generic `_sweep(spec, dev, regimes) -> dict` (the current per-regime loop with the warmup
+ `_bench` + vote logic — lift verbatim from `_tune_clause_mask`, the cleanest instance), one
generic `_print(spec, arch, result)`, and click subcommands generated in a loop:

```python
for spec in KERNELS:
    _register_subcommand(main, spec)     # builds the click command incl. --regime when arity set
```

The atomic-add-safety note (fresh output buffers per `_impl` call ⇒ in-kernel autotune hazard
doesn't apply offline) currently lives on three functions — it moves once onto `_sweep`.
K1's smoke test collapses to `for spec in KERNELS: spec.run(spec.make_inputs(dev, tiny), cfg)`.

## Phase K8 — Test additions

- `test_topk_util.py` (K4.1 — see list there).
- `test_tune_smoke.py` (K1 → reshaped by K7).
- **`evaluate_subset` direct coverage**: for both filters, assert
  `evaluate_subset(qa, ids) == evaluate_mask(qa).gather(1, ids)` on random inputs including
  reverse clauses (exact) and `-1`-inactive queries (both). Today this equivalence is exercised
  only indirectly through `test_combine_filters.py`; check that file first and add only what's
  missing.
- **kw-only lock** (K6.1): `pytest.raises(TypeError)` on positional `clause_is_reverse` for
  both filters.
- **Signature-builder equivalence** (K5): chunked vs loop-free equality.
- **SilverTorch candidates+qa raise** (K4.3.2).
- **`_forward_candidates` short-row semantics**: `SilverTorch._forward_candidates` with
  `P < k` returns `min(k, P)` columns today (no pad) — pin whichever K4.1 table row decides.

## Phase K9 — System-doc sweep (after code lands)

Fix [docs/system/architecture.md](../system/architecture.md) and
[docs/system/kernels.md](../system/kernels.md) in one pass against the refactored tree:

- `RetrievalModule` prose → matches K6.2's minimal ABC; delete `ScorerModule` mentions.
- `SimHashKNN` absent from architecture.md entirely (it ships, is exported, tested, and
  documented in [retrieve/docs/modules.md](../../retrieve/docs/modules.md)); add it + the
  `_PackedBitsKNN` note.
- `OneBitKNN.forward` documented as `(query, mask=None, candidate_ids=None)`; actual is
  `(query, candidate_ids=None, counts=None)` — the "masked path compacts first" story moved
  into the eval algos (linr_v3) long ago.
- `PrefilterKNN` "masked path: `compact_mask` → `fused_masked_knn_topk`" — the layer has no
  mask param; callers pass precompacted `(candidate_ids, counts)`.
- Compact-family return shape: docs say `positive_indices [B, P], P = max(counts.max(), 1)`;
  code returns full-width `[B, N]` with `-1` tails (Stage 2 change;
  [clause_compact.py:169-173](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L169-L173)).
  Same fix in the `compact_mask` helper section (returns full width, not `[:, :P]`).
- `fused_masked_knn_topk` I/O table says query/item_embs fp32; the PrefilterKNN path feeds
  fp16 (K2 adds the dtype assert that documents reality).
- `SilverTorch.forward` documented with a `mask=None` param it doesn't have; `filter` →
  `filter_mode` if K4.3.4 landed.
- `bloom_match` "no autotune today" note: keep, but add the rationale sentence (per-call width
  is dictated by N; a Config would be tuned against exactly one regime).
- While in there: [00-roadmap.md](00-roadmap.md) Stage 4 references `docs/thesis/*.md` files
  that no longer exist in the tree — update or annotate.

## Additional design improvements (catalogued; decide per item)

Beyond dedup — real design gaps found in the audit that deserve their own decisions. None
blocks K1–K9.

1. **State-dict round-trip doesn't work on fresh modules.** Every layer registers its buffers
   inside `register_index` (e.g. [postfilter_knn.py:21-24](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L21-L24),
   [silvertorch/main.py:152-164](../../retrieve/src/retrieve/layers/silvertorch/main.py#L152-L164)),
   so `SilverTorch(...).load_state_dict(saved)` on a fresh module fails (buffers don't exist),
   and even if they did, `SilverTorch._global_scale_f` — a plain Python float cached at
   register time to avoid a per-call `.item()` sync
   ([main.py:160-162](../../retrieve/src/retrieve/layers/silvertorch/main.py#L160-L162)) —
   would be stale/absent after a state-dict load. If checkpoint-shipping of built indexes is
   wanted (it is, for the serving story), add a `load_state_dict` post-hook
   (`register_load_state_dict_post_hook`) that re-derives `_global_scale_f` from the
   `global_scale` buffer, and either register empty buffers in `__init__` or document
   "call `register_index` before `load_state_dict`" as the contract. Decide when the
   checkpointing story matters (torch-export plan is the natural place).
2. **`KMeansTorch` is not k-means++** — the SilverTorch paper specifies "KMeans++-based
   training" (§3, citing Arthur & Vassilvitskii), but
   [kmeans.py:19-21](../../retrieve/src/retrieve/layers/utils/kmeans.py#L19-L21) seeds
   centroids with a plain `randperm` sample and runs Lloyd's. This is a *fidelity* gap, not a
   bug — but k-means++ init typically tightens cluster balance, which directly shrinks
   `max_cluster_size` and therefore `P = n_probe × max_cluster_size`, the codesigned kernel's
   scratch width. A ~15-line `_kmeanspp_init` (distance-weighted sampling, chunked) is cheap;
   measure `max_cluster_size` and probe-kernel latency before/after on goodreads/arxiv.
   Thesis-relevant: this is a claimable improvement with a one-figure ablation. (Also:
   `chunk = 1 << 14` at [kmeans.py:23](../../retrieve/src/retrieve/layers/utils/kmeans.py#L23)
   deserves a constructor knob.)
3. **`OneBitKNN.k_bits` sentinel mutation on re-register**
   ([one_bit_knn.py:63-67](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L63-L67)):
   after the first `register_index`, `k_bits=0` has been resolved to `D₁`; re-registering with
   a different-dim corpus silently keeps `D₁` and `quantize_oporp_1bit` then raises (or worse,
   D₂ % D₁ == 0 and it silently bins). Store the *resolved* value in a separate attr
   (`self._k_bits_resolved`) or reset the sentinel at the top of `register_index`. Tiny; fold
   into K4.2.
4. **Registered-index re-registration generally**: none of the layers guard double
   `register_index` (buffers get re-registered — works, but e.g. `PostfilterKNNInt8._n_real`
   and padding interact). Either document "register once" in the K6.2 ABC docstring (cheap,
   recommended) or add explicit re-registration support when
   [live-update-api.md](live-update-api.md) lands (it makes mutation first-class anyway).
5. **`_int_mm` minimum-M padding constant**:
   [postfilter_knn_int8.py:26](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L26)
   `_PAD_M = 17` encodes cuBLAS LtGemm's `M >= 17` int8 requirement with no comment linking to
   it (the doc note lives only in kernels.md). One comment line; fold into K6.3's pass.
6. **`combine_indices` device sync**: [filters/__init__.py:53](../../retrieve/src/retrieve/layers/filters/__init__.py#L53)
   does `int(new_counts.max().item())` per cascade stage — a documented sync point, fine for
   offline composition, but worth a docstring warning ("host sync per stage; don't put this in
   a cudagraph-captured path"). The eval harness never calls it in a hot loop (only linr algos
   do single-filter compacts); no code change.

## Verification

1. Full GPU suite at every phase boundary: `cd retrieve && uv run pytest tests/` — parity with
   **unchanged tolerances** (strict equality on OPORP/SimHash paths).
2. Perf gate (K2/K3): `uv run tune-kernels <kernel>` medians on shipped regimes within ±5%;
   K3's `clause_pass` regression triggers the documented per-kernel fallback.
3. Compile gate: `tests/compile/test_silvertorch_compile.py` + one eval smoke cell (`linr_v3`
   on a goodreads clause sweep) with `TORCH_LOGS=graph_breaks` — no new graph breaks from
   `masked_topk`, `_PackedBitsKNN`, or shared preps.
4. Export spot check (K2): `torch.export.export` a module whose forward calls
   `codesigned_probe_score_exact`, assert the exported program preserves the kernel reference
   (this is Invariant 2's regression test; add under `tests/compile/` until the export plan
   creates `tests/export/`).
5. Grep gates: no `from retrieve.layers.filters.bloom import _` anywhere; SWAR constants
   appear only in `kernels/common.py` + `layers/utils/quantize.py`; `simhash_knn.py` ≤ ~60
   lines; every `@triton_op` body contains exactly one `wrap_triton(`; `_tune_` prefix count
   in tune.py = 0 after K7 (replaced by `_sweep`).
6. Line-count sanity (leanness is the goal — verify it happened):
   `codesigned_probe_score.py` ≤ ~220, `tune.py` ≤ ~350, `bloom.py` ≤ ~100.

## Critical files

| File | Phases |
|---|---|
| [retrieve/src/retrieve/tune.py](../../retrieve/src/retrieve/tune.py) | K1 bugfix, K7 rework |
| [kernels/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py) | K2 (worked example), K3 |
| [kernels/silvertorch/codesigned_probe_score_exact.py](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) | K2, K3 |
| [kernels/filters/{clause_mask,clause_compact,bloom_compact}.py](../../retrieve/src/retrieve/kernels/filters/) | K2, K3 |
| [kernels/linr/{fused_masked_knn_topk,oporp_1bit_match_topk}.py](../../retrieve/src/retrieve/kernels/linr/) | K2, K3 |
| `kernels/common.py` | K3 — CREATE |
| `layers/utils/topk.py` | K4.1 — CREATE |
| `layers/linr/_bit_knn.py` | K4.2 — CREATE |
| [layers/linr/{one_bit_knn,simhash_knn}.py](../../retrieve/src/retrieve/layers/linr/) | K4.2 |
| [layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py) | K4.3, K5 |
| [layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) → + `bloom_hash.py` | K5 |
| [layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | K4.3.3, K6.1 |
| [src/retrieve/interfaces.py](../../retrieve/src/retrieve/interfaces.py) | K6 |
| [layers/utils/{quantize,retrieval,compact,kmeans}.py](../../retrieve/src/retrieve/layers/utils/) | K4.2, K4.4, K6.3, Additional-2 |
| `retrieve/tests/…` | K1, K4.1, K8 — new files |
| [docs/system/{architecture,kernels}.md](../system/) | K9 |

## Sequencing + effort

| step | phases | est. diff | risk |
|---|---|---|---|
| 1 | K1 | ±15 + test | none |
| 2 | K4.1 `masked_topk` + tests | +120 / −90 | low (behavior pinned per call site) |
| 3 | K2, one kernel file per PR (start `codesigned_probe_score.py`) | −400 net | medium on file 1 (tracing caveat), mechanical after |
| 4 | K3 + perf gates | −80 net | medium (`clause_pass` register pressure — fallback defined) |
| 5 | K4.2 / K4.3 / K5 (independent, any order) | −200 net | low |
| 6 | K6 (+ Additional 3/5 folded in) | ±60 | low; blocks resumption of export/live-update plans, so do before those |
| 7 | K7 | −300 net | low |
| 8 | K9 docs + Additional-item decisions | doc only | none |

After landing, trim this plan per repo convention ([00-roadmap.md](00-roadmap.md)
"Cleanup status").
