# `retrieve` library — architecture and code-design review, 2026-09-06

> **Status:** review only, no source changed. Branch `dev/integration` at `5cf009d`
> (= `development` + B1 official adapter + B5 salt buffer + A2 official pin). Read-only,
> CPU-only; every claim below was checked against the code, not the docs — where the
> two disagree that is itself a finding (§B.4). Numbers quoted from plans are
> **not yet validated** on the A100 (roadmap B2 pending). Reviewed against
> [kernels-layers-design.md](kernels-layers-design.md) K1–K9 (the previous refactor's
> intent) and [silvertorch-official-integration.md §5](silvertorch-official-integration.md#5-adapter-design)
> (the adapter design). Scope: `retrieve/src/retrieve/`, `retrieve/tests/`,
> `retrieve/pyproject.toml`, the four `docs/system` files the library owns.
>
> Line numbers are of `5cf009d`. "Now" = safe on the Mac today, GPU gate at B2;
> "B4" = fold into the CUDA/CuTe deletion; "G-a" = fold into the Triton transposed-bloom
> work; "C1" = coordinate with the harness rewrite.

## A. Executive summary — the five highest-value changes

The library is in good shape: K2 (prep/finish sharing), K4 (`masked_topk`,
`_PackedBitsKNN`), K6 (ABCs, keyword-only filter signature) and K7 (the tune registry)
landed as designed, the `@triton_op` bodies contain exactly one `wrap_triton` each, and
the official adapter is a clean, well-documented single module with its buffers, filter
matrix and eager-only contract exactly as §5 specified. The problems are at the edges:
what the docs and comments still say, what the candidates path does with `-1`, what a
state dict does not carry, and one fairness hole that would leak into citable numbers.

| # | change | effort | risk | when |
|---|---|---|---|---|
| 1 | **Make the official plan cache opt-in for timing.** `parse_plans` is `lru_cache`d on the whole expression tuple ([official.py:357-379](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L357-L379)); the harness times with `do_bench` over one fixed closure (`evaluation/retrieval/measure.py:146` (deleted by C3; v2 times in [bench.py](../../evaluation/retrieval/bench.py))), so after the first call the official bloom arm never pays the parse A3 measured at 58.7 µs/call (B=16) — 10–20 % of an eager bloom forward, silently excluded from B3/D1. Add `OfficialConfig.cache_plans: bool = True`; the harness's `official` cell passes `False` (or reports both, labelled). | 1 h | low | **before B3** |
| 2 | **Stale-record sweep + pin the bit order.** A3 is checked and measured HIGH-first with three probes, 19 launches / 3 syncs; yet `official.py:75-82` says "not yet measured", `kernels.md:931-945` says "the one unmeasured fact … ≈ 12 launches + 2 syncs", `architecture.md:410` repeats 12/2, `architecture.md:384` still defines `Backend` as three-valued, `kernels.md:152` counts eight tune subcommands (there are eleven), `modules.md:52` promises float32 scores from modules that return fp16. Rule 4 says docs move with code; these did not. Pin `OFFICIAL_BIT_ORDER = "high_first"` in `test_official.py:65` at the same time (T3 then asserts the constant on the A100 at B2). | 2 h | none | **now** |
| 3 | **Close the `-1` trap on the candidates paths.** `SilverTorch._forward_candidates` ([main.py:697-698](../../retrieve/src/retrieve/layers/silvertorch/main.py#L697-L698)) and `FullScanKNN._forward_candidates` ([retrieval.py:64](../../retrieve/src/retrieve/layers/utils/retrieval.py#L64)) index buffers with the raw ids; every compact producer in the library returns `-1`-tailed `[B, N]` ids, and these two `forward`s take no `counts`. A `-1` wraps to the last item (on `official`, through `inv_perm`), scores it, and returns it as a real id. Untested (`TestCandidates` never feeds a pad). Fix: `valid = ids >= 0`, gather via `clamp_min(0)`, `masked_topk(..., valid=valid, gather_ids=ids, pad_to_k=False)` — pure tensor flow, no sync. | 2 h + GPU test | low–med | **before D1** |
| 4 | **Make a loaded state dict usable.** `_global_scale_f` and `_max_cluster_size` are Python caches set only in `register_index` ([main.py:245](../../retrieve/src/retrieve/layers/silvertorch/main.py#L245), [:287](../../retrieve/src/retrieve/layers/silvertorch/main.py#L287)); `load_state_dict` leaves them stale. T6 proves it — the test patches both by hand ([test_official.py:691](../../retrieve/tests/parity/test_official.py#L691)). Register a `load_state_dict` post-hook that re-derives them from `global_scale` / `padded_cluster_items.shape[1]` (or `cluster_sizes.max()` on official — a sync at load time is fine) and delete the patch. | 2 h | low | **now** |
| 5 | **Give `Backend` its real shape at B4.** Today one five-valued literal is accepted by every constructor and silently means "torch" for four of them on three of its values; nothing validates it (`OneBitKNN(k, backend="foo")` runs). After B4 the honest types are `LinrBackend = Literal["torch","triton"]` and `SilverTorchBackend = Literal["torch","triton","official"]`, both validated in `__init__`, `SilverTorch.forward` dispatching through a table built once at construction instead of an `if` ladder ([main.py:425-439](../../retrieve/src/retrieve/layers/silvertorch/main.py#L425-L439)). The harness's LiNR cells must stop passing `"cuda"` (C1 already plans a `PATHS` table). | 0.5 d | med (harness cells) | **B4 + C1** |

Everything else in §B is smaller, and §D says where each one lands.

## B. Findings by area

### B.1 Architecture

**A1. Backend dispatch is coherent for `SilverTorch` and vacuous everywhere else.**
`Backend` ([interfaces.py:8](../../retrieve/src/retrieve/interfaces.py#L8)) is
`Literal["torch","triton","cuda","cute","official"]`. Only `SilverTorch.__init__`
validates it ([main.py:116-117](../../retrieve/src/retrieve/layers/silvertorch/main.py#L116-L117));
`_PackedBitsKNN`, `PrefilterKNN`, `PostfilterKNN*`, `BloomFilter`,
`ExactAttributeFilter` store it unvalidated and branch `== "triton"` else torch. The
docs describe this honestly (architecture.md "Backend dispatch"), and the harness
works around it for filters (`sweep.py:48`), but a typo is accepted and `"official"`
on a LiNR module is a silent torch run. *Recommendation:* see summary #5. Until B4,
add the two-line validation now (`if backend not in ("torch","triton"): raise
ValueError`) to the four LiNR/filter constructors **only if** the harness's cuda/cute
LiNR cells are confirmed to pass `"triton"` — otherwise wait for C1. *When:* B4 + C1.

**A2. What `SilverTorch` should look like after B4.** Deleting cuda/cute removes
[main.py:16-26](../../retrieve/src/retrieve/layers/silvertorch/main.py#L16-L26),
`:49`, `:314-339` (the `bloom_sigs_t` branch and its slack warning), `:427-430`,
`:511-591`. What remains is three backends with two index layouts (padded vs CSR) and
two filter-buffer sets. Sketch:

```python
# __init__
self._forward_impl = {
    "triton": self._forward_triton, "torch": self._forward_torch_eager,
    "official": self._forward_official,
}[backend]
# forward (after the candidates / filter_mode checks)
if self.backend == "official" and torch.compiler.is_compiling(): raise ...
return self._forward_impl(query, query_clause_attrs)
```

and merge `_register_filter_buffers` / `_register_official_filter_buffers` — their
`exact` branches are copies ([main.py:346-354](../../retrieve/src/retrieve/layers/silvertorch/main.py#L346-L354)
vs [:385-393](../../retrieve/src/retrieve/layers/silvertorch/main.py#L385-L393); the
only difference is the `[self.sort_perm]` permutation), so one method with
`perm: Tensor | None` covers both. `_TWO_KERNEL_BACKENDS` and `build_transposed_sigs`
go; G-a brings `bloom_sigs_t` back for Triton, at which point the "bloom index differs
per backend" note becomes "bloom index is transposed on triton, row-wise on torch" —
write it that way then, not now. *When:* B4.

**A3. The official adapter fits the design (§5.1) — with three seams.**
Buffers, order, filter matrix, sentinel handling (`indices == -1` → `valid` →
`masked_topk`, [official.py:545-555](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L545-L555))
and the eager-only guard (both `nn.Module.compile` override and `is_compiling`) are as
planned; the loader's three-way outcome (`OfficialMissing` / broken `ImportError` /
missing op) is the right contract and `require_official` uses it correctly.
Seams:
(a) `csr_from_assignments` ([official.py:243-258](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L243-L258))
is dead — `_build_ivf` re-derives `sort_perm`/`offsets` itself ([main.py:255-258](../../retrieve/src/retrieve/layers/silvertorch/main.py#L255-L258))
and `register_index` computes `inv_perm` inline (`:190-191`). Worse, the two disagree:
the adapter's version is `argsort(stable=True)`, the layer's is not. On CUDA int64
`argsort` is radix-sorted and stable in practice, but the layer's slot order (hence
`padded_cluster_items` and every tie-break) rests on an undocumented property.
*Fix:* delete `csr_from_assignments`, use `stable=True` in `_build_ivf`. Changes no
score, may permute ties; F3's provenance story wants this determinism anyway. *When:* now,
validate at B2.
(b) `pack_mask` ([official.py:393-406](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L393-L406))
materialises `padded.view(b, w, 64).to(int64) << shifts` — a `[B, N]` int64 tensor,
8× the mask, 1.3 GB at N = 10 M, B = 16, per forward, on the exact-on-official path.
Pack bytes instead:

```python
bytes_ = padded.view(b, w, 8, 8).to(torch.uint8)          # [B, W, byte, bit]
weights = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)  # high-first within a byte
packed = (bytes_ * weights).sum(-1, dtype=torch.uint8)     # [B, W, 8]; disjoint bits, no carry
return packed.flip(-1).contiguous().view(torch.int64).squeeze(-1)  # byte 0 → bits 63..56 (little-endian host)
```

peak `[B, N]` uint8. Bit-identical output; T3's round trips gate it. *When:* now,
validate at B2 (it is on the timed exact arm, so before B3).
Same family, torch fallback only: `compact_mask` ([compact.py:15](../../retrieve/src/retrieve/layers/utils/compact.py#L15))
casts the mask to fp32 before `argsort(stable=True)` — a `[B, N]` fp32 temp per call;
bool `argsort(stable=True)` works on torch 2.4 (checked on CPU here), so the cast can
go once B2 confirms it on CUDA.
(c) `queries_to_expressions(…, clause_is_reverse=None)` ([official.py:326-354](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L326-L354))
is never called with the second argument from the library (bloom mode rejects reverse
at `register_index`); only T4 uses it. Fine as a test seam — say so in the docstring
("library callers never pass it; NOT is exercised by T4 only").

**A4. Two knobs named `k_hash` and `hash_k` with different meanings.**
`SilverTorch.k_hash` is the official *search* `k` (≤ 10); `OfficialConfig.hash_k`
([official.py:147](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L147))
is the number of raw murmur hashes stored per term. The docstring explains it, the
names do not. Rename `OfficialConfig.hash_k` → `n_stored_hashes` (and `build_k` →
`n_build_bits`?) while B1 is still Mac-side and unpublished. *When:* now.

**A5. K2 prep/epilogue sharing: done, one leftover duplicate.** Every Triton file has
`_<name>_prep` + `_impl` + `@triton_op` with the launch inline; grep confirms one
`wrap_triton(` per op and `.stride(` only inside a `_prep` (the exception is
`bloom_match`, which K2.2 left without a prep on purpose). But `_CpsLaunch`/`_cps_finish`
([codesigned_probe_score.py:227-233](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L227-L233),
[:315-320](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L315-L320))
and `_CpseLaunch`/`_cpse_finish` ([codesigned_probe_score_exact.py:539-545](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py#L539-L545),
[:633-638](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py#L633-L638))
are byte-for-byte the same; K2.2 said "finish identical to `_cps_finish`" and meant
share it. Likewise the 3-D grid split (`tiles`, `tiles_x = cdiv(tiles, 65535)`,
`tiles_y`) is pasted in [clause_mask.py:115-119](../../retrieve/src/retrieve/kernels/filters/clause_mask.py#L115-L119),
[clause_compact.py:298-302](../../retrieve/src/retrieve/kernels/filters/clause_compact.py#L298-L302),
[bloom_compact.py:475-479](../../retrieve/src/retrieve/kernels/filters/bloom_compact.py#L475-L479).
*Fix:* a host-side `kernels/_host.py` (not `common.py`, which is `@triton.jit`-only by
its own docstring) with `ProbeLaunch`, `probe_finish(launch, k)` and
`grid_batch_tiles(b, n, block) -> (b, tiles_y, tiles_x)`. *When:* G-a (it rewrites
`codesigned_probe_score.py` anyway; doing it earlier means two perf gates).

**A6. `@triton_op` pattern and the tune registry (K7): correct.** All ten Triton ops
(seven files) are `triton_op` with `mutates_args=()`; the cuda/cute ops are `custom_op` +
`register_fake` (right — they are opaque), and they are the only reason
`test_export_kernel_ref.py` parametrises three backends. `KERNELS` has eleven specs;
`test_registry_covers_all_kernels` pins the list. After B4 both shrink to seven /
`("triton",)`. Two notes: `tune.py:37-47`, `:173-295`, `:441-529` are the deletion
list (matches O §7); and `bloom_match` still has no config/`_impl`/spec — K2.2 chose
that, keep it (§C).

**A7. `masked_topk` / `_PackedBitsKNN` sharing (K4): done, with one dtype seam.**
All torch-side epilogues route through `masked_topk`; the two bit-KNNs are 52 and 40
lines of subclass. But scores dtype differs by path: `PostfilterKNN` returns fp16
([postfilter_knn.py:32](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L32)),
`PostfilterKNNInt8` fp16 of `dots >> 5` ([postfilter_knn_int8.py:69](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py#L69)),
`PrefilterKNN` fp16 on the torch path ([prefilter_knn.py:65](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L65))
but fp32 on Triton and fp32 in both `p == 0` early returns (`:57-61`, `:77-82`).
`modules.md:52-53` promises `float32 scores` for all. Either cast to fp32 at the layer
boundary (cheap, `[B, k]`) or document per class. The `p == 0` block is also
duplicated between the two `PrefilterKNN` paths — hoist it into `forward`. *When:* now
(doc) / C1 (dtype, since the harness metrics read the scores).

**A8. `RetrievalModule` / `FilterModule` contracts are honoured.** All seven retrievers
subclass `RetrievalModule`; `FilterModule.register_index` is keyword-only on the ABC
and both subclasses (K6.1). `_PackedBitsKNN` uses `raise NotImplementedError` hooks
rather than `abc.abstractmethod` — acceptable (the base is private and never
instantiated), but `@abc.abstractmethod` costs nothing and documents intent.
`ExactAttributeFilter.register_index` does not `.long()` its attrs
([exact_attribute.py:47-48](../../retrieve/src/retrieve/layers/filters/exact_attribute.py#L47-L48))
while `BloomFilter` and `SilverTorch` do; an int32 attrs tensor compiles a second
`clause_mask` variant with a mixed-width compare. One `.long()`. *When:* now.

**A9. Public API surface vs the sdist docs.** `retrieve.__all__` lacks `OfficialConfig`
and `OfficialMissing` (users must know `retrieve.layers.silvertorch`), lacks
`compact_mask` (architecture.md:151-154 presents it as the user-facing mask→candidates
converter; it is in neither `retrieve.__all__`, `retrieve.layers.__all__`, nor any sdist
doc) and `FilterMode`. `kernels/linr/__init__.py:3-11` exports the private
`_fused_masked_knn_topk_impl` in `__all__`. `kernels/silvertorch/__init__.py`
re-exports ops under the *module* names (`codesigned_probe_score` is both a submodule
and an op) — the reason three conftest comments warn "import the full module path".
After B4 that package is three files / four ops; drop the re-exports (every consumer
already imports the submodule — verified by grep) and the footgun disappears. *When:*
`__all__` and the `modules.md` entry now; re-exports at B4.

### B.2 Code design and readability

**D1. The B5 salt buffer has one hole.** `SilverTorch._query_bits`
([main.py:463](../../retrieve/src/retrieve/layers/silvertorch/main.py#L463)) falls back
to on-the-fly salt when the index was registered without attributes, and
`generate_clause_salt` ([bloom_hash.py:56-61](../../retrieve/src/retrieve/layers/filters/bloom_hash.py#L56-L61))
still builds two `torch.tensor(_SALT, device=…)` scalars — i.e. exactly the per-call
H2D copy B5 removed, on that path. Register the salt lazily on first query instead
(or register a `[0]` buffer and derive the `[C]` one with `torch.arange` +
`_mix64` using Python-int constants: `h * c1` with a Python int is a device-side
scalar multiply, no upload). *When:* now, `test_bloom_hash.py` gates it.

**D2. Host syncs in hot paths — all accounted for, one worth a docstring.**
`quantize_int8_global` `.item()` and `_build_ivf` `.item()` are register-time;
`combine_indices` per-stage `.item()` is documented; `queries_to_expressions`
`.tolist()` is intrinsic to the official DSL and documented. `BloomFilter.register_index`
`clause_is_reverse.any().item()` is register-time. Nothing to change; add one line to
`SilverTorch._forward_official`'s docstring naming the `tolist()` as *the* adapter
sync so a profiler reader does not go looking.

**D3. Allocations in forward that could be buffers.** `_cps_prep` allocates two 1×1
int64 dummies per call on the no-bloom path ([codesigned_probe_score.py:272-273](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L272-L273)),
as does `_oporp_prep` (`:434-435`). Under `reduce-overhead` these are captured; eager,
they are two tiny `cudaMalloc`-pool hits per forward — measurable only at B=1. Reuse an
existing int64 tensor (`flat_probed_items`) as the dummy pointer instead — the kernel
never dereferences it under `HAS_QB=False`. *When:* G-a, if the B=1 eager row matters
for B3; otherwise leave.

**D4. `torch.compile` friendliness.** `P`, `D`, `W`, `N` are `tl.constexpr`; the
bucketing policy for the two LiNR indirect kernels is documented and correct; the
official backend is excluded by design. `_max_cluster_size` as a Python int is the
right call (the comment at [main.py:240-244](../../retrieve/src/retrieve/layers/silvertorch/main.py#L240-L244)
explains the SymInt hazard). No graph-break sources found beyond the documented ones.
Keep.

**D5. Duplicated numerics that must stay in lock-step.** The dequant expression
`dot.float() * q_scale * global_scale` (left-associated) appears in the Triton kernel
([codesigned_probe_score.py:217](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py#L217)),
the exact kernel, the eager path ([main.py:687](../../retrieve/src/retrieve/layers/silvertorch/main.py#L687)),
the candidates path (`:703`), the official epilogue ([official.py:552](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L552)),
and `ref_cps_phase23`. That is the bit-exact contract and it is correct in all six;
but it is enforced only by tests. Put one sentence in `kernels.md § Numerics` listing
the six sites, so a future "simplify" pass does not reassociate one of them.

**D6. Error handling.** No bare `except:`; the three `except Exception` sites are the
build/import memoisers and re-raise typed errors — right. Library-code `assert`s:
`main.py:347`, `:386` (type narrowing after validation — fine, comment says so);
`tune.py:372`, `:616` (CLI, fine); `codesigned_probe_score_cuda.py:247`, `:380` and
the cute twin (deleted at B4). No silent fallbacks other than A1.

**D7. `_forward_two_kernel(plain=, bloom=, exact=)`** passes three untyped callables
([main.py:539-554](../../retrieve/src/retrieve/layers/silvertorch/main.py#L539-L554)).
Deleted at B4; do not touch.

**D8. Naming (QuantizedIVF → SilverTorch).** Clean in the library: zero hits for
`QuantizedIVF`, `filter=` is gone, `FilterMode` is the alias. The only residue is
`cuda-silvertorch-handoff.md`, which B4 archives. `_cps`/`_cpse`/`_fmkt`/`_oporp`
prefixes are consistent. Keep.

### B.3 Tests

**T1. Layout matches `testing.md`;** three trees, one root gate, helpers where the doc
says. Parametrisation is right-sized: `test_silvertorch.py` crosses `BACKENDS` (5) ×
filter mode; `test_linr.py` crosses `["torch","triton"]`; `test_official.py` covers
T1–T7 as §5.2 lists. After B4 `BACKENDS` → `["torch","triton","official"]`,
`test_silvertorch_compile.py:33-48` `MODES` loses its twelve cuda/cute rows,
`test_export_kernel_ref.py:127` loses two, and two parity files + `parity/conftest.py:7,40`
(`build_transposed_sigs`, `make_bloom`'s `sigs_t`) go — O §7 lists all of them.

**T2. Gaps.** (a) No test feeds a `-1`-padded `candidate_ids` to `SilverTorch` or
`FullScanKNN` (summary #3). (b) No `load_state_dict`-then-forward test on a fresh
module for any layer — the one that exists (T6) patches the caches by hand. (c) No
test that an unknown `backend` string is rejected (there is nothing to reject, A1).
(d) `test_official.py` T6 transplants the Triton index into the official module
(`_transplant`) because "GPU k-means is not bit-deterministic" — that is the unstable
`argsort` (A3a) plus `torch.cdist` non-determinism; after A3a the transplant is still
needed for cdist, keep it but say which of the two it works around. (e) No test of the
LRU plan cache's *fairness* (summary #1): once `cache_plans` exists, T7's sync count
should be recorded with it off.

**T3. Slow / duplicated.** `assert_topk_matches` in `parity/conftest.py:144-195` and
`assert_topk_id_sets_match` in `conftest.py:191-234` are the same algorithm with a
different signature (per-row vs whole tensor). Fold the latter into a loop over the
former. Nothing else stood out; the suite is 6.8 k lines for 8.1 k of source, which is
proportionate.

**T4. `OFFICIAL_BIT_ORDER = None`** ([test_official.py:65](../../retrieve/tests/parity/test_official.py#L65))
runs T3 over both orders. A3 measured HIGH-first; the roadmap's B1 status note says
"not pinned yet". Pin it now (summary #2) — the adapter constant is already HIGH-first,
so pinning changes what the test *asserts*, not what runs.

### B.4 Bad patterns, dead code, stale text

No mutable defaults, no TODO/FIXME anywhere in `src/`. Stale text is the real list:

| where | says | truth | fix |
|---|---|---|---|
| [official.py:75-82](../../retrieve/src/retrieve/kernels/silvertorch/official.py#L75-L82) | "Read from the source, not yet measured: roadmap A3 … probes it" | A3 done, HIGH-first measured (roadmap A3 record, O §13.2) | "Measured 2026-09-06 (O §13.2): high-first" |
| [kernels.md:931-937](../system/kernels.md) | "Bit order — the one unmeasured fact … Roadmap A3 measures" | same | rewrite paragraph |
| [kernels.md:945](../system/kernels.md), [architecture.md:410](../system/architecture.md) | "≈ 12 launches + 2 syncs" | 19 launches / 3 syncs, bloom ≈ 32 / ≥ 5 (O §13.2) | update both |
| [architecture.md:384](../system/architecture.md) | `Backend = Literal["torch","triton","cuda"]` | five-valued (`interfaces.py:8`) | fix; re-fix at B4 |
| [kernels.md:152-158](../system/kernels.md) | "eight click subcommands" + list of 8 | eleven (`test_tune_smoke.py:17-29`) | "eleven" now, "seven" at B4 |
| [modules.md:52-53](../../retrieve/docs/modules.md) | "`[B, k] float32 scores`" | fp16 from `PostfilterKNN`, `PostfilterKNNInt8`, `PrefilterKNN` (torch path) | per A7 |
| [testing.md:168-170](../system/testing.md) | parity conftest "Three helpers" | six (`make_probe_family`, `make_bloom`, `make_exact` omitted) | list them; three go at B4 |
| [kernels.md:855-866](../system/kernels.md) | "Status: authored … GPU gate has not run yet" | still true — keep, but it will rot at B2 | update at B2 |
| [main.py:46-48](../../retrieve/src/retrieve/layers/silvertorch/main.py#L46-L48) comment | "CuTe DSL backend is a port…" | true | deleted at B4 |

Dead code: `csr_from_assignments` (A3a); `pack_mask_high_first` (a plan-name alias,
unused anywhere — delete, the plan can cite `pack_mask`); `bloom_index_docs`,
`unpack_mask`, `unpack_partial_mask`, `reverse_bits64`, `official_scores_full`,
`padded_rows`, `bloom_full_mask` are test-only — keep them, but move them below a
`# --- test support ---` rule in `official.py` so the 737-line module reads as
"adapter (≈ 450) + probes (≈ 250)". `SilverTorch.has_bloom` / `has_exact` are
properties of a frozen `filter_mode` — fine.

## C. Keep as is (looks odd, is right)

- **`P`, `N`, `D`, `W` as `tl.constexpr`** and the two bucketing ladders: one JIT
  variant per index shape is the deployment model; the module docstrings explain the
  policy and `_bucket_p`/`_bucket_n` keep the parity sweeps from compiling per width.
- **`bloom_match` has no `_prep`/`_impl`/tune spec**: K2.2 decided that for an 80-line,
  one-op file; adding scaffolding would be churn.
- **`fused_masked_knn_topk`'s public op skips bucketing and the pad tail** while
  `_impl` does both: intentional, documented, `PrefilterKNN` guarantees `p ≥ k > 0`.
- **`_max_cluster_size` / `_global_scale_f` as Python scalars**: the SymInt / sync
  reasoning in the comments is correct; the fix is the load hook (#4), not a buffer.
- **`evaluate_indices` returning full-width `[B, N]`** with `-1` (kernels) or arbitrary
  (`compact_mask`) tails: the `counts`-bounded contract is stated in all three places.
- **`PostfilterKNNInt8`'s `>> 5` and `_PAD_M = 17`**: both are cuBLAS constraints,
  both commented, K6.3 already qualified the tie wording.
- **`FullScanKNN`'s post-filter semantics** (tombstone, no backfill): the LiNR baseline,
  documented loudly per K4.4.
- **`combine_indices`' per-stage `.item()`**: offline composition, docstring warns.
- **`KMeansTorch` is Lloyd's, not k-means++**: a paper-fidelity deviation that belongs
  in F1's deviations table, not a code change before the campaign.
- **`SilverTorch.compile()` override plus the `is_compiling()` check**: two guards for
  two entry points (`module.compile()` vs `torch.compile(module)`); both are needed.
- **`OfficialConfig` frozen dataclass with `__post_init__` validation** and
  `DEFAULT_CONFIG` as a module constant: matches the kernel `Config` pattern.
- **`official.py` keeping plans on CPU** and the `[B, P]` `max_tensor_size_per_row`
  choice: both argued in the docstrings and in O §3/§4.

## D. Sequencing against the roadmap

**Now, on the Mac, before B2 (so the B2 gate validates them):** summary #2 (docs sweep
+ pin bit order), #4 (load hook), A3a (`stable=True`, delete `csr_from_assignments`),
A3b (`pack_mask` bytes), A4 (`hash_k` rename), A8 (`.long()`), A9 `__all__`, D1 (salt
fallback), T3 (helper fold), B.4 dead-code moves. One commit per item; all are
`ruff`-checkable and collect-only on the Mac.

**Before B3 (fairness):** summary #1 (`cache_plans`), with the harness's official cell
and T7's sync count updated together.

**Before D1 (campaign):** summary #3 (`-1` on candidates), with a GPU test in
`TestCandidates` and `test_retrieval_utils.py`.

**Fold into B4 (deletion):** summary #5 (`Backend` split + validation, coordinated with
C1's `PATHS` table), A2 (dispatch table, merged filter-buffer registration), A9
re-exports, the test and doc rows in T1/B.4, and re-fixing the two doc lines that #2
sets to "five"/"eleven" down to "three"/"seven".

**Fold into G-a (Triton transposed bloom):** A5 (`_host.py` shared launch/finish/grid),
D3 (dummy pointers), and the `bloom_sigs_t`-on-Triton wording from A2 — all touch
`codesigned_probe_score.py`, which G-a rewrites; doing them once avoids two ±5 % perf
gates.

**Parked plans:** [torch-export-refactor.md](torch-export-refactor.md) assumes
`custom_op` and a `torch_knn` algo — its remaining live question is the Optional
`query_clause_attrs` / `candidate_ids` forward signature; #3 does not change that.
[live-update-api.md](live-update-api.md) needs #4 first (a mutated index must
re-derive the caches). Neither is unblocked by this review; both are re-scoped after B4
as the roadmap says.

## E. Applied 2026-09-06 (`dev/integration`, Mac, CPU-only)

Five commits on top of `12e3927`, one per item, each gated by `ruff check` /
`ruff format --check` (clean apart from the pre-existing `tests/correctness/test_linr.py`
formatting), `pytest tests/ --collect-only` (683 → 699 collected) and
`scripts/check_doc_links.py` (0 broken). **Every GPU claim below is unverified until B2
runs the suite on the A100**; the only execution here was CPU-side smoke checks of the
device-agnostic logic (the load hook and the `-1` candidates semantics on the torch
backend, the salt constants bit-for-bit).

| item | commit | what landed |
|---|---|---|
| #2 docs sweep + bit-order pin | `1ee2f0a` | every row of §B.4's table except the two marked "at B2/B4"; `OFFICIAL_BIT_ORDER = "high_first"` pinned as a test-side constant, T3 asserts the adapter constants against it and keeps `"low_first"` as a negative control — the both-orders parametrisation is gone (683 → 678 tests) |
| #4 load hook | `ee9c339` | `_rederive_cached_scalars` registered as a `load_state_dict` post-hook in `SilverTorch.__init__`; T6's hand patch deleted; new `TestStateDict` (every backend × filter mode) loads a zeroed, cache-poisoned deep copy — a deep copy because GPU k-means is not bit-deterministic and `load_state_dict` checks buffer shapes |
| #3 `-1` on candidates | `37cef19` | both `_forward_candidates` go through `masked_topk(valid=ids >= 0, gather_ids=ids, pad_to_k=False)` with a `clamp_min(0)` gather; `masked_topk` unchanged; bit-identical for all-valid inputs (checked on CPU); GPU tests in `TestCandidates` and `test_retrieval_utils.py`; done now rather than "before D1" |
| #1 `cache_plans` | `55ebb41` | `OfficialConfig.cache_plans: bool = True`, `parse_plans(..., cache=)`; the timing rule in `official.py`, kernels.md, architecture.md and one paragraph in evaluation.md — the harness cell itself is roadmap C4's; T4 covers the uncached path, T6 runs the partial-bloom module with the cache off, T7 records its sync count |
| A3(a), A4, D1, B.4 dead code | `cf1d42b` | `csr_from_assignments` and `pack_mask_high_first` deleted; `OfficialConfig.hash_k` → `n_stored_hashes`; `generate_clause_salt` uses Python-int constants so the attribute-less bloom fallback copies nothing; A3(c)'s docstring note; T2(d)'s `_transplant` wording |

**Deliberately not applied, with the reason.**

- A3(a)'s second half, `argsort(stable=True)` in `_build_ivf`: the dead helper was
  deleted instead. A stable sort could permute tie order in `padded_cluster_items`
  against the checkpoints and golden outputs that exist today; the change is safe only
  when B2 can re-run the parity gate on it. Reconsider at B2 with F3's provenance
  argument. **Applied at B2 (`0521a67`)** after measuring it bit-identical — permutation,
  padded layout, forward ids and scores on all three backends — on nine regimes
  (O §14.6).
- The rest of §D's "now" list was outside this pass and is still open: A3(b)
  (`pack_mask` byte packing — on the timed exact arm, so before B3), A7's `modules.md`
  fix landed but the fp32 boundary cast is C1's, A8 (`.long()` in
  `ExactAttributeFilter.register_index`), A9 (`__all__` exports), T3 (helper fold),
  B.4's `# --- test support ---` move in `official.py`, D2's docstring line, D5's
  numerics sentence in kernels.md.

**Deferred as the review sequences them.** B2: the GPU validation of everything above
plus kernels.md's "Status: authored … GPU gate has not run yet". B4: #5 (`Backend`
split + validation, dispatch table, merged filter-buffer registration), A9 re-exports,
the T1/B.4 test and doc rows, and shrinking "five"/"eleven" to "three"/"seven". G-a: A5
(`_host.py`), D3 (dummy pointers), the `bloom_sigs_t`-on-Triton wording. C1/C4: the
harness's `official` cell passing `cache_plans=False` (or both rows), the LiNR cells'
`"cuda"` → `"triton"`, A7's score dtype at the layer boundary.

**Applied at B4 (2026-09-06, `dev/b4-delete-cuda-cute`, CPU-only; GPU suite pending).** #5
(`LinrBackend` / `SilverTorchBackend` + `check_backend` in every constructor, 18 rejection
cells in `test_linr.py`), A2 (table dispatch in `SilverTorch.__init__`, merged
`_register_filter_buffers(…, perm=)`), A9 re-exports (the kernel package `__init__` is
docstring-only), the T1 / B.4 test and doc rows, "five"/"eleven" → "three"/"seven". D7's
`_forward_two_kernel` went with the backends. Record: O §15.
