# Migrate clean triton kernels to `torch.library.custom_op` to recover single-cudagraph capture

## Context

Every eval-side Algo (`linr_v1`, `linr_v2`, `linr_v3`, `silvertorch`) is an `nn.Module` that calls `self.compile(dynamic=True, mode="reduce-overhead")` in `__init__`. Each algo forward (filter + index + cascade) is supposed to capture as a single cudagraph_trees graph.

Today every triton kernel host wrapper carries `@torch._dynamo.disable`, which works around a known dynamo bug (`AssertionError: Cannot construct ConstantVariable for value of type torch.SymInt` inside `_method_size_stride` when `tensor.stride(i)` is passed as a kwarg under `dynamic=True`). The workaround forces a graph break per kernel call, fragmenting the cudagraph capture. Measured cost on the `linr_v1+clause` path: **0.314 ms/call (disabled) vs 0.170 ms/call (no graph break)** — ~85% slower.

This change replaces `@torch._dynamo.disable` with a `torch.library.custom_op` registration on three "clean" kernels whose output shapes are fully derivable from input shapes. Dynamo then models them as opaque ops (no graph break, single cudagraph stays intact) without needing to trace into them or into `.stride()`. The four remaining kernels (data-dependent output shapes or `Optional[Tensor]` args) keep `@torch._dynamo.disable` — they need caller-side refactors that are out of scope.

## Decision: `custom_op` over `triton_op`

`torch.library.triton_op` (torch ≥ 2.4) is the more structured choice and explicitly handles `@triton.autotune` via `wrap_triton`. However, reading `torch/_library/triton.py:200-202` reveals:

```python
# We require that the user pass us a function that is make_fx traceable,
# so we can just register it as the Fake/meta kernel.
result.register_fake(fn)
```

`triton_op` auto-registers the user's `fn` itself as the fake/meta impl. Under `dynamic=True` compile, the fake runs with FakeTensors that carry SymInt shapes. Two of our three kernel wrappers branch on shape ints at the host:

- `bloom_match`: `block_n = 128 if n >= 128 else triton.next_power_of_2(int(n))`
- `fused_masked_knn_topk`: `if p == 0: return (...)`

Branching on a SymInt either triggers an unwanted specialization guard or fails outright. The `triton_op` body is also re-traced under `FunctionalTensorMode` for AOTDispatcher decomposition (`torch/_library/triton.py:209-251`), repeating the same exposure.

`custom_op` treats the op as fully opaque — no fake re-tracing of the body. We supply an explicit `register_fake` that just allocates the output tensors. The body runs eagerly inside the op boundary on real tensors, where the shape branches are concrete Python ints and work as written. This matches the "opaque-but-not-graph-breaking" intent exactly and avoids any subtle dynamic-shape interaction in the kernel bodies.

Trade-off: Inductor can't inline/fuse the kernel into surrounding ops. For these three kernels there's nothing meaningful to inline (the kernel itself is the work, and the post-kernel `topk`/`gather`/`where`/`cat` in `fused_masked_knn_topk` was already opaque under `@torch._dynamo.disable`), so we lose nothing relative to the baseline.

## Files to modify (3 kernels — decorator + register_fake only)

### 1. [clause_mask.py](retrieve/retrieve/src/retrieve/kernels/triton/filters/clause_mask.py)

- Add `from torch.library import custom_op` import.
- Replace `@torch._dynamo.disable` (line 76) with `@custom_op("retrieve::clause_mask", mutates_args=())`.
- Keep the existing function body as-is (the kernel launch, the input validation, the `.contiguous()`/`.to(int8)` casts, all the `stride(i)` kwargs — all of it lives inside the now-opaque op so dynamo never sees `.stride()`).
- Register the fake right after the function:
  ```python
  @clause_mask.register_fake
  def _(item_clause_attrs, clause_is_reverse, query_clause_attrs):
      n = item_clause_attrs.shape[0]
      b = query_clause_attrs.shape[0]
      return torch.empty((b, n), dtype=torch.bool, device=query_clause_attrs.device)
  ```
- Existing type annotations (`Tensor`, returns `Tensor`) are sufficient for schema inference.

### 2. [bloom_match.py](retrieve/retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py)

- Same recipe. Register name: `retrieve::bloom_match`.
- Fake:
  ```python
  @bloom_match.register_fake
  def _(qb, sigs):
      b = qb.shape[0]
      n = sigs.shape[0]
      return torch.empty((b, n), dtype=torch.bool, device=qb.device)
  ```
- Existing annotations (`qb: Tensor, sigs: Tensor) -> Tensor`) are sufficient.

### 3. [fused_masked_knn_topk.py](retrieve/retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py)

- Same recipe. Register name: `retrieve::fused_masked_knn_topk`.
- Annotation note: signature is `(query: Tensor, item_embs: Tensor, positive_indices: Tensor, counts: Tensor, k: int) -> tuple[Tensor, Tensor]`. Confirm `tuple[Tensor, Tensor]` resolves through `infer_schema` on torch 2.10; if not, fall back to `Tuple[Tensor, Tensor]` from `typing`. (Schema inference at `torch/_library/infer_schema.py:122` requires every parameter be annotated — they all are.)
- Fake (the wrapper always pads/truncates to `k`, so the fake mirrors that):
  ```python
  @fused_masked_knn_topk.register_fake
  def _(query, item_embs, positive_indices, counts, k):
      b = query.shape[0]
      device = query.device
      ids = torch.empty((b, k), dtype=torch.long, device=device)
      scores = torch.empty((b, k), dtype=torch.float32, device=device)
      return ids, scores
  ```
- `@triton.autotune` on the inner kernel is untouched — `custom_op` doesn't care about the kernel body, only the wrapper signature.

### Files NOT to touch

- `bloom_compact.py`, `clause_compact.py`, `oporp_1bit_match_topk.py`, `codesigned_probe_score.py` — keep `@torch._dynamo.disable`. Out of scope (data-dependent shapes or `Optional[Tensor]` args).
- Any `@triton.jit` body. Any algo wrapper in `evaluation/retrieval/algos/`.
- The algo-level `self.compile(dynamic=True, mode="reduce-overhead")`.

## Verification

### Step 1 — Pre-change baseline (must run before any edit)

Write `bench_kernel_migration.py` at `/workspace/retrieve/`. For each of the 3 target paths below, construct the algo on synthetic GPU tensors, run 20 warmup iters, then time 200 iters with `torch.cuda.synchronize()` boundaries and record the median ms/call:

- **clause_mask path**: `LinrV1Algo(item_embs, k=100, filter_mod=ExactAttributeFilter(...), backend="triton")`. Constructor and registration mirror the eval harness (`evaluation/retrieval/algos/__init__.py:60-116`).
- **bloom_match path**: `LinrV1Algo(item_embs, k=100, filter_mod=BloomFilter(m_bits=1024, k_hash=5, ...), backend="triton")`. (LinrV1 uses the *dense* mask path, which calls `bloom_match`. LinrV2/V3 use the sparse `bloom_compact` path, out of scope.)
- **fused_masked_knn_topk path**: `LinrV2Algo(item_embs, k=100, filter_mod=BloomFilter(...), backend="triton")`. PrefilterKNN inside LinrV2 calls `fused_masked_knn_topk` regardless of which filter kind feeds it.

Reuse `measure_forward_cuda()` from [evaluation/retrieval/bench_tools.py](retrieve/evaluation/retrieval/bench_tools.py). Synthetic data: ~100k items, D=64, batch_size=32, plausible attr/bloom shapes.

Save the three median numbers to a JSON sidecar (`bench_before.json`) so the after-comparison is mechanical.

### Step 2 — Apply the 3-file edit

### Step 3 — Post-change measurement

Re-run the same bench, save to `bench_after.json`, print a table with `before`, `after`, `delta_ms`, `delta_pct`. Expected: ~40-90% reduction on each path. If any of the three regress or move <10%, stop and diagnose — likely a graph break still exists or the fake shape is wrong.

### Step 4 — Cudagraph capture check

Around one bench iteration each, run with:
```python
import torch._inductor.config
torch._inductor.config.triton.cudagraph_trees = True
torch._logging.set_logs(inductor=logging.INFO, dynamo=logging.INFO)
```
Capture stderr. Grep for `skipping cudagraphs due to mutated inputs` and `graph break` referencing the migrated kernels. Today these appear; after the migration they should be absent on the three migrated paths (the four still-disabled kernels will still produce them — expected).

### Step 5 — Correctness

1. **Existing parity tests** — `cd retrieve && uv run pytest tests/ -v` must pass 242/242. The parity files [tests/parity/test_clause_mask.py](retrieve/retrieve/tests/parity/test_clause_mask.py), [tests/parity/test_bloom_match.py](retrieve/retrieve/tests/parity/test_bloom_match.py), [tests/parity/test_fused_masked_knn_topk.py](retrieve/retrieve/tests/parity/test_fused_masked_knn_topk.py) (and `test_bloom_compact.py` which calls `bloom_match` as the reference oracle) directly cover the three migrated kernels.

2. **Fake-impl shape parity check** — call each new op once on real tensors and assert `out.shape == expected_shape` and `out.dtype == expected_dtype` and `out.device == expected_device`. This catches a wrong `register_fake` *eagerly* (before compile-time symbolic-shape divergence makes it a silent recall@k regression).

3. **Algo × backend × filter smoke** — programmatic loop over all combinations from [evaluation/retrieval/algos/__init__.py](retrieve/evaluation/retrieval/algos/__init__.py) `ALGORITHMS = ("torch_knn", "triton_knn", "linr_v1_filter_mask", "linr_v3", "linr_v2", "silvertorch")` × backend `("torch", "triton")` × filter_kind `("none", "bloom", "clause")`. Skip combos the eval framework already rejects (silvertorch+clause, linr_v2+none). For each combo: build the algo, run one forward, assert finite scores and id range. Build the filter via `build_filter()` from [evaluation/retrieval/sweep.py:223-288](retrieve/evaluation/retrieval/sweep.py#L223-L288) so the wiring matches production.

4. **Top-K parity against eager reference** — for one query on the `LinrV2 + bloom` path (the most complex, involves `fused_masked_knn_topk`'s tuple return and topk/gather/where/cat tail), build the compiled algo and a parallel uncompiled instance with the same weights; assert `compiled_ids == eager_ids` and `torch.allclose(compiled_scores, eager_scores, rtol=1e-5)`. This is the safety net the user explicitly called out: a wrong `register_fake` for `fused_masked_knn_topk` would silently shift top-K under compile.

## Report at end

- Per-kernel: confirmation that `custom_op` was used (and why `triton_op` was rejected, per the Decision section).
- Before/after timing table for the 3 algo paths with delta_pct.
- Confirmation that `skipping cudagraphs`/`graph break` warnings are absent on the migrated paths.
- `pytest tests/ -v` result (must be 242/242).
- Smoke-test result for all valid algo × backend × filter combos.
- Top-K parity assertion result on `LinrV2 + bloom`.

## Risks and how the verification catches them

- **Wrong fake shape → silent recall@k regression**: caught by Step 5.2 (eager shape check) and Step 5.4 (top-K parity).
- **Type annotation rejected by `infer_schema`**: caught immediately at import — the decorator throws. Fix is in the kernel file, in scope.
- **Decorator doesn't actually eliminate the graph break (e.g., dynamo still sees inner `.stride()` somehow)**: caught by Step 4 (warning grep) and Step 3 (the timing won't improve).
- **`tuple[Tensor, Tensor]` annotation form not handled on torch 2.10**: low probability (`infer_schema` supports modern PEP 585 generics) but a 1-line fallback to `typing.Tuple` is the entire fix. Decided in the edit phase by reading the error message.
