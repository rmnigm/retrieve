# Fix `SilvertorchAlgo` to propagate `clause_is_reverse`

## Context

`SilverTorch` (the library-side IVF + INT8 + fused-filter retriever, [retrieve/src/retrieve/layers/silvertorch/main.py:27](../../../retrieve/src/retrieve/layers/silvertorch/main.py#L27)) **already supports reverse clauses** in its exact-filter mode. The codesigned exact-clause kernel evaluates the AND-of-OR predicate and XORs the per-clause result with a `clause_is_reverse[c]` flag inline ([codesigned_probe_score_exact.py:99](../../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py#L99) — `clause_match = clause_match ^ rev_c`); `SilverTorch.register_index` accepts a `clause_is_reverse: Tensor | None = None` argument ([main.py:132](../../../retrieve/src/retrieve/layers/silvertorch/main.py#L132)) and registers it as a buffer ([main.py:220](../../../retrieve/src/retrieve/layers/silvertorch/main.py#L220)); library-side correctness tests already exercise the path ([retrieve/tests/correctness/test_silvertorch.py:70-144](../../../retrieve/tests/correctness/test_silvertorch.py#L70-L144)).

The eval-side wrapper does NOT thread `clause_is_reverse` through. `SilvertorchAlgo.__init__` ([evaluation/retrieval/algos/silvertorch.py:73, 89](../../../evaluation/retrieval/algos/silvertorch.py#L73-L89)) calls `SilverTorch.register_index(item_embs, item_clause_attrs=item_attrs_narrow)` with no `clause_is_reverse=`. Per `main.py:215-218`, this defaults the buffer to all-zeros (no reverse) — so every clause is treated as a positive equality predicate. On Goodreads, c1 is a reverse-clause ("not equal to query language"); silvertorch evaluates it as the OPPOSITE predicate and matches almost no SASRec-relevant items, producing Recall@100 in the range [0.0005, 0.0425] on `c1_lang_reverse / c0c1 / all4` sweeps across d64 / d128 / d256.

The eval driver already loads `clause_is_reverse` from disk ([loaders.py:296-322](../../../evaluation/retrieval/loaders.py#L296-L322)), threads it into `build_filter` ([sweep.py:256, 265, 279](../../../evaluation/retrieval/sweep.py#L256-L279)), but does NOT thread it through `_build_filter_modules`'s return value or into `build_algorithm`. Fixing this is a wiring change only — no library, kernel, config, dataset, or oracle changes are required.

The registry docstring at [evaluation/retrieval/algos/__init__.py:21-24](../../../evaluation/retrieval/algos/__init__.py#L21-L24) claims "Silvertorch on reverse-clause sweeps is skipped by the driver itself (evaluate.py), not here." This claim is stale: no such skip exists in the current driver, and the empirical data confirms silvertorch DOES run on reverse-clause sweeps. The docstring is updated to reflect the new behavior (silvertorch supports reverse-clause sweeps end-to-end via this plan; no skip required).

Why this matters for the thesis: three of the six Goodreads filter sweeps (`c1_lang_reverse`, `c0c1`, `all4`) contain c1 and so are unusable for silvertorch comparison today. After this fix and a rerun, all six sweeps become legitimate silvertorch cells, restoring the §6.2 algorithm matrix on Goodreads.

## Out of scope

- **Library / kernel changes** — `SilverTorch.register_index` and the codesigned exact-clause kernel already handle `clause_is_reverse` correctly; no edits to `retrieve/src/retrieve/layers/silvertorch/` or `retrieve/src/retrieve/kernels/silvertorch/`.
- **Filter primitives** — `ExactAttributeFilter` already consumes `clause_is_reverse` via `build_filter`; LinR algos (`linr_v1`/`v2`/`v3`/`v4`) consume the resulting mask through `filter_mod.evaluate_mask(qa_narrow)` — they already have correct reverse semantics. No changes to the LinR algo wrappers.
- **Bloom mode** — `BloomFilter` is paper-strict forward-only and the library raises on `clause_is_reverse` with `filter="bloom"` ([main.py:140-141](../../../retrieve/src/retrieve/layers/silvertorch/main.py#L140-L141)). Reverse clauses with bloom are out of scope (and configs already omit them, e.g. [evaluation/config/goodreads/d128-filter.yaml:36-44](../../../evaluation/config/goodreads/d128-filter.yaml#L36-L44)).
- **Stale oracle disk cache for goodreads-d128/d256 filter** — separate issue, separate remediation (delete cache + rerun; see ⚠ block in `docs/thesis/07-results.md` (deleted)). The wrapper fix here is necessary but not sufficient: even with the wrapper fixed, silvertorch on c1_lang_reverse will still be scored against the stale ground truth unless the oracle is rebuilt.
- **d64 numbers for arXiv** — arXiv has no reverse clauses in any of its narrow attribute schema, so silvertorch on arxiv is already correct. No data regen needed for arxiv after this fix.

## Changes

### `evaluation/retrieval/algos/silvertorch.py` — accept and forward `clause_is_reverse`

**`SilvertorchAlgo.__init__` ([silvertorch.py:33-47](../../../evaluation/retrieval/algos/silvertorch.py#L33-L47)):** add a new kwarg `clause_is_reverse: Tensor | None = None` after `item_attrs_narrow`. The clause-mode branch ([silvertorch.py:74-89](../../../evaluation/retrieval/algos/silvertorch.py#L74-L89)) forwards it to `SilverTorch.register_index`:

```python
self.idx.register_index(
    item_embs,
    item_clause_attrs=item_attrs_narrow,
    clause_is_reverse=clause_is_reverse,
)
```

The bloom branch ([silvertorch.py:56-73](../../../evaluation/retrieval/algos/silvertorch.py#L56-L73)) explicitly does NOT forward `clause_is_reverse` — bloom mode raises on it at the library level. If `filter_kind="bloom"` and `clause_is_reverse is not None` (i.e., reverse columns present in the loaded asset tensor), this is fine: bloom configs only iterate over forward-clause sweeps (the YAML enforces it), and `clause_is_reverse` is the full per-clause flag tensor — reverse columns are simply unread by the bloom path. No new validation needed.

Update the module docstring ([silvertorch.py:1-18](../../../evaluation/retrieval/algos/silvertorch.py#L1-L18)) to remove the line "A pre-existing 'post-mask IVF' composition was rejected..." and add one paragraph: "When `filter_kind="clause"`, the wrapper threads `clause_is_reverse` into `SilverTorch.register_index`, so sweeps with reverse predicates (e.g., Goodreads `c1_lang_reverse`) are evaluated correctly. Without this kwarg the codesigned exact-clause kernel would treat every clause as a positive equality, collapsing recall to ≈ 0 on reverse-clause sweeps."

### `evaluation/retrieval/algos/__init__.py` — thread the kwarg through `build_algorithm`

**`build_algorithm` signature ([algos/__init__.py:62-72](../../../evaluation/retrieval/algos/__init__.py#L62-L72)):** add `clause_is_reverse: Tensor | None = None` after `item_attrs_narrow`. Forward to the silvertorch branch:

```python
if name == "silvertorch":
    return SilvertorchAlgo(
        item_embs,
        k,
        filter_kind=filter_kind,
        item_attrs_narrow=item_attrs_narrow,
        clause_is_reverse=clause_is_reverse,   # NEW
        n_lists=int(p.get("n_lists", 1024)),
        ...
    )
```

Other branches ignore the kwarg silently. Drop the stale comment at line 22-24 ("Silvertorch on reverse-clause sweeps is skipped by the driver itself") and replace with: "Silvertorch with `filter_kind='clause'` now accepts `clause_is_reverse` and supports reverse-clause sweeps end-to-end via the fused codesigned exact-clause kernel."

Module docstring ([algos/__init__.py:18-19](../../../evaluation/retrieval/algos/__init__.py#L18-L19)): update the ```filter_kind``/``filter_mod``/``item_attrs_narrow``` enumeration to also mention `clause_is_reverse`.

### `evaluation/retrieval/sweep.py` — thread `clause_is_reverse` from `_build_filter_modules` to `_try_build_algo`

The driver already loads the tensor ([sweep.py:256](../../../evaluation/retrieval/sweep.py#L256)) but drops it on the floor from `_build_filter_modules`'s return.

**`_build_filter_modules` return ([sweep.py:230-295](../../../evaluation/retrieval/sweep.py#L230-L295)):** add `clause_is_reverse` as a fifth tuple element:

```python
return filter_mods, oracle_filter, item_attrs_narrow, clause_is_reverse, n_clauses
```

Adjust the type annotation in the signature.

**`run_filter_kind` ([sweep.py:128-198](../../../evaluation/retrieval/sweep.py#L128-L198)):** unpack the new return tuple at line 156 and pass `clause_is_reverse` into `run_one_sweep` at the call site around line 164.

**`run_one_sweep` signature ([sweep.py:301-394](../../../evaluation/retrieval/sweep.py#L301-L394)):** add `clause_is_reverse: torch.Tensor | None` after `item_attrs_narrow`. Forward into `evaluate_cell` at the call site around line 371.

**`evaluate_cell` signature ([sweep.py:400-503](../../../evaluation/retrieval/sweep.py#L400-L503)):** add `clause_is_reverse: torch.Tensor | None`. Forward into `_try_build_algo` at line 427.

**`_try_build_algo` signature ([sweep.py:517-546](../../../evaluation/retrieval/sweep.py#L517-L546)):** add `clause_is_reverse: torch.Tensor | None`. Forward into `build_algorithm` at line 534.

This is a mechanical pass-through; every function level just adds one kwarg and forwards it. The pattern mirrors how `item_attrs_narrow` is already threaded.

### Tests

#### `retrieve/tests/correctness/test_silvertorch.py` — already has coverage

No change required at the library-test layer. [test_silvertorch.py:70-144](../../../retrieve/tests/correctness/test_silvertorch.py#L70-L144) already builds `SilverTorch` with `clause_is_reverse` and verifies the registered buffer + kernel output.

#### `evaluation/retrieval/` — add a wrapper-level reverse-clause test

There is currently no test under `evaluation/` exercising `SilvertorchAlgo` with `clause_is_reverse`. Add one — this is the test that would have caught today's regression.

New file: `evaluation/retrieval/tests/test_silvertorch_algo_reverse.py` (create `tests/` if absent).

```python
"""SilvertorchAlgo wrapper: reverse-clause propagation.

Regression test for the bug where SilvertorchAlgo dropped clause_is_reverse
on the floor, causing Recall ≈ 0 on reverse-clause sweeps. See
docs/plans/silvertorch-reverse-clause-wrapper-fix.md.
"""

import torch
import pytest
from retrieval.algos import build_algorithm


@pytest.mark.parametrize("backend", ["triton", "torch"])
def test_silvertorch_clause_reverse_matches_full_scan(backend):
    """Build a tiny exact-clause silvertorch with one reverse clause and
    verify its top-K matches a brute-force filtered FullScan oracle."""
    torch.manual_seed(0)
    N, D, B, K = 256, 16, 4, 8
    n_lists, n_probe = 16, 16  # probe all → IVF is exact
    embs = torch.randn(N + 1, D, device="cuda")
    embs[0] = 0
    # 1 clause, 1 attribute per item, 4 possible values.
    item_attrs = torch.randint(0, 4, (N + 1, 1, 1), device="cuda")
    clause_is_reverse = torch.tensor([True], device="cuda")
    qa = torch.randint(0, 4, (B, 1), device="cuda")
    queries = torch.randn(B, D, device="cuda")

    algo = build_algorithm(
        "silvertorch",
        embs,
        k=K,
        filter_kind="clause",
        item_attrs_narrow=item_attrs,
        clause_is_reverse=clause_is_reverse,
        params={"n_lists": n_lists, "n_probe": n_probe, "n_iter": 3, "seed": 0},
        backend=backend,
    )
    ids, _ = algo(queries.contiguous(), qa.contiguous())

    # Brute-force reference: items where attr != query (reverse), top-K by dot.
    scores = queries @ embs.t()
    mask = item_attrs[:, 0, 0].unsqueeze(0) != qa[:, :1]
    scores = scores.masked_fill(~mask, float("-inf"))
    scores[:, 0] = float("-inf")
    ref_ids = scores.topk(K, dim=1).indices

    # Per-row set-equality (kernel and reference may disagree on tie order).
    assert set(ids[0].tolist()) == set(ref_ids[0].tolist())
```

Run with `cd evaluation && uv run pytest retrieval/tests/test_silvertorch_algo_reverse.py -v`.

(If the existing eval-side test layout already has a `tests/` directory or conftest, mirror its style instead of inventing one.)

### `docs/thesis/07-results.md` — retire the methodology caveat once data is regenerated

After (1) the wrapper fix lands, (2) the goodreads-d128/d256 oracle cache is deleted, and (3) the filter sweeps are rerun, the writer must:

- Drop the "INCOMPATIBLE — reverse" annotations from the §6.2.4 algorithm × filter matrices (all three Goodreads dims).
- Replace the §6.2.1 "silvertorch handling of reverse clauses (load-bearing finding)" paragraph with a one-line note: "silvertorch's codesigned exact-clause kernel evaluates reverse clauses via per-clause XOR (see [silvertorch/main.py](../../../retrieve/src/retrieve/layers/silvertorch/main.py)); the eval-side wrapper threads `clause_is_reverse` through end-to-end."
- Re-extract Recall numbers for silvertorch on Goodreads `c1_lang_reverse`, `c0c1`, `all4` from the rerun JSONs and update tables.
- Remove the `[TODO: silvertorch + reverse-clause currently unsupported ...]` line from the §6.2 writer's notes.

## Verification

Stepwise checks to confirm the fix end-to-end:

1. **Library tests stay green:** `cd retrieve && uv run pytest tests/correctness/test_silvertorch.py -v` (no library changes, but smoke-checks that the kwarg-forwarding plan didn't accidentally regress anything).
2. **New wrapper test passes:** `cd evaluation && uv run pytest retrieval/tests/test_silvertorch_algo_reverse.py -v`. Should pass on both `triton` and `torch` backends. If it fails on `torch` only, the issue is in the torch-side codepath of `SilverTorch.forward`, not this plan's scope.
3. **End-to-end smoke (no GPU rerun required if 1 and 2 pass):** call `build_algorithm("silvertorch", ..., clause_is_reverse=torch.tensor([True, False, False, False, False]), ...)` and confirm `algo.idx.clause_is_reverse[0].item() == True`.
4. **Production rerun (requires GPU):** `cd evaluation && uv run run-evaluation --eval-type filter --config config/goodreads/d128-filter.yaml` after `rm data/goodreads-work-id/<gt_subdir>/gt_topk_*.pt`. Verify in the resulting `evaluation/results/goodreads/d128-filter.json`:
   - `linr_v1_filter_mask` on `c1_lang_reverse / clause` Recall@100 ≥ 0.99 (oracle parity, post-cache-rebuild).
   - `silvertorch` on `c1_lang_reverse / clause` Recall@100 is now in the same ballpark as the other algorithms (≥ 0.85 expected, vs the current 0.0005), and within ≤ 5% of `linr_v1_filter_mask`.
   - Same checks on `c0c1` and `all4` (both contain c1).
5. Repeat (4) for `d64-filter` and `d256-filter`.
6. Re-upload to `pinkmeme/retrieval-filter-evals-2026-05-23` (or a fresh dated repo); update the local mirror and re-pull on the writer's machine.

## Critical files

| File                                                                                       | Role |
|--------------------------------------------------------------------------------------------|------|
| [evaluation/retrieval/algos/silvertorch.py](../../../evaluation/retrieval/algos/silvertorch.py) | TO EDIT — add `clause_is_reverse` kwarg, forward to `SilverTorch.register_index` |
| [evaluation/retrieval/algos/__init__.py](../../../evaluation/retrieval/algos/__init__.py)     | TO EDIT — extend `build_algorithm` signature, drop stale skip-claim docstring |
| [evaluation/retrieval/sweep.py](../../../evaluation/retrieval/sweep.py)                       | TO EDIT — thread `clause_is_reverse` through `_build_filter_modules` → `run_filter_kind` → `run_one_sweep` → `evaluate_cell` → `_try_build_algo` |
| `evaluation/retrieval/tests/test_silvertorch_algo_reverse.py`                              | TO CREATE — wrapper-level regression test |
| [retrieve/src/retrieve/layers/silvertorch/main.py](../../../retrieve/src/retrieve/layers/silvertorch/main.py) | READ ONLY — already supports the kwarg |
| [retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py](../../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_exact.py) | READ ONLY — kernel already XORs `clause_is_reverse[c]` per clause |
| [retrieve/tests/correctness/test_silvertorch.py](../../../retrieve/tests/correctness/test_silvertorch.py) | READ ONLY — library-side reverse-clause coverage |
| `docs/thesis/07-results.md` (deleted)                                       | TO EDIT (after rerun) — retire INCOMPATIBLE annotations and the "load-bearing finding" paragraph |

## Sequencing

1. Land the wrapper change (§ Changes 1-3). PR can ship with the new wrapper test passing on a dev GPU; no harness rerun yet.
2. Separately or in the same PR: delete `data/goodreads-work-id/<gt_subdir>/gt_topk_c1_lang_reverse.pt`, `gt_topk_c0c1.pt`, `gt_topk_all4.pt` (and equivalent d64/d256 paths if they exist). This is the goodreads stale-cache remediation from `docs/thesis/07-results.md` (deleted)'s ACTION REQUIRED block; the wrapper fix is necessary but not sufficient — old oracle still poisons recall calculation against silvertorch's now-correct output.
3. Rerun `config/goodreads/d{64,128,256}-filter.yaml` and re-upload to HF.
4. Update `docs/thesis/07-results.md` (deleted) per the "retire methodology caveat" subsection above.
