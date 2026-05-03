# Move filter mask algebra from eval harness into retrieve/ library

## Context

`evaluation/retrieval/algo_registry.py` carries two pure mask-algebra closures
(`bloom_mask`, `combined_mask`) and one input-validation guard
(`check_bloom_no_reverse`) that have no harness-specific concept in them — they
just compose `BloomFilter` / `ClauseIndex` outputs. They belong in
`retrieve/layers/filters/`, where the rest of the mask plumbing lives
(`combine_masks`, `combine_indices`).

The motivating cleanup: shrink `algo_registry.py` to genuine harness wiring
(per-algo branches, cascade composition, silvertorch routing, voyager
post-filter), and grow the lib's public surface so other consumers
(notebooks, future algos, downstream embedders) get the same composition
primitives without re-deriving them.

This also tightens the `BloomFilter` API: today its `evaluate_mask` only
handles `[B, C=1]` cleanly (the harness loops externally for multi-shelf wide
queries), and the "no reverse clauses" rule is a sweep-time check that
silently bypasses for `filter_kind="combined"`. After this change the
multi-shelf loop lives in the lib and the reverse-clause guard fires at
`register_index` time when the harness opts in.

---

## Files touched

**Lib (retrieve/src/retrieve/layers/filters/):**
- `bloom.py` — extend `evaluate_mask` to `[B, M]`; add `clause_is_reverse` kwarg to `register_index`
- `__init__.py` — add `compose_masks`; extend `__all__`

**Lib tests (retrieve/tests/correctness/):**
- `test_bloom_filter.py` — add `[B, M]` parity tests + reverse-clause guard test
- `test_filters.py` — add `compose_masks` dispatch table tests

**Harness (evaluation/retrieval/):**
- `algo_registry.py` — delete inner closures, delete top-level guard, swap call sites, pass `clause_is_reverse` to `bf.register_index` when `filter_kind=="bloom"`, update import, update `__all__`, update module docstring

`evaluation/retrieval/evaluate.py` only imports 5 symbols from `algo_registry`
(`SilvertorchSkippedOnNarrow`, `build_algorithm`, `build_filter_modules`,
`build_filtered_algorithm`, `synthesize_query_attrs_narrow`) — none of which
change. No edit needed there.

---

## Phase A — Lib changes

### A1. `BloomFilter.evaluate_mask` accepts `[B, M]`

File: [retrieve/src/retrieve/layers/filters/bloom.py](retrieve/src/retrieve/layers/filters/bloom.py#L61-L68)

Replace `evaluate_mask` (lines 61–68) with:

```python
def evaluate_mask(self, query_clause_attrs: Tensor) -> Tensor:
    """Returns ``[B, N]`` bool.

    Accepts ``query_clause_attrs`` of shape ``[B]``, ``[B, 1]``, or
    ``[B, M]`` for ``M >= 1``. When ``M > 1`` (multi-shelf wide query),
    runs ``M`` per-shelf passes and ANDs the resulting masks. The
    ``M == 1`` path is byte-identical to the prior single-pass kernel call.
    """
    if query_clause_attrs.dim() == 1:
        query_clause_attrs = query_clause_attrs.unsqueeze(-1)
    m = query_clause_attrs.shape[1]
    if m == 1:
        qb = self._build_query_sigs(query_clause_attrs)  # [B, W]
        if qb.is_cuda:
            from retrieve.kernels.triton.silvertorch.bloom_match import bloom_match
            return bloom_match(qb, self.bloom_sigs)
        match = (qb.unsqueeze(1) & self.bloom_sigs.unsqueeze(0)) == qb.unsqueeze(1)
        return match.all(dim=-1)
    out = self.evaluate_mask(query_clause_attrs[:, 0:1])
    for j in range(1, m):
        out = out & self.evaluate_mask(query_clause_attrs[:, j:j+1])
    return out
```

Notes:
- Math equivalence: `(qb_or & sigs) == qb_or` ⟺ each individual `qb_j` is
  subset-of `sigs`. So existing `[B, C]` callers (e.g. test fixture with
  `c=2`) produce the same final mask under either path. Only the M=1 path
  is byte-identical at the kernel level; M>1 is an algebraic refactor.
- The `dim() == 1` normalization mirrors what the harness `bloom_mask`
  closure does today (line 312–313).

### A2. `BloomFilter.register_index` accepts `clause_is_reverse`

File: [retrieve/src/retrieve/layers/filters/bloom.py](retrieve/src/retrieve/layers/filters/bloom.py#L36-L50)

Replace `register_index` (lines 36–50) with:

```python
def register_index(
    self,
    item_clause_attrs: Tensor,
    item_embs: Tensor | None = None,
    clause_is_reverse: Tensor | None = None,
) -> None:
    if clause_is_reverse is not None and bool(clause_is_reverse.any().item()):
        raise ValueError(
            "BloomFilter is paper-strict: clause_is_reverse must be all-False "
            "(no NOT). Use ClauseIndex (or filter_kind='clause' / 'combined') "
            "for reverse clauses."
        )
    seeds = _generate_seeds(self.k_hash, device=item_clause_attrs.device)
    self.register_buffer("hash_seeds", seeds)
    sigs = _build_signatures(
        item_clause_attrs.long(),
        seeds,
        self.m_bits,
        self.k_hash,
        self.word_count,
    )
    self.register_buffer("bloom_sigs", sigs)
```

Notes:
- The `item_embs` param is kept (interface contract via `FilterModule`).
- Error message preserves the spirit of the harness's
  `check_bloom_no_reverse` message (mentions paper-strict, no NOT,
  suggests `ClauseIndex` / `combined`).
- `clause_is_reverse=None` → no check (lib doesn't second-guess callers
  who don't supply it).

### A3. Add `compose_masks` to filters package

File: [retrieve/src/retrieve/layers/filters/__init__.py](retrieve/src/retrieve/layers/filters/__init__.py)

Add after `combine_masks` (current line 23), before `combine_indices`:

```python
def compose_masks(
    ci: ClauseIndex | None,
    bf: BloomFilter | None,
    qa_narrow: Tensor | None,
    qa_wide: Tensor | None,
) -> Tensor | None:
    """Compose narrow (clause) + wide (bloom) masks via element-wise AND.

    Returns ``None`` if no side contributes a mask (either the matching
    filter or the matching query attrs is absent). Otherwise returns the
    AND of whichever sides are present, as a ``[B, N]`` bool tensor.
    """
    masks: list[Tensor | None] = []
    if ci is not None and qa_narrow is not None:
        masks.append(ci.evaluate_mask(qa_narrow))
    if bf is not None and qa_wide is not None:
        masks.append(bf.evaluate_mask(qa_wide))
    return combine_masks(*masks)
```

Update `__all__` (current lines 73–78) to:

```python
__all__ = [
    "BloomFilter",
    "ClauseIndex",
    "combine_indices",
    "combine_masks",
    "compose_masks",
]
```

---

## Phase B — Lib tests

### B1. Extend [retrieve/tests/correctness/test_bloom_filter.py](retrieve/tests/correctness/test_bloom_filter.py)

Add a new test class at the end of the file, in the same style as
`TestNoFalseNegatives` (helper-call construction with explicit seeds,
`.to("cuda")`, `make_attrs` / `make_query_attrs`):

```python
class TestMultiShelfQuery:
    def test_b1_query_byte_identical_to_legacy(self, attrs):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        q = make_query_attrs(b=32, c=1, n_vocab=100, inactive_rate=0.0, seed=51)
        # Two consecutive runs with same input must be byte-identical.
        a = bf.evaluate_mask(q)
        b = bf.evaluate_mask(q)
        assert torch.equal(a, b)
        assert a.shape == (32, attrs.shape[0])

    def test_bm_equals_and_of_per_shelf_passes(self):
        attrs = make_attrs(n=512, c=1, a_max=4, n_vocab=80, pad_rate=0.2, seed=60)
        bf = BloomFilter(m_bits=2048, k_hash=7).to("cuda")
        bf.register_index(attrs)
        q2 = make_query_attrs(b=32, c=2, n_vocab=80, inactive_rate=0.0, seed=61)

        combined = bf.evaluate_mask(q2)
        per_shelf_0 = bf.evaluate_mask(q2[:, 0:1])
        per_shelf_1 = bf.evaluate_mask(q2[:, 1:2])
        assert torch.equal(combined, per_shelf_0 & per_shelf_1)

    def test_1d_query_normalized_to_b1(self, attrs):
        bf = BloomFilter(m_bits=512, k_hash=5).to("cuda")
        bf.register_index(attrs)
        q1d = make_query_attrs(b=8, c=1, n_vocab=100, inactive_rate=0.0, seed=62).squeeze(-1)
        assert q1d.dim() == 1
        out = bf.evaluate_mask(q1d)
        assert out.shape == (8, attrs.shape[0])


class TestRegisterIndexReverseGuard:
    def test_register_raises_on_any_reverse_clause(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=70)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        is_reverse = torch.tensor([False, True], device="cuda")
        with pytest.raises(ValueError, match="paper-strict"):
            bf.register_index(attrs, clause_is_reverse=is_reverse)

    def test_register_accepts_all_false(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=71)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        is_reverse = torch.tensor([False, False], device="cuda")
        bf.register_index(attrs, clause_is_reverse=is_reverse)  # must not raise
        assert bf.bloom_sigs.shape == (64, 256 // 64)

    def test_register_none_skips_check(self):
        attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=72)
        bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
        bf.register_index(attrs, clause_is_reverse=None)  # default, no check
        assert bf.bloom_sigs.shape == (64, 256 // 64)
```

### B2. Extend [retrieve/tests/correctness/test_filters.py](retrieve/tests/correctness/test_filters.py)

Add new test functions at the end (matching the file's existing
function-style, no class grouping):

```python
def test_compose_masks_neither_returns_none():
    from retrieve.layers.filters import compose_masks
    assert compose_masks(None, None, None, None) is None


def test_compose_masks_ci_only():
    from retrieve.layers.filters import compose_masks
    attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=80)
    qa_narrow = make_query_attrs(b=8, c=2, n_vocab=20, inactive_rate=0.0, seed=81)
    ci = ClauseIndex().to("cuda")
    ci.register_index(attrs)
    out = compose_masks(ci, None, qa_narrow, None)
    assert torch.equal(out, ci.evaluate_mask(qa_narrow))


def test_compose_masks_bf_only():
    from retrieve.layers.filters import compose_masks, BloomFilter
    attrs = make_attrs(n=64, c=1, a_max=2, n_vocab=20, pad_rate=0.0, seed=82)
    qa_wide = make_query_attrs(b=8, c=2, n_vocab=20, inactive_rate=0.0, seed=83)
    bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
    bf.register_index(attrs)
    out = compose_masks(None, bf, None, qa_wide)
    assert torch.equal(out, bf.evaluate_mask(qa_wide))


def test_compose_masks_both_ands():
    from retrieve.layers.filters import compose_masks, BloomFilter
    narrow_attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=84)
    wide_attrs = make_attrs(n=64, c=1, a_max=2, n_vocab=20, pad_rate=0.0, seed=85)
    qa_narrow = make_query_attrs(b=8, c=2, n_vocab=20, inactive_rate=0.0, seed=86)
    qa_wide = make_query_attrs(b=8, c=1, n_vocab=20, inactive_rate=0.0, seed=87)
    ci = ClauseIndex().to("cuda")
    ci.register_index(narrow_attrs)
    bf = BloomFilter(m_bits=256, k_hash=3).to("cuda")
    bf.register_index(wide_attrs)
    out = compose_masks(ci, bf, qa_narrow, qa_wide)
    assert torch.equal(out, ci.evaluate_mask(qa_narrow) & bf.evaluate_mask(qa_wide))


def test_compose_masks_filter_present_query_absent_returns_none():
    from retrieve.layers.filters import compose_masks
    attrs = make_attrs(n=64, c=2, a_max=2, n_vocab=20, pad_rate=0.0, seed=88)
    ci = ClauseIndex().to("cuda")
    ci.register_index(attrs)
    assert compose_masks(ci, None, None, None) is None
```

---

## Phase C — Harness slim-down

File: [evaluation/retrieval/algo_registry.py](evaluation/retrieval/algo_registry.py)

### C1. Update import (line 81)

```python
from retrieve.layers.filters import (
    BloomFilter,
    ClauseIndex,
    combine_indices,
    combine_masks,
    compose_masks,
)
```

### C2. Delete `check_bloom_no_reverse` (lines 206–221) entirely.

### C3. Update `build_filter_modules` (line 252) to pass the guard arg conditionally:

```python
if filter_kind in ("bloom", "combined"):
    if item_attrs_wide is None:
        raise ValueError(f"{filter_kind} requires item_attrs_wide")
    bf = BloomFilter(m_bits=bloom_m_bits, k_hash=bloom_k_hash).to(device)
    bf.register_index(
        item_attrs_wide,
        clause_is_reverse=clause_is_reverse if filter_kind == "bloom" else None,
    )
```

This preserves today's semantics for `filter_kind="combined"` (bypass the
check — `ci` handles narrow reverse clauses, `bf` only handles wide). For
sole-bloom mode, the check now fires at registration; this is stricter
than today's sweep-time check (catches reverse clauses even when the
active sweep doesn't touch them), per the design decision confirmed in
clarification.

### C4. Delete the `check_bloom_no_reverse` call (line 296).

### C5. Delete the `bloom_mask` and `combined_mask` inner closures
(lines 306–330) entirely.

### C6. Replace the four call sites of `combined_mask` with `compose_masks`:

- Line 341 (`fs_forward`): `mask = compose_masks(ci, bf, qa_narrow, qa_wide)`
- Line 359 (`linr_forward`): `mask = compose_masks(ci, bf, qa_narrow, qa_wide)`
- Line 471 (`kt_forward`): `mask = compose_masks(ci, bf, qa_narrow, qa_wide)`
- Line 488 (`voy_forward`): same, but the `.to(item_embs.device)` calls on
  `qa_narrow` / `qa_wide` stay — they're harness device-management:
  ```python
  mask = compose_masks(
      ci,
      bf,
      qa_narrow.to(item_embs.device) if qa_narrow is not None else None,
      qa_wide.to(item_embs.device) if qa_wide is not None else None,
  )
  ```

### C7. Replace the `bloom_mask(qa_wide)` call in `st_forward` (line 453):

```python
if qa_wide is not None and qa_wide.dim() == 2 and qa_wide.shape[1] > 1:
    bloom_mask_external = bf.evaluate_mask(qa_wide)
    external = combine_masks(narrow_mask, bloom_mask_external)
    return idx(q, query_clause_attrs=None, mask=external)
```

`bf` is non-None on this branch (already guarded above). The lib's
`evaluate_mask` now does the M-pass loop internally.

### C8. Update `__all__` (lines 503–514) — drop `check_bloom_no_reverse`:

```python
__all__ = [
    "ALGORITHMS",
    "FilterKind",
    "ForwardFn",
    "FilteredForwardFn",
    "SilvertorchSkippedOnNarrow",
    "build_algorithm",
    "build_filter_modules",
    "build_filtered_algorithm",
    "synthesize_query_attrs_narrow",
]
```

### C9. Update module docstring (top of file, lines 1–61) — drop the
trailing paragraph about composition that explains the M-pass bloom loop
(that's now lib-internal). Keep everything else (algo descriptions,
cascade rationale, voyager note, etc.). Specifically: the paragraph
beginning "Composition: narrow + wide masks AND'd via `combine_masks`…"
becomes "Composition: narrow + wide masks AND'd via `compose_masks` from
`retrieve.layers.filters`."

### C10. Do NOT modify any other branches — the cascade `linr_v3_then_v2`,
the silvertorch `linr_v2_filter_compact` `combine_indices` shelf
fanout, the silvertorch single-shelf fused-bloom path, voyager's
post-filter `mask.gather`, and `SilvertorchSkippedOnNarrow` raising are
all genuine harness wiring and stay byte-identical.

---

## Phase D — Verification

End-to-end checks the implementer must run:

1. **Lib unit tests (CUDA-required)**:
   ```
   cd /workspace/retrieve/retrieve && uv run pytest tests/correctness/ -q
   ```
   All existing tests + the new ones must pass. Two consecutive runs must
   produce identical outputs (no new randomness — multi-shelf path is M
   sequential single-shelf calls AND'd).

2. **Lint**:
   ```
   uv run --with ruff ruff check evaluation/ retrieve/
   uv run --with ruff ruff check evaluation/ retrieve/ --select I
   ```
   Clean modulo any pre-existing E501s (don't chase those).

3. **Import probe** (from `/workspace/retrieve/evaluation`):
   ```bash
   uv run python -c "
   from retrieval.algo_registry import (
       build_algorithm, build_filtered_algorithm, build_filter_modules,
       SilvertorchSkippedOnNarrow, synthesize_query_attrs_narrow,
   )
   from retrieve.layers.filters import (
       BloomFilter, ClauseIndex, combine_masks, combine_indices, compose_masks,
   )
   print('ok')
   "
   ```
   Must print `ok`.

4. **Identifier grep** — `bloom_mask` and `combined_mask` should appear
   only as test names or as method names on `BloomFilter` / `ClauseIndex`,
   never as inner closures in `algo_registry.py`:
   ```
   grep -nE '\b(bloom_mask|combined_mask)\b' /workspace/retrieve/evaluation /workspace/retrieve/retrieve -r
   ```

5. **Harness sanity** — confirm `evaluate.py`'s 5 imports from
   `algo_registry` still resolve (covered by step 3, but worth a
   re-check by reading [evaluation/retrieval/evaluate.py:36-42](evaluation/retrieval/evaluate.py#L36-L42)).

Expected line-count delta on `algo_registry.py`: −~50 lines (16 for
`check_bloom_no_reverse` + ~25 for the two inner closures + 1 call site +
import/`__all__`/docstring tweaks net out to a small additional save).

---

## Out of scope (do not touch)

- Per-algo branches in `build_filtered_algorithm` (cascade composition,
  silvertorch 2-shelf fallback, voyager post-filter, `linr_v2_filter_compact`
  `combine_indices` fanout) — these are bench wiring.
- Eval YAML configs, parquet/pt artifacts.
- Triton kernels (`bloom_match`, `bloom_compact`, `clause_mask`,
  `clause_compact`).
- `evaluation/retrieval/evaluate.py` (its 5 imports remain valid).
- File renames or `_`-prefixed names (use `__all__` for gating, which is
  already in place).

---

## Behavior deltas (intentional)

1. **Multi-shelf bloom equivalence**: `BloomFilter.evaluate_mask([B, M>1])`
   today routes through a single `_build_query_sigs` → `bloom_match` call
   that ORs all M attribute hash sets into one query signature. After
   this change it routes through M sequential M=1 calls AND'd. The two
   are mathematically equivalent (`(qb_or & sigs) == qb_or` ⟺ each
   `qb_j ⊆ sigs`); existing `c=2` test fixtures (e.g.
   `test_bloom_filter.py:22`) continue to produce the same masks. The
   M=1 path remains byte-identical.

2. **Reverse-clause guard timing for `filter_kind="bloom"`**: stricter
   than today. Today it checks only clauses in `sweep.active_clauses`;
   after the move, it raises at registration if any entry of
   `clause_is_reverse` is True. A sole-bloom configuration on a dataset
   with any reverse clause will now fail at `build_filter_modules` time
   even if every actual sweep avoids that clause. This was confirmed as
   the intended trade-off.

3. **Reverse-clause guard timing for `filter_kind="combined"`**:
   unchanged. The harness passes `None` for combined mode, so the lib's
   guard does not fire — matching today's behavior where
   `check_bloom_no_reverse` returned early on `filter_kind != "bloom"`.
