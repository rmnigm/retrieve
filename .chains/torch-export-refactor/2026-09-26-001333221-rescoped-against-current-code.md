---
chain: "torch-export-refactor"
branch: "main"
parent: "2026-05-23-120000000-export-clean-layers-plan-parked.md"
nextStep: "Queued for roadmap phase G (G-e), after F5 (the paper). Not before the campaign or the current kernel-optimization pass (G-a/G-d/Q4). When picked up: fix the three `Tensor | None` forward params in modules/linr.py, then run torch.export.export() on every layer and fix whatever it actually raises on."
created: "2026-09-26T00:13:33Z"
---

# Re-scoped: torch.export-clean layers

## Size/risk estimate
**Small**, not comparable to G-a+Q4. The 2026-05-23 note said "kernel side already in
shape... what remains is layer side" — that remains true, and the layer side turned out
smaller than it read at the time. This is a few hours of focused work, not a phase.

## What changed since 2026-05-23 (verified against current code, 2026-09-26)
- **Kernel side was already finished and is still clean.** No `Optional[...]` anywhere in
  `retrieve/src/retrieve/modules/*.py` forward signatures (checked by grep). The two
  remaining `.item()` syncs in the whole `modules/` + `indexing/` tree
  (`modules/filters.py:54` in `BloomFilter.register_index`, `modules/knn.py:113` in the
  `PostfilterKNNInt8` `load_state_dict` post-hook) are both **setup-time, not forward-time**
  — exactly the pattern `silvertorch.py` documents at lines 585-591 ("the two `.item()`
  syncs here run once, at load time"). No module's `forward()` calls `.item()`.
- **A kernel-reference export test already exists**:
  `retrieve/tests/compile/test_export_kernel_ref.py` exports a module wrapping
  `codesigned_probe_score_exact` and asserts the exported graph keeps a live reference to
  the Triton kernel (via the custom-op node or a `triton_kernel_wrapper_*` HOP) and replays
  bit-identically. Its docstring says explicitly: "Lives under `tests/compile/` until the
  export plan creates `tests/export/`." This is K2's regression gate, already built —
  the parked plan's "Verification as written" section (`tests/export/test_export_roundtrip.py`)
  can start from moving/generalizing this file rather than writing it from scratch.
- **The one real remaining blocker**: `modules/linr.py` has three
  `forward(self, query: Tensor, query_clause_attrs: Tensor | None = None)` signatures
  (lines ~59, ~129, ~163) — `PostfilterKNN`, `PrefilterKNN`(?), the no-filter LiNR paths.
  These are exactly the "Optionals" the original plan's acceptance grep
  ("zero `Tensor | None` in in-scope forward signatures") targets. `torch.export` does
  not handle `Optional[Tensor]` cleanly (it has to specialize per call, or the caller
  passes a sentinel). This is the actual remaining work: give each of these a required
  or sentinel-shaped `query_clause_attrs` at the export boundary (a wrapper module or a
  documented "pass an empty tensor, not None" convention), not a rewrite of the modules.
- **Nothing got harder.** No new Optional args were added anywhere else since 2026-05-23;
  the module surface (`RetrievalModule`, the LiNR family, `SilverTorch`) is stable per
  [decisions.md](../../docs/decisions.md#library) ("op schemas... are stable").

## Updated plan (same shape as the original, anchors fixed)
1. Confirm via `torch.export.export()` on each of `PostfilterKNN`, `PostfilterKNNInt8`,
   `PrefilterKNN`, `FullScanKNN`, `OneBitKNN`, `SilverTorch`, and the four `LiNRV1`-`V4`
   composites (`retrieve/src/retrieve/modules/linr.py`, `bit_knn.py`, `knn.py`,
   `silvertorch.py`) what actually raises today — the grep above says only the three
   `Tensor | None` signatures should, but verify rather than assume.
2. Resolve the `Tensor | None` params: likely an export-time wrapper that takes a required
   tensor and internally treats an all-`-1` / zero-row tensor as "no clause attrs", since
   `SilverTorch`'s own `query_clause_attrs=None` fast path already has a `None`-shaped
   precedent to look at for how the no-filter case is distinguished today.
3. Move/generalize `tests/compile/test_export_kernel_ref.py`'s pattern into
   `tests/export/test_export_roundtrip.py`, parametrized over (layer, filter_mode) per the
   original plan, asserting `torch.equal` ids and `allclose` scores after
   `torch.export.save`/`load`.
4. A `docs/system/architecture.md` or new page note on the export contract (kept out of
   scope for this re-scope note itself — a wiki edit is the implementing step's job, not
   this one's).

## Out of scope (per the original plan and unchanged)
Shipping a `.pt2`, `build_export.py`, end-to-end algo bundling, AOTI, live updates
(composes independently — see the [live-update-api](../live-update-api/) chain).
