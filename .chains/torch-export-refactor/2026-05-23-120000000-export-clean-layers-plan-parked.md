---
chain: "torch-export-refactor"
branch: "main"
nextStep: "Re-scope against the L layout before executing (roadmap G-e): the LiNRV1-V4 composites already give one forward(query, query_clause_attrs=None) signature; decide whether per-primitive mode= flags are still wanted, then write the tests/export round-trip suite."
created: "2026-05-23T12:00:00Z"
---

# `retrieve` modules export-clean under torch.export (parked plan)

Source: `docs/plans/torch-export-refactor.md`, written 2026-05-23, never started. Status banner at archive time: anchors stale (the `torch_knn` algo was deleted in E1.1; `filter=` is now `filter_mode=`; kernels are `@triton_op`; commands used `conf/` and `--algorithms`; the "shelved native CUDA" experiment shipped and was deleted at B4). Goal and design sketch still the best thinking on it. Parked as roadmap G-e.

## Goal
A consumer composes a `retrieve` layer with their own model, calls `torch.export.export(...)`, deploys the `.pt2` to a C++ runtime. The repo ships no `.pt2`. Promise: every layer backing an eval algo traces under `torch.export.export()` without raising on Optionals, data-dependent shapes, `.item()` syncs, or wrapper opacity. Kernel side already in shape (non-Optional op schemas, full-width compact outputs, no `.item()`, pad tail removed); what remains is layer side.

## Design
1. One forward signature per export entry: Optional matrices collapse to a construction-time `mode: Literal[...]` or a sibling method; `backend` stays.
2. No deprecation shims. 3. Tests pass before and after with no relaxed tolerance. 4. No `.pt2` zoo; a smoke harness round-trips export + load.

Mode map as written: `PostfilterKNN` / `PostfilterKNNInt8` `full | masked`; `PrefilterKNN` `full | candidates` (drop the `counts is None` fallback and the layer-side `p == 0` returns); `OneBitKNN` `full | candidates` (1:1 with `oporp_1bit_match_topk_full` / `_indirect`); `FullScanKNN` `full | masked | candidates` (masked keeps `post_filter_topk`; candidates pads to K for a static `[B, k]`); `SilverTorch` sibling methods `forward_ivf_only`, `forward_bloom`, `forward_exact`, `forward_candidates` (sibling methods rather than a flag: each filter mode has a distinct prep block).

Verification as written: `retrieve/tests/export/test_export_roundtrip.py` parametrized over (layer, mode) asserting `torch.equal` ids and `allclose` scores after `torch.export.save` / `load`, plus the four SilverTorch methods; acceptance grep: zero `Tensor | None` in in-scope forward signatures.

Out of scope: shipping `.pt2`; a `build_export.py`; end-to-end algo bundling; AOTI; live updates (composes independently).

## Later context
L2 made V1-V4 library modules with one `forward(query, query_clause_attrs=None)`, which is what this plan wanted at the algo level (roadmap §3). `tests/compile/test_export_kernel_ref.py` already gates that export preserves the kernel reference for `codesigned_probe_score_exact` and replays bit-exactly.
