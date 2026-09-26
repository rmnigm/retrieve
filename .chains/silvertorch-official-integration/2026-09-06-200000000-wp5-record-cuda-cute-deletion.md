---
chain: "silvertorch-official-integration"
branch: "main"
parent: "2026-09-06-140000000-wp2-wp3-record-adapter-and-parity-gate.md"
nextStep: "WP-4 (roadmap B3): the Triton vs official head-to-head, kernel-only, phase 2 and end to end."
created: "2026-09-06T20:00:00Z"
---

# §15 record: WP-5, delete the CUDA C++ and CuTe backends (roadmap B4), 2026-09-06

Branch `dev/b4-delete-cuda-cute` off `development` at `41d4479`; commits `4d92432` (deletion), `010681d` (`Backend` split + table dispatch, review item 5), `6698b4a` (docs), `a435179` (test fix). Gate check before deleting: §14 T1 `torch.equal` on all 8 regime + 4 exact cells; CLAUDE rule 5 satisfied.

## Tag and deletions
`git tag cuda-cute-backends-final 41d4479`. Deleted outright 3,677 lines: `.cu` 672, `codesigned_probe_score_cuda.py` 611, `codesigned_probe_score_cute.py` 650, `cute/codesigned_probe_score.py` 763, `cute/__init__.py` 0, `test_codesigned_probe_score_cuda.py` 454, `..._cute.py` 527. Trimmed: `main.py` 781 -> 639, `tune.py` 643 -> 452, `tests/conftest.py` 247 -> 182, `test_silvertorch.py` 703 -> 599, `test_silvertorch_compile.py` 119 -> 97, `test_export_kernel_ref.py` 169 -> 110, `test_tune_smoke.py` 50 -> 38. `uv.lock`: nine transitive packages gone (`nvidia-cutlass-dsl*`, `cuda-python`, `cuda-bindings`, `cuda-core`, `nvidia-cuda-nvdisasm`, `backports-strenum`), no torch / triton line changed. Commit stat 169 files, +320 / -5,460.

Moved, not deleted: `build_transposed_sigs`, `words_per_cluster` to `bloom_hash.py` (TF-1), their layout test to `test_bloom_hash.py`; `make_bloom` kept for TF-1. Plans archived; `scripts/check_doc_links.py` started skipping `docs/plans/archive/`. Docs: kernels.md 1,621 -> 973 (a 25-line "Historical backends" note), ops 16 -> 10 across 7 files, tuner specs 11 -> 7.

## `Backend` split (`010681d`)
`LinrBackend = Literal["torch", "triton"]`, `SilverTorchBackend = Literal["torch", "triton", "official"]`, `check_backend(backend, literal)` in every constructor: unknown values, or `"official"` on a LiNR module, raise `ValueError("unknown backend ...")` instead of silently running torch (18 new cells). `SilverTorch.forward` dispatches through `self._forward_impl`, a table built in `__init__`. `_register_official_filter_buffers` folded into `_register_filter_buffers(..., perm=sort_perm)`. CPU bit-identity: torch-backend `SilverTorch` on none / bloom / exact, every state-dict key, buffer and output `torch.equal` parent vs `010681d`.

## CPU gates (§15.4)
ruff 0.15.6 clean; library collect 486 after deletion (from 701 = 574 + 127), 504 after the split; evaluation 93 passed / 1 skipped; links 0; `uv lock` 104 ms. The literal grep gate (`git grep -il "cute|codesigned_probe_score_cuda"` only in the archive) is unsatisfiable (`-i` without `-w` matches "execute" in LICENSE, articles, CLAUDE.md); reading applied: no code, test, harness, pyproject, script or live system / sdist doc mentions the backends.

## §15.6 A100 gate, green
`/venvs/b4`, nvcc 12.8, `uv sync --extra official`, `retrieve.__file__` verified inside the worktree; clocks unlocked (210 MHz idle vs 1410 max), wall times are metadata only.
| run | state | result |
|---|---|---|
| 1 | `6698b4a` | 501 passed, 3 failed |
| 2 | `a435179` | 504 passed, 0 failed, 0 skipped, 37.8 s |
The three red cells: the `simhash` row of the new rejection test built `SimHashKNN(k=K, backend=...)` without its required `k_bits`, raising `TypeError` inside `pytest.raises(ValueError)`; invisible to collect-only. Fixed with `k_bits=64`; the `ValueError` still comes from `check_backend` first. Zero skips (official rows genuinely ran): `test_official.py` 43/43, `test_silvertorch.py` all backends x modes, compile 3 rows zero graph breaks through the bound-method dispatch, export, moved transposed-sigs test, tune smoke 7 specs, 18 rejection cells.
Other gates: evaluation 93 passed / 1 skipped; ruff clean on touched files, 11 pre-existing E501s in untouched evaluation files and two unformatted kernel files, all byte-identical to `41d4479`; links 0; grep hits only the mandated tag name in kernels.md.
