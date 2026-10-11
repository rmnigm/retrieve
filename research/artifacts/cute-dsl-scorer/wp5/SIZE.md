# WP-5 size accounting — CuTe DSL port vs CUDA C++ vs Triton

Counted 2026-09-02 on branch `feat/cute-dsl-scorer` (working tree, after the WP-3 R1/R2
fixes landed in the cute files; the benchmark agent's later trims may shift the cute
rows by a few lines). Script: `count.py` in this directory
(`cd retrieve && uv run python .../wp5/count.py`).

**Method.** Python files are classified with `tokenize` + `ast`: a line is *blank* if
whitespace-only, *comment/docstring* if every token on it is a `COMMENT` or belongs to
a docstring (`ast` body[0] string constants of the module / functions / classes), and
*code* otherwise (a code line with a trailing comment counts as code). CUDA lines are
*comment* if they start with `//` or lie inside a `/* … */` block. Table 2 splits the
two device-side files by role over explicit line ranges (listed), so the ranges include
each role's leading comment block / docstring. Like-for-like caveats: (1) the C++
"launchers" role carries the 40 `TORCH_CHECK` lines of validation that the DSL port
keeps in its host module (31 `raise` lines there), so the C++ device file is ~40 lines
"heavier" and the DSL host file correspondingly heavier for the same work; (2) the
Triton row counts only the two kernel files plus the `kernels/common.py` helpers those
two kernels actually import — `bloom_subset_pass` (and `or_combine`, which it uses) for
`codesigned_probe_score.py`, `clause_pass` for `codesigned_probe_score_exact.py`;
`quantize_int8` from `layers/utils/quantize.py` is shared by all three backends and is
counted in none. The two Triton files are one launch each and include their own
`_prep`/`_finish`/`_impl`/`@triton_op` host code, so they are compared against the full
backend (device + host) sums, not against the device files alone.

## Table 1 — per file

| file | total | blank | comment/docstring | code |
|---|---:|---:|---:|---:|
| cuda/codesigned_probe_score.cu (C++ device + launchers + pybind) | 672 | 43 | 127 | 502 |
| codesigned_probe_score_cuda.py (C++ host) | 611 | 80 | 107 | 424 |
| cute/codesigned_probe_score.py (DSL device + launchers + compile cache) | 678 | 85 | 126 | 467 |
| codesigned_probe_score_cute.py (DSL host) | 614 | 61 | 86 | 467 |
| codesigned_probe_score.py (Triton, none/bloom) | 310 | 41 | 40 | 229 |
| codesigned_probe_score_exact.py (Triton, exact) | 294 | 36 | 32 | 226 |
| common.py::bloom_subset_pass (lines 39-48) | 10 | 1 | 5 | 4 |
| common.py::or_combine (lines 17-19) | 3 | 0 | 0 | 3 |
| common.py::clause_pass (lines 51-91) | 41 | 1 | 8 | 32 |
| common.py helpers used, subtotal | 54 | 2 | 13 | 39 |

| **C++ backend (.cu + host .py)** | 1283 | 123 | 234 | 926 |
| **CuTe DSL backend (device .py + host .py)** | 1292 | 146 | 212 | 934 |
| **Triton (2 kernels + used common.py helpers)** | 658 | 79 | 85 | 494 |

## Table 2 — device-side files by role

### C++ `cuda/codesigned_probe_score.cu`

| role | lines | total | blank | comment/docstring | code |
|---|---|---:|---:|---:|---:|
| header: file comment, includes, `#error` sm_61 gate | 1–37 | 37 | 5 | 22 | 10 |
| device helpers: `seg_reduce_add`, constants, namespace | 38–56 | 19 | 2 | 4 | 13 |
| kernel bodies: bloom mask, clause mask, score, generic (incl. their comment blocks) | 57–414 | 358 | 18 | 76 | 264 |
| launchers + pybind: `cdiv`/`i64_ptr`, 3 launchers incl. `TORCH_CHECK`s, dispatch macros/switches, `PYBIND11_MODULE` | 415–672 | 258 | 18 | 25 | 215 |
| **all** | | 672 | 43 | 127 | 502 |

### CuTe DSL `cute/codesigned_probe_score.py`

| role | lines | total | blank | comment/docstring | code |
|---|---|---:|---:|---:|---:|
| header: module docstring, imports, constants | 1–43 | 43 | 8 | 24 | 11 |
| device helpers: `dp4a`, `load_int4`, `cttz64`, `seg_reduce_add`, `ld`, `st`, `@cute.jit` `_ld_id`/`_mask_bit`/`_mask_keep`/`_ld_row` | 44–83, 212–264 | 93 | 23 | 20 | 50 |
| kernel bodies: bloom mask, clause mask, score, generic (incl. docstrings) | 84–211, 265–429 | 293 | 23 | 52 | 218 |
| `@cute.jit` launchers + compile cache: 4 launchers, `gmem_ptr`, `cu_stream`, `_current_device`, `_compile`, 4 `compile_*`, per-device executor cache | 430–678 | 249 | 31 | 30 | 188 |
| **all** | | 678 | 85 | 126 | 467 |

TORCH_CHECK lines in the .cu launchers: 40
raise lines in the cute host module: 31

**Reading.** Like for like, the DSL port is the same size as the C++ backend it
translates: 934 vs 926 code lines over the backend (device + host), 467 vs 502 for the
device file alone (218 vs 264 in kernel bodies — the DSL's `Constexpr` arguments and
list comprehensions absorb the C++ template boilerplate, and its four `@cute.jit`
predicated-load helpers, 50 lines, replace ternaries; the DSL launcher role is 188 vs
215 lines, but the C++ one includes the 40 `TORCH_CHECK` lines). Both two-kernel
backends are ~1.9× the Triton pair (494 code lines) that computes the same function in
one fused launch.

## Dependency footprint

`uv sync --all-packages --extra cute` adds to the venv (`git diff uv.lock`):
`nvidia-cutlass-dsl 4.7.1` + `nvidia-cutlass-dsl-libs-{base,core,cu12}`,
`cuda-python` / `cuda-bindings 12.9.4` / `cuda-core` / `cuda-pathfinder`,
`nvidia-cuda-nvdisasm 13.3.73` (and `backports-strenum`, py<3.11 only). On disk
(`du -sh --apparent-size` under `.venv/lib/python3.11/site-packages`):

| package dir | size |
|---|---:|
| `nvidia_cutlass_dsl/` (DSL + MLIR + bundled `ptxas`; `dsl_packages/` 273 MB, `cu12/` 48 MB) | 323 MB |
| `cuda/` (`cuda-bindings` + `cuda-core` + `cuda-pathfinder`) | 130 MB |
| `nvidia/cu13/` (`nvidia-cuda-nvdisasm`) | 9.1 MB |
| **total added by the `cute` extra** | **~461 MB** |

Everything else under `nvidia/*` (cuBLAS, cuDNN, NCCL, …, ~4.4 GB) is the torch cu128
wheel's and is present in both configurations. No toolkit, `nvcc`, `ninja` or
`CUDA_HOME` is used by the cute path.

The C++ backend needs none of the above but does need a **system CUDA toolkit** whose
major version matches the torch wheel (cu128 → 12.x `nvcc`; this box: `/usr/local/cuda`
= CUDA 12.4, 4.7 GB apparent), `ninja` (retrieve's dev dependency group), and
`CUDA_HOME` if `nvcc` is off PATH; the JIT build is cached in `TORCH_EXTENSIONS_DIR`
after the first ~1-minute compile, where the DSL recompiles each specialization
(~60–100 ms) in every process.
