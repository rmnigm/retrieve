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
