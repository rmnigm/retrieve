# WP-1 notes — CuTe DSL port of `codesigned_probe_score.cu`

Box: A100-SXM4-80GB, `nvidia-cutlass-dsl` 4.7.1, torch 2.10.0+cu128. Branch `feat/cute-dsl-scorer`,
nothing committed. Scripts here rerun with `cd /workspace/retrieve/retrieve && uv run python <script>`.

## Files

| file | lines | role |
|---|---|---|
| `retrieve/src/retrieve/kernels/silvertorch/cute/codesigned_probe_score.py` | 646 | device module: 4 kernels, 4 jit helpers, 4 `@cute.jit` launchers, `compile_*` cache |
| `retrieve/src/retrieve/kernels/silvertorch/cute/__init__.py` | 0 | package marker |
| `retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cute.py` | 606 | host module: config, `CuteMissing`, `_impl`s, 3 custom ops + fakes |

Reference: `.cu` 672 lines (incl. launchers/pybind), cuda host module 611 lines.

## Self-check (`selfcheck.py`, `selfcheck.log`) — 317/317 PASS

`torch.equal` on the full `[B, P]` score buffers **and** on the mask words, cute vs cuda:

- modes none / bloom / exact × D ∈ {64, 128, 256, 96} × B ∈ {1, 16} × cute `unroll` ∈ {1, 2, 4}
  (with `(block_p, num_warps)` = (128, 8) / (256, 4) / (64, 2) resp.) × `max_size` ∈ {3, 64, 100, 768}
  = 288 cases (exact at C=2, A=2, reverse=mixed)
- exact `(C, A)` ∈ {(1,1), (2,2), (3,2), (4,2), (3,4)} × reverse {none, mixed} × D ∈ {128, 96} = 20 cases
  ((3,4) is off the register table and takes the `(0, 0)` runtime loop, as in the C++ switch)
- eager `_impl` path with `k = P`: 3 modes × D ∈ {64, 96} = 6 cases
- `torch.compile(fullgraph=True)` smoke of the three custom ops vs their eager result = 3 cases

Also: `uvx ruff check src/retrieve/kernels/silvertorch/` clean, `ruff format --check` clean on both
new files; `pytest tests/parity/test_codesigned_probe_score_cuda.py -q` → 47 passed (cuda path untouched);
`import_hygiene.py`: importing the host module does not import `cutlass`; `ensure_built()` raises
`CuteMissing` both when `cutlass` cannot be imported and when `CUDA_VISIBLE_DEVICES=""`.

## SASS sanity (`sass_dump.py`, dumps in `dump/`)

`CUTE_DSL_KEEP=ptx,sass,cubin`, kernel body only (the dump also contains the host stub). Register
counts read off the loaded cubin via `cuFuncGetAttribute(NUM_REGS)` (the DSL prints nothing itself;
no `cuobjdump` needed).

| kernel | IDP.4A | LDG.E.EF.128 | REDUX.SUM | FMUL | FFMA | BAR.SYNC | LDL/STL | regs | spill |
|---|---|---|---|---|---|---|---|---|---|
| `cps_score_kernel<8, true, 1>` | 4 | 1 | 1 (+1 REDUX.OR) | 2 | 0 | 0 | 0 | **34** | 0 B |
| `cps_score_kernel<8, true, 4>` | 16 | 4 | 4 (+4 REDUX.OR) | 8 | 0 | 0 | 0 | **48** | 0 B |
| `cps_score_kernel<8, false, 1>` | 20 | 5 | 5 (+5 REDUX.OR) | 10 | 0 | 0 | 0 | **32** | 0 B |
| `cps_score_kernel_generic<true>` | 7 | — (`LDG.E` ×15, 4-byte) | 1 | 2 | 0 | 0 | 0 | 32 | 0 B |
| `cps_bloom_mask_kernel` | — | — | — | — | — | 0 | 0 | 30 | 0 B |
| `cps_clause_mask_kernel<2, 2>` | — | — | — (1 VOTE) | — | — | 0 | 0 | 27 | 0 B |

Per item in `<8, true, 1>`: `LDG.E.EF.128` → 4× `IDP.4A.S8.S8` (x, y, z, w) → `REDUX.SUM.S32` →
`I2F` → `FMUL` → `FMUL` → `STG.E`. Query words: one `LDG.E.128` (no EF) at block start; `q_scale`
one `LDG.E`. The `REDUX.OR` before each row load is the DSL's warp-uniformity check on the
segment-uniform `if keep:` branch (also seen in the spike); the `I2F.U32.RP` / `I2F.U64.RP` are the
64-bit integer divisions (`p_first // max_size`, `idx // wpc`, `word_idx // wpc`) lowered through the
float-reciprocal sequence — one per thread outside the hot loop, as in the C++.

The `<8, false, 1>` build is auto-unrolled 5× by the DSL's LLVM pipeline (4 + remainder copies of the
body: 20 `IDP.4A`, 5 `LDG.E.EF.128`); the C++ build with `UNROLL=1` is one copy. Same arithmetic, so
parity is unaffected; worth remembering when comparing UNROLL sweeps (WP-4).

C++ reference (`cuobjdump -res-usage`, kernels.md): 30 regs at `UNROLL=1`, 40 at `UNROLL=4`. The DSL
build spends +4 / +8 registers — still far from any occupancy cliff.

## Deviations from the `.cu` structure (all forced by the DSL, none change results)

1. **Static loops are `cutlass.range_constexpr`**, not `range(UNROLL)`: in a `@cute.kernel` body every
   builtin `range` becomes an `scf.for` whose induction variable is an IR value, so `keep[u]` with a
   plain-`range` `u` fails ("Cannot use a Runtime value as a list index"). Not in FINDINGS.md — the
   spike's `range(SEG)` "unroll" only worked because `u` was used arithmetically.
2. **Per-item predicated work lives in `@cute.jit` helpers** (`_ld_id`, `_mask_keep`, `_ld_row`) and the
   `UNROLL` arrays are rebuilt as list comprehensions: assigning a list *element* inside a dynamic
   `if`/`while` is not tracked by the DSL's region analysis (only whole names are yielded), so the C++
   `ids_next[u] = p_u < warp_end ? ... : -1` / `keep[u] = ...` / `rw[u] = ...` become
   `xs = [helper(...) for u in range(UNROLL)]`. Issue order is unchanged: all ids, then all mask
   tests, then all row loads, then the dots.
3. **Early returns are inverted into `if` bodies** (`if idx < mask_words:`, `if word_idx < mask_words:`);
   the clause kernel's warp-uniform guard therefore still keeps every lane on the same path to the ballot.
4. **`if (p_u >= warp_end) continue;`** in phase (c) is `if p_u < warp_end:` around the dot/store.
5. **`__ldg` scalar loads are plain `ld.global`** (`LDG.E` / `LDG.E.64`), not `ld.global.nc`
   (`LDG.E.CONSTANT`) — the DSL's `make_tensor(ptr)[0]` load has no read-only-cache hint; the
   `__ldcs` row load keeps its `.cs` (`LDG.E.EF.128`) via `cute.arch.load(cop="cs")`.
6. **The row-0 clamp moved into `_ld_row`** (`select_(keep, id, 0)` then the predicated `.cs` load) —
   same expression as the C++ `(keep ? id : 0)` / `keep ? __ldcs(...) : 0`, just inside the helper.
7. **`seg_reduce_add` is `redux.sync.add` only** (sm_80+); the C++ butterfly for `__CUDA_ARCH__ < 800`
   is not ported (plan §4).
8. **Warp ordinal is `tidx // 32`**, not `cute.arch.warp_idx()` (which costs a `shfl.sync.idx`).
9. **The generic kernel's full-warp reduction** passes no mask (`warp_redux_sync` default 0xFFFFFFFF)
   instead of `seg_reduce_add(v, kFullMask)`.
10. **Block size is a runtime value** (`num_warps * 32`) as in the C++ launcher; the DSL warns it cannot
    emit a `reqntid` hint — that one warning is filtered in `_compile`.

## Gotchas found in this WP (add to FINDINGS.md)

- `for x in range(...)` inside a kernel is *always* `scf.for`; only `cutlass.range_constexpr` (or a
  list comprehension, which is not a `for` statement) keeps a Python-int induction variable.
- List-element assignment inside dynamic control flow silently escapes the region: rebuild the list.
- `@cute.jit` helpers may be called from kernels and may contain dynamic `if`/`while`; they are
  inlined at trace time. A helper that returns a list (the four row words) works through an `if`.
- `Boolean` supports `&`, `|`, `!=` (used for the clause predicate's AND / OR / XOR).
- `make_ptr(dtype, 256, gmem, assumed_align=al)` is a fine `cute.compile` example argument; only the
  dtype / address space / alignment are baked in (verified: the callable runs on other buffers).
- The package `__init__` re-exports the cuda *op* under the module's name, so scripts must
  `importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")`.
- Register counts: `cuFuncGetAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)` on `cuModuleLoadData(fn.__cubin__)`
  (with torch's context current) — no toolkit needed.
