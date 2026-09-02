# CuTe DSL feasibility spike — porting `codesigned_probe_score.cu`

Box: A100-SXM4-80GB, driver 580.159.03, Python 3.11.10, torch 2.10.0+cu128, triton 3.6.0.
Branch `feat/cute-dsl-scorer`, nothing committed. All scripts in this directory rerun with
`cd /workspace/retrieve/retrieve && uv run python <script>`.

## Step 1 — install

`retrieve/pyproject.toml` gained:

```toml
[project.optional-dependencies]
cute = [
    "nvidia-cutlass-dsl>=4.7",
]
```

Sync must be run as **`uv sync --all-packages --extra cute`** — a plain `uv sync --extra cute` from
`retrieve/` pruned the `evaluation` workspace member (its deps got uninstalled; re-running with
`--all-packages` restored them). `uv.lock` changed (+233/-20 lines).

Installed versions (`uv pip list`):

| package | version |
|---|---|
| nvidia-cutlass-dsl | 4.7.1 (`cutlass.__version__ == "4.7.1"`) |
| nvidia-cutlass-dsl-libs-base / -core / **-cu12** | 4.7.1 (cu12 runtime libs were selected; works with the 580 driver) |
| cuda-bindings / cuda-python | 12.9.4 |
| cuda-pathfinder | 1.5.4 |
| nvidia-cuda-nvdisasm | 13.3.73 (pulled in by the DSL for SASS dumps) |

`import cutlass, cutlass.cute as cute` and `import torch, triton` both work. No CUDA toolkit is needed
(the DSL ships its own compiler and runtime libs; `nvcc` is not consulted).

Import cost (cold process): `import torch` 13.6 s on this box, `import cutlass, cutlass.cute` +2.8 s
(3.6 s when imported first). `cuda.bindings.driver` is free.

Package location: `/workspace/retrieve/.venv/lib/python3.11/site-packages/nvidia_cutlass_dsl/dsl_packages/cutlass/`
(no examples ship with the wheel; the useful reference files are `cute/arch/nvvm_wrappers.py`,
`cute/runtime.py`, `base_dsl/env_manager.py`, `base_dsl/jit_executor.py`, `cutlass_dsl/cutlass.py`).

## Common imports used everywhere below

```python
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack, make_ptr
import cuda.bindings.driver as cuda           # cuda.CUstream is the stream arg type
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
```

---

## 1. Launch from `@cute.jit` with torch tensors on torch's stream — **works** (`spike01_launch.py`)

Both routes verified vs torch.

```python
@cute.kernel
def k_dlpack(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int64):
    tidx, _, _ = cute.arch.thread_idx(); bidx, _, _ = cute.arch.block_idx(); bdim, _, _ = cute.arch.block_dim()
    i = bidx * bdim + tidx
    if i < n:
        y[i] = x[i] * 2 + 1

@cute.jit
def host_dlpack(x: cute.Tensor, y: cute.Tensor, n: cutlass.Int64, stream: cuda.CUstream):
    k_dlpack(x, y, n).launch(grid=[(n + 255) // 256, 1, 1], block=[256, 1, 1], stream=stream)

host_dlpack(from_dlpack(x), from_dlpack(y), cutlass.Int64(n), stream)          # route (a)

@cute.kernel
def k_ptr(xp: cute.Pointer, yp: cute.Pointer, n: cutlass.Int64, stride: cutlass.Int64):
    ...
    xt = cute.make_tensor(xp, cute.make_layout(n, stride=stride))   # wrap raw ptr + dynamic stride
    yt = cute.make_tensor(yp, cute.make_layout(n, stride=stride))
    if i < n:
        yt[i] = xt[i] * 2 + 1

xp = make_ptr(cutlass.Int32, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)   # route (b)
host_ptr(xp, yp, cutlass.Int64(n), cutlass.Int64(stride), stream)
```

`make_ptr` asserts `ptr % assumed_align == 0` at construction (torch slices can violate this).
2-D via pointers: `cute.make_tensor(p, cute.make_layout((R, C), stride=(ld, 1)))` (spike09).

## 2. Scalar args and compile-time constants — **works** (`spike02_scalars_constexpr.py`)

```python
@cute.kernel
def k(out: cute.Tensor, a64: cutlass.Int64, a32: cutlass.Int32, f: cutlass.Float32, flag: cutlass.Boolean,
      SEG: cutlass.Constexpr[int], HAS_MASK: cutlass.Constexpr[bool]):
    for u in range(SEG):                      # plain python range over a Constexpr = static unroll
        acc = acc + a64 * (u + 1)
    for u in cutlass.range_constexpr(SEG):    # explicit static unroll (warns if >= 64 iters)
        acc = acc + a32
    if cutlass.const_expr(HAS_MASK):          # `if constexpr`: branch not even traced when False
        acc = acc + 1000000
    if flag:                                  # dynamic branch on a Boolean kernel arg
        acc = acc * 2
```

Call: `host(from_dlpack(out), cutlass.Int64(a64), cutlass.Int32(a32), cutlass.Float32(f), cutlass.Boolean(flag), SEG, HAS_MASK, stream)`.
Each distinct `(SEG, HAS_MASK)` is a separate compilation (like a template instantiation).
`cutlass.range(n, unroll=k)` / `unroll_full=True` exist for dynamic loops with unroll hints
(`base_dsl/ast_helpers.py` docstring: `unroll`, `unroll_full`, `prefetch_stages`, `vectorize`, `at_least_once`).

## 3. dp4a — **works** (`spike03_dp4a.py`)

No intrinsic; `cute.arch.inline_ptx` (a wrapper over `nvvm.inline_ptx` with named operands) does it and
SASS shows `IDP.4A.S8.S8`. Verified against `(A8.int32 * B8.int32).sum()` on random int8.

```python
def dp4a(a: cutlass.Int32, b: cutlass.Int32, c: cutlass.Int32) -> cutlass.Int32:
    return cute.arch.inline_ptx("dp4a.s32.s32 {$w0}, {$r0}, {$r1}, {$r2};",
                                write_only_types=[cutlass.Int32], read_only_args=[a, b, c])
```

Use it on `Int32` views of the int8 rows (`tensor.view(torch.int32)` on the host, or
`cute.recast_ptr(ptr, dtype=cutlass.Int32)` in the kernel).

## 4. Warp collectives and ids — **works** (`spike04_warp.py`)

```python
lane = cute.arch.lane_idx(); warp = cute.arch.warp_idx()          # warp_idx emits one shfl.sync.idx broadcast
tidx, _, _ = cute.arch.thread_idx(); bidx, bidy, _ = cute.arch.block_idx()
bdim, _, _ = cute.arch.block_dim(); gdim, _, _ = cute.arch.grid_dim()
seg = lane // SEG; seg_mask = ((1 << SEG) - 1) << (seg * SEG)     # Int32 expression

r = r + cute.arch.shuffle_sync_bfly(r, offset=off, mask=seg_mask)  # shfl.sync.bfly.b32 (butterfly step)
s = cute.arch.warp_redux_sync(v, "add", mask_and_clamp=seg_mask)  # redux.sync.add.s32 (sm_80+), lane-subset mask
packed = cute.arch.vote_ballot_sync(pred)                          # vote.sync.ballot.b32 -> Int32
```

`shuffle_sync_bfly` is `partial(shuffle_sync_op, kind=bfly)`; `shuffle_sync`, `shuffle_sync_up/down`
exist the same way (`cute/arch/nvvm_wrappers.py:779`). Verified SEG=4/8/16 segment sums, ballot bits, all ids.

## 5. 64-bit integer ops, cttz, dynamic loops — **works with caveat** (`spike05_int64_loops.py`)

`&`, `>>` (arithmetic on `Int64`, logical on `Uint64`), `<<`, `~`, `-`, comparisons all correct.
**Caveat: `//` and `%` on `Int64` follow Python floor semantics, not C truncation**
(`-0x123456789ABCDEF // 1000 == -81985529216487` = floor; `% 1000 == 105`). Identical to C for
non-negative operands (the only case in the kernel).

`__ffsll` equivalent and the bloom set-bit walk (bit-exact vs a Python reference):

```python
from cutlass._mlir.dialects import math as mlir_math

def cttz64(x: cutlass.Int64) -> cutlass.Int32:
    return cutlass.Int32(cutlass.Int64(mlir_math.cttz(x.ir_value())))

result = ~cutlass.Int64(0)
for qw in cutlass.range(W):            # dynamic trip count, W: cutlass.Int32
    bits = qb[qw]
    while bits != 0:                    # dynamic while loop is supported
        j = cttz64(bits)
        bits = bits & (bits - 1)
        result = result & sigs[(cutlass.Int64(qw) * 64 + j) * ncols + col]
```

`for i in range(lo, hi, 2)` with *dynamic* `Int64` `lo`/`hi` also works (preprocessed into `scf.for`).

## 6. 128-bit vectorised load with `.cs` — **works** (`spike06_vec_load.py`)

Both forms produce `LDG.E.EF.128` (evict-first = the `.cs` hint on sm_80) — one instruction per lane:

```python
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm

def load_int4_cs(ptr):                         # ptr: cute.Pointer to Int32, 16 B aligned
    vt = ir.VectorType.get([4], cutlass.Int32.mlir_type)
    v = cute.arch.load(ptr, vt, cop="cs")      # -> ld.global.cs.v4.b32
    return [cutlass.Int32(llvm.extractelement(v, cutlass.Int32(i).ir_value())) for i in range(4)]

def load_int4_ptx(ptr):                        # -> ld.global.cs.v4.s32
    return cute.arch.inline_ptx("ld.global.cs.v4.s32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
                                write_only_types=[cutlass.Int32] * 4, read_only_args=[ptr])
```

Pointer arithmetic is in elements: `xp + i * 4` for chunk `i` of an `Int32` pointer. Note the
single-brace vector syntax in `inline_ptx` — the `{{{$w0},...}}}` escaping form fails in ptxas.
`cute.arch.load` also accepts `level1_eviction_priority=` and `level_prefetch_size=` hints.

**Getting PTX/SASS/cubin out:**
- `CUTE_DSL_KEEP=ptx,sass,cubin,ir CUTE_DSL_DUMP_DIR=<existing dir>` writes
  `cutlass_<fn>_<mangled args>.sm_80.ptx/.cubin/.sass` (+ optional `ir`, `ir-debug`).
  The dir **must already exist**, otherwise you get an "ICE ... Cannot dump SASS: CUBIN file does not exist".
  KEEP also disables the JIT cache (warning printed).
- With KEEP set, the object returned by `cute.compile` exposes **properties** `__ptx__`, `__sass__`,
  `__cubin__`, `__mlir__` (`None` when KEEP is unset). `cute.compile(..., options="dump-ptx-path=... dump-cubin-path=...")` also exists (untested).
- `CUTE_DSL_PRINT_IR=1` prints MLIR; `CUTE_DSL_ARCH=sm_80` overrides auto-detection;
  `CUTE_DSL_LOG_TO_CONSOLE=1 CUTE_DSL_LOG_LEVEL=20` shows cache hits/misses; `CUTE_DSL_SHOW_STACKTRACE=1` for full tracebacks.

## 7. Predicated loads and 32-bit store into an int64 buffer — **works** (`spike07_pred_store32.py`)

```python
v = cutlass.Int32(0)
if keep:                                   # dynamic Boolean; assign-in-branch needs a pre-declared same-typed var
    v = rows[idv]
row = cutlass.select_(keep, idv, cutlass.Int64(0))   # ternary (lives in `cutlass`, not `cute`)
packed = cute.arch.vote_ballot_sync(keep)
if lane == 0:
    m32 = cute.recast_ptr(mask64, dtype=cutlass.Int32)        # Int64 buffer viewed as Int32
    cute.make_tensor(m32, cute.make_layout(n_words * 2))[word * 2 + half] = packed
```

Verified: predicated gather == `torch.where`, ballot halves reassemble to the int64 reference words.
In the mini scorer (item 12 below) the predicated 16-byte load `if keep: rw = load_int4_cs(...)` on a
segment-uniform predicate compiles to plain `LDG.E.EF.128` guarded by a `REDUX.OR` uniformity check.

## 8. fp32 epilogue and `-inf` — **works** (`spike08_fp32_epilogue.py`)

```python
score = cutlass.Float32(-math.inf)
if a >= 0:
    score = cutlass.Float32(a) * qs[i] * g       # (float(a) * q_scale) * global_scale
out[i] = score
```

PTX: exactly two `mul.f32` (no `fma`), constant `0fFF800000`; SASS: `2 FMUL`, `I2F`. `torch.equal`
against `acc.float() * qs * g` on 100k values. (LLVM emits `cvt.rn.f32.u32` inside the `a >= 0` branch —
equivalent to `cvt.rn.f32.s32` there.)

## 9. Dynamic shapes / no recompilation — **works with caveat** (`spike09_dynamic_shapes.py`)

```python
def mk(t):  # row-major 2-D: shapes AND row stride dynamic, contiguous dim 1
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
compiled = cute.compile(host2d, mk(x0), mk(y0), stream)         # 0.11 s
compiled(mk(x), mk(y), stream)                                  # any (R, C), padded rows too: no recompile, correct
# alternative: from_dlpack(t).mark_compact_shape_dynamic(mode=i, stride_order=t.dim_order(), divisibility=d)
#   (d must divide the example tensor's extent, else RuntimeError at marking time)
```

`x.shape` inside the kernel/host then yields dynamic values (used to compute the grid).
**Caveat — no validation:** a callable compiled from an *unmarked* `from_dlpack` tensor bakes the shape and
strides into the code; calling it with another shape raises **no error** and silently reads/writes with the
old layout (4000 of 65536 elements written in the test, plus OOB traffic when the new tensor is smaller).

`make_ptr` route: shapes/strides are ordinary `Int64` kernel args, nothing to mark; compile once with
`cutlass.Int64(...)` examples, then call with plain Python ints (`cp(P(x), P(y), R, C, ld, stream)`) — verified
for three shapes.

## 10. Overheads and caching — measured (`spike10_overheads.py`, `spike10b_cache.py`)

Compile (trivial empty kernel, warm process, after DSL init): `cute.compile` **80 ms** (from_dlpack signature)
/ **57 ms** (make_ptr signature); second compile of the same function 52 ms; first compile in a fresh
process 83–90 ms after the 15 s of imports (torch dominates). First direct `@cute.jit` call: 343 ms
(cache miss) / 251 ms (disk-cache hit).

Per-launch host overhead, compiled callable, 2000 launches, no sync (µs):

| path | µs/launch |
|---|---|
| compiled, `from_dlpack` wrapper pre-built | 8.6 |
| compiled, `make_ptr` wrapper pre-built | 10.3 (kernel has one more scalar arg) |
| compiled, `make_ptr` built per call | 12.0 |
| compiled, `from_dlpack(...).mark_layout_dynamic()` built per call | 25.8 |
| **direct `@cute.jit` call (in-memory cache hit)** | **~4700** |
| baseline `x.add_(0.0)` torch launch | 5.0 |
| `from_dlpack(t)+mark` alone / `make_ptr` alone / `cuda.CUstream(...)` alone | 7.2 / 0.9 / 4.5 |

Persistent cache: yes, but only for the direct-call path. `CUTE_DSL_CACHE_DIR` (default
`/tmp/<user>/cutlass_python_cache`, here `/tmp/root/cutlass_python_cache`) holds `cute_dsl_<hash>.mlir`
with the embedded cubin; a second process logs `JIT cache hit IN-FILE`. **`cute.compile` never consults it**
(`module_hash=[None]`, always recompiles, ~60–100 ms per specialisation). Disable with
`CUTE_DSL_DISABLE_FILE_CACHING=1` / `CUTE_DSL_NO_CACHE=1`; `CUTE_DSL_KEEP=...` disables it implicitly.
AOT export (`cute.compile_to`, `base_dsl/export`) exists but was not tested.

## 11. `torch.compile` / `torch.library.custom_op` — **works** (`spike11_torch_compile.py`)

```python
@torch.library.custom_op("spike::triple", mutates_args=())
def triple(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    compiled(P(x), P(y), x.numel(), cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    return y
@triple.register_fake
def _(x): return torch.empty_like(x)
f = torch.compile(lambda t: triple(t + 1) - 2, fullgraph=True)   # OK, also on a new shape and on a side stream
```

## 12. sm_80 targeting — **works** (`spike01_launch.py`, all others)

Arch auto-detected from the device (`env_manager.detect_gpu_arch`; artifacts are `*.sm_80.ptx`,
`sm_80` cubins). All used features (SIMT loads/stores, dp4a, redux, shfl, vote, inline PTX) compile and run.
No toolkit/driver requirement errors were hit; the DSL runs ptxas in-process from its bundled cu12 libs.
The `nvdisasm warning : Disassembling Std Elf to Old format ...` lines on SASS dumps are harmless.

## End-to-end proof — `spike12_mini_scorer.py`

A SEG-lanes-per-item int8 scorer (per-lane int4 `.cs` load, 4×dp4a, `redux.sync.add` over the segment mask,
predicated loads, fp32 epilogue, `-inf` for `id < 0`, per-warp item ranges, `Constexpr` SEG) is
**bit-exact (`torch.equal`) against the torch reference for D=64/128/256** with ~10% padded ids.
SASS of the D=128 build: `LDG.E.EF.128`, `IDP.4A.S8.S8` (4 per item), `REDUX.SUM.S32`, `I2F`, 2×`FMUL`, `STG.E`.

---

## Gotchas

1. **Python ints given to `@cute.jit` params annotated `cutlass.Int64/Int32` are treated as constants** on the
   direct-call path: they go into the mangled name (`cutlass_host_Ptrgmem_1000_1_...`) and every new value is a
   JIT cache miss (~56 ms). Pass `cutlass.Int64(n)` (mangles as `_`, dynamic). For `cute.compile`, give
   `cutlass.Int64(...)`/`cutlass.Float32(...)` example args; the compiled callable then accepts plain ints/floats.
2. **Never call `@cute.jit` functions directly in the hot path** (~4.7 ms/call even on a cache hit); use
   `cute.compile(fn, *example_args)` once and call the returned object (~9–12 µs).
3. **`Constexpr` params are stripped from the compiled callable's signature.** Call it *without* them
   (`compiled(*runtime_args, stream)`). Passing them shifts the positionals with no validation — a Python int
   ended up in the `CUstream` slot and the process **segfaulted** (exit 139, buffered stdout lost).
4. Kernels must live in real source files: the AST preprocessor calls `inspect.getsource`; `exec`'d strings /
   REPL definitions fail with "could not get source code".
5. `cutlass.select_`, `cutlass.min/max`, `cutlass.const_expr`, `cutlass.range`, `cutlass.range_constexpr`
   are in `cutlass`, not `cutlass.cute`.
6. `Int64 // x` and `% x` are Python floor semantics; use only on non-negative operands or use `Uint64`.
7. A variable assigned inside a dynamic `if` must exist before the `if` with the same type
   (`v = cutlass.Int32(0); if keep: v = ...`). Lists of `Numeric`s (e.g. the four int4 words) can be
   loop-invariant registers and can be reassigned inside a dynamic `if` (worked in spike12).
8. Indexing a size-1 layout tensor with a dynamic coordinate is *not* a pointer offset
   (`cute.make_tensor(p, cute.make_layout(1))[b]` read the wrong element). Offset the pointer:
   `cute.make_tensor(p + b, cute.make_layout(1))[0]`.
9. Unmarked `from_dlpack` tensors bake shape+stride into the compiled code and are **not validated** at call
   time (silent wrong results/OOB). Always `mark_layout_dynamic`/`mark_compact_shape_dynamic`, or use pointers.
10. `mark_compact_shape_dynamic(divisibility=d)` raises if the *example* extent is not divisible by `d`.
11. `make_ptr(..., assumed_align=16)` asserts alignment on the host; `cute.arch.load` of a 16 B vector needs it.
12. `CUTE_DSL_DUMP_DIR` must exist; `CUTE_DSL_KEEP` disables the JIT cache; `KEEP=sass` needs the bundled
    `nvidia-cuda-nvdisasm` (installed automatically).
13. `__ptx__/__sass__/__cubin__/__mlir__` on the compiled object are properties, `None` unless KEEP was set
    at compile time.
14. Many "ICE"/"Runtime Crash"/"This is a bug in the DSL" messages were caused by user errors (missing dump dir,
    wrong argument count). Re-run with `CUTE_DSL_SHOW_STACKTRACE=1` before believing them.
15. `uv sync --extra cute` from the `retrieve/` member prunes the other workspace member; use
    `uv sync --all-packages --extra cute`.
16. `from_dlpack` shapes/strides are i64 by default (`Tensorgmemoi64...`); `use_32bit_stride=True` exists.
17. `cute.arch.warp_idx()` costs a `shfl.sync.idx` broadcast; `lane_idx()` is free.

## Recommendation: tensor-passing route for the per-batch launch

Use **`make_ptr` + `Int64` scalars** for the production scorer:

- Cheapest per call: 0.9 µs per pointer vs 7.2 µs per `from_dlpack(...).mark_layout_dynamic()`; with 6
  tensors per launch that is ~5 µs vs ~45 µs of pure host wrapping. Also pre-build the `cuda.CUstream`
  wrapper per torch stream (4.5 µs each) or cache it keyed on `torch.cuda.current_stream().cuda_stream`.
- The CUDA kernel already takes raw pointers plus `P`, `max_size`, `mask_words`, `wpc`, `items_per_warp`
  as scalars, so the port maps 1:1 and shapes are dynamic by construction — no risk of the silently
  baked static layout of an unmarked `from_dlpack` tensor.
- It keeps the exact `int4`-chunk addressing (`cute.recast_ptr(..., Int32)` + element offsets) that gives
  the single `LDG.E.EF.128` per lane, and the 32-bit half stores into the int64 mask via `recast_ptr`.
- Cost: no dtype/shape validation at all (same as the C++ extension); keep the existing Python-side
  checks and the `assumed_align=16` assumption (rows are 16 B aligned by the layout contract).

Compilation strategy: one `cute.compile` per `(SEG, HAS_MASK, UNROLL)` at first use (~60–100 ms each, no
disk cache for this path), held in a module-level dict; wrap the compiled callable in a
`torch.library.custom_op` so it composes with `torch.compile`. `from_dlpack` remains fine for tests and
for the generic fallback kernel where layout-aware `cute.Tensor` indexing is more convenient.

## Files

`spike01_launch.py`, `spike01b_scalar_dynamic.py`, `spike02_scalars_constexpr.py`, `spike03_dp4a.py`,
`spike04_warp.py`, `spike05_int64_loops.py`, `spike06_vec_load.py`, `spike07_pred_store32.py`,
`spike08_fp32_epilogue.py`, `spike09_dynamic_shapes.py`, `spike10_overheads.py`, `spike10b_cache.py`,
`spike11_torch_compile.py`, `spike12_mini_scorer.py`; PTX/SASS dumps in `dump/`, `dump6/`, `dump8/`;
`cache_test/` is the disk-cache dir used by spike10b.
