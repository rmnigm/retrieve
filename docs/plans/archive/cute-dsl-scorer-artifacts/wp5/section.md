
### `codesigned_probe_score_cute` — the CuTe DSL backend

[`silvertorch/codesigned_probe_score_cute.py`](../../../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cute.py)
(host side) +
[`silvertorch/cute/codesigned_probe_score.py`](../../../../retrieve/src/retrieve/kernels/silvertorch/cute/codesigned_probe_score.py)
(device side), selected by `SilverTorch(backend="cute")`. A one-to-one
port of the CUDA C++ backend above into NVIDIA's CuTe DSL
(`nvidia-cutlass-dsl`: kernels written as Python, traced to MLIR and
compiled in-process). Same three kernels plus the generic fallback, under
the same names (`cps_bloom_mask_kernel`, `cps_clause_mask_kernel`,
`cps_score_kernel`, `cps_score_kernel_generic`); same cluster-major
1-bit mask layout; same three ops argument for argument
(`codesigned_probe_score_cute`, `codesigned_probe_score_bloom_cute`,
`codesigned_probe_score_exact_cute` — `@torch.library.custom_op` with
`register_fake`, same cudagraph reasoning as the cuda ops); same
registered buffers — the layer registers `bloom_sigs_t` for `"cute"`
exactly as for `"cuda"`, and `build_transposed_sigs`,
`words_per_cluster` and the two `_prep`s are *imported* from the cuda
module rather than copied, so a cute checkpoint is byte-identical to a
cuda one. Everything the cuda section says about design, layout, the
perf model and numerics holds here unchanged; this section covers only
what the language changed. Plan, decisions and the spike record:
[cute-dsl-scorer.md](../../cute-dsl-scorer.md).

The correctness gate is the cuda one plus a direct cuda-vs-cute check.
[`tests/parity/test_codesigned_probe_score_cute.py`](../../../../retrieve/tests/parity/test_codesigned_probe_score_cute.py)
asserts `torch.equal` on the `[B, P]` score tensor against **Triton**
(as the cuda file does) and, in `test_cute_matches_cuda_bitexact`,
against the **cuda** backend on both the raw phase-2 mask words and the
full score buffer, across `filter_mode`, `D` and `unroll`. Returned ids
are gated up to permutation within tied scores, for the reason given in
the cuda section.

#### Translation table

| C++ | CuTe DSL |
|---|---|
| template parameters `<SEG, HAS_MASK, UNROLL>`, `<C, A>` | `cutlass.Constexpr[int]` / `Constexpr[bool]` kernel arguments; one `cute.compile` per constexpr tuple, memoized in a module dict keyed `("score", SEG, HAS_MASK, UNROLL)`, `("generic", HAS_MASK)`, `("clause", C, A)`, `("bloom",)`. `Constexpr` parameters are stripped from the compiled callable's signature — it is called with the runtime arguments plus the stream, in launcher order |
| `if constexpr (HAS_MASK)` | `if cutlass.const_expr(HAS_MASK):` |
| `#pragma unroll` over `UNROLL` / `C·A` | `cutlass.range_constexpr` — a plain `range` inside a kernel is an `scf.for` whose induction variable is a runtime value and cannot index a Python list |
| `__dp4a` | `cute.arch.inline_ptx("dp4a.s32.s32 …")` — `IDP.4A.S8.S8` in SASS |
| `__reduce_add_sync(seg_mask, v)` | `cute.arch.warp_redux_sync(v, "add", mask_and_clamp=seg_mask)` — `REDUX.SUM.S32` |
| `__ballot_sync` | `cute.arch.vote_ballot_sync` |
| `__ffsll(bits) - 1` | `math.cttz` on the `Int64` |
| `int4 __ldcs(row)` | `cute.arch.load(ptr, VectorType[4 × i32], cop="cs")` + four `extractelement`s — one `LDG.E.EF.128` per lane |
| 32-bit half store into the int64 mask word | `cute.recast_ptr(mask, dtype=Int32)` |
| `TORCH_CHECK` in the launchers | `ValueError` / `TypeError` in the host module, one for one; the DSL kernels carry no device-side checks at all |
| `at::Tensor` → `data_ptr` in the launcher | raw pointers via `make_ptr(dtype, t.data_ptr(), gmem, assumed_align=)` plus `Int64` / `Int32` scalars for every runtime shape; the launch goes on `cuda.CUstream(torch.cuda.current_stream().cuda_stream)`, with the wrapper cached per stream handle |

The pointer route is a decision, not a default. The DSL's `from_dlpack`
path costs ~7 µs per tensor per call once its layout is marked dynamic,
and *without* that mark it silently bakes the tensor's shape into the
compiled code with no call-time validation — a new `P` would run the old
kernel. Raw pointers plus scalar shapes bake in only dtype, address space
and alignment, so one compiled callable serves every shape and every
buffer, and every shape check stays in Python where it can raise.

#### Where the port had to deviate

Each of these is forced by the DSL and was verified not to change results
(a line-by-line semantic diff against the `.cu`, plus ~400 bit-exact
cases against the cuda backend covering the edges — sub-step `max_size`,
non-multiple `mask_words`, off-table `(C, A)`, misaligned bases):

- **Predicated per-item work lives in `@cute.jit` helpers** (`_ld_id`,
  `_mask_keep`, `_ld_row`) and the `UNROLL` arrays are rebuilt as list
  comprehensions. Assigning a list *element* inside a dynamic `if` /
  `while` escapes the DSL's region analysis (only whole names are
  yielded), so the C++ `keep[u] = …` under a branch is not expressible.
  Issue order is unchanged: all ids, then all mask tests, then all row
  loads, then the dots.
- **Early returns and `continue` are inverted into `if` bodies**
  (`if idx < mask_words:`, `if p_u < warp_end:`). The clause kernel's
  warp-uniform guard still wraps the whole body including the ballot, so
  every lane reaches it together.
- **Predicated loads compile to branches, not `@P LDG`.** nvcc turns the
  C++ ternaries into predicated loads; the DSL emits a real `if` region
  and a `REDUX.OR` warp-uniformity test before each row load. A few extra
  instructions per item, same addresses, same results.
- **No read-only-cache loads.** Every C++ `__ldg` (ids, mask words, query
  words, scales, attributes, `probe_ids`) is a plain `ld.global` here —
  `cute.arch.load` has no `nc` cache operator. Only the row `__ldcs`
  keeps its hint (`.cs`, `LDG.E.EF.128`).
- **`seg_reduce_add` is `redux.sync` only**, so the backend is sm_80+;
  the C++ `__shfl_xor_sync` butterfly for older parts is not ported.
  There is no capability probe in front of the compile, so on a
  pre-Ampere GPU the first compile fails as a plain `ImportError` (tests
  *fail*) rather than a `CuteMissing` skip — the review's R2, a one-line
  fix that has not landed.
- **The no-mask `UNROLL=1` build is auto-unrolled 5× by the DSL's LLVM
  pipeline** (20 `IDP.4A`, 5 `LDG.E.EF.128` per loop body); the C++
  `UNROLL=1` is one copy, and the masked builds are not unrolled (the
  carry `while` blocks it). Same arithmetic, so parity is untouched, but
  that one column of an `UNROLL` sweep is not structurally
  apples-to-apples with cuda's.

Smaller ones: the warp ordinal is `tidx // 32` rather than
`cute.arch.warp_idx()` (which costs a `shfl.sync`); `//` and `%` lower to
floor-semantics `floordivsi` / `remsi`, which equal C truncation because
every operand in these kernels is non-negative; the block size is a
runtime `num_warps * 32` as in the C++ launcher, which the DSL warns it
cannot turn into a `reqntid` hint (that one warning is filtered in
`_compile`).

#### Build, ops, and tuning

Nothing is compiled ahead of time and no toolkit is involved.
`nvidia-cutlass-dsl` is the optional extra `cute`
(`uv sync --all-packages --extra cute`; a plain `--extra cute` from
`retrieve/` prunes the `evaluation` workspace member); it ships its own
`ptxas` and targets the detected arch, so `nvcc`, `ninja` and `CUDA_HOME`
are not needed. Importing the host module never imports `cutlass`;
`is_available()` and `ensure_built()` are the probes, with the same split
as the C++ backend: `CuteMissing(ImportError)` (package absent, or no
CUDA device) is what `tests/conftest`'s `require_cps_cute()` *skips* on;
a DSL that is present and fails to import or compile is a plain
`ImportError` carrying the DSL's text, which the tests *fail* on. The
outcome is memoized once in a module global (`_DEV_MEMO`), as the cuda
module's `_EXT_MEMO` is.

Compilation is per process and per specialization, with no disk cache on
the `cute.compile` path: ~60–100 ms each, after a few seconds of one-time
DSL initialization. `ensure_built()` compiles `cps_score_kernel<8, false,
1>`; the first call of every other `(D, mask, unroll)` / `(C, A)` tuple
pays its own compile, and a sweep that spawns a process per config pays
all of it again per process, where the C++ extension is ninja-cached after
its first build. (The DSL's `@cute.jit` direct-call path does have a disk
cache but costs ~4.7 ms *per call*, so it is never on the hot path.)
Per-launch host overhead is higher than the C++ extension's — the DSL
adapts every pointer and scalar argument in Python on each call, and a
bloom / exact query launches twice; the measured cost, and what it does
to end-to-end numbers at small `B`, are in the plan's
[§5](../../cute-dsl-scorer.md#5-validation-record-wp-4). Registers:
34 at `UNROLL=1`, 48 at `UNROLL=4` for `cps_score_kernel<8, true, ·>`
(vs 30 / 40 for the C++ build), no spills, no `BAR.SYNC`, no shared
memory — read off the loaded cubin via `cuFuncGetAttribute`, since the DSL
prints no resource usage and `cuobjdump` needs a toolkit.

Config is `CodesignedProbeScoreCuteConfig(block_p, num_warps, unroll)`
with its own `DEFAULT_CONFIG`; re-tune with `uv run tune-kernels
codesigned-probe-score-cute` and `… codesigned-probe-score-exact-cute`,
which sweep the same grid as the cuda specs so the `--json-out` files
join on identical keys. Both paste into the same `DEFAULT_CONFIG` line,
so reconcile them first. The cross-backend caveats in the cuda section's
box (sparse vs dense `HAS_QB` inputs, `P` a multiple of the synthetic
`max_size`, no `-1` padding in the probe family) apply unchanged.

#### Constraints the port enforces

Same rows as the cuda table; every one is raised host-side, before any
launch, because the kernels have no device-side checks:

| constraint | raised host-side as |
|---|---|
| `D % 4 == 0` | `ValueError` — `dp4a` consumes 4 int8 lanes per word |
| `unroll ∈ {1, 2, 4}` | `ValueError` — a `Constexpr` of the scorer; the compile cache enumerates exactly these |
| `B ≤ 65535` | `ValueError` — batch rides `grid.y`, as for cuda |
| `block_p % num_warps == 0`, `1 ≤ num_warps ≤ 32` | `ValueError` |
| `P % max_size == 0` | `ValueError` — same slot → `(cluster, offset)` arithmetic in the mask test |
| `query_clause_attrs.size(1) == item_clause_attrs.size(1)`, `clause_is_reverse.shape == (C,)` | `ValueError` — on the `(0, 0)` path `C` is the runtime `n_clauses` argument |
| dtypes (`int8` codes, `fp32` scales / scores, `int64` ids / mask / attrs) and contiguity | `TypeError` / `ValueError`; `clause_is_reverse` is coerced with `.to(torch.bool)` and read as `Uint8`, so the "must be `torch.bool`" row of the cuda table becomes a coercion here |

Not in the table, as for cuda: `D ∈ {64, 128, 256}` is dispatch (anything
else with `D % 4 == 0`, or a code base pointer that is not 16-byte
aligned, takes `cps_score_kernel_generic`), and ids are trusted. Two
things the C++ does not have: a code base pointer that is not even
4-byte aligned raises an `AssertionError` from `make_ptr` on the host,
where the C++ would issue a misaligned load; and compiled callables bind
to the device that is current at their *first* call while the compile
cache is not keyed by device, so a second GPU in the same process would
launch into the wrong context (the review's R1 — single-GPU until the
per-device `JitExecutor.to(dev)` fix lands).

#### Performance

Kernel-only and end-to-end numbers, the head-to-head against the Triton
and cuda backends, and the reading are in the plan's
[§5](../../cute-dsl-scorer.md#5-validation-record-wp-4). This section
documents mechanism only.
