# WP-3 review — CuTe DSL port of `codesigned_probe_score.cu`, line-by-line semantic diff

Reviewed (read-only, 2026-09-02, A100-SXM4-80GB, nvidia-cutlass-dsl 4.7.1, torch 2.10.0+cu128):

- source of truth: `retrieve/src/retrieve/kernels/silvertorch/cuda/codesigned_probe_score.cu`
- port, device: `retrieve/src/retrieve/kernels/silvertorch/cute/codesigned_probe_score.py` (`cute/…` below)
- port, host: `retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cute.py` (`host …` below)
  vs `codesigned_probe_score_cuda.py`
- contract: `docs/system/kernels.md` § codesigned_probe_score_cuda; plan `docs/plans/cute-dsl-scorer.md`;
  spike `FINDINGS.md`; implementer `wp1/NOTES.md`

Method: every statement of the three kernels + generic fallback + three launchers was diffed against
the port; every place the DSL could plausibly diverge from C semantics was checked in the DSL source
(`base_dsl/typing.py`, `cute/arch/nvvm_wrappers.py`, `base_dsl/jit_executor.py`) or in the PTX the
implementer dumped (`wp1/dump/*.ptx`); everything the 317-case self-check did not reach was exercised
by two scripts in this directory (`edge.py`: 86 cases, all `torch.equal` against the cuda backend on the
full `[B, P]` buffer and on the mask words; `host.py` / `host_breakdown.py`: host-side behaviour and
per-call overhead). Rerun: `cd /workspace/retrieve/retrieve && uv run python <script>`.

---

## Findings, ranked

### BUG — none found

No statement in the port computes a different value, touches a different address, or reaches a
warp collective with a different lane set than the `.cu`. Details of what was checked are in
§ "Verified equivalences" below.

### RISK

**R1. Compiled callables are bound to the device that is current at their *first* call; the compile
cache is keyed without a device.** — `cute/codesigned_probe_score.py:599-613` (`_compiled`,
`_compile`), consumers at host `:183, :247, :333-339`.
- C++: `cudaLaunchKernel` under `OptionalCUDAGuard(tensor.device())` works on any device; the runtime
  holds a per-context image of every kernel.
- Port: `cute.compile` returns a `JitExecutor`; its `__call__` lazily creates `_default_executor =
  self.to(None)` bound to `cuCtxGetDevice()` at that moment (`jit_executor.py:1388-1420`) and the
  docstring says "Calling this method [on] multiple devices is not allowed and will result in
  unexpected CUDA errors". The `with torch.cuda.device(...)` guard in the host module makes the
  *first* call land on the right device, but a later call on `cuda:1` reuses the executor bound to
  `cuda:0`'s primary context — a wrong-context launch (invalid handle / silent failure), not an
  exception at the Python level.
- Repro: needs two GPUs (this box has one); reasoning is from the DSL source. Single-GPU use,
  including the eval driver and the tests, is unaffected.
- Fix (small): cache per device — `_executors[(key, dev)] = _compiled[key].to(dev)` with
  `dev = torch.cuda.current_device()` (inside the `with torch.cuda.device` block) or the tensor's
  `device.index`; `JitExecutor.to(device)` exists for exactly this (`jit_executor.py:1295`). One
  compile still serves all devices.

**R2. Pre-Ampere GPUs: a compile failure is reported as "the DSL is broken" (plain `ImportError`,
tests fail) instead of "cannot run here" (`CuteMissing`, tests skip).** — host `:96-103`;
`cute/… :68-70` (`seg_reduce_add` = `redux.sync` only).
- C++: `seg_reduce_add<SEG>` has a `__shfl_xor_sync` butterfly under `__CUDA_ARCH__ < 800`, so the
  extension builds and runs on sm_61..sm_75.
- Port: `redux.sync` is sm_80+ (plan §4 accepts this), but nothing maps the resulting ptxas error to
  `CuteMissing`. On a T4/V100 box `is_available()` is `False` (fine) while `require_cps_cute` will
  raise the loud `ImportError` and every cute test fails rather than skips.
- Fix: in `_import_dev`, before compiling: `if torch.cuda.get_device_capability() < (8, 0): return
  CuteMissing("CuTe DSL kernels need sm_80+ (redux.sync)")`. (Or port the butterfly under a
  `Constexpr` — `cute.arch.shuffle_sync_bfly` was verified in the spike — but the plan chose not to.)

### NIT

**N1. Contiguity checks are dead code.** host `:142-147, :162-164, :214-220`: `_check_i64_contiguous`
runs *after* `.contiguous()`, so the `is_contiguous()` branch can never fire. Harmless (the C++
`TORCH_CHECK` is equally unreachable from its Python wrapper). Either drop the branch or move the
check before the copies if you want it to mean something.

**N2. A base pointer that is not 4-byte aligned raises `AssertionError` (from `make_ptr`), not
`ValueError`.** host `:329-345`. The C++ has no check at all and would issue a misaligned
`ld.global.b32` → sticky device fault, so the port is strictly safer (`edge.py` T6b: `AssertionError:
pointer must be 4 bytes aligned`). Only reachable through a hand-built storage-offset view of an int8
buffer; the layer never produces one. Optional: `if q_codes.data_ptr() % 4 or item_codes.data_ptr()
% 4: raise ValueError(...)` next to the `vec_ok` test, for a message in the house style.

**N3. `B == 0` / `P == 0` surface as an opaque `DSLCudaRuntimeError: <unknown CUDA error code 9>`.**
The C++ raises `AcceleratorError: invalid configuration argument` (grid dim 0). Same class of failure,
context survives on both (`host.py`). A `if b == 0 or p == 0: raise ValueError` in `_cps_cute_scores`
and the two mask impls would be friendlier; not required for parity.

**N4. Only one tensor per launcher is checked `.is_cuda` (`query_bits`, `flat_items`, `q_codes`).**
host `:165, :221, :282`. Identical to the C++ `TORCH_CHECK`s, so not a divergence — but the DSL path
has no device-side safety net at all, and a CPU `clause_is_reverse` (the one tensor a user is most
likely to build by hand) would hand a host pointer to the kernel → sticky illegal-address. A
`t.device == ref.device` loop over all pointer arguments costs ~1 µs. Recommend adding it in both
backends' Python wrappers, not just this one.

**N5. `//` and `%` lower to `arith.floordivsi` / `remsi` (Python floor semantics with sign fix-ups),
not `divsi`.** `cute/… :112, :160, :168, :297-299, :326, :417`. All operands are provably
non-negative (`idx`, `word_idx`, `tidx`, `lane`, `p_first = warp_start + seg ≥ 0`, `p ≥ warp_start`),
so results equal C truncation — verified in the DSL source (`pyir_meta_table.py:382`) and by the
tests. Cost is a few extra compare/select instructions, all outside the hot loop. `tidx // 32`,
`lane // SEG`, `lane % SEG` could be `>> 5`, `>> log2(SEG)`, `& (SEG-1)` for free.

**N6. `warnings.catch_warnings()` mutates process-global warning state** (host-side, `cute/… :608-612`).
Not thread-safe by design; fine because compiles happen under the GIL from one thread in practice.
The module-level `_compiled` / `_streams` / `_DEV_MEMO` dicts are likewise plain dicts: a race between
two threads' first calls would at worst compile the same key twice and keep the last one — harmless.

**N7. `_import_dev` maps *any* `ModuleNotFoundError` whose top-level name is `cutlass` or `cuda` to
`CuteMissing`.** host `:85-91`. An old DSL missing e.g. `cutlass._mlir.dialects.llvm` would then be
a *skip* rather than a loud failure. Acceptable; noting the edge.

### PERF-NOTE (not bugs; for WP-4)

**P1. Per-call host overhead: ~59 µs for `_cps_cute_scores` vs 4.3 µs for the C++ `cps_scores`
(enqueue-only, GPU kept busy; `host.py`, `host_breakdown.py`).** Breakdown: the compiled callable
alone with all arguments prebuilt is 25.8 µs (13 µs of it is the DSL's Python `generate_execution_args`
— `typing.cast` + `_make_owning_c_pointer` for 6 pointers and 8 scalars — plus the launch); six
`gmem_ptr` builds 5.3 µs; `torch.cuda.current_stream()` + stream cache 5.0 µs; `with
torch.cuda.device(...)` 2.1 µs; the remaining ~20 µs is the Python validation block. A bloom/exact
query pays this twice (mask kernel + scorer): ~110 µs of host time per batch against ~10 µs for the
C++ path, on kernels that take 60–90 µs at the handoff regime. Kernel-only (`torch.profiler`)
numbers are unaffected; end-to-end / `do_bench` numbers at small B will show it. Levers if it matters:
fewer scalar arguments (each costs ~1.5 µs of adaptation), skipping `torch.cuda.current_stream()` when
the layer already knows the stream, and trimming the validation to the shape-dependent checks. The
~26 µs DSL floor is intrinsic to `JitExecutor.__call__` at this argument count.

**P2. No read-only-cache loads anywhere.** Every C++ `__ldg` (ids, mask words, query int4, `q_scales`,
`qb`, `sigs_t`, `probe_ids`, attrs, `is_reverse`) is a plain `ld.global` in the port
(`grep ld.global.nc wp1/dump/*.ptx` → 0). Only the row `__ldcs` keeps its hint (`ld.global.cs.v4.b32`,
`LDG.E.EF.128`). On A100 both paths go through L1 so the effect should be small, but it is a
systematic difference to remember when a regime disagrees. `cute.arch.load` has no `nc`
cop; `inline_ptx("ld.global.nc.b64 …")` would restore it (post head-to-head, per plan D8).

**P3. `cps_score_kernel<8, false, 1>` is auto-unrolled 5× by the DSL's LLVM pipeline** (20 `dp4a`,
5 `ld.global.cs.v4`, 5 `redux` in the PTX; NOTES §SASS); the C++ `UNROLL=1` build is one copy, and
the HAS_MASK builds are not unrolled (the carry `while` blocks it). So "cute UNROLL=1, no filter" is
structurally closer to cuda's UNROLL=4 with a prologue/epilogue. Same arithmetic, parity unaffected;
the UNROLL sweep will not be apples to apples for that one column. `cutlass.range(..., unroll=1)` on
the item loop would pin it if WP-4 wants an exact structural match.

**P4. Predicated loads became divergent branches.** `_ld_id` (`cute/… :216-222`) and `_ld_row`
(`:256-263`) are `if p_u < warp_end:` / `if keep:` regions; nvcc turns the C++ ternaries into `@P LDG`.
The SASS shows a `REDUX.OR` uniformity test before each row load. Extra instructions per item,
otherwise identical. (The `select_` clamp to row 0 in `_ld_row` is redundant given the branch but
keeps the address in range should the compiler if-convert — leave it.)

**P5. No disk cache for `cute.compile`**: ~60–100 ms per specialization per process, ~3 s for the
first one (DSL init; `ensure_built` log: "compiled … in 2.98s"). Sweeps that spawn a process per
config pay this on every process; the C++ extension is ninja-cached after the first build.

**P6. Registers: 34 / 48 (UNROLL 1 / 4) vs 30 / 40 for the C++** (NOTES). No occupancy cliff.

**P7. The exact path validates twice** (`_cpse_cuda_prep` then `_clause_partial_mask_cute_impl`);
a few µs, same as the cuda wrapper.

---

## Verified equivalences (so WP-4/5 need not re-check)

Address arithmetic and control flow
- Bloom: `idx = Int64(bidx)*bdim + tidx`, `pi = idx // wpc`, `w_in_c`, `col = cluster*wpc + w_in_c`,
  `m = qw*64 + j`, `m*sig_row_words + col`, store at `b*mask_words + idx` — identical, all Int64
  (Int64 ∘ Int32 promotes to Int64: `typing.py:889-895`). `~Int64(0)` identity verified: empty `QB`
  gives all `-1` words on both backends including the pad tail (T2). Set-bit walk with a word equal
  to `INT64_MIN` (only bit 63) — `bits & (bits - 1)` on *signed* Int64 wraps correctly, the DSL
  emits no `nsw` (T3, and `grep nsw typing.py` → none).
- Clause: `half_idx`, `word_idx = half_idx >> 1`, `h`, `slot = w_in_c*64 + h*32 + lane`,
  `span = b*P + pi*max_size`, `q_row = b*n_clauses`, `idv*(C*A) + i` / `(idv*n_clauses + c)*a_max + a`,
  store at `(b*mask_words + word_idx)*2 + h` through `recast_ptr(Int32)` — identical. Warp-uniform
  guard wraps the whole body incl. the ballot; both lane-divergent branches (`slot < max_size`,
  `idv >= 0`) close before it; `bit` is pre-declared `Boolean(False)` so the pad tail and `id < 0`
  give 0. `mask_words ∉ 4ℤ` (3, 10, 301, 257, 390 words → idle warps in the last block, multi-block
  bloom grid) verified (T4).
- `(C, A)` table `{(1,1),(1,2),(1,4),(2,1),(2,2),(2,4),(3,1),(3,2),(4,1),(4,2)}` equals the C++
  `switch` key set exactly; `(3,4)`, `(4,4)`, `(2,3)`, `(5,1)`, `(0,·)` all route to `(0,0)` on both
  (T7 incl. `C=0` → keep = `id >= 0`, and all-reverse flags). `is_reverse` read as `Uint8 != 0` = C++
  `unsigned char != 0`. Predicate `keep &= (match != rev) | (q == -1)` in the same order;
  `Boolean & Boolean` stays `Boolean` (`typing.py:971`).
- Scorer: `seg_mask`, `qv` at word `b*D/4 + sl*4` (= byte `b*D + sl*16`), `warp_start`,
  `warp_end = min(…, P)`, `p_first`, one `//` outside the loop, carry loops at both ends (`_mask_keep`
  and the loop tail) — identical. `max_size ∈ {1, 3, 7}` with `STEP ∈ {8, 16, 32}` (several spans per
  step), `items_per_warp ∈ {1, 8, 16, 32}` < or ∤ `STEP`, `P % block_p ≠ 0`, `num_warps ∈ {1, 3, 16, 32}`
  verified at D = 64/128/256 (T5). Dead-tail prefetch: `_ld_id` only forms/reads the address under
  `p_u < warp_end`. Phase (c) `if p_u < warp_end` = the C++ `continue`. The 1×1 dummy mask is only
  touched under `const_expr(HAS_MASK)`; `max_size = 0` division is likewise not traced.
- Generic: `q_row = b*d_words`, per-item `p // max_size`, `row = idv*d_words`, `for w in
  range(lane, d_words, 32)` (zero trips for `lane ≥ d_words`, D = 96), full-warp `redux` under a
  warp-uniform `keep` — identical. Misaligned base (`% 16 == 8`) routes to it on both backends and
  the result equals the fast path's (T6).
- Launch geometry: grids `cdiv(mask_words, 256)`, `cdiv(mask_words, 4)`, `cdiv(P, block_p)` × `B`;
  block 256 / `num_warps*32`; `items_per_warp = block_p // num_warps`. `B = 65535` runs, `B = 65536`
  raises on both (T8).

Overflow / dtype
- Every product the C++ does in `long long` is Int64 in the port (`b*P`, `b*mask_words`, `b*W`,
  `b*n_probe`, `id*D`, `m*sig_row_words`, `cl*wpc`, `idv*(C*A)`, `Int64(warp)*items_per_warp`,
  `Int64(bidx)*…`). `items_per_warp * (bdim // 32)` is Int32 × Int32 exactly as the C++ `int × unsigned`.
- `Float32(acc)` is `cvt.rn.f32.s32` in every scorer PTX (signed; the spike's `u32` was an artifact
  of its `a >= 0` guard). All-negative dot products bit-exact at every SEG and generic (T1).

Numerics / warp collectives
- `dp4a(rw[i], qv[i], acc)` in x,y,z,w order (`extractelement` 0..3 = `.x..w`), `redux.sync.add.s32`
  with `mask_and_clamp = seg_mask` (the membermask — `nvvm_wrappers.py:2570`), then exactly two
  `mul.f32`, zero `fma`, `0fFF800000` for `-inf`. `warp_redux_sync` / `vote_ballot_sync` are reached
  by all lanes named in their masks: `keep[u]` and `p_u < warp_end` are segment-uniform (same `p_u`,
  same id, same mask word/bit for the SEG lanes), the clause guard is warp-uniform.
- Large regime `N = 131072, P = 8192, B = 16`, all three modes × UNROLL 1/2/4: bit-exact (T9).

Host module
- Validation set vs the three `TORCH_CHECK` blocks: one-for-one (is_cuda, contiguity, all dtypes,
  `D % 4`, `item_codes D`, batch, `out P`, `B ≤ 65535`, `num_warps ∈ [1,32]`, `block_p` multiple,
  `unroll ∈ {1,2,4}`, `max_size > 0 ∧ P % max_size`, mask shape, `sigs_t` rows/width, `item_attrs.dim
  == 3`, `query_attrs.dim == 2`, C mismatch, `is_reverse` shape). `is_reverse` dtype is coerced with
  `.to(torch.bool)` exactly as the cuda wrapper does before its `TORCH_CHECK(kBool)`. Nothing the C++
  checks is missing.
- Memoization: `_DEV_MEMO` mirrors `_EXT_MEMO`; `CuteMissing` for no-GPU / no-package, plain
  `ImportError` for import-or-compile failure (see R2 for the sm_80 gap).
- Stream cache keyed on `torch.cuda.current_stream().cuda_stream`: `cuda.CUstream(handle)` is a
  non-owning integer wrapper, so a destroyed-and-reused handle simply resolves to the stream that now
  owns it — nothing stale is cached. torch's own streams are pooled (32 distinct handles out of 40
  `Stream()` objects, `host.py`), so the dict is bounded; a side stream works (T11).
- `torch.compile` / fake tensors: no `.item()`, no data-dependent host reads; `.data_ptr()` is only
  read in the real kernel (the `register_fake`s return shapes); `fullgraph=True` smoke passed in the
  self-check. `max_size` is a Python int argument.
- No `.view(torch.int32)` anywhere — pointers are recast via `make_ptr(Int32, …)`, which asserts
  alignment on the host (N2).

DSL-specific hazards
- `cute.compile` example args are `Int64(1)/Int32(1)/Float32(1)` and `make_ptr(dtype, 256, gmem,
  align)` — types only, nothing baked (verified: the callables ran on ~90 distinct shapes/buffers).
- Constexpr parameters are stripped: every call site passes exactly the runtime scalars + stream in
  launcher order (counted for all four launchers; an off-by-one would have segfaulted).
- No `from_dlpack` anywhere. Cache keys: `("bloom",)`, `("clause", C, A)`, `("score", SEG, HAS_MASK,
  UNROLL)`, `("generic", HAS_MASK)` — no shapes.
- `range` vs `range_constexpr`: list-indexed loops (`keep[u]`, `rw[u]`, `v[c*A+a]`, `q[c]`) use
  `range_constexpr`; the UNROLL arrays are rebuilt as whole-name list comprehensions through `@cute.jit`
  helpers (no list-element assignment inside a dynamic region anywhere); dynamic loops use `range` /
  `cutlass.range` with Int64/Int32 bounds; `lb > ub` gives zero trips (T5 tails, D = 96 lanes).
- The `UserWarning` filter is scoped to the compile and matches only "Dynamic variable in block size".

---

## Verdict

**0 BUG, 2 RISK** (R1 multi-GPU executor binding — untestable here, small fix; R2 pre-Ampere
failure mode is loud instead of a skip), 7 NIT, 7 PERF-NOTEs.

The port is semantically identical to the `.cu` on every path a single-GPU sm_80+ box can reach,
bit-exact on 317 + 86 cases including every edge the self-check missed. **It is ready for
benchmarking as-is.** WP-4 should (a) report kernel-only and end-to-end numbers separately, because
the ~55 µs/launch host-overhead delta (P1) is larger than the kernel-time differences being measured,
and (b) treat the no-filter UNROLL=1 column as auto-unrolled (P3). R1 and R2 are one-liners worth
landing before the backend is wired for multi-GPU or CI on older GPUs.
