# CuTe DSL port of the CUDA SilverTorch backend — plan

> **Status:** planned 2026-09-02 on `feat/cute-dsl-scorer` (branched from
> `refactor/kernels-eval` at `0f7792c`). Target: A100-SXM4-80GB, driver 580 / CUDA 13.0
> driver, torch 2.10.0+cu128, triton 3.6.0, `nvidia-cutlass-dsl` 4.7.x, Python 3.11.
> Authored on the GPU box, so every step below is validated as it lands.
>
> Question this plan answers: *how big is the CUDA SilverTorch kernel once rewritten in
> the CuTe DSL, and does it keep the CUDA C++ backend's speed (and its margin over the
> Triton kernel)?* Everything else — API shape, tests, docs — exists to make that
> comparison fair and reproducible.

## 1. Where it starts

The CUDA C++ backend (`backend="cuda"`) is three kernels in
[`kernels/silvertorch/cuda/codesigned_probe_score.cu`](../../retrieve/src/retrieve/kernels/silvertorch/cuda/codesigned_probe_score.cu)
(672 lines incl. launchers/pybind) plus a 611-line Python host module
[`codesigned_probe_score_cuda.py`](../../retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py):

| kernel | role | shape of the work |
|---|---|---|
| `cps_bloom_mask_kernel` | phase 2, bloom | thread-per-output-word; loop over the *set bits* of the query signature (`__ffsll`, `bits &= bits-1`), `and.b64` per 64 items against the transposed index |
| `cps_clause_mask_kernel<C, A>` | phase 2, exact | thread-per-slot, warp-per-32-bit-half-word; `C·A` attribute words into registers, predicate, `__ballot_sync`, lane 0 stores the half |
| `cps_score_kernel<SEG, HAS_MASK, UNROLL>` | phase 3 | `SEG = D/16` lanes per item, one `int4` (`__ldcs`) per lane, 4× `__dp4a`, `__reduce_add_sync` over the segment, fp32 epilogue, `-inf` for rejected; division-free `(cluster, slot)` tracking, ids prefetched one iteration ahead |
| `cps_score_kernel_generic<HAS_MASK>` | phase 3 fallback | warp-per-item, runtime `D/4` word loop — off-table `D` and misaligned rows |

Measured on A100 (handoff §13, `D=128`, `P=58 368`, kernel-only µs): no filter B=16
87.8 (Triton 85.5); bloom B=16 58.6 (Triton 124.6); exact B=16 95.3 + 56.6 clause mask
(Triton 121.7). Those are the numbers the port is held to.

## 2. Decisions

**D1 — a fourth backend, `backend="cute"`, not a replacement.** The C++ backend stays
as-is: it is the bit-exact reference the port is tested against and the baseline the
port is benchmarked against. `Backend = Literal["torch", "triton", "cuda", "cute"]`.
Removing the C++ backend (if the port wins) is a separate decision after the numbers.

**D2 — same op contract, same buffers, same layout.** The cute ops take exactly the
cuda ops' signatures (`codesigned_probe_score_cute`, `..._bloom_cute`, `..._exact_cute`),
consume the same transposed `bloom_sigs_t` (`build_transposed_sigs`, `words_per_cluster`
are imported from the cuda module, not copied), and write the same cluster-major
1-bit mask layout. The layer registers buffers for `"cute"` exactly as for `"cuda"`,
so a cute checkpoint is byte-identical to a cuda checkpoint. `_cps_cuda_prep` /
`_cpse_cuda_prep` are reused for validation + `quantize_int8`.

**D3 — all three kernels are ported, plus the generic fallback.** The scorer is the
headline, but "bloom mode on cute" needs the bloom mask and "exact mode on cute"
needs the clause mask; a backend that only scores is not a backend the layer can
route to. The generic warp-per-item fallback is ~60 lines and keeps off-table `D`
(the opcheck test uses `D=96`) correct.

**D4 — template parameters become `cutlass.Constexpr` kernel arguments and a
compiled-callable cache keyed on them.** `(SEG, HAS_MASK, UNROLL)` for the scorer,
`(C, A)` for the clause mask. Runtime shapes (`P`, `max_size`, `mask_words`, `wpc`,
`items_per_warp`, `sig_row_words`, strides) are `Int64`/`Int32` scalars so that a new
shape never recompiles. Tensor passing route (raw pointers via `make_ptr` vs
`from_dlpack` with dynamic layout marks) is decided by the spike (§4).

**D5 — bit-exactness is the correctness gate, exactly as for cuda.** Same `dp4a`
instruction, same integer segment reduction (`redux.sync.add` on sm_80+, or a
`shfl.bfly` butterfly — both exact), same two left-associated fp32 multiplies, same
`-inf`, same mask bits. Gates: `torch.equal` on the `[B, P]` score tensor and on the
mask words against the **cuda** backend, and against Triton (transitively the same
thing, asserted directly anyway). No `fast-math`-style flags in the DSL compile.

**D6 — config mirrors cuda's.** `CodesignedProbeScoreCuteConfig(block_p, num_warps,
unroll)` with its own `DEFAULT_CONFIG`, tuned by a `codesigned-probe-score-cute` spec
(+ `-exact-cute` twin) on the same grid as the cuda specs so the sweeps join on keys.

**D7 — optional dependency, lazy import, split failure modes.** `nvidia-cutlass-dsl`
is an optional extra `cute` (`uv sync --extra cute`). Importing the module never
imports `cutlass`; `CuteMissing(ImportError)` (package absent / no CUDA device) is
what tests *skip* on, a compile failure is a plain `ImportError` tests *fail* on —
the same split `require_cps_cuda` makes for the C++ backend.

**D8 — no new algorithms.** The port reproduces the C++ kernels' structure one to one
(lane layout, prefetch, predicated loads, carry loop, ballot packing). Any DSL-specific
optimization comes *after* the head-to-head, as a separate commit, so the size and
speed comparison is a comparison of languages, not of designs.

## 3. Work packages (one fresh-context executor each)

| WP | what | depends on |
|---|---|---|
| 0 | **spike** — install the DSL into the venv; prove each primitive the port needs with a running micro-kernel (launch on torch's stream, scalar args, `Constexpr` specialization, `dp4a`, shuffle/redux/ballot, Int64 ops + `cttz`, dynamic `while`, 128-bit loads with `.cs`, predicated loads, fp32 epilogue, dynamic shapes without recompiles, compile + per-call overhead, custom-op/`torch.compile` interplay, sm_80). Deliverable: `FINDINGS.md` with copy-pasteable snippets | — |
| 1 | **kernels + host module** — `kernels/silvertorch/codesigned_probe_score_cute.py`: the four kernels, the compile cache, `_impl`s, three custom ops + fakes, `is_available` / `ensure_built` / `CuteMissing`; a self-check script vs the cuda backend | 0 |
| 2 | **wiring + tests** — `Backend` literal, layer routing (`_forward_cute`, buffer registration), tune specs, `require_cps_cute`, parity test file (bit-exact vs cuda and Triton, mask-vs-mask, config override, tiny-`max_size`, opcheck), backend rows in correctness / compile / export tests, eval driver `--backend cute` | 1 |
| 3 | **review** — fresh-eyes semantic diff of the DSL kernels against the `.cu`, line by line; fix findings; full suite green | 2 |
| 4 | **tune + benchmark** — sweep both cute specs and paste `DEFAULT_CONFIG`; kernel-only `torch.profiler` split and `do_bench` head-to-head triton / cuda / cute at the handoff §13 regimes; SASS check (`IDP.4A`, `LDG.E.128`, no `BAR.SYNC`), registers/thread, compile time, per-call launch overhead; write §5 | 3 |
| 5 | **docs + commit** — `kernels.md` section, `testing.md`, this plan's status; commit(s) | 4 |

## 4. Spike findings (WP-0, 2026-09-02)

Full record with rerunnable scripts: the session scratchpad `spike/FINDINGS.md` (the
durable facts are repeated here). `nvidia-cutlass-dsl 4.7.1` (cu12 libs, cuda-bindings
12.9.4) installs as the `cute` extra; sync with `uv sync --all-packages --extra cute`
(a plain `--extra cute` from `retrieve/` prunes the `evaluation` workspace member). No
toolkit / `nvcc` is needed: the DSL ships ptxas and targets the detected `sm_80`.

Every primitive the port needs exists and was verified bit-exact on the A100:

| need | DSL form | evidence |
|---|---|---|
| `__dp4a` | `cute.arch.inline_ptx("dp4a.s32.s32 {$w0}, {$r0}, {$r1}, {$r2};", write_only_types=[Int32], read_only_args=[a, b, c])` | `IDP.4A.S8.S8` in SASS |
| `__reduce_add_sync(seg_mask)` | `cute.arch.warp_redux_sync(v, "add", mask_and_clamp=seg_mask)` | `REDUX.SUM.S32` |
| `__shfl_xor_sync` / `__ballot_sync` | `cute.arch.shuffle_sync_bfly(v, offset, mask=)` / `cute.arch.vote_ballot_sync(pred)` | verified |
| `int4` `__ldcs` row load | `cute.arch.load(ptr_i32, ir.VectorType.get([4], Int32.mlir_type), cop="cs")` + `llvm.extractelement` | one `LDG.E.EF.128` per lane |
| `__ffsll` set-bit walk | `mlir_math.cttz(x.ir_value())`, `while bits != 0:` (dynamic `while` is supported) | bit-exact vs reference |
| `if constexpr` / templates / `#pragma unroll` | `cutlass.Constexpr[int]` args, `cutlass.const_expr(...)`, plain `range(CONST)` / `cutlass.range_constexpr` | one compilation per constexpr tuple |
| 32-bit half store into an int64 mask | `cute.recast_ptr(mask_ptr, dtype=Int32)` | verified |
| fp32 epilogue | `Float32(acc) * q_scale * global_scale` → exactly two `mul.f32`, no fma; `Float32(-math.inf)` | `torch.equal` on 100k values |
| torch interop | `make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)` + `Int64` scalars; launch on `cuda.CUstream(torch.cuda.current_stream().cuda_stream)` | works eagerly, in a `custom_op`, and under `torch.compile(fullgraph=True)` |

Costs: `cute.compile` ≈ 60–100 ms per specialization (no disk cache on this path; the
`@cute.jit` direct-call disk cache exists but that path costs ~4.7 ms **per call**, so
it is never used in the hot path). Per-launch host overhead of a compiled callable
≈ 9–12 µs (`make_ptr` route; a torch op launch is ≈ 5 µs); `from_dlpack(...)` +
`mark_layout_dynamic` costs ≈ 7 µs per tensor and an *unmarked* `from_dlpack` tensor
silently bakes its shape into the code with no call-time validation.

Decisions taken from the spike:

- **Pointers + scalars** (`make_ptr` route), 1:1 with the C++ launcher signatures; the
  compiled callable is built once per `(SEG, HAS_MASK, UNROLL)` / `(C, A)` and held
  in a module dict. `Constexpr` params are *stripped* from the compiled callable's
  signature — call it with runtime args + stream only.
- The `cuda.CUstream` wrapper is cached per torch stream handle (4.5 µs to build).
- `Int64 // x` and `% x` are floor-semantics; every operand in these kernels is
  non-negative, so they match C. The one divide stays outside the hot loop as in C++.
- sm_80+ only (`redux.sync`); the C++ backend's sub-sm_80 butterfly fallback is not
  ported.

## 5. Validation record (WP-4) — 2026-09-02, A100-SXM4-80GB, driver 580.159.03 / torch 2.10.0+cu128 (CUDA 12.8 runtime), triton 3.6.0, nvidia-cutlass-dsl 4.7.1

Executed on the box in this order (scripts + raw outputs in the session scratchpad
`wp4/`: `run_tune.sh`, `h2h.py`, `kernel_only.py`, `host_overhead.py`, `host_trim.py`,
`p3_bench.py`, `compile_facts.py` / `compile_fresh.py`, `*.txt|json`):

- review fixes R1 (executor per `(specialization, device)` via `JitCompiledFunction.to(device)`)
  and R2 (`get_device_capability() < (8, 0)` → `CuteMissing`) landed first; parity 82/82.
- tune: all six specs in one session (`codesigned-probe-score{,-exact}{,-cuda,-cute}`,
  `--json-out`), 24-point cuda/cute grid × 6 + 5 regimes; then the shared-input head-to-head
  (handoff §6b bloom, §6c exact, plus a no-filter variant; `torch.equal` on the `[B, P]`
  scores across all three backends asserted before every timing; `do_bench(rep=500,
  warmup=100)`), the `torch.profiler` kernel-only split (`ncu` is blocked in this container,
  as in handoff §13), the host-enqueue measurements, the P1 trims, the P3 check, then the
  head-to-head and kernel-only split again on the final code.
- gates: parity 82/82 (cute file) after every change; `tests/parity tests/compile` 286
  passed; `ruff check src tests` clean.

**Tune, best-vs-best** (`do_bench` ms incl. the `topk` epilogue; `cute₀` = as ported,
`cute` = after the host trims below; ratios against `cute`). Triton's `HAS_QB=1` rows are
not input-identical (handoff §6a caveat); the `-exact` rows are.

| regime | triton | cuda | cute₀ | cute | tri/cute | cuda/cute |
|---|---|---|---|---|---|---|
| P=1024, HAS_QB=0 | 0.132 | 0.075 | 0.153 | 0.104 | 1.27× | 0.72× |
| P=1024, HAS_QB=1 | 0.128 | 0.091 | 0.216 | 0.129 | 0.99× | 0.71× |
| P=8192, HAS_QB=0 | 0.139 | 0.137 | 0.157 | 0.137 | 1.01× | 1.00× |
| P=8192, HAS_QB=1 | 0.133 | 0.152 | 0.227 | 0.152 | 0.87× | 1.00× |
| P=65536, HAS_QB=0 | 0.222 | 0.233 | 0.233 | 0.232 | 0.96× | 1.00× |
| P=65536, HAS_QB=1 | 0.209 | 0.229 | 0.297 | 0.227 | 0.92× | 1.01× |
| exact N=797085, B=1, C=4, A=4 | 0.130 | 0.123 | 0.225 | 0.131 | 0.99× | 0.93× |
| exact N=797085, B=16, C=4, A=4 | 0.157 | 0.161 | 0.193 | 0.160 | 0.98× | 1.01× |
| exact N=2988997, B=1, C=5, A=4 | 0.134 | 0.124 | 0.220 | 0.134 | 1.00× | 0.93× |
| exact N=2988997, B=16, C=5, A=4 | 0.168 | 0.173 | 0.225 | 0.172 | 0.98× | 1.01× |
| exact N=15000001, B=16, C=5, A=4 | 0.169 | 0.173 | 0.219 | 0.173 | 0.98× | 1.00× |

Aggregate winners (the tuner's plurality rule): triton `(64, 4)` / `(256, 8)`; cuda
`(128, 8, 4)` / `(256, 8, 4)` (its shipped `(128, 8, 1)` is 1.6 % off the per-regime best
in geomean — not re-pasted, the C++ backend is the fixed reference); cute `(512, 4, 1)`
/ `(256, 4, 1)`. The cute landscape is flat: no config won more than 2 of the 11
regimes of the two sweeps, so the two sweeps were reconciled by geometric mean of
ratio-to-best over all 11 → **`DEFAULT_CONFIG = (block_p=256, num_warps=8, unroll=4)`**
(1.056× the per-regime best, worst 1.144×; the cuda default `(128, 8, 1)` sits at 1.086×
/ 1.311×). A re-sweep after the host trims put every top config within 1.3 % of each
other (`(256, 8, 4)` 1.031×, best 1.018×, per-regime noise 8–40 %), and `(256, 8, 4)`
wins the bloom B=16 kernel outright (below), so it stays. `unroll=4` costs 48
registers/thread (34 at `unroll=1`, WP-1 NOTES) — no occupancy cliff.

**Kernel-only**, `D=128`, layout A (`P=58 368`), `torch.profiler` µs, final code. cute at
its `DEFAULT_CONFIG` and, for the like-for-like column, at cuda's `(128, 8, 1)`:

| regime | triton | cuda (128,8,1) | cute (128,8,1) | cute (256,8,4) |
|---|---|---|---|---|
| no filter, B=1 | 9.9 | 10.6 | 10.1 | 12.3 |
| no filter, B=16 | 85.6 | 89.9 | 90.0 | 91.2 |
| bloom (pass 0.002), B=1 — scorer + bloom mask | 9.9 | 6.0 + 6.2 | 6.1 + 5.9 | 5.9 + 5.9 |
| bloom (pass 0.002), B=16 — scorer + bloom mask | 116.8 | 50.5 + 6.7 | 51.2 + 6.3 | 39.8 + 6.3 |
| exact (pass 0.96), B=1 — scorer + clause mask | 10.3 | 10.2 + 5.7 | 10.0 + 5.7 | 11.3 + 5.7 |
| exact (pass 0.96), B=16 — scorer + clause mask | 129.1 | 88.8 + 54.8 | 101.6 + 54.7 | 87.7 + 54.7 |

(The bloom pass rate of the §6b inputs is 0.2 %, not the 0.19 §13 quotes for its own
profiler run — same inputs as the handoff snippet, recomputed on the full `[B, P]`
buffer; the exact rows are the §6c inputs, pass 0.96.)

**Shared-input head-to-head** incl. the host `topk` epilogue (`do_bench` ms, final code;
`P=1024` is the small layout `(64, 32, 32)`, `58 368` layout A, `46 720` layout B):

| mode | B | P | triton | cuda | cute | tri/cute | cuda/cute |
|---|---|---|---|---|---|---|---|
| none | 1 | 1024 | 0.131 | 0.066 | 0.098 | 1.34× | 0.68× |
| none | 1 | 58368 | 0.201 | 0.140 | 0.172 | 1.17× | 0.81× |
| none | 16 | 1024 | 0.126 | 0.082 | 0.098 | 1.29× | 0.84× |
| none | 16 | 58368 | 0.250 | 0.250 | 0.250 | 1.00× | 1.00× |
| bloom | 1 | 1024 | 0.121 | 0.079 | 0.135 | 0.90× | 0.58× |
| bloom | 1 | 58368 | 0.197 | 0.155 | 0.202 | 0.98× | 0.77× |
| bloom | 16 | 1024 | 0.122 | 0.089 | 0.125 | 0.98× | 0.72× |
| bloom | 16 | 58368 | 0.281 | 0.213 | 0.203 | 1.38× | 1.05× |
| bloom | 16 | 46720 | 0.256 | 0.201 | 0.209 | 1.23× | 0.96× |
| exact | 1 | 1024 | 0.135 | 0.078 | 0.129 | 1.05× | 0.61× |
| exact | 1 | 58368 | 0.214 | 0.156 | 0.209 | 1.02× | 0.75× |
| exact | 16 | 1024 | 0.138 | 0.087 | 0.131 | 1.05× | 0.66× |
| exact | 16 | 58368 | 0.299 | 0.306 | 0.305 | 0.98× | 1.00× |
| exact | 16 | 46720 | 0.270 | 0.278 | 0.277 | 0.98× | 1.00× |

Before the host trims the same table read cute 0.156 / 0.229 / 0.158 / 0.250 (none),
0.230 / 0.312 / 0.240 / 0.317 / 0.318 (bloom), 0.254 / 0.317 / 0.238 / 0.327 / 0.330
(exact) — 0.35–0.67× of cuda everywhere but the GPU-bound `none, B=16, P=58 368` row
(`h2h-base.txt`).

**Host overhead (P1)** — enqueue-only µs per call, GPU kept busy (reviewer's `host.py`
method), eager launchers at B=16 (scorer at `P=8192`, masks at layout A):

| launcher | cuda (C++) | cute as ported | cute after trims |
|---|---|---|---|
| phase 3 `cps_scores` / `_cps_cute_scores` | 4.0 | 63.2 | 16.2 |
| phase 2 `_bloom_partial_mask_*_impl` | 9.9–11.0 | 63.2 | 20.5 |
| phase 2 `_clause_partial_mask_*_impl` | 10.6–10.9 | 72.2 | 24.0 |
| the compiled callable alone, arguments prebuilt | — | 27.4 (`JitExecutor.__call__`) | 9.0 (`_Launch`) |

Trims kept (kernels, op API and results untouched; parity + `tests/compile` cute rows
green after each): (a) the stream handle comes from `torch._C._cuda_getCurrentRawStream`
(0.2 µs; `torch.cuda.current_stream().cuda_stream` was 4.7 µs); (b) the
`torch.cuda.device` guard is skipped when the tensors' device is already current (2.2 →
0.3 µs); (c) the big one — the DSL's per-call argument adaptation
(`generate_execution_args`: `typing.cast`, an owning ctypes cell per value, adapter
lookups — 27 µs for the scorer's 15 arguments, not the 13 the review estimated) is
bypassed by `_Launch` in `cute/codesigned_probe_score.py`: one set of ctypes cells per
thread, values written in place, `run_compiled_program` called directly, the packing
verified once against the DSL's own path on the compile-time example arguments with
a fallback to the ordinary call if a DSL release ever packs differently; the
launchers now take pointer addresses and the stream handle as plain ints, and the
`make_ptr` alignment `AssertionError` (review N2) became an explicit `ValueError`. Not
kept: folding the validation block into fewer Python statements (−1.5 µs, noise).
The floor is now ~9 µs for the launch itself (`_get_invoke_packed_args` + the compiled
host stub + `cuLaunchKernel`) plus ~7 µs of validation and the `torch.empty` of the
mask buffer — 3–4× the C++ launcher, down from 15×. In wall-clock terms at layout A
(`do_bench` minus the profiler's GPU-busy sum): the host-bound residue for cute went
from 102 / 209 / 153 / 209 / 62 µs (none B=1, bloom B=1, bloom B=16, exact B=1, exact
B=16) to 46 / 99 / 38 / 100 / 40 µs, against cuda's 15 / 50 / 38 / 49 / 40 — i.e.
equal to cuda at B=16 and ~2× cuda at B=1, where the remaining gap (≈30–50 µs) is the
regime being entirely host-bound, so every extra host microsecond lands 1:1 on the wall.

**P3 (auto-unroll of the no-mask `UNROLL=1` build).** Pinning the item loop with
`cutlass.range(p_first, warp_end, STEP, unroll=1)` produces the one-copy structure of
the C++ build (PTX: 4 `dp4a`, 1 `ld.global.cs.v4`, 1 `redux` vs 20 / 5 / 5 unpinned)
and is bit-exact (parity 82/82 with the pin). Kernel time, no filter, layout A,
`(128, 8, 1)`: unpinned 89.3 µs (B=16) / 10.5 µs (B=1), pinned 90.6 / 10.7; cuda 89.7 /
10.7. The `UNROLL=4` and every `HAS_MASK` build are identical either way, so
`DEFAULT_CONFIG` is unaffected. Verdict: leave the loop unpinned (marginally faster,
and it is the DSL's own output — D8). The shipped kernel is therefore the WP-1 build.

**Compile / launch facts.** Cold `cute.compile` per specialization in a warm process:
score `<8,true,1>` 173 ms, `<8,false,4>` 134, `<8,true,4>` 167, `<4,false,1>` 116,
`<16,false,1>` 118; bloom mask 120; clause mask `<2,2>` 143, `<0,0>` 153; generic 124
(no disk cache — every process pays them, P5). Fresh process: imports (torch, triton,
retrieve) 12–13 s, `ensure_built` 3.2–3.5 s of which the DSL import is ~3 s and the
probe compile 0.25 s; the probe now builds the default-unroll no-filter scorer, so a
plain query's first call pays only the CUDA-kernel first-load of the `topk` tail. First
`_impl` call in a fresh process (small layout): none 229 ms (before the probe change),
bloom 349 ms, exact 432 ms; second call 0.5–0.6 ms. Specializations per filter mode: none
1 (`score <SEG,false,U>`), bloom 2 (`bloom` + `score <SEG,true,U>`), exact 2
(`clause <C,A>` + `score <SEG,true,U>`); a warm cache lookup is 0.7 µs. SASS/register
facts are WP-1's (`NOTES.md`, kernel unchanged): `IDP.4A.S8.S8` ×4 per item, one
`LDG.E.EF.128` row load, `REDUX.SUM.S32`, exactly two `FMUL`, no `FFMA`, no
`BAR.SYNC`, no `LDL`/`STL`, 0 B spill; registers 34 / 48 (`UNROLL` 1 / 4; C++ 30 / 40)
and 32 for the auto-unrolled no-mask build.

**Reading the numbers.** Kernel for kernel, the port matches the C++ backend at the
same config: no filter 90.0 vs 89.9 µs (B=16) and 10.1 vs 10.6 (B=1), bloom mask 6.3 vs
6.7, clause mask 54.7 vs 54.8, bloom-filtered scorer 51.2 vs 50.5 — all within the
run-to-run noise of ±1 µs. The one kernel where it does not is the `HAS_MASK=true,
UNROLL=1` scorer at a high pass rate (exact, B=16: 101.6 vs 88.8 µs, +14 %); the bloom
column hides it because there almost every row is skipped. That build is the one whose
predicated row load became a divergent branch with a `REDUX.OR` uniformity test in
front of it (P4) and whose scalar `__ldg`s lost their read-only-cache hint (P2) — at
`UNROLL=4` the four in-flight rows cover the extra latency and the same kernel is 87.7
µs, which is why the tuned default is `(256, 8, 4)`; it also makes the bloom-filtered
scorer 20 % faster than cuda's shipped config (39.8 vs 50.5 µs), a config effect, not
a language effect (cuda at `unroll=4` tuned the same way in this session). The port's
real cost was never the kernels but the launch: as ported, a DSL launch cost 63–72 µs
of host time against 4–11 µs for the C++ launcher, which at the handoff regimes is
larger than the kernel-time differences being measured and made cute 0.35–0.67× of
cuda in wall-clock everywhere but the one GPU-bound row. With the trims the two
backends are wall-clock equal at B=16 for every mode and layout (none 1.00×, bloom
1.05× / 0.96×, exact 1.00×), and cute remains 0.6–0.8× of cuda at B=1 and at
`P=1024`, where the whole call is host-bound and the residual ~12 µs per launch (×2
with a filter) is the floor of launching through the DSL's compiled host stub. Against
Triton the port stands where the C++ backend stands: bloom 1.23–1.38× at the large
layouts (kernel-only 2.5×), no filter and exact within ±3 % at B=16, and faster at
small P where Triton's own launch overhead dominates — all of it bit-exact on every
regime above.

### 5.1 Compiled / CUDA-graph replay (WP-6)

2026-09-02, same box as §5. Does the eager host gap survive the deployed path? Full `SilverTorch.forward` (phase 1
+ op + `topk`), `D=128`, `k=64`, modules built with a synthetic balanced IVF (every
cluster exactly `max_size` wide, registered through the module's own steps, no k-means)
so `P` is exactly layout A → 58 368 (N = 3 035 136) and the small layout → 1 024;
attributes as in `bench_common.py` (bloom pass rate 0.0015 at layout A, exact 0.96).
`do_bench(rep=300)` median ms; **eager** `module(q, attrs)`, **compile**
`torch.compile(fullgraph=True)` default mode, **graph** `torch.compile(fullgraph=True,
mode="reduce-overhead")`. `torch.equal` on `(ids, scores)` across the three variants per
backend and on `scores` across backends asserted before every timing, 36/36 cells.
Scripts and raw outputs: session scratchpad `wp6/` (`graphs.py`, `graphs.json`,
`graphs.md`, `diag*.py|txt`).

| mode | B | P | eager tri | cuda | cute | compile tri | cuda | cute | graph tri | cuda | cute | cuda/cute eager | compile | graph |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1 | 1024 | 0.306 | 0.300 | 0.329 | 0.144 | 0.379 | 0.406 | 0.056 | 0.071 | 0.072 | 0.91× | 0.93× | 0.98× |
| none | 1 | 58368 | 0.528 | 0.445 | 0.487 | 0.336 | 0.522 | 0.561 | 0.118 | 0.130 | 0.133 | 0.91× | 0.93× | 0.98× |
| none | 16 | 1024 | 0.368 | 0.295 | 0.332 | 0.180 | 0.362 | 0.408 | 0.059 | 0.082 | 0.082 | 0.89× | 0.89× | 1.00× |
| none | 16 | 58368 | 0.537 | 0.452 | 0.486 | 0.343 | 0.531 | 0.568 | 0.237 | 0.256 | 0.255 | 0.93× | 0.93× | 1.00× |
| bloom | 1 | 1024 | 1.018 | 0.948 | 1.054 | 0.283 | 0.493 | 0.596 | 0.079 | 0.094 | 0.091 | 0.90× | 0.83× | 1.03× |
| bloom | 1 | 58368 | 1.220 | 1.167 | 1.250 | 0.439 | 0.654 | 0.714 | 0.132 | 0.173 | 0.142 | 0.93× | 0.92× | 1.22× |
| bloom | 16 | 1024 | 1.043 | 1.013 | 1.178 | 0.308 | 0.531 | 0.579 | 0.079 | 0.102 | 0.103 | 0.86× | 0.92× | 0.99× |
| bloom | 16 | 58368 | 1.286 | 1.232 | 1.290 | 0.473 | 0.714 | 0.742 | 0.278 | 0.232 | 0.222 | 0.95× | 0.96× | 1.05× |
| exact | 1 | 1024 | 0.428 | 0.346 | 0.405 | 0.223 | 0.420 | 0.481 | 0.059 | 0.084 | 0.087 | 0.85× | 0.87× | 0.96× |
| exact | 1 | 58368 | 0.594 | 0.518 | 0.614 | 0.365 | 0.623 | 0.651 | 0.126 | 0.142 | 0.143 | 0.84× | 0.96× | 0.99× |
| exact | 16 | 1024 | 0.422 | 0.351 | 0.412 | 0.219 | 0.421 | 0.486 | 0.066 | 0.109 | 0.098 | 0.85× | 0.86× | 1.11× |
| exact | 16 | 58368 | 0.609 | 0.535 | 0.582 | 0.395 | 0.615 | 0.661 | 0.287 | 0.315 | 0.316 | 0.92× | 0.93× | 1.00× |

Capture: every cell `cudagraph_skips == 0`, no "skipping cudagraphs" hint, one
`cudaGraphLaunch` per forward for all three backends (the only other host-side CUDA
calls are cudagraph trees' input copies); the cute launch lands on the capture stream via
`torch._C._cuda_getCurrentRawStream`. A manual `torch.cuda.CUDAGraph` of the eager
forward (pure replay) gives cuda/cute 0.97–1.00× for none and exact and fails to capture
bloom on every backend — `build_query_signatures` builds its salt constants with
`torch.tensor(_SALT, device=cuda)`, a pageable H2D copy that invalidates raw capture,
which inductor folds into the graph. Reading: the gap is an eager-path cost (the forward
is host-bound, so the DSL launch lands 1:1, on top of ~0.2 ms of layer + `custom_op`
dispatch — hence milder than the `_impl`-level 0.6–0.8× above); default-mode compile does
not remove it (the cuda/cute ops are opaque `custom_op`s whose body runs eagerly inside
the wrapper, which adds its own fixed dispatch; the Triton ops are see-through
`triton_op`s, hence its default-compile lead); under cudagraph replay it is gone — cuda
and cute equal within noise, the residual per-backend differences being kernel counts in
the graph. The harness compiles every algo with `dynamic=True, mode="reduce-overhead"`
(`AlgoBase._finalize`), so the deployed path never pays the DSL launch cost. Follow-ups
and levers: [kernels.md § Follow-ups](../system/kernels.md#follow-ups).
