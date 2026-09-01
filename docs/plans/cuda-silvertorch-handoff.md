# CUDA SilverTorch backend — validation + benchmark handoff

> **Status:** ready for execution (written 2026-07-06, extended 2026-09-01 with the
> phase-2 work — exact-on-cuda, the `UNROLL` knob, and the two static reviews of
> [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md). Both passes were authored
> on a CUDA-less dev machine — nothing below has run on a GPU yet). This is the runbook a GPU session
> executes **top-to-bottom** to build, validate, and benchmark the new
> `backend="cuda"` SilverTorch implementation against the Triton backend. Target
> hardware: **A100/sm_80** (the paper's own eval GPU and the arch the repo's shipped
> `DEFAULT_CONFIG`s are tuned for); correctness steps run on any CUDA 12.x-capable
> NVIDIA GPU, perf gates are only meaningful on A100.
>
> Do not rebase/merge mid-validation. Every fallback in §8 is a code change — re-run
> the affected gates after applying one.

## 1. Context — what landed

A CUDA C++ implementation of SilverTorch's co-designed ANN + bloom filtering
(`articles/silvertorch.md`, Algorithm 1 "partial_bloom"), wired in as a third
`SilverTorch` backend and benchmarkable against the existing Triton kernel at both
the kernel level (`tune-kernels`) and end-to-end (`evaluation/`).

**Design intent: match the paper, not the Triton kernel.** The Triton
`codesigned_probe_score` fuses a *row-wise* signature read into the scoring pass —
each probed item costs `8·W` bytes of bloom signature (at `m_bits=1024`, W=16 →
128 B/item, as much as the D=128 code row itself). The paper's implementation is
structurally different and cheaper (§3), and this port follows the paper:

| Paper claim (`articles/silvertorch.md`) | Where implemented |
|---|---|
| "Rotate the matrix" — transposed bloom index, iterate only the set bits of QB, one `and.b64` tests 64 items (Fig. 3(b), §Bloom Index) | `cps_bloom_mask_kernel` + `build_transposed_sigs` |
| Partial bloom: filter only items in probed clusters, per-cluster compact masks `M_c` (Alg. 1 Phase 2) | mask kernel runs over `probe_ids` spans only |
| ANN kernel accepts a **1-bit-per-item** mask, not 8-bit bools (§Co-design) | `cps_score_kernel` reads one mask bit per item |
| Fused index-matmul: stream item embeddings straight from the table by id, no intermediate gather tensor (§Fused Int8 ANN) | scoring kernel gathers `item_codes[id]` rows directly |
| Int8 with **dp4a** — 4 MACs/instruction (§Fused Int8 ANN) | `__dp4a` chain per lane |
| One warp per contiguous tile of items, coalesced access (§Fused Int8 ANN) | warp-owned contiguous item ranges, SEG-lane segments |
| Global min/max int8 quantization at publish (§Fused Int8 ANN) | reuses the shared `quantize_int8_global` / `quantize_int8` |
| Phase 1 (centroid top-`n_probe`) & Phase 4 (global top-k) host-side (Alg. 1) | unchanged: layer's `_phase1_probe_with_ids` + `torch.topk` |

### Files landed

| File | What |
|---|---|
| `retrieve/src/retrieve/kernels/silvertorch/cuda/codesigned_probe_score.cu` | NEW — all kernels + host launchers + pybind module. Phase 2 later gained `cps_clause_mask_kernel` + `clause_partial_mask` (WP-B) |
| `retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score_cuda.py` | NEW — lazy JIT loader, `CodesignedProbeScoreCudaConfig`, `build_transposed_sigs`, eager `_impl`s, `retrieve::codesigned_probe_score_cuda` + `retrieve::codesigned_probe_score_bloom_cuda` custom ops + fakes. WP-B added `_clause_partial_mask_cuda_impl`, `_cpse_cuda_prep`, `_cps_cuda_score_topk` (shared phase-3+4 tail) and `retrieve::codesigned_probe_score_exact_cuda` |
| `retrieve/src/retrieve/interfaces.py` | `Backend` literal += `"cuda"` |
| `retrieve/src/retrieve/layers/silvertorch/main.py` | `_forward_cuda`, `_phase1_probe_with_ids`, `bloom_sigs_t` registration for cuda+bloom (WP-D: `RuntimeWarning` above 2× IVF padding slack). The cuda+exact construction `ValueError` was removed by WP-B; `_forward_cuda` routes exact to the new op with `max_size=self._max_cluster_size`, a Python int cached in `_build_ivf` (WP-D — reading it off a buffer shape risked a SymInt under `dynamic=True`) |
| `retrieve/src/retrieve/kernels/silvertorch/__init__.py` | export the three cuda ops |
| `retrieve/src/retrieve/tune.py` | `codesigned-probe-score-cuda` spec (`_cps_cuda_inputs`, `_CPS_CUDA_GRID`); WP-B added `codesigned-probe-score-exact-cuda` (`_cpse_cuda_inputs`) over the shared `_cps_cuda_probe_family` layout builder |
| `retrieve/tests/parity/test_codesigned_probe_score_cuda.py` | NEW — ref parity, **bit-exact vs Triton**, transposed-index roundtrip, mask-kernel-vs-rowwise, opcheck; WP-B added the exact twins of all four (`test_cuda_exact_matches_ref`, `test_cuda_exact_matches_triton_bitexact`, `test_clause_mask_matches_rowwise`); WP-C added `test_config_override_matches_default`, `test_invalid_config_rejected`, `test_unrolled_mask_indexing_tiny_clusters`; WP-D parametrized `test_opcheck` over `d ∈ {64,128,256,96}` |
| `retrieve/tests/parity/conftest.py` | `ref_cps_phase23` moved here (shared with the Triton parity file); WP-B gave it the exact-predicate keywords via `clause_subset_match`, replacing the Triton exact file's local copy; WP-D added `assert_ids_equal_up_to_ties` |
| `retrieve/tests/conftest.py` | `require_cps_cuda()` — WP-D made it *skip* only on a missing toolchain and **fail** on a build error |
| `retrieve/tests/compile/test_export_kernel_ref.py` | WP-D parametrized over `backend ∈ {triton, cuda}` — export must keep the cuda exact op as one opaque node |
| `retrieve/tests/correctness/test_silvertorch.py` | `BACKENDS += "cuda"`, cuda-vs-triton cross-backend (bloom, none, and after WP-B exact + exact-reverse). WP-B dropped the cuda×exact skip and the construction-reject test |
| `retrieve/tests/compile/test_silvertorch_compile.py` | parametrized over `(filter_mode, backend, D)` — all three modes on cuda after WP-B, at both `D=64` and `D=128` after WP-D |
| `retrieve/tests/correctness/test_tune_smoke.py` | registry tuple + cuda skip |
| `evaluation/retrieval/cli/evaluate.py`, `evaluation/retrieval/sweep.py` | `--backend cuda` choice; cuda cells get triton filter modules / oracle |
| `evaluation/config/deep_sweeps/arxiv-d128-silvertorch.yaml` | `backends: [triton, cuda, torch]` |
| `retrieve/pyproject.toml`, `.gitignore` | `ninja` dev dep, wheel `artifacts` guard, `.torch_extensions/` ignore |

### Scope cuts (deliberate, see §7)

- ~~`filter_mode="exact"` has no CUDA kernel.~~ **Lifted.** Work package B of
  [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md) added
  `cps_clause_mask_kernel` (a second phase-2 kernel writing the same 1-bit mask) and
  `retrieve::codesigned_probe_score_exact_cuda`; the construction `ValueError` is
  gone and clause sweeps on cuda cells are legal. Also written on a CUDA-less box —
  §5 gates it. The shipped deep-sweep config stays bloom-only until they pass.
- The paper's stack-machine evaluation of arbitrary AND/OR/NOT operation arrays is
  out of scope; the repo's query model (conjunctive `(clause, value)` bloom keying
  via `build_query_signatures`) is what the mask kernel implements.
- ~~`torch.export` is not a gate for the cuda backend.~~ **Lifted (WP-D).**
  `tests/compile/test_export_kernel_ref.py` is parametrized over
  `backend ∈ {triton, cuda}`: the cuda leg asserts export keeps
  `retrieve::codesigned_probe_score_exact_cuda` as one opaque node and that the
  exported module reproduces eager bit for bit. There is no decomposed fallback
  for a C++ custom op, so that node is the only thing the assertion can accept.

## 2. Environment setup (GPU box)

```bash
cd <repo-root>              # the uv workspace root (contains uv.lock)
git checkout <this-branch>
uv sync                     # installs ninja from retrieve's dev group

nvcc --version              # MUST print CUDA 12.x — same major as torch's +cu128 wheel
```

torch cu12x wheels ship the CUDA *runtime* libraries but **not nvcc** — a system
CUDA toolkit is required for the JIT build. If `nvcc` is not on PATH, point
`CUDA_HOME` at the toolkit root (cpp_extension resolves `CUDA_HOME` → `which nvcc`
→ `/usr/local/cuda`, in that order). Last-resort fallback if the box has no system
toolkit and you cannot install one: `uv pip install nvidia-cuda-nvcc-cu12` puts nvcc
under `site-packages/nvidia/cuda_nvcc/bin/`, but the headers live in *other* wheels'
directories so no coherent `CUDA_HOME` exists — you would have to symlink-farm a fake
toolkit root. Known-awkward; prefer the system toolkit
([NVIDIA forum thread](https://forums.developer.nvidia.com/t/nvidia-cuda-nvcc-pip-wheel-installation/221307)).

```bash
export TORCH_CUDA_ARCH_LIST="8.0+PTX"              # reproducible codegen + PTX fallback
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"  # project-local build cache (gitignored)
```

Optional but recommended before recording perf numbers (thermal drift mostly cancels
in same-session ratios, but pinned clocks make the absolute GB/s meaningful):

```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410      # A100 max SM clock; reset with: sudo nvidia-smi -rgc
```

Data: steps 3–6 need no datasets. Step 7 (end-to-end) needs
`evaluation/data/arxiv-papers/` with `content_d128`/`gt_d128` populated (see
`evaluation/` docs / `eval-fetch`).

## 3. Design & performance model (what to expect and why)

### 3.1 The three kernels

**Phase 2 — `cps_bloom_mask_kernel`.** The bloom index is stored transposed and
cluster-major: `bloom_sigs_t[m]` is a bit-vector over padded IVF slots, each cluster
a contiguous, word-aligned span of `wpc = ceil(max_cluster_size/64)` int64 words
(`build_transposed_sigs`). One thread produces one output word = the verdict for 64
items: it iterates **only the set bits of QB** (`__ffsll` loop) and ANDs the matching
rows — one `and.b64` per 64 items per set query bit, exactly the paper's Fig. 3(b).
A query signature with no set bits yields `~0` (AND identity — everything passes),
matching the row-wise `(qb & ~sig) == 0` semantics bit-for-bit. Filter read traffic
is `popcount(QB)/8` bytes per item (a realistic QB has `k_hash × active clauses` ≈
10–25 set bits → ~1–3 B/item) versus the Triton kernel's `8·W` (64–128 B/item at
production `m_bits`) — a **~40–100× reduction in filter traffic**, which is the
paper's headline bloom-index result (their Table: 12.6–42.7× over the forward
index).

**Phase 2 (exact) — `cps_clause_mask_kernel`.** The second phase-2 filter, added by
work package B of [cuda-silvertorch-phase2.md](cuda-silvertorch-phase2.md). It writes
the *same* `[B, n_probe·wpc]` 1-bit-per-item layout, so phase 3 is reused untouched —
the scorer never learns which filter produced the bit. One **warp** owns one output
word: lane `l` evaluates slot `l` of each 32-slot half and two `__ballot_sync` calls
pack the halves (`word = lo | (hi << 32)`, stored by lane 0), so the early return for
`word_idx >= mask_words` has to be — and is — warp-uniform. Ids come from `flat_items`,
the very tensor phase 3 reads, so exact mode registers **no new buffer** (a cuda+exact
`state_dict` is identical to a triton+exact one) and the 32 lanes of a half read 32
consecutive int64 ids in one coalesced 256-byte run. The predicate is bit-identical to
`common.clause_pass`; `C` and `A_max` are runtime arguments rather than template
parameters (one instantiation per `(C, A_max)` pair would be absurd, and one
`[C, A_max]` int64 row is ~one 32-byte sector at the shipped shapes — same order as the
id read). Expect it to cost roughly what the bloom mask kernel costs: a second launch,
noise next to phase 3 at large `P`, possibly visible at `B=1` (§8).

**Phase 3 — `cps_score_kernel`.** Grid `(cdiv(P, block_p), B)` — identical launch
shape to the Triton kernel. A sub-warp *segment* of `SEG = min(32, D/4)` lanes
cooperates on one item; lane `j` owns packed code words `j, j+SEG, …`, so the
matching **query words live in registers for the whole block** — the kernel uses no
shared memory at all. A code-row read is `SEG` consecutive 4-byte words: fully
coalesced; at D=128 one row is exactly one 128-byte cache line
([CUDA Best Practices Guide §9.2.1: coalesced access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)).
Per item: one broadcast id load, one mask-bit test (1 bit/item; the word covers 64
consecutive slots so it stays L1-resident), then — only for passing items — a
`__dp4a` chain (`WPL = D/4/SEG` instructions/lane; 4 MACs each,
[CUDA Math API integer intrinsics](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html))
and one `__reduce_add_sync` segment reduction (hardware REDUX, sm_80+,
[CUDA C++ Programming Guide, warp reduce functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-reduce-functions)).
The keep verdict is segment-uniform, so a filtered item **genuinely skips** its row
load and the whole dot — not per-lane load predication but a uniform branch.
Specializations: D=64 → 16-lane segments (2 items in flight/warp), D=128 → 32×1,
D=256 → 32×2, any other D%4==0 → a generic runtime-D kernel. The mask test carries
no division: each segment tracks its base item's `(cluster, slot)` incrementally
(one divide before the loop, then a ≤8-subtraction carry loop).

**Phase 3, the `UNROLL` knob** (`CodesignedProbeScoreCudaConfig.unroll ∈ {1,2,4}`,
default 1 = the pre-existing loop). A segment handles `UNROLL` items per iteration
in three phases — ids+verdicts, then *all* `UNROLL` row gathers, then the dots — so
independent row loads overlap instead of serializing one row per iteration. The case
for it is Little's law: A100 needs ≈1555 GB/s × ~600 ns ≈ 0.9 MB in flight ≈ 8.6 KB/SM
≈ 67 rows of 128 B, while ~50 resident warps × 1 row (D=128) ≈ 6.4 KB is borderline
short. Two costs to watch on the GPU box: registers (est. ~55/thread at
`UNROLL=4, WPL=2` vs ~35–45 at 1 — past ~64 occupancy falls under 50%, so read
`ptxas -v`/ncu before trusting a swept winner), and JIT build time (18 scoring-kernel
instantiations now). The generic runtime-D fallback ignores the knob. Arithmetic is
identical at every `UNROLL`, so §3.2's bit-exactness argument is config-independent —
the parity suite checks that directly.

Deliberate non-choices, for the record:

- **No IMMA/tensor cores.** The dot is M=1 (one query row per segment); int8 `mma`
  tiles have M=16 granularity, so 15/16 of every MMA's M-rows would compute zeros,
  plus the operands would have to stage through shared memory + `ldmatrix`. That is
  in fact what Triton emits for this kernel: current Triton's NVIDIA backend sets
  `min_dot_size = (1, 1, 32)` for int8 with the comment *"for small M/N … use
  tensorcores with padding"*
  ([triton nvidia backend compiler.py](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)),
  i.e. the repo comment in the Triton kernel claiming a dp4a lowering is likely
  wrong — step 4 verifies this empirically. Either way the kernel is bandwidth-bound
  (below), so dp4a-vs-IMMA compute throughput is irrelevant; dp4a is what the paper
  names, and it removes the staging/barrier machinery entirely.
- **No cp.async.** Ampere's async copy targets shared memory
  ([Ampere tuning guide §1.4.1.2](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html))
  and this design has none; the data-dependent gather (row address depends on the
  just-loaded id) defeats a long prefetch pipeline anyway.
- **No shared-memory tile GEMM** (mirroring Triton's structure): it would reintroduce
  staging, barriers, the padded-M dot, and unconditional scoring of filtered lanes —
  everything the paper's co-design removes.

### 3.2 Numerics — why CUDA must be bit-identical to Triton

1. The int8×int8→int32 dot is exact: `|dot| ≤ 256·128·128 = 2²² ≪ 2³¹`, and int32
   addition is associative in the no-overflow regime — any accumulation order
   (sequential dp4a + REDUX tree vs Triton's MMA tree) gives the same int32.
2. int32→fp32 cast: `2²² < 2²⁴` → exactly representable.
3. Dequant is the same left-associated pair of fp32 multiplies
   (`(float)acc * q_scale * global_scale`); FMA contraction cannot apply to a pure
   product chain, and the build does **not** pass `--use_fast_math`.
4. The transposed-AND predicate is boolean-identical to the row-wise subset test,
   and the query quantization (`quantize_int8`) is shared torch code.
5. The clause predicate is likewise boolean-identical to `common.clause_pass`, so
   the exact path inherits 1–3 unchanged.
6. Both backends run the identical host `torch.topk` + `gather` epilogue on
   bit-identical `[B, P]` score tensors.

Hence, in `tests/parity/test_codesigned_probe_score_cuda.py`
(`test_cuda_matches_triton_bitexact` and `test_cuda_exact_matches_triton_bitexact`):
**scores bit-identical** (`torch.equal`, a strictly stronger gate than the repo's
tolerance-based comparator) and **ids identical up to permutation within tied
scores** (`assert_ids_equal_up_to_ties`). The id gate is one notch weaker on
purpose: `torch.topk` documents its tie order as "not guaranteed stable across
invocations", and ties are not hypothetical here — int32 dots of ~±1e5 magnitude
over P≈768 candidates land on about one tied pair per row. A permutation *inside* a
run of equal scores is a top-K property; anything outside one still fails.

Corollary for step 7: eval quality columns (`recall@k` etc.) must be **byte-identical**
between cuda and triton rows. (Set-valued metrics are tie-insensitive, so the weaker
id gate does not weaken this one.)

### 3.3 Roofline / traffic model (the acceptance yardstick)

Per probed item, approximate DRAM traffic at bloom pass-rate `s`:

```
Triton (row-wise fused):  8 (id) + 8·W (sigs) + s·D (code row) + 4 (score out)
CUDA  (paper co-design):  8 (id) + popcount(QB)/8 + ~0.25 (mask wr+rd) + s·D + 4
```

Worked example — arxiv-like regime (D=128, W=16 / m_bits=1024, popcount(QB)=25,
s=0.2): Triton ≈ 165 B/item, CUDA ≈ 41 B/item → **~4× less traffic**, so expect the
bloom-mode cuda path to land roughly 2–4× faster at large P where both saturate
DRAM. With the filter off the models coincide (`8 + D + 4`), so expect parity to a
modest cuda win (no smem staging / padded-MMA overhead, cheaper tails). At small P
(≤ a few thousand) both are launch-latency-bound and the second kernel launch on the
cuda path may even cost it a few µs — that is acceptable and worth recording.

Compute intensity ≈ `D` MACs vs ≥ `s·D` bytes → ~1 MAC/B, while A100's dp4a-vs-DRAM
roofline knee sits near ~25 MAC/B (~10¹³ dp4a MAC/s class vs 1 555 GB/s HBM2e,
[A100 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf),
[Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)) —
the kernel is **HBM-bound everywhere**; the metric that matters is achieved GB/s:

```
achieved GB/s = Σ bytes (model above × B × P) / (median_ms × 1e6)
```

Occupancy: the scoring kernel uses no shared memory and an estimated ~35–45
registers/thread → ~50 resident warps/SM of the 64 max (~78%) — ample for a
bandwidth-bound kernel (A100: 64 K regs/SM, 64 warps/SM,
[Ampere tuning guide, occupancy](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)).
Verify with ncu in step 8, don't trust the estimate.

### 3.4 Cost accounting away from the hot path

- `build_transposed_sigs` runs once at `register_index`: `m_bits × 64`-iteration
  loop of small GPU ops (seconds at N=3M, m_bits=1024). Index memory for cuda+bloom:

  ```
  bytes(bloom_sigs_t) = W · 8 · n_lists · wpc · 64
                      = bytes(bloom_sigs) × (n_lists · wpc · 64 / N)
                      ≈ bytes(bloom_sigs) × (n_lists · max_cluster_size / N)
  ```

  The factor is the IVF **padding slack**: every cluster is padded out to the largest
  one, so it is 1.0–1.1× on a balanced index and grows without bound as clusters get
  uneven. The row-wise `bloom_sigs` buffer is **not** registered on the cuda backend,
  so at slack ≈ 1 total filter memory is roughly unchanged vs triton. Above 2× the
  layer emits a `RuntimeWarning` pointing at re-clustering.

  **Record `n_lists · max_cluster_size / N` per dataset** before step 7 — it is the
  predicted `index_mem_mib` delta, and §7's memory gate is "explained by this number",
  not "small":

  ```bash
  cd retrieve && uv run python -c "
  import torch
  from retrieve.layers.silvertorch import build_silvertorch
  # substitute the real item_embs / n_lists for the dataset under test
  m = build_silvertorch(torch.randn(100_000, 128, device='cuda'), k=10,
                        n_lists=1024, n_probe=32, backend='cuda')
  n_lists, max_size = m.padded_cluster_items.shape
  print('slack =', n_lists * max_size / m.item_codes.shape[0])"
  ```
- First forward pays the one-time nvcc JIT build (~1 min) — always absorbed in
  warmup, never in a timed window (`measure_forward_cuda` warms 20 iters; the
  runbook front-loads `ensure_built()` anyway).
- Custom ops are opaque to dynamo and cudagraph-safe by default
  ([torch.library docs](https://docs.pytorch.org/docs/stable/library.html),
  [CUDAGraph Trees — custom ops assumed safe](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)):
  kernels launch on `at::cuda::getCurrentCUDAStream()`
  ([custom C++/CUDA ops tutorial](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)),
  all allocation is `torch.empty` (caching allocator), and `global_scale`/`k` stay
  Python scalars (the layer's cached `_global_scale_f` — no D2H sync during capture).

## 4. Step 1 — build sanity (run this FIRST)

```bash
cd retrieve
uv run python -c "from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import ensure_built; ensure_built(verbose=True)"
```

Expected: ninja compile lines ending in a successful module load (~30–90 s). Run it
**again** — must be a near-instant cache no-op. Failure → §8 F1.

Optional but recommended — confirm the instruction-mix claims from §3.1:

```bash
# CUDA side: dp4a present, no smem staging in the scoring kernel
SO=$(ls "$TORCH_EXTENSIONS_DIR"/retrieve_cps_cuda/*.so)
cuobjdump -sass "$SO" | grep -c "IDP4A"        # expect > 0
cuobjdump -sass "$SO" | grep -c "LDSM"         # expect 0

# Triton side: the M=1 int8 tl.dot lowers to padded IMMA (mma.sync), not dp4a
export TRITON_CACHE_DIR=/tmp/triton_cache && rm -rf $TRITON_CACHE_DIR
uv run pytest tests/parity/test_codesigned_probe_score.py -q
grep -rl "mma.sync"  $TRITON_CACHE_DIR --include='*.ptx'   # expect hits
grep -rl "dp4a"      $TRITON_CACHE_DIR --include='*.ptx'   # expect none
```

Record the outcome in §9 — if Triton *does* emit dp4a, the §3.3 expectations for
`HAS_QB=0` tighten toward parity (the traffic model is unchanged either way).

## 5. Step 2 — correctness gates

```bash
cd retrieve
# 2a. CUDA parity suite: torch-ref parity, BIT-EXACT vs Triton, transposed-index
#     roundtrip, mask-kernel-vs-rowwise, opcheck, config plumbing
uv run pytest tests/parity/test_codesigned_probe_score_cuda.py -v          # → F2/F4

# 2b. Full library suite (triton/torch backends must be untouched; cuda joins
#     correctness + cross-backend + compile + export params)
uv run pytest tests/ -x -q                                                  # → F4

# 2c. Compile gate explicitly (zero graph breaks + reduce-overhead parity on cuda,
#     at both D=64 and D=128) and the export gate
uv run pytest tests/compile/ -v                                             # → F3
```

Every test in 2a is a gate. What each one is actually protecting:

| test | protects |
|---|---|
| `test_cuda_no_filters_matches_ref` / `..._with_bloom_...` / `..._exact_...` | phases 2+3 vs the torch reference, incl. `d=96` on the generic runtime-`D` fallback |
| `test_cuda_matches_triton_bitexact` | scores `torch.equal` to Triton, ids up to tie permutation, at `D ∈ {64,128,256}` × bloom on/off |
| `test_cuda_exact_matches_triton_bitexact` | the same for the clause path, `reverse ∈ {none, mixed}` |
| `test_build_transposed_sigs_bits` | the host-side bit layout the bloom kernel decodes (pad tail, `-1` slots) |
| `test_bloom_mask_matches_rowwise` | phase 2 alone vs `(qb & ~sig) == 0`, splitting mask bugs from scoring bugs |
| `test_clause_mask_matches_rowwise` | phase 2 (exact) alone vs `clause_subset_match`, **plus** the pad tail of a span's last word being 0 |
| `test_config_override_matches_default` | `block_p`/`num_warps`/`unroll` reach the launch and change nothing observable (`torch.equal` on all three paths) |
| `test_invalid_config_rejected` | the launcher's `TORCH_CHECK`s: `block_p=12, num_warps=8` → "block_p must be a positive multiple"; `unroll=3` → "unroll must be 1, 2 or 4" |
| `test_unrolled_mask_indexing_tiny_clusters` | the division-free `(cluster, slot)` carry at `max_size=3 < SPW·UNROLL`, where one iteration crosses several cluster spans |
| `test_opcheck` | the custom-op contract for all three ops at `d ∈ {64,128,256,96}` |

Hard gates: every test green. A cuda test that **skips** now means only "no CUDA
device / no `nvcc`" — step 1 already rules that out, so a skip here is itself a
finding. A toolchain that exists and fails to build is a `pytest.fail` carrying the
nvcc/ninja output, never a skip. `test_cuda_matches_triton_bitexact` failing is a
correctness bug — see F4 before touching tolerances.

## 6. Step 3 — kernel-level benchmark

### 6a. Tune both kernels (best-vs-best)

```bash
cd retrieve
uv run tune-kernels codesigned-probe-score       --json-out /tmp/cps-triton.json
uv run tune-kernels codesigned-probe-score-cuda  --json-out /tmp/cps-cuda.json
```

Both sweep the same regime keys `P ∈ {1024, 8192, 65536} × HAS_QB ∈ {0,1}` (defaults
`--d 128 --b 16 --w 4`) under the same `triton.testing.do_bench` harness (CUDA-event
timing, L2 flushed between iterations,
[do_bench docs](https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html)),
each over its own config grid. Paste the printed `DEFAULT_CONFIG = ...` line into
`codesigned_probe_score_cuda.py` if it differs from the shipped one, and re-run 2a.

The CUDA grid is 3-dimensional — `block_p ∈ {128, 256, 512, 1024} × num_warps ∈ {4, 8}
× unroll ∈ {1, 2, 4}`, 24 points per regime, so this sweep takes ~3× the wall time it
used to and the paste line now carries three fields
(`CodesignedProbeScoreCudaConfig(block_p=…, num_warps=…, unroll=…)`) — paste it whole.
`unroll > 1` has never run anywhere: if it wins, check `ptxas -v` registers/thread
(§8) before adopting it, and confirm the win survives at `D=256` where the register
cost is highest. `unroll=1` reproduces the pre-WP-C kernel exactly, so pasting it is
always safe.

Compare best-vs-best:

```bash
uv run python - <<'EOF'
import json
tri = json.load(open("/tmp/cps-triton.json"))["per_regime"]
cud = json.load(open("/tmp/cps-cuda.json"))["per_regime"]
print(f"{'regime':<38} {'triton':>9} {'cuda':>9} {'speedup':>8}")
for key in tri:
    t, c = tri[key]["winner_ms"], cud[key]["winner_ms"]
    print(f"{key:<38} {t:>9.3f} {c:>9.3f} {t/c:>8.2f}x")
EOF
```

**Caveat:** the `HAS_QB=1` rows are *not* input-identical across the two JSONs — the
Triton spec feeds dense random row-wise signatures (pass rate ≈ 0, a filter-traffic
stress test), the CUDA spec feeds a sparse realistic QB (~25 set bits, pass ≈ 0.2)
against its transposed index. Use them for config selection and the `HAS_QB=0` rows
for direct comparison; the honest bloom head-to-head is 6b.

### 6b. Shared-input head-to-head (the number to report for bloom mode)

Same probe family, same signatures (row-wise for Triton, `build_transposed_sigs` of
those same rows for CUDA — semantically identical inputs), realistic signature
sparsity from the real hash path, bit-exactness asserted before timing:

```bash
cd retrieve && uv run python - <<'EOF'
import torch, triton.testing as ttesting
from retrieve.kernels.silvertorch.codesigned_probe_score import _codesigned_probe_score_impl
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    _codesigned_probe_score_cuda_impl, build_transposed_sigs)
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds

dev = torch.device("cuda:0"); torch.manual_seed(0)
B, D, K, M_BITS, K_HASH = 16, 128, 64, 1024, 5
for (N_LISTS, MAX_SIZE, N_PROBE) in [(1664, 1824, 32), (8192, 365, 128)]:  # arxiv layouts A/B
    N = N_LISTS * MAX_SIZE
    padded = torch.randperm(N, device=dev).reshape(N_LISTS, MAX_SIZE)
    probe_ids = torch.randint(0, N_LISTS, (B, N_PROBE), device=dev)
    flat = padded[probe_ids].reshape(B, -1)
    codes = torch.randint(-128, 128, (N, D), dtype=torch.int8, device=dev)
    query = torch.randn(B, D, device=dev)
    attrs  = torch.randint(0, 50, (N, 2, 2), device=dev)   # realistic sig sparsity
    q_attrs = torch.randint(0, 50, (B, 2), device=dev)
    seeds = generate_seeds(K_HASH, device=dev); W = M_BITS // 64
    sigs = build_signatures(attrs, seeds, M_BITS, K_HASH, W)
    qb = build_signatures(q_attrs.unsqueeze(-1), seeds, M_BITS, K_HASH, W)
    sigs_t = build_transposed_sigs(sigs, padded)

    tri = lambda: _codesigned_probe_score_impl(query, flat, codes, 0.01, K,
                                               query_bits=qb, bloom_sigs=sigs)
    cud = lambda: _codesigned_probe_score_cuda_impl(query, flat, codes, 0.01, K,
                     query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
    (ti, ts), (ci, cs) = tri(), cud()
    assert torch.equal(ts, cs) and torch.equal(ti, ci), "bit-exactness violated"
    print(f"P={N_PROBE*MAX_SIZE}:")
    for name, fn in (("triton", tri), ("cuda", cud)):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        ms, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
        print(f"  {name:>6}: {ms:.3f} ms")
EOF
```

Compute achieved GB/s from the §3.3 model (measure `s` as
`torch.isfinite(scores).float().mean()` on a scored `[B, P]` buffer if you want the
exact byte count) and record both.

### 6c. Shared-input head-to-head, exact mode (the number to report for exact)

Unlike bloom, both backends read *the same* `item_clause_attrs` / `clause_is_reverse`
buffers here — no index rotation, no distribution difference — so this is a clean
head-to-head. The only extra argument is `max_size`, the padded cluster width the
clause-mask kernel needs to lay its output out cluster-major.

```bash
cd retrieve && uv run python - <<'EOF'
import torch, triton.testing as ttesting
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    _codesigned_probe_score_exact_impl)
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import (
    _codesigned_probe_score_exact_cuda_impl)

dev = torch.device("cuda:0"); torch.manual_seed(0)
B, D, K, C, A_MAX, N_VOCAB = 16, 128, 64, 2, 2, 50
for (N_LISTS, MAX_SIZE, N_PROBE) in [(1664, 1824, 32), (8192, 365, 128)]:  # arxiv layouts A/B
    N = N_LISTS * MAX_SIZE
    padded = torch.randperm(N, device=dev).reshape(N_LISTS, MAX_SIZE)
    probe_ids = torch.randint(0, N_LISTS, (B, N_PROBE), device=dev)
    flat = padded[probe_ids].reshape(B, -1)
    codes = torch.randint(-128, 128, (N, D), dtype=torch.int8, device=dev)
    query = torch.randn(B, D, device=dev)
    attrs = torch.randint(0, N_VOCAB, (N, C, A_MAX), device=dev)
    rev = torch.zeros(C, dtype=torch.bool, device=dev); rev[0] = True   # both XOR arms
    q_attrs = torch.randint(0, N_VOCAB, (B, C), device=dev)
    q_attrs[:, -1] = -1                                                 # an inactive clause

    kw = dict(item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs)
    tri = lambda: _codesigned_probe_score_exact_impl(query, flat, codes, 0.01, K, **kw)
    cud = lambda: _codesigned_probe_score_exact_cuda_impl(query, flat, codes, 0.01, K,
                     max_size=MAX_SIZE, **kw)
    (_ti, ts), (_ci, cs) = tri(), cud()
    # Scores are the hard gate. Ids may permute inside a run of tied scores (§3.2) —
    # tests/parity's assert_ids_equal_up_to_ties is the gate for those.
    assert torch.equal(ts, cs), "bit-exactness violated (scores)"
    print(f"P={N_PROBE*MAX_SIZE}: pass rate {torch.isfinite(ts).float().mean():.3f}")
    for name, fn in (("triton", tri), ("cuda", cud)):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        ms, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
        print(f"  {name:>6}: {ms:.3f} ms")
EOF
```

Watch the printed pass rate: `N_VOCAB=50` with `C=2, A_MAX=2` is a low-selectivity
query, so most items survive and the code-row traffic — which both backends pay
identically — dominates. That is the regime §9's ±10% criterion is written for. Push
`N_VOCAB` up if you want a selective-query datapoint too; the cuda side should gain
*relatively* as the pass rate falls, since both skip rows but only cuda pays the
predicate once per item instead of once per scoring tile.

## 7. Step 4 — end-to-end evaluation

Needs `evaluation/data/arxiv-papers/`. The deep-sweep config now carries
`backends: [triton, cuda, torch]`:

```bash
cd evaluation
uv run evaluate --config config/deep_sweeps/arxiv-d128-silvertorch.yaml \
    --algo silvertorch \
    --output results/deep_sweeps/arxiv-d128-silvertorch/silvertorch.json
```

(For a quicker smoke first: append `--backend triton --backend cuda --sweep c0_maincat`.)

Then join cuda vs triton rows on `(filter_kind, sweep, batch_size, k, extra.params)`:

```bash
uv run python - <<'EOF'
import json, collections
rows = json.load(open("results/deep_sweeps/arxiv-d128-silvertorch/silvertorch.json"))
cells = collections.defaultdict(dict)
for r in rows:
    key = (r["filter_kind"], r["sweep"], r["batch_size"], r["k"],
           json.dumps(r["extra"]["params"], sort_keys=True))
    cells[key][r["backend"]] = r
bad_quality = 0
for key, by_b in sorted(cells.items()):
    if "triton" not in by_b or "cuda" not in by_b: continue
    t, c = by_b["triton"], by_b["cuda"]
    qcols = [k for k in t if k.startswith(("recall@", "ndcg@", "precision@", "mrr@"))]
    same = all(t[k] == c[k] for k in qcols)
    bad_quality += (not same)
    print(f"{key[0]:>6} {key[1]:<12} bs={key[2]:<3} k={key[3]:<4} "
          f"triton={t['median_ms']:.3f}ms cuda={c['median_ms']:.3f}ms "
          f"x{t['median_ms']/c['median_ms']:.2f}  peak {t['peak_mem_mib']:.0f}/"
          f"{c['peak_mem_mib']:.0f} MiB  quality {'==' if same else '!= <-- BUG'}")
print("quality mismatches:", bad_quality)
EOF
```

Clause sweeps on cuda cells are now **legal** — the cuda×exact construction
`ValueError` is gone and all three filter modes run there. The shipped
`arxiv-d128-silvertorch.yaml` still lists bloom only; add cuda to an exact config
once §5 and 6c pass, and re-run the join below.

Gates: quality columns **byte-identical** in every cell (→ F4 if not, never
acceptable); latency per §9 criteria; `index_mem_mib`/`peak_mem_mib` deltas explained
by the §3.4 accounting. Concretely, for a bloom cell the cuda `index_mem_mib` should
exceed triton's by the filter-index term times `(n_lists · max_cluster_size / N) − 1`
and nothing else — compute that slack for the dataset first (§3.4) and compare against
the measured delta rather than eyeballing "close enough". An **exact** cell should show
**no** index-memory delta at all: cuda+exact registers the same buffers as
triton+exact.

Note the harness already handles the sharp edges: TF32 pinned off, 20-iteration
warmup absorbs the JIT build + cudagraph capture, and cuda cells reuse the *triton*
standalone filter modules (the filters have no CUDA kernels — only the fused
SilverTorch scorer does).

## 8. Step 5 — profiling (record, don't just eyeball)

Write a one-shot script that runs each impl once at a chosen regime (reuse the 6b
snippet body with a single call instead of do_bench), then:

```bash
ncu --set full --clock-control none --target-processes all \
    -k "regex:cps_score_kernel|cps_bloom_mask_kernel|_codesigned_probe_score_kernel" \
    -o cps_profile -f uv run python /tmp/profile_cps.py
ncu --import cps_profile.ncu-rep --page details | less
```

Record for each kernel:

| Metric | Expectation |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | scoring kernel ≥ Triton's at the same regime; the point of the exercise |
| `dram__bytes.sum` | matches the §3.3 model ±20% (validates the bloom-skip actually removes code-row bytes) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | ≥ 40% on the scoring kernel |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` vs ideal | ≈ 100% sector utilization on code-row loads |
| SASS instruction mix | `IDP4A` present; no `LDSM`, no `BAR.SYNC` in `cps_score_kernel` |

`-lineinfo` is already in the build flags, so ncu's source view maps to the .cu.

**Time phase 2 and phase 3 separately, at `B=1`.** The shipped decision that neither
mask kernel is worth sweeping ("their traffic is noise next to phase 3") is a
*traffic* argument, and at `B=1` with small `P` the cost that matters is not traffic
but the second **launch** — a few µs that phase 3 may not amortize. ncu already
reports per-kernel durations, so read them off the same profile:

```bash
ncu --import cps_profile.ncu-rep --page raw --csv \
  | awk -F, 'NR==1 || /cps_bloom_mask_kernel|cps_clause_mask_kernel|cps_score_kernel/'
```

Record `phase2_us / (phase2_us + phase3_us)` at `B=1, P≈1024` and at
`B=16, P≈65536`. If the small-`B` ratio is material (say > 20%), the "not swept"
decision is what to revisit first — a smaller mask-kernel block or a fused
single-launch variant — *before* touching the scoring kernel. The exact path's clause
mask does more per word than the bloom mask (a `[C, A_max]` row per slot vs one
`and.b64` per 64 slots), so measure both.

## 9. Step 6 — acceptance criteria & sign-off

1. **Correctness (hard):** §5 all green; bit-exact triton↔cuda parity; e2e quality
   columns byte-identical.
2. **Bloom-mode perf (the paper's claim, primary):** 6b shared-input head-to-head —
   cuda ≥ 1.5× faster than triton at the large-P layout (expect 2–4× per §3.3);
   gate: **≥ 1.0×** (no regression) at every layout.
3. **No-filter perf:** tune-sweep `HAS_QB=0` rows — cuda within **±10%** of triton
   per regime (expect parity-to-faster at P ≥ 8192; small-P may be launch-bound).
3b. **Exact-mode perf:** 6c shared-input head-to-head — cuda within **±10%** of the
   Triton exact kernel at every layout. Parity is the honest expectation, not a win:
   the predicate costs one `[C, A_max]` sector per item either way, and the real
   saving — skipping the code row for a rejected item — both backends already have.
   A cuda *loss* beyond 10% points at the second launch (§8) or at the clause loop
   being serialized per lane; a cuda *win* would come from the bit-mask being cheaper
   to re-read than re-evaluating the predicate inside the scoring tile.
4. **E2E:** cuda `median_ms` ≤ triton × 1.10 per bloom cell, and faster on the
   majority of large-P cells; memory deltas explained.
5. If (2)–(4) fail their gates but (1) holds, the backend still merges as
   documented-experimental (F5), not as the default recommendation.

Sign-off template (fill and commit under `docs/plans/` or the PR description):

```markdown
## CUDA backend validation — <date>, <GPU>, driver <v>, CUDA <v>
- build: ok / notes            - SASS check: IDP4A=<n>, LDSM=<n>; triton PTX: <mma|dp4a>
- parity: <pass counts>        - full suite: <counts>   - compile+export gate: <counts>
- tuned DEFAULT_CONFIG: block_p=<>, num_warps=<>, unroll=<>   (regs/thread from ptxas -v: <>)
- IVF padding slack n_lists*max_size/N: <> (predicted bloom index-mem delta)
| regime | triton ms | cuda ms | speedup |   (6a HAS_QB=0 rows + 6b bloom head-to-head)
| exact layout | triton ms | cuda ms | ratio | pass rate |   (6c)
| e2e cell | triton ms | cuda ms | quality identical? | index_mem delta vs predicted |
- ncu: scoring DRAM% triton=<>, cuda=<>; occupancy <>; bytes model err <>%
- phase2/phase3 split: B=1 <>%, B=16 <>%   (bloom mask / clause mask)
- verdict: promote / experimental / revert (reason)
```

## 10. Behavior deltas to be aware of

- `Backend` literal is now `"torch" | "triton" | "cuda"` (propagates to eval config
  typing; YAML `backends:` accepts `cuda` with no schema change).
- All three `filter_mode`s run on `backend="cuda"`. Exact adds a second kernel
  launch (the clause mask) ahead of the same scoring kernel bloom uses.
- A cuda+bloom module's state_dict has `bloom_sigs_t` **instead of** `bloom_sigs`
  (checkpoints are not interchangeable across backends; re-run `register_index`
  when switching). A cuda+**exact** module's state_dict is identical to a
  triton+exact one's — the clause kernel reads `item_clause_attrs` /
  `clause_is_reverse` in place, so exact checkpoints stay portable.
- `register_index` on cuda+bloom additionally pays the one-time transposition
  (`m_bits × 64` small GPU ops).
- Eval cells with `backend=cuda` run *triton* standalone filter modules and oracle
  (unchanged: there are still no CUDA C++ standalone filter kernels — only
  SilverTorch's fused predicates).
- First cuda forward on a cold `TORCH_EXTENSIONS_DIR` compiles for ~1 min.

## 11. Deferred / known-stale

- Paper's stack-machine for arbitrary AND/OR/NOT operation arrays.
- v2 kernel optimizations if step 8 shows headroom: `int2` row loads at D=256,
  config-swept mask-kernel block size. (Per-segment multi-item UNROLL landed as a
  config knob — §3.1, §6a; id prefetch is subsumed by it.)
- Multi-GPU scale-out (paper §Scale Out). (`torch.export` gating for the cuda ops
  landed with WP-D — see §1's scope cuts.)

## 12. Fallback decision tree

- **F1 — build failure:** the wrapper now checks this for you *before* invoking
  nvcc — a message reading `CUDA major-version mismatch: <path> reports CUDA X.Y, but
  this torch wheel was built against CUDA Z` means exactly that and nothing else
  (`cpp_extension.load`'s JIT path never runs torch's own `_check_cuda_version`, which
  is why the wrapper does). Otherwise: set `CUDA_HOME` if the message is
  `no nvcc on PATH or under $CUDA_HOME/bin`; `uv sync` for ninja; nuke
  `$TORCH_EXTENSIONS_DIR` and retry with `ensure_built(verbose=True)`; wheel-nvcc
  symlink farm as documented last resort (§2). Note the build outcome is memoized
  per process, so retry in a **fresh** interpreter after changing the environment.
- **F2 — opcheck/fake mismatch:** fix `register_fake` metadata (shapes
  `[B, k]`, dtypes int64/float32, device from `query`) — never silence opcheck.
- **F3 — graph breaks or cudagraph capture failure on cuda:** confirm the op body
  allocates only via `torch.empty`, launches on `getCurrentCUDAStream`, and no
  `.item()`/CPU tensor sneaked into `_forward_cuda`; last resort, tag the op
  `torch._C.Tag.cudagraph_unsafe` (falls out of the graph, stays correct — record
  the perf cost).
- **F4 — bit-exactness drift:** first suspect a *predicate* bug (run
  `test_bloom_mask_matches_rowwise` + `test_build_transposed_sigs_bits` to split
  mask vs scoring); check no fast-math/fma flag crept into
  `extra_cuda_cflags`; check the dequant multiply order. Do not relax the gate to
  `assert_topk_matches` except as a temporary diagnostic — §3.2 proves equality is
  achievable, so drift is a bug.
- **F5 — perf below gates:** re-tune (`--regime` at the failing shape), try the v2
  items from §11 in order (UNROLL first), profile before/after each. If still
  short, merge as experimental with the recorded gap; do not ship cuda as default.
- **F6 — e2e quality drift:** never acceptable; bisect with the kernel-level parity
  tests at the failing cell's exact shapes.
- Anything else: reproduce on the Triton backend first — only cuda-only failures
  are this branch's regressions.

## 13. References

Paper: `articles/silvertorch.md` (§Bloom Index for Fig. 3 and the transposed AND;
§Fused Int8 ANN Search for dp4a + warp tiling; §ANN and Filtering Co-design +
Algorithm 1 for the phase structure and the 1-bit mask).

- CUDA C++ Programming Guide — warp reduce (`__reduce_add_sync`, cc 8.x+), warp vote,
  `__ldg`, grid limits: <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- CUDA C++ Best Practices Guide — memory coalescing (32-byte sectors):
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory>
- CUDA Math API — `__dp4a` integer intrinsics (signature/semantics; note the page
  does **not** state the arch floor):
  <https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html>
- Mixed-Precision Programming with CUDA 8 — the sm_61 floor for `__dp4a` comes from
  here and from its declaration in the toolkit's `sm_61_intrinsics.h`, not from the
  Math API page: <https://developer.nvidia.com/blog/mixed-precision-programming-cuda-8/>
- Using CUDA Warp-Level Primitives (shuffle reductions, vote):
  <https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/>
- NVIDIA Ampere GPU Architecture Tuning Guide — cp.async semantics, SM limits,
  occupancy: <https://docs.nvidia.com/cuda/ampere-tuning-guide/>
- A100 datasheet (HBM2e 1 555 GB/s): <https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet.pdf>
- Triton NVIDIA backend `min_dot_size` / "tensorcores with padding" for int8:
  <https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py>;
  `tl.dot` minimum-shape history: <https://github.com/triton-lang/triton/issues/4230>
- `triton.testing.do_bench` (CUDA events + L2 flush):
  <https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html>
- torch.utils.cpp_extension (`load`, `TORCH_CUDA_ARCH_LIST`, `TORCH_EXTENSIONS_DIR`,
  ninja): <https://docs.pytorch.org/docs/stable/cpp_extension.html>
- torch.library — `custom_op`, `register_fake`, `opcheck`, opacity to
  compile/export: <https://docs.pytorch.org/docs/stable/library.html>;
  Python custom-ops tutorial:
  <https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html>;
  C++/CUDA custom-ops tutorial (current-stream contract, pybind vs TORCH_LIBRARY):
  <https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html>
- CUDAGraph Trees — custom ops safe-by-default, `cudagraph_unsafe` tag:
  <https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html>
- nvcc pip-wheel / CUDA_HOME caveat:
  <https://forums.developer.nvidia.com/t/nvidia-cuda-nvcc-pip-wheel-installation/221307>
- Hatchling build file selection (wheel `packages` includes non-.py files):
  <https://hatch.pypa.io/latest/config/build/>
