# Stage 1 — Autotune separation

> See [00-roadmap.md](00-roadmap.md). This is the first of three main-thread stages.

## Context

Every Triton kernel in [retrieve/src/retrieve/kernels/triton/](../../retrieve/src/retrieve/kernels/triton/) makes its tuning decisions at the host-wrapper call site — three of them via `@triton.autotune` (`codesigned_probe_score`, `fused_masked_knn_topk`, `oporp_1bit_match_topk`), one via a shape-dependent `BLOCK_N` if-ladder (`bloom_match`), the rest via module-level `_BLOCK_N` / `_NUM_WARPS` constants tuned by hand. The autotune key on `fused_masked_knn_topk` is stabilized by a `_bucket_p` ladder so the cache doesn't compile-per-shape; `oporp_1bit_match_topk` and `codesigned_probe_score` include Optional-presence bools (`HAS_INDICES`, `HAS_QB`) in their autotune keys.

This sprawl works fine for eager execution but blocks the next two stages:

- **Stage 2** (`triton_op`/`custom_op` wrapping) needs every kernel call to have a stable, scalar-typed launch signature with no autotune behind it. `triton_op`'s auto-fake mechanism re-traces the host wrapper under `FakeTensor`, which trips on host-side shape branches; even with `custom_op`'s opaque-boundary semantics, an autotune dispatcher inside the body interacts badly with cudagraph capture and adds invisible compile pressure during warmup.
- **Stage 3** (kernel optimization) wants each per-kernel change measurable in isolation. A runtime-autotune layer makes that hard — a perf delta could be the kernel change or autotune picking a different config because the workload shifted slightly.

The cleanest fix is to lift tuning out of runtime into an offline pass: per-kernel `Config` dataclass + arch-keyed `REGISTRY` + `lookup(device, problem_hint)` helper, resolved once at `register_index` time and stored as a Python attribute on the layer. A standalone `tune-kernels` script populates the REGISTRY entries.

This stage is **structurally** a refactor, not a perf change. Acceptance is "all parity tests pass, eager perf within noise of the prior autotune-selected configs on the local arch."

## Approach

### Per-kernel skeleton

```python
# In each kernel file (e.g. retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py):

@dataclass(frozen=True)
class FusedMaskedKnnTopkConfig:
    block_n: int
    num_warps: int
    num_stages: int = 3

DEFAULT_CONFIG = FusedMaskedKnnTopkConfig(block_n=64, num_warps=4, num_stages=3)

REGISTRY: dict[tuple[str, str], FusedMaskedKnnTopkConfig] = {
    ("sm_80", "p256"):     FusedMaskedKnnTopkConfig(block_n=32,  num_warps=4),
    ("sm_80", "p2048"):    FusedMaskedKnnTopkConfig(block_n=64,  num_warps=4),
    ("sm_80", "p16384"):   FusedMaskedKnnTopkConfig(block_n=128, num_warps=8),
    ("sm_80", "p131072"):  FusedMaskedKnnTopkConfig(block_n=256, num_warps=8),
    # populate with tune-kernels output, one row per arch × bucket
}

_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)

def lookup(device: torch.device, p_hint: int) -> FusedMaskedKnnTopkConfig:
    if device.type != "cuda":
        return DEFAULT_CONFIG
    cap = torch.cuda.get_device_capability(device)
    arch = f"sm_{cap[0]}{cap[1]}"
    bucket = next((f"p{b}" for b in _P_BUCKETS if p_hint <= b), f"p{1 << (p_hint - 1).bit_length()}")
    return REGISTRY.get((arch, bucket), DEFAULT_CONFIG)
```

The kernel's `@triton.jit` body stays exactly as-is. The change is purely at the host-wrapper boundary: replace `@triton.autotune(...)` with a manual REGISTRY lookup; pass `BLOCK_N`, `num_warps`, `num_stages` to the kernel launch from the looked-up `Config`.

### Layer-side plumbing

Each consumer layer (`ExactAttributeFilter`, `BloomFilter`, `SilverTorch`, `OneBitKNN`, `PrefilterKNN`) accepts an optional `<kernel>_config: <Kernel>Config | None = None` in `__init__`, resolves via `lookup(device, problem_hint)` at `register_index` time (when `N` and device are both concrete), and stores the resolved config as a plain Python attribute. The host wrapper takes a `config: <Kernel>Config` arg and pulls `block_n` / `num_warps` / `num_stages` out of it. No autotune at runtime.

### `tune-kernels` script

New file: [evaluation/scripts/tune_kernels.py](../../evaluation/scripts/tune_kernels.py). Click CLI: `--kernel <name>`, `--device cuda:0`, `--shape-grid path.json`. Uses `triton.testing.do_bench`. Output: per-arch JSON (informational) + console output formatted as REGISTRY-pasteable rows like `("sm_80", "p2048"): FusedMaskedKnnTopkConfig(block_n=64, num_warps=4, num_stages=3),`.

Also add `[project.scripts]` entry in [evaluation/pyproject.toml](../../evaluation/pyproject.toml): `tune-kernels = "scripts.tune_kernels:main"`. Add `scripts` to the package list.

The script is run **once per arch** by a human, results pasted into the kernel file. There is no CI; running the script is the gating step before merging a stage 1 PR.

## Per-kernel application

Order: easy → hard. Each kernel ships as its own commit (or PR), measured against the prior autotune baseline on the local arch.

### 1. `clause_mask` — easy

[kernels/triton/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py). Today: fixed `_BLOCK_N = 256, _NUM_WARPS = 4`. No autotune. The "separation" is just wrapping the constants in `ClauseMaskConfig` + REGISTRY + lookup. One REGISTRY row per arch initially. No atomics → safe to widen the REGISTRY later if profiling justifies.

### 2. `clause_compact` — easy

[kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py). Same as `clause_mask`. **MUST NOT autotune** — `tl.atomic_add` corrupts state across autotune trials. One REGISTRY row per arch; document this constraint in the file header.

### 3. `bloom_compact` — easy

[kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py). Same shape and same atomic-add constraint as `clause_compact`. Same recipe.

### 4. `bloom_match` — medium

[kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py). The `block_n = 128 if n >= 128 else triton.next_power_of_2(int(n))` ladder ([line 68](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py#L68)) becomes 3-4 REGISTRY rows keyed by `(arch, "n_<bucket>")`. Layer-side: `BloomFilter` resolves `lookup(device, n)` once in `register_index` (when `N` is concrete) and stores. The if-ladder vanishes from the host wrapper.

### 5. `fused_masked_knn_topk` — medium

[kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py). Strip `@triton.autotune` ([line 36](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L36)). The 8-config grid `block_n × num_warps` becomes REGISTRY rows. The existing `_P_BUCKETS = (256, 2048, 16384, 131072, 1048576)` ladder ([line 9](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L9)) becomes the REGISTRY key dimension. The `do_not_specialize=["P_REAL"]` JIT hint ([line 37](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py#L37)) goes away (it existed to make autotune's cache key independent of runtime `P`; without autotune, the standard JIT specialization on shape is fine).

Caller side: `PrefilterKNN.register_index` (in `prefilter_knn.py`) gets the kernel's `problem_hint = p_bucket` based on its expected candidate width. The host wrapper takes `config` as an arg; the existing `_bucket_p` helper moves into the layer or into `lookup`.

### 6. `oporp_1bit_match_topk` — medium

[kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py). Strip autotune ([line 36](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L36)). The current autotune key includes `HAS_INDICES` — this becomes an additional REGISTRY key dimension: `(arch, regime, has_indices: bool)`. The kernel body's `HAS_INDICES: tl.constexpr` ([line 55](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py#L55)) stays; the host wrapper resolves the right config and passes it.

Note: the cleaner long-term shape is to split the no-indices and has-indices paths into separate kernels (or to wait until stage 2 collapses the Optional via a layer-side `mode` flag and the dummy-tensor trick). For stage 1, just expand the REGISTRY key by one bool dimension. Don't restructure the kernel.

### 7. `codesigned_probe_score` — hard

[kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py). Strip autotune ([line 42](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L42)). Same `HAS_QB` constexpr issue as `oporp_1bit_match_topk`'s `HAS_INDICES` — extend REGISTRY key by one bool. `problem_hint = P (n_probe * max_cluster_size)`.

The kernel's two Optional inputs (`query_bits`, `bloom_sigs`) still bind to dummy tensors at the host wrapper ([lines 187-190](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py#L187-L190)). Don't move the dummy-bind into the layer yet — that's stage 2's job.

## Critical files

| Stage | File | Touch |
|---|---|---|
| Per-kernel | [retrieve/src/retrieve/kernels/triton/filters/clause_mask.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_mask.py) | Add `ClauseMaskConfig` + REGISTRY + lookup. Host wrapper takes `config`. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) | Same. Document atomic-add ⇒ no autotune. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py](../../retrieve/src/retrieve/kernels/triton/filters/bloom_compact.py) | Same. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/bloom_match.py) | Replace shape-dependent BLOCK_N ladder with REGISTRY lookup at register_index time. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/fused_masked_knn_topk.py) | Strip autotune; migrate `_P_BUCKETS` ladder to REGISTRY key; drop `do_not_specialize`. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py](../../retrieve/src/retrieve/kernels/triton/linr/oporp_1bit_match_topk.py) | Strip autotune; REGISTRY key includes `has_indices: bool`. |
| Per-kernel | [retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py) | Strip autotune; REGISTRY key includes `has_qb: bool`. |
| Layer | [retrieve/src/retrieve/layers/filters/exact_attribute.py](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) | `__init__` accepts optional configs; resolve in `register_index`. |
| Layer | [retrieve/src/retrieve/layers/filters/bloom.py](../../retrieve/src/retrieve/layers/filters/bloom.py) | Same. Bucket lookup for `bloom_match` lives here, not in the kernel host. |
| Layer | [retrieve/src/retrieve/layers/silvertorch/main.py](../../retrieve/src/retrieve/layers/silvertorch/main.py) | Same; resolves `codesigned_probe_score` config in `register_index`. |
| Layer | [retrieve/src/retrieve/layers/linr/one_bit_knn.py](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) | Same; resolves `oporp_1bit_match_topk` config. |
| Layer | [retrieve/src/retrieve/layers/linr/prefilter_knn.py](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py) | Same; resolves `fused_masked_knn_topk` config. |
| New | [evaluation/scripts/tune_kernels.py](../../evaluation/scripts/tune_kernels.py) | Click CLI; `triton.testing.do_bench`; one `tune_<kernel>` per kernel. |
| New | [evaluation/pyproject.toml](../../evaluation/pyproject.toml) | Add `tune-kernels` script entry; `scripts` to packages. |
| Tests | [retrieve/tests/parity/test_*.py](../../retrieve/tests/parity/) | All seven parity tests update kernel-call sites to pass `config=DEFAULT_CONFIG` (or a deliberately-different config to check the plumbing). |
| Docs | [docs/system/kernels.md](../../docs/system/kernels.md) | Document the per-kernel `Config` + REGISTRY convention; note the `tune-kernels` workflow. |

## What this stage explicitly does NOT do

- **No `.item()` removal.** That's stage 2's full-width `(ids, counts)` API change.
- **No mode flag / Optional collapse on layer forwards.** That's `torch-export-refactor.md` territory; not needed for stage 1 or stage 2.
- **No `_build_query_signatures_eager` / `_project_oporp_1bit_query_eager` export-path bypass.** Those are for [torch-export-refactor.md](torch-export-refactor.md) (deferred).
- **No new `build_export.py`.** Same.
- **No moving post-launch torch ops (`topk`, `gather`, `cat`-pad) out of host wrappers.** Stage 2 may want this for cleaner `register_fake` impls; defer the decision.
- **No kernel-body changes.** `@triton.jit` bodies stay byte-identical.

## Verification

For each per-kernel PR:

```bash
# 1. Parity tests for the touched kernel must pass.
cd retrieve && uv run pytest tests/parity/test_<kernel>.py -v

# 2. Full correctness suite must stay green.
cd retrieve && uv run pytest tests/ -v

# 3. Eager perf regression check: bench the affected algo path before and after,
#    median ms/call over 200 iters, warmup 20. Acceptable: within ±5% of baseline.
#    Use measure_forward_cuda from evaluation/retrieval/bench_tools.py.

# 4. No autotune left where it was stripped:
grep -n "@triton.autotune" retrieve/src/retrieve/kernels/triton/<path>/<file>.py
# Expect zero.
```

End-of-stage check after all seven kernels:

```bash
grep -rn "@triton.autotune" retrieve/src/retrieve/kernels/triton/   # zero hits
grep -rn "next_power_of_2" retrieve/src/retrieve/kernels/triton/    # zero hits (the bloom_match shape branch is gone)
grep -rn "do_not_specialize" retrieve/src/retrieve/kernels/triton/  # zero hits
```

Also: confirm `tune-kernels` runs cleanly on the local arch and produces REGISTRY-pasteable output for each kernel.

## Risks

- **`bloom_match` bucket policy too coarse**: a 3-4 bucket REGISTRY for BLOCK_N may underperform the current shape-tight `next_power_of_2(n)` selection on workloads with `n < 16`. Likely irrelevant in practice (production `n` is in the millions), but the tune script should include sub-128 shapes to verify.
- **`HAS_INDICES` / `HAS_QB` REGISTRY blowup**: each Optional-bool dimension doubles the row count. With two booleans across two kernels that's 4× rows, manageable. If stage 2 collapses the Optionals via mode flags, the rows collapse back.
- **Eager autotune was masking a config bug**: if removing autotune surfaces a kernel that was silently relying on the autotuner finding a working config (e.g., a config the hand-written REGISTRY initially misses), parity tests catch it but stack traces may be cryptic. Mitigation: ship the per-kernel changes in dependency order; verify each independently.
