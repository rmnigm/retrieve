# `retrieve` module → torch.export-clean (library promise)

> **Status: NOT STARTED. Anchors are stale — re-scope before executing.** Written 2026-05-23,
> before the K/E refactor track and the CUDA backend landed. What rotted:
>
> - the `torch_knn` algo it plans work on **was deleted** (E1.1); its row in the mode map below
>   and the algo-wiring bullet are dead,
> - `SilverTorch(filter=...)` is now `filter_mode=`,
> - the kernels are `@triton_op`, not `@custom_op`,
> - verification commands use `conf/` and `--algorithms`; the real paths are `config/` and
>   `--algo` (singular, required),
> - the "shelved native-CUDA experiment" is no longer shelved — it shipped as
>   `SilverTorch(backend="cuda")`, see [../system/kernels.md](../system/kernels.md).
>
> The *goal* below is still valid and the design sketch is still the best thinking on it.
> Treat the line references and command examples as archaeology.
>
> **Scope:** every layer that backs an algo registered in
> evaluation/retrieval/algos/__init__.py. 6 layer
> classes (5 linr + `SilverTorch`). `ShardedSilverTorch` remains out of scope.

## Goal

`retrieve` ships as a library. A downstream consumer pulls it in, composes one of these retrieval layers with **their own** model and retrieval graph, calls `torch.export.export(my_model_with_index, ...)`, and deploys the resulting `.pt2` to a C++ runtime. **This repo does not ship `.pt2` artifacts.** What we promise is: every layer that backs an eval algo is **export-clean** — its `forward` traces under `torch.export.export()` without raising on Optionals, data-dependent shapes, `.item()` syncs, or compiled-wrapper opacity.

Kernel-side surface is already in shape (`triton_op` / `custom_op` wrappers with non-Optional schemas, `Config` dataclasses, full-width `(ids, counts)` from the compact family, no `.item()` on the export path, host-side `pad` tail eliminated). What remains is layer-side: collapse `forward(... Optional ...)` matrices into one signature per export entry, so a consumer can pick the mode at construction time and trace cleanly.

## Algo → layer → mode map

| Algo (eval name)                       | Layer                          | Modes                                                          |
|----------------------------------------|--------------------------------|----------------------------------------------------------------|
| ~~`torch_knn`~~ (algo deleted)          | `FullScanKNN`                  | `full`, `masked`, `candidates` — layer still exists, no algo uses it |
| `triton_knn` / `linr_v1_filter_mask`   | `PostfilterKNN`                | `full`, `masked`                                                |
| `linr_v2`                              | `PrefilterKNN`                 | `candidates` only (filter required)                             |
| `linr_v3`                              | `OneBitKNN` + `PrefilterKNN`   | stage 1: `full` or `candidates`; stage 2: `candidates`          |
| `linr_v4`                              | `PostfilterKNNInt8`            | `full`, `masked`                                                |
| `silvertorch`                          | `SilverTorch`                  | sibling methods: `forward_ivf_only`, `forward_bloom`, `forward_exact`, `forward_candidates` |

Filter modules (`BloomFilter`, `ExactAttributeFilter`) are already one-signature-per-evaluator — they don't need a mode flag.

## Design principles

1. **One forward signature per export entry.** Optional matrices collapse to a construction-time `mode: Literal[...]` (or a sibling method). The `backend` flag stays — it's also a Python attribute and specializes at export time.
2. **No deprecation shims.** Old kwarg-style Optional-arg signatures are replaced in the same PR as the `mode=` flag; eager-mode callers update at the same time.
3. **Tests must pass before AND after**, with no relaxed tolerances. Parity tests in particular (Triton-vs-torch reference) must continue to assert bitwise-or-near-bitwise equivalence.
4. **No `.pt2` zoo in this repo.** A small smoke harness verifies each (layer, mode) round-trips through `torch.export.export()` + `torch.export.load()`. It is not a deployable artifact and is not wired into eval-time builds.

## Layer-side mode flags

### [PostfilterKNN](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py) (linr_v1)

Add `mode: Literal["full", "masked"]`:

```python
def forward(self, query):                # mode="full"   — no mask path; skip the masked_fill / isfinite wrap
def forward(self, query, mask):          # mode="masked" — apply mask + isfinite → -1 sentinel
```

Drop the `mask is not None` branches at [postfilter_knn.py:50-58](../../retrieve/src/retrieve/layers/linr/postfilter_knn.py#L50-L58).

### [PostfilterKNNInt8](../../retrieve/src/retrieve/layers/linr/postfilter_knn_int8.py) (linr_v4)

Same shape as `PostfilterKNN`. Add `mode: Literal["full", "masked"]`. The int8 quantize + `torch._int_mm` body is common to both modes; the `mask is not None` branch is the only Optional.

### [PrefilterKNN](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py) (linr_v2, linr_v3 stage 2)

Add `mode: Literal["full", "candidates"]`:

```python
def forward(self, query):                                  # mode="full"
def forward(self, query, candidate_ids, counts):           # mode="candidates"
```

Drop:
- the `candidate_ids is None` branch at [prefilter_knn.py:54](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L54);
- the `counts is None → torch.full(p, ...)` fallback at [prefilter_knn.py:117-118](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L117-L118);
- the two layer-side `if p == 0` early returns at [prefilter_knn.py:74-78](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L74-L78) and [prefilter_knn.py:111-116](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py#L111-L116). `fused_masked_knn_topk` handles per-row empty already.

### [OneBitKNN](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py) (linr_v3 stage 1)

Add `mode: Literal["full", "candidates"]`. `forward` dispatches to the existing custom_ops 1:1:

```python
def forward(self, query):                                  # mode="full"        → oporp_1bit_match_topk_full
def forward(self, query, candidate_ids, counts):           # mode="candidates"  → oporp_1bit_match_topk_indirect
```

Same dispatch in `_forward_torch_eager`. Drop the `counts is None` fallback at [one_bit_knn.py:149-155](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py#L149-L155).

### [FullScanKNN](../../retrieve/src/retrieve/layers/utils/retrieval.py) (torch_knn)

Add `mode: Literal["full", "masked", "candidates"]`. The current `forward` matrixes `mask` × `candidate_ids` — three modes cover the practical combinations:

```python
def forward(self, query):                                  # mode="full"
def forward(self, query, mask):                            # mode="masked"   — applies post_filter_topk
def forward(self, query, candidate_ids):                   # mode="candidates"
```

`mode="masked"` keeps `post_filter_topk` ([retrieval.py:9](../../retrieve/src/retrieve/layers/utils/retrieval.py#L9)) post-topk so the K slot count stays static. `mode="candidates"` needs a pad-to-K wrap on top of `_forward_candidates`'s `min(self.k, scores.shape[1])` at [retrieval.py:57](../../retrieve/src/retrieve/layers/utils/retrieval.py#L57) so the output shape is statically `[B, k]`.

### [SilverTorch](../../retrieve/src/retrieve/layers/silvertorch/main.py)

`filter: FilterMode = "none"` ([Literal["none", "bloom", "exact"]](../../retrieve/src/retrieve/layers/silvertorch/main.py#L24)) already exists and `has_bloom` / `has_exact` are derived `@property`s. What remains is collapsing the two Optional arguments in `forward`:

- Collapse `query_clause_attrs: Tensor | None = None` ([main.py:224-244](../../retrieve/src/retrieve/layers/silvertorch/main.py#L224-L244)) into sibling `forward_ivf_only(query)` / `forward_bloom(query, qa)` / `forward_exact(query, qa)` methods (one export entry per filter mode). `forward` becomes a Python dispatcher for eager runtime.
- Extract `_forward_candidates` ([main.py:369-385](../../retrieve/src/retrieve/layers/silvertorch/main.py#L369-L385)) to a public `forward_candidates(query, candidate_ids)`. Drop the `candidate_ids is not None` branch in `forward` at [main.py:236](../../retrieve/src/retrieve/layers/silvertorch/main.py#L236).

Sibling-method form rather than a `mode=` flag because each filter mode has a distinct prep block (`evaluate_bloom_sigs` vs `evaluate_clause_mask` upstream of the probe kernel); inlining all three into one `forward` body via a single dispatch would still need a `mode` check anyway.

## Algo wiring

Eval algos thread the chosen `mode=` into the layer constructor so the eager benchmark path exercises the export-clean code paths. Minimal blast radius — mode is fixed per algo cell:

- `torch_knn.py`: pick `mode="full"` / `"masked"` / `"candidates"` from whether `filter_mod` is wired and which cell is being benched.
- linr_v1.py: `mode="full"` if `filter_mod is None` else `"masked"`.
- linr_v4.py: same shape as linr_v1.
- linr_v2.py: always `mode="candidates"` (filter required).
- linr_v3.py: stage 1 `OneBitKNN(mode="full" or "candidates")` based on whether a filter is wired (linr_v3.py:65); stage 2 always `PrefilterKNN(mode="candidates")`.
- silvertorch.py: already maps `filter_kind` → `filter`; route the eager `forward` dispatcher through it.

## Verifying the library promise

Not a deployable script; just a smoke harness that proves the library promise — every (layer, mode) round-trips through `torch.export.export()` + `torch.export.load()` and produces equal outputs to the eager module. Lives under `retrieve/tests/export/` (not yet created) (new dir) alongside the parity / correctness suites; reuses the existing test fixtures.

```python
# tests/export/test_export_roundtrip.py — parametrized over (layer_cls, mode)
@pytest.mark.parametrize("layer_cls, mode, build_args", [
    (PostfilterKNN,     "full",       _full_args),
    (PostfilterKNN,     "masked",     _masked_args),
    (PostfilterKNNInt8, "full",       _full_args),
    (PostfilterKNNInt8, "masked",     _masked_args),
    (PrefilterKNN,      "full",       _full_args),
    (PrefilterKNN,      "candidates", _cand_args),
    (OneBitKNN,         "full",       _full_args),
    (OneBitKNN,         "candidates", _cand_args),
    (FullScanKNN,       "full",       _full_args),
    (FullScanKNN,       "masked",     _masked_args),
    (FullScanKNN,       "candidates", _cand_args),
])
def test_export_roundtrip(layer_cls, mode, build_args):
    idx = layer_cls(k=K, mode=mode, backend="triton")
    idx.register_index(item_embs)
    args = build_args(idx, batch=B)
    ep = torch.export.export(idx, args)
    out = tmp_path / f"{layer_cls.__name__}_{mode}.pt2"
    torch.export.save(ep, out)
    loaded = torch.export.load(out).module()
    expected = idx(*args)
    got = loaded(*args)
    assert torch.equal(got[0], expected[0])
    assert torch.allclose(got[1], expected[1])


# Plus the four silvertorch entries (sibling methods, not a mode flag):
# torch.export.export(SilverTorch(filter_mode="none"),  (query,), method="forward_ivf_only")
# torch.export.export(SilverTorch(filter_mode="bloom"), (query, qa), method="forward_bloom")
# torch.export.export(SilverTorch(filter_mode="exact"), (query, qa), method="forward_exact")
# torch.export.export(SilverTorch(filter_mode="none"),  (query, cand), method="forward_candidates")
```

Failure mode: if the consumer's `torch.export.export()` raises on one of these layers, the regression is reproducible from this test in-tree.

## Tests

- Parametrize [retrieve/tests/correctness/test_linr.py](../../retrieve/tests/correctness/test_linr.py) over `mode` for every linr layer (PostfilterKNN, PostfilterKNNInt8, PrefilterKNN, OneBitKNN).
- Parametrize [retrieve/tests/correctness/test_silvertorch.py](../../retrieve/tests/correctness/test_silvertorch.py) over `filter ∈ {"none", "bloom", "exact"}` × `candidates ∈ {False, True}`.
- Add a `FullScanKNN` mode-parametrized test under `tests/correctness/` (none today).
- Keep [retrieve/tests/compile/test_silvertorch_compile.py](../../retrieve/tests/compile/test_silvertorch_compile.py)'s zero-graph-breaks assertion intact.
- The export-roundtrip suite above is the load-bearing acceptance check.

## Out of scope

- **Shipping `.pt2` files from this repo.** The library is exportable; consumers compose layers with their own models and produce artifacts on their side.
- **`build_export.py` in `evaluation/`.** Not authored. Eval-time benches use eager / `torch.compile` paths, not `.pt2`.
- **End-to-end-algo bundling** (cascade + filter build captured in one graph). A consumer can do this themselves by composing the exported layers in their model code.
- `ShardedSilverTorch` — shelved per the [roadmap scope note (revised 2026-05-23)](00-roadmap.md). Doesn't exist in the tree.
- Native-CUDA `codesigned_probe_score` — shelved; see the CUDA backend section of [../system/kernels.md](../system/kernels.md).
- AOTI bump (`torch>=2.5`, `aoti_compile_and_package`). Separate later effort.
- Live-update API ([live-update-api.md](live-update-api.md)) — independent feature track; upsert/delete bodies stay export-clean so the two plans compose.

## Verification (when executed)

```bash
cd retrieve && uv run pytest tests/ -v
cd retrieve && uv run pytest tests/export/ -v   # the load-bearing library-promise check
cd evaluation && uv run evaluate --config conf/goodreads/d128-filter.yaml --algorithms linr_v2 --backend triton
cd evaluation && uv run evaluate --config conf/500m/d128-quality.yaml   --algorithms linr_v1_filter_mask linr_v3 linr_v4 silvertorch torch_knn --backend triton
```

## Sweeps (acceptance check)

```bash
grep -rn "Tensor | None\|Optional\[Tensor\]" \
    retrieve/src/retrieve/layers/linr/ \
    retrieve/src/retrieve/layers/silvertorch/ \
    retrieve/src/retrieve/layers/utils/retrieval.py
# Zero hits in forward signatures of the in-scope layers.
```
