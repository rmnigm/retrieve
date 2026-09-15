# The library / harness boundary — what crosses it, in which direction

> **Status:** planned 2026-09-15 on `development` at `76f8985` (nothing
> implemented). A contract, not a work list: its clauses are implemented by
> [library-api-refactor.md](library-api-refactor.md) WP-2 (library side) and
> [evaluation-package-layout.md](evaluation-package-layout.md) WP-V1 (harness
> side), and pinned by two tests (§5). Read it when either side is about to
> grow something the other side might already own.
>
> **Ordering authority:** [00-roadmap.md](00-roadmap.md) §1 (Phase L, C5).

> **Steer 2026-09-15 (user), recorded in every plan it touches.**
> *"We don't care about reproducing old results now, we're improving all code
> and rewriting, then testing and profiling, then running the full evals step
> by step."* The A1 golden baseline stops being a gate and becomes
> information; roadmap **C4 closed** on what the harness proves about itself
> (graph capture 20/20, kill-and-resume, cross-backend parity, the official
> cell end to end), not on equality with the pre-v2 harness. Order of work:
> **code first, then tests and profiling, then the evals one step at a time.**
> Authority: [00-roadmap.md](00-roadmap.md) §1 status block;
> [evaluation-harness-v2.md](evaluation-harness-v2.md) WP-4's amendment block.
>
> **§4's last clause is wrong as worded, and C4 proved it (2026-09-15).** "Bit-
> exactness between backends is the library's parity suite" is right; H WP-4's
> reading of it — `jaccard_vs_first@100 == 1.0` for "the exact algos" — is not.
> *Exact* describes exact **filtering** (no approximation in candidate
> selection), not bit-identical arithmetic between two implementations of an
> fp16 dot product. `linr_v2` scores 0.998743 torch-vs-triton with
> `score_max_abs_diff` 9.77e-3, reproduced independently by the golden at
> 4.2e-4 recall; `linr_v1` and `linr_v4` agree at exactly 0.0 and SilverTorch
> at exactly 1.0 with `score_max_abs_diff` 0.0. Roadmap **L4**
> ([linr-v2-backend-parity.md](linr-v2-backend-parity.md)) settles whether that
> is precision or a defect, and the clause is reworded from its answer.

## 1. The rule

**The library retrieves; the harness measures.** Anything that decides *which
items come back* for a query — a kernel, a filter, a cascade, a quantizer, an
index layout, a query-time knob like `n_probe` or `candidate_pool` — is
library code. Anything that decides *what to run, on what data, how to time
it, and what to write down* is harness code. When a piece of code could go
either way, ask whether a user of `torchretrieve` who has never seen this
repository would want it: the LiNR V3 cascade, yes; the oracle blob keyed by
content fingerprint, no.

Today the rule is broken in one direction, in one file:
[evaluation/retrieval/algos.py](../../evaluation/bench/algos.py) owns the
paper's V1–V4 as wrappers whose only content is the composition of library
primitives, plus `set_query_params` (on `Silvertorch` and `LinrV3`) and the
`capturable` flag — three things that describe the retriever, not the
benchmark. The other direction is clean: nothing under `retrieve/` imports
`evaluation`, and the library's `FullScanKNN` is not used by the harness (its
oracle is `bench/oracle.py`).

## 2. What the harness may import

Exactly this, pinned by `evaluation/tests/test_dependency_direction.py`
([evaluation-package-layout.md](evaluation-package-layout.md) §4):

```python
from retrieve import (LinrBackend, SilverTorchBackend, FilterModule, RetrievalModule,
                      SilverTorch, SilverTorchBuilder, OfficialConfig,
                      LiNRV1, LiNRV2, LiNRV3, LiNRV4, LiNRBuilder,
                      BloomFilter, ExactAttributeFilter)
import retrieve.functional                  # combine_masks / compact_mask, if a sweep needs them
from retrieve.interfaces import DISPATCH    # the one table the PATHS test derives from
```

Today's imports are already close: `algos.py` takes seven classes from
`retrieve` and three names from `retrieve.interfaces`; `data.py` and
`oracle.py` take `FilterModule`; one test reaches into
`retrieve.layers.silvertorch` for `OfficialConfig` (which **L** exports at
the top level). Not allowed after this contract: `retrieve.ops` (the harness
never calls a kernel; a kernel-only number is a `tune-kernels` or `--profile`
run), `retrieve.indexing` (an ablation over k-means init or bloom width is a
*module* constructor argument: `kmeans_init`, `m_bits`, `k_hash`, `n_lists`),
`retrieve.modules.knn` / `bit_knn` (the primitives are composed inside the
library's V1–V4; a harness algo that needs a new composition adds a module to
the library first), `retrieve.modules.official` (Meta's modules are reached
through `SilverTorch(backend="official", official=OfficialConfig(...))`; the
S9 ablation is `OfficialConfig(bloom_path="full")`, as B1 built it).

## 3. Who owns what

| concern | owner | after **L** WP-2 and C5 |
|---|---|---|
| LiNR V1–V4 as compositions of primitives + a filter; `set_query_params`; `capturable` | **library** (`retrieve.modules.linr`, `retrieve.modules.silvertorch`) | harness `algos.py` is a name → class table |
| SilverTorch Algorithm 1 incl. `filter_mode`, `backend`, `OfficialConfig`, the official adapter | library | unchanged |
| filters, their kernels, their composition helpers | library | unchanged |
| k-means (random / k-means++), quantizers, bloom hashing, IVF layouts | library (`retrieve.indexing`) | reached only through module constructor args |
| top-k / compaction glue, subset predicates | library (`retrieve.functional`) | unchanged |
| kernel autotuning | library (`tune-kernels`) | unchanged; the harness records nothing about tuning beyond `code_version` |
| `FullScanKNN` | library, as a *module* (exhaustive KNN is a legitimate retriever for small N) | its docstring drops "oracle"; the harness does not use it |
| brute-force **filtered** oracle, `attrs_digest`, blob v4, `target_in_filter`, `pass_rate` | **harness** (`bench/oracle.py`) | unchanged |
| quality metrics (recall / ndcg / precision / mrr / jaccard) | harness (`bench/metrics.py`) | unchanged (training keeps a frozen local copy, **V** D5) |
| latency, memory, provenance, clocks, graph capture, sync detection | harness (`bench/measure.py`) | unchanged; `graph_callable` reads `module.capturable` instead of a harness-stamped attribute |
| the algorithm table, `PATHS`, eligibility | harness (`bench/algos.py`) | `PATHS` derived from `retrieve.interfaces.DISPATCH` by a test |
| record identity, storage, resume, `flat.csv` | harness (`bench/records.py`) | new file, same behaviour |
| query encoders (SASRec history → vector; text → vector) | harness (`training/encode.py`, `eval_datasets/etl/`) | unchanged; the library has no encoder |
| datasets, the on-disk layout, HF Hub | harness (`eval_datasets`) | `layout.py` owns the contract |
| tables, figures | harness (`bench/report.py`, D4) | — |
| `torch.compile` / CUDA-graph capture of a forward | harness (`measure.graph_callable`) | the library never compiles internally; a module is capturable as-is or says why (`official`: `capturable = False`, `compile()` raises) |

## 4. What the library promises so the harness can measure it

Each clause names the **H** section that needs it and where it is delivered.
A harness that relies on anything not listed here adds a clause, not a
workaround.

| clause | needed by | status / delivered by |
|---|---|---|
| `forward(query, query_clause_attrs=None) -> (ids [B, k] int64, scores [B, k])` on every algo-level module (`SilverTorch`, `LiNRV1`–`V4`); `-1` / `-inf` sentinels | H §2.4 quality pass, one code path for all algos | `SilverTorch` today; V1–V4 in L WP-2 |
| `module.k` is settable after `register_index` and takes effect on the next forward; no module sizes a buffer by `k` at registration | H §2.5 perf per `k` without rebuild | true today per layer (`test_algos.py::test_layer_*_k_not_baked`); the composites forward it (L WP-2) |
| `SilverTorch.set_query_params(n_probe=…)`, `LiNRV3.set_query_params(candidate_pool=…)` re-validate and take effect without rebuild | H §8.2 A (build vs query params) | exists on the harness wrappers today; moves into the library in L WP-2 |
| `SilverTorch.build_timings` = `{"kmeans_s", "quantize_s", "assemble_s", "filter_s"}` after `register_index` | H §2.3 per-phase build time | new, L WP-2 |
| all index state is in `buffers()`, filter included as a submodule; `Σ numel·itemsize` over `buffers()` is the index size | H §2.3 `index_mib`, `filter_mib` | true today (`measure.index_bytes`); the composites register the filter as `self.filter` |
| `capturable: bool` on every algo-level module (`False` on `official`); no `torch.compile`, no `.item()` / `.cpu()` on the `triton` / `torch` forward path | H §2.5 `graph` mode, `cudagraph_skips == 0`, `set_sync_debug_mode` clean | stamped by the harness's `build` today; becomes a class attribute in L WP-2; the sync-free property is C4's finding (i) and holds |
| the bloom salt is a buffer | H §7 | done (B5) |
| every module raises `ValueError` for a backend it cannot run | H §1.5 / `PATHS` | done (review #5, B4) |
| `retrieve.interfaces.DISPATCH`: `{module class: {backend: path label}}` importable data | `PATHS` derived by test | new, L WP-2 |
| builders: `set_state_dict(sd).build()` reproduces `set_item_embeddings(x).build()` for the same seed; the load hook re-derives cached scalars | H §8.2 I (shipped index artifacts, F4) | hook done (review #4); builders in L WP-2 |
| `OfficialConfig.cache_plans` selectable at construction | O §9 fairness (review #1) | done (B1); the harness passes it through `OfficialConfig`, not a setter |
| bit-exactness between backends is the library's parity suite, never the harness's job | H §2.4 (spill-file jaccard is a *wiring* check only) | O T1–T7, B2 done |

## 5. The two gates

- **Harness side** — `evaluation/tests/test_dependency_direction.py`: every
  `import` / `from` in `bench/`, `training/`, `eval_datasets/` resolved with
  `ast`; cross-package and library edges must be in the allow-list of §2 and
  **V** §4. CPU.
- **Library side** — `retrieve/tests/correctness/test_boundary.py` (GPU;
  L WP-2): for each class in `(SilverTorch, LiNRV1, LiNRV2, LiNRV3, LiNRV4)`
  on a tiny index: `k` mutation changes the output width without
  re-registration; `buffers()` covers every tensor attribute (walk `__dict__`
  for stray tensors); a forward under `torch.cuda.set_sync_debug_mode("error")`
  on `triton` and `torch` raises nothing; `set_state_dict` round-trip equals a
  fresh build; `capturable` is a class attribute; `DISPATCH` names every
  class × backend. `tests/bench/test_paths.py` (CPU) then asserts
  `PATHS == derive(DISPATCH)`, replacing today's markdown-parsing test of
  `architecture.md`'s dispatch table.

## 6. Naming map

The three vocabularies must stay aligned in code, docs and prose (CLAUDE.md
"Naming that must stay straight"):

| paper | library class | harness algo key (as built) | notes |
|---|---|---|---|
| LiNR V1 (dense, post-filter mask) | `LiNRV1` over `PostfilterKNN` | `linr_v1_filter_mask` | `backend` accepted, cuBLAS either way (`path: cublas`) |
| LiNR V2 (exact pre-filter → candidates) | `LiNRV2` over `PrefilterKNN` | `linr_v2` | filter required; no `none` cell |
| LiNR V3 (1-bit prefilter → rerank) | `LiNRV3` over `OneBitKNN` → `PrefilterKNN` | `linr_v3` | `candidate_pool` is a query param |
| LiNR V4 (int8 dense) | `LiNRV4` over `PostfilterKNNInt8` | `linr_v4` | cuBLAS either way |
| SilverTorch Algorithm 1 (was "QuantizedIVF") | `SilverTorch` | `silvertorch` | `backend ∈ {triton, torch, official}`; `n_probe` is a query param |
| Meta's official ops | `retrieve.ops.official`, `SilverTorch(backend="official")` | `silvertorch` + `backend: official` | reference arm, `path: official`, eager only |
| exact attribute filter / bloom filter | `ExactAttributeFilter` / `BloomFilter` | `filter_kind: clause` / `bloom` | the harness's `clause` = the library's exact |

## 7. Validation record

*(the two gates of §5 are recorded in L WP-2's and C5's records; nothing is
recorded here.)*
