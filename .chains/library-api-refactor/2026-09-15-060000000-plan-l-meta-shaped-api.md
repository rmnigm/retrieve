---
chain: "library-api-refactor"
branch: "main"
nextStep: "WP-1 (roadmap L1): the move into modules / ops / indexing / functional with no behaviour change, on dev/l1-library-layout; nothing else edits retrieve/ while it is open."
created: "2026-09-15T06:00:00Z"
---

# Plan L: `torchretrieve` 0.2, the Meta-shaped API (`modules` / `ops` / `indexing`)

Source: `docs/plans/library-api-refactor.md` §1-§11 (plan "L"), planned 2026-09-15 on `development` at `76f8985` from three inventories (this package, the harness, Meta's `silvertorch` at `main`, which since `21aa35e` only grew two `*ModuleBuilder` classes). Supersedes the parts of the 2026-09-06 library review deferred to "after B4" (A5 `_host.py`, A9 `__all__` rows, D3, D5). Companions: evaluation-package-layout (V), library-harness-boundary (X). The as-built map is `docs/system/architecture.md` (its move table records the result).

Question: what does `retrieve` look like with Meta's two-level shape (modules with builders, registered ops, nothing else) while containing Meta's modules and ops (imported and wired, never copied), our LiNR V1-V4, our Triton kernels, and a home for helpers Meta never shipped (k-means / k-means++, quantizers, bloom hashing, IVF layouts, top-k glue, the tuner)?

## 1. Starting point (`76f8985`, 36 files, ~6,000 lines)
Layout by history: `kernels/{filters,linr,silvertorch}` (silvertorch held a Triton kernel, a standalone bloom kernel and the official adapter), `layers/{silvertorch,linr,filters,utils}` (utils held an index builder, quantizers, a top-k epilogue and a retriever). Settled already: split backend literals + `check_backend`; table dispatch; `-1` trap closed; load post-hook; salt buffer; `OfficialConfig`; stable argsort; opaque compaction custom ops; deterministic k-means; CUDA / CuTe gone; suite 519. Still open: the torch backend existed twice (inline eager code in every layer + `tests/parity/conftest.py::ref_cps_phase23`, and parity compared against the test copy); `KMeansTorch` random-init only (the paper says k-means++); LiNR V1-V4, `set_query_params` and `capturable` lived in the harness; small debts (private `_quantize_int8_global`, a private `_impl` exported, `__all__` gaps, the launch scaffold, pyproject name `retrieve` while PyPI knows `torchretrieve`); no `build_timings`.

Meta's conventions to copy: two layers and nothing else; ops through one side-effecting import; no `utils`, no config objects beyond constructor args; a module holds its index as buffers and its forward is a thin op call; a `*Builder` per module separating build-from-raw from load-prebuilt. They ship no k-means, quantizer, exact filter, ANN-level module, tuner or Triton.

## 2. Decisions
- D1 two public layers `retrieve.modules` and `retrieve.ops`, plus small `retrieve.indexing` (build-time math) and `retrieve.functional` (query-time torch glue that is not a kernel). `layers`, `kernels`, `layers/utils` gone.
- D2 `retrieve.__init__` exports modules, the two backend literals and the two ABCs; op and helper namespaces are submodules.
- D3 ops keep names and schemas (renaming changes `torch.ops` schemas, invalidates tune JSONs and parity artifacts, buys cosmetics); module names stay.
- D4 three op namespaces with identical signatures: `ops.triton`, `ops.reference` (the torch backend and the parity oracle, one home), `ops.official`.
- D5 LiNR V1-V4 become library modules (`modules/linr.py`), filter as submodule, `forward(query, query_clause_attrs=None)`; bodies moved from the harness; `set_query_params` moves; `capturable` a class attribute.
- D6 builders for the composites only (`SilverTorchBuilder`, `LiNRBuilder`), Meta's fluent shape; primitives and filters keep `register_index`; `build_silvertorch` retired.
- D7 Meta's modules and ops imported and wired, never copied; `OfficialMissing` names `torchretrieve[official]`.
- D8 structure-preserving first (WP-1: no number changes, schemas / buffer names / order unchanged, parity bit-exact, `code_version` changes once), features second (WP-2).
- D9 k-means++ opt-in (`init="random" | "kmeans++"`, default random) until D1's numbers say otherwise; recorded numbers were taken on random init.
- D10 one-release compatibility shim (`retrieve.layers`, `retrieve.kernels` re-exporting under `DeprecationWarning`), removed in 0.3; pyproject `name = "torchretrieve"`, `0.2.0`, import name `retrieve`. (Collision with coding-guidelines D2 resolved by the user as temporary tooling; deleted at L5.)
- D11 sequenced before the C4 rerun and before D1: `code_version` is the library tree hash, so the move must precede any cell meant to survive; C4 gates the final layout once.
- D12 the test tree keeps its by-purpose split; `ref_cps_phase23` moves into the library as `ops.reference.codesigned_probe_score`.

## 3-9. Target (now as built; see architecture.md)
Public API: the 20 names of `retrieve.__all__`; `SilverTorch(k, n_lists, n_probe, filter_mode="none", m_bits=None, k_hash=None, n_iter=10, seed=0, kmeans_init="random", backend="triton", official=None)`; `LiNRV1(k, *, filter=None, backend)`, `LiNRV2(k, *, filter, backend)`, `LiNRV3(k, *, candidate_pool=5000, seed=0, filter=None, backend)`, `LiNRV4(k, *, filter=None, backend)`; builders ending in `build()` with `set_item_embeddings` xor `set_state_dict`. Encoders are not library code. The harness's `PATHS` derived from `interfaces.DISPATCH` by a test. Docs: `retrieve/docs/` rewritten around the builders, new `indexing-and-ops.md`; tests gain composites, builder round trip, k-means++, `test_public_api.py` (no kernel import on `import retrieve`).

Interaction with other plans: O's paths become `ops/official/{__init__,adapter}.py`; O TF-1 becomes `ops/triton/bloom_transposed.py` + a `HAS_MASK` path; review A5 written in WP-1 as `ops/triton/_host.py`; H's `algos.py` becomes ~80 lines (C5); the parked export and live-update plans re-scope against this layout.

## 10. Work packages
- WP-1 the move, no behaviour change (CPU 1.5 d + GPU suite). Gate CPU: ruff, `test_public_api.py`, links 0, harness CPU suite through the shim, `git diff --stat -M` against the move table. Gate GPU: full suite; every parity file bit-exact; `ops.reference.codesigned_probe_score` `torch.equal` to the pre-move `ref_cps_phase23`; pre-move state dicts (3 backends x 3 modes) load and return identical outputs; the shim warns; the golden worktree runs one cell.
- WP-2 composites, builders, k-means++, build timings, boundary clauses (CPU 2 d + GPU suite). Gate: composites `torch.equal` to hand-composed primitives on every backend x filter kind; builder round trip; k-means++ tests; `test_boundary.py`; full suite. Unblocks C5.

## 11. Risks
Collision with in-flight edits (nothing else edits `retrieve/` during WP-1); extracting `ops/reference` must change no number (the `torch.equal` gate); `code_version` invalidation (zero cost if D11 holds); builders must only call `register_index`; k-means++ on 3M x 128 (fall back to k-means‖ if plain D² exceeds a minute at `n_lists = 8192`); if D11 is overruled C4 runs twice.
