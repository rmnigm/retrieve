---
title: meta-published-vs-adapted
created: 2026-10-11
updated: 2026-10-11
type: comparison
tags: [paper]
sources: [official-fork/ (dev/meta-fork), docs/validation.md, docs/artifacts/hub-index.md, docs/paper/reproduction-deviations.md]
---

# Meta's SilverTorch: as published vs adapted

> **Status: NOT CITABLE.** The adapted code is on `dev/meta-fork` and is
> not reviewed yet (roadmap [META-FORK](../roadmap.md#1-meta-fork-review-accept-tag)).
> Every state below is a gate result that has not passed review; no number
> here is paper material (CLAUDE.md rule 2).
>
> **What this page is.** The study's SilverTorch arm is Meta's official
> SilverTorch code, adapted by us ([decisions](../decisions.md#goal-and-scheduling)).
> This page is the paper's account of that adaptation: what it changed
> and why, grouped by the defect in the published code each change removes,
> and the parity gate that holds the adapted code to the published code.
> The per-change list (files, mechanism, commit) has one home,
> `official-fork/CHANGES.md` on `dev/meta-fork`; it is not restated here.
> Meta's code as published is the reference column of this page, not an arm
> of the study. The comparison the study makes is SilverTorch (adapted)
> against LiNR ([claims](claims.md)).

## 1. What the study's SilverTorch is

| part | whose | notes |
|---|---|---|
| base | Meta: `meta-recsys/silvertorch`, vendored unmodified at `21aa35e`, then synced to upstream `main` at `22c2007` | `22c2007` is one upstream commit (CCCL 3 compatibility in the bloom indexer); on CUDA 12.8 it compiles to the same path as `21aa35e` |
| changes | ours, each a row of `official-fork/CHANGES.md` with a gate | [§2](#2-what-the-adaptation-changed) |
| licence | Apache-2.0, as upstream: Meta's `LICENSE` verbatim, every upstream header kept, a "Modified by the retrieve authors" line in each changed file (§4(b)); upstream ships no `NOTICE` | [release §2](release-and-licenses.md#2-third-party-code) |
| phase 1 (k-means, int8 global codes, top-`n_probe` probe selection) and query preparation | ours, shared with `backend="official"` | the same index and the same probes in both columns of this page |
| the reference | `backend="official"`: Meta's code as published at `21aa35e`, built `-O3` (host code; the shipped `setup.py` builds it at `-O0`, [OF-11](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper), bit-identical outputs) | the two load in one process (`torch.ops.st.*` and `torch.ops.stfork.*`), so parity and interleaved timing compare them in one process |

In code the adapted version is still the `official-fork/` workspace
member, package `silvertorch_fork`, `backend="official-fork"`; roadmap
[RENAME](../roadmap.md#5-rename-the-adapted-backend-becomes-silvertorchs-default)
gives it SilverTorch's default backend name.

## 2. What the adaptation changed

By the defect in Meta's code as published that each change removes
(defect ids: [deviations §5](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper)).
"Kind" says what the gate must show: **build** (no behaviour change),
**bit-exact** (`torch.equal` to the published code), **bloom change** (the
filter's semantics change on purpose; recall against the exact oracle and
the false-positive count instead, with Meta's behaviour still selectable
and itself bit-exact).

| defect in the published code | change (CHANGES row) | kind |
|---|---|---|
| OF-6: two test files unparsable after Meta's comment strip, two registered ops not built | F8, its build fix, Meta's suite fixes; rename to `silvertorch_fork` / `stfork` so both versions load together | build |
| OF-11: host code at gcc `-O0` | F5: `-O3` for C++ and nvcc host code; device code unchanged | build (bit-identical) |
| OF-4: host syncs in `fused_kmean_ann` (`repeat_interleave` with no output size, `.item()` on cumsums) | C3: device-side warp index, statically sized payload buffers | bit-exact |
| OF-5: plans decoded on the host every call and uploaded from pageable memory; OF-3: eager only | C4: plans prepared once in `prepare_queries`, decoded per thread on the device; with C3 the forward has no host sync and captures | bit-exact; capture |
| OF-4: launch count (19 per unfiltered forward, ≈ 32 with bloom) | C5: the forward after phase 1 as one op, `stfork::search`; C6: query prep in one kernel; C7: two-level top-k; C8: the partial bloom mask built inside the payload kernels; K fixes 1-14 (one-kernel epilogue, our own top-k select, per-width scorer variants, row-parallel prep) | bit-exact |
| our adapter's full-`N` exact mask in front of Meta's scorer (exact is not in Meta's paper; [ST-3](reproduction-deviations.md#3-silvertorch-as-we-run-it--deviations-from-the-paper)) | C9: the exact clause filter over the probed docs only, inside `search` | bit-exact |
| OF-10: `m_bits` ignored (width per 2048-doc bundle from its max term count) | C12: a configured bloom width (`bloom_width="configured"`); `"upstream"` keeps Meta's | bloom change |
| OF-12 (from OF-1): bundle-bloom false positives in the results | C13: an exact recheck of the bloom's survivors; the adapted backend's default; Meta's bloom by a flag, for the parity gate and C7 ([decisions](../decisions.md#goal-and-scheduling)) | bloom change |
| — (scorer configuration) | C10: none (Meta's scorer already skips failing lanes); C11: scorer variant per `D`, cooperative loads for `D` ≥ 128 (K fix 14) | bit-exact |
| no fake (meta) implementations, an undeclared mutation in `is_topk` | the compile gate: shape functions for all 16 `stfork` ops, `is_topk`'s mutation declared | `torch.compile` / `torch.export` |

## 3. The parity gate

What holds the adapted code to the published code (the META-FORK plan's
definition of done, checked by its review):

- **X, bit-exact.** Every change that is not a bloom change: scores
  `torch.equal`, ids equal up to ties, adapted against published `-O3` in
  one process, int32 path and an fp16 subset, eager and graph, over a grid
  of every bench × width × `n_probe` × batch × k at ≤ 10 M and at 30 M.
- **R, bloom changes.** Recall@100 / @1000 against the exact oracle and the
  false-positive count, at Meta's width and at the configured one; with
  Meta's bloom behaviour selected the adapted code is X-gated as well.
- **S, host work.** Under `torch.cuda.set_sync_debug_mode("error")` no
  forward raises; a profiler trace shows no host sync, no device-to-host
  copy and no pageable upload; no allocation size depends on device data.
- **Capture and compile.** Each forward captures as one CUDA graph whose
  replays equal eager; the layer compiles `fullgraph` with 0 graph breaks
  (default and `reduce-overhead`, static and dynamic shapes) and exports.

State (NOT CITABLE; raw outputs on the Hub, [hub-index](../artifacts/hub-index.md)):

| gate | state | Hub |
|---|---|---|
| the unmodified adapted tree against the published code (C0) | 126 / 126 arXiv d128 cells `torch.equal` | `artifacts/meta-fork/c0` |
| S and capture after C3 / C4 | no device-to-host copy and no pageable upload per forward at C4; capture equal to eager | `artifacts/meta-fork/c3-c4` |
| X at ≤ 10 M, head 82fafb6 | 1,536 / 1,536 cells `torch.equal` to the published `-O3` code (goodreads, arXiv d128 / d256, PubMed d768, YFCC d192) | `artifacts/meta-fork/head-82fafb6` |
| compile gate | green on `dev/meta-fork` (its validation rows) | — |
| R (C12, C13) | running | — |
| X at 30 M (LAION d256) | running | — |
| review | not started | — |

## 4. Speed against the published code

On the ≤ 10 M grid at head 82fafb6 the adapted code is faster than the
published `-O3` code, eager against eager (the published code cannot
capture), beyond the A/A floor (`artifacts/meta-fork/head-82fafb6`). The
paper's numbers for this comparison are T3, timed in the `h2h` cells of
roadmap [NUMBERS](../roadmap.md#2-numbers-adapted-silvertorch-over-the-study-grid)
(regenerated by the final FULL-RERUN),
not the gate grid.

The adaptation's definition of done also asks it to be no slower than our
retired Triton SilverTorch backend on any cell. That is an engineering
target, not a claim of the paper. Four clusters of cells still miss it
(roadmap META-FORK): wide unfiltered scans at `D` ≥ 192 (Meta's scorer),
the full bloom path on multi-clause filters (Meta's full-`N` bloom search),
the partial bloom path at batch 1 under CUDA graph (the payload kernels),
and the exact filter at wide probes at batch 1 (C9's per-doc check).

## 5. What this comparison cannot say

- It is not a comparison with Meta's production system: the end-to-end
  QPS, OverArch and TCO claims (S-1 to S-5, S-15) are untestable here
  ([claims](claims.md)).
- With C13 the adapted arm's bloom returns exact-filter results; Meta's
  bloom admits false positives (OF-12). Where a filter result differs, the
  two columns answer different queries, and the paper says which bloom
  semantics a number used.
- Phase 1 is ours in both columns, so differences in k-means or
  quantization are not Meta's.
- One A100-SXM4-80GB, clocks unlocked and sampled
  ([provenance §4](provenance-and-disclosure.md#4-timing-which-clock-estimator-and-why-the-question-is-not-pedantic));
  the published column is eager only.

## 6. The earlier comparison

The head-to-heads of Meta's code against our Triton reimplementation
(b3, the kernel-opt head-to-head, H2H-FINAL, the C7 cells at
campaign-v2.x) are not in the study. Their records stay on the Hub and
their rows in [validation](../validation.md) are marked "not in the study".
The findings about Meta's published code that they produced (OF-1 to
OF-12) stand in [deviations §5](reproduction-deviations.md#5-deviations-in-metas-released-code-against-metas-paper),
and they are what §2 above removes.
