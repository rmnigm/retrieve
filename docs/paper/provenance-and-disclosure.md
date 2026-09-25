# Provenance and hardware / software disclosure

> **Status:** written 2026-09-15 on `dev/f1-f3-paper` (roadmap **F3**, paper gap
> **G9**). Paper material for §4 "Experimental setup" of
> reproducibility-paper.md §C.4, and for the
> reproducibility-checklist appendix (§B.7).
>
> **References.** Labels such as O §13, H §12 and V §11 name sections of
> the design plans this section was written from (O: official integration,
> H: harness v2, V: harness package layout). Those plans are retired; each
> link now points to the artifacts or wiki page that carries the fact, and
> a label with no link has no public source beyond this text.
>
> Every fact in §1–§4 was verified on the box in the sessions named per line.
> §5 is the disclosure the user's steer of 2026-09-15 requires and its wording
> is deliberate: **no claim of equivalence with the pre-v2 harness may be made
> from the C4 numbers** ([H WP-4's amendment block](../validation.md#harness-gates)).
> §6 lists what is not yet validated and the roadmap step each item waits on
> (CLAUDE.md rule 2).

## 1. Hardware

| item | value |
|---|---|
| GPU | one **NVIDIA A100-SXM4-80GB** |
| driver | **580.159.04** |
| GPUs used per measurement | **1** — every number in this work is single-GPU. Neither paper's multi-GPU configuration is reproduced |
| SM clock control | **none available.** `nvidia-smi -lgc` is denied in this container and there is no `sudo`; the tool reports *"The current user does not have permission to change clocks"* |
| observed SM clocks | 210 MHz idle, **1410 MHz under load** (the boost clock), on every under-load sample of every run recorded to date |
| profiling | `ncu` is **blocked** in the container; per-kernel attribution comes from `torch.profiler` |
| exclusivity | the GPU is **shared**. GPU work is serialized with an advisory lock (`flock /workspace/gpu.lock`), one job at a time, but jobs from different workers alternate on the lock — see §4 |
| storage | `/workspace` is a 100 GB quota volume; virtual environments live on the local disk under `/venvs/` |

The box is a rental and has been re-provisioned during this project: the working
tree, the virtual environments and the staged datasets of the 2026-09-06 session
did not survive into 2026-09-15, and the datasets were restaged
([H §11](../../evaluation/golden/README.md)). Nothing in the results depends on
machine state that is not re-derivable from the repository and the public
datasets.

## 2. Software

| item | value |
|---|---|
| OS / Python | Linux, **Python 3.11** |
| PyTorch | **2.10.0+cu128** (the wheel is built against CUDA 12.8) |
| Triton | **3.6.0** |
| CUDA toolkit on the box | **12.4 only** — `nvcc 12.4` at `/usr/local/cuda`. There is no 12.8 toolkit here; earlier notes claiming a 12.8 build are stale and were corrected on 2026-09-15 |
| toolkit / wheel mismatch | building a C++/CUDA extension with nvcc 12.4 against the cu128 wheel is a **minor** version mismatch: torch warns (`cpp_extension.py: minor version mismatch`) and builds. It raises only on a *major* mismatch, and every CUDA test the upstream package ships passes on the resulting extension |
| Meta's SilverTorch | `silvertorch` **1.0.0** at commit **`21aa35e28b6dd9a91e9ee35efb0857715e86bda7`** (`meta-recsys/silvertorch`, Apache-2.0), installed through the repo's `official` extra. Build: **2 m 17 s**, `ninja`, 10 translation units, `TORCH_CUDA_ARCH_LIST="8.0"`, `MAX_JOBS=32` |
| linter | `ruff` **0.15.6** |
| environment manager | one `uv` workspace, one lockfile. **`uv sync --extra official --all-packages`** — the plain `--extra official` at the workspace root *uninstalls* `pytest`, which lives in the members' `dev` groups |
| our code | the library is pinned per record by `env.code_version`, the **git tree hash of `retrieve/src/retrieve`**, alongside `env.commit`, `env.git_branch` and two dirtiness flags (§3) |

Build friction worth disclosing, because a reader reproducing the "official"
arm will hit it: the upstream README's own verification command
(`pytest silvertorch/`) **fails at collection** on three test files whose Buck
`load_library` lines were mangled by Meta's line-based internal
comment-stripping; excluding them, 99 tests and 3 subtests pass in 11.6 s, every
CUDA test included. Two ops registered in the source tree
(`is_topk`, `take_top_k_and_gather_from_main_and_fresh`) are absent from
`setup.py` and therefore exist in no OSS build. Both from
[O §13.1](../artifacts/official-silvertorch/README.md).

## 3. What every record carries

Provenance is per record, not per paper: the harness writes one JSONL record per
cell, and each record's `env` block is the authority for that cell
([evaluation.md](../system/evaluation.md)).

| field | content |
|---|---|
| `env.gpu`, `env.driver`, `env.cuda`, `env.torch`, `env.triton` | the §1/§2 stack as the process saw it |
| `env.commit`, `env.git_branch`, `env.code_version` | the harness commit, the branch, and the **library subtree tree hash** — the resume key includes `code_version`, so a kernel change invalidates a cell instead of silently reusing it |
| `env.dirty`, `env.repo_dirty` | `dirty` is scoped to `retrieve/src/retrieve` (with a `files:<sha256>` fallback when it is dirty) and a record with `dirty: true` is **not citable**; `repo_dirty` is the whole tracked tree and flags only working-copy noise such as artifact scripts |
| `env.host`, `env.python`, `env.started`, `env.config_sha` | box, interpreter, timestamp, the config that produced the cell |
| `env.sm_mhz_idle`, `env.mem_mhz`, `env.sm_max_mhz`, `env.power_limit_w` | the process-start sample; **provenance only, compared with nothing** |
| `env.sm_mhz_load` | the median of the cell's **under-load** samples |
| `env.clocks_drift` | fires when any under-load sample is more than 5 % from the process's first *under-load* sample |
| `perf[].sm_mhz` | the SM clock sampled immediately after that variant's last timing window's sync, with the GPU still at its load clock. **This is the clock a latency number in the paper is normalised against** |
| `perf[].*` | per `(k, batch, mode)`: median / p20 / p80 / IQR, outlier counts, window spread, `peak_fwd_mib`, `load: closed_loop`, and for a non-capturable path a null median with a `reason` |
| `schema_version` | **2** since C5 |

Seeds: **`seed: 0` only** in everything measured to date. Multi-seed cells and
confidence intervals are roadmap **D1** (paper gap G4) and are listed in §6.

## 4. Timing: which clock estimator, and why the question is not pedantic

Because clocks cannot be locked (§1), a latency comparison must state *which
estimate of the clock* it uses. This is not a formality — it is the single
largest measurement artifact this project has produced.

The harness's C4 gate normalised latency by dividing each run's clock into the
baseline's, and the two numbers were **two different estimators of the same
clock**:

- the baseline's clock was a **whole-run median of a 30 s-cadence trace**:
  n = 65 samples, median **1155 MHz**, max 1410, min 210 — a histogram of
  1155 × 55, 1410 × 9, 210 × 1, i.e. dominated by index-build, oracle and
  between-cell samples rather than by timing windows;
- the run's clock was a **single sample taken under load**, right after the last
  timing window's sync: **1410 MHz** on all 360 perf entries, without exception.

Dividing 1410 by 1155 injects a flat **×1.221** bias — four times the gate's 5 %
threshold — and on that basis **92 of 99 latency rows "failed"**. Filtering the
baseline's own trace to `utilization > 50 %` gives median 1410 and min 1410: the
two runs' GPUs were at the same clock throughout. At the **matched estimator**,
with no threshold changed, **66 of 66 rows pass at batch 8 and 16** (ratios
0.957–1.030), and all 16 remaining failures are at batch 1, 14 of them with the
new harness *faster*. [H §12.4](../validation.md#harness-gates)

Two consequences the paper must carry:

1. **Every latency number states its estimator.** Use the under-load
   `perf[].sm_mhz` (or the cell's `env.sm_mhz_load`); never a whole-run median.
   Records written before C5 carry a schema-1 `env.sm_mhz` whose meaning is the
   whole-run median, and must not be compared with schema-2 numbers directly.
2. **Batch 1 is not a 5 % target on this box.** The *baseline harness's own*
   repeat runs — same code, same box, same data — spread up to **21.1 %** at
   batch 1 (median 14.1 %) against ≤ 0.4 % at batch 8 and ≤ 0.1 % at batch 16,
   while the new harness's own timing windows are tight (≤ 0.3 % at batch 1).
   The noise is in the baseline. Batch-1 latency is reported with its spread and
   is not used for any speedup claim.

Related artifacts, both **fixed at C5** and named here so pre-C5 records are read
correctly: `clocks_drift` used to compare a start-of-process *idle* sample
against under-load samples, so it fired on the GPU boosting (10 of C4's 18
`unstable` flags); and a `clocks_locked` field was recorded `true` on a box that
cannot lock clocks at all, because it meant "within 2 % of an expected value"
and the box happened to be there. `clocks_locked` and `--expected-sm-mhz` are
removed; `env.sm_mhz` is split into `sm_mhz_idle` / `sm_mhz_load`
([V §11](../artifacts/evaluation-package-layout/README.md), L4-c).

**The baseline's latency columns are not clock-controlled**, for a second reason
as well: the GPU was shared while they were produced. Cells of the 2026-09-15
baseline re-derive ran while another worker held the lock in alternation, so
each cell had the GPU to itself for its own duration but the run as a whole was
interleaved. No latency comparison against those columns is clock-controlled,
and none is used as a result.

## 5. What was compared with what — and what is *not* claimed

This section exists because the temptation is to describe the benchmark rewrite
as validated against the old one. It is not, and the paper does not say so.

**What was done.** The pre-v2 ("golden") baseline was **re-derived once**, on
2026-09-15, by running the *old, unchanged* harness against the *current*
library on the same box, the same data, the same cached queries and the same
oracle, so that every delta is a library delta and not a harness delta. It
produced 11 of 11 cells, 99 rows. Six cells came out **bit-identical** to the
original baseline; five moved, by 5e-6 to 1.6e-4 — between 5× and 160× the 1e-6
comparison tolerance — and the cause is the deterministic k-means fix, which
changes the centroids on every backend.
[H §11](../../evaluation/golden/README.md)

The v2 harness was then run against that re-derived baseline. It **matched 9 of
the 11 cells, with a worst passing delta of 7.5e-9**, three orders of magnitude
inside the tolerance; six of the nine are exact to 0.0 on `recall@k`. **Two
residuals did not match and are unresolved**: `linr_v4` at **7.3e-5** (the
chunk-shape attribution was tested and *falsified* — see
[reproduction-deviations.md](reproduction-deviations.md) §8 R-1) and arXiv
`silvertorch` `recall@100` at **2.0e-6**, unattributed.
[H §12.3](../validation.md#harness-gates), correction in
[V §11.2](../artifacts/evaluation-package-layout/README.md)

**What is not claimed.** *No claim of equivalence between the v2 harness and the
pre-v2 harness is made from these numbers.* The baseline comparison is
information, not a gate: on the user's decision of 2026-09-15 the harness step
closed on what the harness proves about **itself**, and that is what the paper
reports:

- 20 of 20 cells `status: ok`;
- `cudagraph_skips == 0` on **20/20** cells, every capturable variant measured;
- a clean kill-and-resume (SIGTERM mid-cell, `--resume` continues at the next
  cell, no duplicate and no recomputation);
- cross-backend parity: SilverTorch `torch` vs `triton` **exactly 1.0**
  `jaccard@100` with `score_max_abs_diff 0.0` on both datasets and both probe
  widths;
- Meta's official kernels end to end on goodreads at `jaccard@100` **0.999849**
  (probe 24) / **0.999805** (probe 32) against our Triton, with the expression
  plan cache **off** so the per-forward parse is paid.

Two things that arm did **not** establish, stated rather than buried: the
official arm reaches only **0.985** on arXiv — since explained by roadmap B3 as
fp16 resolution against arXiv's score distribution rather than a filter effect,
costing 3.1e-4 recall@100
([official-vs-reimplementation.md](official-vs-reimplementation.md) §6.1) — and
the LiNR V2 backend divergence, which turned out to be a real precision defect
in our own kernel (see
[reproduction-deviations.md](reproduction-deviations.md) §7 D-1).

**Superseded, and superseded openly.** The thesis's tables and figures were
produced by the pre-v2 harness. They are **not** cited as results in this paper;
every table and figure comes from `report.py` over the campaign records
(roadmap D1 → D4).

## 6. Not yet validated — the honest list

Each line names the roadmap step it waits on. Until that step's gate is green,
the item is not paper material (CLAUDE.md rule 2).

| item | waits on |
|---|---|
| ~~any Triton-vs-official **kernel or end-to-end speed** comparison~~ — **B3's gate passed 2026-09-15**; the comparison, with its own limits, is [official-vs-reimplementation.md](official-vs-reimplementation.md) | done |
| confidence intervals, paired tests and a second seed behind **any** of B3's ratios | **D1-b** |
| every quality, latency, memory, QPS and pass-rate table in the paper | **D1**, then **D4** |
| fp32 accumulation in `fused_masked_knn_topk`, the reduction-width audit across every Triton kernel, and the fp64-oracle parity file that would have caught the class | **L5** |
| multi-seed cells, bootstrap CIs, paired tests for every "A is faster than B" sentence | **D1** (gap G4) |
| mean / p95 / p99 latency, per-query latency vectors, closed- and open-loop QPS under a P99 budget | **D1** (gap G3) |
| external baselines at matched recall (Faiss GPU/CPU, HNSW, cuBLAS floor; then cuVS, filtered-graph CPU indexes) | **D2** (gaps G5, G13, G14) |
| bloom false-positive rate and memory against filter width on **real** attributes, for our bloom and Meta's | **D3** (gap G6) |
| the SilverTorch co-design ablation (full mask → IVF vs fused partial bloom) as scratch memory and latency vs probe count | **D1** S9 cells / **G-b** |
| the LiNR V1/V2 pass-rate crossover and the liquidity curve on a controlled sweep | **G-b** (gap G11) |
| the V3 bit-width sweep and the "keep 1 %" operating point | **G-b** (gap G15) |
| a Triton transposed bloom index, and any claim about the paper's transposed-index win in current code | **G-a** |
| scale beyond 5.4 M items (10 M real, synthetic ladder, the 240 M and 1 B stress points) | **G-b** (gap G10); the large datasets are **E2**–**E4**, deferred on disk |
| the YFCC-10M, PubMed, Semantic Scholar and KuaiRand cells | **E5** (after D1) |
| the packaged artifact: tagged release, Zenodo DOI including the pinned official sdist, HF data and oracles, one-command reproduction | **F4** |

## 7. Data provenance

| dataset | as used | source |
|---|---|---|
| Goodreads (work-id) | d128 item embeddings (gSASRec), `item_attrs_narrow` `[N, 4, 4]` int64, 313,178 test users of which the first **10,000** are used and **9,859** kept (141 have zero survivors under the `c0_genre` condition and are dropped), oracle top-K from the published `gt_d128/` blobs | `pinkmeme/eval-goodreads-work-id` on the Hugging Face Hub, the project's own mirror; 2.6 GB staged |
| arXiv (papers) | `content_d128` item embeddings (Nomic-Embed), `item_attrs_narrow` `[N, 5, 4]`, 10,000 queries, `c0_maincat` condition | `pinkmeme/eval-arxiv-papers`, 1.3 GB staged |
| Yambda-500M / Yambda-5B | unfiltered cells only, in the thesis; not re-run | `pinkmeme/eval-yambda-500m`, `pinkmeme/eval-yambda-5b` |
| YFCC-10M | ingested, ground-truth gate passed; cells wait on **E5** | Big-ANN filter-track release |

**Disclosure on the mirror's layout.** The Hub mirror still publishes the
**pre-`3b1b5b3` 1-indexed `[N + 1, …]`** artifact layout: goodreads
`item_attrs_narrow` is `[797085, 4, 4]` against 797,084 item ids, arXiv is
`[2988997, …]` against 2,988,996. The harness detects this and drops the
leading pad row. Anyone re-deriving from the mirror gets the 1-indexed files and
must keep that handling; a loader that reads attribute row `i` for item `i`
produces wrong filtered results that are **loud on goodreads and silent on
arXiv** — it showed up only as `cos(query, target)` falling from 0.99 to 0.62
([H §10](../../evaluation/golden/README.md)). Oracle blobs published alongside
the embeddings were re-used where the fingerprint matched, and the one oracle
that was rebuilt (arXiv `c0_maincat`) was cross-checked against the blob the
baseline had used: `torch.equal` on both `[10000, 1000]` int64 top-K tensors,
**exactly equal** ([H §12.1](../validation.md#harness-gates)).

Ground truth is our own exact filtered fp32 full-scan oracle, keyed by a content
fingerprint of the inputs. Where a dataset ships its own filtered ground truth
we checked ours against it: on YFCC-10M our oracle reproduced the shipped
`GT.public.ibin` on **100,000 / 100,000** queries, id-exact, with
`max_abs_distance_error = 0.0` ([roadmap E1](../validation.md#datasets)).

## 8. Reproducing the environment

```bash
uv sync --extra official --all-packages   # workspace + Meta's ops; --all-packages keeps pytest
uv run --directory retrieve pytest tests/ -x -q             # library suite, GPU
cd evaluation && CUDA_VISIBLE_DEVICES="" uv run pytest tests/ -q   # harness suite, CPU-only by construction
```

Three facts that bite anyone reproducing on a similar box, all verified here:
the harness CPU suite must run with `CUDA_VISIBLE_DEVICES=""` because three of
its tests assert CPU-only behaviour in their own text; concurrent GPU jobs each
need their own `TORCHINDUCTOR_CACHE_DIR`, because the shared default is not
invalidated when a Triton host wrapper's Python source changes, and a job will
otherwise silently measure the previous kernel; and GPU work must be serialized
(`flock`) or the latency columns are meaningless.
