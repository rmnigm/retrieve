---
chain: "benchmarking-research"
branch: "main"
nextStep: "Orchestrator triage: dispatch a small CPU-only harness-hardening worker for the low-effort methodology items (continuous clock sampling, trimmed mean, sub-10us unreliable flag, CUDA-event sync-in-loop audit, L2-flush disclosure); add citations to docs/paper/ for Meta's public numbers and the ANN-benchmark trustworthiness critiques; note the medium/large kernel-technique items as background reading, not scheduled work."
created: "2026-09-26T09:00:00Z"
---

# Survey: open-source repro/benchmarking libraries and kernel techniques

Research task (sonnet, web search), dispatched per the user's request to
look at what open-source reproducibility/benchmarking libraries and
kernel techniques this project could learn from, after the kernel-opt
pass landed. Pure research, no code touched. Full findings below,
copied from the subagent's report verbatim.

## Methodology / benchmarking-practice findings

1. **Clock-throttle-aware benchmarking playbook** (standardkernel.com,
   "This Kernel Was Faster Yesterday"): sample `nvidia-smi
   --query-gpu=clocks.sm` continuously during a run, not once; use a
   **trimmed mean** (drop top/bottom 10%) alongside median/IQR; run
   500+ timed trials after 100-200 warmup iters; never `sleep()`
   between trials on shared GPUs; don't trust measurements under
   ~10us; document driver/CUDA version, power cap, PCIe-vs-SXM.
   **Low-effort, directly actionable** — the harness already samples
   `sm_mhz` and has `unstable`; continuous (not single-point) sampling,
   a trimmed-mean stat, and a sub-10us "not reliable" flag would close
   a real gap. Candidates: `docs/decisions.md#harness`,
   `docs/system/testing.md`.
2. **CUDA-event timing loop correctness**: Triton's
   `triton.testing.do_bench` and FlashInfer's
   `bench_gpu_time_with_cuda_event` sync once outside the loop, not
   per-iteration (flashinfer-bench GitHub issue #195: per-iteration
   `torch.cuda.synchronize()` inflates measured time for fast kernels).
   **Low-effort, worth an audit** of `evaluation/bench/measure.py`'s
   timing windows.
3. **L2 cache-state disclosure**: `triton.testing.do_bench` flushes L2
   via a large buffer write between reps by default; cold- vs
   warm-cache measurement is a methodology choice that must be stated
   (large gap for bandwidth-bound ops, small for compute-bound).
   **Low-effort**: audit whether the harness's kernel microbenchmarks
   flush L2, document the choice next to `sm_mhz`/`unstable` in
   validation.md.
4. **Recall-bucketed / Pareto reporting** (NVIDIA cuVS Bench
   methodology): compare indexes only within recall buckets (80-89%,
   90-94%, 95-98%, 99%+), build a Pareto frontier per bucket, fix scope
   (dataset, metric, k, batch, filters, hardware, concurrency) before
   comparing. **Medium effort**: the campaign harness could adopt
   bucketed reporting instead of point comparisons — the de facto
   ANN-community standard (also ann-benchmarks/erikbern). cuVS's own
   methodology doc does *not* address measurement noise/statistical
   significance — a gap this harness already does better on.
5. **Trustworthiness critiques of published ANN benchmarks**: "Towards
   Robustness: A Critique of Current Vector Database Assessments"
   (arXiv 2507.00379) and a YDB.tech blog post document how commonly-
   cited ANN benchmark numbers are non-reproducible or gamed.
   **Low-effort, citable as-is** for
   `docs/paper/provenance-and-disclosure.md`, justifying this project's
   gating rigor (CLAUDE.md rule 2).
6. **Adaptive iteration counts**: `torch.utils.benchmark.Timer
   .blocked_autorange()` picks iteration count adaptively to hit a
   timing budget, reports p50/p99 by default. **Low effort, optional**
   — not a correctness issue, just less manual tuning than fixed
   windows.
7. **BEIR/MTEB**: text-embedding-quality suites, not GPU/ANN-infra
   benchmarks. **Not applicable**, confirmed out of scope.

## Kernel-technique findings

1. **Official SilverTorch repo is public**
   (`github.com/meta-recsys/silvertorch`, matches `backend="official"`)
   plus an Engineering-at-Meta blog post (May 2026) with Meta's own
   claimed numbers: fused Int8 ANN kernel 2.2-14.7x faster than
   Faiss-GPU, bloom index 291-523x faster than a CPU inverted index,
   probe-then-filter co-design cutting filter compute a further 30x.
   **Low effort, high value**: citable target/comparison figures (as
   Meta's own claims, not independently verified) for
   `docs/paper/official-vs-reimplementation.md`.
2. **Tunable-vectorization GPU Bloom filter**: "Optimizing Bloom
   Filters on Modern GPUs" (arXiv 2512.15595, ICS 2026). Per-
   architecture tunable vectorization width instead of a fixed SIMD
   bit-block layout; >92% of the random-access memory bound on B200,
   15.4x lookup / 11.35x construction over prior GPU baselines.
   **Large undertaking** as an implementation change; relevant
   background/citation for the transposed bloom-index kernel
   (`docs/system/filtering.md` or the paper's kernel-technique section).
3. **Cuckoo filter on GPU**: "Cuckoo-GPU: Accelerating Cuckoo Filters
   on Modern GPUs" (arXiv 2603.15486) — alternative predicate-pushdown
   structure, different space/FP tradeoffs, parallel multi-slot
   probing. **Not urgent** — worth a one-line mention as an alternative
   to benchmark against, not implement.
4. **Bucket-based coalesced-access layout for filtered graph search**:
   GRAB-ANNS (arXiv 2604.16402) converts range/filter predicates into
   bucket selection for coalesced SIMT access, dense intra-bucket +
   sparse inter-bucket edges. **Large undertaking** (different index
   topology than IVF); relevant citation for the compact CSR-like
   probe layout already done (same goal, different mechanism).
5. **Warp-ballot candidate selection in IVF kernels**: a pattern across
   recent GPU-IVF kNN work — replace per-thread sequential top-k
   bookkeeping with `__ballot_sync`-based 32-way parallel candidate
   reduction. **Medium effort, concretely actionable** if the probe
   kernel currently does per-thread/per-block top-k without warp-ballot
   reduction — worth a code check against `docs/system/kernels.md`.
6. **Jasper**: "GPU-Accelerated ANNS: Quantized for Speed, Built for
   Change" (arXiv 2601.07048, PVLDB) — overlaps compute/memory for
   data-dependent graph access, batched updates without full rebuild.
   **Large undertaking, likely out of roadmap scope** (this project
   targets static-index gates); background only if incremental-update
   work (G-e's live-update-api) is ever scheduled.

## Negative finding

No public code or third-party reproduction of **LiNR** exists anywhere
(only the CIKM 2024 paper, arXiv 2407.13218, and a LinkedIn blog post).
This project's LiNR reimplementation has no existing open-source
precedent to compare against or borrow from.
