# PubMed 10M × 768 SilverTorch build after the quantization fix

Staging's harness (`8a6096b`, `evaluation/` copied out) run against `dev/kernel-opt`'s library,
the cell E2 reported OOMing (`bench run --dataset pubmed --dim 768 --suite filter --filter-kind
clause --sweep c0_mesh --algo silvertorch --skip-perf --mode eager`), 2026-09-26, A100 80 GB.

- **Build no longer OOMs, either backend.** `pubmed_mem.py` (same inputs, per-phase
  `max_memory_allocated`): fp32 item table 30.1 GiB allocated before the build; k-means +0.3 GiB
  transient; quantize +7.7 GiB transient of which 7.2 GiB is the int8 table it keeps; filter
  buffers +1.5 GiB. Allocated peak ≈ 38 GiB. `nvidia-smi` shows 75.0 GB *reserved* (the caching
  allocator holding freed input-loading blocks), 44.5 GB steady (`mem_*.txt`, 1 s samples).
- **official**: both `n_probe` cells run end to end (`pubmed-d768.jsonl`): build 23.0 s, index
  9006 MiB, recall_oracle@100 0.6640 / 0.7085, held-out recall@100 0.8415 / 0.8799 (n = 8428);
  `partial` only for `skip_perf`. Not validated as campaign numbers.
- **triton**: the build completes; the first forward stops at the op boundary, `ValueError: D=768
  must be a power of two` (`stage: quality`). The probe scorer's `tl.arange(0, D)` cannot take
  D = 768; before the boundary check this was a Triton `CompilationError`. Not an OOM, a separate
  blocker, hidden until now behind it. The same bound applies to `fused_masked_knn_topk` (LiNR V2,
  V3 stage 2) at D = 768 and to OPORP at its default `k_bits = D` (W = 12).
