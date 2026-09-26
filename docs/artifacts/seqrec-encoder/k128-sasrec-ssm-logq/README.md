# k128 memory probe: OOM with one table (KuaiRand-27K d128, `reuse_item_embeddings=true`, `train_on_val=true`)

H100 80GB HBM3 (79.18 GiB usable), 2026-09-26 22:35:50-22:37:22 UTC. `probe.sh`, `probe_oom.txt`.
Not yet validated. Per instruction 011218745: "on OOM, stop and note it". k128-refit was not launched.

**Result:** `torch.OutOfMemoryError` inside `loss.backward()` (`train.py` line 217) on the probe's
**second** step, the first after AdamW has allocated its state. "Tried to allocate 15.28 GiB ... 14.71 GiB
is free ... 62.98 GiB is allocated by PyTorch".

**Mechanism (arithmetic, consistent with the numbers above):** one 32,038,726 x 128 fp32 table is
15.28 GiB.
- weight + AdamW exp_avg + exp_avg_sq = 3 x 15.28 = 45.8 GiB, resident from step 0 on;
- the shared table receives two dense gradients in backward, one from the input lookup
  (`item_embedding`) and one from the output scoring (`scoring_table` / candidate gather), and autograd
  holds both at full size before summing them: 2 x 15.28 = 30.6 GiB;
- plus ~1.9 GiB of activations: ~78 GiB needed against 79.18 GiB, with 62.98 allocated when the
  second 15.28 GiB gradient was requested.

The d64 two-table run fits at 62.7 GB because each d64 table is half the size and gets one gradient.
Getting d128 to fit needs memory engineering (sparse or row-wise gradients or optimizer, a bf16 table,
or a smaller catalog), which the instruction reserves for the user. Nothing was tried.
