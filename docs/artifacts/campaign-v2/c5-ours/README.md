# C5-OURS — `bloom_path="full"` in our Triton SilverTorch (library option and gates)

Roadmap step C5-OURS (campaign-v2.5). C5's reversal comes from Meta's host-bound partial path. Our Triton SilverTorch
is the co-design by construction: the filter is fused into the probe scorer. This step adds the other arm inside our
own kernels, so C5 can compare like with like.
Current state: [validation](../../../validation.md) row *C5-OURS*; mechanism: [kernels](../../../system/kernels.md#bloom_full_mask--the-full-n-mask-bloom_pathfull);
harness: [evaluation](../../../system/evaluation.md) (`codesign`).
A100-SXM4-80GB (pod b, GPU 0). Before = dev/v2-fill 394d3e9 (the tree without the option) as package `retrieve_pre`
(`../st-ids/make_pkg.sh`). Raw outputs: Hub `artifacts/c5-ours/` ([hub-index](../../hub-index.md)).

## The option
`SilverTorch(bloom_path="full")` (bloom, triton or torch) works in two steps:
1. The new op `bloom_full_mask` ANDs the query's rows of the transposed bloom index over all N. The result is a packed
   `[B, ceil(N / 64)]` mask.
2. The unchanged `codesigned_probe_score_bloom` reads that mask as a one-row table per query (`qpos = [[b]]`), with the
   partial path's own pass-rate bound.

The keep set, the dot and the skip-gate decisions are the same, so `full` = `partial` bit for bit. The default
(`partial`) calls the same op with the same tensors as before.

The harness accepts `bloom_path` on `silvertorch / bloom / triton` too. The `codesign` suite gains a triton arm:
- 108 new cells, each carrying `bloom_path` in its key;
- the 108 official keys unchanged;
- partial and full of one backend paired per interleave group.

## Gates ([`phase.sh`](phase.sh))
- **Default path, SASS** ([`sass_layer.py`](sass_layer.py)): the layer's triton forward (none / bloom / exact, D 128 /
  768, bs 1 / 16) launches identical machine code before and after, **8 / 8 cubins**.
- **Bit-exact** ([`c5_gate.py`](c5_gate.py)): **72 / 72 rows**.
  - Setup: four single-value clauses at pass rates ≈ 0.001 / 0.01 / 0.14 / 1; n_lists 1024; n_probe {8, 32, 128} × bs
    {1, 16} × 4 query seeds.
  - 200 k d128, 2 M d128 and 1 M d768.
  - Every row had passing items (1-1,600 returned).

  | comparison | where | check |
  |---|---|---|
  | after's default vs before's | every row | ids + scores `torch.equal` |
  | `full` vs `partial` | every row | scores `torch.equal`, ids equal up to ties |
  | triton `full` vs torch-backend `partial` | 200 k | same as above |
- **Library suite:** **814 passed** on a fresh inductor dir. It includes
  [`test_bloom_full_mask.py`](../../../../retrieve/tests/parity/test_bloom_full_mask.py) (the op against its torch twin
  and the row-wise subset test, and the layer `full` = `partial` on triton and torch at D 64 / 768). It also includes the
  compile test's `bloom-full` cases: `reduce-overhead` replay = eager, 0 cudagraph skips, 0 graph breaks.
- **Harness suite:** 475 passed (CPU).

Timing (the codesign cells partial vs full, goodreads + arXiv, `--interleave`) runs after the campaign-v2.5 tag.
