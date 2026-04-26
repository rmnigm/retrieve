# Random-vector benchmarks

```
uv run pytest tests/bench --bench --bench-output bench.md
```

Adds the `--bench` flag to collect/run the cells (they are skipped without
it). Each cell measures Triton vs the closest pure-PyTorch baseline on the
same random inputs using `triton.testing.do_bench` (CUDA events + L2 flush
+ autorange) and writes a Markdown table per suite to `bench.md`.

Useful options:

- `--bench-rep 200` &nbsp; ms target per `do_bench` call (default 200; lower
  = faster runs, noisier numbers).
- `--bench-csv out.csv` &nbsp; also dump a CSV.
- `--profile` &nbsp; on the slowest cell of each suite, run
  `torch.profiler.profile(record_shapes=True)` and write a Perfetto-loadable
  chrome trace to `traces/`. Stdout gets the
  `key_averages().table(sort_by="cuda_time_total")` summary.

## Reading the verdict column

For every Triton row, the table reports a `verdict` against the same cell's
torch row:

- ✅ &nbsp; Triton median ≤ 0.95× torch median — kernel is pulling its weight
- ➖ &nbsp; within ±5% — neutral
- ❌ &nbsp; Triton ≥ 1.05× torch — kernel is slower; if this persists at large
  `N`, the kernel is not worth keeping in its current form

Correctness is *always* asserted (`✓` column). The bench will fail the cell
if Triton's top-K diverges from the reference, regardless of speed.

## When `torch.profiler` isn't enough

For "this kernel is slow but I don't know why", drop down to NVIDIA's tools:

```
# timeline + GPU-side metrics
nsys profile --stats=true uv run python -c "import tests.bench.bench_kernels as m; m.test_bench_fused_masked_knn_topk(...)"

# per-kernel SASS / occupancy / memory throughput
ncu --set full --target-processes all uv run python <one-call-script>
```

`ncu --set full` is heavyweight; usually `ncu --set basic` on a single
captured kernel name (`--kernel-name regex:_fused_.*_kernel`) is enough to
spot poor occupancy or under-utilized memory bandwidth.
