# V-PROF3: three one-cell profiles from EXHIBITS run 1

Roadmap V-PROF3, pod 1, campaign-v2.1. Each cell is built as `bench run` builds it (`load_matrix`,
`sweep_assets`, `build_module`, `graph_callable`, the rotating pool); k 100, seed 0. NOT CITABLE.

| item | cell | variants |
|---|---|---|
| a | goodreads `codesign`, official bloom `c0_genre`, `n_lists` 1024, `n_probe` 32, bs 16 | `bloom_path` partial vs full (C5) |
| b | goodreads-synth `synth`, V1 clause `p01`, bs 1 | Triton vs `torch.compile(max-autotune)` (C3) |
| c0001, c1 | goodreads-synth `synth`, V1 Triton clause `p0001` / `p1`, bs 16 | eager vs graph replay |
| ac0001, ac1 | arxiv-synth `synth` (3 M), the same | eager vs graph replay |

| file | what |
|---|---|
| [`prof3.py`](prof3.py) | `profile ITEM VARIANT OUT`: 20 calls in one sentinel-bracketed profiler session (padded retry as `measure.profile_once`), writes per-call kernel table, CUDA API counts (launches, graph launches, memcpy, malloc, syncs), peak scratch and a chrome trace; `time ITEM OUT`: both variants in one process, interleaved (`measure.latency_group`; eager vs graph alternates whole windows) |
| [`v-prof3.sh`](v-prof3.sh) | the driver (on [`../v-pod1-run/common.sh`](../v-pod1-run/common.sh)): oracles, twelve profile processes (one profiler session per process), six timing processes |
