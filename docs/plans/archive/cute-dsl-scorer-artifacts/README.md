# cute-dsl-scorer — session artifacts

Raw scripts and outputs behind [cute-dsl-scorer.md](../cute-dsl-scorer.md) (2026-09-02, A100).
Not part of the package or the test suite; kept so the numbers in the plan's §4/§5 can be
re-derived. Run scripts from `retrieve/` with `uv run python <script>`; some hard-code the
original scratchpad path in their output arguments — adjust before rerunning.

| dir | what |
|---|---|
| `spike/` | WP-0 CuTe DSL feasibility micro-kernels (`spike01`–`spike12`) and `FINDINGS.md` |
| `wp1/` | implementer self-check vs the C++ backend, SASS notes, D=128 masked-scorer PTX/SASS dump |
| `wp3/` | read-only review (`REVIEW.md`) and its edge-case scripts |
| `wp4/` | tune sweeps (JSON), shared-input head-to-head, kernel-only profiler split, host-overhead trims |
| `wp5/` | size accounting (`SIZE.md`, `count.py`) |
| `wp6/` | eager vs torch.compile vs CUDA-graph replay (`graphs.py`, `graphs.json`, `graphs.md`) |
