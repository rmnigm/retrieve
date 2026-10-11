# L2 artifacts — `kmeanspp_timing.py`

Plan L §11's risk check on greedy k-means++ seeding. `kmeanspp_timing.py` (run under
`flock /workspace/gpu.lock` from `/workspace/wt/l2` with the L2 venv) times
`KMeans(init=...).fit` for both inits at the C4 gate size (N=200k, D=128, n_lists=1024;
seeding alone and a full 10-iteration fit) and at the plan's worst case (N=3M, D=128,
n_lists=8192; seeding alone), and writes `kmeanspp_timing.json`. SM clocks cannot be locked
here (CLAUDE.md rule 1): `sm_mhz` is the sampled clock; nothing in the JSON is a recorded
number, it decides only whether the k-means‖ variant was needed (it was not: 9.5 s at the
worst case against the plan's one-minute line).
