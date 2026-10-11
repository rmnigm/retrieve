# H-PROFILE: GPU checks

The three GPU checks behind the H-PROFILE row in [validation](../../../validation.md#harness-gates).
They run on any pod with one A100 free; no dataset is needed. Total ≈ 15 min of GPU, one job at a time.
The h2h `--profile` rerun is not part of this: it is folded into H2H-FINAL at `campaign-v2.1`, which
uses the normal h2h driver with this fix.

## Setup

```bash
git -C /workspace/retrieve fetch origin
git -C /workspace/retrieve worktree add /scratch/wt/h-profile origin/dev/h-profile   # or the merged staging
WT=/scratch/wt/h-profile
OUT=/scratch/h-profile/gpu-checks
mkdir -p "$OUT" /scratch/inductor/h-profile
export CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/h-profile HF_HOME=/scratch/hf
# a script run by path imports the venv's installed checkout, not the worktree: pin both packages
export PYTHONPATH=$WT/evaluation:$WT/retrieve/src
LOCK="flock /scratch/gpu0.lock"   # the pod's GPU lock, if the GPU is shared
cd "$WT/evaluation"
/venvs/retrieve/bin/python -m bench.cli env | /venvs/retrieve/bin/python -c 'import json,sys; print(json.load(sys.stdin)["code_version"])' \
  | tee "$OUT/code_version.txt"     # f01255f1… (campaign-v2.1); the fix does not move it
```

## 1. Drift diagnostic, the unfixed profiler body (≈ 6 min)

```bash
$LOCK /venvs/retrieve/bin/python ../docs/artifacts/campaign-v2/h-profile/repro.py 30 20000 \
  > "$OUT/repro-drift.log" 2>&1
```

The header line must show `$WT/evaluation/bench/measure.py` and `$WT/retrieve/src/retrieve/__init__.py`.
Expected (as on pod b): `key_averages_cuda` and `raw_device` fall by about one per session (bloom
30 → ≈ 1, none 16 → 0 by session ≈ 15); `lag_us` stays ≈ 6 µs. The new number is
`skew_us sync_end`: the profiler's timestamp for the end of the closing `cudaDeviceSynchronize`, minus
the wall clock just after it. If it grows across sessions, the window-vs-event drift hypothesis holds.
If it stays flat while the counts fall, the hypothesis is wrong. Report that; the fix still either
records both sentinels or raises.

## 2. The fixed `profile_once` (≈ 6 min)

```bash
$LOCK /venvs/retrieve/bin/python ../docs/artifacts/campaign-v2/h-profile/repro.py 30 20000 fixed \
  > "$OUT/repro-fixed.log" 2>&1
```

Expected: `fixed_kernels` and `calls` constant over all 30 sessions per arm (bloom ≈ 23 kernels /
≈ 30 calls, none ≈ 14 / ≈ 16, sentinels excluded). `s` is the wall time of one `profile_once`. Above
≈ 0.02 s means a padded retry was needed; note the session it starts at. A `RuntimeError: profile_once:
… sentinel kernels recorded at pad 5.0 s` means the pad does not cure the loss: stop and report it.

## 3. The GPU test (< 1 min)

```bash
$LOCK /venvs/retrieve/bin/python -m pytest tests/bench/test_measure.py -q -rs -p no:cacheprovider \
  -k profile_once > "$OUT/pytest-gpu.log" 2>&1
```

Expected: `test_profile_once_records_the_triton_bloom_kernels` passes (1 passed, 1 skipped: the
CPU-only `test_profile_once_is_empty_without_cuda` skips on a GPU).

## Upload

```bash
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv > "$OUT/gpu.csv"
/venvs/retrieve/bin/python -m bench.cli upload --results "$OUT" --path-in-repo artifacts/h-profile/gpu-checks --verify
```

Put the manifest sha256 in the chain note. The pod-b logs of the original reproduction are at
`artifacts/h-profile/pod-b` ([hub index](../../hub-index.md)).
