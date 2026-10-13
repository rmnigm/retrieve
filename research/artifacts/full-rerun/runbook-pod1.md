# FULL-RERUN runbook: pod 1 (host e75980, one A100-SXM4-80GB)

The plan is [`recipe.yaml`](recipe.yaml), the why is in the [README](README.md), and pod e has its own
[runbook](runbook-pode.md). Pod 1 runs these families, each on this pod only, because comparisons are read inside one pod:
- PubMed: curves and timed.
- LAION 1 / 3 / 10 M: curves and timed.
- Every N-sweep leg: C1 and C5 across N.
- The LAION 30 M final-pass suites.
- Exact LiNR on PubMed and LAION.
- The picks (`PICKS_POD=pod1`).

**Never print secrets** (AGENTS.md rule 9): never `cat` / `grep` / `head` an env or secrets file, and never run
`env` / `printenv` / `set` / `declare -x`. Load a token file into the environment with `set -a; . FILE; set +a`, which echoes
nothing.

## 0. Preconditions
- The tag `final-freeze` is on origin (the controller sets it).
- No other job holds `/scratch/gpu0.lock`. Check with `nvidia-smi` showing no process, and `flock -n /scratch/gpu0.lock true`
  returning 0.
- Disk: `/` and `/data` each have more than 60 GB free (`df -h / /data /scratch`). The run writes about 30 GB of `_parity`
  spills at its peak, deleted after each leg's upload.
- Pod 1 reaches pod e over ssh (`rp-a100-x1-e`). Pod e does not reach pod 1.

## 1. Tree and venv at the tag (once per pod)
```bash
git -C /workspace/retrieve fetch origin --tags
git -C /workspace/retrieve worktree add --detach /scratch/wt/fr final-freeze
ln -s /data /scratch/wt/fr/evaluation/data
export TREE=/scratch/wt/fr VENV=/venvs/fr CFG=/scratch/fr/config FR=/scratch/fr LOCK=/scratch/gpu0.lock PICKS_POD=pod1
export UV_PROJECT_ENVIRONMENT=$VENV UV_CACHE_DIR=/scratch/uvcache/fr TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda
mkdir -p $FR
cd $TREE && uv sync --frozen --all-packages --all-groups --extra official     # official-fork's _C.so built here, from the tag
scripts/build_official_o3.sh                                                     # Meta's code as published, -O3
$VENV/bin/python - <<'PY' | tee $FR/build.txt                                    # the .so bytes of this pod
import hashlib, importlib.util, pathlib, retrieve, bench
for pkg in ("silvertorch_fork", "silvertorch"):
    d = pathlib.Path(next(iter(importlib.util.find_spec(pkg).submodule_search_locations)))
    for so in sorted(d.glob("*.so")):
        print(pkg, so, hashlib.sha256(so.read_bytes()).hexdigest())
print("retrieve", retrieve.__file__, "bench", bench.__file__)                  # both under /scratch/wt/fr
PY
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $FR/build.txt
git -C $TREE rev-parse HEAD | tee -a $FR/build.txt
```
`UV_CACHE_DIR` is a new, empty directory, so uv builds official-fork from the tag's sources and does not reuse a cached wheel.
Pod 1's `.so` sha256 differs from pod e's: the build is not byte-reproducible, and pod 1's driver is 570.195.03 against pod e's
580.126.16. Every record carries its own `env.official_fork_build.so_sha256`. This is why each comparison family stays on one
pod.

## 2. Data (CPU, before the queue)
```bash
cd $TREE/evaluation
# oracle blobs are rebuilt at the tag (no reuse): set the old ones aside, reversibly
for d in /data/pubmed-medcpt-dot /data/laion1m /data/laion3m /data/laion10m /data/laion30m /data/arxiv-papers; do
  for g in $d/gt_d*; do [ -d "$g" ] && mv "$g" "$g.pre-fr"; done; done
# synthetic draws (seed 20261008 + d); PubMed's dot build has none yet
for d in 0 1 2; do $VENV/bin/python -m eval_datasets.cli synth-filter --dataset pubmed-synth --draw $d; done
for d in 1 2; do $VENV/bin/python -m eval_datasets.cli synth-filter --dataset laion30m-synth --draw $d; done
for n in 1 3 10; do $VENV/bin/python -m eval_datasets.cli subset laion30m laion${n}m --n-items ${n}000000 --synth-only \
  --config-dir $FR/subset-configs; done                                         # copies draws 1, 2 onto the nested subsets
```
- The draw configs (`<ds>-synth-s<d>.yaml`) are in the tree. `--config-dir $FR/subset-configs` keeps the subset writer from
  touching the tree.
- `synth-filter --dataset pubmed-synth` resolves the data dir from `pubmed-synth.yaml` (`data/pubmed-medcpt-dot`).

Copy PubMed (and the LAION subsets, as spares) to pod e, checked by sha256:
```bash
for d in pubmed-medcpt-dot laion1m laion3m laion10m; do
  rsync -a --exclude 'gt_d*' /data/$d/ rp-a100-x1-e:/data/$d/
  (cd /data/$d && find . -path './gt_d*' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum) > $FR/sha-$d-pod1.txt
  ssh rp-a100-x1-e "cd /data/$d && find . -path './gt_d*' -prune -o -type f -print0 | sort -z | xargs -0 sha256sum" > $FR/sha-$d-pode.txt
  cmp $FR/sha-$d-pod1.txt $FR/sha-$d-pode.txt && echo "COPY OK $d $(wc -l < $FR/sha-$d-pod1.txt) files"
done
```
Tell pod e's runner `COPY OK` for each, with the file counts. Pod e builds its own PubMed synth draws from the same seed;
the sha check of those is in pod e's runbook.

Config and checks:
```bash
$VENV/bin/python ../research/artifacts/full-rerun/prep_config.py $TREE $CFG
$VENV/bin/python ../research/artifacts/full-rerun/check_suites.py $CFG $CFG/suites.yaml | tail -3     # 0 problem(s)
for d in pubmed pubmed-synth pubmed-synth-s1 pubmed-synth-s2 laion1m laion3m laion10m laion30m laion1m-synth laion3m-synth \
         laion10m-synth laion30m-synth laion10m-synth-s1 laion10m-synth-s2 arxiv-synth; do
  $VENV/bin/python -m bench.cli check --dataset $d --config-dir $CFG; done                          # every line: ok
```

## 3. Launch (one sequential driver for the GPU)
```bash
set -a; . /etc/retrieve-pod.env; set +a          # the Hub token for bench upload / fetch (not printed)
cd $TREE/evaluation
nohup setsid bash ../research/artifacts/full-rerun/queue.sh pod1 > $FR/queue-pod1.out 2>&1 &
```
`queue.sh` runs these in order, for each seed: seed 0, then 1, then 2.
1. **Stage 1:** the curves, the synth curves of draw s, the V3 pools; at seed 0 also exact LiNR.
2. **Mark:** it uploads `fr/marks/stage1-s<s>-pod1`.
3. **Picks (seed 0 only):** it runs final-pass suites until pod e's stage-1 mark is on the Hub, then `make_picks.sh` uploads
   `fr/picks`.
4. **Timed bloom.**
5. **Untimed steps.**
6. **The rest of the final pass.**
7. **The META-FORK grid** (seed 0 only).
8. **The clause reference row.**

A leg's records go to `fr/<leg>-pod1-s<seed>`, with the manifest sha appended to `$FR/manifests.tsv`. Every leg begins with
the resolved-arm check: no LiNR bloom arm, SilverTorch bloom `recheck: false`, no YFCC, no V2 bs 256 on PubMed. A failed check
means the leg is not run.

## 4. Watching
- `tail -f $FR/queue-pod1.log`: one line per step.
- `tail $FR/out/<leg>-pod1-s<seed>/logs/driver.log`: what the current leg is doing.
- `ls $FR/out/*/logs/FAILED`: anything that failed. A failure never stops the queue. That leg gets no `.done` and is rerun on
  a restart.
- `nvidia-smi`: GPU 0 should sit near 100 % except in the CPU gaps (data load, `bench check`, the oracle load at 10-30 M:
  minutes).

## 5. Kill and retry
- **Automatic** (`run_leg.sh`):
  - A failed `bench run` is retried once with `--resume`, so finished cells are skipped.
  - A CUDA out-of-memory failure is retried one algo per process. Those cells are then not interleaved with each other, which
    their records show.
  - A run past `MAXH` (12 h) is killed and not retried. Its unfinished cells are reported, never quoted.
- **A hang:** no new record line for 60 min while the GPU is at 0 % outside a CPU gap. Kill the `bench` process
  (`pkill -f "bench.cli run"`). The leg's retry takes over, and `flock` releases with the process.
- **Stop the pod:** `pkill -f queue.sh; pkill -f "bench.cli"`. Resume later with
  `nohup setsid bash ../research/artifacts/full-rerun/queue.sh pod1 <seed> ...`. Legs with `.done` are skipped, and cells
  already `ok` at the tag are skipped (`--resume`).
- **Never** edit the tree, rebuild the venv, or move data under a running queue. A fix to code is a new tag and a rerun of
  what it touches (the controller decides).
- **When the credit runs out,** the last finished seed is a complete set. Report the last mark (`fr/marks/seed<s>-pod1`) and
  `$FR/manifests.tsv`.
