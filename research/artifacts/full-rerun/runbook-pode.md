# FULL-RERUN runbook: pod e (a100-x1-e, one A100-SXM4-80GB)

The plan is [`recipe.yaml`](recipe.yaml), the why is in the [README](README.md), and pod 1 has its own
[runbook](runbook-pod1.md). Pod e runs these families, each on this pod only, because comparisons are read inside one pod:
- goodreads and arXiv: curves and timed, synth draws included.
- LAION 30 M: curves and timed.
- Yambda.
- The V3 pools of goodreads, arXiv and PubMed.
- The bloom width grid.
- The PubMed synthetic crossover (exact LiNR).
- The goodreads / arXiv final-pass suites.

Pod 1 makes the picks; this queue waits for `fr/picks` on the Hub.

**Never print secrets** (AGENTS.md rule 9): never `cat` / `grep` / `head` an env or secrets file, and never run
`env` / `printenv` / `set` / `declare -x`. Load a token file with `set -a; . FILE; set +a`.

## 0. Preconditions
- The tag `final-freeze` is on origin.
- `/scratch/gpu0.lock` is free.
- `/data` has more than 60 GB free.
- Pod e cannot reach pod 1, so pod 1 pushes the PubMed and LAION-subset copies here (pod 1's runbook § 2). Wait for its
  `COPY OK` lines before § 2 below.

## 1. Tree and venv at the tag (once)
```bash
git -C /workspace/retrieve fetch origin --tags
git -C /workspace/retrieve worktree add --detach /scratch/wt/fr final-freeze
ln -s /data /scratch/wt/fr/evaluation/data
export TREE=/scratch/wt/fr VENV=/venvs/fr CFG=/scratch/fr/config FR=/scratch/fr LOCK=/scratch/gpu0.lock PICKS_POD=pod1
export UV_PROJECT_ENVIRONMENT=$VENV UV_CACHE_DIR=/scratch/uvcache/fr TORCH_CUDA_ARCH_LIST="8.0" CUDA_HOME=/usr/local/cuda
mkdir -p $FR
cd $TREE && uv sync --frozen --all-packages --all-groups --extra official
scripts/build_official_o3.sh
```
Then record the build exactly as in pod 1's runbook § 1, into `$FR/build.txt`: the `.so` sha256 of `silvertorch_fork` and
`silvertorch`, where `retrieve` / `bench` import from, the driver and the tag's commit.

This pod's `.so` bytes and driver (580.126.16) differ from pod 1's. No comparison family is split across the pods.

## 2. Data (CPU, before the queue)
```bash
cd $TREE/evaluation
for d in /data/goodreads-work-id /data/arxiv-papers /data/laion30m /data/pubmed-medcpt-dot; do
  for g in $d/gt_d*; do [ -d "$g" ] && mv "$g" "$g.pre-fr"; done; done        # oracle blobs rebuilt at the tag, reversibly
for d in 1 2; do
  for ds in goodreads-synth arxiv-synth laion30m-synth; do $VENV/bin/python -m eval_datasets.cli synth-filter --dataset $ds --draw $d; done
  $VENV/bin/python -m eval_datasets.cli synth-filter --dataset arxiv-corr-synth --correlated --draw $d   # CPU k-means, minutes
done
$VENV/bin/python -m eval_datasets.cli synth-filter --dataset pubmed-synth --draw 0                    # the crossover leg's draw
sha256sum /data/pubmed-medcpt-dot/item_attrs_synth.pt /data/laion30m/item_attrs_synth_s1.pt /data/laion30m/item_attrs_synth_s2.pt
```
The draws are deterministic, seed 20261008 + d. Post the three sha256 lines to pod 1's runner. Pod 1 builds the same files
and must print the same hashes. A mismatch stops both queues: the controller decides.

Check that the synth sidecars match the seeds:
```bash
$VENV/bin/python -c "import json,sys; [print(p, json.load(open(p))['seed']) for p in sys.argv[1:]]" \
  /data/goodreads-work-id/synth_filter_s1.json /data/arxiv-papers/synth_filter_s2.json /data/laion30m/synth_filter_s1.json
```
The expected seeds are 20261009, 20261010 and 20261009.

Config and checks:
```bash
$VENV/bin/python ../research/artifacts/full-rerun/prep_config.py $TREE $CFG
$VENV/bin/python ../research/artifacts/full-rerun/check_suites.py $CFG $CFG/suites.yaml | tail -3     # 0 problem(s)
for d in goodreads arxiv pubmed laion30m yambda-500m goodreads-synth arxiv-synth arxiv-corr-synth laion30m-synth pubmed-synth \
         goodreads-synth-s1 goodreads-synth-s2 arxiv-synth-s1 arxiv-synth-s2 arxiv-corr-synth-s1 arxiv-corr-synth-s2 \
         laion30m-synth-s1 laion30m-synth-s2; do
  $VENV/bin/python -m bench.cli check --dataset $d --config-dir $CFG; done                          # every line: ok
```

## 3. Launch
```bash
set -a; . /etc/retrieve-pod.env; set +a
cd $TREE/evaluation
nohup setsid bash ../research/artifacts/full-rerun/queue.sh pode > $FR/queue-pode.out 2>&1 &
```
The seed-major order is the same as pod 1's: stage 1, then the mark `fr/marks/stage1-s<s>-pode`, then (seed 0) final-pass
suites until `fr/picks` is on the Hub, fetched to `$FR/picks.yaml`. After that come timed bloom, the untimed steps
(fp_adapted at seed 0 here), the rest of the final pass, the META-FORK grid (seed 0), and the clause reference row.

Uploads go to `fr/<leg>-pode-s<seed>`, with manifest shas in `$FR/manifests.tsv`. Every leg starts with the resolved-arm check
(no LiNR bloom, recheck false, no YFCC).

## 4. Watching, kill and retry
Watching and kill / retry are as in pod 1's runbook § 4-5, with `pode` in place of `pod1`.
- **Yambda:** `fr-yambda` scores 10,000 users; its encode cache sits beside each checkpoint (about 5 GB a dim) and is reused.
- **LAION 30 M timed:** an OOM is expected on the LiNR interleave group, as in B2. The automatic per-algo retry handles it, and
  those LiNR cells are then not interleaved, which their records show.
