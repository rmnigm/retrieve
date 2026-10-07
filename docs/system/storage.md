---
title: storage
created: 2026-09-26
updated: 2026-10-07
type: entity
tags: [environment]
sources: [infra/runpod/]
confidence: medium
---

# Storage on the GPU pods

How the disks on a GPU pod behave, what may live on each, and the budget
every agent is expected to keep. `df` is not trustworthy on RunPod: the
measured numbers below come from sized writes and `du`, not from an
advertised capacity.

## What a pod gets

Pods are created by [`infra/runpod/pod.sh`](../../infra/runpod/pod.sh)
from the image in [`infra/runpod/Dockerfile`](../../infra/runpod/Dockerfile):

| | where | default |
|---|---|---|
| container disk (`/`) | ephemeral, recreated with the container | 200 GB (`pod.sh up --disk`) |
| `/workspace` | a pod volume, or a RunPod network volume with `--nv` (the repo then sits at `/workspace/<pod>/retrieve`) | 30 GB pod volume (`--volume`) |
| venv | `/venvs/retrieve` (`UV_PROJECT_ENVIRONMENT`), baked into the image | |

`pod.sh up --gpus N` gives one pod N GPUs, `--pods M` launches M pods. The
image creates `/data` and `/scratch/{campaigns,inductor,parity,hf,wt,tmp}`;
[`bashrc.sh`](../../infra/runpod/rootfs/opt/retrieve-pod/bashrc.sh) sets
`REPO_DIR=/workspace/retrieve`, `RETRIEVE_DATA_ROOT=/data`,
`HF_HOME=/scratch/hf` and `POD_STATE=/workspace/.pod-home`; bootstrap
clones the repo and installs the pre-commit hook. So the image's defaults
follow the placement rule below: only the repository and the pod's own
state live on `/workspace`.

## Measured on a network-volume pod

Measured on an A100-SXM4-80GB pod with a network volume (EUR-IS-1):

| | `/` (container overlay) | `/workspace` (network volume) |
|---|---|---|
| Backing | overlay2 on the host's XFS | MooseFS over the network |
| `df` says | 300 G total | 2.1 PB total, 618 TB free |
| Actually usable | **300 GB**, and `df`'s free figure is honest | **≈ 26 GB** |
| Measured write | 140 GiB resident in one go at **539 / 470 MB/s** | `dd` died at **18 GB**, `Disk quota exceeded` |
| Measured read | — | **≈ 52 MB/s** (5.3 GB dataset copy, 101 s) |
| Survives a pod restart | no | **yes** |

The overlay's ceiling is the container-disk size, enforced as an XFS
project quota, which XFS reports as the filesystem size (so `df /` is
honest). `/dev/shm` and `/run/nvidia-persistenced/socket` are **tmpfs —
RAM, not disk**; never use either as a data root. The host's own NVMe is
not exposed to the container at any writable path (no block devices,
`mknod` and `mount` denied), so the container disk is the ceiling.

## What lives where, and why

**`/workspace` — the git repository.** It is the only disk that survives a
pod restart, and its contents are pushed to `origin` anyway. The harness's
in-flight results tree (`results/` under `evaluation/`) lives inside the checkout for
that reason: it is gitignored, so it is not pushed, but a restart mid-campaign
does not lose the cells already written, and resume reads them locally.

**The container disk — everything large, on one condition.** Datasets,
worktrees that build things, inductor caches, campaign scratch, parity
spills.

> **The rule that makes an ephemeral disk safe:** everything on it must be
> **regenerable without human input** — a dataset re-fetchable from the
> Hub, a venv from one `uv sync`, a cache by re-running the job. If losing
> it would cost a decision, a measurement or an afternoon of someone's
> judgement, it does not belong there. Decisions and code belong in git
> and get pushed; records belong on the Hub (`bench upload`).

**The HF Hub — the archive.** Datasets live at `pinkmeme/eval-*` and
checkpoints alongside them ([checkpoints.md](checkpoints.md)); every
finished results tree (records, samples, `results.parquet`) and the raw
outputs behind documented findings live at `pinkmeme/eval-results`
([evaluation](evaluation.md#results-storage),
[hub-index.md](../artifacts/hub-index.md)). A results tree is finished —
and may be deleted locally — once `bench upload --verify` has passed.
Staging is **pull → use → prune → re-pull**, never hoard.

The private Hub quota is not exposed by the API; it measures between ~95.8
and 116 GB. Deleting a path frees no private quota until the repo's history
is squashed (`HfApi().super_squash_history`): usage after the results
cleanup is 95.76 GB, of which 84.09 GB are current files.

A layout that follows the rule on a network-volume pod:

```
/data/          RETRIEVE_DATA_ROOT — staged datasets, checkpoints, _raw/ ETL input
/venvs/         uv project environments (UV_PROJECT_ENVIRONMENT)
/scratch/
  ├── campaigns/  a results tree kept outside the checkout (`bench campaign --out`)
  ├── inductor/   TORCHINDUCTOR_CACHE_DIR, one subdir per concurrent job
  ├── parity/     parity spill .npz files (~600–680 MB per run)
  ├── hf/         HF_HOME for Hub download caches
  ├── wt/         worktrees that build or generate bulk artifacts
  └── tmp/        general scratch
```

The data root mirrors the layout in [datasets.md](datasets.md) and is
resolved by `evaluation/eval_datasets/hub.py:data_root()`.

### Environment

```bash
export RETRIEVE_DATA_ROOT=/data                           # the image default
export UV_PROJECT_ENVIRONMENT=/venvs/retrieve             # reuse; see the venv budget
export TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>    # private per job
export HF_HOME=/scratch/hf
```

`TORCHINDUCTOR_CACHE_DIR` must be **private per job** (one job per GPU, so per GPU on a multi-GPU pod): inductor's
on-disk FX cache does not invalidate when a `@triton_op` host wrapper's Python
source changes, so a shared `/tmp/torchinductor_root` silently serves stale
kernels ([reproduction-deviations.md](../paper/reproduction-deviations.md) D-7).
`/tmp/inductor-<job>` is on the same disk and works as well.

`rp-sync`, the pod's `uv sync` wrapper, sources `/opt/retrieve-pod/bashrc.sh`. That file
reloads the pod env file, which sets `UV_PROJECT_ENVIRONMENT=/venvs/retrieve`, so an exported
override does not survive.
- A git worktree with its own venv must sync directly:
  `UV_PROJECT_ENVIRONMENT=/venvs/<name> uv sync --all-packages --all-groups --extra official`.
- Otherwise it re-points the shared venv's editable installs at the worktree.
- Never print the pod env file to check this. It holds live tokens (AGENTS.md rule 9).

Memory is capped by the container's cgroup (`RUNPOD_MEM_GB`, 251 GB on the H100 pod), not by
what `free` reports (the host's ~2 TB). Size CPU-heavy ETL against the cgroup.

## The budget

The container disk is one pool: the data root, `/venvs`, `/scratch`, `/tmp`
and `/root` all draw on it. On the measured pod the base image took 15 GB,
`/root` 8.5 GB, and goodreads-d128 + arXiv-d128 5.3 GB.

**Datasets: one or two resident at a time.** Prune each after its cells
run rather than letting the set accumulate. Never stage the whole registry.

**Venvs: one shared environment.** Each is **≈ 7.6 GB** (torch + triton +
CUDA wheels), so one per worktree fills a disk fast. Every worktree points
at the image's:

```bash
export UV_PROJECT_ENVIRONMENT=/venvs/retrieve
uv run --no-sync --directory evaluation bench run ...
```

Create a second only when a concurrent worker genuinely needs different
packages (a different torch, an unmerged dependency), and delete it when
its branch merges.

### Cleanup

```bash
git worktree prune                      # after merging; then remove the dir
git worktree remove /scratch/wt/<name>
rm -rf /venvs/<name>                    # the merged worktree's environment
rm -rf /scratch/inductor/<job> /tmp/inductor-<job>
rm -rf /scratch/parity/*                # between campaign stages
uv cache prune                          # uv's own wheel/source cache
df -h /                                 # confirm
```

## The queued datasets

Size the pod's disks for these at `pod.sh up` (`--disk`, `--volume`):

- **PubMed, as the 10 M slice, streams.** The ETL consumes the
  download shard by shard. Planned peak is about 27 GB for the slice
  (69 GB for the full catalog). Measured on the staged copy: 17 GB of
  layout plus 0.4 GB of PMID lists kept in `_raw/`, and 57 min of
  download-bound `convert` with MEDLINE streaming beside it
  ([datasets](datasets.md#disk-budget-and-the-slice)).
- **OpenAlex (roadmap E5) only fits as a stream.** The works snapshot is 707 GB of
  parquet and is never landed: `openalex convert` reads 297 GB of projected
  columns over S3 (572 s at 64 workers) and stages the filtered, hash-sampled
  rows (16 GB). The 10 M catalog is ~23 GB (papers 6.3, fp16 items 15, attrs
  1.5); the superseded 15 M catalog it was resharded from is 34 GB more while
  it is kept ([datasets](datasets.md#budget)).

## Persistence

`/workspace` persists across pod restarts; the container disk does not.
A network volume is served from outside the pod and outlives it; the
container disk is RunPod's *container disk*, recreated with the container.
That asymmetry is the whole design: regenerable things on the big
ephemeral disk, irreplaceable things pushed off the box — code and prose to
`origin`, records and raw outputs to the Hub.

## See also

- [datasets.md](datasets.md) — the dataset layout under `RETRIEVE_DATA_ROOT` and the ETL commands.
- [checkpoints.md](checkpoints.md) — checkpoints and the HF Hub workflow.
- [evaluation.md](evaluation.md) — the harness and campaign records.
