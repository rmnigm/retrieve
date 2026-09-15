# Storage on the A100 box

How the disks on this machine actually behave, what may live on each, and the
budget every agent is expected to keep. Measured 2026-09-15 on RunPod pod
`5ajcbzrjau5z1f` (A100-SXM4-80GB, EUR-IS-1). `df` is not trustworthy here —
every number below came from a sized write or a `du`, not from an advertised
capacity.

## The two disks

| | `/` (container overlay) | `/workspace` (network volume) |
|---|---|---|
| Backing | overlay2 on the host's XFS (`ubuntu--vg-runpod--data`) | MooseFS over the network (`mfs#eur-is-1.runpod.net:9421`) |
| `df` says | 300 G total | 2.1 PB total, 618 TB free |
| Actually usable | **300 GB**, and `df`'s free figure is honest | **≈ 26 GB** — the petabyte is a lie |
| Measured write | 140 GiB resident in one go at **539 / 470 MB/s** | `dd` died at **18 GB**, `Disk quota exceeded` |
| Measured read | — | **≈ 52 MB/s** (5.3 GB dataset copy, 101 s) |
| Survives a pod restart | **almost certainly not** | **yes** |

The overlay's ceiling is a docker `--storage-opt size=300G` XFS *project* quota
(`prjquota` is in the host mount options, and XFS reports a project quota as the
filesystem size — which is why `df /` says 300 G and means it). It is enforced
the same way MooseFS enforces its 26 GB, but unlike MooseFS it advertises the
truth. A 60 GiB probe and then a further 80 GiB probe both completed, taking the
container to **242 G of 300 G used** with no surprise; both were deleted.

`/dev/shm` (58 GB) and `/run/nvidia-persistenced/socket` (101 GB) are **tmpfs —
RAM, not disk**. Never use either as a data root.

The host has far more storage — `lsblk` shows a 7 TB and a 14 TB NVMe, and the
XFS volume behind the overlay is 21 TB with 19 TB free — but **none of it is
exposed to this container at a writable path**. There are no block device nodes
in `/dev`, `mknod` returns `EPERM`, `mount` is denied (no `CAP_SYS_ADMIN` in the
bounding set), and the volume surfaces only as three file bind mounts
(`/etc/hosts`, `/etc/hostname`, `/etc/resolv.conf`). **300 GB is the ceiling.**

## What lives where, and why

**`/workspace` — the git repo and nothing else of size.** It is the only thing
on this box that survives a pod restart, and its contents are pushed to `origin`
anyway. `/workspace/retrieve` (509 MB) plus worktrees under `/workspace/wt`
(text only) plus `/workspace/gpu.lock`. That is the whole permitted list. It sat
at 6.7 GB of its ~26 GB quota after the 2026-09-15 worktree prune.

**The overlay — everything large, on one condition.** Datasets, venvs,
worktrees that build things, inductor caches, campaign scratch, parity spills.

> **The rule that makes an ephemeral disk safe:** everything on the overlay must
> be **regenerable without human input** — a dataset re-fetchable from the Hub, a
> venv from one `uv sync`, a cache by re-running the job. If losing it would cost
> a decision, a measurement or an afternoon of someone's judgement, it does not
> belong here. Validation records, plans and code belong in git and get pushed.

**The HF Hub — the real archive.** Datasets live at `pinkmeme/eval-*` and
checkpoints alongside them ([checkpoints.md](checkpoints.md)); verbose results
go there too. Staging is **pull → use → prune → re-pull**, never hoard.

## Layout

```
/data/          RETRIEVE_DATA_ROOT — staged datasets, checkpoints, _raw/ ETL input
/venvs/         uv project environments (UV_PROJECT_ENVIRONMENT)
/scratch/
  ├── campaigns/  campaign records and eval outputs before they are promoted
  ├── inductor/   TORCHINDUCTOR_CACHE_DIR, one subdir per concurrent job
  ├── parity/     parity spill .npz files (~600–680 MB per run)
  ├── hf/         HF_HOME for Hub download caches
  ├── wt/         worktrees that build or generate bulk artifacts
  └── tmp/        general scratch
```

`/data` mirrors the layout in [datasets.md](datasets.md) and is resolved by
`evaluation/eval_datasets/hub.py:data_root()`.

### Environment

```bash
export RETRIEVE_DATA_ROOT=/data
export UV_PROJECT_ENVIRONMENT=/venvs/retrieve      # reuse; see the venv budget
export TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>   # private per job
export HF_HOME=/scratch/hf
```

`TORCHINDUCTOR_CACHE_DIR` must be **private per concurrent job**: inductor's
on-disk FX cache does not invalidate when a `@triton_op` host wrapper's Python
source changes, so a shared `/tmp/torchinductor_root` silently serves stale
kernels ([reproduction-deviations.md](../paper/reproduction-deviations.md) D-7).
Existing runbooks use `/tmp/inductor-<job>`, which is on the same overlay and
works; `/scratch/inductor/<job>` is the tidier home for new jobs.

### Cutover

`/workspace/data` has been **copied** to `/data` (5.3 GB, 101 files, verified:
identical name+size manifest, and `sha256` matches on the six largest tensors
including `arxiv-papers/content_d128/text_emb.pt` and
`goodreads-work-id/checkpoints/gsasrec-d128-drop0.5-id/encoded_queries_test.pt`).
The originals are still in place because the D1-a campaign is reading them.
**The cutover step, after D1-a finishes:** switch `RETRIEVE_DATA_ROOT` to
`/data`, repoint the `evaluation/data` symlink, then delete `/workspace/data`
to give the network volume its 5.3 GB back.

## The budget

300 GB is one pool. `/data`, `/venvs`, `/scratch`, `/tmp` and `/root` all draw
on it — `/scratch` is **not** a separate device. Resident on 2026-09-15 after
the cleanup:

| | |
|---|---|
| base image (`/usr`, `/opt`, `/var`) | 15 GB |
| `/root` | 8.5 GB |
| `/venvs` (3 environments) | 23 GB |
| `/data` (goodreads-d128 + arxiv-d128) | 5.3 GB |
| `/tmp` (inductor caches) | 1.3 GB |
| **used / free** | **39 G / 262 G** |

**Datasets: one or two resident at a time.** goodreads-d128 + arxiv-d128
(4.9 GB) is the current pair. D1-c wants yambda-500m (9.2 GB) and yambda-5b
(8.8 GB) — 18 GB together, which now fits comfortably, but prune each after its
cells run rather than letting the set accumulate. Never stage the whole registry.

**Venvs: do not create one per worktree.** Each is **≈ 7.6 GB** (torch + triton
+ CUDA wheels). On 2026-09-15 twelve of them reached **91 GB** before nine were
deleted — a third of the whole disk spent on copies of the same packages.
Prefer **one shared environment** that every worktree points at:

```bash
export UV_PROJECT_ENVIRONMENT=/venvs/retrieve
uv run --no-sync --directory evaluation bench run ...
```

Create a second only when a concurrent worker genuinely needs different packages
(a different torch, an unmerged dependency), and delete it when its branch
merges.

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

## What this unblocks

Against **262 GB free** and the one-or-two-datasets-at-a-time rule:

- **D1-c — unblocked with margin.** yambda-500m (9.2 GB) and yambda-5b (8.8 GB)
  fit *simultaneously* with ~244 GB to spare, plus room for parity spills. The
  26 GB network quota was the only thing forcing the stage-one-at-a-time dance.
- **E2 (PubMed) — feasible, but only as a streaming ETL.** The raw mirror is
  102 GB of 768-d fp32 embeddings + 44 GB of per-PMID JSON (the roadmap's
  ~198 GB figure is the download total). The *processed* item tensor alone is
  36 M × 768 × 4 B ≈ **111 GB**. Raw (198) + processed (111) = 309 GB **does not
  fit**. Landing the full raw mirror first (198 GB) leaves only ~64 GB, which is
  not enough to build the tensor beside it. It fits only if the ETL consumes the
  download shard by shard — fetch a shard, append to a memory-mapped output,
  delete the shard — so that the peak is ~111 GB processed plus one shard.
  With no other dataset resident that leaves ~150 GB of headroom. **E2 is a
  disk-feasible, network-bound job**, not a blocked one.
- **E3 (Semantic Scholar / OpenAlex) — still does not fit as specified.** The
  source is 670–840 GB, more than twice the whole disk, so it can never be
  landed. The 50 M-paper slice would be 50 M × 768 × 4 B ≈ **154 GB** processed,
  which *would* fit alone — but only via a pure streaming pass over 840 GB of
  remote JSONL with nothing else on the disk, and the OpenAlex fallback adds
  9–14 A100-hours of encoding on top. Treat E3 as still deferred: the honest
  answer is that the disk stopped being the first obstacle, not that E3 is now
  cheap.

## Persistence — and how confident to be

**Confidence: high that `/workspace` persists, high that the overlay does not.**

- `/workspace` is the RunPod network volume (`RUNPOD_VOLUME_ID=fnvri6j7xa`),
  a MooseFS mount served from outside the pod. Its whole purpose is to outlive
  the container, and it is the documented persistent store.
- `/` is the container's writable overlay layer, `upperdir` inside
  `/var/lib/docker/.../overlay2/<id>/diff` on the host. RunPod calls this the
  *container disk* and treats it as temporary: it is recreated with the
  container. Supporting evidence from this box — the host has been up 38 days
  but `/.dockerenv` is stamped 2026-09-14 22:49, and nothing under `/venvs`
  predates that (the oldest, `/venvs/retrieve`, is 22:58, nine minutes after the
  container came up). The overlay is a day old on a month-old host.

This asymmetry is the whole design: regenerable things on the fast big
ephemeral disk, irreplaceable things in git and pushed to `origin`.

## See also

- [datasets.md](datasets.md) — the dataset layout under `RETRIEVE_DATA_ROOT` and the ETL commands.
- [checkpoints.md](checkpoints.md) — checkpoints and the HF Hub workflow.
- [evaluation.md](evaluation.md) — the harness, campaign records and the GPU lock.
