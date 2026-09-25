---
name: runpod
description: Launch, reach, log into and tear down RunPod GPU pods (A100 / H100, one or several GPUs, one or several pods) preloaded with this repo's CUDA 12.8 + uv environment, gh, Claude Code and herdr; build and push the pod image. Use when the user asks for a pod, a GPU box, a RunPod machine, to run GPU work remotely, or to update the pod image.
---

# RunPod dev pods

Everything lives in `infra/runpod/`. The laptop-side driver is `infra/runpod/pod.sh`;
it needs `runpodctl` ≥ 2.14 (`brew install runpod/runpodctl/runpodctl`), `jq`, `ssh`,
and `herdr` for remote attach. Every `runpodctl` command prints JSON.

## What a pod is

- Image `ghcr.io/rmnigm/retrieve-pod:<tag>`, the tag pinned as `IMAGE` in `pod.sh`
  (`infra/runpod/Dockerfile`): Ubuntu 22.04 with only nvcc + CUDA 12.8 headers (`CUDA_HOME=/usr/local/cuda-12.8`, matching the torch 2.10.0+cu128 wheel),
  uv, Python 3.11, the workspace venv at `/venvs/retrieve` built from `uv.lock`
  with `--all-packages --all-groups --extra official` (Meta's silvertorch compiled
  for sm_80 + sm_90), gh, Claude Code, herdr, runpodctl, sshd.
- `/workspace` is the pod volume (or a network volume). Boot
  (`rootfs/opt/retrieve-pod/bootstrap.sh`, log `/workspace/.pod-home/bootstrap.log`)
  clones `rmnigm/retrieve` at `staging` into `$REPO_DIR` (default
  `/workspace/retrieve`; `/workspace/<pod>/retrieve` when a network volume is shared),
  runs `rp-sync`, installs herdr's Claude integration and pre-trusts the checkout.
  It never touches an existing checkout. A headless `herdr server` starts at boot
  (log `/workspace/.pod-home/herdr.log`), so herdr sessions outlive SSH disconnects.
- State that survives restarts lives in `/workspace/.pod-home`: `secrets.env`
  (tokens), `gitconfig`, `claude/` (Claude config, `~/.claude` points here).
  `HF_HOME` and `RETRIEVE_DATA_ROOT` are on `/workspace` too.
- In the pod use `rp-sync` rather than bare `uv sync`: a bare sync is exact and
  uninstalls pytest and silvertorch.

## One-time setup (user's laptop)

`infra/runpod/pod.sh init`, which is interactive, so ask the user to run it with `! infra/runpod/pod.sh init`:
RunPod API key (`runpodctl doctor`), SSH key (created if missing, added to the
RunPod account), which RunPod secrets exist, default GPU. It writes
`~/.config/retrieve-pod/config.env` and puts `Include ~/.ssh/retrieve-pods.conf`
at the top of `~/.ssh/config`.

Credentials, two routes, and they combine:

1. RunPod secrets, created by the user at console.runpod.io → Secrets (runpodctl
   cannot create them). Names: `gh_token`, `hf_token`, `wandb_api_key`,
   `claude_code_oauth_token`. Pods then boot logged in. Only secrets marked as
   existing in `init` are referenced, because a missing one fails the deploy.
2. `rp-login` inside a pod (`pod.sh login NAME [gh|hf|wandb|claude]`): prompts for
   each and stores it in `/workspace/.pod-home/secrets.env`, which wins over
   secrets. With a network volume, this is once for all pods on it.

GitHub: prefer a fine-grained PAT limited to `rmnigm/retrieve`, Contents read/write.
A token cannot be limited to one branch; protect `main` on GitHub. Claude: the user
runs `claude setup-token` (laptop or pod) for a subscription token, or uses an API key.

## Image (built on a RunPod CPU pod, no local Docker)

```bash
infra/runpod/pod.sh image [BRANCH]    # default staging; streams the build log, deletes the pod
```

kaniko on a 16-vCPU CPU pod builds `origin/BRANCH` (so push `infra/runpod/` and the
lockfiles first; the command refuses otherwise) and pushes `IMAGE` plus `latest`.
It authenticates with the RunPod secret `ghcr_token`: a classic GitHub token with only
`write:packages`. The pod is created through RunPod's REST API, since `runpodctl`
cannot size CPU pods or override the entrypoint. The log also lands in
`$TMPDIR/retrieve-image-build.log`; it takes several minutes, so run it in the background.

Bump the tag in `pod.sh` for every rebuild (pods cache by tag), then delete the
now-untagged version on GHCR (`gh api user/packages/container/retrieve-pod/versions`,
needs `delete:packages`). Rebuild after `uv.lock` or `infra/runpod/` changes. Pods
pull anonymously, so the GHCR package must be public (GitHub → Packages →
retrieve-pod → settings).

## Commands

```bash
pod.sh up                          # 1× default GPU (config), secure cloud, 100 GB pod volume
pod.sh up -g h100 -n 4             # one pod, 4× H100 SXM
pod.sh up -g a100 -p 3             # three pods, 1× A100 SXM each (names …-1..3)
pod.sh up --nv VOLUME_ID -g h100   # attach a network volume (shared state, per-pod checkout)
pod.sh up --community --branch X --name foo --disk 80 --volume 200
pod.sh ls                          # retrieve-* pods, refreshes ssh aliases rp-NAME
pod.sh log NAME                    # follow bootstrap until "bootstrap done"
pod.sh login NAME                  # rp-login on the pod
pod.sh herdr NAME                  # attach to that pod's herdr directly; `claude` in a pane runs on the pod
pod.sh ssh NAME [cmd…]
pod.sh stop|start|rm NAME
```

`up` also saves each pod as a herdr machine (`herdr machine add rp-NAME --label NAME`),
so plain `herdr` on the laptop lists it in the sidebar; `rm` removes it. SSH goes
through the alias `rp-NAME` in `~/.ssh/retrieve-pods.conf` with the key from
`SSH_KEY` in the local config (it must be a key registered in the RunPod account).
After `stop`/`start` the pod gets a new port: run `pod.sh ls` once it is up to
refresh the alias; the herdr machine keeps working through it.

GPU keys: `a100` (A100-SXM4-80GB), `a100-pcie`, `h100` (H100 80GB HBM3), `h100-pcie`,
`h100-nvl`; any other string is passed to `--gpu-id` verbatim (`runpodctl gpu list`
shows valid ids and availability). `up` blocks until sshd answers (`--wait`), then
bootstrap takes about a minute more.

## Rules for agents

- Pods cost money by the hour. Confirm GPU type, count and pod count with the user
  before `up`. Never `rm` without asking: it destroys the pod volume. `stop` keeps it.
- A pod can stop itself: `runpodctl pod stop $RUNPOD_POD_ID`.
- The GPU rules in `CLAUDE.md` (serialize GPU work, no clock locking, `torch.profiler`)
  apply on pods too; record the GPU model with every number.
- Never echo tokens or `secrets.env` contents into the conversation, logs or commits.
