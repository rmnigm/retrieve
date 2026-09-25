#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$HERE/../.." && pwd)
CONF_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/retrieve-pod
CONF=$CONF_DIR/config.env
SSH_CONF=$HOME/.ssh/retrieve-pods.conf

IMAGE=ghcr.io/rmnigm/retrieve-pod:latest
GPU=a100
SSH_KEY=$HOME/.ssh/runpod_ed25519
SECRETS=
[ -f "$CONF" ] && . "$CONF"

die() { echo "pod.sh: $*" >&2; exit 1; }

gpu_id() {
    case $1 in
        a100) echo "NVIDIA A100-SXM4-80GB" ;;
        a100-pcie) echo "NVIDIA A100 80GB PCIe" ;;
        h100) echo "NVIDIA H100 80GB HBM3" ;;
        h100-pcie) echo "NVIDIA H100 PCIe" ;;
        h100-nvl) echo "NVIDIA H100 NVL" ;;
        *) echo "$1" ;;
    esac
}

cmd_init() {
    command -v runpodctl >/dev/null || brew install runpod/runpodctl/runpodctl
    runpodctl user >/dev/null 2>&1 || runpodctl doctor
    read -rp "ssh key [$SSH_KEY]: " k
    SSH_KEY=${k:-$SSH_KEY}
    [ -f "$SSH_KEY" ] || ssh-keygen -t ed25519 -N "" -C retrieve-pod -f "$SSH_KEY"
    runpodctl ssh list-keys | grep -qF "$(cut -d' ' -f2 "$SSH_KEY.pub")" \
        || runpodctl ssh add-key --key-file "$SSH_KEY.pub" >/dev/null
    local s="" pair a
    for pair in GH_TOKEN=gh_token HF_TOKEN=hf_token WANDB_API_KEY=wandb_api_key \
        CLAUDE_CODE_OAUTH_TOKEN=claude_code_oauth_token; do
        read -rp "RunPod secret '${pair#*=}' exists? [y/N] " a
        [ "$a" = y ] && s="$s $pair"
    done
    read -rp "default gpu [$GPU]: " g
    mkdir -p "$CONF_DIR"
    printf 'IMAGE=%s\nGPU=%s\nSSH_KEY=%s\nSECRETS="%s"\n' \
        "$IMAGE" "${g:-$GPU}" "$SSH_KEY" "${s# }" > "$CONF"
    touch "$SSH_CONF"
    grep -qF "$SSH_CONF" "$HOME/.ssh/config" 2>/dev/null || {
        { echo "Include $SSH_CONF"; cat "$HOME/.ssh/config" 2>/dev/null || true; } > "$HOME/.ssh/config.new"
        mv "$HOME/.ssh/config.new" "$HOME/.ssh/config"
        chmod 600 "$HOME/.ssh/config"
    }
    echo "wrote $CONF"
}

pods() {
    runpodctl pod list | jq '[(if type == "array" then .[] else (.pods // [])[] end)
        | select(.name | startswith("retrieve-"))]'
}

endpoint() {
    local ip port cmd
    ip=$(jq -r '.publicIp // empty' <<<"$1")
    port=$(jq -r '(.portMappings // {})["22"] // empty' <<<"$1")
    if [ -z "$ip" ] || [ -z "$port" ]; then
        cmd=$(runpodctl ssh info "$(jq -r .id <<<"$1")" 2>/dev/null | jq -r '.command // empty' || true)
        ip=$(sed -nE 's/.*@([0-9.]+).*/\1/p' <<<"$cmd")
        port=$(sed -nE 's/.*-p ?([0-9]+).*/\1/p' <<<"$cmd")
    fi
    [ -z "$ip" ] || [ -z "$port" ] || echo "$ip $port"
}

refresh() {
    local all n i pod ep
    all=$(pods)
    n=$(jq length <<<"$all")
    : > "$SSH_CONF.tmp"
    for ((i = 0; i < n; i++)); do
        pod=$(jq ".[$i]" <<<"$all")
        ep=$(endpoint "$pod")
        [ -n "$ep" ] || continue
        printf 'Host rp-%s %s\n  HostName %s\n  Port %s\n  User root\n  IdentityFile %s\n  IdentitiesOnly yes\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n  ServerAliveInterval 30\n\n' \
            "$(jq -r '.name | ltrimstr("retrieve-")' <<<"$pod")" "$(jq -r .id <<<"$pod")" \
            "${ep% *}" "${ep#* }" "$SSH_KEY" >> "$SSH_CONF.tmp"
    done
    mv "$SSH_CONF.tmp" "$SSH_CONF"
    echo "$all"
}

host() {
    local a=${1#rp-}
    grep -qE "^Host rp-$a " "$SSH_CONF" 2>/dev/null || refresh >/dev/null
    grep -qE "^Host rp-$a " "$SSH_CONF" && echo "rp-$a" && return
    grep -qE "^Host .* $1$" "$SSH_CONF" && echo "$1" && return
    die "no reachable pod '$1'"
}

pod_id() {
    pods | jq -r --arg n "retrieve-${1#rp-}" --arg id "$1" '.[] | select(.name == $n or .id == $id) | .id' | head -1
}

cmd_up() {
    local gpu=$GPU gpus=1 npods=1 name="" nv="" cloud=SECURE branch=development disk=60 volume=100
    while [ $# -gt 0 ]; do
        case $1 in
            -g | --gpu) gpu=$2; shift ;;
            -n | --gpus) gpus=$2; shift ;;
            -p | --pods) npods=$2; shift ;;
            --name) name=$2; shift ;;
            --nv) nv=$2; shift ;;
            --community) cloud=COMMUNITY ;;
            --branch) branch=$2; shift ;;
            --disk) disk=$2; shift ;;
            --volume) volume=$2; shift ;;
            *) die "unknown option $1" ;;
        esac
        shift
    done
    [ -f "$CONF" ] || die "run 'pod.sh init' first"
    name=${name:-$gpu-x$gpus-$(date +%m%d%H%M)}
    local i pname env pair args
    for ((i = 1; i <= npods; i++)); do
        pname=$name
        [ "$npods" -eq 1 ] || pname=$name-$i
        env=$(jq -n --arg b "$branch" --arg k "$(cat "$SSH_KEY.pub")" \
            --arg n "$(git -C "$REPO_ROOT" config user.name || true)" \
            --arg e "$(git -C "$REPO_ROOT" config user.email || true)" \
            '{REPO_BRANCH: $b, EXTRA_PUBLIC_KEY: $k, GIT_USER_NAME: $n, GIT_USER_EMAIL: $e}')
        [ -z "$nv" ] || env=$(jq --arg d "/workspace/$pname/retrieve" '. + {REPO_DIR: $d}' <<<"$env")
        for pair in $SECRETS; do
            env=$(jq --arg k "${pair%%=*}" --arg v "{{ RUNPOD_SECRET_${pair#*=} }}" '. + {($k): $v}' <<<"$env")
        done
        args=(--name "retrieve-$pname" --image "$IMAGE" --gpu-id "$(gpu_id "$gpu")" --gpu-count "$gpus"
            --cloud-type "$cloud" --container-disk-in-gb "$disk" --ports 22/tcp
            --min-cuda-version 12.8 --env "$env" --wait --wait-timeout 20m)
        if [ -n "$nv" ]; then args+=(--network-volume-id "$nv"); else args+=(--volume-in-gb "$volume"); fi
        [ "$cloud" = SECURE ] || args+=(--public-ip)
        runpodctl pod create "${args[@]}" | jq -r --arg n "rp-$pname" '"\($n)  \(.id)"'
    done
    refresh >/dev/null
    for ((i = 1; i <= npods; i++)); do
        pname=$name
        [ "$npods" -eq 1 ] || pname=$name-$i
        herdr machine add "rp-$pname" --label "$pname" || echo "herdr machine add rp-$pname failed"
    done
}

machine_id() {
    herdr machine list 2>/dev/null | awk -v l="${1#rp-}" '$0 ~ ("(^|[^-[:alnum:]])" l "([^-[:alnum:]]|$)") {print $1; exit}'
}

cmd=${1:-help}
shift || true
case $cmd in
    init) cmd_init ;;
    up) cmd_up "$@" ;;
    ls) refresh | jq -r '.[] | "rp-\(.name | ltrimstr("retrieve-"))\t\(.id)\t\(.desiredStatus // "?")\t\(.costPerHr // "?")/hr"' ;;
    ssh) h=$(host "${1:?pod}"); shift; exec ssh "$h" "$@" ;;
    herdr) exec herdr --remote "$(host "${1:?pod}")" ;;
    login) exec ssh -t "$(host "${1:?pod}")" rp-login "${@:2}" ;;
    log) exec ssh "$(host "${1:?pod}")" tail -n +1 -f /workspace/.pod-home/bootstrap.log ;;
    stop | start) runpodctl pod "$cmd" "$(pod_id "${1:?pod}")" >/dev/null && refresh >/dev/null ;;
    rm) id=$(pod_id "${1:?pod}"); [ -n "$id" ] || die "no pod $1"; m=$(machine_id "$1"); [ -z "$m" ] || herdr machine remove "$m"; runpodctl pod delete "$id" >/dev/null && refresh >/dev/null ;;
    image) docker buildx build --platform linux/amd64 -f "$HERE/Dockerfile" -t "$IMAGE" --push "$REPO_ROOT" ;;
    *) die "usage: pod.sh init|up|ls|ssh|herdr|login|log|stop|start|rm|image (see .claude/skills/runpod/SKILL.md)" ;;
esac
