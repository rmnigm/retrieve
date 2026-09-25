[ -f /etc/retrieve-pod.env ] && . /etc/retrieve-pod.env

export POD_STATE=/workspace/.pod-home
export REPO_SLUG=${REPO_SLUG:-rmnigm/retrieve}
export REPO_BRANCH=${REPO_BRANCH:-main}
export REPO_DIR=${REPO_DIR:-/workspace/retrieve}
export RETRIEVE_DATA_ROOT=${RETRIEVE_DATA_ROOT:-/workspace/data}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export GIT_CONFIG_GLOBAL=$POD_STATE/gitconfig
export CLAUDE_CONFIG_DIR=$POD_STATE/claude

if [ -f "$POD_STATE/secrets.env" ]; then
    set -a
    . "$POD_STATE/secrets.env"
    set +a
fi

if [[ $- == *i* ]]; then
    VIRTUAL_ENV_DISABLE_PROMPT=1 . /venvs/retrieve/bin/activate
    [ "$PWD" = "$HOME" ] && [ -d "$REPO_DIR" ] && cd "$REPO_DIR"
    [ -n "${GH_TOKEN:-}" ] || echo "retrieve-pod: not logged in, run rp-login"
fi
