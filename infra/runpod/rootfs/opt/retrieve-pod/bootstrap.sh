#!/usr/bin/env bash
set -euo pipefail
. /opt/retrieve-pod/bashrc.sh
echo "== bootstrap $(date -Is) pod=${RUNPOD_POD_ID:-?}"

mkdir -p "$POD_STATE" "$CLAUDE_CONFIG_DIR" "$RETRIEVE_DATA_ROOT" "$HF_HOME"
chmod 700 "$POD_STATE"

if [ ! -L "$HOME/.claude" ]; then
    rm -rf "$HOME/.claude"
    ln -s "$CLAUDE_CONFIG_DIR" "$HOME/.claude"
fi
cfg=$CLAUDE_CONFIG_DIR/.claude.json
[ -s "$cfg" ] || echo '{}' > "$cfg"
jq --arg repo "$REPO_DIR" \
    '.hasCompletedOnboarding = true | .projects[$repo].hasTrustDialogAccepted = true' \
    "$cfg" > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
herdr integration install claude || true

if [ -n "${GIT_USER_NAME:-}" ]; then git config --global user.name "$GIT_USER_NAME"; fi
if [ -n "${GIT_USER_EMAIL:-}" ]; then git config --global user.email "$GIT_USER_EMAIL"; fi
if [ -n "${GH_TOKEN:-}" ]; then gh auth setup-git; fi

if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --branch "$REPO_BRANCH" "https://github.com/$REPO_SLUG.git" "$REPO_DIR"
fi
rp-sync
echo "== bootstrap done $(date -Is)"
