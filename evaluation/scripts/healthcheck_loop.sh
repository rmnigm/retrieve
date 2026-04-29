#!/usr/bin/env bash
# Autonomous Yambda-5B training healthcheck loop.
#
# Runs in its own tmux session (yambda5b-watcher). Polls state every 45 min,
# appends one line per check to evaluation/logs/healthcheck-summary.log,
# exits cleanly when both v4 + v5 are done OR a hard failure is detected.
#
# Replaces the Claude-side ScheduleWakeup chain — the next /loop fire detects
# this script and stops rescheduling.
#
# Auto-actions (only safe ones):
#   - retry HF upload up to 3× per checkpoint if it failed soft (state H)
#   - exit on state D (success) or E/F/G (hard failure)
# Will NOT do:
#   - restart a crashed trainer (user diagnoses)
#   - delete checkpoints/logs to free disk
#   - touch tmux state of training sessions

set -uo pipefail

cd /root/retrieve/evaluation
SUMMARY=/root/retrieve/evaluation/logs/healthcheck-summary.log
V4_DIR=/root/retrieve/evaluation/checkpoints/gsasrec-5b-listens-d64-v4
V5_DIR=/root/retrieve/evaluation/checkpoints/gsasrec-5b-listens-d128-v5
LOG_GLOB_V4='/root/retrieve/evaluation/logs/train-5b-v4-*.log'
LOG_GLOB_V5='/root/retrieve/evaluation/logs/train-5b-v5-*.log'
LOG_GLOB_FU='/root/retrieve/evaluation/logs/followup-*.log'

declare -A upload_retries
upload_retries[v4]=0
upload_retries[v5]=0

log_summary() {
    local state="$1" msg="$2" next="$3"
    printf '%s | watcher %s | %s | next_wake=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$state" "$msg" "$next" \
        >> "$SUMMARY"
}

# Has the trainer crashed? Look for hard failure markers in the latest log.
has_crashed() {
    local glob="$1"
    local f
    f=$(ls -1t $glob 2>/dev/null | head -1) || return 1
    [ -n "$f" ] && grep -aE "Traceback|CUDA out of memory|Killed|RuntimeError" "$f" 2>/dev/null | head -1 | grep -q .
}

# Is the named tmux session alive?
tmux_alive() {
    tmux has-session -t "$1" 2>/dev/null
}

# Latest NDCG@10 from v4 / v5 log (best-effort).
latest_ndcg() {
    local glob="$1"
    local f
    f=$(ls -1t $glob 2>/dev/null | head -1) || return
    [ -n "$f" ] || return
    grep -aoE 'ndcg@10": [0-9.]+' "$f" 2>/dev/null | tail -1 | awk -F': ' '{print $2}'
}

# Try HF upload (single attempt). Returns 0 on success.
try_hf_upload() {
    local ckpt_subdir="$1"
    uv run python -m scripts.upload_checkpoints \
        --owner pinkmeme --checkpoint "$ckpt_subdir" --private 2>&1 \
        | tee -a "$SUMMARY.upload" | tail -3
    return ${PIPESTATUS[0]}
}

# Detect "HF upload failed" line in followup log and grab the checkpoint name.
upload_failed_for() {
    local glob="$LOG_GLOB_FU"
    local f
    f=$(ls -1t $glob 2>/dev/null | head -1) || return 1
    [ -n "$f" ] || return 1
    # We tag failed uploads with the literal "HF upload failed" string in v4_followup.sh.
    # Detect either v4 or v5 missing-on-HF based on which checkpoint mention precedes it.
    grep -aE "HF upload failed|uploading.*v[45]" "$f" 2>/dev/null | tail -10
}

log_summary START "watcher started by healthcheck_loop.sh; replacing Claude-side ScheduleWakeup chain" "yes 45min"

while true; do
    now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

    v4_done=0; [ -f "${V4_DIR}/train_metrics.json" ] && v4_done=1
    v5_done=0; [ -f "${V5_DIR}/train_metrics.json" ] && v5_done=1
    v4_alive=0; tmux_alive yambda5b && v4_alive=1
    fu_alive=0; tmux_alive yambda5b-followup && fu_alive=1

    v4_crashed=0; has_crashed "$LOG_GLOB_V4" && v4_crashed=1
    v5_crashed=0; has_crashed "$LOG_GLOB_V5" && v5_crashed=1

    free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')

    state=""
    msg=""

    if [ "$free_gb" -lt 5 ]; then
        state="G"
        msg="disk pressure: only ${free_gb}G free on /; stopping watcher (no auto-cleanup)"
    elif [ "$v4_crashed" -eq 1 ] || [ "$v5_crashed" -eq 1 ]; then
        state="E"
        msg="crash detected (v4_crashed=${v4_crashed} v5_crashed=${v5_crashed}); stopping watcher"
    elif [ "$v4_done" -eq 1 ] && [ "$v5_done" -eq 1 ]; then
        state="D"
        v4_n=$(python -c "import json; print(json.load(open('${V4_DIR}/eval_quality.json'))['metrics'].get('ndcg@10','?'))" 2>/dev/null || echo "?")
        v5_n=$(python -c "import json; print(json.load(open('${V5_DIR}/eval_quality.json'))['metrics'].get('ndcg@10','?'))" 2>/dev/null || echo "?")
        msg="ALL DONE: v4 test NDCG@10=${v4_n}; v5 test NDCG@10=${v5_n}; uploads attempted in followup log"
    elif [ "$v4_done" -eq 0 ] && [ "$v4_alive" -eq 0 ]; then
        state="F"
        msg="v4 tmux missing but no train_metrics.json or crash signature; stopping watcher"
    elif [ "$v4_done" -eq 1 ] && [ "$v5_done" -eq 0 ] && [ "$fu_alive" -eq 0 ]; then
        state="F"
        msg="v4 done but followup tmux gone and no v5 train_metrics; stopping watcher"
    elif [ "$v4_done" -eq 0 ]; then
        state="A"
        ndcg=$(latest_ndcg "$LOG_GLOB_V4")
        msg="v4 training, last ndcg@10=${ndcg:-?}, GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' '), free=${free_gb}G"
    elif [ "$v5_done" -eq 0 ]; then
        v5_log=$(ls -1t $LOG_GLOB_V5 2>/dev/null | head -1)
        if [ -z "$v5_log" ] || [ ! -s "$v5_log" ]; then
            state="B"
            msg="v4 done, followup uploading or v5 about to start"
        else
            state="C"
            ndcg=$(latest_ndcg "$LOG_GLOB_V5")
            msg="v5 training, last ndcg@10=${ndcg:-?}, GPU $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d ' '), free=${free_gb}G"
        fi
    else
        state="?"
        msg="unclassified: v4_done=${v4_done} v5_done=${v5_done} v4_alive=${v4_alive} fu_alive=${fu_alive}"
    fi

    case "$state" in
        D|E|F|G)
            log_summary "$state" "$msg" "NO terminal"
            exit 0
            ;;
        *)
            log_summary "$state" "$msg" "yes 45min"
            ;;
    esac

    # Auto-retry HF upload if followup soft-failed it (state H component).
    # Looks for "HF upload failed" + the most recent attempted checkpoint name.
    fu_log=$(ls -1t $LOG_GLOB_FU 2>/dev/null | head -1)
    if [ -n "$fu_log" ] && grep -qaE "HF upload failed" "$fu_log" 2>/dev/null; then
        for ckpt in gsasrec-5b-listens-d64-v4 gsasrec-5b-listens-d128-v5; do
            key="${ckpt##*-}"  # v4 or v5
            tag="${key:0:2}"
            # Only retry if local dir exists, train_metrics exists, and not already uploaded
            ckpt_path="/root/retrieve/evaluation/checkpoints/${ckpt}"
            [ -f "${ckpt_path}/train_metrics.json" ] || continue
            [ -f "${ckpt_path}/.uploaded" ] && continue
            retries="${upload_retries[$tag]:-0}"
            [ "$retries" -ge 3 ] && continue
            log_summary "RETRY-UPLOAD" "attempting HF upload for ${ckpt} (try $((retries+1))/3)" "yes 45min"
            if try_hf_upload "$ckpt"; then
                touch "${ckpt_path}/.uploaded"
                log_summary "UPLOAD-OK" "${ckpt} uploaded to HF" "yes 45min"
            else
                upload_retries[$tag]=$((retries+1))
            fi
        done
    fi

    sleep 2700
done
