# Yambda-5B training healthcheck runbook

You are a Claude Code agent that woke up from `ScheduleWakeup` to check on the Yambda-5B training pipeline running on this machine. You have **zero conversation context** from when this was set up — read this file and the artifacts it points to, then act.

## What's running (started 2026-04-29 ~19:32 UTC)

**Two tmux sessions, both on this box:**

| tmux | What | Log file |
|---|---|---|
| `yambda5b` | v4 training: gSASRec d=64, bs=2048, K=128 per-pos, lr=1e-3, gbce-t=0.75, dropout=0.1, tied embeddings, patience=5, num-epochs=100 | `evaluation/logs/train-5b-v4-20260429-193426.log` |
| `yambda5b-followup` | watcher script `evaluation/scripts/v4_followup.sh`: waits for v4 → uploads `pinkmeme/gsasrec-5b-listens-d64-v4` → launches v5 (d=128, K=64) → uploads `pinkmeme/gsasrec-5b-listens-d128-v5` | `evaluation/logs/followup-20260429-195535.log` (followup); `evaluation/logs/train-5b-v5-*.log` (v5 trainer once it starts) |

Checkpoint dirs: `evaluation/checkpoints/gsasrec-5b-listens-d64-v4/` and `evaluation/checkpoints/gsasrec-5b-listens-d128-v5/`. Existence of `train_metrics.json` inside means the corresponding trainer has finished.

GPU: A100-80GB. Disk: `/` is 80 GB overlay; raw HF cache is on `/workspace`, not `/`.

## Step 1 — gather state (run these in parallel)

```bash
date -u +%Y-%m-%dT%H:%M:%SZ
tmux ls
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
df -h /root | tail -1
ls -la /root/retrieve/evaluation/checkpoints/gsasrec-5b-listens-d64-v4/ 2>&1
ls -la /root/retrieve/evaluation/checkpoints/gsasrec-5b-listens-d128-v5/ 2>&1
grep -aE "ndcg@10|Epoch [0-9]+ —|Early stop|Test metrics|Traceback|Error|OOM|killed" \
    /root/retrieve/evaluation/logs/train-5b-v4-*.log \
    /root/retrieve/evaluation/logs/train-5b-v5-*.log \
    /root/retrieve/evaluation/logs/followup-*.log 2>/dev/null | tail -40
```

## Step 2 — decide the situation

Based on the artifacts above, classify which of these you're in:

### A. v4 still training, no errors
- `yambda5b` tmux exists, `train-5b-v4-*.log` shows progressing epochs, GPU >0% utilization, no `Traceback`/`OOM`/`killed`, no `train_metrics.json` in v4 ckpt dir yet.
- **Action**: do nothing. Schedule next wake-up at 2700s (45 min). Exit.

### B. v4 finished, followup actively uploading or v5 launching
- `train_metrics.json` exists in v4 ckpt dir, followup log shows recent activity (HF upload or v5 launch lines).
- **Action**: do nothing. Schedule next wake-up. Exit.

### C. v5 training, no errors
- `yambda5b-followup` tmux still alive, `train-5b-v5-*.log` shows progressing epochs, GPU active, no errors.
- **Action**: do nothing. Schedule next wake-up. Exit.

### D. Both runs finished (success)
- `train_metrics.json` exists in BOTH `gsasrec-5b-listens-d64-v4/` AND `gsasrec-5b-listens-d128-v5/`.
- Followup log contains `[followup] ... all done`.
- **Action**: do **NOT** reschedule. Append a one-line "ALL DONE — final NDCGs: …" summary into `/root/retrieve/evaluation/logs/healthcheck-summary.log` with the final test NDCG@10 from each `eval_quality.json`. Exit. The user will see the summary when they're back.

### E. Trainer crashed / OOM / killed
- log contains `Traceback`, `CUDA out of memory`, `Killed`, OR the tmux session died and the corresponding `train_metrics.json` does not exist.
- **Action**: do **NOT** auto-restart. Append a detailed report to `/root/retrieve/evaluation/logs/healthcheck-summary.log` including: tmux state, last 30 lines of relevant log, GPU memory, disk free. Then **stop scheduling wake-ups**. The user will diagnose and rerun manually.

### F. tmux sessions unexpectedly missing but no crash signature
- e.g. `yambda5b` is gone but no `train_metrics.json` and no traceback in log.
- **Action**: same as E — log a report and stop. Don't restart blindly.

### G. Disk pressure: `df -h /` shows < 5 GB free
- **Action**: log it. Do **NOT** delete anything automatically. Stop scheduling and let the user resolve. (User can re-arm by running ScheduleWakeup themselves once they've cleared space.)

### H. HF upload failure visible in followup log (`HF upload failed`)
- v5 should still launch (the script catches upload errors with `||`), so this is a soft failure.
- **Action**: log it as a warning in the summary file but keep scheduling. The user can retry the upload manually with `uv run python -m scripts.upload_checkpoints --owner pinkmeme --checkpoint <name> --private`.

## Step 3 — schedule next wake-up (only if you're in A, B, or C)

Call `ScheduleWakeup` with:
- `delaySeconds`: 2700 (45 min — stays inside cache TTL × ~10× margin)
- `prompt`: `<<autonomous-loop-dynamic>>`
- `reason`: one short sentence about what you're watching, e.g. "v4 ep 12, healthy, checking again in 45 min"

If you're in D, E, F, or G — **do not call ScheduleWakeup**. Stop the chain.

## Step 4 — log what you did

Append one line to `/root/retrieve/evaluation/logs/healthcheck-summary.log`:

```
<UTC timestamp> | <state A/B/C/D/E/F/G/H> | <one-sentence summary> | next_wake=<yes 45min | NO terminal>
```

Keep this file as the user-visible audit trail.

## Constraints / things NOT to do

- Do not kill or restart any tmux session.
- Do not modify any code in `evaluation/training/`.
- Do not retry the HF upload from this runbook (let the user do it).
- Do not delete checkpoints or logs.
- Do not run `nvidia-smi --gpu-reset` or anything that touches GPU state.
- Do not edit this runbook.
- Do not call any tool other than `Bash` (read-only commands), `Read`, `Write` (only to append to `healthcheck-summary.log`), and `ScheduleWakeup`.
