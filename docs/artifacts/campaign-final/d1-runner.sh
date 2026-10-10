#!/usr/bin/env bash
# Final pass, pod d GPU 1 (d-run): run the recipe's d1-* legs in order, verbatim (env, command, upload_command) from the recipe AT THE TAG in
# /scratch/wt/final. A dependency on another worker's leg is met when a note on .chains/campaign-final/ says "<leg id> done" (the brief's per-worker
# notes) or /scratch/final/deps-ok/<leg id> exists. Bench commands run from evaluation/, `bash docs/...` from the repo root. Stops on the first
# non-zero rc (leg or upload) and says so in d-run's note. Rerun after a crash: finished legs (marker /scratch/final/done/<id>) are skipped and the
# legs' --resume skips finished cells.
# GPU 1's lock is taken per leg (not while waiting on a dependency: laion's M1 needs both pod d locks before d0-01).
#   setsid nohup bash d1-runner.sh > /scratch/final/d1-runner.log 2>&1 &
set -u
WT=/scratch/wt/final; CH=/workspace/retrieve/.chains/campaign-final
NOTE=$CH/2026-10-10-153000000-d-run-final-pass.md
mkdir -p /scratch/final/done /scratch/final/deps-ok
LIB=e16512f5b379fc7e2d370cc8a57c68b9854a06d7
[ "$(git -C $WT rev-parse HEAD:retrieve/src/retrieve)" = "$LIB" ] || { echo "worktree not at $LIB, refusing"; exit 2; }
/venvs/final-o3/bin/python - "$WT/docs/artifacts/campaign-final/recipe.yaml" > /scratch/final/d1-legs.tsv <<'PY'
import sys, yaml, shlex
r = yaml.safe_load(open(sys.argv[1]))
for l in r["legs"]:
    if l["worker"] != "d-run" or not l["id"].startswith("d1"):
        continue
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in l["env"].items())
    print("\t".join([l["id"], ",".join(l.get("after") or []), env, l["command"], l["upload_command"], l["upload"]]))
PY
done_dep() { [ -e /scratch/final/done/$1 ] || [ -e /scratch/final/deps-ok/$1 ] || grep -E "$1([^a-z0-9-]|$)" $CH/*.md 2>/dev/null | grep -vE "queued|running|waiting|failed|FAILED" | grep -qwE "done|DONE" 2>/dev/null; }
status() {  # rewrite d-run's note from the markers
  { printf -- '---\nchain: "campaign-final"\nbranch: "d-run"\ncreated: "%s"\n---\n\n# d-run final pass (pod d GPU 1): d1 legs\n\n**Never print secrets** (AGENTS.md rule 9).\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Tree /scratch/wt/final at campaign-v2.12 (library %s). Updated %s. %s\n\n| leg | state | wall s | Hub | MANIFEST sha256 |\n|---|---|---|---|---|\n' "${LIB:0:8}" "$(date -Is)" "$1"
    while IFS=$'\t' read -r id deps env cmd up hub; do
      if [ -e /scratch/final/done/$id ]; then printf '| %s | done | %s | %s | %s |\n' "$id" "$(sed -n 1p /scratch/final/done/$id)" "$hub" "$(sed -n 2p /scratch/final/done/$id)"
      elif [ "$id" = "${RUNNING:-}" ]; then printf '| %s | running since %s | | %s | |\n' "$id" "$T0S" "$hub"
      else printf '| %s | queued (after %s) | | %s | |\n' "$id" "$deps" "$hub"; fi
    done < /scratch/final/d1-legs.tsv; } > $NOTE
}
status "starting"
while IFS=$'\t' read -r id deps env cmd up hub; do
  [ -e /scratch/final/done/$id ] && continue
  for d in ${deps//,/ }; do
    until done_dep $d; do status "waiting for $d before $id"; sleep 60; done
  done
  RUNNING=$id; T0S=$(date -Is); t0=$(date +%s); status "running $id"
  case "$cmd" in bash\ *) dir=$WT ;; *) dir=$WT/evaluation ;; esac
  ( cd $dir && flock /scratch/gpu1.lock env $env bash -c "echo \"d-run final $id on GPU 1 since $(date -Is)\" > /scratch/gpu1-holder; $cmd" ) > /scratch/final/logs-$id.log 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) $id rc=$rc s=$(( $(date +%s) - t0 ))"
  if [ $rc -ne 0 ]; then RUNNING=; status "**FAILED: $id rc=$rc** (log /scratch/final/logs-$id.log); runner stopped"; exit $rc; fi
  ( cd $WT/evaluation && env $env bash -c "$up" ) > /scratch/final/upload-$id.log 2>&1 < /dev/null
  urc=$?; man=$(grep -o 'MANIFEST.json sha256 [0-9a-f]*' /scratch/final/upload-$id.log | awk '{print $3}')
  if [ $urc -ne 0 ] || ! grep -q 'round trip verified' /scratch/final/upload-$id.log; then RUNNING=; status "**FAILED upload of $id** (/scratch/final/upload-$id.log); runner stopped"; exit 4; fi
  printf '%s\n%s\n' "$(( $(date +%s) - t0 ))" "$man" > /scratch/final/done/$id; RUNNING=; status "$id done"
  echo "$(date -Is) $id uploaded $hub $man"
done < /scratch/final/d1-legs.tsv
status "**all d1 legs done**"
echo "$(date -Is) d1 runner done"
