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
mkdir -p /scratch/final/done /scratch/final/deps-ok /scratch/final/moved
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
done_dep() {  # strict: "<id> done" at line start (or a "| <id> | done" row) in another worker's note
  [ -e /scratch/final/done/$1 ] || [ -e /scratch/final/moved/$1 ] || [ -e /scratch/final/deps-ok/$1 ] ||
    ls $CH/*.md | grep -v -- '-d-run-' | xargs grep -qE "^[[:space:]*|-]*\`?$1\`?[[:space:]]*\|?[[:space:]]*(done|DONE)([^a-zA-Z]|$)" 2>/dev/null; }
status() {  # rewrite d-run's note from the markers
  local id deps env cmd up hub
  { printf -- '---\nchain: "campaign-final"\nbranch: "d-run"\ncreated: "%s"\n---\n\n# d-run final pass (pod d GPU 1): d1 legs\n\n**Never print secrets** (AGENTS.md rule 9).\n\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Tree /scratch/wt/final at campaign-v2.12 (library %s). Updated %s. %s\n\n| leg | state | wall s | Hub | MANIFEST sha256 |\n|---|---|---|---|---|\n' "${LIB:0:8}" "$(date -Is)" "$1"
    while IFS=$'\t' read -r id deps env cmd up hub; do
      if [ -e /scratch/final/moved/$id ]; then printf '| %s | moved: %s | | | |\n' "$id" "$(cat /scratch/final/moved/$id)"
      elif [ -e /scratch/final/done/$id ]; then printf '| %s | done | %s | %s | %s |\n' "$id" "$(sed -n 1p /scratch/final/done/$id)" "$hub" "$(sed -n 2p /scratch/final/done/$id)"
      elif [ "$id" = "${RUNNING:-}" ]; then printf '| %s | running since %s | | %s | |\n' "$id" "$T0S" "$hub"
      else printf '| %s | queued (after %s) | | %s | |\n' "$id" "$deps" "$hub"; fi
    done < /scratch/final/d1-legs.tsv; } > $NOTE
}
upload_leg() {  # controller CORRECTION 15:20Z: the recipe's --results <out>/<suite> uploads 0 records; stage this leg's files as $T/<suite>/
  local suite ds dim out T n rc
  suite=$(grep -oP -- '--suite \K\S+' <<<"$cmd") || { ( cd $WT/evaluation && env $env bash -c "$up" ); return; }  # ceiling: no records, recipe form
  ds=$(grep -oP -- '--dataset \K\S+' <<<"$cmd"); dim=$(grep -oP -- '--dim \K\S+' <<<"$cmd"); out=$(grep -oP -- '--out \K\S+' <<<"$cmd")
  T=$(mktemp -d /scratch/final/upload.XXXXXX); mkdir -p $T/$suite
  cp -al $out/$suite/$ds-d$dim* $T/$suite/ || return 5
  n=$(wc -l < $out/$suite/$ds-d$dim.jsonl)
  ( cd $WT/evaluation && env $env bash -c "\"\$VENV/bin/python\" -m bench.cli upload --results $T --path-in-repo $hub --verify" ) > $T.log 2>&1; rc=$?; cat $T.log; [ $rc -eq 0 ] || return 6
  grep -q " $n record(s) -> " $T.log || { echo "n_records != $n"; return 7; }
  grep -q 'results.parquet' $T.log || { echo "no results.parquet"; return 8; }
  grep -qE '^  (CITABLE|NOT CITABLE)' $T.log || { echo "no provenance line"; return 9; }
  echo "checked: n_records $n, results.parquet, provenance"; rm -rf $T $T.log
}
status "starting"
while IFS=$'\t' read -r id deps env cmd up hub; do
  [ -e /scratch/final/done/$id ] || [ -e /scratch/final/moved/$id ] && continue
  for d in ${deps//,/ }; do
    until done_dep $d; do status "waiting for $d before $id"; sleep 60; done
  done
  RUNNING=$id; T0S=$(date -Is); t0=$(date +%s); status "running $id"
  case "$cmd" in bash\ *) dir=$WT ;; *) dir=$WT/evaluation ;; esac
  ( cd $dir && flock /scratch/gpu1.lock env $env bash -c "echo \"d-run final $id on GPU 1 since $(date -Is)\" > /scratch/gpu1-holder; $cmd" ) > /scratch/final/logs-$id.log 2>&1 < /dev/null
  rc=$?; echo "$(date -Is) $id rc=$rc s=$(( $(date +%s) - t0 ))"
  if [ $rc -ne 0 ]; then RUNNING=; status "**FAILED: $id rc=$rc** (log /scratch/final/logs-$id.log); runner stopped"; exit $rc; fi
  upload_leg > /scratch/final/upload-$id.log 2>&1 < /dev/null
  urc=$?; man=$(grep -o 'MANIFEST.json sha256 [0-9a-f]*' /scratch/final/upload-$id.log | awk '{print $3}')
  if [ $urc -ne 0 ] || ! grep -q 'round trip verified' /scratch/final/upload-$id.log; then RUNNING=; status "**FAILED upload of $id** (/scratch/final/upload-$id.log); runner stopped"; exit 4; fi
  printf '%s\n%s\n' "$(( $(date +%s) - t0 ))" "$man" > /scratch/final/done/$id; RUNNING=; status "$id done"
  echo "$(date -Is) $id uploaded $hub $man"
done < /scratch/final/d1-legs.tsv
status "**all d1 legs done**"
echo "$(date -Is) d1 runner done"
