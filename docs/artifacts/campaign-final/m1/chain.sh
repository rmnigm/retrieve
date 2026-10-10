#!/usr/bin/env bash
# Pod d GPU 0's final pass in order: wait for d-run's "stage-a100-x2-d done" (a d-run note) and GPU 1's lock; M1 under both locks; its
# verdict on $NOTE; then legs.sh over every d0-* leg in recipe order under GPU 0's lock. NOTE=<chain note> bash chain.sh > LOG 2>&1 &
set -u
: "${NOTE:?}"
H=$(dirname "$(readlink -f "$0")")
D=/workspace/retrieve/.chains/campaign-final
done_line() { for f in "$D"/*.md; do grep -q '^branch: "d-run"' "$f" && grep -q -i "stage-a100-x2-d done" "$f" && return 0; done; return 1; }
until done_line; do sleep 60; done
echo "$(date -Is) d-run staging done line seen"
until flock -n /scratch/gpu1.lock true && flock -n /scratch/gpu0.lock true; do sleep 10; done
flock -n /scratch/gpu0.lock flock -n /scratch/gpu1.lock bash "$H/m1.sh" > /scratch/final/m1.driver.log 2>&1
rc=$?
v=$(cat /scratch/final/m1/gate.md 2>/dev/null | grep -i -m1 -E "pass|fail" || echo "no gate.md")
echo "$(date -Is) M1 rc=$rc: $v"
printf -- '- M1 (d0-01 gate) rc=%s at %s: %s (table: /scratch/final/m1/gate.md; GPU 1 released)\n' "$rc" "$(date -u +%H:%MZ)" "$v" >> "$NOTE"
[ $rc -eq 0 ] || exit $rc
legs=$(python3 - /scratch/wt/final/docs/artifacts/campaign-final/recipe.yaml <<'PY'
import sys, yaml
print(" ".join(l["id"] for l in yaml.safe_load(open(sys.argv[1]))["legs"] if l.get("worker") == "laion" and l["index"] == 0))
PY
)
NOTE="$NOTE" flock -n /scratch/gpu0.lock bash "$H/legs.sh" $legs
