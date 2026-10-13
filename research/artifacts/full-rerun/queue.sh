#!/usr/bin/env bash
# FULL-RERUN's one sequential driver per GPU, seed-major (user GO 2026-10-13, controller order): seed 0 / draw 0 of every leg first
# (a complete one-seed set), then seed 1, then seed 2, so a stop leaves a complete set at fewer seeds. Inside a seed the order is the
# paper's priority: stage-1 curves and exact LiNR -> (final-pass suites while the other pod finishes) picks -> timed bloom -> untimed
# -> the rest of the final pass -> META-FORK grid (seed 0) -> the clause reference row. runbook-pod1.md / runbook-pode.md.
#
# usage: queue.sh pod1|pode [FROM_SEED]     env as run_leg.sh (TREE VENV CFG FR LOCK), plus PICKS_POD=pod1
set -u
POD=$1 FROM=${2:-0}
H=$(cd "$(dirname "$0")" && pwd)
R="$H/run_leg.sh" FR=${FR:-/scratch/fr} PY=$VENV/bin/python
: "${TREE:?}" "${CFG:?}"
cd "$TREE/evaluation" || exit 1
say() { echo "$(date -u +%FT%TZ) [$POD] $*" | tee -a "$FR/queue-$POD.log"; }
draw() { [ "$2" = 0 ] && echo "$1" || echo "$1-s$2"; }  # synth dataset name of draw $2
hub_has() {  # path in fr/: is it on the Hub yet
  local t; t=$(mktemp -d); "$PY" -m bench.cli fetch --path-in-repo "$1" --results "$t" > /dev/null 2>&1; local rc=$?; rm -rf "$t"; return $rc
}
mark() {  # a tiny upload other pods poll for
  local t; t=$(mktemp -d); echo "{\"pod\": \"$POD\", \"at\": \"$(date -u +%FT%TZ)\"}" > "$t/marker.json"
  "$PY" -m bench.cli upload --results "$t" --path-in-repo "$1" > /dev/null 2>&1; rm -rf "$t"
}

if [ "$POD" = pod1 ]; then
  REAL_768=pubmed REAL_256="laion1m laion3m laion10m"; SYN_768=pubmed-synth SYN_256="laion1m-synth laion3m-synth laion10m-synth"
  V3_REAL="" V3_LAION="laion1m laion3m laion10m"; BW=""; YAMBDA=0
  BLOOM_768=pubmed BLOOM_256="laion1m laion3m laion10m"; BSYN_768=pubmed-synth BSYN_256=laion10m-synth
  FINAL=("filter:pubmed" "v3bits:pubmed" "c3-real:pubmed" "c3:laion30m-synth" "nsweep:laion1m laion3m laion10m laion30m"
         "nsweep-synth:laion1m-synth laion3m-synth laion10m-synth laion30m-synth arxiv-synth"
         "nsweep-codesign:laion1m laion3m laion10m laion30m" "laion30m:laion30m" "laion30m-synth:laion30m-synth"
         "laion30m-x:laion30m" "codesign-laion30m:laion30m" "c7-scorepath:laion30m" "seeds-c1-laion30m:laion30m-synth"
         "seeds-c7:laion30m")
  CEIL="pubmed laion1m laion3m"
else
  REAL_768="" REAL_256=laion30m; REAL_128="goodreads arxiv"; SYN_768="" SYN_256=laion30m-synth
  SYN_128="goodreads-synth arxiv-synth arxiv-corr-synth"; V3_REAL="goodreads arxiv pubmed" V3_LAION=laion30m; BW="goodreads arxiv pubmed"
  YAMBDA=1; BLOOM_768="" BLOOM_256=laion30m BLOOM_128="goodreads arxiv"; BSYN_768="" BSYN_256=laion30m-synth
  BSYN_128="goodreads-synth arxiv-synth"
  FINAL=("filter:goodreads arxiv" "synth:goodreads-synth arxiv-synth arxiv-corr-synth" "c3:goodreads-synth arxiv-synth"
         "c3-real:goodreads arxiv" "codesign:arxiv goodreads" "bloomwidth:goodreads arxiv pubmed"
         "bloomwidth-timed:goodreads arxiv pubmed" "v3bits:goodreads-synth goodreads" "seeds-c1:arxiv-synth" "deep:goodreads arxiv"
         "h2h:goodreads arxiv")
  CEIL="goodreads goodreads-8192 goodreads-synth arxiv arxiv-synth laion30m laion30m-synth"
fi
# final-pass suites with more than one seed after prep_config.py (seeds 1-2 run only these)
SEEDED="filter synth codesign h2h deep bloomwidth bloomwidth-timed v3bits codesign-laion30m c7-scorepath nsweep-codesign seeds-c1 seeds-c1-laion30m seeds-c7"

final_pass() {  # seed [only-until-file]: the pod's final-pass suites at this seed; stop early when $2 appears on the Hub
  local s=$1 item suite ds
  for item in "${FINAL[@]}"; do
    suite=${item%%:*} ds=${item#*:}
    [ "$s" != 0 ] && [[ " $SEEDED " != *" $suite "* ]] && continue
    [ -f "$FR/out/final-$suite-$POD-s$s/.done" ] && continue
    if [ -n "${2:-}" ] && hub_has "$2"; then return 1; fi
    "$R" "final-$suite-$POD" "$s" timed "$suite" $ds && touch "$FR/out/final-$suite-$POD-s$s/.done"
  done
  return 0
}
leg() {  # name seed mode suite datasets...: run once per (name, seed); the leg is <name>-<pod>, uploaded to fr/<name>-<pod>-s<seed>
  local n=$1-$POD s=$2; shift 2
  [ -f "$FR/out/$n-s$s/.done" ] && return 0
  [ -z "${3:-}" ] && return 0
  "$R" "$n" "$s" "$@" && touch "$FR/out/$n-s$s/.done"
}
syn() { local s=$1 out=""; shift; for d in "$@"; do out="$out $(draw "$d" "$s")"; done; echo $out; }

for s in $(seq "$FROM" 2); do
  say "seed $s: stage 1"
  leg curve-d768 "$s" quality fr-curve-d768 $REAL_768
  leg curve-d256 "$s" quality fr-curve-d256 $REAL_256
  leg curve-d128 "$s" quality fr-curve-d128 ${REAL_128:-}
  leg curve-synth-d768 "$s" quality "fr-curve-synth-s$s-d768" $(syn "$s" $SYN_768)
  leg curve-synth-d256 "$s" quality "fr-curve-synth-s$s-d256" $(syn "$s" $SYN_256)
  leg curve-synth-d128 "$s" quality "fr-curve-synth-s$s-d128" $(syn "$s" ${SYN_128:-})
  leg v3pool "$s" timed fr-v3pool $V3_REAL
  leg v3pool-laion "$s" timed fr-v3pool-laion $V3_LAION
  [ "$YAMBDA" = 1 ] && leg yambda "$s" quality fr-yambda yambda-500m
  if [ "$s" = 0 ]; then  # exact LiNR and the width grid: one seed
    if [ "$POD" = pod1 ]; then
      leg linr "$s" timed fr-linr-bs64,fr-linr-bs256,fr-linr-k2048 pubmed
      leg linr-laion "$s" timed fr-linr-bs64-laion laion1m laion3m laion10m laion30m
    else
      leg linr "$s" timed fr-linr-bs64,fr-linr-bs256,fr-linr-k2048 goodreads arxiv
      leg linr-corr "$s" timed fr-linr-corr arxiv-corr-synth
      leg linr-synth-pubmed "$s" timed fr-linr-synth-pubmed pubmed-synth
      leg bloomwidth "$s" quality fr-bloomwidth $BW
    fi
  fi
  mark "fr/marks/stage1-s$s-$POD"
  say "seed $s: final pass while waiting for the picks"
  if [ "$s" = 0 ]; then
    if [ "$POD" = "${PICKS_POD:-pod1}" ]; then
      other=$([ "$POD" = pod1 ] && echo pode || echo pod1)
      until hub_has "fr/marks/stage1-s0-$other"; do final_pass 0 "fr/marks/stage1-s0-$other" || true
        hub_has "fr/marks/stage1-s0-$other" || sleep 300; done
      say "picks"
      bash "$H/make_picks.sh" || { say "PICKS FAILED: stopping"; exit 3; }
    else
      until hub_has "fr/picks"; do final_pass 0 "fr/picks" || true; hub_has "fr/picks" || sleep 300; done
      t=$(mktemp -d); "$PY" -m bench.cli fetch --path-in-repo fr/picks --results "$t" && cp "$t/suites-timed.yaml" "$FR/picks.yaml"
      rm -rf "$t"
    fi
    "$PY" "$H/prep_config.py" "$TREE" "$CFG" "$FR/picks.yaml" || exit 3
  fi
  say "seed $s: stage 2 (timed bloom)"
  leg timed-bloom-d768 "$s" timed fr-timed-bloom-d768,fr-timed-bloom-k1000-d768,fr-timed-bloom-k2048-d768 $BLOOM_768
  leg timed-bloom-d256 "$s" timed fr-timed-bloom-d256,fr-timed-bloom-k1000-d256 $BLOOM_256
  leg timed-bloom-d128 "$s" timed fr-timed-bloom-d128,fr-timed-bloom-k1000-d128,fr-timed-bloom-k2048-d128 ${BLOOM_128:-}
  leg timed-bloom-synth-d768 "$s" timed "fr-timed-bloom-synth-s$s-d768" $(syn "$s" $BSYN_768)
  leg timed-bloom-synth-d256 "$s" timed "fr-timed-bloom-synth-s$s-d256" $(syn "$s" $BSYN_256)
  leg timed-bloom-synth-d128 "$s" timed "fr-timed-bloom-synth-s$s-d128" $(syn "$s" ${BSYN_128:-})
  say "seed $s: untimed"
  SEED=$s bash "$H/untimed.sh" "$POD" "$s" "$CEIL"
  say "seed $s: final pass"
  final_pass "$s"
  [ "$s" = 0 ] && bash "$H/metafork_grid.sh" "$POD"
  say "seed $s: clause reference row"
  leg timed-clause-d768 "$s" timed fr-timed-clause-d768 $BLOOM_768
  leg timed-clause-d256 "$s" timed fr-timed-clause-d256 $BLOOM_256
  leg timed-clause-d128 "$s" timed fr-timed-clause-d128 ${BLOOM_128:-}
  mark "fr/marks/seed$s-$POD"
  say "seed $s done"
done
