#!/usr/bin/env bash
# V-YFCC synth by chunks (shared with pod c, first come first served): for each chunk of yfcc_chunks.py not yet named
# in a note on .chains/v-yfcc/, write a claim note, run it (`bench run --resume --interleave`, its own tree
# /scratch/v-yfcc-synth/<chunk>), upload it as artifacts/v-yfcc-synth-<chunk> and write a done note. Stops on the first
# non-zero chunk, or between chunks when /scratch/v21/v-yfcc-synth.stop exists. Rerun the same command after a crash:
# a chunk this pod claimed and did not finish is resumed first (its claim note names this pod).
# Launch: setsid nohup flock -n /scratch/gpu0.lock bash synth-chunks.sh > /scratch/v21/v-yfcc-synth/driver.log 2>&1 &
LEG=v-yfcc-synth
R=/scratch/v-yfcc-synth
. "$(dirname "$(readlink -f "$0")")/../v-pod1-run/common.sh"
CH=/workspace/retrieve/.chains/v-yfcc
POD=pod-1
mkdir -p "$CH"
old_oracles yfcc10m-synth
step oracle oracle --dataset yfcc10m-synth --suite synth
$PY "$HERE/../v-yfcc/yfcc_chunks.py" 2>/dev/null | grep -- "--algo" > "$LOG/chunks.txt"
note() {  # kind chunk text
  local f="$CH/$(date -u +%Y-%m-%d-%H%M%S000)-$POD-$1-$2.md"
  printf -- '---\nchain: "v-yfcc"\nbranch: "%s"\ncreated: "%s"\n---\n\nchunk: %s\nstate: %s\npod: %s\n\n%s\n' \
    "$POD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" "$1" "$POD" "$3" > "$f"
}
mine() { grep -lw -- "$1" "$CH"/*-$POD-claim-*.md 2>/dev/null | head -1; }
done_() { local f; for f in $(grep -lw -- "chunk: $1" "$CH"/*.md 2>/dev/null); do grep -qx "state: done" "$f" && return 0; done; return 1; }
taken() { grep -qw -- "chunk: $1" "$CH"/*.md 2>/dev/null; }
while read -r name n args; do
  [ -e /scratch/v21/v-yfcc-synth.stop ] && { echo "$(date -Is) stop file, ending before $name"; break; }
  if done_ "$name"; then continue; fi
  if taken "$name" && [ -z "$(mine "$name")" ]; then echo "$(date -Is) $name claimed elsewhere, skipping"; continue; fi
  [ -n "$(mine "$name")" ] || note claim "$name" "$n cells: bench run --dataset yfcc10m-synth --dim 192 --suite synth $args"
  out=$R/$name
  mkdir -p "$out/_logs"
  log="$out/_logs/synth_yfcc10m-synth-d192_$name.log"
  t0=$(date +%s)
  { echo "=== bench run --dataset yfcc10m-synth --dim 192 --suite synth $args --out $out --resume --interleave"
    echo "=== clocks at start"; clocks; } >> "$log"
  $PIN $PY -m bench.cli run --dataset yfcc10m-synth --dim 192 --suite synth $args --out "$out" --resume --interleave \
    >> "$log" 2>&1 < /dev/null
  rc=$?
  { echo "=== clocks at end"; clocks; } >> "$log"
  s=$(( $(date +%s) - t0 ))
  echo "$(date -Is) chunk $name rc=$rc s=$s"
  [ $rc -eq 0 ] || { note failed "$name" "rc=$rc after ${s}s; log $log"; exit $rc; }
  rm -rf "$out/_parity"
  mkdir -p "$out/logs" && cp -a "$log" "$out/logs/"
  up=$(cd "$REPO/evaluation" && $PY -m bench.cli upload --results "$out" --path-in-repo "artifacts/v-yfcc-synth-$name" --verify 2>&1 \
    | tee "$LOG/upload-$name.log" | grep -E "MANIFEST|round trip")
  echo "$(date -Is) upload $name: $up"
  grep -q "round trip verified" <<< "$up" || { note failed "$name" "upload failed; $LOG/upload-$name.log"; exit 4; }
  note done "$name" "$n cells ok in ${s}s at code_version $cv; Hub artifacts/v-yfcc-synth-$name; $up"
done < "$LOG/chunks.txt"
finish
