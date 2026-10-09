#!/usr/bin/env bash
# V-YFCC synth chunks from the BOTTOM of yfcc_chunks.py (pod 1 takes them from the top; claims on .chains/v-yfcc/,
# protocol in ../../campaign-v2/v-yfcc/README.md). For each chunk, last first: skip it if done, resume it if pod d
# claimed it, stop at the first chunk claimed elsewhere (the meeting point: everything above is pod 1's); otherwise a
# claim note, `bench run --resume --interleave` into /scratch/v-yfcc-synth/<chunk>, `bench upload --verify` to
# artifacts/v-yfcc-synth-<chunk>, a done note. Stops on any non-zero chunk, and between chunks on /scratch/v24/v-yfcc-d.stop.
#   GPU=<i> setsid nohup flock -n /scratch/gpu<i>.lock bash yfcc-chunks.sh > /scratch/v24/v-yfcc-d-gpu<i>.driver.log 2>&1 &
export LEG=v-yfcc-d-gpu${GPU:?} R=/scratch/v-yfcc-synth/_driver-gpu${GPU}
. "$(dirname "$(readlink -f "$0")")/common.sh"
CHY=$CH/v-yfcc
POD=pod-d
DS=yfcc10m-synth
mkdir -p "$CHY"
step oracle oracle --dataset $DS --suite synth --dim 192
$PY "$HERE/../../campaign-v2/v-yfcc/yfcc_chunks.py" 2>/dev/null | grep -- "--algo" | tac > "$LOG/chunks-bottom-up.txt"
ynote() {  # state chunk text
  local f="$CHY/$(date -u +%Y-%m-%d-%H%M%S000)-$POD-$1-$2.md"
  printf -- '---\nchain: "v-yfcc"\nbranch: "%s"\ncreated: "%s"\n---\n\nchunk: %s\nstate: %s\npod: %s\n\n%s\n' \
    "$POD" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" "$1" "$POD" "$3" > "$f"
}
mine() { grep -lx -- "chunk: $1" "$CHY"/*-$POD-claim-*.md 2>/dev/null | head -1; }
done_() { local f; for f in $(grep -lx -- "chunk: $1" "$CHY"/*.md 2>/dev/null); do grep -qx "state: done" "$f" && return 0; done; return 1; }
taken() { grep -qx -- "chunk: $1" "$CHY"/*.md 2>/dev/null; }
while read -r name n args; do
  [ -e /scratch/v24/v-yfcc-d.stop ] && { echo "$(date -Is) stop file, ending before $name"; break; }
  if done_ "$name"; then continue; fi
  if taken "$name" && [ -z "$(mine "$name")" ]; then echo "$(date -Is) $name claimed elsewhere: meeting point, ending"; break; fi
  # a claim by this driver on the other GPU is in flight there
  if [ -n "$(mine "$name")" ] && ! grep -q "gpu $GPU" "$(mine "$name")"; then continue; fi
  [ -n "$(mine "$name")" ] || ynote claim "$name" "gpu $GPU: $n cells: bench run --dataset $DS --dim 192 --suite synth $args"
  out=/scratch/v-yfcc-synth/$name
  mkdir -p "$out/logs"
  log="$out/logs/synth_$DS-d192_$name.log"
  t0=$(date +%s)
  { echo "=== bench run --dataset $DS --dim 192 --suite synth $args --out $out --resume --interleave"
    echo "=== clocks at start"; clocks; } >> "$log"
  $PIN $PY -m bench.cli run --dataset $DS --dim 192 --suite synth $args --out "$out" --resume --interleave >> "$log" 2>&1 < /dev/null
  rc=$?
  { echo "=== clocks at end"; clocks; } >> "$log"
  s=$(( $(date +%s) - t0 ))
  echo "$(date -Is) chunk $name rc=$rc s=$s"
  [ $rc -eq 0 ] || { ynote failed "$name" "gpu $GPU: rc=$rc after ${s}s; log $log"; note pod-d "v-yfcc chunk $name failed rc=$rc on GPU $GPU; driver stopped"; exit $rc; }
  rm -rf "$out/_parity"
  R=$out upload "artifacts/v-yfcc-synth-$name"
  ynote done "$name" "gpu $GPU: $n cells ok in ${s}s at code_version $cv; Hub artifacts/v-yfcc-synth-$name; $UP"
  msg="v-yfcc chunk $name ($n cells) done on pod d GPU $GPU in ${s}s at $cv; Hub artifacts/v-yfcc-synth-$name: $UP"
  note pod-d "$msg"; note control "$msg"
  herdr agent prompt controller "d-run driver: $msg" > /dev/null 2>&1 < /dev/null
done < "$LOG/chunks-bottom-up.txt"
finish
