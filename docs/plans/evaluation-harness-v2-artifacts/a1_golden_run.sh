#!/usr/bin/env bash
# A1 — the GPU lane: the WP-0 golden baseline plus handoff steps 4-7.
#
# Roadmap A1 is "golden baseline on the old harness ... Also closes the
# harness half of refactor-validation-handoff.md (steps 4-7)". This script
# is that lane, end to end, in five stages.
#
#   golden  the 11 golden cells on THIS branch          -> evaluation/golden/
#   step4   TORCH_LOGS=graph_breaks compile smoke        (handoff step 4)
#   step7   run-evaluation kill / --resume smoke         (handoff step 7)
#   step6   the same 11 cells from the `main` worktree
#           + the quality diff                           (handoff step 6)
#   step5   per-kernel tune-kernels +-5% gates, both
#           sides, behind a wall-time estimate           (handoff step 5)
#
# What the golden cells are for: they are the reference quality numbers
# that harness v2's GPU gate (C4 / H WP-4) has to reproduce within 1e-6.
# The latency columns are recorded for context — they are cold-L2
# `do_bench` medians of CUDA-graph replay, the thing v2 replaces (H §1
# verdicts 1-2) — not as a target.
#
# Order: the coordinator's a-d (golden, step4, step7, step5) with step6
# inserted before step5, because step5 is the one stage that may refuse to
# run (see its budget gate) and step6 is a hard gate that must not sit
# behind it. Override with STAGES.
#
# Cells (11 processes per side, one per (dataset, algo, backend) — the
# process boundary H §8.2 K settles on, so no dynamo cache or CUDA-graph
# pool outlives the backend under test):
#
#   goodreads d128-filter, --filter-kind clause --sweep c0_genre
#     {linr_v1_filter_mask, linr_v2, linr_v3, linr_v4, silvertorch}
#     x {triton, torch}                                        = 10
#   arxiv     d128-filter, --filter-kind clause --sweep c0_maincat
#     silvertorch x {triton}                                   =  1
#
# `triton` and `torch` are the ONLY golden backends. H §6 WP-0's text asks
# for `--backend cuda cute` on arxiv; that is void per H's amendment
# (2026-09-05) — the CUDA C++ and CuTe backends are deleted after the
# official backend's parity gate (roadmap B4), so nothing downstream would
# ever compare against a cuda/cute golden column.
#
# Each process writes 9 rows per cell (3 ks x 3 batch sizes).
#
# WHERE TO RUN: the A100 box, from the repository root of THIS worktree,
# with `evaluation/data` pointing at the dataset root. Every stage needs
# CUDA. Needs root for the clock lock (NO_CLOCK_LOCK=1 to skip).
#
#   cd <repo>
#   bash docs/plans/evaluation-harness-v2-artifacts/a1_golden_run.sh
#
# Env knobs:
#   STAGES="golden step4 step7 step6 step5"   which stages to run, in order
#   MAIN_WORKTREE=/workspace/wt/main-golden   the `main` side of steps 5+6
#   NO_CLOCK_LOCK=1     skip nvidia-smi lock/unlock (no sudo). Quality is
#                       unaffected; latency is then not clock-controlled
#                       and the run record must say so.
#   GOLDEN_DIR=...      branch-side cell outputs (default evaluation/golden)
#   A1_DIR=...          step4/5/6/7 artifacts (default
#                       docs/plans/evaluation-harness-v2-artifacts/a1)
#   PER_POINT_S=4.0     step5 cost model: seconds per (shape, config) point
#   STEP5_BUDGET_S=7200 step5 refuses to run above this total estimate
#   STEP7_WAIT_S=1800   how long step7 waits for the first algo before
#                       giving up on the kill/resume interleave
#
# Reruns are safe: a cell whose JSON already exists is skipped, so an
# interrupted run continues where it stopped (the old harness only writes
# rows at the end of a process — H §1 verdict 7 — so a partial cell is
# simply absent). Delete a JSON to force it.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EVAL_DIR="$REPO_ROOT/evaluation"
GOLDEN_DIR="${GOLDEN_DIR:-$EVAL_DIR/golden}"
MAIN_GOLDEN_DIR="$GOLDEN_DIR/_main"
LOG_DIR="$GOLDEN_DIR/_logs"
A1_DIR="${A1_DIR:-$REPO_ROOT/docs/plans/evaluation-harness-v2-artifacts/a1}"
MAIN_WORKTREE="${MAIN_WORKTREE:-/workspace/wt/main-golden}"
MAIN_EVAL_DIR="$MAIN_WORKTREE/evaluation"

STAGES="${STAGES:-golden step4 step7 step6 step5}"
PER_POINT_S="${PER_POINT_S:-4.0}"
STEP5_BUDGET_S="${STEP5_BUDGET_S:-7200}"
STEP7_WAIT_S="${STEP7_WAIT_S:-1800}"

GOODREADS_CONFIG="config/goodreads/d128-filter.yaml"
GOODREADS_SWEEP="c0_genre"
GOODREADS_ALGOS=(linr_v1_filter_mask linr_v2 linr_v3 linr_v4 silvertorch)
GOODREADS_BACKENDS=(triton torch)

ARXIV_CONFIG="config/arxiv/d128-filter.yaml"
ARXIV_SWEEP="c0_maincat"
ARXIV_ALGOS=(silvertorch)
ARXIV_BACKENDS=(triton)

# The six kernels handoff step 5 gates. `codesigned-probe-score-exact` is
# the K1-acceptance run (a tuner that never existed before) and is branch
# only — it has no baseline by construction.
STEP5_KERNELS=(clause-mask clause-compact bloom-compact
               fused-masked-knn-topk oporp-1bit-match-topk codesigned-probe-score)

mkdir -p "$GOLDEN_DIR" "$LOG_DIR" "$A1_DIR"

FAILED=()
SKIPPED_STAGES=()

# ----- clocks ---------------------------------------------------------------
# Locked SM clocks so the latency columns are comparable across processes
# and against C4's rerun (H §2.1; CLAUDE.md hard rule 1).

CLOCKS_LOCKED=0
SAMPLER_PID=""

_nvsmi_priv() {
  # The box may or may not have sudo, and a container may hold neither the
  # capability nor a sudo binary. Try both, quietly.
  if command -v sudo >/dev/null 2>&1; then sudo -n nvidia-smi "$@" 2>&1; else nvidia-smi "$@" 2>&1; fi
}

lock_clocks() {
  [ "${NO_CLOCK_LOCK:-0}" = "1" ] && { echo "[clocks] NO_CLOCK_LOCK=1 — not locking"; return 0; }
  echo "[clocks] locking SM clock to 1410 MHz"
  _nvsmi_priv -pm 1 | tail -1
  if _nvsmi_priv -lgc 1410 | tee /dev/stderr | grep -qi "all done"; then
    CLOCKS_LOCKED=1
  else
    echo "[clocks] WARNING: could not lock clocks (no permission in this container?)."
    echo "[clocks] Falling back to H §7's alternative: clocks are SAMPLED per cell into"
    echo "[clocks] $LOG_DIR/clocks.csv and the run record must say the latencies are not"
    echo "[clocks] clock-controlled. Quality columns are unaffected."
  fi
}

unlock_clocks() {
  [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null
  [ "$CLOCKS_LOCKED" = "1" ] || return 0
  echo "[clocks] resetting SM clock"
  _nvsmi_priv -rgc | tail -1
}

start_clock_sampler() {
  # The fallback H §7 prescribes when locking is unavailable — and useful
  # provenance even when it is not: one sample every 30 s for the whole run.
  local out="$LOG_DIR/clocks.csv"
  # Append, never truncate: a second driver invocation (a resumed run, a
  # single re-run cell) must not erase the trace of the first.
  [ -s "$out" ] || echo "utc,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu" > "$out"
  ( while true; do
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> "$out"
      sleep 30
    done ) &
  SAMPLER_PID=$!
  echo "[clocks] sampling every 30s -> $out (pid $SAMPLER_PID)"
}

trap unlock_clocks EXIT

# ----- one cell -------------------------------------------------------------

run_cell() {
  local eval_dir="$1" side="$2" out_dir="$3" tag="$4" config="$5" algo="$6" backend="$7" sweep="$8"
  local out="$out_dir/${tag}-${sweep}-${algo}-${backend}.json"
  local log="$LOG_DIR/${side}-${tag}-${sweep}-${algo}-${backend}.log"

  if [ -s "$out" ]; then
    echo "[skip] $out exists"
    return 0
  fi

  echo "[run ] ${side}: $algo/$backend -> $out"
  # `uv run` from the side's evaluation/: the configs' data_dir and filter
  # attrs_path are cwd-relative (retrieval.loaders.resolve_path), so the
  # working directory is load-bearing.
  ( cd "$eval_dir" && uv run evaluate \
      --config "$config" \
      --algo "$algo" \
      --backend "$backend" \
      --filter-kind clause \
      --sweep "$sweep" \
      --output "$out" ) >"$log" 2>&1

  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[FAIL] ${side}: $algo/$backend rc=$rc — see $log"
    rm -f "$out"          # never leave a truncated golden behind
    FAILED+=("${side}/${tag}/${algo}/${backend}")
    return 0              # keep going; the summary reports the failures
  fi
  echo "[ ok ] ${side}: $algo/$backend"
}

run_all_cells() {
  local eval_dir="$1" side="$2" out_dir="$3"
  mkdir -p "$out_dir"
  local algo backend
  for algo in "${GOODREADS_ALGOS[@]}"; do
    for backend in "${GOODREADS_BACKENDS[@]}"; do
      run_cell "$eval_dir" "$side" "$out_dir" "goodreads-d128" \
               "$GOODREADS_CONFIG" "$algo" "$backend" "$GOODREADS_SWEEP"
    done
  done
  for algo in "${ARXIV_ALGOS[@]}"; do
    for backend in "${ARXIV_BACKENDS[@]}"; do
      run_cell "$eval_dir" "$side" "$out_dir" "arxiv-d128" \
               "$ARXIV_CONFIG" "$algo" "$backend" "$ARXIV_SWEEP"
    done
  done
}

require_main_worktree() {
  if [ ! -d "$MAIN_EVAL_DIR" ]; then
    echo "[SKIP] no main worktree at $MAIN_WORKTREE — create it with:"
    echo "       git -C $REPO_ROOT worktree add $MAIN_WORKTREE -b tmp/main-users-limit-fix main"
    echo "       (then port the users_limit fix, symlink evaluation/data, uv sync)"
    return 1
  fi
  if [ ! -e "$MAIN_EVAL_DIR/data" ]; then
    echo "[SKIP] $MAIN_EVAL_DIR/data missing — symlink it at the dataset root first"
    return 1
  fi
  return 0
}

# ----- stage: golden --------------------------------------------------------

stage_golden() {
  echo; echo "########## stage golden — 11 cells on $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)"
  run_all_cells "$EVAL_DIR" "branch" "$GOLDEN_DIR"
}

# ----- stage: step4 (compile gate) ------------------------------------------
# handoff step 4: no NEW graph breaks attributable to masked_topk,
# _PackedBitsKNN or the shared kernel preps. "New" is only decidable
# against main, so when the main worktree is there this runs both sides
# and diffs the break reasons; otherwise it prints the branch's breaks for
# a human to read.

_step4_side() {
  local eval_dir="$1" side="$2" d="$3"
  local log="$d/${side}-graph-breaks.log"
  echo "[run ] step4 ${side}"
  ( cd "$eval_dir" && TORCH_LOGS=graph_breaks uv run evaluate \
      --config "$GOODREADS_CONFIG" --algo linr_v3 \
      --backend triton --filter-kind clause --sweep "$GOODREADS_SWEEP" \
      --skip-quality --output "$d/${side}-linr_v3_graphbreak_smoke.json" ) >"$log" 2>&1
  local rc=$?
  # Digits are collapsed so a break reason that differs only by a shape or
  # a line number does not read as "new".
  grep -oE "Graph break.*" "$log" | sed 's/[0-9]\{2,\}/N/g' | sort -u > "$d/${side}-breaks.txt"
  echo "[ rc=$rc ] step4 ${side}: $(wc -l < "$d/${side}-breaks.txt") unique break reason(s) -> $d/${side}-breaks.txt"
  return $rc
}

stage_step4() {
  echo; echo "########## stage step4 — TORCH_LOGS=graph_breaks compile smoke"
  local d="$A1_DIR/step4"; mkdir -p "$d"

  _step4_side "$EVAL_DIR" branch "$d" || FAILED+=("step4/branch")

  if require_main_worktree; then
    _step4_side "$MAIN_EVAL_DIR" main "$d" || FAILED+=("step4/main")
    echo "--- break reasons on the branch but not on main (these are the gate) ---"
    comm -13 "$d/main-breaks.txt" "$d/branch-breaks.txt" | tee "$d/new-breaks.txt"
    if [ -s "$d/new-breaks.txt" ]; then
      echo "[FAIL] step4: $(wc -l < "$d/new-breaks.txt") break reason(s) new on the branch"
      FAILED+=("step4/new-breaks")
    else
      echo "[ ok ] step4: no break reason on the branch that main does not also have"
    fi
  else
    SKIPPED_STAGES+=("step4/main-side comparison")
    echo "branch break reasons (read them against main by hand):"
    cat "$d/branch-breaks.txt"
  fi
}

# ----- stage: step7 (orchestrator smoke) ------------------------------------
# handoff step 7 plus the coordinator's addition: kill it mid-run, resume,
# confirm it continues. Output goes to a throwaway dir via a copied config
# so the checked-in evaluation/results/ tree is never touched.

stage_step7() {
  echo; echo "########## stage step7 — run-evaluation kill / --resume smoke"
  local d="$A1_DIR/step7"; mkdir -p "$d"
  local outdir="$d/out"; rm -rf "$outdir"; mkdir -p "$outdir"
  # The config has to live under evaluation/ — run_evaluation._cfg_name does
  # `cfg.resolve().relative_to(EVAL_DIR)` to name its per-config log and
  # raises otherwise. So it is written next to the real one and deleted at
  # the end of the stage (`.gitignore` covers it in between).
  local cfg="$EVAL_DIR/config/goodreads/_a1_step7-d128-filter.yaml"
  rm -f "$cfg"

  # Same config, `output:` redirected. The anchor block keeps working: the
  # key lives inside `_defaults`, one line, one substitution.
  sed "s|^  output:.*|  output: $outdir|" "$EVAL_DIR/$GOODREADS_CONFIG" > "$cfg"
  cp "$cfg" "$d/$(basename "$cfg")"   # keep a copy with the artifacts
  if ! grep -q "^  output: $outdir$" "$cfg"; then
    echo "[FAIL] step7: could not redirect output: in the config copy"
    FAILED+=("step7/config")
    rm -f "$cfg"
    return 0
  fi

  local first_json="$outdir/${GOODREADS_ALGOS[0]}.json"

  # The child runs in its own session so the whole `run-evaluation` ->
  # `evaluate` subprocess tree dies together. It writes its own PGID: asking
  # `ps` for the background subshell's PGID would hand back THIS script's
  # group, and killing that kills the runbook itself (observed 2026-09-06).
  local pgidfile="$d/pass1.pgid"
  rm -f "$pgidfile"
  echo "[run ] step7 pass 1 (killed as soon as the first algo completes)"
  setsid bash -c "cd '$EVAL_DIR' && echo \$\$ > '$pgidfile' && exec uv run run-evaluation '$cfg' --resume --logdir '$d/runlogs' -- --skip-quality --sweep '$GOODREADS_SWEEP'" \
      >"$d/pass1.log" 2>&1 &
  local pid=$! waited=0 mypgid cpgid
  mypgid="$(ps -o pgid= $$ 2>/dev/null | tr -d ' ')"

  _step7_kill() {
    cpgid="$(cat "$pgidfile" 2>/dev/null | tr -d ' ')"
    if [ -n "$cpgid" ] && [ "$cpgid" != "$mypgid" ]; then
      kill -KILL -- "-$cpgid" 2>/dev/null
    else
      echo "[warn] step7: no distinct child PGID ($cpgid vs mine $mypgid); killing the pid only"
    fi
    kill -KILL "$pid" 2>/dev/null
  }

  while kill -0 "$pid" 2>/dev/null; do
    if [ -s "$first_json" ]; then
      echo "[kill] first algo complete after ${waited}s — killing the child session"
      _step7_kill
      break
    fi
    if [ "$waited" -ge "$STEP7_WAIT_S" ]; then
      echo "[FAIL] step7: no algo finished within ${STEP7_WAIT_S}s (raise STEP7_WAIT_S)"
      _step7_kill
      FAILED+=("step7/timeout")
      rm -f "$cfg"
      return 0
    fi
    sleep 5; waited=$((waited + 5))
  done
  wait "$pid" 2>/dev/null

  if [ ! -s "$first_json" ]; then
    echo "[FAIL] step7: pass 1 produced no complete algo JSON"
    FAILED+=("step7/pass1")
    rm -f "$cfg"
    return 0
  fi

  echo "[run ] step7 pass 2 (--resume; must skip what pass 1 finished)"
  ( cd "$EVAL_DIR" && uv run run-evaluation "$cfg" --resume --logdir "$d/runlogs" \
      -- --skip-quality --sweep "$GOODREADS_SWEEP" ) >"$d/pass2.log" 2>&1
  local rc=$?

  local resumed done_n
  resumed=$(grep -c "resume: skipping" "$d/pass2.log")
  done_n=$(find "$outdir" -maxdepth 1 -name '*.json' -size +0c | wc -l)
  echo "  exit=$rc  resume-skips=$resumed  algo JSONs=$done_n / ${#GOODREADS_ALGOS[@]}"
  ls -1 "$d/runlogs" 2>/dev/null | sed 's/^/    runlog: /'

  if [ "$rc" -ne 0 ] || [ "$resumed" -lt 1 ] || [ "$done_n" -ne "${#GOODREADS_ALGOS[@]}" ]; then
    echo "[FAIL] step7: expected exit 0, >=1 resume skip, ${#GOODREADS_ALGOS[@]} JSONs"
    FAILED+=("step7")
  else
    echo "[ ok ] step7: killed mid-run, resumed, completed"
  fi
  rm -f "$cfg"
}

# ----- stage: step6 (golden diff vs main) -----------------------------------

stage_step6() {
  echo; echo "########## stage step6 — main-side cells + quality diff"
  if ! require_main_worktree; then
    SKIPPED_STAGES+=("step6")
    FAILED+=("step6/no-main-worktree")
    return 0
  fi

  echo "main worktree HEAD: $(git -C "$MAIN_WORKTREE" log --oneline -1)"
  run_all_cells "$MAIN_EVAL_DIR" "main" "$MAIN_GOLDEN_DIR"

  local d="$A1_DIR/step6"; mkdir -p "$d"
  python3 "$REPO_ROOT/docs/plans/evaluation-harness-v2-artifacts/a1_step6_diff.py" \
      --branch "$GOLDEN_DIR" --main "$MAIN_GOLDEN_DIR" --report "$d/report.md" \
      | tee "$d/stdout.txt"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then
    echo "[FAIL] step6: quality diff not clean — see $d/report.md"
    FAILED+=("step6/diff")
  else
    echo "[ ok ] step6: quality byte-identical, additive columns only"
  fi
}

# ----- stage: step5 (per-kernel perf gates) ---------------------------------

step5_estimate() {
  # Points = shapes x grid entries, read from the branch's KernelTuneSpec
  # registry (retrieve/src/retrieve/tune.py). Cost per point is 3 JIT-warm
  # calls + do_bench(warmup=100ms, rep=500ms) — identical on both sides
  # (verified 2026-09-06) — plus a Triton compile the first time a config
  # meets a shape, which is what PER_POINT_S is padded for.
  ( cd "$REPO_ROOT" && uv run python - "$PER_POINT_S" "${STEP5_KERNELS[@]}" <<'PY'
import sys
from retrieve.tune import KERNELS
per_point = float(sys.argv[1])
specs = {s.name: s for s in KERNELS}
total = 0
print(f"  {'kernel':30} {'shapes':>7} {'grid':>6} {'points':>7} {'est':>9}")
for name in sys.argv[2:]:
    s = specs[name]
    if s.default_regimes:
        n = len(s.default_regimes)
    else:
        n = len(s.expand_regimes(**{f.lstrip('-').replace('-', '_'): d for f, d, _ in s.dims}))
    pts = n * len(s.grid)
    total += pts
    print(f"  {name:30} {n:>7} {len(s.grid):>6} {pts:>7} {pts * per_point / 60:>8.1f}m")
print(f"  {'TOTAL (one side)':30} {'':>7} {'':>6} {total:>7} {total * per_point / 60:>8.1f}m")
print(f"POINTS={total}")
PY
  )
}

stage_step5() {
  echo; echo "########## stage step5 — per-kernel tune gates (+-5%)"
  local d="$A1_DIR/step5"; mkdir -p "$d/branch" "$d/main"

  local est_out points total_s
  est_out="$(step5_estimate)"
  echo "$est_out" | grep -v '^POINTS=' | tee "$d/estimate.txt"
  points="$(echo "$est_out" | sed -n 's/^POINTS=//p')"
  if [ -z "$points" ]; then
    echo "[FAIL] step5: could not compute the sweep-point count"
    FAILED+=("step5/estimate")
    return 0
  fi

  # main runs the same six kernels over the same shapes and grids (grid
  # sizes 8/8/8/12/12/12, 5 clause/bloom regimes — verified 2026-09-06), so
  # both sides cost the same. `codesigned-probe-score` on main is the K1
  # bug and dies immediately; counting it only makes the estimate
  # conservative.
  total_s=$(python3 -c "print(int(2 * $points * $PER_POINT_S))")
  {
    echo "both sides: 2 x $points points x ${PER_POINT_S}s = ${total_s}s ($((total_s / 60)) min)"
    echo "budget: ${STEP5_BUDGET_S}s ($((STEP5_BUDGET_S / 60)) min)"
  } | tee -a "$d/estimate.txt"

  if [ "$total_s" -gt "$STEP5_BUDGET_S" ]; then
    echo "[SKIP] step5: estimate ${total_s}s exceeds the ${STEP5_BUDGET_S}s budget — NOT running."
    echo "       Report the estimate and get a decision (raise STEP5_BUDGET_S to override)."
    SKIPPED_STAGES+=("step5 (estimate ${total_s}s > budget ${STEP5_BUDGET_S}s)")
    return 0
  fi

  local kernel rc
  for kernel in "${STEP5_KERNELS[@]}"; do
    if [ ! -s "$d/branch/$kernel.json" ]; then
      echo "[run ] step5 branch $kernel"
      ( cd "$REPO_ROOT/retrieve" && uv run tune-kernels "$kernel" \
          --json-out "$d/branch/$kernel.json" ) >"$d/branch/$kernel.log" 2>&1 \
        || { echo "[FAIL] step5 branch $kernel — see $d/branch/$kernel.log"; FAILED+=("step5/branch/$kernel"); }
    fi
  done

  # The K1-acceptance run: a tuner this kernel never had. Branch only, and
  # its DEFAULT_CONFIG is NOT to be committed (handoff step 5).
  if [ ! -s "$d/branch/codesigned-probe-score-exact.json" ]; then
    echo "[run ] step5 branch codesigned-probe-score-exact (K1 acceptance, no baseline)"
    ( cd "$REPO_ROOT/retrieve" && uv run tune-kernels codesigned-probe-score-exact \
        --json-out "$d/branch/codesigned-probe-score-exact.json" ) \
      >"$d/branch/codesigned-probe-score-exact.log" 2>&1 \
      || { echo "[FAIL] step5 K1-acceptance run"; FAILED+=("step5/k1-acceptance"); }
    grep -h "^DEFAULT_CONFIG" "$d/branch/codesigned-probe-score-exact.log" || true
  fi

  if require_main_worktree; then
    for kernel in "${STEP5_KERNELS[@]}"; do
      if [ ! -s "$d/main/$kernel.json" ]; then
        echo "[run ] step5 main $kernel"
        ( cd "$MAIN_WORKTREE/retrieve" && uv run tune-kernels "$kernel" \
            --json-out "$d/main/$kernel.json" ) >"$d/main/$kernel.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
          if [ "$kernel" = "codesigned-probe-score" ]; then
            echo "[note] main $kernel failed as expected (the K1 bug); no baseline for it"
          else
            echo "[FAIL] step5 main $kernel — see $d/main/$kernel.log"
            FAILED+=("step5/main/$kernel")
          fi
        fi
      fi
    done
    python3 "$REPO_ROOT/docs/plans/evaluation-harness-v2-artifacts/a1_step5_compare.py" \
        --main "$d/main" --branch "$d/branch" --report "$d/report.md" | tee "$d/stdout.txt"
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
      echo "[FAIL] step5: a kernel is outside +-5% — see $d/report.md"
      FAILED+=("step5/gate")
    else
      echo "[ ok ] step5: every joined shape within +-5%"
    fi
  else
    SKIPPED_STAGES+=("step5/main baseline")
  fi

  # Deliberately not tuned here: the cuda / cute variants
  # (codesigned-probe-score{,-exact}-{cuda,cute}) — 528 more points for
  # backends that are deleted after B2. Void, like WP-0's cuda/cute golden
  # columns.
}

# ----- provenance -----------------------------------------------------------

{
  echo "branch:   $(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD) @ $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "dirty:    $(git -C "$REPO_ROOT" status --porcelain | wc -l) paths"
  echo "main wt:  ${MAIN_WORKTREE} @ $(git -C "$MAIN_WORKTREE" rev-parse HEAD 2>/dev/null || echo 'absent')"
  echo "stages:   $STAGES"
  echo "host:     $(hostname)"
  echo "utc:      $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nvidia-smi:"
  nvidia-smi 2>&1 | head -12
  ( cd "$EVAL_DIR" && uv run python -c \
      "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__)" 2>&1 )
} | tee "$LOG_DIR/provenance.txt"

lock_clocks
start_clock_sampler

for stage in $STAGES; do
  case "$stage" in
    golden) stage_golden ;;
    step4)  stage_step4 ;;
    step5)  stage_step5 ;;
    step6)  stage_step6 ;;
    step7)  stage_step7 ;;
    *) echo "unknown stage: $stage"; exit 2 ;;
  esac
done

# ----- summary --------------------------------------------------------------

echo
echo "=== golden cells in $GOLDEN_DIR ==="
for f in "$GOLDEN_DIR"/*.json; do
  [ -e "$f" ] || continue
  n=$( (cd "$EVAL_DIR" && uv run python -c \
        "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$f") 2>/dev/null || echo "?")
  echo "  $(basename "$f")  rows=$n"
done

if [ ${#SKIPPED_STAGES[@]} -ne 0 ]; then
  echo
  echo "=== skipped (${#SKIPPED_STAGES[@]}) ==="
  printf '  %s\n' "${SKIPPED_STAGES[@]}"
fi

if [ ${#FAILED[@]} -ne 0 ]; then
  echo
  echo "=== FAILED (${#FAILED[@]}) ==="
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi

echo
echo "all requested stages green. Next: commit evaluation/golden/*.json (NOT"
echo "_main/, which is gitignored) and docs/plans/evaluation-harness-v2-artifacts/a1/,"
echo "append the validation record to docs/plans/evaluation-harness-v2.md and to"
echo "the harness half (steps 4-7) of docs/plans/refactor-validation-handoff.md,"
echo "then flip A1 in docs/plans/00-roadmap.md."
