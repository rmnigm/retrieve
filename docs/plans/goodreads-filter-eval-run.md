# Goodreads filter-eval — execution runbook

## Context

The harness for the goodreads filter-eval bench is built and currently
undergoing **cosmetic refactor only** (no functional changes; quality
and filter suites already separate, exact / approximate algos already
distinguished). This runbook is for the agent that will actually launch
the eval **after** the refactor lands.

The bench answers two questions on three GSASRec checkpoints (d64 /
d128 / d256, ID-only, drop0.5):

1. **Quality (yambda-style, no filter)**: how well does each retrieval
   algorithm find the user's actual next read? GT is the held-out test
   target. `torch_fullscan recall@10` should match the yambda regression
   number to ±0.0005.
2. **Filter (oracle-based)**: how well does each algo + filter-module
   combination approximate brute-force filtered top-K? GT is the
   filtered FullScan top-K, cached per sweep at
   `data/goodreads/work-id/gt/gt_topk_<sweep>.pt`. Held-out target is
   degenerate under filtering (filters routinely mask the only positive
   out).

Spec: [goodreads-filter-eval.md](goodreads-filter-eval.md).
Driver: [eval_goodreads_retrieval.py](../../evaluation/retrieval/eval_goodreads_retrieval.py).
Algo wrapper: [filter_registry.py](../../evaluation/retrieval/filter_registry.py).
Configs: [conf/goodreads-d{64,128,256}-drop0.5-id.yaml](../../evaluation/conf/).

### Algos per suite

| Suite | Algorithms | Type |
|-------|-----------|------|
| Quality (`filter_kind=none`) | `torch_fullscan`, `linr_v3_then_v2`, `silvertorch` | exact, approx, approx |
| Filter (`clause` / `bloom` / `combined`) | `linr_v2_filter_compact`, `linr_v3_then_v2`, `silvertorch` (wide / combined only) | exact, approx, approx |

`torch_fullscan` is dropped from filter sweeps (= filtered-FullScan
oracle, recall=1.0 trivially). `linr_v2_filter_compact` is dropped from
the no-filter sweep (no filter → no compact pass-list to rerank).
`silvertorch` skips narrow-only sweeps automatically (per the bench
thesis: bloom is wide-only).

---

## Pre-flight checklist

Run this once before kicking off any bench. Each step is a fast file
check or import — no GPU work.

```bash
cd /workspace/retrieve/evaluation

# 1. Checkpoints present (all three).
for d in d64 d128 d256; do
  test -f checkpoints/goodreads-gsasrec-$d-drop0.5-id/best_model.pt \
    || { echo "MISSING: $d checkpoint"; exit 1; }
done

# 2. Data artifacts present.
for f in item_attrs_narrow.pt item_attrs_wide.pt clause_is_reverse_narrow.pt eval_split.parquet; do
  test -f data/goodreads/work-id/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 3. Configs parse and pick up the new exact-LiNR algo.
uv run python -c "
from retrieval.config import load_eval_config
from pathlib import Path
for p in ['conf/goodreads-d64-drop0.5-id.yaml',
          'conf/goodreads-d128-drop0.5-id.yaml',
          'conf/goodreads-d256-drop0.5-id.yaml']:
    cfg = load_eval_config(Path(p))
    assert 'linr_v2_filter_compact' in cfg.algorithms, f'{p}: missing linr_v2_filter_compact'
    assert cfg.checkpoint.endswith('-drop0.5-id/best_model.pt'), f'{p}: stale ckpt path'
    print(f'OK  {p}')
"

# 4. Driver + filter_registry import cleanly.
uv run python -c "
import retrieval.eval_goodreads_retrieval
import retrieval.filter_registry
from retrieval.filter_registry import build_filter_modules, build_filtered_algorithm
print('OK  driver + filter_registry import')
"
```

If any check fails: stop and report. Do not attempt to "fix" missing
checkpoints or stale paths — escalate.

---

## Execution

Run sequentially. d64 first as a sanity check; **only proceed to d128 /
d256 after d64's output validates** (see Verification below).

Each run takes ~20–40 minutes wall on a single GPU. Oracle compute
dominates (11 sweeps × 313 k users × N=797 k items, batched at 64).
Oracle tensors are cached at `data/goodreads/work-id/gt/gt_topk_<sweep>.pt`;
they depend on `item_embs`, so each dim recomputes its own (no cross-dim
caching). Re-runs of the same dim reuse the cache.

```bash
cd /workspace/retrieve/evaluation

# 1. d64 first
uv run evaluate \
  --config conf/goodreads-d64-drop0.5-id.yaml

# 2. d128 once d64 validates
uv run evaluate \
  --config conf/goodreads-d128-drop0.5-id.yaml

# 3. d256 once d128 validates
uv run evaluate \
  --config conf/goodreads-d256-drop0.5-id.yaml
```

### Background execution recipe

Long-running; use `Bash` with `run_in_background=true` and `Monitor` on
the resulting shell ID. Do **not** poll with `sleep` — get a single
notification on completion and check the output file.

```bash
# Start in background, stream stderr to a known log path.
uv run evaluate \
  --config conf/goodreads-d64-drop0.5-id.yaml \
  2>&1 | tee /tmp/bench-d64.log
```

Output JSON path: `<ckpt-dir>/eval_goodreads_retrieval.json` (i.e.
`checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json`).

### Single-cell debugging

If a specific cell fails, narrow the run:

```bash
uv run evaluate \
  --config conf/goodreads-d64-drop0.5-id.yaml \
  --filter-kind clause \
  --sweep c0_genre \
  --algorithms linr_v2_filter_compact
```

`--algorithms` **replaces** (not merges with) the YAML list.
`--filter-kind` and `--sweep` filter to a single slice.

---

## Verification (per-dim, after each run completes)

The output JSON has shape:

```json
{
  "config_path": "...",
  "checkpoint": "...",
  "n_users": 313178,
  "quality_eval": [...],
  "filter_eval": [...]
}
```

Check each item below. **All seven must pass before moving to the next
dim.** A `jq`-based one-liner is given for each.

### 1. Quality cross-check vs yambda regression

`quality_eval` row `impl=torch_fullscan k=10` `recall@10` should match
the yambda regression number (`eval_quality.json` in the same checkpoint
dir) to ±0.0005.

Yambda baselines:

| dim  | yambda recall@10 | yambda recall@100 |
|------|------------------|-------------------|
| d64  | 0.03331          | 0.14861           |
| d128 | 0.03403          | 0.14799           |
| d256 | 0.03429          | 0.14724           |

```bash
jq '.quality_eval[] | select(.impl=="torch_fullscan" and .k==10 and .batch_size==1) | "recall@10="+(."recall@10"|tostring)' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
```

### 2. Exact LiNR sanity (recall=1.0 by construction)

Every `filter_eval` row with `impl=linr_v2_filter_compact` must have
`recall@k = 1.0` (within float noise, ≥ 0.999).

```bash
jq '[.filter_eval[] | select(.impl=="linr_v2_filter_compact") | ."recall@10"] | min' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
# Expect ≥ 0.999
```

If this fails on a sweep, the bug is in either `evaluate_indices` or
the V2 candidate path — escalate before re-running.

### 3. No torch_fullscan in `filter_eval`

```bash
jq '[.filter_eval[] | select(.impl=="torch_fullscan")] | length' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
# Expect 0
```

### 4. No linr_v2_filter_compact in `quality_eval`

```bash
jq '[.quality_eval[] | select(.impl=="linr_v2_filter_compact")] | length' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
# Expect 0
```

### 5. silvertorch on narrow-only sweeps marked skipped

Clause sweeps (`c0_genre`, `c1_lang_reverse`, `c3_year`, `c0c1`, `c0c3`,
`c0c4_author`, `all5`) should have a single skip row per silvertorch
algo, not perf rows.

```bash
jq '[.filter_eval[] | select(.filter_kind=="clause" and .impl=="silvertorch" and (.skipped // false))] | length' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
# Expect 7 (one per clause sweep)
```

### 6. V3 hash quality is plausible

`linr_v3_then_v2` recall@10 in filter sweeps should be in [0.7, 0.99].
Lower than 0.7 means V3's 1-bit hash is missing real top-K under
filter — tunable via `algo_params.linr_v3_then_v2.candidate_pool` in
the YAML (currently 8000); raise to 16000 if needed.

```bash
jq '[.filter_eval[] | select(.impl=="linr_v3_then_v2" and .k==10) | ."recall@10"] | "min=\(min) max=\(max)"' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
```

### 7. Filter sweep coverage is plausible

`n_users_kept` should be a non-trivial fraction of `n_users` for every
sweep — sweeps that drop too many users due to all-`-1` query attrs
indicate a synthesis bug.

```bash
jq '[.filter_eval[] | "\(.filter_kind)/\(.sweep) kept=\(.n_users_kept)"] | unique[]' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/eval_goodreads_retrieval.json
```

Floors (per sweep, of 313 178 total users):

- `clause/c0_genre` ≥ 280 k (genre coverage 93 %)
- `clause/c1_lang_reverse` ≥ 200 k (lang coverage 64 %)
- `clause/c3_year` ≥ 250 k (year coverage 84 %)
- `clause/c0c4_author` ≥ 280 k (genre × author intersect)
- `bloom/1shelf` ≥ 290 k (wide coverage 99.7 %, drop 189 in prep)
- `bloom/2shelf` ≥ 290 k (drop 505 in prep)
- `combined/c0_genre_x_1shelf` ≥ 280 k

---

## Final report (after all three dims complete)

Build a single comparison table for the user. Group by
`(filter_kind, sweep, impl, k=10, batch_size=1)`; columns are d64 / d128
/ d256.

Suggested format:

```
=== Quality suite (no filter, held-out target) ===
                          d64       d128      d256
torch_fullscan recall@10   0.0333    0.0340    0.0343  (yambda parity)
linr_v3_then_v2 recall@10  ?         ?         ?       (V3 quality, no filter)
silvertorch recall@10      ?         ?         ?       (IVF + INT8, no bloom)

=== Filter suite (filtered-FullScan oracle as GT) ===
                                                 d64       d128      d256
clause/c0_genre / linr_v2_filter_compact ms       ?         ?         ?
clause/c0_genre / linr_v3_then_v2 recall@10       ?         ?         ?
bloom/1shelf / linr_v2_filter_compact ms          ?         ?         ?
bloom/1shelf / linr_v3_then_v2 recall@10          ?         ?         ?
bloom/1shelf / silvertorch recall@10              ?         ?         ?
combined/c0_genre_x_1shelf / linr_v2 ms           ?         ?         ?
combined/c0_genre_x_1shelf / linr_v3 recall@10    ?         ?         ?
combined/c0_genre_x_1shelf / silvertorch r@10     ?         ?         ?
```

Use `jq` to extract; one-liner per row. Don't paste the raw JSON — it's
~2000 lines per dim.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RuntimeError: CUDA out of memory` during oracle | batch too large at d256 | drop `compute_filtered_oracle(batch_size=)` from 64 to 32 in [eval_goodreads_retrieval.py](../../evaluation/retrieval/eval_goodreads_retrieval.py); rerun |
| `eval_split rows ≠ test rows` | stale `eval_split.parquet` | regen: `uv run python -m data.goodreads attrs --processed-dir <…> --output-dir data/goodreads/work-id` |
| `filter_kind=bloom + sweep ... clause N is reverse` | YAML bug (bloom + reverse clause) | fix YAML — bloom is paper-strict, no NOT |
| `linr_v2_filter_compact recall < 1.0` on any cell | bug in `evaluate_indices` cascade | escalate, do not work around |
| Oracle file `gt_topk_<sweep>.pt` exists but shape mismatch | stale cache from a different dim | delete the stale file in `data/goodreads/work-id/gt/`; rerun |

---

## Out of scope for this runbook

- Code changes — the harness is functionally complete; only cosmetic
  refactor is in flight. Do **not** touch `filter_registry.py`,
  `eval_goodreads_retrieval.py`, or any YAML beyond the per-cell debug
  flags above.
- Hyperparam tuning — current `candidate_pool=8000`, `n_lists=1024`,
  `n_probe=24`, `m_bits=1024`, `k_hash=5` are bench-thesis defaults.
  Only adjust `candidate_pool` if step 6 verification fails on every
  sweep (a recall floor of 0.7 indicates V3 is fundamentally
  underprovisioned, not a per-sweep issue).
- Re-running data prep (`goodreads.py attrs` /  `prep`) — artifacts on
  disk are validated and final.
