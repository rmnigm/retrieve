# Arxiv filter-eval — execution runbook

## Context

The arxiv filter-bench is functionally complete: the unified driver
([retrieval/evaluate.py](../../evaluation/retrieval/evaluate.py)) handles
arxiv via config-shape dispatch, the YAML
([conf/arxiv-d256.yaml](../../evaluation/conf/arxiv-d256.yaml)) is wired
with all 12 filter sweeps, and pre-encoded text/query embeddings sit on
disk under `data/arxiv/papers/`. This runbook is for the agent that will
launch the full bench end-to-end and produce the comparison table.

Unlike goodreads (three SASRec checkpoints across d64/d128/d256), arxiv
is **single-dim** (d256, nomic-embed-text-v1.5 truncated). There is no
SASRec model and no yambda regression baseline — the unfiltered
`none/full_scan` cell is itself the quality reference for future runs.

The bench answers two questions on one set of pre-encoded embeddings:

1. **Unfiltered baseline (`filter_kind=none`, `sweep=full_scan`)**: how
   well does each retrieval algorithm find the held-out target paper from
   its own abstract embedding? `torch_fullscan recall@10` is the
   reference; record it for future regressions.
2. **Filter (oracle-based)**: how well does each algo + filter-module
   combination approximate brute-force filtered top-K? GT is the
   filtered-FullScan top-K, cached per sweep at
   `data/arxiv/papers/gt/gt_topk_<sweep>.pt`. Held-out target is
   degenerate under filtering (filters can mask the only positive out).

Spec: no standalone arxiv design doc; the bench is a structural mirror of
[goodreads-filter-eval.md](goodreads-filter-eval.md).
Driver: [retrieval/evaluate.py](../../evaluation/retrieval/evaluate.py).
Algo wrapper: [retrieval/algo_registry.py](../../evaluation/retrieval/algo_registry.py).
Config: [conf/arxiv-d256.yaml](../../evaluation/conf/arxiv-d256.yaml).

### Algos per suite

| Suite | Algorithms | Type |
|-------|-----------|------|
| Unfiltered (`filter_kind=none`, `full_scan`) | `torch_fullscan`, `linr_v3_then_v2`, `silvertorch` | exact, approx, approx |
| Filter (`clause` / `bloom` / `combined`) | `linr_v3_then_v2`, `silvertorch` (wide / combined only) | approx, approx |

`torch_fullscan` is dropped from filter sweeps (= filtered-FullScan
oracle, recall=1.0 trivially) via `algo_skip()`
([evaluate.py:410](../../evaluation/retrieval/evaluate.py#L410)).
`silvertorch` skips narrow-only sweeps automatically via
`SilvertorchSkippedOnNarrow` (per the bench thesis: bloom is wide-only).

The arxiv algorithms list does **not** include `linr_v2_filter_compact`,
so there is no per-cell exact-recall sanity check. Plausibility floors
take its place (see Verification §4 below).

---

## Pre-flight checklist

Run this once before kicking off the bench. Each step is a fast file
check or import — no GPU work.

```bash
cd /workspace/retrieve/evaluation

# 1. Pre-encoded embeddings present (item + query side).
for f in content/text_emb.pt content/text_emb.meta.json \
         content/query_emb.pt content/query_emb.meta.json \
         heldout.parquet; do
  test -f data/arxiv/papers/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 2. Filter / attribute artifacts present.
for f in item_attrs_narrow.pt item_attrs_wide.pt \
         clause_is_reverse_narrow.pt eval_split.parquet; do
  test -f data/arxiv/papers/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 3. Encoder prefix sidecars match. The driver runs assert_arxiv_prefixes
#    at startup ([evaluate.py:65](../../evaluation/retrieval/evaluate.py#L65));
#    a doc/query prefix swap silently degrades recall by ~5–15%, so worth
#    catching in pre-flight too.
uv run python -c "
import json, pathlib
d = pathlib.Path('data/arxiv/papers/content')
t = json.load(open(d/'text_emb.meta.json'))
q = json.load(open(d/'query_emb.meta.json'))
assert t['prefix'] == 'search_document: ', t
assert q['prefix'] == 'search_query: ',     q
assert t.get('encoder') == q.get('encoder'), (t, q)
print('OK  prefixes + encoder match')
"

# 4. Config parses and matches expectations.
uv run python -c "
from retrieval.config import load_eval_config
from pathlib import Path
cfg = load_eval_config(Path('conf/arxiv-d256.yaml'))
assert cfg.checkpoint is None, 'arxiv has no SASRec checkpoint'
assert 'linr_v3_then_v2' in cfg.algorithms
assert 'silvertorch'     in cfg.algorithms
assert set(cfg.filters) == {'none','clause','bloom','combined'}
print('OK  conf/arxiv-d256.yaml')
"

# 5. Driver + algo_registry import cleanly.
uv run python -c "
import retrieval.evaluate
from retrieval.algo_registry import build_filter_modules, build_filtered_algorithm
print('OK  driver + algo_registry import')
"
```

If any check fails: stop and report. Do **not** attempt to regenerate
embeddings or attribute tensors — re-encoding is a data-prep task
(`uv run arxiv encode_text` / `encode_queries` / `attrs`), not part of
the eval runbook. Escalate.

---

## Execution

Single dim → single launch. No d64-first sanity gate.

```bash
cd /workspace/retrieve/evaluation
uv run evaluate --config conf/arxiv-d256.yaml
```

Wall-clock estimate: oracle compute is 11 filter sweeps × 10 K queries ×
~2.99 M items (batched at 64 — see
[evaluate.py:329](../../evaluation/retrieval/evaluate.py#L329)). Roughly
1/8 the per-sweep oracle cost of goodreads but with comparable sweep
count → expect 15–30 minutes total on a single GPU. Each sweep's oracle
is cached at `data/arxiv/papers/gt/gt_topk_<sweep>.pt` and reused on
re-runs.

### Background execution recipe

Long-running; use `Bash` with `run_in_background=true` and `Monitor` on
the resulting shell ID. Do **not** poll with `sleep` — get a single
notification on completion and check the output file.

```bash
uv run evaluate --config conf/arxiv-d256.yaml 2>&1 | tee /tmp/bench-arxiv.log
```

**Output JSON path**: `data/arxiv/papers/evaluate.json` (since
`output: null` in the YAML and there is no checkpoint dir to default
into; resolved at [evaluate.py:467-472](../../evaluation/retrieval/evaluate.py#L467-L472)).

> A partial test output already lives at
> `data/arxiv/papers/eval_arxiv_retrieval.json` (9 rows, single sweep,
> `suite="arxiv_filter"` from an older driver version). The new run
> writes to `evaluate.json` and emits `suite="filter"`. Don't confuse
> the two; archive or delete the stale file before kicking off.

### Single-cell debugging

If a specific cell fails, narrow the run:

```bash
uv run evaluate \
  --config conf/arxiv-d256.yaml \
  --filter-kind clause \
  --sweep c0_maincat \
  --algorithms linr_v3_then_v2 \
  --output /tmp/arxiv-debug.json
```

`--algorithms` **replaces** (not merges with) the YAML list.
`--filter-kind` and `--sweep` filter to a single slice. `--output`
overrides the destination so debug runs don't clobber the full output.

---

## Verification (after the run completes)

The output JSON is a flat list of row dicts. Every row carries `suite`,
`filter_kind`, `sweep`, `impl`, `k`, `batch_size`, `n_users_kept`,
latency / memory fields, and `recall@k` / `ndcg@k`. Skip rows have
`"skipped": true` only.

Check each item below. **All five must pass before reporting results.**
A `jq` one-liner is given for each. All paths assume cwd
`/workspace/retrieve/evaluation`.

### 1. Unfiltered baseline plausibility

`filter_kind=none`, `sweep=full_scan`, `impl=torch_fullscan`,
`k=10`, `batch_size=1` row exists with `recall@10 > 0`. There is no
yambda baseline to compare against — record the value as the reference
for future regressions.

```bash
jq '[.[] | select(.filter_kind=="none" and .sweep=="full_scan"
                  and .impl=="torch_fullscan" and .k==10 and .batch_size==1)
        | {recall:."recall@10", ndcg:."ndcg@10"}]' \
  data/arxiv/papers/evaluate.json
```

Also verify the approx algos are within a sane band of fullscan on the
same cell (typically ≥ 0.95 × fullscan recall@10):

```bash
jq '[.[] | select(.filter_kind=="none" and .sweep=="full_scan"
                  and .k==10 and .batch_size==1)
        | {impl, recall:."recall@10"}]' \
  data/arxiv/papers/evaluate.json
```

### 2. No torch_fullscan rows in filter sweeps

```bash
jq '[.[] | select(.filter_kind!="none" and .impl=="torch_fullscan"
                  and (.skipped // false | not))] | length' \
  data/arxiv/papers/evaluate.json
# Expect 0
```

### 3. silvertorch skipped on narrow-only sweeps

Narrow-only clause sweeps: `c0_maincat`, `c2_year`, `c3_nversions`,
`c4_author`, `c0c2`, `c0c4`, `all5` (7 total). Each should have one
skip row per silvertorch algo, not perf rows.

```bash
jq '[.[] | select(.filter_kind=="clause" and .impl=="silvertorch"
                  and (.skipped // false))] | length' \
  data/arxiv/papers/evaluate.json
# Expect 7
```

### 4. V3 hash quality is plausible

`linr_v3_then_v2 recall@10` across all filtered sweeps should land in
[0.7, 1.0]. The existing single-cell test (`c0_maincat`) reaches 0.987,
so most cells should clear comfortably. If any sweep dips below 0.7,
raise `algo_params.linr_v3_then_v2.candidate_pool`
([conf/arxiv-d256.yaml:25](../../evaluation/conf/arxiv-d256.yaml#L25),
currently 12000) toward 16000–24000.

```bash
jq '[.[] | select(.filter_kind!="none" and .impl=="linr_v3_then_v2"
                  and .k==10 and (.skipped // false | not))
        | ."recall@10"] | "min=\(min) max=\(max) avg=\(add/length)"' \
  data/arxiv/papers/evaluate.json
```

### 5. Filter sweep coverage is plausible

`n_users_kept` should be a non-trivial fraction of 10,000 for every
sweep — sweeps that drop too many users due to all-`-1` query attrs or
empty wide bags indicate a synthesis bug.

```bash
jq '[.[] | select((.skipped // false | not))
        | "\(.filter_kind)/\(.sweep) kept=\(.n_users_kept)"] | unique[]' \
  data/arxiv/papers/evaluate.json
```

Floors (per sweep, of 10,000 total queries — see
[prep_log.json](../../evaluation/data/arxiv/papers/prep_log.json) for
the source coverage numbers):

- `clause/c0_maincat`, `c2_year`, `c3_nversions`, `c4_author`, `c0c2`,
  `c0c4`, `all5` ≥ 9,990 (all 5 narrow clauses have 100 % coverage)
- `bloom/1shelf` = 10,000 (`eval_drop_1shelf: 0`)
- `bloom/2shelf` ≈ 4,735 (`eval_drop_2shelf: 5,265` — half the queries
  have <2 leaf categories and are dropped by construction)
- `combined/c0_maincat_x_1shelf` ≥ 9,990
- `combined/c2_year_x_2shelf` ≈ 4,735

If clause-only sweeps drop below 9,990 or 2-shelf sweeps drop below
4,000, the synthesis path in `build_sweep_qa`
([evaluate.py:213](../../evaluation/retrieval/evaluate.py#L213)) or
`eval_split.parquet` is suspect — escalate before re-running.

---

## Final report (after the run completes)

Build a single comparison table for the user. Group by
`(filter_kind, sweep, impl)` × `k=10`, `batch_size=1`. One column for
recall and one for median latency.

Suggested format:

```
=== Unfiltered (none/full_scan, held-out target) ===
                                d256
torch_fullscan  recall@10        ?
linr_v3_then_v2 recall@10        ?
silvertorch     recall@10        ?

=== Filter suite (filtered-FullScan oracle as GT) ===
                                              recall@10  median ms (bs=1)
clause/c0_maincat       linr_v3_then_v2          ?           ?
clause/c2_year          linr_v3_then_v2          ?           ?
clause/c3_nversions     linr_v3_then_v2          ?           ?
clause/c4_author        linr_v3_then_v2          ?           ?
clause/c0c2             linr_v3_then_v2          ?           ?
clause/c0c4             linr_v3_then_v2          ?           ?
clause/all5             linr_v3_then_v2          ?           ?
bloom/1shelf            linr_v3_then_v2          ?           ?
bloom/1shelf            silvertorch              ?           ?
bloom/2shelf            linr_v3_then_v2          ?           ?
bloom/2shelf            silvertorch              ?           ?
combined/c0_maincat_x_1shelf  linr_v3_then_v2    ?           ?
combined/c0_maincat_x_1shelf  silvertorch        ?           ?
combined/c2_year_x_2shelf     linr_v3_then_v2    ?           ?
combined/c2_year_x_2shelf     silvertorch        ?           ?
```

Use `jq` to extract; one-liner per row. Don't paste the raw JSON — it's
hundreds of rows.

---

## Common failure modes

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `RuntimeError: query_emb dim=X ≠ item_embs dim=Y` | encoder/dim mismatch | re-run `arxiv encode_text` and `arxiv encode_queries` with matching `--truncate-dim` |
| `RuntimeError: query_emb rows=X ≠ heldout rows=Y` | stale `query_emb.pt` after heldout regen | re-run `arxiv encode_queries` |
| `RuntimeError: text_emb.meta.json prefix=...` mismatch | doc/query prefix swap (silently kills recall) | re-run `arxiv encode_*` with the correct prefix |
| `eval_split rows ≠ test rows` | stale `eval_split.parquet` | regen: `uv run arxiv attrs --output-dir data/arxiv/papers` |
| `RuntimeError: CUDA out of memory` during oracle | N=2.99M × default batch_size=64 too large | drop `compute_filtered_oracle(batch_size=)` from 64 to 32 in [evaluate.py:329](../../evaluation/retrieval/evaluate.py#L329); rerun |
| Oracle file `gt_topk_<sweep>.pt` exists but shape mismatch | leftover from a prior partial run | delete the stale file in `data/arxiv/papers/gt/`; rerun |
| `linr_v3_then_v2 recall@10 < 0.7` on every sweep | candidate_pool underprovisioned for N=3M | bump `candidate_pool` 12000 → 16000–24000 in YAML |
| `filter_kind=bloom + sweep ... clause N is reverse` | YAML bug (bloom + reverse clause) | fix YAML — bloom is paper-strict, no NOT (note: arxiv has zero reverse clauses today) |

---

## Out of scope for this runbook

- Code changes — the harness is functionally complete. Do **not** touch
  `algo_registry.py`, `evaluate.py`, or the YAML beyond the per-cell
  debug flags above.
- Hyperparam tuning — current `candidate_pool=12000`, `n_lists=1664`,
  `n_probe=32`, `m_bits=1024`, `k_hash=5` are bench-thesis defaults.
  Only adjust `candidate_pool` if step 4 verification fails on every
  sweep (a recall floor of 0.7 indicates V3 is fundamentally
  underprovisioned, not a per-sweep issue).
- Re-running data prep (`arxiv prep` / `attrs` / `encode_text` /
  `encode_queries`) — artifacts on disk are validated and final.
- Adding `linr_v2_filter_compact` to the arxiv algorithms list — would
  require a config change plus thinking through whether the V2
  candidate cascade scales to N=3M, both out of scope here.
