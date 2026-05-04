# Arxiv filter-eval — execution runbook

## Context

The arxiv filter-bench is structurally identical to the goodreads bench
([goodreads-filter-eval.md](goodreads-filter-eval.md)) — same unified
driver, same `build_algorithm`, same `evaluate.json` row schema. Two
arxiv-specific differences:

- **No SASRec**: queries are pre-encoded text embeddings on disk
  (`data/arxiv/papers/content/query_emb.pt`, built by
  `arxiv encode_queries`). The driver branches on config shape:
  `query_emb_path` set + `checkpoint` unset → load text embs.
- **Single dim** (d256, nomic-embed-text-v1.5 truncate-256). No d64/d128
  variants. The unfiltered `none/full_scan` cell is itself the quality
  reference for future runs — there's no yambda-style training-time
  baseline to cross-check against.

The bench answers, per filter sweep:

1. **Quality vs the held-out target** on the unfiltered cell.
2. **Quality vs the filtered-FullScan oracle** on every filter cell.
3. **Latency / memory** across `(algo, k, batch_size)` per cell.

Same unified algorithm dispatch as goodreads:

| algo | unfiltered | filter cell |
|---|---|---|
| `torch_fullscan` | matmul + topk | matmul → topk → post-filter mask |
| `linr_v1_filter_mask` | V1 dense matmul + topk | V1 + `masked_fill(-inf)` → topk = exact filtered top-K |
| `linr_v3_then_v2` | V3 1-bit → V2 rerank | same cascade with mask through V3 |
| `silvertorch` | plain IVF + topk | bloom-fused IVF on `bloom` kind (codesigned), plain IVF + post-mask on `clause` |
| `voyager_hnsw` | HNSW topk (CPU) | HNSW topk + post-filter mask |

The only `algo_skip` rule: `linr_v2_filter_compact` requires a filter
(not in the default arxiv algo list anyway).

## Sweep matrix

**One attribute tensor everywhere**: `item_attrs_narrow.pt` of shape
`[N, 4, 4]` int64 backs both `clause` and `bloom` filter_kinds; the
sweep specifies which `active_clauses` are queried, the kind selects
which algo runs on top (ClauseIndex exact vs BloomFilter approximate).
There is no separate "wide" attribute tensor in active use.

The 4 narrow clauses are all forward — arxiv has no reverse clauses,
so **every active-clause subset is bloom-eligible**.

| C | clause | A_max | source |
|---|---|---|---|
| 0 | main category | 1 | top-level arxiv category (cs, math, physics, …) |
| 1 | license | 1 | CC-BY-* / arxiv-default / unknown |
| 2 | publication year | 1 | bucketed |
| 3 | n_versions | 1 | bucketed (1, 2, 3+) |

```
filters:
  none:   [full_scan]
  clause: [c0_maincat, c2_year, c3_nversions, c0c2, all4]
  bloom:  [c0_maincat, c2_year, c3_nversions, c0c2, all4]
```

Same sweep names on both kinds → exact-vs-bloom comparison directly
side-by-side per cell.

## Pre-flight checklist

Run from `/workspace/retrieve/evaluation`:

```bash
# 1. Pre-encoded embeddings + heldout split.
for f in content/text_emb.pt content/text_emb.meta.json \
         content/query_emb.pt content/query_emb.meta.json \
         heldout.parquet; do
  test -f data/arxiv/papers/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 2. Filter / attribute tensors. Only the narrow tensor is used —
#    the bench has no wide-attribute path right now.
for f in item_attrs_narrow.pt clause_is_reverse_narrow.pt eval_split.parquet; do
  test -f data/arxiv/papers/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 3. Encoder prefix sidecars match. The driver's
#    `assert_arxiv_prefixes` runs at startup; doc/query prefix swap
#    silently degrades recall by 5–15%, worth catching in pre-flight.
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

# 4. Config parses, narrow shapes line up.
uv run python -c "
import torch
from retrieval.config import load_eval_config
from pathlib import Path
cfg = load_eval_config(Path('conf/arxiv/d256-filter.yaml'))
assert cfg.checkpoint is None, 'arxiv has no SASRec checkpoint'
assert set(cfg.filters) == {'none','clause','bloom'}
algos = set(cfg.algorithms)
assert {'torch_fullscan','linr_v1_filter_mask','linr_v3_then_v2','silvertorch','voyager_hnsw'} <= algos, cfg.algorithms
ia = torch.load('data/arxiv/papers/item_attrs_narrow.pt')
isr = torch.load('data/arxiv/papers/clause_is_reverse_narrow.pt')
assert ia.dim() == 3 and ia.shape[1] == 4, ia.shape  # 4 clauses
assert isr.numel() == 4 and not isr.any(), isr.tolist()  # all forward
print('OK  conf + tensors')
"

# 5. Driver imports.
uv run python -c "
import retrieval.evaluate
from retrieval.algo_registry import build_algorithm, build_filter_modules
print('OK  imports')
"
```

If any step fails: stop and report. Don't regenerate embeddings or
attribute tensors — that's a data-prep task (`uv run arxiv encode_text`,
`encode_queries`, `attrs`), not part of the eval runbook. Escalate.

## Execution

Single dim → single launch. Wall-clock estimate: ~30–45 min with
quality, ~5 min with `--skip-quality` (no oracle build, perf timing
only).

```bash
cd /workspace/retrieve/evaluation
uv run evaluate --config conf/arxiv/d256-filter.yaml \
  > /tmp/bench-arxiv.log 2>&1
```

Use `Bash` with `run_in_background=true` and `Monitor` on the resulting
shell ID. Don't poll with `sleep`.

**Output JSON path**: `data/arxiv/papers/evaluate.json` (since
`output: null` in the YAML and there is no checkpoint dir).

> Any older artefacts named `eval_arxiv_retrieval.json` predate the
> driver unification — delete or archive before running so they don't
> get confused with the new output.

### Per-cell debugging

If a specific cell fails, narrow the run:

```bash
uv run evaluate \
  --config conf/arxiv/d256-filter.yaml \
  --filter-kind clause \
  --sweep c0_maincat \
  --algorithms linr_v1_filter_mask \
  --output /tmp/arxiv-debug.json
```

`--algorithms` **replaces** (not merges with) the YAML list.
`--filter-kind` is repeatable (`--filter-kind clause --filter-kind bloom`)
to skip the unfiltered cell.

### Skipping quality for fast latency-only runs

Add `--skip-quality` to skip the quality stream and the oracle build.
Recall/NDCG report `NaN`; perf rows are fully populated. ~5 min wall.

### Cache hygiene between dims

Arxiv uses a single dim, so the `data/arxiv/papers/gt/` cache is dim-correct
once built. Re-running with the same `item_embs` reuses the cache. Delete
`data/arxiv/papers/gt/` only if `item_embs` (i.e. text encoder output)
changes — re-encoding `text_emb.pt` does invalidate it.

## Verification (after the run completes)

The output is a flat JSON list. Each row carries `suite`, `filter_kind`,
`sweep`, `impl`, `k`, `batch_size`, `n_users_kept`, latency / memory
fields, and `recall@k` / `ndcg@k`. Skip rows have `"skipped": true`.

All checks below should pass before reporting results. `jq` one-liner per
check, all paths assume cwd `/workspace/retrieve/evaluation`.

### 1. Unfiltered baseline plausibility

`filter_kind="none", sweep="full_scan", impl="torch_fullscan", k=10,
batch_size=1` row exists with `recall@10 > 0`. There is no training-time
baseline to compare against; record the value as the future regression
reference.

```bash
jq '[.[] | select(.filter_kind=="none" and .sweep=="full_scan"
                  and .impl=="torch_fullscan" and .k==10 and .batch_size==1)
        | {recall:."recall@10", ndcg:."ndcg@10"}]' \
  data/arxiv/papers/evaluate.json
```

Approximate algos on the same unfiltered cell should be within a sane
band of fullscan (typically `recall@10 ≥ 0.95 × fullscan`):

```bash
jq '[.[] | select(.filter_kind=="none" and .sweep=="full_scan"
                  and .k==10 and .batch_size==1)
        | {impl, recall:."recall@10"}]' \
  data/arxiv/papers/evaluate.json
```

### 2. V1 pre-filter is exact on every filter cell

`linr_v1_filter_mask recall@k = 1.0` (≥0.999) on every filter cell —
same scoring + same mask as the oracle by construction. Anything else
indicates a kernel/mask bug.

```bash
jq '[.[] | select(.impl=="linr_v1_filter_mask" and .filter_kind!="none")
        | ."recall@10"] | min' \
  data/arxiv/papers/evaluate.json
# Expect ≥ 0.999
```

### 3. Post-filter degradation is observable

`torch_fullscan` and `voyager_hnsw` on filter cells run as **post-filter**
(matmul → topk → mask). On selective filters their `recall@k` should
drop noticeably below `linr_v1_filter_mask` — that drop is the metric
the bench is designed to surface.

```bash
jq '[.[] | select(.k==10 and .batch_size==1
                  and .impl=="torch_fullscan" and .filter_kind!="none")
        | "\(.filter_kind)/\(.sweep) recall=\(."recall@10")"] | unique[]' \
  data/arxiv/papers/evaluate.json
```

### 4. V3 hash quality is plausible

`linr_v3_then_v2 recall@10` across all filtered sweeps in `[0.7, 1.0]`.
Below 0.7 means V3's 1-bit hash is missing real top-K; tunable via
`algo_params.linr_v3_then_v2.candidate_pool` in the YAML (currently
12000); raise to 16000–24000 if needed.

```bash
jq '[.[] | select(.filter_kind!="none" and .impl=="linr_v3_then_v2"
                  and .k==10 and (.skipped // false | not))
        | ."recall@10"] | "min=\(min) max=\(max) avg=\(add/length)"' \
  data/arxiv/papers/evaluate.json
```

### 5. Codesigned silvertorch on `bloom` kind

`silvertorch` on `bloom` cells runs the bloom-fused IVF probe (the
codesigned path); on `clause` cells it runs plain IVF + post-mask.
The bloom-fused path's per-query `query_clause_attrs=qa_narrow` engages
the inline bloom check; recall reflects the IVF + bloom-FP combined
loss, which on this catalog is typically high (>0.8 at k=10).

```bash
jq '[.[] | select(.filter_kind=="bloom" and .impl=="silvertorch"
                  and .k==10 and .batch_size==1)
        | "\(.sweep) recall=\(."recall@10") ms=\(.median_ms)"] | unique[]' \
  data/arxiv/papers/evaluate.json
```

### 6. Sweep coverage is plausible

`n_users_kept` should be a non-trivial fraction of 10 000 for every
sweep. All-`-1` qa rows are dropped via `skip_mask`.

```bash
jq '[.[] | select((.skipped // false | not))
        | "\(.filter_kind)/\(.sweep) kept=\(.n_users_kept)"] | unique[]' \
  data/arxiv/papers/evaluate.json
```

Floors (per arxiv `prep_log.json`'s coverage):
- All clause / bloom sweeps ≥ 9 990 (every narrow clause has ~100 %
  coverage on arxiv).

If anything drops below 9 000, the sweep synthesis or `eval_split.parquet`
is suspect — escalate before re-running.

## Final report

Build a single comparison table for the user. Group by
`(filter_kind, sweep, impl)` × `k=10`, `batch_size=1`. Two columns:
`recall@10` and `median ms`.

```
=== Unfiltered (none/full_scan, held-out target) ===
                                d256
torch_fullscan  recall@10        ?       ms ?
linr_v1_filter_mask  recall@10   ?       ms ?    (= torch_fullscan numerically)
linr_v3_then_v2 recall@10        ?       ms ?
silvertorch     recall@10        ?       ms ?
voyager_hnsw    recall@10        ?       ms ?

=== Filter suite (vs filtered-FullScan oracle) ===
                                              recall@10  median ms (bs=1)
clause/c0_maincat        torch_fullscan          ?           ?
clause/c0_maincat        linr_v1_filter_mask     1.0         ?
clause/c0_maincat        linr_v3_then_v2         ?           ?
clause/c0_maincat        silvertorch             ?           ?
clause/c0_maincat        voyager_hnsw            ?           ?
…
bloom/c0_maincat         torch_fullscan          ?           ?
bloom/c0_maincat         linr_v1_filter_mask     1.0         ?
bloom/c0_maincat         linr_v3_then_v2         ?           ?
bloom/c0_maincat         silvertorch             ?           ?      (codesigned)
bloom/c0_maincat         voyager_hnsw            ?           ?
…
```

The interesting deltas are:
- `torch_fullscan` / `voyager_hnsw` post-filter recall vs `linr_v1_filter_mask`'s exact 1.0
- `silvertorch` clause (post-mask) vs `silvertorch` bloom (codesigned) — same predicate, two algorithm strategies
- Per-sweep variance as filter selectivity increases (`c0_maincat` loose → `all4` tight)

Use `jq` to extract; one-liner per row. Do **not** paste the raw JSON.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: query_emb dim=X ≠ item_embs dim=Y` | encoder/dim mismatch | re-run `arxiv encode_text` and `arxiv encode_queries` with matching `--truncate-dims` |
| `RuntimeError: query_emb rows=X ≠ heldout rows=Y` | stale `query_emb.pt` after heldout regen | re-run `arxiv encode_queries` |
| `RuntimeError: text_emb.meta.json prefix=...` mismatch | doc/query prefix swap | re-run `arxiv encode_*` with the correct prefix |
| `eval_split rows ≠ test rows` | stale `eval_split.parquet` | regen: `uv run arxiv attrs --output-dir data/arxiv/papers` |
| `RuntimeError: CUDA out of memory` during oracle | N=2.99M × default `batch_size=64` too large | drop `compute_filtered_oracle(batch_size=)` from 64 to 32; rerun |
| Oracle file `gt_topk_<sweep>.pt` exists but shape mismatch | leftover from prior partial run | delete the stale file in `data/arxiv/papers/gt/`; rerun |
| `linr_v3_then_v2 recall@10 < 0.7` on every sweep | `candidate_pool` underprovisioned for N≈3M | bump `candidate_pool` 12000 → 16000–24000 in YAML |
| `linr_v1_filter_mask recall@k < 0.999` on any cell | mask/oracle desync (e.g. item-0 sentinel) | escalate; this is a regression in the unified mask path |

## Out of scope

- Code changes — the harness is functionally complete. Do not touch
  `algo_registry.py`, `evaluate.py`, or the YAML beyond the per-cell
  debug flags above.
- Hyperparam tuning — current `candidate_pool=12000`, `n_lists=1664`,
  `n_probe=32`, `m_bits=1024`, `k_hash=5` are bench-thesis defaults.
  Only adjust `candidate_pool` if step 4 verification fails on every
  sweep.
- Re-running data prep (`arxiv prep` / `attrs` / `encode_text` /
  `encode_queries`) — artifacts on disk are validated and final.
- Adding `linr_v2_filter_compact` to the algo list — V2's
  variable-shape bmm at bs=1 is much slower than V1's GEMV on this
  catalog (see goodreads-filter-eval.md notes).
- Wide-bloom sweeps — the bench has no wide path. There's no separate
  attribute tensor or sweep type beyond the narrow stack described
  here. If `item_attrs_wide.pt` exists on disk from older data-prep
  runs, ignore it.
