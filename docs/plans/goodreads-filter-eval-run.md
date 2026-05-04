# Goodreads filter-eval — execution runbook

## Context

Companion runbook to [arxiv-filter-eval-run.md](arxiv-filter-eval-run.md);
same unified driver, same `build_algorithm`, same `evaluate.json` row
schema. Goodreads-specific differences:

- **SASRec checkpoints** (not pre-encoded text embs). The driver loads
  `<checkpoint>.pt`, encodes the test parquet via `model.predict_last`,
  and feeds the resulting query embeddings through the algos.
- **Three dims** (d64 / d128 / d256, ID-only, drop0.5). Run them
  sequentially via the chain script (`/tmp/run_filter_benches.sh`) or
  one at a time.
- **313k test users** for the unfiltered cell; **50k subsample** for
  filter cells (`filter_users_limit: 50000` in each YAML) so the
  per-cell quality stream stays under ~3 min.

The bench answers, per filter sweep:

1. **Quality vs the held-out target** on `none/full_scan` — the yambda
   parity check (must match `eval_quality.json` to ±0.0005 on
   `torch_fullscan recall@10`).
2. **Quality vs the filtered-FullScan oracle** on every filter cell.
3. **Latency / memory** across `(algo, k, batch_size)`.

Same unified algorithm dispatch as arxiv:

| algo | unfiltered | filter cell |
|---|---|---|
| `torch_fullscan` | matmul + topk | matmul → topk → post-filter mask |
| `linr_v1_filter_mask` | V1 dense matmul + topk | V1 + `masked_fill(-inf)` → topk = exact filtered top-K |
| `linr_v3_then_v2` | V3 1-bit → V2 rerank | same cascade with mask through V3 |
| `silvertorch` | plain IVF + topk | bloom-fused IVF on `bloom` kind (codesigned), plain IVF + post-mask on `clause` |
| `voyager_hnsw` | HNSW topk (CPU) | HNSW topk + post-filter mask |

The only `algo_skip` rule: `linr_v2_filter_compact` requires a filter
(not in the default goodreads algo list anyway).

## Sweep matrix

**One attribute tensor everywhere**: `item_attrs_narrow.pt` of shape
`[N, 4, 4]` int64 backs both `clause` and `bloom` filter_kinds; the
sweep specifies which `active_clauses` are queried, the kind selects
which algo runs on top (ClauseIndex exact vs BloomFilter approximate).
There is no separate "wide" attribute tensor in active use.

The 4 narrow clauses are:

| C | clause | A_max | reverse | source |
|---|---|---|---|---|
| 0 | genre / shelf bucket | 4 | no | top genres aggregated across editions |
| 1 | language code | 1 | **yes** | top-30 langs + `-1` for rest |
| 2 | format bucket | 1 | no | paperback / hardcover / ebook / audio / other |
| 3 | publication year | 1 | no | bucketed |

`clause_is_reverse_narrow.pt = [False, True, False, False]`. The author
clause was removed (single-cardinality forward clause that didn't add
bench signal).

```
filters:
  none:   [full_scan]
  clause: [c0_genre, c1_lang_reverse, c2_format, c3_year, c0c1, c0c3, all_fwd, all4]
  bloom:  [c0_genre,                  c2_format, c3_year,       c0c3, all_fwd        ]
```

`bloom` excludes any sweep that activates the reverse clause (c1) —
BloomFilter is paper-strict (no NOT). Where a sweep is bloom-eligible,
the same sweep name appears under both `clause` and `bloom` so the
exact-vs-bloom comparison is directly side-by-side per cell.

## Pre-flight checklist

Run from `/workspace/retrieve/evaluation`:

```bash
# 1. Checkpoints present (all three).
for d in d64 d128 d256; do
  test -f checkpoints/goodreads-gsasrec-$d-drop0.5-id/best_model.pt \
    || { echo "MISSING: $d checkpoint"; exit 1; }
done

# 2. Filter / attribute tensors. Only the narrow tensor is used —
#    the bench has no wide-attribute path right now.
for f in item_attrs_narrow.pt clause_is_reverse_narrow.pt eval_split.parquet; do
  test -f data/goodreads/work-id/$f \
    || { echo "MISSING: $f"; exit 1; }
done

# 3. Configs parse + tensors line up.
uv run python -c "
import torch
from retrieval.config import load_eval_config
from pathlib import Path
for p in ['conf/goodreads/d64-filter.yaml',
          'conf/goodreads/d128-filter.yaml',
          'conf/goodreads/d256-filter.yaml']:
    cfg = load_eval_config(Path(p))
    assert cfg.checkpoint and cfg.checkpoint.endswith('-drop0.5-id/best_model.pt'), p
    assert set(cfg.filters) == {'none','clause','bloom'}, p
    algos = set(cfg.algorithms)
    assert {'torch_fullscan','linr_v1_filter_mask','linr_v3_then_v2','silvertorch','voyager_hnsw'} <= algos, p
    print(f'OK  {p}')
ia = torch.load('data/goodreads/work-id/item_attrs_narrow.pt')
isr = torch.load('data/goodreads/work-id/clause_is_reverse_narrow.pt')
assert ia.dim() == 3 and ia.shape[1] == 4, ia.shape  # 4 narrow clauses
assert isr.tolist() == [False, True, False, False], isr.tolist()
print('OK  tensors')
"

# 4. Driver imports.
uv run python -c "
import retrieval.evaluate
from retrieval.algo_registry import build_algorithm, build_filter_modules
print('OK  imports')
"
```

If any step fails: stop and report. Don't attempt to "fix" missing
checkpoints or stale tensors — escalate.

## Execution

Three dims, run sequentially. d64 first as a sanity check; only
proceed to d128/d256 after d64's `none/full_scan/torch_fullscan
recall@10` matches the checkpoint's `eval_quality.json` recall@10.
Each dim takes ~30–45 min wall on a single GPU with full quality, ~5
min with `--skip-quality` (no oracle build, perf timing only).

```bash
cd /workspace/retrieve/evaluation

# 1. d64
rm -rf data/goodreads/work-id/gt
uv run evaluate --config conf/goodreads/d64-filter.yaml \
  > /tmp/bench-d64.log 2>&1

# 2. d128 once d64 validates
rm -rf data/goodreads/work-id/gt
uv run evaluate --config conf/goodreads/d128-filter.yaml \
  > /tmp/bench-d128.log 2>&1

# 3. d256 once d128 validates
rm -rf data/goodreads/work-id/gt
uv run evaluate --config conf/goodreads/d256-filter.yaml \
  > /tmp/bench-d256.log 2>&1
```

**Important**: purge `data/goodreads/work-id/gt/` between dims. The
oracle cache is keyed by sweep name only, not by dim — running d128
without purging would reuse d64's oracles (built with d64's
`item_embs`), corrupting the recall scoring.

Use `Bash` with `run_in_background=true` and `Monitor` on the resulting
shell ID. Don't poll with `sleep`.

**Output JSON path**: `<ckpt-dir>/evaluate.json`, e.g.
`checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json`.

### Per-cell debugging

If a specific cell fails, narrow the run:

```bash
uv run evaluate \
  --config conf/goodreads/d64-filter.yaml \
  --filter-kind clause \
  --sweep c0_genre \
  --algorithms linr_v1_filter_mask \
  --output /tmp/gr-debug.json
```

`--algorithms` **replaces** (not merges with) the YAML list.
`--filter-kind` is repeatable (`--filter-kind clause --filter-kind bloom`)
to skip the unfiltered cell.

### Skipping quality for fast latency-only runs

Add `--skip-quality` to skip the quality stream and the oracle build.
Recall/NDCG report `NaN`; perf rows are fully populated. ~5 min wall.

## Verification (per-dim, gates the next dim)

The output is a flat JSON list. Each row carries `suite`,
`filter_kind`, `sweep`, `impl`, `k`, `batch_size`, `n_users_kept`,
latency / memory fields, and `recall@k` / `ndcg@k`. Skip rows have
`"skipped": true`.

All checks below should pass before moving to the next dim. `jq`
one-liner per check, all paths assume cwd
`/workspace/retrieve/evaluation`.

### 1. Yambda parity on the unfiltered cell

`torch_fullscan recall@10 / @100` on `none/full_scan` matches the
checkpoint's `eval_quality.json` to ±0.0005.

Yambda baselines on disk (from training-time eval):

| dim  | recall@10 | recall@100 |
|------|-----------|------------|
| d64  | 0.03331   | 0.14861    |
| d128 | (read from `eval_quality.json`) | |
| d256 | (read from `eval_quality.json`) | |

```bash
jq '.[] | select(.filter_kind=="none" and .impl=="torch_fullscan"
                 and .k==10 and .batch_size==1)
       | {recall:."recall@10", ndcg:."ndcg@10"}' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
```

### 2. V1 pre-filter exact

`linr_v1_filter_mask recall@k = 1.0` (≥ 0.999) on every filter cell.
Same scoring + same mask as the oracle by construction. Anything
else is a regression in the unified mask path.

```bash
jq '[.[] | select(.impl=="linr_v1_filter_mask" and .filter_kind!="none")
        | ."recall@10"] | min' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
# Expect ≥ 0.999
```

### 3. Post-filter degradation visible

`torch_fullscan` and `voyager_hnsw` on filter cells are post-filter.
Recall on selective sweeps (e.g. `all4`) should drop noticeably below
`linr_v1_filter_mask`'s 1.0 — that drop is the metric the bench is
designed to surface.

```bash
jq '[.[] | select(.k==10 and .batch_size==1
                  and .impl=="torch_fullscan" and .filter_kind!="none")
        | "\(.filter_kind)/\(.sweep) recall=\(."recall@10")"] | unique[]' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
```

### 4. V3 hash quality plausible

`linr_v3_then_v2 recall@10` on filter cells in `[0.7, 0.99]`. Below
0.7 means V3's 1-bit hash is missing real top-K under filter; tunable
via `algo_params.linr_v3_then_v2.candidate_pool` (currently 8000);
raise to 16000 if needed.

```bash
jq '[.[] | select(.filter_kind!="none" and .impl=="linr_v3_then_v2"
                  and .k==10) | ."recall@10"]
        | "min=\(min) max=\(max)"' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
```

### 5. Codesigned silvertorch on `bloom` kind

`silvertorch` on `bloom` cells runs the bloom-fused IVF probe (the
codesigned path). The query passes `query_clause_attrs=qa_narrow` so
the inline bloom check filters during probe. Recall reflects the
combined IVF + bloom-FP loss.

```bash
jq '[.[] | select(.filter_kind=="bloom" and .impl=="silvertorch"
                  and .k==10 and .batch_size==1)
        | "\(.sweep) recall=\(."recall@10") ms=\(.median_ms)"] | unique[]' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
```

### 6. Sweep coverage plausible

`n_users_kept` should be a non-trivial fraction of `filter_users_limit`
(50 000) for every filter sweep.

```bash
jq '[.[] | select(.filter_kind!="none" and (.skipped // false | not))
        | "\(.filter_kind)/\(.sweep) kept=\(.n_users_kept)"] | unique[]' \
  checkpoints/goodreads-gsasrec-d64-drop0.5-id/evaluate.json
```

Floors (per sweep, of 50 000 subsampled users):

- `clause/c0_genre` ≥ 47 000 (genre coverage 93 %)
- `clause/c1_lang_reverse` ≥ 32 000 (lang coverage 64 %)
- `clause/c3_year` ≥ 42 000
- `clause/all4` ≥ 30 000 (intersection of all four)
- `bloom/*` mirrors the corresponding `clause/*` floor (same
  active-clauses → same `skip_mask`).

If any clause-only sweep drops below 30 000, the synthesis path in
`build_sweep_qa` or `eval_split.parquet` is suspect — escalate before
re-running.

## Final report (after all three dims complete)

Build a single comparison table for the user. Group by
`(filter_kind, sweep, impl)` × `k=10`, `batch_size=1`. Columns:
`recall@10` and `median ms` per dim.

```
=== Unfiltered (none/full_scan, held-out target) ===
                                 d64       d128      d256
torch_fullscan  recall@10         ?         ?         ?     (yambda parity)
linr_v1_filter_mask  recall@10    ?         ?         ?     (= torch_fullscan)
linr_v3_then_v2 recall@10         ?         ?         ?
silvertorch     recall@10         ?         ?         ?
voyager_hnsw    recall@10         ?         ?         ?

=== Filter suite (vs filtered-FullScan oracle) ===
                                              recall@10 (d64/d128/d256)   median ms (bs=1)
clause/c0_genre     torch_fullscan              ?  ?  ?                      ?  ?  ?
clause/c0_genre     linr_v1_filter_mask         1.0 1.0 1.0                  ?  ?  ?
clause/c0_genre     linr_v3_then_v2             ?  ?  ?                      ?  ?  ?
clause/c0_genre     silvertorch                 ?  ?  ?                      ?  ?  ?
clause/c0_genre     voyager_hnsw                ?  ?  ?                      ?  ?  ?
…
bloom/c0_genre      torch_fullscan              ?  ?  ?                      ?  ?  ?
bloom/c0_genre      linr_v1_filter_mask         1.0 1.0 1.0                  ?  ?  ?
bloom/c0_genre      linr_v3_then_v2             ?  ?  ?                      ?  ?  ?
bloom/c0_genre      silvertorch (codesigned)    ?  ?  ?                      ?  ?  ?
bloom/c0_genre      voyager_hnsw                ?  ?  ?                      ?  ?  ?
…
```

The interesting deltas:
- `torch_fullscan` / `voyager_hnsw` post-filter recall vs `linr_v1_filter_mask`'s exact 1.0
- `silvertorch` clause (post-mask) vs `silvertorch` bloom (codesigned) — same predicate, two strategies
- Per-sweep variance as filter selectivity tightens (`c0_genre` loose → `all4` very tight)
- Dim scaling: d64 vs d128 vs d256 — V3 cascade should win more on larger dims since V3's 1-bit work scales as D/64 vs V1's full GEMV at D

Use `jq` to extract; one-liner per row. Do **not** paste the raw JSON.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `recall@10` near 0 on yambda parity check | encoder NaN (check `bench_tools.encode_queries` autocast) | encoder regression; escalate |
| `linr_v1_filter_mask recall < 0.999` on any cell | item-0 sentinel desync between oracle and combined_mask | escalate; this is a regression in `algo_registry.combined_mask` |
| `eval_split rows ≠ test rows` | stale `eval_split.parquet` after `prep` regen | regen via `uv run python -m data.goodreads attrs --processed-dir <…> --output-dir data/goodreads/work-id` |
| `RuntimeError: CUDA out of memory` during oracle | batch too large at d256 | drop `compute_filtered_oracle(batch_size=)` from 64 to 32 in `evaluate.py` |
| Oracle file `gt_topk_<sweep>.pt` shape mismatch | stale cache from a different dim | delete stale files in `data/goodreads/work-id/gt/`; rerun |
| `linr_v3_then_v2 recall@10 < 0.7` on every sweep | `candidate_pool` underprovisioned | bump `candidate_pool` 8000 → 16000 in YAML |

## Out of scope

- Code changes — the harness is functionally complete. Do not touch
  `algo_registry.py`, `evaluate.py`, `bench_tools.py`, or the
  YAMLs beyond the per-cell debug flags above.
- Hyperparam tuning — current `candidate_pool=8000`, `n_lists=1024`,
  `n_probe=24`, `m_bits=1024`, `k_hash=5`, `voyager (m=32, ef_c=200,
  ef_query=1000)` are bench-thesis defaults. Only adjust
  `candidate_pool` if step 4 verification fails on every sweep.
- Re-running data prep (`goodreads.py attrs` / `prep`) — artifacts on
  disk are validated and final.
- Wide-bloom sweeps — the bench has no wide path. There's no separate
  attribute tensor or sweep type beyond the narrow stack described
  here. If `item_attrs_wide.pt` exists on disk from older data-prep
  runs, ignore it.
