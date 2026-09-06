# GPU validation handoff — the harness half (archived with roadmap C3)

> **Status:** archived 2026-09-06. This is the `evaluation/` half of
> [../refactor-validation-handoff.md](../refactor-validation-handoff.md) (steps 1, 4, 6, 7,
> the harness delta list and fallbacks F4 / F6), split out when roadmap C3 deleted the harness
> these steps validate: `cli/evaluate.py`, `cli/run_evaluation.py`, `sweep.py`, `passes.py`,
> `loaders.py`, `queries_cache.py` and the 19 YAMLs are gone
> ([evaluation-harness-v2.md §5](../evaluation-harness-v2.md)); the live harness is documented in
> [../../system/evaluation.md](../../system/evaluation.md). Steps 4–7 survive as **roadmap A1**
> (the golden baseline run on the *old* harness, on a branch that predates C3), which records
> its result in the live file's "A1 record" section, not here. Commands below run only on a
> checkout that still has the old harness (`main`, or `development` before C3 merges).

## Step 1 — Eval CPU tests (no GPU needed)

```bash
cd evaluation
uv run pytest retrieval/tests/ -v
```

**Pass:** all green (5 files, 24 test functions; parametrization expands the reported count):
config round-trip, metrics padding/idcg/denominator semantics, sweep-qa helpers + `supports`
table coverage, oracle padding + fingerprint cache-hit/invalidation/legacy-blob cases,
silvertorch reverse wrapper. Note: the four E8.1 files are CPU-only; the pre-existing
`test_silvertorch_algo_reverse.py` needs CUDA (fine on the validation box — deselect it if
you ever run this step on a CPU-only machine).

## Step 4 — Compile gate on a real eval cell

```bash
cd evaluation
TORCH_LOGS=graph_breaks uv run evaluate \
    --config config/goodreads/d128-filter.yaml \
    --algo linr_v3 \
    --output /tmp/linr_v3_graphbreak_smoke.json \
    --filter-kind clause --sweep c0_genre --skip-quality
```

**Pass:** the run completes and the `graph_breaks` log shows **no new breaks** attributable to
`masked_topk`, `_PackedBitsKNN`, or the shared kernel preps (if any break appears, confirm it
also fires on `main` with the same command before blaming the refactor — pre-existing breaks
are out of scope). See the step-6 caveat if the run fails *before* reaching the sweep
(`eval_split rows != queries` — the known latent `users_limit` bug, not a refactor break).

## Step 6 — Golden-run diff (quality byte-identical)

The reproducibility gate from evaluation-refactor.md's Conventions. Capture the pre-refactor
golden on `main` (same GPU, same data, same config), then the branch:

```bash
# on main (worktree), from evaluation/:
uv run evaluate --config config/goodreads/d128-filter.yaml --algo linr_v3 \
    --output /tmp/golden-main.json --filter-kind clause --sweep c0_genre

# on refactor/kernels-eval, from evaluation/:
uv run evaluate --config config/goodreads/d128-filter.yaml --algo linr_v3 \
    --output /tmp/golden-branch.json --filter-kind clause --sweep c0_genre
```

> **Known latent bug caveat (pre-existing, NOT introduced or fixed by the refactor):** on the
> checkpoint path, `queries_cache` trims to `users_limit` *before* caching, while
> `load_query_attrs` checks `eval_split.parquet`'s row count against the (trimmed) query
> count — a config with both `filters:` and `users_limit:` set (goodreads d128-filter sets
> `users_limit: 10000`) can raise `eval_split rows=… ≠ queries=…`. If that fires, it must
> fire **identically on main and branch** (then run the golden with `users_limit: null` in a
> copy of the config on both sides, and file the bug for a post-merge fix). If it fires on
> only one side, that is a real regression — stop and report. (Roadmap A1 commits the 3-line
> fix; harness v2's `data.load_inputs` applies `users_limit` after the row check.)

**Pass criteria** (compare row-by-row after joining on the explicit columns
`(filter_kind, sweep, impl, backend, batch_size, k)` — do **not** join on the `cell` string):

1. **Quality columns byte-identical**: `recall@100/200/…` and `ndcg@…` equal exactly on every
   joined row (the first branch run rebuilds the oracle — see below — but the oracle contents
   must be identical, so quality must not move).
2. **Additive columns only**: the branch rows add `precision@<k>`, `mrr@<k>`, and
   `extra.gpu` / `extra.torch` / `extra.commit`. No existing column renamed or dropped
   (`device` is present and `"cuda"`).
3. **Cell keys identical**: the `cell` f-string
   (`<filter_kind>_<sweep>_<backend>_bs<N>_k<K>`) is byte-identical between main and the
   branch (verified with `git show main:evaluation/retrieval/sweep.py`), so the two runs must
   produce the same `cell` key set; any difference is a real regression.
4. **Latency** columns within ~5%.
5. **One-time oracle rebuild**: the first branch run logs
   `building filtered oracle` and writes `gt_topk_v3_c0_genre.pt`; older `gt_topk_*` caches
   are ignored by name and can be deleted. Subsequent branch runs must log
   `loaded oracle from cache`.

Quick diff helper:

```bash
python3 - <<'EOF'
import json
key = lambda r: (r["filter_kind"], r["sweep"], r["impl"], r.get("backend"), r["batch_size"], r["k"])
a = {key(r): r for r in json.load(open("/tmp/golden-main.json"))}
b = {key(r): r for r in json.load(open("/tmp/golden-branch.json"))}
assert a.keys() == b.keys(), f"cell sets differ: {a.keys() ^ b.keys()}"
for k in a:
    qa = {c: v for c, v in a[k].items() if c.startswith(("recall@", "ndcg@"))}
    qb = {c: b[k][c] for c in qa}
    assert qa == qb, f"{k}: quality drift {qa} vs {qb}"
    new = set(b[k]) - set(a[k])
    assert all(c.startswith(("precision@", "mrr@")) for c in new), f"{k}: unexpected new cols {new}"
print(f"OK: {len(a)} rows, quality byte-identical, additive columns only")
EOF
```

## Step 7 — Orchestrator smoke

```bash
cd evaluation
uv run run-evaluation config/goodreads/d128-filter.yaml --resume -- --skip-quality --sweep c0_genre
```

**Pass:** exits 0; one `<algo>.json` per YAML `algorithms` entry appears under
`results/goodreads/d128-filter/` (already-complete ones are skipped with a `resume:` line —
that skip path is part of what's under test); `results/_runlogs/` gains
`goodreads_d128-filter.log`, `full.log`, and `SUMMARY.txt` entries. This exercises the E7.3
output guard, the subprocess flow, and the resume logic; staging is untouched.

## Harness behaviour deltas (E1–E8.1, intentional)

- **Row schema**: `cell` key format unchanged from main; new additive columns `precision@k`,
  `mrr@k`, `extra.{gpu,torch,commit}`; `device` is always `"cuda"` (CPU path removed, column
  kept for stability).
- **Loud failures where there was silence**: unknown YAML `filter_kind` raises; algo
  construction errors kill the run (no more silently-vanishing cells); missing
  attrs/reverse paths raise from `resolve_path` (no basename fallback); `output: null`
  configs get a clear `SystemExit` from the orchestrator.
- **`datasets` → `eval_datasets`**: import paths changed; console-script names unchanged
  (six scripts retargeted). Anything outside the repo importing `datasets.*` from this venv
  breaks (that was the point — the old name shadowed HuggingFace `datasets`).
- **`upload-results`** requires `--repo-id`; campaign prose moved behind `--notes-file`;
  `--private/--public` flags.
- **Oracle**: `gt_topk_v3_` dict-blob cache with content fingerprint; `users_limit`
  participates via the post-limit tensors; expect a one-time rebuild per sweep.
- **Queries cache** key gained `users_limit` (and the cache stores post-limit tensors).
- `torch_knn` algo deleted (was broken *and* unused; resurrect as an `nn.Module` wrapper over
  `FullScanKNN` if an fp32 exact reference is ever wanted again — the oracle never depended
  on it).

## Deferred items that harness v2 closed

- **JSONL streaming row writes** (crash resilience for 6-hour configs) — `run.append_record`,
  one record per cell.
- **`EVAL_TYPES` glob** hand-synced with `evaluation/config/` — gone with the presets; the
  matrix is `config/suites.yaml`.
- **Latent `users_limit` bug** — A1 fixes it on the old harness; `data.load_inputs` has one
  `users_limit` site after the row-count check.
- **`evaluation/` ruff-format debt** (17 files) — the 30 old files are deleted; `ruff check`
  is clean on `retrieval/`.

## Fallbacks

**F4 — golden-diff quality drift** (step 6 criterion 1 fails): quality differences are
**never acceptable** — stop and bisect by phase commit. Ordered suspects: a `masked_topk`
call-site behavior mismatch vs the K4.1 table (check `pad_to_k`/`valid` flags at the failing
algo's layer), stale oracle caches (delete `gt_topk_*` on **both** sides and rerun both),
TF32/precision pins, and only then kernel numerics (which step 3's parity should have
caught). Report the failing `(algo, sweep, k)` cell and the two row dicts.

**F6 — orchestrator smoke failure** (step 7): `output:` unset in the config → the E7.3
guard's `SystemExit` is *correct* behavior (fix the config, not the code); resume skipping
a run you expected to execute → the target JSON already parses as a non-empty list
(`_is_complete`) — delete it or drop `--resume`; nonzero rc from a child `evaluate` → read
`results/_runlogs/current.log`, and remember construction errors are now intentionally fatal
(E3.3).
