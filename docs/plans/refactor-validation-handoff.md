# GPU validation handoff — `refactor/kernels-eval`

> **Status:** **partially executed.** Steps 2–3 passed de facto on an A100 on 2026-09-02:
> the full `retrieve/tests/` suite is green at the branch tip (286 tests,
> [cute-dsl-scorer-artifacts/wp4/pytest-final.txt](cute-dsl-scorer-artifacts/wp4/pytest-final.txt))
> and the K2 tracing caveat did not bite. Steps 1 and 4–7 (eval CPU tests, compile gate on a real
> cell, per-kernel perf gates, golden-run diff, orchestrator smoke) have **not** run — no dataset
> was on the box. Written 2026-07-06, verified against the branch's source
> tree. This is the runbook a GPU test agent executes **top-to-bottom** to validate the
> kernels/layers + evaluation refactor before it merges. The two source plans —
> [kernels-layers-design.md](kernels-layers-design.md) and
> [evaluation-refactor.md](evaluation-refactor.md) — stay intact until every gate here passes;
> after that, trim them per the roadmap's "Cleanup status" convention.
>
> Do not rebase/merge the branch mid-validation. Every fallback below is a code change —
> re-run the affected gates after applying one.

## 1. Context — what landed

Branch `refactor/kernels-eval`, on top of `main` @ `2b1ff80`. All code phases are committed —
12 phase commits, one per work-package (identify them by phase tag in `git log --oneline`, not
by hash); the system docs (`docs/system/*.md`) and the roadmap were brought in line
afterwards. By phase:

**Library (`retrieve/`) — kernels-layers-design.md K1–K8 + rename:**

- **K1** — fixed the broken `tune-kernels codesigned-probe-score` subcommand (was calling the
  public op with kwargs removed in Stage 2b) + a CUDA-gated tuner smoke test.
- **K2** — host-wrapper dedup in all seven kernel files: shared `_<name>_prep` (validation +
  contiguity + buffers + launch-kwargs in a frozen `_<Name>Launch` dataclass) and
  `_<name>_finish` (topk/gather/sentinel epilogue); every entry point is prep → one launch
  line → finish. **Along the way the four remaining `@custom_op` kernels (`clause_mask`,
  `clause_compact`, `bloom_compact`, `fused_masked_knn_topk`) moved to `@triton_op`** — their
  shape-branching (P-bucketing, `p == 0` early-return, pad tails) now lives only in the eager
  `_impl`s behind explicit `bucket=` / `pad_to_k=` flags. All ten registered ops are
  `@triton_op` + inline `wrap_triton(...)[grid](**launch.kwargs)`.
- **K3** — shared `@triton.jit` helpers in `retrieve/src/retrieve/kernels/common.py`
  (`popcount_int64`, `bloom_subset_pass`, `clause_pass`, `compact_store`, `or_combine`);
  bloom kernels standardized on the `qb & ~sig` OR-reduce subset form (boolean-identical).
- **K4** — `layers/utils/topk.py` (`masked_topk` / `counts_to_valid`, replacing six inlined
  epilogue copies); `layers/linr/_bit_knn.py` (`_PackedBitsKNN` base folding `SimHashKNN`'s
  ~95% copy of `OneBitKNN`, plus the `k_bits` re-registration fix); SilverTorch structural
  cleanups (`register_index` split into validate/IVF/quantize/filter-buffer phases with frozen
  buffer order, the `candidate_ids`+`query_clause_attrs` guard, shared eager predicate math);
  `FullScanKNN` post-filter semantics documented.
- **K5** — `layers/filters/bloom_hash.py`: the bloom hash math re-homed behind public names
  (`generate_seeds`, `build_signatures`, `build_query_signatures`, `bloom_subset_match`);
  no more private `_`-imports from `bloom.py` in SilverTorch. Hash output is bit-identical.
- **K6** — `FilterModule.register_index(item_clause_attrs, *, clause_is_reverse=None)`
  unified/kw-only on the ABC and both filters; minimal `RetrievalModule` ABC reintroduced and
  subclassed by every retrieval layer.
- **K7** — `tune.py` rebuilt as a `KernelTuneSpec` registry (`KERNELS` tuple → generated click
  subcommands; one generic `_sweep`/`_print`): **seven** subcommands, including the new
  `codesigned-probe-score-exact`.
- **K8** — new tests: `test_topk_util.py`, `test_bit_knn_base.py`, `test_bloom_hash.py`,
  `test_tune_smoke.py`, `tests/compile/test_export_kernel_ref.py`, plus reworked/extended
  cases in `test_linr.py` / `test_silvertorch.py` / `test_filters.py`.
- **Rename** — `SilverTorch(filter=...)` → `filter_mode=` everywhere (constructor,
  `build_silvertorch`, eval algo wiring, tests, docs). No deprecation shim (pre-1.0).

**Harness (`evaluation/`) — evaluation-refactor.md E1–E8.1:**

- **E8.1** (landed first) — CPU-only unit tests under `evaluation/retrieval/tests/`:
  `test_metrics.py`, `test_sweep_helpers.py`, `test_oracle.py`, `test_config.py` (plus the
  pre-existing `test_silvertorch_algo_reverse.py`).
- **E1** — deleted `TorchKnnAlgo` (broken + unused) and `BACKEND_CAPABLE_ALGOS`; deleted the
  unreachable CPU-timing path (`is_cpu`, `measure_forward_cpu`); inlined
  `expand_param_combos`; dropped `torchvision`/`matplotlib`/`einops` deps; `upload-results`
  parameterized (`--repo-id` required, `--notes-file`, `--private/--public`); **`datasets`
  package renamed to `eval_datasets`** (console-script *names* unchanged; targets changed —
  six scripts: `yambda`, `arxiv`, `goodreads`, `eval-fetch`, `eval-publish`,
  `eval-publish-checkpoint`).
- **E2** — `retrieval/context.py`: frozen `SweepContext` / `FilterAssets` replace the
  15–20-kwarg × 5-level parameter threading; `apply_users_limit` moved to `loaders.py`
  (`build_sweep_qa` also lives in `loaders.py`, not `sweep.py` — plan drift, intentional).
  `SweepContext` gained a `filter_kinds` field the plan block omitted (`--filter-kind` narrow).
- **E3** — `RetrievalAlgo` runtime-checkable protocol; `SUPPORTED_FILTER_KINDS` +
  `supports()` declarative eligibility; `_try_build_algo`'s blanket `except ValueError`
  deleted — construction errors now kill the run loudly; unknown YAML `filter_kind` raises in
  `_select_filter_iter`; `FilterKind` literal alias.
- **E4** — `AlgoBase._finalize(...)`: the one canonical
  `torch.compile(dynamic=True, mode="reduce-overhead")` call; `collect_modules` deleted.
- **E5** — `bench_tools.py` split into `measure.py` / `encode.py` / `passes.py` (then
  deleted); `PerfStats` / `QualityStats` dataclasses; additive row columns
  `precision@k` / `mrr@k` and `extra.{gpu,torch,commit}` provenance.
- **E6** — oracle disk cache is now `gt_topk_v3_<sweep>.pt`, a dict blob with a **content
  fingerprint** (shapes + dtypes + 64-row linspace sample of `item_embs`/`queries`/`qa` +
  `K_GT`); same-shape stale caches recompute automatically. This is the code fix for roadmap
  Stage 4b item 7 (the rerun itself still remains).
- **E7** — `resolve_path` raises instead of the silent basename fallback; one
  `load_raw_config` YAML-strip implementation; `output: null` configs get a clear `SystemExit`
  from the orchestrator; docstring sweep (incl. the `_autotune_prewarm` rationale).

**Invariants that must hold through validation** (from the plans): parity suite passes with
**unchanged tolerances** (strict equality on OPORP/SimHash); `retrieve::*` op schemas frozen;
buffer names + registration order frozen; `wrap_triton` textually inline in every `@triton_op`
body; quality columns byte-identical; latency within ~5%.

## 2. Environment setup

Requirements: CUDA GPU (perf gates are meaningful on the arch the shipped `DEFAULT_CONFIG`s
were tuned for — A100/sm_80; correctness gates run on any recent NVIDIA GPU), `uv`, and the
goodreads eval data + d128 checkpoint for the eval-side steps.

```bash
cd <repo-root>            # the uv workspace root (contains uv.lock)
git checkout refactor/kernels-eval
uv sync
```

`uv sync` is **required**, not optional: `uv.lock` was already regenerated on the branch
(3 direct + 8 transitive dependencies removed — `torchvision`/`matplotlib`/`einops`; no
version bumps), and the `datasets` → `eval_datasets` package rename needs the editable
reinstall to take effect. Verify:

```bash
cd evaluation
uv run python -c "import eval_datasets, retrieval, retrieve; print('imports ok')"
```

Data for steps 4, 6, and 7 (skip if `evaluation/data/goodreads-work-id/` is already
populated):

```bash
cd evaluation
uv run eval-fetch goodreads-work-id
```

## 3. Ordered validation steps

Run in this order — step 2 is deliberately **before** the full suite because it gates the
pattern all seven kernel files share.

### Step 1 — Eval CPU tests (no GPU needed)

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

### Step 2 — K2 tracing-caveat check FIRST (pattern-setter)

The one genuine K2 risk, **unverified without a GPU**: `triton_op` requires its body to be
traceable by make_fx for the fake path, and the branch's launch form is a frozen-dataclass +
kwargs-splat (`wrap_triton(_kernel)[grid](**launch.kwargs)`; the `@triton_op` bodies keep
`def grid(meta)` closures where the grid depends on the config). Check it on
`codesigned_probe_score` (the worked example) before anything else:

```bash
cd retrieve
uv run pytest tests/parity/test_codesigned_probe_score.py -v
uv run pytest tests/compile/test_silvertorch_compile.py -v
uv run pytest tests/compile/test_export_kernel_ref.py -v
```

**Pass:** all green; `test_no_graph_breaks_on_forward` reports zero graph breaks on all three
`filter_mode`s; the export test finds a live kernel reference and replays bit-identically.

**Fail →** apply **Fallback F1** (plain-tuple prep + positional splat) to
`codesigned_probe_score.py` first, re-run this step, then roll to the other six files.

A second, smaller compile risk rides along: the kernel bodies call the shared `@triton.jit`
helpers with keyword args (e.g. `common.clause_pass(..., ids=n_offsets, load_mask=n_valid,
..., C=C, A_MAX=A_MAX)`), which needs one compile pass on the pinned Triton to confirm:

```bash
uv run pytest tests/parity/test_clause_mask.py tests/parity/test_clause_compact.py -v
```

**Fail with a Triton error naming keyword args →** **Fallback F2**.

### Step 3 — Full `retrieve` suite (strict parity)

```bash
cd retrieve
uv run pytest tests/ -x -q
```

**Pass:** everything green with **unchanged tolerances** — strict `torch.equal` on the
OPORP/SimHash parity and cross-backend paths (any bit drift = a popcount/packing bug by
definition). Pay attention to: `test_linr.py` (incl. the zero-counts sentinel case and
cross-backend strict equality for both bit-KNNs), `test_silvertorch.py` (incl. the new
candidates+`query_clause_attrs` raise — two pre-existing tests were reworked because that
combination is now guarded), `test_bloom_filter.py` hash-invariant cases, and the new
`test_bloom_hash.py` (chunked vs loop-free builder equality), `test_bit_knn_base.py`,
`test_topk_util.py`, `test_tune_smoke.py` (CUDA-gated; one tiny sweep point per tuner spec).

### Step 4 — Compile gate on a real eval cell

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

### Step 5 — Kernel perf gates (±5% per kernel)

Gate: per-regime winning medians within ±5% of pre-refactor, per kernel. Capture the baseline
on `main` (a second `git worktree` is convenient — validation must not switch the primary
checkout), then run the branch:

```bash
cd retrieve
uv run tune-kernels clause-mask        --json-out /tmp/tune-clause-mask.json
uv run tune-kernels clause-compact     --json-out /tmp/tune-clause-compact.json
uv run tune-kernels bloom-compact      --json-out /tmp/tune-bloom-compact.json
uv run tune-kernels fused-masked-knn-topk --json-out /tmp/tune-fmkt.json
uv run tune-kernels oporp-1bit-match-topk --json-out /tmp/tune-oporp.json
uv run tune-kernels codesigned-probe-score --json-out /tmp/tune-cps.json
```

Baseline notes:

- On `main`, `tune-kernels codesigned-probe-score` is **broken** (the K1 bug — TypeError on
  the first sweep point). For that kernel, take the baseline from the branch commit that
  landed K2 for `codesigned_probe_score.py` *before* K3 (find it via
  `git log --oneline -- retrieve/src/retrieve/kernels/silvertorch/codesigned_probe_score.py`),
  or accept the parity-suite wall-clock comparison as the gate.
- JSON diagnostics changed shape (benign): the fmkt progress key is now `P=…,D=…,B=…` and the
  `--json-out` per-shape key is `per_regime` (was `per_bucket`). Compare medians, not keys.

**Watch:** `clause_pass` register pressure is the flagged risk (a helper with many pointer
params can perturb allocation on `clause_mask` / `clause_compact` /
`codesigned-probe-score-exact`). Any kernel regressing > 5% → **Fallback F3** (per-kernel
inline revert), then re-run that kernel's gate.

Then exercise the **new** seventh subcommand end-to-end (K1-acceptance analog — this kernel
never had a tuner before):

```bash
uv run tune-kernels codesigned-probe-score-exact
```

**Pass:** completes on the default clause regimes and prints a pasteable
`DEFAULT_CONFIG = CodesignedProbeScoreExactConfig(...)` line. (Do **not** commit a new
DEFAULT_CONFIG from this run unless the perf gate work explicitly calls for retuning.)

### Step 6 — Golden-run diff (quality byte-identical)

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
> only one side, that is a real regression — stop and report.

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

### Step 7 — Orchestrator smoke

```bash
cd evaluation
uv run run-evaluation config/goodreads/d128-filter.yaml --resume -- --skip-quality --sweep c0_genre
```

**Pass:** exits 0; one `<algo>.json` per YAML `algorithms` entry appears under
`results/goodreads/d128-filter/` (already-complete ones are skipped with a `resume:` line —
that skip path is part of what's under test); `results/_runlogs/` gains
`goodreads_d128-filter.log`, `full.log`, and `SUMMARY.txt` entries. This exercises the E7.3
output guard, the subprocess flow, and the resume logic; staging is untouched.

### Step 8 — Sign-off

All green → the refactor is validated. Post-validation follow-ups (not gates):

1. Trim `kernels-layers-design.md` and `evaluation-refactor.md` per the roadmap's
   "Cleanup status" convention, and flip the roadmap Refactor-track status line to validated.
2. Run the roadmap **Stage 4b item 7** rerun (goodreads d128 + d256 filter sweeps against the
   now-fingerprinted oracle) — the publication blocker this branch's E6 unblocked. Expect the
   one-time `gt_topk_v3_` rebuild per sweep.
3. Decide on retuned `DEFAULT_CONFIG`s only if step 5 motivated them.

## 4. Behavior deltas to be aware of (intentional, could surprise)

Library:

- **`PrefilterKNN` (torch path)** now applies the `isfinite → -1` sentinel it previously
  lacked — matches the Triton path (the plan's K4.1 table prescribed it).
- **Pad dtype**: `masked_topk` pads scores in the *scores* dtype, so PostfilterKNN-family
  fp16 paths pad `-inf` in fp16 where the old cat-pad promoted to fp32 — observable only when
  `0 < P < k`.
- **`PostfilterKNN` / `PostfilterKNNInt8` with out-of-contract `k > N`** now pad `-1`/`-inf`
  instead of raising from `torch.topk`.
- **`SilverTorch` eager (torch-backend) epilogue** tombstones `-inf` (filter-rejected)
  winners to id `-1` in rows with < k survivors. The Triton host tail does **not** tombstone;
  suite comparators filter to finite scores, so no test flips — but a downstream consumer
  reading ids at non-finite score positions would now see `-1` on the torch backend.
- **`SilverTorch.forward(query_clause_attrs=…, candidate_ids=…)` now raises `ValueError`**
  (previously silently ignored the predicate on the candidates path).
- **`SilverTorch._forward_candidates` with `P < k`** returns `min(k, P)` columns (no pad) —
  pinned, pre-existing behavior, now explicit via `pad_to_k=False`.
- **`filter=` → `filter_mode=`** on `SilverTorch` / `build_silvertorch` (no shim).
- **`_build_ivf`** registers `centroids` and returns `(padded, cluster_sizes)` for later
  registration by `register_index` — preserves the frozen buffer-registration order
  (commented in code).
- **`OneBitKNN.k_bits`**: pristine `_k_bits_arg` kept from `__init__`; the `0` sentinel is
  re-resolved on every `register_index`, so re-registering a different-dim corpus can't
  silently keep the first `D`.
- **fmkt public op** gained an identity `clamp_max(p-1)` before the gather (bit-identical on
  the unbucketed path; avoids a third policy flag); **public ops now validate inputs** (they
  were the unvalidated path before); the `_impl`'s `p == 0` early-return now precedes
  validation (observable only on malformed dims AND `p == 0`).
- Dropped a redundant `& n_valid` after `clause_pass` in the clause kernels (the helper
  already ANDs `load_mask`; boolean-identical).
- `bloom_match` / `bloom_compact` switched to the `qb & ~sig` OR-reduce subset form
  (boolean-identical; parity-tested).
- **tune.py**: six legacy subcommands keep their CLI contract; `codesigned-probe-score-exact`
  is new; two benign diagnostic deltas — fmkt progress key `P=…,D=…,B=…`, `--json-out` key
  `per_regime` (was `per_bucket`).

Harness:

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

## 5. Known-stale / deferred items (not validation blockers)

- **JSONL streaming row writes** (crash resilience for 6-hour configs) — consciously
  deferred; do after E2's single choke point, per the plan's "Additional smaller
  improvements".
- **State-dict load post-hook** (`_global_scale_f` re-derivation; buffers registered in
  `register_index`) — deferred to the torch-export plan (kernels-layers Additional-1).
- **k-means++ init for `KMeansTorch`** — deferred; needs GPU measurement
  (kernels-layers Additional-2; thesis-relevant ablation).
- **Fused in-kernel top-k** and other kernel optimizations — roadmap Stage 3 /
  future-work-and-research.md; out of scope here.
- **`EVAL_TYPES` glob** — the preset config lists in `run_evaluation.py` stay hand-synced
  with `evaluation/config/` by design.
- **`evaluation/` ruff-format debt**: `ruff format --check` would reformat 17 files, and
  `ruff check .` reports **13 pre-existing violations** (long-line E501s in `eval_datasets/`
  and `training/`-adjacent code; verified 2026-07-06 — ruff version drift may move the count
  slightly). These predate the refactor; the plans' gate was "ruff check clean *on touched
  code*" — do **not** mass-reformat during validation.
- **`retrieve/tests/correctness/test_quantize.py`** uses the SWAR constant
  `0x5555555555555555` as *test data* — the "SWAR constants appear only in
  `kernels/common.py` + `layers/utils/quantize.py`" grep gate has exactly this one benign
  extra hit.
- ~~**`retrieve/docs/filtering-and-quantization.md`** still shows `filter=`~~ — **fixed**;
  the doc sweep caught these and the later `retrieve/README.md` occurrence too. No `filter=`
  remains anywhere in the docs (`grep -rn 'filter=' --include='*.md' .` is clean).
- **Latent bug (observed, NOT fixed — do not silently patch mid-validation)**:
  `users_limit` + `filters:` on a checkpoint dataset can raise the `eval_split` row-count
  check in `load_query_attrs` because the queries cache returns pre-trimmed tensors (see the
  step-6 caveat). Identical on main and branch; file for follow-up.
- The roadmap's Stage 4 thesis links (`docs/thesis/*.md`, `docs/thesis/results-data/`,
  `thesis/figures/scripts/`) point at files no longer in the tree — annotated in
  [00-roadmap.md](00-roadmap.md); resolve against the LaTeX thesis sources.

## 6. Fallback decision tree

**F1 — K2 tracing caveat fires** (step 2: make_fx/fake-path error mentioning the dataclass,
dict kwargs, or the splat during `triton_op` tracing/compile/export):
Plan-sanctioned fallback — same dedup, less sugar. In the failing kernel file, change
`_<name>_prep` to return a **plain tuple** (grid + kernel args in positional order) instead of
the frozen dataclass, and splat **positionally** at each launch:
`wrap_triton(_kernel)[grid](q_ptr, k_ptr, …, BLOCK_N=cfg.block_n, num_warps=cfg.num_warps,
num_stages=cfg.num_stages)`. Keep prep/finish shared; only the carrier changes. Prove it on
`kernels/silvertorch/codesigned_probe_score.py`, re-run step 2, then roll to the other six:
`codesigned_probe_score_exact.py`, `filters/clause_mask.py`, `filters/clause_compact.py`,
`filters/bloom_compact.py`, `linr/fused_masked_knn_topk.py`, `linr/oporp_1bit_match_topk.py`
(`bloom_match.py` has no prep and is unaffected). Re-run steps 2–4 after the roll.

**F2 — keyword args into `@triton.jit` helpers rejected** (Triton compile error at a
`common.clause_pass(...)` / `common.bloom_subset_pass(...)` call naming keyword args on the
pinned Triton): switch the helper call sites to positional arguments — grep
`clause_pass(` (call sites in `filters/clause_mask.py`, `filters/clause_compact.py`,
`silvertorch/codesigned_probe_score_exact.py`). Re-run the step-2 clause parity tests.

**F3 — perf gate regression > 5% on one kernel** (step 5; most plausible on the
`clause_pass` consumers via register pressure): revert **only the regressing kernel's body**
to the inlined predicate, leaving a
`# keep in sync with kernels/common.py::clause_pass` breadcrumb comment (measured perf beats
textual purity — the fallback is written into K3). `bloom_subset_pass` / `popcount_int64` /
`compact_store` regressions are implausible; re-measure before touching those. Re-run that
kernel's tune gate + its parity file + step 3.

**F4 — golden-diff quality drift** (step 6 criterion 1 fails): quality differences are
**never acceptable** — stop and bisect by phase commit. Ordered suspects: a `masked_topk`
call-site behavior mismatch vs the K4.1 table (check `pad_to_k`/`valid` flags at the failing
algo's layer), stale oracle caches (delete `gt_topk_*` on **both** sides and rerun both),
TF32/precision pins, and only then kernel numerics (which step 3's parity should have
caught). Report the failing `(algo, sweep, k)` cell and the two row dicts.

**F5 — export test loses the kernel reference** (step 2, `test_export_kernel_ref`): the
launch line has left the decorated body's source (or the indirection confused the tracer) —
confirm every `@triton_op` body contains its own textual `wrap_triton(` call
(grep gate: exactly one per body) and that no helper wraps the launch; if the body is
correct and export still fails, treat as F1 (the carrier, not the location, is the problem).

**F6 — orchestrator smoke failure** (step 7): `output:` unset in the config → the E7.3
guard's `SystemExit` is *correct* behavior (fix the config, not the code); resume skipping
a run you expected to execute → the target JSON already parses as a non-empty list
(`_is_complete`) — delete it or drop `--resume`; nonzero rc from a child `evaluate` → read
`results/_runlogs/current.log`, and remember construction errors are now intentionally fatal
(E3.3).

**Anything else** — if a failure doesn't fit the tree, reproduce it on `main` with the same
command first. Only branch-only failures are refactor regressions; report with the failing
command, both outputs, and the phase commit `git bisect` lands on.
