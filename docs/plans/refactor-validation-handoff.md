# GPU validation handoff — `refactor/kernels-eval` (library half)

> **Status:** **library half validated; harness half archived.** Steps 2–3 passed de facto on
> an A100 on 2026-09-02: the full `retrieve/tests/` suite is green at the branch tip (286 tests,
> [cute-dsl-scorer-artifacts/wp4/pytest-final.txt](cute-dsl-scorer-artifacts/wp4/pytest-final.txt))
> and the K2 tracing caveat did not bite. Step 5 (per-kernel perf gates) has **not** run.
> The harness steps (1, 4, 6, 7) validated a harness that roadmap C3 deleted on 2026-09-06;
> they moved verbatim to
> [archive/refactor-validation-handoff-harness.md](archive/refactor-validation-handoff-harness.md)
> and survive only as **roadmap A1** — the golden baseline on the old harness, run from a
> pre-C3 checkout — whose result is recorded in the [A1 record](#a1-record) section below.
> Written 2026-07-06, verified against the branch's source tree.
> [kernels-layers-design.md](kernels-layers-design.md) stays intact until step 5 passes;
> [archive/evaluation-refactor.md](archive/evaluation-refactor.md) archived with C3.
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

**Harness (`evaluation/`) — evaluation-refactor.md E1–E8.1:** see the
[archived harness half](archive/refactor-validation-handoff-harness.md); the harness it
describes was replaced by harness v2 ([../system/evaluation.md](../system/evaluation.md)).

**Invariants that must hold through validation** (from the plans): parity suite passes with
**unchanged tolerances** (strict equality on OPORP/SimHash); `retrieve::*` op schemas frozen;
buffer names + registration order frozen; `wrap_triton` textually inline in every `@triton_op`
body.

## 2. Environment setup

Requirements: CUDA GPU (perf gates are meaningful on the arch the shipped `DEFAULT_CONFIG`s
were tuned for — A100/sm_80; correctness gates run on any recent NVIDIA GPU) and `uv`.

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

## 3. Ordered validation steps

Run in this order — step 2 is deliberately **before** the full suite because it gates the
pattern all seven kernel files share. Steps 1, 4, 6 and 7 are in the
[archive](archive/refactor-validation-handoff-harness.md).

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

### Step 8 — Sign-off

All green → the refactor is validated. Post-validation follow-ups (not gates):

1. Trim `kernels-layers-design.md` per the roadmap's "Cleanup status" convention
   (`evaluation-refactor.md` is already archived).
2. Decide on retuned `DEFAULT_CONFIG`s only if step 5 motivated them.

## A1 record

Roadmap A1 (the golden baseline on the old harness: goodreads-d128 `c0_genre`, five algos,
`triton` + `torch`; arxiv-d128 `c0_maincat`, `silvertorch`, `triton`) closes the archived
steps 4–7. Its result goes here (date, box, commit, golden JSONs under `evaluation/golden/`,
what passed and what did not) — nothing has been recorded yet.

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

Harness deltas: in the [archive](archive/refactor-validation-handoff-harness.md).

## 5. Known-stale / deferred items (not validation blockers)

- **State-dict load post-hook** (`_global_scale_f` re-derivation; buffers registered in
  `register_index`) — deferred to the torch-export plan (kernels-layers Additional-1).
- **k-means++ init for `KMeansTorch`** — deferred; needs GPU measurement
  (kernels-layers Additional-2; thesis-relevant ablation).
- **Fused in-kernel top-k** and other kernel optimizations — roadmap Stage 3 /
  future-work-and-research.md; out of scope here.
- **`retrieve/tests/correctness/test_quantize.py`** uses the SWAR constant
  `0x5555555555555555` as *test data* — the "SWAR constants appear only in
  `kernels/common.py` + `layers/utils/quantize.py`" grep gate has exactly this one benign
  extra hit.
- ~~**`retrieve/docs/filtering-and-quantization.md`** still shows `filter=`~~ — **fixed**;
  the doc sweep caught these and the later `retrieve/README.md` occurrence too. No `filter=`
  remains anywhere in the docs (`grep -rn 'filter=' --include='*.md' .` is clean).
- The roadmap's Stage 4 thesis links (`docs/thesis/*.md`, `docs/thesis/results-data/`,
  `thesis/figures/scripts/`) point at files no longer in the tree — annotated in
  [00-roadmap.md](00-roadmap.md); resolve against the LaTeX thesis sources.
- The harness items (JSONL streaming, `EVAL_TYPES`, the `users_limit` bug, the ruff-format
  debt) are closed by harness v2 or A1 — see the archive.

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
(`bloom_match.py` has no prep and is unaffected). Re-run steps 2–3 after the roll.

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

**F5 — export test loses the kernel reference** (step 2, `test_export_kernel_ref`): the
launch line has left the decorated body's source (or the indirection confused the tracer) —
confirm every `@triton_op` body contains its own textual `wrap_triton(` call
(grep gate: exactly one per body) and that no helper wraps the launch; if the body is
correct and export still fails, treat as F1 (the carrier, not the location, is the problem).

F4 and F6 (golden-diff drift, orchestrator smoke) are in the
[archive](archive/refactor-validation-handoff-harness.md).

**Anything else** — if a failure doesn't fit the tree, reproduce it on `main` with the same
command first. Only branch-only failures are refactor regressions; report with the failing
command, both outputs, and the phase commit `git bisect` lands on.
