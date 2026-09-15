# Plans roadmap — the master plan

**Read this first.** This is the single ordered work queue for the
repository. Every other document in [docs/plans/](.) is detail for one
phase of this queue; none of them fixes an order, this file does. If you
are an agent starting a session: read [`../../CLAUDE.md`](../../CLAUDE.md),
then this file top to bottom, then only the plan section your step names.
If you are a **worker** dispatched onto one step, read your step's section
and stop there; who dispatches what, on which model, and how many run at
once is [agent-orchestration.md](agent-orchestration.md).

**Conventions.** A plan lives in this directory while it is either a
live instruction set or a record of intent whose validation hasn't
signed off. Once neither is true it moves to [archive/](archive/). System
behaviour is documented in [../system/](../system/), never here — if you
want to know how something *works*, those are the maintained references;
a plan only tells you why it was built that way.

**How to keep this file true.** Each step below has a checkbox, the plan
section it executes, where it runs (`A100` = needs GPU time; `Mac` =
legacy label, needs no GPU time and runs on the box's CPUs — there is no
Mac target since 2026-09-06), its gate, and what it unblocks. When you finish a
step: run its gate, append the validation record to the *plan's own*
record section (the model is
[cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360)),
then flip the checkbox here with the date and commit. Do not start a
step whose dependencies are unchecked. Do not reorder steps without
rewriting the dependency notes. When a phase is fully checked, move its
plan to `archive/` and shorten its entry here to one line under *Done*.

---

> **Session status 2026-09-06 (late evening, A100 VM, coordinator record).**
> Done and merged into `development`: A1, A2, A3, B1, B2, **B4**
> (`62421b1`, A100 suite 504/0/0), B5, C1, C2, C3, E1's gate; E2
> deferred (skeleton merged); two architecture reviews
> ([library](architecture-review-2026-09-06-library.md),
> [evaluation](architecture-review-2026-09-06-evaluation.md)) with their
> "now" findings applied. **C4's code is merged, its gate is not run**
> (`7a21095` = `dev/c4-harness-gate` + `dev/c4-library-fixes`): the
> harness pieces (legacy 1-indexed layout in `data.py`, official cells
> with the plan cache off, `_check_probe_pool` on the official layout,
> `c4_gate.py` + tests) and the three library fixes the C4 findings
> demanded — (i) `clause_compact` / `bloom_compact` as opaque custom ops so
> compiled LiNR V2/V3 capture (`cudagraph_skips == 0` on all six
> layer × filter cells, regression test in `tests/compile/test_linr_compile.py`);
> (ii) deterministic k-means (`bincount` + float64 one-hot GEMM in panel
> order, no atomics; two seed-0 fits `torch.equal`, 1.22× the old wall
> time); (iii) the O §14.7 `-1` id sentinel at `-inf` slots in both Triton
> epilogues. Record: [evaluation-harness-v2.md §9](evaluation-harness-v2.md).
> Merged state verified: library suite **519 passed** on the A100 with
> the official extra, harness suite 103 passed / 1 skipped, ruff 0.15.6
> clean, links 0. No branch carries unmerged code any more; the
> `dev/*` worktrees under `/workspace/wt/` can be pruned.
> **The eval queue, in order, for the next GPU session** (user
> 2026-09-06: "code now, heavy evals later" — nothing below has run):
> 1. **Re-derive A1's golden** on the old harness
> (`/workspace/wt/main-golden`, `tmp/main-users-limit-fix`) against the
> *new* library — its SilverTorch cells came from the atomic k-means and
> its `linr_v2` / `linr_v3` `graph` cells were compiled-eager, so the
> 1e-6 gate against the existing JSONs is not meaningful. Expect
> SilverTorch-triton quality on rows with < k survivors to move
> *down* only: `metrics.py::_hits` masks on `ids != -1`, never on score
> finiteness, so a bloom-rejected item's id in a `-inf` slot could count
> as a hit before the sentinel (torch / official unchanged).
> 2. **C4 gate rerun** — the 14 goodreads-d128 `c0_genre` cells, the
> arxiv cell, kill-and-resume, `c4_gate.py` against the new golden;
> latency compared normalised on `env.sm_mhz` (golden 1140 MHz vs 1410
> under load; 9 cells came out `unstable` from the idle-first-sample
> rule). Then flip C4.
> 3. **B3** — the Triton vs official head-to-head (O §9a/§9b, gap G2).
> 4. **D1** — the campaign (≈ 24 h wall), then D4 `report.py`.
> Kernel improvement (TF-1 transposed bloom, TF-3/4 retune) stays in
> Phase G: gains are claimed only from B3, and a kernel change after D1
> invalidates the campaign through the tree-hash resume key.
> **Operational finding (C4 agent):** inductor's on-disk FX cache
> (`/tmp/torchinductor_root`) does *not* invalidate when a `@triton_op`
> host wrapper's Python source changes — clear it (or set
> `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`) after editing any kernel
> wrapper, or compiled numbers come from the old code.
> **Environment facts that bind everything below:** the VM *is* the
> A100 box; SM clocks cannot be locked (`nvidia-smi -lgc` denied, no
> sudo — timing uses H §7's fallback: sampled `sm_mhz` + `unstable`);
> **Storage, settled 2026-09-15 — see [../system/storage.md](../system/storage.md).**
> Two writable filesystems and no others: the **300 GB container overlay**
> (a real XFS project quota that reports honestly — an 80 GiB probe succeeded
> at 470 MB/s; **262 GB free**) and `/workspace`. The host's 21 TB is **not
> reachable**: no block device nodes exist in `/dev`, `mknod` returns EPERM,
> `CAP_SYS_ADMIN` is not in the bounding set. **User decision: the overlay is
> the disk for everything**, with datasets staged one or two at a time and
> pulled from / pushed to the Hub. The overlay is **ephemeral** (host up 38
> days, `/.dockerenv` stamped 2026-09-14), so only regenerable things live
> there; the repo stays on `/workspace` and is pushed to `origin`.
> `RETRIEVE_DATA_ROOT=/data` after the D1-a cutover. MooseFS reads at ~52 MB/s,
> so this is a speed win too. **Venvs are ~7.6 GB each and reached 91 GB in
> twelve environments on 2026-09-15** — use one shared venv via
> `UV_PROJECT_ENVIRONMENT`.
> **What it changes: D1-c is unblocked outright** (both yambda datasets fit
> simultaneously, so the stage-run-prune dance is unnecessary); **E2 becomes
> network-bound rather than disk-blocked** (raw 198 GB + a 111 GB processed
> tensor is 309 GB and will not fit, but a shard-by-shard streaming ETL peaks
> at ~111 GB with ~150 GB spare); **E3 stays deferred** — 670–840 GB of source
> is more than twice the disk, and the disk was never its only problem.
>
> *Superseded, kept because the reasoning is instructive —* **`/workspace`'s
> free space is an illusion and the quota is real:** `df` reports a 2.1 PB network volume with ~620 TB free; a
> `dd` probe died at **18 GB** with `Disk quota exceeded`. With 8.2 GB already
> resident the enforced quota is **≈ 26 GB, not the 100 GB** this block
> recorded, and **E2 / E3 stay deferred** — 198 GB and 670 GB do not fit by two
> orders of magnitude. Headroom today ≈ 18 GB, which shapes D1: goodreads-d128
> and arxiv-d128 (4.9 GB) are staged and D1-a/b/d/e fit, but **D1-c's `quality`
> suite needs yambda-500m (9.2 GB) and yambda-5b (8.8 GB), which do not fit
> together** — stage one, run it, prune it, stage the next. Parity spill
> `.npz` files (~600–680 MB per run) must be pruned between stages.
> The local disk (300 GB, ~215 GB free) is the place for anything large, and
> `/venvs` already holds 76 GB of per-worktree environments that can be pruned.
> Superseded text:
> per-worktree venvs go under `/venvs/`, and PubMed / Semantic Scholar
> cannot be staged at native dims until disk grows; the fetched HF
> datasets are the pre-`3b1b5b3` 1-indexed layout (handled in
> `data.py`); Meta's package is built with nvcc 12.8 against the cu128
> wheel (`CUDA_HOME=/usr/local/cuda-12.8`).
> **Decisions taken 2026-09-06, do not reopen:** no Mac target (A0
> dropped); **no PCA anywhere — every dataset at its encoder's native
> dim**; big datasets (E2, E3) deferred until disk allows; A4's local
> merge into `main` is on hold until the user says so.

> **Session status 2026-09-15 (A100 box, orchestrator record).** Supersedes
> the eval queue of the 2026-09-06 block above. Merged into `development` and
> pushed: the three refactor plans (**L**, **V**, **X**) and the two contracts
> ([agent-orchestration.md](agent-orchestration.md),
> [coding-guidelines.md](coding-guidelines.md)); **L1** and **L2**
> (`dc58734`); **eval-queue item 1**, the golden re-derive. Merged state
> verified on the box: library suite **613 passed**, harness CPU suite 103
> passed / 1 skipped, ruff clean, links 0.
> **The box is a fresh rental**: the 2026-09-06 worktrees and
> `evaluation/data` are gone. Datasets restaged to `/workspace/data`
> (`RETRIEVE_DATA_ROOT`, 2.6 GB goodreads d128 + 1.3 GB arxiv `content_d128`;
> the Hub's `gt_d128/` oracle blobs hit, so no oracle rebuild). Only CUDA
> **12.4** is installed — correct, and what O §13.1 recorded; the 12.8 line in
> the block above is stale. `uv sync --extra official` at the workspace root
> **uninstalls pytest** (it lives in the members' `dev` groups): use
> `uv sync --extra official --all-packages`. The harness CPU suite must run
> with `CUDA_VISIBLE_DEVICES=""` — three of its tests assert CPU-only
> behaviour in their own text and fail on a box with a GPU. Concurrent GPU
> workers each need their own `TORCHINDUCTOR_CACHE_DIR`; `/tmp/torchinductor_root`
> is shared and does not invalidate on a kernel-wrapper source change.
> GPU work is serialized with `flock /workspace/gpu.lock`.
> **The golden is re-derived** (H §11): 11/11 cells, 6 bit-identical, 5 moved
> by 5×–160× the C4 tolerance, so the C4 gate against A1's cells would have
> failed unattributably. The 2026-09-06 prediction in the block above is
> **wrong in three of its four claims** and is superseded by H §11.3: the `-1`
> sentinel cannot move a goodreads number (only 141 of 10,000 `c0_genre`
> queries have `< k` survivors and they are exactly the dropped users), the
> `torch` backend moved by the run's largest delta, and moves go up as well as
> down. The cause is the deterministic k-means, which changes the centroids on
> every backend. The finding worth keeping: SilverTorch's `triton` and `torch`
> backends now agree **exactly** on all 9 goodreads rows, where A1 had a
> 1.7e-4 gap it read as tie order.
> **Steer 2026-09-15 (user), which re-points everything below.**
> *"We don't care about reproducing old results now, we're improving all code
> and rewriting, then testing and profiling, then running the full evals step
> by step."* The golden baseline stops being a gate and becomes information
> (H WP-4's amendment block): C4 is closed on what the harness proves about
> *itself* — graph capture, resume, cross-backend parity, the official cell —
> not on equality with A1's numbers. The order of work is therefore **code
> first, then tests and profiling, then the evals one step at a time**, and no
> code step waits on a golden comparison. **The paper may not claim
> equivalence with the pre-v2 harness from these numbers** (P G9).
> **Overnight plan, set 2026-09-15 ~22:40 UTC.** Enough work is queued to run
> unattended; each item is dispatched by the orchestrator when the previous one
> hands back, and every stage is resumable, so an interruption costs one stage.
> In order: **D1-a** (running), then **D1-b** (seeds 1–2, ~8–12 h — the longest
> single block and the one that closes P gap G4), **D1-c** (the `quality` suite;
> stage yambda-500m, run, prune, then yambda-5b — they do not fit together in
> the `/workspace` quota), **D1-d** (the deep Pareto sweeps, the most cuttable),
> **D1-e** (the S9 ablation). `bench report` is re-run on the accumulated
> records after each stage, and results are published per the storage policy.
> **Nothing in `retrieve/src/retrieve` may change until D1-e is done** — the
> resume key is the library tree hash, so a kernel edit silently invalidates
> every cell recorded before it. Kernel work that B3 justified (TF-1's
> transposed bloom, and the new TF-9: our padded probe layout is 97 % `-1` on
> goodreads and costs 10.9–17.8× against Meta's scorer) stays in Phase G,
> after the campaign, exactly as O §8 requires.
> **Housekeeping done the same evening, so the box can run unattended:** eleven
> merged worktrees pruned (every branch reachable from `origin/development`;
> `tmp/golden-rederive` pushed), nine stale venvs deleted (`/venvs` 91 GB →
> 23 GB, local disk 59 GB → 187 GB free), 1.2 GB of stray inductor caches
> removed, `/workspace` down to 6.7 GB of its ~26 GB quota. A venv is ~7.6 GB
> and is regenerable with one `uv sync` — **do not keep one per worktree.**

> **Autonomy rules, set 2026-09-15 (user: "finish most of the evals you can
> autonomously").** The orchestrator runs the campaign chain unattended and
> **does these without asking**: verify a worker's gates independently and merge
> its branch; dispatch the next D1 stage; re-run `bench report` on the
> accumulated records; publish results per the storage policy (private); stage
> and prune datasets one or two at a time; prune merged worktrees, stale venvs
> and caches; cut `RETRIEVE_DATA_ROOT` over to the overlay once D1-a is done;
> fix a harness bug that blocks a stage, and record it.
> **It stops and asks** for: anything that would change a written gate or what
> the paper may claim; any change under `retrieve/src/retrieve` (it invalidates
> every recorded cell through the resume key — the whole reason the kernel work
> is parked); publishing anything publicly or minting a DOI; deleting data that
> is not regenerable; and a gate failure whose fix is a plan amendment rather
> than a bug fix. A falsified expectation is **reported, never smoothed** —
> six were falsified on 2026-09-15 and each was worth more than a green tick.

> **The queue, in order, from here:**
> 1. ~~**L3** — deterministic compaction.~~ **Done**; the two affected golden
> cells were re-derived with it and are byte-identical across runs.
> 2. ~~**C4 gate rerun.**~~ **Done and closed** — see the C4 entry. One
> correction it produced, which any later timing comparison needs: the golden's
> `sm_mhz` of 1155 is a whole-run median **dominated by idle samples**;
> filtered to `utilization > 50 %` the golden trace is median 1410, min 1410,
> which is the estimator v2 records. Compare like with like.
> 3. **Code, in parallel on disjoint trees:** **L4** (the LiNR V2 backend
> divergence — a correctness question about our own code, which the steer does
> *not* demote) and **C5** (the harness package split, **V**).
> 4. **Tests and profiling:** **B3**, the Triton vs official head-to-head
> (O §9a/§9b, gap G2).
> 5. **The evals, step by step:** **D1**, then **D4** `report.py`.
> Kernel *improvement* (TF-1 transposed bloom, TF-3/4 retune) stays in Phase G;
> L3 is a correctness fix, not that work.
> **Decisions taken 2026-09-15, do not reopen:** the `retrieve.layers` /
> `retrieve.kernels` shim stays as **temporary tooling** and is deleted at C5
> (the coding-guidelines D2 collision, resolved by the user); `interfaces.Backend`
> is **not** revived in that shim — it was deleted at B4, not moved by L, and
> the frozen golden worktree declares the literal locally instead; the golden
> was re-derived *before* L landed and L1/L2's gate cells confirm the library
> reproduces it bit-exactly; L3 fixes the kernel rather than widening C4's gate.

## 0. Goal, scheduling rule, branch

**Goal.** A reproducibility paper on SilverTorch (Meta) and LiNR
(LinkedIn) built on this repository: our from-the-paper Triton
reimplementation, Meta's official kernels as the reference, one correct
benchmark harness, public datasets up to the papers' scale.
Plan: [reproducibility-paper.md](reproducibility-paper.md) (venue
research lives there; it does not order anything here).

**Scheduling rule.** There are no dates in this file. Every step is to
be done, in the order given; nothing is optional, deferred, or
conditional on a deadline. The A100 is the bottleneck, so GPU steps are
ordered first and Mac-side steps run in parallel with them.

**Branch.** `development` = `main` + the refactor track + the CUDA/CuTe
backends (deleted by B4, merged 2026-09-06) +
everything the 2026-09-06 session merged (status block above); ≈ 146
commits ahead of `main`, nothing pushed to `main`. Work happens on
`dev/<step>` branches in git worktrees under `/workspace/wt/`, merged into
`development` by the orchestrator once a step's gates and the review fixes
are in, and **pushed to `origin/development`** — a step whose code sits only
in a local branch or worktree is not finished
([agent-orchestration.md](agent-orchestration.md) §6). A4 (merge into
`main`) is on hold.

**Who executes a step.** One user-controlled orchestrator session dispatches
constrained workers, at most three at once, one of which may hold the GPU;
`fable` takes the core library and harness rewrites (L1, L2, C5) as single
large chunks, `opus` takes gates, campaigns, reports, loaders, debugging and
docs; nesting is two levels apart from a `sonnet` subagent for web deep
research; every worker's result is a validation record in the plan it
executed. The contract is
[agent-orchestration.md](agent-orchestration.md); it changes no order here.
What the code it writes should look like — thin, little defensive
programming, no compatibility with shapes we invented, no comment slop,
tests only where a gate names them — is
[coding-guidelines.md](coding-guidelines.md), which also lists the three
places it collides with a decision already taken here.

**Decisions already taken (2026-09-05), do not reopen.** Meta's
`meta-recsys/silvertorch` ops become the reference backend
(`backend="official"`); our CUDA C++ and CuTe DSL backends are deleted
after the official parity gate; Triton stays and gets the kernel effort;
the harness is rewritten rather than patched. Rationale in the four
plans and in *Done* below.

**Decisions taken 2026-09-15, do not reopen.** The library takes Meta's
two-level shape — `retrieve.modules` (high-level `nn.Module`s with
builders) and `retrieve.ops` (registered kernels, one namespace per
backend: `triton`, `reference`, `official`) plus the two small helper
namespaces `retrieve.indexing` and `retrieve.functional`; Meta's modules
and ops stay imported and wired in, never copied; the LiNR paper variants
V1–V4 become library modules and the harness keeps only a name → class
table; the harness is split into three packages with one dependency
direction (`bench` → `training` → `eval_datasets`) and the
`retrieval` ↔ `training` cycle goes. Op names, module names, buffer
names and op schemas do not change; k-means++ is opt-in until D1's
numbers say otherwise. Plans:
[library-api-refactor.md](library-api-refactor.md) (**L**),
[evaluation-package-layout.md](evaluation-package-layout.md) (**V**),
[library-harness-boundary.md](library-harness-boundary.md) (**X**).

## 1. The queue

Effort is in focused days from the plans; "A100 h" is wall time on the
box. Plan section references: **H** =
[evaluation-harness-v2.md](evaluation-harness-v2.md), **O** =
[silvertorch-official-integration.md](silvertorch-official-integration.md),
**P** = [reproducibility-paper.md](reproducibility-paper.md) (gaps G1–G16
in its §B.3), **D** = [dataset-candidates.md](dataset-candidates.md),
**L** = [library-api-refactor.md](library-api-refactor.md), **V** =
[evaluation-package-layout.md](evaluation-package-layout.md), **X** =
[library-harness-boundary.md](library-harness-boundary.md).

### Phase A — unblock the tree

- [x] ~~**A0 — make the Mac able to run this repo's Python.**~~ **Dropped
  2026-09-06 (user decision): GPU environments are the only target; Mac
  support is not a goal.** Nothing below depends on it any more — every
  step runs on the GPU box, and the "Mac" label on a step now only means
  "needs no GPU time", i.e. it can run on the box while the GPU is busy.
- [x] **A1 — run the golden cells on the old harness and commit the baseline.** H §6 WP-0 (A100,
  0.5 d). Done 2026-09-06 on `dev/a1-golden`, commit `9856998`; golden
  JSONs and the run record are under `evaluation/golden/`, the validation record is in
  [evaluation-harness-v2.md](evaluation-harness-v2.md) §6 WP-0. The
  `users_limit` row-count fix landed, plus **two bugs the golden run
  found**, both of which had been hidden by steps 4–7 never having run:
  the fetched datasets are the pre-`3b1b5b3` 1-indexed `[N+1, …]`
  artifacts (loud on goodreads, *silent* on arxiv — `cos(query, target)`
  0.99 → 0.62), and K3's `common.clause_pass` is a `NameError` under
  inductor, which failed every compiled filter algo. Backends are
  `triton` + `torch` only; the cuda / cute columns in H's text are void
  (H's amendment). Of the harness half of
  [refactor-validation-handoff.md](refactor-validation-handoff.md),
  **steps 1, 4 and 7 passed**; **steps 5 and 6 are deferred 2026-09-06
  (user: heavy evals later)** — scripted and ready, see that file's
  status. Unblocks: A4, C4.
- [x] **A2 — pin and build Meta's official SilverTorch package on the A100.** O §10 WP-0 (A100,
  0.5 d). `uv sync --extra official`; upstream suite green; sha, `nvcc`
  and build log in `docs/plans/official-silvertorch-artifacts/`. Gate:
  build < 5 min, `torch.ops.st.fused_kmean_ann` exists. Unblocks: A3, B2.
  **Done 2026-09-06, `d2b9248`** (`dev/a0-a3-deps-official`). Pin
  `21aa35e28b6dd9a91e9ee35efb0857715e86bda7`; build 2 m 17 s with nvcc 12.4
  (a *minor* mismatch against the cu128 wheel — warning, not an error, so
  O §11's 12.8 preference relaxes to 12.x); `torch.ops.st.fused_kmean_ann`
  present. Upstream `pytest silvertorch/` **fails at collection** on three
  files whose Buck `load_library` lines survived Meta's `@oss-disable`
  stripping; excluding them, 99 passed / 3 subtests passed, every CUDA test
  included — green for everything the OSS build ships. Two ops
  (`is_topk`, `take_top_k_and_gather_from_main_and_fresh`) are dead source:
  their `.cpp`/`.cu` are not in `setup.py`. The `cute` extra stays until B4
  (rule 5); `official` is added alongside. Record: O §13.1.
- [x] **A3 — measure how the official ops actually behave on the A100.** O §10 WP-1 (A100, 0.5 d): bit-order
  probe, syncs and launches per op, graph-capture attempt, parse cost,
  the `per_embedding_scale` overflow. Gate: O §3 confirmed or corrected
  in O's record section. Unblocks: B1 (the adapter is written against
  measured facts, not read ones).
  **Done 2026-09-06, `137a3c5`** (`dev/a0-a3-deps-official`). **The answer
  B1 needs: the official bloom mask is HIGH-bit-first — document `d` is bit
  `63 - (d % 64)` of word `d // 64`** (three independent probes agree), so
  take the HIGH-first branch of the two the B1 note below asks for. O §4.2
  (i)–(iii) confirmed, including `per_embedding_scale` returning `inf` in
  every slot at D=128. Corrected: `fused_kmean_ann` costs 3 syncs and 19
  launches, not 2 and ≈ 12; `_with_partial_masks` 4 syncs;
  `bloom_index_search_batch` *captures* into a CUDA graph and then faults on
  replay, so it needs to be on the harness's not-capturable list explicitly
  (D7 unchanged). Parse cost at B=16: 58.7 µs/call, 3.67 µs/query. Record:
  O §13.2; script and raw output in
  [official-silvertorch-artifacts/](official-silvertorch-artifacts/README.md).
- [ ] **A4 — merge `development` into `main`.** (**On hold 2026-09-06 — user decides when**). After A1 passes: merge `development`
  into `main` (library gates passed 2026-09-02, harness golden passed in
  A1). Everything below happens on phase branches off `main`, merged
  back through `development`.

### Phase B — official backend, parity, deletion

- [x] **B1 — write the official-backend adapter and its parity tests.** O §5, §10 WP-2 (Mac, 2 d): `backend=
  "official"` in `SilverTorch`, `require_official`, parity tests T1–T7.
  Gate: `ruff` clean, suite collects and skips on the Mac. Needs A3.
  Authored 2026-09-06 on `dev/b1-official-adapter` (Mac gate green; A3's
  bit order pinned as `OFFICIAL_BIT_ORDER = "high_first"`, with the
  library review's "now" items — see
  [architecture-review-2026-09-06-library.md](architecture-review-2026-09-06-library.md)).
  **Done 2026-09-06, `aadc380`** (`dev/integration`): GPU gate — 43/43 in
  `test_official.py` on the A100 after 11 **test-side** fixes (T1-exact's
  CSR doc space, T3's mirrored-doc negative control, `-inf`-slot id
  normalisation in T6, the fd-2 sync instrument in T7); no adapter or
  kernel change. Record: O §14.3.
- [x] **B2 — prove official and Triton scores agree bit-exactly on the A100.** O §10 WP-3 (A100, 0.5 d). Gate: phase-3
  scores `torch.equal` on the int32 path on every regime, bloom ⊇ check
  and FPR at matched memory recorded in O's record section. **Unblocks
  B4 (deletion) — never delete before this is green.**
  **Done 2026-09-06, `2b09f8e` + `0521a67` + `aadc380`** (`dev/integration`,
  nvcc 12.8 build, clocks unlocked — counts and bit comparisons only).
  T1 `torch.equal` vs the reference and vs Triton on all 8 regime cells +
  4 exact-mask cells; bloom ⊇ exact (AND) and ⊆ exact (NOT) with 0 false
  negatives / positives; FPR at matched memory (byte-exact at
  `b_multiplier = m_bits / (max_terms · 5)`) 0.0000 for both blooms on
  synthetic attrs — the real-attribute calibration stays D3; T7 syncs per
  forward 3 / 3 / 7 / 4 (none / exact / bloom-partial / bloom-full),
  identical with `cache_plans` on and off. Full suite **574 passed, 0
  failed, 127 skipped (all cute — extra not installed)**; upstream suite
  99/99 on the 12.8 build; the three pre-existing red cells fixed (fp32
  literal compare → `torch.equal` on the fp32 inputs, ungated cute cell
  gated); the review's deferred `argsort(stable=True)` applied after
  measuring it bit-identical on nine regimes. Record: O §14. **B4 is now
  unblocked.**
- [x] **B3 — benchmark Triton against the official kernels, kernel-only and end
  to end.** **Done 2026-09-15** (`399c231`, merged at `HEAD`); closes paper gap
  **G2**. Triton is the fastest arm end to end in every cell (1.3–1.6×
  unfiltered, 1.2–1.9× bloom, 2.9–10.1× exact vs `official`; 7.3–32.7× vs
  `torch`), **but Meta's scorer kernel is faster than ours in every cell**
  (1.15–3.2× arxiv, **10.9–17.8× goodreads**) and we win only on payload prep
  (their 268–896 µs over 73–95 launches against our 34–58). The 10× is **our
  padded probe layout, not our scorer**: on goodreads `n_probe ·
  max_cluster_size` is 611,520 slots for ≈18.7 k real items, so **97 % of what
  our kernel walks is `-1` pad** (59 % on the less skewed arxiv IVF). O §9's
  "parity within ±20 %" is falsified in both directions, and so is its
  expectation (iii): their bloom *forward* is 1.8–2.1× slower than our fused
  one, while their **transposed bloom search beats ours 2.0× at 0.8 M items and
  6.1× at 3.0 M** (ours grows 3.3× with N, theirs 1.07×) — the strongest
  evidence yet for TF-1. Parity: the official int32 path is **bit-exact**
  against Triton on both datasets in all three filter modes. Record:
  [silvertorch-official-integration.md §16](silvertorch-official-integration.md). O §9a/§9b, WP-4
  (A100, 1 d). Gate: JSON + tables appended to O. This is paper gap G2.
- [x] **B4 — delete the CUDA C++ and CuTe backends.** (Merged 2026-09-06, A100 gate 504/0/0). O §7, WP-5 (Mac,
  1 d). Tag the parent commit `cuda-cute-backends-final`; move
  [cuda-silvertorch-handoff.md](archive/cuda-silvertorch-handoff.md),
  [cuda-silvertorch-phase2.md](archive/cuda-silvertorch-phase2.md),
  [cute-dsl-scorer.md](archive/cute-dsl-scorer.md) and
  [cute-dsl-scorer-artifacts/](archive/cute-dsl-scorer-artifacts/README.md) to
  `archive/`; rewrite the system docs O §7 lists. Gate: suite green on
  the A100, collect-only on the Mac, `git grep -il "cute\|codesigned_probe_score_cuda"`
  hits only `docs/plans/archive/`. Needs B2.
  **Authored 2026-09-06 on `dev/b4-delete-cuda-cute`; A100 gate green
  2026-09-06** (`4d92432` deletion, `010681d` `Backend` split — review
  item 5, the docs commit, `a435179` a test fix, plus the record): parent
  `41d4479` tagged `cuda-cute-backends-final`; 3,677 lines of kernel/host/
  test code gone, `build_transposed_sigs` moved to `bloom_hash.py` for
  TF-1. **Library suite on the A100 with the `official` extra: 504 passed,
  0 failed, 0 skipped** (the 127 `cute` skips of B2's run 3 are gone with
  the backend, and no official cell skipped); evaluation suite 93 passed /
  1 skipped (C4's cell), links 0, ruff clean on `retrieve` +
  `evaluation/retrieval`, the grep hits only the mandated tag name outside
  `archive/`. The one red cell of the first run — `SimHashKNN` missing
  `k_bits` in the new backend-rejection parametrize, invisible to a
  collect-only gate — is fixed in `a435179`; no library source changed.
  Record and the grep-rule reading: O §15, §15.6. The coordinator flips
  this box after merge.
- [x] **B5 — register the bloom salt as a buffer instead of a per-call host tensor.** O §8 TF-2 (Mac, 0.5 h; validate with
  `test_bloom_hash.py` on the A100). Do before any campaign timing.
  Authored 2026-09-06 on `dev/b1-official-adapter` (commit `2dbee72`).
  **Done 2026-09-06** — GPU gate passed in B2's full-suite run 3
  (`test_bloom_hash.py` incl. the buffer-vs-inline bit-equality on CUDA
  and CPU, and every bloom row of `test_silvertorch.py`, green; O §14.2).
  The raw-capture latency claim is unmeasured until WP-7.

### Phase L — library API refactor (CPU work on the box; two library-suite runs)

The `retrieve` package takes Meta's `modules` / `ops` shape (**L** §3);
structure-preserving first (L1), features second (L2). Both land
**before the C4 gate rerun** (status-block queue item 2) and before D1,
because `code_version` is the library tree hash and every recorded cell
dies with it (**L** D11). One branch, `dev/l-library-layout`, off
`development`; nothing else edits `retrieve/` while it is open.

- [x] **L1 — move the library into the `modules` / `ops` / `indexing`
  layout with no behaviour change.** **Done 2026-09-15** (`df04e74`, merged at
  `dc58734`): library suite 521, 14 reference cells `torch.equal` to the
  pre-move `ref_cps_phase23`, 9 pre-move state dicts returning identical ids
  and scores, and the `goodreads-d128 c0_genre silvertorch triton` golden cell
  bit-identical through the shim. Record:
  [library-api-refactor.md §12.1](library-api-refactor.md). L §10 WP-1 (CPU 1.5 d + one library
  suite run). `modules/`, `ops/{triton,reference,official}/`, `indexing/`,
  `functional.py`, the ops loader and `_host.py` (review A5), the
  `retrieve.layers` / `retrieve.kernels` deprecation shim, pyproject
  `name = "torchretrieve"` 0.2.0, the review's A8 / A9 / T3 / B.4
  leftovers, tests and docs re-pointed. Op schemas, buffer names and
  registration order unchanged; the pure-torch eager branches are
  *extracted* into `ops/reference`, not rewritten. Gate: `ruff` clean,
  `tests/test_public_api.py`, links 0, harness CPU suite green through the
  shim; on the GPU the full library suite green, every parity file
  bit-exact, `ops.reference.codesigned_probe_score` `torch.equal` to the
  pre-move `ref_cps_phase23`, a pre-move state dict (3 backends × 3 filter
  modes) loads and returns identical ids and scores, the golden worktree
  runs one cell through the shim. Needs nothing unchecked.
- [x] **L2 — ship LiNR V1–V4 as library modules, the builders, k-means++
  seeding, build timings and the harness contract.** **Done 2026-09-15**
  (`b81d2ca`, merged at `dc58734`): library suite 613 (+92), composites
  `torch.equal` to the hand-composed primitives on every backend × filter
  kind, builder round-trip against a fresh build, `test_boundary.py` green,
  k-means++ greedy at `n_lists=8192` in 9.5 s so k-means‖ was not needed.
  Record: [library-api-refactor.md §12.2](library-api-refactor.md). L §10 WP-2 (CPU 2 d +
  one library suite run). `LiNRV1`–`LiNRV4` moved from
  `evaluation/retrieval/algos.py`, `SilverTorchBuilder` / `LiNRBuilder`
  (`build_silvertorch` retired into the shim), `KMeans(init="kmeans++")`
  opt-in (L D9), `set_query_params` moved into the library,
  `build_timings`, `capturable` as a class attribute,
  `retrieve.modules.official`, `interfaces.DISPATCH`, the **X** §4 clauses
  and `test_boundary.py`; user guide and system docs per L §8. Gate: as
  L1 plus composites `torch.equal` to the hand-composed primitives on every
  backend × filter kind, builder round-trip, k-means++ tests. Needs L1.
  **Unblocks C5** and, with it, the C4 rerun on the final layout.

- [x] **L3 — make the Triton stream compaction deterministic.** **Done
  2026-09-15** (`4837a6c`, merged at `HEAD`): ascending-order compaction
  `torch.equal` to `ops.reference`, launch-to-launch identity in and across
  processes, library suite 645 (+32), and both affected golden cells
  re-derived and byte-identical across runs. The shipped kernel is **not**
  the plan's D3 — measurement falsified its premise (`clause_compact` is
  34–47 % of a `linr_v2` forward and instruction-bound, so recomputing the
  predicate cost +43–58 % end to end); the tile-stash shape costs ×1.02–1.05
  under graph. D3 amended with the numbers. Two defects found and pinned: a
  `[1]`-view `counts` tripping inductor's `assert_alignment` on the compiled
  bloom V2 forward at `B = 1`, and a count-before-scan epilogue slowing the
  predicate launch 1.7–1.8×. Record:
  [deterministic-compaction.md §7](deterministic-compaction.md).
  [deterministic-compaction.md](deterministic-compaction.md) §5 WP-1 (A100,
  0.5 d). `clause_compact` / `bloom_compact` claim their per-row base with an
  `atomic_add`, so a row's surviving ids land in tile-completion order; the
  tie-breakers downstream (`PrefilterKNN`'s top-k, and `OneBitKNN`'s
  massively-tied integer Hamming ranking, which decides *pool membership*)
  turn that into quality noise of **2.0e-6 on `linr_v2` and 6.8e-5 on
  `linr_v3`** — 2× and 68× C4's `1e-6` gate, measured across three identical
  runs (H §11.8). The invariant becomes ascending item order, matching
  `ops.reference`, via a two-phase kernel (per-tile counts → exclusive scan →
  write at fixed offsets). Gate: order parity `torch.equal` to `ops.reference`,
  launch-to-launch identity in and across processes, full suite green, the cost
  measured, and the two affected golden cells re-derived and bit-identical to
  each other. Needs L2. **Blocks C4** (its gate is unmeetable on two cells
  until this lands) **and D1** (whose WP-5 gate asks for a byte-identical
  rerun). User decision 2026-09-15: fix the kernel rather than widen the gate.

- [x] **L4 — settle the LiNR V2 backend divergence.** **Done 2026-09-15**
  (`91fd352`, merged at `d9a3200`): the candidate sets are **identical**
  (0/9859 rows differ on counts or ids, `torch.equal`), so it is arithmetic —
  `fused_masked_knn_topk` **accumulates in fp16** (`tl.sum` keeps the operand
  dtype; the PTX has 8 `add.f16` and no f32 adds), and on goodreads the partial
  sums reach |s|=31 against a top-100 near 0…-1, i.e. catastrophic
  cancellation: **0.0276** max abs score error against an fp64 dot. All 626
  swapped pairs lie inside that error and the top-k epilogue ranks correctly in
  626/626. `knn.py` and `kernels.md` both claimed fp32 — the code contradicted
  its own contract, and the parity suite could not see it because it compares
  Triton against a reference *at the same precision*. Record:
  [linr-v2-backend-parity.md §6.1](linr-v2-backend-parity.md). **L4-b is
  falsified** (below); **L4-c** is fixed in C5.
- [x] **L5 — fp32 accumulation, and the precision audit it implies.** **Done
  2026-09-15** (`0469741`, merged at `HEAD`): two casts; max abs error vs an
  fp64 dot **0.027644 → 3.2e-6**; `linr_v2` torch-vs-triton **0.998743 →
  0.999751** with all 124 residual rows being `torch`'s own fp16 ties; cost
  B=16 **0.9631 → 0.9404 ms** (faster). The audit found the fixed reduction was
  the **only floating-point reduction in `ops/triton/`** — the int8 scorers use
  `tl.dot(out_dtype=int32)` (exact below 2²⁴), oporp sums int32 popcounts, the
  compaction reductions are integer — so **no SilverTorch number moves** and B3
  is unaffected. `tests/parity/test_accumulation.py` bounds each scoring kernel
  against fp64 *and* asserts an fp16 model of the same computation misses the
  bound, so its discriminating power is checked every run. The shim is deleted.
  Suite 653. Record: [linr-v2-backend-parity.md §8.1](linr-v2-backend-parity.md).
  [linr-v2-backend-parity.md §7](linr-v2-backend-parity.md) (A100, 0.5 d).
  User decision 2026-09-15 on L4's numbers: ship fp32 accumulation — it is not
  a trade-off (B=1 0.05235 → 0.05226 ms, B=16 0.9636 → **0.9408** ms, i.e.
  faster; parity 0.998743 → 0.999751, the residual 124 rows being `torch`'s own
  fp16 ties) — **and sweep every Triton kernel for reduction width**, because
  `tl.sum` inherits the operand dtype and the parity suite is structurally
  blind to the whole class. Adds a parity file against an **fp64 oracle**, the
  test that would have caught it. Also deletes the `retrieve.layers` /
  `retrieve.kernels` shim, whose reason to exist ended when the golden stopped
  being a gate. Gate: suite green (645), existing parity bit-exact, the fp64
  file green, the audit table complete, no cost regression. **Before B3 and
  D1.**
  [linr-v2-backend-parity.md](linr-v2-backend-parity.md) (A100, 0.5 d).
  `linr_v2` is our **exact** filtered top-K, yet `torch` and `triton` return
  different results: jaccard@100 **0.998743**, `score_max_abs_diff` **9.77e-3**,
  and the *golden* reproduces it independently (torch 0.99969470 vs triton
  0.99927376, **4.2e-4** recall) — so it is in the library, not the harness,
  and it predates the v2 rewrite. `linr_v1` and `linr_v4` agree at exactly 0.0
  on the same cells, which is the control. The question to settle first is
  single and decidable: **do the two backends select the same candidate set**
  (post-L3 the compaction is deterministic and ascending, so they should) **and
  differ only in fp16 dot-product rounding near the top-k boundary, or do they
  genuinely disagree?** The first is precision and is documented; the second is
  a bug and is fixed. Carries two smaller open items from C4: **L4-b**, the
  decisive `linr_v4` chunk-64 experiment; **L4-c**, the `clocks_drift` /
  `clocks_locked` harness artifacts (H §8.2 F). Needs L3. **Before D1.**

### Phase C — harness v2 (Mac work in parallel with B; A100 gate at the end)

Backends everywhere in H are now `triton | torch | official` (H's
amendment). The official backend is eager-only (O D7), so H's `graph`
mode applies to `triton` and `torch` only. **H §8** amends H §2/§3 with
the findings of a survey of ann-benchmarks, big-ann-benchmarks,
VectorDBBench, MTEB, cuVS bench, MLPerf, asv, Criterion and the
results-as-data practice of ClickBench and db-benchmark
([artifacts](evaluation-harness-v2-artifacts/README.md)); §8.4 lists what
each WP below gains. Three change behaviour rather than schema: build-time
params (`n_lists`) sweep separately from query-time params (`n_probe`,
`candidate_pool`), which turns the `deep` sweep's 12 builds into 2; the
resume key includes the library subtree's tree hash, so a kernel change
invalidates a cell instead of silently reusing it; and the campaign's
process boundary moves down to `(dataset, dim, algo, backend)` (H §8.2 K,
user decision 2026-09-06) so no dynamo cache or CUDA graph pool outlives
the backend under test — cross-backend parity moves to a spill file, and
bit-exactness stays where it belongs, in B2's library parity suite.

- [x] **C1 — write the measurement primitives, device-side metrics and the algorithm table.** H §6 WP-1
  (Mac, 2 d). Needs A1 (golden exists). Includes O §6.2 / WP-6: the
  `official` path in the `PATHS` table.
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU tests green,
  **gate closed 2026-09-15 with C4** (`c1ae1b5`); the code has since been re-split by C5. `algos.py` landed as `algos_v2.py` (the old
  `algos/` package shadows the name until C3 deletes it); `metrics.py`
  rewritten in place with the old per-row API kept as wrappers.
- [x] **C2 — write the config matrix, per-dataset inputs and the oracle with pass rates.** H §6 WP-2
  (Mac, 1.5 d).
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU tests green,
  **gate closed 2026-09-15 with C4** (`c1ae1b5`); the code has since been re-split by C5. `config.py` and `oracle.py` rewritten in place with
  the old API kept below a divider / as wrappers; `data.py` is new and
  imports `encode.py` (kept as the one `training.*` boundary). Oracle
  caches are now blob v4 with the fingerprint in the file name.
- [x] **C3 — write the cell loop and CLIs, delete the old harness, rewrite the docs.** H §6 WP-3 (Mac,
  1.5 d): delete the old harness files H §5 lists, rewrite
  [../system/evaluation.md](../system/evaluation.md) to H §2, archive
  [evaluation-refactor.md](archive/evaluation-refactor.md) and the harness half
  of the refactor runbook.
  Status: authored 2026-09-06 on `dev/c1-harness-v2`, CPU e2e green, GPU
  gate C4 pending. `run.py` (the cell loop, JSONL + samples sidecar,
  resume by key + `code_version`, parity spill, failures as `status:
  failed`), `cli.py` (`bench run` / `campaign` / `report` stub),
  `upload.py`; the old harness, its 19 YAMLs and the `evaluate` /
  `run-evaluation` / `stage-results` scripts deleted (H §5);
  `algos_v2.py` → `algos.py`; [../system/evaluation.md](../system/evaluation.md)
  rewritten; validation record in H §9.
- [x] **C4 — validate the new harness against the golden baseline on the A100.**
  **Closed 2026-09-15** (`32d4877`, merged at `c1ae1b5`) under the user's
  superseding decision the same day — *"we don't care about reproducing old
  results now"* — which demotes the golden comparison from a gate to
  information. What the run actually established, and what closes this step:
  20 cells all `status: ok`, `cudagraph_skips == 0` 20/20, kill-and-resume
  clean, `official` end to end on goodreads at jaccard 0.9998 with the plan
  cache off, SilverTorch `torch` vs `triton` **exactly 1.0** with
  `score_max_abs_diff 0.0` on both datasets, and quality matching the
  re-derived golden on 9/11 cells at a worst passing delta of 7.5e-9. What it
  did **not** establish, carried forward rather than buried: `linr_v4`'s 7.3e-5
  (chunk-induced int8 boundary ties, attributed but the decisive chunk-64
  experiment unrun), arxiv `silvertorch`'s unattributed 2.0e-6, and the
  `linr_v2` backend divergence, which becomes **L4**. Gate amendments and the
  evidence behind each: [evaluation-harness-v2.md](evaluation-harness-v2.md)
  WP-4's amendment block; run record §12. H §6 WP-4 (A100, 1 d). Gate: quality within
  1e-6 of the A1 golden, graph latency within 5 %, `cudagraph_skips ==
  0`, `jaccard_vs_first@100 == 1.0` torch-vs-triton, one `official` cell
  runs end to end on goodreads-d128 `c0_genre` with jaccard ≥ 0.99 (O
  WP-6's gate). Unblocks: D1.
  **Status 2026-09-06:** partial, on `dev/c4-harness-gate` — see the
  session status block at the top; results JSONL under
  `evaluation-harness-v2-artifacts/c4/` on that branch. The three library
  prerequisites from the C4 findings are done and merged (`7a21095`, from `dev/c4-library-fixes`:
  `dd8b7b5` cudagraph capture verified, `d5d824b` deterministic k-means,
  `8df7e9a` the O §14.7 `-1` sentinel), with the record in
  [evaluation-harness-v2.md §9](evaluation-harness-v2.md); the gate itself
  still needs the cells re-run, and **A1's golden must be re-derived first**
  — its SilverTorch cells were produced with the non-deterministic k-means
  and its `linr_v2` / `linr_v3` `graph` cells were compiled-eager, so
  comparing against them at 1e-6 is not yet meaningful.
  **Amended 2026-09-15:** the rerun happens on the L2 library layout
  (Phase L above), so the gate validates the final code once; the harness
  side of the gate is unchanged by L (the wrappers are moved, not
  rewritten, and `algos.py` is re-tabled in C5 *after* this gate flips).
- [x] **C5 — split the harness into `bench` / `training` / `eval_datasets`
  with one dependency direction, one test tree and three CLIs.** V §9
  WP-V1 + WP-V2 (CPU 2 d). `retrieval/` → `bench/` (`bench.py` →
  `measure.py`, `records.py` gathering `SCHEMA_VERSION` / `KEY_FIELDS` /
  `resume_key` / `append_record` / `read_keys` / `flatten`), `algos.py`
  reduced to the **X** §3 table with `PATHS` derived from
  `retrieve.interfaces.DISPATCH`; `eval_datasets/layout.py` (review §1.11:
  the on-disk contract as code, `bench check`, the readers out of
  `data.py`, `users_limit` once), `hub.py`, `etl/`; `training/encode.py`
  (the cycle goes; `training/evaluate.py` keeps a frozen local recall /
  ndcg, V D5); `evaluation/tests/` with conftest, `gpu` marker,
  dependency-direction test; `bench` / `eval-data` / `train` click groups
  (12 scripts → 3); pre-v2 `results/**` to `results/archive/`, the two
  old-schema YAMLs to v2 shape, the `__pycache__`-only dirs deleted;
  system docs and CLAUDE.md's commands. Gate: `ruff`, `cd evaluation &&
  uv run pytest tests/` green, links 0, every `--help` renders, and one
  `bench run` cell on goodreads-d128 `c0_genre` (`silvertorch`, `triton`)
  reproduces the pre-rename record's key block and `quality` at the same
  `code_version`. Needs C4 (flipped) and L2. **Unblocks D1** (the campaign
  runs on the final package).
  **Done 2026-09-15** (`6900306`, merged at `d9a3200`): 164 passed / 4 skipped,
  ruff clean (the 11 pre-existing E501s cleared), dependency-direction test
  green against **X** §2, `PATHS == derive(DISPATCH)` replacing the
  markdown-parsing test, all 20 `--help`s, links 0, and the one-cell gate
  reproducing C4's pre-rename records with **all 31 `quality` fields equal to
  the digit** at the same `code_version`. **L4-c fixed here**: `clocks_locked`
  and `--expected-sm-mhz` removed (they recorded a coincidence on a box that
  cannot lock clocks), `env.sm_mhz` split into `sm_mhz_idle` / `sm_mhz_load`,
  `clocks_drift` now compares under-load samples with under-load samples;
  `SCHEMA_VERSION` 2. **L4-b falsified**: `linr_v4` at quality chunk 64 lands
  **2.9e-4** from the golden — four times *further* than chunk 16's 7.3e-5 —
  so batch shape moves the cell but the chunk difference is not what separates
  the two harnesses, and that residual is unexplained again. Record:
  [evaluation-package-layout.md §11](evaluation-package-layout.md).

### Phase D — campaign and baselines (A100)

- [ ] **D1 — rerun the full campaign on the new harness.** H §6 WP-5 + O WP-7 (A100 ≈ 24 h wall
  + 0.5 d): four datasets × dims × `{triton, torch, official}` × seeds
  {0, 1, 2} on headline sweeps, `n_probe ∈ {24, 32}`, the S9 co-design
  ablation cells. Closes in one pass: the goodreads oracle rerun (old
  open item 2), P gaps G3 (P99/QPS), G4 (seeds), G7, G8 (cross-dataset
  deep sweeps), old §4b items 1, 3, 4, 5, 7. Gate: H WP-5's. Needs C4,
  L2 and C5 (a library change after D1 invalidates the campaign through
  the tree-hash resume key; a harness rename after it changes the recorded
  command lines and upload paths).
- [ ] **D2 — add Faiss, HNSW, cuBLAS and cuVS baselines as harness algorithms.** P G5 (A100, 2–3 d): Faiss-GPU
  IVF-Flat, Faiss-CPU IVF-Flat, HNSW, cuBLAS brute-force floor at
  matched recall, as harness algos. Then P G13 (cuVS IVF-Flat / IVF-PQ /
  CAGRA with bitset prefilter) and G14 (Filtered-DiskANN, ACORN on the
  CPU box, or FANNBench's harness on one of our datasets).
- [ ] **D3 — measure bloom false-positive rate and memory against filter width.** P G6 (A100, 1–2 d), on
  both blooms (ours and official), real attributes.
- [x] **D4 — generate every thesis and paper table and figure from the records.**
  **Code done 2026-09-15** (`6be479c`, merged at `fcc8d7e`), **run early and
  deliberately**: the campaign needed the GPU that B3 was holding, and D4 needs
  none, so `bench/report.py` was built against the record *schema* and today's
  C4/C5 records. 18 artifacts emit without a traceback, suite 170 passed / 4
  skipped, LaTeX validated structurally (no TeX toolchain on the box).
  **The artifacts are not the paper's**: they are built from pre-campaign
  probe records and every one of them says so — rule 2 is enforced in the tool
  (`--gate` is vetoed by a `failed` / `partial` / dirty / branch-provenance
  record, verified: `--gate D1` over today's records still prints NOT CITABLE).
  **Re-run it on D1's records**; `tab:recall_nofilter` is empty until D1-c
  supplies a `filter_kind: none` cell, and the Pareto and deep-sweep figures
  have never seen a real sweep. Record:
  [evaluation-harness-v2.md §13](evaluation-harness-v2.md). H §6 WP-6 (Mac, 1.5 d): thesis/paper tables
  and figures from the JSONL only, plus the methodology paragraph. Lands
  as `bench/report.py` reading `records.flatten()` (V §5.3).

### Phase E — datasets (GPU box in parallel with D)

**Decision (2026-09-05, final).** The study's dataset set: **arXiv** and
**Goodreads** (rerun on the new harness, D1), **YFCC-10M**, **PubMed +
MedCPT**, **Semantic Scholar SPECTER2** (with **OpenAlex** as the fallback
if the S2 API key is not granted), **KuaiRand-27K** — three
semantic-search corpora beyond arXiv and one recsys corpus beyond
Goodreads, all with real filters and ready or locally-encodable query
encoders. Dropped: Amazon Reviews 2023 (out in general), Yambda-full and
Cohere Wikipedia (scale without meaningful filters, closed query
encoder; the existing Yambda unfiltered runs stay as they are). Survey
and ingestion plans: [dataset-candidates.md](dataset-candidates.md)
§3–§4.

- [ ] **E0 — request the Semantic Scholar API key.** D §3.7 (free
  research partner form). First thing in this phase: it is the long
  pole for E3. If it is refused, E3 runs on OpenAlex.
- [x] **E1 — ingest YFCC-10M with its shipped filtered ground truth.** (GT gate passed 2026-09-06; the two cells run at E5.) D §3.4 / §4.5 (download-bound, ≈ 3 GB; A100
  minutes). Retry the download with the exact URLs in D §3.4 (served on
  2026-09-05). Ingest: uint8 CLIP → fp16/int8 codes; tag bags → the
  narrow clause tensor (cap K per D §3.4 gotcha a); the shipped 100k
  queries with their tag predicates and filtered GT become the query
  set, so this is the one dataset whose filtered ground truth is *not*
  ours. Gate: our exact filtered oracle reproduces the shipped GT on the
  100k queries; one `none` + one filter cell run.
  *Status: download + parse + loader done 2026-09-06, GPU gate pending.*
  (`evaluation/eval_datasets/yfcc.py`, `yfcc_check_gt.py`,
  `config/yfcc10m/d192-filter.yaml`; 200-query CPU subset of the gate
  reproduces the shipped GT bit-exactly. Two deviations from D §3.4 to
  carry into the paper: the tag clause tensor is capped at 32 tags/item,
  which makes the *harness* predicate stricter than the shipped one on
  25.8 % of queries — the shipped GT is validated against the uncapped
  CSR instead — and the harness scores cosine while the shipped GT is
  squared L2. Both in [../system/datasets.md](../system/datasets.md#yfcc10m).)
  **Gate record 2026-09-06 (A100, `development` @ 41d4479):** our exact
  filtered oracle (conjunctive AND over the uncapped tag CSR, squared L2 in
  fp32, TF32 off) reproduces the shipped `GT.public.ibin` on **100,000 /
  100,000** queries, all id-exact, `max_abs_distance_error = 0.0`, 1276 s
  wall — [gt_check-cuda-100k.json](dataset-candidates-artifacts/yfcc10m/gt_check-cuda-100k.json).
  The "one `none` + one filter cell" half is a harness-v2 campaign cell and
  runs with E5 (user 2026-09-06: heavy evals later).
- [ ] **E2 — ingest PubMed with MedCPT embeddings and MeSH filters, ~36 M articles.** D §3.8 / §4.1
  (download-bound: 102 GB of 768-d fp32 embeddings + 44 GB of per-PMID
  JSON from the NCBI FTP, public domain, no registration; no encoding).
  PCA to 256 / 128 / 64 for the dim sweep (fit on a 1 M sample, apply on
  GPU, ≈ 1 h); queries encoded locally with the open
  `ncbi/MedCPT-Query-Encoder` and projected with the same PCA:
  NFCorpus / TREC-style biomedical query sets plus item-as-query;
  attributes: MeSH descriptors (multi-valued, ~30 k), year, journal /
  language via the MEDLINE join. Gate: layout on disk, oracle built,
  one `none` + one filter cell.
  **Status: deferred 2026-09-06 (user decision): no PCA, native 768-d only,
  heavy ETL/evals later; loader skeleton on `dev/e2-pubmed`.** What exists:
  `evaluation/eval_datasets/pubmed.py` (download / verify / medline / convert /
  attrs / queries / encode_queries) with CPU tests on synthetic fixtures,
  `config/pubmed/d768-filter.yaml`, the `pubmed` → `pinkmeme/eval-pubmed`
  registry entry (unpublished) and the docs/system/datasets.md section. What is
  not done: no data staged (the ~198 GB raw mirror does not fit the 100 GiB
  `/workspace` quota — see the section for the budget), no PCA anywhere, no
  oracle, no cells. The `--dims 256,128,64` PCA plan in D §4.1 is superseded.
- [ ] **E3 — ingest a 50 M-paper Semantic Scholar SPECTER2 slice, with OpenAlex as the fallback.** D §3.7
  (`embeddings-specter_v2`: 30 files × 28 GB JSONL ≈ 840 GB for 120 M
  papers, 768-d; `papers` for year / venue / fields of study /
  publication type / open access / citation bucket; `citations` for
  citation-based relevance; SPECTER2 weights are open, so text queries
  encode locally). Stream the 30 files, keep a 50 M slice with
  abstracts + English + year ≥ 2000, PCA to 256 / 128 / 64 as in E2.
  Queries: held-out papers; relevance = cited papers *and* the exact
  filtered oracle; natural filters = year < query year, same field.
  **Fallback if E0 fails: OpenAlex** (D §3.9: same attribute shape,
  citation links, but ~670 GB snapshot pass + ≈ 9–14 A100 h of nomic
  encoding). Gate: layout on disk, oracle built, one `none` + one filter
  cell.
- [ ] **E4 — ingest KuaiRand-27K and train gSASRec over its 32 M videos.** D §3.2 / §4.3. ETL to the
  harness layout; train gSASRec D=128 with a *shared* item table (two
  32 M-row tables are ~100 GB fp32 + Adam) or train on the 5-core
  subset while indexing all 32 M; attributes: video_type, upload_type,
  category hierarchy, tags, duration / upload-date buckets. Two filter
  protocols: target-derived (optimistic) and business-rule (exclude
  ads, duration bucket; pessimistic, LiNR-style pass-rate tiers). Gate:
  checkpoint on HF, layout on disk, one `none` + one filter cell.
- [ ] **E5 — run the campaign on the four new datasets and extend the
  report.** The E1–E4 cells on the D1 harness state. Needs D1.

### Phase F — the paper (F1 and F3 can start any time)

- [x] **F1 — reframe the thesis as SilverTorch Algorithm 1 and write the deviations table.** **Done 2026-09-15**
  (`5dad4df`, merged at `HEAD`): all 33 `QuantizedIVF` occurrences retired from
  `docs/thesis/`, the novelty claims rewritten as an independent
  reimplementation, the SilverTorch paper cited for the first time
  (arXiv 2511.14881, verified against the arXiv record), and
  [`docs/paper/reproduction-deviations.md`](../paper/reproduction-deviations.md)
  written — 42 rows across paper-vs-us, Meta's code vs Meta's paper, defects in
  our own code found by reproducing, and 3 unexplained residuals, each citing
  the plan section that measured it. P G1 (Mac, 1 d): QuantizedIVF
  is SilverTorch Algorithm 1; the deviations table incl. O's findings
  (official bloom hash ≠ ours, official eager-only, the
  `per_embedding_scale` overflow).
- [ ] **F2 — write the "official vs reimplementation" section of the paper.** O §10 WP-9 (1 d),
  from B3 + D1 numbers.
- [x] **F3 — write the provenance and hardware/software disclosure.** P G9.
  **Done 2026-09-15** (`5dad4df`, merged at `HEAD`):
  [`docs/paper/provenance-and-disclosure.md`](../paper/provenance-and-disclosure.md)
  — hardware/software with Meta's pin, which recorded fields may be cited, the
  clock-estimator section, the "what was and was not compared" disclosure (no
  equivalence claim with the pre-v2 harness), and the not-yet-validated list
  with its roadmap step per line.
- [ ] **F4 — package the artifacts: tagged release, Zenodo DOI, HF data, one-command reproduction.** P §B.7: tagged `torchretrieve` release, Zenodo
  DOI (incl. the pinned official sdist), HF datasets + oracles + results,
  one-command `reproduce-paper`, anonymised mirror for review.
- [ ] **F5 — write the paper.** Follow the P §C.3 research questions and
  §C.4 section plan, with every table produced by D4's `report.py`.

### Phase G — kernel follow-ups and extended experiments (after F)

- [ ] **G-a — write the transposed bloom index in Triton and retune the scorer.** O §8,
  WP-8 (2 d + A100). Gate: parity bit-exact, bloom kernel-only within
  1.3× of official. Then rerun B3.
- [ ] **G-b — run the extended experiments: scale ladder, pass-rate
  sweep, ablation depth, bit width, batch grid.** P gaps G10–G12, G15,
  G16 (synthetic scale ladder to
  240 M / 1 B, controlled pass-rate sweep, co-design ablation depth, V3
  bit width, extended batch grid). G13/G14 are in D2.
- [ ] **G-c — write a resource paper about the library itself.** P §A.1
  lists the track; after F5.
- [ ] **G-d — apply the deferred kernel optimizations.**
  `oporp_1bit_match_topk` hardware popcount; allocator hygiene in the LiNR
  kernels. No timeline.
- [ ] **G-e — re-scope the parked export and live-update plans.**
  [torch-export-refactor.md](torch-export-refactor.md),
  [live-update-api.md](live-update-api.md). Not started; both carry
  stale-anchor banners; re-scope before executing. Research menu:
  [future-work-and-research.md](future-work-and-research.md).

## 2. Dependencies at a glance

```
(A0 dropped 2026-09-06 — no Mac target)
A1 ─┬─> A4 (merge)
    └─> C1 ─> C2 ─> C3 ─┬─> C4 ─> C5 ─> D1 ─> D4 ─┐
L1 ─> L2 ───────────────┘    │      │   D2, D3 ───┼─> F2, F5
A2 ─> A3 ─> B1 ─> B2 ─┬─> B4 │      │             │
                      └─> B3 ┘      │             │
E0 first; E1–E4 ingest (any time) ─> E5 (after D1) ┘
F1, F3 (any time); F4 (after D1)
```

The A100 critical path is (golden re-derive) → C4 (on the L2 layout) →
B3 → D1 → D2/D3 → E5. CPU work on the box (L1, L2, C5, D4, F1, F3, the
E-phase loaders) fills the gaps: **L1 → L2 before the C4 rerun, C5 after
C4 and before D1**. E1–E4 ingestion runs whenever the box is otherwise
idle.

### 2.1 Where each step runs

The A100 is the bottleneck, so this is the partition to schedule
against. **Mac** means it needs neither a GPU nor a GPU-produced number,
so it can run on the box's CPUs while the GPU is busy. (A0 was dropped
2026-09-06: there is no Mac target, every step runs on the GPU box.)

| GPU box only | Mac, startable after A0 | Mac, waiting on a GPU number |
|---|---|---|
| A1, A2, A3 | ~~C1, C2, C3~~ (done); **L1, L2** (the library refactor, 3.5 d + two suite runs) | A4 — needs A1 |
| B2, B3 | B1 †, B5, D4 | B4 — needs B2; **C5 — needs C4 + L2** (2 d) |
| C4, D1, D2, D3 | E0, E1 download + parse ‡, E1–E4 loader code + fixture tests | F2 — needs B3 + D1 |
| E1–E4 encode / PCA / gSASRec / oracle builds, E5 | F1, F3, G-e re-scope, G-a kernel authoring | F4 — needs D1; F5 — needs all |
| G-a validation and retune, G-b | | |

† B1's *shape* — the `backend="official"` branch, `require_official`, the
T1–T7 skeletons — is Mac work. A3 supplies the constants the tests
assert, so write them parameterised over the two candidate bit orders and
let A3 pick one.

‡ YFCC-10M is 2.9 GB in total (**D** §3.4) and ships its own filtered
ground truth, so the download, the `.spmat` parse and the narrow attrs
tensor are Mac work on a 588 GB-free disk; only E1's gate ("our exact
oracle reproduces the shipped GT") needs the box. The other three
datasets are 146 GB (PubMed) to 840 GB (Semantic Scholar) and belong on
the box's disk, but their `eval_datasets/<name>.py` modules are ordinary
Python that can be written and unit-tested here against fixtures —
`eval_datasets/` is 3,875 lines with **no tests at all** today, and Phase
E adds four more loaders to it.

CPU-side critical path, startable at once:
**L1 → L2** (3.5 d, before the C4 rerun) → **C5** (2 d, after C4), with
D4, F1, F3 and the E-phase loaders as filler; each of L1 / L2 needs one
library-suite run on the GPU (minutes) and C5 needs one cell.

## 3. Superseded and parked

- **Refactor validation runbook** — library steps passed on the A100
  2026-09-02 (see *Done*); harness steps 4–7 are replaced by A1. The two
  implemented plans ([kernels-layers-design.md](kernels-layers-design.md),
  [evaluation-refactor.md](archive/evaluation-refactor.md)) archive with C3.
- **Thesis results expansion (old item 4)** — its 4a schema items are H
  §2/§3, its 4b reruns are D1; the section is gone from this file.
- **Goodreads oracle rerun (old item 2)** — D1.
- **Deferred kernel optimizations (old item 3)** — G-d.
- **Parked feature plans (G-e)** — re-scope against **L**'s layout: the
  composites' single forward signature is what
  [torch-export-refactor.md](torch-export-refactor.md) wanted, and
  [live-update-api.md](live-update-api.md)'s `LiveIndexMixin` lands on
  `retrieve.modules`.
- **Review items folded into L / V** — library review A5 (`_host.py`),
  A8, A9, D3, T3, B.4 → L1; harness review §1.11 (layout contract), §1.12
  (training / metrics), §1.4 (`resume_key`) → C5.

---

## Done

### CuTe DSL SilverTorch backend — implemented 2026-09-02, deleted at B4 (2026-09-06)

`SilverTorch(backend="cute")`, a one-to-one port of the CUDA C++ backend into
NVIDIA's CuTe DSL, bit-exact against it and against Triton; its finding
(kernel-for-kernel parity with C++, the cost being DSL launch overhead,
gone under CUDA-graph replay) is in the archived plan
[archive/cute-dsl-scorer.md](archive/cute-dsl-scorer.md) §5 / §5.1 with
raw outputs in [archive/cute-dsl-scorer-artifacts/](archive/cute-dsl-scorer-artifacts/README.md).
Citable as `retrieve@cuda-cute-backends-final`.

### CUDA SilverTorch backend — implemented 2026-07-06, validated + tuned on A100 2026-09-02, deleted at B4 (2026-09-06)

`SilverTorch(backend="cuda")`, the paper's two-kernel design (transposed
cluster-major bloom index → 1-bit masks → masked `__dp4a` scoring) in
CUDA C++, bit-identical to Triton on scores; its A100 record (47/47 parity,
the memory-level-parallelism fix that took bloom scoring from 147.8 to
58.6 µs kernel-only against Triton's 124.6 µs at B=16, P=58k, D=128) is
[archive/cuda-silvertorch-handoff.md §13](archive/cuda-silvertorch-handoff.md#13-validation-record--2026-09-02-a100-sxm4-80gb-cuda-124-nvcc--torch-2100cu128-triton-360),
phase 2 in [archive/cuda-silvertorch-phase2.md](archive/cuda-silvertorch-phase2.md).
Deleted after Meta's official ops passed the parity gate (B2, O §14);
the transposed-index idea returns to Triton as O §8 TF-1. Citable as
`retrieve@cuda-cute-backends-final`.

### Refactor track — implemented 2026-07-06, library gates passed 2026-09-02, harness gates pending

Two structure-preserving cleanup plans written from a full audit of both
packages. They change no measured numbers and no op schemas. All code
phases plus the `filter=` → `filter_mode=` rename and the K9/E8.2 doc
sweeps are committed on `refactor/kernels-eval`.

- [kernels-layers-design.md](kernels-layers-design.md) — library-side
  dedup and API consistency: fix the broken `tune-kernels` subcommand
  (K1); shared host-wrapper prep/epilogue so validation reaches the
  production path, and the last four `@custom_op` kernels migrated to
  `@triton_op` (K2); shared `@triton.jit` predicate helpers in
  `kernels/common.py` (K3); `masked_topk` plus the `_PackedBitsKNN` base
  that absorbs `SimHashKNN`'s near-copy of `OneBitKNN` (K4); public
  bloom-hash core in `layers/filters/bloom_hash.py` (K5);
  `FilterModule.register_index` kw-only alignment and the minimal
  `RetrievalModule` ABC (K6); the `KernelTuneSpec` registry (K7); new
  tests (K8); doc sweep (K9).
- [evaluation-refactor.md](archive/evaluation-refactor.md) — harness leanness:
  delete the dead `torch_knn` algo, the unreachable CPU-timing path and
  three unused dependencies; quarantine the upload script behind
  `--repo-id`; rename `datasets` → `eval_datasets` (E1); replace the
  deep kwarg threading in `sweep.py` with `SweepContext` /
  `FilterAssets` (E2); typed `RetrievalAlgo` protocol and a declarative
  eligibility table instead of ValueError-as-control-flow (E3);
  `AlgoBase` for the duplicated compile tail (E4); split `bench_tools.py`
  into `measure.py` / `encode.py` / `passes.py` with `PerfStats` /
  `QualityStats` dataclasses (E5); **oracle cache keyed by content
  fingerprint** — the fix for the blocker in Open work item 2 (E6);
  config/loader hygiene (E7); unit tests and the `evaluation.md` rewrite
  (E8).

Both stay in this directory only until validation signs off; archive
them after.

### Stage 1 — Autotune separation (2026-05-22)

Every kernel exposes a `<Name>Config` dataclass + a single curated
`DEFAULT_CONFIG` next to its `@triton.jit` body. Overrides go through a
private `_<name>_impl(..., *, config=None)` so the public op keeps a
fixed schema. Offline tuning via
[tune.py](../../retrieve/src/retrieve/ops/tune.py). `bloom_match` keeps a
hard-coded tile — its per-call width is dictated by `N`. Convention:
[../system/kernels.md → Autotune separation](../system/kernels.md#autotune-separation).

### Stage 2 — `torch.compile` fixes (2026-05-22)

Replaced `@torch._dynamo.disable` on every kernel host wrapper with
proper op registration, so each algo's
`compile(dynamic=True, mode="reduce-overhead")` captures one
cudagraph_trees graph across filter + index + cascade. The compact
kernels were refactored to return full-width `[B, N]` `(ids, counts)`,
removing the `counts.max().item()` host sync.

### Stage 2b — `custom_op` → `triton_op` (2026-05-23, completed by K2)

`@custom_op` is opaque: inductor and `torch.export` cannot see the
underlying `@triton.jit` kernel. Stage 2b flipped the silvertorch and
bit-KNN wrappers to `@torch.library.triton_op` with a textually-inline
`wrap_triton(...)` launch. The filter/compact kernels initially stayed
behind because their shape-branching prep didn't trace cleanly; K2 moved
that branching into the eager `_impl`s and finished the migration.

**All ten Triton kernel ops are `@triton_op`.** The two hand-written
SilverTorch backends added six `@torch.library.custom_op`s on top while
they lived (a C++ extension has no `@triton.jit` body for inductor to
see); B4 removed them, so the totals are back to 10 registered ops across
7 kernel files, all `@triton_op`.

The host-side `if actual_k < k: pad` tail was eliminated rather than
moved caller-side: `oporp_1bit_match_topk_indirect` widens its score
buffer so `topk(k)` always has ≥ k lanes, and the silvertorch wrappers
rely on index-build asserts. The pad branch was dead in production.

---

## Archived

[archive/](archive/) holds plans whose work fully landed and which are no
longer instructions for anyone. Currently: the SilverTorch reverse-clause
wrapper fix (shipped in `cc85d8f`; the remaining sweep rerun is tracked
as Open work item 2).

Also deleted along the way, subsumed by the system docs: the prototype
custom_op migration doc, the per-item research doc, the deferred
mask-compact-kernel doc, the Stage 1 autotune-separation plan, the Stage
2 `02-triton-op-migration.md` plan, and the `plans-silvertorch-backup/`
tree (which held the `ShardedSilverTorch` sketches and a shelved
native-CUDA experiment — that experiment later shipped as
`backend="cuda"` and was deleted again at B4; its plans are in
`archive/`).
