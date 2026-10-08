---
title: roadmap
created: 2026-09-26
updated: 2026-10-08
type: summary
tags: [roadmap]
sources: [evaluation/config/suites.yaml, evaluation/bench/, infra/runpod/, docs/validation.md, docs/artifacts/hub-index.md]
---

# Roadmap: the open queue

The single ordered work queue: what is left to run or implement, in
dependency order. What already holds is in [validation](validation.md),
the standing decisions in [decisions](decisions.md), work that is not
queued in the [backlog](backlog.md). An agent starting a session reads
[AGENTS.md](../AGENTS.md), then this page, then only the wiki pages its
step names. Who executes a step is
[agent orchestration](contracts/agent-orchestration.md).

**Rules.** No dates: every step is done, in the order given, and nothing
is optional. Do not start a step whose dependencies are open. When a step
finishes, its gate's row in [validation](validation.md) and the affected
`docs/system` page are updated, and the orchestrator **removes** the step
from this page. Step names (D1, H2, ...) are stable identifiers used in
records, commits and code comments; they are not renumbered.

**Estimates** are A100-SXM4-80GB GPU-hours (the reference GPU for every
citable number), from the per-cell wall times the record inventory measured
([artifact](artifacts/campaign-v2/README.md#b-measured-wall-time-per-cell-old-grid-3-bs--3-k--2-modes)); a *stream* is
a slice of a step that can run on its own GPU at the same time as the
others (see [Multi-GPU execution](#multi-gpu-execution)).

## Needs the user

- **`n_probe` > 1024 at k 1000** (optional): the probe scorers' id
  epilogue tile caps `next_pow2(k) · next_pow2(n_probe)` at 2^20
  ([kernels](system/kernels.md)). Tiling it over k would lift the limit; a
  library change (campaign-v2.2). Not needed by the grid as tuned.
- **Contact the original authors** (re-plan decision 7): the LinkedIn LiNR
  team and Meta's SilverTorch team — filter-set details, the V1/V2 setup,
  the SilverTorch paper's FPR inconsistency (0.067 % vs 0.00173 %) — and
  file the official-code defects (OF-3, OF-4, OF-6 in
  [deviations](paper/reproduction-deviations.md)) upstream as issues. The
  ECIR call asks what contact happened; replies take weeks.
- **Clock-normalised or as-measured ratios (T3, F4b).** Clocks cannot be
  locked and follow the load: in the freeze's timed smoke, official arms
  sampled ~1140-1170 MHz while Triton graph sampled 1410 in the same
  interleaved rounds (8 of 9 records `unstable`). Interleaving pairs drift
  but cannot equalise a load-dependent clock. Decide before those exhibits
  get verdicts ([validation](validation.md)).
- **Citability of a narrowed run.** A run with a narrowed mode set is
  recorded `status: partial` and reported NOT CITABLE. The user decides
  once `bench report` runs on the campaign's output (D1-G).
- **A4**: merging `staging` into `main` is on hold until the user decides.
- **HF: squash `pinkmeme/eval-goodreads-work-id`** to reclaim 2.3 GB
  (`HfApi().super_squash_history(repo_id=..., repo_type="dataset")`).
- **`.git` history size.** The JSON/JSONL results left `HEAD`, but their
  bytes stay in history; shrinking `.git` needs a history rewrite (`git
  filter-repo`), which forces every clone and worktree to re-sync
  ([decisions](decisions.md#harness)).

## Running GPU work on a pod

GPU work runs on RunPod pods launched with
[`infra/runpod/pod.sh`](../infra/runpod/pod.sh) (`pod.sh up --gpu a100
--gpus N`); disks and environment in [storage](system/storage.md).

### Restore on a fresh pod

1. Checkout `staging` (the image clones it into `/workspace/retrieve`);
   the venv is `/venvs/retrieve`. A worker that needs a different
   environment gets its own venv (`UV_PROJECT_ENVIRONMENT=/venvs/<name> uv
   sync --all-packages --all-groups --extra official`): a worker that
   re-syncs the shared venv to its worktree breaks every running job.
2. Data under `/data` (`RETRIEVE_DATA_ROOT`), with `evaluation/data ->
   /data` (a gitignored symlink) in every checkout and worktree. `eval-data
   fetch` only what the step reads: `arxiv-papers` (the license-fixed
   version: `item_attrs_narrow.pt` min = -1, `eval_split.parquet`
   `query_attrs_narrow` min = 0), `goodreads-work-id` with its checkpoints
   ([checkpoints](system/checkpoints.md#what-the-harness-reads)). PubMed is not on the Hub: rebuild the 10 M slice with
   `eval-data pubmed` ([datasets](system/datasets.md#pubmed); ~1 h
   download-bound `convert`, 17 GB). YFCC-10M is public: `eval-data yfcc
   all` ([datasets](system/datasets.md#yfcc10m), ~12.5 GB with the raw).
   The synth attrs are built on the pod (*eval-data synth-filter*, CPU,
   deterministic from its seed).
3. Results: the remaining runs do not need the old records locally (the
   narrow scoping below never resumes against them), but the final uploads
   and the report do: `bench fetch --path-in-repo d1/<subtree> --results
   <dir>` per [hub-index](artifacts/hub-index.md) row, merged into
   `evaluation/results/<suite>/`.
4. Every GPU job: its own `TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/<job>`
   (`bench run` keys a default by `code_version` when none is given,
   [evaluation](system/evaluation.md#inductor-cache)), `HF_HOME=/scratch/hf`.
   Run from `evaluation/` as `/venvs/retrieve/bin/python -m bench.cli …`,
   **not** `uv run`, which masks bench's exit code. `bench campaign`'s
   per-group timeout defaults to 48 h. A long
   job runs from one sequential driver script per GPU (one process group,
   no chained waiters, and no log line containing the text a watcher
   greps for). Every job's log carries the pod's `nvidia-smi -q -d CLOCK`
   and its `sm_mhz` histogram, uploaded with the results.

### Multi-GPU execution

On a pod with several GPUs, each job owns one GPU
(`CUDA_VISIBLE_DEVICES=<i>`), its own inductor cache dir and its own
`--out` tree or `--output` file; `aggregate` reads every
`<suite>/*.jsonl`, so per-stream files in one tree merge without a step.
Pin each job's CPU threads to its GPU's NUMA-local cores (`taskset -c`,
the affinity column of `nvidia-smi topo -m`): batch-1 latency is
launch-bound, so a neighbour's host load moves it. Record in the step's
validation row that neighbour GPUs were busy. Split streams by `--algo`,
then by `--filter-kind` / `--sweep` / `--seed`; keep `silvertorch`'s
`triton` and `official` backends in **one** stream, in that order: they
share a `_parity/<dataset, dim, algo>` spill and the second backend's
`jaccard_vs_first` compares against the first's. `bench campaign` has no
`--algo`, so a split stream is a `bench run`. M1 decides whether
multi-GPU numbers are comparable at all.

### Campaign code_version and reruns

Every new campaign cell runs at the `campaign-v2.1` tag's code_version
(the tree hash of `retrieve/src/retrieve`), recorded in
*evaluation/campaign.yaml*: `f01255f106214ec0f540352d0e2a5cd90b84b6a1`
(V2-FIX-A and the V-GRAPH-IDS quantize fix over `campaign-v2`'s
`408b1188`, [decisions](decisions.md#campaign-v2-user-2026-10-08)); its
legs upload under `campaign-v2.1/<dataset>-<suite>`. Records at
`408b1188` stay where they are and count only through manifest entries
(MANIFEST-V21).
New legs run into fresh result trees; old records are never re-stamped and
enter the paper only through the reuse rule and the manifest. A
correctness bug found mid-campaign stops every run; it is fixed and
re-tagged, only the arms its own gates prove changed are rerun, and the
manifest's `log` says so. After each step: `bench upload --verify`, a
[hub-index](artifacts/hub-index.md) row, the manifest and the validation
row updated, `bench report` on the cumulative tree, and a look at the
exhibit the step feeds; if that exhibit already settles its claim, the
claim's remaining runs are skipped and the reason logged.

### Stop rules

- **Pilot gate** (passed 2026-10-08 on goodreads-synth, [validation](validation.md)):
  proceed to the arXiv and YFCC synth legs only if achieved pass rates are
  within 1 % relative of target where N·p ≥ 10⁴ (below that the binomial
  spread exceeds 1 %; goodreads p 0.001 is −1.01 % at N·p 797), V1's
  `recall_oracle` is ≥ 0.99 at every p, and the curves are monotone where
  they must be (V1 latency roughly flat in p, postfilter recall falling
  with p). Otherwise stop and report.
- **Noise gate.** Every timed comparison in an exhibit has a 95 % CI on its
  ratio that excludes 1.0, or it is reported as "no difference". No more
  than 5 repeats to force significance.
- **Densify only on demand**: a parameter point is added only where an
  exhibit's curve has a kink or crossover between two measured points and
  the claim depends on where it is.
- **Budget gate.** If a step's GPU-h exceed 1.25× its estimate, report it
  and update the remaining estimates from the measured rate; the queue
  continues (user, 2026-10-08: the full grid runs).
- **Surprise gate.** A result that contradicts an earlier one (e.g. Triton
  slower than official end to end on the frozen code) stops the queue:
  rerun that one cell interleaved and report before anything else runs.

## Phase P: CPU work before the campaign

The claims and the claim-to-evidence matrix are [claims](paper/claims.md);
the record inventory is [artifacts/campaign-v2](artifacts/campaign-v2/README.md).

## Phase V: the campaign (pods the user creates)

GPU 0 takes all timed work from one sequential driver, cores pinned
NUMA-local; GPU 1 takes quality-only work (`--skip-perf`: oracle builds via
*bench oracle*, bloomwidth quality, the PubMed embedding check, the `n95`
probe). Timed work moves to a second GPU only if M1 passes. Exhibits:
T1 claims, T2 real-filter headline, T3 official vs Triton, F1 latency vs
pass rate, F2 recall vs pass rate, F3 Pareto, F4a bloom FPR/memory, F4b
co-design.

- [ ] **M1: multi-GPU interference control**, before any timed step on a
  multi-GPU pod: one arxiv `filter` cell at bs 1 and 16
  (`silvertorch`/triton, `linr_v2`) alone, then with every other GPU
  loaded, interleaved, cores pinned. *Gate*: loaded medians within each
  arm's own repeat noise; otherwise timed steps run one GPU at a time.
  **0.5 GPU-h** on a ≥ 2-GPU pod.
- [ ] **H-PROFILE + H2H-FINAL at `campaign-v2.1`.** `--profile` stores an
  empty Triton `kernels` list (H2H-FINAL: 40/40 bloom and 13/40 `none`
  eager entries; official always populated), so T3's Triton kernel-only
  column is empty for bloom (C7). Harness fix in `measure.profile_once`
  (code_version unaffected), then the whole `h2h` leg (goodreads + arXiv,
  `--interleave --profile`) once at `campaign-v2.1`: the 408b1188 run is
  stale (eager SilverTorch +30 µs from the quantize fix) and stays the
  record of the old code. The only T3 source. **≈ 1.5 GPU-h.**
- [ ] **ST-DLOOP: SilverTorch Triton probe scorers at wide embeddings**
  (user, 2026-10-09: match Meta's CUDA kernels). D3 PubMed at v2.1 tripped
  the surprise gate: Triton 2.80-2.98 ms vs official 1.99 ms at d768 (bs 16),
  where every d ≤ 256 cell has Triton faster. The scorers do one `tl.dot` over
  the whole padded width (768 → 1024) with no loop over D. Fix: loop over D
  in chunks, D_PAD ≤ 256 code unchanged; plus an architecture comparison with
  the official kernels, other improvements listed, not applied. *Gates*:
  `torch.equal` ids and scores at D 128 / 192 / 768 against `campaign-v2.1`,
  library suite on a pod GPU, interleaved before/after. Then tag
  `campaign-v2.2`; stale: PubMed SilverTorch Triton perf (D3 PubMed timed).
  V-PUBMED waits for it. Pod b. **≈ 1-2 GPU-h** plus the code.
- [ ] **V-RERUN-V21: goodreads reruns at `campaign-v2.1`.** The perf of
  V2 and V3 Triton on goodreads-synth `synth` (V-PILOT) and goodreads
  `filter` (V-GR-FILTER), quality reused (both fixes bit-exact); and the
  SilverTorch arms of both legs whole at IVF-TUNE's goodreads `n_lists`
  4096 (n95 64). Uploads `campaign-v2.1/goodreads-synth-synth`,
  `campaign-v2.1/goodreads-filter`. Needs IVF-TUNE merged. **≈ 2-3 GPU-h.**
- [ ] **MANIFEST-V21: the manifest for two code_versions.** *campaign.yaml*'s
  `default` is `campaign-v2.1`; add entries that keep the `408b1188`
  records the tag did not touch (quality everywhere: both fixes are
  bit-exact on eager outputs; perf of V1 and postfilter triton, and of
  SilverTorch graph mode), and drop the perf half of the 12 arXiv
  `72e5a90` reuse entries for V2 and V3 (their kernels changed). CPU only;
  `bench report --manifest` over the fetched trees reports 0 wrongly
  missing cells. Before D1-G.
- [ ] **V-CODESIGN: `codesign` on goodreads** (arXiv done at v2.1), at `campaign-v2.1`
  (the 408b1188 run, 108/108, is stale: the quantize fix moves every
  SilverTorch eager time; Hub `artifacts/v-codesign-408b`), interleaved
  partial/full, `n_probe` {8, 32, 128}, 3 sweeps, 3 seeds; replaces D1-B2
  and D1-D. F4b, C5. **≈ 2 GPU-h**, GPU 0.
- [ ] **D3: `bloomwidth`**: PubMed `bloomwidth-timed` at `campaign-v2.1`
  remains; goodreads and arXiv done (timed at v2.1), PubMed `bloomwidth`
  (quality) done
  ([validation](validation.md), Hub `campaign-v2/arxiv-bloomwidth[-timed]`):
  both blooms, `m_bits` 64-2048 × `k_hash` {3, 5}, quality-only plus one
  timed point per width at bs 16. F4a, C4. **≈ 3-5 GPU-h**, GPU 1 (the
  timed points on GPU 0).
- [ ] **V-AX-SYNTH: arXiv synth**, uniform 7 points, then the
  cluster-correlated variant (3 points) if the uniform sweep shows the IVF
  recall collapse at low p. F1/F2 3M panel. **≈ 15-45 GPU-h**
  (the pilot measured 8.7 GPU-h at 0.8 M), GPU 0. Its timed cells run at
  `campaign-v2.1`, at IVF-TUNE's arXiv `n_lists`.
- [ ] **V-GR-DEEP: goodreads `deep`**, trimmed (`n_lists` {1024, 4096},
  `n_probe` {8, 16, 32, 64, 128}, V3 pool fractions); replaces D1-C.
  F3; goodreads' `n95`, then the goodreads `filter` n95 cells and the
  189 new goodreads-synth cells of the SilverTorch n_probe sweep (in
  suites.yaml; resume adds only the new cells; driver
  [v-gr-deep](artifacts/campaign-v2/v-gr-deep/driver.sh)). **≈ 6 GPU-h**,
  GPU 0/1.
- [ ] **V-YFCC: YFCC synth (5 points) and a small `deep`** (`n_lists`
  {4096, 16384}). F1/F2 10M panel; YFCC's `n95`. Runs alongside
  V-AX-SYNTH on another GPU (three pods, no fourth: user, 2026-10-08); a
  collapse found on arXiv adds YFCC points afterwards. **≈ 18-120 GPU-h**
  (unmeasured at 10 M; re-estimated from V-AX-SYNTH's first leg), GPU 0/1.
- [ ] **V-SEEDS: arXiv and YFCC `filter`, the cells the manifest does not
  reuse.** arXiv: 117 cells (V1-V3 `c3_nversions` seeds 0-2; SilverTorch
  triton; official bloom; `postfilter` α {1, 8}; SilverTorch torch, plain
  and compiled, for C3); V1-V3 on `c0_maincat` / `all4` are reused
  through the manifest (36 cells, [campaign.yaml](../evaluation/campaign.yaml)).
  YFCC: all 18 cells, seed 0 included (a manifest entry cannot name a
  seed). Plus the `n95` column on the kept sweeps. T2 with CIs. Needs
  V-AX-SYNTH and V-YFCC. **≈ 8-12 GPU-h**, GPU 0.
- [ ] **V-PUBMED: PubMed `filter` under the new grid** (replaces D1-E):
  the embedding identity check on GPU 1 first (`eval-data pubmed
  encode_queries`, `bench check`, one V1 cell equal to `d1/pubmed`'s
  quality, or stop), the `n95` probe suite (quality-only), then the 3
  kept sweeps, 3 seeds, every arm, at `campaign-v2.2` (after ST-DLOOP) and IVF-TUNE's values. T2's 768-d row.
  **≈ 14 GPU-h**, GPU 0.
- [ ] **D1-G: gate reruns and the report.** Needs every step above.
  Eager-vs-graph id identity from the stored hashes, a byte-identical
  quality subset per leg (`--force` into a scratch tree),
  `median_ms(bs=16) < 16 × median_ms(bs=1)`, then `bench report
  --manifest` over everything. **≈ 4 GPU-h**, both GPUs.

## Phase F: the paper

- [ ] **F2: the official-vs-reimplementation section** rewritten from
  H2H-final's numbers, with CIs and paired tests
  ([paper](paper/official-vs-reimplementation.md)). Needs H-PROFILE and
  D1-G.
- [ ] **F4: package the artifacts**: the results public, the PubMed slice
  or its build recipe with checksums, a tagged `torchretrieve` release, a
  Zenodo DOI including the pinned official sdist, a one-command
  reproduction, an anonymised mirror (anonymous.4open.science) for the
  double-blind review. Needs D1-G and the user's accounts.
- [ ] **F5: write the paper** (ECIR 2027 reproducibility format, 12 pages
  LNCS): every table from `bench report`, the `docs/paper/` drafts
  rewritten against claims C1-C7 (they still cite retired plan labels).
  Needs everything above.

## Estimates

| step | A100 GPU-h | GPU | basis |
|---|---|---|---|
| M1 | 0.5 | both | |
| ST-DLOOP | ≈ 1-2 | 0 | gates + before/after on pod b |
| V-RERUN-V21 | ≈ 2-3 | 0 | goodreads-synth + goodreads filter (V2, V3, SilverTorch at n_lists 4096) |
| H-PROFILE + H2H-FINAL | ≈ 1.5 | 0 | 1.18 GPU-h measured at 408b1188 |
| V-CODESIGN | ≈ 1 | 0 | `d1/arxiv-codesign`: 60 cells in 0.4 h |
| D3 bloomwidth | ≈ 3-5 | 1 (+0) | quality-only cells |
| V-AX-SYNTH | ≈ 15-45 | 0 | Triton ~2,000 s per pass point; V1/V2 torch at 3 points ×3 seeds is most of it |
| V-GR-DEEP | ≈ 3-6 | 0/1 | ~210 cells at ~50 s (`d1/arxiv-deep`: 873 cells in 36 h at 3 M) |
| V-YFCC | ≈ 18-120 | 0/1 | V1 / V2 / V3 cells 0.3-1.2 h each at the v2 grid; V2/V3 cost vs p at 10 M unmeasured |
| V-SEEDS | ≈ 8-12 | 0 | YFCC seeds 1-2: V2 and V3 ~1.1 h a cell |
| V-PUBMED | ≈ 20-25 | 0 (+1 for the checks) | V1 ~650 s, V2 ~1,300 s, SilverTorch-Triton ~700 s a cell; 3 sweeps × 3 seeds |
| D1-G | ≈ 4 | both | |

Total ≈ 130-250 GPU-h, against the re-plan's 70-90: the 10 M exact arms
and the torch arms were underestimated there (see "Needs the user"). The
SilverTorch/Triton perf-only rerun the re-plan held in reserve costs
0 GPU-h: the CSR probe layout and transposed bloom predate `72e5a90`, and
the one later library change (per-width probe tiles) moves tiles only at
`D_PAD > 256` ([validation](validation.md#library-gates), *Probe-scorer
tile per width*).

## Dependencies

```
tag campaign-v2 ───────────────────────┬─> V-PILOT ─┬─> V-AX-SYNTH, V-YFCC ─> V-SEEDS ─┐
                                       │            └─> V-GR-DEEP ─────────────────────────┤
                                       ├─> V-CODESIGN, D3 ─────────────────────┤
                                       └─> V-PUBMED ───────────────────────────────────────┴─> D1-G ─> F2, F4, F5
M1: before any timed step on a multi-GPU pod
tag campaign-v2.1 and IVF-TUNE (both done): before every timed V-AX-SYNTH, V-YFCC, V-PUBMED, V-SEEDS cell
```
