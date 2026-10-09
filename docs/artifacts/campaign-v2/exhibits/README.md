# EXHIBITS: charts, comparisons and bug checks after every leg

CPU-only analysis over every published campaign leg (roadmap EXHIBITS). One run fetches each
`campaign-v2/*` and `campaign-v2.1/*` subtree listed in [hub-index](../../hub-index.md), plus `d1/arxiv`
(the manifest's reuse source), `d1/arxiv-deep`, `d1/yfcc10m` and `d1/pubmed` (the only 10 M wall times).
It builds **one results tree per code_version** (`v2` = `408b1188`, `v21` = `f01255f1`, `d1` = `72e5a90`).
A resume key carries the code_version, so a single tree would hold both versions of a re-run cell and
`bench report` would draw them into one curve. Every output is NOT CITABLE. The runs are on the Hub as
`artifacts/exhibits/<YYYYMMDD-HHMM>` ([hub-index](../../hub-index.md)).

| file | what |
|---|---|
| [`run.sh`](run.sh) | one run: fetch (idempotent; `EXTRA_LEGS` adds record trees published under `artifacts/`), trees by each suite's record code_version (a pass that re-records cells already in a tree, such as H-KSUM's profile-only h2h, keeps its own `<tree>-<leg>` tree, read by `t3x.py` only; `artifacts/` legs are placed last), `bench report` per tree, the three scripts below, `bench upload --verify` |
| [`load.py`](load.py) | `records.latest` per tree, plus the key/label helpers |
| [`checks.py`](checks.py) | `checks.md` + `checks.csv`: status / partial / unstable counts, exact-arm recall ≥ 0.99, eager vs graph `ids_sha256`, bs 16 < 16 × bs 1, graph / eager latency, monotone curves (V1 flat in p, postfilter recall in p, SilverTorch recall in `n_probe`), Triton vs official recall, the same cell across code_versions and suites, co-design partial = full recall, and cells planned in `suites.yaml` but absent. Documented causes are tagged `known` |
| [`figures.py`](figures.py) | the figures the report lacks or draws unreadably. `f1x`: V1 / V2 Triton p50 vs pass rate per scale and code_version, with the paired V2/V1 ratio over interleaved rounds and its CI (each row says whether its V2 is pre- or post-Fix A). `f2x`: every synth arm incl. postfilter and V1, real sweeps as per-query pass-rate buckets. `f3x`: deep Pareto, one panel per sweep × bs. `g1`: V1 graph / eager vs pass rate. `synth-arms.csv`: the per-arm medians behind T1's ratios |
| [`v2_old_new.py`](v2_old_new.py) | V2 before / after Fix A by pass rate: the old-side V1 / V2 table of every fetched leg, plus the V2-FIX-A artifact's ABAB ratios and `programs` sweep (`v2_old_new.md`) |
| [`t3x.py`](t3x.py) | T3 with device time next to end to end (`t3x.md` / `.csv`). Per `h2h` cell and arm: the p50, the paired ratio over Triton eager, device ms, launches, the host share 1 − device / p50 (flagged above 0.5), canonical-ids equality, Jaccard and max \|Δs\|, and the scorer scope like for like (ours: `_codesigned_probe_score_kernel`; Meta: `process_cluster*`, its payload kernels and, on bloom, `bloom_search`; blank when the recorded list misses one). Device is `kernels_us` (every kernel, H-KSUM) wherever any given tree holds a record of the same cell, arm and seed with it (`src` all); otherwise it is the top-8 sum (`src` top8, a lower bound), shown next to it as `top-8 ms` | Timing comes from `ok` records; `partial` ones (a profile-only pass) only supply `kernels_us`, and an `ok` record timed with `--profile` supplies its own.
| [`c5.py`](c5.py) | C5: the co-design (`bloom_path` partial) against the full mask, per backend and code_version (`c5.csv`). The paired full / partial ratio over the interleaved rounds per (dataset, sweep, n_probe, bs, mode), with recall equality (S-13), the clocks behind any `unstable` flag, and from campaign-v2.6 on the `query_prep_ms` each path spends outside the timed forward (so served cost stays comparable with older records) |
| [`v3bits.py`](v3bits.py) | C2: LiNR V3 by `k_bits` × candidate pool × pass rate on goodreads-synth (`v3bits.csv`): recall@100 and graph p50 with clocks / spread, next to the default V3 (`k_bits` = D) of another code_version, labelled |
| [`router.py`](router.py) | idea #1, the pass-rate router: a counterfactual per-query policy (V2 below a threshold on the query's own pass rate, IVF above), fitted on one dataset and checked on the others (`router.csv`, `router-querysets.csv`, `router-bs{1,16}.png`); latency is per-cell batch medians, so routing is assumed free |
| [`gls.py`](gls.py) | idea #2: per query, the local pass rate l_q (share of its 100 unfiltered nearest neighbours that pass its filter, read as \|unfiltered top-100 ∩ filtered top-100\| from the synth `p1` and the sweep's oracle blobs, read-only) and GLS = l_q / p_q; their rank correlation with IVF recall against the pass rate's, and the router of `router.py` with l_q as its feature |
| [`qps_bands.py`](qps_bands.py) | idea #6a: QPS at recall_oracle@100 = 0.95 per selectivity band (p < 0.01, 0.01-0.1, 0.1-0.5, > 0.5) per dataset and arm (`qps-bands.csv`). Per sweep: `deep` curves along n_probe / pool, plus the `filter` suite's exact arms at the same code_version. Per query: `codesign` sidecars, band recall over the queries whose own pass rate is in the band, at the cell's latency. A curve already above 0.95 at its cheapest point reports that QPS as a lower bound |
| [`router_pareto.py`](router_pareto.py) | the router decision (user, 2026-10-10): is any router point on the recall-latency Pareto front of IVF (SilverTorch along n_probe, interpolated in recall) and exact search (V1, V2) on PubMed 10 M, per sweep and bs, served mode (graph where the arm has one, else eager) and eager vs eager, same code_version and tree (`router-pareto.md` / `.csv`) |
| [`scaling.py`](scaling.py) | the fixed-p scaling view across N on the uniform synth legs (`scaling.csv` / `.md`): SilverTorch recall at n_probe 24 and at the n_probe nearest 6.25 % scanned, V2 / V1 and SilverTorch / V1 graph ratios inside one leg (absolute latencies never compared across boxes), box and flags (pre-CLAUSE-SKIP, clock-unknown) |
| [`gpuh.py`](gpuh.py) | the V-YFCC GPU-h projection (`gpuh-v-yfcc.csv`); model and assumptions below |

```bash
UV_PROJECT_ENVIRONMENT=<own venv> bash docs/artifacts/campaign-v2/exhibits/run.sh   # --no-upload to stay local
```

## gpuh

Cell cost = overhead + Σ over its timed variants (`ks` × `batch_sizes` × modes; official eager only) of
`measure.latency`'s windows: 50 warm-up calls + 3 × clamp(2 s / t, 1000, 5000) calls of t. Above
t = 2 ms, a variant therefore costs ≈ 3,050 · t. Overheads are `elapsed_s` minus the windows on
`d1/yfcc10m` (V1 119 s, V2 388 s, V3 363 s, SilverTorch 28 s, official 82 s). Seeds 1-2 of the seed-free
arms copy their quality (30 s). Latencies by suite:

- **synth**: V1 = arXiv-synth 3 M at v2.1 (pod c: 1.28 / 3.26 ms at bs 1 / 16) × N ratio (low) or
  N × d ratio (high). V2 / V1 at bs 16 is pod c's 0.69 (p ≤ 0.01) to 1.83 (p 1); every other arm is
  the pilot's arm / V1 ratio at the same p and bs. SilverTorch adds the scanned share `n_probe / n_lists`
  of a V1 scan. Official is Triton × H2H-FINAL's bloom official / Triton ratio (8.0 at bs 1, 6.9 at bs 16).
- **filter and deep**: `d1/yfcc10m`'s per-arm latencies × the 3 M d1 → v2.1 speedup (V1 bs 16
  7.59 → 3.26 ms; low case only).
- **high case**: 1.5 × overhead. Oracle: 0.3 h, not measured.

Not modelled: process start and compile, interleave bookkeeping, unstable reruns.
