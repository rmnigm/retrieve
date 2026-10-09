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
| [`run.sh`](run.sh) | one run: fetch (idempotent), trees, `bench report` per tree, the three scripts below, `bench upload --verify` |
| [`load.py`](load.py) | `records.latest` per tree, plus the key/label helpers |
| [`checks.py`](checks.py) | `checks.md` + `checks.csv`: status / partial / unstable counts, exact-arm recall ≥ 0.99, eager vs graph `ids_sha256`, bs 16 < 16 × bs 1, graph / eager latency, monotone curves (V1 flat in p, postfilter recall in p, SilverTorch recall in `n_probe`), Triton vs official recall, the same cell across code_versions and suites, co-design partial = full recall, and cells planned in `suites.yaml` but absent. Documented causes are tagged `known` |
| [`figures.py`](figures.py) | the figures the report lacks or draws unreadably. `f2x`: every synth arm incl. postfilter and V1, real sweeps as per-query pass-rate buckets. `f3x`: deep Pareto, one panel per sweep × bs. `g1`: V1 graph / eager vs pass rate. `synth-arms.csv`: the per-arm medians behind T1's ratios |
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
