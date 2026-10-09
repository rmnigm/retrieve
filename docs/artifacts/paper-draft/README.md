# paper-draft: first-draft paper materials (NOT CITABLE)

The page behind the private Artifact "LiNR + SilverTorch Repro Materials"
(<https://claude.ai/artifact/7HLZdKki9NiL9pkNrHziyS>): the story, the C1-C7
verdicts at a glance and one chart per main take. Current state only: superseded results and fixed defects stay out of the page. Exploration numbers; the final
pass reruns every cell before anything is cited
([decisions](../../decisions.md#campaign-v2-user-2026-10-08)).

- `extract.py HUB`: reads the legs fetched from `pinkmeme/eval-results` into
  `HUB` (default `/scratch/paper-draft/hub`, records only, no sidecars) and
  writes `data.json`.
- `build.py OUT`: inlines `data.json` into `page.html` and writes the page.
- `data.json`: the chart data as published.

## Sources

| chart | Hub source |
|---|---|
| uniform vs correlated recall | `campaign-v2.5/arxiv-synth-synth`, `campaign-v2.5/arxiv-corr-synth-synth` (pod d) |
| local vs global pass rate | `artifacts/exhibits/20261009-1057-gls/gls-sweeps.csv` |
| recall at 24 probes vs N | uniform synth: `campaign-v2/goodreads-synth-synth`, `campaign-v2.5/{arxiv,yfcc10m,laion30m}-synth-synth`; real filters: V-GR-DEEP (goodreads 0.94), arXiv IVF-TUNE (0.86-0.88), d1 PubMed (0.66), `campaign-v2.5/yfcc10m-deep`, `campaign-v2.5/laion30m-filter` |
| IVF speed-up at 0.95 | exhibits notes: goodreads `c0_genre` (IVF `campaign-v2.9/goodreads-deep` n_probe 32 bs 16 graph 0.196 ms vs v2 `filter` V1 2.147 ms, box `e75980`); 3 M uniform (`campaign-v2.5/arxiv-synth-synth`, pod d, pre-CLAUSE-SKIP, probe counts interpolated); LAION (`campaign-v2.9/laion30m-x`: exact V2 and SilverTorch n_probe 4096 on `tags4`, bs 16 graph, one leg, pod d, not interleaved) |
| int8 mechanism | `artifacts/yfcc-int8` |
| co-design | below 30 M: `artifacts/exhibits/20261009-2125-c5/c5-below30m.csv` (paired over interleaved rounds; ours v2.9 graph, Meta v2.8 -O3 eager); ours 10 M PubMed: `campaign-v2.9/pubmed-codesign-ours` (graph p50 per interleaved pair); Meta at 30 M `campaign-v2.8/laion30m-codesign-laion30m` (final adapter, -O3; eager p50 per pair, equal to exhibits' paired ratios); ours at 30 M `campaign-v2.9/laion30m-codesign-laion30m` (graph p50 per interleaved pair, pod d) |
| V2 / V1 | `campaign-v2.7/arxiv-synth-synth` (pod d `38f5e1`), `campaign-v2.7/yfcc10m-synth-synth` (pod d), `campaign-v2.7/laion30m-synth-laion30m-synth` (pod d), `campaign-v2.5/laion30m-filter`; ratio of graph p50 inside each interleave group; LiNR point = Table 3 PyTorch V2 / V1 at B 1 |
| official vs ours | d128: `campaign-v2.8/goodreads-h2h`, `campaign-v2.8/arxiv-h2h` (final adapter, Meta -O3 fp16 vs Triton, eager p50 and `kernels_us`, k 100, one seed); d768: `campaign-v2.9/pubmed-filter` (ours at v2.9; bloom `c0_mesh`, 4096 lists, interleaved, eager p50; final adapter; Meta's -O3 build, `env.official_build` sha `92c422b2…`), |
| V3 | `campaign-v2.2/goodreads-synth-v3bits`, `campaign-v2.5/pubmed-v3bits`; V2 times from the V3-BITS-PUBMED validation row |
| bloom FPR | `campaign-v2/{arxiv,goodreads,pubmed}-bloomwidth` (Triton, mean over seeds) |
| defect ledger | `artifacts/exhibits/20261009-1101-idea6/ledger.md` |
