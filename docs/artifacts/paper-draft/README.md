# paper-draft: first-draft paper materials (NOT CITABLE)

The page behind the private Artifact "LiNR + SilverTorch Repro Materials"
(<https://claude.ai/artifact/7HLZdKki9NiL9pkNrHziyS>): the story, the C1-C7
verdicts at a glance and one chart per main take. Exploration numbers; the final
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
| IVF speed-up at 0.95 | exhibits notes: goodreads `c0_genre` (V-GR-DEEP vs v2 `filter` V1, box `e75980`); 3 M uniform (`campaign-v2.5/arxiv-synth-synth`, pod d, pre-CLAUSE-SKIP, probe counts interpolated); LAION (`campaign-v2.5/laion30m-filter`) |
| int8 mechanism | `artifacts/yfcc-int8` |
| co-design | `artifacts/exhibits/20261009-1324-c5/c5.csv` (official 0.8-3 M eager, ours graph), `artifacts/codesign-laion30m-v26-official` (ratio of eager p50 per pair; exhibits' paired ratios agree) |
| V2 / V1 | `campaign-v2.7/arxiv-synth-synth` and `campaign-v2.5/arxiv-synth-synth` (both pod d `38f5e1`), `campaign-v2.5/yfcc10m-synth-synth`, `campaign-v2.5/laion30m-synth-synth`, `campaign-v2.5/laion30m-filter`; ratio of graph p50 inside each interleave group; LiNR point = Table 3 PyTorch V2 / V1 at B 1 |
| official vs ours | `artifacts/exhibits/20261009-0843-c7/t3x.csv` (v2.1); d768 0.70 → 2.40 from the ST-DLOOP validation row |
| V3 | `campaign-v2.2/goodreads-synth-v3bits`, `campaign-v2.5/pubmed-v3bits`; V2 times from the V3-BITS-PUBMED validation row |
| bloom FPR | `campaign-v2/{arxiv,goodreads,pubmed}-bloomwidth` (Triton, mean over seeds) |
| defect ledger | `artifacts/exhibits/20261009-1101-idea6/ledger.md` |
