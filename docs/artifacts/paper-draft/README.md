# paper-draft: paper-materials page (NOT CITABLE)

The page behind the private Artifact "LiNR + SilverTorch Repro Materials"
(<https://claude.ai/artifact/7HLZdKki9NiL9pkNrHziyS>): the thesis (the papers'
claims are regime claims), a regime map of claims against scale and batch, one
figure per finding grouped by the three knobs (scale, batch, the filter), the
mechanisms the papers do not state, and the C1-C7 table. Current state only;
exploration numbers, rerun by the final pass before anything is cited
([decisions](../../decisions.md#campaign-v2-user-2026-10-08)).

- `extract.py HUB`: reads the legs and exhibit tables fetched from
  `pinkmeme/eval-results` into `HUB` (default `/scratch/paper-draft/hub`,
  records only, no sidecars) and two repo tables, and writes `data.json`.
  Every number comes from a record or a table; none is typed in from notes.
- `build.py OUT`: inlines `data.json` into `page.html` and writes the page.
  The regime map in `page.html` restates values the charts and T1 carry.

## Sources

| figure | source |
|---|---|
| IVF recall at 24 probes vs N | uniform synth `campaign-v2/goodreads-synth-synth`, `campaign-v2.5/arxiv-synth-synth`, `campaign-v2.9/yfcc10m-synth-synth`, `campaign-v2.5/laion30m-synth-synth`; real filters `campaign-v2.9/{goodreads,arxiv,yfcc10m}-filter`, `pubmed-deep`, `laion30m-filter` |
| IVF vs exact at recall 0.95 | per real sweep, same leg / box / code (v2.9): `campaign-v2.9/{goodreads,arxiv,yfcc10m}-filter`, `pubmed-filter` + `pubmed-deep` (IVF), `laion30m-x`; IVF at the dataset's tuned probe count, exact = the cheaper of V1 / V2, graph p50 |
| co-design (C5) | below 30 M `artifacts/exhibits/20261009-2125-c5/c5-below30m.csv` (paired); `campaign-v2.9/pubmed-codesign-ours`, `campaign-v2.9/laion30m-codesign-laion30m` (ours, graph), `campaign-v2.8/laion30m-codesign-laion30m` (Meta -O3, eager) |
| V3 (C2) | `campaign-v2.5/pubmed-v3bits` against the cheaper of V1 / V2 in `campaign-v2.5/pubmed-router` (same box and code); bits: `campaign-v2.2/goodreads-synth-v3bits` |
| V2 / V1 (C1) | v2.9, nine rates: `campaign-v2.9/goodreads-synth-v1v2`, `arxiv-synth-v1v2-pod1`, `yfcc10m-synth-synth` (seeds pooled), `laion30m-synth-v1v2`; real filters `campaign-v2.9/{goodreads,arxiv,yfcc10m}-filter`, `laion30m-x`; ratio inside each interleave group; LiNR point = its Table 3 V2 / V1 at B 1 |
| C3 | `campaign-v2.10/{goodreads,arxiv,yfcc10m,pubmed}-c3-real` (torch.compile max-autotune V1 vs Triton V1, eager) |
| Meta vs ours (C7) | d128 `campaign-v2.8/{goodreads,arxiv}-h2h` (Meta -O3 fp16, eager, `kernels_us`); d768 `campaign-v2.9/pubmed-filter` (mean over three interleave groups); 30 M `docs/artifacts/campaign-v2.10/st-topk-30m/st-topk-30m.md` (ST-TOPK, Meta fp16 and int32) |
| local pass rate (C6) | `campaign-v2.5/arxiv-synth-synth`, `campaign-v2.5/arxiv-corr-synth-synth`; `artifacts/exhibits/20261009-1057-gls/gls-sweeps.csv` |
| recall ceiling (C6) | `artifacts/ceiling-int8-gr-ax`, `artifacts/ceiling-int8-pubmed`, `artifacts/yfcc-int8` |
| bloom FPR (C4) | `campaign-v2/{arxiv,goodreads,pubmed}-bloomwidth` (Triton, mean over seeds) |
| verdicts, bloom parity, ST-TOPK | T1 `artifacts/exhibits/20261010-0818-parity/t1.md`; [validation](../../validation.md) |
