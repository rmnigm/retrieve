# campaign-v2: claims and the record inventory (roadmap P-INV)

Scripts behind the campaign-v2 plan's measured inputs. Raw outputs are on the Hub under
`artifacts/campaign-v2/` ([hub-index](../hub-index.md)). Nothing here is citable: these are
planning numbers read from records whose gates are not green.

## Inventory

[`inventory.py`](inventory.py) reads every bench record of every
[hub-index](../hub-index.md) subtree (fetched with `bench fetch --path-in-repo <subtree>
--results <root>/<subtree>`, 2026-10-08, 947 MB) and writes, CPU only, standard library:

| file | content |
|---|---|
| `inventory.csv` | one row per (record, perf entry): key, params split build/query, seed, k, bs, mode, code_version, status, `unstable`, fraction of windows below the leg's max sampled clock, pass rate, `recall_oracle@100/@1000`, median ms, `elapsed_s`, `build_s` (1,579 records, 25,556 rows) |
| `pivot.csv` | dataset × inputs × algo × backend × code_version × subtree: records, `unstable`, not-ok |
| `pass_rates.csv` | the exact oracle's mean pass rate per (dataset, inputs, sweep), and whether the sweep is kept |
| `wall_time.csv` | per-cell wall time by dataset, suite, algo, backend, filter kind |
| `reuse.csv` | the record reuse rule per record ([decisions](../../decisions.md#campaign-v2-user-2026-10-08)) |

```bash
python -I docs/artifacts/campaign-v2/inventory.py /scratch/cv2-inventory /scratch/cv2-inventory-out
```

### (a) Pass rates and the kept sweeps

| dataset | sweep (pass rate) | kept |
|---|---|---|
| goodreads | `c1_lang_reverse` 0.621, `c2_format` 0.404, `c3_year` 0.367, `c0_genre` 0.331, `c0c1` 0.199, `all4` 0.041 | `c1_lang_reverse` (high), `c0_genre`, `all4` (conjunctive) |
| arXiv | `c3_nversions` 0.444, `c2_year` 0.234, `c0_maincat` 0.136, `c0c2` 0.039, `all4` 0.0094 | `c3_nversions` (high), `c0_maincat` (low), `all4` (conjunctive) |
| PubMed | `c3_journal_reverse` 0.9993, `c2_year` 0.146, `c0c2` 0.023, `all5` 0.018, `c0_mesh` 0.00019 | `c3_journal_reverse` (reverse, high), `c0_mesh` (low), `all5` (conjunctive) |
| YFCC-10M | `tags_and` 0.018 | `tags_and` |

Goodreads pass rates depend on the attributes only: `c0_genre` is 0.330658 on both the gSASRec and
the E1c records. arXiv: the re-plan named "the lowest of `c2_year` / `c3_nversions`", but its own
rule (one high, one low, one conjunctive) and the measured rates pick `c3_nversions` as the high one.

### (b) Measured wall time per cell (old grid: 3 bs × 3 k × 2 modes)

Median `elapsed_s` per record (the job's build is extra, `build_s`, once per job). The timed
windows are ~31 % of a cell on every dataset; the rest is warm-up, graph capture and the quality
pass. The v2 grid times 2 bs × 2 k (8 variants instead of 18), so a v2 cell costs roughly half.

| dataset | V1 | V2 | V3 | SilverTorch triton | official | torch arms |
|---|---|---|---|---|---|---|
| goodreads (0.8 M) | 129 s | 147 s | 136 s | 103 s | 79 s | ST 917 s, V2 784 s, V1 515 s |
| arXiv (3 M) | 225 s | 306 s | 357 s | 107 s (deep 100 s) | 105 s | — |
| YFCC (10 M, d192) | 2,344 s | 8,513 s | 8,004 s | 142 s | 1,132 s (clause) | — |
| PubMed (10 M, d768) | 1,332 s | 2,612 s | OOM then | 1,404 s (old tile) | 427 s | — |

Whole legs as run: `d1/arxiv-deep` 873 cells in 36.0 h, `d1/arxiv` 126 in 6.5 h, `d1/pubmed` 44 in
15.2 h, `d1/yfcc10m` 7 in 6.0 h, `d1/goodreads` 105 in 3.1 h, `d1/arxiv-codesign` 60 in 0.4 h.

**Budget consequence.** The 10 M exact arms (V1-V3, postfilter) cost 0.4-2.4 GPU-h per cell even at
half the old grid, and the torch arms cost 5-9× Triton at 0.8 M. With 3 seeds everywhere, YFCC's
synth leg (5 points × clause and bloom × V1, V2, V3 × 2 pools, postfilter × 2 α) comes to
roughly 60-120 GPU-h instead of the re-plan's 10-14; PubMed's leg to ~20-25 instead of ~14. How
V2/V3 cost moves with the pass rate at 10 M is not measured yet (the one YFCC sweep sits at
p = 0.018); the goodreads pilot and the arXiv synth leg measure it before YFCC runs. The roadmap's
estimates use these numbers, and the budget gate applies.

### (c) Which records pass the reuse rule

Counted per subtree (`reuse.csv`); "in grid" = bs/k in {1, 16} × {100, 1000} entries exist, kept
sweep, no `n_probe` 4/256, no official clause, no `linr_v4`. Quality reuse also waits on the freeze
gate (golden equality on the frozen tree); SilverTorch-Triton timing waits on CV2-LIB's #16
keep/revert decision ("pending").

| subtree | records | in grid | quality reusable (in grid) | timing reusable | timing pending #16 |
|---|---|---|---|---|---|
| `d1/arxiv` | 126 | 84 | 126 (84) | 72 | 3 |
| `d1/arxiv-deep` | 873 | 309 | 873 (309) | 374 | 143 |
| `d1/arxiv-codesign` | 60 | 24 | 60 (24) | 18 | 0 |
| `d1/yfcc10m` | 8 | 6 | 7 (5) | 5 | 2 |
| `d1/pubmed` | 52 | 22 | 44 (18) | 31 | 0 |
| `d1/goodreads`, `d1-a`, `b3`, `c4`, `c5` | 287 | — | 0 (gSASRec inputs) | 0 | 0 |

PubMed's 13 SilverTorch-Triton records are not timing-reusable whatever #16 decides: they were timed
on the spilling `D_PAD = 1024` tile. The arXiv `codesign` records pass the clock criterion on 18
cells but C5 is a ratio claim, which the plan times interleaved: V-CODESIGN reruns it. The 16
PubMed SilverTorch-Triton cells of the halted D1-E rerun are local to the development pod, not on
the Hub, and are not counted.

## Reuse entries

[`reuse/reuse.py`](reuse/reuse.py) expands today's v2 grid with the harness's own `load_matrix`
(2,343 cells over every suite in `evaluation/config/suites.yaml`), matches every fetched record to
its cell by key block, and judges it by the reuse rule: inputs (E1c goodreads, license-fixed
arXiv subtrees; PubMed held for V-PUBMED's embedding identity check), the arm's evidence (the
per-arm table in the script: golden gates and the library diff `72e5a90..408b1188` read per code
path), and for perf the clock clause against the 1410 MHz device max over the v2-grid windows
(ratio suites `synth` / `codesign` / `h2h`: interleaved only). It runs from `evaluation/` (it
imports `bench`); outputs are on the Hub as `artifacts/campaign-v2-reuse`
([hub-index](../hub-index.md)).

```bash
cd evaluation && uv run python ../docs/artifacts/campaign-v2/reuse/reuse.py /scratch/cv2-inventory /scratch/p-reuse-out
```

A manifest `match` names dataset, suite, algo, backend, filter kind and sweep, never `params` or
`seed`, so an entry is written only where reusable records cover **every** v2 cell under it and no
record off the v2 grid (a dropped sweep or `n_probe` 32) sits under it. That gives 12 entries,
36 cells, quality and perf:

| what | cells |
|---|---|
| v2 cells | 2,343 |
| a quality-reusable record exists | 241 (arXiv `deep` 162, `filter` 56, `codesign` 18, YFCC 4, goodreads 1) |
| a perf-reusable record exists | 100 (arXiv `deep` 51, `filter` 45, YFCC 4) |
| reused through `campaign.yaml` | 36: arXiv `filter` V1 / V2 / V3 triton, clause and bloom, `c0_maincat` and `all4`, seeds 0-2 |

The 36 perf-reused cells took 10,345 s at D1's 18-variant grid (≈ 1.4 GPU-h at the v2 grid's 8
variants). `bench report --manifest evaluation/campaign.yaml` over the fetched `d1/*` subtrees
selects those 36 records; every other old cell reads "missing".
