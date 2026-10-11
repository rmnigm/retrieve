# F-REPRO sizing: the final pass, sized from the campaign's records

CPU-only sizing of the final repro pass (roadmap F-REPRO). The raw outputs are on the Hub under
`artifacts/f-repro/sizing-<stamp>` ([hub-index](../hub-index.md)). The plan built on them is in the
controller's handoff notes. Every number here is NOT CITABLE.

| file | what |
|---|---|
| [`sizing.py`](sizing.py) | expands every suite of `evaluation/config/suites.yaml` through `config.load_matrix`, plus the scratch suites the campaign ran (read from their records; library gates `stw2` / `stt` / `v1g` excluded, `ivf-tune` / `n95` / `router` / `c7-laion30m` marked obsolete). Each cell is matched to its newest record of the same key (`Job.key(params)` = the record's key block); the scratch suites the final grid renamed or folded (`tune`, `tune-t` → `tune-timed`, `v1v2` → `synth`) price the cells they became. Outputs: `inventory.csv` (per source × suite × dataset × arm: cells, with a record, with a v2.9+ record, estimated, cell GPU-h, seed-0 cells and GPU-h, newest code_version), `off-grid.csv` (records under a suites.yaml suite that no current cell matches), `overhead.csv` (per leg, wall time from the driver log over the summed cell time) |
| [`recipe.schema.yaml`](recipe.schema.yaml) | draft schema of the final pass's `recipe.yaml`: pods, GPUs, dataset pins, suites, seeds, commands |

## Duration model

- **Cell time** is the record's `elapsed_s`.
  - An interleave group's records each carry the group's whole time, so it is split evenly across them.
  - It comes from the newest record at v2.9-v2.11 where one exists, else the newest older one.
  - SilverTorch Triton timing windows on the redo ledger are rescaled by the ledger's measured factor (`load.redo`: ST-WIDE-2 `wide` 0.97, `sparse?` 0.9 eager only; ST-TOPK 0.8 at bs ≥ 16). Only windows above 2 ms move, so this changes the total by under 1 %.
- **A cell without a record** takes the median of its (suite, dataset, arm), then (suite, arm), then arm. These are mostly seeds 1-2, never run under claims first.
- **Index builds** are added once per job: `elapsed_s` excludes them (`build_s`; ≈ 400 s per 30 M job). A job's build is its matched records' `build_s`, else the median of the same (dataset, algo, backend, n_lists), else of (algo, backend, n_lists) at a comparable N (within 1.5×). Column `gpu_h_build`.
- **Process overhead** (start, data load, oracle) is the per-leg wall over summed cell time plus builds (median 1.15).
  - The median is 1.15 over 20 legs, most at 1.0-1.4. Without the builds it read 1.37, and PubMed's tune leg 36×: the builds were the missing time.
  - Checked against the final pass's first finished legs: measured wall is 0.78 of this pricing overall (30 M co-design 1.01 h vs 1.17 priced).
