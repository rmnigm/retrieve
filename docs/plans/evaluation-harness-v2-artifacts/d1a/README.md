# D1-a (goodreads leg) — `filter` suite, d128, seed 0, `{triton, torch, official}`

The first of the five staged campaign runs of
[../../evaluation-harness-v2.md](../../evaluation-harness-v2.md) §6 WP-5,
**stopped at the goodreads/arxiv dataset boundary**. D1-a as dispatched was
estimated at 4–6 h and measured out at 25–30 h, so the orchestrator split it at
the dataset boundary: this worker ran the goodreads leg (11 jobs, 126 cells) and
the arxiv leg is a separate worker resuming into the same results tree.
`--resume` keys on `(cell, code_version)`, so nothing recorded here is re-run.
The validation record — environment, command lines, per-cell results, gate
verdicts with numbers — is appended to that plan; this directory holds the
scripts that produced it and `bench report`'s output over the stage's records.

**Nothing here is citable.** D1's gate is not green, the records were produced
on a branch, and `report/` carries the `NOT CITABLE` banner on every artifact
(CLAUDE.md rule 2). Numbers in the validation record are measurements, not
settled results.

| file | what it is |
|---|---|
| [stage_a.py](stage_a.py) | The driver. `bench campaign`'s loop (`bench/cli.py:campaign`) reproduced with the one narrow the CLI has no flag for: `--seed 0`. Same `(dataset, dim, algo, backend)` process boundary, same `python -m bench.cli run …` child on the same interpreter, same `_logs/` layout, same parity-spill lifetime (`rmtree(results/_parity)` when `(dataset, dim, algo)` changes). Without it, `bench campaign` would have run seeds 0/1/2 on the headline sweeps — that is stage b, out of this stage's scope. It is an artifact script: it calls the harness, it does not change it. |
| [d1a_gate.py](d1a_gate.py) | The four gate clauses the records can decide, over the stage's JSONL: (1) every cell `status: ok`, printing stage and traceback for any failure; (2) completeness against `--expected-cells`; (3) `median_ms(bs=16) < 16 x median_ms(bs=1)` over every `(cell, k, mode)`, printing the worst ratio and where; (4) every `graph` entry on a capturable backend measured — `cudagraph_skips` is not a record field, `measure.graph_callable` raises `NotCapturable` instead, so the clause is "null `reason`", and on `official` only `not_capturable` is accepted. Then informational: the `env.sm_mhz_load` range, per-entry `sm_mhz` min/median/max, `sm_mhz_idle`, the `clocks_drift` count, `unstable` entries bucketed by `bs`/`mode`/`algo`, per-group parity minima, and `code_version` / `dirty` / `git_branch`. |
| [d1a_mode_ids.py](d1a_mode_ids.py) | The "ids identical across `mode`" clause, which the records cannot decide — a record stores each mode's latency, not its output. Rebuilds one representative cell per `(dataset, algo, backend)` through `bench.run`'s own `sweep_assets` / `build_module`, then compares eager `module(*args)` against `measure.graph_callable(module, *example)` over 8 pool batches at each batch size and `k in (100, 1000)`, reporting `rows_differing` and `score_max_abs_diff`. `official` is eager-only (O D7) and is skipped with its `NotCapturable` reason printed. |
| [d1a_rerun.sh](d1a_rerun.sh) + [d1a_rerun_diff.py](d1a_rerun_diff.py) | The byte-identical-rerun clause. Re-runs a handful of cells with `--force --skip-perf --output <tmp>` and diffs `json.dumps(quality, sort_keys=True)` against the campaign's record for the same cell. Three keys are dropped before the diff: `jaccard_vs_first@100`, `score_max_abs_diff` and `parity` — a rerun is always its own parity group's first backend, so it is the reference and has nothing to compare against. Everything else must match byte for byte; latency is not part of this clause and is skipped. |
| [gate.txt](gate.txt) | `d1a_gate.py`'s output over the 72 gated cells: the four record-decidable clauses with their numbers, then the informational block (clock range, `unstable` buckets, per-group parity minima, `code_version`). Verdict `PASS`. |
| [driver.log](driver.log) | `stage_a.py`'s own stdout: one line per child with rc and wall time. The harness's per-group logs stay in `evaluation/results/_logs/`. |
| `report/` | `bench report` over the stage's records, no `--gate`. Every file says `NOT CITABLE`; that is correct and stays. |
