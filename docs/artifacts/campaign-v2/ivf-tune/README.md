# IVF-TUNE: SilverTorch `n_lists` and n95 per dataset size

Roadmap IVF-TUNE ([decisions](../../../decisions.md#campaign-v2-user-2026-10-08), *IVF tuned per dataset size*). Quality
only, at campaign-v2 (`408b1188`): SilverTorch triton, clause, the median kept `filter` sweep, seed 0, bs 16, k 100,
`n_probe` doubling per `n_lists` until `recall_oracle@100` ≥ 0.95, capped at `n_lists` / 4 (25 % of the items scanned;
user, 2026-10-09). **The records are artifacts, not paper numbers**, in
their own results tree (`/scratch/ivf-tune/results`), never a campaign tree. How the values reach `suites.yaml`:
[evaluation](../../../system/evaluation.md#ivf-tuning).

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | PubMed rounds of three doublings, each a scratch `ivf-tune` suite appended to a copy of `config/suites.yaml` (`--config-dir`); `ks` {100, 1000} so the existing `all5` oracle (k_gt 1000) is reused |
| [`control.sh`](control.sh) | replaced `driver.sh` mid-run: stops a round as soon as one cell reaches 0.95 or the `n_lists` / 4 cap cell is written, one-point rounds from `n_probe` 256 |
| [`finish-k100.sh`](finish-k100.sh) | the last 16384 points at k 100 (the k-tile limit below) |
| [`tune.py`](tune.py) | the table (recall, items scanned, s per cell), each `n_lists`' n95 and the pick |

## goodreads (pod c): pick `n_lists` 4096, `n_probe` 64

N 797,084, `c0_genre` (achieved pass rate 0.3307, the median of `all4` 0.0409 / `c0_genre` / `c1_lang_reverse` 0.6205),
E1c `sasrec-ssm-logq-d128`, 9,859 oracle rows. Hub `artifacts/ivf-tune/goodreads`, MANIFEST sha256
`f2e20325ea4cecad32a6247c48346ea43cc4ea8f32bda5458477b5a7761777f5`; 0.08 GPU-h.

| n_lists | n_probe | recall_oracle@100 | items scanned |
|---|---|---|---|
| 1024 | 8 / 16 / **32** / 64 / 128 / 256 | 0.8258 / 0.9162 / **0.9601** / 0.9777 / 0.9833 / 0.9845 | 7,251 … **25,933** … 200,295 |
| 4096 | 8 / 16 / 32 / **64** / 128 / 256 | 0.7124 / 0.8385 / 0.9177 / **0.9586** / 0.9764 / 0.9826 | 5,653 … **16,550** … 53,914 |

n95 is 32 at 1024 and 64 at 4096; the pick, 4096 / 64, scans 1.57× fewer items. 4096 ≈ 4.59 √N; 64 = `n_lists` / 64.

## PubMed (pod b): neither n_lists reaches 0.95 under the cap → 4096 / 1024

N 10,000,000, `all5` (pass rate 0.01809, 9,985 oracle rows), seed 0, k 100, at `408b1188`. Hub `artifacts/ivf-tune/pubmed`,
MANIFEST sha256 `e18b49392c060ee11fa6c5d9e0c64e715445e29ba1ccf9f396f2c509835e2548` (19 records: 16 ok, 3 failed; logs,
the scratch suites, the scripts, `table.md`); 1.4 GPU-h (21:25-22:47). `n_lists` 1024 (the `n95` leg, three seeds): 0.748 at 128.

| n_lists | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|---|---|
| 4096 | 0.2642 | 0.3377 | 0.4162 | 0.5000 | 0.5878 | 0.6804 | 0.7758 | **0.8727** (cap) |
| 16384 | 0.1968 | 0.2583 | 0.3228 | 0.3919 | 0.4666 | 0.5435 | 0.6268 | 0.7144 |

Items scanned at 4096 / 1024: 2,504,096 (25 %). 16384 was stopped by the controller's order after 1024 (its 2048 and
4096 points did not run). Decision (controller): **`n_lists` 4096, `n_probe` {24, 1024}**, the higher recall at its cap.
Cell cost at 4096 jumps past `n_probe` 128 (48 → 418 → 485 → 840 s), at 16384 it stays at 14-53 s; not investigated.

**k-tile limit.** Every `n_probe` 2048 cell at k 1000 failed to compile (three `failed` records): the probe scorers' id
epilogue (`retrieve/src/retrieve/ops/triton/common.py`, launched with `KP` = `next_pow2(k)`, `NPP` = `next_pow2(n_probe)`
by `_host.py`) builds a `[KP, NPP]` tile, and Triton allows at most 2^20 elements. At k 1000, `n_probe` ≤ 1024; at k 100,
≤ 8192. [`finish-k100.sh`](finish-k100.sh) reran 16384's last points at k 100 (its own k_gt 100 oracle) until the stop
order; [`control.sh`](control.sh) retried a failing round instead of stopping (fixed in practice by that replacement).

## Values

`suites.yaml` holds the current values (the controller sets arXiv, YFCC and PubMed on staging): `n_lists` by size, n95
only where measured, under the 25 % cap. **n95 / `n_lists` depends on the pass rate, not only on N**: goodreads 1/64 at
0.33, arXiv 1/8 at 0.136 (pod c), PubMed and YFCC above 1/4 at ≈ 0.018 (their cap points).
