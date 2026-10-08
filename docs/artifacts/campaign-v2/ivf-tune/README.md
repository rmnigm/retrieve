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
| [`control.sh`](control.sh) | replaced `driver.sh` mid-run: stops a round as soon as one cell reaches 0.95 or the `n_lists` / 4 cap cell is written, one-point rounds from `n_probe` 256 (a rebuild costs less than an overshoot cell) |
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

## PubMed (pod b): running

N 10,000,000, `all5` (pass rate 0.0181, 9,985 oracle rows). `n_lists` 1024 (the `n95` leg, three seeds): 0.748 at 128.
`n_lists` 4096 so far: 0.2642 / 0.3377 / 0.4162 / 0.5000 / 0.5878 / 0.6804 at `n_probe` 8 … 256 (cells 23-56 s up to
128, 418 s at 256). Not yet at 0.95: n95 / `n_lists` is above 1/16 here, against 1/64 on goodreads.

## Part A in `suites.yaml`

`n_lists` ≈ 4 √N to a power of two: goodreads 4096 (tuned), arXiv (N 2,988,996) 8192, YFCC and PubMed (10 M) 16384;
the synth twins take their real dataset's. n95 is written only where measured: goodreads 64. **n95 / `n_lists` depends on
the pass rate, not only on N**: 1/64 at goodreads' 0.33, above 1/16 at PubMed's 0.018, so no n95 is extrapolated. arXiv's
median sweep `c0_maincat` passes 0.136; YFCC's `tags_and` ≈ 0.018 (a CPU-rehearsal log, not checked on the full set).
arXiv's and YFCC's n95 come from pod c's checks, PubMed's from its capped tuning (part B).
