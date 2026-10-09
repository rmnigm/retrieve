# ST-WIDE-2: the bloom two-pass at `D_PAD` ≤ 256, and the probe scorers' prep in one launch (campaign-v2.10 candidate)

Roadmap ST-WIDE-2, plus the controller's fold-in: v2.9 made goodreads 0.8 M bs-16 eager forwards 1.02-1.22× slower
(exhibits). Raw outputs are on the Hub at `artifacts/st-wide-2` ([hub-index](../../hub-index.md)). Pod b,
A100-SXM4-80GB, 2026-10-09. **NOT CITABLE.**

## 1. Where the two-pass pays at `D_PAD` ≤ 256 (`gate_m*`: two-pass against one-pass + table, same tree)
Graph replay, arxiv-synth (3 M), `n_probe` 128-1024. The sweep names are 10× the rate: `p0001` is p = 0.001.

| p | d128 | d192 / d256 |
|---|---|---|
| ≤ 0.003 | 0.57-0.89 | 0.60-0.89 |
| 0.01 | 0.89-0.95 | 1.00-1.10 |
| 1 | up to 1.23 | up to 1.36 |

At d768 the dot dominates: PubMed `c2_year` 0.95-1.02 and `c0_mesh` / `c0c2` 0.56-0.84, so the two-pass stays always on
there.

**Rule.** `prepare_queries` sets `PreparedFilter.sparse` when every row's rarest queried bit is set on under 1/256 of
the items. The op takes the two-pass at `D_PAD` ≤ 256 only on a sparse batch.

**The bound** (`bounds.py`):

| sweeps | bound vs true pass rate |
|---|---|
| synth | equals it |
| one-clause real sweeps | equals it |
| several clauses (arXiv `all4`) | loose: median 0.11 against 0.005 |

On the real sweeps no batch is marked sparse, so the gain shows on low-pass-rate batches only.

## 2. The goodreads eager regression
**Reproduced.** v2.9 against v2.8, goodreads d128 exact, eager, bs 16, `n_probe` 32-128: 1.085-1.105, +35-40 µs on a
host-bound 0.41 ms forward.

**Mechanism.** The per-row table was its own Triton launch, and that shows wherever the eager forward is host-bound.
Measured on v2.9 (`host_cost.py`, `host_cost/host_cost_v29.json`), one eager exact-scorer call at bs 16, `n_probe` 128:

| | host µs |
|---|---|
| op with the table | 305 |
| op without the table | 273 |
| the table launch alone | 14.6 |

**What did not work.** Gating the table on layout work, then on work or probed bytes, only moved the boundary. Eager
cells of 1.08-1.19 remained either way:
- arxiv-synth d128 / d256, `n_probe` 128, bs 16;
- PubMed `n_probe` 256, bs 2.

**Fix.** `probe_prep_kernel`, one program per row and the probe scorers' first launch, writes:
- the int8 query, bit for bit `quantize_int8`, which replaces its ~10 torch launches. It multiplies by the fp32 1/127 as
  eager torch does: a correctly rounded division differed in the last ulp, caught by `test_bloom_full_mask`;
- the per-row table;
- the two-pass count.

The table rule is v2.9's `B · n_probe ≥ 512` again, in every mode.

## 3. Gate (`gate.py` against tag `campaign-v2.9` as `rv_before`, swapped build order, one process)
516 cells, ids + scores `torch.equal` in every one. After / before medians, 8 interleaved windows:

| cells | eager | graph |
|---|---|---|
| goodreads 0.8 M d128, exact + bloom, n_lists 1024 / 4096, `n_probe` 24-128 × bs 1 / 16 | 0.745-0.900 | 0.859-0.943 |
| arXiv d128 bloom, `n_probe` 24-1024 × bs 1-16 (512-pair switch points included) | 0.744-0.983 | 0.862-0.990 |
| PubMed d768 bloom, `n_probe` 24-1024 × bs 1-16 | 0.768-0.993 | 0.862-0.996 |
| PubMed d768 exact | 0.818-0.990 | 0.882-0.994 |
| arxiv-synth d128 / d192 / d256 sparse | 0.571-0.864 | 0.570-0.918 |
| arxiv-synth d128 / d192 / d256 non-sparse | 0.791-0.997 | 0.868-0.999 |
| arXiv d256 | 0.794-0.997 | 0.906-0.998 |

- d192 is arXiv's d256 embeddings cut to 192 dims (`slice=192`): a kernel keep rule, not a dataset claim. YFCC is not
  staged on pod b.
- **SASS** (`sass_narrow.py`, narrow configs): all 10 scorer / id-epilogue cubins are identical to v2.9. The only new
  cubin is `probe_prep_kernel`.

## 4. Suites
- Library: 873 passed on fresh caches.
- Harness: 570 passed, 6 skipped on the staging-merged tree (`test_laion.py` left out: `tldextract`).
- New tests:
  - `test_prep_quantizes_as_quantize_int8` (d 64-768, zero / constant / extreme / tie rows);
  - `sparse` × the bloom parity cases;
  - wide d64 bloom modes under `reduce-overhead`.

## Files
- `gate.py`: modes `graph`, `swap`, `aa`, `slice=D`; prints the sparse share.
- `bounds.py`: pass-rate bound against the true rate.
- `sass_narrow.py`: narrow-config SASS hashes per tree.
- `host_cost.py`: host µs of the table path on v2.9.

Hub `gate/` also keeps the earlier iterations' outputs, by name in the [hub-index](../../hub-index.md) row; the tables
above are the final gate's.
