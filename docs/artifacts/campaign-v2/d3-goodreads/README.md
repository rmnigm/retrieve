# D3 goodreads: `bloomwidth` (408b1188) and `bloomwidth-timed` (campaign-v2.1)

Roadmap Phase V, D3, the goodreads piece. arXiv is [d3-arxiv](../d3-arxiv/README.md). The state
is in [validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until
D1-G.

| file | what |
|---|---|
| [`d3.sh`](d3.sh) | `bloomwidth` (quality only) at code_version `408b1188`: `bench oracle`, then `bench campaign --suite bloomwidth --dataset goodreads --resume`, under `flock /scratch/gpu0.lock` |
| [`d3-timed-v21.sh`](d3-timed-v21.sh) | `bloomwidth-timed` at `campaign-v2.1` (code_version `f01255f1`) into the fresh tree `/scratch/campaign-v21/results` |

Both source [`../h2h-final/common.sh`](../h2h-final/common.sh). The table comes from
[`../d3-arxiv/summary.py`](../d3-arxiv/summary.py), run on the uploaded slice
(`summary-goodreads.txt` on the Hub, `artifacts/d3-goodreads`).

## `bloomwidth` at `408b1188`

GPU 0 of the one-GPU A100-SXM4-80GB pod, main checkout at staging `f2be5c7`, `taskset -c
0-63,128-191`; goodreads E1c d128, `c0_genre` (pass rate 0.37), seeds 0-2, k {100, 1000}, bs 16.
42 / 42 records ok, rc 0; children triton 2,257 s and official 776 s, 0.85 GPU-h with the oracle.
Hub `campaign-v2/goodreads-bloomwidth`.

- **FPR.** Triton `bloom_fp_rate` is non-zero only at 64 bits: 6.6e-3 at `k_hash` 3 and 3.7e-5 at
  `k_hash` 5. It is 0 at every width ≥ 128, and 0 for official at both `k_hash`.
- **Recall.** At 64 bits / `k_hash` 3, recall_oracle@100 drops to 0.9299 and @1000 to 0.8329,
  against 0.9457 / 0.8398 at every other triton width. At 64 / `k_hash` 5 it is 0.9456 (@1000
  0.8398). Official is 0.9456 / 0.8398 at both `k_hash`.
- **Index size.** Triton `index_mib` 116.1 / 122.1 / 134.3 / 158.6 / 207.3 / 304.6 at 64-2048 bits.
  Official is 130.0 (`k_hash` 3) and 143.3 (`k_hash` 5); it ignores `m_bits`.
- **Surprise check.** At the default width (1024 bits, `k_hash` 5, `n_probe` 24), the triton and
  official cells equal H2H-FINAL's goodreads `bloom` cells at the same code_version, seed by seed:
  recall_oracle@100 and @1000, `index_mib` and `bloom_fp_rate` are identical in all 6 pairs.

## `bloomwidth-timed` at `campaign-v2.1`

Same pod and pinning, main checkout at staging `fa3a730` (tag `campaign-v2.1`, code_version
`f01255f1`), fresh tree `/scratch/campaign-v21/results`; bs 16, k 100, seeds 0-2, `k_hash` 5. 21 / 21
records ok, rc 0; children triton 1,875 s and official 247 s, 0.59 GPU-h with the oracle. Hub
`campaign-v2.1/goodreads-bloomwidth-timed`; logs and clocks `artifacts/d3-goodreads-timed-v21`.

- **Latency does not depend on width.** Triton eager 0.758-0.830 ms and graph 0.177-0.195 ms over
  64-2048 bits; official eager 1.351 ms (not capturable). Quality equals the `bloomwidth` cells.
- **Against H2H-FINAL (not resolved).** At the default width, official / Triton eager is 1.76
  (1.351 / 0.768 ms). H2H-FINAL's goodreads bloom bs 16 k 100 cell at `408b1188` measured 1.785 /
  0.827 ms (2.16), with the three arms interleaved in one process and `--profile`. Graph agrees
  (0.195 vs 0.201 ms). Both eager arms are faster here, although v2.1's quantize fix was expected
  to add ~30 µs. The direction holds. arXiv shows the same gap at one code_version: its
  separate-process `bloomwidth-timed` at `408b1188` measured official eager 1.52 ms on `c0_maincat`
  bloom bs 16 k 100, against H2H-FINAL's 1.806 ms ([d3-arxiv](../d3-arxiv/README.md)), and v2.1 then
  added the expected +≈ 30 µs. So the gap follows H2H-FINAL's protocol (three arms interleaved in
  one process, `--profile`), not v2.1; which part of it is not established, and the surprise gate's
  interleaved one-cell rerun was not run.
- **Clocks.** 60 / 117 timing windows below 1410 MHz (min 1140), 20 / 21 records `unstable`; the
  1 Hz trace under > 50 % load held 1410 MHz (63 / 63 samples), max 39 °C, 318 W.
