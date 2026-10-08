# V-CODESIGN at `408b1188`: official bloom, partial vs full (roadmap Phase V)

**Stale at campaign-v2.1.** The v2.1 quantize fix adds ~30 µs to every SilverTorch eager path,
official included, so this run reruns whole at v2.1 (controller, 2026-10-08). The records are kept
as `artifacts/v-codesign-408b` and are not the F4b source.

The `codesign` suite on arXiv (`n_lists` 1664; `c3_nversions`, `c0_maincat`, `all4`) and goodreads
(E1c, `n_lists` 1024; `c0_genre`, `c2_format`, `c3_year`), d128, official SilverTorch bloom,
`bloom_path` partial vs full interleaved per cell, `n_probe` {8, 32, 128}, seeds 0-2, k 100, bs
{1, 16}. code_version `408b1188` (main checkout at staging `f2be5c7`), A100-SXM4-80GB, GPU 0, cores
0-63,128-191, single-GPU pod. Hub: `artifacts/v-codesign-408b` (records, logs, the 1 Hz clock trace,
summaries, `bench report --manifest` output) and the earlier `campaign-v2/<dataset>-codesign` uploads
of the same records ([hub-index](../../hub-index.md)). Not citable.

| file | what |
|---|---|
| [`v-codesign.sh`](v-codesign.sh) | `bench oracle` per dataset, then `bench campaign --suite codesign --dataset arxiv --dataset goodreads --resume --interleave` (sources [`../h2h-final/common.sh`](../h2h-final/common.sh)) |
| [`codesign_summary.py`](codesign_summary.py) | per `(sweep, n_probe, bs)` medians over seeds, paired full / partial ratio, oracle recall, clock windows |

## Outcome at `408b1188`

- 108 / 108 records ok (54 per dataset: 18 cells × 3 `n_probe`), 92 `unstable`.
- `full` is faster than `partial` in every cell: paired full / partial eager median 0.80-0.88
  (arXiv), 0.78-0.86 (goodreads), every seed below 0.93. Official cannot be captured, so there
  is no `graph` entry. This repeats the official-code finding that the partial-response path is
  slower than Meta's own full-`N` search ([official vs reimplementation](../../../paper/official-vs-reimplementation.md)),
  now interleaved, and it runs against C5 as stated.
- Recall is identical between the two paths at every point (oracle recall@100 arXiv 0.52-0.94,
  goodreads 0.82-0.98 over `n_probe` 8 → 128).
- Wall time: goodreads 1.50 h against arXiv 0.24 h, although goodreads has a quarter of the items.
  About 200 s per goodreads cell goes before the first timed window (the quality pass), against
  about 25 s on arXiv. The query counts are equal (≈ 10k). Not investigated; the timings are not
  affected.
- Clocks: 457 / 648 timing windows below 1410 MHz (min 1140). The 1 Hz trace under > 50 % load
  held 1410 MHz (203 / 203 samples), max 41 °C, 341 W.
- GPU time 1.75 h against the 2 GPU-h estimate.
