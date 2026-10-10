# ST-XQ 30 M profile (st-dloop's ask, control 2026-10-13-190000000)

laion30m d256, bloom (m_bits 1024, k_hash 5), partial path on both arms, n_lists 16384, n_probe 128, bs 16 / 64, k 100, at library
`e16512f5` (dev/v1-topk 8c9b336), Meta's extension at the pin built `-O3` (`official`, int32). Pod d GPU 0, 2026-10-10 11:46-12:20.
Driver [`prof.sh`](prof.sh); [`bounds_share.py`](bounds_share.py) is [campaign-v2.9/st-wide-2/bounds.py](../../campaign-v2.9/st-wide-2/bounds.py)
plus the sparse share; the split is [campaign-v2.10/st-topk/prof.py](../../campaign-v2.10/st-topk/prof.py). Hub `artifacts/st-xq-30m-profile`
(MANIFEST `69d2ae6a6482cf7cccc65ede2e17d22899bd3bbf7044e963cabe59ab2e277a3f`). NOT CITABLE.

## Pass rates and bounds (512 rows per sweep)

| sweep | clause pass rate (v2.9 records, mean) | bloom pass rate median / mean | rarest-bit bound median (max) | rows with bound < 1/256 | sparse batches bs 16 / 64 |
|---|---|---|---|---|---|
| c0_domain | 0.0095 | 0.0025 / 0.0106 | 0.0076 (0.063) | 7.2 % | 0 / 0 |
| tags4 | 0.0027 | 0.0023 / 0.0025 | 0.166 (0.249) | 0 % | 0 / 0 |

`PreparedFilter.sparse` needs every row of the batch below 1/256: it never fires here, so our scorer runs the one-pass scan.
On tags4 (4 sub-queries) the bound is ≈ 72× the bloom pass rate (c0_domain ≈ 3×).

## Kernel split per forward (µs, torch.profiler, 20 calls)

| sweep, bs | ours total / scorer / top-k | Meta total / scorer (7 launches) / top-k |
|---|---|---|
| c0_domain, 16 | 815 / 467 / 236 | 961 / 120 (dot 81) / 473 |
| c0_domain, 64 | 2365 / 1912 / 228 | 2035 / 296 (dot 235, bloom_search 28) / 1008 |
| tags4, 16 | 900 / 550 / 237 | 954 / 102 (dot 41, bloom_search 37) / 478 |
| tags4, 64 | 2705 / 2253 / 227 | 2003 / 252 (dot 138, bloom_search 82) / 1018 |

Our scorer is 3.9-5.4× Meta's at bs 16 and 6.5-8.9× at bs 64; Meta's dot shrinks with the passing items (tags4 < c0_domain), ours does not.
Meta's top-k (at::native on the int32 buffer) costs 2-4.5× ours, so the totals sit within 0.85-1.35× of each other.
