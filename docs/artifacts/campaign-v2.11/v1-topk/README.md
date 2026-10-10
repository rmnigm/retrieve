# V1-TOPK: LiNR V1's top-k on the two-level block top-k (campaign-v2.12 candidate)

Controller's kernel order: after V1-BS1, V1's full-width top-k with ST-TOPK's block top-k. Branch dev/v1-topk, off
dev/v1-bs1 with its 2^24 fix merged. Raw outputs are on the Hub at `artifacts/v1-topk`
([hub-index](../../hub-index.md)). Pod b, A100-SXM4-80GB, 2026-10-10. **NOT CITABLE.**

## The change
ST-TOPK's `_host._two_level_topk` moved to `retrieve.functional.two_level_topk`, and it now takes a ragged tail: the
fewer than 256 slots past the last whole block are always candidates; block maxima cover whole blocks only. Two callers:
- `functional.masked_topk`, LiNR V1's top-k over its unpadded `[B, N]` scores. It engages at B·N ≥ 2²³ with ≥ 8 blocks
  per top-k slot.
- SilverTorch's `_host.probe_topk`. Its buffer is padded, so there is no tail and the op sequence is the same.

`_host` reads the thresholds through `functional`, so one monkeypatch reaches both its padding and the check.

## Gate (`gate_v1.py`, `gate_st.py` against dev/v1-bs1 11242c7, the fixed V1-BS1; one process, swapped build order, 8 windows)
144 cells, ids and scores `torch.equal` in every one, none slower:

| cells | after / before |
|---|---|
| **V1** goodreads d128 clause (`c0_genre`, `all4`) | 0.58-1.004 |
| **V1** arXiv d128 clause / bloom | 0.55-1.004 |
| **V1** arXiv d256 clause | 0.58-1.001 |
| **V1** arXiv k 1000 | 0.67-1.003 |
| **V1** PubMed d768 clause | 0.81-0.98 |
| **SilverTorch** PubMed d768 bloom (`c0_mesh`, `c2_year`) + exact | 0.999-1.008 |
| **SilverTorch** arXiv d128 bloom (`c0_maincat`, `all4`) + exact | 0.988-1.009 |
| **SilverTorch** arXiv d256 bloom | 1.000-1.002 |

- V1 cells ran at bs 1 / 16 / 64, eager + graph, k 100 plus the k 1000 rows. Cells below the thresholds read 1.000.
- SilverTorch cells ran at `n_probe` 24 / 256 / 1024 (d256: 128 / 1024), bs 16 / 64, eager + graph, including ST-TOPK's
  engaging widths. They are unchanged: SilverTorch only moved to the shared helper.
- **SASS:** 14 / 14 narrow cubins identical to dev/v1-bs1 (`sass_narrow.py`).
- **Suites:** library 896, harness 572.
- **New test:** `test_two_level_topk_equals_topk_up_to_ties` (ragged tail, heavy ties, `-inf` rows). The ties test now
  patches `functional`.
- `prefix/` on the Hub holds the first runs against the pre-fix dev/v1-bs1, superseded.

## Files
- `gate_v1.py`: V1 keep-rule gate (`k=K`).
- `gate_st.py`: SilverTorch gate (ST-TOPK's).
- `sass_narrow.py`: narrow SASS hashes.
