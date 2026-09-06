# Artifacts — dataset ingestion (Phase E)

Raw material behind [../dataset-candidates.md](../dataset-candidates.md)
and the Phase E steps of [../00-roadmap.md](../00-roadmap.md). One
subdirectory per dataset, added as each step runs.

## `yfcc10m/` — roadmap E1, 2026-09-06

Produced on the CPU box (128 cores, no GPU used) by
[`evaluation/eval_datasets/yfcc.py`](../../../evaluation/eval_datasets/yfcc.py)
and
[`yfcc_check_gt.py`](../../../evaluation/eval_datasets/yfcc_check_gt.py),
with `RETRIEVE_DATA_ROOT=/workspace/data`.

| file | what it is |
|---|---|
| [download-log.txt](yfcc10m/download-log.txt) | The E1 download. All six URLs in **D** §3.4 served `HTTP/2 200`; every byte count matches the sizes the plan probed on 2026-09-05, including the two files the plan had *not* probed (`query.metadata.public.100K.spmat` 1,907,024 B, `unfiltered.GT.public.ibin` 80,000,008 B). Nothing failed and nothing needed a mirror. The second block is `yfcc download` re-run afterwards, confirming it is idempotent. |
| [raw-manifest.json](yfcc10m/raw-manifest.json) | `yfcc convert --sha256`: sha256 and size of each raw file, plus the header facts parsed out of them — 10,000,000 × 192 uint8 base vectors, 100,000 queries, 108,210,476 item-tag entries over a 200,386-tag vocabulary, filtered GT at k=10, unfiltered GT at k=100. This is the provenance record for **F3**. |
| [prep_log.json](yfcc10m/prep_log.json) | `yfcc prep` + `yfcc attrs`. Notable: 10.821 tags/item mean with a 1,517 max; 61,626 single-tag and 38,374 two-tag queries; only 7,910 of 200,386 tags are ever queried; base-vector norms have cv = 0.011. `attrs.gt_fidelity` is what the K=32 tag cap costs — 74,214 of 100,000 queries keep their whole shipped-GT row. |
| [gt_check-cpu-200.json](yfcc10m/gt_check-cpu-200.json) | `yfcc_check_gt --device cpu --limit 200`, the CPU rehearsal of **E1's gate**: 200/200 queries reproduced, 195 id-exact and 5 tie-equivalent, `max_abs_distance_error = 0.0`. The gate itself is the same command with `--device cuda` and no `--limit`, and is still pending. |
| [gt_check-cpu-100-cosine.json](yfcc10m/gt_check-cpu-100-cosine.json) | `--metric ip` on 100 queries — a diagnostic, not a gate. An exact *cosine* filtered top-10 (the metric the harness actually scores) reproduces the shipped squared-L2 GT on 59 % of queries at mean recall 0.951. This is the size of deviation 1 in [../../system/datasets.md](../../system/datasets.md#yfcc10m). |

Both `gt_check-*` files keep at most two `bad_examples` rows; the scripts
emit five.
