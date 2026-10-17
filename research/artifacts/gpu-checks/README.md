# gpu-checks: three GPU checks of the papers' claims

Three checks for the reproducibility paper (brief `.chains/gpu-checks/2026-10-17-050000000-brief-gpu-checks.md`), run
on one A100-SXM4-80GB (driver 570.172.08, torch 2.10.0+cu128, triton 3.6.0, CUDA 12.8) at `staging` `61045809` on
branch `dev/gpu-checks`, library `code_version` `fed664d6`, clean tree. They are session scripts, not harness suites,
and have no gate: **nothing here is citable** ([validation](../../validation.md#gpu-checks-not-citable)). Raw records
and logs are on the Hub under `artifacts/gpu-checks/<check>/` ([hub-index](../hub-index.md)).

| file | what |
|---|---|
| [`common.py`](common.py) | dataset loading through the harness (`bench.inputs.load_inputs`), the draw-0 synthetic flags (`eval-data synth-filter`'s rule and seed), the JSONL writer |
| [`check1_native.py`](check1_native.py) | check 1: LiNR's native-PyTorch baseline against our Triton V1 / V2 |
| [`check2_v3bits.py`](check2_v3bits.py) | check 2: LiNR V3 at 512 bits against 128 |
| [`check3_bloom_not.py`](check3_bloom_not.py) | check 3: Meta's bloom with NOT on the four kept reverse-clause filters |
| [`summarize.py`](summarize.py) | the tables below from the records |
| [`upload.py`](upload.py) | one commit per check to `pinkmeme/eval-results` with a `MANIFEST.json` (sha256 per file), read back and checked; the records are not harness-schema, so `bench upload` does not apply |

Run from `evaluation/` (`python ../research/artifacts/gpu-checks/check1_native.py`), with `evaluation/data` pointing at the
staged datasets.

**Datasets.** Goodreads 0.8 M d128 (SASRec `sasrec-ssm-logq-d128`), arXiv 3 M d128 and PubMed 10 M d768, from the Hub.
LAION is not on the Hub (it waits for the user's release decision), and rebuilding 10 M from Re-LAION means a download plus
about 20 min of encoding, so PubMed 10 M is the large scale point in every check. Each check uses the configs' first 10,000
queries (`users_limit`).

**Timing protocol.** The harness's: the fixed-seed query pool (`inputs.query_pool`, seed 0), the query side prepared
outside the timed call (`run.prepare_pool`), graph = `measure.graph_callable` (one `cudaGraphLaunch`, zero cudagraph skips),
and `measure.latency_group`'s three windows with the sampled `sm_mhz` and the `unstable` flag (spread > 5 %). The arms of a
variant are timed round-robin in one group.

## Check 1: LiNR's native PyTorch, as LiNR §6 states it

LiNR §6: "Native boolean masking and indexing cause a 100X latency increase". Our own native reference is sync-free, so it
does not test the claim. The script builds the literal forms in eager PyTorch:

- `native_index`: the clause mask by broadcast compare (LiNR §3.1 semantics, `-1` inactive), then per query (LiNR §5.3
  splits a batch per query) `mask[b].nonzero()` (host sync, data-dependent shape), `items[idx]` (a copy), an fp16 matmul on
  the gathered rows, `topk`, and the ids mapped back. A batch therefore costs B host syncs.
- `native_mask`: the same mask, a dense fp16 matmul, failing scores set to `-inf`, then `topk`. This is V1's similarity
  masking in native ops, and it is sync-free.

These are timed against `LiNRV1` / `LiNRV2` (triton backend) in graph replay, as in the final run, and in eager. Each cell
is one synthetic clause (draw 0, `u_i < p`) at p ∈ {0.001, 0.01, 0.1, 0.11, 0.5}, where 0.11 is LiNR's high-pass-rate set
(1.7 M of 15.5 M), with bs 1 and 16 and k 100. All arms read the same fp16 table, the LiNR modules' own buffer.
**Deviation:** the native arms use 3 warm-up calls and windows of at least 10 calls (`target_s` 1 s; the harness's minimum
is 1,000), because a native call reaches 0.4 s.

Each native arm returns the same top-100 as V2: overlap 0.988–1.000 on 4 probe batches. The rest is the native arms' fp16
score rounding; V1 and V2 agree at 1.000.

Medians in ms. "/ best graph" = the native median over the faster of V1 and V2 in graph replay. "unstable" names the arms
whose window spread exceeded 5 % (3 of 30 cells). Most windows ran at 1380–1410 MHz. Some dropped to 1140–1215 MHz:
goodreads 0.11 bs 16 (V1 and V2 eager), pubmed 0.001 bs 1 (V2 graph and eager) and pubmed 0.001 bs 16 (V1 eager, native
index). The ratios in those cells carry that throttle.

| dataset | pass | bs | V1 graph | V2 graph | V1 eager | V2 eager | native mask | native index | mask / best graph | index / best graph | index / best eager | overlap | unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | 0.001 | 1 | 0.294 | 0.171 | 0.385 | 0.547 | 0.516 | 0.289 | 3.0 | 1.7 | 0.8 | 0.9975 | no |
| goodreads | 0.001 | 16 | 0.467 | 0.490 | 0.552 | 0.549 | 0.995 | 2.382 | 2.1 | 5.1 | 4.3 | 0.9997 | no |
| goodreads | 0.01 | 1 | 0.293 | 0.172 | 0.386 | 0.549 | 0.515 | 0.297 | 3.0 | 1.7 | 0.8 | 1.0000 | no |
| goodreads | 0.01 | 16 | 0.468 | 0.502 | 0.555 | 0.561 | 0.995 | 2.757 | 2.1 | 5.9 | 5.0 | 0.9997 | no |
| goodreads | 0.1 | 1 | 0.291 | 0.182 | 0.388 | 0.556 | 0.514 | 0.395 | 2.8 | 2.2 | 1.0 | 1.0000 | no |
| goodreads | 0.1 | 16 | 0.470 | 0.587 | 0.555 | 0.645 | 0.995 | 4.086 | 2.1 | 8.7 | 7.4 | 0.9988 | no |
| goodreads | 0.11 | 1 | 0.291 | 0.185 | 0.390 | 0.558 | 0.517 | 0.399 | 2.8 | 2.2 | 1.0 | 0.9975 | no |
| goodreads | 0.11 | 16 | 0.471 | 0.595 | 0.581 | 0.771 | 1.000 | 4.432 | 2.1 | 9.4 | 7.6 | 0.9986 | triton_v1/eager, triton_v2/eager |
| goodreads | 0.5 | 1 | 0.291 | 0.239 | 0.385 | 0.551 | 0.515 | 0.690 | 2.2 | 2.9 | 1.8 | 1.0000 | no |
| goodreads | 0.5 | 16 | 0.471 | 0.938 | 0.555 | 0.997 | 0.998 | 10.544 | 2.1 | 22.4 | 19.0 | 0.9995 | no |
| arxiv | 0.001 | 1 | 0.735 | 0.227 | 0.765 | 0.544 | 2.169 | 0.295 | 9.5 | 1.3 | 0.5 | 0.9950 | no |
| arxiv | 0.001 | 16 | 1.452 | 1.554 | 1.544 | 1.615 | 3.380 | 3.088 | 2.3 | 2.1 | 2.0 | 0.9964 | no |
| arxiv | 0.01 | 1 | 0.733 | 0.235 | 0.763 | 0.549 | 2.167 | 0.392 | 9.2 | 1.7 | 0.7 | 0.9950 | no |
| arxiv | 0.01 | 16 | 1.453 | 1.591 | 1.549 | 1.649 | 3.379 | 4.483 | 2.3 | 3.1 | 2.9 | 0.9934 | no |
| arxiv | 0.1 | 1 | 0.735 | 0.280 | 0.766 | 0.552 | 2.170 | 0.602 | 7.7 | 2.1 | 1.1 | 0.9925 | no |
| arxiv | 0.1 | 16 | 1.453 | 1.857 | 1.547 | 1.920 | 3.385 | 9.217 | 2.3 | 6.3 | 6.0 | 0.9920 | no |
| arxiv | 0.11 | 1 | 0.735 | 0.287 | 0.765 | 0.554 | 2.167 | 0.637 | 7.6 | 2.2 | 1.2 | 0.9950 | no |
| arxiv | 0.11 | 16 | 1.454 | 1.888 | 1.548 | 1.948 | 3.384 | 9.818 | 2.3 | 6.8 | 6.3 | 0.9931 | no |
| arxiv | 0.5 | 1 | 0.735 | 0.480 | 0.768 | 0.559 | 2.170 | 1.917 | 4.5 | 4.0 | 3.4 | 0.9925 | no |
| arxiv | 0.5 | 16 | 1.453 | 3.305 | 1.547 | 3.376 | 3.384 | 30.187 | 2.3 | 20.8 | 19.5 | 0.9919 | no |
| pubmed | 0.001 | 1 | 9.240 | 0.567 | 9.321 | 0.737 | 9.558 | 0.445 | 16.9 | 0.8 | 0.6 | 0.9975 | triton_v2/graph, native_index/eager |
| pubmed | 0.001 | 16 | 11.584 | 4.666 | 11.719 | 4.723 | 18.762 | 7.809 | 4.0 | 1.7 | 1.7 | 0.9980 | native_index/eager |
| pubmed | 0.01 | 1 | 9.239 | 0.593 | 9.321 | 0.640 | 9.561 | 0.663 | 16.1 | 1.1 | 1.0 | 1.0000 | no |
| pubmed | 0.01 | 16 | 11.595 | 5.646 | 11.719 | 5.695 | 18.770 | 10.361 | 3.3 | 1.8 | 1.8 | 0.9956 | no |
| pubmed | 0.1 | 1 | 9.243 | 1.441 | 9.323 | 1.482 | 9.565 | 3.201 | 6.6 | 2.2 | 2.2 | 1.0000 | no |
| pubmed | 0.1 | 16 | 11.584 | 18.273 | 11.708 | 18.317 | 18.767 | 50.995 | 1.6 | 4.4 | 4.4 | 0.9953 | no |
| pubmed | 0.11 | 1 | 9.244 | 1.538 | 9.319 | 1.579 | 9.563 | 3.540 | 6.2 | 2.3 | 2.2 | 0.9950 | no |
| pubmed | 0.11 | 16 | 11.590 | 19.680 | 11.713 | 19.731 | 18.780 | 56.565 | 1.6 | 4.9 | 4.8 | 0.9962 | no |
| pubmed | 0.5 | 1 | 9.245 | 5.222 | 9.322 | 5.262 | 9.567 | 24.742 | 1.8 | 4.7 | 4.7 | 0.9875 | no |
| pubmed | 0.5 | 16 | 11.592 | 75.272 | 11.720 | 74.486 | 18.765 | 396.114 | 1.6 | 34.2 | 33.8 | 0.9916 | no |

**What it says.** Against our custom kernels in graph replay, the literal native form costs **0.8–34×**:
- **The 100× is not reproduced on any cell.**
- The index form (nonzero, gather, matmul) grows with pass rate and batch: 0.8–4.7× at bs 1 and 1.7–34× at bs 16. The top,
  34×, is PubMed 10 M d768 at 50 % and bs 16, where 16 host syncs each copy 5 M × 768 fp16 rows.
- The masked form is 1.8–17× at bs 1 and 1.6–4.0× at bs 16. Its worst case is a low pass rate at bs 1 (PubMed 16.9×), where
  V2 reads only the passing rows and the masked form scans the whole catalog.
- At LiNR's ~11 % point the index form costs 2.2–2.3× at bs 1 and 4.9–9.4× at bs 16 across the three datasets. The masked
  form costs 1.6–7.6×.
- At bs 1 and up to 11 %, the index form runs at 0.5–1.2× our *eager* V2 on Goodreads and arXiv. One host sync costs less
  than V2's fixed eager launches.

The paper can say the claim was tested as stated, in eager PyTorch 2.10 on an A100, and that we measure up to 34×, not 100×.
LiNR's setup (V100 / A100, TF and PyTorch versions of 2023, 15.5 M items, a serving stack) is not ours, so this neither
confirms nor refutes their number.

## Check 2: LiNR V3 at 512 bits

Not run yet at this commit.

## Check 3: Meta's bloom with NOT

Not run yet at this commit.
