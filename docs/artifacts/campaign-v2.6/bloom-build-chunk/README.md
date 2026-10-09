# BLOOM-BUILD-CHUNK — the triton bloom build fits at 30 M

Roadmap BLOOM-BUILD-CHUNK (campaign-v2.7 bundle). Current state: [validation](../../../validation.md) row
*BLOOM-BUILD-CHUNK*; mechanism: [kernels § SilverTorch kernels](../../../system/kernels.md#silvertorch-kernels)
("Bloom build"). A100-SXM4-80GB, pod b. Raw outputs: Hub `artifacts/bloom-build-chunk/` ([hub-index](../../hub-index.md)).

## Mechanism
`bloom_hash.build_transposed_sigs` materialised, per bloom word, `[64, N_pad]` int64 bit planes and a shifted copy:
14.3 GiB each at 30 M. It also made a full padded copy of the row-wise table. Our triton bloom build OOMed at LAION 30 M
(d-run, v2.6). It now works per word **and** per 2¹⁸-item chunk, with the last chunk padded. Each element goes through
the same arithmetic.

## Gate ([`bb_gate.py`](bb_gate.py), against staging a9748da's one-pass build copied verbatim)
`torch.equal` on all three tables, with peak CUDA memory above the input:

| table | chunked peak | one-pass peak |
|---|---|---|
| goodreads 0.8 M × 16 words | 482 MiB | 1,366 MiB |
| arXiv 3.0 M × 16 words | 749 MiB | 5,110 MiB |
| random 30 M × 16 words | 4,046 MiB (the 3.7 GiB output + ~256 MiB) | 51,270 MiB |

Library suite 821 passed on a fresh inductor dir, including `test_build_transposed_sigs_chunks_equal_one_pass`.
