# D3 PubMed: `bloomwidth` (quality) at campaign-v2

Roadmap D3, the PubMed piece, on pod `a100-x1-b` (1x A100-SXM4-80GB). State and caveats:
[validation](../../../validation.md#campaign-v2-phase-v-not-yet-validated). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`driver.sh`](driver.sh) | `bench oracle --suite bloomwidth`, the quality-only `bloomwidth` campaign, then `bloomwidth-timed` (`--interleave`); every GPU command under `flock /scratch/gpu0.lock` |
| [`summary.py`](summary.py) | FPR / recall / index size per width, and the default width against `d1/pubmed` (pod c's `d3-arxiv/summary.py` on `dev/d3-arxiv`, pointed at pubmed) |

**`bloomwidth-timed` did not run.** It moved to campaign-v2.1 (the quantize fix changes eager SilverTorch). The driver was
stopped after `bloomwidth` and killed (SIGKILL) before reaching the timed step. The calling chain's log line `d3 rc=0`
(`/scratch/v-pubmed/chain-after.log`, 21:24:52) is wrong on that point; `campaign.log` (`finished children=2 rc=0`) is the
record of what ran.

## Result (pubmed d768, bloom `c0_mesh`, seeds 0-2, k {100, 1000}, bs 16, `n_probe` 24, quality only)

42 / 42 `ok`. Hub `campaign-v2/pubmed-bloomwidth`, MANIFEST sha256
`a1257ee3b079729247083a7904294d5da992ca61c1c38c8a931720539cafe62b` (53 files: JSONL, per-query sidecars,
`results.parquet`, `logs/` with the campaign and driver logs, 1 s `clocks.csv`, `nvidia-smi -q -d CLOCK` captures,
`summary.txt`). GPU: oracle 57 s, triton 2,401 s, official 323 s = **0.77 GPU-h**.

Median over seeds (`bloom_fp_rate` is the Triton filter's: the records carry no official FPR):

| backend | m_bits | k_hash | bloom_fp_rate | recall_oracle@100 | @1000 | index_mib |
|---|---|---|---|---|---|---|
| triton | 64 | 3 / 5 | 2.40e-2 / 1.71e-2 | 0.4385 / 0.4646 | 0.3062 / 0.3284 | 7,556.1 |
| triton | 128 | 3 / 5 | 3.95e-3 / 1.09e-3 | 0.5874 / 0.6400 | 0.4094 / 0.4262 | 7,632.4 |
| triton | 256 | 3 / 5 | 5.37e-4 / 4.63e-5 | 0.6566 / 0.6690 | 0.4287 / 0.4306 | 7,785.0 |
| triton | 512 | 3 / 5 | 7.71e-5 / 1.52e-6 | 0.6680 / 0.6699 | 0.4305 / 0.4307 | 8,090.2 |
| triton | 1024 | 3 / 5 | 9.35e-6 / 5.60e-8 | 0.6697 / 0.6699 | 0.4307 / 0.4307 | 8,700.5 |
| triton | 2048 | 3 / 5 | 1.19e-6 / 1.41e-9 | 0.6698 / 0.6699 | 0.4307 / 0.4307 | 9,921.2 |
| official | (its own) | 3 / 5 | — | 0.6551 / 0.6698 | 0.4289 / 0.4307 | 7,765.6 / 7,956.1 |

Recall saturates by 512 bits at k_hash 5; official at k_hash 3 loses 0.015 recall@100 against Triton at the default width
(its bloom admits more false positives:
[validation, *Official vs Triton on bloom cells*](../../../validation.md#library-gates)). Recall@100 sits at 0.67 because
`c0_mesh` passes 0.02 % of items and `n_probe` 24 of 1024 lists finds few of them (the IVF limit `n95` measures).

Against `d1/pubmed` (`72e5a90`) at the default width, seed 0: recall@100 0.6640 and index_mib equal on both backends;
`bloom_fp_rate` 5.6025e-8 → 5.5966e-8 (−0.1 %) with the formula unchanged (`oracle.bloom_fp_rate`). The ratio 0.99895 is
close to 8,419 / 8,428, i.e. nine more kept rows with no false positive averaged in; not verified.

```bash
setsid nohup bash docs/artifacts/campaign-v2/d3-pubmed/driver.sh > /scratch/d3-pubmed/driver.log 2>&1 &
python docs/artifacts/campaign-v2/d3-pubmed/summary.py <tree with bloomwidth/> /scratch/v-pubmed/d1/filter/pubmed-d768.jsonl
```
