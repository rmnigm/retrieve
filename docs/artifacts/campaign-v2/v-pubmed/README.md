# V-PUBMED: PubMed `filter` under the campaign-v2 grid

Roadmap Phase V, V-PUBMED, on pod `a100-x1-b` (1x A100-SXM4-80GB). Current state and caveats:
[validation](../../../validation.md#v-pubmed-pubmed-filter-under-the-new-grid). NOT CITABLE until D1-G.

| file | what |
|---|---|
| [`restage.sh`](restage.sh) | the 10 M slice from NCBI source: [pubmed-restage](../../pubmed-restage/README.md)'s recipe with this pod's NUMA node 1 (cores 96-191) and the shared venv |
| [`identity.sh`](identity.sh) | `encode_queries`, `bench check`, one exact V1 cell (clause `c0_mesh`, seed 0, `--skip-perf`) into a side file |
| [`compare_v1.py`](compare_v1.py) | that cell against `d1/pubmed`'s: every slice fact and every quality metric equal, or exit 1 |
| [`driver.sh`](driver.sh) | `driver.sh n95`: the quality-only `n95` suite; `driver.sh filter`: the `filter` campaign (`--interleave --timeout 48`), refusing unless `bench env` equals *evaluation/campaign.yaml*'s code_version |
| [`n95.py`](n95.py) | n95 from the `n95` records, per seed and on the seed mean, and whether the grid brackets 0.95 (`bench report`'s matched table emits no quality-only curve) |

Every GPU command runs as `flock /scratch/gpu0.lock taskset -c 0-95 …` (GPU 0 is shared on this pod).
`TORCHINDUCTOR_CACHE_DIR=/scratch/inductor/v-pubmed`, results tree `/scratch/campaign-v2/results`.

## Results

| leg | Hub subtree | cells | GPU-h | MANIFEST sha256 |
|---|---|---|---|---|
| identity (V1 `c0_mesh`, seed 0) | in `campaign-v2/pubmed-n95` `logs/identity/` | 1 `partial` | < 0.05 | — |
| `n95` (`all5`, `n_probe` 8-128, seeds 0-2) | `campaign-v2/pubmed-n95` | 15 `ok` | 0.51 | `b89eee7330ed894a9a0852a7d490484feb7386a191d256204b81697a596d7077` |

`n95.py` on the `n95` records (`recall_oracle@100`, pass rate 0.01809, 9,985 oracle rows):

| n_probe | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| 8 | 0.3343 | 0.3550 | 0.3420 | 0.3438 |
| 16 | 0.4280 | 0.4484 | 0.4342 | 0.4369 |
| 32 | 0.5273 | 0.5478 | 0.5355 | 0.5368 |
| 64 | 0.6337 | 0.6498 | 0.6413 | 0.6416 |
| 128 | 0.7414 | 0.7542 | 0.7486 | 0.7481 |

Not bracketed: n95 is above 128.

```bash
cd evaluation && RETRIEVE_DATA_ROOT=/data bash ../docs/artifacts/campaign-v2/v-pubmed/restage.sh
setsid nohup bash docs/artifacts/campaign-v2/v-pubmed/identity.sh > /scratch/v-pubmed/identity.log 2>&1 &
setsid nohup bash docs/artifacts/campaign-v2/v-pubmed/driver.sh n95 > /scratch/v-pubmed/driver-n95.log 2>&1 &
python docs/artifacts/campaign-v2/v-pubmed/n95.py /scratch/campaign-v2/results/n95/pubmed-d768.jsonl
```
