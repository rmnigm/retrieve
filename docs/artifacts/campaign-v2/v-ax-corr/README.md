# V-AX-CORR: cluster-correlated synthetic filter on arXiv

The cluster-correlated variant of the synthetic selectivity axis (IVF's best case, the opposite of
`arxiv-synth`'s uniform flags): `eval-data synth-filter --dataset arxiv --correlated` and the
`synth` suite on `arxiv-corr-synth` (d128, sweeps `c001` / `c003` / `c01`, arxiv-synth's arms and
`n_lists` 2048, seeds {0, 1, 2}; 99 jobs / 198 cells). Builder and files:
[datasets](../../../system/datasets.md#synthetic-selectivity-attrs). Gate and achieved pass rates:
[validation](../../../validation.md#harness-gates). Not citable until D1-G.

| file | what |
|---|---|
| [`PREDICTIONS.md`](PREDICTIONS.md) | the pre-registered predictions, committed before any GPU cell |
| [`driver.sh`](driver.sh) | `quality` (`bench check`, `bench oracle`, `bench campaign --skip-perf`) and `timed` (`--interleave`, re-runs the partial records) passes, own venv, code_version check, `nvidia-smi` at 1 Hz |

## Attrs build (CPU, done)

```bash
cd /scratch/wt/v-ax-corr/evaluation
OMP_NUM_THREADS=64 taskset -c 0-63 /venvs/v-ax-corr/bin/python -m eval_datasets.cli \
  synth-filter --dataset arxiv --correlated --rates 0.01,0.03,0.1 --seed 20261008
```

170 s on cores 0-63 (no GPU). A second build is sha256-identical:

| file | sha256 |
|---|---|
| `item_attrs_corr.pt` | `4025c823985429dfcfb0775545587a645e07d7000f2d56955d1d7486bf8917b5` |
| `query_attrs_corr.pt` | `bef4e61edcbd7f851049721b4bb0d1da53143e2ea6fe01deb70911d9099e88a1` |
| `synth_corr.json` | `036aca8a650c20aa7e47b8fb35a9f7bee1699118783822c999416ffd8ac92b49` |

Achieved pass rates (from `synth_corr.json`; a query's pass rate is its cluster's size over N =
2,988,996; 10,000 queries):

| p | k | mean | std | min | median | max | cluster size min / median / max |
|---|---|---|---|---|---|---|---|
| 0.01 | 100 | 0.0105 | 0.0021 | 0.0050 | 0.0107 | 0.0149 | 14,805 / 30,120 / 44,575 |
| 0.03 | 33 | 0.0325 | 0.0088 | 0.0182 | 0.0300 | 0.0540 | 54,532 / 88,340 / 161,473 |
| 0.1 | 10 | 0.1074 | 0.0271 | 0.0550 | 0.1052 | 0.1465 | 164,348 / 312,508 / 437,952 |

## GPU (pending the orchestrator's go)

```bash
setsid nohup flock /scratch/gpu0.lock bash docs/artifacts/campaign-v2/v-ax-corr/driver.sh quality \
  > /scratch/v21/arxiv-corr-synth-synth/driver-quality.log 2>&1 &
```
