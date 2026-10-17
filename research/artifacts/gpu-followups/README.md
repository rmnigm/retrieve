# gpu-followups: the head-to-head under the current code

Follow-up GPU runs on one A100-SXM4-80GB (pod a100-x1-g) after SilverTorch's IVF moved to inner product end to
end (k-means assignment and probe, `61045809` on staging). Session scripts driving the harness; no gate, **nothing
here is citable** ([validation](../../validation.md#gpu-follow-ups-not-citable)). Raw outputs are on the Hub under
`artifacts/gpu-followups/<run>/` ([hub-index](../hub-index.md)). Yambda is out of the study and in no run.

Run from a checkout's `evaluation/` with `evaluation/data` pointing at the data root; [`chain.sh`](chain.sh) is the
one sequential GPU queue the runs went through.

| file | what |
|---|---|
| [`gen_a.py`](gen_a.py) | run a's suites from the final run's records and `head_to_head_selections.csv`: per (dataset, filter) the selected SilverTorch points, or the best-recall one where the target was not reached, plus n_probe 24 |
| [`run_a.sh`](run_a.sh) | run a: `bench run` per dataset on those suites |
| [`compare_a.py`](compare_a.py) | per point recall / latency against the records; per workload and target the head-to-head rule over the re-run points against the recorded best LiNR, and whether the verdict flips |
| [`gen_a_deeper.py`](gen_a_deeper.py) | run a's second pass: deeper probes (1.5×, 2×, 4×, within n_lists / 4) where a target the final run reached is no longer met |
| [`upload.py`](upload.py) | one commit per run to `pinkmeme/eval-results` with a `MANIFEST.json`, read back and checked |

## Run a

The final run's real-filter head-to-head selections at k 100, bs 1 / 16 (Goodreads `c0_genre`; arXiv `all4`,
`c0_maincat`, `c3_nversions`; PubMed `c0_mesh`; LAION 1M-30M `c0_domain`, `tags4`) and two synthetic rates each
(Goodreads, arXiv, PubMed 1 % / 10 %; LAION 30M 10 % / 50 %): 128 SilverTorch points, seed 0, graph timing. The
LAION 65,536-list points were left out (another pod's targeted rerun measures them). LAION reads the rebuilt inputs
([validation](../../validation.md#datasets)).

**Verdicts** (`compare/final-verdicts.csv`): 0 flips in 84 workload × target pairs; 58 reached in both runs, all
with SilverTorch faster than the recorded best LiNR (ratio ≤ 0.79); 24 reached in neither; 2 covered by the
other pod. The ratios divide a time measured here by LiNR times measured on the final run's pods; run b's
same-box LiNR timings give the same-box view.
