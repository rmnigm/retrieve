# E0c: goodreads-work-id d256 bar re-scored on trainer/test.parquet

`rescore.py` (the E0 path: `load_model_for_eval` + `evaluate`, full catalog, batch 512), H100,
2026-09-26 19:13-19:15 UTC. Not yet validated, not citable.

gsasrec-d256-drop0.5-id: ndcg@10 **0.0354**, ndcg@100 0.0681, recall@10 0.0343, recall@100 **0.1472**,
coverage@10 0.1163 (`rescored.json`). It matches its stored `eval_quality.json` to 4 decimals, as
d64 and d128 did in E0, so the R-g256 bar 0.0354 / 0.1472 stands as a same-file bar. Its
`item_id_map.json` is byte-identical to d64's.
