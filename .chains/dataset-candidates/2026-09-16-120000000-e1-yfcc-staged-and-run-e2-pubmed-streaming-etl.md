---
chain: "dataset-candidates"
branch: "main"
parent: "2026-09-06-120000000-e1-yfcc-ingested-gt-gate-and-layout-contract-stub.md"
nextStep: "Decide the YFCC exact-algo gate (a: fp32 scoring for exact algos, a library change; b: evaluate YFCC at k <= 100; c: exempt YFCC from the gate and report fp16 recall as is); run E1's GPU cell; run the E2 10 M-slice PubMed ETL (~2.5-3 h, network-bound)."
created: "2026-09-16T12:00:00Z"
---

# §7 records: E1 YFCC staged, checked and run (CPU); E2 PubMed streaming ETL, dry run on one shard

## §7.1 E1, 2026-09-16
Branch `dev/e1-e2-datasets` (worktree `/scratch/wt/datasets`, off `development` @ `d808a66`), A100 box, GPU not used (D1-a held `/workspace/gpu.lock` for its whole stage: the lock wraps the stage driver, not a cell), so the cells are CPU rehearsals. Artifacts `docs/artifacts/dataset-candidates/yfcc10m/`.

1. Staged to `/data`: `eval-data yfcc download` (2.97 GB, 2 min, ~19 MB/s); `convert --sha256`, `prep` (35.7 s), `attrs --max-tags 32` (796 s, see surprise 1): `/data/yfcc10m` 9.5 GB (`text_emb.pt` 3.84 GB, `item_attrs_narrow.pt` 5.12 GB, `item_tags_csr.pt` 0.51 GB, `item_id_map.json` 0.18 GB, `gt_shipped.pt` 0.13 GB), 12.5 GB with the raw mirror. `prep_log.json` equals the 2026-09-06 record: 7,910 queried tags, 98.51 % of items uncapped at K=32, 74,214 / 100,000 queries keep their whole shipped-GT row, 84.86 % of GT entries survive the cap.
2. Layout contract: `bench check --dataset yfcc10m` reported the sidecars missing (the loader wrote `emb_provenance.json` to dodge the nomic prefix assertion). Fix is a policy: `layout.prefix_problem` accepts `"prefix": null`, rejects a sidecar without the key; loader and check share it; `yfcc prep` writes both `*.meta.json` with `prefix: null` + provenance. `bench check` -> `yfcc10m d192: ok`. Tests in `test_layout.py` and a new `test_yfcc.py::TestPrepAttrsLayout`.
3. Config: `yfcc10m` joined the `filter` suite (`dims: [128, 192]`; 192 is its only dim, not an ablation); one clause sweep `tags_and = [0, 1]`, no bloom (false positives would defeat the shipped-GT cross-check). No `none` cell in any committed suite (the `quality` suite retired 2026-09-16; unfiltered cells return with E5).
4. Cells on the CPU (`CUDA_VISIBLE_DEVICES=""`, 16 threads, backend torch, `users_limit: 1000`, `--skip-perf --mode eager`, after a full attempt was stopped at 1 h 45 with the oracle still building):
   - `filter`: `linr_v1_filter_mask / torch / clause / tags_and` -> `status: failed`, `QualityGateError: recall_oracle@1000 = 0.9641 < 0.99` (oracle 15 min: pass rate 0.0182, median pass set 15,060 items; quality 1 h 20). Diagnosis (`fp16-gate-diagnostic.py`, `.log`): of 35,273 oracle top-1000 entries the module missed over 1,000 queries, 7,212 are exact ties at the module's 1000th score and 28,061 score strictly higher but never by more than 4.6e-4 cosine. Against an fp64-exact top-1000 over the true capped pass set (49 rows with >= 1,000 survivors): oracle blob 0.9998, plain fp32 top-k 0.9999, module 0.9211. Cause: `PostfilterKNN` stores items fp16 and scores `query.half() @ items_t` (backend-independent). fp16 spacing in [0.5, 1) is 2^-11 = 4.88e-4; YFCC's fp64 rank-1-to-1000 span has median 0.0072 (min 0.00004), ~15 fp16 quanta for a thousand items, plus 4.96 % of the catalog exact duplicate vectors. On goodreads the same gate sits at 0.9996. The GPU cell will fail the same way at k=1000. Choices for the orchestrator: (a) score exact algos in fp32 on this dataset (library change, frozen during the campaign, doubles linr_v1's item bytes); (b) evaluate YFCC at k <= 100 (not measured); (c) exempt YFCC from the exact-algo gate and report fp16 recall as is.
   - `none_e1` (scratch suite): `linr_v1_filter_mask / torch / none / full_scan` -> `partial`, held-out recall@100/500/1000 = 0.998, mrr@100 0.812, build 4.7 s, index 3,662 MiB, 1,952 s CPU; the 0.2 % misses are the cosine vs squared-L2 deviation.
   Both records `env.gpu = "cpu"`, not paper material. GPU command once the lock is free: `flock /workspace/gpu.lock -c '/venvs/retrieve/bin/python -m bench.cli run --dataset yfcc10m --dim 192 --suite filter --algo silvertorch --backend triton --filter-kind clause'`.

Gates: ruff clean except one pre-existing E501 in `bench/records.py:46`; harness suite 197 passed (185 / 4 before; +8 new, and the 4 `TestRealSlice` tests now run against `/data/yfcc10m`); links 0.

Surprises: (1) `/venvs/retrieve/bin/bench` and `eval-data` carried the shebang `#!/workspace/retrieve/.venv/bin/python3`, so `uv run --no-sync bench ...` imported torch and numpy from the MooseFS network volume (52 MB/s, FUSE page faults): why `yfcc attrs` took 796 s vs 20.5 s (9 s user, 240 s system); workaround `/venvs/retrieve/bin/python -m bench.cli ...` with `PYTHONPATH=<worktree>/evaluation` (the venv's editable `.pth` points at `/workspace/retrieve/evaluation`); the fix is a venv re-provision. (2) `evaluation/data` is per worktree: `data_dir: data/<name>` resolves against `evaluation/`, so a fresh worktree needs `ln -s /data evaluation/data`. (3) The container's CPU quota is 27.2 cores (`cpu.max`), not the 255 `nproc` reports.

## §7.2 E2, 2026-09-16
Code, tests and a dry run on `dev/e1-e2-datasets`; no bulk download, nothing staged, no oracle, no cell. PCA plan superseded (native 768-d). Artifacts `docs/artifacts/dataset-candidates/pubmed/`.

Budget from the server (`eval-data pubmed plan` HEADs all 114 MedCPT files and reads each npy header via Range):
| | full catalog | `--keep-items 10000000` |
|---|---|---|
| articles (38 disjoint PMID ranges, 0 duplicates) | 35,920,666 | 10,000,000 |
| MedCPT raw (npy 110.35 GB + JSON 52.89 GB + PMID lists 0.42 GB) | 163.66 GB | 163.66 GB |
| MEDLINE baseline (1334 files) | 54.27 GB | 54.27 GB |
| raw to download | 217.9 GB | 217.9 GB |
| fp16 item shards | 55.17 GB | 15.36 GB |
| article parquet (55.0 B/row measured) | 2.16 GB | 0.60 GB |
| MEDLINE parquet (bound) | 1.20 GB | 1.20 GB |
| raw in flight (1 + prefetch largest shards, 31 + 34) | 9.61 GB | 9.61 GB |
| peak disk | 68.6 GB | 27.2 GB |
| wall, download-bound | 2.54 h at 23.9 MB/s; 1.5 h at 41 MB/s | same |
| items fp32 on the device | 110.3 GB | 30.7 GB |
Corrections to the handed numbers: raw is 163.7 GB of MedCPT, not 146; the processed tensor is 55 GB fp16, not 111 (every text dataset stores fp16 on disk; the harness makes the fp32 copy on the device).

Streaming path (`etl/pubmed.py` rewritten): `download --what pmids` (0.42 GB) fixes `item_id_map.json` (1-indexed dense in shard order, ascending PMID; ranges disjoint, checked as `pmid_ranges_disjoint`); `convert --fetch --prefetch 1 --delete-raw` walks shards (parse chunk JSON to `staging/articles_chunk_i.parquet` keeping the title; gather kept npy rows, L2-normalise, fp16, `content_d768/text_emb_shard_NN.pt`; append `shard_index.json`; delete raw; resumable) into the sharded layout `layout.load_sharded` reads; `--keep-items N` = the N smallest seeded splitmix64 hashes of the PMID (order-free, uniform over 1781-2024, exact; every shard still scanned); `queries` keeps every held-out row (empty strings for untitled), NFCorpus to a separate parquet; `prefix: null` sidecars; `plan` is the dry run. Tests: 33 pubmed tests (selection, hash stability, streaming convert over two fixture shards, keep-items, budget arithmetic, `cmd_plan` with the network monkeypatched).

Dry run on chunk 37 (smallest: 1.17 GB npy + 0.62 GB JSON, 380,761 articles, PMIDs 37,000,000-37,384,379): `convert --shards 37 --fetch --prefetch 1 --delete-raw` in 222 s; output 558 MB (`text_emb_shard_37.pt` 584.9 MB, `articles_chunk_37.parquet` 20.9 MB = 55.0 B/row, `item_id_map.json` 6.7 MB); `attrs` 30 s (C0 MeSH coverage 37 % on this newest chunk; C1 = 0 without the MeSH descriptor file; C3 = 0 without the MEDLINE join; all expected); `queries` 2,000 titles (3 empty, kept); `encode_queries --device cpu` -> `query_emb.pt [2000, 768]`; `bench check --dataset pubmed` -> `pubmed d768: ok`. Dry-run directory deleted; PMID lists kept in `/data/_raw/pubmed`.

Target: the 10 M slice, not 36 M: the harness holds items fp32 on the device (110.3 GB full vs 30.7 GB at 10 M, + 7.7 GB int8 codes + 1.6 GB attrs), 10 M matches the papers' pool and YFCC's, and the harness as written takes ~15 M at 768-d at most; above that is a harness decision (fp16 items + chunked oracle).

The real run (~2.5-3 h, peak 27.2 GB): `pubmed plan --medline --keep-items 10000000`; `download --what mesh`; `convert --fetch --prefetch 1 --delete-raw --keep-items 10000000`; `medline --stream`; `attrs`; `queries`; `encode_queries --device cuda` under the lock; `bench check --dataset pubmed`; then add pubmed to a suite.

Not done: bulk download, MEDLINE join, oracle, cells; MeSH pass rates unmeasured; GPU `encode_queries` untested.
