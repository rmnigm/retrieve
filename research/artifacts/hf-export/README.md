# hf-export: the anonymous input and records archive

Brief `.chains/hf-export/2026-10-17-050500000-brief-hf-export.md` (GUIDELINE artifact issue A1). CPU only. The
archive was built on pod a100-x1-g at `/data/anon-export` (35.98 GB, 4,524 files, `MANIFEST.json` sha256
`c5f52f4737e38a30ba92423e47453c85da95f5a57612624fb869b81e81ea2703`) and **not uploaded**: it waits for the user's
anonymous Hub owner. The export-branch docs that describe it are on `dev/export-inputs`.

Rebuild, in order (each script names its inputs; `/data/work` was the scratch root):

1. `download.sh`: the private Hub copies (`pinkmeme/eval-{goodreads-work-id,arxiv-papers,pubmed@10d48d05,yambda-500m}`
   without the gSASRec checkpoints, `pinkmeme/eval-results` `fr/**`) into `/data/work/hub/`.
2. `assemble.sh`: hardlink the files the final run read into `/data/anon-export/eval-<dataset>/`.
3. `gen_synth.py <dir>|corr`: synthetic draws 0-2 and the correlated draw 0, with the export code.
4. `scrub_runs.py SRC_FR OUT_RUNS MAP_CSV`: the 106 runs under neutral names, scrubbed, `results.parquet` and
   `MANIFEST.json` rebuilt by the export `bench`; writes `run_name_map.csv` and `host_map.json` (here; never in the
   archive).
5. `write_ckpt_cards.py`: neutral checkpoint `README.md` / `config.json`; copy `archive-text/` into place.
6. `write_manifest.py /data/anon-export`, then `verify.py`, `scrub_grep.py /data/anon-export`.

Checks: `check_synth.py` (45 uniform pass counts vs records), `check_real.py <dataset>` (per-query pass counts of
the real filters vs the sidecars), `check_corr.py` / `corr_threads.py` (the correlated filter), `check_runs.py`
(every record / sample line equal outside the scrubbed fields, every npz byte-equal, every manifest sha256 true),
`survey.py` (string fields of the records), `scrub_grep.py` (the identifier scan). Results in
[validation](../../validation.md#datasets).
