---
chain: "seqrec-encoder"
branch: "w13-hub"
parent: "2026-09-27-020041987-w13-brief-hub-upload.md"
nextStep: "w13: finish the 7 checkpoint uploads, then the data uploads below (all private), verify each, commit the docs on dev/hstu-hub, write the w13 report. The VM is deleted after that."
created: "2026-09-26T23:02:20Z"
---

# W13 instruction: the VM will be deleted, so upload the data that exists only here

User, 2026-09-27: "push everything useful into the branch in gh and hf hub; i'll delete this vm". After the 7
checkpoints (unchanged), upload these, with a **dry run first and then for real**. **Private** for anything new;
**never `--public`**. Report visibility for existing repos.

1. **KuaiRand eval inputs** (not on the Hub at all): `eval-data publish kuairand` with `--private`, from
   `/data/kuairand` (the harness layout; `bench check` passed on it). Check first what `publish` includes; the
   attrs files are about 7 GB.
2. **Trainer inputs,** each under a `trainer/` path in its dataset's Hub repo:
   - goodreads-work-id (`/data/goodreads-work-id/trainer/`: train, val and test parquet, item_id_map.json,
     prep_log.json, book_to_work.parquet);
   - yambda-500m (`/data/yambda-500m/trainer/`, the re-prep with timestamps);
   - kuairand (`/data/kuairand/{train,val,test}.parquet` if `publish` did not already include them).

   **Before uploading, check if `hub.py` has a helper for this.** If it does not, use `huggingface_hub`
   `upload_folder` with an explicit `path_in_repo="trainer"`. If an existing repo is **public**, stop before
   uploading trainer data to it and report. Derived goodreads data going public is the user's call.
3. **Not uploaded:** the raw downloads (goodreads UCSD has an academic-use licence; KuaiRand raw is re-downloadable
   from Zenodo), the train-only k64, and the HSTU checkpoints.
4. **Docs, in the same dev/hstu-hub commit or a second one:**
   - `docs/system/checkpoints.md`: the 7 rows;
   - `docs/system/datasets.md`: one line per dataset saying the trainer inputs are on the Hub under `trainer/`, and how to
     fetch them (the command or `hf_hub_download` path); KuaiRand eval inputs are now on the Hub (private).
   - The link checker must be at zero.
5. **Report:** per upload, the repo, path, visibility, bytes, and the verified remote file list.
