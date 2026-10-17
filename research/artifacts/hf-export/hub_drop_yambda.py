from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
REPO = "filtered-retrieval-repro/filtered-retrieval-inputs"
api = HfApi(token=open("/data/hf-anon/token").read().strip())
who = api.whoami()["name"]
print("whoami:", who)
assert who == "filtered-retrieval-repro", who
ops = [CommitOperationDelete("eval-yambda-500m/", is_folder=True),
       CommitOperationDelete("eval-results/runs/yambda-s0/", is_folder=True)]
ops += [CommitOperationAdd(p, f"/data/anon-public/{p}") for p in ("README.md", "LICENSES.md", "MANIFEST.json")]
info = api.create_commit(REPO, ops, repo_type="dataset", commit_message="Drop the Yambda dataset and its runs; four datasets remain")
print("commit:", info.oid)
