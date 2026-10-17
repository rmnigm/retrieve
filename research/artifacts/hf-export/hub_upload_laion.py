from huggingface_hub import HfApi
REPO = "filtered-retrieval-repro/filtered-retrieval-inputs"
api = HfApi(token=open("/data/hf-anon/token").read().strip())
who = api.whoami()["name"]
print("whoami:", who)
assert who == "filtered-retrieval-repro", who
api.upload_large_folder(repo_id=REPO, folder_path="/data/anon-public", repo_type="dataset",
                        allow_patterns=["eval-laion30m/**", "README.md", "LICENSES.md", "MANIFEST.json"], num_workers=16)
print("refs:", api.list_repo_refs(REPO, repo_type="dataset").branches[0].target_commit)
