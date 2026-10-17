import json, sys
from huggingface_hub import HfApi
repo = "filtered-retrieval-repro/filtered-retrieval-inputs"
man = json.load(open(sys.argv[1]))
want = {f["path"]: f for f in man["files"]}
api = HfApi(token=False)
got = {}
for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True, expand=True):
    if hasattr(e, "size"):
        got[e.path] = e
extra = sorted(set(got) - set(want) - {"MANIFEST.json"})
missing = sorted(set(want) - set(got))
bad = []
for p, f in want.items():
    e = got.get(p)
    if e is None:
        continue
    if e.size != f["bytes"]:
        bad.append((p, "size"))
    elif e.lfs is not None and e.lfs.sha256 != f["sha256"]:
        bad.append((p, "sha256"))
lfs = sum(1 for p in want if p in got and got[p].lfs is not None)
print(f"remote files {len(got)}, manifest {len(want)}, missing {len(missing)}, extra {len(extra)}, mismatched {len(bad)}, lfs-sha-checked {lfs}")
print("extra:", extra[:10]); print("missing:", missing[:10]); print("bad:", bad[:10])
print("test.parquet present:", any(p.endswith("eval-goodreads-work-id/test.parquet") for p in got))
