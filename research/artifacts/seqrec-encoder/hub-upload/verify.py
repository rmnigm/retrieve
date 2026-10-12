import os, sys
from huggingface_hub import HfApi
repo, prefix, local = sys.argv[1:4]
i = HfApi().dataset_info(repo, files_metadata=True)
fs = [s for s in i.siblings if s.rfilename.startswith(prefix) and "/" not in s.rfilename[len(prefix):]]
print(f"{repo} {prefix or '<root>'} private={i.private} files={len(fs)} bytes={sum(s.size for s in fs)} used_storage={i.used_storage}")
for s in fs:
    n = s.rfilename[len(prefix):]
    lp = os.path.join(local, n)
    print(f"  {n} {s.size} {'OK' if os.path.exists(lp) and os.path.getsize(lp)==s.size else 'MISMATCH/NOLOCAL'}")
