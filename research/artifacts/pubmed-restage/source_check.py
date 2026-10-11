"""Pubmed restage: the NCBI MedCPT source today against the E2 plan of 2026-09-26
(plan-10m.json from the Hub's artifacts/e2-pubmed): per shard, the three files' sizes and
the npy row count. Two connections at a time (plan's 16-way HEAD burst draws HTTP 503).

    python source_check.py OLD_PLAN_JSON OUT_JSON
"""

import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings"
NAMES = {
    "pmids": "pmids_chunk_{}.json",
    "content": "pubmed_chunk_{}.json",
    "embeds": "embeds_chunk_{}.npy",
}


def size(url):
    with urllib.request.urlopen(
        urllib.request.Request(url, method="HEAD"), timeout=60
    ) as r:
        return int(r.headers["Content-Length"])


def rows(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-255"})
    with urllib.request.urlopen(req, timeout=60) as r:
        m = re.search(rb"'shape':\s*\((\d+),\s*(\d+)\)", r.read(256))
    assert int(m.group(2)) == 768
    return int(m.group(1))


def shard(i):
    out = {k: size(f"{BASE}/{n.format(i)}") for k, n in NAMES.items()}
    out["rows"] = rows(f"{BASE}/{NAMES['embeds'].format(i)}")
    return str(i), out


old = json.load(open(sys.argv[1]))["per_shard"]
with ThreadPoolExecutor(2) as ex:
    new = dict(ex.map(shard, range(38)))
json.dump(new, open(sys.argv[2], "w"), indent=1)
diff = {i: (old[i], new[i]) for i in old if old[i] != new[i]}
print(
    f"shards {len(new)}  rows {sum(s['rows'] for s in new.values()):,}  "
    f"bytes {sum(v for s in new.values() for k, v in s.items() if k != 'rows'):,}  "
    f"differing from the E2 plan: {len(diff)}"
)
for i, (a, b) in diff.items():
    print(i, a, b)
