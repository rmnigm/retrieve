"""IVF-TUNE: write fit.py's values into evaluation/config/suites.yaml: every `filter`
SilverTorch arm gets its dataset's n_lists (build), the triton and official-bloom arms n_probe
{24, n95}; every `synth` SilverTorch arm its synth dataset's n_lists (the explicit n_probe
sweeps stay); the `n95` suite goes. Exact-text replacements, so a drifted file fails loudly.

    python fit.py ... > fit.json && python apply.py fit.json evaluation/config/suites.yaml
"""

import json
import re
import sys
from pathlib import Path

v = json.load(open(sys.argv[1]))["values"]
p = Path(sys.argv[2])
s = p.read_text()


def sub(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


def nl(ds):
    return v[ds]["n_lists"]


def probes(ds):
    return sorted({24, v[ds]["n95"]})


def slot(ds, ind, query=True):
    if ds not in v:  # part A: PubMed not yet tuned, its slot stays open
        return f"{ind}{ds}: {{}}   # IVF-TUNE PubMed pending\n"
    q = f", query: {{n_probe: [{', '.join(map(str, probes(ds)))}]}}" if query else ""
    return f"{ind}{ds}: {{build: {{n_lists: [{nl(ds)}]}}{q}}}\n"


FILTER = ("goodreads", "arxiv", "yfcc10m", "pubmed")
ind = " " * 8
tuned = "".join(slot(d, ind) for d in FILTER)
for backend, extra in (("triton", ""), ("official", "      filter_kinds: [bloom]\n")):
    sub(
        f"    - algo: silvertorch\n      backends: [{backend}]\n{extra}"
        "      query: {n_probe: [24]}"
        + (
            "                                  # 24 = the paper's"
            if backend == "triton"
            else ""
        )
        + "\n      datasets: {goodreads: {}, arxiv: {}, yfcc10m: {}, pubmed: {}}   # n95 slots: [24, n95]\n",
        f"    - algo: silvertorch\n      backends: [{backend}]\n{extra}"
        "      query: {n_probe: [24]}                                  # 24 = the paper's; n95 per dataset\n"
        "      datasets:                                             # IVF-TUNE: n_lists by size, n_probe {24, n95}\n"
        + tuned,
    )
torch_slots = (
    "{"
    + ", ".join(
        f"{d}: {{build: {{n_lists: [{nl(d)}]}}}}" for d in ("goodreads", "arxiv")
    )
    + "}"
)
old = "      query: {n_probe: [24]}\n      datasets: {goodreads: {}, arxiv: {}}\n"
assert s.count(old) == 2, "torch arms"
s = s.replace(
    old,
    f"      query: {{n_probe: [24]}}\n      datasets: {torch_slots}   # the triton arm's index\n",
)

SYN = {
    "goodreads-synth": "goodreads",
    "arxiv-synth": "arxiv",
    "yfcc10m-synth": "yfcc10m",
}
sub(
    "      query: {n_probe: [24, 64, 128, 256, 512, 1024]}       # an explicit sweep, not n95 (user);\n"
    "      datasets:                                             # synth n_lists is 1024: none capped\n"
    "        goodreads-synth: {}\n        arxiv-synth: {}\n"
    "        yfcc10m-synth: {query: {n_probe: [24, 256, 1024]}}\n",
    "      query: {n_probe: [24, 64, 128, 256, 512, 1024]}       # an explicit sweep, not n95 (user)\n"
    "      datasets:                                             # n_lists from IVF-TUNE (real dataset's)\n"
    f"        goodreads-synth: {{build: {{n_lists: [{nl('goodreads')}]}}}}\n"
    f"        arxiv-synth: {{build: {{n_lists: [{nl('arxiv')}]}}}}\n"
    f"        yfcc10m-synth: {{build: {{n_lists: [{nl('yfcc10m')}]}}, query: {{n_probe: [24, 256, 1024]}}}}\n",
)
syn_slots = (
    "{"
    + ", ".join(f"{k}: {{build: {{n_lists: [{nl(d)}]}}}}" for k, d in SYN.items())
    + "}"
)
for backend in ("triton", "official"):
    sub(
        f"    - {{algo: silvertorch, backends: [{backend}], filter_kinds: [bloom], query: {{n_probe: [24, 256]}}}}\n",
        f"    - algo: silvertorch\n      backends: [{backend}]\n      filter_kinds: [bloom]\n"
        f"      query: {{n_probe: [24, 256]}}\n      datasets: {syn_slots}\n",
    )
s2 = re.sub(r"\nn95:  .*?\n\n(?=codesign:)", "\n\n", s, flags=re.S)
assert s2 != s, "n95 suite not found"
p.write_text(s2)
