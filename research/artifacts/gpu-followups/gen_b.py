"""Run b's suites: the LiNR arms (V1, V2, and V3 at its picked pools) of the final run's real-filter
head-to-head suites (`fr-timed-bloom-d{128,256,768}` of the picks' suites-timed.yaml), k 100, bs 1 / 16 / 64,
seeds 1 and 2, interleaved by algo as in the final run. SilverTorch arms dropped.

usage: gen_b.py SUITES_TIMED.yaml OUT_SUITES.yaml"""

import sys

import yaml

src = yaml.safe_load(open(sys.argv[1]))
out = {}
for dim in (128, 256, 768):
    s = dict(src[f"fr-timed-bloom-d{dim}"])
    s.update(
        filter_kinds=["clause"],
        ks=[100],
        batch_sizes=[1, 16, 64],
        seeds=[1, 2],
        interleave=[{"by": ["algo"]}],
    )
    s["arms"] = [a for a in s["arms"] if a["algo"] != "silvertorch"]
    for a in s["arms"]:
        a["filter_kinds"] = ["clause"]
    out[f"gf-b-d{dim}"] = s
yaml.safe_dump(out, open(sys.argv[2], "w"), sort_keys=False)
for k, s in out.items():
    print(
        k, s["datasets"], {d: v for d, v in s["sweeps"].items()}, len(s["arms"]), "arms"
    )
