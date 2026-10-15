"""fr-timed-clause-lite-*: the picks' fr-timed-clause-* suites with only their SilverTorch
(official-fork, exact) arms, seed 0. Every point comes from the picks unchanged.

  python derive_lite.py /scratch/fr/picks.yaml clause-lite.yaml
"""

import sys

import yaml

picks = yaml.safe_load(open(sys.argv[1]))
lite = {}
for name, suite in picks.items():
    if not name.startswith("fr-timed-clause-"):
        continue
    s = dict(suite)
    s["arms"] = [a for a in suite["arms"] if a["algo"] == "silvertorch"]
    s["seeds"] = [0]
    lite[name.replace("fr-timed-clause-", "fr-timed-clause-lite-")] = s
yaml.safe_dump(lite, open(sys.argv[2], "w"), sort_keys=False)
for name, s in lite.items():
    print(name, len(s["arms"]), "arms", s["datasets"], s["batch_sizes"], s["ks"])
