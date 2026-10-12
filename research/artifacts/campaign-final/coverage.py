"""The final grid against the merged tree (README.md § After all legs): every cell `load_matrix` expands from the
config's suites.yaml, keyed like a record without its code_version, set against the tree's latest records.

usage: coverage.py CONFIG_DIR MERGED_TREE OUT.json
"""

import collections
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "evaluation"))
from bench import config, records  # noqa: E402


def bare(key: str) -> str:
    return json.dumps(
        {k: v for k, v in json.loads(key).items() if k != "code_version"},
        sort_keys=True,
    )


def main():
    cfg, tree, out = map(Path, sys.argv[1:4])
    have = {bare(records.record_key(r)) for r in records.latest(tree)}
    planned, missing, miss = set(), [], collections.Counter()
    for suite, spec in yaml.safe_load((cfg / "suites.yaml").read_text()).items():
        if not isinstance(spec, dict) or "datasets" not in spec:
            continue
        for ds in spec["datasets"]:
            for j in config.load_matrix(cfg / f"{ds}.yaml", cfg / "suites.yaml", suite):
                for p in j.cells():
                    k = json.dumps(j.key(p), sort_keys=True)
                    planned.add(k)
                    if k not in have:
                        missing.append(k)
                        miss[(suite, ds, j.dim, j.algo, j.backend)] += 1
    extra = sorted(have - planned)
    print(
        f"planned {len(planned)}, records {len(have)}, missing {len(missing)}, extra {len(extra)}"
    )
    for k, n in sorted(miss.items()):
        print("  missing", k, n)
    out.write_text(json.dumps({"missing": missing, "extra": extra}))


if __name__ == "__main__":
    main()
