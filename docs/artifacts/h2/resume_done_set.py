"""H2 gate: the D1 records' resume keys survive the key change. For one results tree, expand
today's matrix for each record file's (dataset, suite, dim) and print, as JSON, every cell whose
resume key the file holds as ``ok`` — the set ``bench run --resume`` would skip. Run it once
with the pre-H2 ``evaluation/`` tree and once with the H2 one; the two outputs must be equal.

    python resume_done_set.py EVALUATION_DIR RESULTS_DIR > done.json

EVALUATION_DIR provides ``bench`` and ``config/``; the code_version is the records' own (D1
keeps a record's original code_version), so only the key block is under test.
"""

import json
import sys
from pathlib import Path

evaluation, results = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(evaluation))
from bench import records  # noqa: E402
from bench.config import load_matrix  # noqa: E402

done = []
for path in records.record_files(results):
    suite, (dataset, dim) = path.parent.name, path.stem.rsplit("-d", 1)
    recs = records.read_records(path)
    (cv,) = {r["env"]["code_version"] for r in recs}
    keys = records.read_keys(path)
    cfg = evaluation / "config"
    for job in load_matrix(
        cfg / f"{dataset}.yaml", cfg / "suites.yaml", suite, dims=[int(dim)]
    ):
        for p in job.cells():
            if keys.get(records.resume_key(job.key(p), cv)) == "ok":
                k = {f: v for f, v in job.key(p).items() if f != "inputs"}
                done.append(json.dumps(k, sort_keys=True))
    print(
        f"{path}: {len(recs)} records, {list(keys.values()).count('ok')} ok",
        file=sys.stderr,
    )
print(json.dumps(sorted(done), indent=0))
print(f"{len(done)} cells done", file=sys.stderr)
