"""Pod 1's final-pass runner: stage-a100-x1-eval (bench check + bench oracle per (dataset, suite) of the p1 legs), then every
p1-* leg of the tag's recipe.yaml in order with the leg's env and its exact command under flock /scratch/gpu0.lock, then
its upload in the corrected form (upload_leg.sh; the recipe's upload_command passes the suite dir and uploads 0 records). A leg is skipped when its done marker exists; any non-zero rc stops the queue.
P1_TREE=campaign-final selects the extension legs (recipe `tree: campaign-final`); P1_VENV_MAP=/venvs/final=/venvs/final-ext,...
maps the recipe's venvs to that tree's own, and a mapped -O3 venv without a flags file is built (scripts/build_official_o3.sh,
under the GPU lock) before its first leg, then checked to differ from the shipped .so. Usage (from the
FINAL_TAG worktree's evaluation/): python run_p1.py RECIPE LIBRARY_TREE OUT_LOG_DIR"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

recipe, tree, logs = (
    yaml.safe_load(Path(sys.argv[1]).read_text()),
    sys.argv[2],
    Path(sys.argv[3]),
)
logs.mkdir(parents=True, exist_ok=True)
driver = open(logs / "driver.log", "a", buffering=1)


def say(s: str) -> None:
    driver.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())} {s}\n")


UPLOAD = str(Path(__file__).with_name("upload_leg.sh"))
TREE = os.environ.get(
    "P1_TREE"
)  # campaign-final: only the legs the recipe marks with that tree
VENVS = dict(
    kv.split("=") for kv in os.environ.get("P1_VENV_MAP", "").split(",") if kv
)  # recipe venv -> this tree's venv
legs = [
    leg
    for leg in recipe["legs"]
    if leg["id"].startswith("p1-") and leg.get("tree") == TREE
]
got = subprocess.run(
    ["git", "rev-parse", "HEAD:retrieve/src/retrieve"],
    capture_output=True,
    text=True,
    check=True,
).stdout.strip()
if got != tree:
    say(f"refusing: library tree {got} != {tree}")
    sys.exit(1)
say(f"library {tree}, {len(legs)} legs")


def sh(name: str, cmd: str, env: dict, gpu: bool) -> int:
    full = f"flock /scratch/gpu0.lock bash -c {json.dumps(cmd)}" if gpu else cmd
    t0 = time.time()
    with open(logs / f"{name}.log", "a") as f:
        rc = subprocess.run(
            ["bash", "-c", full],
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
        ).returncode
    say(f"step {name} rc={rc} s={time.time() - t0:.0f}")
    return rc


env0 = {k: VENVS.get(str(v), str(v)) for k, v in legs[0]["env"].items()}
if not (logs / "stage.done").exists():
    py = f'"{env0["VENV"]}/bin/python" -m bench.cli'
    pairs = sorted(
        {
            (leg["dataset"], leg["suite"], re.search(r"--dim (\d+)", leg["command"])[1])
            for leg in legs
        }
    )
    for ds in sorted({p[0] for p in pairs}):
        if sh(f"check-{ds}", f"{py} check --dataset {ds}", env0, False):
            sys.exit(1)
    for ds, suite, dim in pairs:
        if sh(
            f"oracle-{ds}-{suite}",
            f"{py} oracle --dataset {ds} --suite {suite} --dim {dim}",
            env0,
            True,
        ):
            sys.exit(1)
    (logs / "stage.done").touch()
for leg in legs:
    done = logs / f"{leg['id']}.done"
    if done.exists():
        continue
    env = {k: VENVS.get(str(v), str(v)) for k, v in leg["env"].items()}
    flags = list(
        Path(env["VENV"]).glob(
            "lib/python*/site-packages/silvertorch/_build_flags.json"
        )
    )
    if env["VENV"].endswith("-o3") and not flags and VENVS:
        build = f"UV_PROJECT_ENVIRONMENT={env['VENV']} ../scripts/build_official_o3.sh"
        if sh(f"build-{Path(env['VENV']).name}", build, {}, True):
            sys.exit(1)
        so = {v: subprocess.run(f"sha256sum {v}/lib/python*/site-packages/silvertorch/_C*.so", shell=True, capture_output=True,
                                text=True).stdout.split()[0] for v in (env["VENV"], env["VENV"].removesuffix("-o3"))}  # fmt: skip
        say(f"official .so sha256 {so}")
        if len(set(so.values())) != 2:
            say(f"stopped at {leg['id']}: the -O3 .so equals the shipped one")
            sys.exit(1)
    if env["VENV"].endswith("-o3") and not list(
        Path(env["VENV"]).glob(
            "lib/python*/site-packages/silvertorch/_build_flags.json"
        )
    ):
        say(f"stopped at {leg['id']}: {env['VENV']} has no -O3 build flags file")
        sys.exit(1)
    dim = re.search(r"--dim (\d+)", leg["command"])[1]
    up = f"{UPLOAD} {leg['id']} {leg['upload']} {leg['suite']} {leg['dataset']} {dim} {env['VENV']} /scratch/final/gpu0 {tree}"
    if sh(leg["id"], leg["command"], env, True) or sh(
        f"{leg['id']}-upload", up, env, False
    ):
        say(f"stopped at {leg['id']}")
        sys.exit(1)
    sha = re.findall(
        r"MANIFEST.json sha256 (\w+)", (logs / f"{leg['id']}-upload.log").read_text()
    )
    say(f"uploaded {leg['upload']} manifest {sha[-1] if sha else '?'}")
    done.touch()
say("driver done")
