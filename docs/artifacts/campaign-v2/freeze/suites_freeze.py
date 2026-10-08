"""Scratch config dirs for the campaign-v2 freeze gate (brief gates 3-5).

    uv run --directory evaluation python ../docs/artifacts/campaign-v2/freeze/suites_freeze.py \
        {smoke|timed|golden} OUT_DIR

Each mode writes ``OUT_DIR/suites.yaml`` and symlinks every ``config/<dataset>.yaml`` next to it,
so ``bench run --config-dir OUT_DIR`` resolves datasets exactly as with the real config (the
``data_dir`` paths are relative to ``evaluation/``). Suite names are kept, so record keys equal
the campaign's.

- ``smoke`` (gate 4): every suite of the real ``suites.yaml`` at seed 0, every arm on every
  dataset it runs on, each grid trimmed to its first value and each arm pinned to one sweep (the
  suite's first sweep for that dataset when the arm allows it, else the arm's first): one cell
  per (arm, filter kind), and one filter kind for the plain-torch SilverTorch arms. Every other key (interleave groups, perf flags, ks) is copied as is.
- ``timed`` (gate 5): ``codesign`` on arxiv's first sweep and ``h2h`` on goodreads bloom, k 100,
  bs 16, seed 0, everything else as the real suite: the interleaved groups, timed.
- ``golden`` (gate 3): the H2 golden rerun's cell set, independent of today's grid: goodreads
  ``c0_genre`` V1 / V2 / V3 triton, SilverTorch triton and official clause at ``n_probe`` 24 / 32;
  arxiv ``c0_maincat`` SilverTorch triton at 24 / 32; ks 100 / 500 / 1000 like the golden JSON.

Prints the expanded cells of every (suite, dataset) through ``bench.config.load_matrix``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml
from bench.config import load_dataset, load_matrix

EVAL = Path(__file__).resolve().parents[4] / "evaluation"
REAL = EVAL / "config"


def first(spec):
    if spec is None:
        return None
    if isinstance(spec, dict):
        return {k: v[:1] for k, v in spec.items()}
    return spec[:1]


def merged(arm: dict, over: dict, key: str):
    base, o = arm.get(key), over.get(key)
    if o is None:
        return base
    if isinstance(base, dict) and isinstance(o, dict):
        return {**base, **o}
    return o


def pick_sweep(suite: dict, arm: dict, over: dict, name: str) -> str | None:
    ds = load_dataset(REAL / f"{name}.yaml", _dim(suite, name))
    kinds = [fk for fk in (arm.get("filter_kinds") or suite["filter_kinds"]) if fk != "none"]
    if not kinds:
        return None
    order = (suite.get("sweeps") or {}).get(name) or list(
        dict.fromkeys(sw for fk in kinds for sw in ds.clauses.get(fk, {}))
    )
    allowed = over.get("sweeps", arm.get("sweeps"))
    cands = [sw for sw in order if allowed is None or sw in allowed]
    if allowed and not cands:
        cands = list(allowed)
    for sw in cands:
        if all(sw in ds.clauses.get(fk, {}) for fk in kinds):
            return sw
    for sw in cands:
        if any(sw in ds.clauses.get(fk, {}) for fk in kinds):
            return sw
    return None


def _dim(suite: dict, name: str) -> int:
    dims = yaml.safe_load((REAL / f"{name}.yaml").read_text())["dims"]
    return next(d for d in dims if d in suite.get("dims", dims))


def smoke(suites: dict) -> dict:
    out = {"bloom": suites.get("bloom")} if "bloom" in suites else {}
    for sname, s in suites.items():
        if sname == "bloom":
            continue
        s = copy.deepcopy(s)
        s["seeds"] = [0]
        arms = []
        for arm in s["arms"]:
            runs_on = list(arm["datasets"]) if "datasets" in arm else list(s["datasets"])
            per_ds = {}
            for name in runs_on:
                over = (arm.get("datasets") or {}).get(name) or {}
                entry = {}
                for key in ("build", "query"):
                    m = first(merged(arm, over, key))
                    if m:
                        entry[key] = m
                sw = pick_sweep(s, arm, over, name)
                if sw is not None:
                    entry["sweeps"] = [sw]
                per_ds[name] = entry
            new = {k: v for k, v in arm.items() if k not in ("build", "query", "sweeps", "datasets")}
            new["datasets"] = per_ds
            if arm["algo"] == "silvertorch" and arm["backends"] == ["torch"]:
                # plain-torch IVF quality is 515-917 s a cell at 0.8 M items: one filter kind
                new["filter_kinds"] = (arm.get("filter_kinds") or s["filter_kinds"])[:1]
            arms.append(new)
        s["arms"] = arms
        out[sname] = s
    return out


def timed(suites: dict) -> dict:
    out = {"bloom": suites.get("bloom")} if "bloom" in suites else {}
    cd = copy.deepcopy(suites["codesign"])
    cd.update(datasets=["arxiv"], ks=[100], batch_sizes=[16], seeds=[0])
    cd["sweeps"] = {"arxiv": cd["sweeps"]["arxiv"][:1]}
    for arm in cd["arms"]:
        if "datasets" in arm:
            arm["datasets"] = {"arxiv": arm["datasets"]["arxiv"]}
    out["codesign"] = cd
    h = copy.deepcopy(suites["h2h"])
    h.update(datasets=["goodreads"], filter_kinds=["bloom"], ks=[100], batch_sizes=[16], seeds=[0])
    if "sweeps" in h:
        h["sweeps"] = {"goodreads": h["sweeps"]["goodreads"][:1]}
    for arm in h["arms"]:
        if "datasets" in arm:
            arm["datasets"] = {"goodreads": arm["datasets"].get("goodreads") or {}}
        if "filter_kinds" in arm:
            arm["filter_kinds"] = [fk for fk in arm["filter_kinds"] if fk == "bloom"]
    h["arms"] = [a for a in h["arms"] if a.get("filter_kinds", ["bloom"])]
    out["h2h"] = h
    return out


def golden(_: dict) -> dict:
    np = {"n_probe": [24, 32]}
    return {
        "filter": {
            "datasets": ["goodreads", "arxiv"],
            "dims": [128],
            "filter_kinds": ["clause"],
            "ks": [100, 500, 1000],
            "batch_sizes": [1, 16],
            "seeds": [0],
            "sweeps": {"goodreads": ["c0_genre"], "arxiv": ["c0_maincat"]},
            "arms": [
                {"algo": "linr_v1_filter_mask", "backends": ["triton"], "datasets": {"goodreads": {}}},
                {"algo": "linr_v2", "backends": ["triton"], "datasets": {"goodreads": {}}},
                {"algo": "linr_v3", "backends": ["triton"], "datasets": {"goodreads": {}}},
                {"algo": "silvertorch", "backends": ["triton"], "query": np},
                {"algo": "silvertorch", "backends": ["official"], "query": np,
                 "datasets": {"goodreads": {}}},
            ],
        }
    }  # fmt: skip


def main(mode: str, out_dir: str) -> None:
    suites = yaml.safe_load((REAL / "suites.yaml").read_text())
    new = {"smoke": smoke, "timed": timed, "golden": golden}[mode](suites)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in REAL.glob("*.yaml"):
        if f.name != "suites.yaml":
            link = out / f.name
            link.unlink(missing_ok=True)
            link.symlink_to(f)
    (out / "suites.yaml").write_text(yaml.safe_dump(new, sort_keys=False))
    for sname, s in new.items():
        if sname == "bloom":
            continue
        for name in s["datasets"]:
            jobs = load_matrix(out / f"{name}.yaml", out / "suites.yaml", sname)
            cells = sum(len(j.query) for j in jobs)
            print(f"{sname:18s} {name:16s} jobs {len(jobs):3d} cells {cells:3d}")
            for j in jobs:
                for q in j.query:
                    print(f"    {j.algo}/{j.backend}/{j.filter_kind}/{j.sweep} {j.build | q}")


if __name__ == "__main__":
    main(*sys.argv[1:])
