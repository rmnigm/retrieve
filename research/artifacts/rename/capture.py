"""RENAME (roadmap §5) gate: the META-FORK gate grid's int32 outputs on one SilverTorch backend,
saved so the tree before the rename and the tree after can be compared cell by cell
(``compare.py``: ``torch.equal`` on ids and scores; a rename changes no behaviour).

    cd <tree>/evaluation && python capture.py GRID OUT_DIR --backend official-fork|cuda

``GRID`` is a ``gate_fork.py`` spec file: ``DATASET DIM MODE SWEEPS N_LISTS NPROBES BSS [--fork-bloom
meta|default] [--m-bits M]`` per line (the timing / arm / ``--recall`` / ``--bloom-path`` flags are
ignored; the adapted code has one bloom path). One ``.pt`` per line under ``OUT_DIR``:
``{"<sweep>/np<n_probe>/bs<bs>/k<k>": [(ids, scores), ...]}`` over the 16 pool batches, on the
CPU; ``manifest.json`` carries a sha256 per cell and the environment. The index (k-means seed 0,
``n_iter`` 10) is built once per (dataset, dim, n_lists) and shared by every mode, as the gate did.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from bench import config, inputs

from retrieve import OfficialConfig, SilverTorch

DEV = torch.device("cuda")
N_POOL = 16
LAYOUT = (
    "centroids",
    "item_codes",
    "global_scale",
    "cluster_offsets",
    "cluster_sizes",
    "sort_perm",
    "inv_perm",
)


def parse_line(line: str) -> dict | None:
    parts = line.split()
    if len(parts) < 7:
        return None
    ds, dim, mode, sweeps, n_lists, nprobes, bss, *extra = parts
    spec = {
        "dataset": ds,
        "dim": int(dim),
        "mode": mode,
        "sweeps": [] if sweeps == "-" else sweeps.split(","),
        "n_lists": int(n_lists),
        "nprobes": [int(x) for x in nprobes.split(",")],
        "bss": [int(x) for x in bss.split(",")],
        "fork_bloom": "meta",
        "m_bits": 1024,
    }
    for i, tok in enumerate(extra):
        if tok == "--fork-bloom":
            spec["fork_bloom"] = extra[i + 1]
        elif tok == "--m-bits":
            spec["m_bits"] = int(extra[i + 1])
    return spec


def tag_of(spec: dict) -> str:
    sw = "_".join(spec["sweeps"]) or "none"
    return (
        f"{spec['dataset']}_d{spec['dim']}_{spec['mode']}_{sw}_nl{spec['n_lists']}"
        f"_np{'-'.join(map(str, spec['nprobes']))}_{spec['fork_bloom']}_m{spec['m_bits']}"
    )


def module_for(spec: dict, base: SilverTorch, inp: dict, backend: str) -> SilverTorch:
    """A ``spec['mode']`` module on ``base``'s index. ``--fork-bloom meta`` is Meta's bloom
    semantics (upstream width, no recheck), ``default`` the adapted code's own."""
    mode = spec["mode"]
    kw = {"k": 100, "n_lists": spec["n_lists"], "n_probe": 24, "n_iter": 10, "seed": 0}
    meta = spec["fork_bloom"] == "meta"
    cfg = {"score_path": "int32"}
    if mode == "bloom":
        kw |= {"k_hash": 5, "m_bits": None if meta else spec["m_bits"]}
        if meta:
            cfg |= {"bloom_width": "upstream", "recheck": False}
    m = SilverTorch(
        filter_mode=mode, backend=backend, official=OfficialConfig(**cfg), **kw
    )
    for b in LAYOUT:
        m.register_buffer(b, getattr(base, b))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    if mode != "none":
        attrs = inp["item_attrs"].to(DEV)
        reverse = inp.get("clause_is_reverse") if mode == "exact" else None
        m._register_filter_buffers(attrs, reverse, base.sort_perm)
    return m


def sha(outs: list[tuple[torch.Tensor, torch.Tensor]]) -> str:
    h = hashlib.sha256()
    for ids, scores in outs:
        h.update(ids.numpy().tobytes())
        h.update(scores.numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid")
    ap.add_argument("out")
    ap.add_argument("--backend", required=True)
    ap.add_argument("--k", default="100,1000")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ks = [int(x) for x in a.k.split(",")]
    specs = [s for s in map(parse_line, Path(a.grid).read_text().splitlines()) if s]
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("cells", {})
    manifest["env"] = {
        "backend": a.backend,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "python": sys.executable,
    }
    loaded: dict[tuple, dict] = {}
    bases: dict[tuple, SilverTorch] = {}
    for spec in specs:
        tag = tag_of(spec)
        if (out / f"{tag}.pt").exists() and tag in manifest["cells"]:
            print(f"{tag}: done", flush=True)
            continue
        key = (spec["dataset"], spec["dim"])
        if (
            key not in loaded
        ):  # one dataset at a time on the GPU: the tail lines rebuild
            loaded.clear()
            bases.clear()
            torch.cuda.empty_cache()
            ds = config.load_dataset(
                Path(f"config/{spec['dataset']}.yaml"), spec["dim"]
            )
            loaded[key] = inputs.load_inputs(ds, DEV, with_filters=True)
            loaded[key]["filters"] = yaml.safe_load(
                Path(f"config/{spec['dataset']}.yaml").read_text()
            )["filters"]
        inp = loaded[key]
        bkey = (*key, spec["n_lists"])
        if bkey not in bases:
            base = SilverTorch(
                k=100,
                n_lists=spec["n_lists"],
                n_probe=24,
                n_iter=10,
                seed=0,
                backend=a.backend,
            )
            base.register_index(inp["item_embs"].to(DEV))
            bases[bkey] = base
            torch.cuda.empty_cache()
        m = module_for(spec, bases[bkey], inp, a.backend)
        sweep_cfg = {} if spec["mode"] == "none" else inp["filters"][
            "clause" if spec["mode"] == "exact" else "bloom"]  # fmt: skip
        cells: dict[str, list] = {}
        hashes: dict[str, str] = {}
        with torch.inference_mode():
            for sweep in spec["sweeps"] or ["none"]:
                qa_s, skip = (
                    (None, None)
                    if spec["mode"] == "none"
                    else inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
                )
                for n_probe in spec["nprobes"]:
                    m.set_query_params(n_probe=n_probe)
                    for bs in spec["bss"]:
                        pool, qa_pool = inputs.query_pool(
                            inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
                        )
                        preps = [
                            None
                            if spec["mode"] == "none"
                            else m.prepare_queries(qa_pool[i])
                            for i in range(N_POOL)
                        ]
                        for k in ks:
                            m.k = k
                            outs = [
                                tuple(t.cpu() for t in m(pool[i], preps[i]))
                                for i in range(N_POOL)
                            ]
                            name = f"{sweep}/np{n_probe}/bs{bs}/k{k}"
                            cells[name] = outs
                            hashes[name] = sha(outs)
        torch.save(cells, out / f"{tag}.pt")
        manifest["cells"][tag] = hashes
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        print(f"{tag}: {len(cells)} cells", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
