"""Roadmap L1 step 0: pin the pre-move numbers, then show the moved library reproduces them.

``capture`` runs at the branch point (``development`` @ 4f52972, before any file moved) and
writes one ``.pt`` per cell under ``--dir``: the parity suite's ``make_probe_family`` inputs
with ``tests/parity/conftest.py::ref_cps_phase23``'s output on them (plain, bloom, exact), and
a ``SilverTorch`` state dict + ``(ids, scores)`` for every backend x filter mode on a small
deterministic index. ``compare`` runs on the moved tree: ``retrieve.ops.reference`` against the
saved reference outputs (``torch.equal`` on scores, ids equal at every finite slot and equal up
to ties after the ``-1`` sentinel), and the saved state dicts loaded into the moved modules
(``torch.equal`` on ids and scores). Results go to ``results.json`` next to the tensors.

    cd retrieve && PYTHONPATH=. python ../docs/artifacts/library-api-refactor/l1/premove_check.py capture --dir <dir>
    cd retrieve && PYTHONPATH=. python ../docs/artifacts/library-api-refactor/l1/premove_check.py compare --dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch

from tests.conftest import make_attrs, make_index, make_query, make_query_attrs
from tests.parity.conftest import make_bloom, make_exact, make_probe_family

# test_official.py T1 cells: (n_lists, max_size, n_probe, d, k) x b.
PLAIN = [(16, 64, 4, 64, 8), (64, 96, 8, 128, 32), (32, 64, 8, 96, 16), (32, 90, 8, 128, 32)]
BATCHES = [1, 16]

# The small index for the state-dict round trip.
N, D, B, K = 2048, 64, 8, 16
N_LISTS, N_PROBE, N_ITER = 32, 4, 3
M_BITS, K_HASH, C, A_MAX = 256, 4, 2, 2
BACKENDS = ("triton", "torch", "official")
FILTER_MODES = ("none", "bloom", "exact")


def _family(b, n_lists, max_size, n_probe, d):
    try:
        from retrieve.indexing.quantize import quantize_int8_global
    except ImportError:  # pre-move tree
        from retrieve.layers.utils.quantize import quantize_int8_global

    _, _, flat, n = make_probe_family(b, n_lists, max_size, n_probe)
    codes, global_scale = quantize_int8_global(make_index(n, d))
    return dict(query=make_query(b, d), flat=flat, codes=codes, global_scale=global_scale, n=n)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def _index_data():
    return dict(
        embs=make_index(N, D),
        query=make_query(B, D),
        attrs=make_attrs(N, c=C, a_max=A_MAX),
        q_attrs=make_query_attrs(B, c=C),
    )


def _module_kwargs(backend, filter_mode):
    kw = dict(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=N_ITER, seed=0, backend=backend)
    if filter_mode == "bloom":
        kw.update(filter_mode="bloom", m_bits=None if backend == "official" else M_BITS, k_hash=K_HASH)
    elif filter_mode == "exact":
        kw.update(filter_mode="exact")
    if backend == "official":
        try:
            from retrieve import OfficialConfig
        except ImportError:  # pre-move tree
            from retrieve.layers.silvertorch import OfficialConfig

        kw["official"] = OfficialConfig(score_path="int32")
    return kw


def _build(build_silvertorch, backend, filter_mode, data):
    kw = _module_kwargs(backend, filter_mode)
    if filter_mode == "none":
        return build_silvertorch(data["embs"], **kw)
    rev = torch.tensor([True, False], device="cuda") if filter_mode == "exact" else None
    return build_silvertorch(data["embs"], item_clause_attrs=data["attrs"], clause_is_reverse=rev, **kw)


def capture(out: Path) -> None:
    from retrieve import build_silvertorch
    from tests.parity.conftest import ref_cps_phase23

    out.mkdir(parents=True, exist_ok=True)
    cells = {}
    for n_lists, max_size, n_probe, d, k in PLAIN:
        for b in BATCHES:
            f = _family(b, n_lists, max_size, n_probe, d)
            ids, scores = ref_cps_phase23(f["query"], f["flat"], f["codes"], f["global_scale"], k)
            name = f"plain_b{b}_l{n_lists}_m{max_size}_p{n_probe}_d{d}_k{k}"
            torch.save({**f, "k": k, "ids": ids, "scores": scores}, out / f"{name}.pt")
            cells[name] = _sha(scores)
    for d in (64, 128):
        b, k = 16, 32
        f = _family(b, 64, 96, 8, d)
        for reverse in ("none", "mixed"):
            attrs, rev, q_attrs = make_exact(f["n"], b, reverse=reverse)
            ids, scores = ref_cps_phase23(
                f["query"], f["flat"], f["codes"], f["global_scale"], k,
                item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs,
            )
            name = f"exact_d{d}_{reverse}"
            torch.save(
                {**f, "k": k, "attrs": attrs, "rev": rev, "q_attrs": q_attrs, "ids": ids, "scores": scores},
                out / f"{name}.pt",
            )
            cells[name] = _sha(scores)
        sigs, qb = make_bloom(f["n"], b)
        ids, scores = ref_cps_phase23(
            f["query"], f["flat"], f["codes"], f["global_scale"], k, qb=qb, bloom_sigs=sigs
        )
        name = f"bloom_d{d}"
        torch.save({**f, "k": k, "qb": qb, "sigs": sigs, "ids": ids, "scores": scores}, out / f"{name}.pt")
        cells[name] = _sha(scores)

    data = _index_data()
    modules = {}
    for backend in BACKENDS:
        for filter_mode in FILTER_MODES:
            m = _build(build_silvertorch, backend, filter_mode, data)
            qa = None if filter_mode == "none" else data["q_attrs"]
            ids, scores = m(data["query"], qa)
            name = f"silvertorch_{backend}_{filter_mode}"
            sd = {k_: v.cpu() for k_, v in m.state_dict().items()}
            torch.save({"state_dict": sd, "ids": ids.cpu(), "scores": scores.cpu()}, out / f"{name}.pt")
            modules[name] = {"keys": list(sd), "scores_sha": _sha(scores)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out / "capture.json").write_text(json.dumps({"commit": commit, "cells": cells, "modules": modules}, indent=2))
    print(f"captured {len(cells)} reference cells and {len(modules)} modules at {commit} -> {out}")


def _ids_agree(ids_a, ids_b, scores) -> dict:
    from tests.parity.conftest import assert_ids_equal_up_to_ties

    finite = torch.isfinite(scores)
    finite_equal = bool(torch.equal(ids_a[finite], ids_b[finite]))
    a = ids_a.masked_fill(~finite, -1)
    b = ids_b.masked_fill(~finite, -1)
    assert_ids_equal_up_to_ties(a, b, scores)
    return {"ids_equal_at_finite_slots": finite_equal, "ids_exactly_equal_after_sentinel": bool(torch.equal(a, b))}


def compare(out: Path) -> None:
    from retrieve.modules.silvertorch import build_silvertorch
    from retrieve.ops import reference

    captured = json.loads((out / "capture.json").read_text())
    results = {"captured_at": captured["commit"], "cells": {}, "modules": {}}
    for name in captured["cells"]:
        c = torch.load(out / f"{name}.pt")
        args = (c["query"], c["flat"], c["codes"], c["global_scale"])
        if name.startswith("plain"):
            ids, scores = reference.codesigned_probe_score(*args, c["k"])
        elif name.startswith("bloom"):
            ids, scores = reference.codesigned_probe_score_bloom(
                c["query"], c["flat"], c["codes"], c["qb"], c["sigs"], c["global_scale"], c["k"]
            )
        else:
            ids, scores = reference.codesigned_probe_score_exact(
                c["query"], c["flat"], c["codes"], c["attrs"], c["rev"], c["q_attrs"], c["global_scale"], c["k"]
            )
        r = {"scores_equal": bool(torch.equal(scores, c["scores"])), **_ids_agree(ids, c["ids"], scores)}
        assert r["scores_equal"], name
        results["cells"][name] = r
        print(name, r)

    data = _index_data()
    for backend in BACKENDS:
        for filter_mode in FILTER_MODES:
            name = f"silvertorch_{backend}_{filter_mode}"
            saved = torch.load(out / f"{name}.pt")
            m = _build(build_silvertorch, backend, filter_mode, data)
            fresh_sd = {k_: v.cpu() for k_, v in m.state_dict().items()}
            fresh_equal = list(fresh_sd) == list(saved["state_dict"]) and all(
                torch.equal(fresh_sd[k_], saved["state_dict"][k_]) for k_ in fresh_sd
            )
            m.load_state_dict({k_: v.cuda() for k_, v in saved["state_dict"].items()})
            qa = None if filter_mode == "none" else data["q_attrs"]
            ids, scores = m(data["query"], qa)
            r = {
                "state_dict_keys_equal": list(fresh_sd) == list(saved["state_dict"]),
                "fresh_build_state_dict_equal": bool(fresh_equal),
                "ids_equal": bool(torch.equal(ids.cpu(), saved["ids"])),
                "scores_equal": bool(torch.equal(scores.cpu(), saved["scores"])),
            }
            assert r["ids_equal"] and r["scores_equal"], name
            results["modules"][name] = r
            print(name, r)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"all {len(results['cells'])} cells and {len(results['modules'])} modules identical -> {out / 'results.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["capture", "compare"])
    p.add_argument("--dir", type=Path, default=Path(__file__).parent / "tensors")
    a = p.parse_args()
    (capture if a.mode == "capture" else compare)(a.dir)
