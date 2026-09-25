#!/usr/bin/env python3
"""c4_linr_v4_probe.py — why does ``linr_v4`` miss C4's gate 1 by 7.3e-5?

C4's gate 1 failed on ``linr_v4`` (both backends, identically) while every other
goodreads cell matched the golden to ~1e-9. The two obvious explanations were tested
first and both are dead:

* **the ``k_max`` slice** (the one difference WP-4 gate 1 permits) — a rerun at
  ``--k 100``, i.e. ``k_max = 100`` exactly as the golden ran it, reproduces the v2
  number to the last digit, not the golden's;
* **the library** — H §11.2 found ``linr_v4`` bit-identical between A1 and the
  re-derive, so the library is not the variable.

What is left is a harness difference. The old harness compiled every algo at build time
(``AlgoBase._finalize``: ``self.compile(dynamic=True, mode="reduce-overhead")``) and ran
its *quality* pass through that compiled forward; harness v2 runs quality eager and only
compiles for the ``graph`` perf variant. ``linr_v4`` is the one algo whose scores come out
of ``_int_mm`` with an fp32 per-row scale-recovery epilogue — the kind of epilogue
inductor is free to fuse and reassociate, which moves score ties and therefore the top-k
boundary. Every other algo scores in fp32 cuBLAS, where there is nothing to reassociate.

This script measures that directly: same module, same inputs, eager vs
``torch.compile(dynamic=True, mode="reduce-overhead")``, ids and scores compared.

    python3 c4_linr_v4_probe.py [--algo linr_v4] [--k 100] [--rows 2048]
"""

from __future__ import annotations

import argparse

import torch
from retrieval import algos, data
from retrieval.config import load_matrix


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="linr_v4")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--dataset", default="goodreads")
    ap.add_argument("--sweep", default="c0_genre")
    ap.add_argument("--chunk2", type=int, default=64)
    ap.add_argument("--compare", choices=("compile", "chunk"), default="compile")
    a = ap.parse_args()

    jobs = load_matrix(
        f"config/{a.dataset}.yaml", "config/suites.yaml", "filter",
        dims=[128], algos=[a.algo], backends=[a.backend], filter_kinds=["clause"],
        sweeps=[a.sweep], seeds=[0],
    )  # fmt: skip
    job = jobs[0]
    dev = torch.device("cuda")
    inputs = data.load_inputs(job.data, dev)
    qa_s, skip = data.sweep_qa(inputs["qa"], job.clauses)
    filters = data.build_filters("clause", inputs, [a.backend], bloom=job.bloom)
    fmod = filters[algos.FILTER_BACKEND[a.backend]]
    keep = (~skip if skip is not None else torch.ones(inputs["n_queries"], dtype=torch.bool))
    rows = keep.nonzero().reshape(-1)[: a.rows]

    mod = algos.build(
        a.algo, inputs["item_embs"], k=a.k, backend=a.backend, filter_kind="clause",
        filter_mod=fmod, item_attrs=inputs["item_attrs"],
        clause_is_reverse=inputs["clause_is_reverse"], params={}, seed=0,
    )  # fmt: skip

    @torch.inference_mode()
    def run(m, chunk):
        ids, sc = [], []
        for s in range(0, rows.numel(), chunk):
            sel = rows[s : s + chunk]
            q = inputs["queries"][sel].to(dev)
            qa = qa_s[sel].to(dev)
            i, v = m(q, qa)
            ids.append(i.cpu())
            sc.append(v.float().cpu())
        return torch.cat(ids), torch.cat(sc)

    ids_e, sc_e = run(mod, a.chunk)
    if a.compare == "compile":
        mod.compile(dynamic=True, mode="reduce-overhead")
        ids_c, sc_c = run(mod, a.chunk)
        label = f"eager vs compiled (chunk {a.chunk})"
    else:
        ids_c, sc_c = run(mod, a.chunk2)
        label = f"chunk {a.chunk} vs chunk {a.chunk2}"
    print(label)

    same_ids = torch.equal(ids_e, ids_c)
    same_sc = torch.equal(sc_e, sc_c)
    n_rows, k = ids_e.shape
    diff_rows = int((ids_e != ids_c).any(dim=1).sum())
    both = torch.isfinite(sc_e) & torch.isfinite(sc_c)
    max_sc = (sc_e - sc_c).abs()[both].max().item() if both.any() else 0.0
    # -1 padding repeats inside a row, so compare distinct real ids only: a short-filled
    # row must not read as "overlap < 1" when the two id rows are in fact identical.
    inter = []
    for i in range(n_rows):
        se = set(ids_e[i].tolist()) - {-1}
        sc = set(ids_c[i].tolist()) - {-1}
        inter.append(1.0 if not se and not sc else len(se & sc) / max(len(se), len(sc)))
    print(f"algo={a.algo} backend={a.backend} k={k} rows={n_rows}")
    print(f"  ids identical:    {same_ids}   rows differing: {diff_rows}/{n_rows}")
    print(f"  scores identical: {same_sc}   max |eager - compiled|: {max_sc:.3e}")
    print(f"  mean set overlap@{k}: {sum(inter) / len(inter):.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
