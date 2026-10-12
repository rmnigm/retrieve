"""META-FORK gate (plan §5, C2; the V1-BS1 / ST-TOPK gate shape). One process, up to three SilverTorch arms
on one index: ``fork`` (``backend="official-fork"``, this tree), ``upstream`` (``backend="official"``, the
pinned Meta build loaded beside it) and ``triton`` (campaign-v2.12's library as the renamed package
``retrieve_v212``, make_pkg.sh). The IVF layout (k-means, codes, CSR) is built once and shared by every arm;
each arm registers its own filter buffers (the two official arms each run their own
``bloom_index_build``, whose outputs must be ``torch.equal``: ``bloom_index_equal``).

Per sweep x n_probe x bs x k, on 16 pool batches (``inputs.query_pool``, seed 0):
  equality  fork against upstream: ``equal`` (ids and scores ``torch.equal``), ``scores_equal``,
            ``ids_up_to_ties`` (ids equal per distinct score above the row's k-th score; distinct inside
            the k-th group, ``-1`` where it is ``-inf``); fork against triton the same, on ``none`` /
            ``exact`` at ``--score-path int32`` only (Triton's bloom is another hash);
  timing    ROUNDS rounds of one window of CALLS calls per arm, arms round-robin (``--swap`` reverses the
            order, builds included); eager for every arm, and ``--graph`` replays of one CUDA graph per pool
            batch for the arms that capture (``capturable``); median ms per arm, ratios fork / upstream and
            fork / triton, the SM clock sampled after every round (clocks cannot be locked).
``--aa`` replaces upstream with a second fork module (the identical-code floor). Bloom: ``--fork-bloom
meta`` gives the fork Meta's semantics (``bloom_width="upstream"``, no recheck; the parity gate X, where
the two builds must be ``torch.equal``), ``default`` its own (``m_bits`` bits per doc, survivors rechecked,
C12 / C13); ``--m-bits`` is the fork's configured and Triton's width. ``--recall`` (gate R) adds a Triton
exact module on the same layout as the oracle at the same probes and records per arm recall@k against it
and the false positives (returned ids failing the exact predicate).

    cd evaluation && PYTHONPATH=.:../retrieve/src:PKGS python gate_fork.py DATASET DIM MODE SWEEPS N_LISTS \
        NPROBES BSS OUT.json [--k 100,1000] [--arms fork,upstream,triton] [--graph] [--swap] [--aa] \
        [--score-path int32|fp16] [--bloom-path partial|full]   (the fork arm always runs partial)
      MODE none | exact | bloom; SWEEPS comma list of the dataset's sweeps (``-`` for none)
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch
import yaml
from bench import config, inputs

from retrieve import OfficialConfig, SilverTorch
from retrieve.functional import clause_subset_match

DEV = torch.device("cuda")
CALLS, ROUNDS, N_POOL = 30, 8, 16
LAYOUT = (
    "centroids",
    "item_codes",
    "global_scale",
    "cluster_offsets",
    "cluster_sizes",
    "sort_perm",
    "inv_perm",
)
BACKEND = {
    "fork": "official-fork",
    "fork2": "official-fork",
    "upstream": "official",
    "triton": "triton",
}


def sm_mhz() -> int:
    q = [
        "nvidia-smi",
        "--query-gpu=clocks.sm",
        "--format=csv,noheader,nounits",
        "-i",
        "0",
    ]
    return int(subprocess.check_output(q, text=True).strip().splitlines()[0])


def same_up_to_ties(x, y) -> bool:
    (ia, sa), (ib, _) = x, y
    kth = sa[:, -1:]
    pad = kth == float("-inf")
    canon = []
    for ids in (ia, ib):
        # (score desc, id asc) per row: stable sort by id, then stable sort by score
        by_id = ids.sort(dim=1, stable=True).indices
        s = sa.gather(1, by_id)
        by_score = s.sort(dim=1, descending=True, stable=True).indices
        canon.append((ids.gather(1, by_id).gather(1, by_score), s.gather(1, by_score)))
    (ca, s), (cb, _) = canon
    above, tied = s > kth, s == kth
    if not torch.equal(ca.masked_fill(~above, 0), cb.masked_fill(~above, 0)):
        return False
    for c, _ in canon:
        if (tied & pad & (c != -1)).any():
            return False
        run = tied[:, 1:] & tied[:, :-1] & ~pad
        if (run & (c[:, 1:] == c[:, :-1])).any():
            return False
    return True


def compare(xs, ys) -> dict[str, bool]:
    pairs = list(zip(xs, ys, strict=True))
    scores = all(torch.equal(x[1], y[1]) for x, y in pairs)
    return {
        "equal": all(
            torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in pairs
        ),
        "scores_equal": scores,
        "ids_up_to_ties": scores and all(same_up_to_ties(x, y) for x, y in pairs),
    }


def make_arm(
    name, base, mode, attrs, reverse, kw, score_path, bloom_path, fork_bloom, m_bits
):
    backend = BACKEND[name]
    cls = (
        importlib.import_module("retrieve_v212.modules.silvertorch").SilverTorch
        if name == "triton"
        else SilverTorch
    )
    official = backend != "triton"
    meta = backend == "official" or (
        backend == "official-fork" and fork_bloom == "meta"
    )
    mkw = dict(kw)
    if mode == "bloom":
        mkw |= {"k_hash": 5, "m_bits": None if meta else m_bits}
        if not official and bloom_path == "full":
            mkw["bloom_path"] = "full"
    semantics = (
        {"bloom_width": "upstream", "recheck": False}
        if backend == "official-fork" and meta
        else {}
    )
    cfg = (
        OfficialConfig(
            score_path=score_path,
            bloom_path="partial" if backend == "official-fork" else bloom_path,
            **semantics,
        )
        if official
        else None
    )
    m = cls(
        filter_mode=mode,
        backend=backend,
        **({"official": cfg} if official else {}),
        **mkw,
    )
    for b in LAYOUT:
        m.register_buffer(b, getattr(base, b))
    m._global_scale_f, m._probe_width = base._global_scale_f, base._probe_width
    if mode != "none":
        m._register_filter_buffers(
            attrs.shape[0], attrs, reverse if mode == "exact" else None, base.sort_perm
        )
    return m


def recall_fp(outs, oracle, item_attrs, qa_pool):
    """Mean recall@k against the oracle's valid ids, and the returned ids failing the exact
    predicate (bloom false positives), over the pool."""
    recall, fp = [], 0
    for (ids, _), (want, _), qa in zip(outs, oracle, qa_pool, strict=True):
        for r in range(ids.shape[0]):
            w = set(want[r][want[r] >= 0].tolist())
            got = set(ids[r][ids[r] >= 0].tolist())
            recall.append(len(w & got) / len(w) if w else 1.0)
        rev = torch.zeros(qa.shape[1], dtype=torch.bool, device=qa.device)
        passing = clause_subset_match(item_attrs[ids.clamp_min(0)], qa, rev)
        fp += int(((ids >= 0) & ~passing).sum())
    return statistics.fmean(recall), fp


def capture(m, pool, prep):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(N_POOL):
            m(pool[i], prep[i])
    torch.cuda.current_stream().wait_stream(s)
    graphs = []
    for i in range(N_POOL):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            m(pool[i], prep[i])
        graphs.append(g)
    return graphs


def window(run, pool, prep, graph):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(CALLS):
        if graph:
            run[i % N_POOL].replay()
        else:
            run(pool[i % N_POOL], prep[i % N_POOL])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / CALLS * 1e3


def timed(arms, pool, preps, graph):
    run = {a: capture(m, pool, preps[a]) if graph else m for a, m in arms.items()}
    t = {a: [] for a in arms}
    mhz = []
    for _ in range(ROUNDS):
        for a in arms:
            t[a].append(window(run[a], pool, preps[a], graph))
        mhz.append(sm_mhz())
    del run
    return {a: statistics.median(v) for a, v in t.items()}, t, mhz


def main():
    ap = argparse.ArgumentParser()
    for p in ("dataset", "dim", "mode", "sweeps", "n_lists", "nprobes", "bss", "out"):
        ap.add_argument(p)
    ap.add_argument("--k", default="100,1000")
    ap.add_argument("--arms", default="fork,upstream,triton")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--swap", action="store_true")
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--score-path", default="int32")
    ap.add_argument("--bloom-path", default="partial")
    ap.add_argument("--fork-bloom", choices=("meta", "default"), default="meta")
    ap.add_argument("--m-bits", type=int, default=1024)
    ap.add_argument("--recall", action="store_true")
    a = ap.parse_args()
    names = [("fork2" if (a.aa and n == "upstream") else n) for n in a.arms.split(",")]
    if a.swap:
        names.reverse()
    cfg_path = Path(f"config/{a.dataset}.yaml")
    ds = config.load_dataset(cfg_path, int(a.dim))
    inp = inputs.load_inputs(ds, DEV, with_filters=True)
    kw = {"k": 100, "n_lists": int(a.n_lists), "n_probe": 24, "n_iter": 10, "seed": 0}
    base = SilverTorch(**kw)
    base.register_index(inp["item_embs"].to(DEV))
    del inp["item_embs"]
    gc.collect()
    torch.cuda.empty_cache()
    attrs = None if a.mode == "none" else inp["item_attrs"].to(DEV)
    reverse = inp.get("clause_is_reverse")
    arms = {
        n: make_arm(n, base, a.mode, attrs, reverse, kw, a.score_path, a.bloom_path,
                    a.fork_bloom, a.m_bits)
        for n in names
    }  # fmt: skip
    oracle = None
    if a.recall and a.mode != "none":
        oracle = SilverTorch(filter_mode="exact", **kw)
        for b in LAYOUT:
            oracle.register_buffer(b, getattr(base, b))
        oracle._global_scale_f, oracle._probe_width = (
            base._global_scale_f,
            base._probe_width,
        )
        oracle._register_filter_buffers(attrs.shape[0], attrs, None, base.sort_perm)
    bloom_equal = None
    if a.mode == "bloom" and {"fork", "upstream"} <= set(arms):
        bloom_equal = torch.equal(
            arms["fork"].bloom_index, arms["upstream"].bloom_index
        ) and torch.equal(
            arms["fork"].bundle_b_offsets, arms["upstream"].bundle_b_offsets
        )
        print(f"bloom_index_equal={bloom_equal}", flush=True)
    sweep_cfg = {} if a.mode == "none" else yaml.safe_load(cfg_path.read_text())["filters"][
        "clause" if a.mode == "exact" else "bloom"]  # fmt: skip
    rows = []
    with torch.inference_mode():
        for sweep in a.sweeps.split(","):
            qa_s, skip = (
                (None, None)
                if a.mode == "none"
                else inputs.sweep_qa(inp["qa"], tuple(sweep_cfg[sweep]))
            )
            for n_probe in map(int, a.nprobes.split(",")):
                for m in [*arms.values(), *([oracle] if oracle is not None else [])]:
                    m.set_query_params(n_probe=n_probe)
                for bs in map(int, a.bss.split(",")):
                    pool, qa_pool = inputs.query_pool(
                        inp, qa_s, skip, bs=bs, seed=0, n_pool=N_POOL, device=DEV
                    )
                    preps = {n: [None if a.mode == "none" else m.prepare_queries(qa_pool[i]) for i in range(N_POOL)]
                             for n, m in arms.items()}  # fmt: skip
                    for k in map(int, a.k.split(",")):
                        for m in arms.values():
                            m.k = k
                        outs = {
                            n: [m(pool[i], preps[n][i]) for i in range(N_POOL)]
                            for n, m in arms.items()
                        }
                        row = {"dataset": a.dataset, "dim": int(a.dim), "mode": a.mode, "sweep": sweep,
                               "bloom_path": a.bloom_path, "score_path": a.score_path, "n_probe": n_probe,
                               "bs": bs, "k": k, "width": base._probe_width, "swap": a.swap, "aa": a.aa,
                               "arms": names, "bloom_index_equal": bloom_equal, "fork_bloom": a.fork_bloom,
                               "m_bits": a.m_bits}  # fmt: skip
                        if oracle is not None:
                            oracle.k = k
                            want = [oracle(pool[i], oracle.prepare_queries(qa_pool[i]))
                                    for i in range(N_POOL)]  # fmt: skip
                            for n in outs:
                                rec, fp = recall_fp(
                                    outs[n], want, inp["item_attrs"].to(DEV), qa_pool
                                )
                                row[f"recall_{n}"], row[f"false_positives_{n}"] = (
                                    rec,
                                    fp,
                                )
                        ref = "fork2" if a.aa else "upstream"
                        if "fork" in outs and ref in outs:
                            row["vs_upstream"] = compare(outs["fork"], outs[ref])
                        if (
                            "fork" in outs
                            and "triton" in outs
                            and a.mode != "bloom"
                            and a.score_path == "int32"
                        ):
                            row["vs_triton"] = compare(outs["fork"], outs["triton"])
                        del outs
                        for n, m in arms.items():
                            for i in range(10):
                                m(pool[i % N_POOL], preps[n][i % N_POOL])
                        med, win, mhz = timed(arms, pool, preps, graph=False)
                        row |= {
                            "eager_ms": med,
                            "eager_windows_ms": win,
                            "eager_sm_mhz": mhz,
                        }
                        if a.graph:
                            cap = {n: m for n, m in arms.items() if m.capturable}
                            gmed, gwin, gmhz = timed(
                                cap, pool, {n: preps[n] for n in cap}, graph=True
                            )
                            row |= {
                                "graph_ms": gmed,
                                "graph_windows_ms": gwin,
                                "graph_sm_mhz": gmhz,
                            }
                        for kind in ("eager", "graph"):
                            ms = row.get(f"{kind}_ms", {})
                            for other in (ref, "triton"):
                                if "fork" in ms and other in ms:
                                    row[f"{kind}_fork_over_{other}"] = (
                                        ms["fork"] / ms[other]
                                    )
                        rows.append(row)
                        brief = {
                            k: v
                            for k, v in row.items()
                            if not k.endswith(("windows_ms", "sm_mhz"))
                        }
                        print(json.dumps(brief), flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
