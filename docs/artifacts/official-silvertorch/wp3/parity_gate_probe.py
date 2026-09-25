"""Roadmap B2 / plan O WP-3: the two measurements the parity gate records beyond the
pass/fail of ``tests/parity/test_official.py``.

1. **T4 — bloom FPR at matched memory (plan §4.3).** Two corpora: T4's own (``make_attrs(4096,
   c=2, a_max=2, n_vocab=50, seed=41)``, 32 all-active AND queries — so selective that both
   blooms read FPR ≈ 0 at every width) and a dense one (vocab 8, 2 × 4 attrs with few pads, one
   active clause per query — where a bloom's width shows in its FPR). The official bloom is
   built at a grid of ``b_multiplier`` values and our row-wise bloom at ``m_bits`` ∈ {256, 512,
   1024}, both with ``k = 5`` hash positions; per row: bits per doc, index bytes, the measured
   false-positive rate against the exact predicate (``clause_mask``), false negatives. The
   official width per bundle is ``int(max_terms_per_doc_in_bundle · k · b_multiplier)`` bits
   per doc (``bundle_b_offsets`` is its running sum), so the matched-memory point for
   ``m_bits`` is ``b_multiplier = m_bits / (max_terms · 5)``: 12.8 / 25.6 / 51.2 on the T4
   corpus (4 terms), 6.4 / 12.8 / 25.6 on the dense one (8 terms) — all in the grid, and
   ``index_bytes`` is read off the real buffers so the 2 % criterion is checked, not assumed.

2. **T7 — launches, memcpys and host syncs per ``SilverTorch(backend="official")`` forward**
   for every filter path, with ``OfficialConfig.cache_plans`` on and off (``torch.profiler``
   for launches, A3's fd-2 + Python-warning instrument for syncs), on the T6 layer regime
   (N = 4096, D = 128, B = 16, n_lists = 64, n_probe = 8, K = 64). The Triton and torch
   forwards are profiled the same way as the reference rows.

Counts only — no timing (the SM clock is unlocked on this box and nothing here is a
performance claim). ``python parity_gate_probe.py --out <dir>`` writes ``parity_gate_probe.json``
and ``parity_gate_probe.txt``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import warnings
from contextlib import contextmanager

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "retrieve"
    ),
)

from tests.conftest import make_attrs, make_index, make_query, make_query_attrs  # noqa: E402

from retrieve.kernels.filters.clause_mask import clause_mask  # noqa: E402
from retrieve.kernels.silvertorch import official as of  # noqa: E402
from retrieve.layers.filters.bloom_hash import (  # noqa: E402
    bloom_subset_match,
    build_query_signatures,
    build_signatures,
    generate_clause_salt,
    generate_seeds,
)
from retrieve.layers.silvertorch import OfficialConfig, build_silvertorch  # noqa: E402

K_SEARCH, HASH_K = 5, 7
B_MULTS = [1.5, 2.0, 3.0, 5.0, 6.4, 10.0, 12.8, 20.0, 25.6, 40.0, 51.2]
M_BITS = [256, 512, 1024]


# --- instruments (A3's, official_facts.py) ------------------------------------------------


@contextmanager
def sync_counter():
    """Host syncs under ``set_sync_debug_mode("warn")``: ``"cpp"`` per c10 warning line on
    fd 2 (a sync inside a C++ op never becomes a Python warning), ``"py"`` per Python warning
    (a sync from Python-dispatched aten never reaches fd 2). Disjoint, so both are counted."""
    msgs: list[str] = []
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.synchronize()
    with (
        tempfile.TemporaryFile(mode="w+") as tf,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            torch.cuda.set_sync_debug_mode("warn")
            yield msgs
        finally:
            torch.cuda.set_sync_debug_mode(prev)
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tf.seek(0)
            msgs.extend("cpp" for ln in tf if "warn_or_error_on_sync" in ln)
    msgs.extend(
        "py"
        for w in caught
        if "called a synchronizing" in str(w.message)
        and "prototype" not in str(w.message)
    )


def count_launches(fn, *, warmup: int = 3) -> dict:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels: dict[str, int] = {}
    memcpy: dict[str, int] = {}
    memset = 0
    for ev in prof.events():
        if "CUDA" not in str(getattr(ev, "device_type", "")):
            continue
        name = ev.name
        if name.startswith("Memcpy"):
            memcpy[name] = memcpy.get(name, 0) + 1
        elif name.startswith("Memset"):
            memset += 1
        else:
            kernels[name] = kernels.get(name, 0) + 1
    with sync_counter() as msgs:
        fn()
    return {
        "kernel_launches": sum(kernels.values()),
        "distinct_kernels": len(kernels),
        "memcpy_d2h": sum(n for k, n in memcpy.items() if "DtoH" in k),
        "memcpy_h2d": sum(n for k, n in memcpy.items() if "HtoD" in k),
        "memset": memset,
        "host_syncs": len(msgs),
        "host_syncs_cpp": sum(1 for m in msgs if m == "cpp"),
        "host_syncs_py": sum(1 for m in msgs if m == "py"),
        "kernels": dict(sorted(kernels.items(), key=lambda kv: -kv[1])),
    }


# --- 1. FPR table -----------------------------------------------------------------------


def fpr_table(corpus: str) -> dict:
    n, c = 4096, 2
    if corpus == "t4":
        a_max, vocab = 2, 50
        attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=vocab, seed=41)
        qa = make_query_attrs(32, c=c, n_vocab=vocab, inactive_rate=0.0, seed=42)
    else:
        a_max, vocab = 4, 8
        attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=vocab, pad_rate=0.1, seed=51)
        qa = make_query_attrs(32, c=c, n_vocab=vocab, inactive_rate=0.0, seed=52)
        qa[:, 1] = -1  # one active clause: pass rate ≈ 1 − (7/8)^3.6
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    exact = clause_mask(attrs, rev, qa)
    neg = int((~exact).sum().item())
    out = {
        "corpus": corpus,
        "n": n,
        "c": c,
        "a_max": a_max,
        "vocab": vocab,
        "queries": 32,
        "k": K_SEARCH,
        "exact_pass_rate": float(exact.float().mean().item()),
        "max_terms_per_doc": int((attrs != -1).sum(dim=(1, 2)).max().item()),
        "official": [],
        "ours": [],
    }
    plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
    for bm in B_MULTS:
        index, boff = of.build_bloom_index(attrs, b_multiplier=bm, build_k=K_SEARCH)
        full = of.bloom_full_mask(
            index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True
        )[:, :n]
        # bundle_b_offsets is the running sum of B (bits per doc) over the bundles
        # (bloom_indexer.cpp:46-66), so its differences are B per bundle.
        bits_per_doc = [int(x) for x in (boff[1:] - boff[:-1]).tolist()]
        out["official"].append(
            {
                "b_multiplier": bm,
                "bits_per_doc_per_bundle": bits_per_doc,
                "index_words": int(index.numel()),
                "index_bytes": int(index.numel() * 8),
                "bytes_per_doc": index.numel() * 8 / n,
                "fpr": float((full & ~exact).sum().item() / max(neg, 1)),
                "false_negatives": int((exact & ~full).sum().item()),
            }
        )
    seeds = generate_seeds(K_SEARCH, device=attrs.device)
    salt = generate_clause_salt(c, device=attrs.device)
    for m in M_BITS:
        w = m // 64
        sigs = build_signatures(attrs.long(), seeds, m, K_SEARCH, w, clause_salt=salt)
        qb = build_query_signatures(
            qa.long().unsqueeze(-1), seeds, m, K_SEARCH, w, clause_salt=salt
        )
        ours = bloom_subset_match(qb, sigs.unsqueeze(0).expand(32, n, w))  # [B, N]
        out["ours"].append(
            {
                "m_bits": m,
                "index_words": int(sigs.numel()),
                "index_bytes": int(sigs.numel() * 8),
                "bytes_per_doc": sigs.numel() * 8 / n,
                "fpr": float((ours & ~exact).sum().item() / max(neg, 1)),
                "false_negatives": int((exact & ~ours).sum().item()),
            }
        )
    return out


# --- 2. launches / syncs per layer forward ---------------------------------------------


def forward_table() -> dict:
    n, d, b, k = 4096, 128, 16, 64
    embs, query = make_index(n, d), make_query(b, d)
    attrs, qa = make_attrs(n, c=2, a_max=2), make_query_attrs(b, c=2)
    base = dict(k=k, n_lists=64, n_probe=8, n_iter=3)
    rows = {}

    def run(name, module, q_attrs):
        rows[name] = count_launches(lambda: module(query, q_attrs))

    for backend in ("triton", "torch"):
        run(f"{backend}/none", build_silvertorch(embs, backend=backend, **base), None)
        run(
            f"{backend}/exact",
            build_silvertorch(
                embs,
                backend=backend,
                filter_mode="exact",
                item_clause_attrs=attrs,
                **base,
            ),
            qa,
        )
        run(
            f"{backend}/bloom",
            build_silvertorch(
                embs,
                backend=backend,
                filter_mode="bloom",
                m_bits=512,
                k_hash=K_SEARCH,
                item_clause_attrs=attrs,
                **base,
            ),
            qa,
        )
    for cache in (True, False):
        tag = f"cache_plans={cache}"
        for score_path in ("fp16", "int32"):
            cfg = dict(score_path=score_path, n_stored_hashes=HASH_K, cache_plans=cache)
            run(
                f"official/none/{score_path}/{tag}",
                build_silvertorch(
                    embs, backend="official", official=OfficialConfig(**cfg), **base
                ),
                None,
            )
            run(
                f"official/exact/{score_path}/{tag}",
                build_silvertorch(
                    embs,
                    backend="official",
                    filter_mode="exact",
                    item_clause_attrs=attrs,
                    official=OfficialConfig(**cfg),
                    **base,
                ),
                qa,
            )
            for path in ("partial", "full"):
                run(
                    f"official/bloom[{path}]/{score_path}/{tag}",
                    build_silvertorch(
                        embs,
                        backend="official",
                        filter_mode="bloom",
                        k_hash=K_SEARCH,
                        item_clause_attrs=attrs,
                        official=OfficialConfig(
                            bloom_path=path, b_multiplier=10.0, **cfg
                        ),
                        **base,
                    ),
                    qa,
                )
    return rows


def environment() -> dict:
    def sh(cmd):
        try:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        except Exception as e:  # noqa: BLE001
            return f"<{e}>"

    return {
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
        "sm_clock_mhz_unlocked_sample": sh(
            "nvidia-smi --query-gpu=clocks.sm --format=csv,noheader"
        ),
        "nvcc": sh("/usr/local/cuda-12.8/bin/nvcc --version | tail -1"),
        "silvertorch_sha": "21aa35e28b6dd9a91e9ee35efb0857715e86bda7",
    }


def render(res: dict) -> str:
    L = [f"### parity gate probe — {res['env']}", ""]
    for t in res["fpr"]:
        L.append(
            f"== T4 FPR table, corpus '{t['corpus']}': N={t['n']} C={t['c']} A_max={t['a_max']} "
            f"vocab={t['vocab']}, {t['queries']} AND queries, k={t['k']}, max terms/doc="
            f"{t['max_terms_per_doc']}, exact pass rate {t['exact_pass_rate']:.4f}"
        )
        L.append(
            f"{'arm':<10}{'width':>12}{'bytes/doc':>11}{'index B':>10}{'FPR':>10}{'FN':>5}"
        )
        for r in t["official"]:
            L.append(
                f"{'official':<10}{'b_mult=' + str(r['b_multiplier']):>12}"
                f"{r['bytes_per_doc']:>11.1f}{r['index_bytes']:>10}{r['fpr']:>10.4f}"
                f"{r['false_negatives']:>5}   bits/doc per bundle {r['bits_per_doc_per_bundle']}"
            )
        for r in t["ours"]:
            L.append(
                f"{'ours':<10}{'m_bits=' + str(r['m_bits']):>12}{r['bytes_per_doc']:>11.1f}"
                f"{r['index_bytes']:>10}{r['fpr']:>10.4f}{r['false_negatives']:>5}"
            )
        L.append("")
    L.append(
        "== T7 launches / memcpys / host syncs per forward (N=4096 D=128 B=16 K=64 n_probe=8)"
    )
    L.append(
        f"{'row':<46}{'launch':>7}{'dist':>6}{'D2H':>5}{'H2D':>5}{'sync':>5}  (cpp+py)"
    )
    for name, r in res["forward"].items():
        L.append(
            f"{name:<46}{r['kernel_launches']:>7}{r['distinct_kernels']:>6}{r['memcpy_d2h']:>5}"
            f"{r['memcpy_h2d']:>5}{r['host_syncs']:>5}  ({r['host_syncs_cpp']}+{r['host_syncs_py']})"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    of.ensure_loaded()
    res = {
        "env": environment(),
        "fpr": [fpr_table("t4"), fpr_table("dense")],
        "forward": forward_table(),
    }
    txt = render(res)
    print(txt)
    with open(os.path.join(args.out, "parity_gate_probe.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(args.out, "parity_gate_probe.txt"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
