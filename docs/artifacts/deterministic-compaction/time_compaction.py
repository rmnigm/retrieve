"""L3 gate 4 — time the two compaction kernels and one V2 / V3 forward, before and after.

Run twice from the retrieve/ package dir with the same venv, once on the old kernel and once on
the new one, and diff the two JSONs::

    cd /workspace/wt/l3/retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/artifacts/deterministic-compaction/time_compaction.py \
        --out ../docs/artifacts/deterministic-compaction/timing-before.json

Kernel timing is ``ops.tune._bench`` (``do_bench`` median, 500 reps) on the tuner's own synthetic
regimes plus the real goodreads ``item_attrs_narrow`` with ``c0_genre``-shaped queries (only clause
0 active, values drawn by item frequency). End-to-end is ``LiNRV2`` / ``LiNRV3`` with the real
attrs and random unit fp16 embeddings, eager and under the harness's compile settings. Clocks
cannot be locked here: ``sm_mhz`` is sampled around every group (H §7).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import torch
from retrieve.indexing.bloom_hash import build_query_signatures
from retrieve.modules import BloomFilter, ExactAttributeFilter, LiNRV2, LiNRV3
from retrieve.ops.tune import _bench, _bloom_inputs, _clause_inputs
from retrieve.ops.triton.bloom_compact import _bloom_compact_impl
from retrieve.ops.triton.clause_compact import _clause_compact_impl

DATA = "/workspace/data/goodreads-work-id"
CLAUSE_REGIMES = [(797_085, 1, 4, 4), (797_085, 16, 4, 4), (2_988_997, 1, 5, 4), (2_988_997, 16, 5, 4)]
BLOOM_REGIMES = [(797_085, 1, 16), (797_085, 16, 16), (2_988_997, 1, 16), (2_988_997, 16, 16)]


def sm_mhz() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def real_queries(attrs: torch.Tensor, b: int, seed: int = 0) -> torch.Tensor:
    c0 = attrs[:, 0, :]
    g = torch.Generator(device=attrs.device).manual_seed(seed)
    pool = c0[c0 >= 0]
    pick = torch.randint(0, pool.numel(), (b,), generator=g, device=attrs.device)
    q = torch.full((b, attrs.shape[1]), -1, dtype=torch.int64, device=attrs.device)
    q[:, 0] = pool[pick]
    return q


def timed(name: str, fn, results: dict, extra: dict | None = None) -> None:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    mhz0 = sm_mhz()
    ms = _bench(fn)
    mhz1 = sm_mhz()
    results[name] = {"ms": ms, "sm_mhz": [mhz0, mhz1], **(extra or {})}
    print(f"{name:60s} {ms:9.4f} ms   sm {mhz0}/{mhz1} MHz", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    results: dict = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "kernels": {}, "e2e": {}}
    kr, er = results["kernels"], results["e2e"]

    for regime in CLAUSE_REGIMES:
        inputs = _clause_inputs(dev, regime)
        timed(f"clause_compact synth N,B,C,A={regime}", lambda: _clause_compact_impl(**inputs), kr)
        del inputs
    for regime in BLOOM_REGIMES:
        inputs = _bloom_inputs(dev, regime)
        timed(f"bloom_compact synth N,B,W={regime}", lambda: _bloom_compact_impl(**inputs), kr)
        del inputs

    attrs = torch.load(f"{DATA}/item_attrs_narrow.pt").to(dev)
    rev = torch.load(f"{DATA}/clause_is_reverse_narrow.pt").to(dev)
    n = attrs.shape[0]
    bf = BloomFilter(m_bits=1024, k_hash=5).to(dev)
    bf.register_index(attrs)
    for b in (1, 16):
        q = real_queries(attrs, b)
        ef = ExactAttributeFilter().to(dev)
        ef.register_index(attrs, clause_is_reverse=rev)
        counts = ef.evaluate_mask(q).sum(1)
        pass_rate = (counts.float() / n).tolist()
        timed(
            f"clause_compact goodreads c0_genre B={b}",
            lambda: _clause_compact_impl(attrs, rev, q),
            kr,
            {"pass_rate": pass_rate},
        )
        qb = build_query_signatures(
            q.unsqueeze(-1), bf.hash_seeds, bf.m_bits, bf.k_hash, bf.word_count, clause_salt=bf.clause_salt
        )
        bcounts = bf.evaluate_mask(q).sum(1)
        timed(
            f"bloom_compact goodreads c0_genre B={b}",
            lambda: _bloom_compact_impl(qb, bf.bloom_sigs),
            kr,
            {"pass_rate": (bcounts.float() / n).tolist()},
        )

    g = torch.Generator(device=dev).manual_seed(0)
    embs = torch.randn(n, 128, generator=g, device=dev)
    embs = (embs / embs.norm(dim=1, keepdim=True)).half()
    for name, make in (
        ("linr_v2 clause", lambda: LiNRV2(100, filter=ExactAttributeFilter())),
        ("linr_v2 bloom", lambda: LiNRV2(100, filter=BloomFilter(m_bits=1024, k_hash=5))),
        ("linr_v3 clause", lambda: LiNRV3(100, candidate_pool=5000, filter=ExactAttributeFilter())),
    ):
        mod = make().to(dev)
        mod.register_index(embs, attrs, None if "bloom" in name else rev)
        for b in (1, 16):
            q = real_queries(attrs, b)
            qe = torch.randn(b, 128, generator=g, device=dev)
            qe = (qe / qe.norm(dim=1, keepdim=True)).half()
            timed(f"{name} eager B={b}", lambda: mod(qe, q), er)
            compiled = torch.compile(mod, mode="reduce-overhead", dynamic=False, fullgraph=True)
            for _ in range(5):
                compiled(qe, q)
            torch.cuda.synchronize()
            timed(f"{name} graph B={b}", lambda: compiled(qe, q), er)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                mod(qe, q)
                torch.cuda.synchronize()
            rows = sorted(
                (
                    (e.key, e.device_time_total, e.count)
                    for e in prof.key_averages()
                    if e.device_time_total > 0
                ),
                key=lambda r: -r[1],
            )[:12]
            er[f"{name} eager B={b}"]["profile_top_us"] = [
                {"kernel": k[:80], "us": round(us, 1), "n": c} for k, us, c in rows
            ]
            torch._dynamo.reset()
        del mod
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
