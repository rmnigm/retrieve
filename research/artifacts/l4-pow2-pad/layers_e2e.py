"""L4 layer smoke: SilverTorch / LiNRV2 / LiNRV3 with ``backend="triton"`` against
``backend="torch"`` at the non-power-of-two widths of yfcc10m (192) and pubmed (768), eager and
``torch.compile``. Run from ``retrieve/`` with ``PYTHONPATH=.`` (it imports the test helpers)."""

import sys

import torch

from retrieve.modules import BloomFilter, ExactAttributeFilter
from retrieve.modules.linr import LiNRV2, LiNRV3
from retrieve.modules.silvertorch import SilverTorch
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs
from tests.parity.conftest import assert_topk_equal, assert_topk_matches

N, B, K = 20_000, 16, 100


def silvertorch(backend, embs, attrs, mode):
    kw = {"filter_mode": mode, "m_bits": 1024, "k_hash": 3} if mode == "bloom" else {}
    if mode == "exact":
        kw = {"filter_mode": "exact"}
    m = SilverTorch(k=K, n_lists=64, n_probe=8, n_iter=3, backend=backend, **kw)
    m.register_index(embs, attrs if mode != "none" else None)
    return m


def linr(cls, backend, embs, attrs, filt):
    f = {
        "bloom": lambda: BloomFilter(m_bits=1024, k_hash=3, backend=backend),
        "exact": lambda: ExactAttributeFilter(backend=backend),
    }[filt]()
    kw = {"candidate_pool": 2000} if cls is LiNRV3 else {}
    m = cls(k=K, filter=f, backend=backend, **kw).cuda()
    m.register_index(embs, attrs)
    return m


def run(m, q, qa, compiled):
    fn = torch.compile(m) if compiled else m
    return fn(q, qa) if qa is not None else fn(q)


for d in (192, 768):
    embs, q = make_index(N, d), make_query(B, d)
    attrs, qa = make_attrs(N, c=2, a_max=3), make_query_attrs(B, c=2)
    cases = {
        **{
            f"silvertorch/{mode}": (
                lambda b, mode=mode: silvertorch(b, embs, attrs, mode),
                None if mode == "none" else qa,
                "exact",
            )
            for mode in ("none", "bloom", "exact")
        },
        **{
            f"{cls.__name__}/{filt}": (
                lambda b, cls=cls, filt=filt: linr(cls, b, embs, attrs, filt),
                qa,
                "fp",
            )
            for cls in (LiNRV2, LiNRV3)
            for filt in ("exact", "bloom")
        },
    }
    for name, (build, qattrs, gate) in cases.items():
        ref = run(build("torch"), q, qattrs, compiled=False)
        tri = build("triton")
        for compiled in (False, True):
            out = run(tri, q, qattrs, compiled)
            if gate == "exact":
                assert_topk_equal(*out, *ref)
            else:
                # fp32 tl.sum vs cuBLAS bmm: the fused_masked_knn_topk parity file's tolerance.
                assert_topk_matches(*out, *ref, atol=1e-6, rtol=0.0)
            fin = torch.isfinite(out[1]).float().mean().item()
            print(
                f"D={d} {name:22s} compiled={compiled!s:5s} ok  finite={fin:.2f}",
                flush=True,
            )
print("all layer cases passed", file=sys.stderr)
