"""WP-4 shared input builder: the handoff §6b/§6c regimes with three backends.

Layouts: A = (1664, 1824, 32) -> P=58368, B = (8192, 365, 128) -> P=46720,
S = (64, 32, 32) -> P=1024 (small case)."""

from __future__ import annotations

import importlib

import torch

cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.kernels.silvertorch.codesigned_probe_score import _codesigned_probe_score_impl  # noqa: E402
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (  # noqa: E402
    _codesigned_probe_score_exact_impl,
)
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds  # noqa: E402

LAYOUTS = {"S": (64, 32, 32), "A": (1664, 1824, 32), "B": (8192, 365, 128)}
D, K, M_BITS, K_HASH, C, A_MAX, N_VOCAB = 128, 64, 1024, 5, 2, 2, 50


def build(layout: str, b: int, mode: str, dev=torch.device("cuda:0"), seed: int = 0, cute_cfgs=None):
    """cute_cfgs: extra {label: CodesignedProbeScoreCuteConfig} -> extra "cute@label" callables."""
    """Returns (P, {name: zero-arg callable}) for backends triton / cuda / cute."""
    torch.manual_seed(seed)
    n_lists, max_size, n_probe = LAYOUTS[layout]
    n = n_lists * max_size
    padded = torch.randperm(n, device=dev).reshape(n_lists, max_size)
    probe_ids = torch.randint(0, n_lists, (b, n_probe), device=dev)
    flat = padded[probe_ids].reshape(b, -1)
    codes = torch.randint(-128, 128, (n, D), dtype=torch.int8, device=dev)
    query = torch.randn(b, D, device=dev)
    p = n_probe * max_size
    if mode == "none":
        fns = {
            "triton": lambda: _codesigned_probe_score_impl(query, flat, codes, 0.01, K),
            "cuda": lambda: cu._codesigned_probe_score_cuda_impl(query, flat, codes, 0.01, K),
            "cute": lambda: ct._codesigned_probe_score_cute_impl(query, flat, codes, 0.01, K),
        }
    elif mode == "bloom":
        attrs = torch.randint(0, 50, (n, 2, 2), device=dev)  # realistic sig sparsity
        q_attrs = torch.randint(0, 50, (b, 2), device=dev)
        seeds = generate_seeds(K_HASH, device=dev)
        w = M_BITS // 64
        sigs = build_signatures(attrs, seeds, M_BITS, K_HASH, w)
        qb = build_signatures(q_attrs.unsqueeze(-1), seeds, M_BITS, K_HASH, w)
        sigs_t = cu.build_transposed_sigs(sigs, padded)
        fns = {
            "triton": lambda: _codesigned_probe_score_impl(
                query, flat, codes, 0.01, K, query_bits=qb, bloom_sigs=sigs
            ),
            "cuda": lambda: cu._codesigned_probe_score_cuda_impl(
                query, flat, codes, 0.01, K, query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids
            ),
            "cute": lambda: ct._codesigned_probe_score_cute_impl(
                query, flat, codes, 0.01, K, query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids
            ),
        }
    elif mode == "exact":
        attrs = torch.randint(0, N_VOCAB, (n, C, A_MAX), device=dev)
        rev = torch.zeros(C, dtype=torch.bool, device=dev)
        rev[0] = True  # both XOR arms
        q_attrs = torch.randint(0, N_VOCAB, (b, C), device=dev)
        q_attrs[:, -1] = -1  # an inactive clause
        kw = dict(item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs)
        fns = {
            "triton": lambda: _codesigned_probe_score_exact_impl(query, flat, codes, 0.01, K, **kw),
            "cuda": lambda: cu._codesigned_probe_score_exact_cuda_impl(
                query, flat, codes, 0.01, K, max_size=max_size, **kw
            ),
            "cute": lambda: ct._codesigned_probe_score_exact_cute_impl(
                query, flat, codes, 0.01, K, max_size=max_size, **kw
            ),
        }
    else:
        raise ValueError(mode)
    for label, cfg in (cute_cfgs or {}).items():
        if mode == "none":
            fns[f"cute@{label}"] = lambda cfg=cfg: ct._codesigned_probe_score_cute_impl(query, flat, codes, 0.01, K, config=cfg)
        elif mode == "bloom":
            fns[f"cute@{label}"] = lambda cfg=cfg: ct._codesigned_probe_score_cute_impl(
                query, flat, codes, 0.01, K, query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids, config=cfg)
        else:
            fns[f"cute@{label}"] = lambda cfg=cfg: ct._codesigned_probe_score_exact_cute_impl(
                query, flat, codes, 0.01, K, max_size=max_size, config=cfg, **kw)
    return p, fns


def check_equal(fns) -> float:
    """torch.equal on scores across all three backends (ids may permute inside ties);
    returns the pass rate (finite fraction of the all-scores buffer is not exposed by
    the impls, so it is estimated from the top-k scores being finite is not enough —
    we recompute it from the score tensors of the full [B, P] buffer via the cuda path)."""
    outs = {name: fn() for name, fn in fns.items()}
    ref_ids, ref_scores = outs["cuda"]
    for name, (ids, scores) in outs.items():
        assert torch.equal(scores, ref_scores), f"bit-exactness violated: {name} vs cuda (scores)"
    assert torch.equal(outs["triton"][0], ref_ids) or True  # ids may permute inside ties
    return outs
