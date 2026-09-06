"""WP-1 self-check: the CuTe DSL backend vs the CUDA C++ backend, bit-exact on the full [B, P]
score buffers and on the mask words, over modes x D x B x unroll x max_size, exact (C, A) shapes,
the topk `_impl` path, and a torch.compile(fullgraph=True) smoke of each custom op.

Run: cd /workspace/retrieve/retrieve && uv run python <this file>
"""

from __future__ import annotations

import sys
import time
import traceback

sys.path.insert(0, "/workspace/retrieve/retrieve")

import torch  # noqa: E402

# Full-module-path imports: the package __init__ re-exports the cuda *op* under the module's name.
import importlib  # noqa: E402

cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds  # noqa: E402
from retrieve.layers.utils.quantize import quantize_int8_global  # noqa: E402
from tests.conftest import make_attrs, make_index, make_query, make_query_attrs  # noqa: E402

# --- probe family construction, as in tests/parity/test_codesigned_probe_score_cuda.py ----


def _make_probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1, seed=7):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n = n_lists * max_size
    padded = torch.randperm(n, generator=g, device="cuda").reshape(n_lists, max_size)
    pad = torch.rand(n_lists, max_size, generator=g, device="cuda") < pad_rate
    padded[pad] = -1
    probe_ids = torch.randint(0, n_lists, (b, n_probe), generator=g, device="cuda")
    flat = padded[probe_ids].reshape(b, -1)
    return padded, probe_ids, flat, n


def _make_bloom(n, b, padded, *, m_bits=512, k_hash=5):
    attrs = make_attrs(n, c=2, a_max=2)
    q_attrs = make_query_attrs(b, c=2)
    seeds = generate_seeds(k_hash=k_hash, device=attrs.device)
    w = m_bits // 64
    sigs = build_signatures(attrs.long(), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    qb = build_signatures(
        q_attrs.long().unsqueeze(-1), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w
    )
    sigs_t = cu.build_transposed_sigs(sigs, padded)
    return sigs, sigs_t, qb


def _make_exact(n, b, *, c=2, a_max=2, reverse="none", n_vocab=8):
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab).long()
    q_attrs = make_query_attrs(b, c=c, n_vocab=n_vocab).long()
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    if reverse == "mixed":
        rev[0] = True
    return attrs, rev, q_attrs


# --- one case: both backends' mask words and full score buffers ----------------------------

CFG_BY_UNROLL = {1: (128, 8), 2: (256, 4), 4: (64, 2)}


def run_case(mode, d, b, unroll, max_size, *, c=2, a_max=2, reverse="none", n_lists=16, n_probe=8):
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, gs = quantize_int8_global(embs)
    query = make_query(b, d)
    ms = max_size if mode != "none" else 0

    # phase 2 masks
    if mode == "bloom":
        _, sigs_t, qb = _make_bloom(n, b, padded)
        m_cuda = cu._bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size)
        m_cute = ct._bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size)
    elif mode == "exact":
        attrs, rev, q_attrs = _make_exact(n, b, c=c, a_max=a_max, reverse=reverse)
        m_cuda = cu._clause_partial_mask_cuda_impl(flat, attrs, rev, q_attrs, max_size)
        m_cute = ct._clause_partial_mask_cute_impl(flat, attrs, rev, q_attrs, max_size)
    else:
        m_cuda = m_cute = torch.empty(1, 1, dtype=torch.int64, device="cuda")
    torch.cuda.synchronize()
    mask_ok = torch.equal(m_cuda, m_cute)

    # phase 3 into full [B, P] buffers (cuda with its default config: the reference is
    # config-independent; cute with a config that varies with unroll)
    q_codes, q_scales, flat_c, s_cuda, has_mask, _ = cu._cps_cuda_prep(
        query, flat, codes, query_bits=None, bloom_sigs_t=None, probe_ids=None
    )
    has_mask = mode != "none"
    cu._load_ext().cps_scores(
        q_codes, q_scales, m_cuda, flat_c, codes, s_cuda, float(gs), has_mask, ms, 128, 8, 1
    )
    s_cute = torch.empty_like(s_cuda)
    block_p, num_warps = CFG_BY_UNROLL[unroll]
    ct._cps_cute_scores(
        q_codes,
        q_scales,
        m_cute,
        flat_c,
        codes,
        s_cute,
        global_scale=float(gs),
        has_mask=has_mask,
        max_size=ms,
        cfg=ct.CodesignedProbeScoreCuteConfig(block_p, num_warps, unroll),
    )
    torch.cuda.synchronize()
    scores_ok = torch.equal(s_cuda, s_cute)
    n_inf = int(torch.isinf(s_cuda).sum())
    return mask_ok, scores_ok, s_cuda.numel(), n_inf


def run_impl_case(mode, d, b, max_size, n_lists=16, n_probe=8):
    """The eager `_impl`s with k = P: scores sorted identically."""
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, gs = quantize_int8_global(embs)
    query = make_query(b, d)
    k = flat.shape[1]
    if mode == "none":
        a = cu._codesigned_probe_score_cuda_impl(query, flat, codes, gs, k)
        c = ct._codesigned_probe_score_cute_impl(query, flat, codes, gs, k)
    elif mode == "bloom":
        _, sigs_t, qb = _make_bloom(n, b, padded)
        kw = dict(query_bits=qb, bloom_sigs_t=sigs_t, probe_ids=probe_ids)
        a = cu._codesigned_probe_score_cuda_impl(query, flat, codes, gs, k, **kw)
        c = ct._codesigned_probe_score_cute_impl(query, flat, codes, gs, k, **kw)
    else:
        attrs, rev, q_attrs = _make_exact(n, b)
        kw = dict(
            item_clause_attrs=attrs, clause_is_reverse=rev, query_clause_attrs=q_attrs,
            max_size=max_size,
        )
        a = cu._codesigned_probe_score_exact_cuda_impl(query, flat, codes, gs, k, **kw)
        c = ct._codesigned_probe_score_exact_cute_impl(query, flat, codes, gs, k, **kw)
    return torch.equal(a[1], c[1])


def run_compile_smoke():
    b, n_lists, max_size, n_probe, d, k = 2, 8, 64, 2, 128, 4
    padded, probe_ids, flat, n = _make_probe_family(b, n_lists, max_size, n_probe)
    embs = make_index(n, d)
    codes, gs = quantize_int8_global(embs)
    query = make_query(b, d)
    _, sigs_t, qb = _make_bloom(n, b, padded)
    attrs, rev, q_attrs = _make_exact(n, b)
    ok = {}
    f = torch.compile(lambda q: ct.codesigned_probe_score_cute(q, flat, codes, gs, k), fullgraph=True)
    ok["none"] = torch.equal(f(query)[1], ct.codesigned_probe_score_cute(query, flat, codes, gs, k)[1])
    f = torch.compile(
        lambda q: ct.codesigned_probe_score_bloom_cute(q, flat, codes, qb, sigs_t, probe_ids, gs, k),
        fullgraph=True,
    )
    ok["bloom"] = torch.equal(
        f(query)[1],
        ct.codesigned_probe_score_bloom_cute(query, flat, codes, qb, sigs_t, probe_ids, gs, k)[1],
    )
    f = torch.compile(
        lambda q: ct.codesigned_probe_score_exact_cute(
            q, flat, codes, attrs, rev, q_attrs, gs, k, max_size
        ),
        fullgraph=True,
    )
    ok["exact"] = torch.equal(
        f(query)[1],
        ct.codesigned_probe_score_exact_cute(query, flat, codes, attrs, rev, q_attrs, gs, k, max_size)[1],
    )
    return ok


def main():
    t0 = time.perf_counter()
    cu.ensure_built()
    ct.ensure_built(verbose=True)
    rows = []
    fails = 0

    def record(name, ok, extra=""):
        nonlocal fails
        fails += not ok
        rows.append((name, "PASS" if ok else "FAIL", extra))

    # 1. modes x D x B x unroll x max_size (exact at C=2, A=2, reverse=mixed)
    for mode in ("none", "bloom", "exact"):
        for d in (64, 128, 256, 96):
            for b in (1, 16):
                for unroll in (1, 2, 4):
                    for max_size in (3, 64, 100, 768):
                        name = f"{mode:5s} D={d:3d} B={b:2d} unroll={unroll} max_size={max_size}"
                        try:
                            m_ok, s_ok, numel, n_inf = run_case(
                                mode, d, b, unroll, max_size, reverse="mixed"
                            )
                            record(name, m_ok and s_ok, f"mask={m_ok} scores={s_ok} P*B={numel} -inf={n_inf}")
                        except Exception as e:
                            record(name, False, f"EXC {type(e).__name__}: {e}")
                            traceback.print_exc()
    # 2. exact (C, A) table + reverse flags, incl. the (3, 4) off-table -> (0, 0) fallback
    for c, a in ((1, 1), (2, 2), (3, 2), (4, 2), (3, 4)):
        for reverse in ("none", "mixed"):
            for d in (128, 96):
                name = f"exact C={c} A={a} rev={reverse:5s} D={d}"
                try:
                    m_ok, s_ok, numel, n_inf = run_case(
                        "exact", d, 16, 1, 100, c=c, a_max=a, reverse=reverse
                    )
                    record(name, m_ok and s_ok, f"mask={m_ok} scores={s_ok} -inf={n_inf}")
                except Exception as e:
                    record(name, False, f"EXC {type(e).__name__}: {e}")
                    traceback.print_exc()
    # 3. eager _impl path (k = P)
    for mode in ("none", "bloom", "exact"):
        for d in (64, 96):
            name = f"_impl {mode} D={d} k=P"
            try:
                record(name, run_impl_case(mode, d, 4, 96))
            except Exception as e:
                record(name, False, f"EXC {type(e).__name__}: {e}")
                traceback.print_exc()
    # 4. torch.compile smoke
    try:
        for mode, ok in run_compile_smoke().items():
            record(f"torch.compile fullgraph {mode}", ok)
    except Exception as e:
        record("torch.compile fullgraph", False, f"EXC {type(e).__name__}: {e}")
        traceback.print_exc()

    width = max(len(r[0]) for r in rows)
    for name, status, extra in rows:
        print(f"{name:{width}s}  {status}  {extra}")
    n = len(rows)
    print(f"\n{n - fails}/{n} passed, {fails} failed, {time.perf_counter() - t0:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
