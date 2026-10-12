"""WP-3 edge-case micro tests: cute vs cuda, torch.equal on full score buffers + mask words,
on the shapes the WP-1 self-check did not reach.

Run: cd /workspace/retrieve/retrieve && uv run python /tmp/.../wp3/edge.py
"""

from __future__ import annotations

import importlib
import sys
import traceback

sys.path.insert(0, "/workspace/retrieve/retrieve")

import torch  # noqa: E402

cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
from retrieve.layers.filters.bloom_hash import build_signatures, generate_seeds  # noqa: E402
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global  # noqa: E402

DEV = "cuda"
rows: list[tuple[str, bool, str]] = []


def record(name, ok, extra=""):
    rows.append((name, ok, extra))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {extra}", flush=True)


def probe_family(b, n_lists, max_size, n_probe, *, pad_rate=0.1, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = n_lists * max_size
    padded = torch.randperm(n, generator=g, device=DEV).reshape(n_lists, max_size)
    pad = torch.rand(n_lists, max_size, generator=g, device=DEV) < pad_rate
    padded[pad] = -1
    probe_ids = torch.randint(0, n_lists, (b, n_probe), generator=g, device=DEV)
    flat = padded[probe_ids].reshape(b, -1)
    return padded, probe_ids, flat, n


def rand_attrs(n, c, a_max, n_vocab=8, seed=3, pad_rate=0.3):
    g = torch.Generator(device=DEV).manual_seed(seed)
    attrs = torch.randint(0, n_vocab, (n, c, a_max), generator=g, dtype=torch.long, device=DEV)
    pad = torch.rand(n, c, a_max, generator=g, device=DEV) < pad_rate
    attrs[pad] = -1
    return attrs


def rand_query_attrs(b, c, n_vocab=8, seed=4, inactive_rate=0.2):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randint(0, n_vocab, (b, c), generator=g, dtype=torch.long, device=DEV)
    q[torch.rand(b, c, generator=g, device=DEV) < inactive_rate] = -1
    return q


def make_bloom(n, b, padded, *, m_bits=512, k_hash=5):
    attrs = rand_attrs(n, 2, 2)
    q_attrs = rand_query_attrs(b, 2)
    seeds = generate_seeds(k_hash=k_hash, device=attrs.device)
    w = m_bits // 64
    sigs = build_signatures(attrs, seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    qb = build_signatures(q_attrs.unsqueeze(-1), seeds, m_bits=m_bits, k_hash=k_hash, word_count=w)
    return sigs, cu.build_transposed_sigs(sigs, padded), qb


def scores_both(q_codes, q_scales, m_cuda, m_cute, flat, codes, gs, has_mask, ms, cfg):
    b, p = flat.shape
    s_cuda = torch.empty((b, p), dtype=torch.float32, device=DEV)
    s_cute = torch.empty_like(s_cuda)
    cu._load_ext().cps_scores(
        q_codes, q_scales, m_cuda, flat, codes, s_cuda, float(gs), has_mask, ms,
        cfg.block_p, cfg.num_warps, cfg.unroll,
    )
    ct._cps_cute_scores(
        q_codes, q_scales, m_cute, flat, codes, s_cute,
        global_scale=float(gs), has_mask=has_mask, max_size=ms, cfg=cfg,
    )
    torch.cuda.synchronize()
    return s_cuda, s_cute


def case(name, *, mode, d, b, max_size, n_lists, n_probe, cfg, codes=None, query=None,
         c=2, a_max=2, rev=None, qb_override=None, seed=7):
    try:
        padded, probe_ids, flat, n = probe_family(b, n_lists, max_size, n_probe, seed=seed)
        if codes is None:
            g = torch.Generator(device=DEV).manual_seed(seed + 1)
            embs = torch.randn(n, d, generator=g, device=DEV)
            codes, gs = quantize_int8_global(embs)
        else:
            gs = 1.0 / 127.0
        if query is None:
            g = torch.Generator(device=DEV).manual_seed(seed + 2)
            query = torch.randn(b, d, generator=g, device=DEV)
        q_codes, q_scales = quantize_int8(query)
        q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
        dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
        if mode == "bloom":
            _, sigs_t, qb = make_bloom(n, b, padded)
            if qb_override is not None:
                qb = qb_override(qb)
            m_cuda = cu._bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size)
            m_cute = ct._bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size)
        elif mode == "exact":
            attrs = rand_attrs(n, c, a_max)
            q_attrs = rand_query_attrs(b, c)
            if rev is None:
                rev = torch.zeros(c, dtype=torch.bool, device=DEV)
            m_cuda = cu._clause_partial_mask_cuda_impl(flat, attrs, rev, q_attrs, max_size)
            m_cute = ct._clause_partial_mask_cute_impl(flat, attrs, rev, q_attrs, max_size)
        else:
            m_cuda = m_cute = dummy
        torch.cuda.synchronize()
        mask_ok = torch.equal(m_cuda, m_cute)
        has_mask = mode != "none"
        s_cuda, s_cute = scores_both(
            q_codes, q_scales, m_cuda, m_cute, flat, codes, gs, has_mask,
            max_size if has_mask else 0, cfg,
        )
        s_ok = torch.equal(s_cuda, s_cute)
        n_inf = int(torch.isinf(s_cuda).sum())
        n_neg = int((s_cuda[torch.isfinite(s_cuda)] < 0).sum())
        record(name, mask_ok and s_ok,
               f"mask={mask_ok} scores={s_ok} B*P={s_cuda.numel()} -inf={n_inf} neg={n_neg}")
        return m_cuda, m_cute, s_cuda, s_cute
    except Exception as e:
        record(name, False, f"EXC {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


Cfg = ct.CodesignedProbeScoreCuteConfig
C_DEF = Cfg(128, 8, 1)


def main():
    cu.ensure_built()
    ct.ensure_built(verbose=True)

    # T1: all-negative dot products (signed int->fp32 conversion) at every SEG + generic.
    for d in (64, 128, 256, 96):
        n_lists, max_size = 8, 64
        n = n_lists * max_size
        g = torch.Generator(device=DEV).manual_seed(11)
        query = torch.rand(4, d, generator=g, device=DEV) + 0.5          # all positive
        embs = -(torch.rand(n, d, generator=g, device=DEV) + 0.5)        # all negative
        codes, _ = quantize_int8_global(embs)
        for u in (1, 4):
            case(f"T1 neg-dots D={d} unroll={u}", mode="none", d=d, b=4, max_size=max_size,
                 n_lists=n_lists, n_probe=4, cfg=Cfg(128, 8, u), codes=codes, query=query)

    # T2: empty query signature -> bloom AND identity (~0 words incl. pad tail).
    def zero_qb(qb):
        return torch.zeros_like(qb)
    r = case("T2 empty QB identity max_size=100 (pad tail)", mode="bloom", d=128, b=3,
             max_size=100, n_lists=8, n_probe=5, cfg=C_DEF, qb_override=zero_qb)
    if r is not None:
        m_cuda, m_cute, *_ = r
        record("T2 all mask words == -1 (both)", bool((m_cuda == -1).all()) and bool((m_cute == -1).all()))

    # T3: INT64_MIN query word (only bit 63 set): bits & (bits - 1) wraparound on signed Int64.
    def top_bit_qb(qb):
        q = torch.zeros_like(qb)
        q[:, 0] = -(2**63)
        q[:, 1] = -(2**63) | 1
        return q
    case("T3 QB word = INT64_MIN (bit 63 walk)", mode="bloom", d=128, b=3, max_size=64,
         n_lists=8, n_probe=5, cfg=C_DEF, qb_override=top_bit_qb)

    # T4: mask_words not a multiple of 4 / of 256; > 256 (multi-block bloom; idle clause warps).
    for mode in ("bloom", "exact"):
        for n_probe, max_size in ((5, 65), (3, 1), (301, 64), (257, 3), (130, 129)):
            case(f"T4 {mode} n_probe={n_probe} max_size={max_size} (mask_words={n_probe * cu.words_per_cluster(max_size)})",
                 mode=mode, d=128, b=2, max_size=max_size, n_lists=max(8, n_probe), n_probe=n_probe,
                 cfg=Cfg(64, 2, 4))

    # T5: launch-geometry tails: items_per_warp < STEP, P % block_p != 0, num_warps = 1 / 32.
    for d in (64, 128, 256):
        for cfg in (Cfg(32, 32, 4), Cfg(8, 1, 4), Cfg(96, 3, 2), Cfg(1024, 32, 1), Cfg(16, 16, 2)):
            for mode in ("none", "bloom"):
                case(f"T5 {mode} D={d} cfg={cfg.block_p}/{cfg.num_warps}/{cfg.unroll} max_size=7",
                     mode=mode, d=d, b=3, max_size=7, n_lists=16, n_probe=11, cfg=cfg)

    # T6: misaligned code base pointers (8 B storage offset) -> generic kernel on both.
    for d in (128, 64):
        n_lists, max_size, n_probe, b = 8, 64, 4, 3
        n = n_lists * max_size
        g = torch.Generator(device=DEV).manual_seed(5)
        embs = torch.randn(n, d, generator=g, device=DEV)
        codes_al, _ = quantize_int8_global(embs)
        buf = torch.empty(n * d + 8, dtype=torch.int8, device=DEV)
        codes_mis = buf[8:].view(n, d)
        codes_mis.copy_(codes_al)
        assert codes_mis.is_contiguous() and codes_mis.data_ptr() % 16 == 8
        for mode in ("none", "bloom"):
            r = case(f"T6 {mode} D={d} item_codes base %16==8 (generic route)", mode=mode, d=d,
                     b=b, max_size=max_size, n_lists=n_lists, n_probe=n_probe, cfg=C_DEF,
                     codes=codes_mis)
            r2 = case(f"T6 {mode} D={d} aligned twin", mode=mode, d=d, b=b, max_size=max_size,
                      n_lists=n_lists, n_probe=n_probe, cfg=C_DEF, codes=codes_al)
            if r and r2:
                record(f"T6 {mode} D={d} generic == fast-path scores", torch.equal(r[3], r2[3]))
    # 1-byte misaligned: the cute host raises on the host, the C++ would hit a device fault.
    try:
        d, n = 128, 64
        buf = torch.empty(n * d + 1, dtype=torch.int8, device=DEV)
        codes_mis1 = buf[1:].view(n, d)
        q = torch.zeros(1, d, dtype=torch.int8, device=DEV)
        qs = torch.ones(1, dtype=torch.float32, device=DEV)
        flat = torch.arange(n, device=DEV).view(1, n)
        out = torch.empty(1, n, dtype=torch.float32, device=DEV)
        ct._cps_cute_scores(q, qs, torch.empty(1, 1, dtype=torch.int64, device=DEV), flat,
                            codes_mis1, out, global_scale=1.0, has_mask=False, max_size=0, cfg=C_DEF)
        record("T6b 1-byte misaligned item_codes", False, "no error raised")
    except Exception as e:
        record("T6b 1-byte misaligned item_codes raises on host", True, f"{type(e).__name__}: {str(e)[:80]}")

    # T7: exact-clause corner shapes: all-reverse, C=4/A=2 table edge, (0,0) with C=5, C=0.
    for (c, a, revk) in (((4, 2, "all"), (4, 4, "all"), (5, 1, "mixed"), (1, 4, "all"), (2, 3, "none"))):
        rev = torch.zeros(c, dtype=torch.bool, device=DEV)
        if revk == "all":
            rev[:] = True
        elif revk == "mixed":
            rev[::2] = True
        case(f"T7 exact C={c} A={a} rev={revk}", mode="exact", d=128, b=5, max_size=100,
             n_lists=8, n_probe=6, cfg=C_DEF, c=c, a_max=a, rev=rev)
    # C = 0 clauses: keep = id >= 0 (runtime (0,0) path with n_clauses = 0)
    try:
        b, n_lists, max_size, n_probe = 2, 8, 64, 3
        padded, probe_ids, flat, n = probe_family(b, n_lists, max_size, n_probe)
        attrs = torch.empty(n, 0, 1, dtype=torch.long, device=DEV)
        q_attrs = torch.empty(b, 0, dtype=torch.long, device=DEV)
        rev = torch.empty(0, dtype=torch.bool, device=DEV)
        m_cuda = cu._clause_partial_mask_cuda_impl(flat, attrs, rev, q_attrs, max_size)
        m_cute = ct._clause_partial_mask_cute_impl(flat, attrs, rev, q_attrs, max_size)
        torch.cuda.synchronize()
        record("T7 exact C=0 (n_clauses=0)", torch.equal(m_cuda, m_cute))
    except Exception as e:
        record("T7 exact C=0", False, f"EXC {type(e).__name__}: {e}")
        traceback.print_exc()

    # T8: B = 65535 (grid.y limit) with tiny P; B = 65536 must raise on both.
    case("T8 B=65535 bloom", mode="bloom", d=64, b=65535, max_size=3, n_lists=4, n_probe=2, cfg=Cfg(8, 1, 1))
    case("T8 B=65535 exact", mode="exact", d=64, b=65535, max_size=3, n_lists=4, n_probe=2, cfg=Cfg(8, 1, 1))
    for be, impl in (("cuda", cu), ("cute", ct)):
        try:
            padded, probe_ids, flat, n = probe_family(65536, 4, 3, 2)
            _, sigs_t, qb = make_bloom(n, 65536, padded)
            getattr(impl, f"_bloom_partial_mask_{be}_impl")(qb, sigs_t, probe_ids, 3)
            torch.cuda.synchronize()
            record(f"T8 B=65536 {be} raises", False, "no error")
        except Exception as e:
            record(f"T8 B=65536 {be} raises", True, f"{type(e).__name__}: {str(e)[:60]}")

    # T9: a realistic regime: N=131072, P=8192, B=16, D=128, all modes, unroll 1/2/4.
    for mode in ("none", "bloom", "exact"):
        for u, cfg in ((1, Cfg(128, 8, 1)), (2, Cfg(256, 4, 2)), (4, Cfg(64, 2, 4))):
            case(f"T9 {mode} N=131072 P=8192 B=16 unroll={u}", mode=mode, d=128, b=16, max_size=128,
                 n_lists=1024, n_probe=64, cfg=cfg)

    # T10: same cluster probed twice by one query (duplicate probe_ids) — same mask words.
    try:
        b, n_lists, max_size, n_probe = 2, 8, 70, 4
        padded, _, _, n = probe_family(b, n_lists, max_size, n_probe)
        probe_ids = torch.tensor([[3, 3, 3, 3], [7, 0, 7, 0]], device=DEV)
        flat = padded[probe_ids].reshape(b, -1)
        _, sigs_t, qb = make_bloom(n, b, padded)
        m_cuda = cu._bloom_partial_mask_cuda_impl(qb, sigs_t, probe_ids, max_size)
        m_cute = ct._bloom_partial_mask_cute_impl(qb, sigs_t, probe_ids, max_size)
        torch.cuda.synchronize()
        record("T10 duplicate probe ids", torch.equal(m_cuda, m_cute))
    except Exception as e:
        record("T10 duplicate probe ids", False, f"EXC {type(e).__name__}: {e}")

    # T11: side stream + stream cache: run on a non-default torch stream.
    try:
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            r = case("T11 side stream bloom", mode="bloom", d=128, b=4, max_size=64, n_lists=8,
                     n_probe=4, cfg=C_DEF)
        torch.cuda.synchronize()
        from retrieve.kernels.silvertorch.cute import codesigned_probe_score as dev
        record("T11 stream cache has 2 entries (default + side)", len(dev._streams) == 2,
               f"keys={list(dev._streams)}")
    except Exception as e:
        record("T11 side stream", False, f"EXC {type(e).__name__}: {e}")

    fails = sum(not ok for _, ok, _ in rows)
    print(f"\n{len(rows) - fails}/{len(rows)} passed, {fails} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
