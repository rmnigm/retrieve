"""Two passes, filter at word level with grouped bit loads; dot persistent over passing tiles."""

import torch
import triton
import triton.language as tl
from retrieve.ops.triton import common
from retrieve.ops.triton.common import probe_dots, probe_tile_table

TPP = 4
NPROG = 108 * 8
FILTER_ONLY = False
FW = 4


@triton.jit
def _filter_kernel(
    table_ptr,
    qpos_ptr,
    bloom_t_ptr,
    out_scores_ptr,
    list_ptr,
    count_ptr,
    n_probe,
    width,
    T,
    n_qbits,
    stride_qpos,
    stride_tm,
    stride_ob,
    FAN: tl.constexpr,
    TPP: tl.constexpr,
    BLOCK_P: tl.constexpr,
    NW: tl.constexpr,
):
    bid = tl.program_id(0)
    tt = tl.program_id(1) * TPP + tl.arange(0, TPP)
    row = table_ptr + bid * 3 * n_probe
    n_tiles = tl.load(row + n_probe - 1)
    total = tl.load(row + 2 * n_probe - 1)
    f = tl.arange(0, FAN)
    ends = (f + 1) * FAN - 1
    e = tl.load(row + ends, mask=ends < n_probe, other=1 << 40)
    j = tl.sum((e[None, :] <= tt[:, None]).to(tl.int32), 1) * FAN
    idx = j[:, None] + f[None, :]
    e = tl.load(row + idx, mask=idx < n_probe, other=1 << 40)
    j += tl.sum((e <= tt[:, None]).to(tl.int32), 1)
    tail = tt >= n_tiles
    inside = j < n_probe
    first = j > 0
    tile_start = tl.load(row + j - 1, mask=inside & first, other=0)
    slot_start = tl.load(row + n_probe + j - 1, mask=inside & first, other=0)
    size = tl.load(row + n_probe + j, mask=inside, other=0) - slot_start
    lo = tl.load(row + 2 * n_probe + j, mask=inside, other=0)
    off = (tt - tile_start) * BLOCK_P
    lane = tl.arange(0, BLOCK_P)
    slot = tl.where(
        tail[:, None],
        (total + (tt - n_tiles) * BLOCK_P)[:, None] + lane[None, :],
        (slot_start + off)[:, None] + lane[None, :],
    )
    n_v = tl.where(tail, 0, tl.minimum(size - off, BLOCK_P))
    valid = lane[None, :] < n_v[:, None]
    live = tt < T
    tl.store(
        out_scores_ptr + bid * stride_ob + slot,
        tl.full([TPP, BLOCK_P], float("-inf"), tl.float32),
        mask=(valid | (tail[:, None] & (slot < width))) & live[:, None],
    )
    pos0 = lo + off
    pos1 = pos0 + n_v
    wi = (pos0 >> 6)[:, None] + tl.arange(0, NW)[None, :]
    lo_b = tl.minimum(tl.maximum(pos0[:, None] - wi * 64, 0), 64)
    hi_b = tl.minimum(tl.maximum(pos1[:, None] - wi * 64, 0), 64)
    one = tl.full([TPP, NW], 1, tl.int64)
    span = tl.where(hi_b >= 64, -1, (one << hi_b) - 1) & ~tl.where(
        lo_b >= 64, -1, (one << lo_b) - 1
    )
    ok = span != 0
    wacc = span
    for i0 in range(0, n_qbits, 4):
        m0 = tl.load(qpos_ptr + bid * stride_qpos + i0)
        m1 = tl.load(
            qpos_ptr + bid * stride_qpos + i0 + 1, mask=i0 + 1 < n_qbits, other=-1
        )
        m2 = tl.load(
            qpos_ptr + bid * stride_qpos + i0 + 2, mask=i0 + 2 < n_qbits, other=-1
        )
        m3 = tl.load(
            qpos_ptr + bid * stride_qpos + i0 + 3, mask=i0 + 3 < n_qbits, other=-1
        )
        if (m0 >= 0) | (m1 >= 0) | (m2 >= 0) | (m3 >= 0):
            v0 = tl.load(
                bloom_t_ptr + m0 * stride_tm + wi, mask=ok & (m0 >= 0), other=-1
            )
            v1 = tl.load(
                bloom_t_ptr + m1 * stride_tm + wi, mask=ok & (m1 >= 0), other=-1
            )
            v2 = tl.load(
                bloom_t_ptr + m2 * stride_tm + wi, mask=ok & (m2 >= 0), other=-1
            )
            v3 = tl.load(
                bloom_t_ptr + m3 * stride_tm + wi, mask=ok & (m3 >= 0), other=-1
            )
            wacc = wacc & v0 & v1 & v2 & v3
    any_pass = (tl.max((wacc != 0).to(tl.int32), 1) > 0) & live
    k = tl.atomic_add(count_ptr + tl.zeros([TPP], tl.int32), 1, mask=any_pass)
    tl.store(list_ptr + k, bid.to(tl.int64) * T + tt, mask=any_pass)


@triton.jit
def _dot_kernel(
    table_ptr,
    list_ptr,
    count_ptr,
    q_codes_ptr,
    q_scales_ptr,
    item_codes_ptr,
    qpos_ptr,
    bloom_t_ptr,
    out_scores_ptr,
    global_scale,
    n_probe,
    T,
    n_qbits,
    stride_qcb,
    stride_cn,
    stride_qpos,
    stride_tm,
    stride_ob,
    D: tl.constexpr,
    D_PAD: tl.constexpr,
    FAN: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NPROG: tl.constexpr,
):
    n = tl.load(count_ptr)
    for i in range(tl.program_id(0), n, NPROG):
        e = tl.load(list_ptr + i)
        bid = (e // T).to(tl.int32)
        t = (e % T).to(tl.int32)
        pos, slot, valid, tail = probe_tile_table(
            table_ptr, bid, t, n_probe, FAN, BLOCK_P
        )
        word = pos >> 6
        bit = pos & 63
        acc = tl.full(pos.shape, -1, tl.int64)
        for i0 in range(0, n_qbits, 4):
            m0 = tl.load(qpos_ptr + bid * stride_qpos + i0)
            m1 = tl.load(
                qpos_ptr + bid * stride_qpos + i0 + 1, mask=i0 + 1 < n_qbits, other=-1
            )
            m2 = tl.load(
                qpos_ptr + bid * stride_qpos + i0 + 2, mask=i0 + 2 < n_qbits, other=-1
            )
            m3 = tl.load(
                qpos_ptr + bid * stride_qpos + i0 + 3, mask=i0 + 3 < n_qbits, other=-1
            )
            if (m0 >= 0) | (m1 >= 0) | (m2 >= 0) | (m3 >= 0):
                w0 = tl.load(
                    bloom_t_ptr + m0 * stride_tm + word,
                    mask=valid & (m0 >= 0),
                    other=-1,
                )
                w1 = tl.load(
                    bloom_t_ptr + m1 * stride_tm + word,
                    mask=valid & (m1 >= 0),
                    other=-1,
                )
                w2 = tl.load(
                    bloom_t_ptr + m2 * stride_tm + word,
                    mask=valid & (m2 >= 0),
                    other=-1,
                )
                w3 = tl.load(
                    bloom_t_ptr + m3 * stride_tm + word,
                    mask=valid & (m3 >= 0),
                    other=-1,
                )
                acc = acc & w0 & w1 & w2 & w3
        keep = valid & (((acc >> bit) & 1) != 0)
        dots_i32, q_scale = probe_dots(
            q_codes_ptr + bid * stride_qcb,
            q_scales_ptr,
            bid,
            item_codes_ptr,
            pos,
            stride_cn,
            keep,
            D,
            D_PAD,
            BLOCK_D,
            BLOCK_P,
        )
        dots = (
            dots_i32.to(tl.float32)
            * tl.cast(q_scale, tl.float32)
            * tl.cast(global_scale, tl.float32)
        )
        dots = tl.where(keep, dots, float("-inf"))
        tl.store(out_scores_ptr + bid * stride_ob + slot, dots, mask=valid)


BP = None
BD = None


def run(la):
    kw = dict(la.kwargs)
    bp = BP or kw["BLOCK_P"]
    tk = dict(la.table.kwargs)
    if BP:
        tk["BLOCK_P"] = bp
        kw["BLOCK_D"] = BD or kw["BLOCK_D"]
    common.probe_table_kernel[la.table.grid](**tk)
    b = la.grid[0]
    T = triton.cdiv(kw["width"], bp) + kw["n_probe"]
    lst = torch.empty(b * T, dtype=torch.int64, device="cuda")
    cnt = torch.zeros(1, dtype=torch.int32, device="cuda")
    _filter_kernel[(b, triton.cdiv(T, TPP))](
        kw["table_ptr"],
        kw["qpos_ptr"],
        kw["bloom_t_ptr"],
        kw["out_scores_ptr"],
        lst,
        cnt,
        kw["n_probe"],
        kw["width"],
        T,
        kw["n_qbits"],
        kw["stride_qpos"],
        kw["stride_tm"],
        kw["stride_ob"],
        FAN=kw["FAN"],
        TPP=TPP,
        BLOCK_P=bp,
        NW=8 if bp == 256 else (4 if bp == 128 else 2),
        num_warps=FW,
    )
    if FILTER_ONLY:
        return la.all_scores, None
    k = _dot_kernel[(NPROG,)](
        kw["table_ptr"],
        lst,
        cnt,
        kw["q_codes_ptr"],
        kw["q_scales_ptr"],
        kw["item_codes_ptr"],
        kw["qpos_ptr"],
        kw["bloom_t_ptr"],
        kw["out_scores_ptr"],
        kw["global_scale"],
        kw["n_probe"],
        T,
        kw["n_qbits"],
        kw["stride_qcb"],
        kw["stride_cn"],
        kw["stride_qpos"],
        kw["stride_tm"],
        kw["stride_ob"],
        D=kw["D"],
        D_PAD=kw["D_PAD"],
        FAN=kw["FAN"],
        BLOCK_P=bp,
        BLOCK_D=kw["BLOCK_D"],
        NPROG=NPROG,
        num_warps=kw["num_warps"],
        num_stages=kw["num_stages"],
    )
    return la.all_scores, k
