"""Two passes: (1) filter: per program TPP tiles of one row, -inf for every slot, bloom test, the passing tiles
appended to a list; (2) persistent dot over the list."""

import torch
import triton
import triton.language as tl
from retrieve.ops.triton import common
from retrieve.ops.triton.common import probe_dots, probe_tile_table, row_base

TPP = 4
FILTER_ONLY = False
P = 108 * 8


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
    STORE: tl.constexpr = True,
    BLOOM: tl.constexpr = True,
    ATOM: tl.constexpr = True,
    NQ: tl.constexpr = 0,
    MEM: tl.constexpr = True,
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
    pos = (lo + off)[:, None] + lane[None, :]
    valid = ((off[:, None] + lane[None, :]) < size[:, None]) & ~tail[:, None]
    out_row = row_base(out_scores_ptr, bid, stride_ob, False)
    live = tt < T
    if STORE:
        tl.store(
            out_row + slot,
            tl.full([TPP, BLOCK_P], float("-inf"), tl.float32),
            mask=(valid | (tail[:, None] & (slot < width))) & live[:, None],
        )
    keep = valid
    word = pos >> 6
    bit = pos & 63
    if BLOOM:
        for i in range(n_qbits if NQ == 0 else NQ):
            m = tl.load(qpos_ptr + bid * stride_qpos + i)
            w = tl.load(
                bloom_t_ptr + m * stride_tm + word,
                mask=valid & (m >= 0) & MEM,
                other=-1,
            )
            keep = keep & (((w >> bit) & 1) != 0)
    any_pass = (tl.max(keep.to(tl.int32), 1) > 0) & live
    if ATOM:
        k = tl.atomic_add(count_ptr + tl.zeros([TPP], tl.int32), 1, mask=any_pass)
        tl.store(list_ptr + k, bid.to(tl.int64) * T + tt, mask=any_pass)
    else:
        tl.store(list_ptr + bid * T + tt, any_pass.to(tl.int64), mask=live)


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
        keep = valid
        word = pos >> 6
        bit = pos & 63
        for q in range(n_qbits):
            m = tl.load(qpos_ptr + bid * stride_qpos + q)
            w = tl.load(
                bloom_t_ptr + m * stride_tm + word, mask=valid & (m >= 0), other=-1
            )
            keep = keep & (((w >> bit) & 1) != 0)
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


FLAGS = {}


def run(la):
    kw = la.kwargs
    common.probe_table_kernel[la.table.grid](**la.table.kwargs)
    b, ty, tx = la.grid
    T = ty * tx
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
        BLOCK_P=kw["BLOCK_P"],
        num_warps=4,
        **FLAGS,
    )
    if FILTER_ONLY:
        return la.all_scores, None
    k = _dot_kernel[(P,)](
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
        BLOCK_P=kw["BLOCK_P"],
        BLOCK_D=kw["BLOCK_D"],
        NPROG=P,
        num_warps=kw["num_warps"],
        num_stages=kw["num_stages"],
    )
    return la.all_scores, k
