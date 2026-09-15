"""L3 — which deterministic epilogue is cheapest? Standalone copies of the clause predicate kernel
with six epilogues, timed on the real goodreads narrow attrs with c0_genre-shaped queries at B=1
and B=16 (do_bench medians). Every deterministic variant is also checked ``torch.equal`` to
``ops.reference.clause_compact`` on the full pipeline.

    cd retrieve && flock /workspace/gpu.lock uv run --no-sync python \
        ../docs/plans/deterministic-compaction-artifacts/epilogue_variants.py

Variants (predicate identical, ``common.clause_pass``):
  atomic        the pre-L3 kernel: cumsum + atomic_add row base + store            (not deterministic)
  count_only    store the tile count and nothing else — the structural floor       (not a compaction)
  bits_reshape  count + bitmask packed with tl.reshape([BLOCK_N//64, 64]).sum(1)   (= branch HEAD)
  bits_loop     count + bitmask packed with a static loop of BLOCK_N//64 masked sums
  bits_atomic   count + bitmask via tl.atomic_or into a zeroed buffer (OR is order-independent)
  ids_scratch   count + cumsum + store ids at tile-local slots of an int32 scratch; copy pass
"""

from __future__ import annotations

import json
import sys

import torch
import triton
import triton.language as tl
from retrieve.ops import reference
from retrieve.ops.tune import _bench
from retrieve.ops.triton._host import grid_batch_tiles
from retrieve.ops.triton.common import clause_pass, compact_scatter_kernel, compact_store

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from time_compaction import DATA, real_queries, sm_mhz  # noqa: E402

BLOCK_N, NUM_WARPS = 512, 2


@triton.jit
def _predicate(
    item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, N, tiles_y,
    C: tl.constexpr, A_MAX: tl.constexpr, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    n_offsets = tile_id * BLOCK_N + tl.arange(0, BLOCK_N)
    pass_mask = clause_pass(
        item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, n_offsets, n_offsets < N, bid,
        stride_in, stride_ic, stride_ia, stride_qb, stride_qc, C=C, A_MAX=A_MAX,
    )
    return bid, tile_id, n_offsets, pass_mask


@triton.jit
def k_atomic(
    item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, out_ptr, counts_ptr, N, tiles_y,
    C: tl.constexpr, A_MAX: tl.constexpr, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
    stride_ob, BLOCK_N: tl.constexpr,
):
    bid, tile_id, ids, pass_mask = _predicate(
        item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, N, tiles_y, C, A_MAX,
        stride_in, stride_ic, stride_ia, stride_qb, stride_qc, BLOCK_N,
    )
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1
    base = tl.atomic_add(counts_ptr + bid, tl.sum(pass_int).to(tl.int64))
    tl.store(out_ptr + bid * stride_ob + base + intra.to(tl.int64), ids.to(tl.int64), mask=pass_mask)


@triton.jit
def k_count_only(
    item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, tile_counts_ptr, N, tiles_y,
    C: tl.constexpr, A_MAX: tl.constexpr, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
    stride_tb, BLOCK_N: tl.constexpr,
):
    bid, tile_id, ids, pass_mask = _predicate(
        item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, N, tiles_y, C, A_MAX,
        stride_in, stride_ic, stride_ia, stride_qb, stride_qc, BLOCK_N,
    )
    tl.store(tile_counts_ptr + bid * stride_tb + tile_id, tl.sum(tl.where(pass_mask, 1, 0).to(tl.int64)))


@triton.jit
def k_bits(
    item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, tile_counts_ptr, bits_ptr, N, tiles_y,
    C: tl.constexpr, A_MAX: tl.constexpr, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
    stride_tb, stride_bb, MODE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    bid, tile_id, ids, pass_mask = _predicate(
        item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, N, tiles_y, C, A_MAX,
        stride_in, stride_ic, stride_ia, stride_qb, stride_qc, BLOCK_N,
    )
    lane = tl.arange(0, BLOCK_N)
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int64)
    tl.store(tile_counts_ptr + bid * stride_tb + tile_id, tl.sum(pass_int))
    contrib = pass_int << (lane % 64).to(tl.int64)
    word_base = bits_ptr + bid * stride_bb + tile_id * (BLOCK_N // 64)
    if MODE == 0:  # reshape
        words = tl.sum(tl.reshape(contrib, (BLOCK_N // 64, 64)), axis=1)
        tl.store(word_base + tl.arange(0, BLOCK_N // 64), words)
    elif MODE == 1:  # static loop of masked reductions
        for w in tl.static_range(BLOCK_N // 64):
            tl.store(word_base + w, tl.sum(tl.where(lane // 64 == w, contrib, 0)))
    else:  # atomic_or into zeroed words
        tl.atomic_or(word_base + lane // 64, contrib, mask=pass_mask)


@triton.jit
def k_ids_scratch(
    item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, tile_counts_ptr, scratch_ptr, N, tiles_y,
    C: tl.constexpr, A_MAX: tl.constexpr, stride_in, stride_ic, stride_ia, stride_qb, stride_qc,
    stride_tb, stride_sb, BLOCK_N: tl.constexpr,
):
    bid, tile_id, ids, pass_mask = _predicate(
        item_attrs_ptr, is_reverse_ptr, query_attrs_ptr, N, tiles_y, C, A_MAX,
        stride_in, stride_ic, stride_ia, stride_qb, stride_qc, BLOCK_N,
    )
    pass_int = tl.where(pass_mask, 1, 0).to(tl.int32)
    intra = tl.cumsum(pass_int, axis=0) - 1
    tl.store(tile_counts_ptr + bid * stride_tb + tile_id, tl.sum(pass_int).to(tl.int64))
    tl.store(scratch_ptr + bid * stride_sb + tile_id * BLOCK_N + intra, ids.to(tl.int32), mask=pass_mask)


@triton.jit
def k_copy(
    scratch_ptr, counts_ptr, offsets_ptr, out_ptr, tiles_y, stride_sb, stride_tb, stride_ob,
    BLOCK_N: tl.constexpr,
):
    bid = tl.program_id(0)
    tile_id = tl.program_id(2) * tiles_y + tl.program_id(1)
    lane = tl.arange(0, BLOCK_N)
    count = tl.load(counts_ptr + bid * stride_tb + tile_id)
    base = tl.load(offsets_ptr + bid * stride_tb + tile_id)
    ids = tl.load(scratch_ptr + bid * stride_sb + tile_id * BLOCK_N + lane, mask=lane < count)
    tl.store(out_ptr + bid * stride_ob + base + lane, ids.to(tl.int64), mask=lane < count)


def main() -> None:
    dev = torch.device("cuda")
    attrs = torch.load(f"{DATA}/item_attrs_narrow.pt").to(dev)
    rev = torch.load(f"{DATA}/clause_is_reverse_narrow.pt").to(dev).to(torch.int8)
    n, c, a_max = attrs.shape
    results = {}
    for b in (1, 16):
        q = real_queries(attrs, b)
        grid, tiles_y = grid_batch_tiles(b, n, BLOCK_N)
        tiles = tiles_y * grid[2]
        common = dict(
            item_attrs_ptr=attrs, is_reverse_ptr=rev, query_attrs_ptr=q, N=n, tiles_y=tiles_y,
            C=c, A_MAX=a_max, stride_in=attrs.stride(0), stride_ic=attrs.stride(1),
            stride_ia=attrs.stride(2), stride_qb=q.stride(0), stride_qc=q.stride(1),
            BLOCK_N=BLOCK_N, num_warps=NUM_WARPS,
        )
        ref = reference.clause_compact(attrs, rev.bool(), q)
        valid = torch.arange(n, device=dev)[None, :] < ref[1][:, None]
        ref_ids = torch.where(valid, ref[0], torch.full_like(ref[0], -1))

        def finish_bits(tile_counts, bits):
            tile_ends = tile_counts.cumsum(1)
            out = torch.full((b, n), -1, dtype=torch.int64, device=dev)
            compact_scatter_kernel[grid](
                bits, tile_ends - tile_counts, out, tiles_y, bits.stride(0), tile_counts.stride(0),
                out.stride(0), out.stride(1), BLOCK_N=BLOCK_N, num_warps=NUM_WARPS,
            )
            return out, tile_ends[:, -1].clone()

        def run_atomic():
            out = torch.full((b, n), -1, dtype=torch.int64, device=dev)
            counts = torch.zeros(b, dtype=torch.int64, device=dev)
            k_atomic[grid](**common, out_ptr=out, counts_ptr=counts, stride_ob=out.stride(0))
            return out, counts

        def run_count_only():
            tc = torch.empty((b, tiles), dtype=torch.int64, device=dev)
            k_count_only[grid](**common, tile_counts_ptr=tc, stride_tb=tc.stride(0))
            return tc

        def make_bits(mode):
            def pred():
                tc = torch.empty((b, tiles), dtype=torch.int64, device=dev)
                alloc = torch.zeros if mode == 2 else torch.empty
                bits = alloc((b, tiles * (BLOCK_N // 64)), dtype=torch.int64, device=dev)
                k_bits[grid](**common, tile_counts_ptr=tc, bits_ptr=bits, stride_tb=tc.stride(0),
                             stride_bb=bits.stride(0), MODE=mode)
                return tc, bits

            return pred, lambda: finish_bits(*pred())

        def pred_ids():
            tc = torch.empty((b, tiles), dtype=torch.int64, device=dev)
            scratch = torch.empty((b, tiles * BLOCK_N), dtype=torch.int32, device=dev)
            k_ids_scratch[grid](**common, tile_counts_ptr=tc, scratch_ptr=scratch,
                                stride_tb=tc.stride(0), stride_sb=scratch.stride(0))
            return tc, scratch

        def full_ids():
            tc, scratch = pred_ids()
            tile_ends = tc.cumsum(1)
            out = torch.full((b, n), -1, dtype=torch.int64, device=dev)
            k_copy[grid](scratch, tc, tile_ends - tc, out, tiles_y, scratch.stride(0), tc.stride(0),
                         out.stride(0), BLOCK_N=BLOCK_N, num_warps=NUM_WARPS)
            return out, tile_ends[:, -1].clone()

        variants = {
            "atomic": (run_atomic, run_atomic, False),
            "count_only": (run_count_only, None, False),
            "bits_reshape": (*make_bits(0), True),
            "bits_loop": (*make_bits(1), True),
            "bits_atomic_or": (*make_bits(2), True),
            "ids_scratch": (pred_ids, full_ids, True),
        }
        for name, (pred, full, deterministic) in variants.items():
            for _ in range(3):
                pred()
                if full is not None:
                    full()
            torch.cuda.synchronize()
            row = {"predicate_ms": _bench(pred)}
            if full is not None:
                row["full_ms"] = _bench(full)
                out, counts = full()
                out = torch.where(torch.arange(n, device=dev)[None, :] < counts[:, None], out, torch.full_like(out, -1))
                row["equal_to_reference"] = bool(torch.equal(counts, ref[1]) and torch.equal(out, ref_ids))
                if deterministic:
                    again = full()
                    row["repeatable"] = bool(torch.equal(again[0], full()[0]))
            row["sm_mhz"] = sm_mhz()
            results[f"B={b} {name}"] = row
            print(f"B={b:2d} {name:15s} " + "  ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    out_path = __file__.replace(".py", ".json")
    json.dump(results, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
