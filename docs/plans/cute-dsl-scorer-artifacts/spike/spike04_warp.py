"""Item 4: shuffle bfly over a lane subset, redux.sync.add (sm_80), ballot, lane/warp/thread/block ids."""
import torch, cutlass, cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda

@cute.kernel
def k(x: cute.Tensor, o_bfly: cute.Tensor, o_redux: cute.Tensor, o_ballot: cute.Tensor, o_ids: cute.Tensor,
      SEG: cutlass.Constexpr[int]):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    gdim, _, _ = cute.arch.grid_dim()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    g = bidx * bdim + tidx
    seg = lane // SEG
    seg_mask = ((1 << SEG) - 1) << (seg * SEG)   # python ints folded... but seg is dynamic -> Int32 expr
    v = x[g]
    # (a) butterfly reduce within SEG-lane segment (mask restricted to the segment)
    r = v
    off = SEG // 2
    while off > 0:          # python-level loop (off is a python int)
        r = r + cute.arch.shuffle_sync_bfly(r, offset=off, mask=seg_mask)
        off //= 2
    o_bfly[g] = r
    # (b) redux.sync.add.s32 over the segment mask (sm_80+)
    o_redux[g] = cute.arch.warp_redux_sync(v, "add", mask_and_clamp=seg_mask)
    # (c) ballot of a predicate; lane 0 stores
    packed = cute.arch.vote_ballot_sync(v % 3 == 0)
    if lane == 0:
        o_ballot[g // 32] = packed
    # (d) ids
    o_ids[g, 0] = tidx; o_ids[g, 1] = bidx; o_ids[g, 2] = bdim; o_ids[g, 3] = gdim
    o_ids[g, 4] = lane; o_ids[g, 5] = warp

@cute.jit
def host(x, o_bfly, o_redux, o_ballot, o_ids, SEG: cutlass.Constexpr[int], stream: cuda.CUstream):
    k(x, o_bfly, o_redux, o_ballot, o_ids, SEG).launch(grid=[2, 1, 1], block=[64, 1, 1], stream=stream)

torch.manual_seed(0)
N = 128
x = torch.randint(-1000, 1000, (N,), device="cuda", dtype=torch.int32)
st = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
for SEG in (4, 8, 16):
    o_bfly = torch.zeros(N, device="cuda", dtype=torch.int32); o_redux = torch.zeros_like(o_bfly)
    o_ballot = torch.zeros(N // 32, device="cuda", dtype=torch.int32); o_ids = torch.zeros(N, 6, device="cuda", dtype=torch.int32)
    host(from_dlpack(x), from_dlpack(o_bfly), from_dlpack(o_redux), from_dlpack(o_ballot), from_dlpack(o_ids), SEG, st)
    torch.cuda.synchronize()
    ref = x.view(-1, SEG).sum(1, dtype=torch.int32).repeat_interleave(SEG)
    bits = (x % 3 == 0).view(-1, 32).to(torch.int64)
    ref_ballot = (bits << torch.arange(32, device="cuda")).sum(1).to(torch.int64)
    got_ballot = o_ballot.to(torch.int64) & 0xFFFFFFFF
    g = torch.arange(N, device="cuda")
    ids_ref = torch.stack([g % 64, g // 64, torch.full_like(g, 64), torch.full_like(g, 2), g % 32, (g % 64) // 32], 1).to(torch.int32)
    print(f"SEG={SEG}: bfly", torch.equal(o_bfly, ref), "redux", torch.equal(o_redux, ref), "ballot", torch.equal(got_ballot, ref_ballot), "ids", torch.equal(o_ids, ids_ref))
