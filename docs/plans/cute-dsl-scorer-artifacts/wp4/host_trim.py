"""WP-4 step 5: host-overhead levers for _cps_cute_scores, measured enqueue-only (GPU busy).

Each variant is a full re-implementation of the phase-3 launcher with one knob changed;
outputs are torch.equal-checked against the module's own launcher first."""

from __future__ import annotations

import ctypes
import importlib
import json
import sys
import time

sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import torch  # noqa: E402

ct = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cute")
cu = importlib.import_module("retrieve.kernels.silvertorch.codesigned_probe_score_cuda")
from retrieve.kernels.silvertorch.codesigned_probe_score_cuda import words_per_cluster  # noqa: E402
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global  # noqa: E402

DEV = "cuda"
ct.ensure_built()
cu.ensure_built()
dev = ct._load_dev()
b, p, n, d = 16, 8192, 131072, 128
codes, gs = quantize_int8_global(torch.randn(n, d, device=DEV))
q_codes, q_scales = quantize_int8(torch.randn(b, d, device=DEV))
q_codes, q_scales = q_codes.contiguous(), q_scales.contiguous()
flat = torch.randint(0, n, (b, p), device=DEV)
out = torch.empty(b, p, dtype=torch.float32, device=DEV)
dummy = torch.empty(1, 1, dtype=torch.int64, device=DEV)
cfg = ct.DEFAULT_CONFIG
GS = float(gs)
big = torch.randn(8192, 8192, device=DEV)
N = 500


def enqueue_us(fn, n=N, warm=30):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    for _ in range(4):
        big @ big
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    t1 = time.perf_counter()
    torch.cuda.synchronize()
    return 1e6 * (t1 - t0) / n


def validate(q_codes, q_scales, mask, flat_items, item_codes, out_scores, has_mask, max_size, cfg):
    """The module's validation block, verbatim (returns b, d, p, wpc, mask_words)."""
    if not q_codes.is_cuda:
        raise ValueError("q_codes must be a CUDA tensor")
    for name, t in (("q_codes", q_codes), ("q_scales", q_scales), ("mask", mask), ("flat_items", flat_items),
                    ("item_codes", item_codes), ("out_scores", out_scores)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if q_codes.dtype != torch.int8 or item_codes.dtype != torch.int8:
        raise TypeError("q_codes and item_codes must be int8")
    if q_scales.dtype != torch.float32 or out_scores.dtype != torch.float32:
        raise TypeError("q_scales and out_scores must be float32")
    if flat_items.dtype != torch.int64 or mask.dtype != torch.int64:
        raise TypeError("flat_items and mask must be int64")
    b, d = q_codes.shape
    p = flat_items.shape[1]
    if d % 4 != 0:
        raise ValueError("D")
    if item_codes.shape[1] != d:
        raise ValueError("item_codes D mismatch")
    if flat_items.shape[0] != b or out_scores.shape[0] != b:
        raise ValueError("batch mismatch")
    if out_scores.shape[1] != p:
        raise ValueError("out_scores P mismatch")
    if b > 65535:
        raise ValueError("B")
    if not 1 <= cfg.num_warps <= 32:
        raise ValueError("num_warps")
    if cfg.block_p <= 0 or cfg.block_p % cfg.num_warps != 0:
        raise ValueError("block_p")
    if cfg.unroll not in (1, 2, 4):
        raise ValueError("unroll")
    wpc = 1
    if has_mask:
        if max_size <= 0 or p % max_size != 0:
            raise ValueError("P")
        wpc = words_per_cluster(max_size)
        if mask.shape != (b, (p // max_size) * wpc):
            raise ValueError("mask")
    return b, d, p, wpc, mask.shape[1]


def validate_min(q_codes, q_scales, mask, flat_items, item_codes, out_scores, has_mask, max_size, cfg):
    """Same checks, fewer Python operations: one tuple of shapes/dtypes/contiguity."""
    b, d = q_codes.shape
    p = flat_items.shape[1]
    if (
        not q_codes.is_cuda
        or q_codes.dtype != torch.int8 or item_codes.dtype != torch.int8
        or q_scales.dtype != torch.float32 or out_scores.dtype != torch.float32
        or flat_items.dtype != torch.int64 or mask.dtype != torch.int64
    ):
        raise TypeError("dtype/device")
    if not (q_codes.is_contiguous() and q_scales.is_contiguous() and mask.is_contiguous()
            and flat_items.is_contiguous() and item_codes.is_contiguous() and out_scores.is_contiguous()):
        raise ValueError("contiguous")
    if d % 4 or item_codes.shape[1] != d or flat_items.shape[0] != b or out_scores.shape != (b, p) or b > 65535:
        raise ValueError("shape")
    if not 1 <= cfg.num_warps <= 32 or cfg.block_p <= 0 or cfg.block_p % cfg.num_warps or cfg.unroll not in (1, 2, 4):
        raise ValueError("cfg")
    wpc = 1
    if has_mask:
        if max_size <= 0 or p % max_size != 0:
            raise ValueError("P")
        wpc = words_per_cluster(max_size)
        if mask.shape != (b, (p // max_size) * wpc):
            raise ValueError("mask")
    return b, d, p, wpc, mask.shape[1]


def make_variant(stream_mode="torch", guard="with", val="full", call="call"):
    vfn = validate if val == "full" else validate_min
    state = {}

    def launcher(q_codes, q_scales, mask, flat_items, item_codes, out_scores, *, global_scale, has_mask, max_size, cfg):
        b, d, p, wpc, mask_words = vfn(q_codes, q_scales, mask, flat_items, item_codes, out_scores, has_mask, max_size, cfg)
        vec_ok = q_codes.data_ptr() % 16 == 0 and item_codes.data_ptr() % 16 == 0
        seg = {64: 4, 128: 8, 256: 16}.get(d if vec_ok else 0)
        align = 16 if seg is not None else 4
        didx = q_codes.device.index
        if guard == "with":
            ctx = torch.cuda.device(didx)
        elif guard == "check":
            ctx = torch.cuda.device(didx) if didx != torch.cuda.current_device() else None
        else:
            ctx = None
        if ctx is not None:
            ctx.__enter__()
        try:
            launch = dev.compile_score(seg, has_mask, cfg.unroll, device=didx)
            if stream_mode == "torch":
                stream = dev.cu_stream(torch.cuda.current_stream().cuda_stream)
            else:
                stream = dev.cu_stream(torch._C._cuda_getCurrentRawStream(didx))
            if call == "call":
                launch(
                    dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), align),
                    dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
                    dev.gmem_ptr(dev.Int64, mask.data_ptr(), 8),
                    dev.gmem_ptr(dev.Int64, flat_items.data_ptr(), 8),
                    dev.gmem_ptr(dev.Int32, item_codes.data_ptr(), align),
                    dev.gmem_ptr(dev.Float32, out_scores.data_ptr(), 4),
                    float(global_scale), p, max_size, mask_words, wpc, b, cfg.block_p, cfg.num_warps, stream,
                )
            else:  # bypass generate_execution_args: persistent ctypes cells per executor
                cells = state.get(launch)
                if cells is None:
                    ptr_cells = [ctypes.c_void_p(0) for _ in range(6)]
                    sc = [ctypes.c_float(0), ctypes.c_int64(0), ctypes.c_int64(0), ctypes.c_int64(0),
                          ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0), ctypes.c_int32(0)]
                    st_cell = ctypes.c_void_p(0)
                    exe = [ctypes.addressof(c) for c in ptr_cells + sc + [st_cell]]
                    cells = state[launch] = (ptr_cells, sc, st_cell, exe)
                ptr_cells, sc, st_cell, exe = cells
                ptr_cells[0].value = q_codes.data_ptr()
                ptr_cells[1].value = q_scales.data_ptr()
                ptr_cells[2].value = mask.data_ptr()
                ptr_cells[3].value = flat_items.data_ptr()
                ptr_cells[4].value = item_codes.data_ptr()
                ptr_cells[5].value = out_scores.data_ptr()
                sc[0].value = float(global_scale); sc[1].value = p; sc[2].value = max_size; sc[3].value = mask_words
                sc[4].value = wpc; sc[5].value = b; sc[6].value = cfg.block_p; sc[7].value = cfg.num_warps
                st_cell.value = int(stream)
                launch.run_compiled_program(exe)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)

    return launcher


ref = torch.empty_like(out)
ct._cps_cute_scores(q_codes, q_scales, dummy, flat, codes, ref, global_scale=GS, has_mask=False, max_size=0, cfg=cfg)
torch.cuda.synchronize()

# what does the DSL actually pass? (to validate the bypass layout)
launch = dev.compile_score(8, False, 1, device=0)
args = [dev.gmem_ptr(dev.Int32, q_codes.data_ptr(), 16), dev.gmem_ptr(dev.Float32, q_scales.data_ptr(), 4),
        dev.gmem_ptr(dev.Int64, dummy.data_ptr(), 8), dev.gmem_ptr(dev.Int64, flat.data_ptr(), 8),
        dev.gmem_ptr(dev.Int32, codes.data_ptr(), 16), dev.gmem_ptr(dev.Float32, out.data_ptr(), 4),
        GS, p, 0, 1, 1, b, cfg.block_p, cfg.num_warps, dev.cu_stream(torch.cuda.current_stream().cuda_stream)]
exe, adapted = launch.generate_execution_args(*args)
print("DSL exe_args:", len(exe), [type(x).__name__ for x in exe][:3], "...; adapted:", [type(a).__name__ for a in adapted])
addr = lambda x: x if isinstance(x, int) else x.value
print("exe elem types:", [type(x).__name__ for x in exe])
print("ptr0 deref:", ctypes.c_void_p.from_address(addr(exe[0])).value, "expected", q_codes.data_ptr())
print("scalar p deref:", ctypes.c_int64.from_address(addr(exe[7])).value, "expected", p)
print("scalar gs deref:", ctypes.c_float.from_address(addr(exe[6])).value, "expected", GS)
print("stream cell deref:", ctypes.c_void_p.from_address(addr(exe[-1])).value, "expected", int(args[-1]))

results = {}
variants = [
    ("module _cps_cute_scores (baseline)", lambda: ct._cps_cute_scores(q_codes, q_scales, dummy, flat, codes, out, global_scale=GS, has_mask=False, max_size=0, cfg=cfg)),
    ("cuda cps_scores (C++ launcher)", lambda: cu._load_ext().cps_scores(q_codes, q_scales, dummy, flat, codes, out, GS, False, 0, cu.DEFAULT_CONFIG.block_p, cu.DEFAULT_CONFIG.num_warps, cu.DEFAULT_CONFIG.unroll)),
]
for stream_mode in ("torch", "raw"):
    for guard in ("with", "check", "none"):
        for val in ("full", "min"):
            for call in ("call", "bypass"):
                if (stream_mode, guard, val, call) in {("torch", "with", "full", "call")}:
                    name = "variant = module (sanity)"
                else:
                    name = f"stream={stream_mode:<5} guard={guard:<5} validate={val:<4} call={call}"
                fn = make_variant(stream_mode, guard, val, call)
                variants.append((name, lambda fn=fn: fn(q_codes, q_scales, dummy, flat, codes, out, global_scale=GS, has_mask=False, max_size=0, cfg=cfg)))
for name, fn in variants:
    out.fill_(0)
    fn()
    torch.cuda.synchronize()
    ok = torch.equal(out, ref)
    us = enqueue_us(fn)
    results[name] = us
    print(f"{name:<58} {us:7.1f} us  {'bit-exact' if ok else 'MISMATCH'}")
json.dump(results, open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), indent=1)
