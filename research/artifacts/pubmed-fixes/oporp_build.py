"""Fix 2 gate on the D1 item tables: the chunked `quantize_oporp_1bit` / `quantize_simhash_1bit`
against the one-shot chain (`project_*_query` over the whole table: the pre-fix build body),
`torch.equal` bits, build seconds and peak allocated memory of each, on the fp32 table the
harness hands LiNR V3. `--no-oneshot` skips the one-shot side (pubmed: it cannot fit); with
`--try-oneshot` it is attempted anyway and its OOM recorded.

    cd evaluation && python oporp_build.py DATASET DIM [--no-oneshot | --try-oneshot]
"""

import sys
import time
from pathlib import Path

import torch

from bench import config
from eval_datasets import layout
from retrieve.indexing.quantize import (
    project_oporp_1bit_query,
    project_simhash_1bit_query,
    quantize_oporp_1bit,
    quantize_simhash_1bit,
)

dataset, dim = sys.argv[1], int(sys.argv[2])
mode = sys.argv[3] if len(sys.argv) > 3 else ""
dev = torch.device("cuda")
ds = config.load_dataset(Path("config") / f"{dataset}.yaml", dim)
embs = layout.load_text_items(ds.content_dir, dev)
print(f"{dataset} table {tuple(embs.shape)} {embs.dtype}, {embs.numel() * 4 / 2**30:.2f} GiB")


def measure(fn):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0, (torch.cuda.max_memory_allocated() - base) / 2**30


for name, build, oneshot in (
    ("oporp", lambda: quantize_oporp_1bit(embs, seed=0), lambda p: project_oporp_1bit_query(embs, p[1], p[2])),
    ("simhash", lambda: quantize_simhash_1bit(embs, k_bits=dim, seed=0), lambda p: project_simhash_1bit_query(embs, p[1])),
):
    new, s_new, gib_new = measure(build)
    line = f"{name}: chunked {s_new:.2f} s, peak +{gib_new:.2f} GiB (output {new[0].numel() * 8 / 2**30:.2f} GiB)"
    if mode != "--no-oneshot":
        try:
            old, s_old, gib_old = measure(lambda: oneshot(new))
            line += f"; one-shot {s_old:.2f} s, peak +{gib_old:.2f} GiB; torch.equal {torch.equal(new[0], old)}"
            del old
        except torch.OutOfMemoryError as e:
            line += f"; one-shot OOM: {str(e).splitlines()[0][:120]}"
    print(line, flush=True)
    del new
    torch.cuda.empty_cache()
