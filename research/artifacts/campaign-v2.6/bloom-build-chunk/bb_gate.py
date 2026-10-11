"""BLOOM-BUILD-CHUNK gate: the chunked `build_transposed_sigs` (this tree) against the one-pass build it replaces
(copied below from staging a9748da), `torch.equal`, with each build's peak CUDA memory above its input:
goodreads and arXiv item signatures as SilverTorch builds them (m_bits 1024, k_hash 5, the registered clause salt),
and a random [30 M, 16] int64 table (LAION 30 M at m_bits 1024; the one-pass build needs ~42 GB, so it fits on an idle
80 GB A100 while the chunked one needs the tables plus ~256 MiB).

    cd evaluation && PYTHONPATH=.:../retrieve/src python bb_gate.py out.json
"""

import json
import sys
from pathlib import Path

import torch
from bench import config, inputs

from retrieve.indexing.bloom_hash import (
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
)

DEV = torch.device("cuda")
M_BITS, K_HASH = 1024, 5


def one_pass(sorted_sigs):
    """staging a9748da's build_transposed_sigs, verbatim."""
    n, w = sorted_sigs.shape
    n_words = (n + 63) // 64
    shifts = torch.arange(64, device=sorted_sigs.device, dtype=torch.int64)
    padded = torch.zeros(n_words * 64, w, dtype=torch.int64, device=sorted_sigs.device)
    padded[:n] = sorted_sigs
    out = torch.empty(w * 64, n_words, dtype=torch.int64, device=sorted_sigs.device)
    for word in range(w):
        bits = (padded[:, word].unsqueeze(0) >> shifts.unsqueeze(1)) & 1
        out[word * 64 : (word + 1) * 64] = (bits.view(64, n_words, 64) << shifts).sum(
            -1
        )
    return out


def peak(fn, sigs):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    t = fn(sigs)
    torch.cuda.synchronize()
    return t, (torch.cuda.max_memory_allocated() - base) / 2**20


rows = []
for ds in ("goodreads", "arxiv"):
    inp = inputs.load_inputs(
        config.load_dataset(Path(f"config/{ds}.yaml"), 128), DEV, with_filters=True
    )
    attrs = inp["item_attrs"].to(DEV).long()
    seeds = generate_seeds(K_HASH, device=DEV)
    salt = generate_clause_salt(attrs.shape[1], device=DEV)
    sigs = build_signatures(
        attrs, seeds, M_BITS, K_HASH, M_BITS // 64, clause_salt=salt
    )
    del inp, attrs
    new, new_mib = peak(build_transposed_sigs, sigs)
    old, old_mib = peak(one_pass, sigs)
    rows.append({"table": ds, "n": sigs.shape[0], "words": sigs.shape[1], "equal": torch.equal(new, old),
                 "chunked_peak_mib": new_mib, "one_pass_peak_mib": old_mib})  # fmt: skip
    print(rows[-1], flush=True)
    del sigs, new, old
g = torch.Generator(device=DEV).manual_seed(30)
ii = torch.iinfo(torch.int64)
sigs = torch.randint(
    ii.min, ii.max, (30_000_000, 16), generator=g, dtype=torch.int64, device=DEV
)
new, new_mib = peak(build_transposed_sigs, sigs)
new_cpu = new.cpu()
del new
old, old_mib = peak(one_pass, sigs)
rows.append({"table": "random-30M", "n": sigs.shape[0], "words": 16, "equal": torch.equal(old.cpu(), new_cpu),
             "chunked_peak_mib": new_mib, "one_pass_peak_mib": old_mib})  # fmt: skip
print(rows[-1], flush=True)
Path(sys.argv[1]).write_text(json.dumps(rows, indent=1))
print("ALL EQUAL" if all(r["equal"] for r in rows) else "MISMATCH", flush=True)
