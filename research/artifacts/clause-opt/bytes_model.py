"""Bytes each filter-payload kernel reads per probed doc, from the real probed docs and attrs.

exact (C14 layout): lane l of a 32-doc payload warp reads, per clause it reaches (clauses in
order, a lane stops at its first failing one), the doc's int32 slots up to its first match or
pad; ``lane_bytes`` counts those 4-byte reads, ``sector_bytes`` the distinct 32-byte sectors
the warp touches per slot (what L1 requests from L2). exact before (final-freeze, int64 [N, C,
A] rows, every slot of a reached clause): one doc's clause row is A * 8 bytes, its own sector(s).
bloom (Meta's index v2): per payload and active term, k distinct 8-byte words of the 64-doc
column; neighbouring payloads share the column, so ``k * 8 / 64`` bytes per doc and term.
"""

import torch


def probed_docs(module, query):
    """Per probed (row, doc): row, cluster-sorted doc index, and the payload group start (the
    entry index of the warp's first doc) so sectors can be counted per warp."""
    probe = module._phase1_probe_ids(query)  # [B, P] cluster ids
    starts = module.cluster_offsets[:-1][probe].reshape(-1)
    lengths = module.cluster_sizes[probe].reshape(-1)
    rows = torch.arange(probe.shape[0], device=probe.device).repeat_interleave(
        probe.shape[1]
    )
    entry_cluster = torch.arange(
        lengths.numel(), device=probe.device
    ).repeat_interleave(lengths)
    first = torch.cumsum(lengths, 0) - lengths
    i = torch.arange(int(lengths.sum()), device=probe.device) - first[entry_cluster]
    doc = starts[entry_cluster] + i
    group = first[entry_cluster] + i // 32 * 32
    return rows.repeat_interleave(lengths), doc, group, entry_cluster


def exact_bytes(sorted_attrs, reverse, qa, rows, doc, group, chunk=1 << 24):
    """(lane_bytes, sector_bytes, before_bytes) summed over the probed (row, doc) entries.
    ``sorted_attrs`` [N, C, A] int64 cluster-sorted, each doc's values sorted descending."""
    n, c_count, a_max = sorted_attrs.shape
    widths = (sorted_attrs != -1).sum(2).amax(0).tolist()
    lane = sector = before = 0
    for lo in range(0, doc.numel(), chunk):
        r, d, g = rows[lo : lo + chunk], doc[lo : lo + chunk], group[lo : lo + chunk]
        alive = torch.ones_like(d, dtype=torch.bool)
        q_rows = qa[r]
        g_first = d - (torch.arange(lo, lo + d.numel(), device=d.device) - g)
        for c in range(c_count):
            q = q_rows[:, c]
            reached = alive & (q != -1)
            before += int(reached.sum()) * ((a_max * 8 + 31) // 32 * 32)
            w = widths[c]
            v = sorted_attrs[d, c, :w]
            hit = (v == q[:, None]) | (v == -1)
            first_hit = (
                torch.where(hit.any(1), hit.int().argmax(1) + 1, w)
                if w
                else torch.zeros_like(d)
            )
            slots = torch.where(reached, first_hit, 0)
            lane += int(slots.sum()) * 4
            for s in range(w):
                m = slots > s
                key = g[m] * 8 + (d[m] // 8 - g_first[m] // 8)
                sector += int(torch.unique(key).numel()) * 32
            match = (v == q[:, None]).any(1)
            alive &= ~reached | (match != bool(reverse[c]))
    return lane, sector, before


def bloom_bytes(qa, rows, k_hash):
    terms = (qa != -1).sum(1)[rows]
    return int(terms.sum()) * k_hash * 8 / 64
