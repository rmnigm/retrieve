"""Pure-torch exact-clause sibling of ``codesigned_probe_score``: the AND-of-OR predicate over
the gathered ``[B, width, C, A_max]`` cluster-sorted attrs (reverse XOR, ``-1`` inactive) gates
the same int8 dot."""

from __future__ import annotations

from torch import Tensor

from retrieve.functional import clause_subset_match
from retrieve.ops.reference.codesigned_probe_score import _probe_score, probe_positions


def codesigned_probe_score_exact(
    query: Tensor,
    probe_ids: Tensor,
    cluster_offsets: Tensor,
    item_codes: Tensor,
    sort_perm: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    global_scale: float,
    k: int,
    width: int,
) -> tuple[Tensor, Tensor]:
    pos, valid = probe_positions(probe_ids, cluster_offsets, width)
    gathered = item_clause_attrs[pos]  # [B, width, C, A_max]
    keep = valid & clause_subset_match(gathered, query_clause_attrs.long(), clause_is_reverse)
    return _probe_score(query, pos, item_codes, sort_perm, global_scale, k, keep)
