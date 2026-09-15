"""Pure-torch exact-clause sibling of ``codesigned_probe_score``: the AND-of-OR predicate over
the gathered ``[B, P, C, A_max]`` attrs (reverse XOR, ``-1`` inactive) gates the same int8 dot."""

from __future__ import annotations

from torch import Tensor

from retrieve.functional import clause_subset_match
from retrieve.ops.reference.codesigned_probe_score import _probe_score


def codesigned_probe_score_exact(
    query: Tensor,
    flat_probed_items: Tensor,
    item_codes: Tensor,
    item_clause_attrs: Tensor,
    clause_is_reverse: Tensor,
    query_clause_attrs: Tensor,
    global_scale: float,
    k: int,
) -> tuple[Tensor, Tensor]:
    valid = flat_probed_items >= 0
    gathered = item_clause_attrs[flat_probed_items.clamp_min(0)]  # [B, P, C, A_max]
    keep = valid & clause_subset_match(gathered, query_clause_attrs.long(), clause_is_reverse)
    return _probe_score(query, flat_probed_items, item_codes, global_scale, k, keep)
