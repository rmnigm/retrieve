import frozen_cps
from retrieve.ops.triton import common


def run(la):
    if la.table is not None:
        common.probe_table_kernel[la.table.grid](**la.table.kwargs)
    k = frozen_cps._codesigned_probe_score_kernel[la.grid](**la.kwargs)
    return la.all_scores, k
