import importlib, torch
cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
from retrieve.ops import reference
from retrieve.indexing.selectivity import bloom_bit_freq
from retrieve.indexing.quantize import quantize_int8_global
from tests.parity.conftest import make_probe_family, make_bloom, assert_topk_equal
from tests.conftest import make_index, make_query
for d in (64, 128, 768):
    for b in (4, 16):
        lay = make_probe_family(b, 512, 60, 300)
        codes, gs = quantize_int8_global(make_index(lay.n, d))
        q = make_query(b, d)
        qpos, bt, _, _ = make_bloom(lay.n, b)
        bf = bloom_bit_freq(bt, lay.n)
        args = (q, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, qpos, bt, bf, gs, 32, lay.width)
        ref = reference.codesigned_probe_score_bloom(*args)
        out = {}
        for flag in (False, True):
            cps.ITEM_TWO_PASS = flag
            out[flag] = cps._codesigned_probe_score_impl(q, lay.probe_ids, lay.cluster_offsets, codes, lay.sort_perm, gs, 32, lay.width,
                query_bit_positions=qpos, bloom_transposed=bt, bloom_bit_freq=bf, sparse=True)
        assert_topk_equal(*out[True], *ref); assert_topk_equal(*out[True], *out[False])
        print(d, b, "ok", torch.isfinite(out[True][1]).float().mean().item())
