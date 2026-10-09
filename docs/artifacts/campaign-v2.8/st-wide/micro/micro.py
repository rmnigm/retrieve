"""ST-WIDE scratch: scorer-only timing on C7's cell (PubMed 10M d768 bloom c0_mesh, n_lists 4096); variants are
modules in this dir exposing run(la) -> scores. Prints regs/spills, us per call, equality vs shipped."""

import importlib
import sys
import time
import statistics
from pathlib import Path
import torch
import yaml
from bench import config, inputs
from retrieve.modules.silvertorch import SilverTorch

sys.path.insert(0, "/scratch/st-wide/micro")
cps = importlib.import_module("retrieve.ops.triton.codesigned_probe_score")
host = importlib.import_module("retrieve.ops.triton._host")
common = importlib.import_module("retrieve.ops.triton.common")
DEV = torch.device("cuda")
cfg = Path("config/pubmed.yaml")
inp = inputs.load_inputs(config.load_dataset(cfg, 768), DEV, with_filters=True)
qa_s, skip = inputs.sweep_qa(
    inp["qa"], tuple(yaml.safe_load(cfg.read_text())["filters"]["bloom"]["c0_mesh"])
)
m = SilverTorch(
    k=100,
    n_lists=4096,
    n_probe=1024,
    filter_mode="bloom",
    m_bits=1024,
    k_hash=5,
    n_iter=10,
    seed=0,
)
m.register_index(inp["item_embs"].to(DEV), item_clause_attrs=inp["item_attrs"].to(DEV))
variants = sys.argv[1].split(",")
cells = [tuple(map(int, c.split("x"))) for c in sys.argv[2].split(",")]


def launches(n_probe, bs):
    m.set_query_params(n_probe=n_probe)
    pool, qa_pool = inputs.query_pool(
        inp, qa_s, skip, bs=bs, seed=0, n_pool=4, device=DEV
    )
    out = []
    for i in range(4):
        prep = m.prepare_queries(qa_pool[i])
        pr = m._phase1_probe_ids(pool[i])
        c = host.tile_for_width(cps.CONFIGS, 768, bs, m._probe_width)
        la = cps._cps_prep(
            pool[i],
            pr,
            m.cluster_offsets,
            m.item_codes,
            m.sort_perm,
            m._global_scale_f,
            m._probe_width,
            query_bit_positions=prep.query_bits,
            bloom_transposed=m.bloom_transposed,
            cfg=c,
            bit_freq=m.bloom_bit_freq,
        )
        out.append(la)
    return out


def shipped(la):
    if la.table is not None:
        common.probe_table_kernel[la.table.grid](**la.table.kwargs)
    k = cps._codesigned_probe_score_kernel[la.grid](**la.kwargs)
    return la.all_scores, k


with torch.inference_mode():
    for n_probe, bs in cells:
        las = launches(n_probe, bs)
        ref = [importlib.import_module("frozen").run(la)[0].clone() for la in las]
        for v in variants:
            run = shipped if v == "shipped" else importlib.import_module(v).run
            if v != "shipped" and las[0].table is None:
                continue
            outs = []
            for la in las:
                la.all_scores.fill_(float("nan"))
                outs.append(run(la))
                outs[-1] = (outs[-1][0].clone(), outs[-1][1])
            eq = all(torch.equal(o[0], r) for o, r in zip(outs, ref))
            k = outs[0][1]
            ts = []
            for _ in range(7):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for i in range(20):
                    run(las[i % 4])
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) / 20 * 1e6)
            from torch.profiler import ProfilerActivity, profile

            with profile(activities=[ProfilerActivity.CUDA]) as p:
                for i in range(20):
                    run(las[i % 4])
                torch.cuda.synchronize()
            kern = sorted(
                (
                    (e.key[:28], round(e.self_device_time_total / 20))
                    for e in p.key_averages()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ),
                key=lambda x: -x[1],
            )[:4]
            print(
                f"np{n_probe} bs{bs} {v:12s} {statistics.median(ts):8.1f} us eq={eq} regs={getattr(k, 'n_regs', None)} "
                f"spills={getattr(k, 'n_spills', None)} grid={las[0].grid} {kern}",
                flush=True,
            )
