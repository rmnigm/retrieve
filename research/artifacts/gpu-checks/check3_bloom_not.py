"""Check 3 (quality only): Meta's bloom with NOT on the four kept reverse-clause filters.

The wrapper refuses reverse clauses in bloom mode, so the bloom index is built forward-only (as
every bloom cell) and the query side passes the reverse flags to the adapter's
``queries_to_expressions`` (its ``clause_is_reverse`` seam): ``NOT c:v`` is Meta's complement of
the bloom's ``c:v`` mask, so a bloom false positive on ``c:v`` becomes a false negative.
Bloom at the study's configuration: m_bits 1024, k_hash 5, configured width, no recheck.

Per query, over the whole catalog (Meta's ``bloom_index_search_batch``, unpacked, mapped back from
the cluster-sorted doc space): false negatives of the NOT clause alone and of the whole filter
against the exact predicate (and the whole filter's false positives). Recall@100 against the exact
filtered top-100 (LiNR V2) of SilverTorch with the bloom (incl. NOT) and with its exact filter,
one k-means (same seed, checked equal), n_probe 24 and the `filter` suite's other probes.

    cd evaluation && python ../research/artifacts/gpu-checks/check3_bloom_not.py
"""

from __future__ import annotations

import functools
import sys
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import Records, env, load, log  # noqa: E402

from bench import algos, measure  # noqa: E402
from bench import inputs as binputs  # noqa: E402
from retrieve import OfficialConfig, SilverTorch  # noqa: E402
from retrieve.ops import official as official_mod  # noqa: E402
from retrieve.ops.official import adapter  # noqa: E402

K = 100
QUALITY_Q = 2000
CHUNK = 16
FILTERS = {
    "goodreads": ["c1_lang_reverse", "all4"],
    "pubmed": ["c3_journal_reverse", "all5"],
}
N_LISTS = {"goodreads": 4096, "pubmed": 4096}  # the IVF tune's picks (validation)
N_PROBES = {"goodreads": [24, 64], "pubmed": [24, 512, 1024]}  # suites.yaml `filter`
BLOOM = algos.BLOOM_DEFAULTS


def bloom_mask(
    st: SilverTorch, expressions: list[str], n_clauses: int, n: int
) -> torch.Tensor:
    """``[B, N]`` bool in original item order: Meta's full bloom search over every doc."""
    cfg = st.official
    plans = adapter.prepare_plans(expressions, cfg.n_stored_hashes, cfg.max_sub_queries,
                                  n_clauses, st.bloom_index.device, slots=len(expressions))  # fmt: skip
    m = adapter.bloom_full_mask(st.bloom_index, st.bundle_b_offsets, plans, st.k_hash,
                                cfg.n_stored_hashes, return_bool_mask=True)  # fmt: skip
    out = torch.zeros(m.shape[0], n, dtype=torch.bool, device=m.device)
    out[:, st.sort_perm] = m[:, :n].bool()
    return out


def recall(ids: torch.Tensor, ref: torch.Tensor, counts: torch.Tensor) -> float:
    hit = ((ids[:, :, None] == ref[:, None, :]) & (ref[:, None, :] >= 0)).any(1).sum(1)
    denom = counts.clamp(max=K)
    keep = denom > 0
    return (hit[keep].double() / denom[keep].double()).mean().item()


def per_query(x: torch.Tensor) -> dict[str, float]:
    x = x.double()
    return {"mean": x.mean().item(), "median": x.median().item(), "max": x.max().item(),
            "frac_queries_nonzero": (x > 0).double().mean().item()}  # fmt: skip


@torch.inference_mode()
def main(names: list[str]) -> None:
    warnings.filterwarnings("ignore")
    device = torch.device("cuda")
    measure.setup(0)
    measure.warm_gpu_once()
    recs = Records("check3", "bloom_not")
    done = recs.done()
    provenance = env()
    for name in names:
        ds, inp = load(name, device)
        n, attrs, rev = inp["n_items"], inp["item_attrs"], inp["clause_is_reverse"]
        n_clauses = attrs.shape[1]
        p = {"n_lists": N_LISTS[name], "n_probe": 24, "n_iter": 10}
        bloom = SilverTorch(k=K, filter_mode="bloom", seed=0, backend="cuda",
                            official=OfficialConfig(recheck=False), **BLOOM, **p)  # fmt: skip
        bloom.register_index(inp["item_embs"], item_clause_attrs=attrs)
        exact = SilverTorch(k=K, filter_mode="exact", seed=0, backend="cuda", **p)
        exact.register_index(
            inp["item_embs"], item_clause_attrs=attrs, clause_is_reverse=rev
        )
        same_index = bool(torch.equal(bloom.sort_perm, exact.sort_perm))
        filt = algos.build_filter("clause", attrs, clause_is_reverse=rev)
        v2 = algos.build("linr_v2", inp["item_embs"], k=K, backend="triton",
                         filter_kind="clause", filter_mod=filt)  # fmt: skip
        del inp["item_embs"]
        torch.cuda.empty_cache()
        log(name, "items", n, "same k-means index", same_index)
        # the reverse flags reach Meta's DSL as NOT (the wrapper's bloom mode never passes them)
        to_expr = functools.partial(
            adapter.queries_to_expressions, clause_is_reverse=rev.cpu()
        )
        for sweep in FILTERS[name]:
            cell = f"{name}/{sweep}"
            if cell in done:
                continue
            clauses = ds.clauses["clause"][sweep]
            rev_c = [c for c in clauses if bool(rev[c])]
            qa_s, skip = binputs.sweep_qa(inp["qa"], clauses)
            rows = (~skip).nonzero()[:, 0][:QUALITY_Q]
            qs = inp["queries"][rows].to(device).split(CHUNK)
            qa_rows = qa_s[rows].to(device).split(CHUNK)
            fn_not, pass_not, fn_all, fp_all, pass_all, counts = [], [], [], [], [], []
            ids = {"v2": [], **{f"bloom/np{x}": [] for x in N_PROBES[name]},
                   **{f"exact/np{x}": [] for x in N_PROBES[name]}}  # fmt: skip
            for q, qa in zip(qs, qa_rows, strict=True):
                ex = filt.evaluate_mask(filt.prepare_queries(qa))
                expr = to_expr(qa)
                bm = bloom_mask(bloom, expr, n_clauses, n)
                fn_all.append((ex & ~bm).sum(1))
                fp_all.append((~ex & bm).sum(1))
                pass_all.append(ex.sum(1))
                for c in rev_c:  # the NOT clause alone
                    qn = torch.full_like(qa, -1)
                    qn[:, c] = qa[:, c]
                    exn = filt.evaluate_mask(filt.prepare_queries(qn))
                    bn = bloom_mask(bloom, to_expr(qn), n_clauses, n)
                    fn_not.append((exn & ~bn).sum(1))
                    pass_not.append(exn.sum(1))
                prep = filt.prepare_queries(qa)
                counts.append(filt.evaluate_indices(prep)[1].long())
                ids["v2"].append(v2(q, prep)[0])
                official_mod.queries_to_expressions = to_expr
                try:
                    bprep = bloom.prepare_queries(qa)
                finally:
                    official_mod.queries_to_expressions = adapter.queries_to_expressions
                eprep = exact.prepare_queries(qa)
                for x in N_PROBES[name]:
                    bloom.set_query_params(n_probe=x)
                    exact.set_query_params(n_probe=x)
                    ids[f"bloom/np{x}"].append(bloom(q, bprep)[0])
                    ids[f"exact/np{x}"].append(exact(q, eprep)[0])
            cat = {k: torch.cat(v) for k, v in ids.items()}
            counts_t, pass_all_t = torch.cat(counts), torch.cat(pass_all)
            fn_all_t, fp_all_t = torch.cat(fn_all), torch.cat(fp_all)
            rec = {"cell": cell, "dataset": name, "dim": ds.dim, "n_items": n, "sweep": sweep,
                   "clauses": list(clauses), "reverse_clauses": rev_c, "n_queries": int(rows.numel()),
                   "bloom": {**BLOOM, "recheck": False, "width": "configured"}, **p,
                   "same_kmeans_index": same_index, "env": provenance,
                   "mean_pass_rate": pass_all_t.double().mean().item() / n,
                   "filter_fn_per_query": per_query(fn_all_t),
                   "filter_fn_rate": (fn_all_t.sum() / pass_all_t.sum().clamp(min=1)).item(),
                   "filter_fn_rate_per_query": per_query(fn_all_t / pass_all_t.clamp(min=1)),
                   "filter_fp_per_query": per_query(fp_all_t),
                   "filter_fp_rate_vs_failing": (fp_all_t.sum() / (n - pass_all_t).sum().clamp(min=1)).item(),
                   "queries_with_zero_pass": int((pass_all_t == 0).sum())}  # fmt: skip
            if fn_not:
                fnn, pn = torch.cat(fn_not), torch.cat(pass_not)
                rec["not_clause_fn_per_query"] = per_query(fnn)
                rec["not_clause_fn_rate"] = (fnn.sum() / pn.sum().clamp(min=1)).item()
                rec["not_clause_fn_rate_per_query"] = per_query(fnn / pn.clamp(min=1))
            rec["recall"] = {
                k: recall(v, cat["v2"], counts_t) for k, v in cat.items() if k != "v2"
            }
            recs.append(rec)
            log(cell, {k: v for k, v in rec.items() if "rate" in k and not isinstance(v, dict)},
                {k: round(v, 4) for k, v in rec["recall"].items()})  # fmt: skip
        del bloom, exact, v2, filt, inp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main(sys.argv[1:] or ["goodreads", "pubmed"])
