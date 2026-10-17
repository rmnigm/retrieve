"""Markdown tables of the gpu-checks records: ``python summarize.py check1|check2|check3``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/scratch/gpu-checks/out")


def rows(check: str, tag: str) -> list[dict]:
    path = OUT / check / f"{tag}.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines() if x]


def f(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def check1() -> None:
    print("| dataset | pass | bs | V1 graph | V2 graph | V1 eager | V2 eager | native mask | native index "
          "| mask / best graph | index / best graph | index / best eager | overlap | unstable |")  # fmt: skip
    print("|" + "---|" * 14)
    for r in rows("check1", "native"):
        m = {a: p.get("median_ms") for a, p in r["perf"].items()}
        best_e = min(m["triton_v1/eager"], m["triton_v2/eager"])
        uns = [a for a, p in r["perf"].items() if p.get("unstable")]
        print(f"| {r['dataset']} | {r['rate']} | {r['bs']} | {f(m['triton_v1/graph'])} | "
              f"{f(m['triton_v2/graph'])} | {f(m['triton_v1/eager'])} | {f(m['triton_v2/eager'])} | "
              f"{f(m['native_mask/eager'])} | {f(m['native_index/eager'])} | "
              f"{f(r['ratio']['native_mask/vs_best_graph'], 1)} | "
              f"{f(r['ratio']['native_index/vs_best_graph'], 1)} | "
              f"{f(m['native_index/eager'] / best_e, 1)} | "
              f"{f(r['overlap_native_index_vs_triton_v2'], 4)} | {', '.join(uns) or 'no'} |")  # fmt: skip


def check2() -> None:
    rs = rows("check2", "v3bits")
    print("Recall@100 vs the exact filtered top-100 (pool label: pool size)\n")
    for r in rs:
        pools = r["pools"]
        arms = sorted({k.split("/")[0] for k in r["recall"] if "/" in k})
        print(
            f"**{r['cell']}** pass {r['pass_rate']:.4f}, mean pass {r['mean_pass_count']:.0f}\n"
        )
        print("| arm | " + " | ".join(f"{p} ({pools[p]})" for p in pools) + " |")
        print("|---|" + "---|" * len(pools))
        for a in arms:
            print(
                f"| {a} | "
                + " | ".join(f(r["recall"][f"{a}/{p}"], 4) for p in pools)
                + " |"
            )
        print()
    print("Timing (graph median ms; ratio = V3 / faster of V1, V2 in the same job)\n")
    print("| cell | bs | V1 | V2 | variant | median ms | ratio | recall | unstable |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        for key, ratio in r.get("ratio_to_best_exact", {}).items():
            v, bs = key.rsplit("/", 1)
            p = r["perf"]
            print(f"| {r['cell']} | {bs[2:]} | {f(p[f'v1/{bs}'].get('median_ms'))} | "
                  f"{f(p[f'v2/{bs}'].get('median_ms'))} | {v} | {f(p[key].get('median_ms'))} | "
                  f"{f(ratio, 2)} | {f(r['recall'][v], 4)} | "
                  f"{'yes' if any(x.get('unstable') for k, x in p.items() if k.endswith(bs)) else 'no'} |")  # fmt: skip


def check3() -> None:
    for r in rows("check3", "bloom_not"):
        print(f"**{r['cell']}** clauses {r['clauses']} (NOT on {r['reverse_clauses']}), "
              f"{r['n_queries']} queries, pass {r['mean_pass_rate']:.4f}, same index {r['same_kmeans_index']}\n")  # fmt: skip
        nf = r.get("not_clause_fn_rate_per_query", {})
        print(f"- NOT clause FN rate {f(r.get('not_clause_fn_rate'), 6)} (per query mean "
              f"{f(nf.get('mean'), 6)}, max {f(nf.get('max'), 6)}, share of queries with any FN "
              f"{f(nf.get('frac_queries_nonzero'), 4)})")  # fmt: skip
        ff = r["filter_fn_rate_per_query"]
        print(f"- whole-filter FN rate {f(r['filter_fn_rate'], 6)} (per query mean {f(ff['mean'], 6)}, "
              f"max {f(ff['max'], 6)}, share with any FN {f(ff['frac_queries_nonzero'], 4)}); "
              f"FP rate over failing items {f(r['filter_fp_rate_vs_failing'], 6)}")  # fmt: skip
        print(
            "- recall@100: "
            + ", ".join(f"{k} {f(v, 4)}" for k, v in r["recall"].items())
            + "\n"
        )


if __name__ == "__main__":
    globals()[sys.argv[1]]()
