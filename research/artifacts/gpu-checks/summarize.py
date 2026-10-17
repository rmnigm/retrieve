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


def first_pool(r: dict, arm: str, bar: float) -> str:
    """The smallest pool (in items) whose recall reaches ``bar``, or ``—``."""
    hits = [n for p, n in r["pools"].items() if r["recall"][f"{arm}/{p}"] >= bar]
    return f"{min(hits):,}" if hits else "—"


def check2() -> None:
    rs = rows("check2", "v3bits")
    arms = ["v3_oporp128", "v3_simhash128", "v3_simhash512", "v3_oporp768"]
    print(
        "Recall@100 against the exact filtered top-100 at LiNR's pool (1 % of N) and at the pool fraction 0.01 of the"
    )
    print(
        "mean pass count (floor 2000); then the smallest pool (items) reaching recall 0.95 per arm.\n"
    )
    print("| cell | pass | " + " | ".join(f"{a[3:]} @1%N / @f0.01" for a in arms) + " | "
          + " | ".join(f"{a[3:]} pool≥0.95" for a in arms) + " |")  # fmt: skip
    print("|---|---|" + "---|" * (2 * len(arms)))
    for r in rs:
        rec = r["recall"]
        cells = [
            f"{f(rec[f'{a}/linrN'], 3)} / {f(rec[f'{a}/f0.01'], 3)}"
            if f"{a}/linrN" in rec
            else "—"
            for a in arms
        ]
        pools = [first_pool(r, a, 0.95) if f"{a}/linrN" in rec else "—" for a in arms]
        print(
            f"| {r['cell']} | {r['pass_rate']:.4f} | "
            + " | ".join(cells)
            + " | "
            + " | ".join(pools)
            + " |"
        )
    print(
        "\nTiming, graph replay: V1 / V2 medians (ms); V3 as a ratio to the faster of V1 and V2 timed in the same group.\n"
    )
    print("| cell | bs | V1 | V2 | oporp128 1%N | oporp128 f0.01 | oporp128 f0.05 | simhash512 1%N | simhash512 f0.01 "
          "| simhash512 f0.05 | unstable |")  # fmt: skip
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        ratio = r.get("ratio_to_best_exact", {})
        for bs in ("bs1", "bs16"):
            if f"v3_oporp128/linrN/{bs}" not in ratio:
                continue
            p = r["perf"]
            vs = [
                f(ratio[f"{a}/{q}/{bs}"], 2)
                for a in ("v3_oporp128", "v3_simhash512")
                for q in ("linrN", "f0.01", "f0.05")
            ]
            uns = [
                k.rsplit("/", 1)[0]
                for k, x in p.items()
                if k.endswith(bs) and x.get("unstable")
            ]
            print(
                f"| {r['cell']} | {bs[2:]} | {f(p[f'v1/{bs}'].get('median_ms'))} | {f(p[f'v2/{bs}'].get('median_ms'))} | "
                + " | ".join(vs)
                + f" | {', '.join(uns) or 'no'} |"
            )


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
