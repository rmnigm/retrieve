"""B3 — render the markdown tables of the validation record from the raw JSON.

Reads the end-to-end records (``e2e/**/*.jsonl``, harness schema 2) and the kernel-only
JSONs (``kernel_<dataset>.json``) written by ``b3_kernel_h2h.py``; prints the tables to
stdout. No measurement happens here — it is the formatting step, kept so the tables in the
plan's record can be re-derived from the raw files.

Usage:  python b3_tables.py <artifacts-dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ARMS = {"triton": "triton", "torch": "torch", "official": "official"}


def load_e2e(root: Path) -> list[dict]:
    recs = []
    for f in sorted(root.glob("*/*.jsonl")):
        if f.name.endswith(".samples.jsonl"):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return [r for r in recs if r.get("algo") == "silvertorch"]


def perf_of(rec: dict, bs: int, k: int, mode: str) -> dict | None:
    for e in rec.get("perf") or []:
        if e["bs"] == bs and e["k"] == k and e["mode"] == mode:
            return e
    return None


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def e2e_tables(recs: list[dict]) -> None:
    print("\n#### End to end — eager latency, one cell per (dataset, filter, backend)\n")
    print("| dataset | filter | sweep | n_probe | backend | k | bs | eager median ms | p99 ms |"
          " qps | spread | sm_mhz | peak MiB | graph median ms |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: (r["dataset"], r["filter_kind"],
                                         r["params"].get("n_probe", 0), r["backend"])):
        for k in sorted({e["k"] for e in (r.get("perf") or [])}):
            for bs in sorted({e["bs"] for e in (r.get("perf") or [])}):
                e = perf_of(r, bs, k, "eager")
                g = perf_of(r, bs, k, "graph")
                if e is None:
                    continue
                print(f"| {r['dataset']} | {r['filter_kind']} | {r['sweep']} | "
                      f"{r['params'].get('n_probe', '—')} | {r['backend']} | {k} | {bs} | "
                      f"{fmt(e['median_ms'])} | {fmt(e['p99_ms'])} | {fmt(e['qps'], 0)} | "
                      f"{fmt(e['spread'], 3)} | {e['sm_mhz']} | {fmt(e['peak_fwd_mib'], 1)} | "
                      f"{fmt(g['median_ms']) if g and g.get('median_ms') else (g or {}).get('reason', '—')} |")

    print("\n#### End to end — quality and parity\n")
    print("| dataset | filter | sweep | n_probe | backend | status | recall@100 (oracle) |"
          " jaccard_vs_first@100 | score_max_abs_diff | parity | build_s | index_mib |"
          " pass_rate | bloom_fp_rate | unstable |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(recs, key=lambda r: (r["dataset"], r["filter_kind"],
                                         r["params"].get("n_probe", 0), r["backend"])):
        q = r.get("quality") or {}
        orc = (q.get("oracle") or {})
        print(f"| {r['dataset']} | {r['filter_kind']} | {r['sweep']} | "
              f"{r['params'].get('n_probe', '—')} | {r['backend']} | {r['status']} | "
              f"{fmt(orc.get('recall@100'), 6)} | {fmt(q.get('jaccard_vs_first@100'), 6)} | "
              f"{fmt(q.get('score_max_abs_diff'), 3 if q.get('score_max_abs_diff') else 4)} | "
              f"{q.get('parity', '—')} | {fmt(r.get('build_s'), 1)} | {fmt(r.get('index_mib'), 1)} | "
              f"{fmt(r.get('pass_rate'), 4)} | {fmt(r.get('bloom_fp_rate'), 6)} | "
              f"{r.get('unstable')} |")


def kernel_tables(js: list[dict]) -> None:
    print("\n#### Kernel-only — Algorithm 1 phases 2+3, phase 1 hoisted out\n")
    print("| dataset | mode | bs | arm | wall median ms | spread | sm_mhz | device µs total |"
          " scorer µs | mask µs | prep µs | topk µs | launches | peak MiB |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for j in js:
        for r in j["tiers"]["A_phase23"]:
            c = r["device_us_by_class"]
            print(f"| {j['dataset']} | {r['mode']} | {r['bs']} | {r['arm']} | "
                  f"{fmt(r['median_ms'])} | {fmt(r['spread'], 3)} | {r['sm_mhz']} | "
                  f"{fmt(r['device_us_total'], 1)} | {fmt(c.get('scorer'), 1)} | "
                  f"{fmt(c.get('mask'), 1)} | {fmt(c.get('prep'), 1)} | {fmt(c.get('topk'), 1)} | "
                  f"{r['n_kernel_launches']} | {fmt(r['peak_fwd_mib'], 1)} |")

    print("\n#### Kernel-only — parity of each official arm against Triton\n")
    print("| dataset | mode | bs | score path | jaccard@100 | score_max_abs_diff | n queries |")
    print("|---|---|---|---|---|---|---|")
    for j in js:
        for r in j["tiers"]["C_parity"]:
            print(f"| {j['dataset']} | {r['mode']} | {r['bs']} | {r['score_path']} | "
                  f"{fmt(r['jaccard_at_k'], 6)} | {r['score_max_abs_diff']:.3e} | "
                  f"{r['n_queries']} |")

    print("\n#### Why the fp16 path costs rank agreement — top-k boundary gaps\n")
    print("| dataset | mode | gap p10 | gap median | score@k median | gap/score median |"
          " fp16 ulp | rows with gap < ulp |")
    print("|---|---|---|---|---|---|---|---|")
    for j in js:
        for r in j["tiers"].get("C_boundary", []):
            if not r:
                continue
            print(f"| {j['dataset']} | {r['mode']} | {r['gap_p10']:.3e} | "
                  f"{r['gap_median']:.3e} | {r['score_at_k_median']:.4f} | "
                  f"{r['gap_over_score_median']:.3e} | {r['fp16_ulp_of_score_median']:.3e} | "
                  f"{r['frac_gap_below_fp16_ulp']:.4f} |")

    print("\n#### Phase 2 alone (bs=16)\n")
    print("| dataset | op | arm | median ms | p99 ms | spread | timer |")
    print("|---|---|---|---|---|---|---|")
    for j in js:
        for r in j["tiers"]["B_phase2"]:
            print(f"| {j['dataset']} | {r['op']} | {r['arm']} | {fmt(r['median_ms'], 4)} | "
                  f"{fmt(r['p99_ms'], 4)} | {fmt(r.get('spread'), 3)} | {r.get('timer', '—')} |")

    print("\n#### Bloom selectivity and memory at the shipped settings\n")
    print("| dataset | exact pass rate | ours FP rate | official FP rate | ours MiB |"
          " official MiB | ours bits/doc | official b_multiplier |")
    print("|---|---|---|---|---|---|---|---|")
    for j in js:
        b = j["tiers"].get("C_bloom")
        if not b:
            continue
        print(f"| {j['dataset']} | {fmt(b['exact_pass_rate'], 4)} | "
              f"{fmt(b['ours_bloom_fp_rate'], 6)} | {fmt(b['official_bloom_fp_rate'], 6)} | "
              f"{fmt(b['ours_index_mib'], 1)} | {fmt(b['official_index_mib'], 1)} | "
              f"{b['ours_bits_per_doc']} | {b['official_b_multiplier']} |")


def main() -> None:
    root = Path(sys.argv[1])
    recs = load_e2e(root / "e2e")
    if recs:
        e2e_tables(recs)
    js = [json.loads(p.read_text()) for p in sorted(root.glob("kernel_*.json"))]
    if js:
        kernel_tables(js)


if __name__ == "__main__":
    main()
