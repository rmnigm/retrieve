"""FULL-RERUN GPU-h per leg and pod (README.md § Sizing), from one seed's cost: measured walls of tonight's legs where one ran
(B1 A1-bloom, B2 A2-bloom / -k1000-e / -synth, A1-small, A1-pubmed, A1b, A3, A4, L1b, L3, L4 on pod 1 and pod e logs), the final
pass's summed `elapsed_s` per suite (merged-4016, out-of-study arms dropped), and NUMBERS plan.yaml estimates otherwise. ×1.2 for
process start, data load and compile (plan.yaml's rule). Credit: ~$115 at ~$3.71/h for both pods, one A100 each.

usage: sizing.py   (prints the markdown table README.md carries)
"""

OVERHEAD, CREDIT_USD, USD_PER_POD_HOUR = 1.2, 115, 3.71
# leg: (pod 1 GPU-h, pod e GPU-h) for one seed / draw, seeds full, seeds lean, note
LEGS = {
    "fr-curve (real, clause + bloom)": (
        (5.0, 5.1),
        3,
        1,
        "A1 + B1; pod 1 PubMed, LAION 1-10 M; pod e goodreads, arXiv, LAION 30 M",
    ),
    "fr-curve-synth (per draw)": (
        (5.3, 4.5),
        3,
        1,
        "pubmed-synth new (~3.5); LAION 10 / 30 M synth bloom new",
    ),
    "fr-linr-* (exact, one seed)": (
        (2.6, 4.9),
        1,
        1,
        "L1 without V2 bs 256 PubMed, L1b, L3, L4; PubMed synth crossover (~3.7) on pod e",
    ),
    "fr-v3pool*": ((2.0, 3.0), 3, 1, "L2 + L2b; PubMed's on pod e"),
    "fr-bloomwidth": ((0.0, 0.35), 1, 1, "A4, on pod e"),
    "fr-yambda": ((0.0, 0.1), 3, 3, "A5 (6 min) + oracles"),
    "fr-timed-bloom (k 100, + V3)": ((1.9, 2.4), 3, 3, "B2 4.2 h measured"),
    "fr-timed-bloom-k1000": ((0.5, 1.5), 3, 3, "B2 k 1000 1.5 h measured (pod e)"),
    "fr-timed-bloom-k2048": ((0.5, 0.6), 3, 3, "new; A3 1.1 h as the basis"),
    "fr-timed-bloom-synth (per draw)": (
        (1.2, 1.8),
        3,
        3,
        "B2 synth 0.5 h + LAION 10 / 30 M + PubMed synth",
    ),
    "fr-timed-clause (reference row)": (
        (8.0, 7.0),
        3,
        0,
        "A2 legs at k 100; trim 2 drops it",
    ),
    "final-pass random suites, seeds 1-2 (prep_config.py)": (
        (0.8, 4.7),
        2,
        2,
        "official / V3 suites raised to 3 seeds: synth, codesign*, c7, v3bits, deep, h2h, bloomwidth*",
    ),
    "final-pass suites": ((17.4, 8.6), 1, 1, "merged-4016 elapsed_s, kept arms 26.0 h"),
    "final-pass trims (filter seed 0, no deep / h2h)": ((-5.5, -3.3), 0, 1, "trim 3"),
    "untimed: grade_bloom, fp / ceiling_adapted": (
        (0.6, 0.4),
        3,
        3,
        "gls / product_recall / score_gaps are CPU",
    ),
    "META-FORK gate grid at the tag": (
        (2.5, 1.5),
        1,
        1,
        "not measured here: size from head-82fafb6's logs",
    ),
}


def table(col):
    rows, tot = [], [0.0, 0.0]
    for leg, ((p1, pe), full, lean, note) in LEGS.items():
        n = {"full": full, "lean": lean, "seed 0 pass": min(full, 1)}[col]
        a, b = p1 * n * OVERHEAD, pe * n * OVERHEAD
        if n:
            rows.append(f"| {leg} | {n} | {a:.1f} | {b:.1f} | {note} |")
        tot[0] += a
        tot[1] += b
    return rows, tot


budget = CREDIT_USD / USD_PER_POD_HOUR
print(
    f"Credit: {CREDIT_USD} USD / {USD_PER_POD_HOUR} USD an hour for both pods = {budget:.0f} h of wall per pod.\n"
)
for col in ("full", "seed 0 pass", "lean"):
    rows, (a, b) = table(col)
    print(
        f"**{col}**\n\n| leg | seeds | pod 1 GPU-h | pod e GPU-h | basis |\n|---|---|---|---|---|"
    )
    print("\n".join(rows))
    wall = max(a, b)
    print(
        f"| **total** | | **{a:.0f}** | **{b:.0f}** | wall ≈ {wall:.0f} h, {wall * USD_PER_POD_HOUR:.0f} USD |\n"
    )
