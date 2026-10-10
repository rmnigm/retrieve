"""C1 across N at one tag (README.md § c1): the paired V2/V1 ratio vs pass rate per scale, bs 1 and 16,
read from `f1x-v2-over-v1.csv` (figures.py), with the crossover interpolated between the bracketing rates.

usage: c1.py OUT [CODE_VERSION]   (default v2.9; OUT holds figures.py's output)
"""

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402


def crossover(pts):
    """First p where the ratio crosses 1, log-p and linear-p interpolated; None if it never does."""
    for (p0, r0), (p1, r1) in zip(pts, pts[1:]):
        if r0 < 1 <= r1:
            t = (1 - r0) / (r1 - r0)
            return math.exp(math.log(p0) + t * math.log(p1 / p0)), p0 + t * (p1 - p0)
    return None


def main():
    out = Path(sys.argv[1])
    c = sys.argv[2] if len(sys.argv) > 2 else "v2.9"
    rows = [
        r
        for r in csv.DictReader(open(out / "f1x-v2-over-v1.csv"))
        if r["code_version"] == c and r["filter_kind"] == "clause"
    ]
    series = sorted({(int(r["n_items"]), r["dataset"], r["box"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    table = []
    for ax, bs in zip(axes, ("1", "16")):
        for n, ds, bx in series:
            pts = sorted(
                (float(r["p"]), float(r["v2_over_v1"]), float(r["ci_lo"]), float(r["ci_hi"]))
                for r in rows
                if (int(r["n_items"]), r["dataset"], r["box"], r["bs"]) == (n, ds, bx, bs)
            )
            if not pts:
                continue
            ax.errorbar(
                [q[0] for q in pts],
                [q[1] for q in pts],
                yerr=[[q[1] - q[2] for q in pts], [q[3] - q[1] for q in pts]],
                marker="o",
                ms=4,
                capsize=2,
                label=f"{n / 1e6:.0f} M {ds} (box {bx})",
            )
            x = crossover([(q[0], q[1]) for q in pts])
            table.append((c, ds, n, bx, bs, len(pts), pts[0][1], pts[-1][1], *(x or ("", ""))))
        ax.axhline(1.0, color="k", lw=0.7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(FixedLocator([0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4]))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("pass rate p")
        ax.set_title(f"B = {bs}")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("V2 / V1 graph p50, paired, 95 % CI (< 1: V2 faster)")
    fig.suptitle(
        f"C1: LiNR V2 / V1 vs pass rate across N at {c}, uniform synth clause, k 100 "
        "(ratios inside one leg and box; NOT CITABLE)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out / f"c1-across-n-{c}.png", dpi=130)
    with open(out / f"c1-across-n-{c}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "code_version",
                "dataset",
                "n_items",
                "box",
                "bs",
                "rates",
                "ratio_at_min_p",
                "ratio_at_p1",
                "crossover_logp",
                "crossover_linp",
            ]
        )
        w.writerows(table)
    print(f"{len(table)} rows")


if __name__ == "__main__":
    main()
