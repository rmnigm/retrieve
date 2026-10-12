"""Writes suites.yaml for the NUMBERS legs (README.md): the LiNR-only legs, the k 1000 check and the quality-only probe
curves of the adapted official SilverTorch (`official-fork`). The timed points come later from pick_points.py.

usage: gen_suites.py > suites.yaml
"""

import sys

import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


P = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BIG_LISTS = [4096, 16384, 65536]  # D-b: PubMed, YFCC, LAION 10 M, LAION 30 M
BENCH = {  # dataset: (dim, n_lists, clause sweeps, bloom sweeps)
    "goodreads": (128, [4096], ["c0_genre", "c1_lang_reverse", "all4"], ["c0_genre"]),
    "goodreads-synth": (128, [4096], ["p0001", "p001", "p01", "p1"], []),
    "arxiv": (
        128,
        [2048],
        ["c3_nversions", "c0_maincat", "all4"],
        ["c0_maincat", "all4"],
    ),
    "arxiv-synth": (128, [2048], ["p0001", "p001", "p01", "p1"], []),
    "arxiv-corr-synth": (128, [2048], ["c001", "c003", "c01"], []),
    "laion1m": (256, [512], ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion1m-synth": (256, [512], ["p0001", "p001", "p01", "p1"], []),
    "laion3m": (256, [2048], ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion3m-synth": (256, [2048], ["p0001", "p001", "p01", "p1"], []),
    "pubmed": (768, BIG_LISTS, ["c0_mesh", "c3_journal_reverse", "all5"], ["c0_mesh"]),
    "yfcc10m": (192, BIG_LISTS, ["tags_and"], []),
    "yfcc10m-synth": (192, BIG_LISTS, ["p0001", "p001", "p01", "p1"], []),
    "laion10m": (256, BIG_LISTS, ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion10m-synth": (256, BIG_LISTS, ["p0001", "p001", "p01", "p1"], []),
    "laion30m": (256, BIG_LISTS, ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion30m-synth": (256, BIG_LISTS, ["p0001", "p001", "p01", "p1"], []),
}
REAL = ["goodreads", "arxiv", "pubmed", "yfcc10m"]
LAION = ["laion1m", "laion3m", "laion10m", "laion30m"]
DIMS = [128, 192, 256, 768]
V12 = [
    {"algo": "linr_v1_filter_mask", "backends": ["triton"]},
    {"algo": "linr_v2", "backends": ["triton"]},
]


def sweeps(names, kinds=("clause",)):
    return {
        d: sorted({s for k in kinds for s in BENCH[d][2 if k == "clause" else 3]})
        for d in names
    }


def suite(
    datasets, arms, ks, bss, kinds=("clause",), perf=True, interleave=None, sw=None
):
    s = {
        "datasets": datasets,
        "dims": sorted({BENCH[d][0] for d in datasets}),
        "filter_kinds": list(kinds),
        "ks": ks,
        "batch_sizes": bss,
        "seeds": [0],
        "sweeps": sw or sweeps(datasets, kinds),
        "arms": arms,
    }
    if not perf:
        s = {"perf": False, **s}
    if interleave:
        s["interleave"] = interleave
    return s


def curve_arms(datasets, wide):
    """One arm per (kind, n_lists): narrow = n_probe <= 1024, wide = 2048 ... n_lists / 4 (the k 1000 limit splits them),
    None = all of them (one build per index when the k 1000 check passes)."""
    arms = []
    for kind in ("clause", "bloom"):
        lists = sorted({n for d in datasets for n in BENCH[d][1]})
        for n in lists:
            probes = [
                p for p in P if p <= n // 4 and (wide is None or (p > 1024) == wide)
            ]
            ds = {
                d: {"sweeps": BENCH[d][2 if kind == "clause" else 3]}
                for d in datasets
                if n in BENCH[d][1] and BENCH[d][2 if kind == "clause" else 3]
            }
            if probes and ds:
                arms.append(
                    {
                        "algo": "silvertorch",
                        "backends": ["official-fork"],
                        "filter_kinds": [kind],
                        "build": {"n_lists": [n]},
                        "query": {"n_probe": probes},
                        "datasets": ds,
                    }
                )
    return arms


SMALL_LISTS = {
    "laion1m": [4096, 8192],
    "laion3m": [8192, 16384],
}  # A1b (user: go bigger at 1-3 M too)


def small_lists_arms(wide, backend="official-fork"):
    arms = []
    for kind in ("clause", "bloom"):
        for ds, lists in SMALL_LISTS.items():
            for n in lists:
                probes = [
                    p for p in P if p <= n // 4 and (wide is None or (p > 1024) == wide)
                ]
                if probes:
                    arms.append(
                        {
                            "algo": "silvertorch",
                            "backends": [backend],
                            "filter_kinds": [kind],
                            "build": {"n_lists": [n]},
                            "query": {"n_probe": probes},
                            "datasets": {ds: {}},
                        }
                    )
    return arms


BLOOM = {  # the user's bloom directive (2026-10-12): SilverTorch bloom, recheck off; dataset: (n_lists, bloom sweeps)
    "goodreads": ([4096], ["c0_genre"]),
    "arxiv": ([2048], ["c0_maincat", "c3_nversions", "all4"]),
    "laion1m": ([512, 4096, 8192], ["c0_domain", "tags4"]),
    "laion3m": ([2048, 8192, 16384], ["c0_domain", "tags4"]),
    "pubmed": (BIG_LISTS, ["c0_mesh"]),
    "laion10m": (BIG_LISTS, ["c0_domain", "tags4"]),
    "laion30m": (BIG_LISTS, ["c0_domain", "tags4"]),
}
BLOOM_SYNTH = {
    d: ([BENCH[d][1][0]], ["p0001", "p001", "p01", "p1"])
    for d in ["goodreads-synth", "arxiv-synth", "laion1m-synth", "laion3m-synth"]
}


def bloom_curve_arms(spec):
    """One arm per (dataset, n_lists): SilverTorch bloom on the adapted backend with the recheck off, n_probe doubling to n_lists / 4."""
    return [
        {
            "algo": "silvertorch",
            "backends": ["official-fork"],
            "filter_kinds": ["bloom"],
            "build": {"n_lists": [n], "recheck": [False]},
            "query": {"n_probe": [p for p in P if p <= n // 4]},
            "datasets": {d: {"sweeps": sweeps}},
        }
        for d, (lists, sweeps) in spec.items()
        for n in lists
    ]


def bloom_curve_suites(spec, name):
    """One suite per dim (a dataset config may list several dims; each dataset runs its own)."""
    out = {}
    for dim in sorted({BENCH[d][0] for d in spec}):
        sub = {d: v for d, v in spec.items() if BENCH[d][0] == dim}
        s = suite(
            list(sub),
            bloom_curve_arms(sub),
            [100, 1000],
            [16],
            kinds=("bloom",),
            perf=False,
            sw={d: sw for d, (_, sw) in sub.items()},
        )
        out[f"{name}-d{dim}"] = {**s, "dims": [dim]}
    return out


ALL = list(BENCH)
BIG = [d for d in ALL if BENCH[d][1] == BIG_LISTS]
YFCC = ["yfcc10m", "yfcc10m-synth"]
WIDE = [
    d for d in BIG if d not in YFCC
]  # YFCC is int8-capped (A1-yfcc): no wide curve, one cap point instead (n-curve-q-yfcc-cap)
blocks = {
    "bloom": {"m_bits": 1024, "k_hash": 5},
    # LiNR-only legs (library e16512f5, start now)
    "n-linr-bs64": suite(
        REAL,
        V12,
        [100, 1000],
        [64, 256],
        interleave=[{"by": "algo", "values": ["linr_v1_filter_mask", "linr_v2"]}],
    ),
    "n-linr-bs64-laion": suite(
        LAION,
        V12,
        [100],
        [64],
        interleave=[{"by": "algo", "values": ["linr_v1_filter_mask", "linr_v2"]}],
    ),
    "n-v3pool": suite(
        REAL,
        [
            {
                "algo": "linr_v3",
                "backends": ["triton"],
                "query": {"candidate_pool_frac": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]},
            }
        ],
        [100, 1000],
        [1, 16],
    ),
    "n-v3pool-laion": suite(
        LAION,
        [
            {
                "algo": "linr_v3",
                "backends": ["triton"],
                "query": {"candidate_pool_frac": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]},
            }
        ],
        [100],
        [1, 16],
    ),
    "n-linr-corr": suite(
        ["arxiv-corr-synth"],
        V12,
        [100, 1000],
        [1, 16],
        interleave=[{"by": "algo", "values": ["linr_v1_filter_mask", "linr_v2"]}],
    ),
    "n-linr-k2048": suite(
        ["goodreads", "arxiv", "pubmed"],
        V12,
        [2048],
        [1, 16],
        sw={"goodreads": ["c0_genre"], "arxiv": ["c0_maincat"], "pubmed": ["c0_mesh"]},
    ),
    # adapted official SilverTorch (frozen fork head)
    "n-k1000-check": suite(
        ["yfcc10m", "laion30m"],
        [
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "build": {"n_lists": [4096]},
                "query": {"n_probe": [1024]},
                "datasets": {"yfcc10m": {}},
            },
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "build": {"n_lists": [16384]},
                "query": {"n_probe": [2048, 4096]},
            },
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "build": {"n_lists": [65536]},
                "query": {"n_probe": [8192, 16384]},
                "datasets": {"yfcc10m": {}},
            },
        ],
        [1000],
        [16],
        sw={"yfcc10m": ["tags_and"], "laion30m": ["tags4"]},
    ),
    "n-curve-q": suite(
        ALL,
        curve_arms(ALL, wide=False),
        [100, 1000],
        [16],
        kinds=("clause", "bloom"),
        perf=False,
    ),
    "n-curve-q-wide": suite(
        WIDE,
        curve_arms(WIDE, wide=True),
        [100, 1000],
        [16],
        kinds=("clause", "bloom"),
        perf=False,
    ),
    "n-curve-q-laion-small": suite(
        list(SMALL_LISTS),
        small_lists_arms(False),
        [100, 1000],
        [16],
        kinds=("clause", "bloom"),
        perf=False,
    ),
    "n-curve-q-laion-small-wide": suite(
        list(SMALL_LISTS),
        small_lists_arms(True),
        [100, 1000],
        [16],
        kinds=("clause", "bloom"),
        perf=False,
    ),
    # C6 ceiling check (ceiling_check.py): one Triton index at the largest n_lists and its cap; quality only, same ids as official-fork
    "n-ceil-laion-small": suite(
        list(SMALL_LISTS),
        [
            {
                "algo": "silvertorch",
                "backends": ["triton"],
                "build": {"n_lists": [8192]},
                "query": {"n_probe": [2048]},
                "datasets": {"laion1m": {}},
            },
            {
                "algo": "silvertorch",
                "backends": ["triton"],
                "build": {"n_lists": [16384]},
                "query": {"n_probe": [4096]},
                "datasets": {"laion3m": {}},
            },
        ],
        [100],
        [16],
        perf=False,
    ),
    # C6 evidence for YFCC (controller): one quality-only point per filter at 16384 lists and their cap; read by no picker
    "n-curve-q-yfcc-cap": suite(
        YFCC,
        [
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "filter_kinds": ["clause"],
                "build": {"n_lists": [16384]},
                "query": {"n_probe": [4096]},
            }
        ],
        [100, 1000],
        [16],
        perf=False,
    ),
    # C4 / F4a on the study's arm (controller c3 ruling 1): the final pass's bloomwidth grid, adapted backend, quality only
    "n-bloomwidth": suite(
        ["goodreads", "arxiv", "pubmed"],
        [
            {
                "algo": "silvertorch",
                "backends": ["official-fork"],
                "build": {"m_bits": [64, 128, 256, 512, 1024, 2048], "k_hash": [3, 5]},
            }
        ],
        [100, 1000],
        [16],
        kinds=("bloom",),
        perf=False,
        sw={
            "goodreads": ["c0_genre"],
            "arxiv": ["all4", "c0_maincat", "c3_nversions"],
            "pubmed": ["c0_mesh"],
        },
    ),
    # B1 (controller bloom plan): bloom curves, recheck off. Priority 1 = real filters, every size; 1b = the small synth sets
    **bloom_curve_suites(BLOOM, "n-bloom-curve"),
    **bloom_curve_suites(BLOOM_SYNTH, "n-bloom-curve-synth"),
    "n-curve-q-all": suite(
        WIDE,
        curve_arms(WIDE, wide=None),
        [100, 1000],
        [16],
        kinds=("clause", "bloom"),
        perf=False,
    ),
}
yaml.dump(blocks, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
