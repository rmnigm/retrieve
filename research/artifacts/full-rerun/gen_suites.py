"""Writes suites.yaml for FULL-RERUN's stage 1 (README.md): the adapted SilverTorch probe curves (clause, and bloom with the
recheck off), the LiNR-only legs, the bloom width grid and Yambda. Stage 2's timed suites come from the curves through
pick.sh. The final pass's suites (evaluation/config/suites.yaml) run beside these unchanged.

Seeds (user, 2026-10-12; controller C3 / C4, 2026-10-13): 0-2 wherever an index build, a candidate pool or a synthetic draw
is random (SilverTorch, LiNR V3, every synth dataset); 1 for the exact LiNR legs. A synth dataset's draw d is the config
`<ds>-synth-s<d>` (draw 0 is `<ds>-synth`) and runs at job seed d, so one suite per draw.

`--lean` (README § Sizing, the first trims): the curves and the V3 pool legs at seed 0 / draw 0 only. The picks read seed 0
either way, and the timed suites still run every seed and draw, so the seeds then sit on the timed points alone.

usage: gen_suites.py [--lean] > suites.yaml
"""

import sys

import yaml


class NoAlias(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


P = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
BIG = [4096, 16384, 65536]
RATES4 = ["p0001", "p001", "p01", "p1"]
SEEDS = [0, 1, 2]
CURVE_SEEDS = [0] if "--lean" in sys.argv[1:] else SEEDS
BENCH = {  # dataset: (dim, n_lists, clause sweeps, bloom sweeps)
    "goodreads": (128, [4096], ["c0_genre", "c1_lang_reverse", "all4"], ["c0_genre"]),
    "arxiv": (
        128,
        [2048],
        ["c3_nversions", "c0_maincat", "all4"],
        ["c0_maincat", "c3_nversions", "all4"],
    ),
    "pubmed": (768, BIG, ["c0_mesh", "c3_journal_reverse", "all5"], ["c0_mesh"]),
    "laion1m": (256, [512, 4096, 8192], ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion3m": (
        256,
        [2048, 8192, 16384],
        ["c0_domain", "tags4"],
        ["c0_domain", "tags4"],
    ),
    "laion10m": (256, BIG, ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
    "laion30m": (256, BIG, ["c0_domain", "tags4"], ["c0_domain", "tags4"]),
}
SYNTH = {  # synth parent: (dim, n_lists, clause sweeps, bloom sweeps); bloom on LAION synth at 10 / 30 M only (C5)
    "goodreads-synth": (128, [4096], RATES4, RATES4),
    "arxiv-synth": (128, [2048], RATES4, RATES4),
    "arxiv-corr-synth": (128, [2048], ["c001", "c003", "c01"], []),
    "pubmed-synth": (768, BIG, RATES4, RATES4),
    "laion1m-synth": (256, [512], RATES4, []),
    "laion3m-synth": (256, [2048], RATES4, []),
    "laion10m-synth": (256, BIG, RATES4, RATES4),
    "laion30m-synth": (256, BIG, RATES4, RATES4),
}
REAL = ["goodreads", "arxiv", "pubmed"]
LAION = ["laion1m", "laion3m", "laion10m", "laion30m"]
V1 = {"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["clause"]}
V2 = {"algo": "linr_v2", "backends": ["triton"], "filter_kinds": ["clause"]}
BY_ALGO = [{"by": "algo", "values": ["linr_v1_filter_mask", "linr_v2"]}]
STUDY = {
    d: v[2] for d, v in BENCH.items()
}  # the study's clause sweeps per real dataset


def draw(ds, d):
    return ds if d == 0 else f"{ds}-s{d}"


def suite(
    datasets,
    dims,
    arms,
    ks,
    bss,
    seeds,
    sweeps,
    kinds=("clause",),
    perf=True,
    interleave=None,
):
    s = {
        "datasets": datasets,
        "dims": dims,
        "filter_kinds": list(kinds),
        "ks": ks,
        "batch_sizes": bss,
        "seeds": seeds,
        **({"sweeps": sweeps} if sweeps else {}),
        "arms": arms,
    }
    if not perf:
        s = {"perf": False, **s}
    if interleave:
        s["interleave"] = interleave
    return s


def curve_arms(spec, names):
    """Per (kind, n_lists): SilverTorch on the adapted backend, n_probe doubling to n_lists / 4, plus the papers' 24 at a
    dataset's smallest n_lists; bloom with the recheck off (user, 2026-10-12)."""
    arms = []
    for kind, col in (("clause", 2), ("bloom", 3)):
        for n in sorted({n for d in spec for n in spec[d][1]}):
            ds = {}
            for d in spec:
                if n not in spec[d][1] or not spec[d][col]:
                    continue
                probes = [p for p in P if p <= n // 4] + (
                    [24] if n == min(spec[d][1]) else []
                )
                ds[names[d]] = {
                    "sweeps": spec[d][col],
                    "query": {"n_probe": sorted(probes)},
                }
            if ds:
                build = {
                    "n_lists": [n],
                    **({"recheck": [False]} if kind == "bloom" else {}),
                }
                arms.append(
                    {
                        "algo": "silvertorch",
                        "backends": ["official-fork"],
                        "filter_kinds": [kind],
                        "build": build,
                        "datasets": ds,
                    }
                )
    return arms


def curve_suites(spec, name, seeds_of):
    """One quality-only suite per (dim, seed group): a dataset config may list several dims, and a synth draw runs at its
    own seed."""
    out = {}
    for dim in sorted({v[0] for v in spec.values()}):
        for tag, seeds, names in seeds_of(sorted(d for d in spec if spec[d][0] == dim)):
            sub = {d: spec[d] for d in names}
            sw = {names[d]: sorted(set(spec[d][2]) | set(spec[d][3])) for d in sub}
            kinds = ("clause", "bloom") if any(spec[d][3] for d in sub) else ("clause",)
            out[f"{name}{tag}-d{dim}"] = suite(
                list(sw),
                [dim],
                curve_arms(sub, names),
                [100, 1000],
                [16],
                seeds,
                sw,
                kinds,
                perf=False,
            )
    return out


blocks = {
    "bloom": {"m_bits": 1024, "k_hash": 5},
    # stage 1: the curves every timed point is picked from (seed 0's) and C6 / F2 reads (all seeds)
    **curve_suites(BENCH, "fr-curve", lambda ds: [("", CURVE_SEEDS, {d: d for d in ds})]),
    **curve_suites(SYNTH, "fr-curve-synth", lambda ds: [(f"-s{s}", [s], {d: draw(d, s) for d in ds}) for s in CURVE_SEEDS]),
    # LiNR exact: one seed (C3)
    "fr-linr-bs64": suite(REAL, [128, 768], [V1, V2], [100, 1000], [64], [0], STUDY, interleave=BY_ALGO),
    # bs 256: V2 on PubMed is out (C2: 11 h for c3_journal_reverse, a TIMEOUT on all5); V1 alone there
    "fr-linr-bs256": suite(
        REAL, [128, 768], [V1, {**V2, "datasets": {"goodreads": {}, "arxiv": {}}}], [100, 1000], [256], [0], STUDY,
        interleave=BY_ALGO,
    ),
    "fr-linr-bs64-laion": suite(LAION, [256], [V1, V2], [100], [64], [0], STUDY, interleave=BY_ALGO),
    "fr-linr-corr": suite(["arxiv-corr-synth"], [128], [V1, V2], [100, 1000], [1, 16], [0], {}, interleave=BY_ALGO),
    "fr-linr-k2048": suite(
        REAL, [128, 768], [V1, V2], [2048], [1, 16], [0],
        {"goodreads": ["c0_genre"], "arxiv": ["c0_maincat"], "pubmed": ["c0_mesh"]},
    ),
    # C1 crossover on PubMed's synthetic filters (the final pass's `synth` / `nsweep-synth` cover the rest)
    "fr-linr-synth-pubmed": suite(["pubmed-synth"], [768], [V1, V2], [100, 1000], [1, 16], [0], {}, interleave=BY_ALGO),
    # LiNR V3 pools: OPORP is random, so 3 seeds; seed 0's pools feed the matched-recall picks
    **{
        name: suite(
            ds, dims,
            [{"algo": "linr_v3", "backends": ["triton"], "query": {"candidate_pool_frac": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]}}],
            ks, [1, 16], CURVE_SEEDS, STUDY,
        )
        for name, ds, dims, ks in (("fr-v3pool", REAL, [128, 768], [100, 1000]), ("fr-v3pool-laion", LAION, [256], [100]))
    },
    # C4 / F4a on the study's arm: the width grid, adapted backend, recheck off, quality only
    "fr-bloomwidth": suite(
        REAL, [128, 768],
        [{"algo": "silvertorch", "backends": ["official-fork"],
          "build": {"m_bits": [64, 128, 256, 512, 1024, 2048], "k_hash": [3, 5], "recheck": [False]}}],
        [100, 1000], [16], [0],
        {"goodreads": ["c0_genre"], "arxiv": ["all4", "c0_maincat", "c3_nversions"], "pubmed": ["c0_mesh"]},
        kinds=("bloom",), perf=False,
    ),
    # Yambda (user, 2026-10-13): adapted SilverTorch only, unfiltered, the exact scan as the reference
    "fr-yambda": suite(
        ["yambda-500m"], [64, 128, 256],
        [{"algo": "linr_v1_filter_mask", "backends": ["triton"], "filter_kinds": ["none"]}]
        + [{"algo": "silvertorch", "backends": ["official-fork"], "filter_kinds": ["none"], "build": {"n_lists": [n]},
            "query": {"n_probe": [p for p in P if p <= n // 4]}} for n in (1024, 4096, 16384)],
        [100, 1000], [16], SEEDS, {}, kinds=("none",), perf=False,
    ),
}  # fmt: skip
yaml.dump(blocks, sys.stdout, sort_keys=False, width=140, Dumper=NoAlias)
