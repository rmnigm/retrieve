"""Offline tuner for the Triton kernels in ``retrieve.ops.triton``: run once on the target arch and
paste the printed ``DEFAULT_CONFIG = ...`` line into the kernel file (or pass it via the
wrapper's ``config=``). Each kernel is one ``KernelTuneSpec`` in ``KERNELS``, from which its click
subcommand is generated (see ``--help``); ``--regime`` overrides the built-in eval shapes for the
regime-swept kernels."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any

import click
import torch
import triton
import triton.testing as ttesting

from retrieve.ops.triton.bloom_compact import (
    DEFAULT_CONFIG as BLOOM_COMPACT_DEFAULT,
    BloomCompactConfig,
    _bloom_compact_impl,
)
from retrieve.ops.triton.clause_compact import (
    DEFAULT_CONFIG as CLAUSE_COMPACT_DEFAULT,
    ClauseCompactConfig,
    _clause_compact_impl,
)
from retrieve.ops.triton.clause_mask import (
    DEFAULT_CONFIG as CLAUSE_MASK_DEFAULT,
    ClauseMaskConfig,
    _clause_mask_impl,
)
from retrieve.ops.triton.codesigned_probe_score import (
    DEFAULT_CONFIG as CODESIGNED_PROBE_SCORE_DEFAULT,
    CodesignedProbeScoreConfig,
    _codesigned_probe_score_impl,
)
from retrieve.ops.triton.codesigned_probe_score_exact import (
    DEFAULT_CONFIG as CODESIGNED_PROBE_SCORE_EXACT_DEFAULT,
    CodesignedProbeScoreExactConfig,
    _codesigned_probe_score_exact_impl,
)
from retrieve.ops.triton.fused_masked_knn_topk import (
    _P_BUCKETS,
    DEFAULT_CONFIG as FUSED_MASKED_KNN_TOPK_DEFAULT,
    FusedMaskedKnnTopkConfig,
    _fused_masked_knn_topk_impl,
)
from retrieve.ops.triton.oporp_1bit_match_topk import (
    _N_BUCKETS,
    DEFAULT_CONFIG as OPORP_1BIT_MATCH_TOPK_DEFAULT,
    Oporp1BitMatchTopkConfig,
    _oporp_1bit_match_topk_impl,
)

# The selection rule (kernels.md § Autotune separation): the shipped default stays unless a
# candidate is faster by more than NOISE_BAND (geometric mean over regimes, the A/A spread of an
# interleaved rerun) and slower by at most WORST_CAP in every single regime.
NOISE_BAND = 0.03
WORST_CAP = 0.05
# A B = 1 regime whose own winner beats the chosen config by this much is flagged: the case for
# a separate batch-1 default.
B1_GAP = 0.10

_FMKT_GRID = tuple((bn, nw) for bn in (32, 64, 128, 256) for nw in (4, 8))
_OPORP_GRID = tuple((bn, nw) for bn in (64, 128, 256, 512) for nw in (4, 8))
_CPS_GRID = tuple((bp, nw) for bp in (32, 64, 128, 256) for nw in (4, 8))
# Filter-index kernels sweep a wider BLOCK_N range since the inner body varies and the optimum can
# land far from 256.
_FILTER_GRID = tuple((bn, nw) for bn in (128, 256, 512, 1024) for nw in (2, 4, 8))

# Default filter regimes mirror real-eval shapes (goodreads N≈800K C=4, arxiv N≈3M C=5, A_MAX=4;
# bloom W=16 from m_bits=1024). End-users override via --regime.
_DEFAULT_CLAUSE_REGIMES = (
    # (N, B, C, A_MAX)
    (797_085, 1, 4, 4),  # goodreads, single-query
    (797_085, 16, 4, 4),  # goodreads, batched
    (2_988_997, 1, 5, 4),  # arxiv-retrieval, single-query
    (2_988_997, 16, 5, 4),  # arxiv-retrieval, batched
    (15_000_001, 16, 5, 4),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)
_DEFAULT_BLOOM_REGIMES = (
    # (N, B, W) — W=16 from m_bits=1024 default
    (797_085, 1, 16),
    (797_085, 16, 16),
    (2_988_997, 1, 16),
    (2_988_997, 16, 16),
    (15_000_001, 16, 16),  # arxiv-synth-15m, batched (DRAM-streaming extreme)
)

# Default CPS P-grid spanning typical IVF sizings; hard-coded since the kernel isn't bucketed on P.
_DEFAULT_CPS_P_GRID = (1024, 8192, 65536)


def _arch_str(device: torch.device) -> str:
    cap = torch.cuda.get_device_capability(device)
    return f"sm_{cap[0]}{cap[1]}"


def _bench(fn) -> float:
    """Median ms for ``fn`` under do_bench — long enough to settle the JIT cache, short enough to
    keep the sweep quick."""
    median, _, _ = ttesting.do_bench(fn, quantiles=[0.5, 0.2, 0.8], rep=500, warmup=100)
    return float(median)


def _parse_regime(spec: str, arity: int) -> tuple[int, ...]:
    """Parse a ``--regime N,B,C,A_MAX`` (or ``N,B,W``) flag value."""
    parts = spec.split(",")
    if len(parts) != arity:
        raise click.BadParameter(
            f"expected {arity} comma-separated ints, got {len(parts)!r}: {spec!r}"
        )
    try:
        return tuple(int(p) for p in parts)
    except ValueError as e:
        raise click.BadParameter(f"non-integer field in {spec!r}: {e}") from e


def _rand_bits(shape: tuple[int, ...], dev: torch.device) -> torch.Tensor:
    ii = torch.iinfo(torch.int64)
    return torch.randint(ii.min, ii.max, shape, dtype=torch.int64, device=dev)


def _fmkt_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    p, d, b = regime
    n = max(p * 2, 1 << 16)  # need N >= P; generous N keeps the gather realistic
    torch.manual_seed(0)
    return {
        "query": torch.randn(b, d, device=dev),
        "item_embs": torch.randn(n, d, device=dev),
        "positive_indices": torch.randint(0, n, (b, p), dtype=torch.long, device=dev),
        "counts": torch.full((b,), p, dtype=torch.long, device=dev),
        "k": min(64, p),
    }


def _oporp_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n_bucket, has_indices, w, b = regime
    torch.manual_seed(0)
    corpus_n = max(n_bucket, 1 << 16) if has_indices else n_bucket
    if has_indices:
        p = n_bucket // 4 if n_bucket >= 4 else 1
        pos = torch.randint(0, corpus_n, (b, p), dtype=torch.long, device=dev)
        counts = torch.full((b,), p, dtype=torch.long, device=dev)
        k = min(64, p)
    else:
        pos = counts = None
        k = 64
    return {
        "query_bits": _rand_bits((b, w), dev),
        "item_bits": _rand_bits((corpus_n, w), dev),
        "k": k,
        "positive_indices": pos,
        "counts": counts,
    }


def _probe_layout(dev: torch.device, n: int, p: int, b: int) -> dict[str, Any]:
    """A synthetic CSR probe: equal 256-item clusters, ``p // 256`` distinct probes per row, so
    every row fills the compact width ``p`` exactly."""
    n_lists, n_probe = n // 256, max(p // 256, 1)
    return {
        "probe_ids": torch.stack([torch.randperm(n_lists, device=dev)[:n_probe] for _ in range(b)]),
        "cluster_offsets": torch.arange(0, n_lists * 256 + 1, 256, device=dev),
        "sort_perm": torch.randperm(n, device=dev),
        "width": n_probe * 256,
    }


def _cps_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    p, has_qb, d, b, w = regime
    n = max(p * 4, 1 << 16)
    torch.manual_seed(0)
    return dict(
        query=torch.randn(b, d, device=dev),
        item_codes=torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=min(64, p),
        # ~C·k_hash = 10 set query bits over a 64·W-bit signature, the shipped bloom shape.
        query_bit_positions=torch.randint(0, 64 * w, (b, 10), device=dev) if has_qb else None,
        bloom_transposed=_rand_bits((64 * w, n // 64), dev) if has_qb else None,
        **_probe_layout(dev, n, p, b),
    )


def _clause_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    """Build realistic (item_attrs, is_reverse, query_attrs) on ``dev`` — small vocab so random
    queries get a non-trivial pass/fail mix."""
    n, b, c, a_max = regime
    torch.manual_seed(0)
    n_vocab = 40
    return {
        "item_clause_attrs": torch.randint(
            0, n_vocab, (n, c, a_max), dtype=torch.int64, device=dev
        ),
        "clause_is_reverse": torch.zeros(c, dtype=torch.bool, device=dev),
        "query_clause_attrs": torch.randint(0, n_vocab, (b, c), dtype=torch.int64, device=dev),
    }


def _cpse_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n, b, c, a_max = regime
    inputs = _clause_inputs(dev, regime)
    # D mirrors the codesigned-probe-score default (--d 128). The width is deployment-specific;
    # the middle of the CPS P-grid (capped by N) stands in for it here.
    d = 128
    inputs.update(
        query=torch.randn(b, d, device=dev),
        item_codes=torch.randint(-128, 128, (n, d), dtype=torch.int8, device=dev),
        global_scale=0.01,
        k=64,
        **_probe_layout(dev, n, min(n, 8192), b),
    )
    return inputs


def _bloom_inputs(dev: torch.device, regime: tuple[int, ...]) -> dict[str, Any]:
    n, b, w = regime
    torch.manual_seed(0)
    return {"qb": _rand_bits((b, w), dev), "sigs": _rand_bits((n, w), dev)}


@dataclass(frozen=True)
class KernelTuneSpec:
    """Everything the generic ``_sweep``/``_print``/CLI machinery needs to tune one kernel.

    A regime is one benchmark shape (a tuple of ints named by ``regime_labels``). Regime-swept
    kernels (``regime_arity`` set) expose a repeatable ``--regime`` flag over ``default_regimes``;
    dimension-swept kernels expose the ``dims`` flags instead and derive their regimes via
    ``expand_regimes`` (crossing the flags with the module's bucket/P axes)."""

    name: str  # click subcommand, e.g. "clause-mask"
    config_cls: type  # ClauseMaskConfig, ... — first dataclass field is the block size
    # Candidate configs as positional (block, num_warps) field values, splatted into config_cls.
    grid: tuple[tuple[int, ...], ...]
    regime_labels: tuple[str, ...]  # names for the regime fields, e.g. ("N", "B", "C", "A_MAX")
    make_inputs: Callable[[torch.device, tuple[int, ...]], dict[str, Any]]  # regime → kwargs
    run: Callable[[dict[str, Any], Any], Any]  # (inputs, config) → one _impl call
    paste_path: str  # file the DEFAULT_CONFIG line goes into
    current: Any  # the kernel module's shipped DEFAULT_CONFIG, the incumbent of the rule
    smoke_regime: tuple[int, ...]  # one tiny regime for the CI smoke test
    # Regime-swept kernels (--regime):
    default_regimes: tuple[tuple[int, ...], ...] = ()
    regime_arity: int | None = None  # 4 = N,B,C,A_MAX; 3 = N,B,W; None = dimension flags instead
    regime_fmt: str = ""  # --regime metavar for --help
    # Dimension-swept kernels (--d/--b/--w):
    dims: tuple[tuple[str, int, str], ...] = ()  # (flag, default, help)
    expand_regimes: Callable[..., tuple[tuple[int, ...], ...]] | None = None


def _config_fields(spec: KernelTuneSpec, entry: tuple[int, ...]) -> tuple[str, ...]:
    """Dataclass field names for a grid entry, positionally — the grid may be shorter than
    the config (the Triton kernels never sweep ``num_stages``)."""
    return tuple(f.name for f in fields(spec.config_cls))[: len(entry)]


def _fmt_entry(spec: KernelTuneSpec, entry: tuple[int, ...]) -> str:
    """One grid entry as ``block_p=256  num_warps=4``, padded so the sweep log
    stays column-aligned across block sizes."""
    return " ".join(
        f"{name}={v:<4}" for name, v in zip(_config_fields(spec, entry), entry, strict=True)
    ).rstrip()


def _entry(spec: KernelTuneSpec, config: Any) -> tuple[int, ...]:
    """A config as a grid entry: its first ``len(grid entry)`` fields."""
    return tuple(getattr(config, n) for n in _config_fields(spec, spec.grid[0]))


def _sm_mhz() -> int | None:
    q = ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", "0"]
    out = subprocess.run(q, capture_output=True, text=True, check=False)
    return int(out.stdout.split()[0]) if out.returncode == 0 and out.stdout.strip() else None


def _commit() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parent,
    )
    return out.stdout.strip() if out.returncode == 0 else None


def _choose(
    current: tuple[int, ...], per_regime: dict[str, dict[tuple[int, ...], float]]
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """The selection rule over ``{regime: {entry: ms}}``: the candidate with the lowest geometric
    mean of ``ms / current_ms`` wins if that mean is below ``1 - NOISE_BAND`` and its worst
    regime is at most ``1 + WORST_CAP``; otherwise ``current`` stays. Returns the pick and the
    per-candidate ``(geomean, worst)`` ratios behind it."""
    ratios: dict[tuple[int, ...], list[float]] = {}
    for timings in per_regime.values():
        for entry, ms in timings.items():
            ratios.setdefault(entry, []).append(ms / timings[current])
    stats = {
        e: (math.exp(sum(map(math.log, r)) / len(r)), max(r))
        for e, r in ratios.items()
        if len(r) == len(per_regime)
    }
    eligible = [e for e, (g, w) in stats.items() if g < 1 - NOISE_BAND and w <= 1 + WORST_CAP]
    pick = min(eligible, key=lambda e: stats[e][0]) if eligible else current
    return pick, {
        "geomean_vs_current": {",".join(map(str, e)): g for e, (g, _) in stats.items()},
        "worst_vs_current": {",".join(map(str, e)): w for e, (_, w) in stats.items()},
    }


def _sweep(spec: KernelTuneSpec, dev: torch.device, regimes: tuple[tuple[int, ...], ...]) -> dict:
    """Time every grid entry (plus the shipped default, if the grid lacks it) on every regime,
    then pick with ``_choose``. Tuning is offline and out-of-kernel on purpose; see
    docs/system/kernels.md § Autotune separation."""
    current = _entry(spec, spec.current)
    grid = spec.grid if current in spec.grid else (*spec.grid, current)
    per_regime: dict[str, dict] = {}
    timings: dict[str, dict[tuple[int, ...], float]] = {}
    sm_mhz: list[int | None] = []
    for regime in regimes:
        key = ",".join(f"{label}={v}" for label, v in zip(spec.regime_labels, regime, strict=True))
        inputs = spec.make_inputs(dev, regime)
        timings[key] = {}
        results: list[dict] = []
        for entry in grid:
            cfg = spec.config_cls(*entry)
            # Warm the JIT cache before measuring; do_bench's warmup wouldn't cover a cold compile
            # of this config.
            for _ in range(3):
                spec.run(inputs, cfg)
            torch.cuda.synchronize()
            ms = _bench(partial(spec.run, inputs, cfg))
            timings[key][entry] = ms
            results.append({**dict(zip(_config_fields(spec, entry), entry, strict=True)), "ms": ms})
            click.echo(f"  [{key}] {_fmt_entry(spec, entry)}  -> {ms:.3f} ms", err=True)
        sm_mhz.append(_sm_mhz())
        best = min(timings[key], key=timings[key].get)
        per_regime[key] = {"winner": best, "winner_ms": timings[key][best], "all": results}
        click.echo(
            f"  [{key}] winner: {_fmt_entry(spec, best)} ({timings[key][best]:.3f} ms)", err=True
        )

    pick, rule = _choose(current, timings)
    b1_gaps = {
        k: timings[k][pick] / per_regime[k]["winner_ms"] - 1
        for k in timings
        if "B=1," in k + "," and timings[k][pick] / per_regime[k]["winner_ms"] - 1 >= B1_GAP
    }
    return {
        "per_regime": per_regime,
        "current": current,
        "default": pick,
        "rule": {"noise_band": NOISE_BAND, "worst_cap": WORST_CAP, **rule},
        "b1_gaps_over_threshold": b1_gaps,
        "env": {
            "sm_mhz_per_regime": sm_mhz,
            "commit": _commit(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "device": torch.cuda.get_device_name(dev),
        },
    }


def _print(spec: KernelTuneSpec, arch: str, result: dict) -> None:
    entry = result["default"]
    args = ", ".join(
        f"{name}={v}" for name, v in zip(_config_fields(spec, entry), entry, strict=True)
    )
    cls_name = spec.config_cls.__name__
    click.echo("")
    if entry == result["current"]:
        click.echo(f"# Keep the shipped default: no candidate beats it by > {NOISE_BAND:.0%}")
        click.echo(f"# within a {WORST_CAP:.0%} worst-regime cap.")
    click.echo(f"# Paste into {spec.paste_path}")
    click.echo(f"# Tuned on {arch}; per-regime details in JSON output if --json-out was used.")
    click.echo(f"DEFAULT_CONFIG = {cls_name}({args})")
    for key, gap in result["b1_gaps_over_threshold"].items():
        click.echo(f"# [{key}] its own winner is {gap:.0%} faster: a batch-1 config may pay.")


KERNELS: tuple[KernelTuneSpec, ...] = (
    KernelTuneSpec(
        name="fused-masked-knn-topk",
        config_cls=FusedMaskedKnnTopkConfig,
        grid=_FMKT_GRID,
        regime_labels=("P", "D", "B"),
        make_inputs=_fmkt_inputs,
        run=lambda inputs, config: _fused_masked_knn_topk_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/fused_masked_knn_topk.py",
        current=FUSED_MASKED_KNN_TOPK_DEFAULT,
        smoke_regime=(256, 64, 2),
        dims=(("d", 128, "Embedding dimension."), ("b", 16, "Batch size.")),
        expand_regimes=lambda d, b: tuple((p, d, b) for p in _P_BUCKETS),
    ),
    KernelTuneSpec(
        name="oporp-1bit-match-topk",
        config_cls=Oporp1BitMatchTopkConfig,
        grid=_OPORP_GRID,
        regime_labels=("N", "HAS_IDX", "W", "B"),
        make_inputs=_oporp_inputs,
        run=lambda inputs, config: _oporp_1bit_match_topk_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/oporp_1bit_match_topk.py",
        current=OPORP_1BIT_MATCH_TOPK_DEFAULT,
        smoke_regime=(4096, 1, 2, 2),
        dims=(("w", 2, "int64 words per bit-vector (D=64*W)."), ("b", 16, "Batch size.")),
        expand_regimes=lambda w, b: tuple((n, hi, w, b) for n in _N_BUCKETS for hi in (0, 1)),
    ),
    KernelTuneSpec(
        name="codesigned-probe-score",
        config_cls=CodesignedProbeScoreConfig,
        grid=_CPS_GRID,
        regime_labels=("P", "HAS_QB", "D", "B", "W"),
        make_inputs=_cps_inputs,
        run=lambda inputs, config: _codesigned_probe_score_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/codesigned_probe_score.py",
        current=CODESIGNED_PROBE_SCORE_DEFAULT,
        smoke_regime=(1024, 1, 64, 2, 4),
        dims=(
            ("d", 128, "Embedding dimension."),
            ("b", 16, "Batch size."),
            ("w", 4, "int64 words per bloom signature."),
        ),
        expand_regimes=lambda d, b, w: tuple(
            (p, hq, d, b, w) for p in _DEFAULT_CPS_P_GRID for hq in (0, 1)
        ),
    ),
    # (N, B, C, A_MAX) defaults mirror the shipped clause regimes; the exact kernel's optimum also
    # tracks P = n_probe × max_cluster_size (approximated inside _cpse_inputs), so re-tune with
    # --regime per deployment rather than trusting these defaults.
    KernelTuneSpec(
        name="codesigned-probe-score-exact",
        config_cls=CodesignedProbeScoreExactConfig,
        grid=_CPS_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_cpse_inputs,
        run=lambda inputs, config: _codesigned_probe_score_exact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/codesigned_probe_score_exact.py",
        current=CODESIGNED_PROBE_SCORE_EXACT_DEFAULT,
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="clause-mask",
        config_cls=ClauseMaskConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_clause_inputs,
        run=lambda inputs, config: _clause_mask_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/clause_mask.py",
        current=CLAUSE_MASK_DEFAULT,
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="clause-compact",
        config_cls=ClauseCompactConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "C", "A_MAX"),
        make_inputs=_clause_inputs,
        run=lambda inputs, config: _clause_compact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/clause_compact.py",
        current=CLAUSE_COMPACT_DEFAULT,
        smoke_regime=(4096, 2, 2, 2),
        default_regimes=_DEFAULT_CLAUSE_REGIMES,
        regime_arity=4,
        regime_fmt="N,B,C,A_MAX",
    ),
    KernelTuneSpec(
        name="bloom-compact",
        config_cls=BloomCompactConfig,
        grid=_FILTER_GRID,
        regime_labels=("N", "B", "W"),
        make_inputs=_bloom_inputs,
        run=lambda inputs, config: _bloom_compact_impl(**inputs, config=config),
        paste_path="retrieve/src/retrieve/ops/triton/bloom_compact.py",
        current=BLOOM_COMPACT_DEFAULT,
        smoke_regime=(4096, 2, 4),
        default_regimes=_DEFAULT_BLOOM_REGIMES,
        regime_arity=3,
        regime_fmt="N,B,W",
    ),
)


def _resolve_device(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise click.ClickException("CUDA is required to tune Triton kernels.")
    return torch.device(device)


def _dump_json(json_out: Path | None, arch: str, kernel: str, result: dict) -> None:
    if json_out is None:
        return
    json_out.write_text(json.dumps({"arch": arch, "kernel": kernel, **result}, indent=2))
    click.echo(f"# Full sweep written to {json_out}", err=True)


_device_opt = click.option("--device", default="cuda:0", show_default=True)
_json_out_opt = click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path for full per-shape sweep results.",
)


@click.group()
def main() -> None:
    """Offline tuner for retrieve's Triton kernels: pick a kernel subcommand (flags vary) and paste
    the printed ``DEFAULT_CONFIG = …`` line into the kernel source."""


def _register_subcommand(group: click.Group, spec: KernelTuneSpec) -> None:
    """Generate ``spec``'s click subcommand: a repeatable ``--regime`` when the spec sweeps
    declared regimes, the spec's dimension flags (``--d``/``--b``/``--w``) otherwise, plus the
    common ``--device``/``--json-out``."""
    kernel = spec.name.replace("-", "_")

    def callback(
        device: str, json_out: Path | None, regime_specs: tuple[str, ...] = (), **dims: int
    ) -> None:
        dev = _resolve_device(device)
        arch = _arch_str(dev)
        click.echo(f"# Tuning {kernel} on {arch} ({dev})", err=True)
        if spec.regime_arity is not None:
            parsed = tuple(_parse_regime(s, spec.regime_arity) for s in regime_specs)
            regimes = parsed or spec.default_regimes
        else:
            assert spec.expand_regimes is not None
            regimes = spec.expand_regimes(**dims)
        result = _sweep(spec, dev, regimes)
        _print(spec, arch, result)
        _dump_json(json_out, arch, kernel, result)

    cmd = _json_out_opt(callback)
    cmd = _device_opt(cmd)
    if spec.regime_arity is not None:
        cmd = click.option(
            "--regime",
            "regime_specs",
            multiple=True,
            metavar=spec.regime_fmt,
            help="Repeatable; tune for these shapes instead of the built-in eval regimes.",
        )(cmd)
    else:
        for flag, default, help_text in reversed(spec.dims):
            cmd = click.option(f"--{flag}", default=default, show_default=True, help=help_text)(cmd)
    group.command(spec.name)(cmd)


for _spec in KERNELS:
    _register_subcommand(main, _spec)


if __name__ == "__main__":
    main()
