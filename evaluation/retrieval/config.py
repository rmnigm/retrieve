"""Config matrix for harness v2 (H §3.3, amended by §8.2 A and J).

One YAML per dataset (``config/<dataset>.yaml``) plus one ``config/suites.yaml``;
``load_matrix`` expands ``(dataset, suite)`` into ``Job``s. A ``Job`` is one index *build*:
``(dataset, dim, filter_kind, sweep, algo, backend, build params, seed)`` carrying the
query-param combos measured against that build (§8.2 A: ``n_probe`` / ``candidate_pool``
mutate via ``set_query_params``, they never rebuild), the suite's ``ks`` / ``batch_sizes``,
the bloom defaults and the resolved ``Dataset`` (paths with ``{dim}`` substituted). One
*cell* is ``(job, params)`` with ``params = build | query combo``; ``Job.key(params)`` is
the record's key block and the input to ``oracle.resume_key``.

Backends that run the same code collapse to one job (``PATHS``: ``linr_v1``/``linr_v4`` on
``none`` are cuBLAS whatever the flag says) and ``(algo, filter_kind, backend)`` triples
``PATHS`` marks ``None`` are skipped; both are logged once. Sweep entries accept the long
form ``{clauses: [...], disabled: true}`` (§8.2 J). No anchors, no env interpolation, no
schema library: unknown keys and malformed values raise ``ConfigError`` naming the file.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from retrieval.algos import ALGOS, BACKENDS, FILTER_KINDS, PATHS, is_valid_combo

QUERY_PARAMS = frozenset({"n_probe", "candidate_pool"})  # set_query_params, never a rebuild
NONE_SWEEP = "full_scan"  # the one sweep of filter_kind ``none`` (the old harness's name)
_DATASET_KEYS = {"data_dir", "checkpoint", "content_dir", "dims", "encode", "users_limit"}
_DATASET_KEYS |= {"filters"}
_FILTER_KEYS = {"attrs", "reverse", "clause", "bloom"}
_SUITE_KEYS = {"datasets", "dims", "filter_kinds", "ks", "batch_sizes", "algos", "seeds", "params"}
_SUITE_KEYS |= {"bloom"}
_ENCODE = {"batch_size": 512, "num_workers": 8, "max_seq_length": 200}
_BLOOM = {"m_bits": 1024, "k_hash": 5}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    """One dataset at one dim, every path resolved. ``checkpoint`` set = SASRec-encoded
    queries; ``content_dir`` set = pre-encoded text embeddings (arxiv). ``attrs`` /
    ``reverse`` are ``None`` without a ``filters:`` block. ``gt_dir`` is derived."""

    name: str
    dim: int
    data_dir: Path
    checkpoint: Path | None
    content_dir: Path | None
    users_limit: int | None
    encode: dict[str, int]
    attrs: Path | None
    reverse: Path | None
    clauses: dict[str, dict[str, tuple[int, ...]]]  # filter_kind -> sweep -> active clauses

    @property
    def gt_dir(self) -> Path:
        return self.data_dir / f"gt_d{self.dim}"


@dataclass(frozen=True)
class Job:
    dataset: str
    dim: int
    suite: str
    filter_kind: str
    sweep: str
    clauses: tuple[int, ...] | None  # None on ``none`` cells
    algo: str
    backend: str
    path: str  # the code path that runs (algos.PATHS)
    build: dict[str, Any]  # build-time params
    query: tuple[dict[str, Any], ...]  # query-time combos, each measured against this build
    ks: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    seed: int
    bloom: dict[str, int]  # m_bits, k_hash (used on bloom cells)
    data: Dataset = field(compare=False, repr=False)

    def cells(self) -> list[dict[str, Any]]:
        """``params`` of every cell of this job, in query order."""
        return [{**self.build, **q} for q in self.query]

    def key(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """The record's key block (H §3.2); ``params`` defaults to the build params."""
        return {
            "dataset": self.dataset,
            "dim": self.dim,
            "suite": self.suite,
            "filter_kind": self.filter_kind,
            "sweep": self.sweep,
            "algo": self.algo,
            "backend": self.backend,
            "params": dict(self.build if params is None else params),
            "seed": self.seed,
        }

    @property
    def group(self) -> tuple[str, int, str, str]:
        """The campaign's process boundary (H §8.2 K)."""
        return (self.dataset, self.dim, self.algo, self.backend)


# ----- parsing --------------------------------------------------------------------------


def _read(path: Path) -> dict:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return raw


def _check_keys(where: str, d: Any, allowed: set[str], required: Iterable[str] = ()) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: expected a mapping, got {d!r}")
    unknown, missing = set(d) - allowed, set(required) - set(d)
    if unknown or missing:
        raise ConfigError(
            f"{where}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}; "
            f"allowed {sorted(allowed)}"
        )


def _ints(where: str, v: Any) -> tuple[int, ...]:
    if not isinstance(v, list) or not v or not all(isinstance(x, int) and x > 0 for x in v):
        raise ConfigError(f"{where}: expected a non-empty list of positive ints, got {v!r}")
    return tuple(v)


def _template(where: str, v: Any, dim: int) -> Any:
    """``{dim}`` in a string, or a per-dim mapping ``{64: a, 128: b}`` (arxiv ``content_dir``)."""
    if isinstance(v, str):
        return v.replace("{dim}", str(dim))
    if isinstance(v, dict):
        if dim not in v:
            raise ConfigError(f"{where}: no entry for dim {dim} in {v!r}")
        return v[dim]
    return v


def _sweeps(where: str, kind: str, raw: Any) -> dict[str, tuple[int, ...]]:
    """``name: [clauses]`` or ``name: {clauses: [...], disabled: true}``; disabled entries
    are dropped here, once, with a log line (§8.2 J)."""
    out: dict[str, tuple[int, ...]] = {}
    for name, spec in (raw or {}).items():
        disabled = False
        if isinstance(spec, dict):
            _check_keys(f"{where}: {kind}/{name}", spec, {"clauses", "disabled"}, ["clauses"])
            disabled, spec = bool(spec.get("disabled", False)), spec["clauses"]
        if not isinstance(spec, list) or not all(isinstance(c, int) and c >= 0 for c in spec):
            raise ConfigError(f"{where}: {kind}/{name}: clauses must be ints, got {spec!r}")
        if disabled:
            logger.info("sweep {}/{} disabled in {}", kind, name, where)
            continue
        out[str(name)] = tuple(spec)
    return out


def load_dataset(path: Path, dim: int) -> Dataset:
    """Resolve ``config/<dataset>.yaml`` at one dim."""
    raw, where = _read(path), str(path)
    _check_keys(where, raw, _DATASET_KEYS, ["data_dir", "dims"])
    if dim not in _ints(f"{where}: dims", raw["dims"]):
        raise ConfigError(f"{where}: dim {dim} not in dims {raw['dims']}")
    if (raw.get("checkpoint") is None) == (raw.get("content_dir") is None):
        raise ConfigError(f"{where}: set exactly one of checkpoint / content_dir")
    data_dir = Path(_template(where, raw["data_dir"], dim))
    encode = {**_ENCODE, **(raw.get("encode") or {})}
    _check_keys(f"{where}: encode", encode, set(_ENCODE))
    filters = raw.get("filters")
    attrs = reverse = None
    clauses: dict[str, dict[str, tuple[int, ...]]] = {}
    if filters is not None:
        _check_keys(f"{where}: filters", filters, _FILTER_KEYS, ["attrs"])
        attrs = data_dir / _template(where, filters["attrs"], dim)
        if filters.get("reverse"):
            reverse = data_dir / _template(where, filters["reverse"], dim)
        clauses = {k: _sweeps(where, k, filters.get(k)) for k in ("clause", "bloom")}
    users_limit = raw.get("users_limit")
    if users_limit is not None and (not isinstance(users_limit, int) or users_limit <= 0):
        raise ConfigError(f"{where}: users_limit must be a positive int or null")
    ckpt, content = raw.get("checkpoint"), raw.get("content_dir")
    return Dataset(
        name=path.stem,
        dim=dim,
        data_dir=data_dir,
        checkpoint=Path(_template(where, ckpt, dim)) if ckpt else None,
        content_dir=data_dir / _template(where, content, dim) if content else None,
        users_limit=users_limit,
        encode=encode,
        attrs=attrs,
        reverse=reverse,
        clauses=clauses,
    )


def _combos(where: str, spec: Any) -> list[dict[str, Any]]:
    """dict-of-lists = grid; list-of-dicts = explicit combos; missing = one empty combo."""
    if spec is None:
        return [{}]
    if isinstance(spec, dict):
        if not all(isinstance(v, list) and v for v in spec.values()):
            raise ConfigError(f"{where}: a grid maps every key to a non-empty list, got {spec!r}")
        return [dict(zip(spec, vals)) for vals in itertools.product(*spec.values())]
    if isinstance(spec, list) and all(isinstance(c, dict) for c in spec):
        return [dict(c) for c in spec] or [{}]
    raise ConfigError(f"{where}: expected a grid or a list of combos, got {spec!r}")


def _params(where: str, spec: Any) -> tuple[list[dict], list[dict]]:
    """``params.<algo>: {build: ..., query: ...}`` → (build combos, query combos) (§8.2 A)."""
    if spec is None:
        return [{}], [{}]
    _check_keys(where, spec, {"build", "query"})
    build = _combos(f"{where}.build", spec.get("build"))
    query = _combos(f"{where}.query", spec.get("query"))
    bad_b = {k for c in build for k in c if k in QUERY_PARAMS}
    bad_q = {k for c in query for k in c if k not in QUERY_PARAMS}
    if bad_b or bad_q:
        raise ConfigError(
            f"{where}: {sorted(QUERY_PARAMS)} are query params, everything else is a build "
            f"param; misplaced: build={sorted(bad_b)} query={sorted(bad_q)}"
        )
    return build, query


def _seeds(where: str, spec: Any, sweep: str, dim: int) -> tuple[int, ...]:
    """``[0, 1]`` | ``{default: [...], headline: {sweeps, dims, seeds}}``; missing → ``(0,)``."""
    if spec is None:
        return (0,)
    if isinstance(spec, list):
        return tuple(int(s) for s in spec)
    _check_keys(where, spec, {"default", "headline"}, ["default"])
    head = spec.get("headline")
    if head:
        _check_keys(f"{where}.headline", head, {"sweeps", "dims", "seeds"}, ["sweeps", "seeds"])
        if sweep in head["sweeps"] and dim in head.get("dims", [dim]):
            return tuple(int(s) for s in head["seeds"])
    return tuple(int(s) for s in spec["default"])


def _narrow(values: Sequence, chosen: Sequence | None, what: str) -> list:
    if chosen is None:
        return list(values)
    out = [v for v in values if v in chosen]
    if not out:
        logger.warning("--{} {} selects nothing from {}", what, list(chosen), list(values))
    return out


# ----- expansion ------------------------------------------------------------------------


def load_matrix(
    dataset_yaml: Path,
    suites_yaml: Path,
    suite: str,
    *,
    dims: Sequence[int] | None = None,
    algos: Sequence[str] | None = None,
    backends: Sequence[str] | None = None,
    filter_kinds: Sequence[str] | None = None,
    sweeps: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    ks: Sequence[int] | None = None,
    batch_sizes: Sequence[int] | None = None,
) -> list[Job]:
    """Expand one ``(dataset, suite)`` into jobs, grouped by ``Job.group`` in the order
    ``dim → algo → backend → filter_kind → sweep → build → seed``. Keyword narrows are the
    CLI overrides (H §3.4): they filter the suite's lists *before* the PATHS collapse, so
    ``backends=["torch"]`` on a ``none`` cell yields the cuBLAS job labelled ``torch``."""
    suites = _read(suites_yaml)
    if suite not in suites:
        raise ConfigError(f"{suites_yaml}: no suite {suite!r}; have {sorted(suites)}")
    where = f"{suites_yaml}: {suite}"
    s = suites[suite]
    _check_keys(where, s, _SUITE_KEYS, ["datasets", "filter_kinds", "ks", "batch_sizes", "algos"])
    name = Path(dataset_yaml).stem
    if name not in s["datasets"]:
        raise ConfigError(f"{where}: dataset {name!r} not in {s['datasets']}")
    _check_keys(f"{where}: algos", s["algos"], set(ALGOS))
    unknown_fk = set(s["filter_kinds"]) - set(FILTER_KINDS)
    unknown_be = {b for bs in s["algos"].values() for b in bs} - set(BACKENDS)
    if unknown_fk or unknown_be:
        raise ConfigError(
            f"{where}: unknown filter_kinds {sorted(unknown_fk)}, backends {sorted(unknown_be)}"
        )
    ks_ = _ints(f"{where}: ks", s["ks"]) if ks is None else _ints("--k", list(ks))
    bs_ = _ints(f"{where}: batch_sizes", s["batch_sizes"])
    bs_ = bs_ if batch_sizes is None else _ints("--bs", list(batch_sizes))
    bloom = {**_BLOOM, **(suites.get("bloom") or {}), **(s.get("bloom") or {})}
    raw_dims = _ints(f"{dataset_yaml}: dims", _read(dataset_yaml).get("dims"))
    dims_ = _narrow([d for d in raw_dims if d in s.get("dims", raw_dims)], dims, "dim")
    fks = _narrow(s["filter_kinds"], filter_kinds, "filter-kind")

    jobs: list[Job] = []
    for dim in dims_:
        ds = load_dataset(Path(dataset_yaml), dim)
        for algo in _narrow(list(s["algos"]), algos, "algo"):
            builds, queries = _params(f"{where}: params.{algo}", (s.get("params") or {}).get(algo))
            seen: dict[tuple[str, str], str] = {}  # (filter_kind, path) -> first backend
            for backend in _narrow(s["algos"][algo], backends, "backend"):
                for fk in fks:
                    path = PATHS[(algo, fk, backend)]
                    if path is None:
                        logger.info("skip {}/{}/{}: no code path", algo, fk, backend)
                        continue
                    if (fk, path) in seen:
                        first = seen[fk, path]
                        logger.info("collapse {}/{}/{} -> {} ({})", algo, fk, backend, first, path)
                        continue
                    seen[fk, path] = backend
                    sweep_defs = ds.clauses.get(fk, {}) if fk != "none" else {NONE_SWEEP: None}
                    for sweep in _narrow(list(sweep_defs), sweeps, "sweep"):
                        for build in builds:
                            qs = tuple(q for q in queries if is_valid_combo(algo, {**build, **q}))
                            if len(qs) < len(queries):
                                logger.info("{} build={}: invalid combos dropped", algo, build)
                            if not qs:
                                continue
                            seed_list = _seeds(f"{where}: seeds", s.get("seeds"), sweep, dim)
                            for seed in _narrow(seed_list, seeds, "seed"):
                                jobs.append(
                                    Job(
                                        dataset=name,
                                        dim=dim,
                                        suite=suite,
                                        filter_kind=fk,
                                        sweep=sweep,
                                        clauses=sweep_defs[sweep],
                                        algo=algo,
                                        backend=backend,
                                        path=path,
                                        build=dict(build),
                                        query=qs,
                                        ks=ks_,
                                        batch_sizes=bs_,
                                        seed=seed,
                                        bloom=dict(bloom),
                                        data=ds,
                                    )
                                )
    logger.info("{}/{}: {} jobs, {} cells", name, suite, len(jobs), sum(len(j.query) for j in jobs))
    return jobs


__all__ = [
    "NONE_SWEEP",
    "QUERY_PARAMS",
    "ConfigError",
    "Dataset",
    "Job",
    "load_dataset",
    "load_matrix",
]
