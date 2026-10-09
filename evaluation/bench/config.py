"""Config matrix for harness v2 (H §3.3, amended by §8.2 A and J).

One YAML per dataset (``config/<dataset>.yaml``) plus one ``config/suites.yaml``;
``load_matrix`` expands ``(dataset, suite)`` into ``Job``s. A ``Job`` is one index *build*:
``(dataset, dim, filter_kind, sweep, algo, backend, build params, seed)`` carrying the query-param
combos measured against that build (§8.2 A: the ``QUERY_PARAMS`` mutate via ``set_query_params``,
they never rebuild), its ``ks`` / ``batch_sizes``, its bloom (the defaults or the arm's gridded
widths) and the resolved ``Dataset`` (paths with ``{dim}`` substituted). A suite is a list of
*arms*: one algo on its backends with optional filter kinds, sweeps, grids and per-dataset
overrides (docs/system/evaluation.md § Config). One *cell* is ``(job, params)``
with ``params = build | query combo``; ``Job.key(params)`` is the record's key block and the input
to ``records.resume_key``.

Backends that run the same code collapse to one job (``PATHS``: ``linr_v1`` on ``none``
is cuBLAS whatever the flag says) and ``(algo, filter_kind, backend)`` triples
``PATHS`` marks ``None`` are skipped; both are logged once. Sweep entries accept the long
form ``{clauses: [...], disabled: true}`` (§8.2 J). No anchors, no env interpolation, no
schema library: unknown keys and malformed values raise ``ConfigError`` naming the file.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from bench.algos import ALGOS, BACKENDS, FILTER_KINDS, PATHS, is_valid_combo, official_config

# set_query_params, no rebuild; candidate_pool_frac is resolved per sweep to candidate_pool (run.py)
QUERY_PARAMS = frozenset({"n_probe", "candidate_pool", "candidate_pool_frac", "alpha"})
BLOOM_PARAMS = frozenset({"m_bits", "k_hash"})  # gridded on silvertorch/bloom: the job's bloom
COMPILE_MODES = ("max-autotune",)
NONE_SWEEP = "full_scan"  # the one sweep of filter_kind ``none`` (the old harness's name)
_DATASET_KEYS = {"data_dir", "checkpoint", "content_dir", "dims", "encode", "users_limit"}
_DATASET_KEYS |= {"filters"}
_FILTER_KEYS = {"attrs", "reverse", "query_attrs", "clause", "bloom"}
_SUITE_KEYS = {"datasets", "dims", "filter_kinds", "ks", "batch_sizes", "arms", "seeds", "bloom"}
_SUITE_KEYS |= {"sweeps", "ks_by_sweep", "perf", "interleave"}
_ARM_KEYS = {"algo", "backends", "filter_kinds", "sweeps", "build", "query", "datasets"}
_ENCODE = {"batch_size": 512, "num_workers": 8, "max_seq_length": 200}
_BLOOM = {"m_bits": 1024, "k_hash": 5}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Dataset:
    """One dataset at one dim, every path resolved. ``checkpoint`` set = SASRec-encoded
    queries; ``content_dir`` set = pre-encoded text embeddings (arxiv). ``attrs`` /
    ``reverse`` are ``None`` without a ``filters:`` block; ``query_attrs`` is the query side,
    ``eval_split.parquet`` unless ``filters.query_attrs`` names a file. ``gt_dir`` is derived."""

    name: str
    dim: int
    data_dir: Path
    checkpoint: Path | None
    content_dir: Path | None
    users_limit: int | None
    encode: dict[str, int]
    attrs: Path | None
    reverse: Path | None
    query_attrs: Path
    clauses: dict[str, dict[str, tuple[int, ...]]]  # filter_kind -> sweep -> active clauses

    @property
    def gt_dir(self) -> Path:
        return self.data_dir / f"gt_d{self.dim}"

    @property
    def inputs(self) -> str:
        """The record key's input identity: the checkpoint's directory name, or the
        ``content_dir`` name (docs/system/evaluation.md, "The key block")."""
        return self.checkpoint.parent.name if self.checkpoint else self.content_dir.name


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
    narrowed: bool = False  # --k / --bs replaced the suite's lists: records are ``partial``
    timed: bool = True  # False on a quality-only suite (``perf: false``): no perf, still ``ok``
    interleave: tuple[tuple[tuple[str, ...], tuple | None], ...] = ()  # the suite's (by, values)

    def cells(self) -> list[dict[str, Any]]:
        """``params`` of every cell of this job, in query order."""
        return [{**self.build, **q} for q in self.query]

    def key(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """The record's key block (H §3.2); ``params`` defaults to the build params."""
        return {
            "dataset": self.dataset,
            "dim": self.dim,
            "inputs": self.data.inputs,
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


def load_dataset(path: Path, dim: int, checkpoint: str | None = None) -> Dataset:
    """Resolve ``config/<dataset>.yaml`` at one dim; ``checkpoint`` (``{dim}`` templated)
    replaces the YAML's, for a run pinned to another encoder (the golden cells)."""
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
    query_attrs = data_dir / "eval_split.parquet"
    clauses: dict[str, dict[str, tuple[int, ...]]] = {}
    if filters is not None:
        _check_keys(f"{where}: filters", filters, _FILTER_KEYS, ["attrs"])
        attrs = data_dir / _template(where, filters["attrs"], dim)
        if filters.get("reverse"):
            reverse = data_dir / _template(where, filters["reverse"], dim)
        if filters.get("query_attrs"):
            query_attrs = data_dir / _template(where, filters["query_attrs"], dim)
        clauses = {k: _sweeps(where, k, filters.get(k)) for k in ("clause", "bloom")}
    users_limit = raw.get("users_limit")
    if users_limit is not None and (not isinstance(users_limit, int) or users_limit <= 0):
        raise ConfigError(f"{where}: users_limit must be a positive int or null")
    ckpt, content = raw.get("checkpoint"), raw.get("content_dir")
    if checkpoint is not None:
        if content is not None:
            raise ConfigError(f"{where}: a checkpoint override on a content_dir dataset")
        ckpt = checkpoint
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
        query_attrs=query_attrs,
        clauses=clauses,
    )


def _combos(where: str, spec: Any) -> list[dict[str, Any]]:
    """dict-of-lists = grid; list-of-dicts = explicit combos; missing = one empty combo."""
    if spec is None:
        return [{}]
    if isinstance(spec, dict):
        if not all(isinstance(v, list) and v for v in spec.values()):
            raise ConfigError(f"{where}: a grid maps every key to a non-empty list, got {spec!r}")
        return [dict(zip(spec, vals, strict=True)) for vals in itertools.product(*spec.values())]
    if isinstance(spec, list) and all(isinstance(c, dict) for c in spec):
        return [dict(c) for c in spec] or [{}]
    raise ConfigError(f"{where}: expected a grid or a list of combos, got {spec!r}")


def _params(where: str, build_spec: Any, query_spec: Any) -> tuple[list[dict], list[dict]]:
    """An arm's ``build:`` / ``query:`` grids → (build combos, query combos) (§8.2 A)."""
    build = _combos(f"{where}.build", build_spec)
    query = _combos(f"{where}.query", query_spec)
    bad_b = {k for c in build for k in c if k in QUERY_PARAMS}
    bad_q = {k for c in query for k in c if k not in QUERY_PARAMS}
    if bad_b or bad_q:
        raise ConfigError(
            f"{where}: {sorted(QUERY_PARAMS)} are query params, everything else is a build "
            f"param; misplaced: build={sorted(bad_b)} query={sorted(bad_q)}"
        )
    return build, query


def _merge(base: Any, override: Any) -> Any:
    """A per-dataset override of one grid: key by key when both are dict grids, else whole."""
    if override is None:
        return base
    if isinstance(base, dict) and isinstance(override, dict):
        return {**base, **override}
    return override


def _check_params(where: str, algo: str, fk: str, backend: str, build: dict, query) -> None:
    """The params that mean something on one arm only (raising, never ignored)."""
    try:
        official_config(algo, fk, backend, build)
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e
    if BLOOM_PARAMS & set(build) and (algo, fk) != ("silvertorch", "bloom"):
        raise ConfigError(f"{where}: m_bits / k_hash are build params of silvertorch/bloom only")
    if "compile" in build and (backend != "torch" or build["compile"] not in COMPILE_MODES):
        raise ConfigError(f"{where}: compile is one of {COMPILE_MODES} on torch arms only")
    if "k_bits" in build and algo != "linr_v3":
        raise ConfigError(f"{where}: k_bits is a linr_v3 build param (its OPORP stage 1)")
    if any("candidate_pool_frac" in q for q in query) and algo != "linr_v3":
        raise ConfigError(f"{where}: candidate_pool_frac is a linr_v3 query param")


def _narrow(values: Sequence, chosen: Sequence | None, what: str) -> list:
    if chosen is None:
        return list(values)
    out = [v for v in values if v in chosen]
    if not out:
        logger.warning("--{} {} selects nothing from {}", what, list(chosen), list(values))
    return out


def _known_sweeps(where: str, names: Any, ds: Dataset) -> list[str]:
    if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
        raise ConfigError(f"{where}: expected a list of sweep names, got {names!r}")
    unknown = set(names) - {sw for sweeps in ds.clauses.values() for sw in sweeps}
    if unknown:
        raise ConfigError(f"{where}: sweeps {sorted(unknown)} not in {ds.name}")
    return names


def _check_suite(where: str, s: dict) -> None:
    _check_keys(where, s, _SUITE_KEYS, ["datasets", "filter_kinds", "ks", "batch_sizes", "arms"])
    unknown_fk = set(s["filter_kinds"]) - set(FILTER_KINDS)
    if unknown_fk:
        raise ConfigError(f"{where}: unknown filter_kinds {sorted(unknown_fk)}")
    if not isinstance(s["arms"], list) or not s["arms"]:
        raise ConfigError(f"{where}: arms must be a non-empty list")
    for i, arm in enumerate(s["arms"]):
        aw = f"{where}: arms[{i}]"
        _check_keys(aw, arm, _ARM_KEYS, ["algo", "backends"])
        if arm["algo"] not in ALGOS:
            raise ConfigError(f"{aw}: unknown algo {arm['algo']!r}")
        bad_be = set(arm["backends"]) - set(BACKENDS)
        bad_fk = set(arm.get("filter_kinds") or []) - set(s["filter_kinds"])
        bad_ds = set(arm.get("datasets") or {}) - set(s["datasets"])
        if bad_be or bad_fk or bad_ds:
            raise ConfigError(
                f"{aw}: unknown backends {sorted(bad_be)}, filter_kinds {sorted(bad_fk)} or "
                f"datasets {sorted(bad_ds)} (filter kinds and datasets must be the suite's)"
            )


def _interleave(where: str, raw: Any) -> tuple[tuple[tuple[str, ...], tuple | None], ...]:
    """A suite's ``interleave:`` list → ``((by fields, values or None), ...)``. ``by`` is
    ``algo``, ``backend`` or build params (one name or a list); ``values`` (one ``by`` field
    only) limits which arms join."""
    out = []
    for i, g in enumerate(raw or []):
        gw = f"{where}: interleave[{i}]"
        _check_keys(gw, g, {"by", "values"}, ["by"])
        by = tuple([g["by"]] if isinstance(g["by"], str) else g["by"])
        if not by or not all(isinstance(f, str) for f in by) or QUERY_PARAMS & set(by):
            raise ConfigError(f"{gw}: by is algo, backend or build params, got {g['by']!r}")
        values = g.get("values")
        if values is not None and (len(by) != 1 or not isinstance(values, list) or not values):
            raise ConfigError(f"{gw}: values is a non-empty list, with one by field only")
        out.append((by, None if values is None else tuple(values)))
    return tuple(out)


def _dim_value(job: Job, field: str) -> Any:
    return getattr(job, field) if field in ("algo", "backend") else job.build.get(field)


def shared_key(job: Job, params: dict[str, Any], by: tuple[str, ...]) -> dict[str, Any]:
    """The key block of one cell minus the group's ``by`` fields: what the arms of one
    interleave group have in common (the seed included, so a group is per seed)."""
    key = job.key(params)
    key["params"] = {k: v for k, v in key["params"].items() if k not in by}
    return {k: v for k, v in key.items() if k not in by}


def interleave_units(jobs: Sequence[Job]) -> list[tuple[tuple[str, ...] | None, list[Job]]]:
    """Partition ``jobs`` into ``(by, members)`` units in first-member order: the jobs of one
    suite comparison group (same key but for the ``by`` fields, query params aside) form one
    unit, every other job is a unit of its own (``by`` None). A job in two groups of two or
    more is a ``ConfigError``."""
    clusters: dict[str, list[Job]] = {}
    for j in jobs:
        for by, values in j.interleave:
            if values is not None and _dim_value(j, by[0]) not in values:
                continue
            c = json.dumps([by, shared_key(j, j.build, by)], sort_keys=True, default=str)
            clusters.setdefault(c, []).append(j)
    unit_of: dict[int, tuple[tuple[str, ...], list[Job]]] = {}
    for c, members in clusters.items():
        if len(members) < 2:
            continue
        for j in members:
            if id(j) in unit_of:
                raise ConfigError(f"{j.suite}: {j.key()} is in two interleave groups")
            unit_of[id(j)] = (tuple(json.loads(c)[0]), members)
    units: list[tuple[tuple[str, ...] | None, list[Job]]] = []
    seen: set[int] = set()
    for j in jobs:
        if id(j) not in seen:
            by, members = unit_of.get(id(j), (None, [j]))
            seen.update(map(id, members))
            units.append((by, members))
    return units


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
    checkpoint: str | None = None,
) -> list[Job]:
    """Expand one ``(dataset, suite)`` into jobs, grouped by ``Job.group`` in the order
    ``dim → algo → backend → arm → filter_kind → sweep → build → seed``. Each of a suite's
    ``arms`` names an algo and its backends, optionally the filter kinds, sweeps, ``build`` /
    ``query`` grids and the datasets it runs on, each with its own overrides
    (docs/system/evaluation.md § Config). Keyword narrows are the CLI overrides (H §3.4): they
    filter the lists *before* the PATHS collapse, so ``backends=["torch"]`` on a ``none`` cell
    yields the cuBLAS job labelled ``torch``. ``ks`` / ``batch_sizes`` *replace* the suite's
    lists rather than select cells, so when they differ the jobs are ``narrowed`` and ``run``
    records them as ``partial``."""
    suites = _read(suites_yaml)
    if suite not in suites:
        raise ConfigError(f"{suites_yaml}: no suite {suite!r}; have {sorted(suites)}")
    where = f"{suites_yaml}: {suite}"
    s = suites[suite]
    _check_suite(where, s)
    name = Path(dataset_yaml).stem
    if name not in s["datasets"]:
        raise ConfigError(f"{where}: dataset {name!r} not in {s['datasets']}")
    suite_ks = _ints(f"{where}: ks", s["ks"])
    suite_bs = _ints(f"{where}: batch_sizes", s["batch_sizes"])
    bs_ = suite_bs if batch_sizes is None else _ints("--bs", list(batch_sizes))
    seed_list = s.get("seeds", [0])
    if not isinstance(seed_list, list) or not all(isinstance(x, int) and x >= 0 for x in seed_list):
        raise ConfigError(f"{where}: seeds must be a list of ints >= 0, got {seed_list!r}")
    bloom = {**_BLOOM, **(suites.get("bloom") or {}), **(s.get("bloom") or {})}
    groups = _interleave(where, s.get("interleave"))
    timed = s.get("perf", True)
    if not isinstance(timed, bool):
        raise ConfigError(f"{where}: perf must be true or false, got {timed!r}")
    raw_dims = _ints(f"{dataset_yaml}: dims", _read(dataset_yaml).get("dims"))
    dims_ = _narrow([d for d in raw_dims if d in s.get("dims", raw_dims)], dims, "dim")
    fks = _narrow(s["filter_kinds"], filter_kinds, "filter-kind")
    sweep_ks = (s.get("ks_by_sweep") or {}).get(name) or {}
    arm_algos = list(dict.fromkeys(a["algo"] for a in s["arms"]))

    jobs: list[Job] = []
    owner: dict[tuple, str] = {}  # (dim, algo, fk, sweep, path, params, seed) -> backend
    for dim in dims_:
        ds = load_dataset(Path(dataset_yaml), dim, checkpoint)
        suite_sweeps = (s.get("sweeps") or {}).get(name)
        if suite_sweeps is not None:
            _known_sweeps(f"{where}: sweeps.{name}", suite_sweeps, ds)
        _known_sweeps(f"{where}: ks_by_sweep.{name}", list(sweep_ks), ds)
        for algo in _narrow(arm_algos, algos, "algo"):
            arms = [(i, a) for i, a in enumerate(s["arms"]) if a["algo"] == algo]
            arm_backends = list(dict.fromkeys(b for _, a in arms for b in a["backends"]))
            for backend in _narrow(arm_backends, backends, "backend"):
                for i, arm in arms:
                    aw = f"{where}: arms[{i}]"
                    if backend not in arm["backends"]:
                        continue
                    if "datasets" in arm and name not in (arm["datasets"] or {}):
                        continue
                    over = (arm.get("datasets") or {}).get(name) or {}
                    _check_keys(f"{aw}.datasets.{name}", over, {"build", "query", "sweeps"})
                    builds, queries = _params(
                        aw,
                        _merge(arm.get("build"), over.get("build")),
                        _merge(arm.get("query"), over.get("query")),
                    )
                    arm_sweeps = over.get("sweeps", arm.get("sweeps"))
                    if arm_sweeps is not None:
                        _known_sweeps(f"{aw}: sweeps", arm_sweeps, ds)
                    for fk in fks:
                        if fk not in (arm.get("filter_kinds") or s["filter_kinds"]):
                            continue
                        path = PATHS[(algo, fk, backend)]
                        if path is None:
                            logger.info("skip {}/{}/{}: no code path", algo, fk, backend)
                            continue
                        sweep_defs = ds.clauses.get(fk, {}) if fk != "none" else {NONE_SWEEP: None}
                        kept = [
                            sw
                            for sw in sweep_defs
                            if fk == "none"
                            or (suite_sweeps is None or sw in suite_sweeps)
                            and (arm_sweeps is None or sw in arm_sweeps)
                        ]
                        for sweep in _narrow(kept, sweeps, "sweep"):
                            job_ks = (
                                _ints(f"{where}: ks_by_sweep", sweep_ks[sweep])
                                if sweep in sweep_ks
                                else suite_ks
                            )
                            ks_ = job_ks if ks is None else _ints("--k", list(ks))
                            narrowed = set(ks_) != set(job_ks) or set(bs_) != set(suite_bs)
                            for build in builds:
                                _check_params(aw, algo, fk, backend, build, queries)
                                qs = tuple(
                                    q for q in queries if is_valid_combo(algo, {**build, **q})
                                )
                                if len(qs) < len(queries):
                                    logger.info("{} build={}: invalid combos dropped", algo, build)
                                if not qs:
                                    continue
                                widths = {k: build[k] for k in BLOOM_PARAMS & set(build)}
                                for seed in _narrow(list(seed_list), seeds, "seed"):
                                    cells = [
                                        (dim, algo, fk, sweep, path, _canon({**build, **q}), seed)
                                        for q in qs
                                    ]
                                    taken = [owner[c] for c in cells if c in owner]
                                    if backend in taken:
                                        raise ConfigError(f"{aw}: a cell twice: {cells[0]}")
                                    if taken:
                                        if len(taken) < len(cells):
                                            raise ConfigError(
                                                f"{aw}: {algo}/{fk}/{backend} shares part of "
                                                f"its cells with {taken[0]} on path {path}"
                                            )
                                        logger.info(
                                            "collapse {}/{}/{} -> {} ({})",
                                            algo, fk, backend, taken[0], path,
                                        )  # fmt: skip
                                        continue
                                    owner.update(dict.fromkeys(cells, backend))
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
                                            bloom={**bloom, **widths},
                                            data=ds,
                                            narrowed=narrowed,
                                            timed=timed,
                                            interleave=groups,
                                        )
                                    )
    logger.info("{}/{}: {} jobs, {} cells", name, suite, len(jobs), sum(len(j.query) for j in jobs))
    interleave_units(jobs)  # a job in two groups fails here, not mid-run
    return jobs


def _canon(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


__all__ = [
    "BLOOM_PARAMS",
    "COMPILE_MODES",
    "NONE_SWEEP",
    "QUERY_PARAMS",
    "ConfigError",
    "Dataset",
    "Job",
    "interleave_units",
    "load_dataset",
    "load_matrix",
    "shared_key",
]
