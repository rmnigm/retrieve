"""The cell loop of harness v2 (H §2, §3.1 ``run.py``, §8.2 A/B/D/K).

``run(jobs, out_dir=...)`` executes ``Job``s in this process, in order, and appends one JSONL
record per cell to ``<out_dir>/<suite>/<dataset>-d<dim>.jsonl`` the moment the cell finishes
(``append_record``: one ``write`` + ``fsync`` per line, so a crash loses at most the cell in
flight). Per ``(dataset, dim)``: inputs once. Per ``(filter_kind, sweep, filter backend,
k_max, bloom)``: the sweep attrs, skip mask, standalone filter, exact oracle (blob v4) and,
on bloom cells, the bloom pass counts. Per job: one build (timed, ``index_bytes``). Per cell
(query combo, applied with ``set_query_params``): quality at ``k_max`` → parity spill →
perf per ``(bs, k, mode)`` → record.

Resume (§8.2 B): a cell is skipped when its ``oracle.resume_key`` — the key block plus the
library tree hash — is already in the file with ``status: ok``; failed and partial records
are re-run, and ``report.py`` reads the last record per key. A record is ``partial`` when
it does not carry everything the suite asked for: ``--skip-quality`` / ``--skip-perf``, a
``--mode`` subset, or ``--k`` / ``--bs`` replacing the suite's lists (``Job.narrowed``);
``partial_reasons`` names which. Failures (§8.2 D, §7): any
exception inside a cell is written as ``status: failed`` with the traceback and the loop
continues — an OOM on the torch path is a finding, not noise. Two things stop the process:
``KeyboardInterrupt`` and the exact-algo recall gate of §2.4 (``QualityGateError``, recorded
first).

Cross-backend parity (§2.4, amended by §8.2 K): the first backend to run a cell writes its
top-``k_max`` ids and scores to ``<out_dir>/_parity/<hash>.npz`` (hash over the key minus
``backend``); later backends record ``jaccard_vs_first@k`` and ``score_max_abs_diff``
against it. ``bench campaign`` deletes the directory when the ``(dataset, dim, algo)`` group
closes; a missing reference records ``null`` with ``parity: "no_reference"``.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import os
import time
import traceback
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

from retrieval import algos, bench, data, oracle
from retrieval.algos import FILTER_BACKEND
from retrieval.config import QUERY_PARAMS, Job
from retrieval.metrics import accumulate, accumulator, finalize, jaccard_at_k

SCHEMA_VERSION = 1
MODES = ("eager", "graph")
QUALITY_CHUNK = 16  # the OOM bound of the old passes.py (H §2.4): [B, P, D] on loose filters
EXACT_ALGOS = ("linr_v1_filter_mask", "linr_v2")  # §2.4: recall_oracle@k_max >= 0.99 or die
EXACT_MIN_RECALL = 0.99
CLOCK_DRIFT = 0.05  # §2.1: warn when clocks.sm drifts > 5 % from the first sample
MiB = bench.MiB
# The stat keys of a perf entry; a variant that cannot run records them as null + ``reason``.
PERF_STAT_KEYS = (
    "n", "median_ms", "mean_ms", "p95_ms", "p99_ms", "min_ms", "iqr_ms", "qps", "host_gap_ms",
    "outliers_std", "outliers_tukey", "spread", "unstable", "peak_fwd_mib", "window_medians_ms",
)  # fmt: skip


class QualityGateError(RuntimeError):
    """An exact algo scored below ``EXACT_MIN_RECALL`` against the oracle (H §2.4)."""


# ----- JSONL ----------------------------------------------------------------------------


def record_path(out_dir: Path, job: Job) -> Path:
    return Path(out_dir) / job.suite / f"{job.dataset}-d{job.dim}.jsonl"


def _clean(o: Any) -> Any:
    """JSON-safe: tensors → lists, paths → str, non-finite floats → null (valid JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, torch.Tensor):
        return _clean(o.tolist())
    if isinstance(o, np.floating | np.integer):
        return _clean(o.item())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, Path):
        return str(o)
    return o


def append_record(path: Path, rec: dict[str, Any]) -> None:
    """Append one record atomically enough: one ``write`` of one line, then ``fsync``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_clean(rec), separators=(",", ":"), allow_nan=False) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_keys(path: Path) -> dict[str, str]:
    """``{resume_key: status}`` of the records in ``path`` (last record per key wins)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = {k: rec[k] for k in oracle.KEY_FIELDS}
            out[oracle.resume_key(key, rec["env"]["code_version"])] = rec.get("status", "ok")
    return out


# ----- per (filter_kind, sweep) assets -----------------------------------------------------


def sweep_assets(job: Job, inputs: dict[str, Any], k_max: int, device: torch.device) -> dict:
    """Sweep attrs + skip mask, the standalone filter for this cell's filter backend, the
    exact oracle blob (filter cells), bloom pass counts (bloom cells) and the row masks the
    quality pass scores: ``keep`` (not skip-masked), ``oracle_rows`` (kept and ≥ 1
    survivor — the old harness's zero-target skip), ``heldout_rows`` (kept, ≥ 1 target and,
    on filter cells, ``target_in_filter``), and ``n_targets_in_filter`` — the held-out
    targets those rows score (on filter cells only the ones the exact mask admits)."""
    fk, n = job.filter_kind, inputs["n_queries"]
    qa_s, skip = data.sweep_qa(inputs["qa"], job.clauses)
    filters = data.build_filters(fk, inputs, [job.backend], bloom=job.bloom)
    filter_mod = filters.get(FILTER_BACKEND[job.backend])
    keep = ~skip if skip is not None else torch.ones(n, dtype=torch.bool)
    heldout = keep & (inputs["n_targets"] > 0)
    blob = None
    bloom_fp = None
    oracle_rows = None
    if fk != "none":
        exact = data.exact_filter(fk, filters, inputs, job.backend)
        blob = oracle.load_or_build(
            job.data.gt_dir,
            job.sweep,
            k_max,
            item_embs=inputs["item_embs"],
            queries=inputs["queries"],
            targets=inputs["targets"],
            qa_sweep=qa_s,
            skip_mask=skip,
            clauses=job.clauses,
            filter_mod=exact,
            attrs_digest=inputs["attrs_digest"],
            device=device,
        )
        heldout &= blob["target_in_filter"]
        oracle_rows = keep & (blob["topk"][:, 0] != -1)
        if fk == "bloom":
            assert filter_mod is not None
            counts = oracle.pass_counts(filter_mod, qa_s, skip, device=device)
            bloom_fp = oracle.bloom_fp_rate(counts, blob["pass_counts"], inputs["n_items"])
    n_tif = (inputs["n_targets"] if blob is None else blob["targets_in_filter"])[heldout].sum()
    return {
        "qa_s": qa_s,
        "skip": skip,
        "filter_mod": filter_mod,
        "blob": blob,
        "keep": keep,
        "oracle_rows": oracle_rows,
        "heldout_rows": heldout,
        "n_targets_in_filter": int(n_tif),
        "pass_rate": blob["pass_rate"] if blob is not None else 1.0,
        "bloom_fp_rate": bloom_fp,
    }


def build_module(job: Job, inputs: dict, assets: dict, k_max: int, params: dict) -> nn.Module:
    kw = dict(params)
    if job.algo == "silvertorch" and job.filter_kind == "bloom":
        kw = {**job.bloom, **kw}
    return algos.build(
        job.algo,
        inputs["item_embs"],
        k=k_max,
        backend=job.backend,
        filter_kind=job.filter_kind,
        filter_mod=assets["filter_mod"],
        item_attrs=inputs["item_attrs"],
        clause_is_reverse=inputs["clause_is_reverse"],
        params=kw,
        seed=job.seed,
    )


# ----- quality + parity ---------------------------------------------------------------------


@torch.inference_mode()
def quality(
    module: nn.Module, inputs: dict, assets: dict, ks: Sequence[int], device: torch.device
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """§2.4: stream the kept rows in chunks of 16 through ``module`` at ``k_max``, accumulate
    every ``k`` from the one top-``k_max`` list (oracle: ranked prefix targets; held-out:
    fixed targets) as device running sums, one sync at the end. On filter cells a held-out
    target the exact mask excludes can never be retrieved, so it is masked to ``-1`` and
    ``nt`` counts only the reachable ones (``blob["targets_in_filter"]``) — Goodreads
    targets are lists, and scoring the unreachable ones biases recall down. Row selection
    uses CPU masks and ``index_select``, so no chunk syncs. Returns the metrics and the
    ``[n_kept, k_max]`` ids / scores (for the parity spill)."""
    rows = assets["keep"].nonzero().reshape(-1)
    blob = assets["blob"]
    acc_o = accumulator(list(ks), device) if blob is not None else None
    acc_h = accumulator(list(ks), device)
    ids_all: list[torch.Tensor] = []
    sc_all: list[torch.Tensor] = []
    for s in range(0, rows.numel(), QUALITY_CHUNK):
        sel = rows[s : s + QUALITY_CHUNK]
        q = inputs["queries"][sel].to(device, non_blocking=True)
        qa = (
            assets["qa_s"][sel].to(device, non_blocking=True)
            if assets["qa_s"] is not None
            else None
        )
        ids, scores = module(q, qa)
        ids_all.append(ids)
        sc_all.append(scores.float())
        if acc_o is not None:
            m = assets["oracle_rows"][sel]
            if bool(m.any()):
                idx = m.nonzero().reshape(-1).to(device, non_blocking=True)
                accumulate(
                    acc_o, ids.index_select(0, idx), blob["topk"][sel][m].to(device), ranked=True
                )
        m = assets["heldout_rows"][sel]
        if bool(m.any()):
            idx = m.nonzero().reshape(-1).to(device, non_blocking=True)
            t = inputs["targets"][sel][m].to(device, non_blocking=True)
            if blob is not None:  # reachable targets only
                t = t.masked_fill(~blob["targets_in_filter"][sel][m].to(device), -1)
            accumulate(acc_h, ids.index_select(0, idx), t, (t != -1).sum(dim=1))
    out: dict[str, Any] = {"heldout": finalize(acc_h)}
    if acc_o is not None:
        out["oracle"] = finalize(acc_o)
    ids_t = torch.cat(ids_all) if ids_all else torch.empty(0, 0, dtype=torch.long)
    sc_t = torch.cat(sc_all) if sc_all else torch.empty(0, 0)
    return out, ids_t, sc_t


def parity(
    out_dir: Path,
    job: Job,
    params: dict,
    ids: torch.Tensor,
    scores: torch.Tensor,
    ks: Sequence[int],
) -> dict[str, Any]:
    """The spill-file wiring check (§2.4 / §8.2 K): write the reference when absent, else
    compare against it."""
    key = {k: v for k, v in job.key(params).items() if k != "backend"}
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
    ref = Path(out_dir) / "_parity" / f"{h}.npz"
    out: dict[str, Any] = {f"jaccard_vs_first@{k}": None for k in ks}
    out["score_max_abs_diff"] = None
    if ref.exists():
        z = np.load(ref)
        r_ids, r_sc = torch.from_numpy(z["ids"]).long(), torch.from_numpy(z["scores"])
        ids, scores = ids.cpu(), scores.cpu()
        if r_ids.shape == ids.shape:
            for k in ks:
                out[f"jaccard_vs_first@{k}"] = jaccard_at_k(ids, r_ids, int(k))
            both = torch.isfinite(scores) & torch.isfinite(r_sc)
            diff = (scores - r_sc).abs()[both]
            out["score_max_abs_diff"] = diff.max().item() if diff.numel() else 0.0
            out["parity"] = f"vs_{z['backend'].item()}"
        else:
            out["parity"] = f"shape_mismatch:{tuple(r_ids.shape)}!={tuple(ids.shape)}"
        return out
    ref.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        ref,
        ids=ids.cpu().numpy().astype(np.int32),
        scores=scores.cpu().numpy().astype(np.float32),
        backend=np.array(job.backend),
    )
    out["parity"] = "reference"
    return out


# ----- perf -----------------------------------------------------------------------------------


def _rotate(callee: Any, pool: torch.Tensor, qa_pool: torch.Tensor | None) -> Any:
    """The zero-arg call ``bench.latency`` times: the pool rotated round-robin (§2.5)."""
    counter = itertools.count()
    n_pool = pool.shape[0]
    if qa_pool is None:
        return lambda: callee(pool[next(counter) % n_pool])

    def fn():
        i = next(counter) % n_pool
        return callee(pool[i], qa_pool[i])

    return fn


def perf(
    module: nn.Module,
    inputs: dict,
    assets: dict,
    job: Job,
    device: torch.device,
    *,
    modes: Sequence[str],
    profile: bool,
    latency_kw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§2.5 per ``(bs, k, mode)``: the fixed-seed pool rotated round-robin, ``module.k = k``
    before each variant, ``graph`` via ``bench.graph_callable`` (one capture per shape) or a
    null entry with the ``reason`` (``official`` → ``not_capturable``, O D7)."""
    entries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for bs in job.batch_sizes:
        pool, qa_pool = data.query_pool(
            inputs, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=device
        )
        for k in job.ks:
            module.k = int(k)
            for mode in modes:
                entry: dict[str, Any] = {"k": int(k), "bs": int(bs), "mode": mode}
                if mode == "graph":
                    example = (pool[0],) if qa_pool is None else (pool[0], qa_pool[0])
                    try:
                        callee = bench.graph_callable(module, *example)
                    except bench.NotCapturable as exc:
                        logger.info("  graph bs={} k={}: {}", bs, k, exc)
                        entries.append(
                            {**entry, **dict.fromkeys(PERF_STAT_KEYS), "reason": str(exc)}
                        )
                        continue
                else:
                    callee = module
                fn = _rotate(callee, pool, qa_pool)
                with torch.inference_mode():
                    d, ms = bench.latency(fn, bs=int(bs), mode=mode, **latency_kw)
                    if profile and mode == "eager":
                        d["kernels"] = bench.profile_once(fn)
                entries.append({**entry, **d})
                samples.append({"k": int(k), "bs": int(bs), "mode": mode, "ms": ms})
                callee = fn = None
            torch._dynamo.reset()
    return entries, samples


# ----- the loop -----------------------------------------------------------------------------


def _release() -> None:
    gc.collect()
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _failed(job: Job, params: dict, env: dict, stage: str, exc: BaseException, t0: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "failed",
        **job.key(params),
        "path": job.path,
        "stage": stage,
        "error": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "elapsed_s": time.perf_counter() - t0,
        "env": env,
    }


def run(
    jobs: Sequence[Job],
    *,
    out_dir: Path,
    out_path: Path | None = None,
    resume: bool = True,
    modes: Sequence[str] = MODES,
    skip_quality: bool = False,
    skip_perf: bool = False,
    profile: bool = False,
    expected_sm_mhz: int | None = 1410,
    env_extra: dict[str, Any] | None = None,
    latency_kw: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> Counter:
    """Run every cell of ``jobs`` in this process (module docstring). ``out_path`` overrides
    the per-``(suite, dataset, dim)`` file; ``latency_kw`` is forwarded to ``bench.latency``
    (tests shrink the windows). Returns ``Counter(ok / partial / failed / skipped)``."""
    counts: Counter = Counter()
    if not jobs:
        return counts
    out_dir = Path(out_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bench.setup(jobs[0].seed)
    bench.warm_gpu_once()
    env0 = {**bench.provenance(), **(env_extra or {})}
    code_version = env0["code_version"]
    clk0 = bench.clocks(expected_sm_mhz)
    latency_kw = dict(latency_kw or {})
    reasons0 = [r for r, on in (("skip_quality", skip_quality), ("skip_perf", skip_perf)) if on]
    if set(modes) != set(MODES):
        reasons0.append("modes")
    with_filters = {(j.dataset, j.dim): False for j in jobs}
    for j in jobs:
        with_filters[j.dataset, j.dim] |= j.filter_kind != "none"

    existing: dict[Path, dict[str, str]] = {}
    inputs: dict[str, Any] | None = None
    inputs_key: tuple | None = None
    assets: dict | None = None
    assets_key: tuple | None = None
    for job in jobs:
        path = out_path or record_path(out_dir, job)
        if path not in existing:
            existing[path] = read_keys(path)
        todo = [
            p
            for p in job.cells()
            if not (
                resume and existing[path].get(oracle.resume_key(job.key(p), code_version)) == "ok"
            )
        ]
        counts["skipped"] += len(job.cells()) - len(todo)
        if not todo:
            logger.info("resume: {} all {} cells done", job.group, len(job.cells()))
            continue
        if (job.dataset, job.dim) != inputs_key:
            inputs = assets = None
            assets_key = None
            _release()
            inputs = data.load_inputs(
                job.data, device, with_filters=with_filters[job.dataset, job.dim]
            )
            inputs_key = (job.dataset, job.dim)
        assert inputs is not None
        k_max = max(job.ks)
        akey = (
            job.filter_kind,
            job.sweep,
            FILTER_BACKEND[job.backend],
            k_max,
            tuple(sorted(job.bloom.items())),
        )
        if akey != assets_key:
            assets = None
            _release()
            assets = sweep_assets(job, inputs, k_max, device)
            assets_key = akey
        assert assets is not None

        bench.setup(job.seed)
        t0 = time.perf_counter()
        logger.info(
            "build {} {}/{} {} build={}",
            job.algo,
            job.filter_kind,
            job.sweep,
            job.backend,
            job.build,
        )
        try:
            module, build_s = bench.timed_build(
                lambda: build_module(job, inputs, assets, k_max, todo[0])
            )
        except Exception as exc:  # noqa: BLE001 — recorded, the loop continues (H §7)
            logger.exception("build failed: {}", job.key(todo[0]))
            for p in todo:
                append_record(path, _failed(job, p, env0, "build", exc, t0))
                counts["failed"] += 1
            _release()
            continue
        index_mib = bench.index_bytes(module) / MiB
        filter_mib = bench.index_bytes(getattr(module, "filter", None)) / MiB

        for params in todo:
            t0 = time.perf_counter()
            stage = "query_params"
            try:
                q = {k: v for k, v in params.items() if k in QUERY_PARAMS}
                if q:
                    module.set_query_params(**q)
                clk = bench.clocks(expected_sm_mhz)
                drift = (
                    clk["sm_mhz"] is not None
                    and clk0["sm_mhz"]
                    and abs(clk["sm_mhz"] - clk0["sm_mhz"]) > CLOCK_DRIFT * clk0["sm_mhz"]
                )
                if drift:
                    logger.warning(
                        "clocks.sm {} MHz drifted from {} MHz", clk["sm_mhz"], clk0["sm_mhz"]
                    )
                env = {**env0, **clk, "clocks_drift": bool(drift)}
                reasons = reasons0 + (["ks_bs"] if job.narrowed else [])
                rec: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "partial" if reasons else "ok",
                    "partial_reasons": reasons or None,
                    **job.key(params),
                    "path": job.path,
                    "n_items": inputs["n_items"],
                    "n_queries": inputs["n_queries"],
                    "n_kept": int(assets["keep"].sum()),
                    "n_queries_heldout": int(assets["heldout_rows"].sum()),
                    "n_targets_in_filter": assets["n_targets_in_filter"],
                    "n_queries_oracle": int(assets["oracle_rows"].sum())
                    if assets["oracle_rows"] is not None
                    else None,
                    "pass_rate": assets["pass_rate"],
                    "bloom_fp_rate": assets["bloom_fp_rate"],
                    "bloom": dict(job.bloom) if job.filter_kind == "bloom" else None,
                    "k_max": k_max,
                    "ks": list(job.ks),
                    "batch_sizes": list(job.batch_sizes),
                    "build_s": build_s,
                    "index_mib": index_mib,
                    "filter_mib": filter_mib,
                    "quality": None,
                    "perf": None,
                }
                samples: list[dict[str, Any]] = []
                if not skip_quality:
                    stage = "quality"
                    module.k = k_max
                    qual, ids, scores = quality(module, inputs, assets, job.ks, device)
                    qual.update(parity(out_dir, job, params, ids, scores, job.ks))
                    del ids, scores
                    rec["quality"] = qual
                    if job.algo in EXACT_ALGOS and "oracle" in qual:
                        r = qual["oracle"][f"recall@{k_max}"]
                        if r < EXACT_MIN_RECALL:
                            raise QualityGateError(
                                f"{job.algo}/{job.backend} recall_oracle@{k_max} = {r:.4f} "
                                f"< {EXACT_MIN_RECALL}"
                            )
                if not skip_perf:
                    stage = "perf"
                    rec["perf"], samples = perf(
                        module,
                        inputs,
                        assets,
                        job,
                        device,
                        modes=modes,
                        profile=profile,
                        latency_kw=latency_kw,
                    )
                rec["unstable"] = bool(drift) or any(e.get("unstable") for e in rec["perf"] or [])
                rec["memory_reserved_mib"] = (
                    torch.cuda.memory_reserved() / MiB if torch.cuda.is_available() else None
                )
                rec["elapsed_s"] = time.perf_counter() - t0
                rec["env"] = env
            except QualityGateError as exc:
                append_record(path, _failed(job, params, env0, stage, exc, t0))
                counts["failed"] += 1
                raise
            except Exception as exc:  # noqa: BLE001 — recorded, the loop continues (H §7)
                logger.exception("cell failed at {}: {}", stage, job.key(params))
                append_record(path, _failed(job, params, env0, stage, exc, t0))
                counts["failed"] += 1
                _release()
                continue
            append_record(path, rec)
            key = job.key(params)
            for s in samples:
                append_record(path.with_suffix(".samples.jsonl"), {**key, **s})
            existing[path][oracle.resume_key(key, code_version)] = rec["status"]
            counts[rec["status"]] += 1
            logger.info(
                "  {} {}/{} {} params={} -> {} ({:.0f}s)",
                job.algo, job.filter_kind, job.sweep, job.backend, params,
                rec["status"], rec["elapsed_s"],
            )  # fmt: skip
        del module
        _release()
    counts = +counts  # drop zero entries
    logger.info("done: {}", dict(counts))
    return counts


__all__ = [
    "EXACT_ALGOS",
    "MODES",
    "PERF_STAT_KEYS",
    "QUALITY_CHUNK",
    "SCHEMA_VERSION",
    "QualityGateError",
    "append_record",
    "read_keys",
    "record_path",
    "run",
]
