"""The cell loop (H §2, §3.1 ``run.py``, §8.2 A/B/D/K).

``run(jobs, out_dir=...)`` executes ``Job``s in this process, in order, and appends one JSONL
record per cell to ``<out_dir>/<suite>/<dataset>-d<dim>.jsonl`` the moment the cell finishes
(``records.append_record``: one ``write`` + ``fsync`` per line, so a crash loses at most the
cell in flight). Per ``(dataset, dim)``: inputs once. Per ``(filter_kind, sweep, filter backend,
k_max, bloom)``: the sweep attrs, skip mask, standalone filter, exact oracle (blob v4) and,
on bloom cells, the bloom pass counts. Per job: one build (timed, ``index_bytes``). Per cell
(query combo, applied with ``set_query_params``): quality at ``k_max`` → parity spill →
perf per ``(bs, k, mode)`` → record.

Resume (§8.2 B): a cell is skipped when its ``records.resume_key`` — the key block plus the
library tree hash — is already in the file with ``status: ok``; failed and partial records
are re-run, and ``report.py`` reads the last record per key. A record is ``partial`` when
it does not carry everything the suite asked for: ``--skip-quality`` / ``--skip-perf``, a
``--mode`` subset, or ``--k`` / ``--bs`` replacing the suite's lists (``Job.narrowed``);
``partial_reasons`` names which. Failures (§8.2 D, §7): any
exception inside a cell is written as ``status: failed`` with the traceback and the loop
continues — an OOM on the torch path is a finding, not noise. Three things stop the
process: ``KeyboardInterrupt``, the exact-algo recall gate of §2.4 (``QualityGateError``,
recorded first) and a *sticky* CUDA error (``STICKY_CUDA``: an illegal memory access or a
device-side assert kills the context, so every later cell would fail in seconds with the
same traceback and ``--resume`` would re-run them all) — recorded, then re-raised so the
campaign moves to the next group.

Crash safety: the oracle blob, encode cache and parity file are written through
``layout.atomic_write`` (tmp + ``os.replace``); the JSONL lines are one ``write`` + ``fsync``
each, samples *before* the record so a crash between the two cannot leave a resumable
record without its vector; ``records.read_keys`` tolerates (and logs) one torn trailing line.

Clocks (H §7's unlocked-clock fallback, C4 finding L4-c): ``env.sm_mhz_idle`` is the
process-start sample and is provenance only; every perf entry's ``sm_mhz`` is sampled under
load right after its last window, ``env.sm_mhz_load`` is the median of those, and
``clocks_drift`` fires when any of them differs by more than ``CLOCK_DRIFT`` from the first
under-load sample of the process — load against load, never against idle.

Cross-backend parity (§2.4, amended by §8.2 K): the first backend to run a cell writes its
top-``k_max`` ids and scores to ``<out_dir>/_parity/<hash>.npz`` (hash over the key minus
``backend``); later backends record ``jaccard_vs_first@k`` and ``score_max_abs_diff``
against it. ``bench campaign`` deletes the directory when the ``(dataset, dim, algo)`` group
closes. There is no "no reference" state: whichever backend runs a cell first *writes* the
reference (``parity: "reference"``), so after ``--resume`` skips the triton cells, ``torch``
becomes the reference and ``official`` is compared against torch (``parity: "vs_torch"``).
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import itertools
import json
import statistics
import time
import traceback
from collections import Counter
from collections.abc import Iterator, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from torch import nn

from bench import algos, inputs, measure, oracle, records
from bench.config import QUERY_PARAMS, Job
from bench.metrics import accumulate, accumulator, finalize, jaccard_at_k
from eval_datasets.layout import atomic_write

MODES = ("eager", "graph")
QUALITY_CHUNK = 16  # the OOM bound of the old passes.py (H §2.4): [B, P, D] on loose filters
EXACT_ALGOS = ("linr_v1_filter_mask", "linr_v2")  # §2.4: recall_oracle@k_max >= 0.99 or die
EXACT_MIN_RECALL = 0.99
CLOCK_DRIFT = 0.05  # §2.1: an under-load sample > 5 % off the process's first one
# An exception whose message carries one of these has killed the CUDA context: recorded, then
# re-raised so the child exits and ``bench campaign`` moves on (review §2.9).
STICKY_CUDA = ("CUDA error", "illegal memory access", "device-side assert")
MiB = measure.MiB
# The stat keys of a perf entry; a variant that cannot run records them as null + ``reason``.
PERF_STAT_KEYS = (
    "n", "median_ms", "mean_ms", "p95_ms", "p99_ms", "min_ms", "iqr_ms", "qps", "host_gap_ms",
    "outliers_std", "outliers_tukey", "spread", "unstable", "peak_fwd_mib", "window_medians_ms",
)  # fmt: skip


class QualityGateError(RuntimeError):
    """An exact algo scored below ``EXACT_MIN_RECALL`` against the oracle (H §2.4)."""


def is_sticky(exc: BaseException) -> bool:
    """A CUDA error after which the process cannot measure anything else."""
    msg = str(exc)
    return any(s in msg for s in STICKY_CUDA)


# ----- per (filter_kind, sweep) assets -----------------------------------------------------


def sweep_assets(job: Job, inp: dict[str, Any], k_max: int, device: torch.device) -> dict:
    """Sweep attrs + skip mask, the standalone filter for this cell's filter backend, the
    exact oracle blob (filter cells), bloom pass counts (bloom cells) and the row masks the
    quality pass scores: ``keep`` (not skip-masked), ``oracle_rows`` (kept and ≥ 1
    survivor — the old harness's zero-target skip), ``heldout_rows`` (kept, ≥ 1 target and,
    on filter cells, ``target_in_filter``), and ``n_targets_in_filter`` — the held-out
    targets those rows score (on filter cells only the ones the exact mask admits)."""
    fk, n = job.filter_kind, inp["n_queries"]
    qa_s, skip = inputs.sweep_qa(inp["qa"], job.clauses)
    filters = inputs.build_filters(fk, inp, [job.backend], bloom=job.bloom)
    filter_mod = filters.get(algos.filter_backend(job.backend))
    keep = ~skip if skip is not None else torch.ones(n, dtype=torch.bool)
    heldout = keep & (inp["n_targets"] > 0)
    blob = None
    bloom_fp = None
    oracle_rows = None
    if fk != "none":
        exact = inputs.exact_filter(fk, filters, inp, job.backend)
        blob = oracle.load_or_build(
            job.data.gt_dir,
            job.sweep,
            k_max,
            item_embs=inp["item_embs"],
            queries=inp["queries"],
            targets=inp["targets"],
            qa_sweep=qa_s,
            skip_mask=skip,
            clauses=job.clauses,
            filter_mod=exact,
            attrs_digest=inp["attrs_digest"],
            device=device,
        )
        heldout &= blob["target_in_filter"]
        oracle_rows = keep & (blob["topk"][:, 0] != -1)
        if fk == "bloom":
            assert filter_mod is not None
            counts = oracle.pass_counts(filter_mod, qa_s, skip, device=device)
            bloom_fp = oracle.bloom_fp_rate(counts, blob["pass_counts"], inp["n_items"])
    n_tif = (inp["n_targets"] if blob is None else blob["targets_in_filter"])[heldout].sum()
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


def build_module(job: Job, inp: dict, assets: dict, k_max: int, params: dict) -> nn.Module:
    kw = dict(params)
    if job.algo == "silvertorch" and job.filter_kind == "bloom":
        kw = {**job.bloom, **kw}
    return algos.build(
        job.algo,
        inp["item_embs"],
        k=k_max,
        backend=job.backend,
        filter_kind=job.filter_kind,
        filter_mod=assets["filter_mod"],
        item_attrs=inp["item_attrs"],
        clause_is_reverse=inp["clause_is_reverse"],
        params=kw,
        seed=job.seed,
    )


# ----- quality + parity ---------------------------------------------------------------------


@torch.inference_mode()
def quality(
    module: nn.Module, inp: dict, assets: dict, ks: Sequence[int], device: torch.device
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
        q = inp["queries"][sel].to(device, non_blocking=True)
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
            t = inp["targets"][sel][m].to(device, non_blocking=True)
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
    atomic_write(
        ref,
        lambda fh: np.savez(
            fh,
            ids=ids.cpu().numpy().astype(np.int32),
            scores=scores.cpu().numpy().astype(np.float32),
            backend=np.array(job.backend),
        ),
    )
    out["parity"] = "reference"
    return out


# ----- perf -----------------------------------------------------------------------------------


def _call_next(callee: Any, pool: torch.Tensor, counter: Iterator[int]) -> Any:
    return callee(pool[next(counter) % pool.shape[0]])


def _call_next_filtered(
    callee: Any, pool: torch.Tensor, qa_pool: torch.Tensor, counter: Iterator[int]
) -> Any:
    i = next(counter) % pool.shape[0]
    return callee(pool[i], qa_pool[i])


def _rotate(callee: Any, pool: torch.Tensor, qa_pool: torch.Tensor | None) -> Any:
    """The zero-arg call ``measure.latency`` times: the pool rotated round-robin (§2.5)."""
    if qa_pool is None:
        return partial(_call_next, callee, pool, itertools.count())
    return partial(_call_next_filtered, callee, pool, qa_pool, itertools.count())


def perf(
    module: nn.Module,
    inp: dict,
    assets: dict,
    job: Job,
    device: torch.device,
    *,
    modes: Sequence[str],
    profile: bool,
    latency_kw: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """§2.5 per ``(bs, k, mode)``: the fixed-seed pool rotated round-robin, ``module.k = k``
    before each variant, ``graph`` via ``measure.graph_callable`` (one capture per shape) or a
    null entry with the ``reason`` (``official`` → ``not_capturable``, O D7). The official
    backend's plan cache is switched *off* for timing — every forward pays the expression
    parse, as serving fresh queries does (kernels.md); ``OfficialConfig.cache_plans`` is read
    per forward, so this is an in-place replace, no rebuild — and every entry records
    ``cache_plans`` (``None`` on backends without such a cache)."""
    entries: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    cache_plans = None
    if module.backend == "official":
        module.official = dataclasses.replace(module.official, cache_plans=False)
        cache_plans = module.official.cache_plans
    for bs in job.batch_sizes:
        pool, qa_pool = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=device
        )
        for k in job.ks:
            module.k = int(k)
            for mode in modes:
                entry: dict[str, Any] = {
                    "k": int(k),
                    "bs": int(bs),
                    "mode": mode,
                    "cache_plans": cache_plans,
                }
                if mode == "graph":
                    example = (pool[0],) if qa_pool is None else (pool[0], qa_pool[0])
                    try:
                        callee = measure.graph_callable(module, *example)
                    except measure.NotCapturable as exc:
                        logger.info("  graph bs={} k={}: {}", bs, k, exc)
                        entries.append(
                            {**entry, **dict.fromkeys(PERF_STAT_KEYS), "reason": str(exc)}
                        )
                        continue
                else:
                    callee = module
                fn = _rotate(callee, pool, qa_pool)
                with torch.inference_mode():
                    d, ms = measure.latency(fn, bs=int(bs), mode=mode, **latency_kw)
                    if profile and mode == "eager":
                        d["kernels"] = measure.profile_once(fn)
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
        "schema_version": records.SCHEMA_VERSION,
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
    env_extra: dict[str, Any] | None = None,
    latency_kw: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> Counter:
    """Run every cell of ``jobs`` in this process (module docstring). ``out_path`` overrides
    the per-``(suite, dataset, dim)`` file; ``latency_kw`` is forwarded to ``measure.latency``
    (tests shrink the windows). Returns ``Counter(ok / partial / failed / skipped)``."""
    counts: Counter = Counter()
    if not jobs:
        return counts
    out_dir = Path(out_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measure.setup(jobs[0].seed)
    measure.warm_gpu_once()
    clk0 = measure.clocks()
    env0 = {
        **measure.provenance(),
        **(env_extra or {}),
        "sm_mhz_idle": clk0["sm_mhz"],
        **{k: v for k, v in clk0.items() if k != "sm_mhz"},
    }
    code_version = env0["code_version"]
    sm_ref: float | None = None  # the process's first under-load sample
    latency_kw = dict(latency_kw or {})
    reasons0 = [r for r, on in (("skip_quality", skip_quality), ("skip_perf", skip_perf)) if on]
    if set(modes) != set(MODES):
        reasons0.append("modes")
    with_filters = {(j.dataset, j.dim): False for j in jobs}
    for j in jobs:
        with_filters[j.dataset, j.dim] |= j.filter_kind != "none"

    existing: dict[Path, dict[str, str]] = {}
    inp: dict[str, Any] | None = None
    inputs_key: tuple | None = None
    assets: dict | None = None
    assets_key: tuple | None = None
    for job in jobs:
        path = out_path or records.record_path(out_dir, job)
        if path not in existing:
            existing[path] = records.read_keys(path)
        todo = [
            p
            for p in job.cells()
            if not (
                resume and existing[path].get(records.resume_key(job.key(p), code_version)) == "ok"
            )
        ]
        counts["skipped"] += len(job.cells()) - len(todo)
        if not todo:
            logger.info("resume: {} all {} cells done", job.group, len(job.cells()))
            continue
        if (job.dataset, job.dim) != inputs_key:
            inp = assets = None
            assets_key = None
            _release()
            inp = inputs.load_inputs(
                job.data, device, with_filters=with_filters[job.dataset, job.dim]
            )
            inputs_key = (job.dataset, job.dim)
        assert inp is not None
        k_max = max(job.ks)
        akey = (
            job.filter_kind,
            job.sweep,
            algos.filter_backend(job.backend),
            k_max,
            tuple(sorted(job.bloom.items())),
        )
        if akey != assets_key:
            assets = None
            _release()
            assets = sweep_assets(job, inp, k_max, device)
            assets_key = akey
        assert assets is not None

        measure.setup(job.seed)
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
            module, build_s = measure.timed_build(
                lambda: build_module(job, inp, assets, k_max, todo[0])  # noqa: B023 — called at once
            )
        except Exception as exc:  # recorded, the loop continues (H §7)
            logger.exception("build failed: {}", job.key(todo[0]))
            for p in todo:
                records.append_record(path, _failed(job, p, env0, "build", exc, t0))
                counts["failed"] += 1
            if is_sticky(exc):
                logger.error("sticky CUDA error: the context is dead, ending this process")
                raise
            _release()
            continue
        index_mib = measure.index_bytes(module) / MiB
        filter_mib = measure.index_bytes(getattr(module, "filter", None)) / MiB

        for params in todo:
            t0 = time.perf_counter()
            stage = "query_params"
            try:
                q = {k: v for k, v in params.items() if k in QUERY_PARAMS}
                if q:
                    module.set_query_params(**q)
                reasons = reasons0 + (["ks_bs"] if job.narrowed else [])
                rec: dict[str, Any] = {
                    "schema_version": records.SCHEMA_VERSION,
                    "status": "partial" if reasons else "ok",
                    "partial_reasons": reasons or None,
                    **job.key(params),
                    "path": job.path,
                    "n_items": inp["n_items"],
                    "n_queries": inp["n_queries"],
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
                    qual, ids, scores = quality(module, inp, assets, job.ks, device)
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
                        inp,
                        assets,
                        job,
                        device,
                        modes=modes,
                        profile=profile,
                        latency_kw=latency_kw,
                    )
                loads = [e["sm_mhz"] for e in rec["perf"] or [] if e.get("sm_mhz")]
                if loads and sm_ref is None:
                    sm_ref = loads[0]
                drift = any(abs(s - sm_ref) > CLOCK_DRIFT * sm_ref for s in loads)
                if drift:
                    logger.warning("under-load clocks.sm {} MHz vs first {} MHz", loads, sm_ref)
                env = {
                    **env0,
                    "sm_mhz_load": statistics.median(loads) if loads else None,
                    "clocks_drift": drift,
                }
                rec["unstable"] = drift or any(e.get("unstable") for e in rec["perf"] or [])
                rec["memory_reserved_mib"] = (
                    torch.cuda.memory_reserved() / MiB if torch.cuda.is_available() else None
                )
                rec["elapsed_s"] = time.perf_counter() - t0
                rec["env"] = env
            except QualityGateError as exc:
                records.append_record(path, _failed(job, params, env0, stage, exc, t0))
                counts["failed"] += 1
                raise
            except Exception as exc:  # recorded, the loop continues (H §7)
                logger.exception("cell failed at {}: {}", stage, job.key(params))
                records.append_record(path, _failed(job, params, env0, stage, exc, t0))
                counts["failed"] += 1
                if is_sticky(exc):
                    logger.error("sticky CUDA error: the context is dead, ending this process")
                    raise
                _release()
                continue
            key = job.key(params)
            for s in samples:  # samples first: a record without its vector is never resumable
                records.append_record(records.samples_path(path), {**key, **s})
            records.append_record(path, rec)
            existing[path][records.resume_key(key, code_version)] = rec["status"]
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
    "CLOCK_DRIFT",
    "EXACT_ALGOS",
    "MODES",
    "PERF_STAT_KEYS",
    "QUALITY_CHUNK",
    "STICKY_CUDA",
    "QualityGateError",
    "is_sticky",
    "run",
]
