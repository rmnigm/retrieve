"""The cell loop (docs/system/evaluation.md § The cell loop): ``run(jobs, out_dir=...)`` runs
``Job``s in this process and appends one JSONL record per cell as it finishes."""

from __future__ import annotations

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
from bench.config import QUERY_PARAMS, Job, interleave_units, shared_key
from bench.metrics import accumulate, accumulator, drop_self, finalize, jaccard_at_k
from eval_datasets.layout import atomic_write

MODES = ("eager", "graph")
QUALITY_CPU_THREADS = 1  # the loop's host work is [chunk]-row masks; more threads spin
QUALITY_CHUNK = 16  # the OOM bound: [B, P, D] on loose filters
EXACT_ALGOS = ("linr_v1_filter_mask", "linr_v2")  # §2.4: recall_oracle@k_max >= 0.99 or die
EXACT_MIN_RECALL = 0.99
# Arms whose quality cannot depend on the seed (it only moves their perf pool): the quality
# cache copies another seed's quality to them. SilverTorch (k-means) and V3 (OPORP) recompute.
SEED_FREE_QUALITY = ("linr_v1_filter_mask", "linr_v2", "postfilter")
# The top-k prefix of these algos' k_max run is not their top-k run (postfilter's pool is
# alpha*k), so quality runs once per k.
PER_K_QUALITY = ("postfilter",)
ARM_FREE_SLACK_MIB = 256
CLOCK_DRIFT = 0.05  # §2.1: an under-load sample > 5 % off the process's first one
POOL_MIN = 2000  # the floor of a resolved candidate_pool_frac
NO_ATTRS = oracle.attrs_digest(None, None)  # a none cell's oracle ignores the attrs
# An exception whose message carries one of these has killed the CUDA context: recorded, then
# re-raised so the child exits and ``bench campaign`` moves on.
STICKY_CUDA = ("CUDA error", "illegal memory access", "device-side assert")
MiB = measure.MiB
# The stat keys of a perf entry; a variant that cannot run records them as null + ``reason``.
PERF_STAT_KEYS = (
    "n", "median_ms", "mean_ms", "trimmed_mean_ms", "p95_ms", "p99_ms", "min_ms", "iqr_ms", "qps",
    "host_gap_ms", "outliers_std", "outliers_tukey", "spread", "unstable", "peak_fwd_mib",
    "window_medians_ms", "window_sm_mhz", "ids_sha256", "ids_sha256_canon",
)  # fmt: skip
IDS_PROBE_BATCHES = 8  # the pool batches ``ids_sha256`` hashes, outside the timed windows


class QualityGateError(RuntimeError):
    """An exact algo scored below ``EXACT_MIN_RECALL`` against the oracle (§2.4)."""


def is_sticky(exc: BaseException) -> bool:
    """A CUDA error after which the process cannot measure anything else."""
    msg = str(exc)
    return any(s in msg for s in STICKY_CUDA)


def sweep_assets(job: Job, inp: dict[str, Any], k_max: int, device: torch.device) -> dict:
    """One sweep's assets: query attrs and skip mask, the standalone filter (clause cells), the
    oracle blob and the row masks the quality pass scores (docs/system/evaluation.md
    § Measurement protocol, § Self-match)."""
    fk, n = job.filter_kind, inp["n_queries"]
    qa_s, skip = inputs.sweep_qa(inp["qa"], job.clauses)
    filters = inputs.build_filters(fk, inp, [job.backend])
    filter_mod = filters.get(algos.filter_backend(job.backend))
    keep = ~skip if skip is not None else torch.ones(n, dtype=torch.bool)
    heldout = keep & (inp["n_targets"] > 0)
    self_ids = self_top1 = None
    blob = sweep_oracle(job, inp, k_max, device, filters, qa_s, skip, warn_missing=True)
    oracle_topk = blob["topk"]
    if fk != "none":
        heldout &= blob["target_in_filter"]
        if job.self_excluded:
            self_ids = inp["targets"][:, 0]
            reach = keep & blob["target_in_filter"]
            self_top1 = (oracle_topk[reach, 0] == self_ids[reach]).double().mean().item()
            logger.info("  self-match {}/{}: oracle top-1 is the query's own item on {:.4f} of "
                        "{} rows", job.dataset, job.sweep, self_top1, int(reach.sum()))  # fmt: skip
            oracle_topk = drop_self(oracle_topk, self_ids)
    oracle_rows = keep & (oracle_topk[:, 0] != -1)
    n_tif = blob["targets_in_filter"][heldout].sum()
    return {
        "qa_s": qa_s,
        "skip": skip,
        "filter_mod": filter_mod,
        "blob": blob,
        "oracle_topk": oracle_topk,
        "self_ids": self_ids,
        "self_top1_share": self_top1,
        "keep": keep,
        "oracle_rows": oracle_rows,
        "heldout_rows": heldout,
        "n_targets_in_filter": int(n_tif),
        "pass_rate": blob["pass_rate"],
    }


def sweep_oracle(
    job: Job,
    inp: dict,
    k_max: int,
    device: torch.device,
    filters: dict,
    qa_s: torch.Tensor | None,
    skip: torch.Tensor | None,
    *,
    warn_missing: bool,
) -> dict[str, Any]:
    """The job's oracle blob at ``k_max`` under the exact mask (``k_max + 1`` on a
    self-excluded cell); a ``none`` cell's blob ignores the attrs, so every suite shares it."""
    return oracle.load_or_build(
        job.data.gt_dir,
        job.sweep,
        k_max + job.self_excluded,
        item_embs=inp["item_embs"],
        queries=inp["queries"],
        targets=inp["targets"],
        qa_sweep=qa_s,
        skip_mask=skip,
        clauses=job.clauses,
        filter_mod=inputs.exact_filter(job.filter_kind, filters, inp, job.backend),
        attrs_digest=inp["attrs_digest"] if job.filter_kind != "none" else NO_ATTRS,
        device=device,
        warn_missing=warn_missing,
        slice_wider=job.self_excluded,
    )


def prebuild_oracles(jobs: Sequence[Job], device: torch.device | None = None) -> Counter:
    """``bench oracle``: every blob ``jobs`` read, built or found, no timing."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measure.setup(0)
    counts: Counter = Counter()
    by_inputs: dict[tuple, dict[tuple, Job]] = {}
    for j in jobs:
        by_inputs.setdefault((j.dataset, j.dim), {})[j.filter_kind, j.sweep, max(j.ks)] = j
    for group in by_inputs.values():
        filtered = any(fk != "none" for fk, _, _ in group)
        inp = inputs.load_inputs(next(iter(group.values())).data, device, with_filters=filtered)
        for j in group.values():
            fk = "clause" if j.filter_kind == "clause" else "none"  # bloom builds its exact one
            filters = inputs.build_filters(fk, inp, [j.backend])
            qa_s, skip = inputs.sweep_qa(inp["qa"], j.clauses)
            sweep_oracle(j, inp, max(j.ks), device, filters, qa_s, skip, warn_missing=False)
            counts["sweeps"] += 1
        del inp
        _release()
    return counts


def resolve_pool(params: dict, assets: dict) -> dict:
    """``candidate_pool_frac`` → ``candidate_pool = max(POOL_MIN, round(frac × mean pass
    count))``; the key keeps the fraction, the module gets the int."""
    if "candidate_pool_frac" not in params:
        return params
    p = dict(params)
    p["candidate_pool"] = max(POOL_MIN, round(p.pop("candidate_pool_frac") * passing_mean(assets)))
    return p


def passing_mean(assets: dict) -> float:
    """The mean exact pass count over the sweep's oracle rows: what a pool fraction is of."""
    return assets["blob"]["pass_counts"][assets["oracle_rows"]].double().mean().item()


def compile_warmup(module: nn.Module, inp: dict, assets: dict, device: torch.device) -> float:
    """Seconds of a compiled module's first forward, kept apart from ``build_s``."""
    sel = assets["keep"].nonzero().reshape(-1)[:QUALITY_CHUNK]
    q = inp["queries"][sel].to(device)
    qa = assets["qa_s"][sel].to(device) if assets["qa_s"] is not None else None
    t0 = time.perf_counter()
    with torch.inference_mode():
        module(q, filter_arg(module, qa))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


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


@torch.inference_mode()
def quality(
    module: nn.Module, inp: dict, assets: dict, ks: Sequence[int], device: torch.device
) -> tuple[dict[str, Any], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """§2.4: the kept rows through ``module`` at ``k_max`` in chunks, every ``k`` scored from
    the one top-``k_max`` list as device running sums. Returns the metrics, the per-query
    recall (NaN where a side does not score the row) and the ids / scores for the parity
    spill. A held-out target the exact mask excludes is masked to ``-1``: it can never be
    retrieved, and scoring it would bias recall down."""
    rows = assets["keep"].nonzero().reshape(-1)
    blob = assets["blob"]
    k_out = max(ks)
    acc_o = accumulator(list(ks), device)
    acc_h = accumulator(list(ks), device)
    nan = partial(torch.full, (rows.numel(),), float("nan"), device=device)
    per_q = {f"{name}@{k}": nan() for name in ("recall_oracle", "heldout_recall") for k in ks}
    ids_all: list[torch.Tensor] = []
    sc_all: list[torch.Tensor] = []
    rows_d = rows.to(device)
    queries = inp["queries"].to(device)
    qa_all = assets["qa_s"].to(device) if assets["qa_s"] is not None else None
    targets = inp["targets"].to(device)
    topk = assets["oracle_topk"].to(device)
    self_ids = assets["self_ids"].to(device) if assets["self_ids"] is not None else None
    in_filter = blob["targets_in_filter"].to(device)
    threads = torch.get_num_threads()
    torch.set_num_threads(QUALITY_CPU_THREADS)
    try:
        for s in range(0, rows.numel(), QUALITY_CHUNK):
            sel = rows[s : s + QUALITY_CHUNK]
            sel_d = rows_d[s : s + QUALITY_CHUNK]
            q = queries.index_select(0, sel_d)
            qa = qa_all.index_select(0, sel_d) if qa_all is not None else None
            ids, scores = module(q, filter_arg(module, qa))
            ids_o = drop_self(ids, self_ids.index_select(0, sel_d)) if self_ids is not None else ids
            ids, scores = ids[:, :k_out], scores[:, :k_out]
            # a compiled arm's outputs live in CUDA-graph buffers the next call overwrites
            ids_all.append(ids.clone())
            sc_all.append(scores.float().clone())
            m = assets["oracle_rows"][sel]
            if bool(m.any()):
                idx = m.nonzero().reshape(-1).to(device, non_blocking=True)
                r = accumulate(
                    acc_o,
                    ids_o.index_select(0, idx),
                    topk.index_select(0, sel_d.index_select(0, idx)),
                    ranked=True,
                )
                for k, v in r.items():
                    per_q[f"recall_oracle@{k}"].index_copy_(0, idx + s, v)
            m = assets["heldout_rows"][sel]
            if bool(m.any()):
                idx = m.nonzero().reshape(-1).to(device, non_blocking=True)
                held = sel_d.index_select(0, idx)
                t = targets.index_select(0, held)
                t = t.masked_fill(~in_filter.index_select(0, held), -1)  # reachable targets only
                r = accumulate(acc_h, ids.index_select(0, idx), t, (t != -1).sum(dim=1))
                for k, v in r.items():
                    per_q[f"heldout_recall@{k}"].index_copy_(0, idx + s, v)
    finally:
        torch.set_num_threads(threads)
    out: dict[str, Any] = {"heldout": finalize(acc_h), "oracle": finalize(acc_o)}
    if self_ids is not None:
        out["self_excluded"] = True
        out["self_top1_share"] = assets["self_top1_share"]
    ids_t = torch.cat(ids_all) if ids_all else torch.empty(0, 0, dtype=torch.long)
    sc_t = torch.cat(sc_all) if sc_all else torch.empty(0, 0)
    return out, {n: t.cpu() for n, t in per_q.items()}, ids_t, sc_t


def per_query_path(job: Job, params: dict, code_version: str) -> str:
    """The per-query sidecar of one cell, relative to the results root."""
    h = hashlib.sha1(records.resume_key(job.key(params), code_version).encode()).hexdigest()
    return f"{job.suite}/{job.dataset}-d{job.dim}.perquery/{h[:20]}.npz"


def write_per_query(root: Path, rel: str, assets: dict, per_q: dict[str, torch.Tensor]) -> None:
    """The per-query sidecar (docs/system/evaluation.md § Per-query sidecar)."""
    rows = assets["keep"].nonzero().reshape(-1)
    counts = assets["blob"]["pass_counts"][rows]
    arrays = {n: t.numpy().astype(np.float32) for n, t in per_q.items()}
    atomic_write(
        Path(root) / rel,
        lambda fh: np.savez_compressed(
            fh,
            rows=rows.numpy().astype(np.int32),
            pass_count=counts.numpy().astype(np.int64),
            **arrays,
        ),
    )


def parity_group(dataset: str, dim: int, algo: str) -> str:
    """The ``_parity/`` subdirectory of one ``(dataset, dim, algo)`` group."""
    return f"{dataset}-d{dim}_{algo}"


def parity(
    out_dir: Path,
    job: Job,
    params: dict,
    ids: torch.Tensor,
    scores: torch.Tensor,
    ks: Sequence[int],
) -> dict[str, Any]:
    """The parity spill (§2.4): write the reference when absent, else compare against it."""
    key = {k: v for k, v in job.key(params).items() if k != "backend"}
    # score_path picks the official kernel's epilogue, a backend knob: its arms share a spill
    key["params"] = {k: v for k, v in key["params"].items() if k != "score_path"}
    h = hashlib.sha1(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:20]
    ref = Path(out_dir) / "_parity" / parity_group(job.dataset, job.dim, job.algo) / f"{h}.npz"
    out: dict[str, Any] = {f"jaccard_vs_first@{k}": None for k in ks}
    out["score_max_abs_diff"] = None
    z = np.load(ref) if ref.exists() else None
    if z is not None and z["backend"].item() != job.backend:  # own spill = re-run after a kill
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


def quality_key(key: dict[str, Any], code_version: str, ks: Sequence[int]) -> str:
    """The quality cache's key: the key block minus ``seed``, plus ``code_version`` and ``ks``."""
    rest = {k: v for k, v in key.items() if k != "seed"}
    return json.dumps({**rest, "code_version": code_version, "ks": list(ks)}, sort_keys=True)


def quality_sources(recs: Sequence[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    """``{quality_key: {seed: record}}`` over the records that computed their quality."""
    out: dict[str, dict[int, dict[str, Any]]] = {}
    for r in recs:
        if (
            r.get("status") in ("ok", "partial")
            and r.get("quality")
            and not r.get("quality_source")
        ):
            qk = quality_key(records.key_block(r), r["env"]["code_version"], r["ks"])
            out.setdefault(qk, {})[r["seed"]] = r
    return out


def cached_quality(
    sources: dict[str, dict[int, dict[str, Any]]], job: Job, params: dict, code_version: str
) -> dict[str, Any] | None:
    """The record whose quality a ``SEED_FREE_QUALITY`` cell copies: the lowest other seed's."""
    if job.algo not in SEED_FREE_QUALITY:
        return None
    by_seed = sources.get(quality_key(job.key(params), code_version, job.ks), {})
    seeds = sorted(s for s in by_seed if s != job.seed)
    return by_seed[seeds[0]] if seeds else None


def _call_next(callee: Any, pool: torch.Tensor, counter: Iterator[int]) -> Any:
    return callee(pool[next(counter) % pool.shape[0]])


def _call_next_filtered(
    callee: Any, pool: torch.Tensor, qa_pool: torch.Tensor, counter: Iterator[int]
) -> Any:
    i = next(counter) % pool.shape[0]
    return callee(pool[i], qa_pool[i])


def filter_arg(module: nn.Module, qa: torch.Tensor | None) -> Any:
    """What the forward takes for a batch's query attrs: ``prepare_queries``'s result or the
    attrs."""
    prepare = getattr(module, "prepare_queries", None)
    return qa if qa is None or prepare is None else prepare(qa)


def prepare_pool(module: nn.Module, qa_pool: torch.Tensor | None) -> tuple[Any, float | None]:
    """Every pool batch's filter argument, prepared before timing, and the mean ms of one
    ``prepare_queries`` (docs/system/evaluation.md § Query preparation)."""
    if qa_pool is None or getattr(module, "prepare_queries", None) is None:
        return qa_pool, None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        prepared = [module.prepare_queries(qa_pool[i]) for i in range(qa_pool.shape[0])]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return prepared, (time.perf_counter() - t0) * 1e3 / len(prepared)


def _rotate(callee: Any, pool: torch.Tensor, qa_pool: Any) -> Any:
    """The zero-arg call ``measure.latency`` times: the pool rotated round-robin (§2.5)."""
    if qa_pool is None:
        return partial(_call_next, callee, pool, itertools.count())
    return partial(_call_next_filtered, callee, pool, qa_pool, itertools.count())


@torch.inference_mode()
def ids_sha256(callee: Any, pool: torch.Tensor, qa_pool: Any) -> tuple[str, str]:
    """``(exact, canon)`` sha256 of the ids ``callee`` returns on the first probe batches: as
    returned, and with each row re-ordered by (score desc, id asc) so tied ids in another
    order hash equal. Each batch is copied out: a graph replay overwrites its buffers."""
    exact, canon = hashlib.sha256(), hashlib.sha256()
    for i in range(min(IDS_PROBE_BATCHES, pool.shape[0])):
        ids, scores = callee(pool[i]) if qa_pool is None else callee(pool[i], qa_pool[i])
        ids, scores = ids.to(torch.int64).cpu(), scores.float().cpu()
        exact.update(ids.contiguous().numpy().tobytes())
        by_id = ids.argsort(dim=1, stable=True)
        ids, scores = ids.gather(1, by_id), scores.gather(1, by_id)
        order = scores.argsort(dim=1, descending=True, stable=True)
        canon.update(ids.gather(1, order).contiguous().numpy().tobytes())
    return exact.hexdigest(), canon.hexdigest()


def perf(
    modules: Sequence[nn.Module],
    inp: dict,
    assets: dict,
    job: Job,
    device: torch.device,
    *,
    modes: Sequence[str],
    profile: bool,
    latency_kw: dict[str, Any],
) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """§2.5 per ``(bs, k, mode)`` for the arms of one cell or interleave group; a variant that
    cannot run gets a null entry with its ``reason``. Returns ``(entries, samples)`` per
    module."""
    out: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = [([], []) for _ in modules]
    for bs in job.batch_sizes:
        pool, qa_pool = inputs.query_pool(
            inp, assets["qa_s"], assets["skip"], bs=bs, seed=job.seed, device=device
        )
        preps = [prepare_pool(m, qa_pool) for m in modules]
        for k in job.ks:
            for m in modules:
                m.k = int(k)
            for mode in modes:
                callees: dict[int, Any] = {}
                if mode == "graph":
                    torch._dynamo.reset()
                for i, m in enumerate(modules):
                    entry = {"k": int(k), "bs": int(bs), "mode": mode, "query_prep_ms": preps[i][1]}
                    if mode == "eager":
                        callees[i] = m
                        continue
                    fa = preps[i][0]
                    example = (pool[0],) if fa is None else (pool[0], fa[0])
                    try:
                        callees[i] = measure.graph_callable(m, *example)
                    except measure.NotCapturable as exc:
                        logger.info("  graph bs={} k={}: {}", bs, k, exc)
                        out[i][0].append(
                            {**entry, **dict.fromkeys(PERF_STAT_KEYS), "reason": str(exc)}
                        )
                if not callees:
                    continue
                fns = {i: _rotate(c, pool, preps[i][0]) for i, c in callees.items()}
                with torch.inference_mode():
                    timed = measure.latency_group(
                        list(fns.values()), bs=int(bs), mode=mode, **latency_kw
                    )
                    for (i, fn), (d, ms) in zip(fns.items(), timed, strict=True):
                        if profile and mode == "eager":
                            d.update(measure.profile_once(fn))
                        d["ids_sha256"], d["ids_sha256_canon"] = ids_sha256(
                            callees[i], pool, preps[i][0]
                        )
                        if len(modules) > 1:
                            d["rounds"] = len(d["window_medians_ms"])
                        entry = {"k": int(k), "bs": int(bs), "mode": mode}
                        out[i][0].append({**entry, "query_prep_ms": preps[i][1], **d})
                        out[i][1].append({**entry, "ms": ms})
                callees = fns = {}
            torch._dynamo.reset()
    return out


def _log_failure(msg: str, exc: BaseException) -> None:
    """``logger.exception`` without loguru's ``diagnose``: it caches ``run``'s locals on the
    frame, so the failed arm's tensors would stay referenced for the rest of the run."""
    logger.error("{}\n{}", msg, "".join(traceback.format_exception(exc)))


def _release() -> None:
    gc.collect()
    torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _allocated() -> int:
    return torch.cuda.memory_allocated() if torch.cuda.is_available() else 0


def _assert_freed(base: int) -> None:
    """A unit's arms are gone once it ends (evaluation.md § Arm release): device memory back to
    the shared inputs' and assets' ``base``, within ``ARM_FREE_SLACK_MIB``."""
    left = (_allocated() - base) / MiB
    if left > ARM_FREE_SLACK_MIB:
        raise RuntimeError(f"{left:.0f} MiB still allocated after the unit's arms were released")


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


def _record_failures(
    path: Path,
    cells: Sequence[tuple[Job, dict]],
    env: dict,
    stage: str,
    exc: BaseException,
    t0: float,
    counts: Counter,
    msg: str,
) -> bool:
    """Log ``exc`` and append one ``failed`` record per cell; returns whether it is sticky."""
    _log_failure(msg, exc)
    for job, params in cells:
        records.append_record(path, _failed(job, params, env, stage, exc, t0))
        counts["failed"] += 1
    if is_sticky(exc):
        logger.error("sticky CUDA error: the context is dead, ending this process")
        return True
    return False


def _new_record(
    job: Job,
    params: dict,
    resolved: dict,
    built: dict[str, Any],
    assets: dict,
    inp: dict,
    interleave: dict[str, Any] | None,
) -> dict[str, Any]:
    """A cell's record before quality and perf: its key block, inputs, sweep assets and build."""
    return {
        "schema_version": records.SCHEMA_VERSION,
        "status": "partial" if built["reasons"] else "ok",
        "partial_reasons": built["reasons"] or None,
        **job.key(params),
        "path": job.path,
        "n_items": inp["n_items"],
        "n_queries": inp["n_queries"],
        "n_kept": int(assets["keep"].sum()),
        "n_queries_heldout": int(assets["heldout_rows"].sum()),
        "n_targets_in_filter": assets["n_targets_in_filter"],
        "n_queries_oracle": int(assets["oracle_rows"].sum()),
        "pass_rate": assets["pass_rate"],
        "bloom": dict(job.bloom) if job.filter_kind == "bloom" else None,
        "k_max": max(job.ks),
        "ks": list(job.ks),
        "batch_sizes": list(job.batch_sizes),
        "build_s": built["build_s"],
        "compile_s": built["compile_s"],
        "candidate_pool": resolved.get("candidate_pool"),
        "pool_passing_mean": passing_mean(assets) if "candidate_pool_frac" in params else None,
        "index_mib": built["index_mib"],
        "filter_mib": built["filter_mib"],
        "seed_scope": "pool" if job.algo in SEED_FREE_QUALITY else "pool+build",
        "quality_source": None,
        "per_query": None,
        "interleave": interleave,
        "quality": None,
        "perf": None,
    }


def _akey(job: Job) -> tuple:
    """The step-3 asset key: what ``sweep_assets`` depends on."""
    return (
        job.filter_kind,
        job.sweep,
        algos.filter_backend(job.backend),
        max(job.ks),
        tuple(sorted(job.bloom.items())),
    )


def group_label(job: Job, by: Sequence[str]) -> str:
    """One arm of an interleave group: ``algo/backend`` plus each ``by`` build param it sets."""
    return "/".join([job.algo, job.backend, *(f"{f}={job.build[f]}" for f in by if f in job.build)])


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
    interleave: bool = False,
    env_extra: dict[str, Any] | None = None,
    latency_kw: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> Counter:
    """Run every cell of ``jobs`` in this process; with ``interleave`` the arms of a comparison
    group are built side by side and timed round-robin. Returns ``Counter(ok / partial /
    failed / skipped)``."""
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
    sm_windows: list[float] = []  # every window's under-load sample: the job log's histogram
    latency_kw = dict(latency_kw or {})
    reasons0 = ["skip_quality"] if skip_quality else []
    skipped_modes = set(MODES) - set(modes)
    with_filters = {(j.dataset, j.dim) for j in jobs if j.filter_kind != "none"}
    units = interleave_units(jobs) if interleave else [(None, [j]) for j in jobs]

    existing: dict[Path, dict[str, str]] = {}
    sources: dict[Path, dict[str, dict[int, dict[str, Any]]]] = {}
    inp: dict[str, Any] | None = None
    inputs_key: tuple | None = None
    assets_by: dict[tuple, dict] = {}
    for by, unit in units:
        path = out_path or records.record_path(out_dir, unit[0])
        if path not in existing:
            recs = records.read_records(path)
            existing[path] = {records.record_key(r): r.get("status", "ok") for r in recs}
            sources[path] = quality_sources(recs)
            del recs
        todo: dict[int, list[dict[str, Any]]] = {id(j): [] for j in unit}
        combos = list({json.dumps(q, sort_keys=True): q for j in unit for q in j.query}.values())
        for q in combos:
            cells = [(j, {**j.build, **q}) for j in unit if q in j.query]
            done = [
                resume and existing[path].get(records.resume_key(j.key(p), code_version)) == "ok"
                for j, p in cells
            ]
            if all(done):
                counts["skipped"] += len(cells)
                continue
            for j, p in cells:  # a group re-runs whole
                todo[id(j)].append(p)
        if not any(todo.values()):
            logger.info("resume: {} all cells done", [j.group for j in unit])
            continue
        if (unit[0].dataset, unit[0].dim) != inputs_key:
            inp = None
            assets_by = {}
            _release()
            inp = inputs.load_inputs(
                unit[0].data, device, with_filters=(unit[0].dataset, unit[0].dim) in with_filters
            )
            inputs_key = (unit[0].dataset, unit[0].dim)
        assert inp is not None
        need = {_akey(j): j for j in unit if todo[id(j)]}
        if set(assets_by) - set(need):
            assets_by = {a: v for a, v in assets_by.items() if a in need}
            _release()
        for a, j in need.items():
            if a not in assets_by:
                assets_by[a] = sweep_assets(j, inp, max(j.ks), device)
        base = _allocated()

        built: dict[int, dict[str, Any]] = {}
        for job in unit:
            if not todo[id(job)]:
                continue
            assets = assets_by[_akey(job)]
            k_max = max(job.ks)
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
            first = todo[id(job)][0]
            module = None
            try:
                module, build_s = measure.timed_build(
                    lambda: build_module(job, inp, assets, k_max, resolve_pool(first, assets))  # noqa: B023 — called at once
                )
                compile_s = (
                    compile_warmup(module, inp, assets, device) if "compile" in job.build else None
                )
            except Exception as exc:  # recorded, the loop continues
                cells = [(job, p) for p in todo[id(job)]]
                msg = f"build failed: {job.key(first)}"
                if _record_failures(path, cells, env0, "build", exc, t0, counts, msg):
                    raise
                module = None
            if module is None:
                _release()  # after the except: its traceback holds the failed build's tensors
                continue
            # A module that cannot capture loses nothing to a skipped graph mode; a quality-only
            # suite (``Job.timed`` false) loses nothing to a skipped perf.
            lost = skipped_modes - (set() if getattr(module, "capturable", True) else {"graph"})
            reasons = reasons0 + (["skip_perf"] if job.timed and skip_perf else [])
            reasons += ["modes"] if lost and job.timed else []
            built[id(job)] = {
                "module": module,
                "build_s": build_s,
                "compile_s": compile_s,
                "reasons": reasons + (["ks_bs"] if job.narrowed else []),
                "index_mib": measure.index_bytes(module) / MiB,
                "filter_mib": measure.index_bytes(getattr(module, "filter", None)) / MiB,
            }

        for q in combos:
            arms = [j for j in unit if id(j) in built and {**j.build, **q} in todo[id(j)]]
            if not arms:
                continue
            t0 = time.perf_counter()
            group = None
            if len(arms) > 1:
                assert by is not None
                shared = shared_key(arms[0], {**arms[0].build, **q}, by)
                gid = json.dumps(shared, sort_keys=True, default=str)
                group = {
                    "group": hashlib.sha1(gid.encode()).hexdigest()[:16],
                    "arms": [group_label(j, by) for j in arms],
                }
            recs_by: dict[int, dict[str, Any]] = {}
            for pos, job in enumerate(arms):
                b, assets, params = built[id(job)], assets_by[_akey(job)], {**job.build, **q}
                module, k_max = b["module"], max(job.ks)
                stage = "query_params"
                try:
                    resolved = resolve_pool(params, assets)
                    qp = {k: v for k, v in resolved.items() if k in QUERY_PARAMS}
                    if qp:
                        module.set_query_params(**qp)
                    member = None if group is None else {**group, "position": pos}
                    rec = _new_record(job, params, resolved, b, assets, inp, member)
                    src = (
                        None
                        if skip_quality
                        else cached_quality(sources[path], job, params, code_version)
                    )
                    if src is not None:
                        rec["quality"], rec["per_query"] = src["quality"], src["per_query"]
                        rec["quality_source"] = {"seed": src["seed"], "code_version": code_version}
                    elif not skip_quality:
                        stage = "quality"
                        extra = int(job.self_excluded)  # the own item, dropped in scoring
                        module.k = k_max + extra
                        qual, per_q, ids, scores = quality(module, inp, assets, job.ks, device)
                        if job.algo in PER_K_QUALITY:
                            for k in (k for k in job.ks if k != k_max):
                                module.k = int(k) + extra
                                qk, pk, _, _ = quality(module, inp, assets, [k], device)
                                for part, m in qk.items():
                                    if isinstance(m, dict):
                                        qual[part].update(m)
                                per_q.update(pk)
                        module.k = k_max
                        qual.update(parity(out_dir, job, params, ids, scores, job.ks))
                        if job.data.self_match:  # the held-out target is the query's own item
                            qual["heldout_label"] = "self_retrieval"
                        del ids, scores
                        rec["quality"] = qual
                        rec["per_query"] = per_query_path(job, params, code_version)
                        write_per_query(out_dir, rec["per_query"], assets, per_q)
                        if job.algo in EXACT_ALGOS and "oracle" in qual:
                            r = qual["oracle"][f"recall@{k_max}"]
                            if r is not None and r < EXACT_MIN_RECALL:  # None: no oracle row
                                raise QualityGateError(
                                    f"{job.algo}/{job.backend} recall_oracle@{k_max} = {r:.4f} "
                                    f"< {EXACT_MIN_RECALL}"
                                )
                    recs_by[id(job)] = rec
                except QualityGateError as exc:
                    records.append_record(path, _failed(job, params, env0, stage, exc, t0))
                    counts["failed"] += 1
                    raise
                except Exception as exc:  # recorded, the loop continues
                    msg = f"cell failed at {stage}: {job.key(params)}"
                    if _record_failures(path, [(job, params)], env0, stage, exc, t0, counts, msg):
                        raise
                if id(job) not in recs_by:
                    _release()

            samples_by: dict[int, list[dict[str, Any]]] = {}
            timed = [j for j in arms if id(j) in recs_by and j.timed and not skip_perf]
            if timed:
                try:
                    results = perf(
                        [built[id(j)]["module"] for j in timed],
                        inp,
                        assets_by[_akey(timed[0])],
                        timed[0],
                        device,
                        modes=modes,
                        profile=profile,
                        latency_kw=latency_kw,
                    )
                except Exception as exc:  # recorded, the loop continues
                    for j in timed:
                        recs_by.pop(id(j))
                    cells = [(j, {**j.build, **q}) for j in timed]
                    msg = f"perf failed: {[j.key(j.build) for j in timed]}"
                    if _record_failures(path, cells, env0, "perf", exc, t0, counts, msg):
                        raise
                    results = None
                if results is None:
                    _release()
                else:
                    for j, (entries, samples) in zip(timed, results, strict=True):
                        recs_by[id(j)]["perf"] = entries
                        samples_by[id(j)] = samples

            for job in arms:
                if id(job) not in recs_by:
                    continue
                rec = recs_by[id(job)]
                loads = [e["sm_mhz"] for e in rec["perf"] or [] if e.get("sm_mhz")]
                if loads and sm_ref is None:
                    sm_ref = loads[0]
                drift = any(abs(s - sm_ref) > CLOCK_DRIFT * sm_ref for s in loads)
                if drift:
                    logger.warning("under-load clocks.sm {} MHz vs first {} MHz", loads, sm_ref)
                rec["unstable"] = drift or any(e.get("unstable") for e in rec["perf"] or [])
                rec["memory_reserved_mib"] = (
                    torch.cuda.memory_reserved() / MiB if torch.cuda.is_available() else None
                )
                rec["elapsed_s"] = time.perf_counter() - t0
                win = [w for e in rec["perf"] or [] for w in e.get("window_sm_mhz") or [] if w]
                sm_windows += win
                sm_max = env0["sm_max_mhz"]  # the device's max SM clock: order-free
                rec["env"] = {
                    **env0,
                    "sm_mhz_load": statistics.median(loads) if loads else None,
                    "clocks_drift": drift,
                    "frac_windows_below_max": sum(w < sm_max for w in win) / len(win)
                    if win and sm_max
                    else None,
                }
                key = job.key({**job.build, **q})
                for smp in samples_by.get(id(job), []):  # samples first: never a record without
                    records.append_record(records.samples_path(path), {**key, **smp})
                records.append_record(path, rec)
                existing[path][records.resume_key(key, code_version)] = rec["status"]
                if rec["quality"] is not None and rec["quality_source"] is None:
                    qk = quality_key(key, code_version, job.ks)
                    sources[path].setdefault(qk, {})[job.seed] = rec
                counts[rec["status"]] += 1
                logger.info(
                    "  {} {}/{} {} params={} -> {} ({:.0f}s)",
                    job.algo, job.filter_kind, job.sweep, job.backend, rec["params"],
                    rec["status"], rec["elapsed_s"],
                )  # fmt: skip
        module = b = built = None  # the last arm's locals still hold its index
        _release()
        _assert_freed(base)
    counts = +counts  # drop zero entries
    logger.info(measure.clock_histogram(sm_windows))
    logger.info("done: {}", dict(counts))
    return counts
