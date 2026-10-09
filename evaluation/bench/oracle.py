"""Exact filtered oracle (H §2.2, §8.2 I).

Brute-force ``q @ E^T``, item chunk by item chunk (``ITEM_CHUNK`` rows, a running top-k
merged with equal scores to the lowest id; no ``[B, N]`` score or mask), under the *exact*
mask (``ExactAttributeFilter`` even on bloom cells — bloom's false positives must never leak
into ground truth), top-``k_max`` with
``-inf`` ties written as ``-1`` so short-filled rows never score against the algos' ``-1``
sentinel. Skipped rows (no live clause) stay all ``-1``. Indices are 0-indexed positions in
the pad-row-dropped ``item_embs``.

**Blob v4** — one dict at ``<gt_dir>/oracle_v4_<sweep>_<fingerprint[:16]>.pt``:

| key | value |
|---|---|
| ``version`` | ``4`` |
| ``topk`` | ``[U, k_gt]`` int64, ``-1`` padded |
| ``pass_counts`` | ``[U]`` int64: items passing the exact mask; ``-1`` on skipped rows |
| ``pass_rate`` | mean over kept rows of ``pass_counts / n_items`` (H §2.2, LiNR's axis) |
| ``targets_in_filter`` | ``[U, T]`` bool: held-out target ``t`` of user ``u`` passes the mask |
| ``target_in_filter`` | ``[U]`` bool: any held-out target passes (``n_queries_heldout = sum``) |
| ``n_items``, ``n_queries``, ``n_kept``, ``k_gt``, ``sweep``, ``clauses`` | the build's shape |
| ``fingerprint`` | the content hash below |
| ``code_version``, ``harness_commit``, ``torch``, ``created`` | provenance of the build |

The **fingerprint** hashes shapes, dtypes and a fixed 64-row linspace sample of
``item_embs``, ``queries``, ``targets`` and the sweep's ``qa``, the *full bytes* of the item
side of the predicate — ``item_attrs`` and ``clause_is_reverse``, via ``attrs_digest``
(``inputs.load_inputs`` computes it once per ``(dataset, dim)``; the ETL edits that tensor
piecemeal, so a row sample is not enough) — plus ``clauses`` and ``k_gt``. Same-shape content
changes — another dim off the same ``data_dir``, regenerated attrs, a retrained checkpoint, a
changed ``users_limit`` — change the file name, so a stale blob is never *read*; the hash in
the name is what makes the blob a portable artifact (§8.2 I). Bloom pass rates are not
cached: they depend on ``m_bits`` / ``k_hash`` and cost one mask pass (``pass_counts``).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from tqdm import tqdm

from bench.measure import _git, code_version
from eval_datasets.layout import atomic_write
from retrieve.interfaces import FilterModule

BLOB_VERSION = 4
_FP_SAMPLE_ROWS = 64
# Item rows per chunk: a [64, 2M] fp32 score block is 512 MB; no [B, N] tensor is ever formed.
ITEM_CHUNK = 2_000_000


def attrs_digest(item_attrs: torch.Tensor | None, clause_is_reverse: torch.Tensor | None) -> str:
    """sha256 over the full bytes (shape, dtype, content) of ``item_attrs`` and
    ``clause_is_reverse`` — the item side of the predicate. ≈ 1 s per GB; computed once per
    ``(dataset, dim)`` by ``inputs.load_inputs`` and passed down to ``fingerprint``."""
    h = hashlib.sha256()
    for t in (item_attrs, clause_is_reverse):
        if t is None:
            h.update(b"none")
            continue
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprint(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor | None,
    qa_sweep: torch.Tensor | None,
    clauses: tuple[int, ...] | None,
    k_gt: int,
    *,
    attrs_digest: str,
) -> str:
    """Deterministic content hash: shapes + dtypes + a fixed row sample of the four query-side
    tensors (<10 ms), the precomputed ``attrs_digest``, ``clauses`` and ``k_gt``."""
    h = hashlib.sha256()
    for t in (item_embs, queries, targets, qa_sweep):
        if t is None:
            h.update(b"none")
            continue
        h.update(repr((tuple(t.shape), str(t.dtype))).encode())
        n = t.shape[0]
        # float32 linspace rounds n - 1 up to n past 2^24 rows; the clamp keeps smaller fingerprints
        idx = torch.linspace(0, n - 1, steps=min(_FP_SAMPLE_ROWS, n)).long().clamp_(max=n - 1)
        h.update(t[idx].detach().float().cpu().contiguous().numpy().tobytes())
    h.update(attrs_digest.encode())
    h.update(repr((None if clauses is None else tuple(clauses), int(k_gt))).encode())
    return h.hexdigest()


def _batches(n_rows: int, skip_mask: torch.Tensor | None, batch_size: int, desc: str):
    keep = ~skip_mask if skip_mask is not None else torch.ones(n_rows, dtype=torch.bool)
    keep_idx = keep.nonzero(as_tuple=False).reshape(-1)
    rng = range(0, keep_idx.numel(), batch_size)
    for s in tqdm(rng, desc=desc, leave=False, disable=not sys.stderr.isatty()):
        yield keep_idx[s : s + batch_size]


def _chunks(n_items: int, item_chunk: int):
    return ((s, min(s + item_chunk, n_items)) for s in range(0, n_items, item_chunk))


@torch.inference_mode()
def pass_counts(
    filter_mod: FilterModule,
    qa_sweep: torch.Tensor,
    skip_mask: torch.Tensor | None,
    n_items: int,
    *,
    batch_size: int = 64,
    item_chunk: int = ITEM_CHUNK,
    device: torch.device,
) -> torch.Tensor:
    """``[U]`` int64 items passing ``filter_mod`` per query (``-1`` on skipped rows), summed
    over item chunks of ``evaluate_mask(qa, start, end)``. With a ``BloomFilter`` this is the
    bloom pass count (H §2.2 ``bloom_fp_rate``)."""
    out = torch.full((qa_sweep.shape[0],), -1, dtype=torch.long)
    for idx in _batches(qa_sweep.shape[0], skip_mask, batch_size, "pass_counts"):
        qa = filter_mod.prepare_queries(qa_sweep[idx].to(device, non_blocking=True))
        n = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        for start, end in _chunks(n_items, item_chunk):
            n += filter_mod.evaluate_mask(qa, start, end).sum(dim=1)
        out[idx] = n.cpu()
    return out


def pass_rate(counts: torch.Tensor, n_items: int) -> float:
    """Mean of ``counts / n_items`` over kept rows (``counts >= 0``); ``nan`` if none."""
    kept = counts[counts >= 0]
    return (kept.double() / n_items).mean().item() if kept.numel() else float("nan")


def bloom_fp_rate(bloom_counts: torch.Tensor, exact_counts: torch.Tensor, n_items: int) -> float:
    """Mean per-query false-positive rate ``(bloom − exact) / (N − exact)`` over kept rows
    with at least one negative; ``nan`` if none (bloom ⊇ exact, so this is ≥ 0)."""
    keep = (exact_counts >= 0) & (bloom_counts >= 0) & (exact_counts < n_items)
    fp = (bloom_counts[keep] - exact_counts[keep]).double()
    neg = (n_items - exact_counts[keep]).double()
    return (fp / neg).mean().item() if keep.any() else float("nan")


def _merge_topk(
    best_s: torch.Tensor, best_i: torch.Tensor, s: torch.Tensor, i: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The top-``k`` of two candidate lists, equal scores to the lowest id: a stable sort by
    id, then a stable sort by descending score."""
    cs, ci = torch.cat([best_s, s], dim=1), torch.cat([best_i, i], dim=1)
    by_id = ci.argsort(dim=1, stable=True)
    cs, ci = cs.gather(1, by_id), ci.gather(1, by_id)
    by_score = cs.argsort(dim=1, descending=True, stable=True)[:, :k]
    return cs.gather(1, by_score), ci.gather(1, by_score)


@torch.inference_mode()
def compute(
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    qa_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    filter_mod: FilterModule | None,
    k_gt: int,
    *,
    targets: torch.Tensor | None = None,
    batch_size: int = 64,
    item_chunk: int = ITEM_CHUNK,
    device: torch.device,
) -> dict[str, Any]:
    """The v4 content fields (``topk``, ``pass_counts``, ``targets_in_filter``,
    ``target_in_filter``, ``pass_rate``) for one sweep, item chunk by item chunk:
    ``q @ chunk.T`` in exact fp32, the filter's ``evaluate_mask(qa, start, end)``, a local
    top-k merged into the running ``[B, k]`` (ids offset by the chunk start, equal scores to
    the lowest id). No ``[B, N]`` score or mask and no transposed copy of ``item_embs``.
    ``filter_mod`` must be exact; with ``None`` (or no ``qa_sweep``) every item passes.
    ``targets`` is ``[U, T]`` ``-1``-padded."""
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("the oracle needs exact fp32 matmuls: TF32 is on (measure.setup)")
    n_users, n_items = queries.shape[0], int(item_embs.shape[0])
    k_eff = min(k_gt, n_items)
    topk = torch.full((n_users, k_gt), -1, dtype=torch.long)
    counts = torch.full((n_users,), -1, dtype=torch.long)
    t_shape = targets.shape if targets is not None else (n_users, 0)
    tif = torch.zeros(t_shape, dtype=torch.bool)
    filtered = filter_mod is not None and qa_sweep is not None
    for idx in _batches(n_users, skip_mask, batch_size, "oracle"):
        q = queries[idx].to(device, non_blocking=True)
        qa = (
            filter_mod.prepare_queries(qa_sweep[idx].to(device, non_blocking=True))
            if filtered
            else None
        )
        best_s = torch.full((idx.numel(), k_eff), float("-inf"), device=device)
        best_i = torch.full((idx.numel(), k_eff), -1, dtype=torch.long, device=device)
        n_pass = torch.zeros(idx.numel(), dtype=torch.long, device=device)
        t = targets[idx].to(device, non_blocking=True) if targets is not None else None
        passes = t != -1 if t is not None else None
        hit = torch.zeros_like(passes) if t is not None and filtered else None
        for start, end in _chunks(n_items, item_chunk):
            scores = q @ item_embs[start:end].t()
            if filtered:
                mask = filter_mod.evaluate_mask(qa, start, end)
                scores = scores.masked_fill(~mask, float("-inf"))
                n_pass += mask.sum(dim=1)
                if t is not None:
                    inside = (t >= start) & (t < end)
                    local = (t - start).clamp(0, end - start - 1)
                    hit |= inside & mask.gather(1, local)
            top = torch.topk(scores, min(k_eff, end - start), dim=1)
            best_s, best_i = _merge_topk(best_s, best_i, top.values, top.indices + start, k_eff)
        # Fewer than k_eff survivors: the tail is -inf and is written as -1.
        topk[idx, :k_eff] = torch.where(torch.isfinite(best_s), best_i, -1).cpu()
        counts[idx] = n_pass.cpu() if filtered else n_items
        if t is not None:
            tif[idx] = (passes & hit if filtered else passes).cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "topk": topk,
        "pass_counts": counts,
        "pass_rate": pass_rate(counts, n_items),
        "targets_in_filter": tif,
        "target_in_filter": tif.any(dim=1),
    }


def blob_path(gt_dir: Path, sweep: str, fp: str) -> Path:
    return gt_dir / f"oracle_v4_{sweep}_{fp[:16]}.pt"


def load_or_build(
    gt_dir: Path,
    sweep: str,
    k_gt: int,
    *,
    item_embs: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor | None,
    qa_sweep: torch.Tensor | None,
    skip_mask: torch.Tensor | None,
    clauses: tuple[int, ...] | None,
    filter_mod: FilterModule | None,
    attrs_digest: str,
    device: torch.device,
    warn_missing: bool = False,
) -> dict[str, Any]:
    """The v4 blob for one sweep: read from ``blob_path`` when present, else build and
    save (atomically). The name carries the fingerprint, so a stale blob is simply never
    found; an unreadable file at the path (a crash mid-save) is rebuilt, not fatal.
    ``warn_missing``: the caller expected ``bench oracle`` to have built it (``bench run``)."""
    fp = fingerprint(
        item_embs, queries, targets, qa_sweep, clauses, k_gt, attrs_digest=attrs_digest
    )
    path = blob_path(gt_dir, sweep, fp)
    if path.exists():
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 — a torn / foreign file at the cache path
            logger.warning("  {}: unreadable ({}: {}); rebuilding", path, type(exc).__name__, exc)
            blob = None
        if (
            isinstance(blob, dict)
            and blob.get("version") == BLOB_VERSION
            and blob.get("fingerprint") == fp
        ):
            logger.info("  loaded oracle from cache: {}", path)
            return blob
        if blob is not None:
            logger.warning("  {}: not a v4 blob for this fingerprint; rebuilding", path)
    if warn_missing:
        logger.warning("  oracle {} missing: building it here; prebuild with `bench oracle`", path)
    logger.info("  building exact oracle (k_gt={}) for sweep {}", k_gt, sweep)
    n_kept = int((~skip_mask).sum()) if skip_mask is not None else int(queries.shape[0])
    blob = {
        "version": BLOB_VERSION,
        **compute(
            item_embs,
            queries,
            qa_sweep,
            skip_mask,
            filter_mod,
            k_gt,
            targets=targets,
            device=device,
        ),
        "n_items": int(item_embs.shape[0]),
        "n_queries": int(queries.shape[0]),
        "n_kept": n_kept,
        "k_gt": int(k_gt),
        "sweep": sweep,
        "clauses": None if clauses is None else [int(c) for c in clauses],
        "fingerprint": fp,
        "code_version": code_version(),
        "harness_commit": _git("rev-parse", "--short", "HEAD") or "unknown",
        "torch": str(torch.__version__),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write(path, lambda fh: torch.save(blob, fh))
    logger.info("  saved oracle → {} (pass_rate={:.4f})", path, blob["pass_rate"])
    return blob


__all__ = [
    "BLOB_VERSION",
    "ITEM_CHUNK",
    "attrs_digest",
    "blob_path",
    "bloom_fp_rate",
    "compute",
    "fingerprint",
    "load_or_build",
    "pass_counts",
    "pass_rate",
]
