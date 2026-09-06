#!/usr/bin/env python3
"""A3 / O WP-1 — measured host-side facts about the official SilverTorch ops.

Answers, on the real A100 with the pinned build, the questions
[`../silvertorch-official-integration.md`](../silvertorch-official-integration.md)
§3 and §4 answer by *reading* Meta's C++:

  1. bit order of the official bloom mask (the trap in O §4.3; B1's adapter
     needs the answer before it can write `pack_mask_high_first`)
  2. host syncs per op, under ``torch.cuda.set_sync_debug_mode("warn")``
  3. kernel launches (and prep work) per op, from ``torch.profiler``
  4. ``torch.cuda.CUDAGraph`` capture per op — the evidence behind O D7
     ("`official` runs eager only"), including what a *replay* does
  5. expression parse cost per query at B=16
  6. the ``per_embedding_scale`` overflow of O §4.2 (iii)

Run (from the repo root, with the `official` extra installed):

    UV_LINK_MODE=copy uv run --extra official python \\
        docs/plans/official-silvertorch-artifacts/official_facts.py \\
        --out docs/plans/official-silvertorch-artifacts

Writes ``official_facts.json`` (machine-readable) and ``official_facts.txt``
(the same thing as a report) into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch

import silvertorch  # noqa: F401  (import order: package, then op registration)
import silvertorch.ops._load_ops  # noqa: F401  registers torch.ops.st.*

DEV = "cuda"
SEED = 0

# ---------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    """A cluster-sorted IVF the way O §4.1 maps our padded layout onto theirs."""

    n_docs: int
    dim: int
    n_lists: int
    n_probe: int
    batch: int
    cluster_size: int

    cluster_offsets: torch.Tensor = field(init=False)  # [n_lists+1] int64
    cluster_ids: torch.Tensor = field(init=False)  # [B, P] int64 (probe ids)
    cluster_length: torch.Tensor = field(init=False)  # [B, P] int64
    embeddings: torch.Tensor = field(init=False)  # [N, D] int8, cluster-sorted
    queries: torch.Tensor = field(init=False)  # [B, D] int8
    max_size: int = field(init=False)

    # bloom side
    feature_ids: torch.Tensor = field(init=False)  # [C] int32
    feature_offsets: torch.Tensor = field(init=False)  # [N*C+1] int64
    feature_values: torch.Tensor = field(init=False)  # [nnz] int64
    doc_values: list[tuple[int, int]] = field(init=False)  # per doc, (v0, v1)
    expressions: list[str] = field(init=False)

    def __post_init__(self) -> None:
        g = torch.Generator().manual_seed(SEED)
        n, d = self.n_docs, self.dim
        assert n == self.n_lists * self.cluster_size

        self.cluster_offsets = torch.arange(
            0, n + 1, self.cluster_size, dtype=torch.long, device=DEV
        )
        self.cluster_ids = torch.randint(
            0, self.n_lists, (self.batch, self.n_probe), generator=g, dtype=torch.long
        ).to(DEV)
        self.cluster_length = torch.full(
            (self.batch, self.n_probe), self.cluster_size, dtype=torch.long, device=DEV
        )
        self.embeddings = torch.randint(
            -127, 128, (n, d), generator=g, dtype=torch.int8
        ).to(DEV)
        self.queries = torch.randint(
            -127, 128, (self.batch, d), generator=g, dtype=torch.int8
        ).to(DEV)
        self.max_size = self.n_probe * self.cluster_size

        # two single-valued clauses per doc, the shape our narrow attrs tensor
        # has (clause index == feature id, O §4.4)
        v0 = torch.randint(100, 150, (n,), generator=g).tolist()
        v1 = torch.randint(200, 250, (n,), generator=g).tolist()
        self.doc_values = list(zip(v0, v1))
        self.feature_ids = torch.tensor([0, 1], dtype=torch.int32)
        self.feature_offsets = torch.arange(0, 2 * n + 1, dtype=torch.long)
        vals: list[int] = []
        for a, b in self.doc_values:
            vals.extend((a, b))
        self.feature_values = torch.tensor(vals, dtype=torch.long)
        self.expressions = [
            f"0:{self.doc_values[i][0]} AND 1:{self.doc_values[i][1]}"
            for i in range(self.batch)
        ]

    def selected_cluster(self) -> tuple[torch.Tensor, torch.Tensor]:
        offs = self.cluster_offsets[:-1][self.cluster_ids]  # [B, P]
        return offs.contiguous(), self.cluster_length.contiguous()


def build_bloom(c: Corpus, *, b_multiplier: float, hash_k: int):
    return torch.ops.st.bloom_index_build(
        c.feature_ids, c.feature_offsets, c.feature_values, b_multiplier, hash_k
    )


def parse_plans(expressions: list[str], hash_k: int):
    ks = torch.ones(1, dtype=torch.long)
    _, tensors = torch.ops.st.parse_expression_query_batch(
        expressions, ks, hash_k, True, 5
    )
    return tensors[0], tensors[1]


# ---------------------------------------------------------------------------
# 1. bit order
# ---------------------------------------------------------------------------


def probe_bit_order(c: Corpus, k: int, hash_k: int) -> dict[str, Any]:
    """Three independent reads of the same question.

    A. `bloom_index_search_batch` bool mask vs packed int64 — which bit of the
       word carries doc 0?
    B. a hand-built `filtering_bit_mask` with exactly one bit set, fed to
       `fused_kmean_ann` — which document survives?
    C. round trip: the packed output of A, used as B's mask, must select
       exactly the documents A's bool mask marks.
    """
    out: dict[str, Any] = {}
    bloom_index, b_offsets = build_bloom(c, b_multiplier=8.0, hash_k=hash_k)
    bloom_index, b_offsets = bloom_index.to(DEV), b_offsets.to(DEV)

    expr = [f"0:{c.doc_values[0][0]} AND 1:{c.doc_values[0][1]}"]
    plans_data, plans_off = parse_plans(expr, hash_k)

    m_bool = torch.ops.st.bloom_index_search_batch(
        bloom_index, b_offsets, plans_data, plans_off, k, hash_k, True
    )
    m_pack = torch.ops.st.bloom_index_search_batch(
        bloom_index, b_offsets, plans_data, plans_off, k, hash_k, False
    )
    passing = torch.nonzero(m_bool[0]).flatten().tolist()
    uword0 = int(m_pack[0, 0].item()) & 0xFFFFFFFFFFFFFFFF
    first64 = [p for p in passing if p < 64]
    out["A_bool_shape"] = list(m_bool.shape)
    out["A_bool_dtype"] = str(m_bool.dtype)
    out["A_packed_shape"] = list(m_pack.shape)
    out["A_packed_dtype"] = str(m_pack.dtype)
    out["A_passing_docs_in_first_64"] = first64
    out["A_first_word_hex"] = f"0x{uword0:016x}"
    out["A_set_bit_indices_lsb0"] = [b for b in range(64) if (uword0 >> b) & 1]
    out["A_doc_to_bit"] = {str(d): 63 - d for d in first64}
    high_first = bool(first64) and all(((uword0 >> (63 - d)) & 1) == 1 for d in first64)
    low_first = bool(first64) and all(((uword0 >> d) & 1) == 1 for d in first64)
    out["A_verdict"] = (
        "HIGH-bit-first (doc d -> bit 63-d)"
        if high_first and not low_first
        else "LOW-bit-first (doc d -> bit d)"
        if low_first and not high_first
        else "ambiguous"
    )

    # -- B: single-bit filtering_bit_mask into the scorer ------------------
    n_words = max((c.max_size + 63) // 64, 1)
    one_ids = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    one_len = torch.full((1, 1), 64, dtype=torch.long, device=DEV)
    q = c.queries[:1]
    b_results: dict[str, list[int]] = {}
    for name, word in (("bit63_set", 1 << 63), ("bit0_set", 1)):
        mask = torch.zeros((1, n_words), dtype=torch.long, device=DEV)
        # torch has no unsigned int64; 1<<63 is INT64_MIN as a signed pattern
        mask[0, 0] = torch.tensor(word, dtype=torch.uint64).view(torch.int64).item()
        _, idx = torch.ops.st.fused_kmean_ann(
            c.cluster_offsets, one_ids, one_len, c.embeddings, q, 64, mask, -1, -1
        )
        b_results[name] = [int(v) for v in idx[0][idx[0] >= 0].tolist()][:8]
    out["B_kept_doc_ids"] = b_results
    out["B_verdict"] = (
        "HIGH-bit-first: bit 63 selects doc 0"
        if b_results["bit63_set"] == [0]
        else "LOW-bit-first: bit 0 selects doc 0"
        if b_results["bit0_set"] == [0]
        else f"ambiguous: {b_results}"
    )

    # -- C: round trip the packed bloom output through the scorer ----------
    cs = c.cluster_size
    mask_c = m_pack[:1, : (cs + 63) // 64].contiguous()
    _, idx = torch.ops.st.fused_kmean_ann(
        c.cluster_offsets,
        torch.zeros((1, 1), dtype=torch.long, device=DEV),
        torch.full((1, 1), cs, dtype=torch.long, device=DEV),
        c.embeddings,
        q,
        cs,
        mask_c,
        -1,
        -1,
    )
    kept = sorted(int(v) for v in idx[0][idx[0] >= 0].tolist())
    expect = sorted(p for p in passing if p < cs)
    out["C_scorer_kept"] = kept[:16]
    out["C_bloom_bool_passing"] = expect[:16]
    out["C_roundtrip_equal"] = kept == expect
    return out


# ---------------------------------------------------------------------------
# 2. syncs
# ---------------------------------------------------------------------------


@contextmanager
def sync_counter():
    """Counts host syncs reported by ``set_sync_debug_mode("warn")``.

    The obvious instrument — ``warnings.catch_warnings(record=True)`` — reads
    **zero** syncs for every ``torch.ops.st.*`` call, and that is an artefact,
    not a fact: a ``TORCH_WARN`` raised inside a C++ custom op is handled by
    c10's own warning handler and printed to fd 2 by the ``[W...]`` logger,
    never converted into a Python warning. So capture fd 2 instead and count
    ``warn_or_error_on_sync`` lines. (Verified against a ``t.item()`` control,
    which *is* visible to both instruments.)
    """
    import os
    import tempfile

    msgs: list[str] = []
    prev = torch.cuda.get_sync_debug_mode()
    torch.cuda.synchronize()
    with tempfile.TemporaryFile(mode="w+") as tf:
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tf.fileno(), 2)
        try:
            torch.cuda.set_sync_debug_mode("warn")
            yield msgs
        finally:
            torch.cuda.set_sync_debug_mode(prev)
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            tf.seek(0)
            msgs.extend(ln.strip()[:200] for ln in tf if "warn_or_error_on_sync" in ln)


# ---------------------------------------------------------------------------
# 3. launches
# ---------------------------------------------------------------------------


def count_launches(fn: Callable[[], Any], *, warmup: int = 3) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels: dict[str, int] = {}
    memcpy: dict[str, int] = {}
    memset = 0
    aten_ops = 0
    for ev in prof.events():
        name = ev.name
        if "CUDA" not in str(getattr(ev, "device_type", "")):
            if name.startswith("aten::"):
                aten_ops += 1
            continue
        if name.startswith("Memcpy"):
            memcpy[name] = memcpy.get(name, 0) + 1
        elif name.startswith("Memset"):
            memset += 1
        else:
            kernels[name] = kernels.get(name, 0) + 1
    return {
        "kernel_launches": sum(kernels.values()),
        "distinct_kernels": len(kernels),
        "kernels": dict(sorted(kernels.items(), key=lambda kv: -kv[1])),
        "memcpy_by_kind": dict(sorted(memcpy.items())),
        "memcpy_d2h": sum(n for k, n in memcpy.items() if "DtoH" in k),
        "memcpy_h2d": sum(n for k, n in memcpy.items() if "HtoD" in k),
        "device_memset": memset,
        "aten_ops_dispatched": aten_ops,
    }


# ---------------------------------------------------------------------------
# 4. cuda graph capture
# ---------------------------------------------------------------------------


def try_capture(fn: Callable[[], Any]) -> dict[str, Any]:
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            try:
                fn()
            except Exception:  # noqa: BLE001
                break
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            fn()
        torch.cuda.synchronize()
        return {"capturable": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).strip().splitlines()
        return {
            "capturable": False,
            "error_type": type(exc).__name__,
            "error": " / ".join(m.strip() for m in msg[:3])[:400],
        }


def probe_graph_replay(c: Corpus, k: int, hash_k: int) -> dict[str, Any]:
    """For the ops that *do* capture, is a replay actually correct?

    ``bloom_index_search_batch`` decodes its query plans on the host and
    uploads them with a fresh ``cudaMemcpyAsync`` every call (O §3). A replay
    cannot re-run host code, so a captured graph can only ever reproduce the
    plan that was live at capture time. This probe makes that concrete:
    capture with predicate A, overwrite the host plan buffer in place with
    predicate B, replay, and compare against both eager answers.
    """
    out: dict[str, Any] = {}
    bloom_index, b_offsets = build_bloom(c, b_multiplier=8.0, hash_k=hash_k)
    bloom_index, b_offsets = bloom_index.to(DEV), b_offsets.to(DEV)

    va, wa = c.doc_values[0]
    vb, wb = c.doc_values[1]
    if (va, wa) == (vb, wb):  # pragma: no cover - the fixed seed avoids this
        return {"skipped": "documents 0 and 1 share a predicate"}
    pa_data, pa_off = parse_plans([f"0:{va} AND 1:{wa}"], hash_k)
    pb_data, pb_off = parse_plans([f"0:{vb} AND 1:{wb}"], hash_k)
    out["plan_shapes_match"] = list(pa_data.shape) == list(pb_data.shape) and list(
        pa_off.shape
    ) == list(pb_off.shape)
    if not out["plan_shapes_match"]:
        return out

    def search(pd, po):
        return torch.ops.st.bloom_index_search_batch(
            bloom_index, b_offsets, pd, po, k, hash_k, True
        )

    eager_a = search(pa_data, pa_off).clone()
    eager_b = search(pb_data, pb_off).clone()
    out["eager_a_passing"] = torch.nonzero(eager_a[0]).flatten()[:8].tolist()
    out["eager_b_passing"] = torch.nonzero(eager_b[0]).flatten()[:8].tolist()
    out["predicates_differ"] = not bool(torch.equal(eager_a, eager_b))

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            search(pa_data, pa_off)
    torch.cuda.current_stream().wait_stream(st)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            captured = search(pa_data, pa_off)
    except Exception as exc:  # noqa: BLE001
        out["captured"] = False
        out["error"] = str(exc).strip().splitlines()[0][:200]
        return out
    out["captured"] = True

    pa_data.copy_(pb_data)
    pa_off.copy_(pb_off)
    try:
        g.replay()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        out["replay_error_type"] = type(exc).__name__
        out["replay_error"] = str(exc).strip().splitlines()[0][:200]
        out["verdict"] = (
            "captures, but replaying the captured graph raises "
            f"{out['replay_error_type']} — the capture is unusable"
        )
        return out
    out["replay_equals_eager_a_stale_plan"] = bool(torch.equal(captured, eager_a))
    out["replay_equals_eager_b_new_plan"] = bool(torch.equal(captured, eager_b))
    out["verdict"] = (
        "captures, but replay silently returns the capture-time plan's answer "
        "— the host plan decode + upload is not in the graph"
        if out["replay_equals_eager_a_stale_plan"]
        and not out["replay_equals_eager_b_new_plan"]
        else "replay tracked the new plan (unexpected)"
        if out["replay_equals_eager_b_new_plan"]
        else "replay produced neither eager answer"
    )
    return out


# ---------------------------------------------------------------------------
# 5 + 6
# ---------------------------------------------------------------------------


def op_table(c: Corpus, k: int, hash_k: int) -> dict[str, Callable[[], Any]]:
    bloom_index, b_offsets = build_bloom(c, b_multiplier=8.0, hash_k=hash_k)
    bloom_index, b_offsets = bloom_index.to(DEV), b_offsets.to(DEV)
    plans_data, plans_off = parse_plans(c.expressions, hash_k)  # kept on CPU (O §3)
    sel_off, sel_len = c.selected_cluster()

    ccc, fio, cmr = torch.ops.st.bloom_index_search_batch_return_partial_response(
        bloom_index, b_offsets, plans_data, plans_off, sel_off, sel_len, k, hash_k
    )
    full_mask = torch.ops.st.bloom_index_search_batch(
        bloom_index, b_offsets, plans_data, plans_off, k, hash_k, False
    )

    def _fused_plain():
        return torch.ops.st.fused_kmean_ann(
            c.cluster_offsets,
            c.cluster_ids,
            c.cluster_length,
            c.embeddings,
            c.queries,
            c.max_size,
            None,
            -1,
            -1,
        )

    def _fused_full_mask():
        return torch.ops.st.fused_kmean_ann(
            c.cluster_offsets,
            c.cluster_ids,
            c.cluster_length,
            c.embeddings,
            c.queries,
            c.max_size,
            full_mask,
            -1,
            -1,
        )

    def _fused_partial():
        return torch.ops.st.fused_kmean_ann_with_partial_masks(
            c.cluster_offsets,
            c.cluster_ids,
            c.cluster_length,
            c.embeddings,
            c.queries,
            c.max_size,
            ccc,
            fio,
            cmr,
            -1,
            -1,
        )

    def _bloom_full():
        return torch.ops.st.bloom_index_search_batch(
            bloom_index, b_offsets, plans_data, plans_off, k, hash_k, False
        )

    def _bloom_partial():
        return torch.ops.st.bloom_index_search_batch_return_partial_response(
            bloom_index, b_offsets, plans_data, plans_off, sel_off, sel_len, k, hash_k
        )

    def _colinfo():
        return torch.ops.st.generate_column_info_for_clusters(sel_off, sel_len)

    def _build():
        return torch.ops.st.bloom_index_build(
            c.feature_ids.to(DEV),
            c.feature_offsets.to(DEV),
            c.feature_values.to(DEV),
            8.0,
            hash_k,
        )

    return {
        "fused_kmean_ann (no filter)": _fused_plain,
        "fused_kmean_ann (full mask)": _fused_full_mask,
        "fused_kmean_ann_with_partial_masks": _fused_partial,
        "bloom_index_search_batch (packed, full N)": _bloom_full,
        "bloom_index_search_batch_return_partial_response": _bloom_partial,
        "generate_column_info_for_clusters": _colinfo,
        "bloom_index_build (CUDA)": _build,
    }


def probe_parse_cost(c: Corpus, hash_k: int) -> dict[str, Any]:
    import torch.utils.benchmark as tb

    ks = torch.ones(1, dtype=torch.long)
    results: dict[str, Any] = {}
    for label, exprs in (
        ("B=16, 2 clauses AND", c.expressions),
        ("B=16, 1 clause", [f"0:{c.doc_values[i][0]}" for i in range(c.batch)]),
        (
            "B=16, 2 clauses + NOT",
            [
                f"0:{c.doc_values[i][0]} AND NOT 1:{c.doc_values[i][1]}"
                for i in range(c.batch)
            ],
        ),
        ("B=1, 2 clauses AND", c.expressions[:1]),
    ):
        t = tb.Timer(
            stmt="torch.ops.st.parse_expression_query_batch(e, ks, hk, True, 5)",
            globals={"torch": torch, "e": exprs, "ks": ks, "hk": hash_k},
        )
        m = t.blocked_autorange(min_run_time=1.0)
        results[label] = {
            "n_expressions": len(exprs),
            "median_us_per_call": m.median * 1e6,
            "us_per_query": m.median * 1e6 / len(exprs),
            "iqr_us": m.iqr * 1e6,
        }
    return results


def probe_per_embedding_scale(c: Corpus) -> dict[str, Any]:
    """O §4.2 (iii): the kernel casts the int32 dot to fp16 *before* dividing."""
    out: dict[str, Any] = {}
    n, d = c.n_docs, c.dim
    emb = torch.full((n, d), 127, dtype=torch.int8, device=DEV)
    q = torch.full((1, d), 127, dtype=torch.int8, device=DEV)
    raw = 127 * 127 * d
    out["dim"] = d
    out["raw_dot"] = raw
    out["fp16_max"] = 65504.0

    one_id = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    one_len = torch.full((1, 1), c.cluster_size, dtype=torch.long, device=DEV)

    scale = torch.ones(n, dtype=torch.float16, device=DEV)
    s_scale, _ = torch.ops.st.fused_kmean_ann(
        c.cluster_offsets,
        one_id,
        one_len,
        emb,
        q,
        c.cluster_size,
        None,
        -1,
        -1,
        None,
        scale,
    )
    finite = torch.isfinite(s_scale.float())
    out["per_embedding_scale"] = {
        "dtype": str(s_scale.dtype),
        "first_8": [float(v) for v in s_scale[0, :8].float().tolist()],
        "n_inf": int((~finite).sum().item()),
        "n_total": int(finite.numel()),
        "all_inf": bool((~finite).all().item()),
    }

    s_i32, _ = torch.ops.st.fused_kmean_ann(
        c.cluster_offsets, one_id, one_len, emb, q, c.cluster_size, None, -1, -1
    )
    out["divisor_minus_1_int32"] = {
        "dtype": str(s_i32.dtype),
        "first_8": [int(v) for v in s_i32[0, :8].tolist()],
        "equals_raw_dot": bool(int(s_i32[0, 0].item()) == raw),
    }

    div = 64
    s_div, _ = torch.ops.st.fused_kmean_ann(
        c.cluster_offsets, one_id, one_len, emb, q, c.cluster_size, None, -1, div
    )
    out[f"divisor_for_int8={div}"] = {
        "dtype": str(s_div.dtype),
        "first_8": [float(v) for v in s_div[0, :8].float().tolist()],
        "all_finite": bool(torch.isfinite(s_div.float()).all().item()),
        "expected": raw / div,
    }
    out["smallest_saturated_dim_that_overflows_fp16"] = next(
        (dd for dd in range(1, 64) if 127 * 127 * dd > 65504), None
    )
    return out


# ---------------------------------------------------------------------------


def run_isolated(args, probe: str, op_index: int) -> dict[str, Any]:
    """Re-invoke this script in a fresh process for one graph probe.

    Necessary, not fastidious: a capture attempt that dies with
    ``cudaErrorStreamCaptureInvalidated`` leaves the context in a state where
    the *next* unrelated synchronize raises ``cudaErrorIllegalAddress``, so
    probing seven ops in one process reports the first failure and then junk.
    """
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe", probe,
        "--op-index", str(op_index),
        "--dim", str(args.dim),
        "--batch", str(args.batch),
        "--n-lists", str(args.n_lists),
        "--n-probe", str(args.n_probe),
        "--cluster-size", str(args.cluster_size),
        "--k", str(args.k),
        "--hash-k", str(args.hash_k),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    return {
        "capturable": False,
        "error_type": "SubprocessFailed",
        "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:300],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).parent)
    # A failed capture attempt leaves the CUDA context unusable, so `all`
    # re-invokes this script once per graph probe in a fresh process.
    ap.add_argument("--probe", default="all", choices=["all", "graph", "graph-replay"])
    ap.add_argument("--op-index", type=int, default=-1)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-lists", type=int, default=64)
    ap.add_argument("--n-probe", type=int, default=8)
    ap.add_argument("--cluster-size", type=int, default=256)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--hash-k", type=int, default=7)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("A3 needs a CUDA device (every official-side check runs on GPU, O §11).")
        return 2
    torch.manual_seed(SEED)

    def corpus() -> Corpus:
        return Corpus(
            n_docs=args.n_lists * args.cluster_size,
            dim=args.dim,
            n_lists=args.n_lists,
            n_probe=args.n_probe,
            batch=args.batch,
            cluster_size=args.cluster_size,
        )

    if args.probe != "all":
        cc = corpus()
        if args.probe == "graph-replay":
            res: dict[str, Any] = probe_graph_replay(cc, args.k, args.hash_k)
        else:
            fn = list(op_table(cc, args.k, args.hash_k).values())[args.op_index]
            res = try_capture(fn)
        print("__JSON__" + json.dumps(res))
        return 0

    c = corpus()
    report: dict[str, Any] = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "silvertorch_rev": "21aa35e28b6dd9a91e9ee35efb0857715e86bda7",
            "config": {
                "N": c.n_docs,
                "D": c.dim,
                "n_lists": c.n_lists,
                "n_probe": c.n_probe,
                "B": c.batch,
                "cluster_size": c.cluster_size,
                "P": c.max_size,
                "k": args.k,
                "hash_k": args.hash_k,
            },
        }
    }

    print("[1/6] bit order")
    report["bit_order"] = probe_bit_order(c, args.k, args.hash_k)

    ops = op_table(c, args.k, args.hash_k)

    print("[2/6] host syncs per op")
    syncs: dict[str, Any] = {}
    for name, fn in ops.items():
        fn()
        torch.cuda.synchronize()
        with sync_counter() as msgs:
            fn()
        syncs[name] = {"count": len(msgs), "messages": msgs[:6]}
    report["syncs"] = syncs

    print("[3/6] kernel launches per op")
    report["launches"] = {name: count_launches(fn) for name, fn in ops.items()}

    print("[4/6] CUDA graph capture per op (one subprocess per op)")
    report["graph_capture"] = {
        name: run_isolated(args, "graph", i) for i, name in enumerate(ops)
    }
    report["graph_replay_semantics"] = run_isolated(args, "graph-replay", -1)

    print("[5/6] expression parse cost")
    report["parse_cost"] = probe_parse_cost(c, args.hash_k)

    print("[6/6] per_embedding_scale overflow")
    report["per_embedding_scale"] = probe_per_embedding_scale(c)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "official_facts.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "official_facts.txt").write_text(render(report))
    print(render(report))
    return 0


def render(r: dict[str, Any]) -> str:
    L: list[str] = []
    m = r["meta"]
    L.append("A3 / O WP-1 — measured facts about the official SilverTorch ops")
    L.append(
        f"generated {m['generated']}  |  {m['gpu']}  |  torch {m['torch']} "
        f"(cuda {m['torch_cuda']})  |  python {m['python']}"
    )
    L.append(f"silvertorch @ {m['silvertorch_rev']}")
    L.append(f"config: {m['config']}")
    L.append("")

    b = r["bit_order"]
    L.append("== 1. bloom mask bit order ==================================")
    L.append(
        f"A  packed word 0 = {b['A_first_word_hex']}, "
        f"docs passing in [0,64) = {b['A_passing_docs_in_first_64']}"
    )
    L.append(f"   set bit indices (LSB=0) = {b['A_set_bit_indices_lsb0']}")
    L.append(f"   -> {b['A_verdict']}")
    L.append(f"B  single-bit filtering_bit_mask into fused_kmean_ann: {b['B_kept_doc_ids']}")
    L.append(f"   -> {b['B_verdict']}")
    L.append(f"C  round trip packed bloom -> scorer: kept {b['C_scorer_kept']}")
    L.append(f"   bool-mask passing        : {b['C_bloom_bool_passing']}")
    L.append(f"   -> equal: {b['C_roundtrip_equal']}")
    L.append("")

    L.append("== 2. host syncs per op (set_sync_debug_mode('warn'), fd-2 capture) ==")
    for k, v in r["syncs"].items():
        L.append(f"  {k:<52} {v['count']}")
    L.append("")

    L.append("== 3. kernel launches per op (torch.profiler) ===============")
    for k, v in r["launches"].items():
        L.append(
            f"  {k:<52} {v['kernel_launches']} launches, "
            f"{v['distinct_kernels']} distinct, "
            f"{v['memcpy_d2h']} D2H, {v['memcpy_h2d']} H2D, "
            f"{v['device_memset']} memset, {v['aten_ops_dispatched']} aten ops"
        )
        for kn, n in v["kernels"].items():
            L.append(f"      {n:>3}x {kn[:110]}")
    L.append("")

    L.append("== 4. CUDA graph capture ====================================")
    for k, v in r["graph_capture"].items():
        if v.get("capturable"):
            L.append(f"  {k:<52} CAPTURED")
        else:
            L.append(f"  {k:<52} FAILED  {v.get('error_type')}")
            L.append(f"      {v.get('error')}")
    gr = r.get("graph_replay_semantics", {})
    if gr:
        L.append("  replay semantics for bloom_index_search_batch:")
        for kk, vv in gr.items():
            L.append(f"      {kk} = {vv}")
    L.append("")

    L.append("== 5. expression parse cost (CPU) ===========================")
    for k, v in r["parse_cost"].items():
        L.append(
            f"  {k:<24} {v['median_us_per_call']:8.1f} us/call  "
            f"{v['us_per_query']:7.2f} us/query  (IQR {v['iqr_us']:.1f} us)"
        )
    L.append("")

    p = r["per_embedding_scale"]
    L.append("== 6. per_embedding_scale overflow (O 4.2 iii) ==============")
    L.append(
        f"  D={p['dim']}, saturated codes -> raw dot = {p['raw_dot']} "
        f"(fp16 max {p['fp16_max']})"
    )
    for key in ("per_embedding_scale", "divisor_minus_1_int32", "divisor_for_int8=64"):
        L.append(f"  {key}: {p[key]}")
    L.append(
        f"  smallest saturated D that overflows fp16: "
        f"{p['smallest_saturated_dim_that_overflows_fp16']}"
    )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
