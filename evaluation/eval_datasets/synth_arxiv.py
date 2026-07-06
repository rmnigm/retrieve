#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.4",
#   "tqdm>=4.66",
# ]
# ///
"""Synthetic-scale arxiv catalog generator.

Reads an existing encoded arxiv dataset directory and writes a drop-in
sibling at arbitrary N (15M / 30M / 50M etc) by SLERPing between
same-cluster parents. Recall is measured against the exact-kNN oracle
on the same catalog, so synthesis is unconstrained — no anti-shadowing.

Subcommands::

    cluster    Spherical k-means on the source ``text_emb.pt``; cache
               ``clusters.pt`` into the source content subdir. Reused
               across all target-N runs from the same source variant.
    synth      Generate sharded ``text_emb_shard_*.pt`` +
               ``item_attrs_narrow.pt``; copy vocabs / queries / heldout /
               eval_split into <output-dir>. Wide attrs are not synthesized
               (the arxiv filter bench is narrow-only).
    all        ``cluster`` + ``synth``.

Examples::

    uv run synth_arxiv all \\
        --source-dir data/arxiv/papers \\
        --output-dir data/arxiv-synth-15m \\
        --target-n 15000000 \\
        --source-content-subdir content_d128

Layout of an output directory (drop-in for the existing arxiv loader)::

    <output-dir>/
      content/
        text_emb_shard_00000.pt        # [shard_size, D] fp16
        text_emb_shard_00001.pt
        ...
        shard_index.json
        text_emb.meta.json             # copied from source content dir
        query_emb.pt                   # copied verbatim
        query_emb.meta.json
      item_attrs_narrow.pt             # [N_target+1, 5, 4] int64
      heldout.parquet                  # copied verbatim
      eval_split.parquet               # copied verbatim
      cat_main_vocab.json              # copied verbatim
      author_vocab.json
      license_vocab.json
      clause_is_reverse_narrow.pt
      item_id_map.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

C_NARROW = 5
A_MAX_NARROW = 4

# Hard cap. Above this the attribute tensors blow past comfortable host RAM
# and the existing single-file oracle path cannot serve them on a single GPU.
# See docs/plans/linr-int8-quantization.md for the path needed at 1B.
MAX_TARGET_N = 200_000_000

# Per-iteration chunk for the assignment matmul in spherical k-means.
KMEANS_ASSIGN_CHUNK = 100_000

# Per-call chunk for synth generation (caps GPU peak during SLERP).
SYNTH_SUBCHUNK = 1_000_000


# ----- spherical k-means ----------------------------------------------------


def _spherical_kmeans(
    items: torch.Tensor,
    k: int,
    *,
    n_iter: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Spherical k-means on L2-normalized items.

    Returns ``(centroids [K, D] fp32 on device, assignment [N] int32 on device)``.
    Centroid update is L2-renormalize-after-sum (Euclidean midpoint projected
    back to the sphere — the standard spherical k-means update).
    """
    n, d = items.shape
    g = torch.Generator(device=device).manual_seed(seed)
    init_idx = torch.randperm(n, generator=g, device=device)[:k]
    centroids = F.normalize(items[init_idx].to(torch.float32), dim=-1).contiguous()

    assignment = torch.empty(n, dtype=torch.int32, device=device)
    ones = torch.ones(n, dtype=torch.float32, device=device)

    for it in range(n_iter):
        c_t = centroids.t().contiguous()
        for s in range(0, n, KMEANS_ASSIGN_CHUNK):
            e = min(s + KMEANS_ASSIGN_CHUNK, n)
            sim = items[s:e].to(torch.float32) @ c_t
            assignment[s:e] = sim.argmax(dim=1).to(torch.int32)

        index_2d = assignment.long().unsqueeze(-1).expand(-1, d)
        new_centroids = torch.zeros((k, d), dtype=torch.float32, device=device)
        new_centroids.scatter_add_(0, index_2d, items.to(torch.float32))
        counts = torch.zeros(k, dtype=torch.float32, device=device)
        counts.scatter_add_(0, assignment.long(), ones)
        has_members = counts > 0
        new_centroids[has_members] = F.normalize(new_centroids[has_members], dim=-1)
        new_centroids[~has_members] = centroids[~has_members]

        shift = (1.0 - (new_centroids * centroids).sum(dim=-1)).clamp(min=0).max().item()
        centroids = new_centroids
        print(f"  iter {it}: max-centroid-shift={shift:.4e}", flush=True)
        if shift < 1e-5:
            print("  converged", flush=True)
            break

    return centroids, assignment


def _build_csr_members(
    assignment: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR (offsets, members) for a per-item cluster assignment.

    ``assignment`` is ``[N_real]`` int32 with values in ``[0, k)``. The members
    list stores 0-indexed positions into ``text_emb`` (no padding row;
    ``text_emb`` is ``[N, D]`` real items only).
    """
    a64 = assignment.to(torch.int64)
    sort_idx = torch.argsort(a64, stable=True).to(torch.int32)
    offsets = torch.zeros(k + 1, dtype=torch.int64)
    counts = torch.bincount(a64, minlength=k)
    offsets[1:] = counts.cumsum(0)
    cluster_members = sort_idx
    return offsets, cluster_members


# ----- SLERP ----------------------------------------------------------------


def _slerp(a: torch.Tensor, b: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """SLERP in fp32. ``a, b`` are L2-normalized ``[B, D]``; ``alpha`` is ``[B]``.

    Falls back to linear interpolation when the parents are near-collinear
    (``omega < 1e-4``) — the SLERP denominator ``sin(omega)`` collapses there.
    """
    dot = (a * b).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    alpha_e = alpha.unsqueeze(-1)
    lerp = (1 - alpha_e) * a + alpha_e * b
    sa = torch.sin((1 - alpha_e) * omega) / sin_omega
    sb = torch.sin(alpha_e * omega) / sin_omega
    slerp = sa * a + sb * b
    near = omega.abs() < 1e-4
    out = torch.where(near, lerp, slerp)
    return F.normalize(out, dim=-1)


# ----- cluster --------------------------------------------------------------


def cmd_cluster(args) -> int:
    source = Path(args.source_dir).expanduser()
    source_content = source / args.source_content_subdir
    text_emb_path = source_content / "text_emb.pt"
    if not text_emb_path.exists():
        print(f"ERROR missing {text_emb_path}", flush=True)
        return 1

    device = torch.device(args.device)
    print(f"START cluster source={source_content} device={device}", flush=True)
    t0 = time.monotonic()

    text_emb = torch.load(str(text_emb_path), map_location=device)
    # text_emb is [N, D] — no padding row (writer drops it at the boundary).
    items = text_emb
    n_real, d = items.shape
    k = args.n_clusters if args.n_clusters else max(2, int(n_real**0.5))
    print(f"  n_real={n_real:,} dim={d} k={k}", flush=True)

    centroids, assignment = _spherical_kmeans(
        items, k, n_iter=args.n_iter, seed=args.seed, device=device,
    )

    cluster_offsets, cluster_members = _build_csr_members(assignment.cpu(), k)

    out_path = source_content / "clusters.pt"
    torch.save(
        {
            "centroids": centroids.cpu(),
            "assignment": assignment.cpu().to(torch.int32),
            "cluster_offsets": cluster_offsets,
            "cluster_members": cluster_members,
            "n_clusters": int(k),
            "n_real": int(n_real),
            "dim": int(d),
            "seed": int(args.seed),
            "n_iter": int(args.n_iter),
        },
        out_path,
    )

    sizes = cluster_offsets[1:] - cluster_offsets[:-1]
    n_small = int((sizes < 2).sum().item())
    print(
        f"DONE cluster in {time.monotonic() - t0:.1f}s — "
        f"k={k} sizes min/median/max={int(sizes.min())}/"
        f"{int(sizes.median())}/{int(sizes.max())} "
        f"clusters_with_<2={n_small} → {out_path}",
        flush=True,
    )
    return 0


# ----- synth ----------------------------------------------------------------


def cmd_synth(args) -> int:
    if args.target_n > MAX_TARGET_N:
        print(
            f"ERROR target-n={args.target_n:,} > {MAX_TARGET_N:,} — at this scale\n"
            f"the attribute tensors exceed comfortable host RAM and the existing\n"
            f"single-file oracle cannot serve the catalog on a single GPU.\n"
            f"See docs/plans/linr-int8-quantization.md for the int8/1-bit path.",
            flush=True,
        )
        return 1

    source = Path(args.source_dir).expanduser()
    output = Path(args.output_dir).expanduser()
    source_content = source / args.source_content_subdir
    output_content = output / "content"
    output.mkdir(parents=True, exist_ok=True)
    output_content.mkdir(parents=True, exist_ok=True)

    text_emb_path = source_content / "text_emb.pt"
    clusters_path = source_content / "clusters.pt"
    source_narrow_path = source / "item_attrs_narrow.pt"
    for p in (text_emb_path, clusters_path, source_narrow_path):
        if not p.exists():
            print(f"ERROR missing {p}", flush=True)
            return 1

    device = torch.device(args.device)
    overall_t0 = time.monotonic()

    print("STEP load source text_emb + clusters", flush=True)
    text_emb = torch.load(str(text_emb_path), map_location=device)
    # text_emb is [N_real, D] — no padding row.
    n_real, dim = text_emb.shape
    if args.target_n < n_real:
        print(f"ERROR target-n={args.target_n} < n_real={n_real}", flush=True)
        return 1
    n_synth = args.target_n - n_real
    n_total = args.target_n
    print(
        f"  n_real={n_real:,} dim={dim} target_n={args.target_n:,} "
        f"n_synth={n_synth:,}",
        flush=True,
    )

    clusters = torch.load(str(clusters_path), map_location=device)
    cluster_offsets = clusters["cluster_offsets"].to(device).to(torch.int64)
    cluster_members = clusters["cluster_members"].to(device).to(torch.int64)
    n_clusters = int(clusters["n_clusters"])
    if clusters["dim"] != dim:
        print(
            f"ERROR clusters dim={clusters['dim']} != text_emb dim={dim}. "
            f"Re-run `cluster` against {args.source_content_subdir}.",
            flush=True,
        )
        return 1

    sizes = cluster_offsets[1:] - cluster_offsets[:-1]
    weights = (sizes - 1).clamp(min=0).to(torch.float32)
    if weights.sum().item() == 0:
        print("ERROR no clusters with >= 2 members; bad clustering", flush=True)
        return 1
    cluster_p = weights / weights.sum()

    g_dev = torch.Generator(device=device).manual_seed(args.seed)
    g_cpu = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    print("STEP load source narrow attrs", flush=True)
    source_narrow = torch.load(str(source_narrow_path), map_location="cpu")

    narrow_bytes = n_total * C_NARROW * A_MAX_NARROW * 8
    print(
        f"  allocating target narrow [{n_total}, {C_NARROW}, {A_MAX_NARROW}] "
        f"≈ {narrow_bytes / 1e9:.2f} GB",
        flush=True,
    )
    target_narrow = torch.empty(
        (n_total, C_NARROW, A_MAX_NARROW), dtype=torch.long
    )
    target_narrow[:n_real] = source_narrow

    def _gen_synth_block(
        n_block: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (embs [n_block, D] fp16 cpu, parent_a_ids [n_block] int64 cpu,
        parent_b_ids [n_block] int64 cpu)."""
        cluster_ids = torch.multinomial(
            cluster_p, n_block, replacement=True, generator=g_dev
        ).to(torch.int64)
        c_off = cluster_offsets[cluster_ids]
        c_sz = (cluster_offsets[cluster_ids + 1] - c_off).to(torch.float32)
        rand_a = (
            torch.rand(n_block, generator=g_dev, device=device) * c_sz
        ).to(torch.int64)
        rand_b = (
            torch.rand(n_block, generator=g_dev, device=device) * (c_sz - 1)
        ).to(torch.int64)
        rand_b = torch.where(rand_b >= rand_a, rand_b + 1, rand_b)
        parent_a = cluster_members[c_off + rand_a]
        parent_b = cluster_members[c_off + rand_b]
        a = text_emb[parent_a].to(torch.float32)
        b = text_emb[parent_b].to(torch.float32)
        if args.method == "slerp":
            alpha = args.alpha_lo + (args.alpha_hi - args.alpha_lo) * torch.rand(
                n_block, generator=g_dev, device=device
            )
            embs = _slerp(a, b, alpha)
        else:
            n_t = torch.randn(n_block, dim, generator=g_dev, device=device)
            n_t = n_t - (n_t * a).sum(dim=-1, keepdim=True) * a
            n_t = F.normalize(n_t, dim=-1)
            embs = F.normalize(a + args.sigma * n_t, dim=-1)
        return embs.to(torch.float16).cpu(), parent_a.cpu(), parent_b.cpu()

    shards_meta: list[dict] = []
    item_offset = 0
    synth_offset = 0
    shard_idx = 0

    print(
        f"STEP synthesize {n_synth:,} items via {args.method!r} "
        f"(shard_size={args.shard_size:,})",
        flush=True,
    )
    pbar = tqdm(total=n_synth, desc="synth", unit="items")
    while item_offset < n_total:
        end = min(item_offset + args.shard_size, n_total)
        shard_rows = end - item_offset
        shard_buf = torch.empty((shard_rows, dim), dtype=torch.float16)

        real_lo, real_hi = item_offset, min(end, n_real)
        if real_hi > real_lo:
            shard_buf[: real_hi - real_lo] = text_emb[real_lo:real_hi].cpu()

        synth_lo = max(item_offset, n_real)
        synth_hi = end
        n_synth_in_shard = synth_hi - synth_lo
        shard_pos = synth_lo - item_offset
        sub = 0
        while sub < n_synth_in_shard:
            sub_n = min(SYNTH_SUBCHUNK, n_synth_in_shard - sub)
            embs, parent_a, parent_b = _gen_synth_block(sub_n)
            shard_buf[shard_pos : shard_pos + sub_n] = embs

            pick_a = torch.rand(sub_n, generator=g_cpu) < 0.5
            inherit_ids = torch.where(pick_a, parent_a, parent_b)

            synth_id_lo = n_real + synth_offset
            synth_id_hi = synth_id_lo + sub_n
            target_narrow[synth_id_lo:synth_id_hi] = source_narrow[inherit_ids]

            shard_pos += sub_n
            sub += sub_n
            synth_offset += sub_n
            pbar.update(sub_n)

        shard_name = f"text_emb_shard_{shard_idx:05d}.pt"
        torch.save(shard_buf, output_content / shard_name)
        shards_meta.append(
            {"filename": shard_name, "start_id": item_offset, "n_rows": shard_rows}
        )
        print(
            f"  wrote {shard_name} rows=[{item_offset}, {end}) "
            f"({shard_rows * dim * 2 / 1e9:.2f} GB)",
            flush=True,
        )
        item_offset = end
        shard_idx += 1
    pbar.close()

    print("STEP write item_attrs_narrow", flush=True)
    torch.save(target_narrow, output / "item_attrs_narrow.pt")

    print("STEP copy vocabs + queries + heldout + eval_split", flush=True)
    top_level_copies = (
        "cat_main_vocab.json",
        "author_vocab.json",
        "license_vocab.json",
        "clause_is_reverse_narrow.pt",
        "heldout.parquet",
        "eval_split.parquet",
        "item_id_map.json",
    )
    for fname in top_level_copies:
        src_p = source / fname
        if src_p.exists():
            shutil.copy(src_p, output / fname)
        else:
            print(f"  WARN missing source {src_p}; skipping copy", flush=True)

    content_copies = ("text_emb.meta.json", "query_emb.pt", "query_emb.meta.json")
    for fname in content_copies:
        src_p = source_content / fname
        if src_p.exists():
            shutil.copy(src_p, output_content / fname)
        else:
            print(f"  WARN missing source {src_p}; skipping copy", flush=True)

    print("STEP write shard_index.json", flush=True)
    shard_index = {
        "n_items": int(n_total),
        "dim": int(dim),
        "dtype": "float16",
        "shard_size": int(args.shard_size),
        "n_shards": len(shards_meta),
        "shards": shards_meta,
        "n_real": int(n_real),
        "synth_id_range": [int(n_real), int(args.target_n)],
        "source_dir": str(source),
        "source_content_subdir": args.source_content_subdir,
        "seed": int(args.seed),
        "method": args.method,
        "alpha_lo": float(args.alpha_lo),
        "alpha_hi": float(args.alpha_hi),
        "sigma": float(args.sigma),
        "n_clusters": int(n_clusters),
    }
    with open(output_content / "shard_index.json", "w") as f:
        json.dump(shard_index, f, indent=2)

    elapsed = time.monotonic() - overall_t0
    print(
        f"ALL DONE synth in {elapsed:.1f}s — n_total={args.target_n:,} "
        f"(n_real={n_real:,} + n_synth={n_synth:,}), {len(shards_meta)} shards, "
        f"narrow={tuple(target_narrow.shape)}",
        flush=True,
    )
    return 0


# ----- all ------------------------------------------------------------------


def cmd_all(args) -> int:
    rc = cmd_cluster(args)
    if rc != 0:
        return rc
    return cmd_synth(args)


# ----- main -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a synthetic arxiv-shape dataset at arbitrary N via "
            "SLERP between same-cluster parents."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def _add_common(p):
        p.add_argument("--source-dir", type=str, required=True,
                       help="top-level source dataset dir (where heldout.parquet, "
                            "item_attrs_*.pt, content/ live)")
        p.add_argument("--source-content-subdir", type=str, default="content",
                       help="which Matryoshka variant to use (content / content_d128 / content_d64)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", type=str, default="cuda")

    def _add_cluster_args(p):
        p.add_argument("--n-clusters", type=int, default=0,
                       help="0 → int(sqrt(N_real))")
        p.add_argument("--n-iter", type=int, default=20)

    def _add_synth_args(p):
        p.add_argument("--output-dir", type=str, required=True)
        p.add_argument("--target-n", type=int, required=True,
                       help=f"total catalog size (max {MAX_TARGET_N:,})")
        p.add_argument("--method", type=str, default="slerp",
                       choices=("slerp", "gaussian"))
        p.add_argument("--alpha-lo", type=float, default=0.30)
        p.add_argument("--alpha-hi", type=float, default=0.70)
        p.add_argument("--sigma", type=float, default=0.10,
                       help="only used when --method gaussian")
        p.add_argument("--shard-size", type=int, default=8_000_000)

    sp_cl = sub.add_parser("cluster", help="spherical k-means on source text_emb")
    _add_common(sp_cl)
    _add_cluster_args(sp_cl)
    sp_cl.set_defaults(func=cmd_cluster)

    sp_sy = sub.add_parser("synth", help="generate sharded synth catalog")
    _add_common(sp_sy)
    _add_synth_args(sp_sy)
    sp_sy.set_defaults(func=cmd_synth)

    sp_al = sub.add_parser("all", help="cluster + synth")
    _add_common(sp_al)
    _add_cluster_args(sp_al)
    _add_synth_args(sp_al)
    sp_al.set_defaults(func=cmd_all)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
