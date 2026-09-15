"""``training``: the frozen recall / ndcg of ``training.evaluate`` agree with
``bench.metrics`` to 1e-9 on random data (plan V D5), and ``encode_split``'s cache is keyed
on the checkpoint's mtime and ``max_seq_length`` (a hit returns the cached tensors, a stale
key re-encodes)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import torch

from bench.metrics import _hits, per_row
from training import encode, evaluate
from training.model import GSASRec


def test_training_metrics_agree_with_bench_metrics():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        ids = torch.stack([torch.randperm(60, generator=g)[:20] for _ in range(16)])
        tgt = torch.randint(0, 15, (16, 6), generator=g)
        nt = torch.randint(0, 7, (16,), generator=g)
        tgt[torch.arange(6).unsqueeze(0) >= nt.unsqueeze(1)] = -1
        hits = evaluate.hits_at(ids, tgt)
        assert torch.equal(hits, _hits(ids, tgt))
        for k in (5, 10, 20):
            ref = per_row(_hits(ids[:, :k], tgt), nt, k)
            assert (evaluate.recall_at_k(hits, nt, k) - ref["recall"]).abs().max() <= 1e-9
            assert (evaluate.ndcg_at_k(hits, nt, k) - ref["ndcg"]).abs().max() <= 1e-9


def _checkpoint(root: Path, num_items: int = 12) -> tuple[Path, Path]:
    params = {
        "max_seq_length": 6, "embedding_dim": 8, "num_heads": 2, "num_blocks": 1,
        "ffn_hidden_dim": 16, "dropout": 0.0, "reuse_item_embeddings": False,
    }  # fmt: skip
    torch.manual_seed(0)
    model = GSASRec(num_items=num_items, **params)
    ckpt_dir = root / "checkpoints" / "tiny"
    ckpt_dir.mkdir(parents=True)
    torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
    (ckpt_dir / "config.json").write_text(json.dumps(params))
    (root / "item_id_map.json").write_text(json.dumps({str(i): i for i in range(1, num_items + 1)}))
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10, 11, 12]]
    pl.DataFrame({"item_ids": seqs, "targets": [[2], [1, 3], [4]]}).write_parquet(
        root / "test.parquet"
    )
    return ckpt_dir / "best_model.pt", root


def test_encode_split_caches_on_ckpt_mtime_and_max_seq_length(tmp_path):
    ckpt, data_dir = _checkpoint(tmp_path)
    kw = dict(max_seq_length=6, batch_size=2, num_workers=0, device=torch.device("cpu"))
    items, queries, targets, n_targets = encode.encode_split(ckpt, data_dir, **kw)
    assert items.shape == (12, 8) and queries.shape == (3, 8)  # the pad row is gone
    assert targets.tolist() == [[1, -1], [0, 2], [3, -1]] and n_targets.tolist() == [1, 2, 1]
    cache = ckpt.parent / encode.ENCODE_CACHE
    assert cache.exists()
    blob = torch.load(cache, weights_only=True)
    blob["queries"][:] = 7.0  # a hit returns the cached tensors verbatim
    torch.save(blob, cache)
    _, again, _, _ = encode.encode_split(ckpt, data_dir, **kw)
    assert bool((again == 7.0).all())
    encode.encode_split(ckpt, data_dir, **{**kw, "max_seq_length": 5})  # a different key
    assert torch.load(cache, weights_only=True)["max_seq_length"] == 5  # re-encoded, rewritten
    os.utime(ckpt, (0, 0))
    _, fresh, _, _ = encode.encode_split(ckpt, data_dir, **kw)
    assert torch.equal(fresh, queries)
    assert torch.load(cache, weights_only=True)["ckpt_mtime"] == 0
