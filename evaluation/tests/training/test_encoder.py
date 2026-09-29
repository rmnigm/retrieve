"""``training``: the encoder's attention mask (a left-padded row stays finite, and the last
position never sees the padding) and ``sampled_softmax_loss`` against a direct
``F.cross_entropy`` over explicitly built candidate lists (with and without logQ), the logQ
expected-count formula, the epochs that write ``_resume.pt``, the ``TrainConfig`` loss /
``normalize`` / ``logq`` boundary, and how ``TrainConfig.load`` resolves them for a legacy or
loss-overridden ``config.json``, and ``train_on_val``: the val rows it trains on (tail of
history ++ targets, only the target positions trained or counted for logQ) and that it runs
no val eval."""

from __future__ import annotations

import dataclasses
import json
import math

import polars as pl
import pytest
import torch
import torch.nn.functional as F

from training import train as train_module
from training.config import TrainConfig
from training.dataset import load_val_transitions, target_mask
from training.losses import sampled_softmax_loss
from training.model import Encoder
from training.train import logq_correction, resume_due, target_frequencies


def test_left_padding_is_finite_and_invisible_to_the_last_position():
    torch.manual_seed(0)
    model = Encoder(num_items=20, max_seq_length=8, embedding_dim=16,
                    num_heads=2, num_blocks=2, ffn_hidden_dim=32).eval()  # fmt: skip
    items = torch.tensor([[0, 0, 0, 0, 0, 3, 7, 9], [1, 2, 3, 4, 5, 6, 7, 8]])
    with torch.no_grad():
        out = model(items)
        assert torch.isfinite(out).all()
        model.item_embedding.weight[0] = 100 * torch.randn(16)
        moved = model(items)
    assert torch.equal(moved[:, -1], out[:, -1])


def _oracle(q, table, pos_ids, cand_ids, temperature, q_of):
    q, table = F.normalize(q, dim=-1), F.normalize(table, dim=-1)
    losses = []
    for row, pos in enumerate(pos_ids.tolist()):
        cands = [c for c in cand_ids.tolist() if c != pos]
        logits = [table[pos] @ q[row] / temperature]
        logits += [table[c] @ q[row] / temperature - math.log(q_of[c]) for c in cands]
        losses.append(F.cross_entropy(torch.stack(logits)[None], torch.tensor([0])))
    return torch.stack(losses).mean()


@pytest.mark.parametrize("logq", [False, True], ids=["plain", "logq"])
def test_sampled_softmax_matches_cross_entropy_over_explicit_candidates(logq):
    g = torch.Generator().manual_seed(0)
    table = torch.randn(10, 4, generator=g)
    q = torch.randn(3, 4, generator=g)
    pos_ids = torch.tensor([2, 5, 7])
    cand_ids = torch.tensor([5, 1, 3, 2, 9, 5])  # rows 0 and 1 each hit their own positive
    # Uneven, so a flipped sign or a corrected positive (2 and 5 are candidates) moves the loss.
    q_of = {5: 0.4, 1: 0.05, 3: 0.1, 2: 0.3, 9: 0.15} if logq else dict.fromkeys(range(10), 1.0)
    log_q = torch.tensor([math.log(q_of[c]) for c in cand_ids.tolist()]) if logq else None
    got = sampled_softmax_loss(q, pos_ids, cand_ids, table, 0.05, normalize=True, log_q=log_q)
    want = _oracle(q, table, pos_ids, cand_ids, 0.05, q_of)
    assert torch.allclose(got, want, rtol=1e-6, atol=0)  # fp32, same values summed in another order


def test_logq_is_the_log_expected_draw_count():
    p_train = torch.tensor([0.0, 0.5, 0.375, 0.125], dtype=torch.float64)
    candidates = torch.tensor([1, 2, 1, 3, 2])
    # M = 2 in-batch, K = 3 uniform over N = 4: q_j = 2·p_j + 3/4.
    want = torch.tensor([math.log(q) for q in [1.75, 1.5, 1.75, 1.0, 1.5]])
    got = logq_correction(p_train, candidates, m=2, k=3, n=4)
    assert torch.allclose(got, want, rtol=0, atol=1e-7)  # float64 log rounded to fp32 on both sides


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"loss": "gbc"}, "expected 'gbce' or 'sampled_softmax'"),
        ({"loss": "gbce", "normalize": True}, "normalize=True is only implemented for"),
        ({"loss": "gbce", "logq": True}, "logq=True is only implemented for"),
    ],
    ids=["unknown-loss", "gbce-normalize", "gbce-logq"],
)
def test_train_config_rejects_what_the_loss_would_silently_ignore(overrides, match):
    with pytest.raises(ValueError, match=match):
        TrainConfig(**overrides)


@pytest.mark.parametrize(
    ("kwargs", "want"),
    [({}, ("sampled_softmax", True, True)), ({"loss": "gbce"}, ("gbce", False, False))],
    ids=["default", "gbce"],
)
def test_train_config_resolves_normalize_and_logq_per_loss(kwargs, want):
    cfg = TrainConfig(**kwargs)
    assert (cfg.loss, cfg.normalize, cfg.logq) == want


_SSM = dataclasses.asdict(TrainConfig())
_GBCE = dataclasses.asdict(TrainConfig(loss="gbce"))


@pytest.mark.parametrize(
    ("saved", "overrides", "want"),
    [
        ({"embedding_dim": 128}, {}, ("gbce", False, False)),
        (_SSM, {"loss": "gbce"}, ("gbce", False, False)),
        (_GBCE, {"loss": "sampled_softmax"}, ("sampled_softmax", True, True)),
        (_GBCE, {"loss": "sampled_softmax", "logq": False}, ("sampled_softmax", True, False)),
        (
            _SSM | {"normalize": False},
            {"loss": "sampled_softmax"},
            ("sampled_softmax", False, True),
        ),
    ],
    ids=["no-loss-key-is-gbce", "ssm-to-gbce", "gbce-to-ssm", "explicit-wins", "same-loss-keeps"],
)
def test_load_resolves_normalize_and_logq_for_the_resulting_loss(tmp_path, saved, overrides, want):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(saved))
    cfg = TrainConfig.load(path, **overrides)
    assert (cfg.loss, cfg.normalize, cfg.logq) == want


@pytest.mark.parametrize(
    ("resume_every", "stop_epoch", "want"),
    [(1, None, [0, 1, 2, 3, 4, 5, 6, 7]), (3, None, [2, 5, 7]), (3, 3, [2, 3]), (4, 5, [3, 5])],
)
def test_resume_is_written_every_n_epochs_and_on_the_last(resume_every, stop_epoch, want):
    config = TrainConfig(num_epochs=8, resume_every=resume_every)
    last = config.num_epochs - 1 if stop_epoch is None else stop_epoch
    got = [e for e in range(last + 1) if resume_due(e, config, stopping=e == stop_epoch)]
    assert got == want


_VAL = {
    "item_ids": [[1, 2, 3], [7, 8, 9, 10, 11, 12], [1], [1]],
    "targets": [[4, 5], [13], [2], [2, 3, 4, 5, 6, 7]],
}


def test_train_on_val_rows_are_the_tail_of_history_then_targets(tmp_path):
    pl.DataFrame(_VAL).write_parquet(tmp_path / "val.parquet")
    items, first = load_val_transitions(str(tmp_path / "val.parquet"), 4, torch.device("cpu"))
    assert items.tolist() == [
        [1, 2, 3, 4, 5],
        [9, 10, 11, 12, 13],
        [0, 0, 0, 1, 2],
        [3, 4, 5, 6, 7],
    ]
    assert first.tolist() == [2, 3, 3, 0]
    trained = [row[1:][m].tolist() for row, m in zip(items, target_mask(items, first), strict=True)]
    assert trained == [[4, 5], [13], [2], [4, 5, 6, 7]]
    p_train = target_frequencies(items, first, num_items=13)
    assert torch.equal(p_train * 8, torch.bincount(torch.tensor([4, 5, 13, 2, 4, 5, 6, 7]),
                                                   minlength=14).double())  # fmt: skip


def test_train_on_val_runs_no_val_eval(tmp_path, monkeypatch):
    (tmp_path / "item_id_map.json").write_text(json.dumps({str(i): i for i in range(1, 14)}))
    pl.DataFrame({"item_ids": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]] * 2}).write_parquet(
        tmp_path / "train.parquet"
    )
    pl.DataFrame(_VAL).write_parquet(tmp_path / "val.parquet")
    calls, seen = [], []
    monkeypatch.setattr(train_module, "evaluate", lambda *a, **k: calls.append(a) or {})
    real_batches = train_module.train_batches
    monkeypatch.setattr(train_module, "train_batches",
                        lambda i, f, b: seen.append((i, f)) or real_batches(i, f, b))  # fmt: skip
    config = TrainConfig(data_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "ckpt"),
                         max_seq_length=4, embedding_dim=8, num_blocks=1, num_heads=1,
                         ffn_hidden_dim=8, num_negatives=4, inbatch_negatives=4, batch_size=2,
                         num_epochs=2, eval_every=1, compile=False, wandb_enabled=False,
                         device="cpu", train_on_val=True)  # fmt: skip
    train_module.train(config)
    assert calls == []
    assert (tmp_path / "ckpt" / "best_model.pt").exists()
    val_items, val_first = load_val_transitions(
        str(tmp_path / "val.parquet"), 4, torch.device("cpu")
    )
    items, first = seen[0]
    assert torch.equal(items[4:], val_items)
    assert first.tolist() == [0, 0, 0, 0, *val_first.tolist()]
    metrics = json.loads((tmp_path / "ckpt" / "train_metrics.json").read_text())
    assert metrics["best_val_metric"] == {}
