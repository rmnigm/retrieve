"""``training``: the encoder's attention mask (a left-padded row stays finite, and the last
position never sees the padding) and ``sampled_softmax_loss`` against a direct
``F.cross_entropy`` over explicitly built candidate lists (with and without logQ), and the
``TrainConfig`` loss / ``normalize`` / ``logq`` boundary."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from training.config import TrainConfig
from training.losses import sampled_softmax_loss
from training.model import Encoder


@pytest.mark.parametrize("encoder", ["sasrec", "hstu"])
def test_left_padding_is_finite_and_invisible_to_the_last_position(encoder):
    torch.manual_seed(0)
    model = Encoder(num_items=20, encoder=encoder, max_seq_length=8, embedding_dim=16,
                    num_heads=2, num_blocks=2, ffn_hidden_dim=32, use_time=True).eval()  # fmt: skip
    items = torch.tensor([[0, 0, 0, 0, 0, 3, 7, 9], [1, 2, 3, 4, 5, 6, 7, 8]])
    timestamps = torch.tensor([[0, 0, 0, 0, 0, 100, 160, 900], list(range(0, 800, 100))])
    with torch.no_grad():
        out = model(items, timestamps)
        assert torch.isfinite(out).all()
        model.item_embedding.weight[0] = 100 * torch.randn(16)
        timestamps[0, :5] = torch.tensor([5, 1, 7, 3, 99])
        moved = model(items, timestamps)
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
