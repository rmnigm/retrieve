"""``training``: the encoder's attention mask (a left-padded row stays finite, and the last
position never sees the padding) and ``sampled_softmax_loss`` against a direct
``F.cross_entropy`` over explicitly built candidate lists."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

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


def _oracle(q, table, pos_ids, cand_ids, temperature):
    q, table = F.normalize(q, dim=-1), F.normalize(table, dim=-1)
    losses = []
    for row, pos in enumerate(pos_ids.tolist()):
        ids = [pos] + [c for c in cand_ids.tolist() if c != pos]
        logits = table[ids] @ q[row] / temperature
        losses.append(F.cross_entropy(logits[None], torch.tensor([0])))
    return torch.stack(losses).mean()


def test_sampled_softmax_matches_cross_entropy_over_explicit_candidates():
    g = torch.Generator().manual_seed(0)
    table = torch.randn(10, 4, generator=g)
    q = torch.randn(3, 4, generator=g)
    pos_ids = torch.tensor([2, 5, 7])
    cand_ids = torch.tensor([5, 1, 3, 2, 9, 5])  # rows 0 and 1 each hit their own positive
    got = sampled_softmax_loss(q, pos_ids, cand_ids, table, temperature=0.05, normalize=True)
    want = _oracle(q, table, pos_ids, cand_ids, 0.05)
    assert torch.allclose(got, want, rtol=1e-6, atol=0)  # fp32, same values summed in another order
