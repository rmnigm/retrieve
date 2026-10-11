"""CPU check: Sign-OPORP at k_bits = D is seed-invariant in its Hamming scores (bits differ,
scores and LiNRV3 output equal); at k_bits < D it is not.

python -I v3_seed_check.py [item_embs.pt] [--keep-zeros]
"""

import sys

import torch
from retrieve.indexing.quantize import project_oporp_1bit_query, quantize_oporp_1bit
from retrieve.modules.linr import LiNRV3


def hamming(qb, bits, k_bits):
    x = qb[:, None, :] ^ bits[None, :, :]
    return k_bits - 2 * sum(((x >> s) & 1).sum(-1) for s in range(64))


def scores(embs, queries, seed, k_bits):
    bits, signs, perm = quantize_oporp_1bit(embs, seed=seed, k_bits=k_bits)
    return bits, hamming(project_oporp_1bit_query(queries, signs, perm, k_bits), bits, k_bits)


args = [a for a in sys.argv[1:] if a != "--keep-zeros"]
torch.manual_seed(123)
# The real table minus its all-zero pad row 0, which the harness drops; a random one otherwise.
embs = torch.load(args[0]).float()[1:20001] if args else torch.randn(20000, 128)
if "--keep-zeros" not in sys.argv:
    embs[embs == 0] = 1e-3  # an exact 0 packs as bit 0 under either sign: the one seed-dependent case
queries = torch.randn(64, embs.shape[1])
d = embs.shape[1]
print(f"table {tuple(embs.shape)}, exact zeros {(embs == 0).sum().item()}")

for k_bits in (d, d // 2):
    b0, s0 = scores(embs, queries, 0, k_bits)
    for seed in (1, 2):
        b, s = scores(embs, queries, seed, k_bits)
        print(f"k_bits {k_bits}, seed {seed} vs 0: bits equal {torch.equal(b, b0)}, "
              f"hamming scores equal {torch.equal(s, s0)}")

out = []
for seed in (0, 1, 2):
    m = LiNRV3(100, candidate_pool=2000, seed=seed, backend="torch")
    m.register_index(embs)
    with torch.no_grad():
        out.append(m(queries))
for seed in (1, 2):
    print(f"LiNRV3 torch seed {seed} vs 0: ids equal {torch.equal(out[seed][0], out[0][0])}, "
          f"scores equal {torch.equal(out[seed][1], out[0][1])}")
