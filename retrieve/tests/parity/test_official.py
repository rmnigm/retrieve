"""Meta's official SilverTorch ops (``torch.ops.st.*``, meta-recsys/silvertorch @ 21aa35e)
vs the pure-torch reference, the Triton kernels and the ``SilverTorch`` layer — the
T1–T7 gates of docs/plans/silvertorch-official-integration.md §5.2.

The official scorer reads a **cluster-sorted** int8 table through a CSR
(``cluster_offsets``) and returns sorted-table positions; the Triton kernels read the
original table through the padded ``flat_items``. Both are built here from one
``make_probe_family`` layout, so both arms score the very same candidate set and the
int32 path must agree with Triton **bit for bit** on the ``[B, k]`` scores
(``torch.equal``), ids up to permutation within tied scores. The fp16 path (the
instantiation Meta ships for int8 serving) is gated by rank agreement and a relative
error bound instead. The official bloom is a different hash from ours, so it is never
bit-compared: the gates are semantic (no false negatives vs the exact predicate, FPR
recorded, partial-mask ≡ full-mask scores).

Every test is gated by ``require_official()``: it skips when the ``official`` extra is
not installed (or there is no CUDA device) and fails when the package is present but
its extension is broken.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from retrieve.kernels.filters.clause_mask import clause_mask
from retrieve.kernels.silvertorch import official as of
from retrieve.kernels.silvertorch.codesigned_probe_score import (
    _codesigned_probe_score_impl,
)
from retrieve.kernels.silvertorch.codesigned_probe_score_exact import (
    _codesigned_probe_score_exact_impl,
)
from retrieve.layers.filters.exact_attribute import clause_subset_match
from retrieve.layers.silvertorch import OfficialConfig, SilverTorch, build_silvertorch
from retrieve.layers.utils.quantize import quantize_int8, quantize_int8_global
from retrieve.layers.utils.topk import masked_topk
from tests.conftest import (
    make_attrs,
    make_index,
    make_query,
    make_query_attrs,
    require_official,
)
from tests.parity.conftest import (
    assert_ids_equal_up_to_ties,
    make_exact,
    make_probe_family,
    ref_cps_phase23,
)

# --- bit-order pin (roadmap A3, footnote † in 00-roadmap.md §2.1) ---------------------------
#
# The order in which the official scorer reads a ``filtering_bit_mask`` word — and in which
# ``bloom_index_search_batch`` packs its output — was measured on the A100 on 2026-09-06
# (plan §13.2, three independent probes): HIGH-first, doc ``d`` at bit ``63 - d % 64``. It is
# pinned here as a *test-side* constant, independent of the adapter's, so T3 asserts
# ``official.MASK_BIT_ORDER`` / ``official.BLOOM_OUTPUT_BIT_ORDER`` against the measurement
# rather than against themselves; the other order is kept as a negative control (a mask
# packed low-first scores nothing / does not round-trip). A3's probes are the tests below.
OFFICIAL_BIT_ORDER: of.BitOrder = "high_first"
OTHER_ORDER: of.BitOrder = "low_first"

K_SEARCH, HASH_K, B_MULT = 5, 7, 10.0


@pytest.fixture(autouse=True)
def _needs_official():
    require_official()


# --- helpers ----------------------------------------------------------------------------


def csr_from_padded(padded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``padded [n_lists, max_size]`` (``-1`` pads) → ``(sort_perm, cluster_offsets,
    cluster_sizes)``: the cluster-sorted CSR the official scorer indexes, listing each
    cluster's real items in slot order. ``sort_perm[j]`` is the original id at sorted
    position ``j``."""
    valid = padded >= 0
    sort_perm = padded[valid]  # row-major → cluster-major
    sizes = valid.sum(dim=1)
    offsets = torch.zeros(padded.shape[0] + 1, dtype=torch.int64, device=padded.device)
    offsets[1:] = sizes.cumsum(0)
    return sort_perm, offsets, sizes


class Family:
    """One probe family in both layouts: padded (``flat`` for the Triton kernels and the
    reference) and CSR (for the official scorer), over the same int8 codes."""

    def __init__(self, b, n_lists, max_size, n_probe, d, *, seed=7):
        self.padded, self.probe_ids, self.flat, self.n = make_probe_family(
            b, n_lists, max_size, n_probe, seed=seed
        )
        self.b, self.d, self.max_size, self.n_probe = b, d, max_size, n_probe
        self.max_row = n_probe * max_size
        embs = make_index(self.n, d)
        self.codes, self.global_scale = quantize_int8_global(embs)
        self.query = make_query(b, d)
        self.sort_perm, self.offsets, self.sizes = csr_from_padded(self.padded)
        self.codes_sorted = self.codes[self.sort_perm].contiguous()

    def official(self, k, **kw):
        return of.official_probe_score(
            self.query,
            self.probe_ids,
            self.offsets,
            self.sizes,
            self.codes_sorted,
            self.sort_perm,
            self.global_scale,
            k,
            self.max_row,
            **kw,
        )

    def official_full(self, **kw):
        return of.official_scores_full(
            self.query,
            self.probe_ids,
            self.offsets,
            self.sizes,
            self.codes_sorted,
            self.sort_perm,
            self.global_scale,
            self.max_row,
            **kw,
        )

    def flat_sorted(self) -> torch.Tensor:
        """``[B, P]`` sorted-table positions in the padded slot order (``-1`` pads) — the
        official doc space in Triton's slot layout, for mask gathers."""
        slot = torch.arange(self.max_size, device=self.padded.device)
        pos = self.offsets[self.probe_ids][:, :, None] + slot[None, None, :]
        valid = slot[None, None, :] < self.sizes[self.probe_ids][:, :, None]
        return torch.where(valid, pos, torch.full_like(pos, -1)).reshape(self.b, -1)


def _sentinel_ids(ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Ids of non-finite slots → ``-1``. The Triton ``_impl`` gathers the padded item id at
    a ``-inf`` slot (a ``-1`` pad or a filtered item's real id) while the official
    epilogue goes through ``masked_topk`` and writes the ``-1`` sentinel; the layer
    contract is the sentinel, so both sides are normalised to it before comparing."""
    return ids.masked_fill(~torch.isfinite(scores), -1)


def _assert_bitexact(out, ref):
    out_ids, out_scores = out
    ref_ids, ref_scores = ref
    assert torch.equal(out_scores, ref_scores), "scores must be bit-identical"
    assert_ids_equal_up_to_ties(
        _sentinel_ids(out_ids, out_scores), _sentinel_ids(ref_ids, ref_scores), out_scores
    )


def _jaccard(ids_a: torch.Tensor, ids_b: torch.Tensor) -> float:
    total = 0.0
    for a, b in zip(ids_a.tolist(), ids_b.tolist()):
        sa = {i for i in a if i >= 0}
        sb = {i for i in b if i >= 0}
        total += 1.0 if not (sa | sb) else len(sa & sb) / len(sa | sb)
    return total / ids_a.shape[0]


def _readme_corpus():
    """The upstream README's 4-doc / 2-feature corpus as our ``[N, C, A_max]`` attrs:
    feature 0 = language, feature 1 = category."""
    return torch.tensor(
        [
            [[100, -1], [200, 201]],  # doc 0: en; music, pop
            [[100, 101], [200, -1]],  # doc 1: en, es; music
            [[102, -1], [202, -1]],  # doc 2: ja; news
            [[100, -1], [200, 203]],  # doc 3: en; music, rock
        ],
        dtype=torch.int64,
        device="cuda",
    )


README_QUERIES = ["0:100", "0:100 AND NOT 1:201", "0:102 OR 1:203"]
README_HITS = [[0, 1, 3], [1, 3], [2, 3]]


# --- T1: int32 path, bit-exact vs the reference and vs Triton --------------------------


# d=96 is official-supported but not a Triton power-of-two width: reference gate only.
@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k",
    [(16, 64, 4, 64, 8), (64, 96, 8, 128, 32), (32, 64, 8, 96, 16), (32, 90, 8, 128, 32)],
)
@pytest.mark.parametrize("b", [1, 16])
def test_t1_int32_path_bitexact(n_lists, max_size, n_probe, d, k, b):
    """``divisor_for_int8=-1`` returns the raw int32 dot; after our host epilogue the
    ``[B, k]`` scores equal the fp32 reference and the Triton kernel **bit for bit**
    (same int32 dot, same two left-associated fp32 multiplies, same topk). Ids equal up
    to permutation within tied scores. ``max_size=90`` exercises probed clusters whose
    length is not a multiple of the warp (the official ``process_cluster_remaining``
    path) and mask words with a pad tail."""
    f = Family(b, n_lists, max_size, n_probe, d)
    out = f.official(k, score_path="int32")
    ref = ref_cps_phase23(f.query, f.flat, f.codes, f.global_scale, k)
    _assert_bitexact(out, ref)
    if d & (d - 1) == 0:
        tri = _codesigned_probe_score_impl(f.query, f.flat, f.codes, f.global_scale, k)
        _assert_bitexact(out, tri)


@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("d", [64, 128])
def test_t1_exact_mask_bitexact_vs_triton(d, reverse):
    """Exact mode on the official scorer is our ``clause_mask`` packed into
    ``filtering_bit_mask`` (phase 2 ours, full ``N``): the same AND-of-OR / reverse /
    inactive predicate the Triton exact kernel fuses, so scores are bit-identical to
    ``codesigned_probe_score_exact`` and to the reference, ids up to ties."""
    b, k = 16, 32
    f = Family(b, 64, 96, 8, d)
    attrs, rev, q_attrs = make_exact(f.n, b, reverse=reverse)
    mask = clause_mask(attrs[f.sort_perm].contiguous(), rev, q_attrs)  # [B, N] sorted ids
    assert mask.shape == (b, f.n)
    out = f.official(k, score_path="int32", filtering_bit_mask=of.pack_mask(mask))
    tri = _codesigned_probe_score_exact_impl(
        f.query,
        f.flat,
        f.codes,
        f.global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
    )
    _assert_bitexact(out, tri)
    ref = ref_cps_phase23(
        f.query,
        f.flat,
        f.codes,
        f.global_scale,
        k,
        item_clause_attrs=attrs,
        clause_is_reverse=rev,
        query_clause_attrs=q_attrs,
    )
    _assert_bitexact(out, ref)


def test_t1_raw_output_contract():
    """The raw op output: int32 scores and int32 indices of width ``round_up(max_row, 32)``,
    pads at ``INT32_MIN`` / ``-1``, every returned index a real sorted position of a probed
    cluster, and the multiset of returned positions equal to the probed candidate set."""
    f = Family(4, 16, 90, 4, 64)
    q_codes, _ = quantize_int8(f.query)
    raw, idx = of.fused_scores(q_codes, f.probe_ids, f.offsets, f.sizes, f.codes_sorted, f.max_row)
    assert raw.dtype == torch.int32 and idx.dtype == torch.int32
    assert raw.shape == idx.shape == (4, of.padded_rows(f.max_row))
    pad = idx < 0
    assert (raw[pad] == torch.iinfo(torch.int32).min).all()
    assert (idx[pad] == -1).all()
    expected = f.flat_sorted()
    for row in range(4):
        got = sorted(idx[row][~pad[row]].tolist())
        want = sorted(expected[row][expected[row] >= 0].tolist())
        assert got == want, f"row {row}: returned positions != probed candidate set"


# --- T2: fp16 path ------------------------------------------------------------------------


@pytest.mark.parametrize("n_lists,max_size,n_probe,d", [(16, 64, 4, 64), (64, 96, 8, 128)])
@pytest.mark.parametrize("b", [1, 16])
def test_t2_fp16_path_close_to_int32(n_lists, max_size, n_probe, d, b, record_property):
    """``divisor_for_int8 = default_divisor(D)`` (a power of two ≥ 127²·D / 65504): the
    kernel writes ``fp16(dot / divisor)`` — exact division, one fp16 rounding — so every
    finite slot is within ``2⁻¹⁰`` relative of the int32 path, no slot overflows, and the
    top-32 sets agree at ``jaccard ≥ 0.99`` (fp16 ties can swap the boundary)."""
    f = Family(b, n_lists, max_size, n_probe, d)
    s32, ids32, v32 = f.official_full(score_path="int32")
    s16, ids16, v16 = f.official_full(score_path="fp16")
    assert torch.equal(ids32, ids16) and torch.equal(v32, v16)
    assert torch.isfinite(s16[v16]).all(), "fp16 path overflowed"
    nonzero = v32 & (s32 != 0)
    rel = ((s16 - s32).abs() / s32.abs())[nonzero]
    assert (s16[v32 & (s32 == 0)] == 0).all()
    max_rel = float(rel.max().item()) if rel.numel() else 0.0
    record_property("fp16_max_rel_err", max_rel)
    assert max_rel <= 2.0**-10, max_rel
    k = 32
    top32, _ = masked_topk(s32, k, valid=v32, gather_ids=ids32)
    top16, _ = masked_topk(s16, k, valid=v16, gather_ids=ids16)
    jac = _jaccard(top32, top16)
    record_property("fp16_jaccard_at_32", jac)
    print(f"T2 d={d} b={b}: max_rel_err={max_rel:.3e} jaccard@32={jac:.4f}")
    assert jac >= 0.99, jac


def test_t2_default_divisor_is_the_overflow_bound():
    for d, want in [(16, 4), (64, 16), (96, 32), (128, 32), (256, 64)]:
        assert of.default_divisor(d) == want
        assert 127 * 127 * d / of.default_divisor(d) <= 65504
        assert 127 * 127 * d / (of.default_divisor(d) // 2) > 65504


# --- T3: bit order --------------------------------------------------------------------------


def test_t3_bloom_output_word_order():
    """``bloom_index_search_batch(return_bool_mask=False)`` packs doc ``d`` at bit
    ``63 - d % 64`` (``store<bool>`` unpacks from bit 63 down — A3 probe A): ``pack_mask(
    bool_mask, OFFICIAL_BIT_ORDER)`` round-trips the packed output and the other order does
    not, and the adapter's constant agrees with the measurement. Also reproduces the
    README's expected hits on its own corpus."""
    assert of.BLOOM_OUTPUT_BIT_ORDER == OFFICIAL_BIT_ORDER
    attrs = _readme_corpus()
    index, boff = of.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    assert of.bloom_index_docs(boff) == of.DOCS_PER_BUNDLE
    plans = of.parse_plans(README_QUERIES, hash_k=7)
    full = of.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=True)
    assert full.dtype == torch.bool and full.shape == (3, of.DOCS_PER_BUNDLE)
    for row, hits in zip(full[:, :4].tolist(), README_HITS):
        assert [i for i, h in enumerate(row) if h] == hits
    packed = of.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=False)
    assert packed.dtype == torch.int64 and packed.shape == (3, of.WORDS_PER_BUNDLE)
    assert torch.equal(of.pack_mask(full, OFFICIAL_BIT_ORDER), packed), (
        f"packed bloom output is not {OFFICIAL_BIT_ORDER} — A3's measurement no longer holds"
    )
    assert not torch.equal(of.pack_mask(full, OTHER_ORDER), packed)
    assert torch.equal(of.unpack_mask(packed, of.DOCS_PER_BUNDLE, of.BLOOM_OUTPUT_BIT_ORDER), full)


def _scored_docs(doc: int, bit_order: of.BitOrder) -> list[int]:
    """One 70-doc cluster (docs 0–63 in the first mask word, 64–69 in the second; docs 0–63
    on the warp-aligned path, 64–69 on the remainder path) scored under a
    ``filtering_bit_mask`` with exactly ``doc`` set, packed in ``bit_order``."""
    n, d = 70, 16
    g = torch.Generator(device="cuda").manual_seed(11)
    codes = torch.randint(-128, 128, (n, d), generator=g, dtype=torch.int8, device="cuda")
    q_codes = torch.randint(-128, 128, (1, d), generator=g, dtype=torch.int8, device="cuda")
    offsets = torch.tensor([0, n], device="cuda")
    sizes = torch.tensor([n], device="cuda")
    probe = torch.zeros(1, 1, dtype=torch.int64, device="cuda")
    mask = torch.zeros(1, n, dtype=torch.bool, device="cuda")
    mask[0, doc] = True
    packed = of.pack_mask(mask, bit_order)
    _, idx = of.fused_scores(q_codes, probe, offsets, sizes, codes, n, filtering_bit_mask=packed)
    return sorted(idx[idx >= 0].tolist())


@pytest.mark.parametrize("doc", [0, 5, 40, 69])
def test_t3_scorer_reads_filtering_bit_mask(doc):
    """A3 probe B as a test: under the pinned read order the one set doc — and only it —
    is scored; under the other order nothing is (the mirrored bit names a doc past the
    cluster). The adapter's ``MASK_BIT_ORDER`` must equal the pinned measurement."""
    assert of.MASK_BIT_ORDER == OFFICIAL_BIT_ORDER
    scored = _scored_docs(doc, OFFICIAL_BIT_ORDER)
    assert scored == [doc], (
        f"with the mask packed {OFFICIAL_BIT_ORDER} the scorer scored docs {scored}, expected "
        f"[{doc}]: A3's measured bit order no longer holds at this upstream sha"
    )
    assert _scored_docs(doc, OTHER_ORDER) == [], (
        f"a mask packed {OTHER_ORDER} scored docs — the scorer no longer reads {OFFICIAL_BIT_ORDER}"
    )


@pytest.mark.parametrize("bit_order", [OFFICIAL_BIT_ORDER, OTHER_ORDER])
def test_t3_partial_mask_decode_order(bit_order):
    """``unpack_partial_mask`` (the test-side decoder of the ``_return_partial_response``
    triple) agrees with the bool full mask under the pinned bloom output order and not the
    other: README corpus as two 2-doc clusters, probed by every query."""
    attrs = _readme_corpus()
    index, boff = of.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    plans = of.parse_plans(README_QUERIES, hash_k=7)
    full = of.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=True)[:, :4]
    sel_off = torch.tensor([[0, 2]] * 3, device="cuda")
    sel_len = torch.tensor([[2, 2]] * 3, device="cuda")
    cumsum, first, words = of.bloom_partial_masks(index, boff, plans, sel_off, sel_len, 3, 7)
    assert cumsum.dtype == torch.int32 and first.dtype == torch.int8 and words.dtype == torch.int64
    assert cumsum.shape == first.shape == (6,)
    assert cumsum.tolist() == [1, 2, 3, 4, 5, 6] and first.tolist() == [0, 2] * 3
    decoded = of.unpack_partial_mask(cumsum, first, words, sel_len, 2, bit_order)
    assert decoded.shape == (3, 4)
    assert torch.equal(decoded, full) == (bit_order == OFFICIAL_BIT_ORDER)


def test_t3_pack_unpack_reverse_roundtrip():
    g = torch.Generator(device="cuda").manual_seed(3)
    mask = torch.rand(5, 300, generator=g, device="cuda") < 0.5
    hi, lo = of.pack_mask(mask, "high_first"), of.pack_mask(mask, "low_first")
    assert hi.shape == lo.shape == (5, 5)
    assert torch.equal(of.unpack_mask(hi, 300, "high_first"), mask)
    assert torch.equal(of.unpack_mask(lo, 300, "low_first"), mask)
    assert torch.equal(of.reverse_bits64(hi), lo) and torch.equal(of.reverse_bits64(lo), hi)


# --- T4: bloom ⊇ exact, FPR ----------------------------------------------------------------


def _bloom_corpus(n=4096, c=2, a_max=2, n_vocab=50, cluster=256):
    attrs = make_attrs(n, c=c, a_max=a_max, n_vocab=n_vocab, seed=41)
    n_lists = n // cluster
    offsets = torch.arange(0, n + 1, cluster, device="cuda")
    sizes = torch.full((n_lists,), cluster, device="cuda")
    index, boff = of.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    return attrs, offsets, sizes, index, boff


def test_t4_bloom_superset_of_exact_and_fpr(record_property):
    """Every doc the exact AND-of-terms predicate accepts, the official bloom accepts too
    (no false negatives) — on the full-``N`` mask and on the partial masks over probed
    clusters, which must equal the full mask restricted to those docs — and the false
    positive rate at ``b_multiplier=10`` is recorded and below 5 %."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    qa = make_query_attrs(32, c=c, n_vocab=50, inactive_rate=0.0, seed=42)
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
    full = of.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True)[:, :n]
    exact = clause_mask(attrs, rev, qa)
    assert not (exact & ~full).any(), "official bloom has false negatives vs the exact predicate"
    fpr = float((full & ~exact).sum().item() / max(int((~exact).sum().item()), 1))
    pass_rate = float(exact.float().mean().item())
    record_property("bloom_fpr_b10", fpr)
    print(f"T4 AND: exact pass rate {pass_rate:.4f}, official bloom FPR {fpr:.4f} at b_mult=10")
    assert fpr < 0.05, fpr

    g = torch.Generator(device="cuda").manual_seed(43)
    n_probe, max_size = 4, int(sizes[0].item())
    probe_ids = torch.randint(0, sizes.numel(), (32, n_probe), generator=g, device="cuda")
    cumsum, first, words = of.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    partial = of.unpack_partial_mask(cumsum, first, words, sizes[probe_ids], max_size)
    slot = torch.arange(max_size, device="cuda")
    flat = (offsets[probe_ids][:, :, None] + slot[None, None, :]).reshape(32, -1)
    assert torch.equal(partial, full.gather(1, flat)), "partial masks != full mask on probed docs"
    assert not (exact.gather(1, flat) & ~partial).any()


def test_t4_not_expressions_have_no_false_positives(record_property):
    """A bloom ``NOT`` is the complement of a bloom term: false positives of the term become
    false *negatives* of the NOT, so the guarantee flips — ``NOT c:v`` on the official bloom
    never admits a doc that has ``v`` (⊆ the exact reverse clause) but may drop docs that
    do not. Plan §5.2 phrases T4 as ⊇ for NOT too; that property does not hold for a
    bloom and this test records the false-negative rate instead (a deviation for the
    paper's table)."""
    attrs, _, _, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    qa = make_query_attrs(32, c=c, n_vocab=50, inactive_rate=0.0, seed=44)
    qa[:, 1] = -1  # single NOT term on clause 0
    rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
    plans = of.parse_plans(of.queries_to_expressions(qa, rev), HASH_K)
    assert plans[0].device.type == "cpu"
    bloom_not = of.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True)[
        :, :n
    ]
    exact_not = clause_mask(attrs, rev, qa)
    assert not (bloom_not & ~exact_not).any(), "bloom NOT admitted a doc that has the value"
    fnr = float((exact_not & ~bloom_not).sum().item() / max(int(exact_not.sum().item()), 1))
    record_property("bloom_not_false_negative_rate", fnr)
    print(f"T4 NOT: false-negative rate {fnr:.4f} at b_mult=10")
    assert fnr < 0.05, fnr


def test_t4_expression_mapping():
    qa = torch.tensor([[7, -1], [7, 9], [-1, -1]], device="cuda")
    assert of.queries_to_expressions(qa) == ["0:7", "0:7 AND 1:9", ""]
    rev = torch.tensor([False, True], device="cuda")
    assert of.queries_to_expressions(qa, rev) == ["0:7", "0:7 AND NOT 1:9", ""]
    a, b = of.parse_plans(["0:7", ""], HASH_K)
    a2, b2 = of.parse_plans(["0:7", ""], HASH_K)
    assert a is a2 and b is b2, "plans are LRU-cached on the expression tuple"
    # EMPTY plan = match all.
    attrs = _readme_corpus()
    index, boff = of.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    full = of.bloom_full_mask(index, boff, of.parse_plans([""], 7), 3, 7, return_bool_mask=True)
    assert full[0, :4].all()


# --- T5: co-design invariance ----------------------------------------------------------------


@pytest.mark.parametrize("b", [1, 16])
def test_t5_partial_masks_equal_full_mask_scores(b):
    """``fused_kmean_ann_with_partial_masks`` (phase-2 masks only over the probed clusters,
    the paper's co-design) returns exactly the scores of ``fused_kmean_ann`` with the
    full-``N`` ``filtering_bit_mask`` from the same bloom search — paper §4.3 "the results
    remain the same" — and both equal the unfiltered scores masked by the bloom."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    n_lists, max_size = sizes.numel(), int(sizes[0].item())
    d, n_probe = 64, 4
    embs = make_index(n, d)
    codes, gs = quantize_int8_global(embs)
    query = make_query(b, d)
    g = torch.Generator(device="cuda").manual_seed(45)
    probe_ids = torch.randint(0, n_lists, (b, n_probe), generator=g, device="cuda")
    sort_perm = torch.arange(n, device="cuda")  # the corpus is already cluster-sorted
    qa = make_query_attrs(b, c=c, n_vocab=50, inactive_rate=0.2, seed=46)
    plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
    partial = of.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    full = of.bloom_filtering_mask(index, boff, plans, K_SEARCH, HASH_K)
    args = (query, probe_ids, offsets, sizes, codes, sort_perm, gs, n_probe * max_size)
    s_p, i_p, v_p = of.official_scores_full(*args, partial=partial)
    s_f, i_f, v_f = of.official_scores_full(*args, filtering_bit_mask=full)
    assert torch.equal(v_p, v_f) and torch.equal(i_p, i_f) and torch.equal(s_p, s_f)
    s_n, i_n, v_n = of.official_scores_full(*args)
    bool_mask = of.unpack_mask(full, n, of.MASK_BIT_ORDER)
    keep = v_n & bool_mask.gather(1, i_n.clamp_min(0))
    assert torch.equal(v_f, keep)
    assert torch.equal(s_f[keep], s_n[keep]) and torch.equal(i_f[keep], i_n[keep])


# --- T6: the layer ----------------------------------------------------------------------------

N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8
C, A_MAX = 2, 2


@pytest.fixture(scope="module")
def data():
    return {
        "embs": make_index(N, D),
        "query": make_query(B, D),
        "attrs": make_attrs(N, c=C, a_max=A_MAX),
        "q_attrs": make_query_attrs(B, c=C),
    }


def _layer(data, backend, filter_mode="none", *, reverse=None, official=None, **kw):
    args = dict(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3, backend=backend, official=official)
    args.update(kw)
    if filter_mode == "bloom":
        args.update(
            filter_mode="bloom", m_bits=512 if backend != "official" else None, k_hash=K_SEARCH
        )
        return build_silvertorch(data["embs"], item_clause_attrs=data["attrs"], **args)
    if filter_mode == "exact":
        args.update(filter_mode="exact")
        return build_silvertorch(
            data["embs"], item_clause_attrs=data["attrs"], clause_is_reverse=reverse, **args
        )
    return build_silvertorch(data["embs"], **args)


def _transplant(off: SilverTorch, tri: SilverTorch) -> None:
    """Give the official module the Triton module's index (GPU k-means is not
    bit-deterministic across runs, so bit-exact layer comparisons share one index):
    centroids and codes verbatim, the CSR rebuilt from the padded layout."""
    sort_perm, offsets, sizes = csr_from_padded(tri.padded_cluster_items)
    inv = torch.empty_like(sort_perm)
    inv[sort_perm] = torch.arange(sort_perm.numel(), device="cuda")
    off.centroids = tri.centroids.clone()
    off.item_codes = tri.item_codes[sort_perm].contiguous()
    off.global_scale = tri.global_scale.clone()
    off._global_scale_f = tri._global_scale_f
    off._max_cluster_size = tri._max_cluster_size
    off.cluster_offsets, off.cluster_sizes, off.sort_perm, off.inv_perm = (
        offsets,
        sizes,
        sort_perm,
        inv,
    )
    if off.filter_mode == "exact":
        off.item_clause_attrs = tri.item_clause_attrs[sort_perm].contiguous()
        off.clause_is_reverse = tri.clause_is_reverse.clone()


@pytest.mark.parametrize(
    "filter_mode,reverse", [("none", None), ("exact", None), ("exact", "mixed")]
)
def test_t6_layer_int32_bitexact_vs_triton(data, filter_mode, reverse):
    """``SilverTorch(backend="official", OfficialConfig(score_path="int32"))`` on the Triton
    module's index: scores ``torch.equal``, ids up to ties, on the plain and the exact paths
    (exact runs our ``clause_mask`` packed into the official scorer's bit mask — the same
    predicate Triton fuses)."""
    rev = torch.tensor([True, False], device="cuda") if reverse else None
    tri = _layer(data, "triton", filter_mode, reverse=rev)
    off = _layer(
        data, "official", filter_mode, reverse=rev, official=OfficialConfig(score_path="int32")
    )
    _transplant(off, tri)
    qa = data["q_attrs"] if filter_mode != "none" else None
    _assert_bitexact(off(data["query"], qa), tri(data["query"], qa))
    if filter_mode == "exact":
        # And with all clauses inactive: the predicate is identically true.
        qa_all = torch.full((B, C), -1, dtype=torch.long, device="cuda")
        _assert_bitexact(off(data["query"], qa_all), tri(data["query"], qa_all))


@pytest.mark.parametrize("filter_mode", ["none", "exact"])
def test_t6_layer_fp16_default_ranks_like_triton(data, filter_mode):
    """The default (fp16, timed) path against Triton on the same index: shapes and dtypes
    as every backend, ``jaccard@K ≥ 0.99`` on the id sets."""
    tri = _layer(data, "triton", filter_mode)
    off = _layer(data, "official", filter_mode)
    assert off.official.score_path == "fp16"
    _transplant(off, tri)
    qa = data["q_attrs"] if filter_mode != "none" else None
    ids_o, sc_o = off(data["query"], qa)
    ids_t, sc_t = tri(data["query"], qa)
    assert ids_o.shape == ids_t.shape == (B, K) and ids_o.dtype == torch.long
    assert sc_o.dtype == torch.float32
    jac = _jaccard(ids_o, ids_t)
    print(f"T6 fp16 {filter_mode}: jaccard@{K} vs triton = {jac:.4f}")
    assert jac >= 0.99, jac


@pytest.mark.parametrize("bloom_path", ["partial", "full"])
def test_t6_layer_bloom(data, bloom_path, record_property):
    """Official bloom through the layer (their hash, ``m_bits=None``): the result is the
    exact-mode Triton result on the same index up to bloom false positives — every
    returned id that passes the exact predicate is in Triton's exact top-K (or tied at its
    boundary), and the two bloom paths (partial vs full mask) agree bit for bit."""
    cfg = OfficialConfig(
        score_path="int32", bloom_path=bloom_path, b_multiplier=B_MULT, hash_k=HASH_K
    )
    tri = _layer(data, "triton", "exact")
    off = _layer(data, "official", "bloom", official=cfg)
    assert off.m_bits == 0 and off.k_hash == K_SEARCH
    assert off.bloom_index.dtype == torch.int64 and off.bundle_b_offsets.numel() == N // 2048 + 1
    _transplant(off, tri)
    ids_o, sc_o = off(data["query"], data["q_attrs"])
    ids_t, sc_t = tri(data["query"], data["q_attrs"])
    exact = clause_subset_match(
        data["attrs"][ids_o.clamp_min(0)], data["q_attrs"], tri.clause_is_reverse
    ) & (ids_o >= 0)
    fp = 0
    for b in range(B):
        t_set = set(ids_t[b][ids_t[b] >= 0].tolist())
        kth = float(sc_t[b][torch.isfinite(sc_t[b])].min().item()) if t_set else float("-inf")
        for j in range(K):
            i = int(ids_o[b, j].item())
            if i < 0:
                continue
            if not bool(exact[b, j].item()):
                fp += 1
                continue
            assert i in t_set or float(sc_o[b, j].item()) <= kth + 1e-6, (
                f"row {b}: exact-passing id {i} (score {sc_o[b, j].item()}) missing from "
                f"Triton's exact top-K (k-th score {kth})"
            )
    record_property("bloom_topk_false_positives", fp)
    print(f"T6 bloom[{bloom_path}]: {fp} bloom false positives among {B * K} returned slots")
    if bloom_path == "full":
        off_p = _layer(
            data,
            "official",
            "bloom",
            official=OfficialConfig(score_path="int32", b_multiplier=B_MULT, hash_k=HASH_K),
        )
        _transplant(off_p, tri)
        _assert_bitexact(off_p(data["query"], data["q_attrs"]), (ids_o, sc_o))


def test_t6_layer_contract(data):
    """Buffers, candidates path, attribute-less bloom index, state-dict round trip."""
    off = _layer(data, "official", "none")
    assert list(off.state_dict()) == [
        "centroids",
        "item_codes",
        "global_scale",
        "cluster_offsets",
        "cluster_sizes",
        "sort_perm",
        "inv_perm",
    ]
    assert not hasattr(off, "padded_cluster_items")
    assert torch.equal(off.item_codes, quantize_int8_global(data["embs"])[0][off.sort_perm])
    # Candidate re-rank maps original ids through inv_perm.
    cand = torch.tensor([[10, 20, 30]] * B, device="cuda")
    ids, sc = off(data["query"], candidate_ids=cand)
    assert set(ids.flatten().tolist()) <= {10, 20, 30}
    tri = _layer(data, "triton", "none")
    _transplant(off, tri)
    ids_t, sc_t = tri(data["query"], candidate_ids=cand)
    ids_o, sc_o = off(data["query"], candidate_ids=cand)
    assert torch.equal(sc_o, sc_t) and torch.equal(ids_o, ids_t)
    # State-dict round trip into a freshly registered module of the same shape.
    twin = _layer(data, "official", "none")
    twin.load_state_dict(off.state_dict())
    twin._max_cluster_size, twin._global_scale_f = off._max_cluster_size, off._global_scale_f
    a, b = off(data["query"]), twin(data["query"])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # Bloom configured without attributes: plain queries work, attribute queries raise.
    bare = SilverTorch(
        k=K,
        n_lists=N_LISTS,
        n_probe=N_PROBE,
        filter_mode="bloom",
        k_hash=K_SEARCH,
        n_iter=3,
        backend="official",
    )
    bare.register_index(data["embs"])
    assert bare.bloom_index.numel() == 0
    assert bare(data["query"])[0].shape == (B, K)
    with pytest.raises(RuntimeError, match="without item_clause_attrs"):
        bare(data["query"], data["q_attrs"])
    with pytest.raises(ValueError, match="k_hash must be <= 10"):
        SilverTorch(
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            filter_mode="bloom",
            k_hash=11,
            backend="official",
        )


# --- T7: eager only; syncs per op -----------------------------------------------------------


def test_t7_compile_refused(data):
    """The official backend is eager-only (plan D7): ``module.compile()`` raises, a
    ``fullgraph`` ``torch.compile`` of the forward raises (our error or dynamo's
    ``Unsupported``, both ``RuntimeError``s), and a default-mode ``torch.compile`` either
    raises or falls back to eager with identical results — never a silently different
    path."""
    off = _layer(data, "official", "none")
    with pytest.raises(RuntimeError, match="eager-only"):
        off.compile()
    eager_ids, eager_scores = off(data["query"])
    torch._dynamo.reset()
    try:
        with pytest.raises(RuntimeError):
            torch.compile(off, fullgraph=True)(data["query"])
        torch._dynamo.reset()
        try:
            out_ids, out_scores = torch.compile(off)(data["query"])
        except RuntimeError:
            pass
        else:
            assert torch.equal(out_ids, eager_ids) and torch.equal(out_scores, eager_scores)
    finally:
        torch._dynamo.reset()


def _count_syncs(fn) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum(1 for w in caught if "synchroniz" in str(w.message).lower())


def test_t7_sync_count_per_op(record_property):
    """Records the host syncs each official op performs under
    ``torch.cuda.set_sync_debug_mode("warn")`` — the fact behind D7 (plan §3 expects 2 for
    ``fused_kmean_ann``, ≥ 1 for the partial-response search and the partial-mask scorer).
    Asserted only to be > 0 for the scorer, which is what makes the backend uncapturable;
    the exact numbers are A3's to confirm and are printed here."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    codes, _ = quantize_int8_global(make_index(n, 64))
    q_codes = torch.randint(-128, 128, (4, 64), dtype=torch.int8, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(47)
    probe_ids = torch.randint(0, sizes.numel(), (4, 4), generator=g, device="cuda")
    qa = make_query_attrs(4, c=c, n_vocab=50, seed=48)
    plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
    max_row = 4 * int(sizes[0].item())
    partial = of.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    counts = {
        "fused_kmean_ann": _count_syncs(
            lambda: of.fused_scores(q_codes, probe_ids, offsets, sizes, codes, max_row)
        ),
        "bloom_index_search_batch_return_partial_response": _count_syncs(
            lambda: of.bloom_partial_masks(
                index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
            )
        ),
        "fused_kmean_ann_with_partial_masks": _count_syncs(
            lambda: of.fused_scores(
                q_codes, probe_ids, offsets, sizes, codes, max_row, partial=partial
            )
        ),
        "bloom_index_search_batch": _count_syncs(
            lambda: of.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K)
        ),
    }
    for name, n_sync in counts.items():
        record_property(f"syncs_{name}", n_sync)
        print(f"T7 syncs per call — {name}: {n_sync}")
    assert counts["fused_kmean_ann"] > 0, (
        "plan §3: the scorer syncs the host (repeat_interleave + .item())"
    )
