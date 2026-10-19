"""Meta's SilverTorch ops, adapted (``torch.ops.st.*``, the ``silvertorch/`` member): the parity
tests of the adapted code against an independent torch oracle, through the ops and through the
``SilverTorch`` layer. The comparison against Meta's code as published (pinned at ``21aa35e``)
is in docs/system/parity.md.

The upstream-shaped ops (``fused_kmean_ann`` and the bloom searches, the parity tests'
reference path)
and the one-op forward ``st::search`` read the same cluster-sorted int8 table through the same
CSR (``cluster_offsets``, built here by ``make_probe_family``), so they score the very same
candidate set. On the int32 path the ``[B, k]`` scores must agree **bit for bit**
(``torch.equal``), ids up to permutation within tied scores. The fp16 path (the instantiation
Meta ships for int8 serving) is gated by rank agreement and a relative error bound instead.
Meta's bloom is gated semantically: no false negatives vs the exact predicate, FPR recorded,
partial masks ≡ the full mask, the rechecked bloom ≡ the exact filter.

Every test is gated by ``require_official()``: it skips when the ``silvertorch`` extension is
not installed (or there is no CUDA device) and fails when the package is present but its
extension is broken.
"""

from __future__ import annotations

import pytest
import torch

from retrieve.functional import clause_subset_match, masked_topk
from retrieve.indexing.quantize import quantize_int8, quantize_int8_global
from retrieve.modules.silvertorch import OfficialConfig, SilverTorch, SilverTorchBuilder
from retrieve.ops import official as of
from retrieve.ops.triton.clause_mask import clause_mask
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
)

# The order in which Meta's scorer reads a ``filtering_bit_mask`` word, and in which
# ``bloom_index_search_batch`` packs its output, measured on the A100 (kernels.md § Meta's ops,
# "Bit order"): HIGH-first, doc ``d`` at bit ``63 - d % 64``. A test-side constant, so T3 asserts
# the adapter's ``MASK_BIT_ORDER`` / ``BLOOM_OUTPUT_BIT_ORDER`` against the measurement rather
# than against themselves; the other order is the negative control.
OFFICIAL_BIT_ORDER: of.BitOrder = "high_first"
OTHER_ORDER: of.BitOrder = "low_first"

K_SEARCH, HASH_K, B_MULT = 5, 7, 10.0


@pytest.fixture(autouse=True)
def _needs_silvertorch():
    require_official()


def _prep(module, qa):
    """The module's prepared filter for ``qa`` (``None`` unfiltered)."""
    return None if qa is None else module.prepare_queries(qa)


# The schemas of every op the adapter calls: the upstream-shaped six exactly as Meta published
# them at 21aa35e28b6d…, plus the adapted forward. A different build fails here, before any gate
# reads a renamed or re-ordered argument.
PINNED_SCHEMAS = {
    "fused_kmean_ann": (
        "st::fused_kmean_ann(Tensor cluster_offsets, Tensor cluster_ids, Tensor cluster_length, "
        "Tensor embeddings, Tensor queries, int max_tensor_size_per_row, "
        "Tensor? filtering_bit_mask=None, int invalid_index_value=-1, int divisor_for_int8=-1, "
        "Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None) -> (Tensor, Tensor)"
    ),
    "fused_kmean_ann_with_partial_masks": (
        "st::fused_kmean_ann_with_partial_masks(Tensor cluster_offsets, Tensor cluster_ids, "
        "Tensor cluster_length, Tensor embeddings, Tensor queries, int max_tensor_size_per_row, "
        "Tensor partial_mask_column_counts_cumsum, "
        "Tensor partial_mask_first_item_offset_in_column, "
        "Tensor partial_mask_column_results, int invalid_index_value=-1, int divisor_for_int8=-1, "
        "Tensor? filtering_bit_index=None, Tensor? per_embedding_scale=None, "
        "Tensor? cluster_warp_size=None, Tensor? cluster_warp_rounded_length_cumsum=None, "
        "Tensor? cluster_remaining_length_cumsum=None, Tensor? cluster_warp_size_cumsum=None, "
        "int total_cluster_rounded_warps=0, int total_cluster_remaining_warps=0) "
        "-> (Tensor, Tensor)"
    ),
    "bloom_index_build": (
        "st::bloom_index_build(Tensor feature_ids, Tensor feature_offsets, Tensor feature_values, "
        "float b_multiplier, int k, bool fast_build=False) -> (Tensor, Tensor)"
    ),
    "parse_expression_query_batch": (
        "st::parse_expression_query_batch(str[] expressions, Tensor silvertorch_ks, "
        "int bloom_hash_k, bool return_query_plan=True, int max_sub_queries=5) -> (int, Tensor[])"
    ),
    "bloom_index_search_batch": (
        "st::bloom_index_search_batch(Tensor bloom_index, Tensor bloom_bundle_b_offsets, "
        "Tensor bloom_query_plans_data, Tensor bloom_query_plans_offsets, int k, int hash_k, "
        "bool return_bool_mask=True) -> Tensor"
    ),
    "bloom_index_search_batch_return_partial_response": (
        "st::bloom_index_search_batch_return_partial_response(Tensor bloom_index, "
        "Tensor bloom_bundle_b_offsets, Tensor bloom_query_plans_data, "
        "Tensor bloom_query_plans_offsets, Tensor selected_cluster_offsets, "
        "Tensor selected_cluster_lengths, int k, int hash_k, Tensor? query_plan_index=None) "
        "-> (Tensor, Tensor, Tensor)"
    ),
    "search": (
        "st::search(Tensor query, Tensor probe_ids, Tensor cluster_offsets, Tensor cluster_sizes, "
        "Tensor item_codes, Tensor sort_perm, Tensor global_scale, int k, "
        "int max_tensor_size_per_row, int divisor_for_int8=-1, Tensor? filtering_bit_mask=None, "
        "Tensor? bloom_index=None, Tensor? bloom_bundle_b_offsets=None, Tensor? plans=None, "
        "int plan_count=0, int max_stack_size=0, Tensor? query_plan_index=None, int bloom_k=0, "
        "int hash_k=0, Tensor? item_clause_attrs=None, Tensor? clause_slots=None, "
        "Tensor? clause_is_reverse=None, Tensor? query_clause_attrs=None) -> (Tensor, Tensor)"
    ),
}


def test_op_schemas_are_pinned():
    assert sorted(PINNED_SCHEMAS) == sorted(of.REQUIRED_OPS)
    for name, schema in PINNED_SCHEMAS.items():
        assert str(getattr(torch.ops.st, name).default._schema) == schema, name


# --- helpers ----------------------------------------------------------------------------


def _oracle(query, probe_ids, offsets, codes, sort_perm, global_scale, k, mask=None):
    """Algorithm 1's phases 2+3 in plain torch, one row at a time: the row's probed positions
    concatenated, the int8 query against their codes as an exact int32 dot, dequantised as
    ``(dot · q_scale) · global_scale`` (the adapter's order), ``mask [B, N]`` (over sorted
    positions) to ``-inf``, then ``masked_topk``. No slot arithmetic is shared with the ops."""
    q_codes, q_scales = quantize_int8(query)
    b = query.shape[0]
    rows, scores = [], []
    for row in range(b):
        pos = torch.tensor(
            [
                j
                for c in probe_ids[row].tolist()
                for j in range(int(offsets[c]), int(offsets[c + 1]))
            ],
            dtype=torch.int64,
            device="cuda",
        )
        dot = (codes[pos].to(torch.int32) * q_codes[row].to(torch.int32)).sum(1)
        s = dot.to(torch.float32) * q_scales[row] * global_scale
        if mask is not None:
            s = torch.where(mask[row, pos], s, float("-inf"))
        rows.append(pos)
        scores.append(s)
    width = max(p.numel() for p in rows)
    ids = torch.full((b, width), -1, dtype=torch.int64, device="cuda")
    sc = torch.full((b, width), float("-inf"), device="cuda")
    for row, (pos, s) in enumerate(zip(rows, scores, strict=True)):
        ids[row, : pos.numel()] = sort_perm[pos]
        sc[row, : pos.numel()] = s
    return masked_topk(sc, k, valid=ids >= 0, gather_ids=ids)


class Family:
    """One CSR probe family (``make_probe_family``) over one cluster-sorted int8 table, scored
    by the upstream-shaped ops (``official``), by the one-op forward (``search``) and by the
    oracle."""

    def __init__(self, b, n_lists, max_size, n_probe, d, *, seed=7):
        lay = make_probe_family(b, n_lists, max_size, n_probe, seed=seed)
        self.probe_ids, self.offsets, self.sort_perm, self.width, self.n = (
            lay.probe_ids, lay.cluster_offsets, lay.sort_perm, lay.width, lay.n,
        )  # fmt: skip
        self.sizes = self.offsets.diff()
        self.b, self.d = b, d
        embs = make_index(self.n, d)
        self.codes, gs = quantize_int8_global(embs)
        # ``st::search`` takes the scale as a tensor, as the layer's buffer is.
        self.global_scale = torch.tensor(gs, dtype=torch.float32, device="cuda")
        self.query = make_query(b, d)

    def _args(self, k):
        return (
            self.query, self.probe_ids, self.offsets, self.sizes, self.codes, self.sort_perm,
            self.global_scale, k, self.width,
        )  # fmt: skip

    def official(self, k, **kw):
        return of.official_probe_score(*self._args(k), **kw)

    def search(self, k, **kw):
        return of.fork_search(*self._args(k), **kw)

    def official_full(self, **kw):
        return of.official_scores_full(
            self.query, self.probe_ids, self.offsets, self.sizes, self.codes, self.sort_perm,
            self.global_scale, self.width, **kw,
        )  # fmt: skip

    def probed_positions(self, row: int) -> list[int]:
        """The sorted-table positions row ``row`` probes, the scorer's doc space."""
        return [
            j for c in self.probe_ids[row].tolist()
            for j in range(int(self.offsets[c]), int(self.offsets[c + 1]))
        ]  # fmt: skip

    def oracle(self, k, mask=None):
        return _oracle(
            self.query, self.probe_ids, self.offsets, self.codes, self.sort_perm,
            self.global_scale, k, mask,
        )  # fmt: skip


def _assert_bitexact(out, ref):
    """Scores ``torch.equal``; ids equal up to ties, the ``-1`` at every ``-inf`` slot included
    (every path writes that sentinel itself)."""
    out_ids, out_scores = out
    ref_ids, ref_scores = ref
    assert torch.equal(out_scores, ref_scores), "scores must be bit-identical"
    assert_ids_equal_up_to_ties(out_ids, ref_ids, out_scores)


def _jaccard(ids_a: torch.Tensor, ids_b: torch.Tensor) -> float:
    total = 0.0
    for a, b in zip(ids_a.tolist(), ids_b.tolist(), strict=True):
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


# --- T1: int32 path, bit-exact vs the oracle ------------------------------------------------


@pytest.mark.parametrize(
    "n_lists,max_size,n_probe,d,k",
    [
        (16, 64, 4, 64, 8),
        (64, 96, 8, 128, 32),
        (32, 64, 8, 96, 16),
        (32, 90, 8, 128, 32),
        (32, 90, 8, 192, 32),
        (32, 90, 8, 768, 32),
    ],
)
@pytest.mark.parametrize("b", [1, 16])
def test_t1_int32_path_bitexact(n_lists, max_size, n_probe, d, k, b):
    """``divisor_for_int8=-1`` returns the raw int32 dot; after the epilogue the ``[B, k]`` scores
    of the upstream-shaped scorer and of ``st::search`` equal the oracle **bit for bit** (same
    int32 dot, same two left-associated fp32 multiplies, same topk), ids up to permutation
    within tied scores. ``max_size=90`` exercises probed clusters whose length is not a multiple
    of the warp (the ``process_cluster_remaining`` path) and mask words with a pad tail;
    d = 96 / 192 / 768 are not powers of two."""
    f = Family(b, n_lists, max_size, n_probe, d)
    want = f.oracle(k)
    _assert_bitexact(f.official(k, score_path="int32"), want)
    _assert_bitexact(f.search(k, score_path="int32"), want)


@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("d", [64, 128, 192, 768])
def test_t1_exact_mask_bitexact(d, reverse):
    """The exact predicate on every path: our ``clause_mask`` packed into the scorer's
    ``filtering_bit_mask`` (full ``N``), the same mask into ``st::search``, and the clauses
    tested inside ``st::search`` over the probed docs (the layer's exact path):
    scores bit-identical to the oracle under the same mask, ids up to ties."""
    b, k = 16, 32
    f = Family(b, 64, 96, 8, d)
    attrs, rev, q_attrs = make_exact(f.n, b, reverse=reverse)
    mask = clause_mask(attrs, rev, q_attrs)  # [B, N] over sorted positions, like the attrs
    want = f.oracle(k, mask=mask)
    packed = of.pack_mask(mask)
    _assert_bitexact(f.official(k, score_path="int32", filtering_bit_mask=packed), want)
    _assert_bitexact(f.search(k, score_path="int32", filtering_bit_mask=packed), want)
    slots, clause_slots = of.fork_clause_attrs(attrs)
    exact = (slots, clause_slots, rev, of.fork_query_attrs(q_attrs))
    _assert_bitexact(f.search(k, score_path="int32", exact=exact), want)


def test_t1_raw_output_contract():
    """The raw op output: int32 scores and int32 indices of width ``round_up(max_row, 32)``,
    pads at ``INT32_MIN`` / ``-1``, every returned index a real sorted position of a probed
    cluster, and the multiset of returned positions equal to the probed candidate set."""
    f = Family(4, 16, 90, 4, 64)
    q_codes, _ = quantize_int8(f.query)
    raw, idx = of.fused_scores(q_codes, f.probe_ids, f.offsets, f.sizes, f.codes, f.width)
    assert raw.dtype == torch.int32 and idx.dtype == torch.int32
    assert raw.shape == idx.shape == (4, of.padded_rows(f.width))
    pad = idx < 0
    assert (raw[pad] == torch.iinfo(torch.int32).min).all()
    assert (idx[pad] == -1).all()
    for row in range(4):
        got = sorted(idx[row][~pad[row]].tolist())
        assert got == sorted(f.probed_positions(row)), f"row {row}: != probed candidate set"


# --- T2: fp16 path ------------------------------------------------------------------------


@pytest.mark.parametrize("n_lists,max_size,n_probe,d", [(16, 64, 4, 64), (64, 96, 8, 128)])
@pytest.mark.parametrize("b", [1, 16])
def test_t2_fp16_path_close_to_int32(n_lists, max_size, n_probe, d, b, record_property):
    """``divisor_for_int8 = default_divisor(D)`` (a power of two ≥ 127²·D / 65504): the
    kernel writes ``fp16(dot / divisor)``, an exact division and one fp16 rounding, so every
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
    ``63 - d % 64``: ``pack_mask(bool_mask, OFFICIAL_BIT_ORDER)`` round-trips the packed output
    and the other order does not, and the adapter's constant agrees with the measurement. Also
    reproduces the README's expected hits on its own corpus."""
    assert of.BLOOM_OUTPUT_BIT_ORDER == OFFICIAL_BIT_ORDER
    attrs = _readme_corpus()
    index, boff = of.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    assert of.bloom_index_docs(boff) == of.DOCS_PER_BUNDLE
    plans = of.parse_plans(README_QUERIES, hash_k=7)
    full = of.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=True)
    assert full.dtype == torch.bool and full.shape == (3, of.DOCS_PER_BUNDLE)
    for row, hits in zip(full[:, :4].tolist(), README_HITS, strict=True):
        assert [i for i, h in enumerate(row) if h] == hits
    packed = of.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=False)
    assert packed.dtype == torch.int64 and packed.shape == (3, of.WORDS_PER_BUNDLE)
    assert torch.equal(of.pack_mask(full, OFFICIAL_BIT_ORDER), packed), (
        f"packed bloom output is not {OFFICIAL_BIT_ORDER}: the measured order no longer holds"
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
    """Under the pinned read order the one set doc, and only it, is scored; under the other order
    the *mirrored* doc ``63 - doc % 64`` of the same word is scored instead, or nothing when the
    mirror lies past the cluster (doc 69 → 122 ≥ 70). The adapter's ``MASK_BIT_ORDER`` must
    equal the pinned measurement."""
    assert of.MASK_BIT_ORDER == OFFICIAL_BIT_ORDER
    scored = _scored_docs(doc, OFFICIAL_BIT_ORDER)
    assert scored == [doc], (
        f"with the mask packed {OFFICIAL_BIT_ORDER} the scorer scored docs {scored}, expected "
        f"[{doc}]: the measured bit order no longer holds"
    )
    mirror = (doc // 64) * 64 + (63 - doc % 64)
    assert _scored_docs(doc, OTHER_ORDER) == ([mirror] if mirror < 70 else []), (
        f"a mask packed {OTHER_ORDER} did not score the mirrored doc: the scorer no longer "
        f"reads {OFFICIAL_BIT_ORDER}"
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
    """Every doc the exact AND-of-terms predicate accepts, Meta's bloom accepts too (no false
    negatives), on the full-``N`` mask and on the partial masks over probed clusters (which must
    equal the full mask restricted to those docs), and the false positive rate at
    ``b_multiplier=10`` is recorded and below 5 %."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    qa = make_query_attrs(32, c=c, n_vocab=50, inactive_rate=0.0, seed=42)
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
    full = of.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True)[:, :n]
    exact = clause_mask(attrs, rev, qa)
    assert not (exact & ~full).any(), "Meta's bloom has false negatives vs the exact predicate"
    fpr = float((full & ~exact).sum().item() / max(int((~exact).sum().item()), 1))
    pass_rate = float(exact.float().mean().item())
    record_property("bloom_fpr_b10", fpr)
    print(f"T4 AND: exact pass rate {pass_rate:.4f}, bloom FPR {fpr:.4f} at b_mult=10")
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
    false *negatives* of the NOT, so the guarantee flips: ``NOT c:v`` on Meta's bloom never
    admits a doc that has ``v`` (⊆ the exact reverse clause) but may drop docs that do not. The
    false-negative rate is recorded (a deviation for the paper's table)."""
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
    assert torch.equal(a2, a) and torch.equal(b2, b) and a.device.type == "cpu"
    # EMPTY plan = match all.
    attrs = _readme_corpus()
    index, boff = of.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    full = of.bloom_full_mask(index, boff, of.parse_plans([""], 7), 3, 7, return_bool_mask=True)
    assert full[0, :4].all()


# --- T5: co-design invariance ----------------------------------------------------------------


@pytest.mark.parametrize("b", [1, 16])
def test_t5_partial_masks_equal_full_mask_scores(b):
    """``fused_kmean_ann_with_partial_masks`` (phase-2 masks only over the probed clusters, the
    paper's co-design) returns exactly the scores of ``fused_kmean_ann`` with the full-``N``
    ``filtering_bit_mask`` from the same bloom search (paper §4.3 "the results remain the
    same"), and both equal the unfiltered scores masked by the bloom."""
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


# --- The adapted code's own gates --------------------------------------------------------------


def test_prepared_searches_equal_the_upstream_shaped_ones():
    """The prepared searches (plans uploaded once, read in place) against the
    upstream-shaped searches over CPU plans, on one index and one batch of plans, over ragged
    clusters that start mid-column. Full mask (packed and bool) ``torch.equal``; the partial
    triple equal up to the prepared response's zero tail (cumsum and first offsets
    ``torch.equal``, the first ``cumsum[-1]`` words ``torch.equal``, the rest 0), with one plan
    per distinct expression and a row → plan index; every parsed plan within
    ``plan_capacity_bytes``."""
    g = torch.Generator(device="cuda").manual_seed(5)
    n_lists, p, b, c = 40, 6, 16, 3
    sizes = torch.randint(0, 200, (n_lists,), generator=g, device="cuda")
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
    attrs = make_attrs(int(offsets[-1]), c=c, a_max=2, seed=41)
    index, boff = of.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    qa = make_query_attrs(b, c=c, seed=6)
    qa[::4] = qa[1::4]  # repeated expressions: fewer plans than rows on the partial path
    exprs = of.queries_to_expressions(qa)
    probe = torch.rand(b, n_lists, generator=g, device="cuda").argsort(1)[:, :p]
    sel_off, sel_len = offsets[probe], sizes[probe]
    width = int(sizes.sort(descending=True).values[:p].sum())

    for bool_mask in (False, True):
        up = of.bloom_full_mask(index, boff, of.parse_plans(exprs, HASH_K), K_SEARCH, HASH_K,
                                return_bool_mask=bool_mask)  # fmt: skip
        dp = of.prepare_plans(exprs, HASH_K, 5, c, torch.device("cuda"), slots=b)
        fk = of.bloom_full_mask(index, boff, dp, K_SEARCH, HASH_K, return_bool_mask=bool_mask)
        assert torch.equal(up, fk)

    unique = list(dict.fromkeys(exprs))
    row_plan = torch.tensor([unique.index(e) for e in exprs], device="cuda")
    up = of.bloom_partial_masks(
        index, boff, of.parse_plans(unique, HASH_K), sel_off, sel_len, K_SEARCH, HASH_K, row_plan
    )
    dp = of.prepare_plans(unique, HASH_K, 5, c, torch.device("cuda"), slots=b)
    cap = of.max_partial_columns(b, p, width)
    fk = of.bloom_partial_masks(
        index, boff, dp, sel_off, sel_len, K_SEARCH, HASH_K, row_plan, max_columns=cap
    )
    total = int(up[0][-1])
    assert torch.equal(up[0], fk[0]) and torch.equal(up[1], fk[1])
    assert fk[2].shape == (cap,) and total <= cap
    assert torch.equal(up[2], fk[2][:total]) and not fk[2][total:].any()

    for n_clauses in (1, 2, 5, 8, 12):
        q = make_query_attrs(b, c=n_clauses, inactive_rate=0.1, seed=7 + n_clauses)
        rev = torch.arange(n_clauses, device="cuda") % 2 == 1
        of.prepare_plans(of.queries_to_expressions(q, rev), HASH_K, 5, n_clauses,
                         torch.device("cuda"), slots=b)  # fmt: skip


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("score_path", ["int32", "fp16"])
def test_search_bloom_equals_the_upstream_shaped_path(partial, score_path):
    """``st::search`` end to end on the bloom (prep kernel, payloads that
    evaluate the bloom in place, scorer, epilogue) against the upstream-shaped search ops +
    scorer + the adapter's epilogue, on ragged clusters starting mid-column with repeated
    expressions; the reference's bloom through its partial masks or its full mask: ids and
    scores ``torch.equal`` up to ties."""
    g = torch.Generator(device="cuda").manual_seed(15)
    n_lists, p, b, c, d, k = 64, 12, 16, 3, 128, 50
    sizes = torch.randint(0, 300, (n_lists,), generator=g, device="cuda")
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
    n = int(offsets[-1])
    attrs = make_attrs(n, c=c, a_max=2, n_vocab=8, seed=43)
    index, boff = of.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    codes, _ = quantize_int8_global(make_index(n, d))
    scale = torch.tensor(0.02, device="cuda")
    perm = torch.randperm(n, generator=g, device="cuda")
    q = make_query(b, d)
    qa = make_query_attrs(b, c=c, n_vocab=8, seed=44)
    qa[1::2] = qa[::2]  # every expression twice: fewer plans than rows
    exprs = of.queries_to_expressions(qa)
    probe = torch.rand(b, n_lists, generator=g, device="cuda").argsort(1)[:, :p]
    width = int(sizes.sort(descending=True).values[:p].sum())
    args = (q, probe, offsets, sizes, codes, perm, scale, k, width)
    if partial:
        unique = list(dict.fromkeys(exprs))
        row_plan = torch.tensor([unique.index(e) for e in exprs], device="cuda")
        masks = of.bloom_partial_masks(index, boff, of.parse_plans(unique, HASH_K), offsets[probe],
                                       sizes[probe], K_SEARCH, HASH_K, row_plan)  # fmt: skip
        ref = of.official_probe_score(*args, score_path=score_path, partial=masks)
        dp = of.prepare_plans(unique, HASH_K, 5, c, torch.device("cuda"), slots=b)
    else:
        row_plan = None
        mask = of.bloom_filtering_mask(index, boff, of.parse_plans(exprs, HASH_K), K_SEARCH, HASH_K)
        ref = of.official_probe_score(*args, score_path=score_path, filtering_bit_mask=mask)
        dp = of.prepare_plans(exprs, HASH_K, 5, c, torch.device("cuda"), slots=b)
    bloom = of.ForkBloom(index, boff, dp, row_plan, K_SEARCH, HASH_K)
    _assert_bitexact(of.fork_search(*args, score_path=score_path, bloom=bloom), ref)


def test_configured_bloom_width():
    """``bloom_width="configured"`` builds every bundle ``m_bits`` bits per doc
    (the index is ``N · m_bits / 8`` bytes rounded to whole bundles) and keeps no false
    negatives against the exact predicate."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n = attrs.shape[0]
    for m_bits in (64, 512, 2048):
        cidx, cboff = of.build_bloom_index(
            attrs, b_multiplier=B_MULT, build_k=K_SEARCH, bits_per_doc=m_bits
        )
        bundles = -(-n // 2048)
        assert torch.equal(cboff, torch.arange(bundles + 1, device="cuda") * m_bits)
        assert cidx.numel() * 8 == bundles * 2048 * m_bits // 8
        qa = make_query_attrs(32, c=attrs.shape[1], n_vocab=50, seed=45)
        plans = of.parse_plans(of.queries_to_expressions(qa), HASH_K)
        bloom = of.bloom_full_mask(cidx, cboff, plans, K_SEARCH, HASH_K, return_bool_mask=True)
        exact = clause_mask(attrs, torch.zeros(attrs.shape[1], dtype=torch.bool, device="cuda"), qa)
        assert not (exact & ~bloom[:, :n]).any(), "a bloom false negative"


@pytest.mark.parametrize("d", [128, 192, 768])
def test_probe_prep_equals_reference(d):
    """The prep kernel (``st::probe_prep``, ``search``'s first launch) against
    our torch path, ``torch.equal`` throughout: codes and scales against ``quantize_int8`` on
    10,000 queries (Gaussian, plus rows of zeros, of one tiny value and of exact ``.5`` code
    quotients), the probed clusters' ids / lengths / offsets against the gathers, and the warp
    sizes and three cumsums against ``round_cluster_to_warp`` + cumsum."""
    g = torch.Generator(device="cuda").manual_seed(9)
    q = torch.randn(10_000, d, generator=g, device="cuda")
    q[0] = 0.0
    q[1] = 0.0
    q[1, 3] = 1e-30
    q[2] = torch.arange(d, device="cuda") % 3 - 1.0
    q[2, 0] = 2.54  # 127 · x / 2.54 lands on .5 for x = ±0.01, 1.27 ...
    q[2, 1:5] = torch.tensor([0.01, -0.01, 1.27, -1.27], device="cuda")
    n_lists, p = 300, 24
    sizes = torch.randint(0, 150, (n_lists,), generator=g, device="cuda")
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
    probe = torch.rand(q.shape[0], n_lists, generator=g, device="cuda").argsort(1)[:, :p]
    out = torch.ops.st.probe_prep(q, probe, offsets, sizes)
    codes, scales = quantize_int8(q)
    assert torch.equal(out[0], codes) and torch.equal(out[1], scales)
    length = sizes[probe]
    assert torch.equal(out[2], probe) and torch.equal(out[3], length)
    assert torch.equal(out[4], offsets[probe])
    warps = (length // 32).to(torch.int32).flatten()
    assert torch.equal(out[5], warps)
    assert torch.equal(out[6], (warps * 32).view_as(probe).cumsum(1, dtype=torch.int32))
    assert torch.equal(out[7], (length % 32).flatten().cumsum(0, dtype=torch.int32))
    assert torch.equal(out[8], warps.cumsum(0, dtype=torch.int32))


@pytest.mark.parametrize("k", [100, 1000])
@pytest.mark.parametrize("score_path", ["int32", "fp16"])
@pytest.mark.parametrize("delta", [-1, 0, 1, None])
def test_two_level_topk_equals_the_upstream_shaped_path(k, score_path, delta):
    """``st::search``'s top-k on wide rows (the chunked
    radix select over ~34 chunks of 8192 slots; at k 1000 its candidates take a second round)
    against the upstream-shaped scorer + the adapter's whole-width ``torch.topk``, under heavy
    ties (four distinct code rows) and with exactly ``k + delta`` passing items a row (a
    full-``N`` mask; ``delta`` None: no filter, the unfiltered scorer with the block prefilter at
    k 100): scores ``torch.equal``, ids up to ties (the boundary tie group included: both take
    ties in index order), ``-1`` / ``-inf`` past the finite count."""
    b, d, n_lists = 64, 64, 1024
    lay = make_probe_family(b, n_lists, 600, n_lists)
    width = of.padded_rows(lay.width)
    assert width > 8192 * 8 and (width // 8192 + 1) * k > (8192 if k == 1000 else 0), "one round"
    g = torch.Generator(device="cuda").manual_seed(13)
    base = torch.randint(-127, 128, (4, d), dtype=torch.int8, device="cuda", generator=g)
    codes = base[torch.randint(0, 4, (lay.n,), device="cuda", generator=g)].contiguous()
    mask = None
    if delta is not None:
        passing = torch.zeros(lay.n, dtype=torch.bool, device="cuda")
        passing[torch.randperm(lay.n, device="cuda", generator=g)[: k + delta]] = True
        mask = of.pack_mask(passing.unsqueeze(0).expand(b, -1).contiguous())
    sizes = lay.cluster_offsets.diff()
    scale = torch.tensor(0.01, device="cuda")
    q = make_query(b, d)
    args = (q, lay.probe_ids, lay.cluster_offsets, sizes, codes, lay.sort_perm, scale, k, lay.width)
    out = of.fork_search(*args, score_path=score_path, filtering_bit_mask=mask)
    ref = of.official_probe_score(*args, score_path=score_path, filtering_bit_mask=mask)
    _assert_bitexact(out, ref)
    want = k if delta is None else min(k, k + delta)
    assert (torch.isfinite(out[1]).sum(1) == want).all()


# --- T6: the layer ----------------------------------------------------------------------------


N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8
C, A_MAX = 2, 2
INT32 = OfficialConfig(score_path="int32")
META_BLOOM = OfficialConfig(score_path="int32", bloom_width="upstream", recheck=False)


def _make_data(d):
    return {
        "embs": make_index(N, d),
        "query": make_query(B, d),
        "attrs": make_attrs(N, c=C, a_max=A_MAX),
        "q_attrs": make_query_attrs(B, c=C),
    }


@pytest.fixture(scope="module")
def data():
    return _make_data(D)


def _layer(data, filter_mode="none", *, reverse=None, official=None, **kw):
    args = {"k": K, "n_lists": N_LISTS, "n_probe": N_PROBE, "n_iter": 3, "official": official}
    args.update(kw)
    if filter_mode == "bloom":
        # Meta's own width with ``bloom_width="upstream"`` (``m_bits`` unused), 512 otherwise.
        upstream = official is not None and official.bloom_width == "upstream"
        args.update(filter_mode="bloom", m_bits=kw.get("m_bits", None if upstream else 512),
                    k_hash=K_SEARCH)  # fmt: skip
    if filter_mode == "exact":
        args.update(filter_mode="exact")
    b = SilverTorchBuilder(**args).set_item_embeddings(data["embs"])
    if filter_mode != "none":
        b.set_item_attributes(data["attrs"], reverse)
    return b.build()


def _transplant(dst: SilverTorch, src: SilverTorch) -> None:
    """Give ``dst`` the index of ``src``, so a layer comparison is of the paths and not of two
    index builds: every buffer is copied verbatim (``KMeans.fit`` is reproducible, so the two
    builds would agree anyway; sharing the index keeps the test independent of that)."""
    for name in ("centroids", "item_codes", "global_scale", "cluster_offsets",
                 "cluster_sizes", "sort_perm", "inv_perm"):  # fmt: skip
        setattr(dst, name, getattr(src, name).clone())
    dst._global_scale_f = src._global_scale_f
    dst._probe_width = src._probe_width
    if dst.filter_mode == "exact":
        for name in ("item_clause_attrs", "clause_slots", "clause_is_reverse"):
            setattr(dst, name, getattr(src, name).clone())


def _layer_oracle(m: SilverTorch, data, qa=None, reverse=None):
    """The oracle on the layer's own index: phase 1 as the layer runs it, then ``_oracle`` over
    its cluster-sorted codes; ``qa`` masks with ``clause_mask`` over the sorted attrs."""
    mask = None
    if qa is not None:
        rev = torch.zeros(C, dtype=torch.bool, device="cuda") if reverse is None else reverse
        mask = clause_mask(data["attrs"][m.sort_perm], rev, qa)
    return _oracle(
        data["query"], m._phase1_probe_ids(data["query"]), m.cluster_offsets, m.item_codes,
        m.sort_perm, m.global_scale, m.k, mask,
    )  # fmt: skip


@pytest.mark.parametrize(
    "filter_mode,reverse", [("none", None), ("exact", None), ("exact", "mixed")]
)
@pytest.mark.parametrize("d", [64, 128, 192, 768])
def test_t6_layer_int32_bitexact_vs_oracle(d, filter_mode, reverse):
    """``SilverTorch(OfficialConfig(score_path="int32"))`` on its own index against the oracle:
    scores ``torch.equal``, ids up to ties, on the plain and the exact paths (the clauses tested
    inside the search op), at power-of-two and padded widths."""
    data = _make_data(d)
    rev = torch.tensor([True, False], device="cuda") if reverse else None
    m = _layer(data, filter_mode, reverse=rev, official=INT32)
    qa = data["q_attrs"] if filter_mode != "none" else None
    _assert_bitexact(m(data["query"], _prep(m, qa)), _layer_oracle(m, data, qa, rev))
    if filter_mode == "exact":
        # And with all clauses inactive: the predicate is identically true.
        qa_all = torch.full((B, C), -1, dtype=torch.long, device="cuda")
        _assert_bitexact(m(data["query"], _prep(m, qa_all)), _layer_oracle(m, data))


@pytest.mark.parametrize("filter_mode", ["none", "exact"])
def test_t6_layer_fp16_default_ranks_like_int32(data, filter_mode):
    """The default (fp16, timed) path against the int32 path on the same index: shapes and dtypes
    as every module, ``jaccard@K ≥ 0.99`` on the id sets."""
    exact = _layer(data, filter_mode, official=INT32)
    off = _layer(data, filter_mode)
    assert off.official.score_path == "fp16"
    _transplant(off, exact)
    qa = data["q_attrs"] if filter_mode != "none" else None
    ids_o, sc_o = off(data["query"], _prep(off, qa))
    ids_t, sc_t = exact(data["query"], _prep(exact, qa))
    assert ids_o.shape == ids_t.shape == (B, K) and ids_o.dtype == torch.long
    assert sc_o.dtype == torch.float32
    jac = _jaccard(ids_o, ids_t)
    print(f"T6 fp16 {filter_mode}: jaccard@{K} vs int32 = {jac:.4f}")
    assert jac >= 0.99, jac


def test_t6_layer_bloom(data, record_property):
    """Meta's bloom semantics through the layer (``bloom_width="upstream"``, ``recheck=False``,
    ``m_bits=None``): the result is the exact-mode result on the same index up to bloom false
    positives: every returned id that passes the exact predicate is in the exact top-K (or tied
    at its boundary); the false positives are recorded."""
    exact = _layer(data, "exact", official=INT32)
    off = _layer(data, "bloom", official=META_BLOOM)
    assert off.m_bits == 0 and off.k_hash == K_SEARCH
    assert off.bloom_index.dtype == torch.int64 and off.bundle_b_offsets.numel() == N // 2048 + 1
    assert not hasattr(off, "item_clause_attrs")  # no recheck: no exact attrs kept
    _transplant(off, exact)
    ids_o, sc_o = off(data["query"], off.prepare_queries(data["q_attrs"]))
    ids_t, sc_t = exact(data["query"], exact.prepare_queries(data["q_attrs"]))
    passing = clause_subset_match(
        data["attrs"][ids_o.clamp_min(0)], data["q_attrs"], exact.clause_is_reverse
    ) & (ids_o >= 0)
    fp = 0
    for b in range(B):
        t_set = set(ids_t[b][ids_t[b] >= 0].tolist())
        t_finite = sc_t[b][torch.isfinite(sc_t[b])]
        kth = float(t_finite.min().item()) if t_finite.numel() else float("-inf")
        for j in range(K):
            i = int(ids_o[b, j].item())
            if i < 0:
                continue
            if not bool(passing[b, j].item()):
                fp += 1
                continue
            assert i in t_set or float(sc_o[b, j].item()) <= kth + 1e-6, (
                f"row {b}: exact-passing id {i} (score {sc_o[b, j].item()}) missing from the "
                f"exact top-K (k-th score {kth})"
            )
    record_property("bloom_topk_false_positives", fp)
    print(f"T6 bloom: {fp} bloom false positives among {B * K} returned slots")


def test_recheck_equals_exact(data, record_property):
    """The bloom with its survivors rechecked (``OfficialConfig.recheck``, the default)
    returns the exact filter's result on the same index: ids and scores
    ``torch.equal`` up to ties at int32, with no false positive; without the recheck the same
    index lets at least one through."""
    cfg = {"score_path": "int32", "n_stored_hashes": HASH_K}
    exact = _layer(data, "exact", official=INT32)
    rechecked = _layer(data, "bloom", official=OfficialConfig(**cfg), m_bits=64)
    plain = _layer(data, "bloom", official=OfficialConfig(recheck=False, **cfg), m_bits=64)
    for m in (rechecked, plain):
        _transplant(m, exact)
    q, qa = data["query"], data["q_attrs"]
    ref = exact(q, exact.prepare_queries(qa))
    _assert_bitexact(rechecked(q, rechecked.prepare_queries(qa)), ref)
    ids, _ = plain(q, plain.prepare_queries(qa))
    passing = clause_subset_match(data["attrs"][ids.clamp_min(0)], qa, exact.clause_is_reverse)
    fp = int(((ids >= 0) & ~passing).sum())
    record_property("bloom_m64_false_positives_without_recheck", fp)
    assert fp > 0, "no false positive at m_bits 64: the gate cannot see the recheck"
    print(f"recheck: {fp} false positives at m_bits 64 without the recheck, 0 with it")


def test_t6_layer_contract(data):
    """Buffers, candidates path, attribute-less bloom index, state-dict round trip."""
    off = _layer(data, "none")
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
    # State-dict round trip into a freshly registered module of the same shape (the CSR
    # layout's buffer shapes do not depend on the k-means outcome); the load hook
    # re-derives the cached scalars, nothing is patched by hand.
    twin = _layer(data, "none")
    twin.load_state_dict(off.state_dict())
    assert twin._probe_width == off._probe_width
    assert twin._global_scale_f == off._global_scale_f
    a, b = off(data["query"]), twin(data["query"])
    assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])
    # A bloom index registered without attributes: an empty index, a filtered query raises.
    empty = _layer(data, "none")
    bloom = SilverTorch(k=K, n_lists=N_LISTS, n_probe=N_PROBE, n_iter=3, filter_mode="bloom",
                        m_bits=512, k_hash=K_SEARCH)  # fmt: skip
    bloom.register_index(data["embs"])
    assert bloom.bloom_index.numel() == 0 and bloom.bundle_b_offsets.numel() == 1
    _transplant(bloom, empty)
    a, b = bloom(data["query"]), empty(data["query"])
    assert torch.equal(a[1], b[1]) and torch.equal(a[0], b[0])
    with pytest.raises(RuntimeError, match="without item_clause_attrs"):
        bloom.prepare_queries(data["q_attrs"])
