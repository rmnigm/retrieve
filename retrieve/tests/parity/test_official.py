"""Meta's official SilverTorch ops (``torch.ops.st.*``, meta-recsys/silvertorch @ 21aa35e) and
the adapted code (``torch.ops.stfork.*``, ``official-fork/``): the parity gates of Meta's code
as published against an independent torch oracle, and of the adapted code against the published
code, through the ops and through the ``SilverTorch`` layer.

Both arms read the same cluster-sorted int8 table through the same CSR (``cluster_offsets``,
built here by ``make_probe_family``), so they score the very same candidate set. On the int32
path the ``[B, k]`` scores must agree **bit for bit** (``torch.equal``), ids up to permutation
within tied scores. The fp16 path (the instantiation Meta ships for int8 serving) is gated by
rank agreement and a relative error bound instead. The official bloom is Meta's hash, so it is
never bit-compared against ours: its gates are semantic (no false negatives vs the exact
predicate, FPR recorded, partial-mask ≡ full-mask scores).

Every test runs once per official backend (``OB``, set by the autouse fixture; ``ofb`` is the
adapter with ``backend=OB`` bound) and is gated by ``require_official(OB)``: it skips when the
``official`` extra is not installed (or there is no CUDA device) and fails when the package is
present but its extension is broken. The adapted-vs-published gates run on the ``official-fork``
parametrisation and need both packages.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from functools import partial

import pytest
import torch

from retrieve.functional import clause_subset_match, masked_topk
from retrieve.indexing.quantize import quantize_int8, quantize_int8_global
from retrieve.interfaces import OFFICIAL_BACKENDS
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

# The order in which the official scorer reads a ``filtering_bit_mask`` word, and in which
# ``bloom_index_search_batch`` packs its output, measured on the A100 (kernels.md § official,
# "Bit order"): HIGH-first, doc ``d`` at bit ``63 - d % 64``. A test-side constant, so T3 asserts
# the adapter's ``MASK_BIT_ORDER`` / ``BLOOM_OUTPUT_BIT_ORDER`` against the measurement rather
# than against themselves; the other order is the negative control.
OFFICIAL_BIT_ORDER: of.BitOrder = "high_first"
OTHER_ORDER: of.BitOrder = "low_first"

K_SEARCH, HASH_K, B_MULT = 5, 7, 10.0


def _prep(module, qa):
    """The module's prepared filter for ``qa`` (``None`` unfiltered)."""
    return None if qa is None else module.prepare_queries(qa)


OB: of.OfficialBackend = "official"  # the backend under test, set per test by _needs_official
_OP_CALLING = {
    "build_bloom_index", "parse_plans", "fused_scores", "official_scores_full",
    "official_probe_score", "bloom_partial_masks", "bloom_full_mask", "bloom_filtering_mask",
}  # fmt: skip


class _Bound:
    """``retrieve.ops.official`` with ``backend=OB`` bound on every op-calling function."""

    def __getattr__(self, name):
        fn = getattr(of, name)
        return partial(fn, backend=OB) if name in _OP_CALLING else fn


ofb = _Bound()


@pytest.fixture(autouse=True, params=OFFICIAL_BACKENDS)
def _needs_official(request, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "OB", request.param)
    require_official(request.param)


def _adapted_only():
    """Gate of the adapted-vs-published tests: they run on the ``official-fork`` parametrisation
    and need the published package too."""
    if OB != "official-fork":
        pytest.skip("adapted vs published: runs on the official-fork parametrisation")
    require_official("official")


# The schemas of every op the adapter calls, at the pinned sha (pyproject: 21aa35e28b6d…). A
# different installed build fails here, before any gate reads a renamed or re-ordered argument.
# The adapted code registers the same strings under its own namespace.
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
}


def test_official_op_schemas_are_pinned():
    assert sorted(PINNED_SCHEMAS) == sorted(of.REQUIRED_OPS)
    namespace = of.PACKAGES[OB][1]
    ops = getattr(torch.ops, namespace)
    for name, schema in PINNED_SCHEMAS.items():
        want = schema.replace("st::", f"{namespace}::", 1)
        assert str(getattr(ops, name).default._schema) == want, name


# --- helpers ----------------------------------------------------------------------------


class Family:
    """One CSR probe family (``make_probe_family``) over one cluster-sorted int8 table, scored
    by the official ops and by the oracle."""

    def __init__(self, b, n_lists, max_size, n_probe, d, *, seed=7):
        lay = make_probe_family(b, n_lists, max_size, n_probe, seed=seed)
        self.probe_ids, self.offsets, self.sort_perm, self.width, self.n = (
            lay.probe_ids, lay.cluster_offsets, lay.sort_perm, lay.width, lay.n,
        )  # fmt: skip
        self.sizes = self.offsets.diff()
        self.b, self.d = b, d
        embs = make_index(self.n, d)
        self.codes, self.global_scale = quantize_int8_global(embs)
        self.query = make_query(b, d)

    def official(self, k, *, backend=None, **kw):
        return of.official_probe_score(
            self.query, self.probe_ids, self.offsets, self.sizes, self.codes, self.sort_perm,
            self.global_scale, k, self.width, backend=OB if backend is None else backend, **kw,
        )  # fmt: skip

    def official_full(self, **kw):
        return ofb.official_scores_full(
            self.query, self.probe_ids, self.offsets, self.sizes, self.codes, self.sort_perm,
            self.global_scale, self.width, **kw,
        )  # fmt: skip

    def probed_positions(self, row: int) -> list[int]:
        """The sorted-table positions row ``row`` probes, the official doc space."""
        return [
            j for c in self.probe_ids[row].tolist()
            for j in range(int(self.offsets[c]), int(self.offsets[c + 1]))
        ]  # fmt: skip

    def oracle(self, k, mask=None):
        """Algorithm 1's phases 2+3 in plain torch, one row at a time: the row's probed positions
        concatenated, the int8 query against their codes as an exact int32 dot, dequantised as
        ``(dot · q_scale) · global_scale`` (the adapter's order), ``mask [B, N]`` (over sorted
        positions) to ``-inf``, then ``masked_topk``. No slot arithmetic is shared with the
        ops."""
        q_codes, q_scales = quantize_int8(self.query)
        rows, scores = [], []
        for row in range(self.b):
            pos = torch.tensor(self.probed_positions(row), dtype=torch.int64, device="cuda")
            dot = (self.codes[pos].to(torch.int32) * q_codes[row].to(torch.int32)).sum(1)
            s = dot.to(torch.float32) * q_scales[row] * self.global_scale
            if mask is not None:
                s = torch.where(mask[row, pos], s, float("-inf"))
            rows.append(pos)
            scores.append(s)
        width = max(p.numel() for p in rows)
        ids = torch.full((self.b, width), -1, dtype=torch.int64, device="cuda")
        sc = torch.full((self.b, width), float("-inf"), device="cuda")
        for row, (pos, s) in enumerate(zip(rows, scores, strict=True)):
            ids[row, : pos.numel()] = self.sort_perm[pos]
            sc[row, : pos.numel()] = s
        return masked_topk(sc, k, valid=ids >= 0, gather_ids=ids)


def _assert_bitexact(out, ref):
    """Scores ``torch.equal``; ids equal up to ties, the ``-1`` at every ``-inf`` slot included
    (every backend writes that sentinel itself)."""
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


# --- T1: int32 path, bit-exact vs the oracle; adapted vs published --------------------------


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
    """``divisor_for_int8=-1`` returns the raw int32 dot; after the host epilogue the ``[B, k]``
    scores equal the oracle **bit for bit** (same int32 dot, same two left-associated fp32
    multiplies, same topk), ids up to permutation within tied scores; on the adapted code they
    also equal the published code's. ``max_size=90`` exercises probed clusters whose length is
    not a multiple of the warp (the ``process_cluster_remaining`` path) and mask words with a
    pad tail; d = 96 / 192 / 768 are not powers of two."""
    f = Family(b, n_lists, max_size, n_probe, d)
    out = f.official(k, score_path="int32")
    _assert_bitexact(out, f.oracle(k))
    if OB == "official-fork":
        require_official("official")
        _assert_bitexact(out, f.official(k, score_path="int32", backend="official"))


@pytest.mark.parametrize("reverse", ["none", "mixed"])
@pytest.mark.parametrize("d", [64, 128, 192, 768])
def test_t1_exact_mask_bitexact(d, reverse):
    """Exact mode on the official scorer is our ``clause_mask`` packed into ``filtering_bit_mask``
    (phase 2 ours, full ``N``): scores bit-identical to the oracle under the same mask, ids up
    to ties; on the adapted code also to the published code's."""
    b, k = 16, 32
    f = Family(b, 64, 96, 8, d)
    attrs, rev, q_attrs = make_exact(f.n, b, reverse=reverse)
    mask = clause_mask(attrs, rev, q_attrs)  # [B, N] over sorted positions, like the attrs
    out = f.official(k, score_path="int32", filtering_bit_mask=of.pack_mask(mask))
    _assert_bitexact(out, f.oracle(k, mask=mask))
    if OB == "official-fork":
        require_official("official")
        pub = f.official(
            k, score_path="int32", filtering_bit_mask=of.pack_mask(mask), backend="official"
        )
        _assert_bitexact(out, pub)


def test_t1_raw_output_contract():
    """The raw op output: int32 scores and int32 indices of width ``round_up(max_row, 32)``,
    pads at ``INT32_MIN`` / ``-1``, every returned index a real sorted position of a probed
    cluster, and the multiset of returned positions equal to the probed candidate set."""
    f = Family(4, 16, 90, 4, 64)
    q_codes, _ = quantize_int8(f.query)
    raw, idx = ofb.fused_scores(q_codes, f.probe_ids, f.offsets, f.sizes, f.codes, f.width)
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
    index, boff = ofb.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    assert of.bloom_index_docs(boff) == of.DOCS_PER_BUNDLE
    plans = ofb.parse_plans(README_QUERIES, hash_k=7)
    full = ofb.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=True)
    assert full.dtype == torch.bool and full.shape == (3, of.DOCS_PER_BUNDLE)
    for row, hits in zip(full[:, :4].tolist(), README_HITS, strict=True):
        assert [i for i, h in enumerate(row) if h] == hits
    packed = ofb.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=False)
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
    _, idx = ofb.fused_scores(q_codes, probe, offsets, sizes, codes, n, filtering_bit_mask=packed)
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
        f"[{doc}]: the measured bit order no longer holds at this upstream sha"
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
    index, boff = ofb.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    plans = ofb.parse_plans(README_QUERIES, hash_k=7)
    full = ofb.bloom_full_mask(index, boff, plans, 3, 7, return_bool_mask=True)[:, :4]
    sel_off = torch.tensor([[0, 2]] * 3, device="cuda")
    sel_len = torch.tensor([[2, 2]] * 3, device="cuda")
    cumsum, first, words = ofb.bloom_partial_masks(index, boff, plans, sel_off, sel_len, 3, 7)
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
    index, boff = ofb.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    return attrs, offsets, sizes, index, boff


def test_t4_bloom_superset_of_exact_and_fpr(record_property):
    """Every doc the exact AND-of-terms predicate accepts, the official bloom accepts too (no
    false negatives), on the full-``N`` mask and on the partial masks over probed clusters
    (which must equal the full mask restricted to those docs), and the false positive rate at
    ``b_multiplier=10`` is recorded and below 5 %."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    qa = make_query_attrs(32, c=c, n_vocab=50, inactive_rate=0.0, seed=42)
    rev = torch.zeros(c, dtype=torch.bool, device="cuda")
    plans = ofb.parse_plans(of.queries_to_expressions(qa), HASH_K)
    full = ofb.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True)[:, :n]
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
    cumsum, first, words = ofb.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    partial = of.unpack_partial_mask(cumsum, first, words, sizes[probe_ids], max_size)
    slot = torch.arange(max_size, device="cuda")
    flat = (offsets[probe_ids][:, :, None] + slot[None, None, :]).reshape(32, -1)
    assert torch.equal(partial, full.gather(1, flat)), "partial masks != full mask on probed docs"
    assert not (exact.gather(1, flat) & ~partial).any()


def test_t4_not_expressions_have_no_false_positives(record_property):
    """A bloom ``NOT`` is the complement of a bloom term: false positives of the term become
    false *negatives* of the NOT, so the guarantee flips: ``NOT c:v`` on the official bloom
    never admits a doc that has ``v`` (⊆ the exact reverse clause) but may drop docs that do
    not. The false-negative rate is recorded (a deviation for the paper's table)."""
    attrs, _, _, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    qa = make_query_attrs(32, c=c, n_vocab=50, inactive_rate=0.0, seed=44)
    qa[:, 1] = -1  # single NOT term on clause 0
    rev = torch.tensor([True, False], dtype=torch.bool, device="cuda")
    plans = ofb.parse_plans(of.queries_to_expressions(qa, rev), HASH_K)
    assert plans[0].device.type == "cpu"
    bloom_not = ofb.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K, return_bool_mask=True)[
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
    a, b = ofb.parse_plans(["0:7", ""], HASH_K)
    a2, b2 = ofb.parse_plans(["0:7", ""], HASH_K)
    assert torch.equal(a2, a) and torch.equal(b2, b) and a.device.type == "cpu"
    # EMPTY plan = match all.
    attrs = _readme_corpus()
    index, boff = ofb.build_bloom_index(attrs, b_multiplier=5.0, build_k=3)
    full = ofb.bloom_full_mask(index, boff, ofb.parse_plans([""], 7), 3, 7, return_bool_mask=True)
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
    plans = ofb.parse_plans(of.queries_to_expressions(qa), HASH_K)
    partial = ofb.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    full = ofb.bloom_filtering_mask(index, boff, plans, K_SEARCH, HASH_K)
    args = (query, probe_ids, offsets, sizes, codes, sort_perm, gs, n_probe * max_size)
    s_p, i_p, v_p = ofb.official_scores_full(*args, partial=partial)
    s_f, i_f, v_f = ofb.official_scores_full(*args, filtering_bit_mask=full)
    assert torch.equal(v_p, v_f) and torch.equal(i_p, i_f) and torch.equal(s_p, s_f)
    s_n, i_n, v_n = ofb.official_scores_full(*args)
    bool_mask = of.unpack_mask(full, n, of.MASK_BIT_ORDER)
    keep = v_n & bool_mask.gather(1, i_n.clamp_min(0))
    assert torch.equal(v_f, keep)
    assert torch.equal(s_f[keep], s_n[keep]) and torch.equal(i_f[keep], i_n[keep])


# --- The adapted code's own gates (official-fork/CHANGES.md) ---------------------------------


def test_c4_fork_prepared_searches_equal_upstream():
    """The adapted code's prepared searches (CHANGES.md C4) against the published ones on one
    index and one batch of plans, over ragged clusters that start mid-column. Full mask (packed
    and bool) ``torch.equal``; the partial triple equal up to the prepared response's zero tail
    (cumsum and first offsets ``torch.equal``, the first ``cumsum[-1]`` words ``torch.equal``,
    the rest 0), with one plan per distinct expression and a row → plan index; the two bloom
    builds ``torch.equal``; every parsed plan within ``plan_capacity_bytes``."""
    _adapted_only()
    g = torch.Generator(device="cuda").manual_seed(5)
    n_lists, p, b, c = 40, 6, 16, 3
    sizes = torch.randint(0, 200, (n_lists,), generator=g, device="cuda")
    offsets = torch.cat([sizes.new_zeros(1), sizes.cumsum(0)])
    attrs = make_attrs(int(offsets[-1]), c=c, a_max=2, seed=41)
    index, boff = of.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    fidx, fboff = ofb.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    assert torch.equal(index, fidx) and torch.equal(boff, fboff)
    qa = make_query_attrs(b, c=c, seed=6)
    qa[::4] = qa[1::4]  # repeated expressions: fewer plans than rows on the partial path
    exprs = of.queries_to_expressions(qa)
    probe = torch.rand(b, n_lists, generator=g, device="cuda").argsort(1)[:, :p]
    sel_off, sel_len = offsets[probe], sizes[probe]
    width = int(sizes.sort(descending=True).values[:p].sum())

    for bool_mask in (False, True):
        up = of.bloom_full_mask(index, boff, of.parse_plans(exprs, HASH_K), K_SEARCH, HASH_K,
                                return_bool_mask=bool_mask)  # fmt: skip
        dp = ofb.prepare_plans(exprs, HASH_K, 5, c, torch.device("cuda"), slots=b)
        fk = ofb.bloom_full_mask(index, boff, dp, K_SEARCH, HASH_K, return_bool_mask=bool_mask)
        assert torch.equal(up, fk)

    unique = list(dict.fromkeys(exprs))
    row_plan = torch.tensor([unique.index(e) for e in exprs], device="cuda")
    up = of.bloom_partial_masks(
        index, boff, of.parse_plans(unique, HASH_K), sel_off, sel_len, K_SEARCH, HASH_K, row_plan
    )
    dp = ofb.prepare_plans(unique, HASH_K, 5, c, torch.device("cuda"), slots=b)
    cap = of.max_partial_columns(b, p, width)
    fk = ofb.bloom_partial_masks(
        index, boff, dp, sel_off, sel_len, K_SEARCH, HASH_K, row_plan, max_columns=cap
    )
    total = int(up[0][-1])
    assert torch.equal(up[0], fk[0]) and torch.equal(up[1], fk[1])
    assert fk[2].shape == (cap,) and total <= cap
    assert torch.equal(up[2], fk[2][:total]) and not fk[2][total:].any()

    for n_clauses in (1, 2, 5, 8, 12):
        q = make_query_attrs(b, c=n_clauses, inactive_rate=0.1, seed=7 + n_clauses)
        rev = torch.arange(n_clauses, device="cuda") % 2 == 1
        ofb.prepare_plans(of.queries_to_expressions(q, rev), HASH_K, 5, n_clauses,
                          torch.device("cuda"), slots=b)  # fmt: skip


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("score_path", ["int32", "fp16"])
def test_c8_fork_bloom_search_equals_upstream(partial, score_path):
    """``stfork.search`` end to end on the bloom paths (CHANGES.md C5 / C8: prep kernel, payloads
    that evaluate the bloom in place on the partial path, the prepared full mask otherwise,
    scorer, epilogue) against the published search ops + scorer + the adapter's epilogue, on
    ragged clusters starting mid-column with repeated expressions: ids and scores
    ``torch.equal`` up to ties."""
    _adapted_only()
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
        dp = ofb.prepare_plans(unique, HASH_K, 5, c, torch.device("cuda"), slots=b)
    else:
        row_plan = None
        mask = of.bloom_filtering_mask(index, boff, of.parse_plans(exprs, HASH_K), K_SEARCH, HASH_K)
        ref = of.official_probe_score(*args, score_path=score_path, filtering_bit_mask=mask)
        dp = ofb.prepare_plans(exprs, HASH_K, 5, c, torch.device("cuda"), slots=b)
    bloom = of.ForkBloom(index, boff, dp, row_plan, K_SEARCH, HASH_K)
    _assert_bitexact(of.fork_search(*args, score_path=score_path, bloom=bloom), ref)


def test_c12_fork_configured_bloom_width():
    """``bloom_width="configured"`` (CHANGES.md C12) builds every bundle ``m_bits`` bits per doc
    (the index is ``N · m_bits / 8`` bytes rounded to whole bundles), keeps no false negatives
    against the exact predicate, and ``bloom_width="upstream"`` builds Meta's index
    ``torch.equal``; ``official`` refuses ``"configured"``."""
    _adapted_only()
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n = attrs.shape[0]
    fidx, fboff = ofb.build_bloom_index(attrs, b_multiplier=B_MULT, build_k=K_SEARCH)
    assert torch.equal(fidx, index) and torch.equal(fboff, boff)
    for m_bits in (64, 512, 2048):
        cidx, cboff = ofb.build_bloom_index(
            attrs, b_multiplier=B_MULT, build_k=K_SEARCH, bits_per_doc=m_bits
        )
        bundles = -(-n // 2048)
        assert torch.equal(cboff, torch.arange(bundles + 1, device="cuda") * m_bits)
        assert cidx.numel() * 8 == bundles * 2048 * m_bits // 8
        qa = make_query_attrs(32, c=attrs.shape[1], n_vocab=50, seed=45)
        plans = ofb.parse_plans(of.queries_to_expressions(qa), HASH_K)
        bloom = ofb.bloom_full_mask(cidx, cboff, plans, K_SEARCH, HASH_K, return_bool_mask=True)
        exact = clause_mask(attrs, torch.zeros(attrs.shape[1], dtype=torch.bool, device="cuda"), qa)
        assert not (exact & ~bloom[:, :n]).any(), "a bloom false negative"
    with pytest.raises(ValueError, match="official-fork"):
        OfficialConfig(bloom_width="configured").width_for("official")


def test_c13_fork_recheck_equals_exact(data, record_property):
    """The adapted bloom with its survivors rechecked (``OfficialConfig.recheck``, the default;
    CHANGES.md C13) returns the exact filter's result on the same index: ids and scores
    ``torch.equal`` up to ties to the adapted code's own exact path and to the published code's
    exact path at int32, with no false positive; without the recheck the same index lets at
    least one through."""
    _adapted_only()
    cfg = {"score_path": "int32", "n_stored_hashes": HASH_K}
    want = _layer(data, "official", "exact", official=OfficialConfig(score_path="int32"))
    exact = _layer(data, OB, "exact", official=OfficialConfig(score_path="int32"))
    rechecked = _layer(data, OB, "bloom", official=OfficialConfig(**cfg), m_bits=64)
    plain = _layer(data, OB, "bloom", official=OfficialConfig(recheck=False, **cfg), m_bits=64)
    for m in (exact, rechecked, plain):
        _transplant(m, want)
    q, qa = data["query"], data["q_attrs"]
    ref = want(q, want.prepare_queries(qa))
    _assert_bitexact(exact(q, exact.prepare_queries(qa)), ref)
    _assert_bitexact(rechecked(q, rechecked.prepare_queries(qa)), ref)
    ids, _ = plain(q, plain.prepare_queries(qa))
    passing = clause_subset_match(data["attrs"][ids.clamp_min(0)], qa, want.clause_is_reverse)
    fp = int(((ids >= 0) & ~passing).sum())
    record_property("bloom_m64_false_positives_without_recheck", fp)
    assert fp > 0, "no false positive at m_bits 64: the gate cannot see the recheck"
    print(f"C13: {fp} false positives at m_bits 64 without the recheck, 0 with it")


@pytest.mark.parametrize("d", [128, 192, 768])
def test_c6_fork_probe_prep_equals_reference(d):
    """The adapted code's prep kernel (``stfork.probe_prep``, ``search``'s first launch;
    CHANGES.md C6) against our torch path, ``torch.equal`` throughout: codes and scales against
    ``quantize_int8`` on 10,000 queries (Gaussian, plus rows of zeros, of one tiny value and of
    exact ``.5`` code quotients), the probed clusters' ids / lengths / offsets against the
    gathers, and the warp sizes and three cumsums against ``round_cluster_to_warp`` + cumsum."""
    if OB != "official-fork":
        pytest.skip("adapted code only")
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
    out = torch.ops.stfork.probe_prep(q, probe, offsets, sizes)
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
def test_c7_fork_two_level_topk_equals_upstream(k, score_path, delta):
    """``stfork.search``'s top-k on wide rows (CHANGES.md C7 and the select fixes: the chunked
    radix select over ~34 chunks of 8192 slots; at k 1000 its candidates take a second round)
    against the published scorer + the adapter's whole-width ``torch.topk``, under heavy ties
    (four distinct code rows) and with exactly ``k + delta`` passing items a row (a full-``N``
    mask; ``delta`` None: no filter, the unfiltered scorer with the block prefilter at k 100):
    scores ``torch.equal``, ids up to ties (the boundary tie group included: both take ties in
    index order), ``-1`` / ``-inf`` past the finite count."""
    _adapted_only()
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
    ref = of.official_probe_score(
        *args, score_path=score_path, filtering_bit_mask=mask, backend="official"
    )
    _assert_bitexact(out, ref)
    want = k if delta is None else min(k, k + delta)
    assert (torch.isfinite(out[1]).sum(1) == want).all()


# --- T6: the layer ----------------------------------------------------------------------------


N, D, B, K = 4096, 128, 16, 64
N_LISTS, N_PROBE = 64, 8
C, A_MAX = 2, 2


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


def _layer(data, backend, filter_mode="none", *, reverse=None, official=None, **kw):
    args = {
        "k": K,
        "n_lists": N_LISTS,
        "n_probe": N_PROBE,
        "n_iter": 3,
        "backend": backend,
        "official": official,
    }
    args.update(kw)
    if filter_mode == "bloom":
        args.update(
            filter_mode="bloom",
            # Meta's own width on official; the adapted code's configured width is m_bits.
            m_bits=kw.get("m_bits", None if backend == "official" else 512),
            k_hash=K_SEARCH,
        )
    if filter_mode == "exact":
        args.update(filter_mode="exact")
    b = SilverTorchBuilder(**args).set_item_embeddings(data["embs"])
    if filter_mode != "none":
        b.set_item_attributes(data["attrs"], reverse)
    return b.build()


def _transplant(dst: SilverTorch, src: SilverTorch) -> None:
    """Give ``dst`` the index of ``src`` (a module on ``dst``'s backend or on the published one),
    so a bit-exact layer comparison is of the ops and not of two index builds: both backends
    register the same CSR, so every buffer is copied verbatim (``KMeans.fit`` is reproducible, so
    the two builds would agree anyway; sharing the index keeps the test independent of that)."""
    for name in ("centroids", "item_codes", "global_scale", "cluster_offsets",
                 "cluster_sizes", "sort_perm", "inv_perm"):  # fmt: skip
        setattr(dst, name, getattr(src, name).clone())
    dst._global_scale_f = src._global_scale_f
    dst._probe_width = src._probe_width
    if dst.filter_mode == "exact":
        dst.clause_is_reverse = src.clause_is_reverse.clone()
        if dst.backend == src.backend:
            dst.item_clause_attrs = src.item_clause_attrs.clone()
            if dst.backend == "official-fork":
                dst.clause_slots = src.clause_slots.clone()
        else:  # published ``[N, C, A]`` attrs into the adapted code's layout (CHANGES.md C14)
            dst.item_clause_attrs, dst.clause_slots = of.fork_clause_attrs(src.item_clause_attrs)


@pytest.mark.parametrize(
    "filter_mode,reverse", [("none", None), ("exact", None), ("exact", "mixed")]
)
@pytest.mark.parametrize("d", [64, 128, 192, 768])
def test_t6_layer_int32_adapted_equals_published(d, filter_mode, reverse):
    """``SilverTorch(backend="official-fork", OfficialConfig(score_path="int32"))`` on the
    published module's index: scores ``torch.equal``, ids up to ties, on the plain and the exact
    paths (the adapted code tests the clauses inside its search op; the published code runs our
    ``clause_mask_packed``), at power-of-two and padded widths."""
    _adapted_only()
    data = _make_data(d)
    rev = torch.tensor([True, False], device="cuda") if reverse else None
    cfg = OfficialConfig(score_path="int32")
    pub = _layer(data, "official", filter_mode, reverse=rev, official=cfg)
    fork = _layer(data, OB, filter_mode, reverse=rev, official=cfg)
    _transplant(fork, pub)
    qa = data["q_attrs"] if filter_mode != "none" else None
    _assert_bitexact(fork(data["query"], _prep(fork, qa)), pub(data["query"], _prep(pub, qa)))
    if filter_mode == "exact":
        # And with all clauses inactive: the predicate is identically true.
        qa_all = torch.full((B, C), -1, dtype=torch.long, device="cuda")
        _assert_bitexact(
            fork(data["query"], _prep(fork, qa_all)), pub(data["query"], _prep(pub, qa_all))
        )


@pytest.mark.parametrize("filter_mode", ["none", "exact"])
def test_t6_layer_fp16_default_ranks_like_int32(data, filter_mode):
    """The default (fp16, timed) path against the int32 path on the same index: shapes and dtypes
    as every backend, ``jaccard@K ≥ 0.99`` on the id sets; on the adapted code the fp16 layer
    also equals the published fp16 layer bit for bit."""
    exact = _layer(data, OB, filter_mode, official=OfficialConfig(score_path="int32"))
    off = _layer(data, OB, filter_mode)
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
    if OB == "official-fork":
        require_official("official")
        pub = _layer(data, "official", filter_mode)
        _transplant(off, pub)
        _assert_bitexact(off(data["query"], _prep(off, qa)), pub(data["query"], _prep(pub, qa)))


@pytest.mark.parametrize("bloom_path", ["partial", "full"])
def test_t6_layer_bloom(data, bloom_path, record_property):
    """The official bloom through the layer (Meta's hash, ``m_bits=None``): the result is the
    exact-mode result on the same index up to bloom false positives: every returned id that
    passes the exact predicate is in the exact top-K (or tied at its boundary), and the two
    bloom paths (partial vs full mask) agree bit for bit. The adapted code has one path and
    refuses ``bloom_path="full"``."""
    if OB == "official-fork" and bloom_path == "full":
        with pytest.raises(ValueError, match="does not apply to backend='official-fork'"):
            _layer(data, OB, "bloom", official=OfficialConfig(bloom_path="full"))
        return
    cfg = OfficialConfig(
        score_path="int32", bloom_path=bloom_path, b_multiplier=B_MULT, n_stored_hashes=HASH_K
    )
    exact = _layer(data, OB, "exact", official=OfficialConfig(score_path="int32"))
    off = _layer(data, OB, "bloom", official=cfg)
    assert off.m_bits == (0 if OB == "official" else 512) and off.k_hash == K_SEARCH
    assert off.bloom_index.dtype == torch.int64 and off.bundle_b_offsets.numel() == N // 2048 + 1
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
                f"row {b}: exact-passing id {i} (score {sc_o[b, j].item()}) missing from "
                f"the exact top-K (k-th score {kth})"
            )
    record_property("bloom_topk_false_positives", fp)
    print(f"T6 bloom[{bloom_path}]: {fp} bloom false positives among {B * K} returned slots")
    if bloom_path == "full":
        off_p = _layer(
            data,
            OB,
            "bloom",
            official=OfficialConfig(
                score_path="int32", b_multiplier=B_MULT, n_stored_hashes=HASH_K
            ),
        )
        _transplant(off_p, exact)
        _assert_bitexact(
            off_p(data["query"], off_p.prepare_queries(data["q_attrs"])), (ids_o, sc_o)
        )


def test_t6_layer_contract(data):
    """Buffers, the candidates path, an attribute-less bloom index, the state-dict round trip."""
    off = _layer(data, OB, "none")
    assert list(off.state_dict()) == [
        "centroids",
        "item_codes",
        "global_scale",
        "cluster_offsets",
        "cluster_sizes",
        "sort_perm",
        "inv_perm",
    ]
    assert torch.equal(off.item_codes, quantize_int8_global(data["embs"])[0][off.sort_perm])
    # The candidate re-rank maps original ids through inv_perm.
    cand = torch.tensor([[10, 20, 30]] * B, device="cuda")
    ids, sc = off(data["query"], candidate_ids=cand)
    assert set(ids.flatten().tolist()) <= {10, 20, 30}
    # State-dict round trip into a freshly registered module of the same shape (the CSR
    # layout's buffer shapes do not depend on the k-means outcome); the load hook re-derives the
    # cached scalars.
    twin = _layer(data, OB, "none")
    twin.load_state_dict(off.state_dict())
    assert twin._probe_width == off._probe_width
    assert twin._global_scale_f == off._global_scale_f
    a, b = off(data["query"]), twin(data["query"])
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    # Bloom configured without attributes: plain queries work, attribute queries raise.
    fork_width = {} if OB == "official" else {"m_bits": 512}
    bare = SilverTorch(
        k=K,
        n_lists=N_LISTS,
        n_probe=N_PROBE,
        filter_mode="bloom",
        k_hash=K_SEARCH,
        n_iter=3,
        backend=OB,
        **fork_width,
    )
    bare.register_index(data["embs"])
    assert bare.bloom_index.numel() == 0
    assert bare(data["query"])[0].shape == (B, K)
    with pytest.raises(RuntimeError, match="without item_clause_attrs"):
        bare(data["query"], bare.prepare_queries(data["q_attrs"]))
    with pytest.raises(ValueError, match="k_hash must be <= 10"):
        SilverTorch(
            k=K,
            n_lists=N_LISTS,
            n_probe=N_PROBE,
            filter_mode="bloom",
            k_hash=11,
            backend=OB,
            **fork_width,
        )


# --- T7: eager only; syncs per op -----------------------------------------------------------


def test_t7_compile_refused(data):
    """The published backend is eager-only: ``module.compile()`` raises, a ``fullgraph``
    ``torch.compile`` of the forward raises (our error or dynamo's ``Unsupported``, both
    ``RuntimeError``s), and a default-mode ``torch.compile`` either raises or falls back to eager
    with identical results, never a silently different path. The adapted code compiles
    ``fullgraph`` to eager's exact result instead."""
    off = _layer(data, OB, "none")
    eager_ids, eager_scores = off(data["query"])
    if OB == "official-fork":
        torch._dynamo.reset()
        try:
            out_ids, out_scores = torch.compile(off, fullgraph=True)(data["query"])
        finally:
            torch._dynamo.reset()
        assert torch.equal(out_ids, eager_ids) and torch.equal(out_scores, eager_scores)
        return
    with pytest.raises(RuntimeError, match="eager-only"):
        off.compile()
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
    """Host syncs ``fn`` performs under ``torch.cuda.set_sync_debug_mode("warn")``. Two disjoint
    instruments: a sync inside a C++ custom op is reported by c10's own warning handler as a
    ``warn_or_error_on_sync`` line on **fd 2** and never becomes a Python warning; a sync from
    Python-dispatched aten (``.item()``, ``nonzero``) surfaces as a Python warning and not on fd
    2. Both are counted; the "prototype feature" notice the mode prints once is not a sync."""
    torch.cuda.synchronize()
    with tempfile.TemporaryFile(mode="w+") as tf, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sys.stderr.flush()
        saved = os.dup(2)
        os.dup2(tf.fileno(), 2)
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
            torch.cuda.synchronize()
        finally:
            torch.cuda.set_sync_debug_mode("default")
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        tf.seek(0)
        from_cpp = sum(1 for line in tf if "warn_or_error_on_sync" in line)
    from_py = sum(
        1
        for w in caught
        if "called a synchronizing" in str(w.message) and "prototype" not in str(w.message)
    )
    return from_cpp + from_py


def test_t7_sync_count_per_op(record_property):
    """The host syncs each official op performs under ``torch.cuda.set_sync_debug_mode("warn")``,
    recorded; asserted ``> 0`` for the published scorer, which is what makes that backend
    uncapturable. On the adapted code both scorers sync 0 times; its upstream-shaped searches
    are the published ones and are only recorded."""
    attrs, offsets, sizes, index, boff = _bloom_corpus()
    n, c, _ = attrs.shape
    codes, _ = quantize_int8_global(make_index(n, 64))
    g = torch.Generator(device="cuda").manual_seed(47)
    q_codes = torch.randint(-128, 128, (4, 64), generator=g, dtype=torch.int8, device="cuda")
    probe_ids = torch.randint(0, sizes.numel(), (4, 4), generator=g, device="cuda")
    qa = make_query_attrs(4, c=c, n_vocab=50, seed=48)
    expressions = of.queries_to_expressions(qa)
    plans = ofb.parse_plans(expressions, HASH_K)
    max_row = 4 * int(sizes[0].item())
    partial = ofb.bloom_partial_masks(
        index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
    )
    counts = {
        # The parse (prepare_queries' work): a CPU op, so 0 device syncs is the expected record;
        # the plan upload is the search ops'.
        "parse_expression_query_batch": _count_syncs(lambda: ofb.parse_plans(expressions, HASH_K)),
        "fused_kmean_ann": _count_syncs(
            lambda: ofb.fused_scores(q_codes, probe_ids, offsets, sizes, codes, max_row)
        ),
        "bloom_index_search_batch_return_partial_response": _count_syncs(
            lambda: ofb.bloom_partial_masks(
                index, boff, plans, offsets[probe_ids], sizes[probe_ids], K_SEARCH, HASH_K
            )
        ),
        "fused_kmean_ann_with_partial_masks": _count_syncs(
            lambda: ofb.fused_scores(
                q_codes, probe_ids, offsets, sizes, codes, max_row, partial=partial
            )
        ),
        "bloom_index_search_batch": _count_syncs(
            lambda: ofb.bloom_full_mask(index, boff, plans, K_SEARCH, HASH_K)
        ),
    }
    for name, n_sync in counts.items():
        record_property(f"syncs_{name}", n_sync)
        print(f"T7 syncs per call: {name}: {n_sync}")
    if OB == "official-fork":
        assert counts["fused_kmean_ann"] == counts["fused_kmean_ann_with_partial_masks"] == 0
        return
    assert counts["fused_kmean_ann"] > 0, "the published scorer syncs the host"


def test_t7_layer_forward_sync_count(data, record_property):
    """Host syncs of one ``SilverTorch`` forward per filter path, the filter prepared beforehand
    (``prepare_queries``): our part (phase 1, the packed exact mask, the epilogue) adds none, so
    every count on the published backend is Meta's ops'; the adapted code's forward syncs 0
    times on every path."""
    cfg = {"b_multiplier": B_MULT, "n_stored_hashes": HASH_K}
    modules = {
        "none": _layer(data, OB, "none"),
        "exact": _layer(data, OB, "exact"),
        "bloom[partial]": _layer(
            data, OB, "bloom", official=OfficialConfig(bloom_path="partial", **cfg)
        ),
    }
    if OB == "official":
        modules["bloom[full]"] = _layer(
            data, OB, "bloom", official=OfficialConfig(bloom_path="full", **cfg)
        )
    for name, module in modules.items():
        prepared = None if name == "none" else module.prepare_queries(data["q_attrs"])
        module(data["query"], prepared)  # warm-up: first-call lazy work is not the steady state
        n_sync = _count_syncs(lambda m=module, p=prepared: m(data["query"], p))
        record_property(f"forward_syncs_{name}", n_sync)
        print(f"T7 forward syncs: {name}: {n_sync}")
        assert n_sync == 0 if OB == "official-fork" else n_sync > 0
