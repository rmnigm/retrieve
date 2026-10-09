"""G-synth: ``eval-data synth-filter`` (docs/system/datasets.md § Synthetic selectivity attrs).
The rates are checked on the flags alone at a large N; the files on the conftest fixture."""

from __future__ import annotations

import json

import pytest
import torch
from click.testing import CliRunner
from conftest import write_tiny_dataset

from bench import oracle
from eval_datasets import cli, layout
from eval_datasets import synth_filter as sf

N_LARGE = 2_000_000


def test_rates_within_one_percent_and_nested():
    u = sf.synth_u(N_LARGE)
    flags = sf.synth_flags(u, sf.RATES)[:, :, 0].bool()
    assert flags.shape == (N_LARGE, len(sf.RATES))
    for j, p in enumerate(sf.RATES):
        if N_LARGE * p >= 1e4:
            assert abs(flags[:, j].sum().item() / (N_LARGE * p) - 1) < 0.01, p
    for j in range(len(sf.RATES) - 1):
        assert not (flags[:, j] & ~flags[:, j + 1]).any()  # pass(p_j) ⊆ pass(p_j+1)
    assert flags[:, -1].all()  # p = 1.0 is the unfiltered point
    assert torch.equal(u, sf.synth_u(N_LARGE)) and not torch.equal(u, sf.synth_u(N_LARGE, 1))


@pytest.mark.parametrize("legacy", [False, True])
def test_build_writes_the_four_files_and_leaves_the_real_attrs(tmp_path, legacy):
    d = write_tiny_dataset(tmp_path / "ds", n=40, u=9, legacy=legacy)
    attrs = d / "item_attrs_narrow.pt"
    before = attrs.read_bytes()
    real = layout.load_item_attrs(attrs, d / "clause_is_reverse_narrow.pt", 40, torch.device("cpu"))
    digest = oracle.attrs_digest(*real)
    meta = sf.build(d)
    assert attrs.read_bytes() == before
    again = layout.load_item_attrs(
        attrs, d / "clause_is_reverse_narrow.pt", 40, torch.device("cpu")
    )
    assert oracle.attrs_digest(*again) == digest
    item = torch.load(d / sf.ITEM_ATTRS, weights_only=True)
    u = torch.load(d / sf.U_FILE, weights_only=True)
    qa = torch.load(d / sf.QUERY_ATTRS, weights_only=True)
    assert (
        item.shape == (40, len(sf.RATES), 1) and item.dtype == torch.long
    )  # the pad row is not an item
    assert torch.equal(item, sf.synth_flags(sf.synth_u(40), sf.RATES))
    assert torch.equal(u, sf.synth_u(40))
    assert qa.dtype == torch.long and qa.shape == (9, len(sf.RATES)) and bool((qa == 1).all())
    assert torch.equal(layout.load_query_attrs(d / sf.QUERY_ATTRS, 9), qa)  # full-split rows
    side = json.loads((d / sf.SIDECAR).read_text())
    assert side == meta and side["seed"] == sf.SEED and side["rates"] == list(sf.RATES)
    assert side["n_items"] == 40 and side["pass_counts"] == item[:, :, 0].sum(0).tolist()
    # The harness reads the synth attrs through the same checked path.
    loaded, rev = layout.load_item_attrs(d / sf.ITEM_ATTRS, None, 40, torch.device("cpu"))
    assert torch.equal(loaded, item) and rev is None


def test_same_seed_is_byte_identical(tmp_path):
    names = (sf.ITEM_ATTRS, sf.U_FILE, sf.QUERY_ATTRS, sf.SIDECAR)
    a, b = (write_tiny_dataset(tmp_path / x, n=40, u=9) for x in "ab")
    sf.build(a)
    sf.build(b)
    assert all((a / f).read_bytes() == (b / f).read_bytes() for f in names)
    sf.build(b, seed=1)
    assert (a / sf.U_FILE).read_bytes() != (b / sf.U_FILE).read_bytes()


def test_cli_resolves_data_dir_from_the_dataset_yaml(tmp_path):
    d = write_tiny_dataset(tmp_path / "data" / "tiny", n=40, u=9)
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "tiny.yaml").write_text("data_dir: data/tiny\n")
    r = CliRunner().invoke(
        cli.main,
        ["synth-filter", "--dataset", "tiny", "--rates", "0.5,1.0", "--config-dir", str(cfg)],
    )
    assert r.exit_code == 0, r.output
    assert torch.load(d / sf.ITEM_ATTRS, weights_only=True).shape == (40, 2, 1)
    assert json.loads((d / sf.SIDECAR).read_text())["rates"] == [0.5, 1.0]


def _write_blobs(d, n_blobs: int, per_blob: int, n_queries: int):
    """Equal, well-separated blobs around random unit centres; each query near one centre."""
    g = torch.Generator().manual_seed(1)
    centres = torch.nn.functional.normalize(torch.randn(n_blobs, 16, generator=g), dim=-1)
    items = centres.repeat_interleave(per_blob, 0) + 0.01 * torch.randn(
        n_blobs * per_blob, 16, generator=g
    )
    queries = centres[torch.arange(n_queries) % n_blobs] + 0.01 * torch.randn(
        n_queries, 16, generator=g
    )
    torch.save(items.half(), d / "content" / "text_emb.pt")
    torch.save(queries.half(), d / "content" / "query_emb.pt")


def test_correlated_pass_rate_and_the_query_own_cluster(tmp_path):
    d = write_tiny_dataset(tmp_path / "ds", n=400, u=30)
    _write_blobs(d, n_blobs=10, per_blob=40, n_queries=30)
    meta = sf.build_correlated(d, d / "content", rates=(0.1, 0.5))
    item = torch.load(d / sf.CORR_ITEM_ATTRS, weights_only=True)
    qa = torch.load(d / sf.CORR_QUERY_ATTRS, weights_only=True)
    assert item.shape == (400, 2, 1) and item.dtype == torch.long
    assert torch.equal(layout.load_query_attrs(d / sf.CORR_QUERY_ATTRS, 30), qa)
    side = json.loads((d / sf.CORR_SIDECAR).read_text())
    assert side == meta and side["n_items"] == 400 and side["rates"] == [0.1, 0.5]
    items = layout.load_text_items(d / "content", torch.device("cpu"))
    queries, _, _ = layout.load_text_queries(d, d / "content", 16)
    for j, r in enumerate(side["per_rate"]):
        passes = item[:, j, 0][None, :] == qa[:, j][:, None]  # [U, N]
        rate = passes.double().mean(1)
        # rel 1e-12: float64 means of the same 30 values, summed in another order
        assert r["query_pass_rate"]["mean"] == pytest.approx(rate.mean().item(), rel=1e-12)
        # Items and queries both take their nearest unit centroid: the least squared distance.
        c = torch.tensor(r["centroids"])
        assert torch.equal(qa[:, j], ((queries[:, None] - c[None]) ** 2).sum(-1).argmin(1))
        assert torch.equal(item[:, j, 0], ((items[:, None] - c[None]) ** 2).sum(-1).argmin(1))
    # 10 equal blobs, k = 10. rel 0.5: a random-init local optimum may split one blob and
    # merge two (here 0.134).
    assert side["per_rate"][0]["query_pass_rate"]["mean"] == pytest.approx(0.1, rel=0.5)
    # The pass set sits around the query: most of its 10 nearest items pass (uncorrelated
    # values would give about p = 0.1; 0.8 leaves room for the boundary queries of a split blob).
    near = (queries @ items.T).topk(10, dim=1).indices
    assert (item[near, 0, 0] == qa[:, :1]).double().mean().item() > 0.8
    assert set(side) >= {"seed", "kmeans", "content_dir", "n_queries"}


def test_correlated_leaves_the_real_and_uniform_attrs(tmp_path):
    d = write_tiny_dataset(tmp_path / "ds", n=400, u=30)
    _write_blobs(d, n_blobs=10, per_blob=40, n_queries=30)
    sf.build(d)
    before = {
        f: (d / f).read_bytes() for f in ("item_attrs_narrow.pt", sf.ITEM_ATTRS, sf.QUERY_ATTRS)
    }
    sf.build_correlated(d, d / "content", rates=(0.1,))
    assert all((d / f).read_bytes() == b for f, b in before.items())
    first = (d / sf.CORR_ITEM_ATTRS).read_bytes(), (d / sf.CORR_QUERY_ATTRS).read_bytes()
    sf.build_correlated(d, d / "content", rates=(0.1,))
    assert first == ((d / sf.CORR_ITEM_ATTRS).read_bytes(), (d / sf.CORR_QUERY_ATTRS).read_bytes())


def test_cli_correlated_reads_the_content_dir_at_dim(tmp_path):
    d = write_tiny_dataset(tmp_path / "data" / "tiny", n=400, u=30)
    _write_blobs(d, n_blobs=10, per_blob=40, n_queries=30)
    cfg = tmp_path / "config"
    cfg.mkdir()
    (cfg / "tiny.yaml").write_text("data_dir: data/tiny\ncontent_dir: {16: content}\n")
    args = ["synth-filter", "--dataset", "tiny", "--correlated", "--dim", "16"]
    r = CliRunner().invoke(cli.main, [*args, "--rates", "0.1", "--config-dir", str(cfg)])
    assert r.exit_code == 0, r.output
    assert torch.load(d / sf.CORR_ITEM_ATTRS, weights_only=True).shape == (400, 1, 1)
    assert not (d / sf.ITEM_ATTRS).exists()
    (cfg / "seq.yaml").write_text("data_dir: data/tiny\n")
    r = CliRunner().invoke(
        cli.main, ["synth-filter", "--dataset", "seq", "--correlated", "--config-dir", str(cfg)]
    )
    assert r.exit_code != 0 and "has no content_dir" in r.output


def test_synth_attrs_for_other_rates_are_refused(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    (d / sf.SIDECAR).write_text(json.dumps({"rates": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]}))
    with pytest.raises(ValueError, match="rerun eval-data synth-filter"):
        layout.check_synth_rates(d / sf.ITEM_ATTRS)
    (d / sf.SIDECAR).write_text(json.dumps({"rates": list(sf.RATES)}))
    layout.check_synth_rates(d / sf.ITEM_ATTRS)
