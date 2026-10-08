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
    assert item.shape == (40, 7, 1) and item.dtype == torch.long  # the pad row is not an item
    assert torch.equal(item, sf.synth_flags(sf.synth_u(40), sf.RATES))
    assert torch.equal(u, sf.synth_u(40))
    assert qa.dtype == torch.long and qa.shape == (9, 7) and bool((qa == 1).all())
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
