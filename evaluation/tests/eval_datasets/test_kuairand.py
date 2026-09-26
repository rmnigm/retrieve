"""CPU-only tests for the KuaiRand-27K loader (roadmap E4): a six-video, two-user tarball in
the upstream member layout, run through ``convert`` -> ``prep`` -> ``attrs``, with every
output value written out by hand."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import tarfile

import numpy as np
import polars as pl
import pytest
import torch

from eval_datasets.etl import kuairand
from eval_datasets.layout import validate_layout

SHANGHAI = dt.timezone(dt.timedelta(hours=8))
LOG_HEADER = "user_id,video_id,date,hourmin,time_ms,is_click,is_rand,tab"
BASIC_HEADER = (
    "video_id,author_id,video_type,upload_dt,upload_type,visible_status,video_duration,"
    "server_width,server_height,music_id,music_type,tag"
)
CATEGORY_HEADER = ",".join(
    ["final_video_id"]
    + [
        f"{lv}_level_category_{f}"
        for lv in kuairand.CATEGORY_LEVELS
        for f in ("id", "name", "prob")
    ]
)


def _unix(when: str) -> int:
    return int(dt.datetime.fromisoformat(when).replace(tzinfo=SHANGHAI).timestamp())


def _row(user: int, video: int, when: str, click: int, rand: int = 0) -> str:
    t = dt.datetime.fromisoformat(when).replace(tzinfo=SHANGHAI)
    return f"{user},{video},{t:%Y%m%d},{t:%H%M},{int(t.timestamp() * 1000)},{click},{rand},1"


STANDARD = [
    _row(1, 0, "2022-05-01T10:00", 1),
    _row(1, 1, "2022-05-02T10:00", 1),
    _row(1, 2, "2022-05-03T10:00", 0),
    _row(1, 3, "2022-05-03T11:00", 1, rand=1),
    _row(1, 4, "2022-05-06T12:00", 1),
    _row(1, 2, "2022-05-07T09:00", 1),
    _row(1, 3, "2022-05-08T09:00", 1),
    _row(2, 1, "2022-05-01T08:00", 1),
    _row(2, 0, "2022-05-02T08:00", 1),
    _row(2, 5, "2022-05-07T01:00", 1),
]
RANDOM = [_row(2, 4, "2022-05-03T08:00", 1, rand=1)]
BASIC = [
    '0,10,NORMAL,2022-05-01,ShortImport,0.0,10000.0,720.0,1280.0,1,4.0,"12,65"',
    "1,11,AD,2022-04-01,LongImport,0.0,45000.0,720.0,1280.0,2,4.0,65",
    "2,12,NORMAL,2020-07-08,ShortImport,0.0,200000.0,720.0,1280.0,3,4.0,",
    "3,13,NORMAL,2022-05-10,ShortImport,0.0,30000.0,720.0,1280.0,4,4.0,12",
    '4,14,NORMAL,2022-02-01,Web,0.0,60000.0,720.0,1280.0,5,4.0,"7,12,65,8,9"',
    "5,15,NORMAL,,ShortImport,0.0,,720.0,1280.0,6,4.0,12",
]


def _cat(vid: int, *ids: float) -> str:
    ids = ids + (-124.0,) * (4 - len(ids))
    return ",".join([str(vid)] + [f"{i},{'UNKNOWN' if i == -124 else 'n'},0.9" for i in ids])


CATEGORIES = [
    _cat(0, 39.0, 698.0),
    _cat(1, 2.0, 724.0, 2550.0),
    _cat(2, 7.0, 126.0, 1056.0, 3000.0),
    _cat(3, 39.0, 698.0),
    _cat(4, 2.0),
]
MEMBERS = {
    "log_standard_4_08_to_4_21_27k_part1.csv": STANDARD[:4],
    "log_standard_4_08_to_4_21_27k_part2.csv": STANDARD[4:6],
    "log_standard_4_22_to_5_08_27k_part1.csv": STANDARD[6:8],
    "log_standard_4_22_to_5_08_27k_part2.csv": STANDARD[8:],
    "log_random_4_22_to_5_08_27k.csv": RANDOM,
}


def _add(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode()
    info = tarfile.TarInfo(f"KuaiRand-27K/data/{name}")
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


@pytest.fixture
def staged(tmp_path, monkeypatch):
    raw, processed, out = tmp_path / "raw", tmp_path / "processed", tmp_path / "kuairand"
    raw.mkdir()
    monkeypatch.setattr(kuairand, "RAW_DIR", raw)
    monkeypatch.setattr(kuairand, "PROCESSED_DIR", processed)
    monkeypatch.setattr(kuairand, "N_VIDEOS", len(BASIC))
    with tarfile.open(raw / "KuaiRand-27K.tar.gz", "w:gz") as tar:
        _add(tar, "video_features_statistic_27k_part1.csv", "video_id,x\n0,1\n")
        for name, rows in MEMBERS.items():
            _add(tar, name, "\n".join([LOG_HEADER, *rows]) + "\n")
        _add(tar, "video_features_basic_27k.csv", "\n".join([BASIC_HEADER, *BASIC]) + "\n")
        _add(tar, "user_features_27k.csv", "user_id,user_active_degree\n1,full_active\n2,high\n")
    (raw / "kuairand_video_categories.csv").write_text(
        "\n".join([CATEGORY_HEADER, *CATEGORIES]) + "\n"
    )
    assert kuairand.cmd_convert(argparse.Namespace()) == 0
    ns = argparse.Namespace(
        processed_dir=str(processed),
        output_dir=str(out),
        max_seq_len=200,
        test_start="2022-05-07",
        val_days=1,
        gap_minutes=30,
        reference_date="2022-05-07",
    )
    assert kuairand.cmd_prep(ns) == 0
    assert kuairand.cmd_attrs(ns) == 0
    return out


def test_train_windows_cover_every_transition_once():
    seq = np.arange(1, 8)
    got = [w.tolist() for w in kuairand.train_windows(seq, 3)]
    assert got == [[4, 5, 6, 7], [1, 2, 3, 4]]
    assert [w.tolist() for w in kuairand.train_windows(np.arange(1, 9), 3)] == [
        [5, 6, 7, 8],
        [2, 3, 4, 5],
        [1, 2],
    ]
    assert kuairand.train_windows(np.array([9]), 3) == []


def test_cumulative_buckets_boundaries_and_unknowns():
    got = kuairand.cumulative_buckets(
        np.array([10.0, 15.0, 16.0, 180.0, 181.0, np.nan]), (15, 30, 60, 180)
    )
    assert got.tolist() == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 2, 3, -1],
        [3, -1, -1, -1],
        [-1, -1, -1, -1],
        [-1, -1, -1, -1],
    ]


def test_shanghai_midnight_is_utc_plus_8():
    assert kuairand.shanghai_midnight("2022-05-07") == int(
        dt.datetime(2022, 5, 6, 16, tzinfo=dt.timezone.utc).timestamp()
    )


def test_convert_keeps_the_needed_members_and_skips_statistics(staged):
    processed = kuairand.PROCESSED_DIR
    names = sorted(p.name for p in processed.glob("*.parquet"))
    assert names == sorted(
        [v[0] for v in kuairand.TAR_MEMBERS.values()] + ["video_categories.parquet"]
    )
    assert pl.read_parquet(processed / "log_standard_1.parquet").height == 4
    assert json.loads((processed / "convert_log.json").read_text())["log_random.parquet"] == 1


def test_prep_split_drops_non_clicks_and_random_exposures(staged):
    train = pl.read_parquet(staged / "train.parquet").rows()
    val = pl.read_parquet(staged / "val.parquet").rows()
    test = pl.read_parquet(staged / "test.parquet").rows()
    u1 = [_unix(f"2022-05-0{d}T{h}:00") for d, h in ((1, 10), (2, 10), (6, 12))]
    u2 = [_unix("2022-05-01T08:00"), _unix("2022-05-02T08:00")]
    assert train == [([1, 2], u1[:2]), ([2, 1], u2)]
    assert val == [([1, 2], u1[:2], [5])]
    assert test == [([1, 2, 5], u1, [3, 4]), ([2, 1], u2, [6])]
    assert json.loads((staged / "item_id_map.json").read_text()) == {
        str(v): v + 1 for v in range(6)
    }
    log = json.loads((staged / "prep_log.json").read_text())["prep"]
    assert (log["n_standard_is_rand"], log["n_random"], log["n_clicks"]) == (1, 1, 8)


def test_attrs_clause_values_row_i_is_item_i_plus_one(staged):
    attrs = torch.load(staged / "item_attrs_narrow.pt")
    assert attrs.shape == (6, 7, 4)
    assert torch.load(staged / "clause_is_reverse_narrow.pt").tolist() == list(
        kuairand.CLAUSE_IS_REVERSE
    )
    expected = {
        0: [[1], [0], [0, 1], [0], [0], [0, 1, 2, 3], [1, 2, 3]],
        1: [[0], [5, 3], [1], [1], [1], [2, 3], []],
        2: [[2], [1, 4, 2], [], [0], [0], [], []],
        3: [[1], [0], [0], [0], [0], [1, 2, 3], [0, 1, 2, 3]],
        4: [[0], [], [2, 0, 1, 3], [2], [0], [2, 3], []],
        5: [[], [], [0], [0], [0], [], []],
    }
    for row, clauses in expected.items():
        want = [c + [-1] * (4 - len(c)) for c in clauses]
        assert attrs[row].tolist() == want, row


def test_eval_split_target_derived_and_business_rule_queries(staged):
    split = pl.read_parquet(staged / "eval_split.parquet")
    assert split["target_id"].to_list() == [3, 6]
    assert split["query_attrs_narrow"].to_list() == [
        [2, 1, -1, 0, 1, 2, 1],
        [-1, -1, 0, 0, 1, 1, 1],
    ]
    vocab = json.loads((staged / "attr_vocab.json").read_text())
    assert vocab["video_type"] == {"NORMAL": 0, "AD": 1}
    assert vocab["tag"] == {"12": 0, "65": 1, "7": 2, "8": 3, "9": 4}


def test_layout_is_clean(staged):
    assert validate_layout(staged) == []


def test_attrs_logs_exact_pass_rates(staged):
    rates = json.loads((staged / "prep_log.json").read_text())["attrs"]["mean_pass_rate"]
    sixths = {
        "c0_cat1": 1,
        "c1_cat_fine": 1,
        "c2_tag": 4,
        "c3_upload_type": 4,
        "c4_no_ads": 5,
        "c5_duration_max": 3,
        "c6_uploaded_within": 2,
    }
    assert rates == {k: round(v / 6, 6) for k, v in sixths.items()}
