"""Yambda Listen+ preprocessing.

Adapted from the original `articles/yambda-benchmarks/sasrec/data.py::preprocess()`
(commit 4594c93). We keep only the listens-Listen+ branch and the function
contract — the rest of the file (TrainDataset/EvalDataset/collate_fn) is
replaced by `evaluation.training.dataset` and `evaluation.training.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .constants import Constants
from .processing import timesplit


@dataclass
class Data:
    train: pl.LazyFrame
    validation: pl.LazyFrame | None
    test: pl.LazyFrame
    item_id_to_idx: dict[int, int]

    @property
    def num_items(self) -> int:
        return len(self.item_id_to_idx)


def preprocess(
    df: pl.LazyFrame,
    interaction: str = "listens",
    val_size: int = Constants.VAL_SIZE,
    max_seq_len: int = 200,
) -> Data:
    """Filter Listen+ events, build a 1-indexed dense id map, temporally split.

    Input `df` must already be in the *sequential* layout: one row per `uid`
    with list-typed `item_id`, `timestamp`, and (for listens) `played_ratio_pct`
    columns. The output train list is sliced to the last `max_seq_len` items;
    val/test lists are left intact so the prep script can compose history+targets.
    """
    if interaction == "listens":
        df = df.select(
            "uid",
            pl.col("item_id", "timestamp").list.gather(
                pl.col("played_ratio_pct").list.eval(
                    pl.arg_where(pl.element() >= Constants.TRACK_LISTEN_THRESHOLD)
                )
            ),
        ).filter(pl.col("item_id").list.len() > 0)

    unique_item_ids = (
        df.select(pl.col("item_id").explode().unique().sort())
        .collect(engine="streaming")["item_id"]
        .to_list()
    )
    item_id_to_idx = {int(item_id): i + 1 for i, item_id in enumerate(unique_item_ids)}

    train, val, test = timesplit.sequential_split_train_val_test(
        df,
        val_size=val_size,
        test_timestamp=Constants.TEST_TIMESTAMP,
        drop_non_train_items=False,
    )

    def replace_strict(frame: pl.LazyFrame) -> pl.LazyFrame:
        return (
            frame.select(
                pl.col("item_id").list.eval(pl.element().replace_strict(item_id_to_idx)),
                pl.all().exclude("item_id"),
            )
            .collect(engine="streaming")
            .lazy()
        )

    train = train.select(
        "uid", pl.all().exclude("uid").list.slice(-max_seq_len, max_seq_len)
    )
    train = replace_strict(train)
    if val is not None:
        val = replace_strict(val)
    test = replace_strict(test)

    return Data(train=train, validation=val, test=test, item_id_to_idx=item_id_to_idx)
