"""Yambda Listen+ preprocessing.

Library entrypoint:

    from datasets.yambda import preprocess

CLI (replaces the old `scripts/prep_yambda.py`):

    uv run yambda prep --variant 500m --output-dir data/yambda/500m-listens

Adapted from the original `articles/yambda-benchmarks/sasrec/data.py::preprocess()`
(commit 4594c93). We keep only the listens-Listen+ branch and the function
contract — the rest of the original file (TrainDataset/EvalDataset/collate_fn)
is replaced by `evaluation.training.dataset` and `evaluation.training.evaluate`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from . import timesplit
from .constants import Constants


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

    train = train.select("uid", pl.all().exclude("uid").list.slice(-max_seq_len, max_seq_len))
    train = replace_strict(train)
    if val is not None:
        val = replace_strict(val)
    test = replace_strict(test)

    return Data(train=train, validation=val, test=test, item_id_to_idx=item_id_to_idx)


# ----- CLI -------------------------------------------------------------------
#
# Downloads `<variant>/sequential/listens.parquet` from `yandex/yambda` on HF,
# runs `preprocess()`, and writes the four artifacts the trainer expects:
#
#     <output>/train.parquet     item_ids: list[int64]
#     <output>/val.parquet       item_ids: list[int64], targets: list[int64]
#     <output>/test.parquet      item_ids: list[int64], targets: list[int64]
#     <output>/item_id_map.json  {raw_yandex_id: dense_int}
#
# Val history = train portion (already sliced to last 200 in preprocess()).
# Test history = train ++ val concatenated, last 200 items.


def concat_then_tail(history: pl.Expr, more: pl.Expr, n: int) -> pl.Expr:
    return history.list.concat(more).list.slice(-n, n)


def cmd_prep(args) -> int:
    from loguru import logger

    from datasets.hf_io import download_raw_file

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    hf_path = f"sequential/{args.variant}/{args.interaction}.parquet"
    logger.info("Downloading {} from HF dataset 'yambda'…", hf_path)
    local = download_raw_file("yambda", hf_path)
    logger.info("Got parquet at {}", local)

    df = pl.scan_parquet(local)
    logger.info("Running Listen+ preprocess (filter + temporal split + dense id map)…")
    data = preprocess(df, interaction=args.interaction, max_seq_len=args.max_seq_len)
    n_items = len(data.item_id_to_idx)
    logger.info("Dense item catalog: {} items", n_items)

    train_lf = data.train
    val_lf = data.validation
    test_lf = data.test
    assert val_lf is not None, "val_size>0 should produce a validation split"

    map_path = output / "item_id_map.json"
    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in data.item_id_to_idx.items()}, f)
    logger.info("Wrote {} ({} entries)", map_path, n_items)

    train_out = train_lf.select(pl.col("item_id").alias("item_ids")).collect(engine="streaming")
    train_path = output / "train.parquet"
    train_out.write_parquet(train_path, compression="zstd")
    logger.info("Wrote {} (n_users={})", train_path, train_out.height)

    train_hist = train_lf.select("uid", pl.col("item_id").alias("history"))

    val_join = (
        train_hist.join(
            val_lf.select("uid", pl.col("item_id").alias("targets")), on="uid", how="inner"
        )
        .select(pl.col("history").alias("item_ids"), pl.col("targets"))
        .collect(engine="streaming")
    )
    val_path = output / "val.parquet"
    val_join.write_parquet(val_path, compression="zstd")
    logger.info("Wrote {} (n_users={})", val_path, val_join.height)

    train_plus_val = (
        train_hist.join(
            val_lf.select("uid", pl.col("item_id").alias("val_items")), on="uid", how="left"
        )
        .with_columns(
            pl.when(pl.col("val_items").is_null())
            .then(pl.col("history"))
            .otherwise(concat_then_tail(pl.col("history"), pl.col("val_items"), args.max_seq_len))
            .alias("history")
        )
        .select("uid", "history")
    )
    test_join = (
        train_plus_val.join(
            test_lf.select("uid", pl.col("item_id").alias("targets")), on="uid", how="inner"
        )
        .select(pl.col("history").alias("item_ids"), pl.col("targets"))
        .collect(engine="streaming")
    )
    test_path = output / "test.parquet"
    test_join.write_parquet(test_path, compression="zstd")
    logger.info("Wrote {} (n_users={})", test_path, test_join.height)

    logger.info("Done. Output: {}", output)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Yambda Listen+ download + preprocess (HF yandex/yambda)."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_pp = sub.add_parser("prep", help="download + preprocess into trainer-format parquets")
    sp_pp.add_argument("--variant", choices=["50m", "500m", "5b"], required=True)
    sp_pp.add_argument("--interaction", choices=["listens"], default="listens")
    sp_pp.add_argument("--output-dir", type=str, required=True)
    sp_pp.add_argument("--max-seq-len", type=int, default=200)
    sp_pp.set_defaults(func=cmd_prep)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Data", "preprocess", "main"]
