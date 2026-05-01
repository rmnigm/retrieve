"""Yambda → trainer-format prep.

Downloads `<variant>/sequential/listens.parquet` from `yandex/yambda` on HF,
runs `evaluation.data.preprocess.preprocess()` (Listen+ filter + temporal
split + 1-indexed dense id map), and writes the four artifacts the trainer
expects:

    <output>/train.parquet   columns: item_ids: list[int64]
    <output>/val.parquet     columns: item_ids: list[int64], targets: list[int64]
    <output>/test.parquet    columns: item_ids: list[int64], targets: list[int64]
    <output>/item_id_map.json   {raw_yandex_id: dense_int}

Val history = train portion (already sliced to last 200 in preprocess).
Test history = train ++ val concatenated, last 200 items.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import click
import polars as pl
from huggingface_hub import hf_hub_download
from loguru import logger

from data.yambda import preprocess


def _concat_then_tail(history: pl.Expr, more: pl.Expr, n: int) -> pl.Expr:
    return history.list.concat(more).list.slice(-n, n)


@click.command()
@click.option("--variant", type=click.Choice(["50m", "500m", "5b"]), required=True)
@click.option("--interaction", type=click.Choice(["listens"]), default="listens")
@click.option("--hf-repo", default="yandex/yambda")
@click.option("--output", type=click.Path(path_type=Path), required=True)
@click.option("--max-seq-len", type=int, default=200)
def main(
    variant: str,
    interaction: str,
    hf_repo: str,
    output: Path,
    max_seq_len: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)

    hf_path = f"sequential/{variant}/{interaction}.parquet"
    logger.info("Downloading {} from HF dataset {}…", hf_path, hf_repo)
    local = hf_hub_download(
        repo_id=hf_repo,
        repo_type="dataset",
        filename=hf_path,
        cache_dir=os.environ.get("HF_HOME"),
    )
    logger.info("Got parquet at {}", local)

    df = pl.scan_parquet(local)
    logger.info("Running Listen+ preprocess (filter + temporal split + dense id map)…")
    data = preprocess(df, interaction=interaction, max_seq_len=max_seq_len)
    n_items = len(data.item_id_to_idx)
    logger.info("Dense item catalog: {} items", n_items)

    train_lf = data.train
    val_lf = data.validation
    test_lf = data.test
    assert val_lf is not None, "val_size>0 should produce a validation split"

    # Write item id map.
    map_path = output / "item_id_map.json"
    with open(map_path, "w") as f:
        json.dump({str(k): v for k, v in data.item_id_to_idx.items()}, f)
    logger.info("Wrote {} ({} entries)", map_path, n_items)

    # train.parquet: just rename item_id → item_ids and keep one row per user.
    train_out = train_lf.select(pl.col("item_id").alias("item_ids")).collect(engine="streaming")
    train_path = output / "train.parquet"
    train_out.write_parquet(train_path, compression="zstd")
    logger.info("Wrote {} (n_users={})", train_path, train_out.height)

    # We need train history (sliced to max_seq_len) to compose val/test eval frames.
    train_hist = train_lf.select("uid", pl.col("item_id").alias("history"))

    # val.parquet: history = train item_ids, targets = val item_ids (intact list).
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

    # test.parquet: history = (train_history ++ val_items) last max_seq_len; targets = test items.
    train_plus_val = (
        train_hist.join(
            val_lf.select("uid", pl.col("item_id").alias("val_items")), on="uid", how="left"
        )
        .with_columns(
            pl.when(pl.col("val_items").is_null())
            .then(pl.col("history"))
            .otherwise(_concat_then_tail(pl.col("history"), pl.col("val_items"), max_seq_len))
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


if __name__ == "__main__":
    main()
