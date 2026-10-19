import polars as pl


def _drop_non_train(df: pl.LazyFrame, unique_train_item_ids: pl.DataFrame) -> pl.LazyFrame:
    return df.select(
        "uid",
        pl.all()
        .exclude("uid")
        .list.gather(
            pl.col("item_id").list.eval(
                pl.arg_where(
                    pl.element().is_in(unique_train_item_ids.get_column("item_id").implode())
                )
            )
        ),
    ).filter(pl.col("item_id").list.len() > 0)


def sequential_split_train_val_test(
    df: pl.LazyFrame,
    test_timestamp: int,
    val_size: int,
    gap_size: int,
    drop_non_train_items: bool = False,
    engine: str = "streaming",
) -> tuple[pl.LazyFrame, pl.LazyFrame | None, pl.LazyFrame]:
    """``(train, val | None, test)`` by timestamp: test from ``test_timestamp``, validation the
    ``val_size`` before it, a ``gap_size`` between the segments; users and items absent from
    training are dropped from the later segments."""
    train_timestamp = test_timestamp - gap_size - val_size - (gap_size if val_size != 0 else 0)
    if gap_size < 0 or val_size < 0 or train_timestamp <= 0:
        raise ValueError(
            f"need gap_size, val_size >= 0 and a positive train cutoff: gap_size={gap_size}, "
            f"val_size={val_size}, train_timestamp={train_timestamp}"
        )

    df_lazy = df.lazy()

    train = df_lazy.select(
        "uid",
        pl.all()
        .exclude("uid")
        .list.gather(pl.col("timestamp").list.eval(pl.arg_where(pl.element() < train_timestamp))),
    ).filter(pl.col("item_id").list.len() > 0)

    unique_train_uids = train.select("uid").unique().collect(engine=engine)
    unique_train_item_ids = (
        train.explode("item_id").select("item_id").unique().collect(engine=engine)
    )

    validation = None
    if val_size != 0:
        validation = (
            df_lazy.select(
                "uid",
                pl.all()
                .exclude("uid")
                .list.gather(
                    pl.col("timestamp").list.eval(
                        pl.arg_where(
                            (pl.element() >= test_timestamp - val_size - gap_size)
                            & (pl.element() < test_timestamp - gap_size)
                        )
                    )
                ),
            )
            .with_columns(
                pl.col("uid")
                .is_in(unique_train_uids.get_column("uid").implode())
                .alias("uid_in_train")
            )  # to prevent filter reordering
            .filter("uid_in_train")
            .drop("uid_in_train")
        )

        if drop_non_train_items:
            validation = _drop_non_train(validation, unique_train_item_ids)
        validation = validation.filter(pl.col("item_id").list.len() > 0)

    test = (
        df_lazy.select(
            "uid",
            pl.all()
            .exclude("uid")
            .list.gather(
                pl.col("timestamp").list.eval(pl.arg_where(pl.element() >= test_timestamp))
            ),
        )
        .with_columns(
            pl.col("uid").is_in(unique_train_uids.get_column("uid").implode()).alias("uid_in_train")
        )  # to prevent filter reordering
        .filter("uid_in_train")
        .drop("uid_in_train")
    )

    if drop_non_train_items:
        test = _drop_non_train(test, unique_train_item_ids)
    test = test.filter(pl.col("item_id").list.len() > 0)

    return train, validation, test
