"""KuaiRand-27K ETL: 27,285 Kuaishou users, 32,038,725 videos, one month of logs.

Subcommands::

    download   KuaiRand-27K.tar.gz + kuairand_video_categories.csv from Zenodo into
               data/_raw/kuairand/raw/ (resumable, md5-verified).
    convert    Stream the tarball once into ZSTD parquet under processed/ (the video
               statistics CSVs are skipped), plus the category supplement.
    prep       Clicked, non-random interactions -> train/val/test.parquet + item_id_map.json
               over the full 32M catalog.
    attrs      item_attrs_narrow.pt [N, 7, 4] + clause_is_reverse_narrow.pt + vocabs +
               eval_split.parquet (target-derived and business-rule clauses).
    all        download -> convert -> prep -> attrs.

Semantics, clause layout and the split: docs/system/datasets.md § kuairand.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from eval_datasets.common import file_hexdigest, merge_prep_log, synthesize_qa_narrow
from eval_datasets.hub import raw_dir
from eval_datasets.timesplit import sequential_split_train_val_test

ROOT = raw_dir("kuairand")
RAW_DIR = ROOT / "raw"
PROCESSED_DIR = ROOT / "processed"

RAW_FILES: dict[str, tuple[str, int, str]] = {
    "KuaiRand-27K.tar.gz": (
        "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz",
        9_892_191_178,
        "3e3c799a24e2d23a4d2c757fbf9adf59",
    ),
    "kuairand_video_categories.csv": (
        "https://zenodo.org/records/18159199/files/kuairand_video_categories.csv",
        3_687_928_520,
        "8a4c772d3d8f8cf65fa1e1eb896f4177",
    ),
}

N_VIDEOS = 32_038_725
TZ = dt.timezone(dt.timedelta(hours=8), "Asia/Shanghai")  # no DST since 1991
DAY = 24 * 60 * 60

LOG_TYPES = {
    "user_id": pa.int32(),
    "video_id": pa.int32(),
    "date": pa.int32(),
    "hourmin": pa.int32(),
    "time_ms": pa.int64(),
    "is_click": pa.int8(),
    "is_like": pa.int8(),
    "is_follow": pa.int8(),
    "is_comment": pa.int8(),
    "is_forward": pa.int8(),
    "is_hate": pa.int8(),
    "long_view": pa.int8(),
    "play_time_ms": pa.int64(),
    "duration_ms": pa.int64(),
    "profile_stay_time": pa.int64(),
    "comment_stay_time": pa.int64(),
    "is_profile_enter": pa.int8(),
    "is_rand": pa.int8(),
    "tab": pa.int8(),
}
VIDEO_BASIC_TYPES = {
    "video_id": pa.int32(),
    "author_id": pa.int64(),
    "video_type": pa.string(),
    "upload_dt": pa.string(),
    "upload_type": pa.string(),
    "visible_status": pa.float64(),
    "video_duration": pa.float64(),
    "server_width": pa.float64(),
    "server_height": pa.float64(),
    "music_id": pa.int64(),
    "music_type": pa.float64(),
    "tag": pa.string(),
}
CATEGORY_LEVELS = ("first", "second", "third", "fourth")
CATEGORY_UNKNOWN = -124

#: tar member (basename) -> (processed parquet, column types or None to infer)
TAR_MEMBERS: dict[str, tuple[str, dict | None]] = {
    "log_standard_4_08_to_4_21_27k_part1.csv": ("log_standard_1.parquet", LOG_TYPES),
    "log_standard_4_08_to_4_21_27k_part2.csv": ("log_standard_2.parquet", LOG_TYPES),
    "log_standard_4_22_to_5_08_27k_part1.csv": ("log_standard_3.parquet", LOG_TYPES),
    "log_standard_4_22_to_5_08_27k_part2.csv": ("log_standard_4.parquet", LOG_TYPES),
    "log_random_4_22_to_5_08_27k.csv": ("log_random.parquet", LOG_TYPES),
    "video_features_basic_27k.csv": ("video_features_basic.parquet", VIDEO_BASIC_TYPES),
    "user_features_27k.csv": ("user_features.parquet", None),
}


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fetch(url: str, dest: Path, expected: int) -> None:
    have = dest.stat().st_size if dest.exists() else 0
    if have > expected:
        dest.unlink()
        have = 0
    if have == expected:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req, timeout=120) as resp:
        if have and resp.status != 206:
            have = 0
        with open(dest, "ab" if have else "wb") as out:
            while chunk := resp.read(1 << 22):
                out.write(chunk)


def cmd_download(args: argparse.Namespace) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, size, md5) in RAW_FILES.items():
        dest = RAW_DIR / name
        for attempt in range(1, args.retries + 1):
            try:
                _fetch(url, dest, size)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                _log(f"retry {attempt}/{args.retries} {name}: {e}")
                continue
            if dest.stat().st_size == size:
                break
        got = file_hexdigest(dest, "md5")
        if got != md5:
            _log(f"ERROR {name}: md5 {got} != {md5}; delete it and re-run")
            return 1
        _log(f"OK {name} ({size / 1e9:.2f} GB, md5 verified)")
    return 0


def csv_to_parquet(src: Path | IO[bytes], dst: Path, column_types: dict | None) -> int:
    """Stream one CSV (a path or a binary file object) into ZSTD parquet; returns the rows."""
    reader = pacsv.open_csv(
        src,
        read_options=pacsv.ReadOptions(block_size=64 << 20),
        convert_options=pacsv.ConvertOptions(column_types=column_types or {}),
    )
    tmp = dst.with_name(dst.name + ".tmp")
    rows = 0
    with pq.ParquetWriter(tmp, reader.schema, compression="zstd") as writer:
        for batch in reader:
            writer.write_batch(batch)
            rows += batch.num_rows
    tmp.replace(dst)
    return rows


def cmd_convert(args: argparse.Namespace) -> int:
    tarball = RAW_DIR / "KuaiRand-27K.tar.gz"
    categories = RAW_DIR / "kuairand_video_categories.csv"
    for p in (tarball, categories):
        if not p.exists():
            _log(f"ERROR missing {p} (run `kuairand download` first)")
            return 1
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows: dict[str, int] = {}
    pending = {k: v for k, v in TAR_MEMBERS.items() if not (PROCESSED_DIR / v[0]).exists()}
    if pending:
        with tarfile.open(tarball, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if name not in pending:
                    continue
                out, types = pending.pop(name)
                _log(f"START {name}")
                rows[out] = csv_to_parquet(tar.extractfile(member), PROCESSED_DIR / out, types)
                _log(f"DONE {name} -> {out} ({rows[out]:,} rows)")
        if pending:
            _log(f"ERROR tarball lacks {sorted(pending)}")
            return 1
    out = PROCESSED_DIR / "video_categories.parquet"
    if not out.exists():
        _log("START kuairand_video_categories.csv")
        rows[out.name] = csv_to_parquet(categories, out, None)
    (PROCESSED_DIR / "convert_log.json").write_text(json.dumps(rows, indent=2))
    _log(f"ALL DONE convert {rows}")
    return 0


def shanghai_midnight(day: str) -> int:
    """Unix seconds of 00:00 Asia/Shanghai on ``day`` (YYYY-MM-DD); the logs' ``date`` is local."""
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=TZ).timestamp())


def train_windows(seq: np.ndarray, max_seq_len: int) -> list[np.ndarray]:
    """Non-overlapping ``max_seq_len`` transitions, cut from the end: windows of ``L + 1`` ids
    sharing one boundary id, so every consecutive pair of ``seq`` is trained exactly once."""
    return [seq[max(0, end - max_seq_len - 1) : end] for end in range(len(seq), 1, -max_seq_len)]


def load_clicks(processed: Path) -> tuple[pl.DataFrame, dict]:
    """``(uid, video_id, ts)`` of every clicked, non-random standard-log impression."""
    logs = pl.scan_parquet([processed / f"log_standard_{i}.parquet" for i in (1, 2, 3, 4)])
    stats = logs.select(
        pl.len().alias("n_standard"),
        pl.col("is_click").sum().alias("n_standard_click"),
        pl.col("is_rand").sum().alias("n_standard_is_rand"),
    ).collect()
    random = pl.scan_parquet(processed / "log_random.parquet").select(
        pl.len().alias("n_random"), pl.col("is_rand").sum().alias("n_random_is_rand")
    )
    log = stats.to_dicts()[0] | random.collect().to_dicts()[0]
    clicks = (
        logs.filter((pl.col("is_click") == 1) & (pl.col("is_rand") == 0))
        .select(
            pl.col("user_id").alias("uid"),
            "video_id",
            (pl.col("time_ms") // 1000).alias("ts"),
            "time_ms",
        )
        .sort(["uid", "time_ms", "video_id"])
        .drop("time_ms")
        .collect()
    )
    log["n_clicks"] = clicks.height
    return clicks, {k: int(v) for k, v in log.items()}


def write_item_id_map(path: Path, n: int) -> None:
    """Identity catalog: ``video_id v`` -> item id ``v + 1`` (upstream ids are dense 0..N-1)."""
    with open(path, "w") as f:
        f.write("{")
        f.write(",".join(f'"{v}":{v + 1}' for v in range(n)))
        f.write("}")


def cmd_prep(args: argparse.Namespace) -> int:
    processed = Path(args.processed_dir).expanduser()
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    L = args.max_seq_len

    videos = pl.scan_parquet(processed / "video_features_basic.parquet").select("video_id")
    n_videos, vmin, vmax = (
        videos.select(
            pl.len(), pl.col("video_id").min().alias("lo"), pl.col("video_id").max().alias("hi")
        )
        .collect()
        .row(0)
    )
    if (n_videos, vmin, vmax) != (N_VIDEOS, 0, N_VIDEOS - 1):
        _log(f"ERROR video ids are not dense 0..{N_VIDEOS - 1}: {(n_videos, vmin, vmax)}")
        return 1
    write_item_id_map(output / "item_id_map.json", N_VIDEOS)

    _log("STEP load clicks")
    clicks, log = load_clicks(processed)
    seq = (
        clicks.with_columns((pl.col("video_id") + 1).cast(pl.Int64).alias("item_id"))
        .group_by("uid", maintain_order=True)
        .agg("item_id", pl.col("ts").alias("timestamp"))
    )
    del clicks
    log |= {"n_users": seq.height, "n_items": N_VIDEOS}

    test_ts = shanghai_midnight(args.test_start)
    gap = args.gap_minutes * 60
    _log(f"STEP time-split (test from {args.test_start} = {test_ts})")
    train_lf, val_lf, test_lf = sequential_split_train_val_test(
        seq.lazy(),
        test_timestamp=test_ts,
        val_size=args.val_days * DAY,
        gap_size=gap,
        drop_non_train_items=False,
    )
    train = train_lf.select(
        "uid", pl.col("item_id").alias("history"), pl.col("timestamp").alias("history_ts")
    ).collect()
    val = val_lf.select(
        "uid", pl.col("item_id").alias("targets"), pl.col("timestamp").alias("val_ts")
    ).collect()
    test = test_lf.select("uid", pl.col("item_id").alias("targets")).collect()

    windows = [
        (w.tolist(), t.tolist())
        for h, ts in train.select("history", "history_ts").iter_rows()
        for w, t in zip(
            train_windows(np.asarray(h, dtype=np.int64), L),
            train_windows(np.asarray(ts, dtype=np.int64), L),
            strict=True,
        )
    ]
    pl.DataFrame(
        windows,
        schema={"item_ids": pl.List(pl.Int64), "timestamps": pl.List(pl.Int64)},
        orient="row",
    ).write_parquet(output / "train.parquet", compression="zstd")
    val_rows = train.join(val, on="uid", how="inner").select(
        pl.col("history").list.tail(L).alias("item_ids"),
        pl.col("history_ts").list.tail(L).alias("timestamps"),
        "targets",
    )
    val_rows.write_parquet(output / "val.parquet", compression="zstd")
    test_rows = (
        train.join(val.rename({"targets": "val_items"}), on="uid", how="left")
        .join(test, on="uid", how="inner")
        .sort("uid")
        .select(
            "uid",
            pl.col("history")
            .list.concat(pl.col("val_items").fill_null([]))
            .list.tail(L)
            .alias("item_ids"),
            pl.col("history_ts")
            .list.concat(pl.col("val_ts").fill_null([]))
            .list.tail(L)
            .alias("timestamps"),
            "targets",
        )
    )
    test_rows.select("item_ids", "timestamps", "targets").write_parquet(
        output / "test.parquet", compression="zstd"
    )
    test_rows.select("uid").write_parquet(output / "test_users.parquet", compression="zstd")

    history_lens = train["history"].list.len()
    log |= {
        "test_start": args.test_start,
        "test_timestamp_unix": test_ts,
        "val_days": args.val_days,
        "gap_minutes": args.gap_minutes,
        "max_seq_len": L,
        "train_users": train.height,
        "train_rows": len(windows),
        "train_positions": int((history_lens - 1).clip(lower_bound=0).sum()),
        "median_train_history": int(history_lens.median()),
        "val_rows": val_rows.height,
        "test_rows": test_rows.height,
        "median_test_targets": int(test_rows["targets"].list.len().median()),
        "items_clicked_in_train": int(train["history"].explode().n_unique()),
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    merge_prep_log(output, "prep", log)
    _log(f"ALL DONE prep {log}")
    return 0


# Clause layout (docs/system/datasets.md § kuairand): C0-C3 carry target-derived query
# values, C4-C6 business-rule ones; the config's sweeps pick one protocol or the other.

C_NARROW = 7
A_MAX = 4
CLAUSE_NAMES = (
    "c0_cat1",
    "c1_cat_fine",
    "c2_tag",
    "c3_upload_type",
    "c4_no_ads",
    "c5_duration_max",
    "c6_uploaded_within",
)
CLAUSE_IS_REVERSE = (False, False, False, False, True, False, False)
DURATION_MAX_S = (15, 30, 60, 180)
UPLOAD_WITHIN_DAYS = (3, 7, 14, 30)
AD_VIDEO_TYPE = "AD"


def cumulative_buckets(values: np.ndarray, thresholds: tuple[int, ...]) -> np.ndarray:
    """``[n, len(thresholds)]`` float ``values`` (NaN = unknown) as cumulative threshold ids:
    an item carries every ``j`` with ``value <= thresholds[j]``, tightest first, so the range
    predicate ``value <= t_j`` is the equality clause ``j`` and a target-derived query value
    (the first slot) is the tightest cap the target fits under."""
    k = len(thresholds)
    first = np.searchsorted(np.asarray(thresholds, dtype=np.float64), values, side="left")
    slots = first[:, None] + np.arange(k)[None, :]
    slots[(slots >= k) | np.isnan(values)[:, None]] = -1
    return slots


def dense_vocab(values: pl.Series) -> dict:
    """Dense 0-indexed ids by descending frequency (ties on value), nulls excluded."""
    freq = (
        values.drop_nulls()
        .value_counts(name="n")
        .sort(["n", values.name], descending=[True, False])
    )
    return {v: i for i, v in enumerate(freq[values.name].to_list())}


def encode(values: pl.Series, vocab: dict) -> np.ndarray:
    return values.replace_strict(vocab, default=-1, return_dtype=pl.Int64).to_numpy()


def pack_left(slots: np.ndarray) -> np.ndarray:
    """Move each row's non-``-1`` entries to the front, keeping their order."""
    order = np.argsort(slots == -1, axis=1, kind="stable")
    return np.take_along_axis(slots, order, axis=1)


def list_slots(lists: pl.Series, vocab: dict, width: int) -> np.ndarray:
    """``[n, width]`` dense ids of a list column, upstream order, ``-1``-padded."""
    ids = lists.list.eval(pl.element().replace_strict(vocab, default=-1, return_dtype=pl.Int64))
    ids = ids.list.eval(pl.element().filter(pl.element() != -1)).fill_null([])
    padded = ids.list.concat(pl.Series([[-1] * width] * len(ids))).list.head(width)
    return padded.list.to_array(width).to_numpy()


def business_query_attrs(
    histories: list[list[int]], duration_s: np.ndarray, ad_id: int, n_clauses: int
) -> np.ndarray:
    """``[U, n_clauses]``, not derived from the target: C4 "no ads" (reverse on ``ad_id``),
    C5 the tightest duration cap at or above the user's median history duration, C6
    "uploaded within 7 days"; every other clause ``-1``."""
    qa = np.full((len(histories), n_clauses), -1, dtype=np.int64)
    qa[:, 4] = ad_id
    for u, hist in enumerate(histories):
        d = duration_s[np.asarray(hist, dtype=np.int64) - 1]
        d = d[~np.isnan(d)]
        if len(d):
            qa[u, 5] = cumulative_buckets(np.array([np.median(d)]), DURATION_MAX_S)[0, 0]
    qa[:, 6] = UPLOAD_WITHIN_DAYS.index(7)
    return qa


def pass_rates(attrs: np.ndarray, qa: np.ndarray) -> dict:
    """Per clause, the mean over live queries of the fraction of items the clause passes."""
    out = {}
    for c, name in enumerate(CLAUSE_NAMES):
        slots = np.sort(attrs[:, c, :], axis=1)
        live = slots != -1
        live[:, 1:] &= slots[:, 1:] != slots[:, :-1]
        items_with = np.bincount(slots[live], minlength=int(max(slots.max(), qa[:, c].max())) + 1)
        q = qa[:, c][qa[:, c] != -1]
        rate = items_with[q] / len(attrs)
        if CLAUSE_IS_REVERSE[c]:
            rate = 1.0 - rate
        out[name] = round(float(rate.mean()), 6) if len(q) else None
    return out


def cmd_attrs(args: argparse.Namespace) -> int:
    import torch

    processed = Path(args.processed_dir).expanduser()
    output = Path(args.output_dir).expanduser()
    for p in (output / "test.parquet", output / "item_id_map.json"):
        if not p.exists():
            _log(f"ERROR missing {p} (run `kuairand prep` first)")
            return 1
    t0 = time.monotonic()

    _log("STEP load video features + categories")
    basic = pl.read_parquet(
        processed / "video_features_basic.parquet",
        columns=["video_id", "video_type", "upload_dt", "upload_type", "video_duration", "tag"],
    ).sort("video_id")
    level_cols = [f"{lv}_level_category_id" for lv in CATEGORY_LEVELS]
    cats = basic.select("video_id").join(
        pl.read_parquet(
            processed / "video_categories.parquet", columns=["final_video_id", *level_cols]
        )
        .rename({"final_video_id": "video_id"})
        .with_columns(pl.col(level_cols).cast(pl.Int64).replace(CATEGORY_UNKNOWN, None)),
        on="video_id",
        how="left",
        maintain_order="left",
    )
    if not basic.height == cats.height == N_VIDEOS:
        _log(f"ERROR {basic.height:,} videos, {cats.height:,} category rows, want {N_VIDEOS:,}")
        return 1
    attrs = np.full((N_VIDEOS, C_NARROW, A_MAX), -1, dtype=np.int64)

    cat1_vocab = dense_vocab(cats["first_level_category_id"])
    attrs[:, 0, 0] = encode(cats["first_level_category_id"], cat1_vocab)

    fine = {lv: cats[f"{lv}_level_category_id"] for lv in ("fourth", "third", "second")}
    fine_keys = {lv: (lv[0] + ":" + s.cast(pl.Utf8)).alias("fine") for lv, s in fine.items()}
    fine_vocab = dense_vocab(pl.concat(list(fine_keys.values())))
    attrs[:, 1, :3] = pack_left(np.stack([encode(k, fine_vocab) for k in fine_keys.values()], 1))

    tags = basic["tag"].str.split(",")
    tag_vocab = dense_vocab(tags.explode().replace("", None).alias("tag"))
    attrs[:, 2] = list_slots(tags, tag_vocab, A_MAX)

    upload_vocab = dense_vocab(basic["upload_type"])
    attrs[:, 3, 0] = encode(basic["upload_type"], upload_vocab)

    type_vocab = dense_vocab(basic["video_type"])
    attrs[:, 4, 0] = encode(basic["video_type"], type_vocab)

    duration_s = (basic["video_duration"].cast(pl.Float64) / 1000.0).fill_null(np.nan).to_numpy()
    attrs[:, 5] = cumulative_buckets(duration_s, DURATION_MAX_S)

    upload = basic["upload_dt"].str.to_date("%Y-%m-%d", strict=False)
    age = (pl.lit(dt.date.fromisoformat(args.reference_date)) - upload).dt.total_days()
    age_days = pl.select(age).to_series().cast(pl.Float64).fill_null(np.nan).to_numpy()
    attrs[:, 6] = cumulative_buckets(age_days, UPLOAD_WITHIN_DAYS)

    narrow_t = torch.from_numpy(attrs)
    torch.save(narrow_t, output / "item_attrs_narrow.pt")
    torch.save(torch.tensor(CLAUSE_IS_REVERSE), output / "clause_is_reverse_narrow.pt")
    vocabs = {
        "clauses": list(CLAUSE_NAMES),
        "clause_is_reverse": list(CLAUSE_IS_REVERSE),
        "cat1": {str(k): v for k, v in cat1_vocab.items()},
        "cat_fine": fine_vocab,
        "tag": tag_vocab,
        "upload_type": upload_vocab,
        "video_type": type_vocab,
        "duration_max_s": list(DURATION_MAX_S),
        "uploaded_within_days": list(UPLOAD_WITHIN_DAYS),
        "reference_date": args.reference_date,
    }
    (output / "attr_vocab.json").write_text(json.dumps(vocabs, ensure_ascii=False, indent=1))

    _log("STEP eval_split")
    test = pl.read_parquet(output / "test.parquet")
    target_first = [t[0] for t in test["targets"].to_list()]
    qa = synthesize_qa_narrow(target_first, narrow_t, C_NARROW).numpy()
    business = business_query_attrs(
        test["item_ids"].to_list(), duration_s, type_vocab[AD_VIDEO_TYPE], C_NARROW
    )
    qa[:, 4:] = business[:, 4:]
    pl.DataFrame(
        {"target_id": target_first, "query_attrs_narrow": qa.tolist()},
        schema={"target_id": pl.Int64, "query_attrs_narrow": pl.List(pl.Int64)},
    ).write_parquet(output / "eval_split.parquet", compression="zstd")

    log = {
        "vocab_sizes": {k: len(v) for k, v in vocabs.items() if isinstance(v, dict)},
        "item_coverage": {
            n: round(float((attrs[:, c, 0] != -1).mean()), 4) for c, n in enumerate(CLAUSE_NAMES)
        },
        "eval_rows": test.height,
        "query_live": {
            n: round(float((qa[:, c] != -1).mean()), 4) for c, n in enumerate(CLAUSE_NAMES)
        },
        "mean_pass_rate": pass_rates(attrs, qa),
        "wall_clock_sec": round(time.monotonic() - t0, 1),
    }
    merge_prep_log(output, "attrs", log)
    _log(f"ALL DONE attrs {log}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for fn in (cmd_download, cmd_convert, cmd_prep, cmd_attrs):
        rc = fn(args)
        if rc:
            return rc
    return 0


def _add_prep_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    sp.add_argument("--output-dir", required=True)
    sp.add_argument("--max-seq-len", type=int, default=200)
    sp.add_argument("--test-start", default="2022-05-07", help="local date (Asia/Shanghai)")
    sp.add_argument("--val-days", type=int, default=1)
    sp.add_argument("--gap-minutes", type=int, default=30)


def _add_attrs_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--reference-date", default="2022-05-07", help="upload ages are counted to this day"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kuairand", description="KuaiRand-27K ETL")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="tarball + category supplement (md5-verified)")
    sp.add_argument("--retries", type=int, default=3)
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("convert", help="raw -> processed/ parquet")
    sp.set_defaults(func=cmd_convert)

    sp = sub.add_parser("prep", help="clicks -> train/val/test.parquet + item_id_map.json")
    _add_prep_args(sp)
    sp.set_defaults(func=cmd_prep)

    sp = sub.add_parser("attrs", help="narrow clause tensor + vocabs + eval_split")
    sp.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    sp.add_argument("--output-dir", required=True)
    _add_attrs_args(sp)
    sp.set_defaults(func=cmd_attrs)

    sp = sub.add_parser("all", help="download -> convert -> prep -> attrs")
    sp.add_argument("--retries", type=int, default=3)
    _add_prep_args(sp)
    _add_attrs_args(sp)
    sp.set_defaults(func=cmd_all)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
