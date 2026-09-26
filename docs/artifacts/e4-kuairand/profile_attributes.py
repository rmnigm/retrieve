"""Attribute and log profile of the converted KuaiRand-27K parquets that fixed the clause
layout and thresholds of eval_datasets/etl/kuairand.py (docs/system/datasets.md § kuairand).
Run from evaluation/: python ../docs/artifacts/e4-kuairand/profile_attributes.py"""

import datetime as dt

import polars as pl

P = "/data/_raw/kuairand/processed/"
logs = pl.scan_parquet([P + f"log_standard_{i}.parquet" for i in (1, 2, 3, 4)])
print(logs.select(pl.len(), pl.col("is_click").sum(), pl.col("is_rand").sum(),
                  pl.col("date").min().alias("d0"), pl.col("date").max().alias("d1")).collect())
r = pl.scan_parquet(P + "log_random.parquet")
print("random log", r.select(pl.len(), pl.col("is_rand").min(), pl.col("video_id").n_unique()).collect())
clicks = logs.filter((pl.col("is_click") == 1) & (pl.col("is_rand") == 0))
u = clicks.group_by("user_id").len().collect()["len"]
print("clicks/user q0,.1,.5,.9,.99,1:", [int(u.quantile(q)) for q in (0, .1, .5, .9, .99, 1)])

b = pl.read_parquet(P + "video_features_basic.parquet")
c = pl.read_parquet(P + "video_categories.parquet")
print("categories rows / id range:", c.height, c["final_video_id"].min(), c["final_video_id"].max())
for col in ("video_type", "upload_type", "music_type"):
    print(b[col].value_counts(sort=True))
tags = b["tag"].str.split(",")
print("tags per video:", tags.list.len().value_counts(sort=True))
j = b.select("video_id", "tag").join(c.rename({"final_video_id": "video_id"}), on="video_id")
agree = j["tag"].str.split(",").list.first().cast(pl.Float64, strict=False) == j["first_level_category_id"]
print("first tag == level-1 category:", round(agree.mean(), 4))
for lv in ("first", "second", "third", "fourth"):
    s = c[f"{lv}_level_category_id"]
    print(lv, "distinct", s.n_unique(), "unknown(-124)", (s == -124).sum(), "null", s.null_count())
d = b["video_duration"] / 1000
print("duration s q.01..q.99:", [round(d.quantile(q), 1) for q in (.01, .1, .25, .5, .75, .9, .99)])
print("duration <= 15/30/60/180 s:", [round((d <= x).mean(), 4) for x in (15, 30, 60, 180)])
up = b["upload_dt"].str.to_date("%Y-%m-%d", strict=False)
age = (pl.Series([dt.date(2022, 5, 7)] * len(up)) - up).dt.total_days()
print("upload range", up.min(), up.max(), "unparsed", up.is_null().sum())
print("age days q.01..q.99:", [age.quantile(q) for q in (.01, .1, .25, .5, .75, .9, .99)])
print("age <= 3/7/14/30/90/365:", [round((age <= x).mean(), 4) for x in (3, 7, 14, 30, 90, 365)])
per_author = b.group_by("author_id").len()["len"]
print("authors", per_author.len(), "videos/author q.5,.9,.99:", [per_author.quantile(q) for q in (.5, .9, .99)])
