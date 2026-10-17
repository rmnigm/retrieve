import sys, polars as pl
rec, out = sys.argv[1], sys.argv[2]
r = pl.read_parquet(rec)
lst = lambda e: e.unique().sort().cast(pl.String).implode().list.join(" ")
x = (r.group_by("experiment","followup","dataset","dim","filter_kind","sweep","algo","backend","params")
     .agg(seeds=lst(pl.col("seed")), batch_sizes=lst(pl.col("perf_bs").drop_nulls().cast(pl.Int64)),
          ks=lst(pl.col("perf_k").drop_nulls().cast(pl.Int64)),
          modes=lst(pl.col("perf_mode").fill_null("quality only")), records=pl.len())
     .rename({"followup":"followup_run","sweep":"filter"})
     .sort("experiment","followup_run","dataset","dim","filter_kind","filter","algo","backend","params"))
x = x.with_columns(pl.col("batch_sizes", "ks").replace("", None))
x.write_csv(out)
