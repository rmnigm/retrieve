"""Re-derive clause 1 (license) of the published arxiv filter assets in place.

The published `item_attrs_narrow.pt` is `[N+1, 4, 4]` (pad row 0, clauses
main/license/year/versions); the current `etl/arxiv.py attrs` writes
`[N, 5, 4]`, so re-running it would change the dataset's shape. This script
touches only clause 1 of that tensor and column 1 of `eval_split.parquet`'s
`query_attrs_narrow`, using the ETL's own `_license_to_bucket` through the
fixed `skip_nulls=False` expression, and asserts every other byte is unchanged.

Usage: python patch_license_clause.py <raw_dir> <data_dir> <out_dir>
  raw_dir   `eval-data arxiv download` output (data/*/*.parquet)
  data_dir  the published arxiv-papers directory (read only)
  out_dir   where the patched item_attrs_narrow.pt / eval_split.parquet land
"""

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from eval_datasets.etl.arxiv import LICENSE_BUCKETS, _license_to_bucket

INT64_MIN = torch.iinfo(torch.int64).min

raw_dir, data_dir, out_dir = (Path(a) for a in sys.argv[1:4])
out_dir.mkdir(parents=True, exist_ok=True)

# Same filter / dedup as `prep`, so the id -> license row is the one prep kept.
raw = (
    pl.scan_parquet(str(raw_dir / "data" / "*" / "*.parquet"))
    .select("id", "abstract", "license")
    .collect()
    .with_columns(pl.col("abstract").fill_null("").str.strip_chars())
    .filter(pl.col("abstract").str.len_chars() > 0)
    .filter(pl.col("id").is_not_null())
    .unique(subset=["id"], keep="first", maintain_order=True)
)
with open(data_dir / "item_id_map.json") as f:
    id_map = json.load(f)
ids = pl.DataFrame(
    {"id": list(id_map), "item_id": list(id_map.values())},
    schema={"id": pl.Utf8, "item_id": pl.Int64},
)
assert raw.height == ids.height, (raw.height, ids.height)
lic = (
    ids.join(raw, on="id", how="left")
    .with_columns(
        pl.col("license")
        .map_elements(_license_to_bucket, return_dtype=pl.Int64, skip_nulls=False)
        .alias("lic_id")
    )
    .sort("item_id")
)
assert lic["item_id"].to_list() == list(range(1, ids.height + 1))
assert lic["lic_id"].null_count() == 0
new_c1 = torch.from_numpy(lic["lic_id"].to_numpy().astype(np.int64))
raw_null = torch.from_numpy(lic["license"].is_null().to_numpy())

old = torch.load(data_dir / "item_attrs_narrow.pt")
assert old.shape == (ids.height + 1, 4, 4), old.shape
old_c1 = old[1:, 1, 0]
bad = old_c1 == INT64_MIN
assert torch.equal(bad, raw_null), "INT64_MIN rows are not exactly the null-license rows"
assert torch.equal(old_c1[~bad], new_c1[~bad]), "non-null license buckets disagree"

new = old.clone()
new[1:, 1, 0] = new_c1
assert (new >= -1).all()
changed = (new != old).nonzero()
assert (changed[:, 1] == 1).all() and (changed[:, 2] == 0).all()
torch.save(new, out_dir / "item_attrs_narrow.pt")

es = pl.read_parquet(data_dir / "eval_split.parquet")
tg = torch.tensor(es["target_id"].to_list())
qa_old = torch.tensor(es["query_attrs_narrow"].to_list())
assert torch.equal(qa_old, old[tg, :, 0])
qa_new = new[tg, :, 0]
qdiff = (qa_new != qa_old).nonzero()
assert (qdiff[:, 1] == 1).all() and (qa_old[qdiff[:, 0], 1] == INT64_MIN).all()
es_new = es.with_columns(
    pl.Series("query_attrs_narrow", qa_new.tolist(), dtype=pl.List(pl.Int64))
)
es_new.write_parquet(out_dir / "eval_split.parquet", compression="zstd")

hist = torch.bincount(new[1:, 1, 0], minlength=len(LICENSE_BUCKETS))
report = {
    "n_items": ids.height,
    "n_null_license": int(raw_null.sum()),
    "n_item_cells_changed": int(changed.shape[0]),
    "n_query_cells_changed": int(qdiff.shape[0]),
    "license_hist": dict(zip(LICENSE_BUCKETS, hist.tolist(), strict=True)),
    "min_value": int(new.min()),
}
print(json.dumps(report, indent=2))
