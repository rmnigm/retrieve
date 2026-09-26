"""G-yambda / G-goodreads: the timestamped trainer inputs against the reference copies.

python gates.py yambda    > gates_yambda.json
python gates.py goodreads > gates_goodreads.json
"""

import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def list_hash(s: pl.Series) -> str:
    h = hashlib.sha256(s.list.len().cast(pl.Int64).to_numpy().tobytes())
    h.update(s.explode().cast(pl.Int64).to_numpy().tobytes())
    return h.hexdigest()


def timestamps_check(df: pl.DataFrame) -> dict:
    ts = df["timestamps"]
    decreasing = df.select(
        pl.col("timestamps").list.eval((pl.element().diff() < 0).sum()).list.first()
    ).to_series()
    return {
        "dtype": str(ts.dtype),
        "lengths_equal_item_ids": bool(
            (ts.list.len() == df["item_ids"].list.len()).all()
        ),
        "rows_with_decrease": int((decreasing > 0).sum()),
        "decreasing_steps": int(decreasing.sum()),
        "min": int(ts.list.min().min()),
        "max": int(ts.list.max().max()),
    }


def by_row_hash(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.select(cols).with_columns(h=pl.struct(cols).hash()).sort("h", *cols)


def compare_columns(new: pl.DataFrame, ref: pl.DataFrame, cols: list[str]) -> dict:
    a, b = by_row_hash(new, cols), by_row_hash(ref, cols)
    out = {
        "rows_new": new.height,
        "rows_ref": ref.height,
        "rows_multiset_equal": all(a[c].equals(b[c]) for c in cols),
        "rows_in_same_position": int(
            pl.DataFrame([new[c] == ref[c] for c in cols])
            .select(pl.all_horizontal(pl.all()))
            .sum()
            .item()
        )
        if new.height == ref.height
        else None,
    }
    for c in cols:
        out[c] = {
            "equal": new[c].equals(ref[c]),
            "sha256_new": list_hash(new[c]),
            "sha256_ref": list_hash(ref[c]),
        }
    return out


def yambda() -> dict:
    root = Path("/data/yambda-500m")
    new, ref = root / "trainer.new", root / "trainer"
    maps = {
        "trainer.new": sha256(new / "item_id_map.json"),
        "trainer": sha256(ref / "item_id_map.json"),
        "hub": sha256(root / "item_id_map.json"),
    }
    res = {"item_id_map_sha256": maps, "maps_identical": len(set(maps.values())) == 1}
    for split, cols in (
        ("train", ["item_ids"]),
        ("val", ["item_ids", "targets"]),
        ("test", ["item_ids", "targets"]),
    ):
        n = pl.read_parquet(new / f"{split}.parquet")
        r = pl.read_parquet(ref / f"{split}.parquet")
        res[split] = compare_columns(n, r, cols) | {"timestamps": timestamps_check(n)}
    res["trainer_test_vs_hub_test"] = compare_columns(
        pl.read_parquet(ref / "test.parquet"),
        pl.read_parquet(root / "test.parquet"),
        ["item_ids", "targets"],
    )
    return res


def tie_only_mismatch(new_ids: list, ref_ids: list, ts: list) -> bool:
    """True when the two id lists differ only by order inside runs of equal timestamps."""
    if len(new_ids) != len(ref_ids):
        return False
    ts = np.asarray(ts)
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    ends = np.r_[starts[1:], len(ts)]
    return all(
        sorted(new_ids[a:b]) == sorted(ref_ids[a:b]) for a, b in zip(starts, ends)
    )


def pair_by_multiset(new: pl.DataFrame, ref: pl.DataFrame) -> dict:
    """Pair rows whose item_ids and targets agree as multisets, then classify each pair."""

    def keyed(df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            k=pl.struct(
                pl.col("item_ids").list.sort(), pl.col("targets").list.sort()
            ).hash()
        )

    a, b = keyed(new), keyed(ref)
    dup = int(a["k"].is_duplicated().sum() + b["k"].is_duplicated().sum())
    j = a.join(b.select("k", "item_ids", "targets"), on="k", how="inner", suffix="_ref")
    same = (j["item_ids"] == j["item_ids_ref"]) & (j["targets"] == j["targets_ref"])
    diff = j.filter(~same)
    tie_only = sum(
        tie_only_mismatch(x, y, t) and u == v
        for x, y, t, u, v in zip(
            diff["item_ids"].to_list(),
            diff["item_ids_ref"].to_list(),
            diff["timestamps"].to_list(),
            diff["targets"].to_list(),
            diff["targets_ref"].to_list(),
            strict=True,
        )
    )
    return {
        "paired": j.height,
        "pairs_identical": int(same.sum()),
        "pairs_item_ids_tie_order_only": tie_only,
        "pairs_targets_order_differs": int(
            (diff["targets"] != diff["targets_ref"]).sum()
        ),
        "unpaired_new": new.height - j.height,
        "unpaired_ref": ref.height - j.height,
        "duplicate_keys": dup,
    }


def positional(new: pl.DataFrame, ref: pl.DataFrame) -> dict:
    """Classify same-position rows that differ (both files keep the user order)."""
    out = collections.Counter()
    for a, b, ts, c, d in zip(
        new["item_ids"].to_list(),
        ref["item_ids"].to_list(),
        new["timestamps"].to_list(),
        new["targets"].to_list(),
        ref["targets"].to_list(),
        strict=True,
    ):
        if a == b and c == d:
            continue
        out["rows_differing"] += 1
        if a != b:
            if tie_only_mismatch(a, b, ts):
                out["item_ids_tie_order_only"] += 1
            elif sorted(a) != sorted(b):
                lead = next((k for k, t in enumerate(ts) if t != ts[0]), len(ts))
                j = next(j for j in range(len(a) + 1) if sorted(a[j:]) == sorted(b[j:]))
                key = (
                    "in_leading_tie_run" if len(a) == len(b) and j <= lead else "other"
                )
                out[f"item_ids_membership_{key}"] += 1
            else:
                out["item_ids_order_other"] += 1
        if c != d:
            same = sorted(c) == sorted(d)
            out["targets_order_only" if same else "targets_membership"] += 1
    return dict(out)


def goodreads() -> dict:
    root = Path("/data/goodreads-work-id")
    new = root / "trainer"
    maps = {
        "trainer": sha256(new / "item_id_map.json"),
        "hub": sha256(root / "item_id_map.json"),
    }
    res = {"item_id_map_sha256": maps, "maps_identical": maps["trainer"] == maps["hub"]}
    n = pl.read_parquet(new / "test.parquet")
    r = pl.read_parquet(root / "test.parquet")
    res["test_vs_hub"] = compare_columns(n, r, ["item_ids", "targets"])
    if not res["test_vs_hub"]["rows_multiset_equal"]:
        res["test_vs_hub"]["pairing"] = pair_by_multiset(n, r)
        res["test_vs_hub"]["positional"] = positional(n, r)
    for split in ("train", "val", "test"):
        df = n if split == "test" else pl.read_parquet(new / f"{split}.parquet")
        res[split] = {"rows": df.height, "timestamps": timestamps_check(df)}
    return res


if __name__ == "__main__":
    print(
        json.dumps({"yambda": yambda, "goodreads": goodreads}[sys.argv[1]](), indent=2)
    )
