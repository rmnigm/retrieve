# Amazon Reviews 2023: Attribute Filtering Research for GPU Retrieval Benchmarks

> Previously: `evaluation/docs/AMAZON_PROMPT.md`.

## Context

The repo has a complete GPU retrieval stack with attribute filtering (`ClauseIndex` at [retrieve/src/retrieve/layers/utils/filters.py](../retrieve/src/retrieve/layers/utils/filters.py), Triton fused `clause_compact` kernel, IVF+INT8 index in [retrieve/src/retrieve/layers/silvertorch/ivf.py](../retrieve/src/retrieve/layers/silvertorch/ivf.py)). The current production benchmark uses Yambda‑500M (N≈1.87M, d=128); see [evaluation/conf/500m.yaml](../evaluation/conf/500m.yaml).

A GSASRec training/eval pipeline for Amazon Reviews 2023 is documented at [amazon-prompt.md](amazon-prompt.md) but the data‑prep script `data/amazon.py` does not exist on disk yet, and `--download-attrs` (which would emit `item_attrs.parquet`) has never been wired up. The eval harness already accepts an `[N, C, A_max]` int tensor at [evaluation/retrieval/eval_quality.py:48-56](../evaluation/retrieval/eval_quality.py#L48-L56) — each item has `C` clauses (filter fields) of up to `A_max` values, with `-1` padding; per‑query filter values are taken from the test target's attrs.

This document is the research output to design the filter set for the Amazon benchmark: which attributes are available, their cardinality, expected selectivity at retrieval time, how to map them onto `ClauseIndex`, and what `data/amazon.py` needs to produce.

**Decisions taken:**
- Catalog scale: **5 categories**, target ≈ 3–5M items post‑filter (Yambda‑500M is N≈1.87M, so this benchmark goes ~2× larger).
- Filter semantics: **exercise all three ClauseIndex paths** — AND‑of‑equalities, multi‑value OR (A_max>1), and negation (clause_is_reverse).
- No price clause — replaced with `store` (high‑cardinality, brand‑like).
- Scope: also include `data/amazon.py` design.

## 1. Amazon Reviews 2023 schema

Source: McAuley‑Lab/Amazon‑Reviews‑2023 on HuggingFace (33 categories, May 1996–Sep 2023, 571.5M reviews / 54.5M users / 48.2M items raw).

### Item metadata fields (`raw_meta_<Category>` HF config)

| Field | Type | Granularity | Use as filter? |
|---|---|---|---|
| `parent_asin` | str | item id | — (item key) |
| `main_category` | str | 33 values dataset‑wide; 3 values in our combined corpus | low‑selectivity discriminator (~33–66 % each) |
| `categories` | list[str] | hierarchical path, e.g. `["Beauty", "Hair Care", "Shampoo & Conditioner", "Shampoo"]` | **yes** — multi‑depth values for OR clause |
| `store` | str | very high (10–100k+ per category) | yes, but long‑tail; bucket by frequency |
| `details` (dict) | dict | varies; common keys: `Brand`, `Color`, `Material`, `Size`, `Manufacturer`, `Item Weight`, `Package Dimensions`, `UPC`, `Date First Available` | **yes** — `Brand` is the most useful single key |
| `price` | float (USD, often null) | continuous | yes — bin into ~5–10 quantile buckets |
| `average_rating` | float in [1.0, 5.0] | ~41 discrete values | yes — bin (≥4★, ≥4.5★) |
| `rating_number` | int | long‑tail | yes — popularity tier buckets |
| `title`, `description`, `features` | text | — | not directly usable as equality filter |
| `bought_together` | list | sparse / often empty | skip |

Field sparsity caveat: `price`, `details["Brand"]`, and `store` are populated for the **majority but not all** items — null rate varies 5–40 % by category. Items missing the field must be encoded as `-1` (clause inactive) or assigned to a dedicated "unknown" bucket.

### Review fields (`raw_review_<Category>`)

`rating, title, text, asin, parent_asin, user_id, timestamp, verified_purchase, helpful_vote, images`. These build the interaction sequences (filtered to rating ≥ 4 in the existing pipeline per [amazon-prompt.md:15-16](amazon-prompt.md#L15-L16)).

## 2. Catalog choice — 5 categories, ≈ 3–5M items

5‑core post‑filter sizes (numbers from amazon‑reviews‑2023.github.io; rating ≥ 4 cut and dense‑id remap shrink them another ~30–40 %):

| Category | Items (5‑core) | Interactions | Why included |
|---|---|---|---|
| Clothing_Shoes_and_Jewelry | 715K | 23.1M | physical goods, deep `categories` hierarchy, dense `Color`/`Size` in `details` |
| Home_and_Kitchen | 763.6K | 28.2M | physical goods, distinct brand distribution |
| Electronics | 368.2K | 15.5M | spec‑heavy, high `Brand` coverage, distinct `store` distribution |
| Books | 495K | 9.5M | media — author/publisher in place of brand, very deep category tree, gives the Zipf tail another shape |
| Kindle_Store | 466.6K | 16.1M | digital media — overlaps Books on author/publisher but disjoint catalog |
| **Combined (5‑core)** | **≈ 2.81M** | **≈ 92M** | |

Catalog scale knob: the 5 categories above give **≈ 2.8M items at 5‑core**. To reach the requested 4–5M item range, **loosen k‑core to 3** in `data/amazon.py` (keep the rating ≥ 4 cut). 3‑core retains roughly 1.6–1.8× more items than 5‑core for these heavy‑tailed catalogs, putting the post‑filter N around **4.5–5M items / ~110M interactions**. The exact number is only known after running prep — confirm and tune k‑core if it overshoots.

Practical implications:
- ItemId space is unified across the 5 categories. `main_category` becomes a 5‑value clause used to test the **negation** path.
- Books + Kindle_Store deliberately overlap on author/publisher semantics (encoded into the `Brand` clause via `details["Brand"]` fallback to `details["Publisher"]` / `details["Author"]`) — useful for studying what filtering looks like when the same "brand" id matches across two main_categories.
- Heterogeneous catalogs give natural diversity in `categories` hierarchy depth (Books is depth 5–7, Clothing depth 4–5, Electronics depth 3–5), which makes the multi‑value OR clause behave very differently per slice.

## 3. Filter design — mapping attributes to ClauseIndex

The filter tensor expected by the index is `item_clause_attrs: [N, C, A_max]` int64 with `-1` padding ([retrieve/src/retrieve/layers/utils/filters.py:25-34](../retrieve/src/retrieve/layers/utils/filters.py#L25-L34)). Within a clause, item values are OR'd against the query value; clauses are AND'd. `clause_is_reverse[c]=True` flips the c'th clause's result before the AND. Query side is `[B, C]` int64 — one value per clause, or `-1` to deactivate.

**Six clauses, exercising all three semantics, no price:**

| C | Clause | Source | A_max | Reverse? | Cardinality (combined ≈3–5M corpus) | Avg selectivity |
|---|---|---|---|---|---|---|
| 0 | leaf_category | `categories[-1]` | 1 | no | 15–40K | 0.005–0.3 % |
| 1 | category_path (multi‑depth) | `categories[1:5]` (depths 2..5 of the path) | **4** | no | 15–40K shared vocab with C0 | 0.5–10 % (broader than leaf) |
| 2 | brand | `details["Brand"]` (fallback `details["Publisher"]` for Books/Kindle, then `details["Author"]`, then `store`) | 1 | no | 200–500K, Zipfian | 0.0005–10 % (heavy tail) |
| 3 | store | `store` field (top‑level, ~95 % populated) | 1 | no | 100–300K, different Zipf shape than brand | 0.001–5 % (heavy tail) |
| 4 | rating_bucket | `{≥4.5, 4.0–4.5, 3.5–4.0, <3.5}` | 1 | no | 4 | ~20–30 % |
| 5 | exclude_main_category | `main_category` | 1 | **yes** | 5 | ~60–90 % "passes" (4‑of‑5 categories) |

**Tensor footprint:** `[N, C, A_max]` with `N≈4.5M, C=6, A_max=4` → 4.5e6 × 6 × 4 × 8 B ≈ **864 MB** for int64. Recommended: store as **int32** (cardinalities all fit in 32‑bit) → 432 MB; or split into a "fat" C1 tensor `[N,1,4]` int32 (72 MB) and a "thin" C0,C2..C5 tensor `[N,5,1]` int32 (90 MB) → 162 MB total. Confirm `clause_compact.py` accepts int32 before committing — if it requires int64, stay with split + int64 ≈ 324 MB.

**Why this mix (and why no price):**
- **C0 (leaf_category, A_max=1)** — tightest clause; tests AND‑equality + sparse compaction at very low selectivity.
- **C1 (category_path, A_max=4)** — same source as C0 but encodes ancestors. The OR‑within‑clause path matches if any ancestor matches, broadening the candidate set. Directly stresses the multi‑value branch in `clause_compact`. `A_max=4` covers depths 2..5; pad with −1.
- **C2 (brand)** and **C3 (store)** — both high‑cardinality, heavy‑tailed, but with **different Zipf shapes**: brand concentrates around manufacturers (~10⁵ unique), store includes resellers (~3×10⁵ unique, even longer tail). Running them separately and together gives two heavy‑tail diagnostics on the same corpus and tests how IVF behaves when two long‑tail clauses are AND'd.
- **C4 (rating_bucket)** — only coarse clause; raises the floor on combined selectivity so AND queries don't hit zero candidates.
- **C5 (exclude_main_category, reverse=True)** — exercises `clause_is_reverse`. With 5 main categories, excluding one leaves ~70–90 % of items; combined with tight C0/C2 this gives a "same niche, different super‑category" benchmark scenario.
- **No price**: replaced by `store` per the user's preference for a brand‑like attribute. Price has high null rate (10–30 %), unstable scales across categories (a $5 book vs a $5K TV), and the quantile bucketing is mostly arbitrary — store gives more reliable diagnostic value for filtered retrieval.

**Selectivity sweep** (toggle clauses per query via `−1`; numbers approximate, on combined ≈4.5M corpus):

| Active clauses | Selectivity | Candidate count | Regime tested |
|---|---|---|---|
| none (full‑scan) | 100 % | 4.5M | baseline |
| C4 only | ~25 % | ~1.1M | very loose |
| C5 only (reverse) | ~60–90 % | 2.7–4M | reverse path |
| C1 only (A_max=4) | ~2–8 % | 90–360K | OR‑within‑clause |
| C2 only | 0.0005–10 % | 25–450K | heavy tail (brand) |
| C3 only | 0.001–5 % | 45–225K | heavy tail (store) |
| C2 ∧ C3 ∧ C4 | ~0.05 % | ~2K | two long tails AND'd |
| C0 only | ~0.05 % | ~2K | tight equality |
| C0 ∧ C5 | ~0.04 % | ~1.8K | reverse + tight equality |
| C0 ∧ C2 | <0.001 % | <50 | very tight |
| C0 ∧ C1 ∧ C2 ∧ C3 ∧ C4 ∧ C5 | <0.0001 % | <5 | all clauses, edge case |

Ten points across this sweep give a clean log‑scale selectivity vs. latency / recall plot covering 6 orders of magnitude.

## 4. `data/amazon.py` — pipeline design

The script does not exist yet; this doc (formerly `AMAZON_PROMPT.md`) describes its contract. The new sections below cover only what's needed for the multi‑category + attribute path.

### CLI surface

```
python -m data.amazon \
  --categories Clothing_Shoes_and_Jewelry,Home_and_Kitchen,Electronics,Books,Kindle_Store \
  --output-dir data/amazon/big5 \
  --rating-threshold 4.0 \
  --kcore 3                  # 3-core to land at ~4-5M items; bump to 5 if tighter is needed
  --download-attrs           # emits item_attrs.parquet + item_attrs.pt + attr_vocab.json
```

### Stage 1 — download & filter (existing in spec, no attrs)

1. For each `--categories` entry, `load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_review_{cat}")` → polars DataFrame.
2. Drop rows with `rating < threshold`. Concatenate categories, retaining a `source_category` column for C5.
3. Iterative k‑core (k from CLI, default 3): drop users with <k interactions, drop items with <k interactions, repeat until stable (typically 3–5 passes).
4. Sort each user's interactions by `timestamp`, dedup consecutive duplicates.
5. Remap `parent_asin → dense_id` (0..N−1). Persist `item_id_map.json`.
6. Leave‑last‑out split → `train.parquet`, `val.parquet`, `test.parquet` with `item_ids` (list[int]) and `targets` (list[int]).

### Stage 2 — `--download-attrs` (NEW; the part this plan adds)

1. For each category, `load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{cat}")`. Keep only rows whose `parent_asin` is in the dense id map.
2. Build six per‑item attribute columns:
   - **leaf_category**: `categories[-1]` if non‑empty else `main_category`. Fold to a vocab `{str → int}` shared across categories.
   - **category_path**: take `categories[1:1+A_max]` (skip the redundant top‑level which equals `main_category`). Pad with `-1` to length 4. Reuse the same vocab as leaf_category — paths share strings with leaves at deeper levels.
   - **brand**: `details.get("Brand")` (case‑normalized, whitespace‑collapsed). Fallback chain for null Brand: `details["Publisher"]` (Books/Kindle), then `details["Author"]`, then `details["Manufacturer"]`, then `-1` (don't fall back to `store` here — store is its own clause). Fold to a brand vocab.
   - **store**: top‑level `store` field, case‑normalized. Items with no store → `-1`. Separate vocab from brand.
   - **rating_bucket**: hard thresholds on `average_rating`: `{<3.5 → 0, [3.5,4.0) → 1, [4.0,4.5) → 2, ≥4.5 → 3}`. Items with no rating → `-1`.
   - **main_category**: 0..4 for the 5 source categories. Always populated.
3. Stack into an `[N, 6, 4]` int32 tensor (most clauses fill only slot 0; slots 1..3 are `-1` except for C1 category_path).
4. Persist:
   - `item_attrs.parquet` (one row per item, columns named per clause; lists for C1).
   - `item_attrs.pt` — the tensor, ready for `eval_quality.py --attrs-path`.
   - `attr_vocab.json` — `{clause_name: {str: int}}` for round‑tripping and sanity histograms.
   - `clause_is_reverse.pt` — `[False, False, False, False, False, True]`.

### Reuse / patterns

- The polars + parquet pattern matches [evaluation/training/dataset.py](../evaluation/training/dataset.py) (`pl.read_parquet`, `item_ids` column).
- The `[N, C, A_max]` tensor convention is enforced by [evaluation/retrieval/eval_quality.py:46-56](../evaluation/retrieval/eval_quality.py#L46-L56) — produce exactly that shape and it drops in.
- For the negation clause (C5), eval‑time semantics differ from the leave‑last‑out target convention: `eval_quality.EvalDataset.__getitem__` currently copies the target's attr value. For a reverse clause, that would say "exclude items with the same main_category as the target" which is the **opposite** of what's wanted. Two clean options — pick one before coding:
  1. Make EvalDataset clause‑aware: read `clause_is_reverse` and for reverse clauses, set the query value to a randomly picked **other** category id rather than the target's own value.
  2. For the C5 reverse experiment, set the query value to a fixed exclude‑target chosen offline (e.g., always exclude Electronics) — simpler but covers only one slice of the reverse benchmark.

## 5. New eval config — `evaluation/conf/amazon-big5.yaml`

Mirror [evaluation/conf/500m.yaml](../evaluation/conf/500m.yaml). The numbers below assume final N≈4.5M post rating‑≥4 + 3‑core; refine after Stage 1 prints actual N.

```yaml
checkpoint: checkpoints/gsasrec-amazon-big5-d128-drop0.5/best_model.pt
data_dir: data/amazon/big5
output: null
split: test
device: cuda
ks: [10, 100, 500]

encode:
  batch_size: 128
  num_workers: 8
  max_seq_length: 200

algorithms:
  - torch_fullscan
  - linr_v3_then_v2
  - silvertorch

# N≈4.5M → sqrt(N)≈2120; n_lists=2048 keeps cluster size ~2.2K avg.
# n_probe=20 visits ~44K items per query (~1.0 % of N) — same regime as 500m.yaml.
algo_params:
  linr_v3_then_v2:
    candidate_pool: 8000
    v3_seed: 0
  silvertorch:
    n_lists: 2048
    n_probe: 20
    n_iter: 10
    seed: 0

# NEW: filter benchmark sweep — eval_quality.py invocations
filters:
  attrs_path: data/amazon/big5/item_attrs.pt
  reverse_path: data/amazon/big5/clause_is_reverse.pt
  sweeps:
    - name: full_scan
      active_clauses: []
    - name: c4_rating
      active_clauses: [4]
    - name: c5_reverse_only
      active_clauses: [5]
    - name: c1_category_path
      active_clauses: [1]
    - name: c3_store
      active_clauses: [3]
    - name: c2_brand
      active_clauses: [2]
    - name: c2c3c4
      active_clauses: [2, 3, 4]
    - name: c0_leaf
      active_clauses: [0]
    - name: c0c5
      active_clauses: [0, 5]
    - name: c0c2
      active_clauses: [0, 2]
    - name: all
      active_clauses: [0, 1, 2, 3, 4, 5]
```

`active_clauses` is the harness contract: at query time, set `query_attrs[:, c] = target_attrs[:, c]` for `c in active_clauses` else `-1`. `eval_quality.py` already supports the per‑clause `-1` deactivation; the sweep just needs a small driver around it.

## 6. Critical files to read / modify

Read‑only references:
- [amazon-prompt.md](amazon-prompt.md) — pipeline spec, rating ≥ 4, 5‑core, leave‑last‑out
- [evaluation/retrieval/eval_quality.py:17-73](../evaluation/retrieval/eval_quality.py#L17-L73) — EvalDataset already builds `query_attrs` from the first target's attrs; the only thing missing is the producer
- [retrieve/src/retrieve/layers/utils/filters.py:25-66](../retrieve/src/retrieve/layers/utils/filters.py#L25-L66) — exact tensor shape contract for ClauseIndex
- [retrieve/src/retrieve/kernels/triton/filters/clause_compact.py](../retrieve/src/retrieve/kernels/triton/filters/clause_compact.py) — fused kernel; confirm A_max and C upper bounds before sizing the tensor
- [evaluation/conf/500m.yaml](../evaluation/conf/500m.yaml) — template

Files to create:
- `evaluation/data/amazon.py` — Stage 1 + Stage 2 above
- `evaluation/conf/amazon-big5.yaml` — config in §5
- Driver in `evaluation/retrieval/` (or extend `eval_quality.py`) that loops over `filters.sweeps` and emits one `eval_quality_<sweep>.json` per active‑clause set

Files to extend:
- `evaluation/retrieval/eval_quality.py` — accept `--active-clauses 0,2,5` CLI arg; mask `query_attrs[:, c] = -1` for clauses not listed; respect `clause_is_reverse` semantics for reverse clauses (the §4 EvalDataset note).

## 7. Verification — end‑to‑end

1. **Stage 1 sanity:** after `data/amazon.py` runs, print `N`, `num_users`, train/val/test counts. Spot‑check that one user's `train + [val_target] + [test_target]` equals their full history sorted by timestamp. If N is far from 4–5M, adjust `--kcore`.
2. **Stage 2 sanity (the new bit):**
   - `(item_attrs[:, 0, 0] == -1).float().mean()` — should be near 0 (every item has *some* category).
   - `(item_attrs[:, 1, :] == -1).all(dim=-1).float().mean()` — same as above (path always has at least the leaf).
   - `(item_attrs[:, 2, 0] == -1).float().mean()` — null brand rate after publisher/author/manufacturer fallback, expect 3–15 %.
   - `(item_attrs[:, 3, 0] == -1).float().mean()` — null store rate, expect 1–8 %.
   - `torch.bincount(item_attrs[:, 5, 0])` — should match the 5 source category sizes (e.g. ≈ Clothing / Home / Electronics / Books / Kindle counts after rating + k‑core).
   - `torch.bincount(item_attrs[:, 2, 0].clamp(min=0))[1:].topk(20)` — top‑20 brands, sanity‑check Zipfian shape.
   - `torch.bincount(item_attrs[:, 3, 0].clamp(min=0))[1:].topk(20)` — top‑20 stores; should be different from top brands (resellers vs manufacturers).
3. **Filter selectivity smoke:** run one query through `clause_compact` for each sweep entry on the full corpus; the returned `counts` mean should match §3's table within an order of magnitude.
4. **Quality smoke:** run `eval_quality.py --use-attrs --active-clauses 4 --index silvertorch` — recall@10 should be ≥ the no‑filter case (filtering to plausible candidates can only help when the target shares attrs with relevant items).
5. **Latency benchmark:** run the full `amazon-big5.yaml` sweep through the existing bench harness ([retrieve/tests/bench/run.py](../retrieve/tests/bench/run.py)). Plot latency, recall, and visit rate against selectivity. Compare the silvertorch curve at ~1.0 % visit rate against [evaluation/conf/500m.yaml](../evaluation/conf/500m.yaml)'s same setting on Yambda — per‑query latency should scale roughly linearly with N (Amazon ~2.4× larger than Yambda).

## 8. Open question — the negation semantics

Decide before coding (§4 lists the two options): does the C5 reverse clause use a **per‑query random other category** (richer benchmark, more code) or a **fixed exclude target** (simpler, exercises the kernel path but covers one slice). The plan otherwise stands either way.
