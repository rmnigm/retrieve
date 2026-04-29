# YFCC-5M attribute-filter benchmark dataset

## Why this exists

Yambda is interaction-only ([data/yambda.py](../../evaluation/data/yambda.py),
[algorithms.py:13-17](../../evaluation/retrieval/algorithms.py#L13-L17)) — no
item attributes, so the whole filter path is dark on that benchmark.
The Amazon big5 prep ([amazon-prompt.md](./amazon-prompt.md)) is design-only
and would need a from-scratch GSASRec encoder run to generate item embeddings
(~6–20 GPU-hours). A leaner alternative is YFCC100M, which carries item-level
tags and metadata and has a known-comparable filtered-ANN benchmark
([Big-ANN'23 filtered track](https://big-ann-benchmarks.com/neurips23.html)).

We will **not** use Big-ANN'23's pre-built embeddings: they are
PCA → 192-d uint8, L2-normalised for L2 distance, and our retrieval
modules score with inner-product / cosine. Plus PCA on top of CLIP
loses fine structure that mattered for the filtered slice. We re-embed
the 5M-item subsample ourselves on a rented A100.

The other reason to prefer YFCC over Amazon: YFCC carries **two
disjoint tag schemas natively** — fixed-enum autotags (1,570 visual
concepts) and free-form user tags (200k+ vocabulary) — which directly
map onto the [filtering.md](../system/filtering.md) two-predicate-shape thesis:
narrow exact (autotags) → `ClauseIndex` expected to win; wide free-form
(user tags) → `BloomFilter` expected to win. Same corpus, both predicate
shapes — minimum-confound bench. The unified API the bench targets is
specified in [filtering-api.md](./filtering-api.md).

## What we keep / discard from Big-ANN'23

| asset | status | source |
|---|---|---|
| YFCC100M photo IDs (subsample 5M from the 10M list) | keep | `https://comp21storage.z5.web.core.windows.net/yfcc100m_images/yfcc100m_id_sampled_10m.txt` (deterministically take the first ~6 M; drop to 5 M after S3 HEAD checks for link rot) |
| pre-built 192-d uint8 embeddings | discard | wrong distance / wrong dtype |
| pre-built 100k query embeddings | discard | same |
| extracted tag bag per item | keep | the BigANN tag file is the cleanest user-tag dump |
| autotags / structured metadata | re-derive | from YFCC100M canonical metadata on AWS Open Data |

## Re-embedding pipeline

We embed both image and text sides of each item into a joint
vision-language vector space, then persist three indexes
(image-only, text-only, fused). The bench can swap between them via
one config key and ablate per-modality signal.

### Plan

- **Catalog**: 5M items, deterministically subsampled from BigANN'23's
  `yfcc100m_id_sampled_10m.txt`; 10k held out as queries → indexed
  N ≈ 4.99M.
- **Encoder**: `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (OpenCLIP B/32,
  512-d output). Joint image+text retrieval head trained on LAION-2B
  (YFCC100M is a subset of that pool, so the model has seen this
  distribution). Image tower runs on 224×224 JPEGs at 32-patch
  tokenization; text tower runs on `title || description ||
  machine_tags` (truncated to 77 tokens, the model's text limit).
  Quality is ~5–10 pt recall@10 below SigLIP-base but the model is
  ~3× faster per image and produces 33% smaller embeddings.
- **Hardware**: rented A100 80GB on Vast.ai, ~$1.30/hr, 500 GB local
  SSD, host located in **Czech Republic datacenter** (per user's pick).
- **Image bytes**: `s3://multimedia-commons/data/images/`, AWS Open
  Data registry, transatlantic egress is **sponsored** so no per-GB
  charges. Resize-on-write to 224×224 q=90 JPEG (raw originals never
  hit disk).
- **Wall clock**: ~2–3.5 hr pipelined; expected cost **~$3–5**, plan
  ~$8 with re-run buffer.
- **Persisted output**: `item_embs_image.pt`, `item_embs_text.pt`,
  `item_embs_fused.pt` (each `[N, 512]` fp16, ~5 GB each).

### Throughput note for the Czech host

S3-to-Czech transatlantic throughput is the dominant unknown. Typical
sustained throughput from a well-peered EU datacenter to AWS
us-east-1 is **80–250 MB/s** with 16–32 parallel S3 connections. The
6700 MB/s figure on a Vast listing is the NIC ceiling, not the
S3-egress ceiling — peering and tenant sharing dominate.

| sustained S3 throughput | download wall clock for ~750 GB raw bytes |
|---|---|
| 250 MB/s (good peering) | ~50 min |
| 150 MB/s (typical) | ~85 min |
| 80 MB/s (fair) | ~2.5 hr |
| < 50 MB/s | abort, switch host |

Run a 1 GB throughput probe in the first 5 minutes after the box
boots; if sustained is < 80 MB/s, terminate (~$0.30 lost) and pick
a different Vast host before kicking off the full pull.

With OpenCLIP B/32 the GPU is fast enough that **download dominates
wall clock at any realistic Czech-to-us-east-1 throughput**. The
A100 idles part of the time even at 250 MB/s.

### Run

1. **Setup (~30 min)**: pull OpenCLIP B/32 weights from HF, install
   `pillow-simd`, write the prep script. Run the 1 GB S3 throughput
   probe; abort and re-pick the host if sustained < 80 MB/s.
   Download `yfcc100m_id_sampled_10m.txt`, take the deterministic
   first ~6 M IDs, hold out the last 10k as queries.
2. **Pipelined download → resize → embed (~1.5 hr at 250 MB/s,
   ~3 hr at 80 MB/s)**: producer threads pull originals from
   `s3://multimedia-commons/data/images/`, decode in memory, resize
   to 224×224 q=90 JPEG, persist resized to disk, push tensor into
   a bounded queue. GPU dataloader drains the queue: OpenCLIP image
   tower → 512-d image vector. The text tower runs the text fields
   of completed items in parallel → 512-d text vector.
3. **Per-shard checkpoint (every 100k items, ~50 shards)**: write
   `item_embs_image_NNN.pt`, `item_embs_text_NNN.pt`. If Vast host
   pre-empts, resume from last good shard — at most ~10 min lost.
4. **Concatenate (~5 min)**: produce `item_embs_image.pt`
   (`[N, 512]` fp16) and `item_embs_text.pt` (`[N, 512]` fp16).
   Compute fused: `item_embs_fused.pt = (img/||img|| + txt/||txt||)
   / 2`, re-normalized.
5. **Stage 4 query embeddings (~3 min)**: encode the 10k held-out
   items separately. Image side → `query_embs_image.pt`. Text side
   on title only → `query_embs_title.pt`. Text side on full fields
   → `query_embs_full.pt` (smoke check). Cold-start prompts:
   ~200 hand-written text prompts → `query_embs_coldstart_text.pt`;
   ~50 image probes → `query_embs_coldstart_image.pt`.
6. **Pull artifacts back, terminate (~5 min)**: scp ~15 GB of
   `item_embs_*.pt` + ~50 MB of `query_embs_*` + metadata to your
   workstation. Done with the GPU.

### Storage budget

On the Vast.ai box during the run:

| artifact | size | notes |
|---|---|---|
| Resized JPEGs (224×224 q=90) | ~50 GB | dies with the box |
| Per-shard embedding files (~50 shards × 2 modalities) | ~15 GB | overwritten by step 4 |
| Final `item_embs_image.pt` + `_text.pt` + `_fused.pt` | ~15 GB | 3 × 5 GB |
| Metadata + script artefacts | < 1 GB | |
| **Peak disk usage** | **~80 GB** | well within Vast's 500 GB |

On your workstation after the run (long-lived):

| artifact | size |
|---|---|
| `item_embs_image.pt`, `_text.pt`, `_fused.pt` | ~15 GB |
| `item_attrs_narrow.pt` (`[5M, 4, 4]` int64) | ~320 MB |
| `item_attrs_wide.pt` (`[5M, 1, 32]` int64) | ~640 MB |
| Query embeddings (5 files: image, title, full, coldstart × 2) | ~50 MB |
| GT oracle (`gt/`, 9 sweeps × 10k × 1000 K_GT × int64) | ~700 MB |
| Vocab + metadata JSONs | < 100 MB |
| **Total** | **~17 GB** |

Resized JPEGs are NOT persisted to the workstation. If a re-embed is
ever needed (different encoder, ablation), step 2's S3 download
repeats from multimedia-commons.

### Risk handling

- **Pre-emption on Vast spot**: per-100k-item checkpointing; resume
  costs ~$1 in re-run.
- **S3 throughput too low**: probe before starting, see the
  throughput-note table above. Czech-to-us-east-1 should normally
  achieve > 100 MB/s on a well-peered host.
- **Link rot on multimedia-commons**: ~10–20 % of YFCC100M URLs are
  dead. Sample ~6 M IDs from `yfcc100m_id_sampled_10m.txt` and drop
  to ~5 M after S3 HEAD checks during step 1. If dropped count is
  materially over 25 %, sample more.
- **Encoder under-spec**: if no-filter recall@10 against a 1k-item
  full-scan probe is < 0.5, swap up to `google/siglip-base-patch16-384`
  (768-d, ~3× slower → ~5 GPU-hours, ~$8). Don't stack this risk;
  budget for the larger encoder up front if the bench needs it.

### Why three indexes

OpenCLIP B/32's image and text towers share a vector space (image-text
contrastive training aligns them), so all three persisted indexes
live in the same 512-d cosine geometry — the bench swaps them via
one config key without touching code.

- **`item_embs_image.pt`** — visual signal only. Primary index for
  the Stage 4 image-as-query path; cleanest comparison against the
  published filtered-ANN literature, which is image-side.
- **`item_embs_text.pt`** — user-generated text signal only.
  Comparable to a sentence-encoder index but in OpenCLIP's space.
  Lets us measure how much YFCC's noisy user text helps vs hurts
  retrieval.
- **`item_embs_fused.pt`** — L2-norm-then-mean of the two. Single
  robust index that responds to either modality of query. Default
  for headline bench numbers.

The bench config (`yfcc-5m.yaml`) picks which file via an
`embs_variant: image|text|fused` key.

### Image-bytes practical notes

- The multimedia-commons key is the YFCC100M `download_url`'s last
  path component hashed in the registry's documented scheme. ~10–20 %
  of YFCC100M URLs are dead on the bucket; record actual delivered
  count in `prep_log.json` and dense-id remap **after** this filter
  so the post-Stage-2 N is known-good.
- Resize at download time and discard originals; keeping ~750 GB of
  variable-resolution JPEGs is unnecessary once embeddings exist.
- Use `pillow-simd` over `pillow`; ~3× faster for batch resize on
  the same CPU.

### If the encoder choice is later revisited

`prep_log.json` records `encoder_name`, `encoder_version`,
`output_dim`, `modalities` (`["image", "text", "fused"]`), and the
date of the run. The bench config loads the corresponding
`item_embs_*.pt` by encoder name, so swapping encoders is a config
change, not a code change. Recall numbers in [bench.md](../system/bench.md)
are tagged with the encoder. The most likely future swap is to a
**Matryoshka multimodal model** (`nomic-ai/nomic-embed-vision-v1.5`,
`jinaai/jina-clip-v2`) — these train at 768/1024 but support
truncation to 64/128/256/512 at query time, so the bench could
ablate `D` with a single re-run.

## Two predicate shapes (the bench thesis)

Encoded as two separate `[N, C, A_max]` tensors so each sweep can load
exactly the schema it needs. Both shapes share `evaluate_quality.py`'s
`item_attrs` contract ([eval_quality.py:46-56](../../evaluation/retrieval/eval_quality.py#L46-L56)).

### Narrow, exact (autotags + structured) — `item_attrs_narrow.pt`

`[N, 4, 4]` int64. Targets `ClauseIndex`.

| C | clause | A_max | reverse | source | cardinality | rough selectivity |
|---|---|---|---|---|---|---|
| 0 | autotag_top | 4 | no | top-4 highest-prob autotags per item | 1,570 | 0.05–10 % |
| 1 | autotag_2nd | 4 | no | next-4 autotags per item | 1,570 | broader, 0.5–20 % |
| 2 | country | 1 | **yes** (C2 reverse) | derived from lat/lon (`pycountry-convert`); items missing geo → `-1` | ~250 | 60–90 % "passes" when reversed |
| 3 | year_bucket | 1 | no | `{<2008, 2008-2011, 2012-2014, >=2015}` from EXIF capture date | 4 | ~25 % each |

`clause_is_reverse_narrow.pt` = `[False, False, True, False]`. Exercises
all three `ClauseIndex` paths: AND-of-equalities, multi-value OR (C0
A_max=4), reverse (C2). Mirrors the [amazon-prompt.md](./amazon-prompt.md)
C5 reverse semantics.

### Wide, free-form (user tags) — `item_attrs_wide.pt`

`[N, 1, A_max_wide]` int64. Targets `BloomFilter`.

- `C = 1`. Single clause holds the user-tag bag.
- `A_max_wide`: empirical p99 of tags-per-item, capped at 32 (the long
  tail of >32-tag items truncates to top-32 by global tag frequency,
  preserving the rarest tags first since those are the most informative
  filter signal).
- Vocabulary: 200k+ user tags, dense-id remapped (lower-cased,
  whitespace-collapsed) and persisted in `wide_tag_vocab.json`.
- Per-query: required tag(s) drawn from the held-out target's tag bag.
  Single-tag queries shape `[B, 1]`; 2-tag conjunctions composed via
  two passes through `BloomFilter.evaluate_indices` AND'd through
  `combine_indices`. (Could be one pass once `[B, C, Q_max]` lands;
  defer.)

No reverse on the wide schema (paper-strict bloom cannot do reverse;
the harness rejects `--filter bloom --reverse-path …`).

## Pipeline (`evaluation/data/yfcc.py`)

CLI mirrors [evaluation/data/yambda.py](../../evaluation/data/yambda.py):

```
python -m data.yfcc \
  --slice-size 5_000_000 \
  --output-dir data/yfcc/5m \
  --encoder laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --download-attrs                     # emits item_attrs_{narrow,wide}.pt
```

Stage 1 — metadata + filter pull:

1. Download `yfcc100m_id_sampled_10m.txt` from the BigANN bucket;
   take the deterministic first ~6 M IDs as the working set
   (over-sampled to absorb link rot).
2. Pull canonical metadata for those IDs from `nateraw/yfcc100m`
   or AWS Open Data multimedia-commons CSV (one of the two — pick
   whichever has the highest tag-coverage on this slice; record
   the choice in `prep_log.json`).
3. S3 HEAD-check images; drop link-rot; drop items with empty
   `title || description || machine_tags`. Target N = 5 M; record
   actual.
4. Dense-id remap → `item_id_map.json`.

Stage 2 — embeddings (rented A100, ~2–3.5 hr; see Re-embedding
pipeline above for the full run sequence).

Stage 3 — `--download-attrs`:

1. Build narrow attrs: top-4 autotags by probability →
   `item_attrs_narrow[:, 0, :]`; next-4 → `[:, 1, :]`; country →
   `[:, 2, 0]`; year_bucket → `[:, 3, 0]`. Pad missing with `-1`.
   Emit `attr_vocab_narrow.json`.
2. Build wide attrs: lowercase + dedupe user tags, dense-id remap,
   stack → `[N, 1, 32]` int64 with `-1` pad. Emit
   `wide_tag_vocab.json`.
3. Write `clause_is_reverse_narrow.pt` = `[False, False, True, False]`.

Stage 4 — held-out queries:

This stage produces three things per query: a query **vector**, a query
**attribute spec** (one per predicate shape), and a **ground-truth
top-K** answer set. The vector and the answer set together let us
measure recall@K; the attribute spec is what `FilterModule` evaluates
against. Yambda's eval harness builds the same triple
([eval_quality.py:35-57](../../evaluation/retrieval/eval_quality.py#L35-L57)),
but Yambda is sequential (last item of a sequence = ground truth) and
YFCC is not, so we have to manufacture the supervision ourselves. The
choice has two parts: where vectors come from, and what counts as the
"right" answer.

### Vector source

Four options, each with a different role. Because Stage 2 produces
both image and text vectors in one shared 512-d space, all four
options below retrieve from the same `item_embs_*.pt` family.

1. **Held-out item, image re-encoded.** Pull 10k items out of the
   index ("query split", disjoint from the indexed N) and run the
   image side of the encoder on the held-out image only. The query
   and the indexed item are not bit-identical embeddings (the
   indexed `item_embs_fused.pt` is the L2-mean of image + text;
   the query is image-only) — recall is non-trivial. **Recommended
   default**, exercises the image-as-query use case and is the
   natural shape for filtered visual search.
2. **Held-out item, text re-encoded with title only.** Same split,
   text side, title field only. Useful ablation alongside (1) — lets
   us compare image-as-query vs text-as-query recall under identical
   filters.
3. **Held-out item, full-text re-encoded.** Same split, full text
   fields. The query equals the text half of the indexed embedding
   for that id; the test reduces to "did the index find the item
   itself?" Useful as a sanity check (recall@1 should be ≈1.0 once
   `self` is unmasked) but not the headline benchmark.
4. **Cold-start prompts.** A small curated set:
   - ~200 hand-written text prompts ("vintage cameras", "birds at a
     feeder", "snow-capped mountains in europe", etc.) embedded via
     the text side;
   - ~50 hand-picked image probes (canonical exemplars of common
     autotags), embedded via the image side.

   No ground truth — used only for qualitative top-K inspection during
   the selectivity sweep. Useful to spot when filtering breaks
   something the unfiltered index would have found (regression smoke).

The query split is **disjoint from the indexed catalog**: choose 10k
items uniformly at random *before* Stage 2 runs, hold them out of
`item_embs_*.pt`, and only embed them as queries. This avoids the
"recall = 1 because the answer is the query" trivial loop. The split
is persisted as `query_ids.txt`; Stage 1's `item_id_map.json` only
contains indexed ids.

### Query-attribute synthesis (per predicate shape)

For each of the 10k query items, emit attribute tensors that exercise
the right semantics.

**Narrow shape** (`query_attrs_narrow: [10k, 4]` int64):

- Active-clause set varies per query, drawn from the sweep config
  (see "New eval config" below). Inactive clauses → `-1`.
- For active non-reverse clauses (`C0`, `C1`, `C3`): pick one
  attribute id uniformly at random from the held-out item's own
  non-pad clause values. For a multi-value clause (e.g. `C0` with
  A_max=4), this picks one of up to 4 values; the harness will OR
  query-side later if `[B, C, Q_max]` lands.
- For the reverse clause (`C2`, country): pick a random country id
  **other than** the held-out item's own country. Mirrors the
  [amazon-prompt.md](./amazon-prompt.md) option-1 fix — without this
  flip, "exclude my own country" matches every other item in the
  index by definition and makes the recall measurement degenerate.
- If the held-out item has no value for a clause (`-1` everywhere),
  the clause is forced inactive for that query (`query_attrs[c] = -1`).

**Wide shape** (`query_attrs_wide_1tag: [10k, 1]` int64,
`query_attrs_wide_2tag: [10k, 1, 2]` int64):

- 1-tag query: pick one user tag uniformly at random from the
  held-out item's tag bag; if the bag is empty, drop the query
  (smaller eval split). Bias the sample toward the **rare** half of
  the tag-frequency distribution (probability ∝ 1 / sqrt(global_freq))
  — common tags ("flower", "blue") give selectivity ≈ corpus, which
  doesn't differentiate filters; rare tags ("siwild:species=lynx")
  are where bloom and clause diverge and the bench thesis lives.
- 2-tag query: pick two tags from the bag, bias toward "one common
  + one rare" so the conjunction is tight without being empty.
  Persisted as `[10k, 1, 2]` so the harness can fan it into two
  `[B, 1]` queries and AND via `combine_indices` until `Q_max>1`
  lands. Drop queries whose bag has < 2 tags.
- No reverse on wide (paper-strict bloom).

### Ground truth — full-scan oracle

For each query, compute the "true" filtered top-K once, offline, on
the same encoder that built `item_embs_*.pt`:

```python
# pseudo-code, runs on a single A100 in ~minutes for 10k queries × 5M items
scores = q_emb @ item_embs.T                  # [10k, N], cosine
mask   = filter.evaluate_mask(query_attrs)    # [10k, N], for the chosen predicate
scores = scores.masked_fill(~mask, -inf)
gt_topk_ids, _ = torch.topk(scores, K_GT, dim=1)   # K_GT = 1000
```

`K_GT = 1000` covers `recall@10`, `recall@100`, `recall@500` (the
default `ks` in [500m-d128.yaml](../../evaluation/conf/500m-d128.yaml));
larger K_GT lets us add `recall@500/1000` later without re-running.
Persist as `gt_topk_<shape>_<sweep>.pt` — one file per
`(predicate shape, active-clause set)` because the mask depends on
both. Bench-time recall is then:

```
recall@K = | index_top_k ∩ gt_top_k[:, :K] | / K
```

Self-id (the held-out item itself, if it sneaks into top-K because
the query is very close to its own indexed embedding) is **not**
masked from the oracle — both oracle and index see the same self,
so it cancels in the recall fraction. If a degenerate sweep makes
self-recall a problem, mask self in both `gt_topk` and `index_top_k`;
that is a one-line patch in the bench harness, not a re-run of
Stage 4.

### Persisted layout

```
data/yfcc/5m/
├── item_embs_image.pt                 # [N, 512] fp16 — Stage 2
├── item_embs_text.pt                  # [N, 512] fp16 — Stage 2
├── item_embs_fused.pt                 # [N, 512] fp16 — Stage 2
├── item_attrs_narrow.pt               # [N, 4, 4] int64 — Stage 3
├── item_attrs_wide.pt                 # [N, 1, 32] int64 — Stage 3
├── clause_is_reverse_narrow.pt        # [4] bool — Stage 3
├── attr_vocab_narrow.json             # Stage 3
├── wide_tag_vocab.json                # Stage 3
├── query_ids.txt                      # 10k held-out item ids — Stage 4
├── query_embs_image.pt                # [10k, 512] fp16 (image side) — Stage 4
├── query_embs_title.pt                # [10k, 512] fp16 (text side, title only) — Stage 4
├── query_embs_full.pt                 # [10k, 512] fp16 (text side, full text) — Stage 4 smoke
├── query_embs_coldstart_text.pt       # [~200, 512] fp16 — Stage 4
├── query_embs_coldstart_image.pt      # [~50, 512] fp16 — Stage 4
├── coldstart_prompts.txt              # Stage 4
├── eval_split.parquet                 # (query_id, query_attrs_narrow,
│                                      #  query_attrs_wide_1tag, query_attrs_wide_2tag)
├── prep_log.json                      # encoder name, version, dim, run date, dropped counts
└── gt/                                # Stage 4 ground-truth oracle
    ├── gt_topk_narrow_full_scan.pt    # [10k, 1000] int64
    ├── gt_topk_narrow_c0_autotag.pt
    ├── gt_topk_narrow_c0c2.pt
    ├── …                              # one per active-clause sweep
    ├── gt_topk_wide_1tag.pt
    └── gt_topk_wide_2tag.pt
```

### Stage-4 cost

For 10k queries on N=5M, D=512:

| step | cost (single A100) |
|---|---|
| Re-encode 10k held-out items (image + title + full + coldstart) | < 1 min |
| Sample query attrs (narrow + wide) | seconds (CPU, polars) |
| Full-scan oracle for one sweep | ~5 GB peak HBM, ~3 s |
| All 9 sweeps (7 narrow + 2 wide) | ~30 s |

Done on the same Vast.ai box at the tail of Stage 2, before
termination. Caching `gt_topk_*.pt` to disk avoids re-running the
oracle on every bench iteration — only invalidated when
`item_embs_*.pt` or `item_attrs_*.pt` changes.

### Why not use Big-ANN'23's queries directly

Their 100k query embeddings are PCA → 192-d uint8 in the same wrong
distance regime as the catalog ([§"What we keep / discard"](#what-we-keep--discard-from-big-ann23)),
so they do not match our re-embedded `item_embs_*.pt`. Re-encoding
*their* held-out images would require image bytes for a query split
disjoint from the catalog — the held-out-item-as-query design above
gives an equivalent benchmark shape (item embedding + required tag
bag) using only the metadata we are already re-embedding.

## New eval config — `evaluation/conf/yfcc-5m.yaml`

Mirrors the existing [500m-d128.yaml](../../evaluation/conf/500m-d128.yaml)
shape; new `filters:` block. Two sweeps (one per predicate shape):

```yaml
checkpoint: null   # no GSASRec encoder — embeddings come from the
                   # multimodal encoder via build_export
data_dir: data/yfcc/5m
embs_variant: fused          # one of: image | text | fused
device: cuda
ks: [10, 100, 500]

algorithms:
  - torch_fullscan
  - linr_v3_then_v2
  - silvertorch       # bloom-fused, native co-design path

algo_params:
  linr_v3_then_v2:
    candidate_pool: 8000
  silvertorch:
    n_lists: 2048
    n_probe: 24
    n_iter: 10
    m_bits: 1024
    k_hash: 5

filters:
  narrow:
    attrs_path: data/yfcc/5m/item_attrs_narrow.pt
    reverse_path: data/yfcc/5m/clause_is_reverse_narrow.pt
    expected_winner: clause      # filtering.md thesis prediction
    sweeps:
      - {name: full_scan, active_clauses: []}
      - {name: c2_reverse, active_clauses: [2]}
      - {name: c3_year, active_clauses: [3]}
      - {name: c0_autotag, active_clauses: [0]}
      - {name: c0c2, active_clauses: [0, 2]}
      - {name: c0c3, active_clauses: [0, 3]}
      - {name: all4, active_clauses: [0, 1, 2, 3]}
  wide:
    attrs_path: data/yfcc/5m/item_attrs_wide.pt
    reverse_path: null
    expected_winner: bloom
    sweeps:
      - {name: 1tag, query_attrs_field: query_attrs_wide_1tag}
      - {name: 2tags, query_attrs_field: query_attrs_wide_2tag}
```

Harness contract: at query time, mask `query_attrs[:, c] = -1` for
clauses not listed in `active_clauses`. The `--filter` flag picks the
`FilterModule` impl (`clause` / `bloom` / `combined`); `--filter bloom`
on a sweep that touches a reverse clause is a hard error.

`embs_variant` selects which `item_embs_*.pt` to load. Recommended
sweep order for the headline run: `fused` → `image` → `text`, each
across all sweep entries; this gives a 3 × 9 = 27-cell matrix.

## Implementation surface

Create:
- `evaluation/data/yfcc.py` — Stage 1–4 above.
- `evaluation/conf/yfcc-5m.yaml` — config above.
- `evaluation/retrieval/filter_sweeps.py` — small driver around
  `eval_quality.py` that loops over `filters.{narrow,wide}.sweeps` and
  emits one `eval_quality_<filter>_<sweep>_<embs_variant>.json` per
  cell.

Modify:
- `evaluation/retrieval/eval_quality.py` —
  - Add `--active-clauses` (csv), `--filter` (`clause`/`bloom`/`combined`),
    and `--embs-variant` (`image`/`text`/`fused`) CLI flags.
  - For clauses not in `--active-clauses`, set `query_attrs[:, c] = -1`.
  - Honour `--reverse-path`; for reverse clauses, sample a *different*
    target attribute (the [amazon-prompt.md](./amazon-prompt.md)
    option-1 fix).
  - Build the `FilterModule` per `--filter`; thread its mask /
    indices into the index forward.
- `evaluation/retrieval/algorithms.py` —
  - Broaden `ForwardFn` to accept optional `query_attrs`.
  - For LiNR variants, the wrapper builds a mask via the configured
    `FilterModule` (per algo entry) and threads `mask=` /
    `candidate_ids=` into the LinR forward; this fixes the bug noted
    in [algorithms.py:13-17](../../evaluation/retrieval/algorithms.py#L13-L17)
    where attrs never reached the index.
  - For `silvertorch`, switch from `IVF_INT8_ANN` to `SilverTorch`
    whenever a filter dataset is provided.
- `evaluation/retrieval/eval_perf.py:177-282` — extend `_run_sweep` to
  iterate over the YFCC selectivity sweep and emit
  `eval_perf_filter_<sweep>.json` with latency, recall, bloom FPR.
- `evaluation/retrieval/build_export.py:54` — widen `SUPPORTS_CLAUSES`
  to include both LiNR variants paired with both filters.

## Verification

1. **Stage-1 sanity** — print `(N, num_tags_total, autotag_coverage,
   country_coverage, year_coverage)`; user-tag p99 / p50 length;
   confirm `(item_attrs_narrow[:, 2, 0] == -1).float().mean() < 0.20`.
2. **Stage-2 sanity** — for a held-out 1k items, check
   `cos(query_embs_image, item_embs_image[query_id]) > 0.7` median
   (image-side self-similarity sanity), and same for text side.
3. **Selectivity smoke** — run one query through `clause_compact` and
   `bloom_match` for each sweep entry; passing-count mean within an
   order of magnitude of the table above.
4. **Quality smoke (narrow)** — `eval_quality.py --filter clause
   --active-clauses 0,2 --use-attrs --index linr_v3_then_v2
   --embs-variant fused` — recall@10 ≥ no-filter case (filtering to
   plausible candidates can only help when target shares attrs with
   relevant items).
5. **Quality smoke (wide)** — same with `--filter bloom`,
   `--active-clauses 0` against `item_attrs_wide.pt`.
6. **Bench thesis** — run [tests/bench/run.py](../../retrieve/tests/bench/run.py)
   on the full sweep matrix `(linr_v3_then_v2, silvertorch) ×
   (clause, bloom) × (narrow, wide) × (fused, image, text)`. Predict +
   verify per [filtering.md](../system/filtering.md): `ClauseIndex` wins
   narrow on latency at any selectivity; `BloomFilter` wins wide.
   Numbers go into [bench.md](../system/bench.md).
7. **Reverse-clause hard-error** — `eval_quality.py --filter bloom
   --reverse-path …` exits non-zero with a clear error message.

## Open questions / follow-ups

- If the BigANN'23 tag dump turns out cleaner than the YFCC100M
  canonical metadata, prefer it for user-tag bags (record in
  `prep_log.json`).
- If OpenCLIP B/32 recall@10 on the no-filter baseline is < 0.5,
  swap up to `google/siglip-base-patch16-384` (768-d, ~3× slower,
  ~$8 re-run cost). Decision is one config-key change in `yfcc.py`
  and a re-run of Stage 2.
- A Matryoshka multimodal encoder (`nomic-ai/nomic-embed-vision-v1.5`
  or `jinaai/jina-clip-v2`) would let us ablate `D ∈ {64, 128, 256,
  512}` from a single dump. Worth doing once the bench is settled, to
  characterise filter-vs-D interactions.
- Image-side queries against an image-only index, image-side queries
  against a fused index — the bench will produce both. If image-only
  indexes consistently lose to fused, the user-text signal is real
  and we can drop the text-only variant from the headline numbers.
- Amazon big5 ([amazon-prompt.md](./amazon-prompt.md)) remains
  design-only; YFCC-5M covers the bench thesis without it.
