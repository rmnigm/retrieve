# OFFICIAL-REWORK — Meta's official backend run as Meta intends, query preparation outside the timed call

Roadmap step OFFICIAL-REWORK (user, 2026-10-10); ADAPTER-FIX is milestone 1. Design and milestone plan:
`.chains/st-dloop/2026-10-10-140000000-official-rework-design.md`. Current state: [validation](../../validation.md) row
*OFFICIAL-REWORK*; mechanisms: [evaluation § Query preparation](../../system/evaluation.md#query-preparation) and
[kernels § official](../../system/kernels.md#official--metas-torchopsst-kernels-as-the-reference-backend).
A100-SXM4-80GB (pod b, GPU 0). Raw outputs: Hub `artifacts/official-rework/` ([hub-index](../hub-index.md)).

## M1 — query preparation outside the timed call (every arm), no host syncs of ours, packed exact mask
- **`prepare_queries` on every filtered arm**, giving a batch's query-side filter encoding once, before timing:
  - SilverTorch → `PreparedFilter`: query bit positions (triton / torch bloom), Meta's parsed CPU plans (official bloom),
    or the int64 attrs (exact).
  - `FilterModule` (bloom signatures / int64 attrs), and the LiNR variants, `postfilter` and the router through it.
  - `forward(query, prepared)` takes that encoding. The attrs-in-forward path is gone (rule 8).
- **Harness:** `run.prepare_pool` prepares each module's whole pool before the first timed call. `query_prep_ms` is on
  every entry. `OfficialConfig.cache_plans` is gone.
- **Exact on official:** `clause_mask_packed` (Triton, ours, filling a step Meta does not provide) writes the scorer's
  `filtering_bit_mask` words directly. It replaces `clause_mask` → `pack_mask` (a `[B, N/64, 64]` int64 intermediate).

## Gates (`m1_gate.py`, `m1_time.py`; before = staging 5618bb6 as package `retrieve_pre`)
- **Bit-exact, every row equal:** goodreads (0.8 M) and arXiv (3 M) d128.
  - Paths: triton none / exact / bloom partial / bloom full, and official the same four at int32 and fp16.
  - Cells: bs 1 / 16 × k 100 / 1000.
  - Ids up to ties, scores `torch.equal`, on one shared index per path.
- **Keep rule** (interleaved, 8 windows, median [min, max] of the paired ratios, 1410 MHz):

  | path | forward, after / before | prep + forward, after / before |
  |---|---|---|
  | triton none / exact | 0.98-1.00 | 0.98-1.00 |
  | triton bloom partial / full | 0.68-0.75 | 1.00-1.01 |
  | official none | 0.99-1.00 | 0.99-1.01 |
  | official exact | 0.85-0.94 | 0.85-0.95 |
  | official bloom partial / full | 0.78-0.90 | 1.00-1.03 |

  - **Bloom:** the forward lost the query hashing (0.17-0.19 ms) or render + parse (0.11-0.29 ms); the work moved to
    `prepare_queries`, which is reported as `query_prep_ms`.
  - **Official exact:** actually faster (the packed kernel). Its forward is still slow on arXiv bs 16 (6.0 ms against
    Triton's 0.50): Meta's scorer with a full-N mask, for M3.
- **Library suite** 820 passed on a fresh inductor dir; harness suite 498 passed.

## A defect of our gate tooling, found and fixed here
`../campaign-v2/st-ids/make_pkg.sh` did not rename quoted module paths. `interfaces.DISPATCH` and `ops.__getattr__`
import the backends by string, so a renamed "before" package's **modules** dispatched to the current tree's ops. Gates
that import kernel files directly were unaffected; the C5-OURS layer gate was affected (validation rows corrected). Fixed
in `make_pkg.sh`.

## M2 — the epilogue and plan dedup
- **Top-k on the raw scores:** a top-k on the scorer's raw `[B, M]` (int32 or `fp16(dot / divisor)`, pads at the dtype's
  minimum), then the `[B, k]` winners are dequantized and mapped through `sort_perm`. It replaces dequantizing and
  id-mapping all `M` slots before `masked_topk`.
- **Plan dedup (partial bloom):** one plan per distinct expression, plus Meta's `query_plan_index`.
- **Gates:**
  - bit-exact against staging 5618bb6, every row (goodreads and arXiv, every path, int32 / fp16, bs 1 / 16,
    k 100 / 1000);
  - official tests 60 passed;
  - against M1 ([`m2_time.py`](m2_time.py)), forward: official none 0.82-0.91, bloom partial 0.87-0.93, bloom full
    0.92-0.93, exact 0.92-0.99; triton unchanged (0.98-1.00). Partial-path prep at bs 1: 0.035 → 0.049 ms (the index
    upload).
- **Not done:** the per-cluster warp tables for the partial scorer. The totals Meta's op reads back are data-dependent,
  so a host value costs a sync either way, and M3 shows the scorer's device share is small.

## M3 — profile, capture, Meta's build
**Per path** ([`m3_prof.py`](m3_prof.py); before = pre-M1 with the parse inside, after = M2), bs 16:

| dataset, path | device µs | launches | peak MiB |
|---|---|---|---|
| goodreads none | 360 → 291 | 75 → 65 | 25 → 11 |
| goodreads exact | 2088 → 690 | 84 → 67 | 222 → 14 |
| goodreads bloom partial / full | 420 → 343 / 383 → 312 | 90 → 80 / 76 → 66 | |
| arXiv exact | 6943 → 1872 | | 827 → 51 |

Syncs per call: 3-7 → 3-6.

**Where the device time goes (after):**
- **Meta's kernels:** 50-222 µs.
- **Ours, Triton:** only `clause_mask_packed`, 47-1469 µs on exact. It is evaluated per batch row; a variant that loads
  each item tile once for all B rows is the open follow-up.
- **Torch glue:** probe matmul + top-k + gathers, 142-279 µs.

The single-window walls in this profile are not interleaved; the keep-rule numbers are M1's and M2's interleaved
ratios.

**First host sync per path** (`set_sync_debug_mode("error")`): inside Meta's op on every path:
- `fused_kmean_ann` for none / exact / bloom full;
- the partial search for bloom partial.

Ours: none. A whole-forward CUDA-graph capture fails on all four paths (the capture is invalidated inside Meta's op), so
the arm stays eager.

**Meta's build** ([`o3_check.py`](o3_check.py)): as shipped (`setup.py` passes no nvcc `-O`, so host code is at gcc
`-O0`) against the same sources with `-O3 -Xcompiler -O3`, kernels unmodified, same sm_80 / sm_90 SASS layout. Three
alternating process rounds per build:
- **outputs:** bit-identical across all six processes on goodreads and arXiv;
- **`-O3` / shipped:** bloom full 0.77-0.85, partial 0.82-0.87, none 0.90-0.92, exact 0.90-0.98, with non-overlapping
  ranges.

**Adoption:** build the `official` extra with `NVCC_APPEND_FLAGS='-O3 -Xcompiler -O3'` (no source change). That means
rebuilding the shared venvs, which is the controller's call (OF-11).
