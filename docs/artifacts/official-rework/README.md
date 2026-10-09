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
