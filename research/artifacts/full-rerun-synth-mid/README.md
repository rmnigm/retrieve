# SYNTH-MID: SilverTorch and LiNR at 20 / 30 / 50 % synthetic pass rate, and a timed n_probe sweep

The user asked for this on 2026-10-15, in `.chains/control/2026-10-15-193000000-synth-mid-leg-brief.md`. It is a post-freeze addition
at the FULL-RERUN tag `final-freeze` (b3bddc88), seed 0, run on pod e. The FR picks and records stay unchanged. FR timed SilverTorch
at synthetic pass rates 0.1 / 1 / 10 / 100 %, so this leg adds 20 / 30 / 50 % on the 10 M and 30 M LAION sets, with LiNR V1 / V2 / V3
at the same rates. It also times SilverTorch's whole n_probe range on one dataset, so time against recall can be shown per filter.
Nothing here is citable until its gate is green in [validation](../../validation.md).

| file | what |
|---|---|
| [`gen_mid.py`](gen_mid.py) → [`suites-synth-mid.yaml`](suites-synth-mid.yaml) | stage 1 and 3 suites, derived from the tag's `full-rerun/suites-lean.yaml` (`fr-curve-synth-s0-d256`, `fr-v3pool-laion`) so the arms and grids cannot drift |
| [`pick_mid.py`](pick_mid.py) | stage 2: pick.py's rule at k 100 over the stage-1 curves, then `fr-synth-mid-timed-d256` in pick.py's timed-suite shape |
| [`run_mid.sh`](run_mid.sh) | the driver as run on pod e (stage 3, stage 1, picks, stage 2), each leg through the FR `run_leg.sh` |

**Stage 1, quality only.** Two suites:
- `fr-synth-mid-curve-d256`: SilverTorch official-fork bloom with the recheck off, at n_lists 4096 and 16384 with the seed-0 n_probe
  grids up to n_lists / 4. laion10m-synth and laion30m-synth, sweeps `p02` / `p03` / `p05`, k 100, bs 16.
- `fr-synth-mid-v3pool-d256`: LiNR V3 at fr-v3pool-laion's pools (0.005 - 0.2) on the same cells. It exists so that V3 can be timed at
  pick.py's matched-recall pools; FR has no V3 pool curve for synthetic datasets.

**Stage 2, timed.** `pick_mid.py` takes, per (dataset, sweep, n_lists), the smallest n_probe reaching oracle recall@100 0.90 and 0.95,
plus 24 at 4096 lists. It skips a curve that stops short of n_lists / 4. V3 gets the smallest pool reaching 0.90 / 0.95.
`fr-synth-mid-timed-d256` then times SilverTorch bloom (recheck off) at those points. It is interleaved by (algo, n_lists) with LiNR
V1 and V2 (exact) and V3, at k 100 and bs 1 / 16 / 64. A LiNR cell that runs out of memory at 30 M is rerun one algo per process
(run_leg's fallback), and is then not interleaved; its record shows that.

**Stage 3, timed n_probe sweep.** `fr-synth-mid-probesweep-d256` runs on laion30m-synth for all seven synthetic filters (`p0001`,
`p001`, `p01`, `p1`, `p02`, `p03`, `p05`):
- SilverTorch bloom with the recheck off, at 4096 lists (n_probe 24, 32, 64, 128, 256, 512, 1024) and 16384 lists (24, 64, 256,
  1024, 2048, 4096, 8192), k 100, bs 16;
- LiNR V1 interleaved once per filter as the time anchor.

**Trim (user, GO 2026-10-15 ~20:0xZ).** Stages 1 and 2 are bloom only and k 100 only, at 4096 / 16384 lists:
- no clause curves at 20 - 50 %;
- no k 1000;
- no 65536 lists.

Stage 3 runs at bs 16 only; bs 1 would have added more than an hour.

**Provenance.** The tree is `/scratch/wt/fr` at final-freeze, the venv `/venvs/fr`, on pod e's A100-SXM4-80GB (driver 580.126.16)
with its own official-fork build. laion10m-synth reads pod e's copy of laion10m. In FR, laion10m was timed on pod 1. Every SilverTorch
/ LiNR ratio in this leg is read inside one run on one copy.

Raw outputs are on the Hub at `fr/synth-mid-curve-pode-s0`, `fr/synth-mid-timed-pode-s0` and `fr/synth-mid-probesweep-pode-s0`
(plus `-rerun` for unstable cells). Their manifests are in the hub index.
