# NUMBERS: the paper's measurement legs

The legs that produce the paper's numbers. The study pits SilverTorch, Meta's official code adapted (`backend="official-fork"`), against LiNR
V1-V3. [`plan.yaml`](plan.yaml) lists every leg: id, GPU queue, dependency, command, upload path, estimate. Raw outputs go to the Hub
under `numbers/<leg>` ([hub-index](../hub-index.md)). Nothing here is citable until its gate is green in [validation](../../validation.md).

| file | what |
|---|---|
| [`gen_suites.py`](gen_suites.py) → `suites.yaml` | the LiNR-only suites, the k 1000 check and the quality-only probe curves of the adapted code |
| [`pick_points.py`](pick_points.py) → `suites-timed.yaml` | from the curves: per (dataset, filter, n_lists) the smallest n_probe reaching recall 0.90 / 0.95 (at k 100 and k 1000), the cap, and the paper's 24; writes `n-timed` and `n-st-k2048`, each SilverTorch arm interleaved with LiNR V1 / V2 |
| [`prep_config.sh`](prep_config.sh) | a leg's `--config-dir`: the tree's dataset yamls plus these suites |

**Grid.** Probe curves double n_probe from 8 to the cap of a quarter of the lists. n_lists is the tuned one at ≤ 3 M (goodreads 4096,
arXiv 2048, LAION 1 M 512, LAION 3 M 2048) and {4096, 16384, 65536} at PubMed, YFCC, LAION 10 M and LAION 30 M (user, D-b). Recall targets
are 0.90 and 0.95 (user, D-a). Curves run quality-only; only the picked points are timed, at bs 1 / 16 / 64, k 100 / 1000, eager and
graph, interleaved with LiNR. The k 1000 check runs first: today's scorers stop at n_probe 1024 at k 1000.

**Order.** The LiNR-only legs need no adapted code, so they run first in every queue. The adapted legs wait for the frozen adapted-code
head. Each timed leg waits for its curve leg and a `pick_points.py` run.

**Timed suites are per dim.** A dataset config may list several dims (goodreads and arXiv list 128 and 256), so `pick_points.py`
picks per (dataset, dim) and writes `n-timed-d<dim>` / `n-st-k2048-d<dim>`, each pinned to one dim. A timed leg runs its dataset's own
dim.

**Item 9 (arXiv d-sweep): dropped by the user (2026-10-11).** The extra d256 curves of goodreads / arXiv from A1-small are kept but
not timed.

**Left for the final full rerun (user, 2026-10-11).** Seeds and error bars: every NUMBERS leg runs seed 0, and the final full rerun adds
the seeds and the confidence intervals. Not in the study: a GPU ANN library baseline, a separate index memory / build-time leg, a d4096
cross-check, a second GPU model.
