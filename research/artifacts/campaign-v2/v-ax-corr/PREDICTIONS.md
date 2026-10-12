# V-AX-CORR: pre-registered predictions

Written before any GPU cell of `arxiv-corr-synth` (2026-10-09). The attrs were already built on CPU;
their achieved pass rates are known (see the [README](README.md)), the recalls are not. "Uniform" is
`arxiv-synth` at the same p, n_probe, n_lists 2048, k and bs.

1. **SilverTorch recall is roughly flat in p on correlated filters.** `recall_oracle@100` of
   `silvertorch` triton clause at a fixed `n_probe` varies by at most 0.10 across `c001`, `c003`
   and `c01`, at every `n_probe` in {24, 64, 128, 256, 512, 1024}.
2. **Correlated beats uniform at equal p and n_probe**, and the gap grows as p falls: at
   `n_probe` 24 correlated is above uniform at every p, by the most at p 0.01.
3. **The n_probe needed for 0.95 does not scale like 1/p** on correlated filters (it does on
   uniform): the smallest swept `n_probe` reaching 0.95 is the same at all three p or differs by one
   grid step.
4. **Postfilter is no longer hopeless at low p**: alpha 1 recall on correlated is far above
   uniform's (which tracks about p × alpha), because a query's unfiltered neighbours mostly share
   its coarse cluster; expect above 0.3 at every p.
5. **Exact arms do not care.** V1 / V2 recall stays at about 1; their latency at a given p is
   within 10 % of uniform's at the same p (it depends on the pass count, not on where the passing
   items sit).
6. **The achieved per-query pass rate exceeds p on average** (a query lands in a cluster with a
   probability that grows with the cluster's size, so the mean is size-biased), with a wide spread
   over queries.
