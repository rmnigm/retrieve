# Wiki Log

> Append-only record of structural wiki changes, one line each.
> Format: `## [YYYY-MM-DD] action | subject`

## [2026-09-26] init | wiki at `docs/`: SCHEMA, index, log, roadmap, decisions, validation, contracts; `system/` pages given frontmatter; `docs/plans/` removed, artifacts moved to `docs/artifacts/`
## [2026-09-26] lint | history and retired-plan labels removed from system/, decisions, validation, contracts; page/code mismatches fixed; storage.md marked contested
## [2026-09-26] move | results and raw artifact outputs off git onto the Hub (H1): `evaluation/results/` gitignored, `artifacts/hub-index.md` added
## [2026-09-29] lint | roadmap trimmed to open work (history out, resolved defects removed); stale claims fixed in validation and system/ (kernels, evaluation, datasets, architecture, testing) against the code
## [2026-10-07] lint | roadmap rewritten as the pure queue (pod restore, multi-GPU execution, rerun policy, D1-A..G, H3-H7, L6, M1, D5 postfilter baseline, GPU-hour estimates); D2, KuaiRand and yambda out (decisions); new backlog.md for unqueued work; AGENTS rule 1 to pods; storage contested resolved; D1 cross-scale and audit facts into validation; stale claims fixed (arxiv all4, TF-1/TF-9, L5)
## [2026-10-08] rewrite | campaign v2: roadmap rewritten as the re-plan's queue (code batch CV2-*, freeze, claims/inventory, campaign V-*, stop rules); decisions gain the Campaign v2 section (claims drive cells, code-first freeze, reuse rule, 4 datasets, 3 seeds everywhere, grid); OpenAlex/LAION/top-k fusion to backlog; superseded D1/b3 marked in validation; E5/D5/D1-x wording fixed across pages
## [2026-10-08] add | paper/claims.md (claims of both papers + C1-C7 matrix); artifacts/campaign-v2 (record inventory script + README, outputs on the Hub); roadmap estimates from measured per-cell times
## [2026-10-10] add | V-LAION30 (Re-LAION 30 M, d256, `filter` only) and H-OVIEW (oracle without its fp32 copy) on the roadmap; decisions/Datasets records the user's call; backlog LAION entry narrowed to > 30 M
## [2026-10-10] decide | ECIR replicability criteria + library/benchmark goal, in-graph scope, V3 one bit per coordinate (decisions); roadmap: SYNTH-TRIM, V3-BITS-PUBMED, ROUTER-LIB, H-ADDDATA, REL-LIC, F-CRIT; H-OVIEW closed (already done)
## [2026-10-10] add | paper/release-and-licenses.md (REL-LIC: per-dataset license and redistribution, third-party code, Hub/GitHub inventory, leak scan, double-blind plan, decisions for the user); paper/ecir-criteria.md (F-CRIT: every ECIR question mapped to claims, exhibits, evidence, state, gap step); datasets: eval-yfcc10m holds the staged files
## [2026-10-10] decide | YFCC exact gate: per-dataset fp16-storage allowance at k 100, conditional on the probe (decisions § Datasets)
## [2026-10-10] decide | router kept only if on the recall-latency Pareto front between IVF and exact on PubMed 10 M
