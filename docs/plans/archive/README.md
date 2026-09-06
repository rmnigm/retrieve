# Archived plans

Plans whose work has fully landed and which are no longer instructions
for anyone. Kept for provenance — the reasoning behind a decision is
often worth more later than the decision itself — but **not maintained**:

- line references point at the tree as it was when the plan was written,
- links may target files that have since moved or been deleted,
- "next steps" are historical, not a queue.

Do not execute anything here. If a plan still has live follow-up work,
that work belongs in [`../00-roadmap.md`](../00-roadmap.md) instead.

| plan | landed | superseded by |
|---|---|---|
| [`silvertorch-reverse-clause-wrapper-fix.md`](silvertorch-reverse-clause-wrapper-fix.md) | commit `cc85d8f` | roadmap Stage 4b item 7 tracks the remaining sweep rerun |
| [`evaluation-refactor.md`](evaluation-refactor.md) | E1–E8 on `refactor/kernels-eval`, 2026-07-06 | harness v2 ([`../evaluation-harness-v2.md`](../evaluation-harness-v2.md)), roadmap Phase C; archived with C3 |
| [`refactor-validation-handoff-harness.md`](refactor-validation-handoff-harness.md) | the harness half of [`../refactor-validation-handoff.md`](../refactor-validation-handoff.md) | roadmap A1 (golden baseline on the old harness) and C4 (harness v2 gate); archived with C3 |
| [`cuda-silvertorch-handoff.md`](cuda-silvertorch-handoff.md) | CUDA C++ SilverTorch backend, implemented 2026-07-06, validated + tuned on the A100 2026-09-02 (its §13 is the model validation record) | deleted at roadmap B4 after the official parity gate (B2, [`../silvertorch-official-integration.md`](../silvertorch-official-integration.md) §7 / §14); last commit holding it: tag `cuda-cute-backends-final` |
| [`cuda-silvertorch-phase2.md`](cuda-silvertorch-phase2.md) | the CUDA backend's phase 2 (exact mode, `UNROLL`, two static reviews), 2026-09-01 | same |
| [`cute-dsl-scorer.md`](cute-dsl-scorer.md) | the CuTe DSL port of the CUDA backend, authored and benchmarked 2026-09-02; §5 / §5.1 hold the language head-to-head and the CUDA-graph replay result | same; raw scripts and outputs in [`cute-dsl-scorer-artifacts/`](cute-dsl-scorer-artifacts/README.md) |
