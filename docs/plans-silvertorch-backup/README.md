# plans-silvertorch-backup

Snapshot of [docs/plans/](../plans/) taken on 2026-05-19, before silvertorch was removed from planning scope.

Files here are frozen and may not match the current code tree. Active plans live in [../plans/](../plans/); they have been rewritten to cover only the **linr** family (`SimilarityMasking` / `PrefilterKNN` / `OneBitKNN`) plus the filter layers (`ExactAttributeFilter`, `BloomFilter`).

Contents:
- All twelve plan files as they existed on 2026-05-19 — including the two silvertorch-only plans that have been removed from the live dir entirely: `sharding.md` (the `ShardedSilverTorch` scatter-gather design) and `silvertorch-cuda-shelved.md` (the shelved native-CUDA `codesigned_probe_score` experiment).
- The active rewrites in `../plans/` reference this directory by relative path (e.g. for the original Phase 3 of `torch-export-refactor.md`, the original `bloom_match` recipe in `migrate-clean-triton-custom-op.md`, etc.).
