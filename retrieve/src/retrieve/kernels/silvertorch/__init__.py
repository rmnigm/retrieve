"""SilverTorch kernels: ``codesigned_probe_score`` (IVF + INT8 + bloom),
``codesigned_probe_score_exact`` (IVF + INT8 + exact AND-of-OR), ``bloom_match`` (standalone
subset test) and ``official`` (the adapter over Meta's ``torch.ops.st.*``). Import the
submodules directly — nothing is re-exported here, so a module name never shadows the op
registered under the same name."""
