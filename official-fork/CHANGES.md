# official-fork: Meta's SilverTorch, forked

This directory is a fork of [meta-recsys/silvertorch](https://github.com/meta-recsys/silvertorch) at commit
`21aa35e28b6dd9a91e9ee35efb0857715e86bda7`, vendored unmodified by `git archive` (repository commit 94d8f69), synced to
upstream `main` at `22c2007ed5d28ec8304de081cc2393778708ad12` (the "upstream sync" row), and changed by the retrieve authors to remove the bottlenecks measured in `docs/paper/reproduction-deviations.md`.

**Licence.** Apache License 2.0, as upstream: [`LICENSE`](LICENSE) is upstream's, verbatim; every file keeps upstream's
copyright and licence header; every file we changed carries a "Modified by the retrieve authors" line under it (§4(b)).
Upstream ships no `NOTICE` file. The rest of the retrieve repository is under its own licence; this directory stays
Apache-2.0.

**Coexistence.** The fork installs as `silvertorch-fork` (package `silvertorch_fork`) and registers its ops under
`torch.ops.stfork.*`, so it loads in the same process as the pinned upstream (`silvertorch`, `torch.ops.st.*`): the
parity gates and the interleaved timing compare the two in one process.

## Changes (newest last; each with its gate in `docs/validation.md`)
| change | files | what |
|---|---|---|
| rename | every file under `silvertorch_fork/`, `setup.py`, `README.md` | package `silvertorch` → `silvertorch_fork`, distribution `silvertorch` → `silvertorch-fork`, op namespace `st` → `stfork`; no behaviour change |
| F8 (OF-6) | `ops/csrc/tests/test_bloom_search_integration.py`, `ops/csrc/tests/test_fresh_index_post_processing.py`, `setup.py` | the two test files upstream ships unparsable (Meta's line-based `@oss-disable` strip left two `torch.ops.load_library("//…")` calls open) get those calls commented out like their siblings; `is_topk` and `fresh_index_post_processing` (`take_top_k_and_gather_from_main_and_fresh`), registered upstream but not built, join `setup.py`'s sources |
| F5 (OF-11) | `setup.py` | `-O3` for the C++ sources and for nvcc's host code (`-O3 -Xcompiler -O3`, what the `-O3` reference arm sets through `NVCC_APPEND_FLAGS`); upstream's `.cu` host code builds at gcc's `-O0`. Device code is unchanged |
| packaging | `pyproject.toml`, `setup.py` | static `[project]` metadata (name `silvertorch-fork`, `torch>=2.0`) moved out of `setup()`, so the directory is a uv workspace member; `retrieve[official]` installs it beside upstream |
| upstream sync 21aa35e → 22c2007 | `ops/csrc/bloom_indexer_cuda.cu` | upstream's one commit since the vendored base ("Make silvertorch bloom_indexer_cuda.cu CCCL 3 (CUDA 13.x) compatible", +41/−8), applied verbatim: `ST_CUB_MAXIMUM` / `ST_CUB_COUNTING_ITERATOR` / `ST_CUB_TRANSFORM_ITERATOR` select `::cuda::maximum` and thrust iterators when `CUB_VERSION >= 200800`, else the old cub names. On CUDA 12.8 (CCCL 2.7) the old path is selected, so the compiled behaviour is unchanged. The pinned `backend="official"` arm stays at 21aa35e for the final pass (frozen; the same compiled path on CUDA 12.8) |
