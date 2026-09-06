# Official SilverTorch artifacts

Raw scripts and outputs behind
[`../silvertorch-official-integration.md`](../silvertorch-official-integration.md)
(plan **O**). Everything here is re-derivable: the commands that produced each
file are in this README, and the pinned upstream sha never moves without a
deliberate PR that reruns the parity gate (O D2).

**Upstream pin.** `https://github.com/meta-recsys/silvertorch`
@ `21aa35e28b6dd9a91e9ee35efb0857715e86bda7` (short `21aa35e`, `main` HEAD on
2026-09-06, no tags upstream, Apache-2.0). Recorded in the workspace root
`pyproject.toml` under `[tool.uv.sources]` and in `uv.lock`.

**Box.** A100-SXM4-80GB (sm_80, driver 570.195.03), Python 3.11.10,
torch 2.10.0+cu128, triton 3.6.0, nvcc 12.4 (`/usr/local/cuda`, the only
toolkit installed).

## Files

| file | step | what it is |
|---|---|---|
| `wp0_environment.txt` | A2 / O WP-0 | the pin, `nvcc --version`, `nvidia-smi`, torch/triton versions, the nine registered `st::` ops, the build flags and the build time |
| `wp0_build.txt` | A2 / O WP-0 | full `uv sync --extra official -v` log, including the nvcc invocations and torch's CUDA minor-mismatch warning |
| `wp0_upstream_pytest.txt` | A2 / O WP-0 | upstream `pytest silvertorch/` output: the README's command, the same minus the three unimportable files, and a scratch-patched run that quantifies why those three fail |
| `official_facts.py` | A3 / O WP-1 | the probe script: bit order, host syncs, kernel launches, graph capture + replay, parse cost, the `per_embedding_scale` overflow |
| `official_facts.json` | A3 / O WP-1 | its raw output, including the full per-op kernel-name lists |
| `official_facts.txt` | A3 / O WP-1 | the same, rendered as a report |

## How to reproduce

```bash
export UV_LINK_MODE=copy                      # cache is on another filesystem
export CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=32
export PATH=$CUDA_HOME/bin:$PATH
uv sync --extra official                      # builds silvertorch._C, ~2.5 min

uv run --extra official python -c "
import torch, silvertorch.ops._load_ops
print(sorted(n for n in torch._C._dispatch_get_all_op_names() if n.startswith('st::')))"

# upstream suite, from a clone at the pinned sha with the built _C.so copied in
git clone https://github.com/meta-recsys/silvertorch && cd silvertorch
git checkout 21aa35e28b6dd9a91e9ee35efb0857715e86bda7
cp <venv>/lib/python3.11/site-packages/silvertorch/_C.cpython-311-*.so silvertorch/
uv run --extra official --with pytest python -m pytest silvertorch/ -q
```

Then A3's probes (each graph probe runs in its own subprocess — a failed capture
poisons the CUDA context):

```bash
uv run --extra official python \
    docs/plans/official-silvertorch-artifacts/official_facts.py \
    --out docs/plans/official-silvertorch-artifacts
```

## A2 findings (corrections and additions to O §1.1, §6.1, §11)

1. **Build gate: PASS.** `uv sync --extra official` built `silvertorch._C` in
   **2 m 17 s** (2 m 25 s wall for the whole sync), against a 5-minute gate,
   with `ninja` (10 translation units) and one arch. `torch.ops.st.fused_kmean_ann`
   exists.
2. **nvcc 12.4 against a cu128 wheel is fine.** O §11 preferred
   `CUDA_HOME=/usr/local/cuda-12.8`; this box only has 12.4. Torch emits
   `cpp_extension.py:525] The detected CUDA version (12.4) has a minor version
   mismatch with the version that was used to compile PyTorch (12.8)` and
   proceeds — it raises only on a *major* mismatch. Every GPU test in the
   upstream suite passes on the resulting extension. **The 12.8 preference in
   O §11 can be relaxed to "12.x".**
3. **Nine `st::` ops, not eleven.** `is_topk.{cpp,cu}` and
   `fresh_index_post_processing.{cpp,cu}` are present in `ops/csrc/` and do
   contain `TORCH_LIBRARY_FRAGMENT(st, …)` registrations, but they are **not in
   `setup.py`'s `cpu_sources` / `cuda_sources` lists** at this sha, so they are
   never compiled and `torch.ops.st.is_topk` /
   `torch.ops.st.take_top_k_and_gather_from_main_and_fresh` do not exist in any
   OSS build. O §1.1 cites both as if they were available (top-k row, live-update
   row) — they are dead source, alongside the dead `process_cluster_v4_pipelined`
   kernel and the unreachable bloom-index-v1 template branches O already lists.
   Neither is needed by the adapter (we use our own `masked_topk`, and live
   update is out of scope), so this changes nothing in §5, only the audit
   paragraph that feeds the paper's build-friction note.
   `faster_repeat_interleave.cu` compiles but registers no op — its
   `_with_cumsum_raw` helpers are header-internal, which is consistent with
   O §3's remark that `fused_kmean_ann_cuda.cu` never calls them.
4. **The upstream suite is not green as shipped.** `pytest silvertorch/` — the
   README's own verification command — **fails at collection** with 3 errors,
   before running a single test. Cause: Meta's internal `@oss-disable`
   comment-stripping is line-based and mangled three files, all of which try to
   `torch.ops.load_library("//silvertorch/oss/ops/csrc:<buck target>")`, a Buck
   label that cannot resolve outside Meta:
   - `test_fresh_index_post_processing.py` and `test_bloom_search_integration.py`
     have the *closing paren* of a multi-line call left commented out
     (`# @oss-disable[end= ]: )`) → **`SyntaxError: '(' was never closed`**;
   - `test_is_topk.py:24` has a single-line call that was not commented at all
     → **`OSError: Could not load this library: /silvertorch/oss/ops/csrc:is_topk`**.

   With those three files excluded: **99 passed, 3 subtests passed in 11.6 s**,
   including every CUDA test. With the bogus `load_library` lines removed in a
   scratch copy, `test_bloom_search_integration.py` passes in full (9 tests), so
   that file is sound; the other two fail on the missing ops of finding 3.
   **Gate reading: the suite is green for everything the OSS build actually
   ships (99/99); the 3 collection errors are upstream packaging defects, not
   ours, and are reproducible from a bare clone with no involvement of this
   repository.** Two paper-ready build-friction facts.
5. **Seven open upstream items, and all seven are pull requests.** O §1.1/§11
   say "7 open issues". The GitHub issues API returns 7 open items for this repo
   — `#5`–`#10` and `#16` — every one of them a PR exported from Meta's internal
   Phabricator; the repo has **zero** issues, open or closed, in its whole
   history. None affects the pin: `#16` makes `bloom_indexer_cuda.cu` CCCL-3
   (CUDA 13.x) compatible, which is the incompatibility we avoid by staying on
   CUDA 12.x; `#5`/`#6` add a MovieLens benchmark; `#7` adds lazy imports to
   internal test rules; `#8`/`#9` are README wording; `#10` adds a JAX/TPU bloom
   path. There is no open report of a correctness or build bug to weigh against
   `21aa35e`.
6. **Packaging notes for O §6.1.** The recipe there works as written, with two
   additions this box needed: the workspace root is virtual (`package = false`),
   so `uv sync --extra official` only resolves if the root re-exports the extra
   (`[project.optional-dependencies] official = ["retrieve[official]"]`); and
   `no-build-isolation` means `setuptools`/`wheel`/`ninja` must already be in the
   shared `.venv`, which a member's dev group does not guarantee — they are now a
   root `[dependency-groups] dev`. `[[tool.uv.dependency-metadata]]` does its job:
   `uv lock` takes 1 s and never executes upstream's `setup.py`.
7. **Our own library suite is red on `development`, independently of this step.**
   `uv run --directory retrieve --extra official pytest tests/ -q` gives
   **3 failed, 485 passed, 122 skipped** in 5 m 25 s. All three fail identically
   on `development` with none of A2's changes applied (verified by stashing):
   `tests/correctness/test_topk_util.py::test_gather_ids_mapping` and
   `::test_gather_ids_with_mask_sentinel` (an fp32-vs-Python-float comparison:
   `0.10000000149011612 != 0.1`) and
   `tests/correctness/test_silvertorch.py::TestEdgeCases::test_n_lists_equals_n[cute]`.
   Not caused by, and not fixed by, the official extra — flagged here so the next
   step does not read them as regressions.

## A3 findings

Full record, claim by claim against O §3 and §4, in
[`../silvertorch-official-integration.md` §13](../silvertorch-official-integration.md).
The headline for whoever writes the adapter next:

- **The official bloom mask is HIGH-bit-first**: document `d` is bit
  `63 - (d % 64)` of word `d // 64`. Confirmed three ways (packed search output,
  a hand-built single-bit `filtering_bit_mask` into the scorer, and a round trip
  of one through the other). B1 takes the HIGH-first branch.
- `fused_kmean_ann` costs **3 host syncs and 19 kernel launches** (O §3 read 2
  and ≈ 12); only 2 of those launches are the scorer itself.
- `bloom_index_search_batch` **captures into a CUDA graph and then faults on
  replay** — it must be on the harness's not-capturable list explicitly. O D7
  is unchanged.
- `per_embedding_scale` returns `inf` in every slot at D=128 with full-range
  codes, exactly as O §4.2 (iii) predicted from the source.
- Instrument note: `warnings.catch_warnings` reads zero syncs for every
  `torch.ops.st.*` call. That is an artefact of where c10 routes a `TORCH_WARN`
  raised inside a C++ op; capture fd 2 instead.
