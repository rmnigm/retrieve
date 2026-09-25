# L1 artifacts — the pre-move reference and the post-move comparison

`premove_check.py capture` ran on `development` @ `4f52972` (the L1 branch point, before any
file moved) and pinned, under `tensors/` (13 MB, not committed — `tensors/*.pt` is ignored):

- 14 reference cells: the parity suite's `make_probe_family` / `make_exact` / `make_bloom`
  inputs with `tests/parity/conftest.py::ref_cps_phase23`'s output (8 plain cells = the
  `test_official.py` T1 grid × B ∈ {1, 16}, 4 exact cells at D ∈ {64, 128} × reverse
  ∈ {none, mixed}, 2 bloom cells at D ∈ {64, 128});
- 9 `SilverTorch` modules (backend ∈ {triton, torch, official} × filter_mode ∈ {none, bloom,
  exact}) on a deterministic N=2048, D=64 index: `state_dict()` and the `(ids, scores)` of a
  fixed 8-query batch.

`capture.json` records the commit and a score digest per cell; `results.json` is
`premove_check.py compare` on the moved tree (`torch.equal` on scores, ids equal at every
finite slot and equal up to ties after the `-1` sentinel; the state dicts loaded into the
moved modules and returning identical ids and scores; the fresh builds' state dicts equal to
the saved ones).

Regenerate the tensors from the branch point:

```bash
git worktree add /tmp/l1-base 4f52972 && cd /tmp/l1-base/retrieve
PYTHONPATH=. python ../docs/artifacts/library-api-refactor/l1/premove_check.py capture --dir <dir>
```
