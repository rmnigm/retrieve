# release-cleanup: what left the tree for the release, and the gate that it changed nothing the final run executes

The user's cleanup (2026-10-13): drop the code and records that will not go into the release (our Triton SilverTorch, LiNR's
bloom filter, YFCC, one-off scripts, superseded results), so an agent drafting the paper finds only the final run's sources
and the rulings. Under the user's rule of the same day ([decisions](../../decisions.md#goal-and-scheduling): the final run is
final), the cleanup is a pure deletion of files the FULL-RERUN never imports, plus docs; every deletion that would need an edit
to a file the final run imports waits for roadmap step 6
([TRITON-ST-DELETE](../../roadmap.md#6-triton-st-delete-delete-what-the-release-cleanup-had-to-leave)), which lists it.

Tags: `triton-silvertorch-final` and `pre-release-cleanup` (both at staging `3174acd`) hold everything removed here.

| file | what |
|---|---|
| [`fr_imports.py`](fr_imports.py) | the repository files a final-run leg imports (bench / eval-data entry points, every op namespace and both official backends, the FR scripts' imports) |
| [`resolve_dump.py`](resolve_dump.py) | every FULL-RERUN job as `load_matrix` resolves it, one sorted JSON line per job |

**Gate** (both trees in the same venv; `prep_config.py` builds each tree's config dir):

1. Every file `fr_imports.py` lists at `final-freeze` is byte-identical in the cleaned tree (`git diff --quiet final-freeze -- <files>`).
2. `resolve_dump.py` at `pre-release-cleanup` and on the cleaned tree: identical output.
3. `check_suites.py` on the cleaned tree: 0 problems.
