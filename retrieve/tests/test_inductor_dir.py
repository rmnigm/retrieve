"""H-INDCACHE: the conftest gives every pytest run its own inductor cache (no CUDA needed)."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.mark.cpu
def test_conftest_inductor_dir_is_this_runs_own(pytestconfig: pytest.Config):
    run = Path(pytestconfig.inductor_dir)
    assert run.is_dir() and run.parent == Path(tempfile.gettempdir())
    assert run.name.startswith("pytest-inductor-")
    env = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])  # a `bench run` earlier in the run appends
    assert env == run or env.parent == run  # its code_version under the run's dir
