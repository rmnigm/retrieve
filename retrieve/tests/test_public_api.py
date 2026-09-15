"""The 0.2 public surface (plan L §4.1) and the laziness of ``import retrieve``: the two
``__all__`` lists are pinned, and importing the package registers no kernel — the Triton and
reference op namespaces are imported by a module's constructor, not by the package."""

from __future__ import annotations

import subprocess
import sys

import pytest

import retrieve
import retrieve.modules

MODULES = [
    "BloomFilter",
    "ExactAttributeFilter",
    "FullScanKNN",
    "LiNRBuilder",
    "LiNRV1",
    "LiNRV2",
    "LiNRV3",
    "LiNRV4",
    "OfficialConfig",
    "OneBitKNN",
    "PostfilterKNN",
    "PostfilterKNNInt8",
    "PrefilterKNN",
    "SilverTorch",
    "SilverTorchBuilder",
    "SimHashKNN",
]
INTERFACES = ["FilterModule", "LinrBackend", "RetrievalModule", "SilverTorchBackend"]

pytestmark = pytest.mark.cpu


def test_all_lists():
    assert retrieve.modules.__all__ == MODULES
    assert retrieve.__all__ == sorted(MODULES + INTERFACES)
    assert len(retrieve.__all__) == 20
    for name in retrieve.__all__:
        assert getattr(retrieve, name) is not None


def test_import_registers_no_kernel():
    code = (
        "import sys, retrieve; "
        "print(sorted(m for m in sys.modules if m.startswith(('retrieve.ops.triton', "
        "'retrieve.ops.reference'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout
