"""Names the old harness imported from the library, gone in the current one.

`retrieve.interfaces.Backend` was a single `Literal["torch", "triton", "cuda",
"cute"]`. The current library (`4f52972`) splits it into `LinrBackend` and
`SilverTorchBackend` and drops the two deleted backends (roadmap B4). This
harness is frozen at `dev/a1-golden` and is only ever run to re-derive the A1
golden cells, so it keeps the old alias here instead of being ported.

It is a typing alias only — nothing in this package evaluates it at runtime
(no `get_args(Backend)`), so re-declaring it changes no behaviour and no number.
"""

from __future__ import annotations

from typing import Literal

Backend = Literal["torch", "triton", "cuda", "cute"]
