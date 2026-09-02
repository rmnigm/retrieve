"""Toggle the P3 pin on the scorer's item loop (run with 'pin' or 'unpin')."""
import sys
p = "/workspace/retrieve/retrieve/src/retrieve/kernels/silvertorch/cute/codesigned_probe_score.py"
s = open(p).read()
unpinned = "    for p0 in range(p_first, warp_end, STEP):\n"
pinned = "    for p0 in cutlass.range(p_first, warp_end, STEP, unroll=1):\n"
if sys.argv[1] == "pin":
    assert unpinned in s; s = s.replace(unpinned, pinned)
else:
    assert pinned in s; s = s.replace(pinned, unpinned)
open(p, "w").write(s); print("ok", sys.argv[1])
