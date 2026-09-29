"""Run a command, append its wall time and children's peak RSS to a JSON-lines file."""

import json
import resource
import subprocess
import sys
import time

out, cmd = sys.argv[1], sys.argv[2:]
t0 = time.monotonic()
rc = subprocess.call(cmd)
peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024**2
rec = {
    "cmd": " ".join(cmd),
    "rc": rc,
    "wall_s": round(time.monotonic() - t0, 1),
    "peak_rss_gb": round(peak_gb, 2),
}
with open(out, "a") as f:
    f.write(json.dumps(rec) + "\n")
sys.exit(rc)
