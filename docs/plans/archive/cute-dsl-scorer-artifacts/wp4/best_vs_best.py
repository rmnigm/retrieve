"""WP-4 step 2: best-vs-best per regime, triton / cuda / cute (handoff §6a snippet, 3 columns)."""
import json, sys
W = "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4"
for kind in ("cps", "cpse"):
    js = {be: json.load(open(f"{W}/{kind}-{be}.json")) for be in ("triton", "cuda", "cute")}
    per = {be: js[be]["per_regime"] for be in js}
    print(f"\n### {kind}: winners  " + "  ".join(f"{be}={js[be]['default']}" for be in js))
    print(f"{'regime':<38} {'triton':>8} {'cuda':>8} {'cute':>8} {'tri/cute':>9} {'cuda/cute':>9}  cute winner cfg")
    for key in per["cuda"]:
        t = per["triton"].get(key, {}).get("winner_ms", float("nan"))
        c, k = per["cuda"][key]["winner_ms"], per["cute"][key]["winner_ms"]
        print(f"{key:<38} {t:>8.3f} {c:>8.3f} {k:>8.3f} {t/k:>8.2f}x {c/k:>8.2f}x  {per['cute'][key].get('winner')}")
