"""WP-4 step 7: first-call latency in a truly fresh process, one filter mode per run."""
import sys, time
mode = sys.argv[1]
sys.path.insert(0, "/tmp/claude-0/-workspace-retrieve/1a91087e-6afe-4d43-a38e-7d1801b945a5/scratchpad/wp4")
import torch
t_imp = time.perf_counter()
import bench_common as bc
ct = bc.ct
print(f"imports: {time.perf_counter() - t_imp:.2f}s")
p, fns = bc.build("S", 1, mode)
torch.cuda.synchronize()
t0 = time.perf_counter()
ct.ensure_built(verbose=True)
t1 = time.perf_counter()
fns["cute"](); torch.cuda.synchronize()
t2 = time.perf_counter()
fns["cute"](); torch.cuda.synchronize()
t3 = time.perf_counter()
dev = ct._load_dev()
print(f"{mode}: ensure_built {t1 - t0:.2f}s; first _impl call {1e3 * (t2 - t1):.1f} ms; second {1e3 * (t3 - t2):.2f} ms; "
      f"specializations: {sorted(dev._compiled)}")
