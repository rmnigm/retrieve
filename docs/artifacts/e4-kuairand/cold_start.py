import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

N = 32038726
seen = np.zeros(N, bool)
t = pq.read_table("data/kuairand/train.parquet", columns=["item_ids"])
seen[pc.list_flatten(t.column("item_ids")).to_numpy()] = True
vt = pc.list_flatten(pq.read_table("data/kuairand/val.parquet").column("targets")).to_numpy()
print("val targets seen in train: %.3f" % seen[vt].mean())
seen_tv = seen.copy()
seen_tv[vt] = True
tt = pc.list_flatten(pq.read_table("data/kuairand/test.parquet").column("targets")).to_numpy()
print("test targets seen in train: %.3f, in train+val: %.3f" % (seen[tt].mean(), seen_tv[tt].mean()))
