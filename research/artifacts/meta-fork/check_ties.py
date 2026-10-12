"""Old (per-row loop) vs vectorised same_up_to_ties: synthetic edge cases, then every compare() of one real
gate_fork block. Usage: check_ties.py <old gate_fork.py> <new gate_fork.py> <gate_fork args...>"""

import ast
import importlib.util
import json
import sys
import time
import torch


def load_fn(path):
    src = open(path).read()
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "same_up_to_ties"
    )
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), path, "exec"), ns)
    return ns["same_up_to_ties"]


old, new = load_fn(sys.argv[1]), load_fn(sys.argv[2])
dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(0)
bad = 0


def chk(x, y, label):
    global bad
    a, b = old(x, y), new(x, y)
    if a != b:
        bad += 1
        print("DISAGREE", label, a, b)
    return a


n_true = n_false = 0
for t in range(3000):
    B, k = (
        int(torch.randint(1, 5, (1,), generator=g)),
        int(torch.randint(1, 12, (1,), generator=g)),
    )
    s = (
        torch.randint(0, 4, (B, k), generator=g)
        .float()
        .sort(dim=1, descending=True)
        .values
    )
    npad = torch.randint(0, k + 1, (B,), generator=g)
    for r in range(B):
        if npad[r] and t % 3 == 0:
            s[r, k - int(npad[r]) :] = float("-inf")
    ia = torch.stack([torch.randperm(50, generator=g)[:k] for _ in range(B)])
    ia[s == float("-inf")] = -1
    ib = ia.clone()
    for r in range(B):
        for v in s[r].unique():
            sel = (s[r] == v).nonzero().flatten()
            ib[r, sel] = ib[r, sel[torch.randperm(len(sel), generator=g)]]
    m = t % 5
    if m == 1:
        r, c = (
            int(torch.randint(0, B, (1,), generator=g)),
            int(torch.randint(0, k, (1,), generator=g)),
        )
        ib[r, c] = 99
    elif m == 2 and k > 1:
        r = int(torch.randint(0, B, (1,), generator=g))
        ib[r, -1] = ib[r, -2]
    elif m == 3 and k > 1:
        r = int(torch.randint(0, B, (1,), generator=g))
        ib[r, 0], ib[r, -1] = ib[r, -1].clone(), ib[r, 0].clone()
    x, y = (ia.to(dev), s.to(dev)), (ib.to(dev), s.to(dev))
    if chk(x, y, f"synthetic t={t} m={m}"):
        n_true += 1
    else:
        n_false += 1
print(
    json.dumps(
        {
            "synthetic_cases": 3000,
            "old_true": n_true,
            "old_false": n_false,
            "disagree": bad,
        }
    )
)

spec = importlib.util.spec_from_file_location("gate_fork_old", sys.argv[1])
gm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gm)
stats = {
    "pairs": 0,
    "disagree": 0,
    "old_s": 0.0,
    "new_s": 0.0,
    "old_true": 0,
    "old_false": 0,
}
orig = gm.compare


def compare(xs, ys):
    for x, y in zip(xs, ys, strict=True):
        if not torch.equal(x[1], y[1]):
            continue
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = old(x, y)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        b = new(x, y)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        stats["pairs"] += 1
        stats["old_s"] += t1 - t0
        stats["new_s"] += t2 - t1
        stats["old_true" if a else "old_false"] += 1
        if a != b:
            stats["disagree"] += 1
            print("DISAGREE real pair", x[0].shape)
    return orig(xs, ys)


gm.compare = compare
gm.same_up_to_ties = new
sys.argv = [sys.argv[1]] + sys.argv[3:]
gm.main()
print(json.dumps({"real": stats}))
