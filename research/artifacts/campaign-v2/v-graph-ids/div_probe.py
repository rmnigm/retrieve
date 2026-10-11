"""Which division formulations does inductor compile to eager's correctly rounded fp32 quotient?
Every arXiv d128 query, `torch.compile` vs eager `x / abs_max`; the generated kernel of each.

usage: div_probe.py OUT_DIR
"""

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl
from bench import config, inputs
from torch._inductor.utils import run_and_get_code
from torch.library import triton_op, wrap_triton

CFG = Path(__file__).resolve().parents[4] / "evaluation" / "config"


def m_of(x):
    return x.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)


def f32(x):
    return x / m_of(x)


def f64(x):
    m = m_of(x)
    return (x.double() / m.double()).float()


def true_div(x):
    return torch.true_divide(x, m_of(x))


@triton.jit
def _div_rn_kernel(x_ptr, m_ptr, out_ptr, D: tl.constexpr, D_PAD: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, D_PAD)
    x = tl.load(x_ptr + r * D + c, mask=c < D)
    m = tl.load(m_ptr + r)
    tl.store(out_ptr + r * D + c, tl.div_rn(x, m), mask=c < D)


@triton_op("probe::div_rn_rows", mutates_args=())
def div_rn_rows(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    d = x.shape[1]
    wrap_triton(_div_rn_kernel)[(x.shape[0],)](
        x, m, out, D=d, D_PAD=triton.next_power_of_2(d)
    )
    return out


def triton_rn(x):
    return div_rn_rows(x.contiguous(), m_of(x).reshape(-1).contiguous())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    jobs = config.load_matrix(
        CFG / "arxiv.yaml",
        CFG / "suites.yaml",
        "h2h",
        algos=["silvertorch"],
        backends=["triton"],
        seeds=[1],
    )
    q = inputs.load_inputs(jobs[0].data, torch.device("cuda"), with_filters=False)[
        "queries"
    ]
    q = q.cuda()
    ref = f32(q)
    res = {}
    for name, fn in [
        ("f32", f32),
        ("f64", f64),
        ("true_div", true_div),
        ("triton_div_rn", triton_rn),
    ]:
        torch._dynamo.reset()
        out, code = run_and_get_code(
            torch.compile(fn, dynamic=False, fullgraph=True), q
        )
        res[name] = {
            "eager_equals_f32": bool(torch.equal(fn(q), ref)),
            "compiled_elems_diff": int((out != ref).sum()),
        }
        (a.out / f"{name}.py").write_text("\n\n".join(code))
        print(name, json.dumps(res[name]), flush=True)
    (a.out / "div_probe.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
