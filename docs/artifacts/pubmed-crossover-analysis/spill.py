"""Compile _codesigned_probe_score_exact_kernel for (D_PAD, BLOCK_P, num_warps) offline (ptxas, sm_80;
no kernel launch) and print registers / spill stack from cuobjdump -res-usage."""
import os, re, subprocess, sys, tempfile
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp()
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from retrieve.ops.triton.codesigned_probe_score_exact import _codesigned_probe_score_exact_kernel as K

CU = os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/cuobjdump")
sig = {"q_codes_ptr": "*i8", "q_scales_ptr": "*fp32", "probe_ids_ptr": "*i64", "offsets_ptr": "*i64",
       "item_codes_ptr": "*i8", "item_attrs_ptr": "*i64", "is_reverse_ptr": "*i8", "query_attrs_ptr": "*i64",
       "out_scores_ptr": "*fp32", "global_scale": "fp32", "n_probe": "i32", "width": "i32", "tiles_y": "i32",
       "stride_qcb": "i32", "stride_cn": "i32", "stride_ian": "i32", "stride_iac": "i32", "stride_iaa": "i32",
       "stride_qab": "i32", "stride_qac": "i32", "stride_ob": "i32"}
for d, dpad, c, a in [(128, 128, 5, 4), (192, 256, 2, 32), (384, 512, 5, 4), (768, 1024, 5, 4)]:
    for bp, nw in [(256, 4), (256, 8), (128, 4), (64, 4), (32, 4), (16, 4)]:
        const = dict(D=d, D_PAD=dpad, NPP=32, C=c, A_MAX=a, BLOCK_P=bp, WIDE=False, stride_iaa=1, stride_qac=1)
        s = {**sig, **{k: "constexpr" for k in const}}
        div16 = [k for k in sig if k.endswith("_ptr")] + ["stride_qcb", "stride_cn"] + (["stride_ian", "stride_iac"] if a == 32 else [])
        src = ASTSource(fn=K, signature=s, constexprs={(K.arg_names.index(k),): v for k, v in const.items()},
                        attrs={(K.arg_names.index(k),): [["tt.divisibility", 16]] for k in div16})
        ck = triton.compile(src, target=GPUTarget("cuda", 80, 32), options={"num_warps": nw, "num_stages": 3})
        with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
            f.write(ck.asm["cubin"]); f.flush()
            ru = re.search(r"REG:\d+ STACK:\d+", subprocess.run([CU, "-res-usage", f.name], capture_output=True, text=True).stdout).group()
            sass = subprocess.run([CU, "-sass", f.name], capture_output=True, text=True).stdout
        print(f"D={d} D_PAD={dpad} C={c} A={a} BLOCK_P={bp} warps={nw}: {ru} LDL={sass.count('LDL')} STL={sass.count('STL')}", flush=True)
