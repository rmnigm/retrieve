# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by the retrieve authors, 2026-10-14: package silvertorch -> silvertorch_fork, ops torch.ops.st -> torch.ops.stfork; builds is_topk / fresh_index_post_processing; -O3 host code; metadata in pyproject.toml (official-fork/CHANGES.md).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import os

from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDA_HOME,
    CUDAExtension,
)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

cpu_sources = [
    "silvertorch_fork/ops/csrc/bloom_index_util.cpp",
    "silvertorch_fork/ops/csrc/bloom_indexer.cpp",
    "silvertorch_fork/ops/csrc/bloom_index_search.cpp",
    "silvertorch_fork/ops/csrc/expression_query_parser.cpp",
    "silvertorch_fork/ops/csrc/fused_kmean_ann.cpp",
    # fork (OF-6): registered upstream but absent from its build
    "silvertorch_fork/ops/csrc/is_topk.cpp",
    "silvertorch_fork/ops/csrc/fresh_index_post_processing.cpp",
]

cuda_sources = [
    "silvertorch_fork/ops/csrc/bloom_indexer_cuda.cu",
    "silvertorch_fork/ops/csrc/bloom_index_search_cuda.cu",
    "silvertorch_fork/ops/csrc/expression_query_parser_cuda.cu",
    "silvertorch_fork/ops/csrc/faster_repeat_interleave.cu",
    "silvertorch_fork/ops/csrc/fused_kmean_ann_cuda.cu",
    # fork (OF-6): registered upstream but absent from its build
    "silvertorch_fork/ops/csrc/is_topk.cu",
    "silvertorch_fork/ops/csrc/fresh_index_post_processing.cu",
]

# Resolve paths relative to this file
cpu_sources = [os.path.join(ROOT_DIR, s) for s in cpu_sources]
cuda_sources = [os.path.join(ROOT_DIR, s) for s in cuda_sources]
include_dirs = [os.path.join(ROOT_DIR, "silvertorch_fork", "ops", "csrc")]

extra_compile_args = {
    "cxx": [
        "-O3",
        "-Wall",
        "-Wextra",
        "-Wno-sign-conversion",
        "-Wno-sign-compare",
        "-Wno-unused-parameter",
        "-Wno-unknown-pragmas",
        "-Wno-unused-function",
    ],
}

if CUDA_HOME is not None:
    # fork (OF-11): upstream passes no -O to nvcc, so the .cu host code built at gcc's -O0
    extra_compile_args["nvcc"] = [
        "-O3",
        "-Xcompiler",
        "-O3",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
    ]
    ext_modules = [
        CUDAExtension(
            name="silvertorch_fork._C",
            sources=cpu_sources + cuda_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ]
else:
    ext_modules = [
        CppExtension(
            name="silvertorch_fork._C",
            sources=cpu_sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ]

setup(
    packages=[
        "silvertorch_fork",
        "silvertorch_fork.ops",
        "silvertorch_fork.modules",
        "silvertorch_fork.modules.tests",
    ],
    package_dir={
        "silvertorch_fork": "silvertorch_fork",
        "silvertorch_fork.ops": "silvertorch_fork/ops",
        "silvertorch_fork.modules": "silvertorch_fork/modules",
        "silvertorch_fork.modules.tests": "silvertorch_fork/modules/tests",
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
