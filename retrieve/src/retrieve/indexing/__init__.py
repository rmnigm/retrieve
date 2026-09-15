"""Index-build-time math: clustering, the IVF layouts, quantizers and bloom hashing. Reached
as ``retrieve.indexing.<name>``; the modules consume it through their constructor arguments."""

from retrieve.indexing.bloom_hash import (
    build_query_signatures,
    build_signatures,
    build_transposed_sigs,
    generate_clause_salt,
    generate_seeds,
    words_per_cluster,
)
from retrieve.indexing.ivf import csr_layout, padded_layout
from retrieve.indexing.kmeans import KMeans
from retrieve.indexing.quantize import (
    project_oporp_1bit_query,
    project_simhash_1bit_query,
    quantize_int8,
    quantize_int8_global,
    quantize_int8_global_codes,
    quantize_oporp_1bit,
    quantize_simhash_1bit,
)

__all__ = [
    "KMeans",
    "build_query_signatures",
    "build_signatures",
    "build_transposed_sigs",
    "csr_layout",
    "generate_clause_salt",
    "generate_seeds",
    "padded_layout",
    "project_oporp_1bit_query",
    "project_simhash_1bit_query",
    "quantize_int8",
    "quantize_int8_global",
    "quantize_int8_global_codes",
    "quantize_oporp_1bit",
    "quantize_simhash_1bit",
    "words_per_cluster",
]
