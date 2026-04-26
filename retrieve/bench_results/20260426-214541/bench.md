# Benchmark results

_run id_: `20260426-214541`
_device_: `NVIDIA A100-SXM4-40GB` (40441 MiB)
_versions_: torch=`2.10.0+cu128` triton=`3.6.0` git=``

## linr_v1_full

| cell | impl | median (ms) | p20 | p80 | index (MiB) | fwd peak (MiB) | transient (MiB) | vs baseline | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| B=1,N=1048576,D=128,K=200 | torch | 0.639 | 0.637 | 0.641 | 512.0 | 524.4 | 4.3 | baseline |  |
| B=1,N=1048576,D=128,K=200 | triton | 0.588 | 0.586 | 0.589 | 512.0 | 524.4 | 4.3 | ✅ |  |
| B=1,N=2097152,D=128,K=200 | torch | 1.036 | 1.033 | 1.037 | 1024.0 | 1040.4 | 8.3 | baseline |  |
| B=1,N=2097152,D=128,K=200 | triton | 0.997 | 0.995 | 0.998 | 1024.0 | 1040.4 | 8.3 | ➖ |  |
| B=16,N=1048576,D=128,K=200 | torch | 1.359 | 1.356 | 1.369 | 512.0 | 584.7 | 64.6 | baseline |  |
| B=16,N=1048576,D=128,K=200 | triton | 1.104 | 1.101 | 1.106 | 512.0 | 584.7 | 64.6 | ✅ |  |
| B=16,N=2097152,D=128,K=200 | torch | 2.508 | 2.506 | 2.510 | 1024.0 | 1161.2 | 129.1 | baseline |  |
| B=16,N=2097152,D=128,K=200 | triton | 1.932 | 1.930 | 1.933 | 1024.0 | 1161.2 | 129.1 | ✅ |  |

## silvertorch

| cell | impl | median (ms) | p20 | p80 | index (MiB) | fwd peak (MiB) | transient (MiB) | vs baseline | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | codesigned | 0.331 | 0.330 | 0.333 | 205.4 | 218.1 | 4.4 | ✅ | recall@K=0.0026 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | composed_ivf_bloom | 4.973 | 4.970 | 4.976 | 205.4 | 1365.7 | 1152.0 | baseline | recall@K=0.0026 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | codesigned | 0.382 | 0.381 | 0.383 | 205.4 | 227.7 | 13.9 | ✅ | recall@K=0.007 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | composed_ivf_bloom | 6.477 | 6.473 | 6.480 | 205.4 | 1365.8 | 1152.0 | baseline | recall@K=0.007 |
| B=16,N=2097152,D=128,K=1024,n_lists=2048,n_probe=16 | codesigned | 0.335 | 0.334 | 0.335 | 410.5 | 422.6 | 3.7 | ✅ | recall@K=0.002 |
| B=16,N=2097152,D=128,K=1024,n_lists=2048,n_probe=16 | composed_ivf_bloom | 9.042 | 9.041 | 9.045 | 410.5 | 2722.8 | 2304.0 | baseline | recall@K=0.002 |
| B=16,N=2097152,D=128,K=1024,n_lists=2048,n_probe=64 | codesigned | 0.424 | 0.397 | 0.486 | 410.5 | 432.6 | 13.8 | ✅ | recall@K=0.0045 |
| B=16,N=2097152,D=128,K=1024,n_lists=2048,n_probe=64 | composed_ivf_bloom | 10.622 | 10.620 | 10.625 | 410.5 | 2722.8 | 2304.0 | baseline | recall@K=0.0045 |

