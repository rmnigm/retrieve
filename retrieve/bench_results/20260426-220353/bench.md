# Benchmark results

_run id_: `20260426-220353`
_device_: `NVIDIA A100-SXM4-40GB` (40441 MiB)
_versions_: torch=`2.10.0+cu128` triton=`3.6.0` git=``

## silvertorch

| cell | impl | median (ms) | p20 | p80 | index (MiB) | fwd peak (MiB) | transient (MiB) | vs baseline | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | codesigned | 0.331 | 0.330 | 0.331 | 205.4 | 218.2 | 4.5 | ✅ | pass_rate_measured=0.0253 recall@K=0.0443 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | composed_ivf_bloom | 4.920 | 4.895 | 5.003 | 205.4 | 1365.8 | 1152.0 | baseline | pass_rate_measured=0.0253 recall@K=0.0443 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | codesigned | 0.390 | 0.387 | 0.421 | 205.4 | 227.7 | 14.0 | ✅ | pass_rate_measured=0.0253 recall@K=0.1398 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | composed_ivf_bloom | 6.508 | 6.473 | 6.555 | 205.4 | 1365.7 | 1152.0 | baseline | pass_rate_measured=0.0253 recall@K=0.1399 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=16 | codesigned | 0.317 | 0.313 | 0.337 | 51.3 | 64.1 | 4.4 | ✅ | pass_rate_measured=0.0255 recall@K=0.0815 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=16 | composed_ivf_bloom | 1.875 | 1.873 | 1.876 | 51.3 | 347.6 | 288.0 | baseline | pass_rate_measured=0.0255 recall@K=0.0815 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64 | codesigned | 0.387 | 0.385 | 0.398 | 51.3 | 73.4 | 13.8 | ✅ | pass_rate_measured=0.0255 recall@K=0.269 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64 | composed_ivf_bloom | 3.415 | 3.412 | 3.417 | 51.3 | 787.9 | 728.2 | baseline | pass_rate_measured=0.0255 recall@K=0.269 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=16 | codesigned | 0.327 | 0.324 | 0.336 | 12.8 | 25.4 | 4.3 | ✅ | pass_rate_measured=0.0254 recall@K=0.1737 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=16 | composed_ivf_bloom | 1.057 | 1.056 | 1.058 | 12.8 | 194.1 | 173.0 | baseline | pass_rate_measured=0.0254 recall@K=0.1737 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64 | codesigned | 0.398 | 0.393 | 0.430 | 12.8 | 34.2 | 13.1 | ✅ | pass_rate_measured=0.0254 recall@K=0.5241 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64 | composed_ivf_bloom | 2.504 | 2.502 | 2.507 | 12.8 | 710.5 | 689.4 | baseline | pass_rate_measured=0.0254 recall@K=0.5241 |

## silvertorch_mask

| cell | impl | median (ms) | p20 | p80 | index (MiB) | fwd peak (MiB) | transient (MiB) | vs baseline | extra |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.01 | codesigned | 0.279 | 0.278 | 0.279 | 205.4 | 243.7 | 14.0 | ✅ | pass_rate_measured=0.01 recall@K=0.1431 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.01 | composed_ivf | 2.389 | 2.387 | 2.391 | 141.4 | 900.3 | 734.5 | baseline | pass_rate_measured=0.01 recall@K=0.1431 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.1 | codesigned | 0.286 | 0.285 | 0.288 | 205.4 | 243.7 | 14.0 | ✅ | pass_rate_measured=0.0999 recall@K=0.2126 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.1 | composed_ivf | 2.406 | 2.401 | 2.409 | 141.4 | 900.3 | 734.5 | baseline | pass_rate_measured=0.0999 recall@K=0.2126 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.5 | codesigned | 0.331 | 0.330 | 0.332 | 205.4 | 243.7 | 14.0 | ✅ | pass_rate_measured=0.4998 recall@K=0.2564 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64,pass=0.5 | composed_ivf | 2.393 | 2.390 | 2.407 | 141.4 | 900.3 | 734.5 | baseline | pass_rate_measured=0.4998 recall@K=0.2563 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.01 | codesigned | 0.261 | 0.260 | 0.261 | 51.3 | 77.4 | 13.8 | ✅ | pass_rate_measured=0.01 recall@K=0.3296 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.01 | composed_ivf | 2.347 | 2.343 | 2.348 | 35.3 | 771.9 | 724.2 | baseline | pass_rate_measured=0.01 recall@K=0.3296 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.1 | codesigned | 0.267 | 0.267 | 0.268 | 51.3 | 77.4 | 13.8 | ✅ | pass_rate_measured=0.0999 recall@K=0.4442 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.1 | composed_ivf | 2.332 | 2.316 | 2.334 | 35.3 | 771.9 | 724.2 | baseline | pass_rate_measured=0.0999 recall@K=0.4442 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.5 | codesigned | 0.281 | 0.280 | 0.282 | 51.3 | 77.4 | 13.8 | ✅ | pass_rate_measured=0.4996 recall@K=0.5056 |
| B=16,N=262144,D=128,K=1024,n_lists=256,n_probe=64,pass=0.5 | composed_ivf | 2.349 | 2.348 | 2.350 | 35.3 | 771.9 | 724.2 | baseline | pass_rate_measured=0.4996 recall@K=0.5056 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.01 | codesigned | 0.247 | 0.247 | 0.248 | 12.8 | 35.2 | 13.1 | ✅ | pass_rate_measured=0.01 recall@K=0.6464 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.01 | composed_ivf | 2.227 | 2.223 | 2.242 | 8.8 | 706.8 | 688.6 | baseline | pass_rate_measured=0.01 recall@K=0.6464 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.1 | codesigned | 0.252 | 0.251 | 0.253 | 12.8 | 35.2 | 13.1 | ✅ | pass_rate_measured=0.0999 recall@K=0.9961 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.1 | composed_ivf | 2.226 | 2.224 | 2.242 | 8.8 | 706.8 | 688.6 | baseline | pass_rate_measured=0.0999 recall@K=0.9961 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.5 | codesigned | 0.256 | 0.256 | 0.257 | 12.8 | 35.2 | 13.1 | ✅ | pass_rate_measured=0.4993 recall@K=0.9947 |
| B=16,N=65536,D=128,K=1024,n_lists=64,n_probe=64,pass=0.5 | composed_ivf | 2.239 | 2.209 | 2.244 | 8.8 | 706.8 | 688.6 | baseline | pass_rate_measured=0.4993 recall@K=0.9947 |

