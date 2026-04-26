# Benchmark results

## linr_v1_unmasked

| cell | impl | median (ms) | p20 (ms) | p80 (ms) | peak mem (MiB) | correct | verdict | notes |
|---|---|---:|---:|---:|---:|---|---|---|
| B=1,N=1048576,D=128,K=200 | torch | 0.607 | 0.603 | 0.610 | 524.4 | ✓ | baseline |  |
| B=1,N=1048576,D=128,K=200 | triton | 0.591 | 0.590 | 0.594 | 524.4 | ✓ | ➖ | vs_torch=➖ |
| B=16,N=1048576,D=128,K=200 | torch | 1.371 | 1.368 | 1.385 | 584.8 | ✓ | baseline |  |
| B=16,N=1048576,D=128,K=200 | triton | 1.107 | 1.104 | 1.111 | 584.8 | ✓ | ✅ | vs_torch=✅ |

## silvertorch

| cell | impl | median (ms) | p20 (ms) | p80 (ms) | peak mem (MiB) | correct | verdict | notes |
|---|---|---:|---:|---:|---:|---|---|---|
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | ivf+bloom (composed) | 4.919 | 4.919 | 4.921 | 2131.3 | ✓ |  | recall@K=0.003 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=16 | silvertorch (codesigned) | 0.330 | 0.329 | 0.331 | 983.6 | ✓ |  | recall@K=0.003 vs_ivf+bloom (composed)=✅ speedup=14.92x mem_save=54% |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | ivf+bloom (composed) | 6.515 | 6.512 | 6.516 | 2131.3 | ✓ |  | recall@K=0.007 |
| B=16,N=1048576,D=128,K=1024,n_lists=1024,n_probe=64 | silvertorch (codesigned) | 0.418 | 0.399 | 0.429 | 993.3 | ✓ |  | recall@K=0.007 vs_ivf+bloom (composed)=✅ speedup=15.59x mem_save=53% |

