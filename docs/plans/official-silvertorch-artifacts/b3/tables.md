
#### End to end — eager latency, one cell per (dataset, filter, backend)

| dataset | filter | sweep | n_probe | backend | k | bs | eager median ms | p99 ms | qps | spread | sm_mhz | peak MiB | graph median ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | bloom | c0_maincat | 24 | official | 100 | 1 | 1.1761 | 1.3782 | 861 | 0.002 | 1215.0 | 3.8 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 100 | 8 | 1.3697 | 1.5584 | 5829 | 0.005 | 1395.0 | 30.4 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 100 | 16 | 1.2923 | 1.5045 | 12355 | 0.081 | 1410.0 | 61.3 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 500 | 1 | 1.0101 | 1.2248 | 929 | 0.151 | 1365.0 | 3.8 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 500 | 8 | 1.1303 | 1.3820 | 6722 | 0.143 | 1410.0 | 31.3 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 500 | 16 | 1.2641 | 1.4615 | 12234 | 0.014 | 1410.0 | 61.3 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 1000 | 1 | 1.1443 | 1.4665 | 845 | 0.069 | 1275.0 | 3.8 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 1000 | 8 | 1.1714 | 1.4143 | 6500 | 0.076 | 1410.0 | 31.3 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | official | 1000 | 16 | 1.3535 | 1.5643 | 12100 | 0.044 | 1410.0 | 61.3 | not_capturable |
| arxiv | bloom | c0_maincat | 24 | torch | 100 | 1 | 1.1621 | 1.3490 | 861 | 0.002 | 1410.0 | 107.7 | 0.2638 |
| arxiv | bloom | c0_maincat | 24 | torch | 100 | 8 | 4.2533 | 4.3877 | 1878 | 0.000 | 1410.0 | 862.1 | 1.2326 |
| arxiv | bloom | c0_maincat | 24 | torch | 100 | 16 | 8.1699 | 8.2771 | 1957 | 0.000 | 1410.0 | 1725.1 | 2.3188 |
| arxiv | bloom | c0_maincat | 24 | torch | 500 | 1 | 1.1629 | 1.3053 | 864 | 0.007 | 1410.0 | 107.7 | 0.2763 |
| arxiv | bloom | c0_maincat | 24 | torch | 500 | 8 | 4.2720 | 4.3874 | 1870 | 0.000 | 1410.0 | 861.7 | 1.2479 |
| arxiv | bloom | c0_maincat | 24 | torch | 500 | 16 | 8.1874 | 8.3084 | 1953 | 0.000 | 1410.0 | 1725.1 | 2.3310 |
| arxiv | bloom | c0_maincat | 24 | torch | 1000 | 1 | 1.1554 | 1.3405 | 871 | 0.021 | 1410.0 | 107.7 | 0.2785 |
| arxiv | bloom | c0_maincat | 24 | torch | 1000 | 8 | 4.2733 | 4.4475 | 1869 | 0.000 | 1410.0 | 861.7 | 1.2469 |
| arxiv | bloom | c0_maincat | 24 | torch | 1000 | 16 | 8.1894 | 8.3583 | 1952 | 0.000 | 1410.0 | 1725.1 | 2.3396 |
| arxiv | bloom | c0_maincat | 24 | triton | 100 | 1 | 1.0833 | 1.2900 | 899 | 0.197 | 1155.0 | 2.1 | 0.2025 |
| arxiv | bloom | c0_maincat | 24 | triton | 100 | 8 | 1.0755 | 1.5948 | 4666 | 0.001 | 1395.0 | 16.7 | 0.2867 |
| arxiv | bloom | c0_maincat | 24 | triton | 100 | 16 | 1.0944 | 1.2836 | 14518 | 0.017 | 1410.0 | 31.8 | 0.4276 |
| arxiv | bloom | c0_maincat | 24 | triton | 500 | 1 | 1.0767 | 1.5499 | 850 | 0.005 | 1155.0 | 2.1 | 0.2024 |
| arxiv | bloom | c0_maincat | 24 | triton | 500 | 8 | 1.0655 | 1.2661 | 7291 | 0.007 | 1410.0 | 16.1 | 0.2996 |
| arxiv | bloom | c0_maincat | 24 | triton | 500 | 16 | 1.0831 | 1.2693 | 14629 | 0.007 | 1410.0 | 31.9 | 0.4437 |
| arxiv | bloom | c0_maincat | 24 | triton | 1000 | 1 | 1.0674 | 1.2576 | 887 | 0.081 | 1155.0 | 2.1 | 0.1990 |
| arxiv | bloom | c0_maincat | 24 | triton | 1000 | 8 | 1.0661 | 90.4968 | 2370 | 0.008 | 1410.0 | 16.1 | 0.3022 |
| arxiv | bloom | c0_maincat | 24 | triton | 1000 | 16 | 1.0702 | 1.2462 | 14924 | 0.001 | 1410.0 | 32.0 | 0.4438 |
| arxiv | bloom | c0_maincat | 32 | official | 100 | 1 | 1.3004 | 1.4967 | 761 | 0.001 | 1155.0 | 5.0 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 100 | 8 | 1.3719 | 1.5809 | 5766 | 0.173 | 1410.0 | 40.3 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 100 | 16 | 1.4506 | 1.6412 | 11102 | 0.008 | 1410.0 | 80.5 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 500 | 1 | 1.2970 | 1.6932 | 763 | 0.005 | 1155.0 | 5.0 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 500 | 8 | 1.2703 | 1.6041 | 6209 | 0.152 | 1410.0 | 40.3 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 500 | 16 | 1.3974 | 1.5642 | 11337 | 0.082 | 1410.0 | 80.5 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 1000 | 1 | 1.1676 | 1.3535 | 877 | 0.018 | 1380.0 | 5.0 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 1000 | 8 | 1.1331 | 1.4729 | 6672 | 0.128 | 1410.0 | 40.3 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | official | 1000 | 16 | 1.4748 | 1.9190 | 10809 | 0.148 | 1410.0 | 80.5 | not_capturable |
| arxiv | bloom | c0_maincat | 32 | torch | 100 | 1 | 1.2402 | 1.4086 | 807 | 0.002 | 1410.0 | 143.6 | 0.3126 |
| arxiv | bloom | c0_maincat | 32 | torch | 100 | 8 | 5.5653 | 5.6881 | 1436 | 0.000 | 1410.0 | 1149.0 | 1.5711 |
| arxiv | bloom | c0_maincat | 32 | torch | 100 | 16 | 10.8056 | 10.9480 | 1480 | 0.000 | 1410.0 | 2298.9 | 3.0168 |
| arxiv | bloom | c0_maincat | 32 | torch | 500 | 1 | 0.9936 | 1.1926 | 948 | 0.002 | 1410.0 | 143.6 | 0.3228 |
| arxiv | bloom | c0_maincat | 32 | torch | 500 | 8 | 5.5844 | 5.7327 | 1431 | 0.000 | 1410.0 | 1149.7 | 1.5857 |
| arxiv | bloom | c0_maincat | 32 | torch | 500 | 16 | 10.8264 | 10.8925 | 1477 | 0.000 | 1410.0 | 2298.9 | 3.0320 |
| arxiv | bloom | c0_maincat | 32 | torch | 1000 | 1 | 1.1431 | 1.2450 | 897 | 0.128 | 1410.0 | 143.6 | 0.3266 |
| arxiv | bloom | c0_maincat | 32 | torch | 1000 | 8 | 5.5841 | 5.7166 | 1431 | 0.000 | 1410.0 | 1149.7 | 1.5857 |
| arxiv | bloom | c0_maincat | 32 | torch | 1000 | 16 | 10.8263 | 10.9502 | 1477 | 0.001 | 1410.0 | 2298.9 | 3.0284 |
| arxiv | bloom | c0_maincat | 32 | triton | 100 | 1 | 1.0685 | 1.2708 | 958 | 0.014 | 1155.0 | 2.7 | 0.1997 |
| arxiv | bloom | c0_maincat | 32 | triton | 100 | 8 | 1.0865 | 1.2677 | 7331 | 0.108 | 1410.0 | 21.3 | 0.3338 |
| arxiv | bloom | c0_maincat | 32 | triton | 100 | 16 | 1.0843 | 1.3510 | 14751 | 0.001 | 1410.0 | 42.3 | 0.5262 |
| arxiv | bloom | c0_maincat | 32 | triton | 500 | 1 | 1.0748 | 1.2668 | 932 | 0.004 | 1155.0 | 2.7 | 0.2006 |
| arxiv | bloom | c0_maincat | 32 | triton | 500 | 8 | 1.0702 | 1.2646 | 7465 | 0.009 | 1410.0 | 21.3 | 0.3475 |
| arxiv | bloom | c0_maincat | 32 | triton | 500 | 16 | 1.0721 | 1.2395 | 15132 | 0.004 | 1410.0 | 42.3 | 0.5424 |
| arxiv | bloom | c0_maincat | 32 | triton | 1000 | 1 | 1.0534 | 1.2283 | 949 | 0.023 | 1155.0 | 2.7 | 0.2006 |
| arxiv | bloom | c0_maincat | 32 | triton | 1000 | 8 | 1.0860 | 1.2568 | 7425 | 0.006 | 1410.0 | 21.4 | 0.3506 |
| arxiv | bloom | c0_maincat | 32 | triton | 1000 | 16 | 1.0738 | 1.2622 | 14883 | 0.001 | 1410.0 | 42.4 | 0.5422 |
| arxiv | clause | c0_maincat | 24 | official | 100 | 1 | 1.1673 | 1.3684 | 844 | 0.008 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 100 | 8 | 3.9564 | 4.1514 | 2020 | 0.002 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 100 | 16 | 7.1096 | 7.2361 | 2255 | 0.001 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 500 | 1 | 1.1698 | 1.4313 | 843 | 0.003 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 500 | 8 | 3.9504 | 4.0942 | 2028 | 0.003 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 500 | 16 | 7.1092 | 7.2894 | 2253 | 0.002 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 1000 | 1 | 1.1477 | 1.3530 | 869 | 0.045 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 1000 | 8 | 3.9451 | 4.1067 | 2029 | 0.009 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 24 | official | 1000 | 16 | 7.1113 | 7.2515 | 2253 | 0.001 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 24 | torch | 100 | 1 | 0.7346 | 0.7679 | 1352 | 0.000 | 1410.0 | 128.7 | 0.2608 |
| arxiv | clause | c0_maincat | 24 | torch | 100 | 8 | 4.0759 | 4.2131 | 1960 | 0.000 | 1410.0 | 1029.3 | 1.2065 |
| arxiv | clause | c0_maincat | 24 | torch | 100 | 16 | 7.8626 | 8.0007 | 2033 | 0.000 | 1410.0 | 2059.6 | 2.2757 |
| arxiv | clause | c0_maincat | 24 | torch | 500 | 1 | 0.9292 | 1.1054 | 1075 | 0.002 | 1410.0 | 128.7 | 0.2747 |
| arxiv | clause | c0_maincat | 24 | torch | 500 | 8 | 4.0954 | 4.2203 | 1950 | 0.000 | 1410.0 | 1029.3 | 1.2222 |
| arxiv | clause | c0_maincat | 24 | torch | 500 | 16 | 7.8796 | 8.0188 | 2029 | 0.000 | 1410.0 | 2059.7 | 2.2944 |
| arxiv | clause | c0_maincat | 24 | torch | 1000 | 1 | 0.9288 | 1.1239 | 1072 | 0.004 | 1410.0 | 128.7 | 0.2764 |
| arxiv | clause | c0_maincat | 24 | torch | 1000 | 8 | 4.0955 | 4.2382 | 1950 | 0.000 | 1410.0 | 1029.3 | 1.2215 |
| arxiv | clause | c0_maincat | 24 | torch | 1000 | 16 | 7.8830 | 8.0510 | 2028 | 0.000 | 1410.0 | 2059.7 | 2.2924 |
| arxiv | clause | c0_maincat | 24 | triton | 100 | 1 | 0.5601 | 0.6495 | 1705 | 0.006 | 1380.0 | 2.1 | 0.1941 |
| arxiv | clause | c0_maincat | 24 | triton | 100 | 8 | 0.6885 | 0.7399 | 11414 | 0.000 | 1410.0 | 16.0 | 0.2959 |
| arxiv | clause | c0_maincat | 24 | triton | 100 | 16 | 0.7013 | 0.8340 | 22750 | 0.007 | 1410.0 | 31.8 | 0.4524 |
| arxiv | clause | c0_maincat | 24 | triton | 500 | 1 | 0.6911 | 0.8648 | 1369 | 0.014 | 1170.0 | 2.1 | 0.1948 |
| arxiv | clause | c0_maincat | 24 | triton | 500 | 8 | 0.7017 | 0.7774 | 11290 | 0.002 | 1410.0 | 16.1 | 0.3090 |
| arxiv | clause | c0_maincat | 24 | triton | 500 | 16 | 0.6976 | 0.7728 | 23044 | 0.011 | 1410.0 | 31.9 | 0.4678 |
| arxiv | clause | c0_maincat | 24 | triton | 1000 | 1 | 0.6928 | 0.7398 | 1425 | 0.013 | 1275.0 | 2.1 | 0.1962 |
| arxiv | clause | c0_maincat | 24 | triton | 1000 | 8 | 0.7032 | 0.7414 | 11223 | 0.002 | 1410.0 | 16.1 | 0.3121 |
| arxiv | clause | c0_maincat | 24 | triton | 1000 | 16 | 0.7014 | 0.8885 | 22027 | 0.009 | 1410.0 | 32.0 | 0.4679 |
| arxiv | clause | c0_maincat | 32 | official | 100 | 1 | 1.1497 | 1.2600 | 875 | 0.012 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 100 | 8 | 3.9238 | 4.1017 | 2034 | 0.008 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 100 | 16 | 7.1312 | 7.2967 | 2240 | 0.001 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 500 | 1 | 1.1542 | 1.3594 | 873 | 0.013 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 500 | 8 | 3.9576 | 4.1407 | 2027 | 0.002 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 500 | 16 | 7.1421 | 7.3178 | 2238 | 0.001 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 1000 | 1 | 1.1607 | 1.4235 | 855 | 0.007 | 1410.0 | 51.7 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 1000 | 8 | 3.9533 | 4.0900 | 2032 | 0.010 | 1410.0 | 413.3 | not_capturable |
| arxiv | clause | c0_maincat | 32 | official | 1000 | 16 | 7.1472 | 7.3252 | 2236 | 0.001 | 1410.0 | 827.2 | not_capturable |
| arxiv | clause | c0_maincat | 32 | torch | 100 | 1 | 0.9019 | 1.0826 | 1083 | 0.002 | 1410.0 | 171.6 | 0.3081 |
| arxiv | clause | c0_maincat | 32 | torch | 100 | 8 | 5.3340 | 5.4938 | 1498 | 0.000 | 1410.0 | 1372.4 | 1.5582 |
| arxiv | clause | c0_maincat | 32 | torch | 100 | 16 | 10.4099 | 10.5546 | 1536 | 0.000 | 1410.0 | 2744.9 | 2.9900 |
| arxiv | clause | c0_maincat | 32 | torch | 500 | 1 | 0.9194 | 1.0414 | 1081 | 0.000 | 1410.0 | 171.6 | 0.3214 |
| arxiv | clause | c0_maincat | 32 | torch | 500 | 8 | 5.3532 | 5.5194 | 1492 | 0.000 | 1410.0 | 1372.7 | 1.5733 |
| arxiv | clause | c0_maincat | 32 | torch | 500 | 16 | 10.4276 | 10.5659 | 1533 | 0.000 | 1410.0 | 2744.9 | 3.0166 |
| arxiv | clause | c0_maincat | 32 | torch | 1000 | 1 | 0.9259 | 1.0586 | 1073 | 0.000 | 1410.0 | 171.6 | 0.3235 |
| arxiv | clause | c0_maincat | 32 | torch | 1000 | 8 | 5.3535 | 5.5060 | 1492 | 0.000 | 1410.0 | 1372.7 | 1.5739 |
| arxiv | clause | c0_maincat | 32 | torch | 1000 | 16 | 10.4288 | 10.5898 | 1533 | 0.000 | 1410.0 | 2744.9 | 3.0061 |
| arxiv | clause | c0_maincat | 32 | triton | 100 | 1 | 0.6948 | 0.8808 | 1350 | 0.004 | 1155.0 | 2.7 | 0.1932 |
| arxiv | clause | c0_maincat | 32 | triton | 100 | 8 | 0.7040 | 0.8793 | 10337 | 0.225 | 1410.0 | 21.3 | 0.3494 |
| arxiv | clause | c0_maincat | 32 | triton | 100 | 16 | 0.7089 | 2.0361 | 19981 | 0.019 | 1410.0 | 42.3 | 0.5604 |
| arxiv | clause | c0_maincat | 32 | triton | 500 | 1 | 0.6775 | 0.8538 | 1440 | 0.003 | 1260.0 | 2.7 | 0.1953 |
| arxiv | clause | c0_maincat | 32 | triton | 500 | 8 | 0.7006 | 0.8695 | 11295 | 0.005 | 1410.0 | 21.3 | 0.3623 |
| arxiv | clause | c0_maincat | 32 | triton | 500 | 16 | 0.7095 | 0.9235 | 19712 | 0.191 | 1410.0 | 42.3 | 0.5762 |
| arxiv | clause | c0_maincat | 32 | triton | 1000 | 1 | 0.6863 | 0.8697 | 1419 | 0.016 | 1395.0 | 2.7 | 0.1961 |
| arxiv | clause | c0_maincat | 32 | triton | 1000 | 8 | 0.6910 | 0.8502 | 11561 | 0.003 | 1410.0 | 21.4 | 0.3657 |
| arxiv | clause | c0_maincat | 32 | triton | 1000 | 16 | 0.7007 | 0.8008 | 22639 | 0.013 | 1410.0 | 42.4 | 0.5761 |
| arxiv | none | full_scan | — | official | 100 | 1 | 0.8337 | 1.0155 | 1185 | 0.015 | 1395.0 | 3.8 | not_capturable |
| arxiv | none | full_scan | — | official | 100 | 8 | 0.8344 | 1.0175 | 9410 | 0.016 | 1410.0 | 30.8 | not_capturable |
| arxiv | none | full_scan | — | official | 100 | 16 | 0.8456 | 1.0175 | 18851 | 0.004 | 1410.0 | 60.8 | not_capturable |
| arxiv | none | full_scan | — | torch | 100 | 1 | 0.5885 | 0.6859 | 1712 | 0.132 | 1410.0 | 107.5 | 0.2463 |
| arxiv | none | full_scan | — | torch | 100 | 8 | 2.5492 | 2.6484 | 3131 | 0.000 | 1410.0 | 860.4 | 1.1325 |
| arxiv | none | full_scan | — | torch | 100 | 16 | 4.8524 | 4.9000 | 3294 | 0.000 | 1410.0 | 1721.8 | 2.1285 |
| arxiv | none | full_scan | — | triton | 100 | 1 | 0.5408 | 0.6534 | 1837 | 0.012 | 1350.0 | 2.1 | 0.1669 |
| arxiv | none | full_scan | — | triton | 100 | 8 | 0.6587 | 0.7143 | 11923 | 0.002 | 1410.0 | 16.9 | 0.2400 |
| arxiv | none | full_scan | — | triton | 100 | 16 | 0.6680 | 0.7123 | 23821 | 0.007 | 1410.0 | 31.8 | 0.3484 |
| goodreads | bloom | c0_genre | 24 | official | 100 | 1 | 1.3301 | 1.5274 | 742 | 0.019 | 1200.0 | 13.4 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 100 | 8 | 1.4113 | 1.6024 | 5747 | 0.092 | 1410.0 | 107.9 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 100 | 16 | 2.0629 | 2.2420 | 7501 | 0.000 | 1410.0 | 215.2 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 500 | 1 | 1.2177 | 1.3210 | 803 | 0.207 | 1350.0 | 13.4 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 500 | 8 | 1.3041 | 1.5120 | 5916 | 0.001 | 1410.0 | 107.3 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 500 | 16 | 2.0374 | 2.3176 | 7362 | 0.008 | 1410.0 | 215.2 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 1000 | 1 | 1.2024 | 1.3885 | 825 | 0.098 | 1155.0 | 13.4 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 1000 | 8 | 1.3077 | 1.6036 | 5898 | 0.096 | 1410.0 | 107.3 | not_capturable |
| goodreads | bloom | c0_genre | 24 | official | 1000 | 16 | 2.0228 | 2.2265 | 7904 | 0.002 | 1410.0 | 215.2 | not_capturable |
| goodreads | bloom | c0_genre | 24 | torch | 100 | 1 | 2.0877 | 2.2266 | 477 | 0.000 | 1410.0 | 383.7 | 0.5943 |
| goodreads | bloom | c0_genre | 24 | torch | 100 | 8 | 14.3254 | 14.4703 | 558 | 0.000 | 1410.0 | 3070.7 | 3.6095 |
| goodreads | bloom | c0_genre | 24 | torch | 100 | 16 | 28.5159 | 28.6971 | 561 | 0.000 | 1410.0 | 6140.3 | 7.1924 |
| goodreads | bloom | c0_genre | 24 | torch | 500 | 1 | 2.1052 | 2.2280 | 474 | 0.000 | 1410.0 | 383.7 | 0.6079 |
| goodreads | bloom | c0_genre | 24 | torch | 500 | 8 | 14.3451 | 14.5025 | 557 | 0.000 | 1410.0 | 3069.9 | 3.6451 |
| goodreads | bloom | c0_genre | 24 | torch | 500 | 16 | 28.5320 | 28.6877 | 561 | 0.000 | 1410.0 | 6140.3 | 7.2074 |
| goodreads | bloom | c0_genre | 24 | torch | 1000 | 1 | 2.1119 | 2.2403 | 472 | 0.000 | 1410.0 | 383.7 | 0.6104 |
| goodreads | bloom | c0_genre | 24 | torch | 1000 | 8 | 14.3468 | 14.5076 | 557 | 0.000 | 1410.0 | 3069.9 | 3.6264 |
| goodreads | bloom | c0_genre | 24 | torch | 1000 | 16 | 28.5348 | 28.7160 | 561 | 0.000 | 1410.0 | 6140.3 | 7.2074 |
| goodreads | bloom | c0_genre | 24 | triton | 100 | 1 | 0.9989 | 1.2894 | 956 | 0.004 | 1185.0 | 7.3 | 0.2172 |
| goodreads | bloom | c0_genre | 24 | triton | 100 | 8 | 1.0199 | 1.7973 | 7493 | 0.011 | 1410.0 | 56.3 | 0.5060 |
| goodreads | bloom | c0_genre | 24 | triton | 100 | 16 | 1.1118 | 1.2513 | 14305 | 0.000 | 1410.0 | 112.3 | 0.9702 |
| goodreads | bloom | c0_genre | 24 | triton | 500 | 1 | 1.0705 | 1.4204 | 885 | 0.070 | 1170.0 | 7.3 | 0.2319 |
| goodreads | bloom | c0_genre | 24 | triton | 500 | 8 | 1.0301 | 1.4090 | 7492 | 0.014 | 1410.0 | 56.4 | 0.5225 |
| goodreads | bloom | c0_genre | 24 | triton | 500 | 16 | 1.1298 | 1.2825 | 14081 | 0.000 | 1410.0 | 112.4 | 0.9856 |
| goodreads | bloom | c0_genre | 24 | triton | 1000 | 1 | 1.0036 | 1.2754 | 951 | 0.004 | 1260.0 | 7.3 | 0.2350 |
| goodreads | bloom | c0_genre | 24 | triton | 1000 | 8 | 1.0309 | 1.4986 | 7327 | 0.169 | 1410.0 | 56.4 | 0.5234 |
| goodreads | bloom | c0_genre | 24 | triton | 1000 | 16 | 1.1322 | 1.3627 | 14023 | 0.000 | 1410.0 | 112.5 | 0.9873 |
| goodreads | bloom | c0_genre | 32 | official | 100 | 1 | 1.3443 | 1.5248 | 734 | 0.004 | 1230.0 | 18.6 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 100 | 8 | 1.6366 | 90.9847 | 1275 | 0.093 | 1410.0 | 143.8 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 100 | 16 | 2.2598 | 2.4293 | 7088 | 0.008 | 1410.0 | 286.9 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 500 | 1 | 1.3220 | 1.7074 | 739 | 0.004 | 1335.0 | 18.6 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 500 | 8 | 1.5837 | 1.7748 | 5084 | 0.004 | 1410.0 | 143.8 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 500 | 16 | 2.2693 | 2.4708 | 7148 | 0.078 | 1410.0 | 286.9 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 1000 | 1 | 1.2157 | 1.4667 | 841 | 0.101 | 1410.0 | 18.6 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 1000 | 8 | 1.5888 | 1.7555 | 5096 | 0.001 | 1410.0 | 143.8 | not_capturable |
| goodreads | bloom | c0_genre | 32 | official | 1000 | 16 | 2.2926 | 2.5990 | 6007 | 0.024 | 1410.0 | 286.9 | not_capturable |
| goodreads | bloom | c0_genre | 32 | torch | 100 | 1 | 2.6681 | 2.7917 | 374 | 0.000 | 1410.0 | 511.7 | 0.7472 |
| goodreads | bloom | c0_genre | 32 | torch | 100 | 8 | 19.0647 | 19.2236 | 419 | 0.000 | 1410.0 | 4094.6 | 4.8016 |
| goodreads | bloom | c0_genre | 32 | torch | 100 | 16 | 37.8697 | 38.0248 | 422 | 0.000 | 1410.0 | 8186.4 | 9.3464 |
| goodreads | bloom | c0_genre | 32 | torch | 500 | 1 | 2.6877 | 2.7668 | 371 | 0.000 | 1410.0 | 511.7 | 0.7604 |
| goodreads | bloom | c0_genre | 32 | torch | 500 | 8 | 19.0839 | 19.2222 | 419 | 0.000 | 1410.0 | 4094.6 | 4.8225 |
| goodreads | bloom | c0_genre | 32 | torch | 500 | 16 | 37.8870 | 38.0488 | 422 | 0.000 | 1410.0 | 8186.4 | 9.3614 |
| goodreads | bloom | c0_genre | 32 | torch | 1000 | 1 | 2.6904 | 2.8457 | 371 | 0.000 | 1410.0 | 511.7 | 0.7610 |
| goodreads | bloom | c0_genre | 32 | torch | 1000 | 8 | 19.0854 | 19.2137 | 419 | 0.000 | 1410.0 | 4094.6 | 4.8185 |
| goodreads | bloom | c0_genre | 32 | torch | 1000 | 16 | 37.8892 | 38.0625 | 422 | 0.000 | 1410.0 | 8186.4 | 9.3641 |
| goodreads | bloom | c0_genre | 32 | triton | 100 | 1 | 1.0555 | 1.4810 | 920 | 0.011 | 1230.0 | 9.7 | 0.2381 |
| goodreads | bloom | c0_genre | 32 | triton | 100 | 8 | 1.0958 | 1.3687 | 7072 | 0.050 | 1410.0 | 75.0 | 0.6649 |
| goodreads | bloom | c0_genre | 32 | triton | 100 | 16 | 1.3501 | 1.4068 | 11797 | 0.000 | 1410.0 | 149.7 | 1.2060 |
| goodreads | bloom | c0_genre | 32 | triton | 500 | 1 | 1.0265 | 1.2084 | 938 | 0.025 | 1290.0 | 9.7 | 0.2527 |
| goodreads | bloom | c0_genre | 32 | triton | 500 | 8 | 1.2233 | 1.5018 | 6644 | 0.107 | 1410.0 | 75.0 | 0.6820 |
| goodreads | bloom | c0_genre | 32 | triton | 500 | 16 | 1.3679 | 1.4961 | 11644 | 0.000 | 1410.0 | 149.8 | 1.2215 |
| goodreads | bloom | c0_genre | 32 | triton | 1000 | 1 | 1.0601 | 1.2558 | 930 | 0.033 | 1260.0 | 9.7 | 0.2536 |
| goodreads | bloom | c0_genre | 32 | triton | 1000 | 8 | 1.1025 | 1.3569 | 7060 | 0.041 | 1410.0 | 75.1 | 0.6834 |
| goodreads | bloom | c0_genre | 32 | triton | 1000 | 16 | 1.3704 | 1.5034 | 11621 | 0.000 | 1410.0 | 149.9 | 1.2240 |
| goodreads | clause | c0_genre | 24 | official | 100 | 1 | 1.0528 | 1.2515 | 969 | 0.011 | 1410.0 | 14.5 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 100 | 8 | 1.6769 | 1.8196 | 4744 | 0.009 | 1410.0 | 114.2 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 100 | 16 | 3.0441 | 3.2090 | 5240 | 0.002 | 1410.0 | 228.3 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 500 | 1 | 0.9091 | 1.2265 | 1035 | 0.160 | 1410.0 | 14.5 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 500 | 8 | 1.7047 | 1.9571 | 4668 | 0.000 | 1410.0 | 114.2 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 500 | 16 | 3.0602 | 3.2392 | 5212 | 0.001 | 1410.0 | 228.3 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 1000 | 1 | 1.0477 | 1.2892 | 999 | 0.078 | 1410.0 | 14.5 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 1000 | 8 | 1.7084 | 1.8860 | 4540 | 0.001 | 1410.0 | 114.2 | not_capturable |
| goodreads | clause | c0_genre | 24 | official | 1000 | 16 | 3.0665 | 3.2315 | 5202 | 0.001 | 1410.0 | 228.3 | not_capturable |
| goodreads | clause | c0_genre | 24 | torch | 100 | 1 | 1.9800 | 2.0818 | 503 | 0.000 | 1410.0 | 458.9 | 0.5776 |
| goodreads | clause | c0_genre | 24 | torch | 100 | 8 | 13.8426 | 13.9952 | 578 | 0.000 | 1410.0 | 3668.7 | 3.4544 |
| goodreads | clause | c0_genre | 24 | torch | 100 | 16 | 27.5755 | 27.7436 | 580 | 0.000 | 1410.0 | 7334.7 | 6.8110 |
| goodreads | clause | c0_genre | 24 | torch | 500 | 1 | 2.0002 | 2.1585 | 498 | 0.000 | 1410.0 | 458.9 | 0.5925 |
| goodreads | clause | c0_genre | 24 | torch | 500 | 8 | 13.8641 | 14.0223 | 577 | 0.000 | 1410.0 | 3668.7 | 3.4764 |
| goodreads | clause | c0_genre | 24 | torch | 500 | 16 | 27.5848 | 27.7487 | 580 | 0.000 | 1410.0 | 7334.7 | 6.8281 |
| goodreads | clause | c0_genre | 24 | torch | 1000 | 1 | 2.0062 | 2.1472 | 497 | 0.000 | 1410.0 | 458.9 | 0.5944 |
| goodreads | clause | c0_genre | 24 | torch | 1000 | 8 | 13.8646 | 14.0331 | 577 | 0.000 | 1410.0 | 3668.7 | 3.4667 |
| goodreads | clause | c0_genre | 24 | torch | 1000 | 16 | 27.5873 | 27.7210 | 580 | 0.000 | 1410.0 | 7334.7 | 6.8297 |
| goodreads | clause | c0_genre | 24 | triton | 100 | 1 | 0.5189 | 0.6774 | 1843 | 0.006 | 1410.0 | 7.3 | 0.2025 |
| goodreads | clause | c0_genre | 24 | triton | 100 | 8 | 0.6492 | 0.8315 | 12327 | 0.005 | 1410.0 | 56.3 | 0.4382 |
| goodreads | clause | c0_genre | 24 | triton | 100 | 16 | 0.9214 | 1.0443 | 17253 | 0.000 | 1410.0 | 112.3 | 0.8417 |
| goodreads | clause | c0_genre | 24 | triton | 500 | 1 | 0.6385 | 0.8538 | 1490 | 0.004 | 1410.0 | 7.3 | 0.2177 |
| goodreads | clause | c0_genre | 24 | triton | 500 | 8 | 0.6560 | 0.8465 | 11898 | 0.003 | 1410.0 | 56.4 | 0.4553 |
| goodreads | clause | c0_genre | 24 | triton | 500 | 16 | 0.9401 | 1.0934 | 16911 | 0.000 | 1410.0 | 112.4 | 0.8568 |
| goodreads | clause | c0_genre | 24 | triton | 1000 | 1 | 0.6479 | 0.8310 | 1476 | 0.003 | 1410.0 | 7.3 | 0.2200 |
| goodreads | clause | c0_genre | 24 | triton | 1000 | 8 | 0.6553 | 0.8396 | 11901 | 0.005 | 1410.0 | 56.4 | 0.4559 |
| goodreads | clause | c0_genre | 24 | triton | 1000 | 16 | 0.9422 | 1.0636 | 16878 | 0.000 | 1410.0 | 112.5 | 0.8591 |
| goodreads | clause | c0_genre | 32 | official | 100 | 1 | 1.0512 | 1.4755 | 984 | 0.144 | 1410.0 | 18.7 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 100 | 8 | 1.8373 | 2.0014 | 4332 | 0.003 | 1410.0 | 149.9 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 100 | 16 | 3.2912 | 3.4304 | 4849 | 0.000 | 1410.0 | 299.8 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 500 | 1 | 0.9084 | 1.1414 | 1057 | 0.029 | 1410.0 | 18.7 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 500 | 8 | 1.8633 | 2.0393 | 4276 | 0.008 | 1410.0 | 149.9 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 500 | 16 | 3.3399 | 95.9773 | 1436 | 0.010 | 1410.0 | 299.8 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 1000 | 1 | 0.9053 | 1.0960 | 1061 | 0.007 | 1410.0 | 18.7 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 1000 | 8 | 1.8669 | 2.0456 | 4267 | 0.001 | 1410.0 | 149.9 | not_capturable |
| goodreads | clause | c0_genre | 32 | official | 1000 | 16 | 3.3351 | 3.5838 | 4478 | 0.005 | 1410.0 | 299.8 | not_capturable |
| goodreads | clause | c0_genre | 32 | torch | 100 | 1 | 2.5440 | 2.6388 | 392 | 0.000 | 1410.0 | 611.2 | 0.7253 |
| goodreads | clause | c0_genre | 32 | torch | 100 | 8 | 18.4073 | 18.5716 | 434 | 0.000 | 1410.0 | 4890.5 | 4.5859 |
| goodreads | clause | c0_genre | 32 | torch | 100 | 16 | 36.6383 | 36.8101 | 437 | 0.000 | 1410.0 | 9778.9 | 9.0217 |
| goodreads | clause | c0_genre | 32 | torch | 500 | 1 | 2.5643 | 2.7157 | 389 | 0.000 | 1410.0 | 611.2 | 0.7383 |
| goodreads | clause | c0_genre | 32 | torch | 500 | 8 | 18.4259 | 18.5645 | 434 | 0.000 | 1410.0 | 4890.5 | 4.6013 |
| goodreads | clause | c0_genre | 32 | torch | 500 | 16 | 36.6560 | 36.8062 | 436 | 0.000 | 1410.0 | 9778.9 | 9.0374 |
| goodreads | clause | c0_genre | 32 | torch | 1000 | 1 | 2.5665 | 2.6471 | 389 | 0.000 | 1410.0 | 611.2 | 0.7391 |
| goodreads | clause | c0_genre | 32 | torch | 1000 | 8 | 18.4273 | 18.6040 | 434 | 0.000 | 1410.0 | 4890.5 | 4.6026 |
| goodreads | clause | c0_genre | 32 | torch | 1000 | 16 | 36.6558 | 36.8189 | 436 | 0.000 | 1410.0 | 9778.9 | 9.0405 |
| goodreads | clause | c0_genre | 32 | triton | 100 | 1 | 0.6482 | 0.8408 | 1476 | 0.003 | 1410.0 | 9.7 | 0.2202 |
| goodreads | clause | c0_genre | 32 | triton | 100 | 8 | 0.6547 | 0.8263 | 12008 | 0.005 | 1410.0 | 75.0 | 0.5576 |
| goodreads | clause | c0_genre | 32 | triton | 100 | 16 | 1.1191 | 1.2262 | 14224 | 0.000 | 1410.0 | 149.7 | 1.0368 |
| goodreads | clause | c0_genre | 32 | triton | 500 | 1 | 0.6502 | 0.8438 | 1473 | 0.007 | 1410.0 | 9.7 | 0.2356 |
| goodreads | clause | c0_genre | 32 | triton | 500 | 8 | 0.6611 | 0.8258 | 11919 | 0.001 | 1410.0 | 75.0 | 0.5770 |
| goodreads | clause | c0_genre | 32 | triton | 500 | 16 | 1.1367 | 1.2669 | 14002 | 0.000 | 1410.0 | 149.8 | 1.0528 |
| goodreads | clause | c0_genre | 32 | triton | 1000 | 1 | 0.6401 | 0.8334 | 1521 | 0.003 | 1410.0 | 9.7 | 0.2364 |
| goodreads | clause | c0_genre | 32 | triton | 1000 | 8 | 0.6621 | 0.8130 | 11961 | 0.001 | 1410.0 | 75.1 | 0.5785 |
| goodreads | clause | c0_genre | 32 | triton | 1000 | 16 | 1.1390 | 1.2943 | 13964 | 0.000 | 1410.0 | 149.9 | 1.0552 |
| goodreads | none | full_scan | — | official | 100 | 1 | 0.8967 | 1.0924 | 1101 | 0.013 | 1410.0 | 13.6 | not_capturable |
| goodreads | none | full_scan | — | official | 100 | 8 | 0.9043 | 1.1963 | 8846 | 0.013 | 1410.0 | 107.3 | not_capturable |
| goodreads | none | full_scan | — | official | 100 | 16 | 1.3249 | 1.5020 | 11995 | 0.001 | 1410.0 | 214.6 | not_capturable |
| goodreads | none | full_scan | — | torch | 100 | 1 | 1.2745 | 1.3740 | 781 | 0.000 | 1410.0 | 383.7 | 0.5663 |
| goodreads | none | full_scan | — | torch | 100 | 8 | 8.4903 | 8.5966 | 942 | 0.000 | 1410.0 | 3066.1 | 3.3201 |
| goodreads | none | full_scan | — | torch | 100 | 16 | 16.9176 | 17.0669 | 945 | 0.000 | 1410.0 | 6131.0 | 6.6391 |
| goodreads | none | full_scan | — | triton | 100 | 1 | 0.5338 | 0.6040 | 1833 | 0.008 | 1410.0 | 7.3 | 0.2011 |
| goodreads | none | full_scan | — | triton | 100 | 8 | 0.6669 | 0.7664 | 12265 | 0.003 | 1410.0 | 56.3 | 0.4630 |
| goodreads | none | full_scan | — | triton | 100 | 16 | 0.8570 | 0.9861 | 18539 | 0.000 | 1410.0 | 112.3 | 0.8939 |

#### End to end — quality and parity

| dataset | filter | sweep | n_probe | backend | status | recall@100 (oracle) | jaccard_vs_first@100 | score_max_abs_diff | parity | build_s | index_mib | pass_rate | bloom_fp_rate | unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | bloom | c0_maincat | 24 | official | ok | 0.883735 | 0.984950 | 0.095 | vs_triton | 2.1 | 482.3 | 0.1363 | 0.000000 | True |
| arxiv | bloom | c0_maincat | 24 | torch | ok | 0.884044 | 1.000000 | 0.0000 | vs_triton | 2.1 | 786.1 | 0.1363 | 0.000000 | False |
| arxiv | bloom | c0_maincat | 24 | triton | ok | 0.884044 | — | — | reference | 2.2 | 786.1 | 0.1363 | 0.000000 | True |
| arxiv | bloom | c0_maincat | 32 | official | ok | 0.903010 | 0.984797 | 0.098 | vs_triton | 2.1 | 482.3 | 0.1363 | 0.000000 | True |
| arxiv | bloom | c0_maincat | 32 | torch | ok | 0.903351 | 1.000000 | 0.0000 | vs_triton | 2.1 | 786.1 | 0.1363 | 0.000000 | True |
| arxiv | bloom | c0_maincat | 32 | triton | ok | 0.903351 | — | — | reference | 2.2 | 786.1 | 0.1363 | 0.000000 | True |
| arxiv | clause | c0_maincat | 24 | official | ok | 0.883757 | 0.985033 | 0.000 | vs_triton | 2.2 | 775.9 | 0.1363 | — | False |
| arxiv | clause | c0_maincat | 24 | torch | ok | 0.884044 | 1.000000 | 0.0000 | vs_triton | 2.2 | 786.1 | 0.1363 | — | False |
| arxiv | clause | c0_maincat | 24 | triton | ok | 0.884044 | — | — | reference | 2.3 | 786.1 | 0.1363 | — | True |
| arxiv | clause | c0_maincat | 32 | official | ok | 0.903036 | 0.984882 | 0.000 | vs_triton | 2.2 | 775.9 | 0.1363 | — | False |
| arxiv | clause | c0_maincat | 32 | torch | ok | 0.903351 | 1.000000 | 0.0000 | vs_triton | 2.2 | 786.1 | 0.1363 | — | False |
| arxiv | clause | c0_maincat | 32 | triton | ok | 0.903351 | — | — | reference | 2.3 | 786.1 | 0.1363 | — | True |
| arxiv | none | full_scan | — | official | partial | — | 0.983778 | 0.000 | vs_triton | 2.4 | 411.0 | 1.0000 | — | False |
| arxiv | none | full_scan | — | torch | partial | — | 1.000000 | 0.0000 | vs_triton | 2.5 | 421.3 | 1.0000 | — | True |
| arxiv | none | full_scan | — | triton | partial | — | — | — | reference | 2.4 | 421.3 | 1.0000 | — | False |
| goodreads | bloom | c0_genre | 24 | official | ok | 0.912796 | 0.999849 | 0.006 | vs_triton | 0.6 | 143.3 | 0.3307 | 0.000000 | True |
| goodreads | bloom | c0_genre | 24 | torch | ok | 0.912800 | 1.000000 | 0.0000 | vs_triton | 0.6 | 394.2 | 0.3307 | 0.000000 | False |
| goodreads | bloom | c0_genre | 24 | triton | ok | 0.912800 | — | — | reference | 0.6 | 394.2 | 0.3307 | 0.000000 | True |
| goodreads | bloom | c0_genre | 32 | official | ok | 0.936910 | 0.999805 | 0.006 | vs_triton | 0.6 | 143.3 | 0.3307 | 0.000000 | True |
| goodreads | bloom | c0_genre | 32 | torch | ok | 0.936908 | 1.000000 | 0.0000 | vs_triton | 0.6 | 394.2 | 0.3307 | 0.000000 | False |
| goodreads | bloom | c0_genre | 32 | triton | ok | 0.936908 | — | — | reference | 0.6 | 394.2 | 0.3307 | 0.000000 | True |
| goodreads | clause | c0_genre | 24 | official | ok | 0.912796 | 0.999849 | 0.006 | vs_triton | 0.8 | 207.3 | 0.3307 | — | True |
| goodreads | clause | c0_genre | 24 | torch | ok | 0.912800 | 1.000000 | 0.0000 | vs_triton | 0.9 | 394.2 | 0.3307 | — | False |
| goodreads | clause | c0_genre | 24 | triton | ok | 0.912800 | — | — | reference | 0.9 | 394.2 | 0.3307 | — | False |
| goodreads | clause | c0_genre | 32 | official | ok | 0.936910 | 0.999805 | 0.006 | vs_triton | 0.8 | 207.3 | 0.3307 | — | True |
| goodreads | clause | c0_genre | 32 | torch | ok | 0.936908 | 1.000000 | 0.0000 | vs_triton | 0.9 | 394.2 | 0.3307 | — | False |
| goodreads | clause | c0_genre | 32 | triton | ok | 0.936908 | — | — | reference | 0.9 | 394.2 | 0.3307 | — | False |
| goodreads | none | full_scan | — | official | partial | — | 0.999885 | 0.004 | vs_triton | 1.1 | 110.0 | 1.0000 | — | False |
| goodreads | none | full_scan | — | torch | partial | — | 1.000000 | 0.0000 | vs_triton | 0.9 | 296.9 | 1.0000 | — | False |
| goodreads | none | full_scan | — | triton | partial | — | — | — | reference | 1.1 | 296.9 | 1.0000 | — | False |

#### Kernel-only — Algorithm 1 phases 2+3, phase 1 hoisted out

| dataset | mode | bs | arm | wall median ms | spread | sm_mhz | device µs total | scorer µs | mask µs | prep µs | topk µs | launches | peak MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| arxiv | none | 1 | triton | 0.3807 | 0.017 | 1410.0 | 128.5 | 11.0 | — | 27.6 | 83.9 | 39 | 0.7 |
| arxiv | none | 1 | official-fp16 | 0.9389 | 0.001 | 1155.0 | 277.9 | 15.6 | — | 134.3 | 102.3 | 77 | 3.8 |
| arxiv | none | 1 | official-int32 | 0.9148 | 0.004 | 1170.0 | 271.6 | 15.2 | — | 130.5 | 100.2 | 75 | 4.1 |
| arxiv | none | 16 | triton | 0.4977 | 0.005 | 1410.0 | 301.6 | 129.8 | — | 39.6 | 123.7 | 38 | 10.8 |
| arxiv | none | 16 | official-fp16 | 0.9362 | 0.006 | 1410.0 | 529.4 | 113.3 | — | 268.2 | 119.6 | 75 | 60.9 |
| arxiv | none | 16 | official-int32 | 0.9235 | 0.022 | 1410.0 | 542.7 | 118.8 | — | 281.0 | 119.3 | 73 | 65.5 |
| arxiv | bloom | 1 | triton | 0.8690 | 0.000 | 1155.0 | 222.7 | 28.7 | — | 80.1 | 102.0 | 59 | 0.7 |
| arxiv | bloom | 1 | official-fp16 | 1.4013 | 0.010 | 1155.0 | 330.6 | 15.6 | 10.7 | 161.5 | 101.1 | 95 | 3.8 |
| arxiv | bloom | 1 | official-int32 | 1.4671 | 0.035 | 1155.0 | 330.3 | 14.7 | 10.7 | 162.1 | 101.6 | 94 | 4.1 |
| arxiv | bloom | 16 | triton | 0.8953 | 0.009 | 1410.0 | 451.6 | 228.6 | — | 86.0 | 124.0 | 58 | 10.8 |
| arxiv | bloom | 16 | official-fp16 | 1.6064 | 0.001 | 1395.0 | 542.7 | 76.8 | 10.4 | 294.1 | 119.6 | 93 | 61.0 |
| arxiv | bloom | 16 | official-int32 | 1.5852 | 0.013 | 1410.0 | 550.6 | 74.7 | 11.4 | 304.4 | 120.5 | 91 | 65.6 |
| arxiv | exact | 1 | triton | 0.5068 | 0.016 | 1275.0 | 147.4 | 28.7 | — | 25.1 | 91.0 | 35 | 0.7 |
| arxiv | exact | 1 | official-fp16 | 1.2499 | 0.003 | 1410.0 | 282.0 | 11.0 | — | 141.6 | 81.9 | 80 | 51.7 |
| arxiv | exact | 1 | official-int32 | 1.2475 | 0.007 | 1410.0 | 280.7 | 10.3 | — | 141.6 | 81.4 | 79 | 51.7 |
| arxiv | exact | 16 | triton | 0.4770 | 0.003 | 1410.0 | 422.8 | 260.8 | — | 33.0 | 124.8 | 34 | 10.8 |
| arxiv | exact | 16 | official-fp16 | 7.2117 | 0.001 | 1410.0 | 815.1 | 81.7 | — | 270.6 | 119.2 | 77 | 826.7 |
| arxiv | exact | 16 | official-int32 | 7.1936 | 0.001 | 1410.0 | 501.2 | 73.6 | — | 279.5 | 119.4 | 75 | 826.7 |
| goodreads | none | 1 | triton | 0.3800 | 0.004 | 1410.0 | 190.8 | 24.7 | — | 28.0 | 132.4 | 39 | 2.8 |
| goodreads | none | 1 | official-fp16 | 0.9391 | 0.004 | 1410.0 | 284.8 | 8.9 | — | 119.1 | 132.7 | 77 | 13.6 |
| goodreads | none | 1 | official-int32 | 0.8326 | 0.139 | 1410.0 | 283.5 | 9.2 | — | 119.5 | 130.4 | 75 | 15.6 |
| goodreads | none | 16 | triton | 0.7517 | 0.000 | 1410.0 | 721.2 | 337.0 | — | 41.2 | 334.8 | 38 | 37.7 |
| goodreads | none | 16 | official-fp16 | 1.3074 | 0.006 | 1410.0 | 1146.3 | 30.9 | — | 709.5 | 338.0 | 75 | 214.6 |
| goodreads | none | 16 | official-int32 | 1.3750 | 0.003 | 1410.0 | 1202.6 | 33.1 | — | 771.2 | 334.5 | 73 | 233.3 |
| goodreads | bloom | 1 | triton | 0.8460 | 0.007 | 1410.0 | 248.9 | 40.8 | — | 65.4 | 133.0 | 59 | 2.9 |
| goodreads | bloom | 1 | official-fp16 | 1.2014 | 0.139 | 1230.0 | 374.4 | 9.3 | 9.5 | 157.9 | 152.1 | 95 | 13.7 |
| goodreads | bloom | 1 | official-int32 | 1.3680 | 0.008 | 1185.0 | 389.4 | 9.3 | 10.2 | 165.7 | 159.2 | 94 | 15.7 |
| goodreads | bloom | 16 | triton | 1.0069 | 0.000 | 1410.0 | 960.2 | 520.8 | — | 87.5 | 338.9 | 58 | 37.7 |
| goodreads | bloom | 16 | official-fp16 | 2.1563 | 0.070 | 1410.0 | 1180.7 | 29.2 | 9.1 | 730.3 | 331.0 | 93 | 214.7 |
| goodreads | bloom | 16 | official-int32 | 2.2258 | 0.003 | 1410.0 | 1243.8 | 24.5 | 9.2 | 795.6 | 333.6 | 91 | 233.3 |
| goodreads | exact | 1 | triton | 0.4946 | 0.002 | 1410.0 | 192.1 | 33.5 | — | 23.9 | 132.2 | 36 | 3.5 |
| goodreads | exact | 1 | official-fp16 | 1.2138 | 0.006 | 1410.0 | 305.1 | 8.0 | — | 131.1 | 132.1 | 81 | 15.2 |
| goodreads | exact | 1 | official-int32 | 1.1973 | 0.003 | 1410.0 | 307.4 | 8.6 | — | 132.4 | 132.8 | 80 | 16.8 |
| goodreads | exact | 16 | triton | 0.8177 | 0.000 | 1410.0 | 778.2 | 397.6 | — | 37.0 | 339.4 | 35 | 37.7 |
| goodreads | exact | 16 | official-fp16 | 3.0496 | 0.001 | 1410.0 | 1343.4 | 26.8 | — | 834.4 | 323.6 | 79 | 228.3 |
| goodreads | exact | 16 | official-int32 | 3.1124 | 0.001 | 1410.0 | 1411.5 | 27.8 | — | 895.9 | 329.3 | 77 | 247.0 |

#### Kernel-only — parity of each official arm against Triton

| dataset | mode | bs | score path | jaccard@100 | score_max_abs_diff | n queries |
|---|---|---|---|---|---|---|
| arxiv | none | 16 | int32 | 1.000000 | 0.000e+00 | 512 |
| arxiv | none | 16 | fp16 | 0.982928 | 4.814e-04 | 512 |
| arxiv | bloom | 16 | int32 | 1.000000 | 0.000e+00 | 512 |
| arxiv | bloom | 16 | fp16 | 0.985788 | 4.814e-04 | 512 |
| arxiv | exact | 16 | int32 | 1.000000 | 0.000e+00 | 512 |
| arxiv | exact | 16 | fp16 | 0.985788 | 4.814e-04 | 512 |
| goodreads | none | 16 | int32 | 1.000000 | 0.000e+00 | 512 |
| goodreads | none | 16 | fp16 | 1.000000 | 2.885e-03 | 512 |
| goodreads | bloom | 16 | int32 | 0.999961 | 0.000e+00 | 512 |
| goodreads | bloom | 16 | fp16 | 0.999961 | 2.916e-03 | 512 |
| goodreads | exact | 16 | int32 | 0.999961 | 0.000e+00 | 512 |
| goodreads | exact | 16 | fp16 | 0.999961 | 2.916e-03 | 512 |

#### Why the fp16 path costs rank agreement — top-k boundary gaps

| dataset | mode | gap p10 | gap median | score@k median | gap/score median | fp16 ulp | rows with gap < ulp |
|---|---|---|---|---|---|---|---|
| arxiv | none | 1.384e-05 | 7.918e-05 | 0.8363 | 9.336e-05 | 4.083e-04 | 0.9531 |
| arxiv | bloom | 1.408e-05 | 8.565e-05 | 0.8303 | 1.021e-04 | 4.054e-04 | 0.9453 |
| arxiv | exact | 1.408e-05 | 8.565e-05 | 0.8303 | 1.021e-04 | 4.054e-04 | 0.9453 |
| goodreads | none | 9.461e-04 | 6.967e-03 | 0.5623 | 1.200e-02 | 2.746e-04 | 0.0332 |
| goodreads | bloom | 1.103e-03 | 6.854e-03 | 0.6320 | 1.119e-02 | 3.086e-04 | 0.0312 |
| goodreads | exact | 1.103e-03 | 6.854e-03 | 0.6320 | 1.119e-02 | 3.086e-04 | 0.0312 |

#### Phase 2 alone (bs=16)

| dataset | op | arm | median ms | p99 ms | spread | timer |
|---|---|---|---|---|---|---|
| arxiv | ours_query_bits (bloom hash, [B,W]) | ours | 0.3780 | 0.4580 | 0.015 | cuda_events |
| arxiv | ours_bloom_match (row-wise, full N) | ours | 1.4809 | 1.4898 | 0.000 | cuda_events |
| arxiv | official_bloom_index_search_batch (full N, packed) | official | 0.2443 | 0.2814 | 0.003 | cuda_events |
| arxiv | official_return_partial_response (probed only) | official | 0.4513 | 0.5039 | 0.016 | cuda_events |
| arxiv | official_queries_to_expressions (host, incl. D2H) | official | 0.0273 | 0.0363 | — | perf_counter (host-side op) |
| arxiv | official_parse_expression_query_batch (host) | official | 0.0468 | 0.0559 | — | perf_counter (host-side op) |
| goodreads | ours_query_bits (bloom hash, [B,W]) | ours | 0.3687 | 0.4191 | 0.004 | cuda_events |
| goodreads | ours_bloom_match (row-wise, full N) | ours | 0.4517 | 0.5610 | 0.060 | cuda_events |
| goodreads | official_bloom_index_search_batch (full N, packed) | official | 0.2279 | 0.2588 | 0.003 | cuda_events |
| goodreads | official_return_partial_response (probed only) | official | 0.4488 | 0.4971 | 0.029 | cuda_events |
| goodreads | official_queries_to_expressions (host, incl. D2H) | official | 0.0272 | 0.0376 | — | perf_counter (host-side op) |
| goodreads | official_parse_expression_query_batch (host) | official | 0.0468 | 0.0594 | — | perf_counter (host-side op) |

#### Bloom selectivity and memory at the shipped settings

| dataset | exact pass rate | ours FP rate | official FP rate | ours MiB | official MiB | ours bits/doc | official b_multiplier |
|---|---|---|---|---|---|---|---|
| arxiv | 0.1357 | 0.000000 | 0.000000 | 364.9 | 71.3 | 1024 | 10.0 |
| goodreads | 0.3323 | 0.000000 | 0.000000 | 97.3 | 33.3 | 1024 | 10.0 |
