| dataset | tree | mode | bs | arm | width | wall ms | scorer µs | mask µs | topk µs | prep µs | device µs | sm_mhz | unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| goodreads | padded | none | 1 | triton | 611520 | 0.3871 | 25.1 | 0.0 | 139.8 | 33.8 | 205.2 | 1410.0 | False |
| goodreads | padded | none | 1 | official-fp16 | 611520 | 0.9228 | 9.6 | 0.0 | 139.0 | 133.3 | 306.7 | 1410.0 | False |
| goodreads | padded | none | 1 | official-int32 | 611520 | 0.9095 | 9.4 | 0.0 | 138.1 | 132.0 | 304.3 | 1410.0 | False |
| goodreads | padded | none | 16 | triton | 611520 | 0.7550 | 338.0 | 0.0 | 339.2 | 42.0 | 727.8 | 1410.0 | False |
| goodreads | padded | none | 16 | official-fp16 | 611520 | 1.3191 | 32.1 | 0.0 | 326.2 | 713.6 | 1139.4 | 1410.0 | False |
| goodreads | padded | none | 16 | official-int32 | 611520 | 1.3843 | 30.6 | 0.0 | 335.6 | 779.3 | 1207.6 | 1410.0 | False |
| goodreads | padded | bloom | 1 | triton | 611520 | 0.8037 | 41.4 | 0.0 | 137.0 | 77.2 | 266.7 | 1410.0 | False |
| goodreads | padded | bloom | 1 | official-fp16 | 611520 | 1.3595 | 8.0 | 9.7 | 154.7 | 169.0 | 382.9 | 1275.0 | False |
| goodreads | padded | bloom | 1 | official-int32 | 611520 | 1.3402 | 9.6 | 9.8 | 154.2 | 174.4 | 388.8 | 1275.0 | False |
| goodreads | padded | bloom | 16 | triton | 611520 | 1.0173 | 524.5 | 0.0 | 340.3 | 92.7 | 970.7 | 1410.0 | False |
| goodreads | padded | bloom | 16 | official-fp16 | 611520 | 2.1371 | 28.6 | 9.3 | 333.6 | 743.3 | 1192.5 | 1410.0 | False |
| goodreads | padded | bloom | 16 | official-int32 | 611520 | 2.1983 | 28.1 | 9.2 | 335.8 | 806.0 | 1259.0 | 1410.0 | False |
| goodreads | padded | exact | 1 | triton | 611520 | 0.4825 | 31.9 | 0.0 | 138.7 | 29.2 | 202.3 | 1410.0 | False |
| goodreads | padded | exact | 1 | official-fp16 | 611520 | 1.1840 | 9.2 | 0.0 | 138.5 | 146.5 | 327.3 | 1410.0 | False |
| goodreads | padded | exact | 1 | official-int32 | 611520 | 1.2185 | 9.2 | 0.0 | 139.0 | 147.5 | 329.1 | 1410.0 | False |
| goodreads | padded | exact | 16 | triton | 611520 | 0.8218 | 400.0 | 0.0 | 342.1 | 37.2 | 783.2 | 1410.0 | False |
| goodreads | padded | exact | 16 | official-fp16 | 611520 | 3.0576 | 28.4 | 0.0 | 333.7 | 844.2 | 1364.4 | 1410.0 | False |
| goodreads | padded | exact | 16 | official-int32 | 611520 | 3.1218 | 27.3 | 0.0 | 335.2 | 906.8 | 1427.2 | 1410.0 | False |
| goodreads | compact | none | 1 | triton | 64757 | 0.3422 | 6.8 | 0.0 | 74.6 | 18.5 | 112.9 | 1380.0 | False |
| goodreads | compact | none | 1 | official-fp16 | 64757 | 0.8524 | 10.0 | 0.0 | 78.6 | 129.8 | 240.1 | 1275.0 | False |
| goodreads | compact | none | 1 | official-int32 | 64757 | 0.8341 | 9.5 | 0.0 | 79.6 | 126.0 | 236.8 | 1275.0 | False |
| goodreads | compact | none | 16 | triton | 64757 | 0.4216 | 26.2 | 0.0 | 92.0 | 17.5 | 150.2 | 1410.0 | False |
| goodreads | compact | none | 16 | official-fp16 | 64757 | 0.8676 | 32.2 | 0.0 | 91.2 | 172.4 | 319.1 | 1410.0 | False |
| goodreads | compact | none | 16 | official-int32 | 64757 | 0.8314 | 31.8 | 0.0 | 91.8 | 172.9 | 315.4 | 1410.0 | False |
| goodreads | compact | bloom | 1 | triton | 64757 | 0.6014 | 10.1 | 0.0 | 79.5 | 48.5 | 151.9 | 1275.0 | False |
| goodreads | compact | bloom | 1 | official-fp16 | 64757 | 1.2918 | 8.1 | 8.5 | 80.4 | 155.7 | 289.1 | 1275.0 | False |
| goodreads | compact | bloom | 1 | official-int32 | 64757 | 1.2710 | 8.4 | 8.4 | 80.5 | 156.3 | 290.2 | 1275.0 | False |
| goodreads | compact | bloom | 16 | triton | 64757 | 0.6205 | 34.3 | 0.0 | 99.4 | 49.9 | 198.9 | 1305.0 | False |
| goodreads | compact | bloom | 16 | official-fp16 | 64757 | 1.4653 | 27.6 | 10.5 | 102.4 | 218.8 | 398.1 | 1275.0 | False |
| goodreads | compact | bloom | 16 | official-int32 | 64757 | 1.4657 | 29.5 | 10.5 | 102.0 | 218.0 | 398.9 | 1275.0 | False |
| goodreads | compact | exact | 1 | triton | 64757 | 0.4328 | 13.9 | 0.0 | 79.0 | 14.5 | 116.9 | 1275.0 | False |
| goodreads | compact | exact | 1 | official-fp16 | 64757 | 1.1881 | 8.9 | 0.0 | 78.5 | 145.4 | 265.0 | 1275.0 | False |
| goodreads | compact | exact | 1 | official-int32 | 64757 | 1.1028 | 7.9 | 0.0 | 73.1 | 135.0 | 245.9 | 1365.0 | False |
| goodreads | compact | exact | 16 | triton | 64757 | 0.4357 | 53.1 | 0.0 | 93.0 | 12.5 | 168.3 | 1410.0 | False |
| goodreads | compact | exact | 16 | official-fp16 | 64757 | 2.5187 | 24.7 | 0.0 | 90.9 | 299.4 | 529.4 | 1410.0 | False |
| goodreads | compact | exact | 16 | official-int32 | 64757 | 2.5023 | 24.9 | 0.0 | 91.6 | 299.7 | 530.8 | 1410.0 | False |
| arxiv | padded | none | 1 | triton | 171648 | 0.3520 | 11.8 | 0.0 | 90.2 | 33.7 | 142.4 | 1410.0 | False |
| arxiv | padded | none | 1 | official-fp16 | 171648 | 0.8569 | 15.1 | 0.0 | 96.8 | 136.7 | 271.3 | 1275.0 | False |
| arxiv | padded | none | 1 | official-int32 | 171648 | 0.8959 | 15.6 | 0.0 | 94.6 | 132.0 | 264.4 | 1290.0 | True |
| arxiv | padded | none | 16 | triton | 171648 | 0.4497 | 130.9 | 0.0 | 125.9 | 42.1 | 307.4 | 1410.0 | False |
| arxiv | padded | none | 16 | official-fp16 | 171648 | 0.8758 | 110.9 | 0.0 | 121.3 | 277.1 | 537.0 | 1410.0 | False |
| arxiv | padded | none | 16 | official-int32 | 171648 | 0.8524 | 115.8 | 0.0 | 122.2 | 287.1 | 547.7 | 1410.0 | False |
| arxiv | padded | bloom | 1 | triton | 171648 | 0.7676 | 27.0 | 0.0 | 98.8 | 87.3 | 225.0 | 1275.0 | False |
| arxiv | padded | bloom | 1 | official-fp16 | 171648 | 1.2771 | 14.1 | 9.7 | 99.6 | 165.1 | 325.5 | 1275.0 | False |
| arxiv | padded | bloom | 1 | official-int32 | 171648 | 1.2723 | 12.8 | 9.8 | 98.8 | 164.5 | 323.4 | 1275.0 | False |
| arxiv | padded | bloom | 16 | triton | 171648 | 0.7776 | 237.7 | 0.0 | 127.2 | 91.8 | 469.6 | 1410.0 | False |
| arxiv | padded | bloom | 16 | official-fp16 | 171648 | 1.4533 | 81.8 | 10.3 | 121.5 | 304.7 | 558.7 | 1410.0 | False |
| arxiv | padded | bloom | 16 | official-int32 | 171648 | 1.4569 | 83.0 | 10.6 | 121.3 | 315.5 | 569.1 | 1410.0 | False |
| arxiv | padded | exact | 1 | triton | 171648 | 0.4510 | 25.3 | 0.0 | 90.1 | 26.4 | 144.2 | 1410.0 | False |
| arxiv | padded | exact | 1 | official-fp16 | 171648 | 1.1924 | 10.3 | 0.0 | 87.7 | 156.3 | 301.1 | 1410.0 | False |
| arxiv | padded | exact | 1 | official-int32 | 171648 | 1.1701 | 15.8 | 0.0 | 89.9 | 152.3 | 304.9 | 1410.0 | False |
| arxiv | padded | exact | 16 | triton | 171648 | 0.4762 | 268.2 | 0.0 | 125.4 | 36.6 | 433.9 | 1410.0 | False |
| arxiv | padded | exact | 16 | official-fp16 | 171648 | 7.1511 | 80.8 | 0.0 | 121.1 | 732.0 | 1278.7 | 1410.0 | False |
| arxiv | padded | exact | 16 | official-int32 | 171648 | 7.1473 | 82.8 | 0.0 | 121.1 | 290.3 | 522.8 | 1410.0 | False |
| arxiv | compact | none | 1 | triton | 128027 | 0.3421 | 11.6 | 0.0 | 85.6 | 18.0 | 128.0 | 1410.0 | True |
| arxiv | compact | none | 1 | official-fp16 | 128027 | 0.8612 | 15.5 | 0.0 | 95.0 | 135.8 | 268.5 | 1275.0 | False |
| arxiv | compact | none | 1 | official-int32 | 128027 | 0.8316 | 14.4 | 0.0 | 94.2 | 130.9 | 261.7 | 1275.0 | True |
| arxiv | compact | none | 16 | triton | 128027 | 0.4569 | 99.1 | 0.0 | 114.3 | 17.6 | 246.2 | 1410.0 | True |
| arxiv | compact | none | 16 | official-fp16 | 128027 | 0.9262 | 112.6 | 0.0 | 110.8 | 236.2 | 488.5 | 1410.0 | False |
| arxiv | compact | none | 16 | official-int32 | 128027 | 0.8487 | 111.0 | 0.0 | 111.1 | 244.8 | 490.8 | 1410.0 | False |
| arxiv | compact | bloom | 1 | triton | 128027 | 0.6080 | 15.6 | 0.0 | 97.0 | 49.3 | 176.0 | 1275.0 | False |
| arxiv | compact | bloom | 1 | official-fp16 | 128027 | 1.2899 | 13.7 | 9.4 | 91.9 | 164.1 | 316.4 | 1275.0 | False |
| arxiv | compact | bloom | 1 | official-int32 | 128027 | 1.2723 | 15.6 | 9.7 | 93.7 | 163.2 | 318.8 | 1275.0 | False |
| arxiv | compact | bloom | 16 | triton | 128027 | 0.6222 | 105.0 | 0.0 | 114.1 | 46.4 | 280.5 | 1410.0 | False |
| arxiv | compact | bloom | 16 | official-fp16 | 128027 | 1.4699 | 73.7 | 10.9 | 109.7 | 260.8 | 496.0 | 1410.0 | False |
| arxiv | compact | bloom | 16 | official-int32 | 128027 | 1.4519 | 69.0 | 10.5 | 109.1 | 269.0 | 497.2 | 1410.0 | False |
| arxiv | compact | exact | 1 | triton | 128027 | 0.4325 | 21.5 | 0.0 | 89.3 | 10.8 | 130.6 | 1380.0 | False |
| arxiv | compact | exact | 1 | official-fp16 | 128027 | 1.2037 | 11.4 | 0.0 | 87.5 | 152.7 | 298.3 | 1410.0 | False |
| arxiv | compact | exact | 1 | official-int32 | 128027 | 1.1870 | 11.8 | 0.0 | 87.0 | 150.7 | 296.6 | 1410.0 | False |
| arxiv | compact | exact | 16 | triton | 128027 | 0.4477 | 197.3 | 0.0 | 114.9 | 10.5 | 333.0 | 1410.0 | False |
| arxiv | compact | exact | 16 | official-fp16 | 128027 | 7.1645 | 79.7 | 0.0 | 112.6 | 690.4 | 1229.1 | 1410.0 | False |
| arxiv | compact | exact | 16 | official-int32 | 128027 | 7.1525 | 81.6 | 0.0 | 112.8 | 248.0 | 471.5 | 1410.0 | False |

| dataset | tree | mode | ours scorer+mask µs (bs16) | official-fp16 scorer+mask µs | ours / official |
|---|---|---|---|---|---|
| goodreads | padded | none | 338.0 | 32.1 | 10.52 |
| goodreads | padded | bloom | 524.5 | 38.0 | 13.81 |
| goodreads | padded | exact | 400.0 | 28.4 | 14.09 |
| goodreads | padded | parity none | jaccard 1.000000 | dmax 0.00e+00 | |
| goodreads | padded | parity bloom | jaccard 1.000000 | dmax 0.00e+00 | |
| goodreads | padded | parity exact | jaccard 1.000000 | dmax 0.00e+00 | |
| goodreads | compact | none | 26.2 | 32.2 | 0.82 |
| goodreads | compact | bloom | 34.3 | 38.1 | 0.90 |
| goodreads | compact | exact | 53.1 | 24.7 | 2.15 |
| goodreads | compact | parity none | jaccard 1.000000 | dmax 0.00e+00 | |
| goodreads | compact | parity bloom | jaccard 1.000000 | dmax 0.00e+00 | |
| goodreads | compact | parity exact | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | padded | none | 130.9 | 110.9 | 1.18 |
| arxiv | padded | bloom | 237.7 | 92.1 | 2.58 |
| arxiv | padded | exact | 268.2 | 80.8 | 3.32 |
| arxiv | padded | parity none | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | padded | parity bloom | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | padded | parity exact | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | compact | none | 99.1 | 112.6 | 0.88 |
| arxiv | compact | bloom | 105.0 | 84.5 | 1.24 |
| arxiv | compact | exact | 197.3 | 79.7 | 2.47 |
| arxiv | compact | parity none | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | compact | parity bloom | jaccard 1.000000 | dmax 0.00e+00 | |
| arxiv | compact | parity exact | jaccard 1.000000 | dmax 0.00e+00 | |
