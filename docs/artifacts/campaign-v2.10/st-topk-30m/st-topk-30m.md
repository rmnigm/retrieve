| sweep | bs | mode | v2.10 ms | ST-TOPK ms | ST-TOPK / v2.10 | ids A = B | flag | Meta fp16 ms | ST-TOPK / Meta fp16 | Meta int32 ms | ST-TOPK / Meta int32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| c0_domain | 1 | eager | 0.428 | 0.429 | **1.002 (0.974-1.029)** | True | **> 1.00** | 1.062 | 0.405 (0.397-0.412) | 1.066 | 0.404 (0.396-0.412) |
| c0_domain | 1 | graph | 0.314 | 0.313 | **0.999 (0.997-1.002)** | True |  | - | - | - | - |
| c0_domain | 16 | eager | 1.003 | 0.834 | **0.831 (0.825-0.837)** | True |  | 1.496 | 0.559 (0.549-0.569) | 1.518 | 0.551 (0.544-0.559) |
| c0_domain | 16 | graph | 0.977 | 0.771 | **0.789 (0.788-0.790)** | True |  | - | - | - | - |
| c0_domain | 64 | eager | 3.005 | 2.349 | **0.782 (0.778-0.786)** | True |  | 2.212 | 1.064 (1.051-1.078) | 2.594 | 0.907 (0.898-0.917) |
| c0_domain | 64 | graph | 2.959 | 2.254 | **0.762 (0.761-0.763)** | True |  | - | - | - | - |
| tags4 | 1 | eager | 0.498 | 0.500 | **1.003 (0.998-1.008)** | True | **> 1.00** | 1.383 | 0.361 (0.361-0.362) | 1.391 | 0.359 (0.358-0.361) |
| tags4 | 1 | graph | 0.317 | 0.318 | **1.001 (0.999-1.003)** | True | **> 1.00** | - | - | - | - |
| tags4 | 16 | eager | 1.075 | 0.907 | **0.843 (0.837-0.850)** | True |  | 1.508 | 0.603 (0.590-0.616) | 1.528 | 0.595 (0.582-0.609) |
| tags4 | 16 | graph | 1.049 | 0.844 | **0.805 (0.804-0.806)** | True |  | - | - | - | - |
| tags4 | 64 | eager | 3.320 | 2.658 | **0.801 (0.797-0.805)** | True |  | 2.117 | 1.256 (1.245-1.268) | 2.491 | 1.068 (1.060-1.076) |
| tags4 | 64 | graph | 3.275 | 2.565 | **0.783 (0.783-0.783)** | True |  | - | - | - | - |

cells above 1.00 of v2.10: 3

score dumps (first 64 kept queries per sweep; bs 1 on 8):
('c0_domain', 1) ids equal True scores equal True max |d| nan
('c0_domain', 16) ids equal True scores equal True max |d| nan
('c0_domain', 64) ids equal True scores equal True max |d| nan
('tags4', 1) ids equal True scores equal True max |d| 0.0
('tags4', 16) ids equal True scores equal True max |d| nan
('tags4', 64) ids equal True scores equal True max |d| nan

