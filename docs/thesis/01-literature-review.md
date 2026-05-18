# Глава 1. Обзор литературы

> Первая итерация: структура подразделов + аннотированная библиография со связью с
> модулями пакета [`retrieve`](../../retrieve/src/retrieve/). Не итоговая прозаическая
> глава — это скелет, который будет развёрнут в текст на следующем этапе.

## Введение к главе

Цель главы — позиционировать работу относительно (а) промышленных систем model-based
retrieval (LiNR, SilverTorch, Merlin), (б) классических ANN-алгоритмов и их GPU-реализаций,
(в) подходов к фильтрации, квантизации и обучаемым similarity, а также (г) инструментов
(Triton, PyTorch) и датасетов, используемых в эмпирической части. Все ссылки сгруппированы
по подразделам в порядке от широкого контекста к узким техническим деталям. В конце главы
даётся сводная аннотированная библиография и формулируется зазор, который закрывает
разрабатываемый пакет.

---

## 1.1. Задача retrieval в современных рекомендательных системах

**Содержание раздела.** Сначала описывается стандартная двухстадийная архитектура промышленных
рекомендательных систем (retrieval → ranking) и накладываемые ограничения по latency, throughput
и масштабу каталога. Затем разбирается, почему two-tower модель с дот-произведением стала
доминирующим подходом к embedding-based retrieval, и в каких сценариях её выразительной
способности перестаёт хватать (что мотивирует MoL/OverArch в Главе 1.5).

**Ключевые ссылки:**
- Naumov et al. 2019 — *Deep Learning Recommendation Model (DLRM)* — каноническая
  индустриальная RecSys-архитектура.
- Covington, Adams, Sargin 2016 — *Deep Neural Networks for YouTube Recommendations* —
  образец многостадийной retrieval/ranking-схемы.
- Huang et al. 2020 — *Embedding-Based Retrieval in Facebook Search* — классический пример
  EBR в продакшене.
- Yi et al. 2019 — *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item
  Recommendations* — двухбашенный подход для рекомендаций.
- Liu et al. 2021 — *Que2Search* — query/document understanding для поиска FB.
- Mudigere et al. 2022 — *Software-Hardware Co-Design for Fast Training of DLRMs*.

---

## 1.2. Sequential модели для генерации user embeddings

**Содержание раздела.** В evaluation-харнессе пакета query-embeddings для Yambda и Goodreads
генерируются обученной GSASRec-моделью (см. [`evaluation/training/model.py`](../../evaluation/training/model.py)
и [`evaluation/training/train_sasrec.py`](../../evaluation/training/train_sasrec.py)). Раздел
последовательно проходит от RNN-подходов к трансформерным sequential моделям и описывает
проблему overconfidence, которую решает gBCE-лосс в gSASRec.

**Ключевые ссылки:**
- Hidasi et al. 2015 — *Session-Based Recommendations with RNNs (GRU4Rec)*.
- Kang & McAuley 2018 — *Self-Attentive Sequential Recommendation (SASRec)* — основа архитектуры
  query-encoder в пакете.
- Sun et al. 2019 — *BERT4Rec* — двунаправленный энкодер, конкурент SASRec.
- Petrov & Macdonald 2023 — *gSASRec: Reducing Overconfidence in Sequential Recommendation
  Trained with Negative Sampling* (RecSys'23 Best Paper) — фактическая модель trainer-а в
  пакете.
- Hochreiter & Schmidhuber 1997 — *LSTM* — исторический фон.

---

## 1.3. Классические ANN-методы (CPU)

**Содержание раздела.** Здесь систематизируются «model-free» индексы, с которыми сравнивается
model-based retrieval: квантизационные (PQ, IVF/PQ), графовые (HNSW, DiskANN) и гибридные
(IVF+PQ). Особое внимание уделяется HNSW, поскольку он используется как CPU-baseline в
[`evaluation/retrieval/algos/voyager.py`](../../evaluation/retrieval/algos/voyager.py).

**Ключевые ссылки:**
- Jégou, Douze, Schmid 2011 — *Product Quantization for Nearest Neighbor Search* — основа
  IVF/PQ.
- Malkov & Yashunin 2020 — *Efficient and Robust ANN Search Using HNSW Graphs* — графовый ANN,
  baseline пакета.
- Guo et al. 2020 — *Accelerating Large-Scale Inference with Anisotropic Vector Quantization
  (ScaNN)*.
- Subramanya et al. 2019 — *DiskANN: Fast Accurate Billion-Point NN Search on a Single Node*.
- Douze et al. 2024 — *The Faiss Library* (arXiv:2401.08281) — современный technical report.
- Brin & Page 1998 — *The Anatomy of a Large-Scale Hypertextual Web Search Engine* —
  исторический контекст inverted-index подходов.

---

## 1.4. GPU-реализации ANN

**Содержание раздела.** Разбираются существующие GPU-ANN библиотеки, их ограничения по top-k,
n_probe и кастомизируемости фильтров. Эти ограничения и являются мотивацией для построения
собственных Triton-ядер в пакете (см. [`retrieve/src/retrieve/kernels/triton/silvertorch/`](../../retrieve/src/retrieve/kernels/triton/silvertorch/)).

**Ключевые ссылки:**
- Johnson, Douze, Jégou 2019/2021 — *Billion-Scale Similarity Search with GPUs (FAISS-GPU)*.
- Ootomo et al. 2024 — *CAGRA: Highly Parallel Graph Construction and ANN Search for GPUs*.
- Wang et al. 2021 — *Milvus: A Purpose-Built Vector Data Management System*.
- Zhao, Tan, Li 2020 — *SONG: Approximate Nearest Neighbor Search on GPU*.
- Nolet 2023 — *Reusable Computational Patterns for ML and IR with RAPIDS RAFT* (NVIDIA blog).
- Zhao, Tan, Li 2022 — *Constrained Approximate Similarity Search on Proximity Graph* —
  filtered HNSW, релевантно нашему filter benchmark.
- Spotify Engineering 2023 — *Introducing Voyager: Spotify's New Nearest-Neighbor Search
  Library* — фактический CPU-baseline в [`voyager.py`](../../evaluation/retrieval/algos/voyager.py).
- Pinterest 2023 — *Manas HNSW Realtime: Powering Realtime Embedding-Based Retrieval* (блог).
- *Faiss on the GPU — Limitations* (FAISS Wiki) и *Milvus GPU Index Limitations* (docs) —
  фиксация ограничений top-k и probe, мотивирующих собственный Triton-стек.

---

## 1.5. Model-based retrieval (центральный prior art)

**Содержание раздела.** Главный раздел обзора. Анализируются обе базовые статьи — LiNR
и SilverTorch — с акцентом на (а) их предложенные алгоритмы, (б) границы применимости,
(в) различие в подходе к фильтрации (декуплированная в LiNR vs. inline co-design в
SilverTorch). Дополнительно разбираются параллельные направления: NVIDIA Merlin как
GPU-стек, MoL и HSTU как развитие dot-product similarity, и generative retrieval как
альтернативная парадигма.

**Ключевые ссылки (прямые предшественники):**
- Borisyuk et al. 2024 — *LiNR: Model Based Neural Retrieval on GPUs at LinkedIn* (CIKM'24) —
  оригинал LiNR-подмножества (V1 similarity masking, V2 prefilter, V3 1-bit OPORP). Файлы пакета:
  [`similarity_masking.py`](../../retrieve/src/retrieve/layers/linr/similarity_masking.py),
  [`prefilter_knn.py`](../../retrieve/src/retrieve/layers/linr/prefilter_knn.py),
  [`one_bit_knn.py`](../../retrieve/src/retrieve/layers/linr/one_bit_knn.py).
- SilverTorch (Meta) — статья из [`articles/silvertorch.md`](../../articles/silvertorch.md);
  первоисточник co-designed IVF+INT8+Bloom, реимплементированного в
  [`main.py`](../../retrieve/src/retrieve/layers/silvertorch/main.py) +
  [`codesigned_probe_score.py`](../../retrieve/src/retrieve/kernels/triton/silvertorch/codesigned_probe_score.py).
- Oldridge et al. 2020 — *Merlin: A GPU-Accelerated Recommendation Framework* (NVIDIA) — broader
  GPU-стек для RecSys.

**Ключевые ссылки (learned similarities, родственные направления):**
- Zhai et al. 2023 — *Revisiting Neural Retrieval on Accelerators* (KDD'23) — Mixture-of-Logits
  и h-indexer; обоснование «learned similarities» поверх dot product.
- Zhai et al. 2024 — *Actions Speak Louder than Words* (arXiv:2402.17152) — HSTU, generative
  recommenders.
- Ding & Zhai 2025 — *Retrieval with Learned Similarities* (WWW'25) — обобщающий взгляд на MoL.
- Zhang et al. 2025 — *Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for
  Item-to-Item Retrieval* (KDD'25 V.2) — контекст multi-task retrieval из SilverTorch.

**Ключевые ссылки (generative / semantic retrieval как альтернатива):**
- Tay et al. 2022 — *Transformer Memory as a Differentiable Search Index (DSI)* (NeurIPS).
- Rajput et al. 2023 — *Recommender Systems with Generative Retrieval (TIGER)* (NeurIPS).
- Zhang et al. 2023 — *Model-Enhanced Vector Index (MEVI)*.

---

## 1.6. Feature filtering и attribute matching

**Содержание раздела.** Здесь обосновывается проблема «liquidity» в post-filter ANN
(низкая selectivity → деградация recall), сравниваются inverted-index, forward-index
и bloom-based подходы. Раздел напрямую мотивирует
[`ExactAttributeFilter`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py) (clause
matching по словарям атрибутов) и
[`BloomFilter`](../../retrieve/src/retrieve/layers/filters/bloom.py) (signature-based)
в пакете, а также композиционные хелперы `combine_masks` / `combine_indices`.

**Ключевые ссылки:**
- Curtiss et al. 2013 — *Unicorn: A System for Searching the Social Graph* (Facebook, VLDB) —
  inverted-index с posting-list-композицией.
- Chen, Lassance, Lin 2023 — *End-to-End Retrieval with Learned Dense and Sparse Representations
  Using Lucene* (arXiv:2311.18503).
- Goodwin et al. 2017 — *BitFunnel: Revisiting Signatures for Search* (SIGIR) — идейная база
  bloom-index в SilverTorch.
- Zhao, Tan, Li 2022 — *Constrained Approximate Similarity Search on Proximity Graph* — filtered
  HNSW, демонстрирующий деградацию recall при post-filter.
- Cambazoglu & Baeza-Yates 2016 — *Scalability and Efficiency Challenges in Large-Scale Web
  Search Engines* — обзорный фон для inverted-index систем.
- Bloom 1970 — *Space/Time Trade-offs in Hash Coding with Allowable Errors* — оригинальный
  bloom filter.

---

## 1.7. Квантизация и компактные представления embedding

**Содержание раздела.** Теоретические основы под две используемые в пакете квантизационные
схемы: INT8 для SilverTorch ANN ([`quantize_int8`](../../retrieve/src/retrieve/layers/utils/quantize.py))
и 1-bit Sign-OPORP для LiNR V3 ([`quantize_oporp_1bit`](../../retrieve/src/retrieve/layers/utils/quantize.py)).
Обсуждается trade-off память/качество и совместимость с pre-filter подходом.

**Ключевые ссылки:**
- Jégou et al. 2011 — *Product Quantization* (см. также 1.3) — теоретический фон для всех
  квантизационных схем.
- Li & Li 2023 — *OPORP: One Permutation + One Random Projection* (arXiv:2302.03505) — основа
  V3 LiNR (Sign-OPORP), оценивающего косинусную близость через popcount(xor).
- NVIDIA — *dp4a instruction* (CUDA documentation) — аппаратная основа Int8 ANN-кёрнела
  SilverTorch.
- Zhao et al. 2023 — *Embedding in Recommender Systems: A Survey* (arXiv:2310.18608) — общий
  обзор embedding-методов.

---

## 1.8. GPU-программирование для retrieval-моделей

**Содержание раздела.** Инструменты, на которых построен пакет. Ключевой аргумент — оригинальные
LiNR и SilverTorch используют CUDA C++ и закрытые компоненты, тогда как пакет демонстрирует, что
открытый Triton-DSL достаточно выразителен для воспроизведения обоих подходов и допускает
торнадо-стиля autotune (см. [`docs/system/kernels.md`](../system/kernels.md)).

**Ключевые ссылки:**
- Tillet, Kung, Cox 2019 — *Triton: An Intermediate Language and Compiler for Tiled Neural
  Network Computations* (MAPL'19) — обязательная ссылка, обосновывает выбор DSL.
- Paszke et al. 2019 — *PyTorch: An Imperative Style, High-Performance Deep Learning Library*
  (NeurIPS'19, arXiv:1912.01703).
- Ivchenko et al. 2022 — *TorchRec: A PyTorch Domain Library for Recommendation Systems*
  (RecSys'22).
- Abadi et al. 2016 — *TensorFlow: A System for Large-Scale Machine Learning* (OSDI'16) — для
  сопоставления стеков.
- Frostig, Johnson, Leary 2019 — *Compiling Machine Learning Programs via High-Level Tracing
  (JAX)* (SysML'18) — альтернативный стек.
- Moritz et al. 2018 — *Ray* (OSDI'18) — distributed compute, фоновая ссылка.

**Технические анонсы и документация:**
- OpenAI 2021 — *Introducing Triton: Open-Source GPU Programming for Neural Networks* (блог-пост).

---

## 1.9. Live update в model-based индексах

**Содержание раздела.** Контекст для будущей работы. На данный момент пакет реализует только
offline-индексирование; раздел перечисляет работы, описывающие near-realtime ingestion в
model-based retrieval, и формулирует, какие именно компоненты потребовалось бы добавить.

**Ключевые ссылки:**
- Liu et al. 2022 — *Monolith: Real-Time Recommendation System with Collisionless Embedding
  Table* (arXiv:2209.07663).
- Lian et al. 2022 — *PERSIA: Scaling Deep Learning-Based Recommenders up to 100 Trillion
  Parameters* (KDD'22).
- Jiang et al. 2019 — *XDL: An Industrial DL Framework for High-Dimensional Sparse Data*
  (KDD'19).
- GV 2008 — *Open Sourcing Venice — LinkedIn's Derived Data Platform* (LinkedIn engineering
  blog) — CDC-стрим для LiNR.

---

## 1.10. Датасеты и протокол оценки

**Содержание раздела.** Обоснование выбора трёх датасетов в evaluation-харнессе
([`evaluation/conf/`](../../evaluation/conf/)): Yambda (unfiltered retrieval, 500M/5B
listening events), Goodreads (filter benchmark по жанрам/категориям) и ArXiv
(text retrieval по nomic-embed). Описываются протоколы Global Temporal Split в Yambda и
filter sweep matrix в Goodreads/ArXiv (см. [`docs/system/filtering.md`](../system/filtering.md)).

**Ключевые ссылки:**
- *Yambda: Yandex Music Billion-Interactions Dataset* (статья из
  [`articles/yambda.md`](../../articles/yambda.md), 2025) — основной retrieval-benchmark
  пакета.
- Wan & McAuley 2018 — *Item Recommendation on Monotonic Behavior Chains* (RecSys'18) —
  оригинальная публикация Goodreads dataset, основа filter benchmark по жанрам.
- Wan, Misra, Nakashole, McAuley 2019 — *Fine-Grained Spoiler Detection from Large-Scale
  Review Corpora* (ACL'19) — расширение Goodreads с детальными аннотациями.
- Nussbaum, Morris, Duderstadt, Mulyar 2024 — *Nomic Embed: Training a Reproducible Long
  Context Text Embedder* (arXiv:2402.01613) — модель, генерирующая ArXiv item/query embeddings
  в pipeline.
- Harper & Konstan 2015 — *The MovieLens Datasets* (TIIS) — сравнительный фон.
- Ni, Li, McAuley 2019 — *Justifying Recommendations Using Distantly-Labeled Reviews*
  (EMNLP-IJCNLP'19) — Amazon Reviews 2018, сравнительный фон.
- Akiba et al. 2019 — *Optuna: A Next-Generation Hyperparameter Optimization Framework*
  (KDD'19) — HPO в Yambda-обзоре и для обучения SASRec.

---

## 1.11. Позиционирование данной работы

**Содержание раздела.** Финальный сводный параграф, формализующий зазор:

- LiNR полностью закрыт исходным кодом (LinkedIn), SilverTorch — частично; обе работы
  используют CUDA C++ и не предоставляют воспроизводимого open-source стека.
- Ни одна из открытых GPU-ANN-библиотек (FAISS-GPU, CAGRA, Milvus, RAFT) не поддерживает
  inline attribute filtering, co-designed с ANN-поиском.
- Triton-реимплементация впервые делает обе модели воспроизводимыми и сравнимыми на едином
  стеке через флаг `backend="torch" | "triton"`.
- Модульный filter API
  ([`FilterModule`](../../retrieve/src/retrieve/interfaces.py),
  [`ExactAttributeFilter`](../../retrieve/src/retrieve/layers/filters/exact_attribute.py),
  [`BloomFilter`](../../retrieve/src/retrieve/layers/filters/bloom.py),
  [`combine_masks`/`combine_indices`](../../retrieve/src/retrieve/layers/filters/__init__.py))
  обобщает оба подхода под общий контракт.
- Эмпирическая оценка на трёх независимых датасетах (Yambda, Goodreads, ArXiv) демонстрирует
  поведение моделей вне индустриальных условий, в которых они изначально замерялись.

---

## Сводная аннотированная библиография

Дедуплицированный список со всеми ссылками, сгруппированными по тематике. Пометки в колонке
«Источник»: **[L]** — есть в библиографии LiNR; **[S]** — есть в библиографии SilverTorch;
**[Y]** — есть в библиографии Yambda; **[+]** — добавлено сверх трёх базовых статей.

### A. Model-based retrieval & learned similarities

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Borisyuk et al. — *LiNR: Model Based Neural Retrieval on GPUs at LinkedIn* | CIKM | 2024 | **[S]** Реимплементируется в [`layers/linr/`](../../retrieve/src/retrieve/layers/linr/). |
| SilverTorch | [`articles/silvertorch.md`](../../articles/silvertorch.md) | — | Реимплементируется в [`layers/silvertorch/`](../../retrieve/src/retrieve/layers/silvertorch/). |
| Oldridge et al. — *Merlin* | Proc. of IRS | 2020 | **[S]** NVIDIA-стек, broader контекст. |
| Zhai et al. — *Revisiting Neural Retrieval on Accelerators (MoL)* | KDD | 2023 | **[L][S]** Learned similarities. |
| Zhai et al. — *Actions Speak Louder than Words (HSTU)* | arXiv:2402.17152 | 2024 | **[S]** Generative recommenders. |
| Ding & Zhai — *Retrieval with Learned Similarities* | WWW | 2025 | **[S]** Обобщение MoL. |
| Zhang et al. — *Multi-Task Multi-Head Item-to-Item Retrieval* | KDD V.2 | 2025 | **[S]** Multi-task retrieval. |
| Tay et al. — *Transformer Memory as a DSI* | NeurIPS | 2022 | **[L]** Generative-retrieval. |
| Rajput et al. — *TIGER: Recommender Systems with Generative Retrieval* | NeurIPS | 2023 | **[L][Y]** Generative-retrieval. |
| Zhang et al. — *Model-Enhanced Vector Index (MEVI)* | — | 2023 | **[L]** Гибрид модели и индекса. |

### B. Классический и GPU-ориентированный ANN

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Jégou, Douze, Schmid — *Product Quantization* | TPAMI | 2011 | **[L]** Основа IVF/INT8. |
| Malkov & Yashunin — *HNSW* | TPAMI | 2020 | **[L]** Графовый ANN. |
| Johnson, Douze, Jégou — *FAISS-GPU* | IEEE TBD | 2019/2021 | **[L][S]** Главный GPU-baseline. |
| Douze et al. — *The Faiss Library* | arXiv:2401.08281 | 2024 | **[S]** Обзор. |
| Guo et al. — *ScaNN / Anisotropic VQ* | ICML | 2020 | **[L]** Альтернативная квантизация. |
| Subramanya et al. — *DiskANN* | NeurIPS | 2019 | **[S]** Disk-resident billion-scale ANN. |
| Ootomo et al. — *CAGRA* | ICDE / arXiv:2308.15136 | 2024 | **[L][S]** GPU graph ANN. |
| Wang et al. — *Milvus* | SIGMOD | 2021 | **[S]** Vector database. |
| Zhao, Tan, Li — *SONG* | ICDE | 2020 | **[L]** ANN на GPU. |
| Nolet — *RAPIDS RAFT* | NVIDIA dev blog | 2023 | **[L]** GPU primitives. |
| Spotify — *Voyager: Spotify's NN Search Library* | Spotify Engineering blog + GitHub | 2023 | **[+]** CPU-baseline в [`voyager.py`](../../evaluation/retrieval/algos/voyager.py). |
| Pinterest — *Manas HNSW Realtime* | Pinterest blog | 2023 | **[S]** Production HNSW. |
| Pace et al. — *Lance* | arXiv:2504.15247 | 2025 | **[S]** Columnar storage с random access. |

### C. Filtering и attribute matching

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Curtiss et al. — *Unicorn* | VLDB | 2013 | **[L]** Inverted-index в социальном графе. |
| Chen, Lassance, Lin — *End-to-End Retrieval with Lucene* | arXiv:2311.18503 | 2023 | **[L]** Lucene. |
| Goodwin et al. — *BitFunnel* | SIGIR | 2017 | **[S]** Signature-based search → bloom-index. |
| Zhao, Tan, Li — *Constrained Approximate Similarity Search on Proximity Graph* | arXiv:2210.14958 | 2022 | **[L]** Filtered HNSW. |
| Cambazoglu & Baeza-Yates — *Scalability and Efficiency Challenges in WebSearch* | SIGIR | 2016 | **[S]** Обзор inverted-index. |
| Brin & Page — *Anatomy of a Hypertextual Search Engine* | Computer Networks | 1998 | **[S]** Исторический фон. |
| Bloom — *Space/Time Trade-offs in Hash Coding* | CACM | 1970 | **[+]** Оригинальный bloom filter. |

### D. Квантизация и компактные представления

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Jégou et al. — *Product Quantization* (см. B) | TPAMI | 2011 | **[L]** Теоретический фон. |
| Li & Li — *OPORP* | arXiv:2302.03505 | 2023 | **[L]** Основа V3 (Sign-OPORP). |
| NVIDIA — *dp4a instruction (CUDA docs)* | NVIDIA docs | — | **[+]** Hardware-основа INT8 ANN. |
| Zhao et al. — *Embedding in RecSys: A Survey* | arXiv:2310.18608 | 2023 | **[S]** Общий обзор. |

### E. GPU-инфраструктура и DL frameworks

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Tillet, Kung, Cox — *Triton: Intermediate Language and Compiler...* | MAPL | 2019 | **[+]** DSL пакета. |
| Paszke et al. — *PyTorch* | NeurIPS / arXiv:1912.01703 | 2019 | **[S]** Фреймворк. |
| Ivchenko et al. — *TorchRec* | RecSys | 2022 | **[S]** PyTorch для RecSys. |
| Abadi et al. — *TensorFlow* | OSDI | 2016 | **[S]** Сопоставление стеков. |
| Frostig, Johnson, Leary — *JAX (Compiling ML via High-Level Tracing)* | SysML | 2019 | **[S]** Альтернативный стек. |
| Moritz et al. — *Ray* | OSDI | 2018 | **[S]** Distributed compute. |
| OpenAI — *Introducing Triton* | OpenAI blog | 2021 | **[+]** Анонс Triton. |
| FAISS Wiki — *Faiss on the GPU — Limitations* | github.com | 2023 | **[S]** Лимиты top-k. |
| Milvus — *GPU Index Limitations* | milvus.io docs | 2023 | **[S]** Лимиты top-k. |

### F. Sequential модели рекомендаций

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Kang & McAuley — *SASRec* | ICDM | 2018 | **[Y]** Baseline в [`evaluation/training/`](../../evaluation/training/). |
| Sun et al. — *BERT4Rec* | CIKM | 2019 | **[Y]** Сопоставление с SASRec. |
| Hidasi et al. — *GRU4Rec* | arXiv:1511.06939 | 2015 | **[Y]** RNN-предшественник. |
| Petrov & Macdonald — *gSASRec* | RecSys (Best Paper) / arXiv:2308.07192 | 2023 | **[+]** Фактическая модель trainer-а. |
| Hochreiter & Schmidhuber — *LSTM* | Neural Computation | 1997 | **[Y]** Исторический фон. |

### G. RecSys foundation и two-tower

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Naumov et al. — *DLRM* | arXiv:1906.00091 | 2019 | **[S]** Industrial RecSys-архитектура. |
| Covington, Adams, Sargin — *YouTube DNN* | RecSys | 2016 | **[S]** Многостадийный retrieval. |
| Huang et al. — *EBR in Facebook Search* | KDD | 2020 | **[L][S]** EBR в продакшене. |
| Yi et al. — *Sampling-Bias-Corrected Two-Tower* | RecSys | 2019 | **[S]** Two-tower retrieval. |
| Liu et al. — *Que2Search* | KDD | 2021 | **[L]** Query understanding. |
| Rangadurai et al. — *NxtPost* | KDD | 2022 | **[L]** User-to-post FB Groups. |
| Pal et al. — *PinnerSage* | KDD | 2020 | **[L]** Multi-modal user embeddings. |
| Borisyuk et al. — *LiGNN* | KDD | 2024 | **[L]** GNN-фон LinkedIn-стека. |
| Shen et al. — *Learning to Retrieve for Job Matching* | arXiv:2402.13435 | 2024 | **[L]** Применение LiNR. |
| Wang et al. — *Billion-scale Commodity Embedding (Alibaba)* | KDD | 2018 | **[S]** Industrial embeddings. |
| Zhai, Wu, Tzeng et al. — *Unified Embedding at Pinterest* | KDD | 2019 | **[S]** Pinterest two-tower. |
| Baltescu et al. — *ItemSage* | KDD | 2022 | **[S]** Pinterest two-tower. |
| Zhou et al. — *Deep Interest Network (DIN)* | KDD | 2018 | **[S]** Attention в ranking. |
| Mudigere et al. — *SW/HW Co-Design for DLRM* | ISCA | 2022 | **[S]** Hardware-aware DLRM. |
| Ardalani et al. — *Scaling Laws for Recommendation Models* | arXiv:2208.08489 | 2022 | **[S][Y]** Scaling-laws. |
| Zhang et al. — *Wukong* | arXiv:2403.02545 | 2024 | **[S][Y]** Scaling-law for RecSys. |

### H. Live update / streaming

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Liu et al. — *Monolith* | arXiv:2209.07663 | 2022 | **[L]** Real-time RecSys. |
| Lian et al. — *PERSIA* | KDD | 2022 | **[L]** 100T-параметрный RecSys. |
| Jiang et al. — *XDL* | KDD | 2019 | **[L]** Industrial DL framework. |
| GV — *Open Sourcing Venice (LinkedIn)* | LinkedIn blog | 2008 | **[L]** CDC stream для LiNR. |

### I. Датасеты и evaluation tools

| Ссылка | Венью | Год | Источник / релевантность |
|--------|-------|-----|--------------------------|
| Yandex — *Yambda Dataset* | [`articles/yambda.md`](../../articles/yambda.md) | 2025 | **[Y]** Unfiltered retrieval benchmark. |
| Wan & McAuley — *Item Recommendation on Monotonic Behavior Chains* (Goodreads) | RecSys | 2018 | **[+]** Goodreads dataset paper. |
| Wan, Misra, Nakashole, McAuley — *Fine-Grained Spoiler Detection* | ACL | 2019 | **[+]** Расширение Goodreads. |
| Nussbaum, Morris, Duderstadt, Mulyar — *Nomic Embed* | arXiv:2402.01613 | 2024 | **[+]** Embedding-модель для ArXiv eval. |
| Harper & Konstan — *MovieLens* | TIIS | 2015 | **[Y]** Сравнительный фон. |
| Ni, Li, McAuley — *Amazon Reviews 2018* | EMNLP-IJCNLP | 2019 | **[Y]** Сравнительный фон. |
| Akiba et al. — *Optuna* | KDD | 2019 | **[Y]** HPO в обучении SASRec. |
| Hou et al. — *Bridging Language and Items (Amazon Reviews 2023)* | arXiv:2403.03952 | 2024 | **[Y]** Современный large-scale dataset. |
| Schedl — *LFM-1b* | ICMR | 2016 | **[Y]** Music-RecSys фон. |
| Schedl et al. — *LFM-2b* | CHIIR | 2022 | **[Y]** Music-RecSys фон. |
| Kaplan et al. — *Scaling Laws for Neural Language Models* | arXiv:2001.08361 | 2020 | **[Y]** Scaling-laws background. |
| Hoffmann et al. — *Training Compute-Optimal LLMs (Chinchilla)* | arXiv:2203.15556 | 2022 | **[Y]** Scaling-laws background. |

### J. Прочие технические ссылки

| Ссылка | Тип | Год | Источник / релевантность |
|--------|-----|-----|--------------------------|
| Lewis et al. — *Retrieval-Augmented Generation (RAG)* | NeurIPS | 2020 | **[S]** Vector search в LLM. |
| Arthur & Vassilvitskii — *k-means++* | Technical Report (Stanford) | 2006 | **[S]** Init для IVF clustering. |
| Vantage — *AWS p4d.24xlarge / r6i.8xlarge / x2idn.24xlarge cost pages* | vantage.sh | 2025 | **[S]** Для cost-efficiency сравнения. |

---

## TODO для следующих итераций

- Уточнить с научным руководителем стиль цитирования (ГОСТ vs. ACM/IEEE numeric).
- Собрать полный `bibliography.bib` с DOI / arXiv-id / URL для всех приведённых ссылок.
- Развернуть подразделы 1.1–1.11 в прозаический текст (~3–5 страниц на подраздел в финальной версии).
- Решить, какие ссылки из категории J (cost benchmarks, GitHub Wiki) оставлять как первоклассные
  цитаты, а какие — выносить в сноски.
- Возможно, выделить отдельный sub-section по Triton-кёрнел-методологии (autotune, block-tiling)
  при описании реализации, если глубины Tillet et al. 2019 не хватит.
