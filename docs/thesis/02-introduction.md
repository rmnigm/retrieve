# Введение (Introduction) — Reference Notes

**Status:** English reference notes for the downstream Russian-prose writing
agent. Not finished prose. Every claim is anchored to either a `path:line`
code reference, a `[DATA: ...]` pointer into the results, or a flagged
`[CITE: ... — KEEP/DROPPED]` candidate.

**Length target for finished Russian prose:** 3–5 pages
(00-thesis-plan.md §95–§102).

**Hard constraint:** the LinR-lineage statement at §"LinR-lineage statement"
below MUST appear in the finished prose. The Citation Policy mandates it
both here in Введение AND in §1.5 of the literature review
(00-thesis-plan.md §85, §102, §118).

---

## Voice / scope reminder for writer

- Audience: HSE MSc thesis committee; Russian academic prose;
  third-person passive ("работа реализует", "продемонстрировано",
  "построен бенчмарк").
- Body register: software-engineering — reproducibility, measurement,
  kernel-level detail. The thesis is not a novel-algorithm contribution;
  it is a reproducibility-and-extension contribution.
- The internal code symbol `SilverTorch` MUST NOT appear in body text;
  use neutral phrasing: "ко-дизайнерный IVF+INT8+Bloom-ретривер",
  "ко-дизайнерный ретривер", "IVF+INT8-модуль" (00-thesis-plan.md §82).

---

## Актуальность (Relevance)

- Two converging trends define why the work is timely:
  1. **The shift from heuristic ANN indices to model-based GPU
     retrieval.** Classical pipelines paired a heuristic index (HNSW,
     IVF-PQ, ScaNN-style trees) running on CPU with a separate scoring
     model on GPU. Model-based GPU retrieval collapses the stack: the
     index access pattern is co-designed with the model that produced
     the embeddings, exploits tensor cores, fuses kernels (probe + dot
     + filter test), and adopts aggressive quantization (1-bit / INT8
     surrogates of the inner product). The LinR research line is the
     canonical published instance of this paradigm
     [CITE: Borisyuk et al. 2024, CIKM — LinkedIn — KEEP].
  2. **The open-source gap.** Most reported numbers for model-based
     GPU retrieval are internal to large platforms; published results
     are reproducible in principle but not via a drop-in tool. There
     is no widely-used pure-PyTorch+Triton implementation of LinR
     primitives, and no open implementation of an IVF+INT8+Bloom
     co-designed retriever for filtered top-K.
- **Why now.** Triton 3.x and the `torch.library.custom_op` /
  `triton_op` decorators make it possible to ship Triton kernels as
  `torch.export`-clean modules. This removes the historical hard
  dependency on hand-written CUDA C++ for shipping GPU retrieval code
  outside large vendors. Stage 2 of `docs/plans/00-roadmap.md` has
  already replaced the legacy `@dynamo.disable` decorators with these
  `triton_op`-based wrappers, demonstrating that the contract is
  achievable in practice for the kernels at issue.
- **Why GPU retrieval at this catalogue scale.** At $N \in [10^6, 10^7]$
  with $d \in [128, 256]$, a single-GPU fp16 dense matmul over the
  full item table already achieves sub-millisecond median latency for
  $K = 100$, $B = 1$ on A100. Approximate kernels (1-bit OPORP,
  IVF+INT8+Bloom) extend the regime by one to two orders of magnitude
  in latency, with measured quality losses in the 0.5–2.2% range on
  the quality-suite cells [DATA: docs/thesis/07-results.md §6.2].
- **Citation candidates for the relevance framing:**
  - [CITE: Borisyuk et al. 2024, CIKM (LinR) — LinkedIn — KEEP]
  - [CITE: TIGER — Rajput et al. 2023 — Google — KEEP] (generative
    retrieval as an alternative paradigm; positions LinR within the
    landscape)
  - [CITE: SCANN — Guo et al. 2020 — Google — KEEP] (learned
    similarity / anisotropic quantization line)
  - [CITE: CAGRA — Ootomo et al. 2024 — NVIDIA — KEEP] (GPU-native
    ANN state of the art; vendor-neutral substitute for FAISS-GPU
    references)
  - [CITE: FAISS / FAISS-GPU — Johnson, Douze, Jégou — **DROPPED**,
    Meta (Douze and Jégou at FAIR)]
  - [CITE: DLRM — Naumov et al. 2019 — **DROPPED**, Meta]
  - [CITE: HSTU / "Actions Speak Louder than Words" — Zhai et al. 2024
    — **DROPPED**, Meta]

---

## Цель (Goal)

**Suggested single-sentence formulation (Russian):**
«Воспроизвести семейство алгоритмов LinR и ко-дизайнерный
IVF+INT8+Bloom-ретривер как чистый PyTorch+Triton open-source пакет,
а также провести репродуцируемую оценку качества, латентности и
памяти на трёх открытых датасетах».

- **Writer note.** Keep the two halves explicit: "reproduce
  (воспроизвести)" + "evaluate (оценить)". The thesis defends the
  reproducibility AND the empirical characterisation, not the
  underlying algorithms.

---

## Задачи (Tasks — five core + two optional)

Seven candidate bullets; the writer should select 5–7. List ordered
to match the achievement-of-задач mirror in Заключение §"Achievement
of задач".

1. **Реализовать Triton-ядра для LinR V1/V2/V3/V4 и для
   ко-дизайнерного IVF+INT8+Bloom-ретривера**, удовлетворяющие
   контракту `torch.export`-чистоты (без `.item()`, `.cpu()`,
   `Optional[Tensor]` в публичном API).
2. **Собрать единый бенчмарк-харнесс** с Hydra-style конфигами и
   едиными протоколами Quality / Filter / Deep sweep; оракулы top-K
   валидируются на стабильность размера и содержимого.
3. **Обучить SASRec query-кодировщики** на Goodreads и Yambda
   (двух масштабов — 500m и 5b событий) с gBCE-потерей; для arXiv
   использовать предзаписанные эмбеддинги nomic-embed-v1.5 как корпус
   с точным ground-truth (по построению, Recall@100 = 1.0).
4. **Прогнать quality-, filter- и deep-sweep на трёх открытых
   датасетах** (Goodreads ≈ 797k items, arXiv 2.99M, Yambda 500m
   3.06M / Yambda 5b 5.37M); агрегировать результаты в long-format
   CSV для воспроизводимого построения графиков и таблиц.
5. **Провести абляции**: размер кандидатского пула для LinR V3
   ($\text{candidate\_pool} \in \{2k, 4k, 8k, 16k, 32k\}$), число
   IVF-кластеров и параметра $n_{\text{probe}}$ для ко-дизайнерного
   ретривера, бит-бюджет Bloom-фильтра, масштабирование по
   $d \in \{64, 128, 256\}$.
6. **Опубликовать пакет** под open-source лицензией
   (`pip install torchretrieve`); опубликовать конфиги, оракулы и
   сырые JSON-результаты для полной воспроизводимости.
7. *(Optional 7th)* Сравнить полученные числа с опубликованным
   диапазоном для LinR-V3 и зафиксировать зоны соответствия и
   расхождения; в частности, проверить совпадение
   paper-strict OPORP-варианта (Sign-OPORP с bin-sum и L2-нормализацией)
   с эталоном при $k_{\text{bits}} = D$ (byte-identical, см. commit
   `4f9b1a6`).

---

## Объект и предмет (HSE convention)

- **Объект:** системы model-based GPU-retrieval (с поддержкой
  фильтрации и без) для плотных эмбеддингов в задачах top-K по
  скалярному произведению.
- **Предмет:** алгоритмы семейства LinR (V1 post-filter masking, V2
  pre-filter reduction, V3 1-bit OPORP, V4 INT8 dense) и
  ко-дизайнерный IVF+INT8+Bloom-ретривер; их реализация в виде
  Triton-ядер, методология оценки и эмпирические свойства
  (качество, латентность, потребление памяти) на открытых датасетах.

---

## Методы (Methods)

- **Экспериментальное ML и GPU-программирование.** Реализация
  Triton-ядер для перечисленных алгоритмов, обучение SASRec-моделей
  с gBCE-потерей, репродуцируемая оценка с фиксированными сидами,
  валидируемыми оракулами и long-format агрегацией результатов в
  `docs/thesis/results-data/`.
- **Литературный обзор.** Алгоритмы LinR, классические примитивы
  (IVF — [CITE: Jégou et al. 2011 — INRIA — KEEP]; BitFunnel-style
  Bloom-сигнатуры — [CITE: Goodwin et al. 2017 — Microsoft — KEEP];
  INT8-квантование), GPU-нативные ANN-библиотеки, заменяющие
  FAISS-семейство (см. политику цитирования в §1.0 / §1.11).
- **Анализ Pareto-фронтов** «качество × латентность × память» по
  типам нагрузки (filter-on / filter-off), batch-размерам
  $B \in \{1, 8, 16\}$ и значениям $K \in \{100, 200, 400\}$.

---

## Эмпирическая база (Empirical base)

- **Goodreads:** ~797k книг, 227.5M прочтений, 313 178 тестовых
  пользователей; SASRec обучен в трёх размерностях $d \in \{64, 128,
  256\}$ с gBCE-потерей. [CITE: Goodreads — Wan & McAuley 2018 —
  UCSD — KEEP]
- **arXiv:** 2.99M статей, эмбеддинги предзаписаны
  nomic-embed-v1.5 (SASRec не используется); 10 000 тестовых
  запросов; ceiling Recall@100 = 1.0 по построению (ground truth —
  exact top-K над тем же индексом). [CITE: nomic-embed-v1.5 —
  Nussbaum et al. 2024 — Nomic AI — KEEP]
- **Yambda (Yandex Music 2025):** два масштаба обучения —
  500m (3.06M треков, 466.5M событий, $d \in \{64, 128, 256\}$) и
  5b (5.37M треков, 4.65B событий, $d \in \{64, 128\}$). Yambda-5b-d128
  даёт абсолютный максимум Recall@100 в исследовании (0.1779,
  `linr_v1`). [CITE: Yambda — Yandex 2025 — Yandex — KEEP]
- **Аппаратура:** NVIDIA A100 (single-GPU), PyTorch 2.x, Triton 3.x.
- **SASRec базовая модель:** [CITE: Kang & McAuley 2018 — UCSD —
  KEEP] (исходная архитектура self-attentive sequential
  recommendation).
- [DATA: docs/thesis/results-data/results/sasrec_quality_ceiling.csv]
- [DATA: docs/thesis/04-datasets-notes.md]

---

## Краткое изложение результатов (Brief result statement)

Для заполнения после окончательной фиксации чисел §6. Кандидатные
формулировки:

- **Ко-дизайнерный IVF+INT8+Bloom-ретривер занимает Pareto-фронт во
  всех 11 ячейках quality-suite:** ускорение в $1.7\times$–$10.9\times$
  относительно exact-baseline (`linr_v1`) при потере Recall@100
  в 0.5–2.2%. Максимум ускорения зафиксирован на Yambda-5b-d128:
  $10.94\times$ (0.17 мс vs 1.87 мс на A100, $B = 1$, $K = 100$).
  [DATA: docs/thesis/07-results.md §6.2.3]
- **LinR V4 (INT8 dense) достигает нулевой измеренной потери
  Recall@100** при 50%-ном сокращении памяти индекса
  (например, 49.1 МиБ vs 194.6 МиБ на Goodreads-d128), но проигрывает
  V1 по латентности при $B = 1$ из-за per-row дeквантизационного
  эпилога и INT32-аккумулятора, не амортизирующихся при единичных
  запросах. [DATA: docs/thesis/07-results.md §6.6.3 speedup matrix;
  §6.3.4 memory tables]
- **LinR V3 (1-bit OPORP) при `candidate_pool = 5000` достигает exact
  Recall@100 на arXiv при $d \in \{128, 256\}$ с ускорением около
  $3\times$**; на большом каталоге Yambda-5b теряет 10–16%
  Recall при дефолтном пуле; deep sweep §6.5.2 показывает, что
  $\text{candidate\_pool} = 32{,}000$ восстанавливает Recall@100 до
  $\approx 0.99$ при минимальном росте латентности
  ($0.36 \to 0.42$ мс). [DATA: docs/thesis/07-results.md §6.5.2]
- **Масштабирование по $d$:** ко-дизайнерный ретривер — почти
  плоское (оценка $\alpha \approx 0.07$–$0.14$ в $\text{median\_ms}
  \propto d^{\alpha}$); LinR V1 — линейное ($\alpha \approx 0.78$–
  $1.25$). [DATA: docs/thesis/07-results.md §6.6.3 dim-scaling]
- [DATA: docs/thesis/results-data/results/all_results_long.csv]

[TODO: clarify with author — какие из четырёх кратких выводов выше
оставить в Введении (1–2 фразы), а какие отнести в §6 и упомянуть
во Введении только агрегированной формулой типа
«ко-дизайнерный ретривер достигает 1.7–10.9× ускорения при
0.5–2.2% потери Recall@100, см. §6.2»?]

**Caveat for the writer.** The Goodreads d128 / d256 filter-suite
cells are tainted by a stale-oracle cache (§7.8.5 / §6.0 ACTION
REQUIRED); the numbers above are taken from the QUALITY suite only,
which is unaffected. Do not quote any filter-suite number for
Goodreads d128 / d256 in Введение until the oracle is rebuilt.

---

## LinR-lineage statement (MUST APPEAR VERBATIM in Введение AND §1.5)

> **"Algorithms in the LinR family (V1 post-filter masking, V2
> pre-filter reduction, V3 1-bit OPORP) are reproduced as published
> in Borisyuk et al. (2024, CIKM). The co-designed IVF+INT8+Bloom
> system in this work is the author's own continuation of that line,
> motivated by the limitations of LinR V3 on filtered retrieval at
> high catalogue sizes; it composes classical primitives (Jégou 2011
> IVF, INT8 quantization, BitFunnel-style bloom signatures (Goodwin
> 2017)) under the unified `FilterModule` contract introduced in
> this thesis."**

- **Writer note.** This English text is the canonical source
  (00-thesis-plan.md §85). The Russian translation MUST preserve
  three technical claims 1:1:
  1. The three LinR variants (V1 / V2 / V3) are reproduced as
     published by Borisyuk et al. 2024 CIKM.
  2. The co-designed IVF+INT8+Bloom system is the **author's own
     continuation** of the LinR line, motivated by V3's filtered-retrieval
     limitations at high catalogue sizes.
  3. The co-designed retriever composes **classical primitives**
     (Jégou 2011 IVF, INT8 quantization, BitFunnel Bloom signatures
     [Goodwin 2017]) under the **unified `FilterModule` contract**
     introduced in this thesis.
- **Placement guidance.** Per 00-thesis-plan.md §102 ("Constraint:
  lineage statement (see Citation Policy) must appear here too")
  and §118 ("§1.5 prose explicitly contains the lineage statement"),
  the statement appears in BOTH §1.5 of Ch.1 AND in Введение.
- **LinR V4** is the author's INT8 dense variant of LinR V1; it is
  NOT in the original Borisyuk et al. 2024 paper. The lineage
  statement does not include V4 because V4 is presented as an
  author-side extension (see Ch.2 §2.2). [TODO: clarify with author
  — should Введение explicitly note that V4 is an author-side INT8
  variant of V1, or leave that distinction for Ch.2?]

---

## Структура работы (Structure of the work — one paragraph)

Suggested Russian text:

«Работа структурирована следующим образом. **Глава 1** — обзор
литературы по model-based GPU-retrieval, классическим методам ANN и
политике цитирования. **Глава 2** — методы: формальные определения
семейства LinR (V1 post-filter masking, V2 pre-filter reduction,
V3 1-bit OPORP, V4 INT8 dense) и ко-дизайнерного IVF+INT8+Bloom-
ретривера. **Глава 3** — датасеты и обучение query-моделей.
**Глава 4** — реализация пакета `torchretrieve`. **Глава 5** —
протокол оценки. **Глава 6** — результаты и анализ. **Глава 7** —
ограничения и направления развития. Завершают работу заключение
и приложения».

Chapter mapping (from 00-thesis-plan.md):

| # | RU title | Notes file |
|---|----------|------------|
| — | Введение | `02-introduction.md` (this file) |
| 1 | Обзор литературы | `01-literature-review.md` |
| 2 | Методы модельного GPU-retrieval | `03-methods.md` |
| 3 | Датасеты и обучение query-моделей | `04-datasets-notes.md` |
| 4 | Реализация (пакет `retrieve`) | `05-implementation.md` |
| 5 | Протокол оценки | `06-eval-protocol.md` |
| 6 | Результаты и анализ | `07-results.md` |
| 7 | Ограничения и направления развития | `08-limitations.md` |
| — | Заключение | `09-conclusion.md` |

---

## Citation candidates — kept vs dropped (Введение only)

| Citation | Affiliation | Decision | Used in |
|----------|-------------|----------|---------|
| Borisyuk et al. 2024, CIKM (LinR) | LinkedIn | KEEP | Актуальность, Lineage |
| Jégou et al. 2011 (IVF) | INRIA | KEEP | Lineage, Методы |
| Goodwin et al. 2017 (BitFunnel) | Microsoft | KEEP | Lineage, Методы |
| CAGRA — Ootomo et al. 2024 | NVIDIA | KEEP | Актуальность |
| SCANN — Guo et al. 2020 | Google | KEEP | Актуальность |
| TIGER — Rajput et al. 2023 | Google | KEEP | Актуальность |
| SASRec — Kang & McAuley 2018 | UCSD | KEEP | Эмпирическая база |
| nomic-embed-v1.5 — Nussbaum et al. 2024 | Nomic AI | KEEP | Эмпирическая база |
| Yambda — Yandex 2025 | Yandex | KEEP | Эмпирическая база |
| Goodreads — Wan & McAuley 2018 | UCSD | KEEP | Эмпирическая база |
| FAISS / FAISS-GPU — Johnson, Douze, Jégou | Meta (Douze, Jégou at FAIR) | **DROPPED** | — |
| DLRM — Naumov et al. 2019 | Meta | **DROPPED** | — |
| HSTU — Zhai et al. 2024 | Meta | **DROPPED** | — |
| EBR-Facebook — Huang et al. 2020 | Meta | **DROPPED** | — |

**Kept:** 10 citations.
**Dropped:** 4 citations (FAISS family, DLRM, HSTU, EBR-Facebook;
all Meta-affiliated).

---

## Writer's notes (Введение)

- **Result-statement freshness.** The "brief result statement"
  numbers above are read from the current `07-results.md` draft.
  They carry the stale-oracle caveat for Goodreads d128 / d256 filter
  cells (see Ch.7 §7.8.5 and §6.0 ACTION REQUIRED). Quote ONLY the
  quality-suite headlines in Введение; do NOT quote Goodreads
  d128 / d256 FILTER numbers until the oracle is rebuilt.
- **Lineage statement placement.** The Citation Policy requires the
  statement in both §1.5 of Ch.1 AND in Введение. Two options for
  Введение: (a) quote-block the English statement verbatim and
  follow with a Russian translation paragraph; (b) paraphrase in
  Russian with a back-reference to §1.5. Option (a) is safer for
  the audit; option (b) reads more naturally in academic Russian.
  [TODO: clarify with author — verbatim quote-block + Russian
  translation, or paraphrase + back-reference to §1.5?]
- **HSE Объект / Предмет convention.** The notes provide the
  Russian phrasing inline (under "Объект и предмет"); the writer
  should NOT retranslate — copy the wording as-is or adjust only
  for grammatical flow, since this section is judged on its
  adherence to the HSE convention.
- **Mirror with Заключение.** The 5–7 "Задачи" listed here must
  match 1:1 with the "Achievement of задач" mirror in Заключение
  §"Achievement of задач". Both files were drafted in sync. If the
  writer adds or removes a task bullet, the Заключение mirror must
  be updated correspondingly.
- **No SilverTorch in body.** The internal code symbol `SilverTorch`
  is used in repo, but the Введение body text must use neutral
  phrasing. The writer should run a grep for "silvertorch" /
  "SilverTorch" / "сильверторч" against the finished Russian prose
  before submission.
- **Trim guidance.** If 3–5 page target is overrun: cut from
  "Brief result statement" first (move detail into §6); the
  Актуальность and Цель / Задачи / Объект-Предмет sections are
  structurally required and cannot shrink below ~2 pages combined.

---

## Sources consulted (Введение)

- `docs/thesis/00-thesis-plan.md` (Введение contract at §95–§102;
  Citation Policy at §46–§94; lineage statement at §85; chapter
  map throughout)
- `docs/thesis/07-results.md` (headline quality-suite numbers for
  the brief result statement; failure-mode caveats)
- `docs/thesis/01-literature-review.md` (citation candidates,
  positioning vs LinR)
- `docs/thesis/03-methods.md` (algorithm formal definitions —
  V1/V2/V3/V4/co-designed retriever)
- `docs/thesis/04-datasets-notes.md` (dataset characteristics)
- `docs/thesis/06-eval-protocol.md` (methods description, oracle
  validation, sweep grids)
- `docs/thesis/08-limitations.md` (this thesis Ch.7 — for cross-
  reference of stale-oracle caveat)
- `docs/plans/00-roadmap.md` (Triton 3.x / `triton_op` migration
  context for the "Why now" claim)
- Git log entries (e.g. commit `4f9b1a6` — Sign-OPORP `k_bits` fix
  for byte-identical match at $k_{\text{bits}} = D$, referenced
  in Задача 7)
