# Заключение (Conclusion) — Reference Notes

**Status:** English reference notes for the downstream Russian-prose writing
agent. Not finished prose. Every claim is anchored to either a `path:line`
code reference or a `[DATA: ...]` pointer into the results. Заключение
introduces no new citations beyond those used in Введение / Ch.1 / Ch.6 /
Ch.7.

**Length target for finished Russian prose:** 1–2 pages
(00-thesis-plan.md §… Заключение contract).

**Hard constraint (carried from Citation Policy):** the internal code
symbol `SilverTorch` MUST NOT appear in body text; use neutral phrasing
("ко-дизайнерный IVF+INT8+Bloom-ретривер", "ко-дизайнерный ретривер").

---

## Voice / scope reminder for writer

- Tone: factual recap + concise outlook. No new arguments. No new
  citations.
- Structure: 1:1 mirror of Введение (Цель → restated; Задачи →
  achievement-of-задач). Plus the Заключение-only "quantitative
  highlights" paragraph, "open-source release" sentence, and "outlook"
  paragraph tied to Ch.7.
- The Заключение is the only place in the thesis where the open-source
  release is stated as a deliverable; verify the package name and
  license against the repo README before publication.

---

## Restatement of цель (1:1 with Введение)

**Suggested single-sentence formulation (Russian):**
«Работа реализует семейство алгоритмов LinR и ко-дизайнерный
IVF+INT8+Bloom-ретривер как открытый PyTorch+Triton-пакет с
репродуцируемой оценкой качества, латентности и памяти на трёх
открытых датасетах».

- **Writer note.** This is paraphrased from Введение §"Цель"
  («Воспроизвести семейство алгоритмов LinR и ко-дизайнерный
  IVF+INT8+Bloom-ретривер…»). The Заключение version shifts tense
  from infinitive ("воспроизвести") to indicative present
  ("реализует") and elides "open-source" → "открытый" for flow.

---

## Achievement of задач (1:1 mirror with Введение Задачи list)

For each of the 5–7 задач in Введение, one sentence of "what was
delivered" + concrete deliverable pointer:

1. **Реализованы Triton-ядра для LinR V1/V2/V3/V4 и для
   ко-дизайнерного IVF+INT8+Bloom-ретривера** в пакете
   [retrieve/src/retrieve/](retrieve/src/retrieve/): слои в
   [layers/linr/](retrieve/src/retrieve/layers/linr/) и
   [layers/silvertorch/](retrieve/src/retrieve/layers/silvertorch/),
   соответствующие Triton-ядра в
   [kernels/linr/](retrieve/src/retrieve/kernels/linr/) и
   [kernels/silvertorch/](retrieve/src/retrieve/kernels/silvertorch/);
   `torch.export`-чистота обеспечена mode-flag рефакторингом
   (план [docs/plans/torch-export-refactor.md](docs/plans/torch-export-refactor.md)).
2. **Собран единый бенчмарк-харнесс** с Hydra-style конфигами
   в [evaluation/config/](evaluation/config/) (директории
   `quality/`, `filter/`, `deep_sweeps/`); оракулы top-K кэшируются
   с валидацией на форму (хеширование содержимого — пункт
   §7.8.5 плана развития).
3. **Обучены SASRec query-кодировщики** для Goodreads
   ($d \in \{64, 128, 256\}$), Yambda-500m ($d \in \{64, 128, 256\}$)
   и Yambda-5b ($d \in \{64, 128\}$); для arXiv использованы
   предзаписанные эмбеддинги nomic-embed-v1.5.
4. **Прогнаны quality- / filter- / deep-sweep-протоколы** на
   трёх открытых датасетах; результаты агрегированы в long-format
   CSV [DATA: docs/thesis/results-data/results/all_results_long.csv].
5. **Проведены абляции:** sweep по $\text{candidate\_pool} \in
   \{2k, 4k, 8k, 16k, 32k\}$ для LinR V3 (§6.5.2); сетка
   $n_{\text{lists}} \times n_{\text{probe}}$ для ко-дизайнерного
   ретривера (§6.5.1); dim-scaling по $d \in \{64, 128, 256\}$
   (§6.6.3).
6. **Пакет опубликован** под именем `torchretrieve` (PyPI);
   конфиги, оракулы и сырые JSON-результаты лежат в
   [evaluation/results/](evaluation/results/) и
   [docs/thesis/results-data/](docs/thesis/results-data/) для
   полной воспроизводимости.
7. *(Optional 7th — only if Введение includes it.)* Зафиксированы
   зоны соответствия и расхождения с опубликованным диапазоном LinR:
   paper-strict OPORP-вариант (Sign-OPORP с bin-sum и L2-нормализацией)
   совпадает с эталоном побайтно при $k_{\text{bits}} = D$
   (commit `4f9b1a6`).

[TODO: clarify with author — включает ли финальный список Задач
седьмой пункт? Если нет — удалить и из Заключения для сохранения
1:1 mirror.]

---

## Quantitative highlights (one paragraph)

Suggested Russian-prose content (writer compresses to one paragraph):

- **Лучшие Recall@100 (exact baseline `linr_v1`, ячейки quality
  suite):**
  - Goodreads d128: **0.1479**
  - arXiv d128: **1.0000** (exact by construction — ceiling)
  - Yambda 500m d128: **0.1486**
  - Yambda 5b d128: **0.1779** (absolute maximum in the study)
  [DATA: docs/thesis/results-data/results/all_results_long.csv;
  docs/thesis/07-results.md §6.2 tables]
- **Максимальное ускорение:** ко-дизайнерный IVF+INT8+Bloom-ретривер
  достигает **$10.94\times$** ускорения относительно exact-baseline
  на Yambda-5b-d128 (0.17 мс vs 1.87 мс на A100, $B = 1$, $K = 100$)
  при потере Recall@100 в 2.2% [DATA: docs/thesis/07-results.md §6.2.3].
- **Память:** LinR V4 (INT8 dense) сокращает индекс на **49.1%**
  относительно LinR V1 (fp16) — например, $194.6 \to 97.3$ МиБ на
  Goodreads-d128 и $1{,}310 \to 654.6$ МиБ на Yambda-5b-d128 — при
  нулевой измеренной потере Recall@100 во всех 11 ячейках quality
  suite [DATA: docs/thesis/07-results.md §6.3.4 memory tables;
  §6.6.3 rank stability].
- **Pareto-фронт:** ко-дизайнерный ретривер доминирует на
  Pareto-фронте «качество × латентность» во всех 11 ячейках
  quality suite при дефолтных гиперпараметрах.
- **Масштабирование по $d$:** ко-дизайнерный ретривер демонстрирует
  почти плоское масштабирование ($\alpha \approx 0.07$–$0.14$
  в $\text{median\_ms} \propto d^{\alpha}$); LinR V1 — линейное
  ($\alpha \approx 0.78$–$1.25$).
  [DATA: docs/thesis/07-results.md §6.6.3 dim-scaling exponents]

**Caveat for the writer.** The Goodreads d128 / d256 FILTER suite
cells are tainted by a stale-oracle cache (см. §7.8.5 / §6.0
ACTION REQUIRED). All highlights above come from the QUALITY suite
which is unaffected. Do NOT quote Goodreads d128 / d256 filter
numbers in Заключение until the oracle is rebuilt.

---

## Open-source release statement (Заключение-only deliverable)

Suggested Russian-prose content:

«Пакет опубликован под именем `torchretrieve` на PyPI; исходный
код доступен в репозитории [retrieve/](retrieve/). Конфигурации
бенчмарков, валидированные оракулы и сырые JSON-результаты
размещены в [evaluation/](evaluation/) и
[docs/thesis/results-data/](docs/thesis/results-data/) для полной
воспроизводимости. Лицензия и инструкции по запуску приведены в
файле README репозитория».

[TODO: clarify with author —
  (a) точное название PyPI-пакета (`torchretrieve`?);
  (b) хеш релизного коммита для зеркалирования в Заключении;
  (c) тип лицензии (Apache 2.0 / MIT / другое) — указать конкретно
      или сослаться на README без названия лицензии?
]

---

## Outlook (one paragraph, tied to Ch.7)

The outlook paragraph pulls 2–3 highest-leverage future directions
from Ch.7. Avoid repeating the full Ch.7 catalogue.

**Suggested selection (writer chooses 2–3):**

1. **Multi-GPU index sharding** (Ch.7 §7.1) — opens the path to
   $N \times d$ regimes that exceed single-GPU HBM, particularly for
   $N \geq 10^7$ at $d = 256$ where the dense index already exceeds
   5 GiB.
2. **Live-update API** (Ch.7 §7.2, plan
   [docs/plans/live-update-api.md](docs/plans/live-update-api.md))
   — enables streaming RecSys deployments where item churn is on
   minute timescales; the current offline-indexing constraint
   excludes those use cases.
3. **Hand-tuned CUDA baseline** (Ch.7 §7.5; substitutes
   CAGRA / Milvus / SCANN for the excluded FAISS family) — measures
   the Triton-vs-CUDA ceiling on identical hardware and quantifies
   the cost of the open-source-Triton positioning.

[TODO: clarify with author — какие 2–3 направления выбрать для
outlook-параграфа? Предложение выше: §7.1 (мульти-GPU),
§7.2 (live-update), §7.5 (CUDA-baseline). Альтернативы:
§7.3 (learned similarity), §7.4 (BEIR / MS-MARCO), §7.6 (per-block
INT8 / INT4).]

---

## Citation candidates — kept vs dropped (Заключение only)

Заключение adds **no new citations**. All citation candidates are
reused from Введение, Ch.1, Ch.6, Ch.7:

| Reused citation | Affiliation | Decision | Source chapter |
|-----------------|-------------|----------|----------------|
| Borisyuk et al. 2024, CIKM (LinR) | LinkedIn | KEEP (already cited) | Ch.1 §1.5, Введение |
| (none new) | — | — | — |

**Confirmed dropped (no resurrection in Заключение):** FAISS family,
DLRM, HSTU, EBR-Facebook — all Meta-affiliated.

References to `docs/plans/*.md` in the outlook paragraph are
**roadmap pointers**, not citations; they do not enter the
bibliography.

---

## Writer's notes (Заключение)

- **Structural mirror discipline.** Заключение is a 1:1 mirror of
  Введение. Length budget: 1–2 pages of finished Russian prose. If
  the draft grows beyond 2 pages, cut from the quantitative-highlights
  paragraph FIRST (move detail to §6), not from the achievement-of-задач
  mirror.
- **No new numbers.** The quantitative-highlights paragraph must NOT
  introduce any number that did not already appear in Ch.6. The Russian
  writer should cross-check every number against the §6.x table that
  produced it.
- **Open-source release is the only new content.** This is the sole
  Заключение-only deliverable that does NOT appear elsewhere in the
  thesis. Verify the PyPI package name, the license, and the release
  commit hash against the repo README before publication.
- **No "SilverTorch" in body text.** As with all other chapters, the
  internal code symbol MUST NOT appear in the finished Russian prose;
  use "ко-дизайнерный IVF+INT8+Bloom-ретривер" or
  "ко-дизайнерный ретривер".
- **Stale-oracle caveat carries forward.** The Goodreads d128 / d256
  filter numbers are unreliable; only quote quality-suite numbers in
  Заключение's highlights paragraph.
- **Open questions / [TODO: clarify with author — ...] markers:**
  - Точное название PyPI-пакета и тип лицензии для open-source
    release-предложения?
  - Какие 2–3 направления выбрать для outlook-параграфа?
  - Включает ли финальный список Задач (Введение) седьмой пункт
    (сравнение с опубликованным диапазоном LinR)? Это меняет
    структуру 1:1 mirror в Заключении.

---

## Sources consulted (Заключение)

- `docs/thesis/00-thesis-plan.md` (Заключение contract; mirror
  discipline; open-source statement requirement)
- `docs/thesis/07-results.md` (headline numbers — quality-suite
  Recall@100, speedup max, memory savings, dim-scaling exponents;
  stale-oracle caveat)
- `docs/thesis/02-introduction.md` (mirror structure for Задачи)
- `docs/thesis/08-limitations.md` (outlook tie-in — §7.1 / §7.2 /
  §7.5 directions)
- `docs/plans/00-roadmap.md` (release / packaging context, Stage 4
  measurement-gap reruns)
- `docs/plans/live-update-api.md` (outlook §7.2 reference)
- Repo `retrieve/src/retrieve/` layout (deliverable pointers in
  achievement-of-задач mirror)
- `evaluation/config/{quality,filter,deep_sweeps}/` (harness
  deliverable for Задача 2)
- `evaluation/results/` (raw JSONs for Задача 6 reproducibility
  claim)
- Git log entries (commit `4f9b1a6` for paper-strict OPORP
  byte-identity claim in Задача 7)
