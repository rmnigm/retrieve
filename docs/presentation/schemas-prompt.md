# Prompt for a follow-up agent: draw the 5 algorithm schemas (S1–S5) for the defense deck

You are continuing work on a Master's-thesis defense presentation at HSE FCS. The deck `docs/presentation/pre-defense-ru.tex` is finished except for five algorithm-diagram placeholders that currently render as dashed grey TODO boxes. Your job: replace each placeholder with a real TikZ schema inside the beamer file, using the visual style already established by the deck and the existing TikZ block on slide 2.

This prompt is self-contained — you do not need to ask the user anything. Compile after every schema and ship the working file.

## Repository orientation

Working dir: `/Users/rmnigm/work/retrieve`. Key files:

- `docs/presentation/pre-defense-ru.tex` — the deck. The five placeholders are calls to `\schemaTODO{width}{height}{caption}`. The macro definition is in the preamble (search for `\newcommand{\schemaTODO}`).
- `docs/presentation/HSE-theme/beamerthemeHSE.sty` — HSE colour palette. Use `HSEblue` for arrows/highlights (already loaded). Other available colours: `HSEorange`, `HSEgreen`, `HSEred`, `HSEdarkblue`.
- `docs/thesis/main.tex` — source of truth for the algorithms. Pseudocode lives in chapter 3:
  - LiNR V1 → algorithm `algo:linr_v1` (lines ~392–402)
  - LiNR V2 → `algo:linr_v2` (~406–417)
  - LiNR V3 → `algo:linr_v3` (~422–440)
  - LiNR V4 → `algo:linr_v4` (~444–458)
  - QuantizedIVF → `algo:quantized_ivf` (~474–498)
  Architecture / API description is in chapter 4 (~509–552).
- `docs/thesis/figures/` — already-rendered result PNGs (used elsewhere in the deck, not source for schemas).
- `articles/linr.md`, `articles/silvertorch.md` — original papers with their own figures; use as conceptual references only, do not copy them. SilverTorch is the conceptual ancestor of QuantizedIVF; LiNR is the direct ancestor of V1–V4.

## Visual style — match this exactly

The deck already establishes the look in the slide-2 pipeline (search `node distance=0.42cm, box/.style`). Reuse:

```
box/.style={draw, rounded corners=3pt, fill=blue!6,
            text width=<...>, align=center,
            font=\footnotesize, minimum height=0.55cm},
opt/.style={draw, dashed, rounded corners=3pt, fill=blue!3,
            text width=<...>, align=center,
            font=\footnotesize, minimum height=0.55cm},
arr/.style={-Stealth, thick, color=HSEblue}
```

Conventions to follow:
- Solid blue-tinted boxes (`box`) for required compute steps.
- Dashed boxes (`opt`) for optional/parametric steps or offline-only blocks.
- All arrows `-Stealth` thick HSEblue.
- Font: `\footnotesize` inside boxes, `\scriptsize` for tiny labels on arrows.
- Use TikZ libraries already loaded: `arrows.meta, positioning, shapes.geometric, fit, backgrounds`. Do NOT load new libraries.
- Keep each schema within the size the placeholder was reserving: ≈ `0.55\textwidth` wide × `5.6cm` tall (S5 = `5.4cm`). The `\schemaTODO{w}{h}{label}` call tells you the exact reserved box.
- No external images, no `\includegraphics`. Pure TikZ.
- Group offline blocks into a shaded `\node[fit=...]` rectangle labelled "offline" (use the `backgrounds` library + `on background layer`). Mark online blocks similarly when both phases coexist (S2, S3, S4).

## The five schemas

For each, the bullets below are the data flow — translate each bullet into a node, then connect with arrows in left-to-right or top-to-bottom flow. Russian inside boxes (matches surrounding text); math in standard LaTeX. Keep box labels short — 2–4 words plus one inline formula at most.

### S1 — LiNR V1 vs V2 (slide 8, replaces the `\schemaTODO` after "Базовая пара алгоритмов из статьи LiNR")

Two parallel horizontal flows stacked vertically, with a small "V1" / "V2" tag on the left margin of each row.

**V1 row** (top):
`Q, X` → `S = QX^⊤` (FP16) → `mask: S[M=0] ← −∞` → `row-wise top-K` → `I, S`

**V2 row** (bottom):
`Q, M` → `gather: P_i = {j: M[i,j]=1}` (and `X[P_i]`) → `Q[i]·X[P_i]^⊤` → `top-K` → `remap: I_i ← P_i[top]`

Visual cue for the contrast: shade the "S = QX^⊤" node in V1 a slightly darker blue and add a tiny note below: "весь $X$". Shade "Q[i]·X[P_i]^⊤" in V2 the same way, with note "только $X[P_i]$".

### S2 — LiNR V3 (slide 9)

Two phases. Make the split explicit with a thin horizontal separator or two `fit` boxes.

**Offline (top, in a labelled "offline" fit box, dashed border):**
`X` → `permute by π` → `multiply R^⊤` → `sign` → `B_X ∈ {0,1}^{N×W}` (FP16 stored as packed 64-bit words)

**Online (bottom):**
`Q` → (same offline-style transform: `π, R^⊤, sign`) → `B_Q` →
`Hamming: H = popcount(B_Q ⊕ B_X)` → `mask M` → `top-P_c by H` → `rescore: Q[i]·X[C_i]^⊤ FP16` → `top-K`

Label the arrow leaving the `top-P_c` node with "P_c кандидатов" so the reader sees the speed↔recall knob.

### S3 — LiNR V4 (slide 10)

Looks like V1 with an explicit INT8 branch. Single horizontal flow, with a small "offline" fit box on the left for the index-side quantization.

**Offline (small box, dashed border, top-left):**
`X` → `INT8-quantize` → `X̃, s_X` (общий scale)

**Online (main row):**
`Q` → `INT8-quantize` → `Q̃, s_Q` → **`Triton kernel: INT8 GEMM`** (highlight this box — fill `HSEblue!18` or similar, bold label) → `S̃` → `scale: S = s_Q·s_X·S̃` → `mask` → `top-K`

Add a small annotation under the scale step: "общий $s_X$ → top-K без полного dequant" (`\scriptsize`, HSEorange text).

### S4 — QuantizedIVF (slide 11) — the headline schema of the work

Three sequential phases inside one big rounded rectangle labelled **"Triton kernel"** (HSEblue border, fill `HSEblue!4`). Use `\node[fit=...]` from the `fit` library.

**Offline (small box on the left, outside the kernel rectangle, dashed):**
- `k-means++` → `µ, C` (cluster centroids and assignments)
- `INT8-quantize X` → `X̃, s_X`
- `Bloom-encode attrs(j)` → `b[j]` (1024-бит)

**Inside the Triton-kernel rectangle, three labelled phases left-to-right:**

1. **Probe**: `q` → compare to centroid matrix `µ` → `top-n_probe clusters P_i`
2. **Filter** (per cluster `C_ℓ ∈ P_i`, per item `j`): `FilterModule` →
   - Branch A: `ExactAttributeFilter` (solid)
   - Branch B: `BloomFilter`: bitwise `b_query AND b[j]` (solid, parallel branch)
   - Both branches merge into "выживает $j$?" → drop or pass
3. **Score**: `INT8 GEMM ⟨q̃, x̃[j]⟩ · s_Q·s_X` → push into `top-K heap` → `top-K`

Show n_probe and the heap as the two visible knobs. The Filter phase is the only one where the two filter variants exist — draw them as two parallel sub-boxes inside that phase. Add small label "одно ядро, без graph break" under the kernel rectangle.

### S5 — torchretrieve architecture (slide 13)

System-level diagram, not an algorithm flow. Layout: one big `torch.compile`-блок rectangle (rounded, `HSEblue!4` fill, thick HSEblue border) containing the online path, with offline blocks outside it.

**Inside the compile block, left-to-right:**
- `query_attrs, query` →
- `query encoder (nn.Module)` produces `q` →
- `FilterModule.evaluate_mask(query_attrs)` produces `mask` (split: one branch produces mask, q passes through) →
- `RetrieverModule(q, mask)` — show two stacked sub-boxes inside: `backend="triton"` and `backend="pytorch"`, indicating they are interchangeable; one arrow leaves the merged module →
- `top-K`

**Outside the compile block (label area "offline / pre-deployment"):**
- `register_index(item_embs)` → arrows into `RetrieverModule`
- `register_index(item_attrs)` → arrows into `FilterModule`
- `autotuning CLI` → arrow into both modules (parameter injection)

Bottom annotation (HSEorange `\scriptsize`): "численная верификация: $\Delta\mathrm{Recall@}100 = 3{,}5\cdot10^{-4}$".

## Workflow

1. Read the current `pre-defense-ru.tex` end-to-end so you understand the surrounding text and the macro.
2. Locate the five `\schemaTODO{...}{...}{...}` calls — one each on slides 8, 9, 10, 11, 13. Note the width/height each placeholder reserves; design your TikZ to fit within those dims (with a small margin).
3. Replace one placeholder at a time. After each replacement: `cd docs/presentation && xelatex -interaction=nonstopmode pre-defense-ru.tex` and check for compile errors and Overfull `\vbox` warnings on the affected slide. Fix sizing before moving on.
4. When all five are done, run `xelatex` twice in a row to settle aux files, then open `pre-defense-ru.pdf` and visually scan each replaced slide for crowding / overlap / cut-off labels. Tighten `node distance` and `text width` if needed.
5. Keep the `\schemaTODO` macro defined in the preamble even if unused at the end — harmless.

## Constraints

- Do not edit any file other than `docs/presentation/pre-defense-ru.tex`.
- Do not add new LaTeX packages or TikZ libraries — work with what is already in the preamble (`arrows.meta, positioning, shapes.geometric, fit, backgrounds`).
- Do not change Russian text on the slides outside the schemas. Do not renumber slides.
- Do not touch the `\graphicspath`, the title page, or the existing TikZ on slide 2.
- Keep every schema self-contained inside its `\begin{tikzpicture}…\end{tikzpicture}` so it can be cut/pasted later if the user wants to re-arrange slides.

## Verification before reporting done

- `xelatex` exits 0; PDF has exactly 20 pages.
- No `Overfull \vbox` warnings (Overfull `\hbox` of 1–2 pt is OK).
- Each of slides 8, 9, 10, 11, 13 shows a real schema where a dashed TODO box used to be.
- Visual check: arrows actually connect their endpoints (no floating arrows); no text leaves its bounding box; nothing overlaps the column-text on the left half of the slide.
