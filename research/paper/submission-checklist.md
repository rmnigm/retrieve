---
title: submission-checklist
created: 2026-10-16
updated: 2026-10-16
type: summary
tags: [paper]
sources: [research/paper/venue.md, research/paper/release-and-licenses.md, research/paper/abstract.md, research/paper/outline.md, research/paper/tex/]
---

# Submission checklist

Everything the ECIR 2027 replicability submission (19 October 2026,
EasyChair `ecir27`) and the camera-ready need, beyond the prose. The rules
are quoted in [venue](venue.md); the licence facts come from
[release and licenses](release-and-licenses.md). **User** marks a decision or
an action only the user can take.

## Format

- [ ] LNCS via [`tex/`](tex/README.md): `make` builds the anonymous
      `main.pdf`, `make camera` the camera-ready `camera.pdf`.
- [ ] At most 12 pages before the references, appendices included (the
      outline puts nothing in an appendix). Check the page the bibliography
      starts on, not the total.
- [ ] No `\todo{}` left, and the `\nocite{*}` in `main.tex` removed once the
      prose cites, so the reference list holds only cited works.
- [ ] Title as registered in EasyChair ([abstract](abstract.md): the
      recommended title is not confirmed against the record). Abstract
      verbatim from the EasyChair record; keywords.
- [ ] Sources ASCII only (accents as `{\"u}`): `make` runs tectonic, which
      is XeTeX, and `T1` font encoding misprints raw Unicode there (`§`
      came out as `ğ`). Springer typesets with pdfLaTeX; `make latexmk`
      needs a TeX Live install, which the pods do not have, so the
      pdfLaTeX build is **unverified**.

## Anonymity

The switch is `\ifanonymous` in `main.tex` (true unless `camera.tex`
defines `\camera`). Anonymous build verified on 2026-10-16: author block
"Anonymous Author(s)", no acknowledgements, `\codeurl` → the mirror
placeholder, no `github.com/rmnigm` in the PDF text.

- [ ] **User:** the anonymous mirror on anonymous.4open.science of the
      repository at the submission tag, its term list replacing the
      author's name, `rmnigm`, `pinkmeme`, the e-mail address and "thesis"
      ([release §4](release-and-licenses.md#4-double-blind-review)). Put its
      URL in `\codeurl`.
- [ ] Which tree the mirror shows: `staging` at the submission tag, or the
      export cleanup branches once merged (they merge only after their own
      gates). **User / controller.**
- [ ] The paper names no GitHub account, Hub account, `pinkmeme/*` repo,
      thesis, or the `torchretrieve` package name (it leads to GitHub; the
      user may accept that, [release §4](release-and-licenses.md#4-double-blind-review)).
- [ ] No self-citation of a technical report; no arXiv preprint of this
      paper during review (the call's preprint rule).
- [ ] PDF metadata: `hyperref` sets no `pdfauthor`; check the built PDF's
      properties before upload.
- [ ] Supplementary archive file names and contents carry no name or
      account (the results files' `path` fields hold container paths only,
      [release §3](release-and-licenses.md#3-what-is-on-the-hub-and-github-now)).

## Artifacts for review

The call's options: an anonymous repository linked in the paper, or upload
to EasyChair ([venue](venue.md#artifacts)).

- [ ] **Code**: the anonymous mirror (above).
- [ ] **Results**: one archive with the cited `results.parquet`, the
      `MANIFEST.json`s and `campaign.yaml`, from the private Hub
      `eval-results` subtrees the paper cites (the Hub repo itself is not
      linked during review). Into the mirror or as EasyChair supplementary.
- [ ] **Data**: the `eval-data … all` recipes and checksums in the mirror's
      docs; derived arXiv files only through an unlinked account, if at
      all. Goodreads is recipe-only.
- [ ] **User:** release decisions 1-6 of
      [release §5](release-and-licenses.md#5-decisions-for-the-user)
      (goodreads checkpoints, Re-LAION vectors, PubMed slice, the public May
      result repos, `eval-results` shape, GitHub visibility during review).
      The paper's availability paragraph (§8) depends on them.

## Licences and attribution the paper carries

From [release §1-§2](release-and-licenses.md#1-datasets); each line needs a
citation or a sentence in §3 or §8.

| input | terms | the paper must |
|---|---|---|
| Goodreads | academic use only, no redistribution | cite `wan2018item` and `wan2019fine`; say the data is not redistributed (recipe only) |
| arXiv metadata (`open-index/open-arxiv`) | CC0 | courtesy: arXiv, `openindex2026openarxiv`; name `nussbaum2025nomic` as the encoder |
| PubMed / MedCPT | public domain (NCBI); NLM download terms | "Courtesy of the U.S. National Library of Medicine"; say the 2026 baseline is not current (`nlm2026medline`); cite `jin2023medcpt`; share no titles or abstracts |
| Re-LAION-2B-en-research-safe | Apache-2.0; LAION's research-use wording | cite `laion2024relaion` (and `schuhmann2022laion` for LAION-5B); never captions or URLs |
| Yambda | Apache-2.0 | cite `ploshkin2025yambda` |
| Meta's SilverTorch code | Apache-2.0 | cite `xue2026silvertorch` and `meta2026silvertorch` (pin `21aa35e`); say the adapted code is modified, changes listed in CHANGES, no endorsement by Meta |
| our code | Apache-2.0 | §8 availability; the `torchretrieve` sdist has no LICENSE file yet (F4) |

## Author information (camera-ready; placeholders in `main.tex`)

- [ ] **User:** author names, order, affiliations, e-mail, ORCID iDs
      (Springer encourages them), the corresponding author (must match the
      Consent-to-Publish signer), the running head (`\authorrunning`).
- [ ] Authorship is frozen at submission: "Addition of authors after the
      submission deadline" is a desk-rejection ground, so the EasyChair
      author list must be final on 19 October.

## Statements

- [ ] **Disclosure of Interests** (LNCS mandatory, both builds):
      `sections/credits.tex`. **User:** the statement (default wording is
      in the placeholder).
- [ ] **AI use** — **User decision.** The study's code, harness and
      documentation were produced with AI coding agents under the author's
      direction. Springer's policy (as quoted in
      [venue](venue.md#statements-ethics-ai-use-reproducibility)): an LLM is
      not an author; substantive LLM use is documented in the Methods
      section (or a suitable alternative); AI copy editing need not be
      declared. The call requires no statement at submission. Options: one
      sentence in §3 (setup) or §8 at submission, or at camera-ready only.
      The [abstract page](abstract.md#open-before-the-paper) notes a GenAI
      statement may be requested at camera-ready.
- [ ] **Ethics**: no human subjects; public datasets used under their
      terms (table above). No section required.
- [ ] **Communication with the original authors** (the call asks for it):
      §7 states what was asked and answered, "no reply" included
      ([outline](outline.md#open-for-the-user), item 1). **User.**

## Bibliography

[`tex/refs.bib`](tex/refs.bib), 35 entries, sorted, keys
`firstauthorYEARword`. Every DOI resolved through doi.org and its Crossref
record on 2026-10-16; the arXiv-only entries through the arXiv API; HSTU on
PMLR, PyTorch on the NeurIPS proceedings page, Nomic Embed on OpenReview
(TMLR 2025); the web entries (Meta's repo and the commit, cuVS, open-arxiv
revision, Re-LAION post, NLM page) by fetching them. Open points:

- [ ] SilverTorch's author list: the ACM record (used) has 29 authors; the
      arXiv v5 has 32 (adds Jiaqi Zhai, Kevin Greer, Rui Zhang). Check the
      ACM PDF's first page.
- [ ] Yambda: the ACM record (used) has 8 authors; arXiv lists a ninth
      (N. Savushkin).
- [ ] Big-ANN: author list and order from Crossref (21 names), not the
      arXiv page the material page pointed to.
- [ ] Efron 1979: pages not in Crossref; the field is left out.
- [ ] The Faiss library is cited as its journal version (IEEE TBD 2026);
      Johnson et al. 2021 is the GPU Faiss paper.
- [ ] Not added: the ACM "Artifact Review and Badging" page (the call links
      it; the page blocked the fetch) and the five style-reference papers of
      the [abstract research](../artifacts/abstract-draft/repro-abstracts.md),
      which a parallel reference-paper list may add.

## Camera-ready (after 7 December 2026, if accepted)

- [ ] `make camera`: authors, ORCIDs, acknowledgements, `\codeurl` to the
      public GitHub repository, Zenodo DOI (F4) in §8.
- [ ] The public releases of [release §4](release-and-licenses.md#4-double-blind-review)
      ("After acceptance"): repos flipped, PyPI, Zenodo with the tag, the
      results subset and Meta's pinned sdist.
- [ ] The Consent-to-Publish form, signed by the corresponding author.
- [ ] One author registers and presents in person (Southampton, 21-25 March
      2027).
- [ ] LaTeX sources for Springer, built once with pdfLaTeX.
