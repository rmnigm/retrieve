# The paper's LaTeX sources

`main.tex` is the ECIR 2027 replicability submission (LNCS, double-blind);
`camera.tex` builds the same paper de-anonymised. The section files under
`sections/` follow the [outline](../outline.md) one to one and are
placeholders; `figures/` holds one stub per exhibit (T1, T-data, F1-F4, T2,
T3), each naming its source. Venue rules: [venue](../venue.md); what remains
before upload: [submission checklist](../submission-checklist.md).

## Build

```bash
make            # main.pdf, anonymous (tectonic)
make camera     # camera.pdf, authors and links in
make latexmk    # pdfLaTeX via latexmk, needs a TeX Live install
make clean
```

`make` needs `tectonic` on `PATH` (or `make TECTONIC=/path/to/tectonic`).
It fetches its TeX bundle on first run. On a pod:

```bash
mkdir -p /scratch/tools/tectonic && cd /scratch/tools/tectonic
curl -sL https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz | tar xz
```

Tectonic is XeTeX: keep the sources ASCII (accents as `{\"u}`, `\S` for §),
so the XeTeX and pdfLaTeX builds agree. Tectonic 0.15 prints "internal
consistency problem when checking if main.bbl changed" and stops at six
passes; the PDF and its references are complete. PDFs and build outputs are
not committed (`.gitignore`).

## Class files

`llncs.cls` 2.26 and `splncs04.bst` are Springer's LNCS package from
[CTAN](https://ctan.org/pkg/llncs), "distributed and/or modified under the
conditions of the CC BY license, version 4.0 or later"
([`LLNCS-README.md`](LLNCS-README.md), copyright Springer); `splncs04.bst`
carries its own LPPL notice. Springer's template zip ships the older 2.24 and
the same `.bst`.

## Bibliography

`refs.bib`: one entry per key, sorted by key, keys `firstauthorYEARword`.
Add entries in key order; how each was verified and what is still open is in
the [checklist](../submission-checklist.md#bibliography).
