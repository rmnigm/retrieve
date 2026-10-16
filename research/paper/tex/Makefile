# TeX toolchain: tectonic (fetches its TeX bundle on first run) or latexmk.
TECTONIC ?= tectonic

.PHONY: all camera latexmk clean

all: main.pdf

main.pdf: main.tex refs.bib llncs.cls splncs04.bst $(wildcard sections/*.tex figures/*.tex)
	$(TECTONIC) main.tex

camera: camera.pdf

camera.pdf: camera.tex main.pdf
	$(TECTONIC) camera.tex

latexmk:
	latexmk -pdf main.tex

clean:
	rm -f *.pdf *.aux *.bbl *.blg *.log *.out *.fls *.fdb_latexmk *.synctex.gz
