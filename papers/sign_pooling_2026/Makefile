PY    ?= python3
PAPER  = paper
REPL   = replication

.PHONY: help paper sims eqtl figures all clean

help:
	@echo "Targets:"
	@echo "  make paper    - compile paper/article.pdf (pdflatex + bibtex + pdflatex x2)"
	@echo "  make sims     - run the simulation (R=200) and regenerate Fig S1-S5, Tables S1-S4"
	@echo "  make eqtl     - run the eQTL pipeline and regenerate Fig E1-E3, Table E1"
	@echo "  make figures  - regenerate all figures and tables into paper/"
	@echo "  make all      - figures + paper"
	@echo "  make clean    - remove LaTeX build artifacts"

paper:
	cd $(PAPER) && pdflatex -interaction=nonstopmode article.tex && bibtex article && pdflatex -interaction=nonstopmode article.tex && pdflatex -interaction=nonstopmode article.tex

sims:
	cd $(REPL) && $(PY) sign_pooling_simulation.py && $(PY) sign_pooling_robustness.py && $(PY) exhibits.py

eqtl:
	cd $(REPL) && $(PY) eqtl_pipeline.py && $(PY) eqtl_exhibits.py

figures: sims eqtl

all: figures paper

clean:
	cd $(PAPER) && rm -f *.aux *.bbl *.blg *.log *.out *.spl
