PY ?= python3
REPL = replication

.PHONY: help paper sims eqtl figures cached-figures

help:
	@echo "Set PY to the configured external interpreter. Outputs stay outside the checkout."
	@echo "  make paper          - build approved current assets in external latex-build/"
	@echo "  make sims           - full simulations and new simulation exhibits"
	@echo "  make eqtl           - eQTL fits and new exhibits (requires MOSEK)"
	@echo "  make figures        - sims + eqtl; does not replace approved paper assets"
	@echo "  make cached-figures - simulation exhibits from frozen reference tables"

paper:
	$(PY) $(REPL)/build_paper.py

sims:
	$(PY) $(REPL)/sign_pooling_simulation.py
	$(PY) $(REPL)/sign_pooling_robustness.py
	$(PY) $(REPL)/exhibits.py --from-run

eqtl:
	$(PY) $(REPL)/eqtl_pipeline.py
	$(PY) $(REPL)/eqtl_exhibits.py

figures: sims eqtl

cached-figures:
	$(PY) $(REPL)/exhibits.py
