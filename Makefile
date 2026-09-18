# Pipeline entry points for the analysis harness.
#
# Targets are grouped by what they need. Everything under "offline" runs with no
# model weights and no real data, which is most of the harness; the "online"
# targets need DINOv3 access and UNICORNv2 in config.yaml.
#
#   make setup      create the virtualenv and install everything
#   make offline    everything that runs without gated weights or data
#   make online     reproduction, then the corrected experiment, then analysis
#
# Run `make help` for the full list.

PYTHON := .venv/bin/python
UV := uv

.DEFAULT_GOAL := help
.PHONY: help setup test offline online clean \
        stats plots fakedata smoke variants analyse reproduce features check

help:
	@echo "Setup"
	@echo "  setup        create .venv and install dependencies"
	@echo ""
	@echo "Offline - no model weights or real data needed"
	@echo "  test         run the test suite"
	@echo "  stats        analyse the published results -> analysis/reported_stats.*"
	@echo "  plots        generate figures -> analysis/*.png"
	@echo "  fakedata     synthetic UNICORNv2-shaped chips -> data/fake"
	@echo "  smoke        end-to-end pipeline check on the stub backbone"
	@echo "  variants     variant experiment on stub data (protocol check)"
	@echo "  offline      all of the above, in order"
	@echo ""
	@echo "Online - needs DINOv3 weights and UNICORNv2 in config.yaml"
	@echo "  features     extract and cache real features"
	@echo "  reproduce    run the five published methods and compare (D4)"
	@echo "  variants-real  the corrected experiment, 15 seeds (D5)"
	@echo "  analyse      analyse experiment output (D6)"
	@echo "  online       reproduce, then variants-real, then analyse"
	@echo ""
	@echo "  check        verify the environment and report what is blocked"
	@echo "  clean        remove generated artifacts (keeps figures and .venv)"

setup:
	HF_HUB_DISABLE_XET=1 $(UV) venv .venv --python 3.12
	HF_HUB_DISABLE_XET=1 $(UV) pip install --python $(PYTHON) \
		-r requirements.txt -r requirements-analysis.txt
	@echo "\nDone. Run 'make check' to see what is available."

# --- offline ---------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -q

stats:
	$(PYTHON) src/run_reported_stats.py

plots:
	$(PYTHON) src/make_plots.py

fakedata:
	$(PYTHON) src/make_fake_unicorn.py --per-class 10

smoke: fakedata
	$(PYTHON) src/smoke_pipeline.py

variants: fakedata
	$(PYTHON) src/run_variant_experiment.py --stub --data-root data/fake \
		--seeds 3 --epochs 2
	$(PYTHON) src/analyse_experiment.py analysis/variant_experiment_stub.json

offline: test stats plots smoke variants
	@echo "\nOffline pipeline complete. Findings that need no compute are in"
	@echo "analysis/reported_stats.json and analysis/CONFOUNDS.md."

# --- online ----------------------------------------------------------------

features:
	$(PYTHON) src/extract_and_save_features.py --split train_eo
	$(PYTHON) src/extract_and_save_features.py --split train_sar
	$(PYTHON) src/extract_and_save_features.py --split test_sar

reproduce:
	$(PYTHON) src/run_reproduction.py

variants-real:
	$(PYTHON) src/run_variant_experiment.py --seeds 15

analyse:
	$(PYTHON) src/analyse_experiment.py analysis/variant_experiment_dinov3.json

online: reproduce variants-real analyse
	@echo "\nOnline pipeline complete."

# --- utilities -------------------------------------------------------------

check:
	@$(PYTHON) -c "import sys, pathlib; \
	cfg = pathlib.Path('config.yaml'); \
	print('config.yaml      ', 'present' if cfg.is_file() else 'MISSING - copy config-example.yaml'); \
	import yaml; \
	paths = (yaml.safe_load(cfg.read_text()).get('paths', {}) if cfg.is_file() else {}); \
	[print(f'{k:17}', 'ok' if pathlib.Path(str(v)).exists() else f'MISSING ({v})') for k, v in paths.items()]; \
	print(); \
	print('Offline targets always work. Online targets need every path above.')"

clean:
	rm -rf outputs/features data/fake analysis/variant_experiment_stub*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "Removed generated artifacts. Figures and .venv kept."
