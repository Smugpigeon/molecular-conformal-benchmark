# Per CLAUDE.md §5. Run `make help` to list all targets.
# Tabs are required for recipe lines (do not convert to spaces).

SHELL := /bin/bash
PYTHON := python
ENV_NAME := drug
DATA_DIR := data

.DEFAULT_GOAL := help

.PHONY: help setup setup-cpu lint type test test-fast \
        baseline-rf baseline-chemprop baseline-all \
        train sweep \
        data canonicalize \
        check-repro \
        clean-runs clean-cache

help:  ## Show this help
	@echo "Fudan AIDD — Makefile targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
setup:  ## Create conda env from env.yml (requires mamba)
	@command -v mamba >/dev/null 2>&1 || { echo "ERROR: mamba not found. Install with: conda install -n base -c conda-forge mamba"; exit 1; }
	mamba env create -f env.yml -n $(ENV_NAME)
	@echo ""
	@echo "Next: conda activate $(ENV_NAME)"

setup-cpu:  ## Lighter env without CUDA (for laptop dev)
	mamba env create -f env.yml -n $(ENV_NAME)-cpu
	@echo "Edit env.yml to remove pytorch-cuda if needed"

# -----------------------------------------------------------------------------
# Code quality (CLAUDE.md §6)
# -----------------------------------------------------------------------------
lint:  ## Ruff check + format
	ruff check src/ tests/ scripts/
	ruff format src/ tests/ scripts/

type:  ## Mypy type check
	mypy src/

test:  ## Pytest with coverage
	pytest --cov=src tests/

test-fast:  ## Pytest skipping slow + GPU tests
	pytest -x -m "not slow and not gpu" tests/

# -----------------------------------------------------------------------------
# Data (CLAUDE.md §7)
# -----------------------------------------------------------------------------
data: canonicalize binarize  ## Regenerate all processed data from data/raw

canonicalize:  ## SMILES canonicalize all 6 datasets (CLAUDE.md §7.4)
	$(PYTHON) -m src.data.preprocess

binarize:  ## Build optional classification variants (esol_cls/bace_cls/bbbp_cls)
	$(PYTHON) -m src.data.binarize

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
baseline-rf:  ## RF + Morgan FP baseline on all 6 datasets
	bash scripts/baseline_rf.sh

baseline-chemprop:  ## Chemprop v2 baseline on all 6 datasets
	bash scripts/baseline_chemprop.sh

baseline-all: baseline-rf baseline-chemprop  ## Run all baselines

train:  ## Train a single config (override with DATASET=gsht MODEL=rf)
	$(PYTHON) -m src.train dataset=$(or $(DATASET),gsht) model=$(or $(MODEL),rf)

sweep:  ## Hyperparameter sweep via wandb (configure configs/sweep_*.yaml first)
	@echo "Edit configs/sweep_<task>.yaml then run:"
	@echo "  wandb sweep configs/sweep_gsht.yaml"
	@echo "  wandb agent <sweep_id>"

# -----------------------------------------------------------------------------
# Pre-submission (CLAUDE.md §13)
# -----------------------------------------------------------------------------
check-repro:  ## JCIM reproducibility checklist
	bash scripts/check_repro.sh

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
clean-runs:  ## DESTRUCTIVE: delete all experiment outputs
	@echo "About to delete: runs/ wandb/ outputs/ multirun/"
	@read -p "Type YES to confirm: " confirm && [ "$$confirm" = "YES" ] || exit 1
	rm -rf runs/ wandb/ outputs/ multirun/ lightning_logs/

clean-cache:  ## Clean Python/test/type-checker caches (safe)
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
