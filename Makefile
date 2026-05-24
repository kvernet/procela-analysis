# Default target
.DEFAULT_GOAL := help

.PHONY: help \
        install dev-install docs-install notebooks-install \
        requirements test lint format isort type-check \
        pre-commit clean ci

help:  ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------
# Installation
# --------------------------------------------------------------------

install:  ## Install package in editable mode
	pip install --upgrade pip
	pip install -e .

dev-install: install  ## Install development dependencies
	pip install -e ".[dev]"

docs-install: install  ## Install documentation dependencies
	pip install -e ".[docs]"

notebooks-install: install  ## Install notebook dependencies
	pip install -e ".[notebooks]"

# --------------------------------------------------------------------
# Requirements generation (from pyproject.toml)
# --------------------------------------------------------------------

requirements:  ## Generate requirements files from pyproject.toml
	mkdir -p requirements
	pip install --upgrade pip pip-tools
	pip-compile --strip-extras \
		--output-file=requirements/base.txt pyproject.toml
	pip-compile --strip-extras \
		--extra=dev --output-file=requirements/dev.txt pyproject.toml
	pip-compile --strip-extras \
		--extra=docs --output-file=requirements/docs.txt pyproject.toml
	pip-compile --strip-extras \
		--extra=notebooks --output-file=requirements/notebooks.txt pyproject.toml
	@echo "Requirements files generated with --strip-extras (future default)"

# --------------------------------------------------------------------
# Quality checks (no mutation)
# --------------------------------------------------------------------

lint:  ## Run linters (no code modification)
	ruff check procela_analysis tests
	pydocstyle procela_analysis

type-check:  ## Run static type checking
	mypy procela_analysis

test:  ## Run tests
	pytest

# --------------------------------------------------------------------
# Formatting (code mutation)
# --------------------------------------------------------------------

format: isort  ## Format code
	black procela_analysis tests
	ruff check --fix procela_analysis tests

isort:  ## Sort imports
	isort procela_analysis tests

# --------------------------------------------------------------------
# Tooling
# --------------------------------------------------------------------

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

# --------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------

clean:  ## Clean build and cache artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs: docs-install ## Build documentation
	mkdocs build --clean

# --------------------------------------------------------------------
# CI
# --------------------------------------------------------------------

ci: isort lint type-check test  ## Run CI checks (no mutation)


ci-act:  ## Run CI checks with act and no upload
	act -W .github/workflows/ci.yaml -j ci --artifact-server-path \
		/tmp/act-artifacts --env ACT_SKIP_UPLOAD=true
