.DEFAULT_GOAL := help
SHELL := /bin/bash
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Install dev + extras (where wheels exist)
	$(UV) sync --extra dev --extra embed

.PHONY: install-min
install-min: ## Install only core + dev (no torch — Intel macOS path)
	$(UV) sync --extra dev

.PHONY: fmt
fmt: ## Format with ruff
	$(UV) run ruff format src tests

.PHONY: lint
lint: ## Lint with ruff
	$(UV) run ruff check src tests

.PHONY: typecheck
typecheck: ## mypy --strict on src
	$(UV) run mypy src

.PHONY: check
check: lint typecheck ## Lint + typecheck

.PHONY: test
test: ## Unit tests
	$(UV) run pytest -m "not integration"

.PHONY: test-integration
test-integration: ## Integration tests (real Postgres)
	$(UV) run pytest -m integration

.PHONY: test-all
test-all: ## All tests
	$(UV) run pytest

.PHONY: index
index: ## Index a corpus into pgvector + the in-memory call graph
	$(UV) run atlas index --help

.PHONY: ask
ask: ## Run a single agent query end-to-end
	$(UV) run atlas ask --help

.PHONY: eval
eval: ## Run the eval harness against the golden set
	$(UV) run atlas eval --help

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
