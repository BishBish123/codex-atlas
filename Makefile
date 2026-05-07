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
	$(UV) run atlas eval

.PHONY: calibrate
calibrate: ## Run Cohen's kappa calibration for the LLM-as-judge
	$(UV) run atlas calibrate --labels evals/calibration.csv --out evals/CALIBRATION.md

.PHONY: pg-up
pg-up: ## Start local Postgres + pgvector (port 5433; ATLAS_PG_DSN=postgresql://atlas:atlas@localhost:5433/atlas)
	docker compose -f docker-compose.pgvector.yml up -d --wait

.PHONY: pg-down
pg-down: ## Stop and remove the local pgvector container + data volume
	docker compose -f docker-compose.pgvector.yml down -v

.PHONY: neo4j-up
neo4j-up: ## Start local Neo4j Community (bolt 7687, browser 7474; password neo4j-dev)
	docker compose -f docker-compose.neo4j.yml up -d --wait

.PHONY: neo4j-down
neo4j-down: ## Stop and remove the local Neo4j container + data volumes
	docker compose -f docker-compose.neo4j.yml down -v

.PHONY: docker-build
docker-build: ## Build the Docker image locally
	docker build --tag codex-atlas:local .

.PHONY: docker-run
docker-run: ## Run the container locally (HTTP transport on port 8000)
	docker run --rm \
		--publish 8000:8000 \
		--env ATLAS_STORE=memory \
		--name codex-atlas-local \
		codex-atlas:local

.PHONY: deploy
deploy: ## Deploy to Fly.io (requires fly CLI + auth; set FLY_APP_NAME or --app)
	./scripts/deploy.sh

.PHONY: cli-help
cli-help: ## Regenerate docs/CLI.md from atlas --help output
	$(UV) run python scripts/generate_cli_help.py

.PHONY: demo-cast
demo-cast: ## Generate assets/demo.cast from real CLI output
	$(UV) run python scripts/generate_demo_cast.py

.PHONY: clean
clean: ## Wipe caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +
