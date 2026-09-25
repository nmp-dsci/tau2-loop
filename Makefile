# tau2-loop — every target is a thin wrapper over `uv run tau2loop …`.
.DEFAULT_GOAL := help
DOMAIN ?= airline
AGENT ?= v0
SPLIT ?= train
TRIALS ?= 1
CONCURRENCY ?= 3
CYCLES ?= 1
OPTIMISER ?= sonnet
MLFLOW_TRACKING_URI ?= http://localhost:5000
API_PORT ?= 8081

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## submodule at the pin, python deps (uv) and frontend deps (npm)
	git submodule update --init --depth 1
	uv sync
	cd frontend && npm ci

splits: ## cut the 20 / 20 train / test split per domain (seed 300) → data/splits, data/tasks
	uv run tau2loop splits

platform-up: ## start the central MLflow (nmp-central-ai: postgres + minio + mlflow on :5000)
	$(MAKE) -C ../nmp-central-ai up

platform-status: ## preflight: the central MLflow must answer /health (runs before every tracked eval)
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## the adapter on the mock domain (10 tasks): agent, user simulator and judge on the subscription
	uv run tau2loop smoke --concurrency $(CONCURRENCY)

eval: platform-status ## run AGENT on DOMAIN's SPLIT (TRIALS=, CONCURRENCY=)
	uv run tau2loop eval --domain $(DOMAIN) --agent $(AGENT) --split $(SPLIT) --trials $(TRIALS) --concurrency $(CONCURRENCY)

baselines: platform-status ## v0 on the train split of all four domains, one trial each
	for d in airline retail telecom banking_knowledge; do uv run tau2loop eval --domain $$d --agent v0 --split train --concurrency $(CONCURRENCY) || exit 1; done

score: ## summarise RUN=<run id>
	uv run tau2loop score $(RUN)

rescore: ## replay RUN=<run id> through tau2's evaluators offline and compare verdicts
	uv run tau2loop rescore $(RUN)

compare: ## gate CHAMPION=<run> CHALLENGER=<run>
	uv run tau2loop compare $(CHAMPION) $(CHALLENGER)

register: ## register RUN=<run id> as challenger
	uv run tau2loop register $(RUN)

promote: ## promote RUN=<run id> to champion of its domain
	uv run tau2loop promote $(RUN)

loop: platform-status ## the error loop on DOMAIN: CYCLES=1 of eval → diagnose → new version → gate (OPTIMISER=sonnet)
	uv run tau2loop loop --domain $(DOMAIN) --cycles $(CYCLES) --optimiser $(OPTIMISER) --concurrency $(CONCURRENCY)

ledger: ## print DOMAIN's loop ledger
	uv run tau2loop ledger --domain $(DOMAIN)

snapshot: platform-status ## export MLflow to loop/mlflow_snapshot.json
	uv run tau2loop snapshot

leaderboard: ## ingest tau2-bench's published submissions into data/index/leaderboard.json
	uv run tau2loop leaderboard

db-migrate: ## apply infra/roles.sql to the central Postgres (database `tau2`, idempotent)
	uv run python -c "from tau2_loop.data import pg; pg.migrate(); print('tau2_loop schema ready')"

db-smoke: ## zero-LLM proof this project can reach its database and read its own tables
	uv run python -c "from tau2_loop.data import pg; print('tables:', pg.tables()); print('read-only role ok:', bool(pg.connect_ro()))"

gate: ## the CI gate: every champion re-scores offline to what its registry says
	uv run tau2loop gate

dev: ## run the API on :$(API_PORT) (frontend: cd frontend && npm run dev)
	uv run tau2loop serve --port $(API_PORT)

viewer: ## build the frontend and serve it with the API on :$(API_PORT)
	cd frontend && npm run build
	uv run tau2loop serve --port $(API_PORT)

demo-up: ## build and run the demo image locally (no keys, DEMO_MODE=1)
	docker build -t tau2loop-demo . && docker run --rm -p 8081:8080 tau2loop-demo

test: ## pytest (offline)
	uv run pytest -q

lint: ## ruff + mypy (+ frontend design lint when node_modules exist)
	uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy
	@if [ -d frontend/node_modules ]; then cd frontend && npm run lint:design; fi

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

.PHONY: help setup splits platform-up platform-status smoke eval baselines score rescore compare register promote loop ledger snapshot gate dev viewer demo-up test lint fmt
