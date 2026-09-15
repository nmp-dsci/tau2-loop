# tau2-loop — every target is a thin wrapper over `uv run tau2loop …`.
.DEFAULT_GOAL := help
DOMAIN ?= airline
AGENT ?= v0
SPLIT ?= train
TRIALS ?= 1
CONCURRENCY ?= 3
CYCLES ?= 1
OPTIMISER ?= sonnet
MLFLOW_PORT ?= 5601
API_PORT ?= 8081

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## submodule at the pin, python deps (uv) and frontend deps (npm)
	git submodule update --init --depth 1
	uv sync
	cd frontend && npm ci

splits: ## cut the 20 / 20 train / test split per domain (seed 300) → data/splits, data/tasks
	uv run tau2loop splits

mlflow-up: ## start the self-hosted MLflow tracking server on :$(MLFLOW_PORT)
	mkdir -p .mlflow
	uv run mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT) \
	  --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artifacts

smoke: ## the adapter on the mock domain (10 tasks): agent, user simulator and judge on the subscription
	uv run tau2loop smoke --concurrency $(CONCURRENCY)

eval: ## run AGENT on DOMAIN's SPLIT (TRIALS=, CONCURRENCY=)
	uv run tau2loop eval --domain $(DOMAIN) --agent $(AGENT) --split $(SPLIT) --trials $(TRIALS) --concurrency $(CONCURRENCY)

baselines: ## v0 on the train split of all four domains, one trial each
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

loop: ## the error loop on DOMAIN: CYCLES=1 of eval → diagnose → new version → gate (OPTIMISER=sonnet)
	uv run tau2loop loop --domain $(DOMAIN) --cycles $(CYCLES) --optimiser $(OPTIMISER) --concurrency $(CONCURRENCY)

ledger: ## print DOMAIN's loop ledger
	uv run tau2loop ledger --domain $(DOMAIN)

snapshot: ## export MLflow to loop/mlflow_snapshot.json
	uv run tau2loop snapshot

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
	@test -d frontend/node_modules && (cd frontend && npm run lint:design) || true

fmt: ## ruff format + fix
	uv run ruff format src tests && uv run ruff check --fix src tests

.PHONY: help setup splits mlflow-up smoke eval baselines score rescore compare register promote loop ledger snapshot gate dev viewer demo-up test lint fmt
