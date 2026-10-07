# tau2-loop — every target is a thin wrapper over `uv run tau2loop …`.
.DEFAULT_GOAL := help
DOMAIN ?= airline
# AGENT unset: eval runs the domain's v0, or its oldest version where v0 was retired (banking: v1); smoke runs v0
AGENT ?=
SPLIT ?= train
TRIALS ?= 1
CONCURRENCY ?= 3
CYCLES ?= 1
# OPTIMISER and MODE unset: loop and ab take the domain's optimiser profile (optimisers/DOMAIN/profile.yaml)
OPTIMISER ?=
MODE ?=
MLFLOW_TRACKING_URI ?= http://localhost:5000
API_PORT ?= 8081
AGENT_PORT ?= 8091
KIND ?= gate
# USER_MODEL unset: the customer is Haiku (eval/runner.USER_MODEL); another makes the run an experiment the loop never reuses
USER_MODEL ?=

help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup: ## submodule at the pin, python deps (uv) and frontend deps (npm)
	git submodule update --init --depth 1
	uv sync
	cd frontend && npm ci

splits: ## cut each domain's base set in half, train / test (seed 300) → data/splits, data/tasks
	uv run tau2loop splits

documents: ## index banking's knowledge base (title and size per document) → data/tasks, for Evals
	uv run tau2loop documents

fork: ## a new DOMAIN version from FROM= (default the champion) on MODEL=, EFFORT=, RETRIEVAL=, TOOL_MODE=, IDENTITY_NOTE=true, PARALLEL_CALLS=true or WORKFLOWS=rN, by hand
	uv run tau2loop fork --domain $(DOMAIN) $(if $(FROM),--from $(FROM)) $(if $(MODEL),--model $(MODEL)) $(if $(EFFORT),--effort $(EFFORT)) $(if $(RETRIEVAL),--retrieval $(RETRIEVAL)) $(if $(TOOL_MODE),--tool-mode $(TOOL_MODE)) $(if $(filter true,$(IDENTITY_NOTE)),--identity-note) $(if $(filter true,$(PARALLEL_CALLS)),--parallel-calls) $(if $(WORKFLOWS),--workflows $(WORKFLOWS))

rag-build: ## s20/s21: build a merging workflow_rag version's library, RAG= (default r2): r2 one at a time in id order, r3 with WORKERS= (default its own 4); resumable (TASKS= some, LIMIT=)
	uv run tau2loop rag-build --domain banking_knowledge --rag $(or $(RAG),r2) $(foreach t,$(TASKS),--task $(t)) $(if $(LIMIT),--limit $(LIMIT)) $(if $(WORKERS),--workers $(WORKERS))

rag-locks: ## s21: who is updating which workflow in RAG='s library (default r3): held and waiting locks
	uv run tau2loop rag-locks --domain banking_knowledge --rag $(or $(RAG),r3)

rag-replay: ## s21: RAG='s library (default r3) as of commit AT= (default the latest), as JSON
	uv run tau2loop rag-replay --domain banking_knowledge --rag $(or $(RAG),r3) $(if $(AT),--at $(AT))

rag-rubric: ## s20: the rubric's checks over RAG='s whole library (default r2), no model
	uv run tau2loop rag-rubric --domain banking_knowledge --rag $(or $(RAG),r2)

agent-service: ## the agent's model call as a container on 127.0.0.1:$(AGENT_PORT); runs reach it with AGENT_SERVICE_URL
	uv run tau2loop agent-service --docker --port $(AGENT_PORT)

platform-up: ## start the central MLflow (nmp-central-ai: postgres + minio + mlflow on :5000)
	$(MAKE) -C ../nmp-central-ai up

platform-status: ## preflight: the central MLflow must answer /health (runs before every tracked eval)
	@curl -fsS $(MLFLOW_TRACKING_URI)/health >/dev/null || (echo "central MLflow down at $(MLFLOW_TRACKING_URI): run make platform-up"; exit 1)

smoke: platform-status ## the adapter on the mock domain (10 tasks, AGENT=v0): agent, user simulator and judge on the subscription
	uv run tau2loop smoke --agent $(or $(AGENT),v0) --concurrency $(CONCURRENCY)

eval: platform-status ## run AGENT on DOMAIN's SPLIT (TRIALS=, CONCURRENCY=; TASKS="task_1 task_2" for some of its tasks; USER_MODEL= the customer)
	uv run tau2loop eval --domain $(DOMAIN) $(if $(AGENT),--agent $(AGENT)) --split $(SPLIT) --trials $(TRIALS) --concurrency $(CONCURRENCY) $(foreach t,$(TASKS),--task $(t)) $(if $(USER_MODEL),--user-model $(USER_MODEL))

baselines: platform-status ## each domain's base version on its train split, one trial each: v0, or the oldest version where v0 was retired (banking, v1)
	for d in airline retail telecom banking_knowledge; do uv run tau2loop eval --domain $$d --split train --concurrency $(CONCURRENCY) || exit 1; done

score: ## summarise RUN=<run id>
	uv run tau2loop score $(RUN)

extend: platform-status ## play only the tasks RUN=<run id>'s split has gained since, joined to it as one run of the split (a champion's train run stays its record)
	uv run tau2loop extend $(RUN) --concurrency $(CONCURRENCY)

rescore: ## replay RUN=<run id> through tau2's evaluators offline and compare verdicts
	uv run tau2loop rescore $(RUN)

compare: ## gate CHAMPION=<run> CHALLENGER=<run>
	uv run tau2loop compare $(CHAMPION) $(CHALLENGER)

register: ## register RUN=<run id> as challenger
	uv run tau2loop register $(RUN)

promote: ## promote RUN=<run id> to champion of its domain (KIND="model swap" for a fork by fiat)
	uv run tau2loop promote $(RUN) --kind "$(KIND)"

loop: platform-status ## the error loop on DOMAIN: CYCLES=1 of eval → diagnose → new version → gate → test under DOMAIN's optimiser profile (AGENT= to start from a held version; OPTIMISER=, MODE=classic|routing override the profile for this run; TRIALS=1)
	uv run tau2loop loop --domain $(DOMAIN) --cycles $(CYCLES) $(if $(AGENT),--agent $(AGENT)) $(if $(OPTIMISER),--optimiser $(OPTIMISER)) --concurrency $(CONCURRENCY) --trials $(TRIALS) $(if $(MODE),--mode $(MODE))
optimise: ## one optimiser session from FROM= (default the champion) on DOMAIN's train failures, writing the next version and scoring nothing; MODEL=, EFFORT=, PARALLEL_CALLS=true set the new version's agent.yaml; OPTIMISER=, MODE= override the profile
	uv run tau2loop optimise --domain $(DOMAIN) $(if $(FROM),--from $(FROM)) $(if $(OPTIMISER),--optimiser $(OPTIMISER)) $(if $(MODE),--mode $(MODE)) $(if $(MODEL),--agent-model $(MODEL)) $(if $(EFFORT),--agent-effort $(EFFORT)) $(if $(filter true,$(PARALLEL_CALLS)),--parallel-calls)
ab: platform-status ## two challengers from DOMAIN's champion on the same failures, the classic and the routing optimiser, under DOMAIN's optimiser profile (OPTIMISER= overrides its model); both gated and tested, at most one crowned
	uv run tau2loop ab --domain $(DOMAIN) $(if $(OPTIMISER),--optimiser $(OPTIMISER)) --concurrency $(CONCURRENCY) --trials $(TRIALS)

challenge: platform-status ## score AGENT= (a fork) against DOMAIN's champion through the loop's gate, ledger and test report; no optimiser (TRIALS=1, NO_TEST=1)
	@if [ "$(origin AGENT)" = "file" ]; then echo "make challenge needs AGENT=vN: the version to score against the champion"; exit 1; fi
	uv run tau2loop challenge --domain $(DOMAIN) --agent $(AGENT) --concurrency $(CONCURRENCY) --trials $(TRIALS) $(if $(NO_TEST),--no-test)

judge-labels: ## J0 (s11): label every checkpoint of DOMAIN's scored train conversations from gold → data/judge/DOMAIN.json (no model)
	uv run tau2loop judge-labels --domain $(DOMAIN)

judge-gold: platform-status ## J1 (s11): a golden answer per DOMAIN train conversation, Opus 5.5 high, sees gold → data/judge/DOMAIN_gold.jsonl (resumable; CONCURRENCY=)
	uv run tau2loop judge-gold --domain $(DOMAIN) --concurrency $(CONCURRENCY)

judge-gold-freeze: ## J1 (s11): copy the person's golden-answer checks (Review › golden answers, Postgres) into data/judge/DOMAIN_gold.jsonl
	uv run tau2loop judge-gold-freeze --domain $(DOMAIN)

judge-probe: ## J2 (s11): one call proving the SDK's output_format holds under the sealed core → data/judge/probe.json
	uv run tau2loop judge-probe

JUDGE ?= j1
judge-replay: platform-status ## J2 (s11): replay JUDGE= on every DOMAIN train checkpoint, scored on the golden answers → judge_runs/ (CONCURRENCY=)
	uv run tau2loop judge-replay --domain $(DOMAIN) --judge $(JUDGE) --split $(SPLIT) --concurrency $(CONCURRENCY)

judge-loop: platform-status ## J3 (s11): CYCLES= of the plan judge's loop on DOMAIN: an optimiser adds lessons from the read half, the gate half decides → judges/DOMAIN/plan/ (CONCURRENCY=)
	uv run tau2loop judge-loop --domain $(DOMAIN) --cycles $(CYCLES) --concurrency $(CONCURRENCY)

judge-score: ## J2 (s11): rewrite REPLAY=<judge_runs id>'s summary with today's scorer and gold (WHY="…"); verdicts never change
	uv run tau2loop judge-score $(REPLAY) $(if $(WHY),--why "$(WHY)")

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

dev: ## run the API on :$(API_PORT), reloading on code changes (frontend: cd frontend && npm run dev)
	uv run tau2loop serve --port $(API_PORT) --reload

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

.PHONY: rag-build rag-locks rag-replay rag-rubric judge-labels judge-gold judge-gold-freeze judge-probe judge-replay judge-loop judge-score help setup splits documents fork agent-service platform-up platform-status smoke eval baselines score rescore compare register promote loop challenge ledger snapshot gate dev viewer demo-up test lint fmt ab
