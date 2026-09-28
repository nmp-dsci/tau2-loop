# tau2-loop

A policy-following customer-support agent for [τ²-bench](https://github.com/sierra-research/tau2-bench)
on four domains (airline, retail, telecom, banking), run **entirely on the Claude
subscription** through the Claude Agent SDK, no API key — the agent on the model its
version's `agent.yaml` names (Sonnet 5 on airline since v3, Haiku 4.5 elsewhere), the
user simulator and judge on Haiku 4.5 — with a scored, versioned, self-improving
optimisation loop: the same mechanics as [DABStep-loop](https://github.com/nmp-dsci/DABStep-loop).

- **Plan:** `.lavish/s00_tau2-loop-init-plan.html` — the discovery (splits, scoring,
  judge, tools, budget) and the milestones this build follows.
- **How it is built and why:** [`AGENTS.md`](./AGENTS.md). Visual brief: [`DESIGN.md`](./DESIGN.md).
- **Findings from the end-to-end build:** `.lavish/s01_build-findings.html` (written at M4b).

## What it does

```
make smoke                     # v0 on tau2's mock domain: proves the adapter on all three roles
make eval DOMAIN=airline       # a version on its domain's train half → runs/<ts>_airline_v0_train/
make loop DOMAIN=airline       # champion → failures → one Sonnet optimiser session → challenger → gate → test → ledger
make fork DOMAIN=airline MODEL=sonnet   # the champion's prompt on another model: a model swap, promoted by hand
make agent-service             # the agent's model call as a container on :8091; AGENT_SERVICE_URL routes a run to it
make viewer                    # the run viewer on :8081 — eight tabs, see AGENTS.md §4b
```

tau2-bench v1.0.1 is a submodule pinned at `2174a60` and never edited. Our
code is a registry factory (`tau2_loop`) that turns `agents/<domain>/vN/system.md`
and `helper.py` into the next reply, and a litellm provider under the prefix
`claude-sdk/` that carries every model call tau2 makes — the agent's, the user
simulator's, the retail NL judge's — to the subscription. Tool calls travel as a
JSON contract in the prompt and come back as `tool_calls`, so tau2's orchestrator
executes them unchanged.

The agent's model call is a sealed core (`llm/core.py`: a prompt and tool
schemas in, one reply out, the Agent SDK and nothing else). The SDK child has no
tools, no MCP servers, no settings and an allow-listed environment. The same
core runs in-process and behind `POST /v1/chat/completions`
(`llm/service.py`, `Dockerfile.agent`, a bearer token). With
`AGENT_SERVICE_URL` set, the harness reaches the agent only over HTTP, through
litellm's `openai/` provider, so where the agent runs is configuration.

Each domain's split is half its public base set for train and half for test
(seed 300, committed under `data/splits/`; the earlier 20 / 20 cut sits inside
it on the same sides). Each domain also has its own agent versions, ledger and
registry. The gate pairs the two runs by task and promotes when the challenger
fixes at least one task and breaks none, or when a one-sided exact sign test on
the tasks that changed gives p < 0.05 (McNemar at one trial; a task's pass
fraction over trials otherwise). The challenger then runs the test split once
whatever the verdict. The champion-vs-challenger comparison on test is recorded
beside the verdict and never decides it.

## Status

| Milestone | State |
|---|---|
| M0 scaffold, splits, viewer shell | done |
| M1 subscription adapter on mock | done — 8/10 on mock with all three roles live |
| M2 v0 baselines on train, all four domains | done — gate re-scores 80/80 |
| M3 loop cycles | done — airline ×2, retail, telecom; banking not run (subscription window) |
| M4 holdout · M4b findings page | done — `.lavish/s01_build-findings.html` |
| s07 Sonnet agent through a sealed service, split v2, the gate with trials | built; airline done: v3 (Sonnet) champion 18/25 train · 21/25 test, v4 held. Retail, telecom, banking next (`.lavish/s07_next-challenger-plan.html`) |
| M5 keyless demo image on App Runner | parked — bootstrap role, ECR repo and image (`5518ad7`) are in AWS; the service is blocked by the account's 2-per-region App Runner cap (both regions full). Resume: lift the quota or free a slot, then `terraform apply` in `infra/terraform/demo` |

## Results

### Split v1 (20 / 20), Haiku 4.5 in every role

Haiku 4.5 for agent, user simulator and judge; one trial; concurrency 3; seed 300;
lean harness. Train is the 20-task split the optimiser sees; test is the 20 it
never sees, run once on promotion. p is the one-sided exact McNemar test on the
paired train tasks; the verdicts were made at p < 0.05 alone. The dominance rule
(fixed ≥ 1, broke 0) came later, so airline's v2 (4 fixed / 0 broke) would promote
under today's gate; its recorded verdict is kept as made.

| Domain | v0 train | Challenger train | Fixed / broke | p | Verdict | Champion test |
|---|---|---|---|---|---|---|
| airline | 12/20 `20260915T075151Z_airline_v0_train` | v1 15/20 `20260915T124751Z_airline_v1_train` · v2 16/20 `20260915T132148Z_airline_v2_train` | 4/1 · 4/0 | 0.188 · 0.062 | hold ×2, v0 champion | — |
| retail | 14/20 `20260915T080153Z_retail_v0_train` | v1 16/20 `20260915T172708Z_retail_v1_train` | 3/1 | 0.312 | hold, v0 champion | — |
| telecom | 14/20 `20260915T081700Z_telecom_v0_train` | v1 19/20 `20260915T174430Z_telecom_v1_train` | 5/0 | 0.031 | **promote**, v1 champion | **19/20** `20260915T181840Z_telecom_v1_test` |
| banking_knowledge | 4/20 `20260915T121036Z_banking_knowledge_v0_train` | — | — | — | no cycle yet | — |

### Split v2 (25 / 25), airline — Sonnet 5 agent through the service

Agent `claude-sonnet-5` at medium effort, reached over HTTP (`agent_route:
service:127.0.0.1:8091`); user simulator `claude-haiku-4-5` at medium; one trial;
concurrency 3; code `228e3d6`. v3 is v0's prompt on Sonnet, promoted by hand as
a model swap. v4 is loop cycle 3: the optimiser read v3's 7 train failures and
the ledger, including v1's and v2's held surfaces. The test columns are
reported only; the gate reads train.

| Version | Train | Test | Gate on train | Test, v3 → v4 |
|---|---|---|---|---|
| v3 · v0's prompt, Sonnet 5 | **18/25** `20260928T060029Z_airline_v3_train` | **21/25** `20260928T073602Z_airline_v3_test` | champion by fiat (`model swap`) | — |
| v4 · cycle 3 | 20/25 `20260928T075613Z_airline_v4_train` | 18/25 `20260928T081324Z_airline_v4_test` | hold: fixed 6, broke 4, p = 0.377 | 21 → 18: fixed 1, broke 4 |

Every run folder under `runs/` holds `run.json`, `results.jsonl`, one trace per
conversation and the agent version it ran; `tau2loop gate` replays them through
tau2's evaluators. Per-domain ledgers are in `loop/<domain>/ledger.jsonl`.

Two harness findings worth knowing before reading the traces: the Claude CLI
tells every session the real date and the account's email, and Haiku used both
inside the simulation (airline v0 refused "already flown" 2024 flights; retail
v0 looked customers up by the account address). Replies and run folders are now
redacted (`tau2_loop.llm.redact`); the date is handled in the airline prompts.
Details and next steps: `.lavish/s01_build-findings.html`.

## Setup

```
make setup          # submodule at the pin, uv sync, npm ci
cp .env.example .env
claude login        # the subscription; no ANTHROPIC_API_KEY anywhere
make platform-up    # central MLflow + Postgres (make -C ../nmp-central-ai up) → http://localhost:5000
make db-migrate      # app-state schema on the central Postgres (database `tau2`, idempotent)
make test · make lint
```
