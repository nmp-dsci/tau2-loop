# tau2-loop

A policy-following customer-support agent for [τ²-bench](https://github.com/sierra-research/tau2-bench)
on four domains (airline, retail, telecom, banking), run **entirely on the Claude
subscription** — agent, user simulator and judge all Haiku 4.5 through the Claude
Agent SDK, no API key — with a scored, versioned, self-improving optimisation
loop: the same mechanics as [DABStep-loop](https://github.com/nmp-dsci/DABStep-loop).

- **Plan:** `.lavish/s00_tau2-loop-init-plan.html` — the discovery (splits, scoring,
  judge, tools, budget) and the milestones this build follows.
- **How it is built and why:** [`AGENTS.md`](./AGENTS.md). Visual brief: [`DESIGN.md`](./DESIGN.md).
- **Findings from the end-to-end build:** `.lavish/s01_build-findings.html` (written at M4b).

## What it does

```
make smoke                     # v0 on tau2's mock domain: proves the adapter on all three roles
make eval DOMAIN=airline       # v0 on the 20-task train split → runs/<ts>_airline_v0_train/
make loop DOMAIN=airline       # champion → failures → one Sonnet optimiser session → challenger → gate → ledger
make viewer                    # the run viewer on :8081 (data, tasks, agents, runs, gate, loop, evolution)
```

tau2-bench v1.0.1 is a submodule pinned at `2174a60` and never edited. Our
code is a registry factory (`tau2_loop`) that turns `agents/<domain>/vN/system.md`
and `helper.py` into the next reply, and a litellm provider under the prefix
`claude-sdk/` that carries every model call tau2 makes — the agent's, the user
simulator's, the retail NL judge's — to the subscription. Tool calls travel as a
JSON contract in the prompt and come back as `tool_calls`, so tau2's orchestrator
executes them unchanged.

Each domain has its own random 20 train / 20 test split (seed 300, committed
under `data/splits/`), its own agent versions, its own ledger and registry. The
gate is a one-sided exact McNemar test on the paired train tasks (promote at
p < 0.05); a promotion runs the test split once, for the record.

## Status

| Milestone | State |
|---|---|
| M0 scaffold, splits, viewer shell | done |
| M1 subscription adapter on mock | done — 8/10 on mock with all three roles live |
| M2 v0 baselines on train, all four domains | done — gate re-scores 80/80 |
| M3 loop cycles | done — airline ×2, retail, telecom; banking not run (subscription window) |
| M4 holdout · M4b findings page | done — `.lavish/s01_build-findings.html` |
| M5 keyless demo image on App Runner | parked — bootstrap role, ECR repo and image (`5518ad7`) are in AWS; the service is blocked by the account's 2-per-region App Runner cap (both regions full). Resume: lift the quota or free a slot, then `terraform apply` in `infra/terraform/demo` |

## Results

Haiku 4.5 for agent, user simulator and judge; one trial; concurrency 3; seed 300;
lean harness. Train is the 20-task split the optimiser sees; test is the 20 it
never sees, run once on promotion. p is the one-sided exact McNemar test on the
paired train tasks; promote at p < 0.05.

| Domain | v0 train | Challenger train | Fixed / broke | p | Verdict | Champion test |
|---|---|---|---|---|---|---|
| airline | 12/20 `20260915T075151Z_airline_v0_train` | v1 15/20 `20260915T124751Z_airline_v1_train` · v2 16/20 `20260915T132148Z_airline_v2_train` | 4/1 · 4/0 | 0.188 · 0.062 | hold ×2, v0 champion | — |
| retail | 14/20 `20260915T080153Z_retail_v0_train` | v1 16/20 `20260915T172708Z_retail_v1_train` | 3/1 | 0.312 | hold, v0 champion | — |
| telecom | 14/20 `20260915T081700Z_telecom_v0_train` | v1 19/20 `20260915T174430Z_telecom_v1_train` | 5/0 | 0.031 | **promote**, v1 champion | **19/20** `20260915T181840Z_telecom_v1_test` |
| banking_knowledge | 4/20 `20260915T121036Z_banking_knowledge_v0_train` | — | — | — | no cycle yet | — |

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
make platform-up    # central MLflow (make -C ../nmp-central-ai up) → http://localhost:5000
make test · make lint
```
