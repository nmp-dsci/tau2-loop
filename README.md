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
| M2 v0 baselines on train, all four domains | running |
| M3 loop cycles | — |
| M4 holdout · M4b findings page | — |
| M5 keyless demo image on App Runner | image builds and runs locally; deploy needs the author's one-off bootstrap |

Results tables land here at M4, every number with its denominator and its run folder.

## Setup

```
make setup          # submodule at the pin, uv sync, npm ci
cp .env.example .env
claude login        # the subscription; no ANTHROPIC_API_KEY anywhere
make mlflow-up      # http://127.0.0.1:5601
make test · make lint
```
