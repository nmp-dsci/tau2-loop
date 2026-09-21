# AGENTS.md — tau2-loop

The source of truth for how this project is built and why. `CLAUDE.md` points
here. `DESIGN.md` governs anything visual. The plan this was built to is
`.lavish/s00_tau2-loop-init-plan.html`; the findings page at the end of the
build is `.lavish/s01_build-findings.html`.

## 1 · What it is — a Haiku policy agent on τ²-bench, and the loop that improves it

[τ²-bench](https://github.com/sierra-research/tau2-bench) (v1.0.1, pinned at
`2174a60` as the `vendor/tau2-bench` submodule) simulates a customer-support
conversation: a user simulator plays a scripted customer, the agent follows a
policy document and calls the domain's tools, and the evaluator scores the
final database, the actions taken, what was said, and (retail) an LLM judge's
reading of natural-language assertions. Four domains are scored here:
airline, retail, telecom, banking_knowledge. This project:

1. runs our agent (Haiku 4.5, `agents/<domain>/vN/{system.md, helper.py}`)
   through tau2's own harness — orchestrator, tools, user simulator, evaluator
   all unmodified — with **every model call on the Claude subscription** via a
   litellm provider over the Claude Agent SDK (`src/tau2_loop/llm/`);
2. scores it on our own committed 20 train / 20 test split per domain
   (`data/splits/`, seed 300) and keeps every run as a folder under `runs/`,
   indexed in the central MLflow (nmp-central-ai);
3. runs an **error loop per domain**: one optimiser session (Sonnet) reads
   every failed conversation and the ledger of earlier attempts, writes
   `agents/<domain>/v(N+1)/`, and a gate promotes it when a one-sided McNemar
   test on the paired train tasks clears p < 0.05; a promotion runs the test
   split once, for the record;
4. serves a read-only viewer (React + FastAPI) of the data, the architecture,
   the runs, the gate and the ledgers, deployable as a keyless demo image.

The build's goal is the working pipeline, not a score. There is no
leaderboard submission in this build (that is M6, a separate decision).

## 2 · Decisions, and the reasons

| Decision | Choice | Why |
|---|---|---|
| Split | our own random 20/20 per domain from tau2's public `base` set, seed 300, committed | nothing is held out upstream; tau2's train/test are fixed lists and banking has none |
| Model under test | `claude-haiku-4-5` in every role: agent, user simulator, NL judge | the only billing path is the subscription; a Claude judge makes retail's score `custom` relative to the board |
| Optimiser | `claude-sonnet-5`, effort medium, one session per cycle, may write two files | must read ~20 transcripts and the policy in one context |
| Tool calls | a JSON reply contract in the prompt (`tool_mode: json`), parsed back into `tool_calls` | the SDK cannot return a native tool call without executing it; the contract is the same either-message-or-tools rule tau2 enforces |
| Sampling | the CLI default; tau2's `temperature: 0.0` does not apply | the SDK exposes no temperature; recorded as `sampling: cli-default` on every run |
| Trials | one per cycle for the gate; `pass^k` over trials when `TRIALS>1` | a cycle is 20 conversations; the board's ≥4 trials is a submission requirement, not a loop one |
| Tracking | MLflow 3 on the central platform (`nmp-central-ai`, http://localhost:5000; `MLFLOW_TRACKING_URI` overrides); `loop/<domain>/registry.json` is the truth | one server for the portfolio (DABStep-loop logs to the same one, experiment `dabstep-loop`); the run folder is the record, MLflow the index; the old sqlite store under `.mlflow/` is an archive |
| Billing | `require_live()` refuses a key alongside `BILLING=subscription`, scrubs a key tau2's dotenv search injects from `~/.env`, blanks the key in the SDK child | tau2's `utils.py` calls `load_dotenv()` with a directory search on import |
| Deploy | DABStep-loop's pattern: ECR + App Runner, OIDC role, `workflow_run` after CI, `DEMO_MODE=1` in the Dockerfile | keyless by construction |
| Frontend | React 18 + Vite + TS, plain CSS on `tokens.css` from DESIGN.md | the Field Guide brief; no Tailwind/DaisyUI |

## 3 · Layout

```
vendor/tau2-bench/        τ³-bench at 2174a60 (submodule): harness, domains, data/tau2/, evaluator
data/splits/<domain>.json 20 train / 20 test ids, seed 300, base_n, reserve_n     committed
data/tasks/<domain>.json  the forty tasks (tau2's dump), policy, tool list       committed, for the viewer
agents/<domain>/vN/       system.md ({policy} slot) · helper.py (hooks) · agent.yaml (frozen) · diagnosis.json
runs/<ts>_<domain>_<vN>_<split>/  run.json · results.jsonl · traces/<task>.json · tau2_results.json · agent/
loop/<domain>/            ledger.jsonl · registry.json;  loop/mlflow_snapshot.json for the demo
src/tau2_loop/
  config.py               paths, Settings (boots keyless), DOMAINS, split seed
  llm/                    __init__ (models, billing) · sdk_provider (litellm CustomLLM → Agent SDK) · prompting (contract)
  agent/                  versions (per domain, fingerprint) · factory (LoopAgent, registered as "tau2_loop")
  data/splits.py          cut, extract, read
  eval/                   runner (tau2 run_tasks → run folder) · results · compare (McNemar) · rescore (offline replay)
  loop/                   run (cycle) · optimiser (Sonnet session, hooks) · ledger
  tracking/               registry · mlflow_log · snapshot · gate (CI)
  serving/app.py          FastAPI + SPA
frontend/                 Vite + React; src/tokens.css verbatim from DESIGN.md; scripts/design_lint.mjs
.github/workflows/        ci.yml · deploy-aws.yml
```

## 4 · Commands

```
make setup                submodule + uv sync + npm ci
make splits               cut the splits and task extracts (only when seed/size change)
make platform-up          central MLflow (make -C ../nmp-central-ai up) → http://localhost:5000
make platform-status      preflight: is the central MLflow up? (smoke/eval/loop run it first)
make smoke                v0 on the mock domain: proves the adapter on all three roles
make eval DOMAIN=airline AGENT=v0 SPLIT=train [TRIALS=1] [CONCURRENCY=3]
make baselines            v0 on train for all four domains
make promote RUN=<run>    make register RUN=<run>
make loop DOMAIN=airline CYCLES=1     eval → optimiser → challenger → gate → ledger (→ test run on promote)
make ledger DOMAIN=…      make snapshot
make gate                 CI gate: champions re-score offline to their registry entries
make dev                  API on :8081; `cd frontend && npm run dev` for the UI on :5174
make viewer               build the frontend and serve it from the API
make demo-up              build + run the demo image locally
make test · make lint
```

## 5 · The loop, precisely

1. `run_cycle` loads the domain's registry champion; reuses its train run
   when the folder's fingerprint matches, else evaluates it (and promotes it
   when the registry was empty).
2. Failures = `correct is False or error`. None → ledger `nothing to fix`.
3. `run_optimiser` copies the champion to `agents/<domain>/v(N+1)/`, builds
   one prompt (both surfaces; per failure the scenario, relevant policy
   clauses, expected actions, which reward components failed and a condensed
   transcript; `render_history()` of the ledger incl. per-task prior
   attempts) and runs one `ClaudeSDKClient` session with
   Read/Write/Edit/Bash/Glob/Grep. A `PreToolUse` hook denies writes outside
   the new folder and any Bash that runs a simulation or a model; a checksum
   of the guarded paths is compared before and after; `agent.yaml` must be
   byte-identical; `diagnosis.json` must exist.
4. The ledger entry is appended **before** the challenger runs (verdict
   `pending`), then `update_entry` fills `outcome` after `compare()`.
5. Promote → registry champion and one test-split run recorded in the
   outcome; hold → registry challenger. Either way the version folder and
   its runs stay in the repo.

## 6 · The helper contract

`helper.py` is optional and may define any of: `on_tool_call(name, arguments)
-> (name, arguments)` (normalise arguments before the harness executes a
tool), `on_reply(text) -> text` (post-process a message to the user),
`extra_context(policy) -> str` (appended to the system prompt). A helper that
fails to import or raises is ignored for that turn; it can never break a
conversation, only shape it.

## 7 · Conventions

- `uv` for everything Python; Ruff (line 100); mypy strict; pytest offline only.
- Models are named in `llm/__init__.py` and nowhere else.
- Never commit `.env`, keys, `.mlflow/`, `workspace/`. Never add `.lavish/`
  to `.gitignore`.
- A number on a page has its denominator; a figure has a committed source.
- Run folders are immutable once scored. Fix the code and re-run.
- Commit messages: imperative subject, a body that says why.

## 8 · Prerequisites that sit with the author

- `claude login` (subscription) for any live run.
- One-off `terraform apply` in `infra/terraform/bootstrap` with admin
  credentials before the first deploy; then `DEPLOY_ROLE_ARN` in
  `deploy-aws.yml` matches its output.
