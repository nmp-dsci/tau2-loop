# AGENTS.md — tau2-loop

The source of truth for how this project is built and why. `CLAUDE.md` points
here. `DESIGN.md` governs anything visual. The plan this was built to is
`.lavish/s00_tau2-loop-init-plan.html`; the findings page at the end of the
build is `.lavish/s01_build-findings.html`.

## 1 · What it is — a Claude policy agent on τ²-bench, and the loop that improves it

[τ²-bench](https://github.com/sierra-research/tau2-bench) (v1.0.1, pinned at
`2174a60` as the `vendor/tau2-bench` submodule) simulates a customer-support
conversation: a user simulator plays a scripted customer, the agent follows a
policy document and calls the domain's tools, and the evaluator scores the
final database, the actions taken, what was said, and (retail) an LLM judge's
reading of natural-language assertions. Four domains are scored here:
airline, retail, telecom, banking_knowledge. This project:

1. runs our agent (`agents/<domain>/vN/{system.md, helper.py}`, and since s09
   optionally `checks.py`, `memory.py`, `guidance.py`, on the model its
   `agent.yaml` names: Sonnet 5 on airline since v3 and banking since v1, Haiku 4.5 elsewhere)
   through tau2's own harness — orchestrator, tools, user simulator, evaluator
   all unmodified — with **every model call on the Claude subscription** via a
   sealed core over the Claude Agent SDK (`src/tau2_loop/llm/`), in-process or
   as a service the harness reaches over HTTP;
2. scores it on our own committed split per domain, half the base set train and
   half test (`data/splits/`, seed 300, version 2), and keeps every run as a
   folder under `runs/`, indexed in the central MLflow (nmp-central-ai);
3. runs an **error loop per domain**: one optimiser session (Sonnet) reads
   every failed conversation and the ledger of earlier attempts, writes
   `agents/<domain>/v(N+1)/`, and a gate promotes it when it fixes at least one
   train task and breaks none, or when a one-sided exact test on the tasks that
   changed clears p < 0.05; the challenger then runs the test split once,
   promoted or held, and the champion-vs-challenger test comparison is recorded
   beside the verdict;
4. serves a read-only viewer (React + FastAPI) of the data, the architecture,
   the runs, the gate and the ledgers, deployable as a keyless demo image.

The build's goal is the working pipeline, not a score. There is no
leaderboard submission in this build (that is M6, a separate decision).

## 2 · Decisions, and the reasons

| Decision | Choice | Why |
|---|---|---|
| Split | our own, per domain, from tau2's public `base` set, seed 300, committed. v2 (now): half train, half test, no reserve (airline 25/25, retail 57/57, telecom 57/57, banking 48/49 before its test cap: 48/25, 24 in reserve), with v1's 20/20 kept inside it on the same sides | nothing is held out upstream; tau2's train/test are fixed lists and banking has none; keeping v1's sides means no task that was ever test has been trained on |
| Test cap (s09) | banking's test is the first 25 of its split v2 test list (v1's 20, then 5 dealt), the other 24 held back in reserve (`TEST_CAP`, `test_cap` in the split file) | 49 test conversations cost ~50 minutes a version; nothing held back moves to train |
| Halves (s09) | banking's train is dealt into a read half (24) and a gate half (24), seed 300, under `halves` in its split file (`HALVED_DOMAINS`); the optimiser reads only read-half failures, the gate decides on the gate half, test stays reported; train and test keep their members, so runs still read split v2 | a gate on the tasks the optimiser read grades its own homework; this makes it honest without spending the test or adding a conversation |
| Retrieval (s09) | banking runs tau2's `bm25_grep` variant (BM25 `KB_search` plus `grep` over the 698 documents), recorded as `retrieval` in `run.json`; runs before it recorded nothing and ran `bm25`, which replay and rescore still use for them | both are local; the board's AllTools adds OpenAI embeddings, which the subscription cannot call |
| Models | agent: `agent.yaml`'s model and effort (airline v3+: `claude-sonnet-5`, medium; elsewhere `claude-haiku-4-5`, medium); user simulator and NL judge: `claude-haiku-4-5`, medium; every effort is recorded in `run.json` | the only billing path is the subscription; a Claude judge makes retail's score `custom` relative to the board |
| Model swap | `tau2loop fork` copies a version's two surfaces with a new `agent.yaml`; it is evaluated once and promoted by hand (`KIND="model swap"`), or gated against the champion by `make challenge` (a ledger cycle with `kind: model swap` and no optimiser); its `diagnosis.json` names `forked_from`, and the optimiser follows that lineage to the held challengers | the optimiser may not change `agent.yaml`, so a model change is not a loop cycle |
| Ringfence | `llm/core.py` imports the SDK, `prompting.py` and the standard library only; the SDK child runs with no tools, no MCP, no settings, a temp working directory and an allow-listed environment (the SDK merges `options.env` over the parent's, so every other name is passed blank); `llm/service.py` serves it as `POST /v1/chat/completions` with a bearer token; `Dockerfile.agent` ships those three files; `AGENT_SERVICE_URL` routes a run's agent to it | the agent must depend on nothing but its prompt, its tool schemas and the SDK, and run here or in the cloud unchanged |
| Gate | pair by task; a task's score is its pass fraction over trials; promote on fixed ≥ 1 and broke 0 (dominance), or on a one-sided exact sign test p < 0.05 (McNemar at one trial); runs over different tasks or trials are refused | the rule the loop's owner set; pairing trial k with trial k would treat independent samples as pairs |
| Optimiser | `claude-opus-5-5`, effort medium (`OPTIMISER=`; `claude-sonnet-5` to airline cycle 3). `MODE=classic`: one session per cycle, may write `system.md` and `helper.py`. `MODE=routing` (s09): a read-only diagnosis session names each failure's root cause and surfaces, then a writing session may change only those, among five. Both are fenced (`loop/guards.py`): they read the policy and tools from a copy in the version folder and never `runs/`, `data/`, `loop/` or `vendor/` | must read ~20 transcripts and the policy in one context; `data/tasks` holds the test split's expected actions, which the optimiser was pointed at before s09 |
| Code surfaces (s09) | `checks.py` (a write is blocked once with a fix-it message, then its retry goes through), `memory.py` (facts from each tool result, one conversation only), `guidance.py` (a reminder on that call's system prompt, never in the transcript); standard-library imports only; no customer id, name or email from any task; a routing `system.md` grows at most 1,500 characters | write-time checks and per-turn guidance moved published airline agents more than prompt text (`.lavish/s08`); code is checked every turn, a prompt rule competes with every other |
| Tool calls | a JSON reply contract in the prompt (`tool_mode: json`), parsed back into `tool_calls` | the SDK cannot return a native tool call without executing it; the contract is the same either-message-or-tools rule tau2 enforces |
| Prompt cache (s10) | the history goes to the SDK as one text block per turn (`build_blocks`), with one `cache_control` mark (ttl 1h) on the last block only; nothing closes the transcript, so each call's blocks are the start of the next call's; cache read and write counts reach tau2's usage in-process and through the service | the Claude CLI already spends 3 of the API's 4 marks, and a second mark returned a 400; one text block changed every call and was written afresh (`.lavish/s10_banking-token-audit.html`) |
| Sampling | the CLI default; tau2's `temperature: 0.0` does not apply | the SDK exposes no temperature; recorded as `sampling: cli-default` on every run |
| Trials | one per cycle for the gate (`TRIALS=` on `make loop` for more); `pass^k` over trials when `TRIALS>1` | one trial keeps a cycle inside a subscription window; the board's ≥4 trials is a submission requirement, not a loop one |
| Tracking | MLflow 3 on the central platform (`nmp-central-ai`, http://localhost:5000; `MLFLOW_TRACKING_URI` overrides); `loop/<domain>/registry.json` is the truth | one server for the portfolio (DABStep-loop logs to the same one, experiment `dabstep-loop`); the run folder is the record, MLflow the index; the old sqlite store under `.mlflow/` is an archive |
| Billing | `require_live()` refuses a key alongside `BILLING=subscription`, scrubs a key tau2's dotenv search injects from `~/.env`, blanks the key in the SDK child; the service image sets `BILLING=subscription` and logs in with `CLAUDE_CODE_OAUTH_TOKEN` | tau2's `utils.py` calls `load_dotenv()` with a directory search on import |
| Deploy | DABStep-loop's pattern: ECR + App Runner, OIDC role, `workflow_run` after CI, `DEMO_MODE=1` in the Dockerfile | keyless by construction |
| Frontend | React 18 + Vite + TS, plain CSS on `tokens.css` from DESIGN.md | the Field Guide brief; no Tailwind/DaisyUI |

**Platform migration (2026-09-21).** Tracking moved from this repo's own MLflow (`make mlflow-up`, sqlite
under `.mlflow/`, `:5601`) to the portfolio's central server in `../nmp-central-ai` (experiment `tau2-loop`,
`http://localhost:5000`; `MLFLOW_TRACKING_URI` overrides). The
only coupling is that env var — nothing here imports the platform. Old runs stay in `.mlflow/` as a read-only
archive (platform decision D2; set `MLFLOW_TRACKING_URI=sqlite:///.mlflow/mlflow.db` to read them, never to
log). The platform's `registry/projects.yaml` lists this project and `make -C ../nmp-central-ai check
ARGS="--only P3"` re-logs one committed run as proof; run `make snapshot` only after real runs have been
re-logged, or it will overwrite `loop/mlflow_snapshot.json` with an empty index. Contract:
`../nmp-central-ai/PLATFORM.md`; receipt: `../nmp-central-ai/ai_specs/s01_m0_m1_build_receipt.md`.

## 3 · Layout

```
vendor/tau2-bench/        τ³-bench at 2174a60 (submodule): harness, domains, data/tau2/, evaluator
data/splits/<domain>.json train / test ids (v2: half each), seed 300, base_n, v1's 20 / 20   committed
data/tasks/<domain>.json  every base task (tau2's dump), policy, tool list       committed, for the viewer
data/index/leaderboard.json  τ²-bench's published submissions, ingested from the submodule   committed
agents/<domain>/vN/       system.md ({policy} slot) · helper.py (hooks) · checks.py · memory.py · guidance.py
                          (s09 code surfaces, each optional) · agent.yaml (frozen) · diagnosis.json
runs/<ts>_<domain>_<vN>_<split>/  run.json · results.jsonl · traces/<task>.json · tau2_results.json · agent/
loop/<domain>/            ledger.jsonl · registry.json;  loop/mlflow_snapshot.json for the demo
src/tau2_loop/
  config.py               paths, Settings (boots keyless), DOMAINS, split seed
  llm/                    core (sealed: models, billing check, env allow-list, one SDK answer) · prompting (contract)
                          service (POST /v1/chat/completions over core) · sdk_provider (litellm CustomLLM → core)
                          __init__ (the harness's side: prefix, dotenv scrub, the optimiser's env, redaction)
  agent/                  versions (per domain, fingerprint) · factory (LoopAgent, registered as "tau2_loop")
  data/                   splits (cut, extract, read) · leaderboard (ingest the published board) · pg (central Postgres)
  eval/                   runner (tau2 run_tasks → run folder) · results · profile (the cost distribution)
                          compare (the gate: pass fractions, sign test) · rescore (offline replay) · review
  loop/                   run (cycle, challenge) · optimiser (Opus session, hooks) · ledger
                          history (every version per domain: how made, the gate's runs, the reigns)
  tracking/               registry · mlflow_log (runs, required tags, preflight) · tracing (a trace per
                          conversation) · prompts (the prompt registry) · snapshot · gate (CI)
  serving/app.py          FastAPI + SPA; one write route (POST /api/review/…)
infra/roles.sql           schema tau2_loop on the central Postgres: review · submission (app state only)
frontend/                 Vite + React; routes.tsx is the address table, lib/url.ts the grammar,
                          lib/ui.tsx the shared pieces; src/tokens.css per DESIGN.md §5;
                          scripts/design_lint.mjs; routes.test.tsx (vitest) drives every address
Dockerfile.agent          the agent service image: core.py, prompting.py, service.py on the SDK + FastAPI
.github/workflows/        ci.yml · deploy-aws.yml
```

## 4 · Commands

```
make setup                submodule + uv sync + npm ci
make splits               cut the splits and task extracts (only when the seed or the cut's version changes)
make fork DOMAIN=… MODEL=sonnet [EFFORT=…]   a model-swap version from the champion's surfaces
make challenge DOMAIN=… AGENT=vN [TRIALS=1] [NO_TEST=1]   that version vs the champion on train through
                          the loop's own gate, ledger entry and test report, with no optimiser; runs of
                          its exact bytes are reused (`run_challenge` and `run_cycle` share `_score_challenger`)
make agent-service        the agent service container on 127.0.0.1:8091 (AGENT_PORT=); then
                          AGENT_SERVICE_URL=http://127.0.0.1:8091 make eval … routes the agent to it
make platform-up          central MLflow (make -C ../nmp-central-ai up) → http://localhost:5000
make platform-status      preflight: is the central MLflow up? (smoke/eval/loop run it first)
make smoke [AGENT=v0]     a mock-domain version: proves the adapter on all three roles
make eval DOMAIN=airline AGENT=v0 SPLIT=train [TRIALS=1] [CONCURRENCY=3]
make baselines            v0 on train for all four domains
make promote RUN=<run> [KIND="model swap"]    make register RUN=<run>
make loop DOMAIN=airline CYCLES=1 [TRIALS=1] [MODE=classic|routing]   eval → optimiser → challenger → gate → test → ledger
make ab DOMAIN=banking_knowledge   two challengers from the champion on the same failures, the classic and the
                          routing optimiser; both gated and tested; at most one crowned (s09 §6)
make ledger DOMAIN=…      make snapshot
make gate                 CI gate: champions re-score offline to their registry entries
make leaderboard          ingest τ²-bench's published submissions → data/index/leaderboard.json
make db-migrate           apply infra/roles.sql to the central Postgres (database `tau2`, idempotent)
make db-smoke             zero-LLM proof this project can reach its database
make dev                  API on :8081; `cd frontend && npm run dev` for the UI on :5174
make viewer               build the frontend and serve it from the API
make demo-up              build + run the demo image locally
make test · make lint
```

## 4b · The viewer — eight tabs, DataAgentBench's slots (plan s04)

| tab | address | what it answers |
|---|---|---|
| Overview | `/` | where each domain stands |
| Domains & tasks | `/domains/<domain>/<task>` | what the benchmark is; a task's answer key |
| Rubric | `/rubric` | how τ² decides a conversation passed, and which check failed |
| Leaderboard | `/leaderboard` | who else has tried, and why we are not comparable yet |
| Runs | `/runs/<run>[?vs=<run>]` | who holds each domain and how the title moved; what we ran, what it cost, and the gate against another run |
| Optimise | `/optimise/<domain>/<version>` | every version of a domain, train and test; a round: diagnose → propose → outcome; every cycle across the domains |
| Agent | `/agent/<domain>/<version>?run=&trial=&node=&step=` | one conversation drawn as the agent in its harness; each node's inputs and outputs; a tool playground (plan s05) |
| Review | `/review/<run>/<task>/t<n>` | what a person thought of what the judge scored (writes) |

The grammar is `frontend/src/lib/url.ts`: one id per thing, the path names the
subject, the query holds the lens, and a detail opens inside its list. Every
address published before it redirects — `routes.test.tsx` asserts each one.

The Runs and Optimise figures (`lib/versions.tsx`, after DataAgentBench's) read one
route, `GET /api/versions`, built by `loop/history.py` from the version folders, the
ledger, the registry and the run folders. A version's train number is the run the gate
read (the ledger's challenger run, else its latest promotion); its test number is the
ledger's test run, else its newest test run on the same cut. Runs draws each domain's
champion and newest challenger, then the champions over the versions (a re-baseline is a
second point in one column); Optimise draws the chosen domain's versions, train and test
with the champion each faced, and every cycle in every domain as champion against
challenger. No line joins numbers from two different cuts.

The Agent tab (`lib/agentgraph.tsx`) reads three routes. `GET /api/runs/<run>/<task>/t<n>`
returns every message whole (ids pair a result with its call), the task spec, and the
domain's tools typed read/write from `data/tasks/<domain>.json` (the extract carries
tau2's own `tool_type`, and telecom's and banking's customer-side tools). `GET
/api/runs/<run>/agent` serves the run's own agent snapshot, with the prompt composed by
`agent/compose.py`, the one function `LoopAgent.system_prompt()` also calls. `POST
/api/runs/<run>/<task>/t<n>/tool` is the playground: `eval/replay.py` rebuilds tau2's
environment at a message (`set_state`, as the evaluator does) and runs one call in it.
It writes nothing and calls no model; it needs tau2, so the demo image answers 503.
`/agent` alone reopens the last view in the browser tab, else a champion's first failure.

## 5 · The loop, precisely

1. `run_cycle` loads the domain's registry champion; reuses its train run
   when the fingerprint matches and the run covers the split's current train
   ids at the cycle's trials, else evaluates it (and promotes that run: as the
   first champion when the registry was empty, as a `re-baseline` when the
   same bytes had a run on an older cut).
2. Failures = `correct is False or error`, on the read half where train is
   halved. None → ledger `nothing to fix`.
3. `run_optimiser` copies the champion to `agents/<domain>/v(N+1)/`, writes
   the policy and tools to its `.context/`, builds the prompt (the surfaces; per
   failure the scenario, relevant policy clauses, expected actions, which
   reward components failed and a condensed transcript; `render_history()` of
   the ledger incl. per-task prior attempts, with only read-half ids where
   train is halved) and runs `ClaudeSDKClient` sessions with
   Read/Write/Edit/Bash/Glob/Grep: one (`classic`), or a diagnosis then a
   writer (`routing`). `PreToolUse` hooks deny a write to any file the
   session was not given, a read of a fenced folder or an A/B partner's, and
   any Bash that runs a simulation or a model; a checksum of the guarded paths
   is compared before and after; `agent.yaml` must be byte-identical;
   `diagnosis.json` must exist; the package guards (imports, customer data,
   the routing budget) reject the cycle before the challenger runs.
4. The ledger entry is appended **before** the challenger runs (verdict
   `pending`), then `update_entry` fills `outcome` after `compare()`.
5. Promote → registry champion; hold → registry challenger. Either way the
   challenger then runs the test split once, and the champion's test run on
   the same tasks (reused, or run once) is compared with it: `test_compare`
   in the outcome — passes, fixed, broke, p — reported, never used by the
   gate. The version folder and its runs stay in the repo.

## 6 · The helper contract, and the code surfaces

`helper.py` is optional and may define any of: `on_tool_call(name, arguments)
-> (name, arguments)` (normalise arguments before the harness executes a
tool), `on_reply(text) -> text` (post-process a message to the user),
`extra_context(policy) -> str` (appended to the system prompt). A helper that
fails to import or raises is ignored for that turn; it can never break a
conversation, only shape it.

Since s09 three more files, each optional, each one hook (`agent/factory.py`):

- `memory.py` `remember(state, name, arguments, result)` after every tool
  result (with the call's name and arguments) and user message (`name ==
  "user"`); `state` is a dict that starts empty in each conversation.
- `guidance.py` `guidance(state, trigger) -> str | None` before every model
  call (`trigger` is `user` or `tool`); the reminder, cut to 600 characters,
  rides on that call as a system message, which the provider folds into the
  call's system prompt, so the transcript never holds it.
- `checks.py` `check_write(name, arguments, state) -> str | None` on each call
  to a tool tau2 types `write`; a string blocks the reply once: the model reads
  it as those calls' results and replies again, and that reply goes through.

What they did is kept on the reply as `raw_data["tau2_loop"]` and shown on the
Agent tab. A version without them has the fingerprint it always had.

## 7 · Conventions

- `uv` for everything Python; Ruff (line 100); mypy strict; pytest offline only.
- Models are named in `llm/core.py` and nowhere else.
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
