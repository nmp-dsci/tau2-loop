# CLAUDE.md — tau2-loop

> Read [`AGENTS.md`](./AGENTS.md) first: what this is, the decisions, the layout,
> the loop. [`DESIGN.md`](./DESIGN.md) governs anything visual (viewer, Lavish
> artifacts, README figures). This file is a pointer plus the rules that bite.

## Quick reference

- `make smoke` · `make eval DOMAIN=airline` · `make loop DOMAIN=airline CYCLES=1`
- `make dev` + `cd frontend && npm run dev` (UI on :5174) · `make viewer`
- `uv run pytest -q` · `make lint` · `make gate`
- MLflow: central (`make platform-up` → `make -C ../nmp-central-ai up`) → http://localhost:5000; `MLFLOW_TRACKING_URI` overrides

## Rules

- **Billing.** Every model call runs on the subscription through the Claude Agent
  SDK. `.env` has `BILLING=subscription` and no `ANTHROPIC_API_KEY`;
  `llm.require_live()` refuses to start otherwise and scrubs a key that tau2's
  `load_dotenv()` search pulls from `~/.env`. The demo image has `DEMO_MODE=1`
  baked in and cannot call a model.
- **tau2 is a pinned, unmodified submodule.** Never edit `vendor/tau2-bench`;
  route a change through our own code (`sdk_provider`, `factory`, `runner`).
- **Neither side may know the real date, and the customer is ours.** The Claude
  CLI tells every session today's date; the customer (`eval/user.py`, a subclass
  of tau2's) is told the world's time and to ignore it, the agent's prompt ends
  with `compose.CLOCK_NOTE`. The customer also holds a stop sent with words until
  the agent's turn. Runs record `sim_rules`; the loop never compares runs across
  it, so bump `runner.SIM_RULES` when these rules change.
- **Banking's base is v1, its split is v3** (3 Oct 2026): v0 is retired and
  deleted; every banking experiment is scored on split v3's 60 train / 37 test
  tasks. v1's runs were extended to it (`make extend RUN=`: only the new tasks
  played, joined to the old run): train `20261002T235638Z`, test
  `20261003T003126Z`. Never re-create banking v0 or score banking on another
  task set. A plain `make eval` runs v1, the oldest version left, never a later
  champion (`versions.base_version`).
- **The optimiser may edit the version's surfaces only**: `system.md` and
  `helper.py` (`MODE=classic`), plus `checks.py`, `memory.py`, `guidance.py`
  when a routing diagnosis names them (`MODE=routing`, s09). It never reads
  `runs/`, `data/`, `loop/` or `vendor/`. "Improve the agent" means
  `make loop DOMAIN=…`, not a hand edit — a hand edit without a re-run fails
  the CI gate.
- **Run folders are immutable** once scored. Fix the code and re-run.
- **The test split is never shown to an optimiser.** It runs once per challenger,
  promoted or held, and the gate never reads it, except banking's
  (`config.GATE_ON_TEST`, the person's call, 3 Oct 2026): there the optimiser
  reads all 60 train tasks and promotion compares the challenger's test run with
  the champion's. No test conversation or test task id reaches a prompt
  (`optimiser._redact`), only the gate's pass counts.
- **The LLM judge learns from optimised answering agents only** (s11): never
  v0's traces. J0's labels skip them (`labels.UNOPTIMISED`), so the golden
  answers, replays and the judge loop never see them. Never add a v0 run to
  the judge's data, and never re-deal its folds.
- **The LLM judge is called only at a write or a transfer, before it runs**
  (s11): never on a text reply. J0 marks those checkpoints `judged`
  (`labels.JUDGED_KINDS`); Evals, the review queue and the judge's bar show
  only them. The plan judge (j1–j3, text replies) is retired. No judge is
  replayed until the person has confirmed the golden answers.
- **A passed conversation is confirmed by the grader** (s11, the passed rule):
  tau2 matched its database to gold's, so every write and transfer in it is
  golden `allow` (`review.effective_verdicts`), whatever the annotator said; no
  person checks it and `make judge-gold` calls no model for it. A person ticks
  each failed one in Evals: the first call the judge must block, or none.
- **Visuals follow DESIGN.md**: tokens verbatim, assertion headings, one `<em>`
  per page, every number with its denominator. Never the Tailwind/DaisyUI fallback.
- Never add `.lavish/` to `.gitignore`.
