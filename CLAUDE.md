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
- **The optimiser may edit the version's surfaces only**: `system.md` and
  `helper.py` (`MODE=classic`), plus `checks.py`, `memory.py`, `guidance.py`
  when a routing diagnosis names them (`MODE=routing`, s09). It never reads
  `runs/`, `data/`, `loop/` or `vendor/`. "Improve the agent" means
  `make loop DOMAIN=…`, not a hand edit — a hand edit without a re-run fails
  the CI gate.
- **Run folders are immutable** once scored. Fix the code and re-run.
- **The test split is reported, never optimised on.** It runs once per challenger, promoted or held; the gate never reads it. Where train is halved (banking), the gate half is never shown to an optimiser either.
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
