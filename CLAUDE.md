# CLAUDE.md — tau2-loop

> Read [`AGENTS.md`](./AGENTS.md) first: what this is, the decisions, the layout,
> the loop. [`DESIGN.md`](./DESIGN.md) governs anything visual (viewer, Lavish
> artifacts, README figures). This file is a pointer plus the rules that bite.

## Quick reference

- `make smoke` · `make eval DOMAIN=airline` · `make loop DOMAIN=airline CYCLES=1`
- `make dev` + `cd frontend && npm run dev` (UI on :5174) · `make viewer`
- `uv run pytest -q` · `make lint` · `make gate`
- MLflow: `make mlflow-up` → http://127.0.0.1:5601

## Rules

- **Billing.** Every model call runs on the subscription through the Claude Agent
  SDK. `.env` has `BILLING=subscription` and no `ANTHROPIC_API_KEY`;
  `llm.require_live()` refuses to start otherwise and scrubs a key that tau2's
  `load_dotenv()` search pulls from `~/.env`. The demo image has `DEMO_MODE=1`
  baked in and cannot call a model.
- **tau2 is a pinned, unmodified submodule.** Never edit `vendor/tau2-bench`;
  route a change through our own code (`sdk_provider`, `factory`, `runner`).
- **The optimiser may edit two files** (`agents/<domain>/vN/system.md`,
  `helper.py`). "Improve the agent" means `make loop DOMAIN=…`, not a hand edit
  — a hand edit without a re-run fails the CI gate.
- **Run folders are immutable** once scored. Fix the code and re-run.
- **The test split is reported, never optimised on.** It runs once per promotion.
- **Visuals follow DESIGN.md**: tokens verbatim, assertion headings, one `<em>`
  per page, every number with its denominator. Never the Tailwind/DaisyUI fallback.
- Never add `.lavish/` to `.gitignore`.
