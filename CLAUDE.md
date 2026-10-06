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
- **Banking's champion is v4** (5 Oct 2026, promoted by the gate): written by `make optimise`
  from v3 under banking's s14 guide (routing: all five surfaces), Sonnet at high effort with
  `parallel_calls: true`. Test 6 → 14 (fixed 8, broke 0, p = 0.004); train 12 → 29 of 60
  (`20261004T222002Z`, test `20261004T113341Z`). v3 before it (4 Oct, the person's call): v1's
  prompt with AllTools (`alltools_minilm`), native calls and the identity note, a `tool change`
  that removed v1's harness defects. Loop cycles start from v4; CI's gate needs the sandbox and
  MiniLM to re-score it (`.github/workflows/ci.yml`).
- **Banking's base is v1, its split is v3** (3 Oct 2026): v0 is retired and
  deleted; every banking experiment is scored on split v3's 60 train / 37 test
  tasks. v1's runs were extended to it (`make extend RUN=`: only the new tasks
  played, joined to the old run): train `20261002T235638Z`, test
  `20261003T003126Z`. Never re-create banking v0 or score banking on another
  task set. A plain `make eval` runs v1, the oldest version left, never a later
  champion (`versions.base_version`).
- **Retrieval, tool mode and the identity note are version settings** (s13, 4 Oct 2026),
  frozen in `agent.yaml` like the model: `retrieval:` (banking's variant; absent = today's
  `bm25_grep`), `tool_mode: json|native` (native: the model calls the tools and each call
  returns to tau2 unrun, `llm/core.py`), `identity_note: true` (`compose.IDENTITY_NOTE`, the
  CLI's account email is not the customer's), `parallel_calls: true` (s14, native only: every
  tool call of a reply kept; before it the core kept the first, so v1–v3 made one call a turn).
  Change them with `make fork … RETRIEVAL= TOOL_MODE= IDENTITY_NOTE=true PARALLEL_CALLS=true`,
  or on `make optimise`, never a loop edit. The local AllTools variants
  (`eval/retrieval.py`: `alltools_minilm`, `alltools_qwen3_0_6b`) need sandbox-runtime,
  ripgrep and sentence-transformers wherever their runs are played or re-scored.
- **`make optimise` writes a version and scores nothing** (s14, 4 Oct 2026): one optimiser
  session on the source's train failures (default the champion). `EFFORT=`, `MODEL=`,
  `PARALLEL_CALLS=true` set the new version's `agent.yaml`, written and frozen by the harness
  before the session, never by the optimiser. Play it with `make eval`, gate it with `make
  challenge` (its ledger entry carries the diagnosis, kind `optimised`). Every optimiser reads
  the policy its source's run had, banking's documents in `.context/kb/`, and never `.lavish/`
  (fenced: the pages quote test results). `checks.py` may also define `check_reply(text,
  state)`, which sends a text reply back once, as `check_write` does a write.
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
  the judge's data, and never re-deal its folds. The data is closed at the runs
  its committed labels hold (the person's call, 3 Oct 2026,
  `labels.CLOSED_AT`): a run scored later, of any agent or domain, never joins
  the labels, golden answers, replays, the judge loop, the pending queue or Evals.
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
- **workflow_rag researches train only** (s16, 6–7 Oct 2026): banking's second agent,
  `rag_agents/<domain>/rN/`, is frozen like an answering version (a change is the next rN), and
  its sessions under `rag_agent_runs/` are immutable. The library v6 and v7 look up
  (`find_workflow`) holds only train-question sessions; a mid-conversation `request_workflow`
  reads only the agent's words and is never cached. Test is held out of optimisation, not hidden
  (the person's call): Evals shows each test question's workflows as `golden.RULES` map them,
  and `test_counts.json` stays counts only. v7 (v6 at Sonnet medium) is a held experiment:
  train 37/60 and test 12/37 against v4's 29 and 14 (`.lavish/s17_…`); v4 stays champion.
- **Visuals follow DESIGN.md**: tokens verbatim, assertion headings, one `<em>`
  per page, every number with its denominator. Never the Tailwind/DaisyUI fallback.
- Never add `.lavish/` to `.gitignore`.
