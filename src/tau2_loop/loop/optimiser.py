"""The optimiser: Agent SDK sessions that read every failure and write the next version.

Two modes (s09). `classic` is one session, as every cycle before: it sees the
champion's surfaces, every failed conversation's scenario, policy clauses,
expected actions, the reward components that failed, what tau2 graded rebuilt
offline (`diff_block`: every field where the final database differs from gold's,
and each expected action matched or missing beside the agent's calls) and a
condensed transcript, plus the ledger's history, and may write only `system.md`, `helper.py` and
`diagnosis.json`. `routing` splits that in two: a read-only diagnosis session
names each failure's root cause and the surfaces that fit it (among `system.md`,
`helper.py`, `checks.py`, `memory.py`, `guidance.py`), then a writing session may
change only those. Either way a PreToolUse hook refuses any other write, a
checksum of the guarded tree is compared after the session, and the session is
fenced: it reads the policy and tools from a copy in the version folder and
never `runs/`, `data/`, `loop/` or `vendor/`, which hold the test split's and the
gate half's tasks and results (`loop/guards.py`). The package is then checked:
code surfaces import only the standard-library allow-list, no file carries a
customer's id, name or email, and a routing `system.md` stays within its budget.
It may not run a simulation, and it must finish with `diagnosis.json` — the
structured record the ledger stores.

Each dataset has its own optimiser profile (s13 §5, `loop/profiles.py`): its model,
effort, mode, turn limit, failure-reading budgets and surfaces, a guide of its own
lessons appended to these prompts when non-empty, and optional guards of its own.
Every default is the one optimiser above, so an untuned dataset's prompts are
byte for byte what they were.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from tau2_loop.agent.compose import CHECKED_TOOLS
from tau2_loop.agent.versions import (
    AGENT_YAML_HEADER,
    CODE_SURFACES,
    FROZEN,
    SURFACES,
    AgentConfig,
    AgentVersion,
    lineage,
    load_version,
    next_version_name,
    version_dir,
)
from tau2_loop.config import AGENTS_DIR, BANKING_RETRIEVAL, GATE_ON_TEST, ROOT, RUNS_DIR
from tau2_loop.data.splits import halves, read_task_extract, split_ids, tool_kinds
from tau2_loop.eval.results import TaskResult, read_results
from tau2_loop.llm import (
    Effort,
    model_label,
    redact,
    redact_tree,
    require_live,
    resolve_model,
    subscription_env,
)
from tau2_loop.loop.guards import (
    ALLOWED_IMPORTS,
    PROMPT_GROWTH,
    bash_fence_reason,
    fence_reason,
    import_violations,
    leak_values,
    leaks,
    profile_violations,
)
from tau2_loop.loop.ledger import read_ledger, render_history, seen_changes, swap_note
from tau2_loop.loop.profiles import (  # today's values: every profile's defaults (s13 §5)
    DIFF_BUDGET,
    MAX_TURNS,
    MODE_SURFACES,
    MODES,
    SCENARIO_BUDGET,
    TRANSCRIPT_BUDGET,
    OptimiserProfile,
    load_profile,
)

CONTEXT_DIR = ".context"  # the policy and tools the session reads, removed when it ends
TRACE_CHARS = 9000
TOOL_RESULT_CHARS = 700
FIELD_ROWS = 8  # differing fields shown per record; the rest counted
MIN_SHARE = 1000


def _share(budget: int, n: int, cap: int | None = None) -> int:
    """One failure's share of a budget, never under MIN_SHARE nor over `cap`."""
    share = max(MIN_SHARE, budget // max(n, 1))
    return min(share, cap) if cap else share


def _calls(pairs: list[tuple[str, int | str]], noun: str) -> str:
    """Calls grouped by name: `cancel_reservation at message 12`, or for a tool called many
    times `call_discoverable_agent_tool at messages 40, 42, 44 … 88 (16 calls)`."""
    by: dict[str, list[int | str]] = {}
    for name, at in pairs:
        by.setdefault(name, []).append(at)
    out = []
    for name, ats in by.items():
        if len(ats) == 1:
            out.append(f"{ats[0]} {name}" if noun == "id" else f"{name} at message {ats[0]}")
            continue
        shown = ", ".join(str(a) for a in ats[:3]) + (f" … {ats[-1]}" if len(ats) > 3 else "")
        out.append(
            f"{name} ×{len(ats)} ({shown})"
            if noun == "id"
            else f"{name} at messages {shown} ({len(ats)} calls)"
        )
    return ", ".join(out)


@dataclass
class OptimiserOutput:
    new_version: str
    diagnosis: dict[str, Any]
    n_turns: int = 0
    duration_ms: int = 0
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    transcript: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    # s09: which optimiser, the surfaces it was allowed and the ones it changed; a guard or a
    # write outside the version rejects the cycle before the challenger runs
    mode: str = "classic"
    routed: list[str] = field(default_factory=list)
    surfaces_changed: list[str] = field(default_factory=list)
    rejected: bool = False


def condense_trace(trace_path: Path, limit: int = TRACE_CHARS) -> str:
    """The conversation in order: user, agent, tool calls and their results; the parts a diagnosis needs."""
    if not trace_path.exists():
        return "(no trace)"
    d = json.loads(trace_path.read_text())
    lines: list[str] = []
    for m in d.get("messages") or []:
        role = m.get("role")
        if role == "user":
            if m.get("tool_calls"):
                lines.append("### user → tool\n" + json.dumps(m["tool_calls"], ensure_ascii=False))
            else:
                lines.append("### user\n" + str(m.get("content") or "").strip())
        elif role == "assistant":
            if m.get("tool_calls"):
                calls = [
                    {"name": c.get("name"), "arguments": c.get("arguments")}
                    for c in m["tool_calls"]
                ]
                lines.append("### agent → tool\n" + json.dumps(calls, ensure_ascii=False))
            else:
                lines.append("### agent\n" + str(m.get("content") or "").strip())
        elif role == "tool":
            tag = "tool result · ERROR" if m.get("error") else "tool result"
            lines.append(f"### {tag}\n" + str(m.get("content") or "").strip()[:TOOL_RESULT_CHARS])
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[: limit // 2] + "\n…[middle of conversation elided]…\n" + text[-limit // 2 :]
    meta = {
        "termination_reason": d.get("termination_reason"),
        "duration_s": round(float(d.get("duration") or 0), 1),
    }
    return f"{json.dumps(meta)}\n{text}"


def failure_details(trace_path: Path, actions: bool = True) -> str:
    """Which reward components failed, from the saved reward_info: expected vs matched actions
    (unless `actions=False`: `diff_block` lists them against the agent's calls), unmet checks."""
    if not trace_path.exists():
        return "(no reward_info)"
    ri = (json.loads(trace_path.read_text()).get("reward_info")) or {}
    out: list[str] = [f"reward {ri.get('reward')} · basis {ri.get('reward_basis')}"]
    db = ri.get("db_check")
    if db is not None:
        out.append(
            f"DB check: {'match' if db.get('db_match') else 'MISMATCH — the final database differs from the gold one'}"
        )
    for a in ri.get("env_assertions") or []:
        if not a.get("met"):
            out.append(
                f"ENV ASSERTION UNMET: {json.dumps(a.get('env_assertion'), ensure_ascii=False)}"
            )
    for c in (ri.get("action_checks") or []) if actions else []:
        act = c.get("action") or {}
        mark = "matched" if c.get("action_match") else "MISSING"
        out.append(
            f"expected action {mark}: {act.get('name')}({json.dumps(act.get('arguments'), ensure_ascii=False)})"
        )
    for c in ri.get("communicate_checks") or []:
        if not c.get("met"):
            out.append(f"COMMUNICATE UNMET: the agent never told the user {c.get('info')!r}")
    for a in ri.get("nl_assertions") or []:
        if not a.get("met"):
            out.append(
                f"NL ASSERTION UNMET: {a.get('nl_assertion')} — judge: {str(a.get('justification'))[:300]}"
            )
    info = ri.get("info") or {}
    if isinstance(info, dict) and info.get("note"):
        out.append(f"note: {info['note']}")
    return "\n".join(out)


DIFF_VALUE_CHARS = 300


def _cut(v: str | None, cap: int = DIFF_VALUE_CHARS) -> str:
    if v is None:
        return "not set"
    return v if len(v) <= cap else f"{v[: cap - 1]}…"


def diff_block(run_id: str, r: TaskResult, domain: str, limit: int | None = None) -> str | None:
    """What tau2 graded, laid out for a diagnosis: the database difference (when DB is in the
    task's reward basis and failed) and the expected actions against the agent's calls, both
    rebuilt offline from the trace and the task (`eval.replay`). None when they cannot be
    rebuilt here; the caller then keeps tau2's matched/missing list. `limit` is this failure's
    share of the profile's diff budget (`DIFF_BUDGET` by default): arguments are cut shorter and
    the lines past it are counted."""
    from tau2_loop.eval import replay

    if not r.trace or not replay.available():
        return None
    try:
        conv = replay.load_conversation(run_id, r.trace, domain, r.task_id)
        acts = replay.action_diff(conv)
        db = replay.db_diff(conv) if "DB" in acts["basis"] and r.db_check is False else None
    except Exception:  # noqa: BLE001 - a diff that cannot be rebuilt never stops a cycle
        return None
    out: list[str] = []
    if db is not None:
        out.append(
            "DATABASE DIFFERENCE (DB is in this task's reward basis; this is the check that failed): "
            f"{db['fields']} field(s) of {len(db['records'])} record(s) differ"
            + (f", the first {db['shown']} listed" if db["shown"] < db["fields"] else "")
        )
        for rec in db["records"]:
            agent = _calls(
                sorted({(c["name"], c["msg"]) for c in rec["agent_calls"]}, key=lambda x: x[1]),
                "msg",
            )
            gold = _calls([(a["name"], a["action_id"]) for a in rec["gold_actions"]], "id")
            why = {
                "missed write": f"gold's {gold} changes it; the agent never wrote to it",
                "wrong write": f"the agent's {agent} changed it; gold leaves it as it was",
                "wrong arguments": f"both change it, differently: the agent's {agent}; gold's {gold}",
            }.get(rec["verdict"], "no recorded write on either side changed it directly")
            out.append(f"- {rec['record']} — {rec['verdict']}: {why}")
            for f in rec["fields"][:FIELD_ROWS]:
                out.append(
                    f"    {f['field']}: before {_cut(f['before'])} · the agent left {_cut(f['agent'])} "
                    f"· gold expects {_cut(f['gold'])}"
                )
            if len(rec["fields"]) > FIELD_ROWS:
                out.append(
                    f"    … and {len(rec['fields']) - FIELD_ROWS} more differing fields in this record"
                )
        for e in db["gold_errors"]:
            out.append(
                f"  (expected action {e['action_id']} {e['name']} raised on gold's side: {e['error']})"
            )
    args = DIFF_VALUE_CHARS if limit is None or limit >= 6000 else 160
    graded = (
        "ACTION is in this task's reward basis: every expected action must be matched"
        if acts["graded"]
        else "ACTION is not in this task's reward basis: an action counts only through the database it "
        "produces, so a missing read or transfer costs nothing here"
    )
    out.append(f"EXPECTED ACTIONS AGAINST THE AGENT'S CALLS ({graded}):")
    matched = [e for e in acts["expected"] if e["matched_at"] is not None]
    if matched:
        out.append(
            "- matched: "
            + ", ".join(
                f"{e['action_id']} {e['name']} at message {e['matched_at']}" for e in matched
            )
        )
    for e in acts["expected"]:
        if e["matched_at"] is not None:
            continue
        kind = "write" if e["write"] else "read" if acts["graded"] else "read, not graded"
        line = f"- MISSING {e['action_id']} {e['name']} ({kind}) {_cut(json.dumps(e['arguments'], ensure_ascii=False), args)}"
        n = e["nearest"]
        if not e["write"] and not acts["graded"]:
            pass  # a lookup gold made and the agent did not: nothing to trace
        elif n is None:
            line += (
                f" — every {e['name']} call the agent made matches another expected action"
                if e["name"] in acts["called"]
                else f" — the agent never called {e['name']}"
            )
        else:
            differs = "; ".join(
                f"{d['argument']}: the agent {_cut(d['agent'], 120)}, expected {_cut(d['expected'], 120)}"
                for d in n["differs"]
            )
            line += (
                f" — nearest: the agent's {n['name']} at message {n['msg']}"
                + (" (refused by the tool)" if n["refused"] else "")
                + f", differing in {differs}"
            )
        out.append(line)
    for u in acts["unexpected_writes"]:
        out.append(
            f"- UNEXPECTED WRITE at message {u['msg']}: {u['name']} "
            f"{_cut(json.dumps(u['arguments'], ensure_ascii=False), args)}"
            + (" — refused by the tool, so it changed nothing" if u["refused"] else " — it ran")
        )
    if limit is None or sum(len(x) + 1 for x in out) <= limit:
        return "\n".join(out)
    kept: list[str] = []
    for x in out:
        if sum(len(k) + 1 for k in kept) + len(x) > limit:
            break
        kept.append(x)
    return "\n".join(
        [*kept, f"… {len(out) - len(kept)} more lines cut, this failure's share of the prompt"]
    )


def task_block(
    domain: str, task_id: str, goal_chars: int | None = None, actions: bool = True
) -> str:
    """Scenario, purpose, relevant policy clauses and evaluation criteria from the committed extract."""
    ext = read_task_extract(domain)
    t = next((x for x in ext.get("tasks", []) if x.get("id") == task_id), None)
    if not t:
        return "(task not in the committed extract)"
    desc = t.get("description") or {}
    scenario = t.get("user_scenario") or {}
    sc = scenario.get("instructions") or {}
    if not isinstance(sc, dict):  # banking: one block of prose, beside a persona
        sc = {"task_instructions": sc}
    ev = t.get("evaluation_criteria") or {}
    parts = [
        f"PURPOSE: {desc.get('purpose')}",
        f"RELEVANT POLICIES: {desc.get('relevant_policies')}"
        if desc.get("relevant_policies")
        else "",
        f"NOTES: {_cut(str(desc['notes']), goal_chars) if goal_chars else desc['notes']}"
        if desc.get("notes")
        else "",
        f"USER'S PERSONA: {scenario.get('persona')}" if scenario.get("persona") else "",
        f"USER'S GOAL (what the simulated user was told to do): {_cut(str(sc.get('task_instructions') or sc), goal_chars) if goal_chars else sc.get('task_instructions') or sc}",
        f"USER KNOWS: {sc.get('known_info')}" if sc.get("known_info") else "",
        f"USER DOES NOT KNOW: {sc.get('unknown_info')}" if sc.get("unknown_info") else "",
        f"EXPECTED ACTIONS: {json.dumps([{k: a.get(k) for k in ('name', 'arguments')} for a in ev.get('actions') or []], ensure_ascii=False)}"
        if actions
        else "EXPECTED ACTIONS: listed against the agent's calls under WHAT FAILED",
        f"REQUIRED DOCUMENTS (tau2's list for this task, each at .context/kb/<id>.md): "
        f"{', '.join(t['required_documents'])}"
        if t.get("required_documents")
        else "",
        f"MUST COMMUNICATE: {ev.get('communicate_info')}" if ev.get("communicate_info") else "",
        f"NL ASSERTIONS: {ev.get('nl_assertions')}" if ev.get("nl_assertions") else "",
        f"REWARD BASIS: {ev.get('reward_basis')}",
    ]
    return "\n".join(p for p in parts if p)


def _unseen_ids(domain: str) -> set[str]:
    """Ids of the tasks no optimiser may read: test, and the gate half where train is halved.
    Numeric ids (airline's) are left out: a bare number in a task's notes is rarely a task."""
    try:
        h = halves(domain)
        ids = set(split_ids(domain, "test")) | (set(h[1]) if h else set())
    except (FileNotFoundError, KeyError, ValueError):
        return set()
    return {t for t in ids if not t.isdigit()}


def _redact(text: str, unseen: set[str]) -> str:
    """A held-out task named in a train task's notes ("Adversarial variant of task_026") becomes
    "a held-out task": its id never reaches a prompt."""
    if not unseen:
        return text
    return re.sub(
        r"\b(" + "|".join(sorted(map(re.escape, unseen))) + r")\b", "a held-out task", text
    )


@dataclass
class Context:
    """What both optimiser modes read, built once per cycle so an A/B pair sees the same input."""

    domain: str
    blocks: list[str]
    history: str
    held_block: str
    fork_note: str
    tool_names: str
    read_traces: list[Path]
    halves: tuple[list[str], list[str]] | None
    # every run of the champion and of its challengers, task by task (`champion_record`)
    record: str = ""
    # the dataset's optimiser profile the context was built under (s13 §5): its budgets cut the
    # blocks above, its guide and surfaces reach the prompts; None reads as today's optimiser
    profile: OptimiserProfile | None = None
    # banking: the knowledge base is copied read-only into the version's `.context/kb/` (s14 P8b)
    kb: bool = False


def build_context(
    champion: AgentVersion,
    run_id: str,
    failures: list[TaskResult],
    profile: OptimiserProfile | None = None,
) -> Context:
    """Everything the prompts read of a cycle, under the domain's profile (loaded when not given)."""
    domain = champion.domain
    profile = profile or load_profile(domain)
    run_dir = RUNS_DIR / run_id
    blocks: list[str] = []
    traces: list[Path] = []
    trace_chars = _share(profile.transcript_budget, len(failures), TRACE_CHARS)
    goal_chars = _share(profile.scenario_budget, len(failures))
    diff_chars = _share(profile.diff_budget, len(failures))
    unseen = _unseen_ids(domain)
    for r in failures:
        tp = run_dir / "traces" / r.trace
        traces.append(tp.resolve())
        diffs = diff_block(run_id, r, domain, diff_chars)
        what = failure_details(tp, actions=diffs is None) + (f"\n{diffs}" if diffs else "")
        blocks.append(
            f"## Task {r.task_id} (trial {r.trial})\n"
            f"{_redact(task_block(domain, r.task_id, goal_chars, actions=diffs is None), unseen)}\n"
            f"WHAT FAILED:\n{what}\n"
            f"ERROR: {r.error or 'none'} · {r.n_agent_turns} agent turns · {r.n_tool_calls} tool calls · ended by {r.termination_reason}\n"
            f"TRANSCRIPT (condensed; the whole conversation is runs/{run_id}/traces/{r.trace}, readable):\n"
            f"{condense_trace(tp, trace_chars)}\n"
        )
    held = held_challengers(domain, champion.name)
    record, record_traces = champion_record(champion, run_id, failures)
    traces += record_traces
    for h in held:
        traces += [(ROOT / t).resolve() for t in h.get("broken_traces") or []]
    ancestors = lineage(domain, champion.name)[1:]
    fork_note = (
        f" `{champion.name}` is a fork of `{ancestors[0]}` (same system.md and helper.py, a different "
        f"agent.yaml), so the held challengers of {', '.join(f'`{a}`' for a in ancestors)} are listed "
        "below as its own; they were scored on the earlier model and the earlier cut."
        if ancestors
        else ""
    )
    held_block = (
        "\n".join(
            f"- `agents/{domain}/{h['challenger']}/` (cycle {h['cycle']}, challenger of {h['champion']}, held: {h['reason']}; passes {h['passes']}; fixed {h['fixed']}; "
            f"broke {h['broken']}). Its system.md and helper.py are on disk: read them and copy what held — a held "
            "challenger is a starting point, not a rejected one. "
            + (
                f"Which of its fixes and breaks were real, against every run of `{champion.name}`, and the "
                "conversations it broke are under # What the champion's challengers moved."
                if record and h["champion"] == champion.name
                else "Do not repeat what broke: read the traces of the broken tasks "
                f"({', '.join(h.get('broken_traces') or []) or 'none'}) before you decide what to keep."
            )
            for h in held
        )
        or "none"
    )
    ext = read_task_extract(domain)
    return Context(
        domain=domain,
        blocks=blocks,
        history=render_history(domain, [r.task_id for r in failures]),
        held_block=held_block,
        fork_note=fork_note,
        tool_names=", ".join(t["name"] for t in ext.get("tools", [])),
        read_traces=traces,
        halves=halves(domain),
        record=record,
        profile=profile,
        kb=_kb_source(domain) is not None,
    )


def _intro(champion: AgentVersion, ctx: Context) -> str:
    from tau2_loop.eval.runner import USER_MODEL

    return f"""You are the optimiser in a benchmark improvement loop for a customer-support agent on τ²-bench,
domain `{ctx.domain}`. The agent is {model_label(champion.config.model)} on the Claude subscription; it follows a policy document and calls the
domain's tools ({ctx.tool_names}) {"as native tool calls" if champion.config.tool_mode == "native" else "through a JSON contract"}; a simulated user ({model_label(USER_MODEL)}) plays the customer. A
conversation scores 1 only if every component in its task's REWARD BASIS passes (each failure lists it): DB, the
final database equals gold's, which tau2 builds by running the expected actions on the same starting database;
ACTION, every expected action was called with its arguments; COMMUNICATE, the required facts were said to the
user; NL_ASSERTION, an LLM judge finds the assertions met; ENV_ASSERTION, the environment checks hold. Where ACTION
is not in the basis (airline, retail, most of banking), the expected actions are graded only through the database
they produce: the writes count, a missed lookup does not."""


READING_GUIDE = """Each failure lists what tau2 graded. Start from the component that failed.
- **DATABASE DIFFERENCE** is the DB check itself: every field where the final database differs from gold's, with
  its value before the conversation, what the agent left and what gold expects, and the calls on each side that
  changed the record. Each record is one of three:
  - *missed write*: gold changes it and the agent never did. Find where the agent stopped short: it refused
    something the policy allows, asked for a confirmation and never acted on it, transferred, or ran out of steps.
  - *wrong write*: the agent changed it and gold leaves it alone. Find the rule it broke: a write the policy
    forbids, one the customer never asked for, a second write where one was due.
  - *wrong arguments*: both changed it, differently. The field rows say which value (a destination, a cabin, a
    payment method, an amount); trace where the agent took it from: the customer's words, a tool result, its
    own arithmetic.
- **EXPECTED ACTIONS AGAINST THE AGENT'S CALLS** names the call behind each difference: a missing write with the
  agent's nearest call of that tool and the arguments that differ, and every write no expected action matches.
  When ACTION is not in the reward basis a missing read changes nothing: do not add a rule to force a lookup gold
  happened to make.
- **COMMUNICATE / NL ASSERTION UNMET**: the database can match and the conversation still fail on what was said.
- A difference the simulated user caused (it broke its instructions, or ended the call before the agent could
  act) is not the agent's: say so and change nothing for it.
The values in these differences are this task's answer: never copy a customer, id, date or amount from them into
a surface. Fix the rule or the mechanism that produced the wrong value; the test split has other customers."""


def _record_section(ctx: Context) -> str:
    """The champion's challengers, task by task: what to carry forward and what to drop."""
    if not ctx.record:
        return ""
    return f"""
# What the champion's challengers moved
Every challenger of this champion was an experiment on the same tasks. Read this before you edit: one trial per
task means a task can pass in one run and fail in the next with nothing changed, so a fix or a break counts only
against every run of the champion.
- A **real fix** (the champion fails the task in every run, a challenger passes it): carry the change behind it
  into your version, from that challenger's folder.
- A **real break** (the champion passes the task in every run, a challenger fails it): find in its conversation
  the edit that caused it, and leave that edit out or narrow it. Do not carry a real break into your version.
- A task the champion itself **flips** on is neither: credit and blame nothing to it.
{ctx.record}
"""


def _where_failed(ctx: Context) -> str:
    if ctx.halves:
        return (
            f"It failed the conversations below on the read half of the train split ({len(ctx.halves[0])} tasks, "
            "the ones you may learn from)."
        )
    return "It failed the conversations below on the train split."


def _reading(ctx: Context, new_name: str) -> str:
    return (
        f"The policy is at `agents/{ctx.domain}/{new_name}/.context/policy.md` and the tool list with descriptions at "
        f"`.context/tools.json` in the same folder. `runs/`, `data/`, `loop/` and `vendor/` are closed to you: the "
        "failures below are all the task data you get, and the traces they came from are the only runs you may read."
        + (
            " The knowledge base the agent searches is copied read-only to `.context/kb/` in the same folder, one file "
            "per document named by its id: read the documents a failure needed before you write a rule, and cite "
            "the document, not the graded values."
            if ctx.kb
            else ""
        )
    )


def _gate(ctx: Context, new_name: str) -> str:
    rule = (
        f"it promotes when `{new_name}` fixes at least one task and breaks none, or when a one-sided exact "
        "(McNemar) test on the tasks that changed gives p < 0.05 (five fixes with no break; seven with one "
        "break; nine with two)"
    )
    if ctx.domain in GATE_ON_TEST:
        return (
            f"The harness evaluates `{new_name}` on the train split, then on the held-out test split "
            f"({len(split_ids(ctx.domain, 'test'))} tasks, none of them among the failures above, never shown to "
            f"an optimiser), and applies the gate there, against the champion's test run on the same tasks: {rule}. "
            "It records the outcome next to your diagnosis in the ledger. A rule that only fits the conversations "
            "above will not pass it: the test split has other customers and other requests."
        )
    where = (
        f"on the {len(ctx.halves[1])} train tasks of the gate half, which are not among the failures above and are "
        "never shown to an optimiser"
        if ctx.halves
        else "on the same train tasks"
    )
    return (
        f"The harness evaluates `{new_name}` on the train split and applies the gate {where}: {rule}. It records the "
        f"outcome next to your diagnosis in the ledger, then runs `{new_name}` on the held-out test split for the record."
    )


def _surfaces_of(ctx: Context, mode: str) -> tuple[str, ...]:
    """What the dataset's profile lets `mode` edit (today's, without a profile)."""
    return ctx.profile.surfaces_for(mode) if ctx.profile else MODE_SURFACES[mode]


def _narrowed(ctx: Context, mode: str) -> str:
    """A profile that narrows its mode's surfaces says so beside the shared text; one that keeps
    today's adds nothing, so the prompt stays as it was."""
    allowed = _surfaces_of(ctx, mode)
    closed = [n for n in MODE_SURFACES[mode] if n not in allowed]
    if not closed:
        return ""
    names = ", ".join(f"`{n}`" for n in allowed)
    shut = ", ".join(f"`{n}`" for n in closed)
    if mode == "classic":
        return (
            f"\nThis dataset's optimiser profile narrows that to {names}: the harness refuses a write "
            f"to {shut}."
        )
    return (
        f"\nThis dataset's optimiser profile lets a diagnosis route only to {names}: {shut} "
        f"{'is' if len(closed) == 1 else 'are'} closed, and a fix routed there is dropped."
    )


def _guide_section(ctx: Context) -> str:
    """The dataset's own lessons (`optimisers/<domain>/guide.md`), after the shared prompt and
    under their own heading; nothing at all when the guide is empty, as every default is."""
    guide = ctx.profile.guide.strip() if ctx.profile else ""
    if not guide:
        return ""
    return (
        "\n# This dataset's lessons\n"
        f"What `{ctx.domain}`'s optimiser has been taught beyond the shared method above, kept for this "
        "dataset alone by the loop's owner. Hold every diagnosis and every edit to them.\n\n"
        f"{guide}\n"
    )


def build_prompt(
    champion: AgentVersion,
    new_name: str,
    run_id: str,
    failures: list[TaskResult],
    ctx: Context | None = None,
) -> str:
    """The classic optimiser: one session that diagnoses and edits `system.md` and `helper.py`."""
    ctx = ctx or build_context(champion, run_id, failures)
    domain = ctx.domain
    return f"""{_intro(champion, ctx)}

The champion is `agents/{domain}/{champion.name}/`. {_where_failed(ctx)}{ctx.fork_note}
A copy of the champion is already at `agents/{domain}/{new_name}/`. Your job is to turn that copy into a better
version by editing **only two files**: `agents/{domain}/{new_name}/system.md` (the agent's system prompt; the
literal `{{policy}}` is replaced by the domain policy at run time — keep it) and
`agents/{domain}/{new_name}/helper.py` (optional deterministic hooks the harness calls around the model:
`on_tool_call(name, arguments) -> (name, arguments)` to normalise arguments before a tool runs,
`on_reply(text) -> text` to post-process a message to the user, `extra_context(policy) -> str` to append text
to the system prompt). `agent.yaml` is frozen; do not touch it, and do not edit anything outside
`agents/{domain}/{new_name}/` — the harness refuses the cycle if you do. You may not run a simulation or call a
model; verify helper code with `uv run python -c "..."` on the example arguments from the traces.{_narrowed(ctx, "classic")}

{_reading(ctx, new_name)} Read the policy sections a failure cites before deciding a root cause.

# What was tried before
{ctx.history}

# Held challengers of this champion (their folders still exist)
{ctx.held_block}

# The champion's surfaces
## agents/{domain}/{champion.name}/system.md
{champion.system_prompt}

## agents/{domain}/{champion.name}/helper.py
```python
{champion.helper or "(no helper yet)"}
```

# Reading a failure
{READING_GUIDE}

# The failures ({len(failures)} of the train split)
{chr(10).join(ctx.blocks) or "none"}
{_record_section(ctx)}
# Method
0. Start from what the champion's challengers moved, when that section is there: your version begins as the
   champion plus every change behind a real fix and none behind a real break.
1. For each failed conversation, start from the database difference and the expected actions against the
   agent's calls, then find in the transcript why the agent made that call or skipped it. The root cause: a policy
   rule the agent skipped or misread (confirmation before a write, an eligibility condition, an order of
   operations), a tool called with wrong or missing arguments, a required fact never stated to the user, the
   agent refusing something the policy allows or allowing something it forbids, a user-simulator turn the agent
   mishandled, a conversation that ran out of steps, a JSON-contract slip (text and tool calls in one reply).
   Classify each as a prompt problem or a helper problem. A task that failed for a reason already tried and not
   fixed needs a different fix, not the same one again.
2. Make the change in the two surfaces. Prefer short, concrete prompt rules that name the policy clause and the
   exact tool sequence (which tool first, what to confirm, when to stop); prefer helper hooks for mechanical
   argument fixes (date formats, id casing, enum values). Keep the reply contract: one JSON object per turn,
   either a message or tool calls. Do not hard-code any task's answer, user name or ids: the test split has
   different customers with the same policy.
3. Do not regress: keep everything that made the champion pass its other tasks.
4. Finish by writing `agents/{domain}/{new_name}/diagnosis.json` with exactly this shape:
{{
  "diagnoses": [
    {{"task_id": "…", "symptom": "…",
      "graded_difference": "the record and field that differ and the call behind them, e.g. 'reservations.X status: gold cancels it (expected action N), the agent never called cancel_reservation' — or 'database matched: <the unmet component>'",
      "root_cause": "…", "surface": "system.md|helper.py|both",
      "change": "one sentence", "verified_in_session": true|false, "verification": "what you ran and saw"}}
  ],
  "prompt_diff_summary": "what changed in system.md, one line",
  "helper_diff_summary": "what changed in helper.py, one line",
  "expected_to_fix": ["task ids"],
  "risks": ["what might regress and why you think it will not"],
  "carried_forward": [{{"from": "vN", "what": "the challenger's edit you kept", "evidence": "the real fix it made"}}],
  "dropped": [{{"from": "vN", "what": "the challenger's edit you left out or narrowed", "evidence": "the real break it caused: task and transcript line"}}],
  "changes": [
    {{"file": "system.md|helper.py", "anchor": "the function name, or the first five words of the edited block",
      "what": "what the edit does, one sentence", "why": "the evidence that made you do it — a transcript line, a policy clause",
      "task_ids": ["tasks this edit is for"]}}
  ]
}}
`changes` is the change log: one entry per distinct edit, so a reader looking at the diff can find the reason next
to the hunk. Then stop. {_gate(ctx, new_name)}
""" + _guide_section(ctx)


SURFACE_GUIDE = """- `system.md` — the system prompt, once per conversation. For a rule the agent did not know or misread.
- `helper.py` — `on_tool_call(name, arguments) -> (name, arguments)`, `on_reply(text) -> text`,
  `extra_context(policy) -> str`. For mechanical slips: date formats, id casing, enum values.
- `checks.py` — `def check_write(name: str, arguments: dict, state: dict) -> str | None`, called for every call
  to a write tool ({writes}) before tau2 runs it. Return None to let it through, or one short sentence saying
  what is wrong: the call is then not run, the agent reads your sentence as the call's result and replies once
  more, and that reply goes through unchecked. For the right rule applied with the wrong arguments, or a write
  out of order. Check arguments against facts in `state`, never against one task's answer. The same file may also
  define `def check_reply(text: str, state: dict) -> str | None`, called on every text reply to the customer
  before it is sent: a sentence sends the reply back once with it (the model writes again, unchecked). For a rule
  about what the agent says or refuses, e.g. a transfer refused after the customer has asked four times.
- `memory.py` — `def remember(state: dict, name: str, arguments: dict, result: str) -> None`, called after every
  tool result (the call's name and arguments, the tool's text output, usually JSON) and every user message
  (`name == "user"`, `arguments == {{}}`). Update `state` in place. It starts empty in each conversation and is
  what `checks.py` and `guidance.py` read. For facts lost over a long conversation.
- `guidance.py` — `def guidance(state: dict, trigger: str) -> str | None`, called before every model call;
  `trigger` is `"user"` after a user message and `"tool"` after tool results. Return a short reminder (cut to
  600 characters) or None; it is added to that one call only, never to the transcript. For steps skipped in a
  multi-step procedure.
The three code surfaces run inside every conversation: standard-library imports only ({imports}), no files,
no network, no model calls; the harness swallows an exception, so a hook that raises does nothing."""

ROUTING_RULES = """| root cause | surface | not this |
|---|---|---|
| the agent didn't know a rule, or misread it | system.md | a rule restated that the agent already follows |
| a format slip in an argument | helper.py | a prompt line about formats |
| the right rule, the wrong arguments or order at a write | checks.py (+ memory.py for the facts it checks) | a check that blocks every attempt |
| lost a fact or a step in the middle of a procedure | memory.py + guidance.py | a longer procedure in the prompt |
| the task or the simulated user is at fault | none: say so, change nothing | any edit |"""


def _surfaces_text(domain: str) -> str:
    kinds = tool_kinds(domain)
    writes = (
        ", ".join(
            [n for n, k in kinds.items() if k == "write"] + sorted(CHECKED_TOOLS & set(kinds))
        )
        or "none known"
    )
    return SURFACE_GUIDE.format(
        writes=writes, imports=", ".join(sorted(ALLOWED_IMPORTS - {"__future__"}))
    )


def _champion_files(champion: AgentVersion) -> str:
    out = []
    for name in SURFACES:
        text = champion.files().get(name)
        if name == "system.md":
            out.append(f"## agents/{champion.domain}/{champion.name}/system.md\n{text}")
        else:
            out.append(
                f"## agents/{champion.domain}/{champion.name}/{name}\n```python\n{text or '(none yet)'}\n```"
            )
    return "\n\n".join(out)


def build_diagnose_prompt(
    champion: AgentVersion, new_name: str, failures: list[TaskResult], ctx: Context
) -> str:
    """The routing optimiser's first session: read-only, one root cause and its surfaces per failure."""
    domain = ctx.domain
    return f"""{_intro(champion, ctx)}

The champion is `agents/{domain}/{champion.name}/`. {_where_failed(ctx)}{ctx.fork_note}
This session is the **diagnosis** step. For each failed conversation, find the root cause and decide where the
fix belongs. You write one file, `agents/{domain}/{new_name}/diagnosis.json`; a second session then makes the
edits, and the harness lets it write only the surfaces you name. You may not run a simulation or call a model.

{_reading(ctx, new_name)} Read the policy sections a failure cites before deciding a root cause.

# The five surfaces a version may have
{_surfaces_text(domain)}

# Where a fix belongs
{ROUTING_RULES}
Prefer the fewest surfaces that fix the failures you can explain. A fix that lives in code is checked every
turn; a rule in the prompt competes with every other rule. Name `none` for a failure that is the task's or the
simulated user's fault, and change nothing for it.{_narrowed(ctx, "routing")}

# Reading a failure
{READING_GUIDE}

# What was tried before
{ctx.history}

# Held challengers of this champion (their folders still exist)
{ctx.held_block}

# The champion's surfaces
{_champion_files(champion)}

# The failures ({len(failures)})
{chr(10).join(ctx.blocks) or "none"}
{_record_section(ctx)}
# Finish
Write `agents/{domain}/{new_name}/diagnosis.json` with exactly this shape, then stop:
{{
  "diagnoses": [
    {{"task_id": "…", "symptom": "…",
      "graded_difference": "the record and field that differ and the call behind them — or 'database matched: <the unmet component>'",
      "root_cause": "…",
      "class": "knowledge|format|write-arguments|lost-state|task-fault",
      "surfaces": ["checks.py", "memory.py"], "why_this_surface": "…",
      "evidence": "a transcript line or a policy clause", "change": "one sentence"}}
  ],
  "surfaces": {{"checks.py": "what to add or change there, and for which tasks"}},
  "expected_to_fix": ["task ids"],
  "risks": ["what might regress and why you think it will not"],
  "carried_forward": [{{"from": "vN", "what": "the challenger's edit to keep", "evidence": "the real fix it made"}}],
  "dropped": [{{"from": "vN", "what": "the challenger's edit to leave out or narrow", "evidence": "the real break it caused"}}]
}}
`surfaces` has one key per file to change (from {", ".join(SURFACES)}); those keys are the only files the next
session may write. Do not hard-code any task's answer, customer name or id anywhere: the test split has other
customers. {_gate(ctx, new_name)}
""" + _guide_section(ctx)


def build_write_prompt(
    champion: AgentVersion,
    new_name: str,
    failures: list[TaskResult],
    ctx: Context,
    diagnosis: dict[str, Any],
    routed: list[str],
) -> str:
    """The routing optimiser's second session: the edits, to the routed surfaces only."""
    domain = ctx.domain
    base = len(champion.system_prompt)
    return f"""{_intro(champion, ctx)}

This session is the **writing** step. A diagnosis session has read the failures below and routed the fixes to
these files: {", ".join(f"`{r}`" for r in routed)}. A copy of the champion `{champion.name}` is at
`agents/{domain}/{new_name}/`. Make the diagnosed changes there. You may write only those files and
`agents/{domain}/{new_name}/changes.json`; `agent.yaml` and `diagnosis.json` are frozen, and the harness
refuses the cycle if anything else changes. You may not run a simulation or call a model; verify every code
surface with `uv run python -c "..."`, importing it from the version folder and calling its hook on arguments
and results taken from the transcripts below.

{_reading(ctx, new_name)}

# The surfaces and their contracts
{_surfaces_text(domain)}

# Reading a failure
{READING_GUIDE}

# Budgets and guards the harness applies to what you write
- `system.md` may grow by at most {PROMPT_GROWTH} characters over the champion's ({base} now, so at most
  {base + PROMPT_GROWTH}). Keep the literal `{{policy}}`.
- A code surface may import only the modules above and may not call open, exec or eval.
- No file may contain a customer id, name or email from any task.

# The diagnosis
```json
{json.dumps(diagnosis, indent=1, ensure_ascii=False)}
```

# The champion's surfaces
{_champion_files(champion)}

# The failures ({len(failures)})
{chr(10).join(ctx.blocks) or "none"}
{_record_section(ctx)}
# Finish
Write `agents/{domain}/{new_name}/changes.json` with exactly this shape, then stop:
{{
  "changes": [
    {{"file": "one of the routed files", "anchor": "the function name, or the first five words of the edited block",
      "what": "what the edit does, one sentence", "why": "the evidence — a transcript line, a policy clause",
      "task_ids": ["tasks this edit is for"], "verified_in_session": true, "verification": "what you ran and saw"}}
  ],
  "prompt_diff_summary": "what changed in system.md, one line (or none)",
  "helper_diff_summary": "what changed in the code surfaces, one line (or none)"
}}
{_gate(ctx, new_name)}
""" + _guide_section(ctx)


def _settings_note(changes: list[str]) -> str:
    """What the harness set in the new version's agent.yaml (`make optimise`), for every session."""
    if not changes:
        return ""
    return (
        "\n# The new version's agent settings\nThe harness has already written the new version's "
        f"`agent.yaml` with {'; '.join(changes)} (the failures above ran on the champion's). It is "
        "frozen: do not edit it, and write for the agent as it will run.\n"
    )


def _broken_trace_paths(outcome: dict[str, Any]) -> list[str]:
    """Map each broken task id to its own trace file in the challenger's run — the read half's
    breaks only, where train is halved."""
    broken = seen_changes(outcome)[1]
    challenger_run = outcome.get("challenger_run")
    if not challenger_run or not broken:
        return []
    try:
        results = read_results(RUNS_DIR / str(challenger_run) / "results.jsonl")
    except (OSError, FileNotFoundError):
        return []
    by_task = {r.task_id: r.trace for r in results}
    return [f"runs/{challenger_run}/traces/{by_task[tid]}" for tid in broken if tid in by_task]


# hand-made forks (`versions.fork_version`): their surfaces are copies of their source's
FORK_KINDS = ("model swap", "tool change")


def held_challengers(domain: str, champion_name: str) -> list[dict[str, Any]]:
    """Earlier challengers of this champion — or of the version it was forked from — that the
    gate held: real progress the next version may reuse. A held fork (a model swap or a tool
    change) is not: its surfaces are its source's, so there is nothing in them to copy."""
    names = set(lineage(domain, champion_name))
    out: list[dict[str, Any]] = []
    for e in read_ledger(domain):
        o = e.get("outcome") or {}
        if (
            e.get("champion") in names
            and o.get("verdict") == "hold"
            and e.get("kind") not in FORK_KINDS
            and e.get("challenger")
            and version_dir(domain, str(e["challenger"])).exists()
        ):
            on_test = str(o.get("gate_on") or "").startswith("test")
            out.append(
                {
                    "cycle": e.get("cycle"),
                    "challenger": e["challenger"],
                    "champion": e.get("champion"),
                    # the reason of a gate on the gate half or on test names its ids: its counts
                    # stand in, and the fixes and breaks beside them are the train moves
                    "reason": (
                        f"the gate's test passes {o.get('passes')}"
                        if on_test
                        else f"gate-half passes {o.get('passes')}"
                        if "read_fixed" in o
                        else o.get("reason")
                    ),
                    "passes": f"{o.get('train_passes')} on train" if on_test else o.get("passes"),
                    "fixed": seen_changes(o)[0],
                    "broken": seen_changes(o)[1],
                    "broken_traces": _broken_trace_paths(o),
                }
            )
    return out


RECORD_TRACE_CHARS = 4000  # a conversation a challenger broke: shorter than a failure's own


def _cell(pn: tuple[int, int] | None) -> str:
    """A task in one run: ✓ every trial passed, ✗ none did, p/n some did, – not scored."""
    if not pn or not pn[1]:
        return "–"
    p, n = pn
    return "✓" if p == n else "✗" if p == 0 else f"{p}/{n}"


@dataclass
class _Run:
    label: str
    meta: Any  # RunMeta
    rows: dict[str, list[TaskResult]]  # task id → its trials
    cells: dict[str, str]
    entry: dict[str, Any] | None = None  # the ledger's cycle, for a challenger


def _run_column(
    label: str, run_id: str, ids: set[str], entry: dict[str, Any] | None = None
) -> _Run | None:
    from tau2_loop.eval.compare import pass_fractions
    from tau2_loop.eval.runner import load_run

    try:
        meta, results = load_run(run_id)
    except (FileNotFoundError, OSError, ValueError):
        return None
    if not meta.summary or not ids <= set(meta.task_ids):
        return None
    rows: dict[str, list[TaskResult]] = {}
    for r in results:
        if r.task_id in ids:
            rows.setdefault(r.task_id, []).append(r)
    fr = pass_fractions([r for rs in rows.values() for r in rs])
    return _Run(label, meta, rows, {t: _cell(fr.get(t)) for t in ids}, entry)


def _task_order(t: str) -> tuple[int, str]:
    return (int(t), t) if t.isdigit() else (1 << 30, t)


def champion_record(
    champion: AgentVersion, run_id: str, failures: list[TaskResult]
) -> tuple[str, list[Path]]:
    """Every train run of the champion's bytes and of each challenger the ledger scored against
    it, task by task, over the tasks an optimiser may read (the read half where train is halved):
    which moves were real and which the champion's own runs make too, what each challenger
    changed, and the conversations it broke with what tau2 graded. ("", []) when there is only
    the one run of the champion to show."""
    from tau2_loop.eval.runner import SIM_RULES, list_runs, load_run

    domain, name = champion.domain, champion.name
    try:
        _, read_results = load_run(run_id)
    except (FileNotFoundError, OSError, ValueError):
        return "", []
    h = halves(domain)
    ids = {r.task_id for r in read_results} & (
        set(h[0]) if h else {r.task_id for r in read_results}
    )
    if not ids:
        return "", []
    champ_ids = [
        m.run_id
        for m in list_runs(domain)
        if m.agent == name
        and m.fingerprint == champion.fingerprint
        and m.split == "train"
        and not m.dry_run
        and m.run_id != run_id
    ]
    champ_ids = sorted({*champ_ids, run_id})
    champs = [
        c
        for i, rid in enumerate(champ_ids)
        if (c := _run_column(f"{name}·{i + 1}" if len(champ_ids) > 1 else name, rid, ids))
    ]
    challs = [
        c
        for e in read_ledger(domain)
        if e.get("champion") == name
        and e.get("challenger")
        and (rid := (e.get("outcome") or {}).get("challenger_run"))
        and (c := _run_column(str(e["challenger"]), str(rid), ids, e))
    ]
    if len(champs) + len(challs) < 2:
        return "", []
    cols = champs + challs
    label_of = {c.meta.run_id: c.label for c in cols}

    def kind(t: str) -> str:
        cells = {c.cells[t] for c in champs}
        return "pass" if cells == {"✓"} else "fail" if cells == {"✗"} else "flips"

    reading: dict[str, str] = {}
    shown = []
    for t in sorted(ids, key=_task_order):
        cells = [c.cells[t] for c in cols]
        if set(cells) == {"✓"}:
            continue
        shown.append(t)
        k = kind(t)
        if k == "pass":
            broke = [c.label for c in challs if c.cells[t] != "✓"]
            reading[t] = f"{name} passes it in every run: a real break by {', '.join(broke)}"
        elif k == "fail":
            fixed = [c.label for c in challs if c.cells[t] == "✓"]
            reading[t] = (
                f"{name} fails it in every run: a real fix by {', '.join(fixed)}"
                if fixed
                else f"fails in every run of {name} and of every challenger"
            )
        else:
            reading[t] = (
                f"{name} itself passes and fails it: a move here is as likely luck as a change"
            )
    old = [c for c in cols if c.meta.sim_rules != SIM_RULES]
    n_train = len(ids)
    lines = [
        f"`{name}` has {len(champs)} scored train run{'s' if len(champs) != 1 else ''} and "
        f"{len(challs)} challenger{'s' if len(challs) != 1 else ''} the ledger scored against it, one "
        f"column each, over the {n_train} train tasks you may read. Columns, oldest first:"
    ]
    for c in cols:
        passed = sum(1 for t in ids if c.cells[t] == "✓")
        what = ""
        if c.entry is not None:
            e, o = c.entry, c.entry.get("outcome") or {}
            gate_run = str(e.get("champion_run") or "")
            what = (
                f" · cycle {e.get('cycle')}, "
                + (
                    f"a {swap_note(e)}, no optimiser"
                    if e.get("kind")
                    else f"the {e.get('optimiser_mode') or 'classic'} optimiser"
                )
                + f", verdict {o.get('verdict', 'pending')}, "
                + (
                    f"its gate compared its test run with {e.get('champion')}'s (test passes {o.get('passes')})"
                    if str(o.get("gate_on") or "").startswith("test")
                    else "its gate compared it with " + label_of.get(gate_run, f"runs/{gate_run}")
                )
            )
        elif c.meta.run_id == run_id:
            what = " · the run whose failures are listed above"
        lines.append(
            f"- {c.label} = runs/{c.meta.run_id} ({model_label(c.meta.model)}; {passed}/{n_train} "
            f"passed every trial; {c.meta.trials} trial{'s' if c.meta.trials != 1 else ''})"
            + (" †" if c in old else "")
            + what
        )
    if old:
        lines.append(
            "† ran before the simulation rules of 2 Oct 2026: tau2's own customer, which could end the "
            'call on its "yes" before the agent acted, and both sides were told the real date. A move '
            "seen only in † runs may be the old customer's, not the agent's."
        )
    lines += [
        "",
        "| task | " + " | ".join(c.label for c in cols) + " | reading |",
        "|---|" + "---|" * len(cols) + "---|",
    ]
    lines += [
        f"| {t} | " + " | ".join(c.cells[t] for c in cols) + f" | {reading[t]} |" for t in shown
    ]
    lines.append(f"The other {n_train - len(shown)} tasks passed in every run.")

    traces: list[Path] = []
    for c in cols:
        for t in shown:
            traces += [
                (RUNS_DIR / c.meta.run_id / "traces" / r.trace).resolve()
                for r in c.rows.get(t, [])
                if r.trace
            ]
    failed_ids = {r.task_id for r in failures}
    scenario_shown: set[str] = set()
    for c in challs:
        e = c.entry or {}
        o = e.get("outcome") or {}
        swap = bool(e.get("kind"))
        real_fix = [t for t in shown if kind(t) == "fail" and c.cells[t] == "✓"]
        real_break = [t for t in shown if kind(t) == "pass" and c.cells[t] != "✓"]
        noise = [f"{t} {c.cells[t]}" for t in shown if kind(t) == "flips"]
        lines += [
            "",
            f"## {c.label} — cycle {e.get('cycle')}, verdict {o.get('verdict', 'pending')}",
        ]
        if swap:
            lines.append(
                f"A {swap_note(e)}: `{name}`'s surfaces on another model. What moved here is the model "
                "(and luck); there is no surface in its folder to copy."
            )
        else:
            lines.append(
                f"Its folder `agents/{domain}/{c.label}/` is on disk. What it changed (its own change log):"
            )
            try:
                diag = json.loads((version_dir(domain, c.label) / "diagnosis.json").read_text())
            except (OSError, ValueError):
                diag = {}
            changes = diag.get("changes") if isinstance(diag, dict) else None
            if changes:
                lines += [
                    f"- {ch.get('file')} · {ch.get('anchor')}: {ch.get('what')} (for tasks {ch.get('task_ids')})"
                    for ch in changes
                    if isinstance(ch, dict)
                ]
            else:
                lines += [
                    f"- prompt: {e.get('prompt_diff_summary', '')}",
                    f"- helper: {e.get('helper_diff_summary', '')}",
                ]
        if real_fix:
            lines.append(
                f"Real fixes (every run of `{name}` fails them, {c.label} passes): {', '.join(real_fix)}. "
                + (
                    "The other model made them, not a surface: read the passing conversation for what "
                    "it did differently, and write that down as a rule only if the policy supports it:"
                    if swap
                    else "Keep the change behind each:"
                )
            )
            diagnosed = {str(d.get("task_id")): d for d in e.get("diagnoses") or []}
            for t in real_fix:
                d = diagnosed.get(t)
                said = (
                    f"its diagnosis: {d.get('root_cause')} → {d.get('change')}"
                    if d
                    else "no edit was made for it."
                    if swap
                    else "it did not diagnose this task: an edit made for another task fixed it."
                )
                tr = next((r.trace for r in c.rows.get(t, []) if r.correct), "")
                lines.append(
                    f"- task {t}: {said} Its passing conversation: runs/{c.meta.run_id}/traces/{tr}"
                )
        if real_break:
            lines.append(
                f"Real breaks (every run of `{name}` passes them, {c.label} fails): {', '.join(real_break)}. "
                + (
                    "Each follows, with what tau2 graded."
                    if swap
                    else "Each follows with what tau2 graded and the conversation: find the edit that caused it, "
                    "and drop or narrow that edit rather than keep it."
                )
            )
        if noise:
            lines.append(
                f"Where `{name}` itself flips ({', '.join(noise)} here): neither a fix nor a break."
            )
        if not (real_fix or real_break):
            lines.append(f"No task moved that `{name}`'s own runs do not move too.")
        for t in real_break:
            r = next((x for x in c.rows.get(t, []) if not x.correct), None)
            if r is None:
                continue
            tp = RUNS_DIR / c.meta.run_id / "traces" / r.trace
            diffs = diff_block(c.meta.run_id, r, domain)
            what = failure_details(tp, actions=diffs is None) + (f"\n{diffs}" if diffs else "")
            if t in scenario_shown or t in failed_ids:
                scenario = f"(task {t}'s scenario is above)"
            else:
                scenario = task_block(domain, t)
                scenario_shown.add(t)
            lines += [
                "",
                f"### {c.label} broke task {t} (trial {r.trial}) — runs/{c.meta.run_id}/traces/{r.trace}",
                scenario,
                f"WHAT FAILED:\n{what}",
                f"ended by {r.termination_reason} · {r.n_agent_turns} agent turns · {r.n_tool_calls} tool calls",
            ]
            if not swap:
                lines.append(f"TRANSCRIPT:\n{condense_trace(tp, RECORD_TRACE_CHARS)}")
    return "\n".join(lines), traces


def _copy_champion(champion: AgentVersion, new_name: str) -> Path:
    new_dir = version_dir(champion.domain, new_name)
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True)
    for name in SURFACES + FROZEN:
        src = champion.path / name
        if src.exists():
            shutil.copyfile(src, new_dir / name)
    return new_dir


def _run_policy(run_id: str | None) -> str | None:
    """The policy the champion's agent read in its own run (tau2 records it per conversation): its
    retrieval variant's text, naming the tools it had, not the extract's default variant's."""
    if not run_id:
        return None
    try:
        sims = json.loads((RUNS_DIR / run_id / "tau2_results.json").read_text()).get("simulations")
    except (OSError, ValueError):
        return None
    return next((str(x["policy"]) for x in sims or [] if x.get("policy")), None)


def _version_tools(champion: AgentVersion | None, extract_tools: list[Any]) -> list[Any]:
    """The extract's tools with banking's default knowledge tools swapped for the champion's
    variant's (as the Domains page shows them); the extract's own when that cannot be read."""
    if champion is None or not champion.retrieval or champion.retrieval == BANKING_RETRIEVAL:
        return list(extract_tools)
    try:
        from tau2_loop.eval import retrieval

        rows = retrieval.variant_tool_rows(champion.retrieval)
        default = set(retrieval.variant_tools(BANKING_RETRIEVAL))
    except Exception:  # noqa: BLE001 - without tau2's spec the extract's list still helps
        return list(extract_tools)
    if not rows:
        return list(extract_tools)
    return [*rows, *(t for t in extract_tools if t.get("name") not in default)]


def _kb_source(domain: str) -> Path | None:
    """The domain's knowledge-base documents in the tau2 checkout, or None (no documents, or no tau2)."""
    if domain != "banking_knowledge":
        return None
    from tau2_loop.data.documents import source_dir

    d = source_dir(domain)
    return d if d.is_dir() else None


def _copy_kb(domain: str, dest: Path) -> int:
    """Each document as `<id>.md` (title, id, text) under `dest`, for the session to read and grep."""
    src = _kb_source(domain)
    if src is None:
        return 0
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        (dest / f"{f.stem}.md").write_text(
            f"# {d.get('title') or f.stem}\n\nID: {f.stem}\n\n{d.get('content') or ''}\n"
        )
        n += 1
    return n


def _write_context(
    domain: str, new_dir: Path, champion: AgentVersion | None = None, run_id: str | None = None
) -> Path:
    """The policy, tools and (banking) knowledge base the session may read, copied beside the
    version: `data/`, `runs/` and `vendor/` are fenced. The policy and tools are the ones the
    champion's agent had in `run_id` (s14 P8a), not the extract's default variant's."""
    ctx_dir = new_dir / CONTEXT_DIR
    ctx_dir.mkdir(exist_ok=True)
    ext = read_task_extract(domain)
    policy = _run_policy(run_id) or str(ext.get("policy") or "")
    (ctx_dir / "policy.md").write_text(policy)
    tools = {
        "tools": _version_tools(champion, ext.get("tools") or []),
        "user_tools": ext.get("user_tools") or [],
    }
    (ctx_dir / "tools.json").write_text(json.dumps(tools, indent=1, ensure_ascii=False))
    _copy_kb(domain, ctx_dir / "kb")
    return ctx_dir


SESSION_TOOLS = ("Read", "Write", "Edit", "Bash", "Glob", "Grep")

GUARDED = (
    "agents",
    "src/tau2_loop",
    "tests",
    "data/splits",
    "data/tasks",
    "loop",
    "runs",
    "optimisers",  # each dataset's optimiser profile: the session follows it, never edits it
    "Makefile",
    "pyproject.toml",
)


def tree_checksum(exclude: Path) -> dict[str, int]:
    """mtime+size of every file the optimiser could cheat with, outside its own version folder."""
    out: dict[str, int] = {}
    for rel in GUARDED:
        base = ROOT / rel
        if not base.exists():
            continue
        files = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for p in files:
            if exclude in p.parents or p == exclude or "__pycache__" in p.parts:
                continue
            st = p.stat()
            out[str(p.relative_to(ROOT))] = int(st.st_mtime_ns) ^ st.st_size
    return out


@dataclass
class Session:
    turns: int = 0
    cost_usd: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


async def _run_session(
    prompt: str,
    *,
    label: str,
    domain: str,
    new_name: str,
    writable: set[Path],
    readable: set[Path],
    hidden: list[Path],
    model: str,
    effort: Effort,
    transcript: list[dict[str, Any]],
    where: str | None = None,
    max_turns: int = MAX_TURNS,
) -> Session:
    """One Agent SDK session inside the fence: it writes only `writable` and reads nothing fenced.
    `where` names the folder it writes in, for the refusal (an agent version's by default);
    `max_turns` is the dataset profile's turn limit (today's, for the judge's optimiser)."""
    names = ", ".join(sorted(p.name for p in writable))
    where = where or f"agents/{domain}/{new_name}/"

    def deny(reason: str) -> dict[str, Any]:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def guard_writes(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Refuse a Write/Edit to anything but this session's files."""
        path = str(input_data.get("tool_input", {}).get("file_path", ""))
        resolved = Path(path).resolve() if path else None
        if resolved is not None and resolved in writable:
            return {}
        return deny(f"This step may write only {where}: {names}.")

    async def guard_reads(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """Refuse a Read/Glob/Grep of a fenced folder or of this cycle's other challenger."""
        ti = input_data.get("tool_input", {}) or {}
        path = str(ti.get("file_path") or ti.get("path") or "")
        reason = fence_reason(path, readable, hidden)
        return deny(reason) if reason else {}

    async def guard_bash(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """No simulations, no model calls and no fenced folders from inside the optimiser session."""
        cmd = str(input_data.get("tool_input", {}).get("command", ""))
        banned = (
            "tau2loop eval",
            "tau2loop judge",
            "make eval",
            "make loop",
            "make judge",
            "tau2 run",
            "claude ",
            "make smoke",
        )
        if any(b in cmd for b in banned):
            return deny(
                "The harness runs the evaluation after you finish; do not run simulations or models here."
            )
        reason = bash_fence_reason(cmd, hidden)
        return deny(reason) if reason else {}

    options = ClaudeAgentOptions(
        model=resolve_model(model),
        effort=effort,
        # the only tools it has: a sub-agent (Agent/Task) ran outside the turn and the session
        # ended before writing its file (s15), so `tools` closes every other built-in
        tools=list(SESSION_TOOLS),
        allowed_tools=list(SESSION_TOOLS),
        strict_mcp_config=True,
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        cwd=str(ROOT),
        env=subscription_env(),
        setting_sources=[],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[guard_writes]),
                HookMatcher(matcher="Read|Glob|Grep", hooks=[guard_reads]),
                HookMatcher(matcher="Bash", hooks=[guard_bash]),
            ]
        },
    )
    out = Session()
    transcript.append({"role": "session", "content": label})
    transcript.append({"role": "user", "content": prompt})
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for blk in msg.content:
                        if isinstance(blk, TextBlock) and blk.text.strip():
                            transcript.append({"role": "assistant", "content": blk.text})
                        elif isinstance(blk, ToolUseBlock):
                            transcript.append(
                                {"role": "tool_use", "name": blk.name, "input": blk.input}
                            )
                elif isinstance(msg, ResultMessage):
                    out.turns = msg.num_turns
                    out.cost_usd = msg.total_cost_usd
                    u = msg.usage or {}
                    out.input_tokens = (
                        int(u.get("input_tokens", 0))
                        + int(u.get("cache_read_input_tokens", 0))
                        + int(u.get("cache_creation_input_tokens", 0))
                    )
                    out.output_tokens = int(u.get("output_tokens", 0))
                    if msg.is_error:
                        out.error = f"{msg.subtype}: {(msg.errors or [''])[0]}"[:500]
    except Exception as e:  # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"[:500]
    return out


def _add(out: OptimiserOutput, sess: Session) -> None:
    out.n_turns += sess.turns
    out.input_tokens += sess.input_tokens
    out.output_tokens += sess.output_tokens
    if sess.cost_usd is not None:
        out.cost_usd = (out.cost_usd or 0.0) + sess.cost_usd
    if sess.error:
        out.error = f"{out.error}; {sess.error}" if out.error else sess.error


def _fail(out: OptimiserOutput, msg: str, reject: bool = True) -> None:
    out.error = f"{out.error}; {msg}" if out.error else msg
    out.rejected = out.rejected or reject


def _read_json(path: Path, out: OptimiserOutput, what: str) -> dict[str, Any] | None:
    if not path.exists():
        _fail(out, f"no {what} written")
        return None
    try:
        d = json.loads(redact(path.read_text()))
    except json.JSONDecodeError as e:
        _fail(out, f"{what} unreadable: {e}")
        return None
    return d if isinstance(d, dict) else None


async def run_optimiser(
    champion: AgentVersion,
    run_id: str,
    failures: list[TaskResult],
    model: str | None = None,
    effort: Effort | None = None,
    mode: str | None = None,
    hidden: list[Path] | None = None,
    ctx: Context | None = None,
    profile: OptimiserProfile | None = None,
    settings: dict[str, Any] | None = None,
) -> OptimiserOutput:
    """`classic`: one session edits `system.md` and `helper.py`, as every cycle before s09.
    `routing`: a read-only diagnosis session names each failure's surfaces, then a writing
    session may change only those, among all five. Both run inside the fence; `hidden` closes
    an A/B partner's folder, and `ctx` lets a pair share the exact same input.

    `profile` is the dataset's optimiser (s13 §5; the context's, else loaded from
    `optimisers/<domain>/`): `model`, `effort` and `mode` left None are its own, and its turn
    limit, surfaces, guide and guards apply to every session and to the package.

    `settings` (`make optimise`, s14) gives the new version agent settings that differ from the
    champion's (`effort`, `model`, `parallel_calls`, …): the harness writes that `agent.yaml`
    before any session and freezes it there, and `diagnosis.json` records the change as
    `agent_yaml`. A loop cycle passes none, so a cycle still compares prompts and code only."""
    domain = champion.domain
    profile = profile or (ctx.profile if ctx else None) or load_profile(domain)
    model = model or profile.model
    effort = effort or profile.effort
    mode = mode or profile.mode
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    require_live()
    ctx = ctx or build_context(champion, run_id, failures, profile)
    new_name = next_version_name(domain)
    new_dir = _copy_champion(champion, new_name)
    yaml_changes: list[str] = []
    if settings:
        cfg = replace(champion.config, **settings)
        if cfg == champion.config:
            raise ValueError(f"settings {settings} change nothing in {champion.ref}'s agent.yaml")
        (new_dir / "agent.yaml").write_text(AGENT_YAML_HEADER + cfg.yaml())
        yaml_changes = [
            f"{k}: {getattr(champion.config, k)} → {getattr(cfg, k)}"
            for k in AgentConfig.__dataclass_fields__
            if getattr(champion.config, k) != getattr(cfg, k)
        ]
    frozen_yaml = (new_dir / "agent.yaml").read_bytes()
    before = tree_checksum(new_dir)
    ctx_dir = _write_context(domain, new_dir, champion, run_id)
    readable = set(ctx.read_traces) | {p.resolve() for p in ctx_dir.iterdir()}
    diag_path = (new_dir / "diagnosis.json").resolve()
    out = OptimiserOutput(new_version=new_name, diagnosis={}, mode=mode)
    hidden = [h.resolve() for h in hidden or []]
    started = time.time()
    common: dict[str, Any] = {
        "domain": domain,
        "new_name": new_name,
        "readable": readable,
        "hidden": hidden,
        "model": model,
        "effort": effort,
        "transcript": out.transcript,
        "max_turns": profile.max_turns,
    }
    allowed = profile.surfaces_for(mode)  # today: system.md + helper.py, or all five to route

    if mode == "classic":
        writable = {(new_dir / n).resolve() for n in allowed} | {diag_path}
        prompt = build_prompt(champion, new_name, run_id, failures, ctx) + _settings_note(
            yaml_changes
        )
        _add(out, await _run_session(prompt, label="classic", writable=writable, **common))
        out.diagnosis = _read_json(diag_path, out, "diagnosis.json") or {}
        out.routed = list(allowed)
    else:
        prompt = build_diagnose_prompt(champion, new_name, failures, ctx) + _settings_note(
            yaml_changes
        )
        _add(out, await _run_session(prompt, label="diagnose", writable={diag_path}, **common))
        diagnosis = _read_json(diag_path, out, "diagnosis.json") or {}
        routed = [n for n in allowed if n in (diagnosis.get("surfaces") or {})]
        out.routed = routed
        if not routed:
            _fail(out, "the diagnosis routed no surface")
        else:
            frozen = diag_path.read_bytes()
            changes_path = (new_dir / "changes.json").resolve()
            writable = {(new_dir / n).resolve() for n in routed} | {changes_path}
            prompt = build_write_prompt(
                champion, new_name, failures, ctx, diagnosis, routed
            ) + _settings_note(yaml_changes)
            _add(out, await _run_session(prompt, label="write", writable=writable, **common))
            if diag_path.read_bytes() != frozen:
                _fail(out, "diagnosis.json changed in the writing step")
            changes = _read_json(changes_path, out, "changes.json") or {}
            diagnosis.update(
                {
                    k: changes.get(k)
                    for k in ("changes", "prompt_diff_summary", "helper_diff_summary")
                }
            )
            diagnosis["routed"] = routed
            changes_path.unlink(missing_ok=True)
            diag_path.write_text(json.dumps(diagnosis, indent=2, ensure_ascii=False) + "\n")
        out.diagnosis = diagnosis
    out.duration_ms = int((time.time() - started) * 1000)
    shutil.rmtree(ctx_dir, ignore_errors=True)

    # the package: what changed, and the guards on it
    after = tree_checksum(new_dir)
    touched = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if touched:
        _fail(out, f"optimiser changed files outside agents/{domain}/{new_name}: {touched[:10]}")
    if (new_dir / "agent.yaml").read_bytes() != frozen_yaml:
        _fail(out, "agent.yaml was modified (frozen)")
    if yaml_changes:
        out.diagnosis["agent_yaml"] = yaml_changes
        diag_file = new_dir / "diagnosis.json"
        try:
            d = json.loads(diag_file.read_text())
            d["agent_yaml"] = yaml_changes
            diag_file.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n")
        except (OSError, ValueError):
            pass  # no diagnosis to annotate: the run is already failed for it
    expected = set(SURFACES + FROZEN) | {"diagnosis.json"}
    # importing helper.py to check it, as the prompt asks, compiles it: bytecode, not a stray file
    shutil.rmtree(new_dir / "__pycache__", ignore_errors=True)
    strays = sorted(p.name for p in new_dir.iterdir() if p.name not in expected)
    for name in strays:  # the write guard should make this impossible; a stray file never ships
        p = new_dir / name
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    if strays:
        _fail(out, f"stray files removed: {strays}")
    new_version = load_version(domain, new_name)
    mine, theirs = new_version.files(), champion.files()
    out.surfaces_changed = [n for n in SURFACES if mine.get(n) != theirs.get(n)]
    if not out.surfaces_changed:
        _fail(out, "no change to any surface")
    for name in CODE_SURFACES:
        if name in out.surfaces_changed and name in mine:
            bad = import_violations(mine[name])
            if bad:
                _fail(out, f"guard: {name} {'; '.join(bad)}")
    found = leaks({n: mine[n] for n in out.surfaces_changed if n in mine}, leak_values(domain))
    if found:
        _fail(out, f"guard: customer data in the version: {found[:5]}")
    where = f"optimisers/{domain}/guards.py"  # the dataset's own guards (s13 §5)
    for msg in profile_violations(profile.guards, mine, theirs, where):
        _fail(out, f"guard ({domain}'s optimiser profile): {msg}")
    if (
        mode == "routing"
        and len(new_version.system_prompt) > len(champion.system_prompt) + PROMPT_GROWTH
    ):
        _fail(
            out,
            f"guard: system.md grew {len(new_version.system_prompt) - len(champion.system_prompt)} "
            f"characters, over the {PROMPT_GROWTH} budget",
        )
    (new_dir / "optimiser_transcript.json").write_text(
        redact(json.dumps(out.transcript, ensure_ascii=False, indent=1))
    )
    redact_tree(new_dir)  # the optimiser quotes traces; the account email must not land in agents/
    return out


__all__ = [
    "DIFF_BUDGET",
    "MAX_TURNS",
    "MODES",
    "SCENARIO_BUDGET",
    "TRANSCRIPT_BUDGET",
    "run_optimiser",
    "build_prompt",
    "build_context",
    "build_diagnose_prompt",
    "build_write_prompt",
    "condense_trace",
    "OptimiserOutput",
    "AGENTS_DIR",
]
