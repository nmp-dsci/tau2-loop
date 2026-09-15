"""The optimiser: one Agent SDK session that reads every failure and writes the next version.

It sees the champion's two surfaces, every failed conversation's scenario,
policy clauses, expected actions, the reward components that failed and a
condensed transcript, plus the ledger's history of what was tried before. It
may write only `agents/<domain>/v(n+1)/system.md` and `helper.py` (a
PreToolUse hook refuses any other path; a checksum of the guarded tree is
compared after the session as well), it may not run a simulation, and it must
finish by writing `diagnosis.json` — the structured record the ledger stores.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
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

from tau2_loop.agent.versions import (
    SURFACES,
    AgentVersion,
    load_version,
    next_version_name,
    version_dir,
)
from tau2_loop.config import AGENTS_DIR, ROOT, RUNS_DIR
from tau2_loop.data.splits import read_task_extract
from tau2_loop.eval.results import TaskResult
from tau2_loop.llm import EFFORT, Effort, require_live, resolve_model, subscription_env
from tau2_loop.loop.ledger import read_ledger, render_history

MAX_TURNS = 120
TRACE_CHARS = 9000
TOOL_RESULT_CHARS = 700


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


def failure_details(trace_path: Path) -> str:
    """Which reward components failed, from the saved reward_info: expected vs matched actions, unmet checks."""
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
    for c in ri.get("action_checks") or []:
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


def task_block(domain: str, task_id: str) -> str:
    """Scenario, purpose, relevant policy clauses and evaluation criteria from the committed extract."""
    ext = read_task_extract(domain)
    t = next((x for x in ext.get("tasks", []) if x.get("id") == task_id), None)
    if not t:
        return "(task not in the committed extract)"
    desc = t.get("description") or {}
    sc = (t.get("user_scenario") or {}).get("instructions") or {}
    ev = t.get("evaluation_criteria") or {}
    parts = [
        f"PURPOSE: {desc.get('purpose')}",
        f"RELEVANT POLICIES: {desc.get('relevant_policies')}",
        f"NOTES: {desc.get('notes')}" if desc.get("notes") else "",
        f"USER'S GOAL (what the simulated user was told to do): {sc.get('task_instructions') or sc}",
        f"USER KNOWS: {sc.get('known_info')}" if sc.get("known_info") else "",
        f"USER DOES NOT KNOW: {sc.get('unknown_info')}" if sc.get("unknown_info") else "",
        f"EXPECTED ACTIONS: {json.dumps([{k: a.get(k) for k in ('name', 'arguments')} for a in ev.get('actions') or []], ensure_ascii=False)}",
        f"MUST COMMUNICATE: {ev.get('communicate_info')}" if ev.get("communicate_info") else "",
        f"NL ASSERTIONS: {ev.get('nl_assertions')}" if ev.get("nl_assertions") else "",
        f"REWARD BASIS: {ev.get('reward_basis')}",
    ]
    return "\n".join(p for p in parts if p)


def build_prompt(
    champion: AgentVersion, new_name: str, run_id: str, failures: list[TaskResult]
) -> str:
    domain = champion.domain
    run_dir = RUNS_DIR / run_id
    blocks: list[str] = []
    for r in failures:
        tp = run_dir / "traces" / r.trace
        blocks.append(
            f"## Task {r.task_id} (trial {r.trial})\n"
            f"{task_block(domain, r.task_id)}\n"
            f"WHAT FAILED:\n{failure_details(tp)}\n"
            f"ERROR: {r.error or 'none'} · {r.n_agent_turns} agent turns · {r.n_tool_calls} tool calls · ended by {r.termination_reason}\n"
            f"TRANSCRIPT:\n{condense_trace(tp)}\n"
        )
    held = held_challengers(domain, champion.name)
    held_block = (
        "\n".join(
            f"- `agents/{domain}/{h['challenger']}/` (cycle {h['cycle']}, held: {h['reason']}; passes {h['passes']}; fixed {h['fixed']}; "
            f"broke {h['broken']}). Its system.md and helper.py are on disk: read them and copy what held — a held "
            "challenger is a starting point, not a rejected one. Do not repeat what broke: read the traces of the broken "
            f"tasks ({', '.join(h.get('broken_traces') or []) or 'none'}) before you decide what to keep."
            for h in held
        )
        or "none"
    )
    ext = read_task_extract(domain)
    tool_names = ", ".join(t["name"] for t in ext.get("tools", []))
    return f"""You are the optimiser in a benchmark improvement loop for a customer-support agent on τ²-bench,
domain `{domain}`. The agent is Claude Haiku 4.5 on the Claude subscription; it follows a policy document and calls the
domain's tools ({tool_names}) through a JSON contract; a simulated user (also Haiku) plays the customer. A
conversation scores 1 only if every component in the task's reward basis passes: the final database equals the
gold one produced by the expected actions, the expected actions were called with the expected arguments, the
required information was said to the user, and (retail) an LLM judge finds the NL assertions met.

The champion is `agents/{domain}/{champion.name}/`. It failed the conversations below on the train split.
A copy of the champion is already at `agents/{domain}/{new_name}/`. Your job is to turn that copy into a better
version by editing **only two files**: `agents/{domain}/{new_name}/system.md` (the agent's system prompt; the
literal `{{policy}}` is replaced by the domain policy at run time — keep it) and
`agents/{domain}/{new_name}/helper.py` (optional deterministic hooks the harness calls around the model:
`on_tool_call(name, arguments) -> (name, arguments)` to normalise arguments before a tool runs,
`on_reply(text) -> text` to post-process a message to the user, `extra_context(policy) -> str` to append text
to the system prompt). `agent.yaml` is frozen; do not touch it, and do not edit anything outside
`agents/{domain}/{new_name}/` — the harness refuses the cycle if you do. You may not run a simulation or call a
model; verify helper code with `uv run python -c "..."` on the example arguments from the traces.

The policy is at `data/tasks/{domain}.json` (key `policy`) and the tool list with descriptions at key `tools`;
the tasks with their scenarios and expected actions at key `tasks`. Read the policy sections a failure cites
before deciding a root cause.

# What was tried before
{render_history(domain, [r.task_id for r in failures])}

# Held challengers of this champion (their folders still exist)
{held_block}

# The champion's surfaces
## agents/{domain}/{champion.name}/system.md
{champion.system_prompt}

## agents/{domain}/{champion.name}/helper.py
```python
{champion.helper or "(no helper yet)"}
```

# The failures ({len(failures)} of the train split)
{chr(10).join(blocks) or "none"}

# Method
1. For each failed conversation, find the root cause from the transcript and the failed components: a policy
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
    {{"task_id": "…", "symptom": "…", "root_cause": "…", "surface": "system.md|helper.py|both",
      "change": "one sentence", "verified_in_session": true|false, "verification": "what you ran and saw"}}
  ],
  "prompt_diff_summary": "what changed in system.md, one line",
  "helper_diff_summary": "what changed in helper.py, one line",
  "expected_to_fix": ["task ids"],
  "risks": ["what might regress and why you think it will not"],
  "changes": [
    {{"file": "system.md|helper.py", "anchor": "the function name, or the first five words of the edited block",
      "what": "what the edit does, one sentence", "why": "the evidence that made you do it — a transcript line, a policy clause",
      "task_ids": ["tasks this edit is for"]}}
  ]
}}
`changes` is the change log: one entry per distinct edit, so a reader looking at the diff can find the reason next
to the hunk. Then stop. The harness evaluates `{new_name}` on the same train tasks, applies the gate (a one-sided
McNemar test on the paired tasks: fixes must outweigh breaks with p < 0.05 — one break costs three extra fixes),
and records the outcome next to your diagnosis in the ledger.
"""


def held_challengers(domain: str, champion_name: str) -> list[dict[str, Any]]:
    """Earlier challengers of this champion that the gate held: real progress the next version may reuse."""
    out: list[dict[str, Any]] = []
    for e in read_ledger(domain):
        o = e.get("outcome") or {}
        if (
            e.get("champion") == champion_name
            and o.get("verdict") == "hold"
            and e.get("challenger")
            and version_dir(domain, str(e["challenger"])).exists()
        ):
            out.append(
                {
                    "cycle": e.get("cycle"),
                    "challenger": e["challenger"],
                    "reason": o.get("reason"),
                    "passes": o.get("passes"),
                    "fixed": o.get("fixed"),
                    "broken": o.get("broken"),
                    "broken_traces": [
                        f"runs/{o.get('challenger_run')}/traces/"
                        for _ in (o.get("broken") or [])[:1]
                    ]
                    if o.get("challenger_run")
                    else [],
                }
            )
    return out


def _copy_champion(champion: AgentVersion, new_name: str) -> Path:
    new_dir = version_dir(champion.domain, new_name)
    if new_dir.exists():
        shutil.rmtree(new_dir)
    new_dir.mkdir(parents=True)
    for name in ("system.md", "agent.yaml", "helper.py"):
        src = champion.path / name
        if src.exists():
            shutil.copyfile(src, new_dir / name)
    return new_dir


GUARDED = (
    "agents",
    "src/tau2_loop",
    "tests",
    "data/splits",
    "data/tasks",
    "loop",
    "runs",
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


async def run_optimiser(
    champion: AgentVersion,
    run_id: str,
    failures: list[TaskResult],
    model: str = "sonnet",
    effort: Effort = EFFORT,
) -> OptimiserOutput:
    require_live()
    domain = champion.domain
    new_name = next_version_name(domain)
    new_dir = _copy_champion(champion, new_name)
    allowed_prefix = str(new_dir.resolve())
    before = tree_checksum(new_dir)

    async def guard_writes(
        input_data: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        """Refuse a Write/Edit outside agents/<domain>/v(n+1)/ and any touch of agent.yaml."""
        path = str(input_data.get("tool_input", {}).get("file_path", ""))
        resolved = str(Path(path).resolve()) if path else ""
        ok = resolved.startswith(allowed_prefix) and not resolved.endswith("agent.yaml")
        if ok:
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Only agents/{domain}/{new_name}/system.md, helper.py and diagnosis.json may be written.",
            }
        }

    async def guard_bash(input_data: Any, tool_use_id: str | None, context: Any) -> dict[str, Any]:
        """No simulations and no model calls from inside the optimiser session."""
        cmd = str(input_data.get("tool_input", {}).get("command", ""))
        banned = ("tau2loop eval", "make eval", "make loop", "tau2 run", "claude ", "make smoke")
        if any(b in cmd for b in banned):
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "The harness runs the evaluation after you finish; do not run simulations or models here.",
                }
            }
        return {}

    options = ClaudeAgentOptions(
        model=resolve_model(model),
        effort=effort,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        max_turns=MAX_TURNS,
        cwd=str(ROOT),
        env=subscription_env(),
        setting_sources=[],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[guard_writes]),
                HookMatcher(matcher="Bash", hooks=[guard_bash]),
            ]
        },
    )
    prompt = build_prompt(champion, new_name, run_id, failures)
    out = OptimiserOutput(new_version=new_name, diagnosis={})
    out.transcript.append({"role": "user", "content": prompt})
    started = time.time()
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock) and b.text.strip():
                            out.transcript.append({"role": "assistant", "content": b.text})
                        elif isinstance(b, ToolUseBlock):
                            out.transcript.append(
                                {"role": "tool_use", "name": b.name, "input": b.input}
                            )
                elif isinstance(msg, ResultMessage):
                    out.n_turns = msg.num_turns
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
    out.duration_ms = int((time.time() - started) * 1000)

    after = tree_checksum(new_dir)
    touched = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    if touched:
        out.error = (
            out.error + "; " if out.error else ""
        ) + f"optimiser changed files outside agents/{domain}/{new_name}: {touched[:10]}"
    if (new_dir / "agent.yaml").read_bytes() != (champion.path / "agent.yaml").read_bytes():
        out.error = (out.error + "; " if out.error else "") + "agent.yaml was modified (frozen)"
    diag_path = new_dir / "diagnosis.json"
    if diag_path.exists():
        try:
            out.diagnosis = json.loads(diag_path.read_text())
        except json.JSONDecodeError as e:
            out.error = (out.error + "; " if out.error else "") + f"diagnosis.json unreadable: {e}"
    else:
        out.error = (out.error + "; " if out.error else "") + "no diagnosis.json written"
    new_version = load_version(domain, new_name)
    if all(new_version.files().get(s) == champion.files().get(s) for s in SURFACES):
        out.error = (out.error + "; " if out.error else "") + "no change to either surface"
    (new_dir / "optimiser_transcript.json").write_text(
        json.dumps(out.transcript, ensure_ascii=False, indent=1)
    )
    return out


__all__ = ["run_optimiser", "build_prompt", "condense_trace", "OptimiserOutput", "AGENTS_DIR"]
