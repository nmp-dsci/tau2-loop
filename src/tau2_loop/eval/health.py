"""A run's harness health: whether the agent's tools and tool calls did what they are for (s13).

`profile.py` says what a run cost; this says how the agent used its harness, from the
conversations tau2 kept (`runs/<id>/tau2_results.json`). Each number is one of s13's milestone
checks, so a run shows on Runs whether a milestone worked:

- slipped calls: text replies that are really a tool call (the JSON contract written as prose,
  or v2's "One moment please" line that stood in for one). Native tool calls should make it 0.
- the knowledge tools: shell commands, `INDEX.md` reads, BM25 and dense searches, and the
  sandbox or dense failures among them.
- bare discoverable calls: a discoverable tool (`…_7291`) called by its name instead of through
  `call_discoverable_agent_tool`; it runs, but the grader counts nothing.
- look-ups before the first action: knowledge searches before the first verification, unlock,
  discoverable call, user-tool hand-over or transfer (the leaders 9–13, v1 and v2 2–3).
- account-email calls: a call carrying the CLI's account address (`core.REDACTED_EMAIL`)
  instead of the customer's, the identity leak.
- required documents read: of the documents tau2 lists for a task, the share the agent had in
  front of it in full (search results, or a file it printed through the shell).
- required documents seen: the ones it was never shown in full but whose id (or file name,
  `<id>.md`) came back in a knowledge tool's result: a listing, a `grep` line, a shell output.

Arithmetic only: no model, no tracking server, no environment.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

from tau2_loop.config import RUNS_DIR
from tau2_loop.llm.core import REDACTED_EMAIL
from tau2_loop.llm.prompting import _first_json_object

SEARCH_TOOLS = frozenset({"KB_search", "KB_search_bm25", "KB_search_dense", "grep", "shell"})
DOC_SEARCH_TOOLS = frozenset({"KB_search", "KB_search_bm25", "KB_search_dense", "grep"})
ACTION_TOOLS = frozenset(
    {
        "log_verification",
        "unlock_discoverable_agent_tool",
        "call_discoverable_agent_tool",
        "give_discoverable_user_tool",
    }
)
# v2's helper put this line in place of a slipped call (agents/banking_knowledge/v2/helper.py)
HOLD_PREFIX = "One moment please, I'm still working on that step"
BARE_DISCOVERABLE = re.compile(r"^[a-z][a-z0-9_]*_\d{4}$")
DOC_ID = re.compile(r"doc_[A-Za-z0-9_\-]+?(?=\.md\b|\b)")
RESULT_ID = re.compile(r"ID: (doc_[^\s]+)")
PRINTS_FILE = re.compile(r"\b(cat|head|tail|sed|awk|less)\b")


def is_slipped(content: str | None) -> bool:
    """A text reply that is really a tool call."""
    if not content:
        return False
    if content.startswith(HOLD_PREFIX):
        return True
    obj = _first_json_object(content)
    return isinstance(obj, dict) and ("name" in obj or "tool_calls" in obj)


def _is_action(name: str) -> bool:
    return name in ACTION_TOOLS or "transfer" in name


def _mentions(text: str, doc_id: str) -> bool:
    """Whether a result names a document: its id whole, or as its file name `<id>.md`. Ids hold
    parentheses (`…_(general)_001`), so the id is matched as a literal, never as a word."""
    return re.search(rf"(?<![\w-]){re.escape(doc_id)}(?![\w-])", text) is not None


def conversation_health(sim: dict[str, Any], required: set[str] | frozenset[str]) -> dict[str, Any]:
    """One conversation's numbers."""
    msgs = sim.get("messages") or []
    calls: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    call_turns = 0
    replies = slipped = 0
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls") or []
        if tcs:
            call_turns += 1
            for tc in tcs:
                calls.append(tc)
                by_id[str(tc.get("id"))] = tc
        elif m.get("content"):
            replies += 1
            slipped += is_slipped(m.get("content"))
    names = [str(tc.get("name")) for tc in calls]
    commands = [
        str((tc.get("arguments") or {}).get("command", ""))
        for tc in calls
        if tc.get("name") == "shell"
    ]
    first_action = next((i for i, n in enumerate(names) if _is_action(n)), len(names))
    read: set[str] = set()
    for c in commands:
        if PRINTS_FILE.search(c):
            read |= set(DOC_ID.findall(c))
    tool_errors = sandbox_failures = dense_errors = shell_blocked = 0
    seen: set[str] = set()
    for m in msgs:
        if m.get("role") != "tool":
            continue
        text = str(m.get("content") or "")
        tc = by_id.get(str(m.get("id")))
        name = str(tc.get("name")) if tc else ""
        if text.startswith("Error") or m.get("error"):
            tool_errors += 1
            low = text.lower()
            if name == "shell" and "command blocked" in low:
                shell_blocked += 1
            elif name == "shell" and ("srt" in low or "sandbox" in low):
                sandbox_failures += 1
            elif name == "KB_search_dense":
                dense_errors += 1
        if name in DOC_SEARCH_TOOLS:
            read |= set(RESULT_ID.findall(text))
        if name in SEARCH_TOOLS:
            seen |= {d for d in required if _mentions(text, d)}
    bare = [n for n in names if BARE_DISCOVERABLE.match(n)]
    read_ids = set(required) & read
    return {
        "task_id": sim.get("task_id"),
        "trial": sim.get("trial"),
        "passed": (sim.get("reward_info") or {}).get("reward") == 1.0,
        "replies": replies,
        "slipped": slipped,
        "tool_calls": len(calls),
        "calls_per_turn": round(len(calls) / call_turns, 3) if call_turns else 0.0,
        "shell_calls": names.count("shell"),
        "index_reads": sum("INDEX" in c for c in commands),
        "bm25_calls": names.count("KB_search_bm25") + names.count("KB_search"),
        "dense_calls": names.count("KB_search_dense"),
        "grep_calls": names.count("grep"),
        "lookups_before_first_action": sum(n in SEARCH_TOOLS for n in names[:first_action]),
        "bare_discoverable_calls": len(bare),
        "account_email_calls": sum(
            REDACTED_EMAIL in json.dumps(tc.get("arguments")) for tc in calls
        ),
        "tool_errors": tool_errors,
        "sandbox_failures": sandbox_failures,
        "shell_blocked": shell_blocked,
        "dense_errors": dense_errors,
        "required_docs": len(required),
        "required_docs_read": len(read_ids),
        "required_docs_read_ids": sorted(read_ids),
        "required_docs_seen": len(seen - read_ids),
        "required_docs_seen_ids": sorted(seen - read_ids),
    }


def summarise_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The run's totals, means and conversation counts over its conversations."""
    n = len(rows)
    if not n:
        return {"conversations": 0}

    def total(k: str) -> int:
        return int(sum(r[k] for r in rows))

    def convs(k: str) -> int:
        return sum(1 for r in rows if r[k] > 0)

    replies = total("replies")
    with_need = [r for r in rows if r["required_docs"]]
    return {
        "conversations": n,
        "passed": sum(1 for r in rows if r["passed"]),
        "replies": replies,
        "slipped": total("slipped"),
        "slipped_rate": round(total("slipped") / replies, 4) if replies else 0.0,
        "slipped_conversations": convs("slipped"),
        "tool_calls": total("tool_calls"),
        "calls_per_turn": round(statistics.mean(r["calls_per_turn"] for r in rows), 3),
        "shell_calls": total("shell_calls"),
        "shell_conversations": convs("shell_calls"),
        "index_read_conversations": convs("index_reads"),
        "bm25_calls": total("bm25_calls"),
        "dense_calls": total("dense_calls"),
        "grep_calls": total("grep_calls"),
        "lookups_before_first_action": round(
            statistics.mean(r["lookups_before_first_action"] for r in rows), 2
        ),
        "bare_discoverable_calls": total("bare_discoverable_calls"),
        "bare_discoverable_conversations": convs("bare_discoverable_calls"),
        "account_email_calls": total("account_email_calls"),
        "account_email_conversations": convs("account_email_calls"),
        "tool_errors": total("tool_errors"),
        "sandbox_failures": total("sandbox_failures"),
        "shell_blocked": total("shell_blocked"),
        "dense_errors": total("dense_errors"),
        "required_docs_read_share": round(
            statistics.mean(r["required_docs_read"] / r["required_docs"] for r in with_need), 4
        )
        if with_need
        else None,
        # of the same documents, the mean share only seen: named in a result, never shown whole
        "required_docs_seen_share": round(
            statistics.mean(r.get("required_docs_seen", 0) / r["required_docs"] for r in with_need),
            4,
        )
        if with_need
        else None,
    }


def results_health(results: dict[str, Any]) -> dict[str, Any]:
    """Health of a tau2 results object: `{summary, conversations}`."""
    required = {
        str(t.get("id")): frozenset(t.get("required_documents") or [])
        for t in results.get("tasks") or []
    }
    rows = [
        conversation_health(s, required.get(str(s.get("task_id")), frozenset()))
        for s in results.get("simulations") or []
    ]
    return {"summary": summarise_health(rows), "conversations": rows}


def run_health(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any] | None:
    """A run folder's health, or None when it kept no tau2 results."""
    p = runs_dir / run_id / "tau2_results.json"
    if not p.exists():
        return None
    return results_health(json.loads(p.read_text()))
