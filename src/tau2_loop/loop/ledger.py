"""The loop's memory, per domain: `loop/<domain>/ledger.jsonl`, one entry per cycle, committed.

Every diagnosis the optimiser makes, every change it proposes and what the gate
then said about it lives here, so the next cycle's optimiser reads what was
tried on a task before proposing it again. An entry is written before the
challenger is evaluated (so a crashed cycle still leaves its reasoning) and the
`outcome` is filled in by the harness after the gate — never by the optimiser.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from tau2_loop.config import ledger_path
from tau2_loop.llm import redact


def _redact_entry(value: Any) -> Any:
    """Recursively redact the account email out of every string an entry carries."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _redact_entry(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_entry(v) for v in value]
    return value


def read_ledger(domain: str) -> list[dict[str, Any]]:
    p = ledger_path(domain)
    if not p.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def append_entry(domain: str, entry: dict[str, Any]) -> None:
    p = ledger_path(domain)
    p.parent.mkdir(parents=True, exist_ok=True)
    entry.setdefault("domain", domain)
    entry.setdefault("at", datetime.now(UTC).isoformat())
    entry = _redact_entry(entry)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def update_entry(domain: str, cycle: int, **fields: Any) -> None:
    """Rewrite the ledger with `fields` merged into the entry for `cycle`."""
    fields = _redact_entry(fields)
    entries = read_ledger(domain)
    for e in entries:
        if e.get("cycle") == cycle:
            e.update(fields)
    ledger_path(domain).write_text(
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8"
    )


def next_cycle_number(domain: str) -> int:
    cycles = [int(e.get("cycle", 0)) for e in read_ledger(domain)]
    return (max(cycles) + 1) if cycles else 1


def swap_note(e: dict[str, Any]) -> str:
    """A cycle no optimiser wrote (`make challenge`): what changed instead, e.g.
    `model swap (model: sonnet → opus)`; "" for a loop cycle."""
    if not e.get("kind"):
        return ""
    return f"{e['kind']} ({'; '.join(e.get('agent_yaml') or []) or 'agent.yaml unchanged'})"


def seen_changes(outcome: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(fixed, broken) as an optimiser may see them: the read half's where train is halved
    (s09), since the gate half's ids must never reach a prompt, and all of train's where the gate
    is on test, whose ids must not either (`read_fixed` holds both); else the gate's own lists."""
    if "read_fixed" in outcome:
        return list(outcome.get("read_fixed") or []), list(outcome.get("read_broken") or [])
    return list(outcome.get("fixed") or []), list(outcome.get("broken") or [])


def prior_attempts(domain: str, task_id: str) -> list[dict[str, Any]]:
    """Every earlier diagnosis of one task, with the cycle's verdict and whether the task got fixed."""
    out: list[dict[str, Any]] = []
    for e in read_ledger(domain):
        outcome = e.get("outcome") or {}
        fixed_ids, broken_ids = seen_changes(outcome)
        if task_id in broken_ids and not any(
            str(d.get("task_id")) == str(task_id) for d in e.get("diagnoses", [])
        ):
            out.append(
                {
                    "cycle": e.get("cycle"),
                    "challenger": e.get("challenger"),
                    "root_cause": "not diagnosed: this task passed on the champion and BROKE on the challenger",
                    "surface": "agent.yaml" if e.get("kind") else "both",
                    "change": swap_note(e)
                    or f"the cycle's edits ({e.get('prompt_diff_summary', '')[:160]} / {e.get('helper_diff_summary', '')[:160]})",
                    "verdict": outcome.get("verdict", "pending"),
                    "task_outcome": "broken",
                }
            )
        for d in e.get("diagnoses", []):
            if str(d.get("task_id")) != str(task_id):
                continue
            fixed = task_id in fixed_ids
            broken = task_id in broken_ids
            still_failed = task_id in (outcome.get("still_failed") or [])
            out.append(
                {
                    "cycle": e.get("cycle"),
                    "challenger": e.get("challenger"),
                    "root_cause": d.get("root_cause"),
                    "surface": d.get("surface"),
                    "change": d.get("change"),
                    "verdict": outcome.get("verdict", "pending"),
                    "task_outcome": "fixed"
                    if fixed
                    else "broken"
                    if broken
                    else "still failed"
                    if still_failed
                    else "unknown",
                }
            )
    return out


def render_history(domain: str, task_ids: list[str]) -> str:
    """The history block for the optimiser prompt: what was tried, and what happened."""
    entries = read_ledger(domain)
    if not entries:
        return "No earlier cycles on this domain. This is the first optimisation; there is nothing to avoid repeating yet."
    lines = [
        "Earlier cycles (newest last). Do not repeat a change whose task outcome was 'still failed' or 'broken' unless you explain what is different this time."
    ]
    for e in entries:
        o = e.get("outcome") or {}
        swap = swap_note(e)
        fixed_ids, broken_ids = seen_changes(o)
        hidden = "read_fixed" in o
        on_test = str(o.get("gate_on") or "").startswith("test")
        reason = (
            "" if hidden else str(o.get("reason") or "")
        )  # the reason of a gate on the gate half or on test names its ids
        lines.append(
            f"- cycle {e.get('cycle')}: {e.get('champion')} → {e.get('challenger')}"
            + (f", a {swap}, no optimiser" if swap else "")
            + (f", the {e['optimiser_mode']} optimiser" if e.get("optimiser_mode") else "")
            + f" · verdict {o.get('verdict', 'pending')}"
            + (
                f" · the gate's test passes {o.get('passes', '?')} · on train {o.get('train_passes', '?')}, fixed {fixed_ids} · broken {broken_ids}"
                if on_test
                else f" · gate-half passes {o.get('passes', '?')} · on the read half fixed {fixed_ids} · broken {broken_ids}"
                if hidden
                else f" · passes {o.get('passes', '?')} · fixed {fixed_ids} · broken {broken_ids}"
            )
            + (f" · reason: {reason[:300]}" if reason else "")
        )
        if e.get("surfaces_changed"):
            lines.append(f"    surfaces changed: {', '.join(e['surfaces_changed'])}")
        lines.append(f"    prompt: {e.get('prompt_diff_summary', '')}")
        lines.append(f"    helper: {e.get('helper_diff_summary', '')}")
    for tid in task_ids:
        attempts = prior_attempts(domain, tid)
        if attempts:
            lines.append(f"- task {tid} was diagnosed before:")
            for a in attempts:
                lines.append(
                    f"    cycle {a['cycle']} ({a['surface']}): {a['change']} → {a['task_outcome']} (cycle verdict {a['verdict']})"
                )
    return "\n".join(lines)
