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


def prior_attempts(domain: str, task_id: str) -> list[dict[str, Any]]:
    """Every earlier diagnosis of one task, with the cycle's verdict and whether the task got fixed."""
    out: list[dict[str, Any]] = []
    for e in read_ledger(domain):
        outcome = e.get("outcome") or {}
        if task_id in (outcome.get("broken") or []) and not any(
            str(d.get("task_id")) == str(task_id) for d in e.get("diagnoses", [])
        ):
            out.append(
                {
                    "cycle": e.get("cycle"),
                    "challenger": e.get("challenger"),
                    "root_cause": "not diagnosed: this task passed on the champion and BROKE on the challenger",
                    "surface": "both",
                    "change": f"the cycle's edits ({e.get('prompt_diff_summary', '')[:160]} / {e.get('helper_diff_summary', '')[:160]})",
                    "verdict": outcome.get("verdict", "pending"),
                    "task_outcome": "broken",
                }
            )
        for d in e.get("diagnoses", []):
            if str(d.get("task_id")) != str(task_id):
                continue
            fixed = task_id in (outcome.get("fixed") or [])
            broken = task_id in (outcome.get("broken") or [])
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
        lines.append(
            f"- cycle {e.get('cycle')}: {e.get('champion')} → {e.get('challenger')} · verdict {o.get('verdict', 'pending')}"
            f" · passes {o.get('passes', '?')} · fixed {o.get('fixed', [])} · broken {o.get('broken', [])}"
            + (f" · reason: {o.get('reason', '')[:300]}" if o.get("reason") else "")
        )
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
