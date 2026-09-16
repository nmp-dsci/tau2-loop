"""One row per simulated conversation: `runs/<id>/results.jsonl`.

tau2 scores a conversation with `evaluate_simulation()`: the reward is the
product of the components named in the task's `reward_basis` (DB hash, env
assertions, action checks, communicate checks, NL assertions), so it is 1.0 or
0.0 except in the rare partial cases. `correct` is `reward >= 1`. The row keeps
each component's verdict so a failure can be read without opening the trace,
and `partial_action_reward` as the diagnostic it is — never as the score.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    task_id: str
    trial: int
    reward: float
    correct: bool | None
    reward_basis: list[str] = field(default_factory=list)
    db_check: bool | None = None
    env_assertions: str | None = None  # "passed/total" or None
    action_checks: str | None = None  # "matched/total" or None
    communicate_checks: str | None = None
    nl_assertions: str | None = None
    partial_action_reward: float | None = None
    termination_reason: str = ""
    n_messages: int = 0
    n_agent_turns: int = 0
    n_tool_calls: int = 0
    n_tool_errors: int = 0
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    user_input_tokens: int = 0
    user_output_tokens: int = 0
    duration_ms: int = 0
    cost_usd_est: float | None = None
    error: str | None = None
    purpose: str = ""  # task.description.purpose, for tables
    trace: str = ""  # traces/<file>.json


@dataclass
class Summary:
    n: int
    n_scored: int
    passed: int
    pass_rate: float | None
    n_tasks: int
    trials: int
    pass_hat_k: dict[str, float]  # pass^k for k in 1..trials, on tasks with all trials scored
    failed_ids: list[str]
    errored_ids: list[str]
    by_termination: dict[str, int]
    mean_agent_turns: float
    mean_tool_calls: float
    partial_action_mean: float | None
    duration_ms: int
    cost_usd_est: float


def summarise(results: list[TaskResult]) -> Summary:
    from math import comb

    scored = [r for r in results if r.correct is not None]
    passed = [r for r in scored if r.correct]
    by_task: dict[str, list[TaskResult]] = {}
    for r in scored:
        by_task.setdefault(r.task_id, []).append(r)
    trials = max((len(v) for v in by_task.values()), default=0)
    pass_hat: dict[str, float] = {}
    for k in range(1, trials + 1):
        vals = []
        for rows in by_task.values():
            if len(rows) < k:
                continue
            n, c = len(rows), sum(1 for r in rows if r.correct)
            vals.append(comb(c, k) / comb(n, k) if comb(n, k) else 0.0)
        if vals:
            pass_hat[f"pass^{k}"] = round(sum(vals) / len(vals), 4)
    term: dict[str, int] = {}
    for r in results:
        term[r.termination_reason or "unknown"] = term.get(r.termination_reason or "unknown", 0) + 1
    partials = [r.partial_action_reward for r in scored if r.partial_action_reward is not None]
    return Summary(
        n=len(results),
        n_scored=len(scored),
        passed=len(passed),
        pass_rate=(len(passed) / len(scored)) if scored else None,
        n_tasks=len(by_task),
        trials=trials,
        pass_hat_k=pass_hat,
        failed_ids=sorted({r.task_id for r in scored if not r.correct}),
        errored_ids=sorted({r.task_id for r in results if r.error}),
        by_termination=term,
        mean_agent_turns=round(sum(r.n_agent_turns for r in results) / len(results), 2)
        if results
        else 0.0,
        mean_tool_calls=round(sum(r.n_tool_calls for r in results) / len(results), 2)
        if results
        else 0.0,
        partial_action_mean=round(sum(partials) / len(partials), 3) if partials else None,
        duration_ms=sum(r.duration_ms for r in results),
        cost_usd_est=round(sum(r.cost_usd_est or 0.0 for r in results), 4),
    )


def write_results(path: Path, results: list[TaskResult]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def read_results(path: Path) -> list[TaskResult]:
    out: list[TaskResult] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(TaskResult(**json.loads(line)))
    return out


def slug(task_id: str) -> str:
    """A filesystem-safe name for a task id (telecom ids carry brackets, pipes and colons)."""
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("_")
    return s[:80] if len(s) > 80 else s


def from_simulation(sim: Any, task: Any, trace_name: str) -> TaskResult:
    """A row from a tau2 `SimulationRun` (with `reward_info`) and its `Task`."""
    ri = sim.reward_info
    reward = float(ri.reward) if ri is not None else 0.0
    basis = (
        [str(b.value if hasattr(b, "value") else b) for b in (ri.reward_basis or [])] if ri else []
    )
    msgs = sim.messages or []
    agent_msgs = [m for m in msgs if getattr(m, "role", "") == "assistant"]
    tool_calls = sum(len(m.tool_calls or []) for m in agent_msgs if hasattr(m, "tool_calls"))
    tool_errors = sum(
        1 for m in msgs if getattr(m, "role", "") == "tool" and getattr(m, "error", False)
    )
    a_in = a_out = u_in = u_out = 0
    cost = 0.0
    for m in msgs:
        usage = getattr(m, "usage", None) or {}
        if getattr(m, "role", "") == "assistant":
            a_in += int(usage.get("prompt_tokens", 0) or 0)
            a_out += int(usage.get("completion_tokens", 0) or 0)
        elif getattr(m, "role", "") == "user":
            u_in += int(usage.get("prompt_tokens", 0) or 0)
            u_out += int(usage.get("completion_tokens", 0) or 0)
        cost += float(getattr(m, "cost", 0.0) or 0.0)
    partial = None
    if ri is not None and ri.action_checks:
        partial = round(
            sum(1 for c in ri.action_checks if c.action_match) / len(ri.action_checks), 3
        )
    desc = getattr(task, "description", None)
    purpose = str(getattr(desc, "purpose", "") or "") if desc is not None else ""
    return TaskResult(
        task_id=str(sim.task_id),
        trial=int(sim.trial or 0),
        reward=reward,
        correct=reward >= 1.0,
        reward_basis=basis,
        db_check=(bool(ri.db_check.db_match) if ri and ri.db_check is not None else None),
        env_assertions=_ratio([a.met for a in ri.env_assertions])
        if ri and ri.env_assertions
        else None,
        action_checks=_ratio([c.action_match for c in ri.action_checks])
        if ri and ri.action_checks
        else None,
        communicate_checks=_ratio([c.met for c in ri.communicate_checks])
        if ri and ri.communicate_checks
        else None,
        nl_assertions=_ratio([a.met for a in ri.nl_assertions])
        if ri and ri.nl_assertions
        else None,
        partial_action_reward=partial,
        termination_reason=str(getattr(sim.termination_reason, "value", sim.termination_reason)),
        n_messages=len(msgs),
        n_agent_turns=len(agent_msgs),
        n_tool_calls=tool_calls,
        n_tool_errors=tool_errors,
        agent_input_tokens=a_in,
        agent_output_tokens=a_out,
        user_input_tokens=u_in,
        user_output_tokens=u_out,
        duration_ms=int(float(sim.duration or 0.0) * 1000),
        cost_usd_est=round(cost, 5) if cost else None,
        error=_sim_error(sim),
        purpose=purpose,
        trace=trace_name,
    )


def _ratio(flags: list[bool]) -> str:
    return f"{sum(1 for f in flags if f)}/{len(flags)}"


def _sim_error(sim: Any) -> str | None:
    reason = str(getattr(sim.termination_reason, "value", sim.termination_reason))
    info = getattr(sim, "info", None) or {}
    if isinstance(info, dict) and info.get("error"):
        return f"{reason}: {str(info['error'])[:160]}"
    if reason not in {"agent_stop", "user_stop"}:
        return reason
    return None
