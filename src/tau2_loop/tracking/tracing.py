"""One MLflow trace per conversation, rebuilt from the transcript τ² already wrote.

`mlflow.anthropic.autolog()` sees nothing here: the agent's model call happens
inside the `claude` CLI child the Agent SDK spawns, and the user simulator and
the judge are τ²'s own calls through litellm. So this reconstructs the trace
from `runs/<id>/traces/<file>.json`, which carries a timestamp, a cost and a
token usage on every message:

    conversation                 root, the whole simulation
      turn 1                     one assistant (agent) message
        get_reservation_details  a tool call, ended by its own tool result
      user 1                     one simulated-user message — it is a model call
                                 too, and half the bill

Both sides are logged because a τ² conversation is two agents talking: reading
only the agent's turns hides why a conversation ran long. Tags are what the
optimiser and the viewer filter on; the four the platform requires
(`project · git_sha · env · billing`, PLATFORM.md) are on every trace.

Volume: a full-split run is 4 domains × up to 114 tasks × 4 trials. Keep the
`CUT` trimming below and never set `MLFLOW_TRACE_FULL=1` on a shared run
(PLATFORM.md rule 4).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from tau2_loop.config import settings
from tau2_loop.tracking.mlflow_log import EXPERIMENT, PROJECT, required_tags

if TYPE_CHECKING:
    from tau2_loop.eval.results import TaskResult
    from tau2_loop.eval.runner import RunMeta

CUT = 2_000


def _cut(v: Any, n: int = CUT) -> Any:
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + f"…[{len(v) - n} more]"
    if isinstance(v, dict):
        return {k: _cut(x, n) for k, x in v.items()}
    if isinstance(v, list):
        return [_cut(x, n) for x in v[:50]]
    return v


def _ns(stamp: Any, fallback: int) -> int:
    """A τ² timestamp (`2026-09-16T03:10:33.517219`) as nanoseconds since the epoch."""
    if not isinstance(stamp, str):
        return fallback
    try:
        return int(datetime.fromisoformat(stamp).timestamp() * 1e9)
    except ValueError:
        return fallback


def _tokens(usage: Any) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def emit_spans(
    client: Any, trace_id: str, parent_id: str, messages: list[dict[str, Any]], t0: int
) -> int:
    """A span per message in stream order; returns the last end time in nanoseconds.

    A tool call opens a span that its own `tool` message closes, matched on the
    call id. τ² gives the tool message the same timestamp as the call that made
    it, so a tool span is usually zero-width — that is honest: the environment is
    in-process and answers instantly. What takes the time is the turn above it.
    """
    pending: dict[str, str] = {}
    last = t0
    turns = {"assistant": 0, "user": 0}
    for m in messages:
        role = str(m.get("role") or "")
        at = _ns(m.get("timestamp"), last)
        if role in ("assistant", "user"):
            turns[role] += 1
            secs = m.get("generation_time_seconds")
            start = at - int(float(secs) * 1e9) if isinstance(secs, int | float) else last
            prompt, completion = _tokens(m.get("usage"))
            calls = m.get("tool_calls") or []
            span = client.start_span(
                name=f"{'turn' if role == 'assistant' else 'user'} {turns[role]}",
                trace_id=trace_id,
                parent_id=parent_id,
                span_type="LLM",
                inputs={"prompt_tokens": prompt},
                start_time_ns=min(start, at),
            )
            client.end_span(
                trace_id=trace_id,
                span_id=span.span_id,
                outputs={
                    "text": _cut(str(m.get("content") or "")),
                    "tool_calls": [c.get("name") for c in calls],
                },
                attributes={
                    "mlflow.chat.tokenUsage": {
                        "input_tokens": prompt,
                        "output_tokens": completion,
                        "total_tokens": prompt + completion,
                    },
                    "cost_usd": float(m.get("cost") or 0.0),
                },
                end_time_ns=at,
            )
            for c in calls:
                tool = client.start_span(
                    name=str(c.get("name")),
                    trace_id=trace_id,
                    parent_id=span.span_id,
                    span_type="TOOL",
                    inputs=_cut(c.get("arguments") or {}),
                    start_time_ns=at,
                )
                pending[str(c.get("id"))] = tool.span_id
            last = at
        elif role == "tool":
            span_id = pending.pop(str(m.get("id")), None)
            if span_id is not None:
                client.end_span(
                    trace_id=trace_id,
                    span_id=span_id,
                    outputs={"result": _cut(str(m.get("content") or ""))},
                    status="ERROR" if m.get("error") else "OK",
                    end_time_ns=at,
                )
            last = at
    # a call whose result never arrived: the conversation was cut short
    for span_id in pending.values():
        client.end_span(trace_id=trace_id, span_id=span_id, status="ERROR", end_time_ns=last)
    return last


def log_conversation(meta: RunMeta, result: TaskResult, transcript: dict[str, Any]) -> str | None:
    """Build and log one conversation's trace; None when tracking is unreachable.

    Never raises: a trace is an index entry, and the run folder is the record.
    """
    try:
        import mlflow
        from mlflow import MlflowClient

        s = settings()
        mlflow.set_tracking_uri(s.mlflow_tracking_uri)
        exp = mlflow.set_experiment(EXPERIMENT)
        client = MlflowClient()
        messages = list(transcript.get("messages") or [])
        t0 = _ns(transcript.get("start_time"), int(datetime.now().timestamp() * 1e9))
        reward = transcript.get("reward_info") or {}
        root = client.start_trace(
            name=f"{result.task_id} t{result.trial}",
            span_type="AGENT",
            inputs={"task_id": result.task_id, "purpose": _cut(result.purpose)},
            attributes={
                "agent": meta.agent,
                "fingerprint": meta.fingerprint,
                "model": meta.model,
                "user_model": meta.user_model,
                "judge_model": meta.judge_model,
            },
            tags={
                **required_tags(),
                "run_id": meta.run_id,
                "domain": meta.domain,
                "task": result.task_id,
                "trial": str(result.trial),
                "split": meta.split,
                "passed": "" if result.correct is None else str(int(result.correct)),
                "termination": result.termination_reason,
                "agent": meta.agent,
                "fingerprint": meta.fingerprint,
                "model": meta.model,
                "kind": "conversation",
            },
            experiment_id=exp.experiment_id,
            start_time_ns=t0,
        )
        last = emit_spans(client, root.trace_id, root.span_id, messages, t0)
        agent_in, agent_out = result.agent_input_tokens, result.agent_output_tokens
        user_in, user_out = result.user_input_tokens, result.user_output_tokens
        client.end_trace(
            trace_id=root.trace_id,
            outputs={
                "reward": result.reward,
                "correct": result.correct,
                "reward_basis": result.reward_basis,
                "breakdown": _cut(reward.get("reward_breakdown") or {}),
            },
            attributes={
                # the key MLflow's Traces tab reads for its Tokens column
                "mlflow.chat.tokenUsage": {
                    "input_tokens": agent_in + user_in,
                    "output_tokens": agent_out + user_out,
                    "total_tokens": agent_in + user_in + agent_out + user_out,
                },
                "agent_turns": result.n_agent_turns,
                "tool_calls": result.n_tool_calls,
                "tool_errors": result.n_tool_errors,
                "agent_input_tokens": agent_in,
                "agent_output_tokens": agent_out,
                "user_input_tokens": user_in,
                "user_output_tokens": user_out,
                "cost_usd": result.cost_usd_est or 0.0,
                "error": result.error or "",
            },
            status="ERROR" if result.error else "OK",
            end_time_ns=max(last, t0 + int(result.duration_ms * 1e6)),
        )
        return str(root.trace_id)
    except Exception:  # noqa: BLE001 - a missing index never fails a completed run
        return None


def log_run_traces(meta: RunMeta, results: list[TaskResult], run_dir: Any) -> int:
    """One trace per conversation of a finished run. Returns how many were logged."""
    import json
    from pathlib import Path

    n = 0
    for r in results:
        path = Path(run_dir) / "traces" / r.trace
        if not r.trace or not path.is_file():
            continue
        if log_conversation(meta, r, json.loads(path.read_text())) is not None:
            n += 1
    return n


def flush() -> None:
    import contextlib

    with contextlib.suppress(Exception):
        import mlflow

        mlflow.flush_trace_async_logging()


__all__ = ["PROJECT", "flush", "log_conversation", "log_run_traces"]
