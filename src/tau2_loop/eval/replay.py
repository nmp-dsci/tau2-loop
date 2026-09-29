"""Rebuild a conversation's environment at any message, and run one tool call in it.

The playground behind the Agent tab: pick a message in a committed trace, edit a
tool call's arguments, and see what tau2's own tool code returns against the
database as it stood at that point. The state is rebuilt the way the evaluator
builds it for the DB check — the task's initial state, then every state-changing
call before the chosen message, replayed through `Environment.set_state` — so
the playground and the grade cannot disagree about what the database held.

Nothing here reaches a model, and nothing is written: each call gets a fresh
environment that is discarded when the call returns. tau2 is imported lazily
because the demo image does not ship it; `available()` is how the API asks.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from typing import Any

from tau2_loop.config import RUNS_DIR, quiet_tau2
from tau2_loop.data.splits import read_task_extract

# The playground's bounds: one call per request, arguments this size at most.
MAX_ARGS_BYTES = 8192
# How many changed fields a diff reports before it stops listing them.
MAX_DIFF_FIELDS = 60


class ReplayError(ValueError):
    """A request the playground cannot serve: the message is wrong, the tool unknown."""


@dataclass(frozen=True)
class Conversation:
    """One committed conversation, loaded as tau2 objects."""

    domain: str
    task: Any  # tau2.data_model.tasks.Task
    messages: list[Any]  # tau2.data_model.message.Message


def available() -> bool:
    """Whether tau2 is importable here: true in a checkout, false in the demo image."""
    return importlib.util.find_spec("tau2") is not None


def env_kwargs_for(domain: str, task: Any) -> dict[str, Any]:
    """The environment the live evaluation built. Banking's retrieval must be the run's own
    `bm25` (`eval/runner.py`), never a default: the dense variants call an embedding API."""
    if domain != "banking_knowledge":
        return {}
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import _build_env_kwargs

    return dict(_build_env_kwargs(TextRunConfig(domain=domain, retrieval_config="bm25"), task))


def load_conversation(run_id: str, trace_name: str, domain: str, task_id: str) -> Conversation:
    _import_tau2()
    from tau2.data_model.simulation import SimulationRun
    from tau2.data_model.tasks import Task

    p = RUNS_DIR / run_id / "traces" / trace_name
    if "/" in trace_name or not p.is_file():
        raise ReplayError("no trace for this conversation")
    sim = SimulationRun.model_validate(json.loads(p.read_text()))
    spec = next(
        (t for t in read_task_extract(domain).get("tasks", []) if t.get("id") == task_id), None
    )
    if spec is not None:
        task = Task.model_validate(spec)
    else:  # the smoke domain has no extract: ask tau2 for it
        from tau2.runner.helpers import load_tasks

        task = next((t for t in load_tasks(domain, None) if t.id == task_id), None)
        if task is None:
            raise ReplayError(f"task {task_id} is not in {domain}")
    return Conversation(domain=domain, task=task, messages=list(sim.messages))


def build_env(conv: Conversation, at: int, strict: bool = False) -> Any:
    """The environment as it stood before message `at`: initial state, then each earlier write.

    `strict=False` is what `eval/rescore.py` uses: a replayed write whose recorded output
    drifted cosmetically still applies its change. The tests replay with `strict=True`."""
    from tau2.registry import registry

    if not 0 <= at <= len(conv.messages):
        raise ReplayError(f"message {at} is outside this conversation (0–{len(conv.messages)})")
    init = conv.task.initial_state
    env = registry.get_env_constructor(conv.domain)(
        solo_mode=False, **env_kwargs_for(conv.domain, conv.task)
    )
    try:
        env.set_state(
            initialization_data=init.initialization_data if init else None,
            initialization_actions=init.initialization_actions if init else None,
            # the trajectory alone, as the evaluator replays it for the DB check
            message_history=conv.messages[:at],
            strict=strict,
        )
    except ValueError as e:
        if "Tool message expected" in str(e) or "Tool call id mismatch" in str(e):
            raise ReplayError(
                f"message {at} falls between a tool call and its result; pick the message that made the call"
            ) from e
        raise
    return env


def run_tool(
    conv: Conversation,
    *,
    name: str,
    arguments: dict[str, Any],
    at: int,
    after_calls: int = 0,
    requestor: str = "assistant",
) -> dict[str, Any]:
    """One call, in the environment rebuilt to message `at`.

    `after_calls` runs the first N calls of message `at` first, so a call that was the
    second in its message sees the first one's effect, as it did live."""
    from tau2.data_model.message import ToolCall

    if len(json.dumps(arguments)) > MAX_ARGS_BYTES:
        raise ReplayError(f"arguments over {MAX_ARGS_BYTES} bytes")
    if requestor not in ("assistant", "user"):
        raise ReplayError("requestor is `assistant` or `user`")
    env = build_env(conv, at)
    kit = env.tools if requestor == "assistant" else env.user_tools
    if kit is None or not kit.has_tool(name):
        side = "the agent's" if requestor == "assistant" else "the customer's"
        raise ReplayError(f"{name} is not one of {side} tools in {conv.domain}")

    siblings = (
        list(getattr(conv.messages[at], "tool_calls", None) or [])
        if at < len(conv.messages)
        else []
    )
    for tc in siblings[: max(0, after_calls)]:
        env.get_response(tc)

    before, hashes = _dump(env), _hashes(env)
    msg = env.get_response(
        ToolCall(id="playground", name=name, arguments=arguments, requestor=requestor)
    )
    wrote = _hashes(env) != hashes
    recorded = _recorded_result(conv, at, after_calls, name, arguments)
    return {
        "name": name,
        "arguments": arguments,
        "at": at,
        "after_calls": after_calls,
        "requestor": requestor,
        "content": msg.content,
        "error": bool(msg.error),
        "wrote": wrote,
        "diff": diff(before, _dump(env)) if wrote else [],
        "replayed_writes": _writes_before(conv, at),
        # the same call as the trace made, at the same point: did it come back the same?
        "same_as_recorded": None if recorded is None else _same(recorded, msg.content),
    }


def diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """The records that changed, each with the field paths that differ.

    A record is the first two levels of the database (`reservations.FQ8APE`); a field is
    the rest of the path (`payment_methods.gift_card_8190333.amount`)."""
    fields: list[tuple[str, str, Any, Any]] = []
    _leaves(before, after, [], fields)
    out: dict[str, list[dict[str, Any]]] = {}
    for record, field, a, b in fields[:MAX_DIFF_FIELDS]:
        out.setdefault(record, []).append({"field": field, "before": _short(a), "after": _short(b)})
    return [{"record": r, "fields": fs} for r, fs in out.items()]


def _leaves(a: Any, b: Any, path: list[str], out: list[tuple[str, str, Any, Any]]) -> None:
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            _leaves(a.get(k), b.get(k), [*path, str(k)], out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(path) < 2:
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _leaves(x, y, [*path, str(i)], out)
        return
    record = ".".join(path[:2]) or "(database)"
    field = ".".join(path[2:]) or "(record)"
    out.append((record, field, a, b))


def _short(v: Any) -> str | None:
    if v is None:
        return None
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= 400 else f"{s[:399]}…"


def _dump(env: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if env.tools is not None and env.tools.db is not None:
        out.update(env.tools.db.model_dump(mode="json"))
    user_db = env.user_tools.db if env.user_tools is not None else None
    if user_db is not None and (env.tools is None or user_db is not env.tools.db):
        out["user"] = user_db.model_dump(mode="json")
    return out


def _hashes(env: Any) -> tuple[str | None, str | None]:
    return env.get_db_hash(), env.get_user_db_hash()


def _writes_before(conv: Conversation, at: int) -> int:
    """How many state-changing calls the rebuild re-ran to reach message `at`."""
    ext = read_task_extract(conv.domain)
    mutating = {
        t["name"] for t in [*ext.get("tools", []), *ext.get("user_tools", [])] if t.get("mutates")
    }
    return sum(
        1
        for m in conv.messages[:at]
        for tc in (getattr(m, "tool_calls", None) or [])
        if tc.name in mutating
    )


def _recorded_result(
    conv: Conversation, at: int, after_calls: int, name: str, arguments: dict[str, Any]
) -> str | None:
    if at >= len(conv.messages):
        return None
    calls = list(getattr(conv.messages[at], "tool_calls", None) or [])
    if after_calls >= len(calls):
        return None
    tc = calls[after_calls]
    if tc.name != name or tc.arguments != arguments:
        return None
    for m in conv.messages[at + 1 :]:
        if getattr(m, "role", None) == "tool" and m.id == tc.id:
            return str(m.content or "")
    return None


def _same(a: str, b: str) -> bool:
    try:
        return bool(json.loads(a) == json.loads(b))
    except (json.JSONDecodeError, TypeError):
        return a == b


def _import_tau2() -> None:
    """Import tau2 quietly, and drop any key its dotenv search pulled in from `~/.env`."""
    if not available():
        raise ReplayError("the playground needs tau2, which this deployment does not ship")
    quiet_tau2()
    import tau2  # noqa: F401 - the import is the point: it runs tau2's dotenv search once

    from tau2_loop.llm import scrub_injected_key

    scrub_injected_key()
