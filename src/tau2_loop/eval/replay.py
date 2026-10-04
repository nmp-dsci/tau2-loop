"""Rebuild a conversation's environment at any message, and run one tool call in it.

The playground behind the Agent tab: pick a message in a committed trace, edit a
tool call's arguments, and see what tau2's own tool code returns against the
database as it stood at that point. `db_diff` rebuilds both sides of the DB check
whole, the conversation's final database and gold's, and says where they differ. The state is rebuilt the way the evaluator
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
    retrieval: str | None = None  # banking: the run's retrieval variant


def available() -> bool:
    """Whether tau2 is importable here: true in a checkout, false in the demo image."""
    return importlib.util.find_spec("tau2") is not None


# The variant every banking run used before `RunMeta.retrieval` recorded one.
LEGACY_RETRIEVAL = "bm25"


def env_kwargs_for(domain: str, task: Any, retrieval: str | None = None) -> dict[str, Any]:
    """The environment the live evaluation built. Banking's retrieval must be the run's own
    variant (`RunMeta.retrieval`; `bm25` before the field existed), never tau2's default:
    its dense search calls an embedding API. The local variants (`eval.retrieval`) need the
    sandbox and the embedding model installed, as their live runs did."""
    if domain != "banking_knowledge":
        return {}
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import _build_env_kwargs

    from tau2_loop.eval.retrieval import register

    register()  # a run on a local AllTools variant rebuilds the same tools
    cfg = TextRunConfig(domain=domain, retrieval_config=retrieval or LEGACY_RETRIEVAL)
    return dict(_build_env_kwargs(cfg, task))


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
    return Conversation(
        domain=domain, task=task, messages=list(sim.messages), retrieval=_run_retrieval(run_id)
    )


def _run_retrieval(run_id: str) -> str | None:
    """The retrieval variant a run recorded in its `run.json`; None for a run before the field."""
    try:
        meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
    except (OSError, ValueError):
        return None
    v = meta.get("retrieval")
    return str(v) if v else None


def build_env(conv: Conversation, at: int, strict: bool = False) -> Any:
    """The environment as it stood before message `at`: initial state, then each earlier write.

    `strict=False` is what `eval/rescore.py` uses: a replayed write whose recorded output
    drifted cosmetically still applies its change. The tests replay with `strict=True`."""
    from tau2.registry import registry

    if not 0 <= at <= len(conv.messages):
        raise ReplayError(f"message {at} is outside this conversation (0–{len(conv.messages)})")
    init = conv.task.initial_state
    env = registry.get_env_constructor(conv.domain)(
        solo_mode=False, **env_kwargs_for(conv.domain, conv.task, conv.retrieval)
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


def gold_env(conv: Conversation, upto: int | None = None) -> tuple[Any, list[dict[str, Any]]]:
    """Gold's environment as the evaluator builds it: the task's initial state and initial
    history, then the first `upto` expected actions (all of them by default). Returns it with
    each action that raised, which the evaluator logs and skips."""
    from tau2.registry import registry

    init = conv.task.initial_state
    env = registry.get_env_constructor(conv.domain)(
        **env_kwargs_for(conv.domain, conv.task, conv.retrieval)
    )
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=list(init.message_history or []) if init else [],
        strict=False,
    )
    actions = list(
        (conv.task.evaluation_criteria.actions or []) if conv.task.evaluation_criteria else []
    )
    errors = []
    for a in actions[:upto]:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception as e:  # noqa: BLE001 - the evaluator skips a failing gold action too
            errors.append({"action_id": a.action_id, "name": a.name, "error": str(e)[:200]})
    return env, errors


def db_diff(conv: Conversation) -> dict[str, Any]:
    """Where the conversation's final database differs from gold's, as the DB check compares them.

    The agent's side replays the whole trajectory on the task's initial state; gold's applies
    every expected action to it. Each differing field carries its value before the conversation
    too, and each record the agent's calls and the expected actions that changed it, so a missed
    write, a wrong write and a write with the wrong arguments read apart."""
    agent = build_env(conv, len(conv.messages))
    base, _ = gold_env(conv, upto=0)
    before = _dump(base)
    gold_side, gold_errors = gold_env(conv)
    a_dump, g_dump = _dump(agent), _dump(gold_side)
    match = _hashes(agent) == _hashes(gold_side)

    paths: list[list[str]] = []
    _paths(g_dump, a_dump, [], paths)
    by_record: dict[str, dict[str, Any]] = {}
    for path in paths[:MAX_DIFF_FIELDS]:
        record = ".".join(path[:2]) or "(database)"
        field = ".".join(path[2:])
        g, a = _at(g_dump, path), _at(a_dump, path)
        rec = by_record.setdefault(
            record,
            {
                "record": record,
                "kind": "changed",
                "fields": [],
                "agent_calls": [],
                "gold_actions": [],
            },
        )
        if not field:
            rec["kind"] = "agent_only" if g is None else "gold_only" if a is None else "changed"
        rec["fields"].append(
            {
                "field": field or "(record)",
                "before": _short(_at(before, path), 4000 if not field else 400),
                "agent": _short(a, 4000 if not field else 400),
                "gold": _short(g, 4000 if not field else 400),
            }
        )
    for record, who in _agent_writes(conv).items():
        if record in by_record:
            by_record[record]["agent_calls"] = who
    for record, who in _gold_writes(conv).items():
        if record in by_record:
            by_record[record]["gold_actions"] = who
    for rec in by_record.values():
        rec["verdict"] = record_verdict(rec)
    return {
        "match": match,
        "records": list(by_record.values()),
        "fields": len(paths),
        "shown": min(len(paths), MAX_DIFF_FIELDS),
        "gold_errors": gold_errors,
    }


def record_verdict(rec: dict[str, Any]) -> str:
    """Why one record differs: gold changed it and the agent did not (`missed write`), the agent
    changed it and gold did not (`wrong write`), or both did and the results differ (`wrong
    arguments`); `differs` when no recorded write on either side touched it directly."""
    mine, gold = bool(rec["agent_calls"]), bool(rec["gold_actions"])
    if gold and not mine:
        return "missed write"
    if mine and not gold:
        return "wrong write"
    return "wrong arguments" if mine and gold else "differs"


def action_diff(conv: Conversation) -> dict[str, Any]:
    """The expected actions against the conversation's calls, paired as tau2's action check pairs
    them (`Action.compare_with_tool_call`: the same tool, the same arguments).

    Each expected action is matched at a message or missing; a missing one names the nearest call
    of the same tool that matches no other expected action, and the arguments that differ. `unexpected_writes` are the state-changing
    calls no expected action matches, each marked when the tool refused it. `graded` says whether
    the task's reward basis scores the actions themselves (ACTION); without it an action counts
    only through the database it produces."""
    crit = conv.task.evaluation_criteria
    actions = list((crit.actions or []) if crit else [])
    basis = [str(getattr(b, "value", b)) for b in ((crit.reward_basis or []) if crit else [])]
    mutating = _mutating(conv.domain)
    refused = {
        m.id
        for m in conv.messages
        if getattr(m, "role", None) == "tool" and getattr(m, "error", False)
    }
    calls = [
        (i, tc)
        for i, m in enumerate(conv.messages)
        if getattr(m, "role", None) in ("assistant", "user")
        for tc in (getattr(m, "tool_calls", None) or [])
    ]

    def call(i: int, tc: Any) -> dict[str, Any]:
        return {"msg": i, "name": tc.name, "arguments": tc.arguments, "refused": tc.id in refused}

    expected = []
    for a in actions:
        hit = next(((i, tc) for i, tc in calls if a.compare_with_tool_call(tc)), None)
        row: dict[str, Any] = {
            "action_id": a.action_id,
            "name": a.name,
            "arguments": a.arguments,
            "write": a.name in mutating,
            "matched_at": hit[0] if hit else None,
            "nearest": None,
        }
        if hit is None:
            # a call that is another expected action's match is not a near miss for this one
            same = [
                (i, tc, _arg_diff(a, tc))
                for i, tc in calls
                if tc.name == a.name and not any(b.compare_with_tool_call(tc) for b in actions)
            ]
            if same:
                i, tc, differs = min(same, key=lambda x: len(x[2]))
                row["nearest"] = {**call(i, tc), "differs": differs}
        expected.append(row)
    unexpected = [
        call(i, tc)
        for i, tc in calls
        if tc.name in mutating and not any(a.compare_with_tool_call(tc) for a in actions)
    ]
    return {
        "graded": "ACTION" in basis,
        "basis": basis,
        "expected": expected,
        "called": sorted({tc.name for _, tc in calls}),
        "unexpected_writes": unexpected,
    }


def _arg_diff(action: Any, tc: Any) -> list[dict[str, Any]]:
    """The arguments tau2 compares that differ between an expected action and a call: the
    action's `compare_args`, else every argument the call passed (an expected argument the call
    left out is not compared)."""
    keys = action.compare_args if action.compare_args is not None else tc.arguments.keys()
    return [
        {
            "argument": k,
            "agent": _short(tc.arguments.get(k)),
            "expected": _short(action.arguments.get(k)),
        }
        for k in sorted(keys)
        if tc.arguments.get(k) != action.arguments.get(k)
    ]


def _agent_writes(conv: Conversation) -> dict[str, list[dict[str, Any]]]:
    """Each record a state-changing call of the conversation changed, with the calls that did."""
    mutating = _mutating(conv.domain)
    out: dict[str, list[dict[str, Any]]] = {}
    msgs = conv.messages
    for i, m in enumerate(msgs):
        calls = [tc for tc in (getattr(m, "tool_calls", None) or []) if tc.name in mutating]
        if not calls:
            continue
        j = i + 1
        while j < len(msgs) and getattr(msgs[j], "role", None) == "tool":
            j += 1
        pre, post = build_env(conv, i), build_env(conv, j)
        if _hashes(pre) == _hashes(post):
            continue  # every call here was refused: nothing changed
        for record in _records(_dump(pre), _dump(post)):
            out.setdefault(record, []).extend(
                {"msg": i, "name": tc.name, "arguments": tc.arguments, "requestor": tc.requestor}
                for tc in calls
            )
    return out


def _gold_writes(conv: Conversation) -> dict[str, list[dict[str, Any]]]:
    """Each record an expected action changed, with the actions that did."""
    actions = list(
        (conv.task.evaluation_criteria.actions or []) if conv.task.evaluation_criteria else []
    )
    out: dict[str, list[dict[str, Any]]] = {}
    env, _ = gold_env(conv, upto=0)
    prev, h = _dump(env), _hashes(env)
    for a in actions:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:  # noqa: BLE001, S112 - a refused gold action changes nothing
            continue
        if _hashes(env) == h:
            continue
        cur, h = _dump(env), _hashes(env)
        for record in _records(prev, cur):
            out.setdefault(record, []).append(
                {"action_id": a.action_id, "name": a.name, "arguments": a.arguments}
            )
        prev = cur
    return out


def _mutating(domain: str) -> set[str]:
    ext = read_task_extract(domain)
    return {
        t["name"] for t in [*ext.get("tools", []), *ext.get("user_tools", [])] if t.get("mutates")
    }


def _records(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    paths: list[list[str]] = []
    _paths(a, b, [], paths)
    return {".".join(p[:2]) or "(database)" for p in paths}


def _paths(a: Any, b: Any, path: list[str], out: list[list[str]]) -> None:
    """Every leaf path where `a` and `b` differ, walked as `_leaves` walks them."""
    if a == b:
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            _paths(a.get(k), b.get(k), [*path, str(k)], out)
        return
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and len(path) < 2:
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _paths(x, y, [*path, str(i)], out)
        return
    out.append(path)


def _at(d: Any, path: list[str]) -> Any:
    for k in path:
        if isinstance(d, dict):
            d = d.get(k)
        elif isinstance(d, list) and k.isdigit() and int(k) < len(d):
            d = d[int(k)]
        else:
            return None
    return d


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


def _short(v: Any, cap: int = 400) -> str | None:
    if v is None:
        return None
    s = json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= cap else f"{s[: cap - 1]}…"


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
