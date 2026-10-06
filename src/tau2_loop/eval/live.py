"""A live conversation (s16): one train task played by one version, streamed to the viewer.

The Agent tab starts it and polls it. It is built exactly as a run builds a conversation (the
run's own `TextRunConfig`, tau2's `build_orchestrator`, our agent and customer, the same seed)
and scored by tau2's own `run_simulation`, so the reward at the end is the one an eval would
give. Only the loop is watched: each orchestrator step appends the messages it added to
`events.jsonl`, and a version with workflow tools (v6) also streams each lookup and each
workflow_rag research session from inside the agent's turn.

It is not a run: it lives in `live_runs/<id>/`, is never logged to MLflow, never joins a
ledger, a gate or the judge's data, and plays only train tasks.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau2_loop.agent.versions import load_version
from tau2_loop.config import ROOT, SPLIT_SEED, quiet_tau2
from tau2_loop.data.splits import split_ids
from tau2_loop.llm import sdk_model

LIVE_DIR = ROOT / "live_runs"


class LiveError(ValueError):
    """A conversation that cannot be started: an unknown version, a test task."""


@dataclass
class LiveMeta:
    id: str
    domain: str
    agent: str
    fingerprint: str
    task_id: str
    model: str
    user_model: str
    workflows: str | None
    started: str
    status: str = "running"  # running | done | failed | interrupted
    n_messages: int = 0
    reward: float | None = None
    correct: bool | None = None
    action_checks: str | None = None
    actions_done: float | None = None
    termination: str | None = None
    duration_ms: int = 0
    error: str | None = None


_running: set[str] = set()
_lock = threading.Lock()


def _dir(live_id: str) -> Path:
    import re

    if not re.fullmatch(r"[0-9TZ]+_[a-z_]+_v\d+_task_\d+", live_id):
        raise LiveError(f"no live conversation {live_id}")
    return LIVE_DIR / live_id


def _write_meta(d: Path, meta: LiveMeta) -> None:
    (d / "run.json").write_text(json.dumps(asdict(meta), indent=1) + "\n")


def start(
    domain: str, agent: str, task_id: str, user_model: str = "", background: bool = True
) -> LiveMeta:
    """Start a conversation on a train task; returns at once, the conversation in a thread."""
    from tau2_loop.eval.runner import USER_MODEL

    if task_id not in set(split_ids(domain, "train")):
        raise LiveError(f"{task_id} is not a train task: test is sealed")
    try:
        version = load_version(domain, agent)
    except FileNotFoundError as e:
        raise LiveError(str(e)) from e
    from tau2_loop.llm import require_live

    require_live()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = LiveMeta(
        id=f"{stamp}_{domain}_{agent}_{task_id}",
        domain=domain,
        agent=agent,
        fingerprint=version.fingerprint,
        task_id=task_id,
        model=sdk_model(version.config.model),
        user_model=sdk_model(user_model or USER_MODEL),
        workflows=version.config.workflows,
        started=stamp,
    )
    d = _dir(meta.id)
    d.mkdir(parents=True, exist_ok=False)
    _write_meta(d, meta)
    with _lock:
        _running.add(meta.id)
    if not background:
        _play(d, meta, user_model or USER_MODEL)
        return meta
    threading.Thread(
        target=_play, args=(d, meta, user_model or USER_MODEL), daemon=True, name=f"live-{meta.id}"
    ).start()
    return meta


def _play(d: Path, meta: LiveMeta, user_model: str) -> None:
    events = d / "events.jsonl"
    t0 = time.time()
    lock = threading.Lock()

    def emit(row: dict[str, Any]) -> None:
        row = {"t": int((time.time() - t0) * 1000), **row}
        with lock, events.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    try:
        quiet_tau2()
        from tau2.evaluator.evaluator import EvaluationType
        from tau2.runner.build import _build_env_kwargs, build_orchestrator
        from tau2.runner.helpers import get_tasks
        from tau2.runner.simulation import run_simulation

        from tau2_loop.agent.factory import LoopAgent
        from tau2_loop.agent.factory import register as register_agent
        from tau2_loop.eval.results import from_simulation
        from tau2_loop.eval.runner import _point_judge_at_sdk, _run_config
        from tau2_loop.llm import scrub_injected_key

        register_agent()
        _point_judge_at_sdk()
        scrub_injected_key()
        version = load_version(meta.domain, meta.agent)
        config = _run_config(meta.domain, version, [meta.task_id], 1, 1, SPLIT_SEED, user_model)
        task = get_tasks(meta.domain, config.task_split_name, task_ids=[meta.task_id])[0]
        orch = build_orchestrator(config, task, seed=SPLIT_SEED, simulation_id=meta.id)
        emit({"kind": "start", "task_id": meta.task_id, "agent": meta.agent})
        sent = {"n": 0}
        if isinstance(orch.agent, LoopAgent):

            def agent_event(row: dict[str, Any]) -> None:
                # placed before the message the agent is writing: the trajectory's length now
                emit({**row, "at": len(orch.trajectory)})

            orch.agent.live = agent_event
            orch.agent.conversation_id = meta.id

        def flush() -> None:
            traj = list(orch.trajectory)
            for i in range(sent["n"], len(traj)):
                emit({"kind": "message", "i": i, "message": traj[i].model_dump(mode="json")})
            sent["n"] = len(traj)
            meta.n_messages = len(traj)

        initialize, step = orch.initialize, orch.step

        def initialize_and_flush() -> None:
            initialize()
            flush()

        def step_and_flush() -> None:
            try:
                step()
            finally:
                flush()

        orch.initialize = initialize_and_flush
        orch.step = step_and_flush
        sim = run_simulation(
            orch,
            evaluation_type=EvaluationType.ALL,
            env_kwargs=_build_env_kwargs(config, task) or None,
        )
        flush()
        (d / "simulation.json").write_text(
            json.dumps(sim.model_dump(mode="json"), ensure_ascii=False, indent=1)
        )
        row = from_simulation(sim, task, "simulation.json")
        meta.reward = row.reward
        meta.correct = row.correct
        meta.action_checks = row.action_checks
        meta.actions_done = row.partial_action_reward
        meta.termination = row.termination_reason
        meta.status = "done"
        ri = sim.reward_info
        emit(
            {
                "kind": "done",
                "reward": row.reward,
                "correct": row.correct,
                "action_checks": [
                    {
                        "name": c.action.name,
                        "requestor": c.action.requestor,
                        "arguments": c.action.arguments,
                        "matched": bool(c.action_match),
                    }
                    for c in (ri.action_checks or [] if ri else [])
                ],
                "db_match": bool(ri.db_check.db_match) if ri and ri.db_check else None,
            }
        )
    except Exception as e:  # noqa: BLE001 - a failed conversation is recorded, not raised
        meta.status = "failed"
        meta.error = f"{type(e).__name__}: {e}"[:500]
        emit({"kind": "error", "text": meta.error, "trace": traceback.format_exc()[-2000:]})
    finally:
        meta.duration_ms = int((time.time() - t0) * 1000)
        _write_meta(d, meta)
        with _lock:
            _running.discard(meta.id)


def _read_meta(d: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((d / "run.json").read_text())
    with _lock:
        running = meta["id"] in _running
    if meta["status"] == "running" and not running:
        meta["status"] = "interrupted"  # the server restarted under it
    return meta


def conversations(domain: str, agent: str | None = None) -> list[dict[str, Any]]:
    if not LIVE_DIR.is_dir():
        return []
    out = []
    for d in sorted(LIVE_DIR.iterdir(), reverse=True):
        if not (d / "run.json").is_file():
            continue
        m = _read_meta(d)
        if m["domain"] == domain and (agent is None or m["agent"] == agent):
            out.append(m)
    return out


def conversation(live_id: str, since: int = 0) -> dict[str, Any]:
    """The conversation as it stands; `since` skips events the viewer already has."""
    d = _dir(live_id)
    if not (d / "run.json").is_file():
        raise LiveError(f"no live conversation {live_id}")
    lines = (d / "events.jsonl").read_text().splitlines() if (d / "events.jsonl").is_file() else []
    events = [json.loads(x) for x in lines[since:] if x.strip()]
    return {"meta": _read_meta(d), "events": events, "next": len(lines)}
