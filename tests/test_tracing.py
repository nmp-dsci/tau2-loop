"""The trace rebuilt from a τ² transcript, offline: no server, no client library.

`emit_spans` takes whatever client it is handed, so a recorder stands in for
MlflowClient and the shape of the tree can be asserted without the platform.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from tau2_loop.tracking.mlflow_log import required_tags
from tau2_loop.tracking.prompts import prompt_name
from tau2_loop.tracking.tracing import CUT, _cut, emit_spans, fresh_ids


@dataclass
class _Span:
    span_id: str


class Recorder:
    """Records what a client was asked to do, in order."""

    def __init__(self) -> None:
        self.started: list[dict[str, Any]] = []
        self.ended: list[dict[str, Any]] = []
        self._n = 0

    def start_span(self, **kw: Any) -> _Span:
        self._n += 1
        kw["span_id"] = f"s{self._n}"
        self.started.append(kw)
        return _Span(kw["span_id"])

    def end_span(self, **kw: Any) -> None:
        self.ended.append(kw)


def _msg(role: str, at: str, **kw: Any) -> dict[str, Any]:
    return {"role": role, "timestamp": f"2026-09-16T03:10:{at}", **kw}


TRANSCRIPT = [
    _msg(
        "assistant", "33.000000", content="Hi", usage={"prompt_tokens": 10, "completion_tokens": 3}
    ),
    _msg("user", "37.000000", content="I need help", generation_time_seconds=4.0),
    _msg(
        "assistant",
        "45.000000",
        content=None,
        generation_time_seconds=7.0,
        tool_calls=[{"id": "c1", "name": "get_reservation_details", "arguments": {"id": "H9ZU1C"}}],
    ),
    _msg("tool", "45.100000", id="c1", content="{}", error=False),
    _msg("assistant", "50.000000", content="Done"),
]


def test_a_transcript_becomes_a_turn_tree() -> None:
    rec = Recorder()
    end = emit_spans(rec, "tr-1", "root", TRANSCRIPT, 0)
    names = [s["name"] for s in rec.started]
    assert names == ["turn 1", "user 1", "turn 2", "get_reservation_details", "turn 3"]
    # both sides are traced: the simulated user is a model call too
    assert [s["span_type"] for s in rec.started] == ["LLM", "LLM", "LLM", "TOOL", "LLM"]
    # the tool span hangs off the turn that called it, not off the root
    tool = next(s for s in rec.started if s["name"] == "get_reservation_details")
    turn2 = next(s for s in rec.started if s["name"] == "turn 2")
    assert tool["parent_id"] == turn2["span_id"]
    # every span is closed, and the conversation ends at the last message
    assert len(rec.ended) == len(rec.started)
    assert end == int(datetime.fromisoformat("2026-09-16T03:10:50").timestamp() * 1e9)


def test_a_turn_is_as_long_as_the_model_took() -> None:
    rec = Recorder()
    emit_spans(rec, "tr-1", "root", TRANSCRIPT, 0)
    turn2 = next(s for s in rec.started if s["name"] == "turn 2")
    end = rec.ended[2]
    assert round((end["end_time_ns"] - turn2["start_time_ns"]) / 1e9, 2) == 7.0


def test_an_unanswered_tool_call_is_closed_as_an_error() -> None:
    rec = Recorder()
    cut_short = TRANSCRIPT[:3]  # the tool result never arrives
    emit_spans(rec, "tr-1", "root", cut_short, 0)
    tool_end = next(e for e in rec.ended if e["span_id"] == "s4")
    assert tool_end["status"] == "ERROR"


def test_long_values_are_trimmed_with_the_length_kept() -> None:
    long = "x" * (CUT + 500)
    out = _cut(long)
    assert out.startswith("x" * 100) and out.endswith("[500 more]") and len(out) < len(long)
    assert _cut({"a": long, "b": 3})["b"] == 3
    assert len(_cut([long] * 80)) == 50


def test_two_runs_with_one_seed_never_share_a_trace_id() -> None:
    """τ² seeds the global `random` with the run's seed; OpenTelemetry draws its ids from it."""
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    ids = RandomIdGenerator()

    def run() -> list[int]:
        random.seed(300)  # what τ² does at the start of every run
        with fresh_ids():
            return [ids.generate_trace_id() for _ in range(20)]

    first, second = run(), run()
    assert not set(first) & set(second)
    # unprotected, the same seed gives the same ids: the collision this guards against
    random.seed(300)
    bare = ids.generate_trace_id()
    random.seed(300)
    assert ids.generate_trace_id() == bare
    # the seeded stream outside the block is left where it was
    random.seed(300)
    expected = [random.random() for _ in range(3)]
    random.seed(300)
    with fresh_ids():
        ids.generate_trace_id()
    assert [random.random() for _ in range(3)] == expected


def test_a_domain_logs_to_its_own_experiment(monkeypatch: Any) -> None:
    """Since 6 Oct 2026 each domain has `tau2-loop/<domain>`; the flat `tau2-loop` takes nothing new."""
    import mlflow

    from tau2_loop.tracking.mlflow_log import log_cycle

    chosen: list[str] = []

    def stop(**_: Any) -> None:
        raise RuntimeError("no server in tests")

    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda _uri: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: chosen.append(name))
    monkeypatch.setattr(mlflow, "start_run", stop)
    assert log_cycle({"domain": "banking_knowledge", "cycle": 1}) is None
    assert chosen == ["tau2-loop/banking_knowledge"]


def test_the_platform_tags_are_on_every_trace() -> None:
    assert set(required_tags()) == {"project", "git_sha", "env", "billing"}
    assert required_tags()["project"] == "tau2-loop"
    assert required_tags(billing="api")["billing"] == "api"


def test_a_prompt_is_named_for_its_domain() -> None:
    assert prompt_name("banking_knowledge") == "tau2-loop.banking_knowledge.system"
