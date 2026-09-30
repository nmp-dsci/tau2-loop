"""The s09 code surfaces around a turn: memory, per-turn guidance and write-time checks.

No model is called: tau2's `generate` is replaced by a script of replies, and the
tests read what each call was sent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tau2.utils.llm_utils as llm_utils
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

import tau2_loop.agent.versions as versions
from tau2_loop.agent.compose import GUIDANCE_CHARS, hooks_defined, load_code_surfaces
from tau2_loop.agent.factory import LoopAgent
from tau2_loop.agent.versions import load_version

MEMORY = """
def remember(state, name, arguments, result):
    state.setdefault("seen", []).append(name)
    if name == "get_user_details":
        state["user"] = arguments.get("user_id")
"""
GUIDANCE = """
def guidance(state, trigger):
    if trigger == "user":
        return "Reminder: verified user is " + str(state.get("user")) + "." + " pad" * 400
    return None
"""
CHECKS = """
def check_write(name, arguments, state):
    if arguments.get("origin") == arguments.get("destination"):
        return "a round trip's destination cannot be its origin."
    return None
"""


def make_version(root: Path, monkeypatch: pytest.MonkeyPatch, **files: str) -> Any:
    monkeypatch.setattr(versions, "AGENTS_DIR", root)
    d = root / "airline" / "v9"
    d.mkdir(parents=True)
    (d / "system.md").write_text("You are an airline agent.\n{policy}")
    (d / "agent.yaml").write_text("model: haiku\neffort: medium\n")
    for name, text in files.items():
        (d / name.replace("_py", ".py")).write_text(text)
    return load_version("airline", "v9")


class Script:
    """tau2's generate, replaced: returns the scripted replies in order and keeps each call's messages."""

    def __init__(self, replies: list[AssistantMessage]):
        self.replies = list(replies)
        self.calls: list[list[Any]] = []

    def __call__(self, *, messages: list[Any], **_: Any) -> AssistantMessage:
        self.calls.append(list(messages))
        return self.replies.pop(0)


def say(text: str) -> AssistantMessage:
    return AssistantMessage(role="assistant", content=text)


def calls(*tcs: tuple[str, str, dict[str, Any]]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[ToolCall(id=i, name=n, arguments=a, requestor="assistant") for i, n, a in tcs],
    )


def test_a_version_without_code_surfaces_keeps_its_fingerprint_and_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = make_version(tmp_path, monkeypatch)
    assert v.code == {} and set(v.files()) == {"system.md", "agent.yaml"}
    assert load_version("airline", "v9").fingerprint == v.fingerprint
    hooks = hooks_defined(None, load_code_surfaces(v.path, "t"))
    assert not any(hooks.values()) and set(hooks) >= {"remember", "guidance", "check_write"}


def test_guidance_rides_on_the_call_only_and_memory_reads_each_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = make_version(tmp_path, monkeypatch, memory_py=MEMORY, guidance_py=GUIDANCE)
    assert set(v.files()) >= {"memory.py", "guidance.py"}
    script = Script([calls(("c1", "get_user_details", {"user_id": "u_1"})), say("Found you.")])
    monkeypatch.setattr(llm_utils, "generate", script)
    agent = LoopAgent(tools=[], domain_policy="P", version=v)
    state = agent.get_init_state()

    first, state = agent.generate_next_message(UserMessage(role="user", content="hi"), state)
    sent = script.calls[0]
    assert isinstance(sent[-1], SystemMessage) and sent[-1].content.startswith("Reminder")
    assert len(sent[-1].content) == GUIDANCE_CHARS
    assert first.raw_data and first.raw_data["tau2_loop"]["guidance"] == sent[-1].content
    # the reminder is not in the transcript the next call is built from
    assert not any(isinstance(m, SystemMessage) for m in state.messages)

    result = ToolMessage(id="c1", role="tool", requestor="assistant", content='{"name": "Ana"}')
    second, state = agent.generate_next_message(result, state)
    assert state.memory == {"seen": ["user", "get_user_details"], "user": "u_1"}
    # a tool result is not a user turn: this guidance says nothing, so nothing rides on the call
    assert not isinstance(script.calls[1][-1], SystemMessage)
    assert second.content == "Found you." and not (second.raw_data or {}).get("tau2_loop")
    # a new conversation starts with an empty memory
    assert agent.get_init_state().memory == {}


def test_a_failed_check_blocks_the_reply_once_and_the_retry_goes_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = make_version(tmp_path, monkeypatch, checks_py=CHECKS)
    bad = {"origin": "JFK", "destination": "JFK"}
    script = Script(
        [
            calls(("c1", "book_reservation", bad), ("c2", "get_user_details", {"user_id": "u"})),
            calls(("c3", "book_reservation", bad)),  # the retry is not checked again
        ]
    )
    monkeypatch.setattr(llm_utils, "generate", script)
    agent = LoopAgent(tools=[], domain_policy="P", version=v)
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="book it"), agent.get_init_state()
    )
    assert len(script.calls) == 2
    retry_call = script.calls[1]
    blocked, held = retry_call[-2], retry_call[-1]
    assert blocked.id == "c1" and "destination cannot be its origin" in blocked.content
    assert held.id == "c2" and "another call" in held.content
    assert [tc.id for tc in reply.tool_calls] == ["c3"]
    note = reply.raw_data["tau2_loop"]
    assert note["retried"] and note["blocked"] == [
        {
            "name": "book_reservation",
            "arguments": bad,
            "check": "a round trip's destination cannot be its origin.",
        }
    ]
    # the blocked attempt is not in the transcript tau2 keeps
    assert state.messages[-1] is reply and len(state.messages) == 2 and state.n_blocked == 1


def test_reads_are_never_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    v = make_version(
        tmp_path, monkeypatch, checks_py="def check_write(n, a, s):\n    return 'no'\n"
    )
    script = Script([calls(("c1", "get_user_details", {"user_id": "u"}))])
    monkeypatch.setattr(llm_utils, "generate", script)
    agent = LoopAgent(tools=[], domain_policy="P", version=v)
    reply, _ = agent.generate_next_message(
        UserMessage(role="user", content="hi"), agent.get_init_state()
    )
    assert len(script.calls) == 1 and reply.tool_calls[0].id == "c1"
