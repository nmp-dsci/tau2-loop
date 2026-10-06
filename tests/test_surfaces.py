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


def test_a_transfer_is_checked_though_tau2_does_not_type_it_a_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """s15: a reason-code rule (banking doc _042's tiers) must see the transfer before it ends the call."""
    checks = (
        "def check_write(name, arguments, state):\n"
        "    if name == 'transfer_to_human_agents' and arguments.get('reason') == 'other':\n"
        "        return 'a more specific reason code applies.'\n"
        "    return None\n"
    )
    v = make_version(tmp_path, monkeypatch, checks_py=checks)
    script = Script(
        [
            calls(("c1", "transfer_to_human_agents", {"reason": "other"})),
            calls(("c2", "transfer_to_human_agents", {"reason": "fraud_or_security_concern"})),
        ]
    )
    monkeypatch.setattr(llm_utils, "generate", script)
    agent = LoopAgent(tools=[], domain_policy="P", version=v)
    assert agent.kinds.get("transfer_to_human_agents") != "write"
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="get me a person"), agent.get_init_state()
    )
    assert len(script.calls) == 2 and "more specific reason" in script.calls[1][-1].content
    assert [tc.arguments["reason"] for tc in reply.tool_calls] == ["fraud_or_security_concern"]
    assert state.n_blocked == 1


def test_a_refused_text_reply_goes_back_once_and_never_reaches_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checks = (
        "def check_write(name, arguments, state):\n    return None\n\n"
        "def check_reply(text, state):\n"
        "    if state.get('asks', 0) >= 4 and 'transfer' not in text.lower():\n"
        "        return 'the customer has asked for a human four times: transfer them.'\n"
        "    return None\n"
    )
    memory = (
        "def remember(state, name, arguments, result):\n"
        "    if name == 'user' and 'human' in result:\n"
        "        state['asks'] = state.get('asks', 0) + 1\n"
    )
    v = make_version(tmp_path, monkeypatch, checks_py=checks, memory_py=memory)
    script = Script([say("Let me help you first."), say("I will transfer you now.")])
    monkeypatch.setattr(llm_utils, "generate", script)
    agent = LoopAgent(tools=[], domain_policy="P", version=v)
    state = agent.get_init_state()
    state.memory["asks"] = 3
    reply, state = agent.generate_next_message(
        UserMessage(role="user", content="a human, please"), state
    )
    assert len(script.calls) == 2
    sent_back = script.calls[1][-1]
    assert (
        isinstance(sent_back, SystemMessage) and "asked for a human four times" in sent_back.content
    )
    assert reply.content == "I will transfer you now." and state.n_blocked == 1
    assert reply.raw_data["tau2_loop"]["blocked_reply"]["content"] == "Let me help you first."
    assert state.messages[-1] is reply and len(state.messages) == 2
    # a reply the check lets through is sent as written, with one model call
    script2 = Script([say("Hello.")])
    monkeypatch.setattr(llm_utils, "generate", script2)
    agent2 = LoopAgent(tools=[], domain_policy="P", version=v)
    r2, _ = agent2.generate_next_message(
        UserMessage(role="user", content="hi"), agent2.get_init_state()
    )
    assert len(script2.calls) == 1 and r2.content == "Hello."
