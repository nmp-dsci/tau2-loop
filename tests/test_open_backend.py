"""s36: the answering agent on DeepSeek through Pydantic AI (`backend: pydantic_ai`).

No test reaches DeepSeek. A mock HTTP transport stands where the API would be, so each request
is the one Pydantic AI's DeepSeek model really builds and each reply goes through its real
parsing; the history, the reasoning sent back, the tools, the reply tau2 keeps, the trajectory
file and the trace spans are read from what the code produces.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from tau2_loop.agent import versions
from tau2_loop.llm import open_backend

SENTINEL = "sk-test-not-a-real-key-36"


def _reply(
    content: str | None = None,
    reasoning: str | None = "Check the line before anything else.",
    calls: list[dict[str, Any]] | None = None,
    hit: int = 900,
) -> dict[str, Any]:
    """A DeepSeek chat completion as its API sends it."""
    return {
        "id": "resp-1",
        "object": "chat.completion",
        "created": 1_761_350_400,
        "model": "deepseek-v4-flash-1017",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if calls else "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                    "tool_calls": calls,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 120,
            "total_tokens": 1120,
            "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": 1000 - hit,
            "completion_tokens_details": {"reasoning_tokens": 80},
        },
    }


def _call(name: str = "get_customer_by_phone", args: str = '{"phone_number": "555-0100"}') -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": args}}


class _DeepSeek:
    """The API: records every request body and answers with the replies given, in order."""

    def __init__(self, *replies: dict[str, Any], status: int = 200) -> None:
        self.replies = list(replies)
        self.status = status
        self.requests: list[dict[str, Any]] = []
        self.auth: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.auth.append(request.headers.get("authorization", ""))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": "no"}})
        return httpx.Response(200, json=self.replies.pop(0))


@pytest.fixture
def api(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Point the backend at a mock API, with a key it must never write anywhere."""
    monkeypatch.setenv(open_backend.KEY, SENTINEL)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    real = open_backend.build_model
    box: dict[str, _DeepSeek] = {}

    def install(*replies: dict[str, Any], status: int = 200) -> _DeepSeek:
        box["api"] = _DeepSeek(*replies, status=status)
        transport = httpx.MockTransport(box["api"])
        monkeypatch.setattr(
            open_backend, "build_model", lambda m, e: real(m, e, transport=transport)
        )
        return box["api"]

    open_backend.reset()
    yield install
    open_backend.reset()


def _tools() -> list[Any]:
    from tau2.registry import registry

    return list(registry.get_env_constructor("telecom")().get_tools())


def _history() -> list[Any]:
    first = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="call_0", name="get_customer_by_phone", arguments={"phone_number": "5"})
        ],
        raw_data=_reply(reasoning="Find the customer first.", calls=[_call()]),
        usage={"prompt_tokens": 800, "completion_tokens": 40, "prompt_cache_hit_tokens": 0},
    )
    return [
        SystemMessage(role="system", content="You are telecom's agent. POLICY"),
        UserMessage(role="user", content="My mobile data stopped working."),
        first,
        ToolMessage(
            id="call_0", role="tool", requestor="assistant", content='{"customer_id": "C1"}'
        ),
    ]


# ── the request ──────────────────────────────────────────────────────────


def test_each_earlier_reply_goes_back_with_its_reasoning_and_the_tools_go_as_definitions(
    api: Any,
) -> None:
    deepseek = api(_reply(calls=[_call("get_details_by_id", '{"id": "C1"}')]))
    tools = _tools()
    open_backend.generate("deepseek-flash", "high", _history(), tools)

    body = deepseek.requests[0]
    assert body["model"] == "deepseek-flash"
    assert body["reasoning_effort"] == "high"  # thinking at the version's effort
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    earlier = body["messages"][2]
    # DeepSeek 400s on a tool-call turn sent back without its reasoning
    assert earlier["reasoning_content"] == "Find the customer first."
    assert earlier["tool_calls"][0]["function"]["name"] == "get_customer_by_phone"
    assert body["messages"][3]["tool_call_id"] == "call_0"
    sent = {t["function"]["name"]: t["function"] for t in body["tools"]}
    assert set(sent) == {t.openai_schema["function"]["name"] for t in tools}
    for t in tools:
        fn = t.openai_schema["function"]
        assert sent[fn["name"]]["description"] == fn["description"]
        assert set(sent[fn["name"]]["parameters"]["properties"]) == set(
            fn["parameters"]["properties"]
        )
    assert body.get("tool_choice") in (None, "auto")  # never forced while thinking


def test_the_reply_is_taus_message_with_the_reasoning_word_for_word_in_raw_data(api: Any) -> None:
    api(_reply(content=None, reasoning="Line C1: check data roaming.", calls=[_call()]))
    m = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())

    assert m.role == "assistant" and m.content is None
    assert [(tc.id, tc.name, tc.arguments) for tc in m.tool_calls] == [
        ("call_1", "get_customer_by_phone", {"phone_number": "555-0100"})
    ]
    msg = m.raw_data["choices"][0]["message"]  # the shape tau2 saves for a litellm reply
    assert msg["reasoning_content"] == "Line C1: check data roaming."
    assert msg["tool_calls"][0]["function"]["arguments"] == '{"phone_number":"555-0100"}'
    assert m.raw_data["model"] == "deepseek-v4-flash-1017"  # the model DeepSeek served
    assert m.raw_data["backend"]["effort"] == "high"
    assert m.usage["completion_tokens_details"]["reasoning_tokens"] == 80
    assert m.usage["prompt_cache_hit_tokens"] == 900
    # the card: 900 hits, 100 misses, 120 out, at peak or half off it
    full = (900 * 0.006 + 100 * 0.30 + 120 * 1.20) / 1e6
    assert m.cost == pytest.approx(full if m.raw_data["backend"]["peak"] else full / 2)


def test_the_reasoning_survives_a_saved_run_file_and_rebuilds_the_next_request(api: Any) -> None:
    deepseek = api(
        _reply(reasoning="Ask which phone.", content="Which phone number is it?"),
        _reply(reasoning="Now look it up.", calls=[_call()]),
    )
    m = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    saved = json.loads(json.dumps(m.model_dump(mode="json")))  # as traces/<task>.json holds it
    back = AssistantMessage.model_validate(saved)
    assert open_backend.reasoning_of(back.raw_data) == "Ask which phone."

    history = [*_history()[:2], back, UserMessage(role="user", content="555-0100")]
    open_backend.generate("deepseek-flash", "high", history, _tools())
    replayed = deepseek.requests[1]["messages"][2]
    assert replayed["content"] == "Which phone number is it?"


def test_a_reply_with_no_text_and_no_call_is_asked_for_again_and_charged(api: Any) -> None:
    deepseek = api(_reply(content=None, reasoning="Hmm."), _reply(content="Hello, how can I help?"))
    m = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert m.content == "Hello, how can I help?" and len(deepseek.requests) == 2
    assert m.raw_data["backend"]["attempts"] == 2
    assert open_backend.spent() == pytest.approx(2 * m.cost)


def test_the_harness_keeps_one_call_and_no_text_beside_it_unless_the_version_keeps_them_all(
    api: Any,
) -> None:
    two = [_call(), {**_call("get_details_by_id", '{"id": "C1"}'), "id": "call_2"}]
    api(_reply(content="Let me check.", calls=two), _reply(content="Let me check.", calls=two))
    one = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert one.content is None and [tc.id for tc in one.tool_calls] == ["call_1"]
    assert one.raw_data["backend"]["dropped"] == {"text": True, "calls": 1}
    assert len(one.raw_data["choices"][0]["message"]["tool_calls"]) == 2  # DeepSeek's reply, whole
    both = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools(), parallel=True)
    assert [tc.id for tc in both.tool_calls] == ["call_1", "call_2"]


# ── the key and the bill ─────────────────────────────────────────────────


def test_the_key_goes_in_the_request_header_only(api: Any) -> None:
    deepseek = api(_reply(content="Hi."))
    m = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert deepseek.auth == [f"Bearer {SENTINEL}"]
    assert SENTINEL not in json.dumps(m.model_dump(mode="json"))


def test_the_key_is_not_in_the_run_config_tau2_writes_or_a_session_with_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.data.splits import split_ids
    from tau2_loop.eval import runner
    from tau2_loop.llm import subscription_env

    monkeypatch.setenv(open_backend.KEY, SENTINEL)
    monkeypatch.delenv("AGENT_SERVICE_URL", raising=False)
    v3 = _fork(tmp_path, monkeypatch)
    config = runner._run_config("telecom", v3, split_ids("telecom", "train")[:1], 1, 1, 300)
    assert config.llm_agent == "pydantic_ai/deepseek/deepseek-flash"
    assert SENTINEL not in config.model_dump_json()
    assert subscription_env()[open_backend.KEY] == ""


def test_the_demo_image_refuses_before_any_request(
    api: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.llm import BillingError

    deepseek = api(_reply(content="Hi."))
    monkeypatch.setenv("DEMO_MODE", "1")
    with pytest.raises(BillingError):
        open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert deepseek.requests == []


def test_the_cap_keeps_the_reply_that_crosses_it_and_stops_every_call_after(api: Any) -> None:
    deepseek = api(_reply(content="One."), _reply(content="Two."))
    open_backend.reset(cap_usd=1e-9)
    m = open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert m.content == "One."
    with pytest.raises(open_backend.BackendStoppedError, match="cap"):
        open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert len(deepseek.requests) == 1


def test_no_balance_stops_the_run_at_once(api: Any) -> None:
    deepseek = api(status=402)
    with pytest.raises(open_backend.BackendStoppedError):
        open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    with pytest.raises(open_backend.BackendStoppedError, match="402"):
        open_backend.generate("deepseek-flash", "high", _history()[:2], _tools())
    assert len(deepseek.requests) == 1


# ── the version ──────────────────────────────────────────────────────────


def _fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> versions.AgentVersion:
    real = versions.AGENTS_DIR
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    shutil.copytree(real / "telecom" / "v2", tmp_path / "telecom" / "v2")
    return versions.fork_version(
        "telecom",
        "v2",
        model="deepseek-flash",
        effort="high",
        tool_mode="native",
        backend="pydantic_ai",
        why="s36",
    )


def test_a_fork_onto_deepseek_freezes_the_backend_and_keeps_v2s_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v = _fork(tmp_path, monkeypatch)
    v2 = versions.load_version("telecom", "v2")
    assert v.name == "v3" and v.system_prompt == v2.system_prompt
    assert (v.config.backend, v.config.model, v.config.effort, v.config.tool_mode) == (
        "pydantic_ai",
        "deepseek-flash",
        "high",
        "native",
    )
    assert "backend: pydantic_ai\n" in (v.path / "agent.yaml").read_text()
    d = json.loads((v.path / "diagnosis.json").read_text())
    assert d["kind"] == "model swap" and d["forked_from"] == "v2"
    assert "backend: None → pydantic_ai" in d["agent_yaml"]
    # a later fork of v3 keeps the backend unless told otherwise
    v4 = versions.fork_version("telecom", "v3", effort="max")
    assert v4.config.backend == "pydantic_ai"


@pytest.mark.parametrize(
    ("fields", "why"),
    [
        (
            {
                "model": "deepseek-flash",
                "effort": "high",
                "tool_mode": "json",
                "backend": "pydantic_ai",
            },
            "native",
        ),
        (
            {
                "model": "deepseek-flash",
                "effort": "xhigh",
                "tool_mode": "native",
                "backend": "pydantic_ai",
            },
            "thinks at",
        ),
        (
            {
                "model": "claude-haiku-5-5",
                "effort": "high",
                "tool_mode": "native",
                "backend": "pydantic_ai",
            },
            "runs",
        ),
        ({"model": "deepseek-flash", "effort": "high", "tool_mode": "native"}, "only"),
        ({"model": "haiku", "backend": "litellm"}, "backend must be"),
    ],
)
def test_a_config_the_backend_cannot_run_is_refused(fields: dict[str, Any], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        versions.AgentConfig(**fields)


def test_the_agent_sends_its_composed_prompt_through_the_backend(
    api: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.agent import factory

    deepseek = api(_reply(content="Hello! How can I help with your line today?"))
    v3 = _fork(tmp_path, monkeypatch)
    agent = factory.LoopAgent(tools=_tools(), domain_policy="THE POLICY", version=v3)
    state = agent.get_init_state()
    out, state = agent.generate_next_message(UserMessage(role="user", content="Hi"), state)

    assert out.content.startswith("Hello!") and state.n_model_calls == 1
    system = deepseek.requests[0]["messages"][0]
    assert system["role"] == "system" and system["content"] == agent.system_prompt()
    assert v3.system_prompt.strip()[:200] in system["content"] and "THE POLICY" in system["content"]
    assert agent.system_prompt() == factory.system_prompt_for(v3, "THE POLICY")
    assert open_backend.reasoning_of(state.messages[-1].raw_data)


# ── what a run keeps ─────────────────────────────────────────────────────


def _sim() -> Any:
    """A finished conversation as tau2 returns it, the customer's own tool use included."""
    reply = AssistantMessage(
        role="assistant",
        content="Please toggle airplane mode off.",
        raw_data=_reply(reasoning="Airplane mode is on: ask them to turn it off."),
    )
    return SimpleNamespace(
        id="sim-1",
        task_id="[mobile_data_issue]airplane_mode_on[PERSONA:Easy]",
        trial=0,
        termination_reason="user_stop",
        reward_info=SimpleNamespace(reward=1.0),
        messages=[
            *_history()[1:],
            reply,
            UserMessage(
                role="user",
                content=None,
                tool_calls=[
                    ToolCall(id="u1", name="toggle_airplane_mode", arguments={}, requestor="user")
                ],
            ),
            ToolMessage(id="u1", role="tool", requestor="user", content="Airplane mode OFF"),
            UserMessage(role="user", content="Done, it works. ###STOP###"),
        ],
    )


def test_the_trajectory_is_the_agents_view_with_every_replys_reasoning() -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ThinkingPart, ToolReturnPart

    row = open_backend.trajectory("THE PROMPT", _sim())
    assert (row["task_id"], row["trial"], row["reward"]) == (
        "[mobile_data_issue]airplane_mode_on[PERSONA:Easy]",
        1,
        1.0,
    )
    history = ModelMessagesTypeAdapter.validate_python(row["messages"])
    assert history[0].parts[0].content == "THE PROMPT"
    thinking = [p.content for m in history for p in m.parts if isinstance(p, ThinkingPart)]
    assert thinking == ["Find the customer first.", "Airplane mode is on: ask them to turn it off."]
    returns = [p.tool_name for m in history for p in m.parts if isinstance(p, ToolReturnPart)]
    assert returns == ["get_customer_by_phone"]  # the customer's own call is not the agent's
    assert history[-1].parts[-1].content == "Done, it works. ###STOP###"


def test_the_trajectory_file_round_trips(tmp_path: Path) -> None:
    p = tmp_path / open_backend.TRAJECTORIES
    rows = [open_backend.trajectory("P", _sim())]
    assert open_backend.write_trajectories(p, rows) == 1
    assert open_backend.read_trajectories(p) == json.loads(json.dumps(rows))


class _Spans:
    def __init__(self) -> None:
        self.ended: list[dict[str, Any]] = []

    def start_span(self, **kw: Any) -> Any:
        return SimpleNamespace(span_id=f"s{len(self.ended)}-{kw['name']}")

    def end_span(self, **kw: Any) -> None:
        self.ended.append(kw)


def test_each_turns_span_carries_its_reasoning_cut_like_the_text() -> None:
    from tau2_loop.tracking import tracing

    long = "Reasoning. " * 400
    reply = AssistantMessage(
        role="assistant",
        content="Turn airplane mode off.",
        raw_data=_reply(reasoning=long),
        usage=_reply()["usage"],
    )
    plain = UserMessage(role="user", content="ok")
    client = _Spans()
    tracing.emit_spans(client, "t", "root", [m.model_dump(mode="json") for m in (reply, plain)], 0)
    turn, user = client.ended
    assert turn["outputs"]["reasoning"].startswith("Reasoning. ")
    assert len(turn["outputs"]["reasoning"]) < len(long)
    assert turn["outputs"]["reasoning"][: tracing.CUT] == long[: tracing.CUT]
    assert turn["attributes"]["reasoning_tokens"] == 80
    assert "reasoning" not in user["outputs"] and "reasoning_tokens" not in user["attributes"]


def test_two_long_task_ids_cut_alike_get_two_trace_files() -> None:
    from tau2_loop.eval.results import slug, trace_file

    a = "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting[PERSONA:Hard]"
    b = "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting[PERSONA:None]"
    assert slug(a) == slug(b)  # what overwrote v2's files
    assert trace_file(a, 1, 1) != trace_file(b, 1, 1)
    assert len(trace_file(a, 1, 1)) <= 85
    short = "[mobile_data_issue]airplane_mode_on[PERSONA:Easy]"
    assert trace_file(short, 1, 1) == f"{slug(short)}.json"  # a name that fit is unchanged
    assert trace_file(short, 2, 3) == f"{slug(short)}_t2.json"
