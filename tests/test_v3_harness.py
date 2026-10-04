"""s13 milestones 0–2, offline: per-version retrieval, tool mode and identity note; the local
AllTools variants; native tool calls through the sealed core; and the run-health numbers."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.agent import compose as composing
from tau2_loop.agent import versions
from tau2_loop.eval import health
from tau2_loop.llm import core, prompting

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_information_by_email",
            "description": "Look a customer up.",
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "KB_search_bm25",
            "description": "Search the knowledge base.",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    },
]
MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a bank's agent."},
    {"role": "user", "content": "I'm jane@example.com; what is the ATM fee?"},
]


# ── milestone 0: settings that live in the version ───────────────────────────────────────────
def test_v1_and_v2_keep_their_bytes_and_banking_default_retrieval() -> None:
    """Adding the fields changed no older version: v1's fingerprint is still the one its
    champion runs were logged on, and both still run tau2's bm25_grep."""
    from tau2_loop.eval.runner import load_run

    for name, run in (
        ("v1", "20261002T235638Z_banking_knowledge_v1_train"),
        ("v2", "20261003T005018Z_banking_knowledge_v2_train"),
    ):
        v = versions.load_version("banking_knowledge", name)
        meta, _ = load_run(run)
        assert v.fingerprint == meta.fingerprint
        assert v.retrieval == "bm25_grep" == meta.retrieval
        assert v.config.tool_mode == "json" and v.config.identity_note is False
    assert versions.load_version("airline", "v0").retrieval is None


def test_a_bad_tool_mode_is_refused_and_unset_fields_stay_out_of_agent_yaml() -> None:
    with pytest.raises(ValueError):
        versions.AgentConfig(tool_mode="xml")
    text = versions.AgentConfig(model="sonnet").yaml()
    assert "retrieval" not in text and "identity_note" not in text
    full = versions.AgentConfig(retrieval="alltools_minilm", tool_mode="native", identity_note=True)
    assert "retrieval: alltools_minilm\n" in full.yaml() and "identity_note: true\n" in full.yaml()


def test_a_tool_fork_records_a_tool_change_and_its_runs_follow_its_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = versions.AGENTS_DIR
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    shutil.copytree(real / "banking_knowledge" / "v1", tmp_path / "banking_knowledge" / "v1")
    src = versions.load_version("banking_knowledge", "v1")
    v = versions.fork_version(
        "banking_knowledge",
        "v1",
        retrieval="alltools_minilm",
        tool_mode="native",
        identity_note=True,
    )
    assert v.name == "v2" and v.system_prompt == src.system_prompt
    assert (v.config.retrieval, v.config.tool_mode, v.config.identity_note) == (
        "alltools_minilm",
        "native",
        True,
    )
    assert v.retrieval == "alltools_minilm" and v.config.model == src.config.model
    d = json.loads((v.path / "diagnosis.json").read_text())
    assert d["kind"] == "tool change" and d["forked_from"] == "v1"
    assert any("retrieval: None → alltools_minilm" in c for c in d["agent_yaml"])
    with pytest.raises(ValueError):
        versions.fork_version("banking_knowledge", "v2", tool_mode="native")  # nothing changes


def test_the_identity_note_is_added_only_for_a_version_that_asks() -> None:
    plain = composing.compose("Be exact.\n{policy}", "POLICY", None)
    noted = composing.compose("Be exact.\n{policy}", "POLICY", None, identity=True)
    assert composing.IDENTITY_NOTE not in plain.text and plain.identity_note is None
    assert noted.text == f"{plain.text}\n\n{composing.IDENTITY_NOTE}"
    assert noted.text.index(composing.CLOCK_NOTE) < noted.text.index(composing.IDENTITY_NOTE)


# ── milestone 1: the local AllTools variants ─────────────────────────────────────────────────
def test_the_local_variants_are_alltools_with_a_local_dense_model() -> None:
    from tau2_loop.eval import retrieval

    retrieval.register()
    retrieval.register()  # idempotent
    assert retrieval.variant_tools("bm25_grep") == ["KB_search", "grep"]
    for name in retrieval.LOCAL_VARIANTS:
        s = retrieval.variant_summary(name)
        assert s["tools"] == ["KB_search_bm25", "KB_search_dense", "shell"]
        assert s["template"] == "all_tools.md"
        assert s["dense_model"] == f"local:{retrieval.variant_model(name)}"
    assert retrieval.variant_tools("no_such_variant") == [] and retrieval.variant_tools(None) == []


def test_the_runner_hands_tau2_the_versions_own_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.eval import runner

    real = versions.AGENTS_DIR
    monkeypatch.setattr(versions, "AGENTS_DIR", tmp_path)
    shutil.copytree(real / "banking_knowledge" / "v1", tmp_path / "banking_knowledge" / "v1")
    v = versions.fork_version("banking_knowledge", "v1", retrieval="alltools_minilm")
    monkeypatch.delenv("AGENT_SERVICE_URL", raising=False)
    cfg = runner._run_config("banking_knowledge", v, ["task_073"], 1, 1, 300)
    assert cfg.retrieval_config == "alltools_minilm"
    old = runner._run_config(
        "banking_knowledge", versions.load_version("banking_knowledge", "v1"), ["t"], 1, 1, 300
    )
    assert old.retrieval_config == "bm25_grep"


# ── milestone 2: native tool calls ───────────────────────────────────────────────────────────
def test_native_blocks_carry_no_contract_and_the_json_blocks_are_unchanged() -> None:
    system, blocks = prompting.build_blocks(MESSAGES, TOOLS, native=True)
    assert "# Tools" not in system and blocks[0] == prompting.NATIVE_ASK
    j_system, j_blocks = prompting.build_blocks(MESSAGES, TOOLS)
    assert j_system.startswith("You are a bank's agent.") and "# Tools" in j_system
    assert prompting.build_blocks(MESSAGES, TOOLS, native=False) == (j_system, j_blocks)
    assert blocks[1:] == j_blocks[1:]  # the transcript itself is the same either way


class _Native:
    def __init__(self, uses: list[dict[str, Any]], text: str = "") -> None:
        self.uses, self.text = uses, text
        self.seen: list[Any] = []

    def __call__(self, system: str, user: Any, model: str, effort: str = "medium", **kw: Any):
        self.seen.append(kw.get("native_tools"))
        r = core.SdkResult(self.text, 10, 2, None, 1, "s")
        r.tool_uses = list(self.uses)
        return r


def test_a_native_answer_returns_the_models_calls_and_drops_text_beside_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _Native(
        [{"id": "toolu_1", "name": "KB_search_bm25", "input": {"query": "ATM fee"}}],
        text="Let me check.",
    )
    monkeypatch.setattr(core, "run_query", stub)
    a = core.answer(MESSAGES, TOOLS, "sonnet", "medium", tool_mode="native")
    assert stub.seen == [TOOLS]
    assert a.content is None and a.parsed and a.finish_reason == "tool_calls"
    assert [(c["name"], c["arguments"]) for c in a.tool_calls] == [
        ("KB_search_bm25", {"query": "ATM fee"})
    ]
    stub2 = _Native([], text="Could you tell me your account number?")
    monkeypatch.setattr(core, "run_query", stub2)
    b = core.answer(MESSAGES, TOOLS, "sonnet", "medium", tool_mode="native")
    assert b.content == "Could you tell me your account number?" and not b.tool_calls
    monkeypatch.setattr(core, "run_query", _Native([], text='{"content": "hi", "tool_calls": []}'))
    c = core.answer(MESSAGES, TOOLS, "sonnet", "medium")  # json: the contract, no native tools
    assert c.content == "hi"


def test_the_native_session_offers_only_the_stubs_and_defers_every_call() -> None:
    opts = core._native_options(TOOLS)
    assert opts["allowed_tools"] == [
        "mcp__tau2__get_user_information_by_email",
        "mcp__tau2__KB_search_bm25",
    ]
    assert list(opts["mcp_servers"]) == [core.NATIVE_SERVER]
    hook = opts["hooks"]["PreToolUse"][0].hooks[0]
    out = asyncio.run(hook({"tool_name": "x"}, "toolu_1", None))
    assert out["hookSpecificOutput"]["permissionDecision"] == "defer"


def test_the_core_reads_the_deferred_calls_off_the_stream_and_redacts_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk as sdk

    monkeypatch.setattr(core, "account_email", lambda: "owner@corp.example")
    seen: dict[str, Any] = {}

    async def fake_query(prompt: Any, options: Any):  # noqa: ANN202
        seen["options"] = options
        async for _ in prompt:
            pass
        yield sdk.AssistantMessage(
            content=[
                sdk.ToolUseBlock(
                    id="toolu_1",
                    name="mcp__tau2__get_user_information_by_email",
                    input={"email": "owner@corp.example"},
                ),
                sdk.ToolUseBlock(
                    id="toolu_2", name="mcp__tau2__KB_search_bm25", input={"query": "fee"}
                ),
            ],
            model="claude-sonnet-5",
        )
        yield sdk.AssistantMessage(content=[sdk.TextBlock(text="after")], model="claude-sonnet-5")
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s1",
            usage={"input_tokens": 5, "output_tokens": 3},
        )

    monkeypatch.setattr(sdk, "query", fake_query)
    res = asyncio.run(
        core._query("sys", ["ask", "[user]\nhi"], "claude-sonnet-5", "medium", native_tools=TOOLS)
    )
    assert [u["name"] for u in res.tool_uses] == ["get_user_information_by_email", "KB_search_bm25"]
    assert res.tool_uses[0]["input"] == {"email": core.REDACTED_EMAIL}
    assert res.text == ""  # what followed the deferred calls is not this turn's
    assert seen["options"].tools == [] and seen["options"].allowed_tools[0].startswith(
        "mcp__tau2__"
    )


def _stream(*replies: tuple[str | None, list[Any]]):  # noqa: ANN202
    """A fake SDK stream: one AssistantMessage per (message_id, blocks), then the result."""
    import claude_agent_sdk as sdk

    async def fake_query(prompt: Any, options: Any):  # noqa: ANN202
        async for _ in prompt:
            pass
        for mid, blocks in replies:
            yield sdk.AssistantMessage(content=blocks, model="claude-sonnet-5", message_id=mid)
        yield sdk.ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="s1",
            usage={"input_tokens": 5, "output_tokens": 3},
        )

    return fake_query


def test_parallel_keeps_every_call_of_one_reply_and_plain_native_keeps_the_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk as sdk

    # the CLI streams each block of one reply as its own message under one message_id (s14 P0a)
    def use(i: int, name: str) -> Any:
        return sdk.ToolUseBlock(id=f"toolu_{i}", name=f"mcp__tau2__{name}", input={"query": str(i)})

    replies = (
        ("msg_a", [sdk.ThinkingBlock(thinking="search both", signature="x")]),
        ("msg_a", [use(1, "KB_search_bm25")]),
        ("msg_a", [use(2, "KB_search_bm25")]),
        ("msg_b", [use(3, "KB_search_bm25")]),  # a later reply: never this turn's
    )

    def run(parallel: bool) -> list[str]:
        monkeypatch.setattr(sdk, "query", _stream(*replies))
        res = asyncio.run(
            core._query(
                "sys",
                ["ask", "[user]\nhi"],
                "claude-sonnet-5",
                "medium",
                native_tools=TOOLS,
                parallel=parallel,
            )
        )
        return [u["id"] for u in res.tool_uses]

    assert run(parallel=False) == ["toolu_1"]
    assert run(parallel=True) == ["toolu_1", "toolu_2"]


def test_a_parallel_version_asks_for_every_call_and_needs_native_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tau2_loop.agent.factory import with_tool_mode

    assert with_tool_mode({}, parallel=True)[core.TOOL_MODE_PARAM] == core.NATIVE_PARALLEL
    assert with_tool_mode({}, service=True, parallel=True)["extra_body"] == {
        core.TOOL_MODE_PARAM: core.NATIVE_PARALLEL
    }
    with pytest.raises(ValueError, match="parallel_calls needs tool_mode: native"):
        versions.AgentConfig(tool_mode="json", parallel_calls=True)
    cfg = versions.AgentConfig(model="sonnet", tool_mode="native", parallel_calls=True)
    assert "parallel_calls: true" in cfg.yaml()
    assert "parallel_calls" not in versions.AgentConfig(tool_mode="native").yaml()
    seen: list[dict[str, Any]] = []

    def stub(system: str, user: Any, model: str, effort: str = "medium", **kw: Any):  # noqa: ANN202
        seen.append(kw)
        r = core.SdkResult("", 10, 2, None, 1, "s")
        r.tool_uses = [{"id": "t", "name": "KB_search_bm25", "input": {"query": "q"}}]
        return r

    monkeypatch.setattr(core, "run_query", stub)
    core.answer(MESSAGES, TOOLS, "sonnet", "medium", tool_mode=core.NATIVE_PARALLEL)
    core.answer(MESSAGES, TOOLS, "sonnet", "medium", tool_mode="native")
    assert seen[0].get("parallel") is True and seen[0]["native_tools"] == TOOLS
    assert "parallel" not in seen[1] and seen[1]["native_tools"] == TOOLS


def test_only_a_native_version_asks_the_provider_for_native_calls() -> None:
    from tau2_loop.agent.factory import with_tool_mode

    args = with_tool_mode(
        {"reasoning_effort": "medium", "allowed_openai_params": ["reasoning_effort"]}
    )
    assert args[core.TOOL_MODE_PARAM] == "native"
    assert args["allowed_openai_params"] == ["reasoning_effort", core.TOOL_MODE_PARAM]
    svc = with_tool_mode({}, service=True)
    assert svc["extra_body"] == {core.TOOL_MODE_PARAM: "native"} and core.TOOL_MODE_PARAM not in svc


def test_the_provider_passes_the_tool_mode_to_the_core(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    import tau2_loop.llm.sdk_provider as provider

    stub = _Native([{"id": "t", "name": "KB_search_bm25", "input": {"query": "q"}}])
    monkeypatch.setattr(core, "run_query", stub)
    monkeypatch.setattr(provider, "require_live", lambda: None)
    provider.register()
    r = litellm.completion(
        model="claude-sdk/claude-sonnet-5",
        messages=MESSAGES,
        tools=TOOLS,
        **{core.TOOL_MODE_PARAM: "native", "allowed_openai_params": [core.TOOL_MODE_PARAM]},
    )
    assert stub.seen == [TOOLS]
    assert r.choices[0].message.tool_calls[0].function.name == "KB_search_bm25"


# ── run health ───────────────────────────────────────────────────────────────────────────────
def _sim(messages: list[dict[str, Any]], reward: float = 0.0) -> dict[str, Any]:
    return {
        "task_id": "task_1",
        "trial": 0,
        "messages": messages,
        "reward_info": {"reward": reward},
    }


def test_health_counts_slips_bare_calls_the_leak_and_documents_read() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": '{"name": "KB_search", "arguments": {"query": "x"}}'},
        {"role": "assistant", "content": health.HOLD_PREFIX + " and haven't completed it yet."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "name": "shell", "arguments": {"command": "cat INDEX.md"}},
                {"id": "b", "name": "shell", "arguments": {"command": "cat doc_fees_001.md"}},
            ],
        },
        {"role": "tool", "id": "a", "content": "# Knowledge Base Index"},
        {"role": "tool", "id": "b", "content": "ATM fees are $2."},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "c", "name": "KB_search_dense", "arguments": {"query": "fee"}},
            ],
        },
        {"role": "tool", "id": "c", "content": "1. Fees\n   ID: doc_cards_002\n   Content: …"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "d",
                    "name": "get_user_information_by_email",
                    "arguments": {"email": core.REDACTED_EMAIL},
                },
                {"id": "e", "name": "get_user_dispute_history_7291", "arguments": {}},
            ],
        },
        {"role": "tool", "id": "d", "content": "Error: no user"},
        {"role": "tool", "id": "e", "content": "[]"},
        {"role": "assistant", "content": "Your fee is $2."},
    ]
    row = health.conversation_health(
        _sim(msgs), frozenset({"doc_fees_001", "doc_cards_002", "doc_other_003"})
    )
    assert (row["replies"], row["slipped"]) == (3, 2)
    assert (row["shell_calls"], row["index_reads"], row["dense_calls"]) == (2, 1, 1)
    assert row["lookups_before_first_action"] == 3  # the email lookup is not an action
    assert (row["bare_discoverable_calls"], row["account_email_calls"]) == (1, 1)
    assert (row["required_docs"], row["required_docs_read"]) == (3, 2)
    assert row["calls_per_turn"] == round(5 / 3, 3) and row["tool_errors"] == 1
    s = health.summarise_health([row])
    assert s["slipped_rate"] == round(2 / 3, 4) and s["shell_conversations"] == 1


def test_health_reproduces_what_s12_counted_by_hand_on_v1_and_v2() -> None:
    def both(*runs: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for r in runs:
            h = health.run_health(r)
            assert h is not None
            rows += h["conversations"]
        return health.summarise_health(rows)

    v1 = both(
        "20261002T235638Z_banking_knowledge_v1_train", "20261003T003126Z_banking_knowledge_v1_test"
    )
    v2 = both(
        "20261003T005018Z_banking_knowledge_v2_train", "20261003T051036Z_banking_knowledge_v2_test"
    )
    assert (v1["bare_discoverable_calls"], v1["bare_discoverable_conversations"]) == (30, 10)
    assert (v2["bare_discoverable_calls"], v2["bare_discoverable_conversations"]) == (86, 20)
    assert (v1["replies"], v2["replies"]) == (1017, 1690)
    assert v1["account_email_conversations"] == 30 and v2["account_email_conversations"] == 0
    assert v1["shell_calls"] == 0 and v1["dense_calls"] == 0


# ── promoting a tool change by the person's call ─────────────────────────────────────────────
def test_a_tool_change_is_a_promotion_kind_and_never_a_held_challenger_to_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tau2_loop.loop import optimiser
    from tau2_loop.tracking import registry

    assert "tool change" in registry.PROMOTION_KINDS
    ledger = [
        {"cycle": 1, "champion": "v1", "challenger": "v2", "outcome": {"verdict": "hold"}},
        {
            "cycle": 2,
            "champion": "v1",
            "challenger": "v3",
            "kind": "tool change",
            "outcome": {"verdict": "hold"},
        },
    ]
    monkeypatch.setattr(optimiser, "read_ledger", lambda d: ledger)
    held = optimiser.held_challengers("banking_knowledge", "v3")
    assert [h["cycle"] for h in held] == [1]  # v2's edits, never v3 as its own held challenger
