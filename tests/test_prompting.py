"""The JSON reply contract, offline: what the provider sends and how it reads the answer back."""

from __future__ import annotations

from tau2_loop.llm.prompting import build_prompt, parse_reply, render_transcript

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user",
            "description": "Look up a user",
            "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}}},
        },
    }
]


def test_build_prompt_puts_system_and_tools_in_system_prompt() -> None:
    system, user = build_prompt(
        [{"role": "system", "content": "POLICY"}, {"role": "user", "content": "hi"}], TOOLS
    )
    assert system.startswith("POLICY")
    assert "get_user" in system and '"tool_calls"' in system
    assert "[user]\nhi" in user and "single JSON object" in user


def test_build_prompt_without_tools_asks_for_plain_text() -> None:
    system, user = build_prompt(
        [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}], None
    )
    assert "Tools" not in system
    assert "plain text" in user


def test_transcript_renders_tool_calls_and_results() -> None:
    turns = [
        {"role": "user", "content": "book it"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "get_user", "arguments": '{"user_id": "u1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"name": "A"}'},
        {"role": "assistant", "content": "Found you, A."},
    ]
    t = render_transcript(turns)
    assert "[assistant → tool calls]" in t and '"user_id": "u1"' in t
    assert "[tool result · call c1]" in t
    assert t.endswith("[assistant]\nFound you, A.")


def test_parse_message_reply() -> None:
    r = parse_reply('{"content": "Hello there", "tool_calls": []}', tools_present=True)
    assert r.content == "Hello there" and r.tool_calls == [] and r.parsed


def test_parse_tool_call_reply_with_fences_and_prose() -> None:
    text = 'Sure.\n```json\n{"content": null, "tool_calls": [{"name": "get_user", "arguments": {"user_id": "u1"}}]}\n```'
    r = parse_reply(text, tools_present=True)
    assert r.content is None
    assert r.tool_calls == [{"name": "get_user", "arguments": {"user_id": "u1"}}]


def test_parse_arguments_given_as_string() -> None:
    r = parse_reply(
        '{"tool_calls": [{"name": "x", "arguments": "{\\"a\\": 1}"}]}', tools_present=True
    )
    assert r.tool_calls == [{"name": "x", "arguments": {"a": 1}}]


def test_parse_mixed_reply_drops_text_keeps_calls() -> None:
    r = parse_reply('{"content": "one sec", "tool_calls": [{"name": "x", "arguments": {}}]}', True)
    assert r.content is None and len(r.tool_calls) == 1


def test_parse_non_json_becomes_message_and_is_flagged() -> None:
    r = parse_reply("I will look that up for you.", tools_present=True)
    assert r.content == "I will look that up for you." and not r.parsed


def test_parse_without_tools_is_verbatim() -> None:
    r = parse_reply('{"content": "x"}', tools_present=False)
    assert r.content == '{"content": "x"}' and r.parsed


def test_parse_braces_inside_strings() -> None:
    r = parse_reply('{"content": "use {curly} braces", "tool_calls": []}', True)
    assert r.content == "use {curly} braces"
