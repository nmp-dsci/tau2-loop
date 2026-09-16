"""Turning a chat-completion request into one Agent SDK prompt, and the reply back.

The Agent SDK takes a system prompt and one user prompt per query; a chat
request arrives as a role-tagged history plus, sometimes, a tool list. This
module serialises the history into a transcript block and, when tools are
present, adds the JSON reply contract the model must follow: exactly one
object with either `content` (a message to the other party) or `tool_calls`
(a list of `{name, arguments}`), never both — the same rule tau2's
orchestrator enforces on the agent.

Pure functions, no I/O, so the contract is unit-tested offline.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CONTRACT = """
# Tools

You can call the tools below. Reply with EXACTLY ONE JSON object and nothing else — no prose
before or after it, no code fences:

{"content": "<your message to the other party>", "tool_calls": []}
  — to send a message, or

{"content": null, "tool_calls": [{"name": "<tool name>", "arguments": {<arguments as JSON>}}]}
  — to call one or more tools. You will receive each tool's result as the next message.

A reply has either a message or tool calls, never both, and never neither.
Argument values must match the tool's JSON schema (types, required fields, enums).

Available tools (name, description, parameters as JSON schema):
"""

_JSON_START = re.compile(r"\{")


@dataclass
class Reply:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""
    parsed: bool = True


def split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """System text (joined) and the non-system turns, in order."""
    system = "\n\n".join(
        str(m.get("content") or "") for m in messages if m.get("role") == "system"
    ).strip()
    rest = [m for m in messages if m.get("role") != "system"]
    return system, rest


def render_tools(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return ""
    lines = [TOOL_CONTRACT.strip(), ""]
    for t in tools:
        fn = t.get("function", t)
        lines.append(
            json.dumps(
                {
                    "name": fn.get("name"),
                    "description": fn.get("description"),
                    "parameters": fn.get("parameters"),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)


def render_transcript(turns: list[dict[str, Any]]) -> str:
    """The conversation so far, one block per turn; tool calls and results inline."""
    out: list[str] = []
    for m in turns:
        role = str(m.get("role"))
        content = m.get("content")
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if calls:
                rendered = [
                    {
                        "id": c.get("id"),
                        "name": (c.get("function") or {}).get("name") or c.get("name"),
                        "arguments": _args((c.get("function") or {}).get("arguments")),
                    }
                    for c in calls
                ]
                out.append("[assistant → tool calls]\n" + json.dumps(rendered, ensure_ascii=False))
            if content:
                out.append(f"[assistant]\n{content}")
        elif role == "tool":
            out.append(f"[tool result · call {m.get('tool_call_id')}]\n{content}")
        elif role == "user":
            out.append(f"[user]\n{content}")
        else:
            out.append(f"[{role}]\n{content}")
    return "\n\n".join(out)


def _args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw


def build_prompt(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> tuple[str, str]:
    """(system_prompt, user_prompt) for one SDK query."""
    system, turns = split_messages(messages)
    tool_block = render_tools(tools)
    if tool_block:
        system = f"{system}\n\n{tool_block}".strip()
    transcript = render_transcript(turns)
    if tools:
        ask = (
            "The conversation so far is below. Produce the next assistant reply as the single JSON "
            "object described in the Tools section."
        )
    else:
        ask = (
            "The conversation so far is below. Produce the next assistant reply as plain text — "
            "the reply only, no preamble and no role label."
        )
    user_prompt = f"{ask}\n\n<conversation>\n{transcript}\n</conversation>"
    return system, user_prompt


def parse_reply(text: str, tools_present: bool) -> Reply:
    """The model's text back into a Reply; a malformed JSON reply becomes a plain message."""
    raw = text.strip()
    if not tools_present:
        # A caller that asked for JSON (tau2's NL judge does json.loads on the reply) gets the
        # object, not a fenced block around it; a plain sentence passes through untouched.
        return Reply(content=_strip_fences(raw), raw=raw)
    candidate = _strip_fences(raw)
    obj = _first_json_object(candidate)
    if obj is None:
        return Reply(content=raw, raw=raw, parsed=False)
    calls_raw = obj.get("tool_calls") or []
    calls: list[dict[str, Any]] = []
    for c in calls_raw:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        args = c.get("arguments")
        if isinstance(args, str):
            args = _args(args)
        calls.append({"name": str(c["name"]), "arguments": args if isinstance(args, dict) else {}})
    content = obj.get("content")
    content = str(content) if content not in (None, "") else None
    if calls:
        # The contract forbids both; tool calls win and the text is dropped, as tau2's
        # own LLM agent would be told off for a mixed message.
        return Reply(content=None, tool_calls=calls, raw=raw)
    if content is None:
        return Reply(content=raw, raw=raw, parsed=False)
    return Reply(content=content, raw=raw)


def _strip_fences(s: str) -> str:
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


def _first_json_object(s: str) -> dict[str, Any] | None:
    """The first balanced `{…}` in `s` that parses, scanning from each `{`."""
    for m in _JSON_START.finditer(s):
        start = m.start()
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return obj if isinstance(obj, dict) else None
        # unbalanced from this start: try the next `{`
    return None
