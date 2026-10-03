"""One judge call: a sealed session through `llm.core.run_query`, a verdict parsed and checked.

The verdict contract is s11 Fig 7, schema-enforced by the SDK's `output_format` (the J2 probe
showed it holds under the sealed core: no tools, `max_turns=4`). A block stands only when its
confidence clears the version's threshold and its rule is a verbatim policy span or `transcript`;
otherwise it is kept as a note and the reply goes through. The call fails open: an error or an
unparsable verdict allows the reply, and the record says why.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from tau2_loop.tooljudge import prompt
from tau2_loop.tooljudge.gold import norm

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_plan": {"type": "boolean"},
        "verdict": {"type": "string", "enum": ["allow", "block"]},
        "confidence": {"type": "number"},
        "check": {"type": ["integer", "null"]},
        "rule": {"type": ["string", "null"]},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"msg": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["msg", "quote"],
                "additionalProperties": False,
            },
        },
        "fix": {"type": ["string", "null"]},
        "why": {"type": "string"},
    },
    "required": ["is_plan", "verdict", "confidence", "check", "rule", "evidence", "fix", "why"],
    "additionalProperties": False,
}

QueryFn = Callable[..., Any]


def decide(raw: dict[str, Any], threshold: float, policy: str) -> tuple[str, str | None]:
    """The verdict that stands, and why a raw block was turned into a note (None when it stands)."""
    if raw.get("verdict") != "block":
        return "allow", None
    if float(raw.get("confidence") or 0) < threshold:
        return "allow", f"confidence {raw.get('confidence')} under the threshold {threshold}"
    rule = str(raw.get("rule") or "").strip()
    if rule.lower() != "transcript" and (not rule or norm(rule) not in norm(policy)):
        return "allow", "the rule is not verbatim in the policy"
    return "block", None


def judge_reply(
    judge: prompt.JudgeVersion,
    system: str,
    policy: str,
    msgs: list[dict[str, Any]],
    at: int,
    reply: str,
    query: QueryFn | None = None,
) -> dict[str, Any]:
    """Judge the reply the agent proposes at message `at`. Never raises: it fails open."""
    if query is None:
        from tau2_loop.llm import core

        query = core.run_query
    started = time.time()
    rec: dict[str, Any] = {"verdict": "allow", "raw": None, "note": None, "error": None}
    try:
        res = query(
            system,
            prompt.blocks(msgs, at, reply),
            judge.model,
            judge.effort,
            output_format={"type": "json_schema", "schema": VERDICT_SCHEMA}
            if judge.structured
            else None,
        )
        rec["tokens"] = {
            "input": res.input_tokens,
            "output": res.output_tokens,
            "cache_read": res.cache_read,
        }
        raw = res.structured
        if raw is None and res.text:
            try:
                raw = json.loads(res.text)
            except ValueError:
                raw = None
        if not isinstance(raw, dict) or raw.get("verdict") not in ("allow", "block"):
            rec["error"] = f"no verdict ({res.error or 'unparsable'}); failed open"
        else:
            rec["raw"] = raw
            rec["verdict"], rec["note"] = decide(raw, judge.threshold, policy)
    except Exception as e:  # noqa: BLE001 - the judge must never stop a conversation
        rec["error"] = f"{type(e).__name__}: {e}"[:300] + "; failed open"
    rec["ms"] = int((time.time() - started) * 1000)
    return rec


def probe(model: str = "claude-sonnet-5", query: QueryFn | None = None) -> dict[str, Any]:
    """One call that settles whether the SDK's `output_format` holds under the sealed core.

    s11 left it unverified with `tools=[]` and `max_turns=4`; this asks a toy plan for a verdict
    in the Fig 7 schema and records whether it came back schema-shaped in `structured_output`."""
    from importlib.metadata import version

    if query is None:
        from tau2_loop.llm import core as llm_core

        query = llm_core.run_query
    res = query(
        "You review a plan an airline agent proposes. Answer in the schema.",
        "Policy: a round trip's destination is never its origin.\n"
        "[0] customer: Book me a round trip from JFK to SFO.\n"
        "[1] agent (proposed): I'll book a round trip JFK to JFK. Shall I proceed?",
        model,
        "low",
        output_format={"type": "json_schema", "schema": VERDICT_SCHEMA},
    )
    s = res.structured
    return {
        "sdk": version("claude-agent-sdk"),
        "model": model,
        "holds": isinstance(s, dict) and set(VERDICT_SCHEMA["required"]) <= set(s),
        "structured_output": s,
        "error": res.error,
        "tokens": {"input": res.input_tokens, "output": res.output_tokens},
        "ms": res.duration_ms,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
