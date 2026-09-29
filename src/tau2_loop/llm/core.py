"""The sealed core: a chat request in, one Agent SDK answer out.

Everything that turns a role-tagged history plus tool schemas into a model
reply lives here, and nothing else does: the in-process litellm shim
(`sdk_provider.py`) and the HTTP service (`service.py`) both call `answer()`,
so the two routes cannot drift. The module imports the Agent SDK,
`prompting.py` and the standard library, and nothing else of ours
(`tests/test_core.py` pins that), which is what lets `Dockerfile.agent` ship
it with its two siblings and no repo.

The session it opens can only answer: no built-in tools, no MCP servers, no
settings files, a working directory outside any checkout, and an environment
reduced to an allow-list. The SDK merges `options.env` over the parent's
environment, so leaving a variable out is not enough; every name outside the
list is passed blank.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from tau2_loop.llm.prompting import build_prompt, parse_reply

MODELS: dict[str, str] = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5-5",
}

# Every Agent SDK session in this app runs at this effort unless a version's
# `agent.yaml` pins another one for the task agent.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT: Effort = "medium"
EFFORTS: tuple[str, ...] = ("low", "medium", "high", "xhigh", "max")

RETRIES = 3
RETRY_WAIT_S = (2.0, 6.0, 15.0)
MAX_WAIT_S = 6 * 3600

# What the CLI child may inherit. PATH, HOME and the locale run it; the proxy
# and certificate names let it reach the API from behind a proxy;
# CLAUDE_CONFIG_DIR and CLAUDE_CODE_OAUTH_TOKEN are its login, the second the
# only one a container has (`claude setup-token`).
ENV_ALLOW: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "LANG",
        "TERM",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "NODE_EXTRA_CA_CERTS",
    }
)
ENV_ALLOW_PREFIXES: tuple[str, ...] = ("LC_",)
# Set by us, whatever the parent had: the entrypoint the SDK would set, and the
# switch that stops the CLI naming every session with a separate Haiku call
# (≈1.9k tokens a call here, where every model call is its own session).
ENV_FIXED: dict[str, str] = {
    "CLAUDE_CODE_ENTRYPOINT": "sdk-py",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDECODE": "",
}


class BillingError(RuntimeError):
    """The environment would bill the wrong way, or cannot bill at all."""


class CoreError(RuntimeError):
    """The SDK gave no reply after its retries."""


def resolve_model(name: str) -> str:
    """`haiku` → `claude-haiku-4-5`; a full model id passes through; a route prefix is stripped."""
    name = name.strip()
    if "/" in name:
        name = name.rsplit("/", 1)[1]
    return MODELS.get(name.lower(), name)


def billing() -> str:
    return os.environ.get("BILLING", "subscription").strip().lower()


def demo_mode() -> bool:
    return os.environ.get("DEMO_MODE", "").strip() in {"1", "true", "yes"}


def check_billing() -> None:
    """Refuse a model call in a state where the bill would be a surprise."""
    if demo_mode():
        raise BillingError(
            "DEMO_MODE=1: this deployment serves committed runs and never calls a model"
        )
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if billing() == "subscription" and key:
        raise BillingError(
            "ANTHROPIC_API_KEY is set while BILLING=subscription. Unset one: with a key present the "
            "CLI bills per token even though the subscription would cover the call."
        )
    if billing() == "api" and not key:
        raise BillingError("BILLING=api but ANTHROPIC_API_KEY is not set")


def _allowed(name: str) -> bool:
    return name in ENV_ALLOW or name.startswith(ENV_ALLOW_PREFIXES)


def sealed_env(parent: dict[str, str] | None = None) -> dict[str, str]:
    """`options.env` for a sealed session: the allow-list kept, every other inherited name blank.

    The key is passed only when `BILLING=api` asks for per-token billing; under
    the subscription it is blank, so the CLI cannot bill per token.
    """
    parent = dict(os.environ if parent is None else parent)
    env = {k: (v if _allowed(k) else "") for k, v in parent.items()}
    env.update(ENV_FIXED)
    env["ANTHROPIC_API_KEY"] = parent.get("ANTHROPIC_API_KEY", "") if billing() == "api" else ""
    return env


def session_dir() -> Path:
    """The sessions' working directory: one fixed folder under the temp dir, outside any checkout."""
    p = Path(tempfile.gettempdir()) / "tau2loop-sdk"
    p.mkdir(parents=True, exist_ok=True)
    return p


REDACTED_EMAIL = "account-email@redacted.invalid"


def account_email() -> str:
    """The subscription account's email, from the CLI's own config; '' when unknown.

    The CLI tells every session whose account it runs under. That sentence
    reaches the task agent as context, and Haiku on retail v0 used the address
    as the customer's in 20/20 conversations (`find_user_id_by_email`), which
    both wasted the first turn and put a real address in every trace.
    """
    try:
        cfg = json.loads((Path.home() / ".claude.json").read_text())
        return str(cfg.get("oauthAccount", {}).get("emailAddress", "")).strip()
    except (OSError, ValueError):
        return ""


def redact(text: str) -> str:
    email = account_email()
    return text.replace(email, REDACTED_EMAIL) if email else text


@dataclass
class SdkResult:
    text: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    duration_ms: int
    session_id: str | None
    error: str | None = None


async def _query(system_prompt: str, user_prompt: str, model: str, effort: str) -> SdkResult:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools: the model can only answer
        allowed_tools=[],
        strict_mcp_config=True,  # no inherited connector tools: 27k tokens a call otherwise (s02)
        permission_mode="bypassPermissions",
        max_turns=4,  # one reply; headroom because the CLI has ended a tool-less reply as 'max turns (1)'
        cwd=str(session_dir()),
        env=sealed_env(),
        setting_sources=[],
        effort=effort,
    )
    texts: list[str] = []
    res = SdkResult("", 0, 0, None, 0, None)
    started = time.time()
    async for msg in query(prompt=user_prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for b in msg.content:
                if isinstance(b, TextBlock):
                    texts.append(b.text)
        elif isinstance(msg, ResultMessage):
            u = msg.usage or {}
            res.input_tokens = (
                int(u.get("input_tokens", 0))
                + int(u.get("cache_read_input_tokens", 0))
                + int(u.get("cache_creation_input_tokens", 0))
            )
            res.output_tokens = int(u.get("output_tokens", 0))
            res.cost_usd = msg.total_cost_usd
            res.session_id = msg.session_id
            if msg.is_error:
                res.error = f"{msg.subtype}: {(msg.errors or [''])[0]}"[:500]
            if not texts and msg.result:
                texts.append(str(msg.result))
    # The CLI names the account in every session; the address must not reach a conversation.
    res.text = redact("\n".join(t for t in texts if t).strip())
    res.duration_ms = int((time.time() - started) * 1000)
    return res


_RESET_RE = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)", re.I)


def seconds_until_reset(error: str, now: datetime | None = None) -> int | None:
    """The wait a subscription 'session limit · resets 4:40pm' message asks for, in local time.

    The CLI reports the window's reset as a local clock time; the next such time
    (today or tomorrow) is the earliest a call can succeed. None when the error
    is not a session-limit one."""
    if "session limit" not in error.lower():
        return None
    m = _RESET_RE.search(error)
    now = now or datetime.now()
    if not m:
        return 15 * 60
    hour, minute, ampm = int(m.group(1)), int(m.group(2) or 0), m.group(3).lower()
    hour = hour % 12 + (12 if ampm == "pm" else 0)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return min(int((target - now).total_seconds()) + 60, MAX_WAIT_S)


def run_query(system_prompt: str, user_prompt: str, model: str, effort: str = EFFORT) -> SdkResult:
    """One SDK query on a private event loop, so it works from worker threads.

    A subscription window that has run out is waited for, not failed: the
    caller pauses until the reset the CLI names, then carries on."""
    check_billing()
    last: SdkResult | None = None
    attempt = 0
    while attempt < RETRIES:
        loop = asyncio.new_event_loop()
        try:
            last = loop.run_until_complete(_query(system_prompt, user_prompt, model, effort))
        except Exception as e:  # noqa: BLE001 - transport errors are retried like HTTP ones
            last = SdkResult("", 0, 0, None, 0, None, error=f"{type(e).__name__}: {e}"[:500])
        finally:
            loop.close()
        if last.text:
            return last  # a reply came back; an error next to it (e.g. a max-turns note) is recorded, not retried
        wait = seconds_until_reset(last.error or "")
        if wait is not None:
            print(
                f"[claude-sdk] session limit reached; waiting {wait // 60} min for the window to reset",
                flush=True,
            )
            time.sleep(wait)
            continue  # the wait is not an attempt
        attempt += 1
        if attempt < RETRIES:
            time.sleep(RETRY_WAIT_S[attempt - 1])
    assert last is not None
    if last.error is None:
        last.error = "empty reply"
    return last


@dataclass
class Answer:
    """One reply in the shape both routes return: a message or tool calls, never both."""

    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # {id, name, arguments}
    parsed: bool = True
    result: SdkResult | None = None

    @property
    def finish_reason(self) -> str:
        return "tool_calls" if self.tool_calls else "stop"


def answer(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    effort: str | None = None,
) -> Answer:
    """The chat request as the SDK sees it, and its reply parsed back; raises CoreError on no reply."""
    effort = effort if effort in EFFORTS else EFFORT
    system_prompt, user_prompt = build_prompt(messages, tools)
    res = run_query(system_prompt, user_prompt, resolve_model(model), effort)
    if res.error and not res.text:
        raise CoreError(f"claude-sdk: {res.error}")
    reply = parse_reply(res.text, tools_present=bool(tools))
    calls = [
        {"id": f"call_{uuid.uuid4().hex[:12]}", "name": c["name"], "arguments": c["arguments"]}
        for c in reply.tool_calls
    ]
    return Answer(content=reply.content, tool_calls=calls, parsed=reply.parsed, result=res)
