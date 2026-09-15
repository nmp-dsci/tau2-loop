"""`claude-sdk/<model>`: a litellm provider that answers a chat request through the Agent SDK.

tau2 sends every model call — the agent's, the user simulator's, the NL judge's
— through `tau2.utils.llm_utils.generate()`, which calls `litellm.completion()`.
litellm routes a model string with an unknown prefix to a registered custom
provider, so registering this class under the prefix `claude-sdk` makes
`--agent-llm claude-sdk/claude-haiku-4-5` a Haiku call on the subscription with
nothing in tau2 touched.

Each request is one `query()` on a fresh CLI process: the history is
serialised into the prompt (`prompting.build_prompt`), the reply parsed back
(`prompting.parse_reply`). Tool calls travel as the JSON contract in the
prompt, so the reply carries `tool_calls` the same way an OpenAI response
would and tau2's orchestrator executes them unchanged. Temperature and other
sampling arguments have no counterpart in the SDK and are ignored — recorded
in the run's `run.json` as `sampling: cli-default`.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import CustomLLM
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Function,
    Message,
    ModelResponse,
    Usage,
)

from tau2_loop.config import WORKSPACE_DIR
from tau2_loop.llm import EFFORT, SDK_PREFIX, require_live, resolve_model, subscription_env
from tau2_loop.llm.prompting import Reply, build_prompt, parse_reply

PROVIDER = SDK_PREFIX.rstrip("/")
RETRIES = 3
RETRY_WAIT_S = (2.0, 6.0, 15.0)

_registered = threading.Lock()
_is_registered = False


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

    cwd = WORKSPACE_DIR / "sdk"
    cwd.mkdir(parents=True, exist_ok=True)
    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        tools=[],  # no built-in tools: the model can only answer
        allowed_tools=[],
        permission_mode="bypassPermissions",
        max_turns=1,
        cwd=str(cwd),
        env=subscription_env(),
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
    res.text = "\n".join(t for t in texts if t).strip()
    res.duration_ms = int((time.time() - started) * 1000)
    return res


def run_query(system_prompt: str, user_prompt: str, model: str, effort: str = EFFORT) -> SdkResult:
    """One SDK query on a private event loop, so it works from tau2's worker threads."""
    require_live()
    last: SdkResult | None = None
    for attempt in range(RETRIES):
        loop = asyncio.new_event_loop()
        try:
            last = loop.run_until_complete(_query(system_prompt, user_prompt, model, effort))
        except Exception as e:  # noqa: BLE001 - transport errors are retried like HTTP ones
            last = SdkResult("", 0, 0, None, 0, None, error=f"{type(e).__name__}: {e}"[:500])
        finally:
            loop.close()
        if last.error is None and last.text:
            return last
        if attempt < RETRIES - 1:
            time.sleep(RETRY_WAIT_S[attempt])
    assert last is not None
    if last.error is None:
        last.error = "empty reply"
    return last


class ClaudeSdkProvider(CustomLLM):  # type: ignore[misc]
    """litellm handler: `completion()` is what `tau2.utils.llm_utils.generate()` reaches."""

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        model: str = kwargs["model"]
        messages: list[dict[str, Any]] = kwargs["messages"]
        optional = kwargs.get("optional_params") or {}
        tools = optional.get("tools")
        full_model = resolve_model(model)
        system_prompt, user_prompt = build_prompt(messages, tools)
        res = run_query(system_prompt, user_prompt, full_model)
        if res.error and not res.text:
            raise RuntimeError(f"claude-sdk: {res.error}")
        reply = parse_reply(res.text, tools_present=bool(tools))
        return _to_model_response(reply, res, model)


def _to_model_response(reply: Reply, res: SdkResult, model: str) -> ModelResponse:
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            type="function",
            function=Function(name=c["name"], arguments=json.dumps(c["arguments"])),
        )
        for c in reply.tool_calls
    ]
    message = Message(content=reply.content, role="assistant", tool_calls=tool_calls or None)
    choice = Choices(finish_reason="tool_calls" if tool_calls else "stop", index=0, message=message)
    response = ModelResponse(
        id=f"sdk-{res.session_id or uuid.uuid4().hex[:8]}",
        choices=[choice],
        model=model,
        usage=Usage(
            prompt_tokens=res.input_tokens,
            completion_tokens=res.output_tokens,
            total_tokens=res.input_tokens + res.output_tokens,
        ),
    )
    response._hidden_params = {  # noqa: SLF001 - litellm's own extension point
        "sdk_cost_usd": res.cost_usd,
        "sdk_duration_ms": res.duration_ms,
        "sdk_session_id": res.session_id,
        "reply_parsed": reply.parsed,
    }
    return response


def register() -> None:
    """Install the provider into litellm once per process; safe to call repeatedly."""
    global _is_registered
    with _registered:
        if _is_registered:
            return
        handler = ClaudeSdkProvider()
        existing = [m for m in (litellm.custom_provider_map or []) if m.get("provider") != PROVIDER]
        litellm.custom_provider_map = [*existing, {"provider": PROVIDER, "custom_handler": handler}]
        _is_registered = True
