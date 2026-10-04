"""`claude-sdk/<model>`: a litellm provider that answers a chat request through the Agent SDK.

tau2 sends every model call — the agent's, the user simulator's, the NL judge's
— through `tau2.utils.llm_utils.generate()`, which calls `litellm.completion()`.
litellm routes a model string with an unknown prefix to a registered custom
provider, so registering this class under the prefix `claude-sdk` makes
`--agent-llm claude-sdk/claude-haiku-4-5` a Haiku call on the subscription with
nothing in tau2 touched.

This file is only the litellm side of it: the request goes to `core.answer()`,
the sealed core the agent service (`service.py`) calls too, and the answer
comes back as a litellm `ModelResponse` whose `tool_calls` tau2's orchestrator
executes unchanged. `reasoning_effort` arrives in `optional_params` when the
caller passes `allowed_openai_params=["reasoning_effort"]` (litellm drops it for
a custom provider otherwise), and a version's native tool mode the same way, as
`tau2_loop_tool_mode`; temperature and other sampling arguments have no
counterpart in the SDK and are ignored — recorded in the run's `run.json` as
`sampling: cli-default`.
"""

from __future__ import annotations

import json
import threading
import uuid
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

from tau2_loop.llm import SDK_PREFIX, require_live
from tau2_loop.llm.core import (  # re-exported: tests and callers import them from here
    MAX_WAIT_S,
    TOOL_MODE_PARAM,
    Answer,
    SdkResult,
    answer,
    cache_usage,
    run_query,
    seconds_until_reset,
)

PROVIDER = SDK_PREFIX.rstrip("/")

_registered = threading.Lock()
_is_registered = False

__all__ = [
    "MAX_WAIT_S",
    "ClaudeSdkProvider",
    "SdkResult",
    "register",
    "run_query",
    "seconds_until_reset",
]


class ClaudeSdkProvider(CustomLLM):  # type: ignore[misc]
    """litellm handler: `completion()` is what `tau2.utils.llm_utils.generate()` reaches."""

    def completion(self, *args: Any, **kwargs: Any) -> ModelResponse:
        model: str = kwargs["model"]
        optional = kwargs.get("optional_params") or {}
        require_live()  # scrubs a key tau2's dotenv search injected, then checks billing
        a = answer(
            kwargs["messages"],
            optional.get("tools"),
            model,
            optional.get("reasoning_effort"),
            optional.get(TOOL_MODE_PARAM),
        )
        return to_model_response(a, model)


def to_model_response(a: Answer, model: str) -> ModelResponse:
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=c["id"],
            type="function",
            function=Function(name=c["name"], arguments=json.dumps(c["arguments"])),
        )
        for c in a.tool_calls
    ]
    message = Message(content=a.content, role="assistant", tool_calls=tool_calls or None)
    choice = Choices(finish_reason=a.finish_reason, index=0, message=message)
    res = a.result or SdkResult("", 0, 0, None, 0, None)
    response = ModelResponse(
        id=f"sdk-{res.session_id or uuid.uuid4().hex[:8]}",
        choices=[choice],
        model=model,
        usage=Usage(
            prompt_tokens=res.input_tokens,
            completion_tokens=res.output_tokens,
            total_tokens=res.input_tokens + res.output_tokens,
            **cache_usage(res),
        ),
    )
    response._hidden_params = {  # noqa: SLF001 - litellm's own extension point
        "sdk_cost_usd": res.cost_usd,
        "sdk_duration_ms": res.duration_ms,
        "sdk_session_id": res.session_id,
        "reply_parsed": a.parsed,
    }
    return response


def register() -> None:
    """Install the provider into litellm once per process; safe to call repeatedly."""
    global _is_registered
    with _registered:
        if _is_registered:
            return
        litellm.suppress_debug_info = (
            True  # the cost lookup fails on our model id; tau2 records 0.0
        )
        handler = ClaudeSdkProvider()
        existing = [m for m in (litellm.custom_provider_map or []) if m.get("provider") != PROVIDER]
        litellm.custom_provider_map = [*existing, {"provider": PROVIDER, "custom_handler": handler}]
        _is_registered = True
