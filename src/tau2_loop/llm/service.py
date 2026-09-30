"""The agent as a service: `POST /v1/chat/completions` over the sealed core.

An OpenAI-shaped endpoint, so the harness reaches it through litellm's own
`openai/` provider with an `api_base` and tau2 stays unmodified. A request
carries `{model, messages, tools?, reasoning_effort?}`; the reply carries
`choices[0].message.{content, tool_calls}` and `usage`, a tool call as
`{id, type: "function", function: {name, arguments}}`. Everything between is
`core.answer()`, the same call the in-process shim makes, so a conversation
through the service and one in-process differ only in the transport.

Every call needs the bearer token (`AGENT_SERVICE_TOKEN`): an open endpoint
would let anyone spend the subscription. Only the three models the app names
are served. The module imports FastAPI and the core, nothing else of ours, so
`Dockerfile.agent` ships it with `core.py` and `prompting.py` and no repo.
"""

from __future__ import annotations

import hmac
import json
import os
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException

from tau2_loop.llm import core

TOKEN_ENV = "AGENT_SERVICE_TOKEN"


def to_openai(a: core.Answer, model: str) -> dict[str, Any]:
    """An Answer as an OpenAI chat completion."""
    res = a.result or core.SdkResult("", 0, 0, None, 0, None)
    message: dict[str, Any] = {"role": "assistant", "content": a.content}
    if a.tool_calls:
        message["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
            }
            for c in a.tool_calls
        ]
    return {
        "id": f"chatcmpl-{res.session_id or uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": a.finish_reason}],
        "usage": {
            "prompt_tokens": res.input_tokens,
            "completion_tokens": res.output_tokens,
            "total_tokens": res.input_tokens + res.output_tokens,
            **core.cache_usage(res),
        },
    }


def create_app(token: str | None = None) -> FastAPI:
    token = token if token is not None else os.environ.get(TOKEN_ENV, "").strip()
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV} is not set: an open endpoint would let anyone spend the subscription"
        )
    expected = f"Bearer {token}".encode()
    app = FastAPI(title="tau2-loop agent", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    def healthz() -> dict[str, Any]:
        """Which models, which billing path, which login; no model call."""
        return {
            "ok": True,
            "billing": core.billing(),
            "demo_mode": core.demo_mode(),
            "login": "oauth-token" if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") else "cli-session",
            "models": sorted(core.MODELS.values()),
            "effort": core.EFFORT,
        }

    # sync on purpose: FastAPI runs it on its thread pool, and core.run_query owns its event loop
    @app.post("/v1/chat/completions")
    def chat(
        body: dict[str, Any], authorization: str | None = Header(default=None)
    ) -> dict[str, Any]:
        if not authorization or not hmac.compare_digest(authorization.encode(), expected):
            raise HTTPException(status_code=401, detail="missing or wrong bearer token")
        model = core.resolve_model(str(body.get("model") or ""))
        if model not in core.MODELS.values():
            raise HTTPException(status_code=400, detail=f"model {model!r} is not served here")
        effort = body.get("reasoning_effort")
        try:
            a = core.answer(
                list(body.get("messages") or []),
                body.get("tools") or None,
                model,
                str(effort) if effort else None,
            )
        except core.BillingError as e:
            raise HTTPException(status_code=403, detail=str(e)) from e
        except core.CoreError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        return to_openai(a, model)

    return app
