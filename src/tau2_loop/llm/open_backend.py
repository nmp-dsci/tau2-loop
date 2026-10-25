"""The answering agent on DeepSeek through Pydantic AI (s36, 25 Oct 2026, `backend: pydantic_ai`).

ConvFinQA-agent's open runtime (its s25–s28) lifted the champion's prompt onto `deepseek-flash`
through Pydantic AI's DeepSeek provider and kept every conversation as a trajectory file for a
student's training rows. This is the same model call in tau2-loop's agent seat, and only the
model call: tau2 still owns the turn loop, holds the history, runs every tool and gives the
customer its turn. Each agent turn is one request:

- the history tau2 holds (the composed prompt, the customer's words, the agent's earlier replies
  and their tool results) is rebuilt as Pydantic AI messages (`to_model_messages`), DeepSeek's
  reasoning put back as a thinking part on each earlier reply, which DeepSeek requires on every
  tool-call turn;
- tau2's tools go as tool definitions only. Pydantic AI never runs one (`model_request_sync`
  makes one request and returns), so a tool call comes back to tau2 unrun, as the subscription
  core's native mode does;
- the reply comes back as tau2's `AssistantMessage` (`to_assistant_message`) with `raw_data` in
  the shape tau2 saves for a litellm reply, `choices[0].message.reasoning_content` holding the
  reasoning word for word. That is the field the leaderboard's submissions use and
  tau-telecom-sft's reader counts, so a run folder carries every turn's reasoning.

Billing (the person's call, s36): this is the one model call in the repo not on the Claude
subscription. It bills DeepSeek's API with `DEEPSEEK_API_KEY`, read at call time from the
environment or `~/.env` and never put in `os.environ`, `llm_args` or a file a run writes. The
customer, the judge, the optimiser and workflow_rag stay on the subscription; `check_billing()`
still runs first, so the demo image refuses. Prices are DeepSeek's card, declared here so a run
is never repriced, and a spend cap stops a run: the reply that crosses it is kept (it is paid
for), and every call after it fails, so the rest of the run's conversations are infrastructure
errors, unscored, never fails.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

BACKEND = "pydantic_ai"
BILLING = "deepseek-api"
KEY = "DEEPSEEK_API_KEY"
PROVIDER = "deepseek"
# what the open backend may run; DeepSeek remaps an unknown effort silently (`xhigh` → high), and
# a run named for one effort that ran at another is not that version's evidence
OPEN_MODELS: tuple[str, ...] = ("deepseek-flash", "deepseek-v4-pro")
OPEN_EFFORTS: tuple[str, ...] = ("low", "high", "max")
# Pydantic AI knows a V4 model only by its `deepseek-v4-` prefix; profiled under its own name,
# `deepseek-flash` would be treated as a model that cannot think (ConvFinQA-agent, s25)
PROFILE_AS = {"deepseek-flash": "deepseek-v4-flash"}
# USD per 1M tokens (cache hit, cache miss, output), peak; off-peak is half. DeepSeek's card,
# api-docs.deepseek.com/quick_start/pricing, read 25 Oct 2026 (ConvFinQA-agent tracking/cost.py)
CARD: dict[str, tuple[float, float, float]] = {
    "deepseek-flash": (0.006, 0.30, 1.20),
    "deepseek-v4-pro": (0.044, 1.32, 3.96),
}
TIMEOUT_S = 120.0
ATTEMPTS = 4  # each request, on a dropped connection, a 429 or a 5xx
EMPTY_RETRIES = 2  # a reply with neither text nor a tool call is asked for again
_RETRY_STATUS = {429, 500, 502, 503, 504}
REASONING_ID = "reasoning_content"


class BackendStoppedError(RuntimeError):
    """The run's DeepSeek calls are over: the cap was reached or the balance ran out."""


def is_open_model(model: str) -> bool:
    return model in OPEN_MODELS


def check(model: str, effort: str) -> None:
    """Refuse a model or effort outside the declared sets rather than pass it through."""
    if model not in OPEN_MODELS:
        raise ValueError(f"backend {BACKEND} runs {OPEN_MODELS}, not {model!r}")
    if effort not in OPEN_EFFORTS:
        raise ValueError(f"backend {BACKEND} thinks at {OPEN_EFFORTS}, not {effort!r}")


def route(model: str) -> str:
    """The agent's model as a run records it (`run.json`, tau2's `llm_agent`): provenance only."""
    return f"{BACKEND}/{PROVIDER}/{model}"


def api_key() -> str:
    """The key, from the process environment or `~/.env`, never written back to either."""
    key = os.environ.get(KEY, "").strip()
    if not key:
        from dotenv import dotenv_values

        key = str(dotenv_values(Path.home() / ".env").get(KEY) or "").strip()
    if not key:
        raise RuntimeError(
            f"{KEY} is not set: the {BACKEND} backend calls DeepSeek's API. Put it in ~/.env."
        )
    return key


# ── cost and the cap ─────────────────────────────────────────────────────


def peak(at: datetime) -> bool:
    """DeepSeek's peak window: 01:00–04:00 and 06:00–10:00 UTC, Monday to Friday."""
    utc = at.astimezone(UTC)
    return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)


def cost_usd(model: str, *, hit: int, miss: int, output: int, at: datetime) -> float:
    """One request on the card in force at `at`."""
    c_hit, c_miss, c_out = CARD[model]
    scale = 1.0 if peak(at) else 0.5
    return scale * (hit * c_hit + miss * c_miss + output * c_out) / 1_000_000


_lock = threading.Lock()
_spent = 0.0
_cap: float | None = None
_stopped: str | None = None


def reset(cap_usd: float | None = None) -> None:
    """A run starts clean: nothing spent, nothing stopped, its own cap (None: no cap)."""
    global _spent, _cap, _stopped
    with _lock:
        _spent, _cap, _stopped = 0.0, cap_usd, None


def spent() -> float:
    return _spent


def _charge(usd: float) -> None:
    global _spent, _stopped
    with _lock:
        _spent += usd
        if _cap is not None and _spent >= _cap and _stopped is None:
            _stopped = f"DeepSeek spend cap ${_cap:.2f} reached (${_spent:.4f} spent)"
            print(f"[{BACKEND}] {_stopped}: every further call fails, unscored", flush=True)


def _stop(reason: str) -> None:
    global _stopped
    with _lock:
        if _stopped is None:
            _stopped = reason
            print(f"[{BACKEND}] {reason}: every further call fails, unscored", flush=True)


# ── the model ────────────────────────────────────────────────────────────


class _RetryTransport(httpx.AsyncHTTPTransport):
    """A dropped connection, a 429 or a 5xx is retried with backoff; a 402 (no balance) and a
    400 (a request this code got wrong) are not."""

    def __init__(self) -> None:
        super().__init__(retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import asyncio

        for attempt in range(ATTEMPTS):
            last = attempt == ATTEMPTS - 1
            try:
                response = await super().handle_async_request(request)
            except (httpx.TransportError, httpx.TimeoutException):
                if last:
                    raise
            else:
                if response.status_code not in _RETRY_STATUS or last:
                    return response
                await response.aread()
                await response.aclose()
            await asyncio.sleep(min(20.0, 2**attempt) + random.random())
        raise RuntimeError("unreachable")


# one model per thread and setting: tau2 plays conversations on worker threads, and an async
# HTTP client belongs to the event loop of the thread that made it
_local = threading.local()


def build_model(model: str, effort: str, transport: httpx.AsyncBaseTransport | None = None) -> Any:
    """DeepSeek's chat model thinking at `effort`, one per thread. `transport` replaces the
    network (a test's mock), and such a model is never cached."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.settings import ModelSettings

    check(model, effort)
    cache: dict[tuple[str, str], Any] = getattr(_local, "models", None) or {}
    _local.models = cache
    if transport is not None or (model, effort) not in cache:
        provider = DeepSeekProvider(
            api_key=api_key(),
            http_client=httpx.AsyncClient(
                timeout=httpx.Timeout(TIMEOUT_S, connect=10.0),
                transport=transport or _RetryTransport(),
            ),
        )
        built = OpenAIChatModel(
            model,
            provider=provider,
            profile=DeepSeekProvider.model_profile(PROFILE_AS.get(model, model)),
            settings=ModelSettings(thinking=effort),  # type: ignore[typeddict-item]
        )
        if transport is not None:
            return built
        cache[(model, effort)] = built
    return cache[(model, effort)]


# ── tau2 ⇄ Pydantic AI ───────────────────────────────────────────────────


def tool_definitions(tools: Iterable[Any]) -> list[Any]:
    """tau2's tools as definitions only: the schema tau2 would send, nothing Pydantic AI can run."""
    from pydantic_ai.tools import ToolDefinition

    out = []
    for t in tools:
        fn = t.openai_schema["function"]
        out.append(
            ToolDefinition(
                name=fn["name"],
                description=fn.get("description"),
                parameters_json_schema=fn.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return out


def reasoning_of(raw: dict[str, Any] | None) -> str:
    """The reasoning a reply's `raw_data` holds (`choices[0].message`), or ''."""
    choices = (raw or {}).get("choices") or [{}]
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("reasoning_content") or msg.get("reasoning") or "")


def _flatten(messages: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for m in messages:
        inner = getattr(m, "tool_messages", None)
        out.extend(inner if inner is not None else [m])
    return out


def to_model_response(m: Any) -> Any:
    """An agent reply tau2 holds, as the Pydantic AI response it was: reasoning, text, calls."""
    from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
    from pydantic_ai.usage import RequestUsage

    raw = m.raw_data or {}
    parts: list[Any] = []
    if reasoning := reasoning_of(raw):
        parts.append(ThinkingPart(content=reasoning, id=REASONING_ID, provider_name=PROVIDER))
    if m.content:
        parts.append(TextPart(content=m.content))
    for tc in m.tool_calls or []:
        parts.append(
            ToolCallPart(tool_name=tc.name, args=dict(tc.arguments or {}), tool_call_id=tc.id)
        )
    usage = m.usage or {}
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    return ModelResponse(
        parts=parts,
        usage=RequestUsage(
            input_tokens=int(usage.get("prompt_tokens") or 0),
            cache_read_tokens=hit,
            output_tokens=int(usage.get("completion_tokens") or 0),
        ),
        model_name=raw.get("model"),
        provider_name=PROVIDER if raw.get("model") else None,
    )


def to_model_messages(messages: Sequence[Any]) -> list[Any]:
    """tau2's history (system, user, assistant and tool messages) as Pydantic AI messages.

    A system message is a system part where it stands, the customer's words a user part, a tool
    result a tool-return part named for the call it answers; an agent reply closes the request
    before it and is a response of its own."""
    from pydantic_ai.messages import (
        ModelRequest,
        SystemPromptPart,
        ToolReturnPart,
        UserPromptPart,
    )

    out: list[Any] = []
    parts: list[Any] = []
    names: dict[str, str] = {}

    def flush() -> None:
        if parts:
            out.append(ModelRequest(parts=list(parts)))
            parts.clear()

    for m in _flatten(messages):
        role = getattr(m, "role", None)
        if role == "system":
            parts.append(SystemPromptPart(content=m.content or ""))
        elif role == "user":
            parts.append(UserPromptPart(content=m.content or ""))
        elif role == "tool":
            parts.append(
                ToolReturnPart(
                    tool_name=names.get(m.id, "unknown"), content=m.content or "", tool_call_id=m.id
                )
            )
        elif role == "assistant":
            flush()
            out.append(to_model_response(m))
            names.update({tc.id: tc.name for tc in m.tool_calls or []})
    flush()
    return out


def _usage(response: Any) -> dict[str, Any]:
    """The reply's tokens in litellm's keys, which every reader in this repo uses. DeepSeek's
    input count includes the cache hits."""
    u = response.usage
    details = dict(u.details or {})
    total_in = int(u.input_tokens or 0)
    hit = int(u.cache_read_tokens or details.get("prompt_cache_hit_tokens") or 0)
    out = int(u.output_tokens or 0)
    reasoning = int(u.output_reasoning_tokens or details.get("reasoning_tokens") or 0)
    return {
        "prompt_tokens": total_in,
        "completion_tokens": out,
        "total_tokens": total_in + out,
        "prompt_cache_hit_tokens": hit,
        "prompt_cache_miss_tokens": max(0, total_in - hit),
        "completion_tokens_details": {"reasoning_tokens": reasoning},
    }


def to_assistant_message(
    response: Any, model: str, effort: str, seconds: float, at: datetime, attempts: int = 1
) -> Any:
    """A Pydantic AI response as tau2's `AssistantMessage`, its reasoning in `raw_data`."""
    from pydantic_ai.messages import TextPart, ThinkingPart, ToolCallPart
    from tau2.data_model.message import AssistantMessage, ToolCall

    text = "".join(p.content for p in response.parts if isinstance(p, TextPart))
    reasoning = "\n\n".join(p.content for p in response.parts if isinstance(p, ThinkingPart))
    calls, raw_calls = [], []
    for p in response.parts:
        if not isinstance(p, ToolCallPart):
            continue
        try:
            args = p.args_as_dict()
        except ValueError:
            args = {}  # arguments that are not JSON: tau2 answers the call with its error
        calls.append(ToolCall(id=p.tool_call_id, name=p.tool_name, arguments=args))
        raw_calls.append(
            {
                "id": p.tool_call_id,
                "type": "function",
                "function": {"name": p.tool_name, "arguments": p.args_as_json_str()},
            }
        )
    usage = _usage(response)
    cost = cost_usd(
        model,
        hit=usage["prompt_cache_hit_tokens"],
        miss=usage["prompt_cache_miss_tokens"],
        output=usage["completion_tokens"],
        at=at,
    )
    raw = {
        "id": response.provider_response_id,
        "object": "chat.completion",
        "created": int(at.timestamp()),
        "model": response.model_name,
        "choices": [
            {
                "index": 0,
                "finish_reason": response.finish_reason,
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "reasoning_content": reasoning or None,
                    "tool_calls": raw_calls or None,
                },
            }
        ],
        "usage": usage,
        "backend": {
            "harness": BACKEND,
            "pydantic_ai": _pydantic_ai_version(),
            "provider": PROVIDER,
            "model": model,
            "effort": effort,
            "attempts": attempts,
            "peak": peak(at),
        },
    }
    return AssistantMessage(
        role="assistant",
        content=text or None,
        tool_calls=calls or None,
        cost=round(cost, 8),
        usage=usage,
        raw_data=raw,
        generation_time_seconds=seconds,
    )


def _pydantic_ai_version() -> str:
    from importlib.metadata import version

    return version("pydantic-ai-slim")


def unavailable(exc: BaseException) -> bool:
    """A provider that will not answer after the transport's retries (by status or class)."""
    seen: BaseException | None = exc
    while seen is not None:
        status = getattr(seen, "status_code", None)
        if status in _RETRY_STATUS or isinstance(
            seen, httpx.TransportError | httpx.TimeoutException
        ):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _status(exc: BaseException) -> int | None:
    seen: BaseException | None = exc
    while seen is not None:
        if isinstance(status := getattr(seen, "status_code", None), int):
            return status
        seen = seen.__cause__ or seen.__context__
    return None


def generate(model: str, effort: str, messages: Sequence[Any], tools: Iterable[Any]) -> Any:
    """One agent turn: tau2's history and tools in, tau2's `AssistantMessage` out.

    `model_request_sync` makes one request and never runs a tool. A reply with neither text nor
    a call is asked for again (`EMPTY_RETRIES`); every attempt is charged."""
    from pydantic_ai.direct import model_request_sync
    from pydantic_ai.models import ModelRequestParameters

    from tau2_loop.llm.core import check_billing

    check_billing()
    if _stopped is not None:
        raise BackendStoppedError(_stopped)
    llm = build_model(model, effort)
    history = to_model_messages(messages)
    params = ModelRequestParameters(function_tools=tool_definitions(tools))
    message = None
    for attempt in range(1 + EMPTY_RETRIES):
        started, at = time.perf_counter(), datetime.now(UTC)
        try:
            response = model_request_sync(llm, history, model_request_parameters=params)
        except Exception as e:
            if _status(e) == 402:
                _stop(f"DeepSeek returned 402 (insufficient balance): {e}"[:300])
                raise BackendStoppedError(_stopped or "402") from e
            raise
        message = to_assistant_message(
            response, model, effort, time.perf_counter() - started, at, attempt + 1
        )
        _charge(float(message.cost or 0.0))
        if (message.content or "").strip() or message.tool_calls:
            return message
    assert message is not None
    raise RuntimeError(
        f"{model} gave no text and no tool call in {1 + EMPTY_RETRIES} attempts "
        f"(finish reason {response.finish_reason})"
    )


# ── the trajectory file ──────────────────────────────────────────────────

TRAJECTORIES = "trajectories.jsonl.gz"


def trajectory(system_prompt: str, sim: Any) -> dict[str, Any]:
    """One conversation as the agent saw it, as Pydantic AI messages: the composed prompt, the
    customer's words, every reply with its reasoning, every call and result. The customer's own
    tool calls and results are left out, as tau2 leaves them out of the agent's history."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from tau2.agent.base_agent import is_valid_agent_history_message
    from tau2.data_model.message import SystemMessage

    seen = [m for m in _flatten(sim.messages or []) if is_valid_agent_history_message(m)]
    history = to_model_messages([SystemMessage(role="system", content=system_prompt), *seen])
    ri = sim.reward_info
    return {
        "task_id": sim.task_id,
        "trial": int(sim.trial or 0) + 1,
        "simulation_id": sim.id,
        "reward": float(ri.reward) if ri is not None else None,
        "termination_reason": str(getattr(sim.termination_reason, "value", sim.termination_reason)),
        "messages": ModelMessagesTypeAdapter.dump_python(history, mode="json"),
    }


def write_trajectories(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    import gzip

    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_trajectories(path: Path) -> list[dict[str, Any]]:
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
