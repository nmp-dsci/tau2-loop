"""Our agent, as tau2 sees it: a registry factory named `tau2_loop`.

The orchestrator owns the turn loop and executes every tool; our only code in
the conversation is this class. Each turn it calls tau2's `generate()` with the
version's system prompt (the policy stuffed in), the history and the domain's
tools, over the `claude-sdk/` provider. The version's `helper.py` may define
three optional hooks — `on_tool_call`, `on_reply`, `extra_context` — the
deterministic guards an optimiser can put around the model without touching
the harness.

`agent.yaml` (frozen) picks the model, effort and tool mode; the effort
reaches the provider as `reasoning_effort`. The version to run is passed by the
runner through `llm_args["tau2_loop_version"]`, so it lands in tau2's own
results file as provenance. When the runner routes the agent to the service
(`openai/<model>` with an `api_base`), the bearer token is added here, at run
time, so it never enters the run config tau2 writes to disk.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.tool import Tool

from tau2_loop.agent.compose import call_hook, compose, load_helper_file
from tau2_loop.agent.versions import AgentVersion, load_version, parse_ref
from tau2_loop.config import settings
from tau2_loop.llm import sdk_model

AGENT_NAME = "tau2_loop"
VERSION_KEY = "tau2_loop_version"
SERVICE_PREFIX = "openai/"


def with_effort(llm_args: dict[str, Any], effort: str) -> dict[str, Any]:
    """`reasoning_effort` for a call, and the litellm switch without which a custom provider never sees it."""
    out = dict(llm_args)
    out.setdefault("reasoning_effort", effort)
    allowed = list(out.get("allowed_openai_params") or [])
    if "reasoning_effort" not in allowed:
        allowed.append("reasoning_effort")
    out["allowed_openai_params"] = allowed
    return out


@dataclass
class LoopAgentState:
    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage] = field(default_factory=list)
    n_model_calls: int = 0
    parse_failures: int = 0


class LoopAgent(HalfDuplexAgent[LoopAgentState]):  # type: ignore[misc]
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        version: AgentVersion,
        llm: str | None = None,
        llm_args: dict[str, Any] | None = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.version = version
        self.llm = llm or sdk_model(version.config.model)
        args = {k: v for k, v in (llm_args or {}).items() if k != VERSION_KEY}
        self.llm_args = with_effort(args, version.config.effort)
        if self.llm.startswith(SERVICE_PREFIX):
            token = settings().agent_service_token.get_secret_value()
            if not token:
                raise RuntimeError(
                    "the agent is routed to the service but AGENT_SERVICE_TOKEN is not set"
                )
            self.llm_args["api_key"] = token
        self.helper = load_helper(version)

    def system_prompt(self) -> str:
        # the one composition, shared with the viewer's `GET /api/runs/{id}/agent`
        return compose(self.version.system_prompt, self.domain_policy, self.helper).text

    def get_init_state(self, message_history: list[Message] | None = None) -> LoopAgentState:
        return LoopAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt())],
            messages=list(message_history) if message_history else [],
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LoopAgentState
    ) -> tuple[AssistantMessage, LoopAgentState]:
        from tau2.utils.llm_utils import generate

        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=state.system_messages + state.messages,
            call_name="tau2_loop_agent",
            **self.llm_args,
        )
        state.n_model_calls += 1
        if looks_unparsed(response.content):
            state.parse_failures += 1
        response = self._apply_helper(response)
        state.messages.append(response)
        return response, state

    def _apply_helper(self, response: AssistantMessage) -> AssistantMessage:
        if self.helper is None:
            return response
        if response.tool_calls:
            fixed: list[ToolCall] = []
            for tc in response.tool_calls:
                out = call_hook(self.helper, "on_tool_call", tc.name, dict(tc.arguments or {}))
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                    fixed.append(ToolCall(id=tc.id, name=str(out[0]), arguments=out[1]))
                else:
                    fixed.append(tc)
            response.tool_calls = fixed
        elif response.content:
            out = call_hook(self.helper, "on_reply", response.content)
            if isinstance(out, str) and out.strip():
                response.content = out
        return response

    def is_stop(self, message: AssistantMessage) -> bool:
        return bool(message.content) and "###STOP###" in message.content

    def set_seed(self, seed: int) -> None:
        return None


def looks_unparsed(content: str | None) -> bool:
    """A text reply that is really a JSON contract object the provider could not parse."""
    if not content:
        return False
    s = content.lstrip()
    return s.startswith(("{", "```")) and '"tool_calls"' in s


def load_helper(version: AgentVersion) -> types.ModuleType | None:
    """Import `agents/<domain>/vN/helper.py` as a module, if present; a broken helper is None."""
    return load_helper_file(
        version.helper_path, f"tau2_loop_helper_{version.domain}_{version.name}"
    )


def create_loop_agent(tools: list[Tool], domain_policy: str, **kwargs: Any) -> LoopAgent:
    llm_args = dict(kwargs.get("llm_args") or {})
    ref = llm_args.get(VERSION_KEY)
    if not ref:
        raise ValueError(f"llm_args_agent must carry {VERSION_KEY}=<domain>/<vN>")
    domain, name = parse_ref(str(ref))
    version = load_version(domain, name)
    return LoopAgent(
        tools=tools,
        domain_policy=domain_policy,
        version=version,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
    )


def register() -> None:
    """Register the factory with tau2 and the SDK provider with litellm; idempotent."""
    from tau2.registry import registry

    from tau2_loop.llm import sdk_provider

    sdk_provider.register()
    if AGENT_NAME not in registry.get_agents():
        registry.register_agent_factory(create_loop_agent, AGENT_NAME)


# Message and ToolMessage are re-exported for type checkers reading this module.
__all__ = ["LoopAgent", "LoopAgentState", "create_loop_agent", "register", "Message", "ToolMessage"]
