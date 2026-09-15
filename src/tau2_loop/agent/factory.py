"""Our agent, as tau2 sees it: a registry factory named `tau2_loop`.

The orchestrator owns the turn loop and executes every tool; our only code in
the conversation is this class. Each turn it calls tau2's `generate()` with the
version's system prompt (the policy stuffed in), the history and the domain's
tools, over the `claude-sdk/` provider. The version's `helper.py` may define
three optional hooks — `on_tool_call`, `on_reply`, `extra_context` — the
deterministic guards an optimiser can put around the model without touching
the harness.

`agent.yaml` (frozen) picks the model, effort and tool mode. The version to run
is passed by the runner through `llm_args["tau2_loop_version"]`, so it lands
in tau2's own results file as provenance.
"""

from __future__ import annotations

import importlib.util
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

from tau2_loop.agent.versions import AgentVersion, load_version, parse_ref
from tau2_loop.llm import sdk_model

AGENT_NAME = "tau2_loop"
VERSION_KEY = "tau2_loop_version"

# A default prompt for a version folder whose system.md has no {policy} slot.
POLICY_SLOT = "{policy}"


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
        self.llm_args = {k: v for k, v in (llm_args or {}).items() if k != VERSION_KEY}
        self.helper = load_helper(version)

    def system_prompt(self) -> str:
        text = self.version.system_prompt
        if POLICY_SLOT in text:
            text = text.replace(POLICY_SLOT, self.domain_policy)
        else:
            text = f"{text}\n\n# Domain policy\n{self.domain_policy}"
        extra = _call(self.helper, "extra_context", self.domain_policy)
        if isinstance(extra, str) and extra.strip():
            text = f"{text}\n\n{extra.strip()}"
        return text

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
                out = _call(self.helper, "on_tool_call", tc.name, dict(tc.arguments or {}))
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                    fixed.append(ToolCall(id=tc.id, name=str(out[0]), arguments=out[1]))
                else:
                    fixed.append(tc)
            response.tool_calls = fixed
        elif response.content:
            out = _call(self.helper, "on_reply", response.content)
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
    p = version.helper_path
    if p is None:
        return None
    spec = importlib.util.spec_from_file_location(
        f"tau2_loop_helper_{version.domain}_{version.name}", p
    )
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001 - a helper that does not import is a version without one
        return None
    return mod


def _call(helper: types.ModuleType | None, name: str, *args: Any) -> Any:
    fn = getattr(helper, name, None) if helper is not None else None
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001 - a helper exception never breaks a turn
        return None


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
