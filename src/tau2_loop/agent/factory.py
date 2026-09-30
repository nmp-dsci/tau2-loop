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

Since s09 a version may also carry three code surfaces, each one hook:
`memory.py` `remember(state, name, arguments, result)` keeps facts from every tool
result and user message within one conversation; `guidance.py`
`guidance(state, trigger)` returns a short reminder that rides on that one model
call as a system message (the provider folds it into the call's system prompt,
so the transcript never holds it); `checks.py` `check_write(name, arguments,
state)` sees each write call before tau2 runs it, and a string back blocks the
reply once: the model gets the message as the calls' tool results and replies
again, and that reply goes through unchecked. What a hook did is recorded on the
reply tau2 keeps, under `raw_data["tau2_loop"]`, for the Agent tab.
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

from tau2_loop.agent.compose import (
    GUIDANCE_CHARS,
    call_hook,
    compose,
    load_code_surfaces,
    load_helper_file,
)
from tau2_loop.agent.versions import AgentVersion, load_version, parse_ref
from tau2_loop.config import settings
from tau2_loop.data.splits import tool_kinds
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
    # memory.py's state: this conversation's only, never carried into another
    memory: dict[str, Any] = field(default_factory=dict)
    n_blocked: int = 0


# The tool result a blocked call gets; the model sees it and replies again.
BLOCKED = "NOT EXECUTED. A check on this call failed: {msg} Fix the call and send it again, or ask the user."
HELD_BACK = (
    "NOT EXECUTED: another call in this reply failed a check. Send it again if it is still needed."
)


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
        self.code = load_code_surfaces(
            version.path, f"tau2_loop_code_{version.domain}_{version.name}"
        )
        self.kinds = tool_kinds(version.domain)

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

        incoming = (
            list(message.tool_messages) if isinstance(message, MultiToolMessage) else [message]
        )
        self._remember(state, incoming)
        state.messages.extend(incoming)
        trigger = "tool" if all(isinstance(m, ToolMessage) for m in incoming) else "user"
        note: dict[str, Any] = {}
        reminder = self._guidance(state, trigger)
        call = state.system_messages + state.messages
        if reminder:
            note["guidance"] = reminder
            call = call + [SystemMessage(role="system", content=reminder)]

        def ask(messages: list[Any]) -> AssistantMessage:
            out = generate(
                model=self.llm,
                tools=self.tools,
                messages=messages,
                call_name="tau2_loop_agent",
                **self.llm_args,
            )
            state.n_model_calls += 1
            if looks_unparsed(out.content):
                state.parse_failures += 1
            return self._apply_helper(out)

        response = ask(call)
        blocked = self._check(response, state)
        if blocked:
            # once per turn: the model sees why, replies again, and that reply goes through
            state.n_blocked += 1
            calls = list(response.tool_calls or [])
            results = [
                ToolMessage(
                    id=tc.id,
                    role="tool",
                    requestor="assistant",
                    error=True,
                    content=BLOCKED.format(msg=blocked[i]) if i in blocked else HELD_BACK,
                )
                for i, tc in enumerate(calls)
            ]
            note["blocked"] = [
                {"name": tc.name, "arguments": dict(tc.arguments or {}), "check": blocked[i]}
                for i, tc in enumerate(calls)
                if i in blocked
            ]
            response = ask(call + [response, *results])
            note["retried"] = True
        if note:
            response.raw_data = {**(response.raw_data or {}), "tau2_loop": note}
        state.messages.append(response)
        return response, state

    def _remember(self, state: LoopAgentState, incoming: list[Any]) -> None:
        """memory.py's hook on each tool result (with the call that asked for it) and user message."""
        mod = self.code.get("memory.py")
        if mod is None:
            return
        calls = {
            tc.id: tc
            for m in state.messages
            if isinstance(m, AssistantMessage)
            for tc in (m.tool_calls or [])
        }
        for m in incoming:
            if isinstance(m, ToolMessage):
                tc = calls.get(m.id)
                name = tc.name if tc else "unknown"
                args = dict(tc.arguments or {}) if tc else {}
                call_hook(mod, "remember", state.memory, name, args, m.content or "")
            elif getattr(m, "content", None):
                call_hook(mod, "remember", state.memory, "user", {}, m.content)

    def _guidance(self, state: LoopAgentState, trigger: str) -> str | None:
        """guidance.py's reminder for this call, cut to the budget; None when it has nothing to say."""
        out = call_hook(self.code.get("guidance.py"), "guidance", state.memory, trigger)
        if not isinstance(out, str) or not out.strip():
            return None
        return out.strip()[:GUIDANCE_CHARS]

    def _check(self, response: AssistantMessage, state: LoopAgentState) -> dict[int, str]:
        """checks.py's verdict on each write call in a reply: the calls it blocks, by position."""
        mod = self.code.get("checks.py")
        if mod is None or not response.tool_calls:
            return {}
        out: dict[int, str] = {}
        for i, tc in enumerate(response.tool_calls):
            if self.kinds.get(tc.name) != "write":
                continue
            msg = call_hook(mod, "check_write", tc.name, dict(tc.arguments or {}), state.memory)
            if isinstance(msg, str) and msg.strip():
                out[i] = msg.strip()
        return out

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
