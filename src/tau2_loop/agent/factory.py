"""Our agent, as tau2 sees it: a registry factory named `tau2_loop`.

The orchestrator owns the turn loop and executes every tool; our only code in
the conversation is this class. Each turn it calls tau2's `generate()` with the
version's system prompt (the policy stuffed in), the history and the domain's
tools, over the `claude-sdk/` provider. The version's `helper.py` may define
three optional hooks — `on_tool_call`, `on_reply`, `extra_context` — the
deterministic guards an optimiser can put around the model without touching
the harness.

`agent.yaml` (frozen) picks the model, effort, tool mode and (banking) retrieval;
the effort reaches the provider as `reasoning_effort`, and `tool_mode: native` as
the core's `tau2_loop_tool_mode`, so the model calls the tools natively instead of
writing the JSON contract (s13, milestone 2). The version to run is passed by the
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
state)` sees each write call (and, since s15, a transfer) before tau2 runs it, and a string back blocks the
reply once: the model gets the message as the calls' tool results and replies
again, and that reply goes through unchecked. What a hook did is recorded on the
reply tau2 keeps, under `raw_data["tau2_loop"]`, for the Agent tab.

Since s16 a version whose `agent.yaml` names `workflows: rN` gets two harness tools beside tau2's:
`find_workflow` (look the customer's job up in the workflows workflow_rag rN wrote) and
`request_workflow` (have workflow_rag research it now). The harness runs them inside the turn and
asks the model again with the result, so tau2 never sees them and they change nothing it grades;
the calls and results stay in the agent's own history, and are recorded under
`raw_data["tau2_loop"]["workflow"]`. A live conversation (`eval/live.py`) also streams them.
"""

from __future__ import annotations

import contextlib
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
from tau2.environment.tool import Tool, as_tool

from tau2_loop.agent.compose import (
    CHECKED_TOOLS,
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
from tau2_loop.llm.core import NATIVE_PARALLEL, TOOL_MODE_PARAM

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


def with_tool_mode(
    llm_args: dict[str, Any], service: bool = False, parallel: bool = False
) -> dict[str, Any]:
    """Native tool calls for this agent's calls only (the customer and the judge stay on the
    contract): the core's parameter, passed the way `reasoning_effort` is, or in the request
    body when the agent is the service. `parallel` keeps every call of a reply (s14 P0a)."""
    out = dict(llm_args)
    mode = NATIVE_PARALLEL if parallel else "native"
    if service:
        out["extra_body"] = {**(out.get("extra_body") or {}), TOOL_MODE_PARAM: mode}
        return out
    out[TOOL_MODE_PARAM] = mode
    allowed = list(out.get("allowed_openai_params") or [])
    if TOOL_MODE_PARAM not in allowed:
        allowed.append(TOOL_MODE_PARAM)
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
# s16: the harness's workflow tools, run inside the turn; another call in the same reply waits.
WORKFLOW_TOOLS = ("find_workflow", "request_workflow")
WORKFLOW_ROUNDS = 4  # workflow calls in one turn before the model is asked to go on without them
WORKFLOW_HELD = (
    "NOT EXECUTED: a workflow tool in the same reply ran first. Send this call again if it is "
    "still needed."
)
WORKFLOW_DONE = "Workflow tools are used up for this turn: continue with your other tools or reply."
WORKFLOW_RESULT_CHARS = 4000  # of each result, kept on the reply for the trace


def find_workflow(request: str, job: str = "") -> str:
    """Look up the workflow library: procedures workflow_rag researched ahead of time, by job.

    Call it as soon as you know what the customer wants, before you act. It lists the library's
    jobs ranked for your request and gives the best match in full: its steps and who takes each,
    the facts to ask the customer for, and the rules that decide. Follow a workflow only if it is
    the customer's job; it is guidance from the knowledge base, never a reason to skip a step the
    policy or a document requires.

    Args:
        request: What the customer wants, in one sentence of your own words.
        job: Optional. A job name from an earlier lookup, to read that workflow in full.

    Returns:
        The library's jobs ranked for the request, and one workflow as JSON.
    """
    raise RuntimeError("run by the harness, never by tau2")


def request_workflow(request: str) -> str:
    """Ask workflow_rag to research the procedure for a request no workflow in the library fits.

    workflow_rag searches the knowledge base and writes a workflow: the steps and who takes each,
    the facts to ask for and the rules, each with its source. It takes a few minutes while the
    customer waits, so use it only after find_workflow found no job that fits.

    Args:
        request: What the customer wants and the facts so far that bear on the procedure, in your
            own words. Leave out personal details such as names, ids and contact details.

    Returns:
        The workflow as JSON.
    """
    raise RuntimeError("run by the harness, never by tau2")


# A text reply checks.py's check_reply refused (s14); the model sees it and writes again.
REPLY_BLOCKED = (
    "NOT SENT: your last message did not reach the customer. A check on it failed: {msg} "
    "Write your reply again, or make the tool call it calls for."
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
        if version.config.tool_mode == "native":
            self.llm_args = with_tool_mode(
                self.llm_args,
                service=self.llm.startswith(SERVICE_PREFIX),
                parallel=version.config.parallel_calls,
            )
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
        # s16: workflow_rag's library and research, as two harness tools (None before v6)
        self.workflows = version.config.workflows
        self.workflow_tools: list[Tool] = (
            [as_tool(find_workflow), as_tool(request_workflow)] if self.workflows else []
        )
        # set by a live conversation: each workflow call and result is streamed as it happens
        self.live: Any = None
        self.conversation_id: str | None = None

    def system_prompt(self) -> str:
        # the one composition, shared with the viewer's `GET /api/runs/{id}/agent`
        return compose(
            self.version.system_prompt,
            self.domain_policy,
            self.helper,
            identity=self.version.config.identity_note,
        ).text

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

        def ask(messages: list[Any], workflow: bool = True) -> AssistantMessage:
            out = generate(
                model=self.llm,
                tools=self.tools + (self.workflow_tools if workflow else []),
                messages=messages,
                call_name="tau2_loop_agent",
                **self.llm_args,
            )
            state.n_model_calls += 1
            if looks_unparsed(out.content):
                state.parse_failures += 1
            return self._apply_helper(out)

        response = ask(call)
        if self.workflow_tools:
            response, extra, used = self._workflow_rounds(call, response, ask)
            if extra:
                # the lookups stay in the agent's own history, so later turns still have them
                state.messages.extend(extra)
                call = call + extra
                note["workflow"] = used
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
        elif not response.tool_calls and response.content:
            refused = call_hook(
                self.code.get("checks.py"), "check_reply", response.content, state.memory
            )
            if isinstance(refused, str) and refused.strip():
                # once per turn, as for writes: the model sees why, and its next reply goes through
                state.n_blocked += 1
                note["blocked_reply"] = {"content": response.content, "check": refused.strip()}
                why = SystemMessage(
                    role="system", content=REPLY_BLOCKED.format(msg=refused.strip())
                )
                response = ask(call + [response, why])
                note["retried"] = True
        if note:
            response.raw_data = {**(response.raw_data or {}), "tau2_loop": note}
        state.messages.append(response)
        return response, state

    def _emit(self, row: dict[str, Any]) -> None:
        if self.live is not None:
            # a viewer that cannot keep up never breaks a turn
            with contextlib.suppress(Exception):
                self.live(row)

    def _workflow_rounds(
        self, call: list[Any], response: AssistantMessage, ask: Any
    ) -> tuple[AssistantMessage, list[Any], list[dict[str, Any]]]:
        """Run the reply's workflow calls and ask again, until a reply makes none."""
        extra: list[Any] = []
        used: list[dict[str, Any]] = []
        for _ in range(WORKFLOW_ROUNDS):
            calls = list(response.tool_calls or [])
            if not any(tc.name in WORKFLOW_TOOLS for tc in calls):
                return response, extra, used
            results = []
            for tc in calls:
                content = WORKFLOW_HELD
                if tc.name in WORKFLOW_TOOLS:
                    content, rec = self._run_workflow_tool(tc)
                    used.append(rec)
                results.append(
                    ToolMessage(id=tc.id, role="tool", requestor="assistant", content=content)
                )
            extra += [response, *results]
            response = ask(call + extra)
        if any(tc.name in WORKFLOW_TOOLS for tc in response.tool_calls or []):
            response = ask(
                call + extra + [SystemMessage(role="system", content=WORKFLOW_DONE)], False
            )
        return response, extra, used

    def _run_workflow_tool(self, tc: ToolCall) -> tuple[str, dict[str, Any]]:
        from tau2_loop.workflows import library

        args = dict(tc.arguments or {})
        text = str(args.get("request") or "")
        session: str | None = None
        self._emit({"kind": "workflow_call", "id": tc.id, "name": tc.name, "args": args})
        try:
            if tc.name == "find_workflow":
                content = library.find(
                    self.version.domain, str(self.workflows), text, str(args.get("job") or "")
                )
            else:
                content, session = library.request(
                    self.version.domain,
                    str(self.workflows),
                    text,
                    conversation=self.conversation_id,
                    on_start=lambda m: self._emit(
                        {"kind": "workflow_research", "id": tc.id, "session": m.id}
                    ),
                )
        except Exception as e:  # noqa: BLE001 - a failed lookup is the model's to work around
            content = f"{tc.name} failed: {type(e).__name__}: {e}"[:500]
        self._emit(
            {
                "kind": "workflow_result",
                "id": tc.id,
                "name": tc.name,
                "content": content,
                "session": session,
            }
        )
        return content, {
            "name": tc.name,
            "arguments": args,
            "result": content[:WORKFLOW_RESULT_CHARS],
            "session": session,
        }

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
        """checks.py's verdict on each write call (or transfer) in a reply: the calls it blocks, by position."""
        mod = self.code.get("checks.py")
        if mod is None or not response.tool_calls:
            return {}
        out: dict[int, str] = {}
        for i, tc in enumerate(response.tool_calls):
            if self.kinds.get(tc.name) != "write" and tc.name not in CHECKED_TOOLS:
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
