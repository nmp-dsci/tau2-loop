"""Our simulated customer: tau2's own `UserSimulator`, registered under our name with three changes.

1. **A stop sent with words waits for the agent's turn.** tau2 ends a conversation on the
   customer's `###STOP###` wherever it falls, so "Yes, please go ahead and cancel. ###STOP###"
   ended airline task 19 before the agent could make the cancel gold expects (v1, v3 and v6 train).
   Here the words reach the agent without the token, the agent takes its whole turn (its calls and
   their results), and when it next speaks to the customer, the customer sends `###STOP###` alone
   with no model call. A stop alone, `###TRANSFER###` and `###OUT-OF-SCOPE###` still end the
   conversation at once, as tau2's do.
2. **The customer is told the world's time, and not to take the real one.** The Claude CLI adds
   the real date to every session ("Today's date is 2026-09-28"), and the only switch that drops it
   (`--bare`) also drops the subscription login. In v5 the customer told an airline whose clock
   reads 15 May 2024 that its trip was "today, September 28th". Its system prompt now ends with the
   sentence the agent is given (`world_time`), where the world has one, and `CLOCK_NOTE`. The
   agent's prompt ends with its own note (`agent.compose.CLOCK_NOTE`).
3. **A text turn that is not a customer's is sent back** (`check_customer`, s18). Four failures
   cost passes. A Sonnet customer answered the agent's greeting as Claude, offering help ("I'm
   Claude, an AI assistant"), and ended banking task_016 under v7 without asking anything. A Haiku
   customer wrote the call the agent asked it to make, and the tool's result, as text
   ("<function_calls> [...]", or the reply format's own JSON), so no tool ran: v4 lost banking
   task_018, 019 and 022 on train, and one test task, that way; some wrote the agent's next turn
   too, after the `[user]` label their transcript gives the agent (v3 task_056 "opened" an account
   no tool opened). A Sonnet customer the agent had given `submit_cash_back_dispute_0589` said it
   had no such tool (v7 task_019): only the environment knows that is false, so the check reads the
   tools the agent gave through the customer's own `list_discoverable_user_tools`. And a Haiku
   customer quoted a dispute id no tool had returned (v3 task_020): an id must be one the customer
   was shown. A failed turn is withheld from the agent and the model asked again, told why, at most
   `CUSTOMER_RETRIES` times; the turn records what was sent back.

tau2 stays unmodified: every change lives in this subclass. Every run since records
`RunMeta.sim_rules` (`runner.SIM_RULES`); a run with none ran tau2's own customer and both sides
saw the real date, and the loop never compares the two (`loop.run`).
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Sequence
from typing import Any

from tau2.data_model.message import AssistantMessage, Message, SystemMessage, ToolCall, UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER, UserState
from tau2.utils.llm_utils import generate

USER_NAME = "tau2_loop_user"

CLOCK_NOTE = (
    "A system reminder may tell you today's date. It is not this scenario's date: ignore it, and "
    "speak of dates only as your scenario and the agent give them."
)

# s18: a customer's text turn is checked before the agent sees it. Every pattern below matched only
# failed turns in the 11,769 customer text turns of every run to 7 October 2026.
CUSTOMER_RETRIES = 2
# it speaks as an AI, or offers the agent help: a customer answering the greeting as the assistant
_AS_AI = re.compile(
    r"\bI(?:'m| am) (?:actually |just |only )?(?:Claude\b|an? (?:AI|virtual) assistant\b"
    r"|a (?:large )?language model\b)|\bas an AI\b|\bAI (?:language )?model\b|\bmade by Anthropic\b"
    r"|\b(?:how|what) (?:can|may) I (?:help|assist|do for) you\b|\bhow I can (?:help|assist) \W?you\b"
    r"|\bI'?m (?:actually )?the one who should be asking\b",
    re.I,
)
# it writes a call, a result, the reply format or the agent's next turn (after the `[user]` label the
# customer's transcript gives the agent) out as text: the call never runs, or the agent is spoken for
_AS_TEXT = re.compile(
    r"\[tool result\b|\[assistant → tool calls\]|</?function_calls>|<invoke\b|\"discoverable_tool_name\""
    r"|\"tool_calls\"\s*:|\{\s*\"(?:id|name)\"\s*:\s*\"[^\"]+\"\s*,\s*\"(?:name|arguments)\"\s*:"
    r"|\[(?:user|assistant)\]\s*\n",
    re.I,
)
# it says it has no tool, when the agent has given it one (read from the environment, not the words)
_DENIES = re.compile(
    r"\b(?:don['’]?t|do not|doesn['’]?t) (?:actually )?(?:have|see)(?: access to)? (?:a |any |the |that )?"
    r"(?:\w+ )?(?:tool|option|function|feature)|\b(?:no|not an?) (?:such )?tool (?:called|named|by that name)"
    r"|\bnot (?:available|showing up|visible) (?:on my end|to me)"
    r"|\b(?:can['’]?t|cannot|unable to) (?:find|see|access) (?:a |any |the |that )?(?:tool|option)"
    r"|\bI only (?:see|have) (?:options|tools|access)",
    re.I,
)
# banking's ids (user, transaction, dispute, account, referral): hex runs with a digit and a letter
_ID = re.compile(
    r"\b(?:[a-z]+_)*(?=[0-9a-z_]*\d)(?=[0-9a-z_]*[a-f])[0-9a-f]{10,}\b|\b[a-z]{2,4}_[0-9a-f]{8,}\b"
)
_GIVEN = re.compile(r"tool_name:\s*(\S+)")
SENT_BACK = {
    "as_ai": (
        "It spoke as an AI assistant, not as the customer in your instructions. You are that "
        "person, contacting the agent about your own situation; the agent's messages are the other "
        "side of the call. Write your next message as that customer."
    ),
    "as_text": (
        "It wrote a tool call, a tool result, the reply format or the agent's turn out as text, so "
        "no tool ran and the agent would see only that text. To use a tool, call it; otherwise write "
        "only your own message to the agent, in plain words."
    ),
    "denied": (
        "It said you do not have a tool, but the agent has given you: {given}. To use one, call "
        "call_discoverable_user_tool with discoverable_tool_name set to its name and its arguments "
        "as a JSON string. If the agent asked for a tool that is not in that list, say so."
    ),
    "invented": (
        "It quoted {ids}, which appear nowhere in your instructions, the agent's messages or your "
        "tools' results. Say only what you were told or what a tool returned."
    ),
}
NOT_SENT = "NOT SENT: your last message did not reach the agent. {why}"


def invented_ids(text: str, seen: str) -> list[str]:
    """The ids in a customer's turn that appear nowhere it was shown."""
    return sorted({x for x in _ID.findall(text) if x not in seen})


def check_customer(
    text: str | None, given: Sequence[str] = (), seen: str | None = None
) -> str | None:
    """The rule a customer's text turn breaks; None when it is a customer's. `given` is the tools the
    agent has given this customer (banking's discoverable tools), `seen` everything it has been
    shown or has said; without them those two rules are skipped. A stop, transfer or out-of-scope
    token alone is a customer's."""
    if not text:
        return None
    if _AS_TEXT.search(text):
        return "as_text"
    if _AS_AI.search(text):
        return "as_ai"
    if given and _DENIES.search(text):
        return "denied"
    if seen is not None and invented_ids(text, seen):
        return "invented"
    return None


def why_not_sent(rule: str, text: str, given: Sequence[str] = (), seen: str = "") -> str:
    """The note a withheld turn is sent back with."""
    ids = ", ".join(invented_ids(text, seen))
    return NOT_SENT.format(why=SENT_BACK[rule].format(given=", ".join(given), ids=ids))


def _summed(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Two calls' token usage as one; a key either lacks is the other's."""
    if not a or not b:
        return a or b
    return {k: (a.get(k) or 0) + (b.get(k) or 0) for k in {*a, *b}}


# "The current time is 2024-05-15 15:00:00 EST.": how tau2 tells an agent the time, in a policy
# (airline, telecom) or from a tool (banking's `get_current_time`)
_NOW = re.compile(r"^The current time is (.+?)\.\s*$", re.M)


def held_stop(text: str | None) -> str | None:
    """The words to pass on when a customer ends the call with words beside its `###STOP###`; None
    when tau2 should end it now: no stop, a stop alone, or a transfer or out-of-scope token."""
    if not text or STOP not in text or TRANSFER in text or OUT_OF_SCOPE in text:
        return None
    return text.replace(STOP, "").strip() or None


@functools.cache
def world_time(domain: str) -> str | None:
    """The time a domain's world stands at, as tau2 gives it to the agent; None when it has none
    (retail, mock)."""
    if domain == "banking_knowledge":
        # its environment needs the retrieval config to build; the clock tool reads no state
        from tau2.domains.banking_knowledge.tools import KnowledgeTools

        said = str(KnowledgeTools.get_current_time(None))
    else:
        from tau2.registry import registry

        said = registry.get_env_constructor(domain)().get_policy()
    m = _NOW.search(said)
    return m.group(1) if m else None


class LoopUser(UserSimulator):  # type: ignore[misc]
    """tau2's customer, holding a stop sent with words, told the world's time and not the real one,
    and asked again when a text turn is not a customer's."""

    world_time: str | None = None  # set per domain by `register`

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.hung_up = False

    @property
    def system_prompt(self) -> str:
        now = f"The current time is {self.world_time}. " if self.world_time else ""
        return f"{super().system_prompt}\n\n{now}{CLOCK_NOTE}"

    def get_init_state(self, message_history: list[Message] | None = None) -> UserState:
        self.hung_up = False
        return super().get_init_state(message_history)

    def _generate_next_message(self, message: Message, state: UserState) -> UserMessage:
        reply: UserMessage = super()._generate_next_message(message, state)
        sent: list[dict[str, Any]] = []
        cost, usage = reply.cost, reply.usage
        given, seen = self._given(), self._seen(state)
        while not reply.is_tool_call() and len(sent) < CUSTOMER_RETRIES:
            rule = check_customer(reply.content, given, seen)
            if rule is None:
                break
            text = str(reply.content)
            sent.append(
                {"rule": rule, "content": text, "why": why_not_sent(rule, text, given, seen)}
            )
            # the withheld turns stay in front of the model, each with why it was not sent
            told: list[Message] = []
            for s in sent:
                told += [
                    AssistantMessage(role="assistant", content=s["content"]),
                    SystemMessage(role="system", content=s["why"]),
                ]
            reply = self._ask([*state.system_messages, *state.flip_roles(), *told])
            cost = None if cost is None or reply.cost is None else cost + reply.cost
            usage = _summed(usage, reply.usage)
        if not sent:
            return reply
        fixed = reply.is_tool_call() or check_customer(reply.content, given, seen) is None
        note = {"customer_sent_back": sent, "fixed": fixed}
        raw = reply.raw_data if isinstance(reply.raw_data, dict) else {}
        return reply.model_copy(
            update={"cost": cost, "usage": usage, "raw_data": {**raw, "tau2_loop": note}}
        )

    def _given(self) -> tuple[str, ...]:
        """The tools the agent has given this customer, read from the environment through the
        customer's own `list_discoverable_user_tools` (banking's); none in a domain without it."""
        tool = next((t for t in self.tools or [] if t.name == "list_discoverable_user_tools"), None)
        if tool is None:
            return ()
        return tuple(_GIVEN.findall(str(tool())))

    def _seen(self, state: UserState) -> str:
        """Everything this customer has been shown or has said: its prompt, the agent's words, its
        tools' results, its own turns and calls."""
        parts = [self.system_prompt]
        for m in state.messages:
            parts.append(str(getattr(m, "content", None) or ""))
            for c in getattr(m, "tool_calls", None) or []:
                parts.append(json.dumps(c.arguments, default=str))
        return "\n".join(parts)

    def _ask(self, messages: list[Message]) -> UserMessage:
        """One more reply from the customer's model, as tau2's own turn makes it."""
        out = generate(
            model=self.llm,
            messages=messages,
            tools=self.tools,
            call_name="user_simulator_response",
            **self.llm_args,
        )
        calls = [
            ToolCall(id=c.id, name=c.name, arguments=c.arguments, requestor="user")
            for c in out.tool_calls or []
        ]
        return UserMessage(
            role="user",
            content=out.content,
            tool_calls=calls or None,
            cost=out.cost,
            usage=out.usage,
            raw_data=out.raw_data,
        )

    def generate_next_message(
        self, message: Message, state: UserState
    ) -> tuple[UserMessage, UserState]:
        if self.hung_up:
            # the customer said its last words and the agent has had its turn: the call ends here
            bye = UserMessage(role="user", content=STOP)
            state.messages.append(bye)
            return bye, state
        msg, state = super().generate_next_message(message, state)
        words = None if msg.is_tool_call() else held_stop(msg.content)
        if words:
            msg = msg.model_copy(update={"content": words})
            state.messages[-1] = msg
            self.hung_up = True
        return msg, state


def register(domain: str) -> str:
    """Register the domain's customer with tau2 and return the name a run config gives it."""
    from tau2.registry import registry

    name = f"{USER_NAME}_{domain}"
    if name not in registry.get_users():
        cls = type(f"LoopUser_{domain}", (LoopUser,), {"world_time": world_time(domain)})
        registry.register_user(cls, name)
    return name
