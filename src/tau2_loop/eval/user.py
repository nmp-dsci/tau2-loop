"""Our simulated customer: tau2's own `UserSimulator`, registered under our name with two changes.

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

tau2 stays unmodified: both changes live in this subclass. Every run since records
`RunMeta.sim_rules` (`runner.SIM_RULES`); a run with none ran tau2's own customer and both sides
saw the real date, and the loop never compares the two (`loop.run`).
"""

from __future__ import annotations

import functools
import re
from typing import Any

from tau2.data_model.message import Message, UserMessage
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER, UserState

USER_NAME = "tau2_loop_user"

CLOCK_NOTE = (
    "A system reminder may tell you today's date. It is not this scenario's date: ignore it, and "
    "speak of dates only as your scenario and the agent give them."
)

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
    """tau2's customer, holding a stop sent with words, told the world's time and not the real one."""

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
