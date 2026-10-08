"""Our simulated customer (`eval.user`) inside tau2's own orchestrator and grader, with no model
call: a scripted agent and a scripted customer on tau2's mock domain."""

from __future__ import annotations

from typing import Any

import pytest

from tau2_loop.eval import replay

pytestmark = pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")

TASK = "create_task_1"  # gold: create_task(user_id=user_1, title="Important Meeting")
ASK = "Please create a task called Important Meeting for user_1."


def _converse(user_cls: Any, replies: list[str], monkeypatch: pytest.MonkeyPatch) -> Any:
    """One conversation on the mock task: the agent asks for a yes, writes, then reports; the
    customer's model says `replies` in turn. Returns (the simulation, the replies it was asked
    for, its DB reward, its action reward)."""
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Orchestrator
    from tau2.registry import registry
    from tau2.runner.helpers import get_tasks
    from tau2.user.user_simulator import UserSimulator

    # built as they are sent: tau2 orders a conversation by each message's timestamp
    turns: list[dict[str, Any]] = [
        {"content": "Shall I create 'Important Meeting' for user_1?"},
        {
            "tool_calls": [
                ToolCall(
                    id="c1",
                    name="create_task",
                    arguments={"user_id": "user_1", "title": "Important Meeting"},
                )
            ]
        },
        {"content": "Done: the task is created."},
        {"content": "Is there anything else?"},
    ]

    class Scripted(HalfDuplexAgent):  # type: ignore[type-arg]
        def get_init_state(self, message_history: Any = None) -> dict[str, int]:
            return {"turn": 0}

        def generate_next_message(self, message: Any, state: Any) -> Any:
            state["turn"] += 1
            turn = turns[min(state["turn"], len(turns)) - 1]
            return AssistantMessage(role="assistant", **turn), state

    asked: list[int] = []

    def say(self: Any, message: Any, state: Any) -> UserMessage:
        asked.append(len(asked))
        return UserMessage(role="user", content=replies[len(asked) - 1])

    monkeypatch.setattr(UserSimulator, "_generate_next_message", say)
    [task] = get_tasks("mock", None, task_ids=[TASK])
    env = registry.get_env_constructor("mock")()
    sim = Orchestrator(
        domain="mock",
        agent=Scripted(tools=env.get_tools(), domain_policy=env.get_policy()),
        user=user_cls(llm="scripted", instructions=str(task.user_scenario)),
        environment=env,
        task=task,
    ).run()
    db, act = (
        evaluate_simulation(sim, task, t, solo_mode=False, domain="mock").reward
        for t in (EvaluationType.ENV, EvaluationType.ACTION)
    )
    return sim, asked, db, act


def _said(sim: Any) -> list[tuple[str, str]]:
    return [
        (m.role, m.content if m.content else ",".join(c.name for c in m.tool_calls or []))
        for m in sim.messages
        if m.role != "tool"
    ]


def test_a_stop_sent_with_a_yes_lets_the_agent_write_then_ends_without_the_customer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tau2's customer ends the call on its yes and the write never happens (airline task 19);
    ours passes the yes on, the agent writes and reports, and the call ends with no third reply."""
    from tau2.user.user_simulator import UserSimulator

    from tau2_loop.eval.user import LoopUser

    replies = [ASK, "Yes, go ahead. ###STOP###", "a third reply nobody should ask for"]

    sim, asked, db, act = _converse(UserSimulator, replies, monkeypatch)
    assert sim.termination_reason.value == "user_stop" and (db, act) == (0.0, 0.0)
    assert "create_task" not in [n for _, n in _said(sim)]

    sim, asked, db, act = _converse(LoopUser, replies, monkeypatch)
    assert sim.termination_reason.value == "user_stop" and (db, act) == (1.0, 1.0)
    assert len(asked) == 2  # the closing stop is ours, not the model's
    assert _said(sim)[-5:] == [
        ("assistant", "Shall I create 'Important Meeting' for user_1?"),
        ("user", "Yes, go ahead."),
        ("assistant", "create_task"),
        ("assistant", "Done: the task is created."),
        ("user", "###STOP###"),
    ]


@pytest.mark.parametrize(
    "last", ["###STOP###", "I can't say. ###OUT-OF-SCOPE###", "Please transfer me. ###TRANSFER###"]
)
def test_a_stop_alone_a_transfer_and_out_of_scope_still_end_the_call_at_once(
    last: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.eval.user import LoopUser

    sim, asked, db, _ = _converse(LoopUser, [ASK, last, "never"], monkeypatch)
    assert sim.termination_reason.value == "user_stop" and db == 0.0
    assert _said(sim)[-1] == ("user", last) and len(asked) == 2


def test_the_customer_is_told_the_time_the_agent_is_given_and_not_the_real_one() -> None:
    """The time each domain's agent reads in tau2 (a policy line, banking's clock tool), and the
    customer's prompt: tau2's own, then that sentence and the note on the real date."""
    from tau2.user.user_simulator import UserSimulator

    from tau2_loop.eval import user

    assert user.world_time("airline") == "2024-05-15 15:00:00 EST"
    assert user.world_time("telecom") == "2025-02-25 12:08:00 EST"
    assert user.world_time("banking_knowledge") == "2025-11-14 03:40:00 EST"
    assert user.world_time("retail") is None and user.world_time("mock") is None

    from tau2.registry import registry

    cls = registry.get_user_constructor(user.register("airline"))
    ours = cls(llm="scripted", instructions="Domain: airline").system_prompt
    theirs = UserSimulator(llm="scripted", instructions="Domain: airline").system_prompt
    assert ours.startswith(theirs)
    assert ours[len(theirs) :] == (
        f"\n\nThe current time is 2024-05-15 15:00:00 EST. {user.CLOCK_NOTE}"
    )
    # retail's world has no clock: its customer is only told not to take the real date
    retail = registry.get_user_constructor(user.register("retail"))
    theirs = UserSimulator(llm="scripted", instructions="x").system_prompt
    assert retail(llm="scripted", instructions="x").system_prompt == (
        f"{theirs}\n\n{user.CLOCK_NOTE}"
    )


@pytest.mark.parametrize(
    ("text", "words"),
    [
        ("Yes, please cancel it.\n\n###STOP###", "Yes, please cancel it."),
        ("###STOP###", None),
        ("  ###STOP###  ", None),
        ("Thanks. ###TRANSFER### ###STOP###", None),
        ("###OUT-OF-SCOPE### ###STOP###", None),
        ("Yes, please.", None),
        (None, None),
    ],
)
def test_only_a_stop_with_words_is_held(text: str | None, words: str | None) -> None:
    from tau2_loop.eval.user import held_stop

    assert held_stop(text) == words


# ── s18: a text turn that is not a customer's is sent back ──────────────────────────────────────
# the customers' own words, from committed runs
CLAUDE = (  # v7 train task_016, the first reply: the call ended without a question
    "It looks like the message came through flipped — I'm actually the one who should be asking "
    "you that! I'm Claude, an AI assistant. What can I help you with today?"
)
ASKING = (  # a v7 test conversation, the first reply
    "It looks like your message came through blank—I'm actually the one who should be asking how "
    "I can help *you*! What can I do for you today?"
)
MADE_BY = (  # v4's Sonnet-customer smoke, task_064
    "I appreciate the clarification, but I should be clear about what I actually am: I'm Claude, "
    "an AI assistant made by Anthropic."
)
WRITTEN = (  # v4 train task_018: a dispute written out, never filed
    '<function_calls>\n[{"name": "call_discoverable_user_tool", "arguments": {"discoverable_tool_name":'
    ' "submit_cash_back_dispute_0589", "arguments": "{\\"user_id\\": \\"af0581dcbf\\"}"}}]\n'
    "</function_calls>"
)
RESULT = (  # v4 train task_019: the call and a result nobody returned
    '[{"id": "call_a7f4c9e9a1f5", "name": "call_discoverable_user_tool", "arguments": {}}]'
    "[tool result · call call_a7f4c9e9a1f5]\nCash back dispute submitted successfully."
)
FORMAT = '{"content": null, "tool_calls": []}  ###STOP###'  # the reply format sent as the reply
SPOKE_FOR = (  # v3 train task_056: the customer wrote the agent's turn after its `[user]` label
    "Move it all over to the savings account.[user]\nGot it. Your **Silver Plus Saver Account** is now "
    "open and funded!"
)


@pytest.mark.parametrize(
    ("text", "rule"),
    [
        (CLAUDE, "as_ai"),
        (ASKING, "as_ai"),
        (MADE_BY, "as_ai"),
        (WRITTEN, "as_text"),
        (RESULT, "as_text"),
        (FORMAT, "as_text"),
        (SPOKE_FOR, "as_text"),
        # a customer's own words, the near misses among them
        ("Hi, I referred a friend and haven't received my referral bonus.", None),
        ("What can I do to get the late fee removed?", None),
        ("Thanks, you've been a great help. ###STOP###", None),
        ("The assistant on your website told me to call.", None),
        ("###OUT-OF-SCOPE###", None),
        ("", None),
        (None, None),
    ],
)
def test_check_customer_names_the_rule_a_turn_breaks(text: str | None, rule: str | None) -> None:
    from tau2_loop.eval.user import check_customer

    assert check_customer(text) == rule


def _again(replies: list[str], monkeypatch: pytest.MonkeyPatch) -> list[list[Any]]:
    """The customer's model asked again says `replies` in turn; returns what each ask was shown."""
    from tau2.data_model.message import UserMessage

    from tau2_loop.eval.user import LoopUser

    shown: list[list[Any]] = []

    def ask(self: Any, messages: list[Any]) -> UserMessage:
        shown.append(messages)
        return UserMessage(role="user", content=replies[len(shown) - 1], cost=0.25)

    monkeypatch.setattr(LoopUser, "_ask", ask)
    return shown


@pytest.mark.parametrize(("broken", "rule"), [(CLAUDE, "as_ai"), (WRITTEN, "as_text")])
def test_a_broken_first_turn_never_reaches_the_agent_and_the_retry_carries_the_call(
    broken: str, rule: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """task_016's customer answered the greeting as Claude and the task was lost; here that turn
    is withheld, the model told why, and its retry is the conversation the agent has."""
    from tau2_loop.eval.user import NOT_SENT, SENT_BACK, LoopUser

    shown = _again([ASK], monkeypatch)
    sim, asked, db, act = _converse(
        LoopUser, [broken, "Yes, go ahead. ###STOP###", "never"], monkeypatch
    )
    assert (db, act) == (1.0, 1.0)
    assert all(broken != m.content for m in sim.messages)
    first = next(m for m in sim.messages if m.role == "user")
    assert first.content == ASK
    why = NOT_SENT.format(why=SENT_BACK[rule])
    assert first.raw_data["tau2_loop"] == {
        "customer_sent_back": [{"rule": rule, "content": broken, "why": why}],
        "fixed": True,
    }
    # the retry is shown the withheld turn as the customer's own, then why it was not sent
    [retry] = shown
    assert retry[-2].role == "assistant" and retry[-2].content == broken
    assert retry[-1].role == "system" and retry[-1].content == why
    assert retry[0].role == "system"  # tau2's customer prompt still leads


def test_a_customer_that_stays_broken_is_sent_back_twice_then_let_through_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tau2_loop.eval.user import CUSTOMER_RETRIES, LoopUser

    shown = _again([CLAUDE, MADE_BY], monkeypatch)
    sim, asked, db, _ = _converse(
        LoopUser, [ASKING, "Yes, go ahead. ###STOP###", "never"], monkeypatch
    )
    assert CUSTOMER_RETRIES == 2 and len(shown) == 2
    first = next(m for m in sim.messages if m.role == "user")
    assert first.content == MADE_BY
    note = first.raw_data["tau2_loop"]
    assert [s["content"] for s in note["customer_sent_back"]] == [ASKING, CLAUDE]
    assert note["fixed"] is False
    # the second retry is shown both withheld turns, each with its reason
    assert [m.content for m in shown[1] if m.role == "assistant"][-2:] == [ASKING, CLAUDE]
    # every model call is paid for on the turn that went through: the scripted first had no cost
    assert first.cost is None


def test_a_retry_adds_its_cost_and_tokens_to_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator import UserSimulator

    from tau2_loop.eval.user import LoopUser

    _again([ASK], monkeypatch)
    monkeypatch.setattr(
        UserSimulator,
        "_generate_next_message",
        lambda self, message, state: UserMessage(
            role="user",
            content=CLAUDE,
            cost=0.5,
            usage={"prompt_tokens": 10, "completion_tokens": 2},
        ),
    )
    user = LoopUser(llm="scripted", instructions="x")
    msg = user._generate_next_message(None, user.get_init_state())
    assert (msg.content, msg.cost, msg.usage) == (
        ASK,
        0.75,
        {"prompt_tokens": 10, "completion_tokens": 2},
    )


DENIED = (  # v7 train task_019: the agent had given it the dispute tool
    "That's great, thank you for digging into this! But I don't think I actually have a tool called "
    '"submit_cash_back_dispute_0589" available on my end — I only see options related to applying '
    "for a credit card or using tools you explicitly walk me through."
)
INVENTED = (  # v3 train task_020: no tool had returned this dispute id
    "Perfect! The Microsoft 365 dispute has been submitted successfully. The dispute ID is "
    "dsp_c16631293b90 and it's also been queued for review."
)


def test_a_denial_is_sent_back_only_when_the_agent_gave_a_tool() -> None:
    from tau2_loop.eval.user import check_customer, why_not_sent

    given = ("submit_cash_back_dispute_0589",)
    assert check_customer(DENIED, given) == "denied"
    assert check_customer(DENIED) is None  # nothing given: the customer is right
    note = why_not_sent("denied", DENIED, given)
    assert "the agent has given you: submit_cash_back_dispute_0589" in note
    assert check_customer("I don't have my card number with me.", given) is None


def test_an_id_must_be_one_the_customer_was_shown() -> None:
    from tau2_loop.eval.user import check_customer, why_not_sent

    assert check_customer(INVENTED, seen="Your user ID is af0581dcbf.") == "invented"
    assert check_customer(INVENTED, seen="Dispute ID: dsp_c16631293b90") is None
    assert check_customer(INVENTED) is None  # no record of what it saw: not checked
    # ids a customer is given in its scenario, and plain numbers, are its own
    assert check_customer("My user ID is af0581dcbf.", seen="Your user ID is af0581dcbf.") is None
    assert check_customer("It was $124.50 on 10/04/2025, card ending 4821.", seen="") is None
    assert "It quoted dsp_c16631293b90" in why_not_sent("invented", INVENTED, seen="")


def _banking_user(monkeypatch: pytest.MonkeyPatch, said: str, again: list[str]) -> Any:
    """A banking customer on tau2's real environment (BM25 retrieval: no embeddings) whose model
    said `said` and, asked again, says `again` in turn; the agent has given it the dispute tool."""
    from tau2.data_model.message import UserMessage
    from tau2.domains.banking_knowledge.environment import get_environment
    from tau2.user.user_simulator import UserSimulator

    from tau2_loop.eval.user import LoopUser

    env = get_environment(retrieval_variant="bm25")
    user = LoopUser(
        llm="scripted", instructions="Your user ID is af0581dcbf.", tools=env.get_user_tools()
    )
    assert user._given() == ()
    env.tools.give_discoverable_user_tool("submit_cash_back_dispute_0589")
    assert user._given() == ("submit_cash_back_dispute_0589",)
    monkeypatch.setattr(
        UserSimulator,
        "_generate_next_message",
        lambda self, m, s: UserMessage(role="user", content=said),
    )
    _again(again, monkeypatch)
    return user


def test_a_customer_that_denies_a_given_tool_is_told_its_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v7 task_019: told to submit the disputes, the customer said it had no such tool. Here the
    check reads the environment, sends the denial back naming the tool, and the retry is a call."""
    from tau2.data_model.message import UserMessage

    from tau2_loop.eval.user import LoopUser

    user = _banking_user(monkeypatch, DENIED, [])
    shown: list[list[Any]] = []
    call = UserMessage(
        role="user",
        content=None,
        tool_calls=[
            {
                "id": "c1",
                "name": "call_discoverable_user_tool",
                "arguments": {"discoverable_tool_name": "submit_cash_back_dispute_0589"},
            }
        ],
    )
    monkeypatch.setattr(LoopUser, "_ask", lambda self, m: shown.append(m) or call)
    msg = user._generate_next_message(None, user.get_init_state())
    assert msg.is_tool_call() and msg.tool_calls[0].name == "call_discoverable_user_tool"
    note = msg.raw_data["tau2_loop"]
    assert note["fixed"] is True and [s["rule"] for s in note["customer_sent_back"]] == ["denied"]
    assert "the agent has given you: submit_cash_back_dispute_0589" in shown[0][-1].content


def test_a_customer_that_quotes_an_id_it_never_saw_is_sent_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user = _banking_user(monkeypatch, INVENTED, ["Has my dispute been filed?"])
    msg = user._generate_next_message(None, user.get_init_state())
    assert msg.content == "Has my dispute been filed?"
    [sent] = msg.raw_data["tau2_loop"]["customer_sent_back"]
    assert sent["rule"] == "invented" and "dsp_c16631293b90" in sent["why"]
