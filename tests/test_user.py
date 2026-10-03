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
