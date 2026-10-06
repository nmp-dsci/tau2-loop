"""v6's workflow tools, the workflow library and the live conversation routes (s16). No model is
called: the agent's replies are scripted and workflow_rag's research is faked."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from tau2.data_model.message import AssistantMessage, ToolCall

from tau2_loop.agent import factory
from tau2_loop.agent.versions import load_version
from tau2_loop.data.splits import split_ids
from tau2_loop.eval import live
from tau2_loop.serving.app import create_app
from tau2_loop.workflows import library, rag_agent

D = "banking_knowledge"
CASH = {
    "job": "cash_back_dispute",
    "when": {
        "quote": "Use this process when a customer believes there is a discrepancy between the cash back they received",
        "doc": "d",
    },
    "steps": [{"id": "give", "by": "harness", "do": "give the customer the dispute tool"}],
}
CLOSE = {
    "job": "credit_card_closure",
    "when": {"quote": "Use this when a customer wants to close a credit card", "doc": "d"},
    "steps": [],
}


def session(root: Path, sid: str, jobs: list[dict[str, Any]], **meta: Any) -> None:
    d = root / sid
    d.mkdir(parents=True)
    m = {
        "id": sid,
        "domain": D,
        "agent": "r1",
        "status": "done",
        "source": "question",
        "task_id": "task_019",
        **meta,
    }
    (d / "run.json").write_text(json.dumps(m))
    (d / "output.json").write_text(json.dumps({"jobs": jobs}))


def test_the_library_is_what_r1_wrote_on_train_questions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session(tmp_path, "20261006T010000Z_banking_knowledge_r1_task_019", [CASH])
    session(tmp_path, "20261006T020000Z_banking_knowledge_r1_task_044", [CLOSE])
    # a later session's job replaces an earlier one of the same name
    session(
        tmp_path,
        "20261006T030000Z_banking_knowledge_r1_task_018",
        [{**CASH, "steps": []}],
        task_id="task_018",
    )
    # never in the library: a conversation's request, a failed session, another RAG version
    session(
        tmp_path,
        "20261006T040000Z_banking_knowledge_r1_req_abcdef",
        [{**CASH, "job": "from_a_conversation"}],
        source="conversation",
    )
    session(
        tmp_path,
        "20261006T050000Z_banking_knowledge_r1_task_001",
        [{**CASH, "job": "failed_one"}],
        status="failed",
    )
    session(
        tmp_path,
        "20261006T060000Z_banking_knowledge_r2_task_001",
        [{**CASH, "job": "r2_job"}],
        agent="r2",
    )
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    lib = library.entries(D, "r1")
    assert sorted(e["job"] for e in lib) == ["cash_back_dispute", "credit_card_closure"]
    assert next(e for e in lib if e["job"] == "cash_back_dispute")["task_id"] == "task_018"

    out = library.find(D, "r1", "my cash back on two cards looks wrong")
    assert out.index("cash_back_dispute") < out.index("credit_card_closure")
    assert "Workflow cash_back_dispute" in out and "Workflow credit_card_closure" not in out
    assert "Workflow credit_card_closure" in library.find(D, "r1", "", job="credit_card_closure")
    assert "No job named" in library.find(D, "r1", "", job="nope")
    # sharing "credit card" with the cash-back job is not a match: closing a card is no dispute
    session(
        tmp_path,
        "20261006T070000Z_banking_knowledge_r1_task_020",
        [{**CLOSE, "job": "credit_card_closure"}],
        task_id="task_020",
    )
    only_cash = [e for e in library.entries(D, "r1") if e["job"] == "cash_back_dispute"]
    ranked = library.rank("I want to close my credit card account", only_cash)
    assert ranked[0][1] is False


def test_a_rare_shared_word_outranks_a_common_one() -> None:
    def job(name: str, quote: str) -> dict[str, Any]:
        return {"job": name, "workflow": {"job": name, "when": {"quote": quote}, "steps": []}}

    lib = [
        job(
            "recommend_and_apply_for_personal_credit_card", "Choose a credit card for the customer"
        ),
        job("submitting_a_cash_back_dispute", "A dispute about the cash back on a transaction"),
        job("credit_card_closure", "Close a credit card account"),
        job("credit_limit_increase", "Raise a credit card's limit"),
    ]
    request = "The customer thinks the cash back on two of their credit cards was credited wrong"
    # both share two words with the request, a tie broken by the alphabet before; every job says
    # "credit card" and only one says "cash back", so the rarer words win
    assert library.rank(request, lib)[0][2]["job"] == "submitting_a_cash_back_dispute"


def test_an_empty_library_points_to_request_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    assert "request_workflow" in library.find(D, "r1", "anything")


def test_only_a_version_that_names_workflow_rag_gets_the_two_tools() -> None:
    v6 = factory.LoopAgent(tools=[], domain_policy="p", version=load_version(D, "v6"))
    v4 = factory.LoopAgent(tools=[], domain_policy="p", version=load_version(D, "v4"))
    assert [t.name for t in v6.workflow_tools] == ["find_workflow", "request_workflow"]
    assert v4.workflow_tools == [] and load_version(D, "v4").config.workflows is None


def test_v6_runs_a_lookup_inside_its_turn_and_tau2_never_sees_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = factory.LoopAgent(tools=[], domain_policy="p", version=load_version(D, "v6"))
    monkeypatch.setattr(library, "find", lambda d, r, req, job="": f"found for {req}")
    streamed: list[dict[str, Any]] = []
    agent.live = streamed.append
    first = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="a", name="find_workflow", arguments={"request": "cash back looks wrong"}),
            ToolCall(id="b", name="get_current_time", arguments={}),
        ],
    )
    replies = [AssistantMessage(role="assistant", content="Could you give me your name?")]
    asked: list[list[Any]] = []

    def ask(messages: list[Any], workflow: bool = True) -> AssistantMessage:
        asked.append(messages)
        return replies.pop(0)

    out, extra, used = agent._workflow_rounds([], first, ask)
    # what goes back to tau2 is the reply after the lookup, with no workflow call in it
    assert out.content == "Could you give me your name?" and not out.tool_calls
    assert [m.role for m in extra] == ["assistant", "tool", "tool"]
    assert extra[1].content == "found for cash back looks wrong"
    assert extra[2].content.startswith("NOT EXECUTED")  # the other call waits for the next reply
    assert asked[0] == extra and used[0]["name"] == "find_workflow"
    assert [e["kind"] for e in streamed] == ["workflow_call", "workflow_result"]


def test_a_request_mid_conversation_is_researched_and_never_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_rag_agent import WORKFLOW, FakeEnv, Reply

    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    started: list[str] = []
    sent: list[list[dict[str, Any]]] = []

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        sent.append(messages)
        return Reply(json.dumps(WORKFLOW))

    meta = rag_agent.start_request(
        D,
        "The customer thinks cash back on two cards was credited wrong",
        conversation="20261006T080838Z_banking_knowledge_v6_task_019",
        on_start=lambda m: started.append(m.id),
        ask=ask,
        run_tool=lambda n, a: "",
        env=FakeEnv(),
    )
    assert started == [meta.id] and "_req_" in meta.id and meta.status == "done"
    run = json.loads((tmp_path / meta.id / "run.json").read_text())
    assert run["source"] == "conversation" and run["conversation"].endswith("task_019")
    # it reads the agent's words, introduced as a live request, never a task's script
    assert sent[0][1]["content"].startswith("An answering agent is talking to a customer")
    assert library.entries(D, "r1") == []


def test_live_plays_only_train_tasks() -> None:
    with pytest.raises(live.LiveError, match="sealed"):
        live.start(D, "v6", split_ids(D, "test")[0])
    with pytest.raises(live.LiveError, match="no live conversation"):
        live.conversation("not-an-id")


def test_the_live_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    c = TestClient(create_app())
    body = c.get(f"/api/live/{D}/v6").json()
    assert len(body["tasks"]) == len(split_ids(D, "train"))
    assert not {t["id"] for t in body["tasks"]} & set(split_ids(D, "test"))
    # the conversation route is not read as a domain called "conversations"
    r = c.get("/api/live/conversations/20990101T000000Z_banking_knowledge_v6_task_019")
    assert r.status_code == 404 and "no live conversation" in r.json()["detail"]
    r = c.post(f"/api/live/{D}/v6", json={"task_id": split_ids(D, "test")[0]})
    assert r.status_code == 422
    monkeypatch.setenv("DEMO_MODE", "1")
    assert (
        TestClient(create_app()).post(f"/api/live/{D}/v6", json={"task_id": "task_019"}).status_code
        == 503
    )
