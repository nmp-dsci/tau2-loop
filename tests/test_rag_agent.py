"""workflow_rag's RAG agent (s16): the live demo's session loop, its guards and its score.

No model is called: each test drives the loop with a scripted model and scripted tool results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tau2_loop.data.splits import split_ids
from tau2_loop.serving.app import create_app
from tau2_loop.workflows import rag_agent

D = "banking_knowledge"
CASH_BACK_DOC = "doc_credit_cards_credit_cards_(general)_003"
WHEN = (
    "Use this process when a customer believes there is a discrepancy between the cash back they "
    "received for a particular transaction and the cash back they should have received."
)


@dataclass
class Reply:
    content: str | None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    result: Any = None


@dataclass
class FakeTool:
    name: str

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.name} tool",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }


class FakeEnv:
    policy = "# Policy\nVerify the customer first."

    def get_tools(self) -> list[FakeTool]:
        names = [
            "KB_search_bm25",
            "KB_search_dense",
            "shell",
            "log_verification",
            "give_discoverable_user_tool",
        ]
        return [FakeTool(n) for n in names]

    def get_user_tools(self) -> list[FakeTool]:
        return [FakeTool("call_discoverable_user_tool")]


WORKFLOW = {
    "jobs": [
        {
            "job": "cash_back_dispute",
            "when": {"quote": WHEN, "doc": CASH_BACK_DOC},
            "info": [],
            "steps": [
                {"id": "verify", "by": "harness", "call": "log_verification"},
                {
                    "id": "give",
                    "by": "harness",
                    "call": "give_discoverable_user_tool",
                    "quote": "Tool to provide to the user",
                    "doc": "doc_credit_cards_credit_cards__general__003.md",
                },
                {"id": "submit", "by": "customer", "call": "submit_cash_back_dispute_0589"},
            ],
            "rules": [
                {
                    "rule": "made up",
                    "quote": "Cash back is tripled on Tuesdays.",
                    "doc": CASH_BACK_DOC,
                }
            ],
            "done_when": "every dispute submitted",
        }
    ],
    "documents": [{"id": CASH_BACK_DOC, "why": "the procedure"}],
    "open_questions": [],
}


def scripted(replies: list[Reply]) -> tuple[rag_agent.Ask, list[list[dict[str, Any]] | None]]:
    seen_tools: list[list[dict[str, Any]] | None] = []

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        seen_tools.append(tools)
        return replies.pop(0)

    return ask, seen_tools


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"id": f"c_{name}_{len(args)}", "name": name, "arguments": args}


def test_a_test_question_is_refused_before_anything_runs() -> None:
    test_id = split_ids(D, "test")[0]
    with pytest.raises(rag_agent.RagAgentError, match="sealed"):
        rag_agent.train_question(D, test_id)
    assert "Priya Sharma" in rag_agent.train_question(D, "task_019")


@pytest.mark.parametrize(
    "command",
    ["cat ../tasks/task_001.json", "ls /", "cat /etc/hosts", "grep -ril task .", "ls ~"],
)
def test_the_shell_may_not_reach_for_the_task_files(command: str) -> None:
    assert rag_agent.shell_refusal(command)


@pytest.mark.parametrize(
    "command",
    [
        "grep -il 'cash back' *.md | head",
        "cat doc_credit_cards_gold_rewards_card_001.md",
        "awk '{print $1}' INDEX.txt",
        "ls | wc -l",
    ],
)
def test_ordinary_searches_of_the_documents_pass(command: str) -> None:
    assert rag_agent.shell_refusal(command) is None


def test_the_workflow_is_read_from_a_fenced_or_wrapped_reply() -> None:
    body = json.dumps(WORKFLOW)
    assert rag_agent.parse_workflow(body) == WORKFLOW
    assert rag_agent.parse_workflow(f"```json\n{body}\n```") == WORKFLOW
    assert rag_agent.parse_workflow(f"Here it is:\n{body}\nDone.") == WORKFLOW
    assert rag_agent.parse_workflow('{"steps": []}') is None
    assert rag_agent.parse_workflow("no json") is None


def test_a_quote_counts_only_when_its_document_holds_it() -> None:
    q = rag_agent.quote_check(D, WORKFLOW)
    by_quote = {r["quote"]: r for r in q["rows"]}
    assert by_quote[WHEN]["found"] is True
    # the shell's file name for a document resolves to its id
    assert by_quote["Tool to provide to the user"]["doc_id"] == CASH_BACK_DOC
    assert by_quote["Tool to provide to the user"]["found"] is True
    assert by_quote["Cash back is tripled on Tuesdays."]["found"] is False
    assert (q["n"], q["found"]) == (3, 2)


def test_a_quote_from_the_policy_is_looked_up_in_the_policy() -> None:
    wf = {
        "jobs": [
            {
                "job": "x",
                "steps": [
                    {
                        "id": "v",
                        "by": "model",
                        "quote": "Verify the customer first.",
                        "doc": "policy",
                    }
                ],
            }
        ]
    }
    assert rag_agent.quote_check(D, wf, FakeEnv.policy)["found"] == 1
    assert rag_agent.quote_check(D, wf, "")["found"] == 0


def test_a_session_runs_the_tools_then_scores_the_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    ask, offered = scripted(
        [
            Reply(
                None,
                [
                    call("KB_search_bm25", query="cash back dispute"),
                    call("transfer_to_human_agents"),
                ],
            ),
            Reply(None, [call("shell", command="cat ../tasks/task_019.json")]),
            Reply("Let me think about it."),  # not JSON: asked once more
            Reply(json.dumps(WORKFLOW)),
        ]
    )
    ran: list[tuple[str, dict[str, Any]]] = []

    def run_tool(name: str, args: dict[str, Any]) -> str:
        ran.append((name, args))
        return f"1. Submitting a Cash Back Dispute (Internal)\n   ID: {CASH_BACK_DOC}\n"

    meta = rag_agent.start(
        D, "task_019", ask=ask, run_tool=run_tool, env=FakeEnv(), background=False
    )
    assert meta.status == "done" and meta.steps == 4 and meta.tool_calls == 3
    # only the knowledge tools are offered, and only an allowed call reaches the environment
    assert {t["function"]["name"] for t in offered[0] or []} == {
        "KB_search_bm25",
        "KB_search_dense",
        "shell",
    }
    assert ran == [("KB_search_bm25", {"query": "cash back dispute"})]

    got = rag_agent.session(D, meta.id)
    tools = [e for e in got["events"] if e["kind"] == "tool"]
    assert [t["refused"] for t in tools] == [False, True, True]
    assert any(e["kind"] == "note" for e in got["events"])
    assert got["output"] == WORKFLOW
    sc = got["score"]
    # gold is read only by the score: task_019's six required documents and its tools
    assert sc["documents_referenced"]["required"] == 6
    assert sc["documents_referenced"]["hit"] == [CASH_BACK_DOC]
    assert sc["documents_seen"]["hit"] == [CASH_BACK_DOC]
    assert set(sc["tools"]["hit"]) == {"log_verification", "submit_cash_back_dispute_0589"}
    assert sc["golden"]["workflows"] == ["cash_back_dispute"]
    assert sc["jobs"] == ["cash_back_dispute"]
    # the prompt the session ran with is kept beside it, and names no gold
    system = (tmp_path / meta.id / "system.md").read_text()
    assert "Verify the customer first." in system and "evaluation_criteria" not in system


def test_a_session_out_of_steps_is_told_to_write_with_no_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    agent = rag_agent.load(D, "r1")
    replies = [
        Reply(None, [call("KB_search_dense", query=f"q{i}")]) for i in range(agent.max_steps)
    ]
    ask, offered = scripted([*replies, Reply(json.dumps(WORKFLOW))])
    meta = rag_agent.start(
        D, "task_019", ask=ask, run_tool=lambda n, a: "nothing", env=FakeEnv(), background=False
    )
    assert meta.status == "done" and meta.steps == agent.max_steps + 1
    assert offered[-1] is None and offered[0] is not None


def test_the_api_refuses_a_test_question_and_lists_r1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEMO_MODE", raising=False)
    c = TestClient(create_app())
    r = c.get(f"/api/workflows/{D}/rag")
    assert r.status_code == 200 and r.json()["versions"][0]["name"] == "r1"
    assert c.get("/api/workflows/airline/rag").status_code == 404
    test_id = split_ids(D, "test")[0]
    r = c.post(f"/api/workflows/{D}/rag/sessions", json={"task_id": test_id})
    assert r.status_code == 422 and "sealed" in r.json()["detail"]


def test_the_demo_image_starts_no_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMO_MODE", "1")
    c = TestClient(create_app())
    assert (
        c.post(f"/api/workflows/{D}/rag/sessions", json={"task_id": "task_019"}).status_code == 503
    )


# ── s20: r2 merges each question into the library r1 left ──
OLD = {
    "job": "cash_back_dispute",
    "when": {"quote": WHEN, "doc": CASH_BACK_DOC},
    "info": [],
    "steps": [
        {"id": "verify", "by": "harness", "call": "log_verification"},
        {
            "id": "give",
            "by": "harness",
            "call": "give_discoverable_user_tool",
            "if": "a discrepancy",
        },
    ],
    "rules": [],
    "done_when": "every dispute submitted",
}
DRAFT = {"jobs": [{**OLD, "job": "submitting_a_cash_back_dispute", "steps": OLD["steps"][:1]}]}


def seed_r1(root: Path) -> None:
    d = root / "20261006T010000Z_banking_knowledge_r1_task_018"
    d.mkdir(parents=True)
    meta = {"id": d.name, "domain": D, "agent": "r1", "status": "done", "source": "question"}
    (d / "run.json").write_text(json.dumps({**meta, "task_id": "task_018"}))
    (d / "output.json").write_text(json.dumps({"jobs": [OLD]}))


def merged(steps: list[dict[str, Any]], changelog: list[dict[str, Any]] | None = None) -> str:
    job = {
        **OLD,
        "into": ["cash_back_dispute"],
        "aliases": ["submitting_a_cash_back_dispute"],
        "steps": steps,
        "changelog": changelog or [],
    }
    return json.dumps({"jobs": [job], "documents": [], "open_questions": []})


def test_r2_merges_a_question_into_the_job_the_library_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.workflows import library

    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    seed_r1(tmp_path)
    assert [e["job"] for e in library.entries(D, "r2")] == ["cash_back_dispute"]  # r1's, seeded
    branch = {"id": "explain", "by": "model", "if": "no discrepancy", "do": "explain the rate"}
    ask, _ = scripted(
        [
            Reply(json.dumps(DRAFT)),  # the draft, under another name
            Reply(merged([*OLD["steps"], branch], [{"change": "added", "what": "explain"}])),
        ]
    )
    meta = rag_agent.start(
        D, "task_019", "r2", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    assert meta.status == "done" and meta.steps == 2
    out = rag_agent.session(D, meta.id)
    # the merge message offered r1's job as the drafted job's match, in full
    merge_ev = next(e for e in out["events"] if e["kind"] == "merge")
    assert merge_ev["candidates"] == {"submitting_a_cash_back_dispute": ["cash_back_dispute"]}
    rec = out["output"]["merge"][0]
    assert rec["taken"] and rec["into"] == ["cash_back_dispute"] and not rec["failed"]
    assert rec["changelog"] == [{"change": "added", "what": "explain"}]
    job = out["output"]["jobs"][0]
    assert "changelog" not in job and "into" not in job
    assert job["aliases"] == ["submitting_a_cash_back_dispute"]
    # the library's next version contains the old one; the draft's name now finds it
    lib = library.entries(D, "r2")
    assert [e["job"] for e in lib] == ["cash_back_dispute"] and lib[0]["session"] == meta.id
    assert [s["id"] for s in lib[0]["workflow"]["steps"]] == ["verify", "give", "explain"]
    assert library.by_name(lib, "submitting_a_cash_back_dispute") is lib[0]
    assert "Workflow cash_back_dispute" in library.find(
        D, "r2", "", job="submitting_a_cash_back_dispute"
    )
    # r1's own library is untouched
    assert [s["id"] for s in library.entries(D, "r1")[0]["workflow"]["steps"]] == ["verify", "give"]


def test_a_merge_that_drops_a_step_is_sent_back_then_the_library_keeps_its_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tau2_loop.workflows import library

    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    seed_r1(tmp_path)
    dropped = merged(OLD["steps"][:1])  # "give" gone, and no changelog says why
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(dropped), Reply(dropped)])
    seen: list[list[dict[str, Any]]] = []

    def watch(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        seen.append(list(messages))
        return ask(messages, tools, model, effort, tool_mode)

    meta = rag_agent.start(
        D, "task_019", "r2", ask=watch, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    assert meta.status == "done" and meta.steps == 3
    sent_back = seen[-1][-1]["content"]
    assert "lost steps the changelog does not name: give" in sent_back
    # no gold in anything the model was shown
    assert all("evaluation_criteria" not in json.dumps(m) for m in seen[-1])
    out = rag_agent.session(D, meta.id)["output"]
    assert out["jobs"] == [] and out["rejected"][0]["job"] == "cash_back_dispute"
    assert out["merge"][0]["taken"] is False
    lib = library.entries(D, "r2")
    assert [s["id"] for s in lib[0]["workflow"]["steps"]] == ["verify", "give"]  # r1's version


def test_a_removal_the_changelog_names_passes_the_gate() -> None:
    from tau2_loop.workflows import rubric

    job = {
        **OLD,
        "steps": OLD["steps"][:1],
        "changelog": [{"change": "removed", "what": "give", "why": "the doc says so"}],
    }
    assert rubric.lost(job, [OLD]) == {"steps": [], "tools": [], "quotes": []}
    assert rubric.lost({**job, "changelog": []}, [OLD])["steps"] == ["give"]


def test_the_build_queue_is_train_in_id_order_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    train = split_ids(D, "train")
    q = rag_agent.build_queue(D, "r2")
    assert sorted(q) == sorted(train) and q == sorted(q, key=lambda t: int(t.split("_")[1]))
    d = tmp_path / f"20261007T010000Z_banking_knowledge_r2_{q[0]}"
    d.mkdir()
    meta = {"id": d.name, "domain": D, "agent": "r2", "status": "done", "source": "question"}
    (d / "run.json").write_text(json.dumps({**meta, "task_id": q[0]}))
    assert rag_agent.build_queue(D, "r2")[0] == q[1]  # a finished question is not redone
    with pytest.raises(rag_agent.RagAgentError, match="sealed"):
        rag_agent.build_queue(D, "r2", [split_ids(D, "test")[0]])


def test_a_conversation_request_is_never_merged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    seed_r1(tmp_path)
    ask, _ = scripted([Reply(json.dumps(DRAFT))])
    meta = rag_agent.start_request(
        D, "my cash back looks wrong", "r2", ask=ask, run_tool=lambda n, a: "", env=FakeEnv()
    )
    out = rag_agent.session(D, meta.id)
    assert meta.status == "done" and meta.steps == 1 and "merge" not in out["output"]
