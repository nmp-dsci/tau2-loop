"""workflow_rag's RAG agent (s16): the live demo's session loop, its guards and its score.

No model is called: each test drives the loop with a scripted model and scripted tool results.
"""

from __future__ import annotations

import json
import threading
import time
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


def test_a_session_another_process_runs_shows_running_until_it_goes_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os
    import time

    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", tmp_path)
    d = tmp_path / "20261007T010000Z_banking_knowledge_r2_task_041"
    d.mkdir()
    meta = {"id": d.name, "domain": D, "agent": "r2", "status": "running", "task_id": "task_041"}
    (d / "run.json").write_text(json.dumps(meta))
    (d / "events.jsonl").write_text("{}\n")
    assert rag_agent.sessions(D)[0]["status"] == "running"  # the build is writing it
    old = time.time() - rag_agent.STALE_S - 60
    for f in ("run.json", "events.jsonl"):
        os.utime(d / f, (old, old))
    assert rag_agent.sessions(D)[0]["status"] == "interrupted"


# ── s21: r3 builds the library with several workers; each locks the jobs it writes ──
DECIDED = json.dumps({"decision": {"submitting_a_cash_back_dispute": ["cash_back_dispute"]}})
BRANCH_A = {"id": "explain_a", "by": "model", "if": "no discrepancy", "do": "explain the rate"}
BRANCH_B = {"id": "refund_b", "by": "model", "if": "a posted refund", "do": "explain the reversal"}


@pytest.fixture
def r3_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tau2_loop.workflows import store

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", runs)
    monkeypatch.setattr(store, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(store, "POLL_S", 0.02)
    seed_r1(runs)  # r2 has merged nothing here, so r3 starts from r1's one job
    return runs


def test_two_r3_workers_on_one_job_keep_both_cases(r3_library: Path) -> None:
    from tau2_loop.workflows import library

    st = library.store_for(D, "r3")
    a_locked = threading.Event()
    b_saw: list[list[dict[str, Any]]] = []

    def ask_a(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        n = len([m for m in messages if m["role"] == "assistant"])
        if n == 0:
            return Reply(json.dumps(DRAFT))
        if n == 1:
            return Reply(DECIDED)
        a_locked.set()  # A holds the job and is merging: B arrives now
        deadline = time.time() + 10
        while not st.locks()["waiting"] and time.time() < deadline:
            time.sleep(0.02)
        return Reply(merged([*OLD["steps"], BRANCH_A], [{"change": "added", "what": "explain_a"}]))

    def ask_b(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        n = len([m for m in messages if m["role"] == "assistant"])
        if n < 2:
            return Reply(json.dumps(DRAFT) if n == 0 else DECIDED)
        b_saw.append(list(messages))
        return Reply(
            merged([*OLD["steps"], BRANCH_A, BRANCH_B], [{"change": "added", "what": "refund_b"}])
        )

    out: dict[str, rag_agent.SessionMeta] = {}

    def go(tid: str, ask: rag_agent.Ask) -> None:
        out[tid] = rag_agent.start(
            D, tid, "r3", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
        )

    ta = threading.Thread(target=go, args=("task_019", ask_a))
    ta.start()
    assert a_locked.wait(10)
    tb = threading.Thread(target=go, args=("task_020", ask_b))
    tb.start()
    ta.join(20)
    tb.join(20)
    a, b = (rag_agent.session(D, out[t].id) for t in ("task_019", "task_020"))
    assert a["meta"]["status"] == b["meta"]["status"] == "done"
    assert a["output"]["lock"]["committed"] == {"cash_back_dispute": 1}
    assert b["output"]["lock"]["committed"] == {"cash_back_dispute": 2}
    # B waited for A, and its log says who it waited for
    wait = next(e for e in b["events"] if e["kind"] == "wait")
    assert wait["in_the_way"][0]["task"] == "task_019" and b["output"]["lock"]["waited_ms"] > 0
    # B merged A's version, the newer one, and was told it changed since B read it
    shown = b_saw[0][-1]["content"]
    assert (
        "explain_a" in shown
        and "changed since you read it above: version 1, from task_019" in shown
    )
    head = library.entries(D, "r3")[0]
    assert (head["job"], head["version"]) == ("cash_back_dispute", 2)
    assert [s["id"] for s in head["workflow"]["steps"]] == [
        "verify",
        "give",
        "explain_a",
        "refund_b",
    ]
    # every message the harness sent is in the log, in full: the decision asked, then the merge
    said = [e["text"] for e in b["events"] if e["kind"] == "user"]
    assert "name the library jobs you will write" in said[0] and shown == said[1]
    # r2's library is not touched, and the commit log names both questions in order
    assert [s["id"] for s in library.entries(D, "r2")[0]["workflow"]["steps"]] == ["verify", "give"]
    assert [c["task"] for c in st.commits()] == ["task_019", "task_020"]
    assert st.locks() == {"held": [], "waiting": []}


def test_an_r3_merge_into_a_job_it_did_not_name_is_sent_back_then_not_taken(
    r3_library: Path,
) -> None:
    from tau2_loop.workflows import library

    new = json.dumps({"decision": {"submitting_a_cash_back_dispute": []}})  # "a new job"
    into_old = merged([*OLD["steps"], BRANCH_A], [{"change": "added", "what": "explain_a"}])
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(new), Reply(into_old), Reply(into_old)])
    meta = rag_agent.start(
        D, "task_019", "r3", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    out = rag_agent.session(D, meta.id)["output"]
    assert out["lock"]["jobs"] == ["submitting_a_cash_back_dispute"]
    rec = out["merge"][0]
    assert rec["taken"] is False and any("did not name" in w for w in rec["failed"])
    assert out["lock"]["committed"] == {} and out["jobs"] == []
    head = library.entries(D, "r3")[0]
    assert (head["job"], head["version"]) == ("cash_back_dispute", 0)  # r1's, as seeded


def test_an_r3_job_that_keeps_its_new_name_while_it_merges_an_old_one_is_committed(
    r3_library: Path,
) -> None:
    from tau2_loop.workflows import library

    # task_089's shape: the draft is new, the decision merges an old job into it, and the merge
    # keeps the draft's name and retires the old one
    kept = json.loads(merged([*OLD["steps"], BRANCH_A], [{"change": "added", "what": "explain_a"}]))
    kept["jobs"][0] = {
        **kept["jobs"][0],
        "job": "submitting_a_cash_back_dispute",
        "aliases": ["cash_back_dispute"],
    }
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(DECIDED), Reply(json.dumps(kept))])
    meta = rag_agent.start(
        D, "task_089", "r3", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    out = rag_agent.session(D, meta.id)["output"]
    assert out["lock"]["jobs"] == ["cash_back_dispute", "submitting_a_cash_back_dispute"]
    assert out["lock"]["error"] is None
    assert out["lock"]["committed"] == {"submitting_a_cash_back_dispute": 1}
    [head] = library.entries(D, "r3")
    assert (head["job"], head["version"]) == ("submitting_a_cash_back_dispute", 1)
    assert library.store_for(D, "r3").commits()[0]["retired"] == ["cash_back_dispute"]


def test_an_r3_merge_that_renames_its_job_is_sent_back_then_not_taken(r3_library: Path) -> None:
    from tau2_loop.workflows import library

    renamed = json.loads(merged([*OLD["steps"], BRANCH_A], [{"change": "added", "what": "a"}]))
    renamed["jobs"][0]["job"] = "cash_back_disputes_all_cards"  # neither name the lock holds
    again = json.dumps(renamed)
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(DECIDED), Reply(again), Reply(again)])
    meta = rag_agent.start(
        D, "task_089", "r3", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    out = rag_agent.session(D, meta.id)["output"]
    rec = out["merge"][0]
    assert rec["taken"] is False and any("a name it did not name" in w for w in rec["failed"])
    assert out["lock"]["error"] is None and out["lock"]["committed"] == {}
    [head] = library.entries(D, "r3")
    assert (head["job"], head["version"]) == ("cash_back_dispute", 0)


def test_an_r3_question_whose_commit_was_refused_is_queued_again(r3_library: Path) -> None:
    for tid, error in (("task_089", "CommitRefusedError: x not locked"), ("task_094", None)):
        d = r3_library / f"20261007T105040Z_banking_knowledge_r3_{tid}"
        d.mkdir()
        meta = {"id": d.name, "domain": D, "agent": "r3", "status": "done", "source": "question"}
        (d / "run.json").write_text(json.dumps({**meta, "task_id": tid}))
        (d / "output.json").write_text(json.dumps({"jobs": [], "lock": {"error": error}}))
    q = rag_agent.build_queue(D, "r3")
    assert "task_089" in q and "task_094" not in q  # nothing of 089's was written


def test_r3_keeps_what_r2_merged_and_queues_the_rest_shuffled(r3_library: Path) -> None:
    import random

    d = r3_library / "20261007T010000Z_banking_knowledge_r2_task_001"
    d.mkdir()
    meta = {"id": d.name, "domain": D, "agent": "r2", "status": "done", "source": "question"}
    (d / "run.json").write_text(json.dumps({**meta, "task_id": "task_001"}))
    train = sorted(split_ids(D, "train"), key=lambda t: int(t.split("_")[1]))
    random.Random(rag_agent.load(D, "r3").order_seed).shuffle(train)
    q = rag_agent.build_queue(D, "r3")
    assert "task_001" not in q and q == [t for t in train if t != "task_001"]
    assert rag_agent.build_queue(D, "r3") == q  # the same order on every resume


def test_the_r3_build_runs_its_questions_on_several_workers(
    r3_library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from tau2_loop.cli import app

    running, peak, started = [0], [0], []
    guard = threading.Lock()

    def fake_start(
        domain: str, tid: str, rag: str, background: bool = True, split: str = "train"
    ) -> rag_agent.SessionMeta:
        with guard:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            started.append(tid)
        time.sleep(0.2)
        with guard:
            running[0] -= 1
        return rag_agent.SessionMeta(
            id=f"x_{tid}",
            domain=domain,
            agent=rag,
            task_id=tid,
            model="m",
            effort="e",
            retrieval="r",
            started="s",
            status="done",
            duration_ms=60000,
        )

    monkeypatch.setattr(rag_agent, "start", fake_start)
    monkeypatch.setattr(
        rag_agent, "session", lambda d, sid: {"output": {"merge": [], "lock": {"waited_ms": 0}}}
    )
    r = CliRunner().invoke(app, ["rag-build", "--rag", "r3", "--workers", "3", "--limit", "6"])
    assert r.exit_code == 0, r.output
    assert "6 train questions to research and merge, 3 workers, shuffled order" in r.output
    assert len(started) == len(set(started)) == 6 and peak[0] == 3  # each once, three at a time
    assert sorted(started) == sorted(rag_agent.build_queue(D, "r3")[:6])  # the queue's first six


# ── r4: r3 at less cost: write turns keep the tools (the cache holds), merges may be edits ──
def edited(*ops: dict[str, Any]) -> str:
    job = {
        "job": "cash_back_dispute",
        "into": ["cash_back_dispute"],
        "aliases": ["submitting_a_cash_back_dispute"],
        "edits": list(ops),
    }
    return json.dumps({"jobs": [job], "documents": [], "open_questions": []})


def test_an_r4_merge_written_as_edits_becomes_the_next_version(r3_library: Path) -> None:
    from tau2_loop.workflows import library

    add = {"op": "add", "list": "steps", "item": BRANCH_A, "why": "the no-discrepancy case"}
    ask, seen = scripted([Reply(json.dumps(DRAFT)), Reply(DECIDED), Reply(edited(add))])
    meta = rag_agent.start(
        D, "task_019", "r4", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    out = rag_agent.session(D, meta.id)["output"]
    assert out["lock"]["committed"] == {"cash_back_dispute": 1}
    rec = out["merge"][0]
    assert rec["taken"] is True and rec["written"]["edits"] == 1
    assert rec["changelog"][0]["change"] == "added" and "explain_a" in rec["changelog"][0]["what"]
    [head] = library.entries(D, "r4")
    assert [s["id"] for s in head["workflow"]["steps"]] == ["verify", "give", "explain_a"]
    assert head["workflow"]["done_when"] == OLD["done_when"]  # what it did not edit is r3's
    assert out["jobs"][0]["steps"] == head["workflow"]["steps"]  # the output holds the whole job
    # every turn offered the tools, the decide turn too, so the prompt cache reads it back
    assert len(seen) == 3 and all(t for t in seen)
    # r3's library is not touched
    assert [s["id"] for s in library.entries(D, "r3")[0]["workflow"]["steps"]] == ["verify", "give"]


def test_an_r4_write_turn_refuses_a_tool_call_instead_of_running_it(r3_library: Path) -> None:
    ran: list[str] = []
    ask, _ = scripted(
        [
            Reply(json.dumps(DRAFT)),
            Reply(None, [call("KB_search_bm25", query="cash back")]),  # in the decide turn
            Reply(DECIDED),
            Reply(edited({"op": "add", "list": "steps", "item": BRANCH_A, "why": "x"})),
        ]
    )
    meta = rag_agent.start(
        D,
        "task_019",
        "r4",
        ask=ask,
        run_tool=lambda n, a: ran.append(n) or "",
        env=FakeEnv(),
        background=False,
    )
    s = rag_agent.session(D, meta.id)
    assert ran == []  # nothing ran
    tool = next(e for e in s["events"] if e["kind"] == "tool")
    assert tool["refused"] and "no more tool calls" in tool["content"]
    assert s["output"]["decision"] == {"submitting_a_cash_back_dispute": ["cash_back_dispute"]}
    assert s["output"]["lock"]["committed"] == {"cash_back_dispute": 1}


def test_r4_edits_that_name_nothing_are_sent_back_then_not_taken(r3_library: Path) -> None:
    from tau2_loop.workflows import library

    bad = edited({"op": "remove", "list": "steps", "key": "explain", "why": "x"})
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(DECIDED), Reply(bad), Reply(bad)])
    meta = rag_agent.start(
        D, "task_019", "r4", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    s = rag_agent.session(D, meta.id)
    rec = s["output"]["merge"][0]
    assert rec["taken"] is False and "no single item of steps is 'explain'" in rec["failed"][0]
    said = [e["text"] for e in s["events"] if e["kind"] == "user"]
    assert "applied again to the library's newest version" in said[-1]  # the send-back
    assert s["output"]["lock"]["error"] is None and s["output"]["lock"]["committed"] == {}
    [head] = library.entries(D, "r4")
    assert (head["job"], head["version"]) == ("cash_back_dispute", 0)


def test_r4_queues_nothing_r3s_library_already_merged(r3_library: Path) -> None:
    for agent, tid in (("r2", "task_001"), ("r3", "task_005")):
        d = r3_library / f"20261007T010000Z_banking_knowledge_{agent}_{tid}"
        d.mkdir()
        meta = {"id": d.name, "domain": D, "agent": agent, "status": "done", "source": "question"}
        (d / "run.json").write_text(json.dumps({**meta, "task_id": tid}))
    q = rag_agent.build_queue(D, "r4")
    assert "task_001" not in q and "task_005" not in q  # r2's and r3's merges are r4's seed
    assert "task_018" in q  # r1 researched it but merged nothing
    assert "task_018" in rag_agent.build_queue(D, "r3")  # r3's own queue is as it was


# ── r3 on test (8 Oct 2026, the person's call): only when the build names the test split ──
def test_a_build_on_the_test_split_queues_test_questions_and_records_the_split(
    r3_library: Path,
) -> None:
    import random

    test = sorted(split_ids(D, "test"), key=lambda t: int(t.split("_")[1]))
    random.Random(rag_agent.load(D, "r3").order_seed).shuffle(test)
    assert rag_agent.build_queue(D, "r3", split="test") == test
    train_id = split_ids(D, "train")[0]
    with pytest.raises(rag_agent.RagAgentError, match="not test questions"):
        rag_agent.build_queue(D, "r3", [train_id], split="test")
    with pytest.raises(rag_agent.RagAgentError, match="test is sealed"):
        rag_agent.build_queue(D, "r3", [test[0]])  # the default is still train only
    ask, _ = scripted([Reply(json.dumps(DRAFT)), Reply(DECIDED), Reply(merged(OLD["steps"]))])
    meta = rag_agent.start(
        D,
        test[0],
        "r3",
        ask=ask,
        run_tool=lambda n, a: "",
        env=FakeEnv(),
        background=False,
        split="test",
    )
    assert rag_agent.session(D, meta.id)["meta"]["split"] == "test"
    assert test[0] not in rag_agent.build_queue(D, "r3", split="test")  # done: not queued again
