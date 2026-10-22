"""s25 (r19, the person's call, 15 Oct 2026): a better merge on the same research.

r18's merges kept the old library: a merge that left out an old quote was refused whole, and the
reviewer's "no" was only advice. r19 replays each r18 session's research, reuses its draft and
decision, and merges again with the draft as the base, logging what it leaves out instead of
refusing. No model is called: each test drives the merge with scripted replies.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.data import documents as kb_docs
from tau2_loop.workflows import provenance, rag_agent, rubric
from test_rag_agent import (
    CASH_BACK_DOC,
    OLD,
    VERIFY_PARAMS,
    D,
    FakeEnv,
    Reply,
    seed_r1,
)

VERIFY = {
    "id": "verify",
    "by": "harness",
    "call": "log_verification",
    "args": {p: "from the customer" for p in VERIFY_PARAMS},
}
DRAFT = {"jobs": [{**OLD, "job": "submitting_a_cash_back_dispute", "steps": [VERIFY]}]}
DECISION = {"submitting_a_cash_back_dispute": ["cash_back_dispute"]}


@pytest.fixture
def library_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from tau2_loop.workflows import store

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", runs)
    monkeypatch.setattr(store, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(store, "POLL_S", 0.02)
    seed_r1(runs)  # every version down the seed chain starts from r1's one job, OLD
    return runs


def r18_session(
    runs: Path,
    task: str,
    started: str,
    locked_ms: int,
    draft: dict[str, Any] = DRAFT,
    decision: dict[str, list[str]] = DECISION,
) -> str:
    """A finished r18 session as the build left it: research events up to the draft, then its merge."""
    sid = f"{started}_{D}_r18_{task}"
    d = runs / sid
    d.mkdir()
    meta = {"id": sid, "domain": D, "agent": "r18", "task_id": task, "status": "done"}
    meta |= {"source": "question", "started": started, "split": "train", "model": "m"}
    (d / "run.json").write_text(json.dumps(meta))
    call = {"id": "c1", "name": "KB_search_bm25", "args": {"query": "cash back"}}
    events = [
        {"t": 0, "kind": "start", "question": "The customer's script."},
        {"t": 10, "kind": "model", "step": 1, "text": None, "calls": [call]},
        {
            "t": 20,
            "kind": "tool",
            "step": 1,
            "id": "c1",
            "name": "KB_search_bm25",
            "content": "a doc",
        },
        {"t": 30, "kind": "model", "step": 2, "text": "not JSON", "calls": []},
        {"t": 31, "kind": "user", "step": 3, "text": "Reply with the JSON."},
        {"t": 40, "kind": "model", "step": 3, "text": json.dumps(draft), "calls": []},
        {
            "t": 50,
            "kind": "merge",
            "candidates": {k: v or [] for k, v in decision.items()},
        },
        {"t": 60, "kind": "decided", "decision": decision},
        {
            "t": locked_ms,
            "kind": "locked",
            "token": 1,
            "jobs": sorted({j for v in decision.values() for j in v}),
        },
        {"t": locked_ms + 10, "kind": "user", "step": 4, "text": "r18's merge message"},
        {"t": locked_ms + 20, "kind": "model", "step": 4, "text": "r18's merge", "calls": []},
    ]
    (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    out = {"jobs": [], "merge": [], "draft": draft["jobs"], "decision": decision}
    (d / "output.json").write_text(json.dumps(out))
    return sid


def test_r19_is_a_draft_based_remerge_of_r18s_drafts_onto_r17() -> None:
    r19 = rag_agent.load(D, "r19")
    assert (r19.drafts_from, r19.seed, r19.merge_base, r19.merge_write) == (
        "r18",
        "r17",
        "draft",
        "full",
    )
    assert r19.review_binding and "D18" in r19.rubric and "D12" not in r19.rubric
    r18 = rag_agent.load(D, "r18")
    assert (r18.drafts_from, r18.merge_base, r18.review_binding) == (None, "library", False)


def test_the_replay_is_the_research_as_the_model_saw_it_up_to_the_draft(library_dirs: Path) -> None:
    sid = r18_session(library_dirs, "task_019", "20261013T050000Z", 5000)
    events = rag_agent.session(D, sid)["events"]
    msgs = rag_agent.replay_messages(events, "r19's system prompt")
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert msgs[0]["content"] == "r19's system prompt"
    assert msgs[1]["content"].endswith("The customer's script.")
    assert msgs[2]["tool_calls"][0]["function"] == {
        "name": "KB_search_bm25",
        "arguments": '{"query": "cash back"}',
    }
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "a doc"}
    assert json.loads(msgs[-1]["content"]) == DRAFT  # nothing of r18's merge is replayed


def test_a_remerge_runs_only_the_merge_and_code_logs_what_the_draft_left_out(
    library_dirs: Path,
) -> None:
    from tau2_loop.workflows import library

    sid = r18_session(library_dirs, "task_019", "20261013T050000Z", 5000)
    asked: list[list[dict[str, Any]]] = []

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        if tools is None and len(messages) == 1:  # the reviewer, in its own context
            return Reply(json.dumps({"verdicts": []}))
        asked.append(list(messages))
        # the draft as the job: OLD's `give` is left out, and the changelog does not name it
        job = {**DRAFT["jobs"][0], "job": "cash_back_dispute", "into": ["cash_back_dispute"]}
        return Reply(
            json.dumps({"jobs": [{**job, "changelog": []}], "documents": [], "open_questions": []})
        )

    meta = rag_agent.remerge(D, "r19", sid, ask=ask, run_tool=lambda n, a: "", env=FakeEnv())
    s = rag_agent.session(D, meta.id)
    assert (s["meta"]["status"], s["meta"]["source"], s["meta"]["drafts_of"]) == (
        "done",
        "remerge",
        sid,
    )
    # one model turn, the merge: no research and no decide turn ran
    assert len(asked) == 1
    said = [e["text"] for e in s["events"] if e["kind"] == "user"]
    assert len(said) == 1 and "Your\ndraft is the newest research" in said[0]
    assert asked[0][: len(asked[0]) - 1][-1]["content"] == json.dumps(DRAFT)  # the replayed draft
    rec = s["output"]["merge"][0]
    assert rec["taken"] is True and rec["failed"] == []
    assert any(x["what"] == "step give" and "logged by code" in x["why"] for x in rec["settled"])
    assert any(x["what"] == "step give" for x in rec["changelog"])
    [head] = library.entries(D, "r19")
    assert [st["id"] for st in head["workflow"]["steps"]] == ["verify"]
    # the libraries it started from are untouched
    assert [st["id"] for st in library.entries(D, "r17")[0]["workflow"]["steps"]] == [
        "verify",
        "give",
    ]


def test_the_same_merge_under_r18s_gate_is_refused_for_what_it_left_out() -> None:
    c = rubric.check(D, {**OLD, "steps": [VERIFY]}, [OLD])
    assert c["D8"]["steps"] == ["give"]
    assert any("lost steps" in w for w in rubric.gate(c))
    assert not any("lost" in w for w in rubric.gate(c, draft_base=True))


@pytest.mark.skipif(not kb_docs.available(D), reason="tau2's documents are not here")
def test_a_quote_still_not_found_loses_its_place_not_the_merge() -> None:
    bad = {"rule": "made up", "quote": "Cash back is tripled on Tuesdays.", "doc": CASH_BACK_DOC}
    clause = {
        "clause": "tripled",
        "quote": "Cash back is tripled on Tuesdays.",
        "doc": CASH_BACK_DOC,
    }
    job = {**OLD, "steps": [VERIFY, {**OLD["steps"][1], "basis": [clause]}], "rules": [bad]}
    c = rubric.check(D, job, [OLD])
    row = {
        "job": job,
        "previous": [{"workflow": OLD}],
        "checks": c,
        "why": rubric.gate(c, draft_base=True),
    }
    assert any(
        "not found" in w for w in row["why"]
    )  # an old quote is gated too, so it goes back first
    rag_agent._settle(D, [row], "", ("D9",), binding=True)
    assert row["why"] == [] and "quote" not in job["rules"][0] and "unquoted" in job["rules"][0]
    assert job["steps"][1]["basis"][0]["source"] == "none"
    assert sum("quote not found" in x["what"] for x in row["settled"]) == 2


def test_a_step_the_reviewer_still_finds_unsupported_is_removed() -> None:
    job = {**OLD, "steps": [VERIFY, OLD["steps"][1]]}
    review = {
        "verdicts": [
            {"step": "give", "clause": "a discrepancy", "verdict": "no", "why": "no quote"}
        ]
    }
    row = {"job": job, "previous": [{"workflow": OLD}], "checks": {}, "why": [], "review": review}
    rag_agent._settle(D, [row], "", (), binding=True)
    assert [st["id"] for st in job["steps"]] == ["verify"]
    assert any(x["what"] == "step give" and "reviewer" in x["why"] for x in row["settled"])
    kept = {**OLD, "steps": [VERIFY, OLD["steps"][1]]}
    row2 = {"job": kept, "previous": [{"workflow": OLD}], "checks": {}, "why": [], "review": review}
    rag_agent._settle(D, [row2], "", (), binding=False)
    assert [st["id"] for st in kept["steps"]] == ["verify", "give"]  # advice only, as in r18


def test_d18_asks_r18s_undefined_period_for_its_bounds_and_not_the_drafts() -> None:
    # r18's library kept this step; r18's own draft for task 090 had the second
    merged = (
        "For a card that cannot be unlocked, look in the txns results for successful (posted or "
        "pending) transactions in the suspicious period that the customer did not make. Ask the "
        "customer to confirm any you find, and tell them whether any money was taken."
    )
    drafted = (
        "When the card cannot be unlocked, look in get_transactions for SUCCESSFUL (non-declined) "
        "transactions during the suspicious period, which is the window of the declined attempts "
        "the customer denied. Ask the customer whether each one in that window was theirs."
    )
    job = {
        "steps": [
            {"id": "check_unauthorized", "do": merged},
            {"id": "cannot_unlock_check", "do": drafted},
        ]
    }
    assert provenance.unbounded_periods(job) == ["check_unauthorized"]
    assert (
        provenance.unbounded_periods(
            {"steps": [{"id": "a", "do": "within 60 days of the statement period"}]}
        )
        == []
    )


def test_the_remerge_queue_takes_r18s_sessions_in_the_order_they_locked_and_resumes(
    library_dirs: Path,
) -> None:
    late = r18_session(library_dirs, "task_019", "20261013T050000Z", 600_000)  # locked at 05:10
    early = r18_session(library_dirs, "task_020", "20261013T050500Z", 1_000)  # locked at 05:05
    assert rag_agent.remerge_queue(D, "r19") == [early, late]
    done = library_dirs / f"20261015T050000Z_{D}_r19_task_020"
    done.mkdir()
    meta = {"id": done.name, "domain": D, "agent": "r19", "task_id": "task_020", "status": "done"}
    (done / "run.json").write_text(json.dumps({**meta, "source": "remerge", "drafts_of": early}))
    (done / "output.json").write_text(json.dumps({"jobs": [], "lock": {}}))
    assert rag_agent.remerge_queue(D, "r19") == [late]
    with pytest.raises(rag_agent.RagAgentError):
        rag_agent.remerge_queue(D, "r18")  # r18 takes no other version's drafts


# ── s34 (r20, the person's call, 21 Oct 2026): the merge bench's option D2 as workflow_rag's merge ──
def test_r20_is_the_benchs_d2_on_r18s_drafts_onto_r17() -> None:
    import dataclasses

    from tau2_loop.workflows import bench

    r20 = rag_agent.load(D, "r20")
    d2 = bench.load_option(D, "D2")
    assert dataclasses.replace(d2, name="r20") == r20  # field for field, the bench's D2
    assert (r20.drafts_from, r20.seed, r20.merge_mode, r20.post_audit, r20.review_binding) == (
        "r18",
        "r17",
        "audit",
        True,
        False,
    )


def test_r20_merges_a_session_of_two_jobs_auditing_the_library_job_before_and_after(
    library_dirs: Path,
) -> None:
    from tau2_loop.workflows import library

    new_job = {
        "job": "checking_interest_question",
        "when": OLD["when"],
        "info": [],
        "steps": [VERIFY],
    }
    new_job |= {"rules": [], "done_when": "the question is answered"}
    draft = {"jobs": [DRAFT["jobs"][0], new_job]}
    decision = {"submitting_a_cash_back_dispute": ["cash_back_dispute"], new_job["job"]: []}
    sid = r18_session(library_dirs, "task_019", "20261013T050000Z", 5000, draft, decision)
    audits: list[str] = []
    turns: list[list[dict[str, Any]]] = []

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        first = str(messages[0]["content"] or "")
        if first.startswith("You review one workflow job"):
            return Reply(json.dumps({"verdicts": []}))
        if first.startswith("You audit one workflow job"):
            audits.append(first)
            ids = sorted(set(re.findall(r'"id": "([^"]+)"', first)))
            return Reply(
                json.dumps({"steps": [{"step": i, "verdict": "valid"} for i in ids], "rules": []})
            )
        turns.append(list(messages))
        jobs = [
            {
                **OLD,
                "steps": [VERIFY, OLD["steps"][1]],  # verify with every parameter (D9)
                "into": ["cash_back_dispute"],
                "aliases": ["submitting_a_cash_back_dispute"],
            },
            {**new_job, "into": []},
        ]
        return Reply(
            json.dumps(
                {
                    "jobs": [{**j, "changelog": []} for j in jobs],
                    "documents": [],
                    "open_questions": [],
                }
            )
        )

    meta = rag_agent.remerge(D, "r20", sid, ask=ask, run_tool=lambda n, a: "", env=FakeEnv())
    s = rag_agent.session(D, meta.id)
    assert s["meta"]["status"] == "done", s["meta"].get("error")
    assert len(turns) == 1  # one merge turn wrote both jobs; no research or decide turn ran
    assert {r["job"] for r in s["output"]["merge"] if r["taken"]} == {
        "cash_back_dispute",
        "checking_interest_question",
    }
    kinds = [
        (e["kind"], e.get("job")) for e in s["events"] if e["kind"] in ("audited", "post_audited")
    ]
    assert ("audited", "cash_back_dispute") in kinds  # the library's version, before the merge
    assert ("post_audited", "cash_back_dispute") in kinds  # the merged job, after it
    assert not any(k == "audited" and j == "checking_interest_question" for k, j in kinds)
    # the audit as the auditor received it holds the gap rule (D2), before and after the merge
    assert audits and all(
        "A gap in the documents is not permission" in " ".join(a.split()) for a in audits
    )
    assert {e["job"] for e in library.entries(D, "r20")} == {
        "cash_back_dispute",
        "checking_interest_question",
    }
