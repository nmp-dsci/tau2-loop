"""workflow_rag r18 (s23): every clause from its source, what a session read, and the tools' replies.

No model is called: the reviewer and the writer are scripted. The KB-backed tests skip when tau2's
documents are not here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tau2_loop.data import documents as kb_docs
from tau2_loop.workflows import provenance, rag_agent, toolspec

D = "banking_knowledge"
SILVER = "doc_savings_accounts_silver_account_002"
MAY = (
    "If you maintain multiple Rho-Bank products, you may also qualify for a relationship bonus of "
    "0.025% that stacks on top of your base Silver APY."
)
needs_kb = pytest.mark.skipif(not kb_docs.available(D), reason="tau2's documents are not here")


@needs_kb
def test_a_hedged_quote_under_add_is_picked_for_the_reviewer() -> None:
    """s23's 097: "may also qualify" became "add the bonus"; D1 found the quote, nothing read it."""
    step = {
        "id": "bonus",
        "do": "Add the relationship bonus of 0.025%.",
        "quote": MAY,
        "doc": SILVER,
    }
    assert provenance.step_flags(D, step)["C3"] is True
    picks = provenance.flagged(D, {"job": "j", "steps": [step]})
    assert picks == [{"step": "bonus", "flags": provenance.step_flags(D, step)}]


@needs_kb
def test_a_clause_with_its_own_basis_entry_is_not_counted_as_unsourced() -> None:
    """C4: two deciding clauses under one quote are picked; a basis entry for each is not."""
    do = "Only when the customer asks, explain the bonus. Never credit it if the card is closed."
    one = {"id": "s", "do": do, "quote": MAY, "doc": SILVER}
    assert provenance.step_flags(D, one)["C4"] == {"deciding": 2, "quotes": 1}
    two = {
        **one,
        "basis": [
            {"clause": "never credit it if the card is closed", "source": "none", "why": "x"}
        ],
    }
    assert "C4" not in provenance.step_flags(D, two)


@needs_kb
def test_a_value_the_documents_define_but_the_quote_lacks_is_picked() -> None:
    """C2, s23's 088: pin_purchase offered under a quote that defines neither value."""
    step = {
        "id": "fraud",
        "do": "File transaction_type pin_purchase (or signature_purchase).",
        "quote": MAY,
        "doc": SILVER,
    }
    assert {"pin_purchase", "signature_purchase"} <= set(provenance.step_flags(D, step)["C2"])


def _shell(cmd: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "shell", "arguments": json.dumps({"command": cmd})}}],
    }


@needs_kb
def test_a_document_printed_in_full_must_be_cited_or_set_aside() -> None:
    """D15, s23's 084: r8 printed doc 037 with cat and cited it nowhere."""
    d37 = "doc_bank_accounts_bank_accounts_(general)_037"
    messages = [
        _shell("cat doc_bank_accounts_bank_accounts__general__037.md"),
        _shell("ls | grep debit"),  # a listing prints no document
    ]
    assert provenance.read_in_full(D, messages) == [d37]
    assert provenance.unaccounted(D, messages, {"jobs": []}) == [d37]
    cited = {"jobs": [{"job": "j", "steps": [{"id": "s", "quote": "x", "doc": d37}]}]}
    assert provenance.unaccounted(D, messages, cited) == []
    aside = {"jobs": [], "set_aside": [{"doc": "doc_bank_accounts_bank_accounts__general__037.md"}]}
    assert provenance.unaccounted(D, messages, aside) == []
    assert provenance.unaccounted(D, messages, {"jobs": [], "documents": [{"id": d37}]}) == [d37]


@needs_kb
def test_a_tools_error_replies_are_read_from_tau2s_source() -> None:
    """M6, s23's 051: an identical credit limit request always returns this reply."""
    reply = "Error: A similar request may already exist."
    name = "submit_credit_limit_increase_request_7392"
    assert reply in toolspec.error_replies(D)[name]
    assert reply not in (toolspec.describe(D, name) or "")  # r17 never saw it
    assert reply in (toolspec.describe(D, name, errors=True) or "")
    said = rag_agent.tool_doc_reply(D, {"tool_name": name}, errors=True)
    assert "Replies it can return besides its result" in said


@needs_kb
def test_an_idea_a_tools_description_names_by_title_is_followed() -> None:
    """D16, s23's 084: the dispute tool's description names Regulation E, doc 037's title."""
    d37 = "doc_bank_accounts_bank_accounts_(general)_037"
    job: dict[str, Any] = {
        "job": "dispute",
        "steps": [{"id": "f", "call": "file_debit_card_transaction_dispute_6281"}],
    }
    found = {m["idea"]: m["docs"] for m in toolspec.named_titles(D, job)}
    assert found["Regulation E"] == [d37]
    job["steps"].append({"id": "q", "quote": "x", "doc": d37})
    assert "Regulation E" not in {m["idea"] for m in toolspec.named_titles(D, job)}
    assert "Regulation E" not in {
        m["idea"] for m in toolspec.named_titles(D, job, frozenset({d37}))
    }


def test_the_reviewers_no_is_sent_back_and_its_entails_is_not() -> None:
    text = json.dumps(
        {
            "verdicts": [
                {"step": "bonus", "clause": "add the bonus", "verdict": "no", "why": "may"},
                {"step": "bonus", "clause": "for Silver", "verdict": "entails"},
                {"step": "x", "clause": "y", "verdict": "maybe"},  # not a verdict: dropped
            ]
        }
    )
    verdicts = provenance.parse_review(text)
    assert [v["verdict"] for v in verdicts] == ["no", "entails"]
    [line] = provenance.review_reasons(verdicts)
    assert line.startswith('step bonus: the reviewer finds "add the bonus" not supported')
    assert provenance.parse_review("not json") == []


# ── r18 end to end: the reviewer's "no" goes back once, then the merge is taken and recorded ──
@needs_kb
def test_an_r18_merge_sends_the_reviewers_no_back_once_then_takes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_rag_agent import DECIDED, DRAFT, VERIFY_PARAMS, FakeEnv, Reply, edited, seed_r1

    from tau2_loop.workflows import library, store

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(rag_agent, "RAG_RUNS_DIR", runs)
    monkeypatch.setattr(store, "LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(store, "POLL_S", 0.02)
    seed_r1(runs)
    bonus = {"id": "bonus", "by": "model", "do": "Add the relationship bonus of 0.025%."}
    bonus |= {"quote": MAY, "doc": SILVER}
    add = {"op": "add", "list": "steps", "item": bonus, "why": "the bonus case"}
    verify = {
        "op": "change",
        "list": "steps",
        "key": "verify",
        "fields": {"args": {p: "from the customer" for p in VERIFY_PARAMS}},
        "why": "every parameter of log_verification",
    }
    merged = json.loads(edited(add, verify))
    merged["jobs"][0]["kept"] = {"steps": ["give"]}
    no = json.dumps(
        {"verdicts": [{"step": "bonus", "clause": "Add", "verdict": "no", "why": "may"}]}
    )
    replies = [
        Reply(json.dumps(DRAFT)),
        Reply(DECIDED),
        Reply(json.dumps(merged)),
        Reply(no),  # the reviewer
        Reply(json.dumps(merged)),  # the writer, after the send-back, unchanged
        Reply(no),  # the reviewer again
    ]
    asked: list[tuple[str, list[dict[str, Any]] | None]] = []

    def ask(messages, tools, model, effort, tool_mode):  # type: ignore[no-untyped-def]
        asked.append((effort, tools))
        return replies.pop(0)

    meta = rag_agent.start(
        D, "task_019", "r18", ask=ask, run_tool=lambda n, a: "", env=FakeEnv(), background=False
    )
    s = rag_agent.session(D, meta.id)
    reviews = [e for e in s["events"] if e["kind"] == "review"]
    assert len(reviews) == 2 and "### step bonus" in reviews[0]["text"]
    assert [a for a in asked if a[1] is None] == [("medium", None)] * 2  # its own call, no tools
    sent_back = [e["text"] for e in s["events"] if e["kind"] == "user"][-1]
    assert 'the reviewer finds "Add" not supported by its quote' in sent_back
    rec = s["output"]["merge"][0]
    assert rec["taken"] is True and not rec["failed"]  # advice is recorded, never blocking
    assert rec["review"]["verdicts"][0]["verdict"] == "no" and rec["advice"]
    assert rec["checks"]["D14"] == {"picked": 2, "no": 1, "partly": 0}
    # the kept step is read too (M3): `give` runs "if a discrepancy" with no quote at all
    assert [p["step"] for p in rec["review"]["picked"]] == ["give", "bonus"]
    assert s["output"]["unaccounted"] == [] and s["output"]["set_aside"] == []
    [head] = library.entries(D, "r18")
    assert [x["id"] for x in head["workflow"]["steps"]][-1] == "bonus"
    assert rag_agent.load(D, "r17").reviewer is None  # r17 never calls a reviewer
