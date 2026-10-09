"""s22: rubric D9 (every write names its tool's parameters) and D10 (a write's documented status
requirement is met), read from tau2's tools and the documents (r5, r6), and D12 (a merge re-reads
every step and rule of the version it merges, r7). No model."""

from __future__ import annotations

from typing import Any

import pytest

from tau2_loop.data import documents as kb
from tau2_loop.workflows import rubric, toolspec

D = "banking_knowledge"
CLOSING = "doc_bank_accounts_bank_accounts_(general)_025"
FREEZING = "doc_bank_accounts_bank_accounts_(general)_026"
pytestmark = pytest.mark.skipif(not kb.available(D), reason="tau2's documents are not here")


def call(sid: str, tool: str, **args: Any) -> dict[str, Any]:
    return {
        "id": sid,
        "by": "harness",
        "call": "call_discoverable_agent_tool",
        "args": {"agent_tool_name": tool, **args},
    }


def job(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"job": "closing_a_card", "steps": list(steps), "rules": []}


def test_d9_names_each_parameter_a_write_leaves_out_and_what_it_is() -> None:
    j = job(
        {
            "id": "unlock",
            "call": "unlock_discoverable_agent_tool",
            "args": {"agent_tool_name": "close_debit_card_4721"},
        },  # an unlock calls nothing
        call("close", "close_debit_card_4721", card_id="from get_cards"),
    )
    [m] = toolspec.missing_params(D, j)
    assert (m["step"], m["tool"], m["missing"]) == ("close", "close_debit_card_4721", ["reason"])
    assert "lost, stolen" in m["about"]["reason"]  # tau2's own description of the parameter
    whole = job(call("close", "close_debit_card_4721", card_id="x", reason="as the customer said"))
    assert toolspec.missing_params(D, whole) == []
    # parameters given inside an `arguments` object count too
    nested = job(call("close", "close_debit_card_4721", arguments={"card_id": "x", "reason": "y"}))
    assert toolspec.missing_params(D, nested) == []


def test_d10_a_close_the_job_can_meet_frozen_needs_the_documented_unfreeze() -> None:
    freeze = call("freeze", "freeze_debit_card_3892", card_id="x")
    close = call("close", "close_debit_card_4721", card_id="x", reason="stolen")
    [m] = toolspec.unmet_status(D, job(freeze, close))  # the job itself freezes cards
    assert (m["step"], m["meets"], m["remedy"]) == ("close", "FROZEN", ["unfreeze_debit_card_3893"])
    assert (m["doc"], m["remedy_doc"], m["leaves"]) == (CLOSING, FREEZING, "ACTIVE")
    # a close step that says it handles a frozen card, with no freeze in the job, is flagged too
    frozen = {**close, "do": "Close the card the customer froze earlier; it shows FROZEN."}
    assert [x["step"] for x in toolspec.unmet_status(D, job(frozen))] == ["close"]
    # unfreezing first meets the requirement
    unfreeze = call("unfreeze", "unfreeze_debit_card_3893", card_id="x")
    assert toolspec.unmet_status(D, job(freeze, unfreeze, close)) == []
    # a close of an ACTIVE card in a job that never meets a frozen one needs nothing
    assert toolspec.unmet_status(D, job(close)) == []


def test_d10_accepts_a_prohibition_only_from_the_procedures_themselves() -> None:
    freeze = call("freeze", "freeze_debit_card_3892", card_id="x")
    close = call("close", "close_debit_card_4721", card_id="x", reason="stolen")
    elsewhere = {
        "rule": "no unfreeze",
        "quote": "DO NOT immediately offer to unfreeze or reactivate.",
        "doc": "doc_bank_accounts_bank_accounts_(general)_037",
    }  # a decline code's rule
    assert toolspec.unmet_status(D, {**job(freeze, close), "rules": [elsewhere]})
    own = {**elsewhere, "doc": FREEZING}
    assert toolspec.unmet_status(D, {**job(freeze, close), "rules": [own]}) == []


def test_the_gate_adds_d9_and_d10_only_for_a_version_that_names_them() -> None:
    j = job(
        call("freeze", "freeze_debit_card_3892", card_id="x"),
        call("close", "close_debit_card_4721", card_id="x"),
    )
    c = rubric.check(D, j, [j])
    assert c["D9"] and c["D10"]  # reported for every version
    assert rubric.gate(c) == []  # r2-r4 gate on D8 and D1 only
    why = rubric.gate(c, ("D9", "D10"))
    assert any("without reason" in w for w in why)
    assert any("unfreeze_debit_card_3893" in w and FREEZING in w for w in why)


def test_d12_names_each_old_step_and_rule_a_merge_left_untouched_without_keeping_it() -> None:
    rule = {"rule": "Provisional credit is not required when the merchant was not contacted"}
    old = {
        **job(
            call("close", "close_debit_card_4721", card_id="x", reason="y"),
            {"id": "explain", "by": "model", "do": "Explain the timeline."},
        ),
        "rules": [rule],
    }
    added = {"id": "new_case", "by": "model", "do": "This customer's case."}
    merged = {**old, "steps": [*old["steps"], added]}
    assert rubric.unread(merged, [old], None) == {
        "steps": ["close", "explain"],
        "rules": [rule["rule"]],
    }
    # kept by name (a rule by its start), rewritten, or removed: each counts as re-read
    kept = {"steps": ["close"], "rules": ["Provisional credit is not required"]}
    rewritten = {
        **merged,
        "steps": [merged["steps"][0], {**merged["steps"][1], "do": "Say 45 days."}, added],
    }
    assert rubric.unread(rewritten, [old], kept) == {"steps": [], "rules": []}
    removed = {**merged, "steps": [merged["steps"][0], added], "rules": []}
    assert rubric.unread(removed, [old], {"steps": ["close"]}) == {"steps": [], "rules": []}
    # a rule named by too short a start is not named
    assert rubric.unread(merged, [old], {"steps": ["close", "explain"], "rules": ["Prov"]})["rules"]


def test_the_gate_sends_back_unread_steps_only_for_a_version_that_names_d12() -> None:
    old = job(call("close", "close_debit_card_4721", card_id="x", reason="y"))
    merged = {**old, "steps": [*old["steps"], {"id": "new_case", "by": "model", "do": "x"}]}
    c = rubric.check(D, merged, [old])
    assert c["D12"]["steps"] == ["close"]  # reported for every version
    assert not any("re-read" in w for w in rubric.gate(c, ("D9", "D10")))
    [why] = [w for w in rubric.gate(c, ("D12",)) if "re-read" in w]
    assert "steps close" in why and "`kept`" in why
    assert rubric.gate(rubric.check(D, merged, [old], kept={"steps": ["close"]}), ("D12",)) == []
