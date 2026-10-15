"""s25 (v27, the person's calls, 15 Oct 2026): v26 reading r19, plus the runtime pieces.

A1 the harness keeps each job find_workflow shows in the hooks' state; A2 `workflow_view: steps`
shows a job without the quotes and basis behind it; A3 guidance names the shown job's next call
steps; H1 check D2 lets a liability equal to a disputed amount under the tier through; H2 a
transfer before the customer's 4th request for a human is sent back once. Each is driven through
the version's own hooks or the harness function, with no model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tau2_loop.agent.compose import load_code_surfaces
from tau2_loop.agent.versions import AgentConfig, load_version, version_dir
from tau2_loop.workflows import library

D = "banking_knowledge"
DISPUTE = "file_debit_card_transaction_dispute_6281"
JOB = {
    "job": "filing_a_credit_card_transaction_dispute",
    "aliases": ["credit card dispute"],
    "when": {"quote": "Use this process when …", "doc": "doc_x"},
    "steps": [
        {
            "id": "verify",
            "call": "log_verification",
            "quote": "Verify the customer",
            "doc": "policy",
        },
        {"id": "ask_priority", "do": "Ask which disputes come first.", "after": ["verify"]},
        {
            "id": "file",
            "call": "file_credit_card_transaction_dispute_4829",
            "after": ["verify", "ask_priority"],
            "basis": [{"clause": "one at a time", "quote": "File each separately", "doc": "doc_x"}],
        },
    ],
    "rules": [{"rule": "credit is limited", "quote": "at most two", "doc": "doc_y"}],
    "done_when": "every dispute filed",
}


def hooks(version: str) -> dict[str, Any]:
    return load_code_surfaces(version_dir(D, version), f"t27_{version}")


def dispute(liability: float, amount: float, tid: str = "btxn_1") -> dict[str, Any]:
    args = {
        "transaction_id": tid,
        "disputed_amount": amount,
        "customer_max_liability_amount": liability,
    }
    return {"agent_tool_name": DISPUTE, "arguments": json.dumps(args)}


def test_v27_is_v26_reading_r19_with_the_steps_view() -> None:
    v26, v27 = load_version(D, "v26"), load_version(D, "v27")
    assert (v27.config.workflows, v27.config.workflow_view) == ("r19", "steps")
    assert (v26.config.workflows, v26.config.workflow_view) == ("r18", None)
    with pytest.raises(ValueError):
        AgentConfig(workflow_view="summary")


def test_h1_a_liability_equal_to_a_disputed_amount_under_the_tier_is_filed() -> None:
    v26, v27 = hooks("v26")["checks.py"], hooks("v27")["checks.py"]
    late = dispute(412.88, 412.88)
    assert (
        v26.check_write("call_discoverable_agent_tool", late, {}) is not None
    )  # v24's D2 sent it back
    assert v27.check_write("call_discoverable_agent_tool", late, {}) is None
    assert (
        v27.check_write("call_discoverable_agent_tool", dispute(500, 412.88), {}) is None
    )  # the tier passes too
    wrong = v27.check_write("call_discoverable_agent_tool", dispute(100, 412.88), {})
    assert wrong is not None and "a maximum" in wrong  # neither a tier nor the amount


def test_h2_a_transfer_before_the_fourth_request_is_sent_back_once() -> None:
    checks = hooks("v27")["checks.py"]
    state: dict[str, Any] = {"human_requests": 3}
    first = checks.check_write("transfer_to_human_agents", {"reason": "x"}, state)
    assert first is not None and "3 time(s)" in first and "Rule 5" in first
    assert (
        checks.check_write("transfer_to_human_agents", {"reason": "x"}, state) is None
    )  # once only
    assert checks.check_write("transfer_to_human_agents", {}, {"human_requests": 4}) is None
    assert (
        hooks("v26")["checks.py"].check_write("transfer_to_human_agents", {}, {"human_requests": 1})
        is None
    )


def test_a3_guidance_names_the_shown_jobs_next_call_step_and_what_comes_before_it() -> None:
    v27 = hooks("v27")
    state: dict[str, Any] = {}
    v27["memory.py"].remember(state, "log_verification", {}, "Verification logged successfully.")
    state["workflows"] = {JOB["job"]: JOB}
    note = v27["guidance.py"].guidance(state, "tool")
    assert note is not None
    assert (
        f"Workflow {JOB['job']}" in note
        and "file (file_credit_card_transaction_dispute, after ask_priority)" in note
    )
    assert "verify (" not in note  # it ran
    v27["memory.py"].remember(
        state,
        "call_discoverable_agent_tool",
        {"agent_tool_name": "file_credit_card_transaction_dispute_4829"},
        "Dispute filed successfully.",
    )
    assert "next call steps" not in (v27["guidance.py"].guidance(state, "tool") or "")
    assert "next call steps" not in (
        hooks("v26")["guidance.py"].guidance({"workflows": {JOB["job"]: JOB}}, "tool") or ""
    )


def test_a2_the_steps_view_drops_the_evidence_and_keeps_what_the_agent_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = {"job": JOB["job"], "workflow": JOB, "session": "s1"}
    monkeypatch.setattr(library, "entries", lambda domain, rag: [entry])
    full = library.find(D, "r19", "", job=JOB["job"])
    steps = library.find(D, "r19", "", job=JOB["job"], view="steps")
    assert '"basis"' in full and '"quote"' in full and '"aliases"' in full
    shown = json.loads(steps.split(":\n", 1)[1])
    assert "aliases" not in shown and '"quote"' not in steps and '"basis"' not in steps
    assert [s["id"] for s in shown["steps"]] == ["verify", "ask_priority", "file"]
    assert (
        shown["steps"][2]["after"] == ["verify", "ask_priority"]
        and shown["done_when"] == JOB["done_when"]
    )
    assert len(steps) < len(full)
    assert library.shown_job(steps) == JOB["job"] and library.shown_job("No job named x") is None


def test_a1_the_harness_keeps_each_shown_job_newest_last(monkeypatch: pytest.MonkeyPatch) -> None:
    from tau2_loop.agent.factory import LoopAgent

    other = {**JOB, "job": "closing_cancelling_a_debit_card"}
    lib = [{"job": j["job"], "workflow": j, "session": "s"} for j in (JOB, other)]
    monkeypatch.setattr(library, "entries", lambda domain, rag: lib)
    agent = SimpleNamespace(version=SimpleNamespace(domain=D), workflows="r19")
    state = SimpleNamespace(memory={})
    used = [
        {"name": "find_workflow", "job": JOB["job"]},
        {"name": "find_workflow"},
        {"name": "find_workflow", "job": other["job"]},
    ]
    LoopAgent._record_workflows(agent, state, used)  # type: ignore[arg-type]
    LoopAgent._record_workflows(agent, state, [{"name": "find_workflow", "job": JOB["job"]}])  # type: ignore[arg-type]
    assert list(state.memory["workflows"]) == [other["job"], JOB["job"]]  # shown again: newest last
    assert state.memory["workflows"][JOB["job"]] is JOB


def test_v27s_surfaces_differ_from_v26s_only_where_the_bundle_says() -> None:
    a, b = version_dir(D, "v26"), version_dir(D, "v27")
    differ = {
        f
        for f in ("system.md", "checks.py", "guidance.py", "memory.py", "helper.py")
        if (a / f).read_text() != (b / f).read_text()
    }
    assert differ == {"system.md", "checks.py", "guidance.py", "memory.py"}
    text = (b / "system.md").read_text()
    assert (
        "never the disputed amount" not in text
        and "pin_purchase an in-store purchase with PIN" in text
    )
    assert Path(b / "agent.yaml").read_text().count("workflow_view: steps") == 1
