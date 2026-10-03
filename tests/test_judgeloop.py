"""J3 (s11 §7): the judge loop's deterministic parts. The rubric's head is fixed and its lessons change
by number; a lesson naming a customer, an id or a conversation is refused; the optimiser reads only
the read half; the gate pairs the two judges by conversation on the gate half."""

from __future__ import annotations

import json

import pytest

from tau2_loop.config import ROOT
from tau2_loop.data.splits import read_task_extract
from tau2_loop.loop.guards import fence_reason, leak_values
from tau2_loop.tooljudge import labels, loop, replay
from tau2_loop.tooljudge import prompt as judge_prompt


def test_a_rubric_keeps_its_head_and_round_trips_its_lessons() -> None:
    j1 = judge_prompt.load("airline", "j1")
    head, lessons = loop.split_rubric(j1.rubric)
    assert lessons == []
    assert loop.render_rubric(head, [], "j1") == j1.rubric
    md = loop.render_rubric(head, ["Check 2: one.", "Check 3: two."], "j9")
    assert md.startswith("# The plan judge · j9\n")
    head2, lessons2 = loop.split_rubric(md)
    assert lessons2 == ["Check 2: one.", "Check 3: two."]
    assert head2.split("\n", 1)[1] == head.split("\n", 1)[1]


def test_changes_apply_by_number_and_a_rewrite_or_a_no_op_is_refused() -> None:
    four = ["one", "two", "three", "four"]
    new, delta, errors = loop.apply_changes(
        four,
        {"edit": [{"n": 2, "lesson": "TWO"}], "remove": [{"n": 4}], "add": [{"lesson": "five"}]},
    )
    assert (new, errors) == (["one", "TWO", "three", "five"], [])
    assert delta["added"] == ["five"] and delta["removed"] == [{"n": 4, "lesson": "four"}]
    assert "does not exist" in loop.apply_changes(four, {"remove": [{"n": 9}]})[2][0]
    assert loop.apply_changes(four, {})[2] == ["changes.json changes no lesson"]
    rewrite = {"edit": [{"n": k, "lesson": "x"} for k in (1, 2, 3)]}
    assert any("delta edits" in e for e in loop.apply_changes(four, rewrite)[2])
    many = {"add": [{"lesson": f"lesson {k}"} for k in range(loop.MAX_ADDED + 1)]}
    assert any("at most" in e for e in loop.apply_changes([], many)[2])


def test_a_lesson_naming_an_id_a_customer_or_a_conversation_is_refused() -> None:
    policy = str(read_task_extract("airline")["policy"])
    from_policy = " ".join(policy.split()[40:52])
    said = "honestly I would really like to cancel the whole trip because plans changed"
    conversations = [f"[3] customer: {said}. {from_policy}"]

    def problems(lesson: str) -> list[str]:
        return loop.lesson_problems([lesson], "airline", conversations, policy)

    assert problems("Check 2: a payment method must appear in the user's profile.") == []
    assert problems(f"Check 3: the policy says {from_policy}.") == []
    assert problems("Check 1: reservation 4WQ150 is fine.") == ["lesson 1 names a reservation id"]
    assert "lesson 1 names a flight number" in problems("Check 2: flight HAT123 is fine.")
    assert problems("Check 1: pat_doe_1234 owns it.") == ["lesson 1 names a user id"]
    assert problems("Check 2: as in d07, block.") == ["lesson 1 names a disagreement id"]
    assert problems(
        "Block when they would really like to cancel the whole trip because of it."
    ) == ["lesson 1 quotes a conversation"]
    name = next(v for v in sorted(leak_values("airline")) if v.isalpha())
    found = problems(f"Check 1: ask {name} for the id.")
    assert found == ["lesson 1 names a customer from a task"] and name not in found[0]
    assert "over the 60 limit" in loop.lesson_problems(["x"] * 61, "airline", [], policy)[0]


def test_the_optimiser_reads_the_read_half_only_each_with_its_conversation() -> None:
    champ = replay.latest("airline", "j1")
    assert champ is not None
    dis = loop.disagreements("airline", replay.read_verdicts(champ))
    data = labels.read_labels("airline") or {}
    half = {c["key"]: c["half"] for c in data["conversations"]}
    synth = {s["id"]: s for s in data["synthetic"]}
    assert dis and all(half[d["key"]] == "read" for d in dis)
    rows = replay.score("airline", replay.read_verdicts(champ))["rows"]
    gate_keys = {r["key"] for r in rows if r["half"] == "gate"}
    assert not gate_keys & {d["key"] for d in dis}
    assert [d["id"] for d in dis] == [f"d{i:02d}" for i in range(1, len(dis) + 1)]
    for d in dis:
        assert "## The agent's proposed reply, not yet shown to the customer" in d["excerpt"]
        assert d["golden"]["verdict"] != d["judge"]["verdict"]
    by_key_msg = {(s["key"], s["msg"]): s for s in synth.values()}
    for d in (d for d in dis if d["item"] == "synthetic"):
        assert by_key_msg[(d["key"], d["msg"])]["half"] == "read"


def test_the_fence_closes_every_replay_the_plan_pages_and_the_ledger() -> None:
    hidden = loop.hidden_paths("airline")
    closed = [
        ROOT / "judge_runs" / "x" / "verdicts.jsonl",
        ROOT / ".lavish" / "s11_tool-judge-plan.html",
        loop.ledger_path("airline"),
        ROOT / "data" / "judge" / "airline_gold.jsonl",
    ]
    for p in closed:
        assert fence_reason(str(p), set(), hidden), p
    assert fence_reason(str(ROOT / "judges/airline/plan/j1/judge.md"), set(), hidden) is None


def test_the_gate_pairs_conversations_on_the_gate_half() -> None:
    champ_dir = replay.latest("airline", "j1")
    assert champ_dir is not None
    champ = replay.read_verdicts(champ_dir)
    same = loop.gate("airline", champ, champ)
    assert (same["fixed"], same["broken"], same["promote"]) == ([], [], False)
    assert same["conversations"] > 50
    # leave every gate-half pass alone: each pass the champion interrupted is a fix, and nothing breaks
    passed = {c["key"]: c["passed"] for c in (labels.read_labels("airline") or {})["conversations"]}
    calm = [
        {**v, "verdict": "allow"} if v["half"] == "gate" and passed.get(v["key"]) else v
        for v in champ
    ]
    g = loop.gate("airline", champ, calm)
    assert g["fixed"] and not g["broken"] and g["promote"] and g["rule"] == "mcnemar"
    assert {f["kind"] for f in g["fixed"]} == {"pass"}
    # block everything in the gate half's passes: every pass the champion left alone breaks
    loud = [
        {**v, "verdict": "block"} if v["half"] == "gate" and passed.get(v["key"]) else v
        for v in champ
    ]
    assert not loop.gate("airline", champ, loud)["promote"]
    with pytest.raises(ValueError, match="different gate conversations"):
        loop.gate("airline", champ, [v for v in champ if v["half"] != "gate"])


def test_a_bar_is_met_exactly_at_its_limit() -> None:
    rate = {"k": 0, "n": 0}
    s = {k: rate for k in loop.BARS} | {"balanced_accuracy": 0.73}
    s["passes_interrupted"] = {"k": 2, "n": 55}
    b = loop.bars(s)
    assert b["balanced_accuracy"] is True and b["passes_interrupted"] is True
    assert b["right_plans_blocked"] is None


def test_stock_wording_several_conversations_share_is_not_a_quote() -> None:
    """Regression, cycle 1: the agent restates the policy's 24-hour rule in several read-half
    conversations, and a lesson using the same words was refused as a quote."""
    champ = replay.latest("airline", "j1")
    assert champ is not None
    convs = loop.by_conversation(loop.disagreements("airline", replay.read_verdicts(champ)))
    policy = str(read_task_extract("airline")["policy"])
    lesson = "Check 3: a cancellation that qualifies because it was booked within the last 24 hours is allowed."
    assert loop.lesson_problems([lesson], "airline", convs, policy) == []
    one = "the trip to the coast was for my sister's wedding next spring"
    assert loop.quotes(f"Block {one} plans.", [one, "something else"], policy)
    assert not loop.quotes(f"Block {one} plans.", [one, one], policy)
    assert loop._words("24 hrs") == ["24", "hours"]


def test_a_recheck_stands_for_its_cycle(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(loop, "ledger_path", lambda domain: ledger)
    lines = [
        {"cycle": 1, "verdict": "rejected"},
        {"cycle": 1, "verdict": "held"},
        {"cycle": 2, "verdict": "held"},
    ]
    ledger.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert loop.last_verdicts("airline") == ["held", "held"]


def test_optimise_on_the_judge_serves_its_ledger_and_champion() -> None:
    from fastapi.testclient import TestClient

    from tau2_loop.serving.app import create_app

    c = TestClient(create_app())
    o = c.get("/api/judge/airline").json()
    assert o["cycles"] == loop.read_ledger("airline")
    assert (o["registry"] or {}).get("champion", "j1") == loop.read_registry("airline")["champion"]
    r = c.get("/api/judge/retail").json()
    assert (r["cycles"], r["registry"]) == ([], None)
