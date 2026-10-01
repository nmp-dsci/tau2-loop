"""The tool judge (s11): J0's labels on a fixture conversation and on the committed runs."""

from __future__ import annotations

import json
from typing import Any

from tau2_loop.config import RUNS_DIR
from tau2_loop.data.splits import read_task_extract
from tau2_loop.llm.core import SdkResult
from tau2_loop.tooljudge import core as judge_core
from tau2_loop.tooljudge import gold, labels, replay
from tau2_loop.tooljudge import prompt as judge_prompt

KINDS = {
    "get_user_details": "read",
    "book_reservation": "write",
    "cancel_reservation": "write",
    "transfer_to_human_agents": "generic",
}
GOLD = [
    {"name": "get_user_details", "arguments": {"user_id": "u1"}, "compare_args": None},
    {
        "name": "book_reservation",
        "arguments": {"user_id": "u1", "origin": "JFK", "destination": "SFO"},
        "compare_args": None,
    },
]


def _trace(
    destination: str, reward: float, plan: str = "Shall I proceed with JFK to SFO?"
) -> dict[str, Any]:
    return {
        "task_id": "7",
        "trial": 1,
        "reward_info": {"reward": reward},
        "messages": [
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Book me JFK to SFO."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "name": "get_user_details", "arguments": {"user_id": "u1"}}
                ],
            },
            {"role": "tool", "id": "c1", "content": "{}"},
            {"role": "assistant", "content": "Would you like me to look at your options?"},
            {"role": "user", "content": "sure"},
            {"role": "assistant", "content": plan},
            {"role": "user", "content": "yes"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "book_reservation",
                        "arguments": {"user_id": "u1", "origin": "JFK", "destination": destination},
                    }
                ],
            },
            {"role": "tool", "id": "c2", "content": "{}"},
            {"role": "assistant", "content": "Done."},
        ],
    }


def _label(
    t: dict[str, Any], slips: frozenset[tuple[str, str, int]] = frozenset()
) -> dict[str, Any]:
    return labels.label_conversation("r1", {"agent": "v9"}, "7.json", t, GOLD, KINDS, slips)


def test_a_right_plan_and_its_good_write_are_allowed_and_pinned() -> None:
    c = _label(_trace("SFO", 1.0))
    kinds = [(cp["msg"], cp["kind"], cp["label"]) for cp in c["checkpoints"]]
    assert kinds == [(4, "reply", "allow"), (6, "plan", "right"), (8, "write", "good")]
    assert c["mode"] == "pass" and c["first_wrong"] is None
    assert all(cp["pinned"] for cp in c["checkpoints"])


def test_a_wrong_write_makes_its_plan_wrong_and_the_plan_the_first_wrong_step() -> None:
    c = _label(_trace("JFK", 0.0))
    plan = next(cp for cp in c["checkpoints"] if cp["kind"] == "plan")
    assert (plan["label"], plan["pinned"], plan["writes"]) == ("wrong", False, [8])
    assert c["mode"] == "wrong_write"
    assert c["first_wrong"] == {"msg": 6, "kind": "plan"}
    # in a failed conversation, a flagged reply that is not a plan is left to the golden answers
    assert next(cp for cp in c["checkpoints"] if cp["kind"] == "reply")["label"] == "unlabelled"


def test_a_hand_labelled_slip_keeps_the_plan_right_in_spirit() -> None:
    c = _label(_trace("JFK", 0.0), frozenset({("r1", "7", 8)}))
    assert next(cp for cp in c["checkpoints"] if cp["kind"] == "plan")["label"] == "slip"


def test_a_plan_the_trigger_misses_is_reached_only_at_its_write() -> None:
    c = _label(_trace("JFK", 0.0, plan="Booking JFK to SFO now, one moment."))
    plan = next(cp for cp in c["checkpoints"] if cp["kind"] == "plan")
    assert (plan["trigger"], plan["live"]) == (False, False)
    assert c["first_wrong"] == {"msg": 8, "kind": "write"}


def test_an_errored_write_has_no_plan_and_changes_nothing() -> None:
    t = _trace("SFO", 0.0)
    t["messages"][9]["content"] = "Error: flight not available"
    c = _label(t)
    assert [cp["kind"] for cp in c["checkpoints"]] == ["reply", "reply", "write"]
    assert c["checkpoints"][-1]["label"] == "errored"
    assert c["mode"] == "missed_write_other"


def test_a_route_perturbation_sends_the_trip_back_to_its_origin() -> None:
    got = labels._perturb_route(
        "Outbound: HAT023, JFK → SFO, May 26", {"origin": "JFK", "destination": "SFO"}
    )
    assert got is not None
    assert got[0] == "Outbound: HAT023, JFK → JFK, May 26"


def test_match_agrees_with_tau2_on_every_committed_train_write() -> None:
    from tau2.data_model.message import ToolCall
    from tau2.data_model.tasks import Action

    ext = read_task_extract("airline")
    kinds = {t["name"]: t.get("type") for t in ext["tools"]}
    gold = {
        str(t["id"]): (t.get("evaluation_criteria") or {}).get("actions") or []
        for t in ext["tasks"]
    }
    pairs = 0
    for c in (labels.read_labels("airline") or {})["conversations"][:60]:
        t = json.loads((RUNS_DIR / c["run"] / "traces" / c["trace"]).read_text())
        for m in t["messages"]:
            for tc in m.get("tool_calls") or []:
                if kinds.get(tc["name"]) != "write":
                    continue
                for g in gold[c["task"]]:
                    ours = labels.match(g, tc["name"], tc.get("arguments") or {})
                    theirs = Action(
                        **{**g, "action_id": g.get("action_id", "x")}
                    ).compare_with_tool_call(
                        ToolCall(id="x", name=tc["name"], arguments=tc.get("arguments") or {})
                    )
                    assert ours == theirs
                    pairs += 1
    assert pairs > 50


def test_the_committed_runs_reproduce_the_plan_pages_counts() -> None:
    data = labels.build("airline")
    tr = data["summary"]["train"]
    assert (tr["conversations"], tr["passed"], tr["failed"]) == (206, 151, 55)
    assert tr["plans"] == {"right": 83, "wrong": 15, "slip": 4}
    assert tr["writes"] == {"good": 117, "bad": 20, "errored": 4}
    assert tr["trigger_fires"] == {
        "plans": 101,
        "plans_missed": 1,
        "replies": 416,
        "other_text_replies": 1153,
    }
    assert data["folds"]["F4"] == ["11", "14", "20", "25", "46"]
    gate = data["summary"]["by_half"]["gate"]
    assert (gate["passed"], gate["plans"]["wrong"], gate["first_wrong"]) == (
        55,
        5,
        {"plan": 4, "write": 3},
    )
    assert data["summary"]["test"]["conversations"] == 125  # counted, never written
    assert {c["split"] for c in data["conversations"]} == {"train"}
    assert all(s["fold"] and s["label"] == "wrong" for s in data["synthetic"])


# ── J1: the annotator's machine checks ───────────────────────────────────────

POLICY = (
    "Before taking any action that updates the booking database, obtain explicit user confirmation."
)


def _answer(conv: dict[str, Any], **over: Any) -> dict[str, Any]:
    cps = []
    for cid, cp in gold.checkpoint_ids(conv):
        block = cp["label"] in ("wrong", "bad")
        cps.append(
            {
                "id": cid,
                "verdict": "block" if block else "allow",
                "check": 2 if block else None,
                "rule": "transcript" if block else None,
                "evidence": [{"msg": 1, "quote": "Book me JFK to SFO."}] if block else [],
                "detectable": True if block else None,
                "why": "one sentence",
                "fix": "Book SFO." if block else None,
            }
        )
    fw = (
        None
        if conv["passed"]
        else {"msg": 6, "kind": "plan", "blame": "agent", "why": "wrong destination"}
    )
    return {"summary": "s", "first_wrong_step": fw, "checkpoints": cps, **over}


def test_a_sound_golden_answer_passes_every_machine_check() -> None:
    t = _trace("JFK", 0.0)
    c = _label(t)
    assert gold.check_answer(_answer(c), c, t["messages"], POLICY) == []


def test_the_machine_checks_catch_a_flipped_pin_a_made_up_quote_and_a_missing_first_step() -> None:
    t = _trace("JFK", 0.0)
    c = _label(t)
    a = _answer(c, first_wrong_step=None)
    write = next(x for x in a["checkpoints"] if x["verdict"] == "block" and x["id"] == "c3")
    write["verdict"] = "allow"  # a bad write is pinned block
    plan = next(x for x in a["checkpoints"] if x["id"] == "c2")
    plan["evidence"] = [{"msg": 1, "quote": "Book me JFK to LAX."}]
    plan["rule"] = "Agents must always book the cheapest flight."
    problems = gold.check_answer(a, c, t["messages"], POLICY)
    assert any("pinned block" in p for p in problems)
    assert any("not verbatim in message 1" in p for p in problems)
    assert any("not verbatim in the policy" in p for p in problems)
    assert any("needs its first wrong step" in p for p in problems)


class FakeQuery:
    """llm.core.run_query stand-in: returns the queued structured replies in order, and records each ask."""

    def __init__(self, replies: list[Any]) -> None:
        self.replies, self.asks = list(replies), []  # type: ignore[var-annotated]

    def __call__(
        self, system: str, blocks: Any, model: str, effort: str, output_format: Any = None
    ) -> SdkResult:
        self.asks.append((system, blocks, model, effort, output_format))
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return SdkResult(
            json.dumps(r) if r is not None else "", 100, 10, None, 5, "s", structured=r
        )


def test_a_failing_answer_is_asked_for_again_once_with_its_problems_named() -> None:
    t = _trace("JFK", 0.0)
    c = _label(t)
    c.update(fold=1, half="read")
    bad = _answer(c, first_wrong_step=None)
    q = FakeQuery([bad, _answer(c)])
    rec = gold.annotate_one(c, t, {"evaluation_criteria": {"actions": GOLD}}, "system", q)
    assert (rec["attempts"], rec["problems"]) == (2, [])
    assert "needs its first wrong step" in q.asks[1][1][-1]
    assert q.asks[0][4]["schema"] is gold.SCHEMA
    # the wrong plan and the first wrong step are what a person checks
    assert [it["asks"] for it in rec["review"]] == [
        ["plan · wrong or a slip", "the first wrong step"]
    ]


# ── J2: the judge's input, its call, and the scorer ──────────────────────────


def test_the_judges_input_stops_before_the_reply_and_never_carries_gold() -> None:
    t = _trace("JFK", 0.0)
    b = judge_prompt.blocks(t["messages"], 6, "Shall I book JFK to JFK?")
    assert (
        b[-1].startswith("## The agent's proposed reply")
        and "[6] agent: Shall I book JFK to JFK?" in b[-1]
    )
    assert not any(
        x.startswith("[6]") or x.startswith("[8]") for x in b[1:-1]
    )  # nothing at or after the checkpoint
    j = judge_prompt.load("airline", "j1")
    system = judge_prompt.system_prompt(
        j, POLICY, [{"name": "book_reservation", "type": "write", "description": "Book."}]
    )
    assert "gold" not in " ".join(b).lower() and "compare_args" not in system


def test_a_block_stands_only_above_the_threshold_and_with_a_real_rule() -> None:
    raw = {"verdict": "block", "confidence": 0.9, "rule": "obtain explicit user confirmation"}
    assert judge_core.decide(raw, 0.5, POLICY) == ("block", None)
    assert judge_core.decide({**raw, "confidence": 0.3}, 0.5, POLICY)[0] == "allow"
    assert judge_core.decide({**raw, "rule": "never fly on Tuesdays"}, 0.5, POLICY) == (
        "allow",
        "the rule is not verbatim in the policy",
    )
    assert judge_core.decide({**raw, "rule": "transcript"}, 0.5, POLICY)[0] == "block"


def test_the_judge_fails_open_on_an_error_or_an_unparsable_verdict() -> None:
    j = judge_prompt.load("airline", "j1")
    msgs = _trace("JFK", 0.0)["messages"]
    for q in (FakeQuery([RuntimeError("cli died")]), FakeQuery([None])):
        v = judge_core.judge_reply(j, "system", POLICY, msgs, 6, "Shall I?", query=q)
        assert v["verdict"] == "allow" and "failed open" in v["error"]


def test_an_allow_everything_judge_scores_a_half_and_a_perfect_one_stops_every_wrong_plan() -> None:
    items = [i for i in replay.items("airline") if i["item"] == "checkpoint"]
    allow = [{**i, "verdict": "allow"} for i in items]
    s = replay.score("airline", allow)["scores"]
    assert s["all"]["balanced_accuracy"] == 0.5
    assert s["all"]["passes_interrupted"]["k"] == 0
    first = {c["key"]: c["first_wrong"] for c in replay.score("airline", allow)["conversations"]}
    perfect = [
        {**i, "verdict": "block" if (first.get(i["key"]) or {}).get("msg") == i["msg"] else "allow"}
        for i in items
    ]
    p = replay.score("airline", perfect)["scores"]["all"]
    assert p["wrong_plans_stopped"]["k"] == p["wrong_plans_stopped"]["n"] > 0
    assert p["balanced_accuracy"] == 1.0


# ── the viewer's judge routes ────────────────────────────────────────────────


def _client():  # type: ignore[no-untyped-def]
    from fastapi.testclient import TestClient

    from tau2_loop.serving.app import create_app

    return TestClient(create_app())


def test_a_labelled_train_conversation_carries_the_judges_evidence_and_a_test_one_none() -> None:
    c = _client()
    j = c.get("/api/runs/20260928T075613Z_airline_v4_train/44/t1").json()["judge"]
    assert (
        j["labels"]["first_wrong"] == {"msg": 34, "kind": "plan"} and j["labels"]["half"] == "gate"
    )
    test_conv = c.get("/api/runs/20260928T073602Z_airline_v3_test/29/t1")
    assert (
        test_conv.status_code == 200 and test_conv.json()["judge"] is None
    )  # test is never labelled


def test_the_golden_review_write_is_refused_in_the_demo_image_and_without_a_database(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    body = {"item_id": "x/1/t1#2", "verdict": "agree"}
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    monkeypatch.setenv("RO_DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    c = _client()
    assert c.post("/api/review/golden/airline", json=body).status_code == 503
    queue = c.get("/api/judge/airline/gold").json()
    assert queue["writable"] is False and queue["current"] == {}
    monkeypatch.setenv("DEMO_MODE", "1")
    assert _client().post("/api/review/golden/airline", json=body).status_code == 403
