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
from tau2_loop.tooljudge import review as judge_review

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
    assert (tr["conversations"], tr["passed"], tr["failed"]) == (165, 129, 36)
    assert tr["plans"] == {"right": 80, "wrong": 15, "slip": 4}
    assert tr["writes"] == {"good": 114, "bad": 20, "errored": 4}
    assert tr["trigger_fires"] == {
        "plans": 98,
        "plans_missed": 1,
        "replies": 358,
        "other_text_replies": 964,
    }
    assert data["folds"]["F4"] == ["11", "14", "20", "25", "46"]
    gate = data["summary"]["by_half"]["gate"]
    assert (gate["passed"], gate["plans"]["wrong"], gate["first_wrong"]) == (
        50,
        5,
        {"plan": 4, "write": 3},
    )
    assert data["summary"]["test"]["conversations"] == 125  # counted, never written
    assert {c["split"] for c in data["conversations"]} == {"train"}
    assert all(s["fold"] and s["label"] == "wrong" for s in data["synthetic"])


def test_the_judge_is_called_only_at_a_write_or_a_transfer() -> None:
    """s11's second rule: a checkpoint is judged exactly when the agent's message issues a write or
    a transfer; a text reply, plan or not, never is."""
    data = labels.build("airline")
    cps = [cp for c in data["conversations"] for cp in c["checkpoints"]]
    assert all(cp["judged"] == (cp["kind"] in ("write", "transfer")) for cp in cps)
    assert not any(cp["judged"] for cp in cps if cp["kind"] in ("plan", "reply"))
    assert data["summary"]["train"]["judged"] == {
        "checkpoints": 181,
        "messages": 174,
        "conversations": 118,
    }


def test_the_judge_learns_from_optimised_agents_only() -> None:
    """s11's rule: v0, the baseline before any loop cycle, never enters the labels, the golden
    answers or a replay's score, and the folds J3 already read stay where they were."""
    data = labels.build("airline")
    versions = {c["version"] for c in data["conversations"]}
    assert versions and not versions & labels.UNOPTIMISED
    assert not any("_v0_" in c["run"] for c in data["conversations"])
    assert not labels.optimised({"agent": "v0"}) and labels.optimised({"agent": "v3"})
    keys = {c["key"] for c in data["conversations"]}
    assert {r["key"] for r in gold.read_gold("airline")} <= keys
    assert data["folds"] == {
        f"F{f}": sorted(
            (t for t, k in labels.PINNED_FOLDS["airline"].items() if k == f),
            key=lambda t: int(t),
        )
        for f in range(1, 6)
    }
    for r in replay.list_replays("airline"):
        scored = replay.score(
            "airline", replay.read_verdicts(replay.JUDGE_RUNS_DIR / r["replay_id"])
        )
        assert not any("_v0_" in row["key"] for row in scored["rows"])


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
    # the wrong plan is a text reply the judge never sees, and the bad write it led to is pinned
    # by structure, so nothing here is left for a person to decide
    assert {cp["kind"] for cp in c["checkpoints"] if cp["judged"]} == {"write"}
    assert rec["review"] == []


def test_a_pass_is_written_by_the_passed_rule_and_only_a_failure_calls_the_model(
    monkeypatch, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """A new experiment's passes cost nothing: the grader confirmed them, so `make judge-gold`
    writes `allow` at every checkpoint with no model call, and asks the annotator about failures."""
    ok = _label(_trace("SFO", 1.0))
    t = _trace("JFK", 0.0)
    bad = labels.label_conversation("r2", {"agent": "v9"}, "7.json", t, GOLD, KINDS)
    for c in (ok, bad):
        c.update(fold=1, half="read")
    (tmp_path / "r2" / "traces").mkdir(parents=True)
    (tmp_path / "r2" / "traces" / "7.json").write_text(json.dumps(t))
    monkeypatch.setattr(gold, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(gold, "read_labels", lambda d: {"conversations": [ok, bad]})
    monkeypatch.setattr(
        gold,
        "read_task_extract",
        lambda d: {
            "tasks": [{"id": "7", "evaluation_criteria": {"actions": GOLD}}],
            "policy": POLICY,
        },
    )
    monkeypatch.setattr(gold, "gold_path", lambda d: tmp_path / "gold.jsonl")
    monkeypatch.setattr(gold, "summary_path", lambda d: tmp_path / "summary.json")
    q = FakeQuery([_answer(bad)])
    gold.run("airline", concurrency=1, query=q, log=lambda m: None)
    assert len(q.asks) == 1 and q.asks[0][1][0]
    recs = {r["key"]: r for r in gold.read_gold("airline")}
    rule = recs[ok["key"]]
    assert rule["annotator"]["model"] == "rule" and rule["attempts"] == 0
    assert {a["verdict"] for a in rule["answer"]["checkpoints"]} == {"allow"}
    assert rule["answer"]["first_wrong_step"] is None and rule["review"] == []
    assert recs[bad["key"]]["annotator"]["model"] == gold.ANNOTATOR_MODEL


def test_the_passed_rule_allows_every_write_and_transfer_whatever_the_annotator_said() -> None:
    c = _label(_trace("SFO", 1.0))
    a = _answer(c)
    for x in a["checkpoints"]:
        x["verdict"] = "block"
    rec = {"answer": a, "human": {}}
    eff = judge_review.effective_verdicts(rec, c)
    judged = {cid for cid, cp in gold.checkpoint_ids(c) if cp["judged"]}
    assert judged and all(eff[cid] == "allow" for cid in judged)
    # the retired plan judge's text replies keep the annotator's answer, so its history stands
    assert all(eff[cid] == "block" for cid in eff if cid not in judged)
    c["passed"] = False
    assert all(v == "block" for v in judge_review.effective_verdicts(rec, c).values())


def test_a_check_is_taken_back_only_whole_and_a_conversation_is_checked_as_a_whole(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import pytest

    with pytest.raises(ValueError, match="carries no correction"):
        judge_review.add("airline", "r/1/t1#4", "withdraw", {"verdict": "allow"})
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    monkeypatch.setenv("RO_DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    c = _client()
    convs = c.get("/api/judge/airline/evals").json()["conversations"]
    callless = next(x for x in convs if not x["passed"] and not x["calls"])
    case = c.get("/api/judge/airline/gold/item", params={"id": f"{callless['key']}#-1"}).json()
    assert case["asks"] == ["the conversation as a whole"] and case["checkpoint"] is None
    bad = c.get("/api/judge/airline/gold/item", params={"id": f"{callless['key']}#-2"})
    assert bad.status_code == 404


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


def test_a_golden_answer_opens_for_correction_at_any_write_or_transfer_never_a_text_reply(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """A person corrects a golden answer where they read it: a case opens at any write or transfer
    the judge reviews, queued or not, and never at a text reply, where the judge is not called."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    monkeypatch.setenv("RO_DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    c = _client()
    key = "20260915T124751Z_airline_v1_train/14/t1"
    queued = {i["id"] for i in c.get("/api/judge/airline/gold").json()["items"]}
    j = c.get(f"/api/runs/{key}").json()["judge"]
    assert j["checks"] == {} and j["writable"] is False and j["reason"]
    judged = [cp["msg"] for cp in j["labels"]["checkpoints"] if cp["judged"]]
    replies = [cp["msg"] for cp in j["labels"]["checkpoints"] if cp["kind"] in ("plan", "reply")]
    off = [m for m in judged if f"{key}#{m}" not in queued]
    assert off and replies, "task 14 has an unqueued write and a text reply"
    case = c.get("/api/judge/airline/gold/item", params={"id": f"{key}#{off[0]}"}).json()
    assert case["asks"] == ["opened from the conversation"]
    assert case["checkpoint"]["kind"] in ("write", "transfer")
    for m in replies:
        assert c.get("/api/judge/airline/gold/item", params={"id": f"{key}#{m}"}).status_code == 404
    assert {
        i["id"].rsplit("#", 1)[1]
        for i in c.get("/api/judge/airline/gold").json()["items"]
        if i["key"] == key
    } <= {str(m) for m in judged}


def test_the_newest_correction_of_the_first_wrong_step_wins() -> None:
    fw = {"msg": 10, "kind": "transfer", "blame": "agent", "why": "made-up tool problems"}
    older = {
        "verdict": "correct",
        "correction": {"first_wrong_msg": None},
        "created_at": "2026-10-02 09:00:00+00",
    }
    newer = {
        "verdict": "correct",
        "correction": {"first_wrong_msg": 14},
        "created_at": "2026-10-02 10:00:00+00",
    }
    rec = {"answer": {"first_wrong_step": fw}, "human": {"8": older, "10": newer}}
    got = judge_review.effective_first_wrong(rec)
    assert got is not None and got["msg"] == 14
    rec["human"] = {
        "8": newer | {"created_at": older["created_at"]},
        "10": older | {"created_at": newer["created_at"]},
    }
    assert judge_review.effective_first_wrong(rec) is None


def test_the_judge_agent_and_its_replays_are_listed_for_the_scope_bar() -> None:
    c = _client()
    vs = c.get("/api/judge/airline/versions").json()["versions"]
    refs = [v["ref"] for v in vs]
    assert refs[0] == "airline/plan/j1"
    assert all("## The checks" in v["files"]["judge.md"] and v["model"] for v in vs)
    # a version is listed only once it has a rubric: a rejected challenger's folder has none
    on_disk = sorted(p.parent.name for p in judge_prompt.JUDGES_DIR.glob("airline/plan/*/judge.md"))
    assert sorted(r.rsplit("/", 1)[1] for r in refs) == on_disk
    reps = c.get("/api/judge/airline/replays").json()
    assert reps and all(r["judge"] in refs for r in reps)
    # a domain with no judge yet is an empty list, never an error
    assert c.get("/api/judge/retail/versions").json()["versions"] == []
    assert c.get("/api/judge/retail/replays").json() == []


def test_evals_on_the_judge_lists_its_whole_eval_set_and_none_where_it_has_none() -> None:
    c = _client()
    e = c.get("/api/judge/airline/evals").json()
    convs = e["conversations"]
    # every conversation is listed; the judge is called only at a write or a transfer: 181 calls
    assert len(convs) == 165 and sum(x["checkpoints"] for x in convs) == 181
    labelled = {x["key"]: x for x in labels.read_labels("airline")["conversations"]}
    for x in convs:
        cps = labelled[x["key"]]["checkpoints"]
        assert [k["msg"] for k in x["calls"]] == sorted(
            {cp["msg"] for cp in cps if cp["kind"] in ("write", "transfer")}
        )
    # the passed rule: 129 passes, confirmed by the grader, allow every call and wait for no one
    passes = [x for x in convs if x["passed"]]
    assert len(passes) == 129 and all(x["review"] == [] for x in passes)
    assert {k["golden"] for x in passes for k in x["calls"]} == {"allow"}
    # a person confirms the 36 failures: each call in 25, and 11 with none as a whole
    fails = [x for x in convs if not x["passed"]]
    assert len(fails) == 36
    assert sum(x["review"] == [judge_review.WHOLE] for x in fails) == 11
    assert sum(len(x["review"]) for x in fails if x["calls"]) == 50
    # the annotator blocked a write in three passing task-25 runs; the grader overrules it
    assert sum(x["golden_blocks"] for x in convs) == 27
    first = {
        x["key"]: next((k["msg"] for k in x["calls"] if k["golden"] == "block"), None)
        for x in fails
    }
    assert first["20260928T075613Z_airline_v4_train/6/t1"] == 50
    assert sum(m is not None for m in first.values()) == 22
    assert e["pending"] == {"runs": [], "no_gold": {"failed": 0, "passed": 0}}
    assert {x["half"] for x in convs} == {"read", "gate"} and len(e["synthetic"]) == 31
    assert c.get("/api/judge/retail/evals").json()["conversations"] is None
    # nothing in a pass waits in the review queue either
    items = c.get("/api/judge/airline/gold").json()["items"]
    assert items and not any(i["passed"] for i in items)


def test_a_run_scored_after_the_judges_data_closed_never_joins_it_or_its_pending_queue(
    tmp_path: Any, monkeypatch: Any
) -> None:
    src = RUNS_DIR / "20260928T060029Z_airline_v3_train"
    new = "20261003T000000Z_airline_v8_train"
    assert labels.open_to_judge(src.name) and not labels.open_to_judge(new)
    run = tmp_path / new
    (run / "traces").mkdir(parents=True)
    meta = json.loads((src / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**meta, "agent": "v8"}))
    for f in (src / "traces").iterdir():
        (run / "traces" / f.name).write_text(f.read_text())
    monkeypatch.setattr(labels, "RUNS_DIR", tmp_path)
    assert [r for r, *_ in labels._scored_traces("airline")] == []
    assert labels.unlabelled_runs("airline") == []
