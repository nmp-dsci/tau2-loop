"""The Agent tab's playground: a tool call run against a rebuilt environment.

It trusts the replay (`Environment.set_state` re-running every earlier write), so
the replay is what these pin: a whole committed run rebuilds with `strict=True`,
and each recorded write re-runs to its recorded result. Then the route's bounds,
and that nothing it does reaches the disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tau2_loop.config import RUNS_DIR
from tau2_loop.eval import replay
from tau2_loop.eval.results import read_results
from tau2_loop.serving.app import create_app

pytestmark = pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")

RUN = "20260915T132148Z_airline_v2_train"


def conversations(run_id: str) -> list[tuple[str, replay.Conversation]]:
    meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
    return [
        (r.task_id, replay.load_conversation(run_id, r.trace, meta["domain"], r.task_id))
        for r in read_results(RUNS_DIR / run_id / "results.jsonl")
    ]


def test_every_conversation_in_a_run_rebuilds_strictly() -> None:
    """strict=True makes tau2 raise if any replayed write returns other than it did live."""
    convs = conversations(RUN)
    assert len(convs) == 20
    for _, conv in convs:
        replay.build_env(conv, len(conv.messages), strict=True)


def test_each_recorded_write_re_runs_to_its_recorded_result() -> None:
    writes = 0
    for _, conv in conversations(RUN):
        for i, m in enumerate(conv.messages):
            for k, tc in enumerate(getattr(m, "tool_calls", None) or []):
                if tc.name not in {
                    "book_reservation",
                    "cancel_reservation",
                    "update_reservation_flights",
                    "update_reservation_passengers",
                    "update_reservation_baggages",
                    "send_certificate",
                }:
                    continue
                out = replay.run_tool(
                    conv, name=tc.name, arguments=tc.arguments, at=i, after_calls=k
                )
                assert out["same_as_recorded"] is True, (tc.name, i)
                writes += 1
    assert writes > 0


def test_the_call_the_agent_refused_on_task_22() -> None:
    """The s05 walkthrough: v2 would not change a basic-economy cabin; τ² accepts the call."""
    conv = dict(conversations(RUN))["22"]
    gold = next(
        a for a in conv.task.evaluation_criteria.actions if a.name == "update_reservation_flights"
    )
    out = replay.run_tool(conv, name=gold.name, arguments=gold.arguments, at=len(conv.messages))
    assert out["error"] is False and out["wrote"] is True and out["same_as_recorded"] is None
    fields = {
        (d["record"], f["field"]): (f["before"], f["after"])
        for d in out["diff"]
        for f in d["fields"]
    }
    assert fields[("reservations.FQ8APE", "cabin")] == ('"basic_economy"', '"economy"')
    paid = fields[("users.omar_rossi_1241", "payment_methods.gift_card_8190333.amount")]
    assert float(paid[0]) - float(paid[1]) == 209.0


def test_the_route_refuses_what_it_cannot_serve() -> None:
    c = TestClient(create_app())
    url = f"/api/runs/{RUN}/22/t1/tool"
    n = len(c.get(f"/api/runs/{RUN}/22/t1").json()["messages"])
    assert (
        c.post(url, json={"name": "drop_everything", "arguments": {}, "at": 2}).status_code == 422
    )
    assert (
        c.post(
            url, json={"name": "get_user_details", "arguments": {"x": "y" * 9000}, "at": 2}
        ).status_code
        == 422
    )
    assert (
        c.post(url, json={"name": "get_user_details", "arguments": {}, "at": n + 1}).status_code
        == 422
    )
    # message 5 is the result of the call in message 4: state is undefined between the two
    mid = c.post(
        url, json={"name": "get_user_details", "arguments": {"user_id": "omar_rossi_1241"}, "at": 5}
    )
    assert mid.status_code == 422 and "between a tool call and its result" in mid.json()["detail"]
    assert (
        c.post(
            f"/api/runs/{RUN}/22/t9/tool", json={"name": "get_user_details", "at": 0}
        ).status_code
        == 404
    )


def test_without_tau2_the_route_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay, "available", lambda: False)
    c = TestClient(create_app())
    assert c.get("/healthz").json()["playground"] is False
    r = c.post(f"/api/runs/{RUN}/22/t1/tool", json={"name": "get_user_details", "at": 0})
    assert r.status_code == 503 and "demo image" in r.json()["detail"]


def test_a_write_in_the_playground_touches_no_file() -> None:
    from tau2.domains.airline.utils import AIRLINE_DB_PATH

    watched = [Path(AIRLINE_DB_PATH), *sorted((RUNS_DIR / RUN).rglob("*.json*"))]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in watched]
    c = TestClient(create_app())
    body = c.get(f"/api/runs/{RUN}/22/t1").json()
    gold = body["task"]["evaluation_criteria"]["actions"][0]
    r = c.post(
        f"/api/runs/{RUN}/22/t1/tool",
        json={"name": gold["name"], "arguments": gold["arguments"], "at": len(body["messages"])},
    )
    assert r.status_code == 200 and r.json()["wrote"] is True
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in watched] == before


def _diff(run_id: str, task: str) -> dict:
    meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
    return replay.db_diff(replay.load_conversation(run_id, f"{task}.json", meta["domain"], task))


def test_the_db_diff_names_a_missed_write_a_wrong_write_and_a_wrong_argument() -> None:
    """Both sides of tau2's DB check, rebuilt: each differing field, and who changed its record."""
    # v6 task 39: gold cancels MSJ4OA; the agent never wrote to it
    missed = _diff("20260928T101605Z_airline_v6_train", "39")
    assert missed["match"] is False
    (rec,) = missed["records"]
    assert rec["record"] == "reservations.MSJ4OA" and rec["agent_calls"] == []
    assert [g["action_id"] for g in rec["gold_actions"]] == ["39_10"]
    status = next(f for f in rec["fields"] if f["field"] == "status")
    assert (status["before"], status["agent"], status["gold"]) == (None, None, '"cancelled"')
    # v4 task 6: the agent's second booking at message 50 made every difference
    wrong = _diff("20260928T075613Z_airline_v4_train", "6")
    assert {r["record"]: r["kind"] for r in wrong["records"]}["reservations.HATHAT"] == "agent_only"
    assert all([c["msg"] for c in r["agent_calls"]] == [50] for r in wrong["records"])
    assert all(r["gold_actions"] == [] for r in wrong["records"])
    # v5 task 14: both book HATHAT; the agent's destination is JFK where gold's is SFO
    args = _diff("20260928T094017Z_airline_v5_train", "14")
    (rec,) = args["records"]
    assert rec["fields"] == [
        {"field": "destination", "before": None, "agent": '"JFK"', "gold": '"SFO"'}
    ]
    assert rec["agent_calls"][0]["msg"] == 24 and rec["gold_actions"][0]["action_id"] == "14_1"
    # each record says which of the three it is, the word the viewer and the optimiser both read
    assert [r["verdict"] for r in missed["records"]] == ["missed write"]
    assert {r["verdict"] for r in wrong["records"]} == {"wrong write"}
    assert [r["verdict"] for r in args["records"]] == ["wrong arguments"]


def _acts(run_id: str, task: str) -> dict:
    meta = json.loads((RUNS_DIR / run_id / "run.json").read_text())
    return replay.action_diff(
        replay.load_conversation(run_id, f"{task}.json", meta["domain"], task)
    )


def test_the_action_diff_pairs_expected_actions_with_calls_as_tau2_does() -> None:
    """Matched or missing as tau2's action check has it; a missing write's nearest call names the
    argument that differs; a write no expected action matches is listed, refused or not."""
    for run_id, task in [
        ("20260928T101605Z_airline_v6_train", "39"),
        ("20260928T075613Z_airline_v4_train", "6"),
        ("20260928T094017Z_airline_v5_train", "14"),
        ("20260928T060029Z_airline_v3_train", "25"),
    ]:
        trace = json.loads((RUNS_DIR / run_id / "traces" / f"{task}.json").read_text())
        tau2_says = [c["action_match"] for c in trace["reward_info"]["action_checks"]]
        got = _acts(run_id, task)
        assert [e["matched_at"] is not None for e in got["expected"]] == tau2_says
        # airline grades the database and what was said, never the actions themselves
        assert got["graded"] is False and got["basis"] == ["DB", "COMMUNICATE"]

    # v6 task 39: the third cancel is missing, and both cancels the agent made are other actions'
    a = _acts("20260928T101605Z_airline_v6_train", "39")
    (miss,) = [e for e in a["expected"] if e["matched_at"] is None]
    assert (miss["action_id"], miss["write"], miss["nearest"]) == ("39_10", True, None)
    assert "cancel_reservation" in a["called"] and a["unexpected_writes"] == []
    # v5 task 14: the booking at message 24 differs from 14_1 in its destination only
    a = _acts("20260928T094017Z_airline_v5_train", "14")
    (miss,) = [e for e in a["expected"] if e["matched_at"] is None]
    assert miss["action_id"] == "14_1" and miss["nearest"]["msg"] == 24
    assert miss["nearest"]["differs"] == [
        {"argument": "destination", "agent": '"JFK"', "expected": '"SFO"'}
    ]
    assert [(u["msg"], u["name"]) for u in a["unexpected_writes"]] == [(24, "book_reservation")]
    # v4 task 6: one expected lookup, matched; the second booking is a write gold never makes
    a = _acts("20260928T075613Z_airline_v4_train", "6")
    assert [(u["msg"], u["name"], u["refused"]) for u in a["unexpected_writes"]] == [
        (50, "book_reservation", False)
    ]
    # v3 task 25: the booking the API refused is still a write no expected action matches
    a = _acts("20260928T060029Z_airline_v3_train", "25")
    assert any(u["name"] == "book_reservation" and u["refused"] for u in a["unexpected_writes"])
    # a task whose basis has ACTION says so
    assert _acts("20260915T014228Z_mock_v0_all", "update_task_with_user_tools")["graded"] is True


def test_a_passing_conversation_has_no_db_diff_and_the_route_agrees_with_the_grade() -> None:
    assert _diff(RUN, "0")["records"] == []
    c = TestClient(create_app())
    got = c.get("/api/runs/20260928T101605Z_airline_v6_train/39/t1/db").json()
    assert got["match"] is False and got["graded"] is False and got["records"]
