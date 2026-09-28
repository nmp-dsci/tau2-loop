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
