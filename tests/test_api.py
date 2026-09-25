"""The read-only API over the committed files, offline."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tau2_loop.serving.app import create_app, trace_events


def client() -> TestClient:
    return TestClient(create_app())


def test_healthz_lists_domains_and_mode(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DEMO_MODE", "1")
    r = client().get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "demo" and body["domains"] == [
        "airline",
        "retail",
        "telecom",
        "banking_knowledge",
    ]


def test_domains_and_tasks() -> None:
    c = client()
    ds = c.get("/api/domains").json()
    assert [d["domain"] for d in ds] == ["airline", "retail", "telecom", "banking_knowledge"]
    assert all(d["train"] == 20 and d["test"] == 20 for d in ds)
    airline = c.get("/api/domains/airline").json()
    assert airline["policy_words"] > 1000 and len(airline["tasks"]) == 40
    first = airline["tasks"][0]
    t = c.get(f"/api/domains/airline/tasks/{first['id']}").json()
    assert t["id"] == first["id"] and t["split"] in {"train", "test"}
    assert c.get("/api/domains/nope").status_code == 404


def test_agents_and_registry() -> None:
    c = client()
    body = c.get("/api/agents").json()
    refs = {v["ref"] for v in body["versions"]}
    assert {"airline/v0", "retail/v0", "telecom/v0", "banking_knowledge/v0"} <= refs
    assert set(body["registry"]) == {"airline", "retail", "telecom", "banking_knowledge"}
    v = c.get("/api/agents/airline/v0").json()
    assert "{policy}" in v["files"]["system.md"]
    d = c.get("/api/agents/diff", params={"domain": "airline", "a": "v0", "b": "v0"}).json()
    assert all(not f["changed"] for f in d["files"])


def test_runs_ledger_experiments_shapes() -> None:
    c = client()
    assert isinstance(c.get("/api/runs").json(), list)
    assert set(c.get("/api/ledger").json()) == {"airline", "retail", "telecom", "banking_knowledge"}
    assert "runs" in c.get("/api/experiments").json()
    assert c.get("/api/runs/nope").status_code == 404


def test_trace_endpoint_404s_on_directory_name_instead_of_crashing() -> None:
    c = client()
    run_id = "20260915T014228Z_mock_v0_all"
    ok = c.get(f"/api/runs/{run_id}/traces/create_task_1.json")
    assert ok.status_code == 200
    assert c.get(f"/api/runs/{run_id}/traces/%2e%2e").status_code == 404
    assert c.get(f"/api/runs/{run_id}/traces/nope.json").status_code == 404


def test_trace_events_flatten_messages() -> None:
    t = {
        "messages": [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "book"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"name": "get_user", "arguments": {"id": 1}}],
            },
            {"role": "tool", "content": "{}", "error": False},
        ]
    }
    ev = trace_events(t)
    assert [e["type"] for e in ev] == ["assistant", "user", "tool_call", "tool_result"]
    assert ev[2]["name"] == "get_user"


def test_stats_is_the_overview_in_one_object() -> None:
    s = client().get("/api/stats").json()
    assert [d["domain"] for d in s["domains"]] == [
        "airline",
        "retail",
        "telecom",
        "banking_knowledge",
    ]
    # the four base splits are the benchmark: 50 + 114 + 114 + 97
    assert s["base_total"] == 375
    assert all(d["train"] == 20 and d["test"] == 20 for d in s["domains"])
    assert s["runs_scored"] <= s["runs"] and s["conversations"] > 0


def test_rubric_counts_the_checks_and_what_failed() -> None:
    r = client().get("/api/rubric").json()
    airline = next(d for d in r["by_domain"] if d["domain"] == "airline")
    assert airline["n_tasks"] == 40
    # every airline task is judged on NL assertions; only some carry expected actions
    assert airline["uses"]["nl_assertions"] == 40
    assert 0 < airline["uses"]["actions"] <= 40
    f = r["failures"]
    assert f["failed"] > 0
    assert set(f["by_check"]) == {
        "db_check",
        "actions",
        "communicate_info",
        "nl_assertions",
        "error",
    }
    # a conversation can miss several checks, so the mix never sums below the worst one
    assert max(f["by_check"].values()) <= f["failed"]


def test_run_detail_carries_a_profile() -> None:
    c = client()
    run_id = c.get("/api/runs").json()[0]["run_id"]
    body = c.get(f"/api/runs/{run_id}").json()
    p = body["profile"]
    assert p["n"] == len(body["results"])
    turns = p["metrics"]["turns"]
    # the distribution is ordered by construction, and the sum is the total
    assert turns["p50"] <= turns["p95"] <= turns["max"]
    assert turns["sum"] == sum(r["n_agent_turns"] for r in body["results"])


def test_a_conversation_is_addressed_by_task_and_trial() -> None:
    c = client()
    run_id = next(r["run_id"] for r in c.get("/api/runs").json() if r["summary"])
    row = c.get(f"/api/runs/{run_id}").json()["results"][0]
    r = c.get(f"/api/runs/{run_id}/{row['task_id']}/t{row['trial']}")
    assert r.status_code == 200
    body = r.json()
    assert body["task_id"] == row["task_id"] and body["result"]["trial"] == row["trial"]
    assert c.get(f"/api/runs/{run_id}/{row['task_id']}/t99").status_code == 404
    assert c.get(f"/api/runs/{run_id}/{row['task_id']}/nope").status_code == 404
    assert c.get(f"/api/runs/no-such-run/{row['task_id']}/t1").status_code == 404


def test_healthz_says_whether_a_review_can_be_recorded(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DEMO_MODE", "1")
    # the demo image is never writable, whatever the database says
    assert client().get("/healthz").json()["writable"] is False


def test_the_review_routes_degrade_without_a_database(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    monkeypatch.setenv("RO_DATABASE_URL", "postgresql://nobody:nobody@127.0.0.1:1/none")
    c = client()
    body = c.get("/api/review").json()
    # the tab still renders: an empty list and the remedy, never a 500
    assert body["writable"] is False and body["current"] == {}
    assert "nmp-central-ai" in body["reason"]
    assert c.post("/api/review/run/task/t1", json={"verdict": "agree"}).status_code == 503


def test_a_review_is_refused_in_the_demo_image(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("DEMO_MODE", "1")
    r = client().post("/api/review/run/task/t1", json={"verdict": "agree"})
    assert r.status_code == 403
