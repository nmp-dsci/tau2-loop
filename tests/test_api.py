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
