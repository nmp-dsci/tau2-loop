"""The read-only API over the committed files, offline."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tau2_loop.agent.compose import CLOCK_NOTE, compose
from tau2_loop.config import RUNS_DIR
from tau2_loop.serving.app import create_app, trace_events, trace_messages

V2_RUN = "20260915T132148Z_airline_v2_train"


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
    # split v2: half of each base set on each side; banking at split v3, 60 train / 37 test
    assert [(d["train"], d["test"]) for d in ds] == [(25, 25), (57, 57), (57, 57), (60, 37)]
    airline = c.get("/api/domains/airline").json()
    assert airline["policy_words"] > 1000 and len(airline["tasks"]) == 50
    first = airline["tasks"][0]
    t = c.get(f"/api/domains/airline/tasks/{first['id']}").json()
    assert t["id"] == first["id"] and t["split"] in {"train", "test"}
    assert c.get("/api/domains/nope").status_code == 404


def test_a_task_lists_its_conversation_in_every_run_that_played_it() -> None:
    """The scope bar's task on Runs: one row per run and trial, newest run first."""
    rows = client().get("/api/domains/airline/conversations", params={"task": "39"}).json()
    runs = [r["run_id"] for r in rows]
    assert runs == sorted(runs, reverse=True) and len(set(runs)) == len(runs)
    v6 = next(r for r in rows if r["run_id"] == "20260928T101605Z_airline_v6_train")
    assert (v6["agent"], v6["split"], v6["correct"], v6["db_check"]) == (
        "v6",
        "train",
        False,
        False,
    )
    # a train task is never in a test run
    assert all(r["split"] != "test" for r in rows)
    assert client().get("/api/domains/airline/conversations", params={"task": "nope"}).json() == []


def test_agents_and_registry() -> None:
    c = client()
    body = c.get("/api/agents").json()
    refs = {v["ref"] for v in body["versions"]}
    assert {"airline/v0", "retail/v0", "telecom/v0", "banking_knowledge/v1"} <= refs
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


def test_runs_carry_mean_tokens_per_conversation() -> None:
    c = client()
    row = next(
        r for r in c.get("/api/runs").json() if r["run_id"] == "20260915T014228Z_mock_v0_all"
    )
    tokens = row["tokens_per_conversation"]
    rows = [json.loads(line) for line in (RUNS_DIR / row["run_id"] / "results.jsonl").open()]
    total = sum(
        r["agent_input_tokens"]
        + r["agent_output_tokens"]
        + r["user_input_tokens"]
        + r["user_output_tokens"]
        for r in rows
    )
    assert tokens["all"] == round(total / len(rows))
    assert 0 < tokens["agent"] <= tokens["all"]


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
    # every base task is train or test (banking's reserve was dealt out at split v3)
    assert all(
        d["train"] + d["test"] + (d.get("reserve_n") or 0) == d["base_n"] for d in s["domains"]
    )
    assert s["runs_scored"] <= s["runs"] and s["conversations"] > 0


def test_rubric_counts_the_checks_and_what_failed() -> None:
    r = client().get("/api/rubric").json()
    airline = next(d for d in r["by_domain"] if d["domain"] == "airline")
    assert airline["n_tasks"] == 50  # the whole base set since split v2
    # every airline task is judged on NL assertions; only some carry expected actions
    assert airline["uses"]["nl_assertions"] == 50
    assert 0 < airline["uses"]["actions"] <= 50
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


def test_every_committed_tool_result_pairs_with_exactly_one_call() -> None:
    """The Agent tab pairs a result with its call by id; every committed trace must allow it."""
    checked = 0
    for trace in sorted(RUNS_DIR.glob("*/traces/*.json")):
        msgs = trace_messages(json.loads(trace.read_text()))
        calls = [c["id"] for m in msgs for c in m["tool_calls"]]
        results = [m["id"] for m in msgs if m["role"] == "tool"]
        assert len(calls) == len(set(calls)), f"{trace}: a call id repeats"
        assert sorted(results) == sorted(calls), (
            f"{trace}: a result without its call, or a call without its result"
        )
        checked += len(calls)
    assert checked > 1000


def test_a_conversation_carries_messages_the_task_and_typed_tools() -> None:
    body = client().get(f"/api/runs/{V2_RUN}/22/t1").json()
    msgs = body["messages"]
    assert [m["i"] for m in msgs] == list(range(len(msgs)))
    # τ²'s greeting is a fixed string: no usage; every later agent message has some
    assert msgs[0]["role"] == "assistant" and msgs[0]["usage"] is None
    assert all(m["usage"] for m in msgs[1:] if m["role"] == "assistant")
    assert body["task"]["id"] == "22" and body["task"]["evaluation_criteria"]["actions"]
    types = {t["name"]: t["type"] for t in body["tools"]}
    assert types["update_reservation_flights"] == "write" and types["get_user_details"] == "read"
    assert body["user_tools"] == []  # airline's customer has no tools of their own


def test_telecom_lists_the_customers_own_tools() -> None:
    body = client().get("/api/domains/telecom").json()
    ext = json.loads((RUNS_DIR.parent / "data" / "tasks" / "telecom.json").read_text())
    assert body["tools"] == ext["tools"]
    assert {t["name"] for t in ext["user_tools"]} >= {
        "toggle_airplane_mode",
        "check_network_status",
    }


def test_a_run_serves_the_agent_it_ran_with() -> None:
    c = client()
    v2 = c.get(f"/api/runs/{V2_RUN}/agent").json()
    assert v2["agent"] == "v2" and v2["hooks"]["extra_context"] is True
    p = v2["prompt"]
    # the policy sits in the slot, and extra_context() is appended after it
    assert p["slotted"] and p["policy"] in p["text"] and p["extra_context"]
    assert p["text"].endswith(p["extra_context"])
    v0 = c.get("/api/runs/20260915T075151Z_airline_v0_train/agent").json()
    assert v0["hooks"] == {
        "extra_context": False,
        "on_tool_call": False,
        "on_reply": False,
        "remember": False,
        "guidance": False,
        "check_write": False,
        "check_reply": False,
    }
    assert "helper.py" not in v0["files"] and v0["prompt"]["extra_context"] is None
    assert c.get("/api/runs/nope/agent").status_code == 404


def test_the_prompt_shown_is_the_prompt_the_agent_builds() -> None:
    """One composition, two callers: the tab's prompt and `LoopAgent.system_prompt()`."""
    from tau2_loop.agent.factory import LoopAgent
    from tau2_loop.agent.versions import load_version

    shown = client().get(f"/api/runs/{V2_RUN}/agent").json()
    version = load_version("airline", "v2")
    assert version.fingerprint == shown["fingerprint"][:12]
    agent = LoopAgent(tools=[], domain_policy=shown["prompt"]["policy"], version=version)
    # the run was made before the clock note, so the tab shows the prompt it was sent, without one;
    # the agent built today ends with it
    assert shown["prompt"]["clock_note"] is None
    assert agent.system_prompt() == f"{shown['prompt']['text']}\n\n{CLOCK_NOTE}"


def test_the_agent_is_told_not_to_take_the_real_date() -> None:
    """The CLI's date reaches every session; the composed prompt ends with the note against it,
    unless it is composed for a run made before the note."""
    assert compose("Be kind.\n{policy}", "P.", None).text == f"Be kind.\nP.\n\n{CLOCK_NOTE}"
    old = compose("Be kind.\n{policy}", "P.", None, clock=False)
    assert old.text == "Be kind.\nP." and old.clock_note is None


BANKING_V1_TRAIN = "20261002T235638Z_banking_knowledge_v1_train"


def test_a_version_carries_its_architecture_without_a_run() -> None:
    """The Agent tab draws a version from its own folder (s13 §2): retrieval and the tools it
    gives (read from tau2's spec, no environment), tool mode, surfaces, prompt layers, how it
    was made and the parent it is compared with."""
    c = client()
    body = c.get("/api/agents", params={"domain": "banking_knowledge"}).json()
    by = {v["name"]: v for v in body["versions"]}
    v1, v2 = by["v1"], by["v2"]
    assert v1["retrieval"] == "bm25_grep" and v1["config"]["retrieval"] is None
    assert v1["retrieval_info"]["tools"] == ["KB_search", "grep"]
    assert v1["retrieval_info"]["dense_model"] is None
    assert v1["retrieval_info"]["template"].endswith(".md")
    assert v1["tool_mode"] == "json" == v1["config"]["tool_mode"]
    assert v1["surfaces_present"] == ["system.md"]
    assert v1["prompt_layers"] == [
        "system.md",
        f"policy: {v1['retrieval_info']['template']}",
        "CLOCK_NOTE",
    ]
    # v1 was forked from v0, whose folder was retired: nothing to compare it with
    assert (v1["made_by"]["kind"], v1["made_by"]["from"], v1["parent"]) == (
        "model swap",
        "v0",
        None,
    )
    # v2 was written by banking's first loop cycle from v1, and adds helper.py
    assert (v2["made_by"]["kind"], v2["made_by"]["from"], v2["made_by"]["cycle"]) == (
        "loop cycle",
        "v1",
        1,
    )
    assert v2["parent"] == "v1"
    assert v2["surfaces_present"] == ["system.md", "helper.py"]
    assert v2["surfaces"]["system.md"]["sha"] != v1["surfaces"]["system.md"]["sha"]
    assert v2["surfaces"]["helper.py"]["chars"] > 0
    assert v2["diagnosis"] is not None  # the Optimise link still reads it
    # the single-version route says the same
    one = c.get("/api/agents/banking_knowledge/v2").json()
    for k in ("retrieval", "retrieval_info", "tool_mode", "surfaces", "prompt_layers", "parent"):
        assert one[k] == v2[k], k


def test_a_domain_without_retrieval_takes_tau2s_policy() -> None:
    v0 = client().get("/api/agents/airline/v0").json()
    assert v0["retrieval"] is None and v0["retrieval_info"] is None
    assert v0["prompt_layers"] == ["system.md", "policy: tau2 domain policy", "CLOCK_NOTE"]
    assert v0["made_by"]["kind"] == "base" and v0["parent"] is None
    # a helper with extra_context() adds its layer between the policy and the clock note
    v2 = client().get("/api/agents/airline/v2").json()
    if "extra_context" in v2["files"].get("helper.py", ""):
        assert v2["prompt_layers"][2] == "extra_context(): helper.py"


def test_a_run_carries_its_harness_health() -> None:
    """s12's hand-measured numbers, now in the run profile Runs reads (s13 milestone 0)."""
    c = client()
    h = c.get(f"/api/runs/{BANKING_V1_TRAIN}").json()["health"]
    assert h["conversations"] == 60 and 0 < h["slipped"] < h["replies"]
    assert h["slipped_rate"] == round(h["slipped"] / h["replies"], 4)
    assert h["shell_calls"] == 0 and h["dense_calls"] == 0 and h["bm25_calls"] > 0
    assert 0 < h["bare_discoverable_conversations"] <= h["conversations"]
    assert 0 <= h["required_docs_read_share"] <= 1
    # one conversation's row rides on the trial route, matched by task and trial
    row = c.get(f"/api/runs/{BANKING_V1_TRAIN}").json()["results"][0]
    t = c.get(f"/api/runs/{BANKING_V1_TRAIN}/{row['task_id']}/t{row['trial']}").json()
    assert t["health"]["task_id"] == row["task_id"] and t["health"]["trial"] == row["trial"] - 1


def test_a_run_without_tau2s_results_has_no_health() -> None:
    c = client()
    missing = [
        r["run_id"]
        for r in c.get("/api/runs").json()
        if not (RUNS_DIR / r["run_id"] / "tau2_results.json").exists()
    ]
    if missing:
        assert c.get(f"/api/runs/{missing[0]}").json()["health"] is None


def test_the_run_agent_shows_the_identity_note_only_when_the_version_asks() -> None:
    p = client().get(f"/api/runs/{BANKING_V1_TRAIN}/agent").json()["prompt"]
    assert p["identity_note"] is None and p["clock_note"] == CLOCK_NOTE


def test_a_v3_with_alltools_native_calls_and_the_identity_note(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The shape milestone 1 and 2's v3 will have, drawn from a folder, with no environment."""
    from tau2_loop.agent.versions import AgentConfig, AgentVersion
    from tau2_loop.serving.app import _architecture

    (tmp_path / "system.md").write_text("Be careful.\n{policy}\n")
    (tmp_path / "diagnosis.json").write_text(
        json.dumps({"kind": "tool change", "forked_from": "v1", "agent_yaml": ["retrieval: …"]})
    )
    v3 = AgentVersion(
        domain="banking_knowledge",
        name="v3",
        path=tmp_path,
        system_prompt="Be careful.\n{policy}\n",
        config=AgentConfig(
            model="sonnet", retrieval="alltools_minilm", tool_mode="native", identity_note=True
        ),
        helper=None,
        fingerprint="x",
    )
    a = _architecture(v3, ["v1", "v2", "v3"], {})
    assert a["retrieval"] == "alltools_minilm" and a["tool_mode"] == "native"
    assert a["retrieval_info"]["tools"] == ["KB_search_bm25", "KB_search_dense", "shell"]
    assert a["retrieval_info"]["dense_model"].endswith("all-MiniLM-L6-v2")
    tpl = a["retrieval_info"]["template"]
    assert a["prompt_layers"] == ["system.md", f"policy: {tpl}", "CLOCK_NOTE", "identity note"]
    assert (a["made_by"]["kind"], a["parent"]) == ("tool change", "v1")
