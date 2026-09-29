"""The version history the Runs and Optimise figures draw, on synthetic files and on the committed ones."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tau2_loop.loop.history import version_history
from tau2_loop.serving.app import create_app


def _run(
    run_id: str, agent: str, split: str, passed: int, n: int = 10, cut: int = 2
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "agent": agent,
        "split": split,
        "cut": cut,
        "passed": passed,
        "n": n,
        "model": "claude-sdk/claude-sonnet-5",
        "effort": "medium",
    }


def test_a_history_reads_the_gates_runs_the_verdicts_and_the_reigns() -> None:
    runs = [
        _run("r0", "v0", "train", 5),
        _run("r1", "v1", "train", 7),
        _run("r1t", "v1", "test", 6),
        _run("r0b", "v0", "train", 6),  # a re-run of v0, newer than v1's
        _run("r2", "v2", "train", 4),
        _run("r2t", "v2", "test", 8),
        _run("r3", "v3", "train", 9),  # scored, never gated
    ]
    versions = [
        {"name": "v0", "model": "haiku", "effort": "medium", "diagnosis": None},
        {"name": "v1", "model": "haiku", "effort": "medium", "diagnosis": {"diagnoses": []}},
        {
            "name": "v2",
            "model": "opus",
            "effort": "medium",
            "diagnosis": {
                "kind": "model swap",
                "forked_from": "v1",
                "agent_yaml": ["model: haiku → opus"],
            },
        },
        {"name": "v3", "model": "haiku", "effort": "medium", "diagnosis": None},
        {"name": "v4", "model": "haiku", "effort": "high", "diagnosis": None},
    ]
    ledger = [
        {
            "cycle": 1,
            "champion": "v0",
            "champion_run": "r0",
            "challenger": "v1",
            "optimiser_model": "opus",
            "outcome": {
                "verdict": "promote",
                "challenger_run": "r1",
                "test_run": "r1t",
                "fixed": ["a", "b"],
                "broken": [],
                "p_value": 0.25,
            },
        },
        {
            "cycle": 2,
            "kind": "model swap",
            "forked_from": "v1",
            "champion": "v1",
            "champion_run": "r1",
            "challenger": "v2",
            "agent_yaml": ["model: haiku → opus"],
            "outcome": {
                "verdict": "hold",
                "challenger_run": "r2",
                "test_run": "r2t",
                "fixed": [],
                "broken": ["a", "c", "d"],
                "p_value": 1.0,
                "test_compare": {"champion_run": "r1t"},
            },
        },
    ]
    registry = {
        "champion": {"agent": "v1"},
        "history": [
            {"event": "promote", "agent": "v0", "run_id": "r0", "passed": 5, "n_scored": 10},
            {
                "event": "register:challenger",
                "agent": "v1",
                "run_id": "r1",
                "passed": 7,
                "n_scored": 10,
            },
            {"event": "promote", "agent": "v1", "run_id": "r1", "passed": 7, "n_scored": 10},
        ],
    }
    h = version_history("airline", versions, runs, ledger, registry)
    assert h["champion"] == "v1"
    assert [(r["version"], r["kind"], r["passed"]) for r in h["reigns"]] == [
        ("v0", "first", 5),
        ("v1", "gate", 7),
    ]
    v = {x["version"]: x for x in h["versions"]}
    # v0's train number is the run the registry promoted, not its newer re-run
    assert (
        v["v0"]["train"]["run_id"] == "r0"
        and v["v0"]["verdict"] == "first"
        and v["v0"]["made"]["kind"] == "base"
    )
    assert v["v1"]["made"] == {
        "kind": "optimise",
        "cycle": 1,
        "source": "v0",
        "detail": "opus optimiser",
    }
    assert (v["v1"]["verdict"], v["v1"]["fixed"], v["v1"]["broke"], v["v1"]["held_title"]) == (
        "promote",
        2,
        0,
        True,
    )
    assert (
        v["v1"]["test"]["passed"] == 6
        and v["v1"]["vs"]["train"]["passed"] == 5
        and v["v1"]["vs"]["test"] is None
    )
    assert v["v2"]["made"] == {
        "kind": "model swap",
        "cycle": 2,
        "source": "v1",
        "detail": "model: haiku → opus",
    }
    assert (
        v["v2"]["verdict"] == "hold"
        and v["v2"]["broke"] == 3
        and v["v2"]["vs"]["test"]["passed"] == 6
    )
    assert (
        v["v3"]["verdict"] == "not gated"
        and v["v3"]["train"]["passed"] == 9
        and v["v3"]["test"] is None
    )
    # no run at all: nothing to draw, and the effort is the agent.yaml's
    assert v["v4"]["train"] is None and v["v4"]["verdict"] is None and v["v4"]["effort"] == "high"


def test_the_committed_airline_history() -> None:
    h = (
        TestClient(create_app())
        .get("/api/versions", params={"domain": "airline"})
        .json()["airline"]
    )
    assert h["champion"] == "v3"
    # v0 first, then v3 twice: promoted by hand as a model swap, re-run and re-baselined on the new SDK
    assert [(r["version"], r["kind"], r["passed"], r["n"]) for r in h["reigns"]] == [
        ("v0", "first", 12, 20),
        ("v3", "model swap", 18, 25),
        ("v3", "re-baseline", 21, 25),
    ]
    v = {x["version"]: x for x in h["versions"]}
    assert list(v) == ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]
    assert [v[k]["train"]["cut"] for k in v] == [1, 1, 1, 2, 2, 2, 2]
    assert v["v3"]["made"]["kind"] == "model swap" and v["v3"]["verdict"] == "by hand"
    assert (v["v3"]["train"]["passed"], v["v3"]["test"]["passed"]) == (21, 21)
    assert v["v5"]["made"] == {
        "kind": "model swap",
        "cycle": 4,
        "source": "v3",
        "detail": "model: sonnet → opus",
    }
    assert (v["v6"]["verdict"], v["v6"]["fixed"], v["v6"]["broke"]) == ("hold", 2, 2)
    assert v["v6"]["vs"]["train"]["passed"] == 21 and v["v6"]["vs"]["test"]["passed"] == 21
