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
    # v0 first, then v3 three times: promoted by hand as a model swap, re-baselined on the new
    # SDK, and again under our simulation rules (2 Oct 2026) before cycle 6's gate
    assert [(r["version"], r["kind"], r["passed"], r["n"]) for r in h["reigns"]] == [
        ("v0", "first", 12, 20),
        ("v3", "model swap", 18, 25),
        ("v3", "re-baseline", 21, 25),
        ("v3", "re-baseline", 19, 25),
    ]
    v = {x["version"]: x for x in h["versions"]}
    assert list(v) == ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    assert [v[k]["train"]["cut"] for k in v] == [1, 1, 1, 2, 2, 2, 2, 2]
    assert v["v3"]["made"]["kind"] == "model swap" and v["v3"]["verdict"] == "by hand"
    assert (v["v3"]["train"]["passed"], v["v3"]["test"]["passed"]) == (19, 21)
    assert (v["v7"]["verdict"], v["v7"]["fixed"], v["v7"]["broke"]) == ("hold", 5, 1)
    assert v["v7"]["vs"]["train"]["passed"] == 19 and v["v7"]["test"]["passed"] == 21
    assert v["v5"]["made"] == {
        "kind": "model swap",
        "cycle": 4,
        "source": "v3",
        "detail": "model: sonnet → opus",
    }
    assert (v["v6"]["verdict"], v["v6"]["fixed"], v["v6"]["broke"]) == ("hold", 2, 2)
    assert v["v6"]["vs"]["train"]["passed"] == 21 and v["v6"]["vs"]["test"]["passed"] == 21


def test_a_version_played_on_part_of_a_split_or_never_is_not_full() -> None:
    runs = [
        _run("r4", "v4", "train", 29, n=60, cut=3),
        _run("r4t", "v4", "test", 14, n=37, cut=3),
        # `make eval TASKS=`: no cut, and the split side it was part of
        {**_run("r5", "v5", "train", 10, n=20), "cut": None, "of": 60},
        _run("r7", "v7", "train", 37, n=60, cut=3),
        # a check on 3 test tasks is not v7's test number, which must be on its train run's cut
        {**_run("r7t", "v7", "test", 1, n=3), "cut": None, "of": 37},
        _run("r0", "v0", "train", 12, n=20, cut=1),  # before the loop ran test: still full
    ]
    fork = {
        "kind": "workflow tools",
        "forked_from": "v4",
        "agent_yaml": ["workflows: None → r1"],
        "diagnoses": [],
        "changes": [
            {
                "file": "agent.yaml",
                "why": "s16: the workflow tools (a hand-made fork: the optimiser may not change agent.yaml)",
            },
            {"file": "system.md", "why": "tells the model when to call them"},
        ],
    }
    versions = [
        {"name": "v0", "model": "sonnet", "effort": "medium", "diagnosis": None},
        {"name": "v4", "model": "sonnet", "effort": "high", "diagnosis": None},
        {
            "name": "v5",
            "model": "sonnet",
            "effort": "high",
            "diagnosis": {
                "diagnoses": [{}],
                "optimised_from": "v4",
                "prompt_diff_summary": "Adds lesson 10.",
            },
        },
        {"name": "v6", "model": "sonnet", "effort": "high", "diagnosis": fork},
        {"name": "v7", "model": "sonnet", "effort": "medium", "diagnosis": None},
    ]
    v = {
        x["version"]: x
        for x in version_history("banking_knowledge", versions, runs, [], {})["versions"]
    }
    assert {k: x["full"] for k, x in v.items()} == {
        "v0": True,
        "v4": True,
        "v5": False,
        "v6": False,
        "v7": True,
    }
    assert v["v7"]["test"] is None
    assert (v["v5"]["train"]["n"], v["v5"]["train"]["of"], v["v4"]["train"]["of"]) == (20, 60, None)
    # an optimiser's version says what it changed; a fork, why, without the note every fork carries
    assert v["v5"]["why"] == "Adds lesson 10." and v["v5"]["made"]["source"] == "v4"
    assert v["v6"]["why"] == "s16: the workflow tools; tells the model when to call them"
    assert v["v4"]["why"] is None


def test_a_later_replay_of_some_tasks_does_not_stand_for_a_whole_run() -> None:
    """s23 replayed v23 on its 34 failures after its whole runs; v23 stays played in full."""
    runs = [
        _run("r23", "v23", "train", 43, n=60, cut=3),
        _run("r23t", "v23", "test", 20, n=37, cut=3),
        {**_run("c23", "v23", "train", 1, n=17), "cut": None, "of": 60},
        {**_run("c23t", "v23", "test", 4, n=17), "cut": None, "of": 37},
    ]
    versions = [{"name": "v23", "model": "sonnet", "effort": "medium", "diagnosis": None}]
    [v] = version_history("banking_knowledge", versions, runs, [], {})["versions"]
    assert v["full"] and (v["train"]["run_id"], v["test"]["run_id"]) == ("r23", "r23t")


def test_the_committed_banking_history_leaves_the_checks_off() -> None:
    h = (
        TestClient(create_app())
        .get("/api/versions", params={"domain": "banking_knowledge"})
        .json()["banking_knowledge"]
    )
    v = {x["version"]: x for x in h["versions"]}
    assert [k for k, x in v.items() if x["full"]] == ["v1", "v2", "v3", "v4", "v7", "v8", "v23"]
    assert all(
        (x["train"]["n"], x["train"]["of"], x["test"]["n"], x["test"]["of"]) == (60, 60, 37, 37)
        for x in v.values()
        if x["full"]
    )
    # s22's checks: v9 played 2 train and 3 test tasks of the 60 and 37; v18-v22 played test only
    assert (
        v["v9"]["train"]["n"],
        v["v9"]["train"]["of"],
        v["v9"]["test"]["n"],
        v["v9"]["test"]["of"],
    ) == (2, 60, 3, 37)
    assert v["v22"]["train"] is None and v["v22"]["test"]["cut"] is None
    assert v["v6"]["train"] is None and v["v6"]["test"] is None and not v["v6"]["full"]
