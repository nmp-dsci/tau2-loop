"""The published board, ingested from the pinned submodule into a committed index."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from tau2_loop.config import DOMAINS
from tau2_loop.data.leaderboard import LEADERBOARD_PATH, best_per_domain, read
from tau2_loop.serving.app import create_app


def test_the_index_is_committed_and_matches_the_manifest() -> None:
    board = read()
    assert board["entries"], "run `make leaderboard`"
    # every listed submission that is actually committed upstream was ingested
    assert 0 < len(board["entries"]) <= board["listed"]
    assert board["tau2_sha"], "the submodule sha is what pins this index"


def test_every_entry_keeps_its_verification_and_at_least_one_domain() -> None:
    for e in read()["entries"]:
        assert e["model"] and e["date"]
        # self-reported: whether the prompts were modified is the only guard, so it
        # must survive the ingest even when upstream left it unset
        assert "modified_prompts" in e
        assert set(e["domains"]) <= set(DOMAINS)
        for domain, s in e["scores"].items():
            assert domain in DOMAINS
            if s["pass_1"] is not None:
                assert 0 <= s["pass_1"] <= 100, "the site reports pass^k as a percentage"


def test_the_best_per_domain_is_the_highest_pass_1() -> None:
    entries = read()["entries"]
    best = best_per_domain(entries)
    for domain, top in best.items():
        highest = max(
            e["scores"][domain]["pass_1"]
            for e in entries
            if e["scores"].get(domain, {}).get("pass_1") is not None
        )
        assert top["pass_1"] == highest


def test_the_api_serves_the_board_beside_our_own_runs() -> None:
    body = TestClient(create_app()).get("/api/leaderboard").json()
    assert len(body["entries"]) == len(read()["entries"])
    # our runs are computed, never written into the index
    assert "ours" in body and all(o["domain"] in DOMAINS for o in body["ours"])
    assert set(body["our_caveats"]) == {"user_simulator", "tool_calling", "prompts", "split"}
    assert "ours" not in json.loads(LEADERBOARD_PATH.read_text())
