"""Evals on a banking task: the customer goal derived from the scenario, the committed index of
the knowledge base, which required documents each version read or only saw, and the tools the
domain's champion is given. Offline: committed files, and tau2's documents where present."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tau2_loop.data import documents as kb_docs
from tau2_loop.data.goals import customer_goal, is_placeholder
from tau2_loop.data.splits import read_task_extract
from tau2_loop.eval import health, replay
from tau2_loop.serving.app import create_app

BOILER = (
    "You are playing the role of a customer contacting a customer service representative agent. "
)
GOLD = "doc_credit_cards_gold_rewards_card_001"
SILVER = "doc_credit_cards_silver_rewards_card_001"
BRONZE = "doc_credit_cards_bronze_rewards_card_001"
PLATINUM = "doc_credit_cards_platinum_rewards_card_001"


def client() -> TestClient:
    return TestClient(create_app())


# ── the customer goal ────────────────────────────────────────────────────────────────────────
def test_only_taus_placeholder_purpose_is_replaced() -> None:
    assert is_placeholder("Task: task_001") and is_placeholder(None) and is_placeholder("")
    assert not is_placeholder("Test resolution path: MMS (Picture/Group Messaging) Issues.")


def test_the_goal_skips_the_boilerplate_and_the_character_sentence() -> None:
    s = (
        BOILER + "Your character is a consultant named Sarah who earns $100,000 annually. "
        "You travel frequently for work.\n\n"
        "You're looking for a credit card to use for your everyday purchases. You want cash back."
    )
    assert (
        customer_goal(s) == "You're looking for a credit card to use for your everyday purchases."
    )


def test_a_goal_section_wins_over_a_wanting_sentence_in_the_situation() -> None:
    s = (
        "**Your character:** You are Fatima, a 31-year-old owner. You want everything.\n\n"
        "**Your situation:** You want to see your recent transactions.\n\n"
        "**Your goal:** Review your recent transactions and get your card replaced.\n\n"
        '**Opening:** "Hi, I need help."'
    )
    assert customer_goal(s) == "Review your recent transactions and get your card replaced."


def test_without_a_narrative_the_opening_line_is_quoted() -> None:
    s = (
        "**Your character:** You are Jordan, a consultant. You're friendly.\n\n"
        "**Verification info:**\n- Name: Jordan\n\n"
        "## Conversation Flow\n\n"
        '1. **Opening:** "Hi, I need help with some disputes on my debit cards. Thanks."\n\n'
        '2. **If asked:** "That is why I am calling."'
    )
    assert customer_goal(s) == "“I need help with some disputes on my debit cards.”"


def test_a_sentence_ending_in_a_colon_takes_its_list() -> None:
    s = (
        "**Your character:** You are Taylor.\n\n"
        "**Your situation:** You've saved up $20,000 and want to:\n"
        "1. Open a credit card\n2. Open a savings account."
    )
    assert customer_goal(s) == (
        "You've saved up $20,000 and want to: Open a credit card; Open a savings account"
    )


def test_a_structured_scenario_has_no_derived_goal() -> None:
    assert customer_goal({"reason_for_call": "You want to cancel."}) is None
    assert customer_goal(None) is None


def test_every_banking_task_gets_a_goal_and_task_001_the_one_the_person_expects() -> None:
    tasks = read_task_extract("banking_knowledge")["tasks"]
    goals = {t["id"]: customer_goal(t["user_scenario"]["instructions"]) for t in tasks}
    assert all(goals.values()) and len(goals) == 97
    assert goals["task_001"] == (
        "You're looking for a credit card to use for your everyday purchases."
    )
    assert all("playing the role" not in g for g in goals.values() if g)


# ── the knowledge base's index ───────────────────────────────────────────────────────────────
def test_the_committed_index_names_every_document_tasks_require() -> None:
    index = kb_docs.read_index("banking_knowledge")
    assert len(index) == 698
    required = {
        d for t in read_task_extract("banking_knowledge")["tasks"] for d in t["required_documents"]
    }
    assert required <= set(index)
    assert index[GOLD] == {
        "title": "Gold Rewards Card: Apply for a Premium Credit Card",
        "chars": 1684,
    }
    assert kb_docs.read_index("airline") == {}


@pytest.mark.skipif(not kb_docs.available("banking_knowledge"), reason="tau2's documents absent")
def test_the_committed_index_is_what_tau2s_files_give() -> None:
    built = kb_docs.build_index("banking_knowledge")
    assert built["documents"] == kb_docs.read_index("banking_knowledge")
    assert built["source"] == "vendor/tau2-bench/data/tau2/domains/banking_knowledge/documents"


def test_a_document_is_read_only_from_the_documents_folder() -> None:
    assert kb_docs.document("banking_knowledge", "../db") is None
    assert kb_docs.document("banking_knowledge", "doc_nope_001") is None
    assert kb_docs.described("banking_knowledge", ["doc_nope_001"]) == [
        {"id": "doc_nope_001", "title": None, "chars": None}
    ]


# ── read, seen, never reached ────────────────────────────────────────────────────────────────
def _sim(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {"task_id": "task_1", "trial": 0, "messages": messages, "reward_info": {"reward": 1.0}}


def test_a_document_named_in_a_result_but_never_shown_whole_is_seen() -> None:
    paren = "doc_bank_accounts_bank_accounts_(general)_001"
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "a", "name": "KB_search_bm25", "arguments": {"query": "gold"}},
                {"id": "b", "name": "shell", "arguments": {"command": "ls | grep cards"}},
                {"id": "c", "name": "shell", "arguments": {"command": "grep -l fee *.md"}},
                {"id": "d", "name": "get_user_information_by_id", "arguments": {}},
            ],
        },
        {"role": "tool", "id": "a", "content": f"1. Gold\n   ID: {GOLD}\n   Content: …"},
        # a listing names silver, and a file whose name only ends in bronze's id
        {"role": "tool", "id": "b", "content": f"{SILVER}.md\nold_{BRONZE}.md\n{GOLD}.md"},
        {"role": "tool", "id": "c", "content": f"{paren}.md"},
        # a non-knowledge tool naming a document is not the agent finding it
        {"role": "tool", "id": "d", "content": PLATINUM},
    ]
    row = health.conversation_health(_sim(msgs), frozenset({GOLD, SILVER, BRONZE, PLATINUM, paren}))
    assert row["required_docs_read"] == 1 and row["required_docs_read_ids"] == [GOLD]
    assert row["required_docs_seen_ids"] == sorted([SILVER, paren])  # gold was read, not seen
    assert row["required_docs_seen"] == 2
    s = health.summarise_health([row])
    assert s["required_docs_read_share"] == 0.2 and s["required_docs_seen_share"] == 0.4


def test_task_001_reads_per_version_are_what_the_person_counted() -> None:
    r = client().get("/api/domains/banking_knowledge/tasks/task_001/reads")
    assert r.status_code == 200
    body = r.json()
    assert body["required"] == [GOLD, SILVER, BRONZE, PLATINUM]
    by = {v["version"]: v for v in body["versions"]}
    # the four the person counted; a later version's run adds its own row (v7's train run did)
    assert {"v1", "v2", "v3", "v4"} <= set(by)
    assert (by["v1"]["passed"], by["v1"]["read"], by["v1"]["seen"]) == (False, [SILVER], [])
    assert (by["v2"]["passed"], by["v2"]["read"], by["v2"]["seen"]) == (True, [BRONZE, GOLD], [])
    # v3 found the cards through the shell (listings, grep lines, a loop over a wildcard)
    assert (by["v3"]["passed"], by["v3"]["read"]) == (True, [])
    assert by["v3"]["seen"] == sorted([GOLD, SILVER, BRONZE, PLATINUM])
    assert by["v1"]["run_id"] == "20261002T235638Z_banking_knowledge_v1_train"
    assert by["v3"]["run_id"] == "20261004T023602Z_banking_knowledge_v3_train"
    # v4 (champion since 5 Oct) also printed the four cards through the shell
    assert (by["v4"]["passed"], by["v4"]["read"]) == (True, [])
    assert by["v4"]["seen"] == sorted([GOLD, SILVER, BRONZE, PLATINUM])
    assert by["v4"]["run_id"] == "20261004T222002Z_banking_knowledge_v4_train"
    assert all(v["trial"] == 1 and v["split"] == "train" for v in body["versions"])
    assert client().get("/api/domains/banking_knowledge/tasks/nope/reads").status_code == 404


# ── the routes ──────────────────────────────────────────────────────────────────────────────
def test_a_banking_task_lists_its_documents_by_title() -> None:
    t = client().get("/api/domains/banking_knowledge/tasks/task_001").json()
    assert [d["id"] for d in t["documents"]] == t["required_documents"]
    assert t["documents"][0]["title"].startswith("Gold Rewards Card")
    airline = client().get("/api/domains/airline/tasks/0").json()
    assert airline["documents"] == []


def test_the_task_table_carries_a_goal_and_a_document_count_where_tau2_has_none() -> None:
    c = client()
    rows = {r["id"]: r for r in c.get("/api/domains/banking_knowledge").json()["tasks"]}
    assert rows["task_001"]["purpose"] == "Task: task_001"  # tau2's own field is kept
    assert rows["task_001"]["goal"].startswith("You're looking for a credit card")
    assert rows["task_001"]["n_documents"] == 4
    assert max(r["n_documents"] for r in rows.values()) == 30
    for d in ("airline", "telecom"):
        assert all(
            r["goal"] is None and r["n_documents"] == 0
            for r in c.get(f"/api/domains/{d}").json()["tasks"]
        )


def test_a_document_route_serves_tau2s_text_or_says_it_is_not_here() -> None:
    c = client()
    assert c.get("/api/domains/banking_knowledge/documents/doc_nope_001").status_code == 404
    r = c.get(f"/api/domains/banking_knowledge/documents/{GOLD}")
    if kb_docs.available("banking_knowledge"):
        body = r.json()
        assert r.status_code == 200 and body["chars"] == 1684 == len(body["content"])
        src = kb_docs.source_dir("banking_knowledge") / f"{GOLD}.json"
        assert body["content"] == json.loads(src.read_text())["content"]
    else:
        assert r.status_code == 404 and "not in this image" in r.json()["detail"]


@pytest.mark.skipif(not replay.available(), reason="tau2 is not installed")
def test_the_tools_follow_bankings_champion_retrieval() -> None:
    d = client().get("/api/domains/banking_knowledge").json()
    h = d["harness_tools"]
    assert (h["version"], h["retrieval"], h["known"]) == ("v4", "alltools_minilm", True)
    names = [t["name"] for t in h["tools"]]
    assert names[:3] == ["KB_search_bm25", "KB_search_dense", "shell"]
    assert "KB_search" not in names and "grep" not in names
    assert h["replaced"] == ["KB_search", "grep"]
    assert h["tools"][0]["description"] == "Search the knowledge base using BM25 sparse retrieval."
    # the extract's own list is kept beside it, and the other tools pass through unchanged
    assert [t["name"] for t in d["tools"]][:2] == ["KB_search", "grep"]
    assert names[3:] == [t["name"] for t in d["tools"]][2:]
    airline = client().get("/api/domains/airline").json()["harness_tools"]
    assert airline["retrieval"] is None and airline["replaced"] == []
