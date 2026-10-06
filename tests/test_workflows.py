"""workflow_rag's golden set (s16): train open, test sealed, and agents listed under their dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tau2_loop.config import DATASET_AGENTS, DOMAINS
from tau2_loop.data.splits import read_task_extract, split_ids
from tau2_loop.serving.app import create_app
from tau2_loop.workflows import golden

D = "banking_knowledge"


def test_every_dataset_lists_its_own_agents_and_banking_has_no_judge() -> None:
    assert set(DATASET_AGENTS) == set(DOMAINS)
    assert all(kinds[0] == "answering" for kinds in DATASET_AGENTS.values())
    assert DATASET_AGENTS["airline"] == ("answering", "judge")
    assert DATASET_AGENTS[D] == ("answering", "workflow_rag")
    assert "judge" not in DATASET_AGENTS[D]


def test_the_golden_set_is_every_train_question_and_nothing_else() -> None:
    g = golden.golden_set(D)
    assert g is not None and g["status"] == "draft"
    ids = [e["task_id"] for e in g["entries"]]
    assert sorted(ids) == sorted(split_ids(D, "train"))
    assert not set(ids) & set(split_ids(D, "test"))
    names = {w["name"] for w in g["workflows"]}
    # every entry names known workflows, and every workflow is used on train
    assert all(set(e["workflows"]) <= names and e["workflows"] for e in g["entries"])
    assert all(w["train"] > 0 for w in g["workflows"])
    assert sum(len(e["workflows"]) > 1 for e in g["entries"]) == 27


def test_an_entry_reads_gold_as_a_person_would() -> None:
    g = golden.golden_set(D)
    assert g is not None
    e = next(x for x in g["entries"] if x["task_id"] == "task_019")
    assert e["workflows"] == ["cash_back_dispute"]
    tools = [c["tool"] for c in e["calls"]]
    # a discoverable tool appears under its own name; the unlock before it is not a step
    assert tools[0] == "log_verification" and tools.count("submit_cash_back_dispute_0589") == 4
    assert "unlock_discoverable_agent_tool" not in tools
    assert [c["by"] for c in e["calls"]][-1] == "customer"
    assert len(e["required_documents"]) == 6 and e["situation"]
    assert all(d["title"] for d in e["required_documents"])


def test_a_dataset_without_workflow_rag_has_no_golden_set() -> None:
    assert golden.golden_set("airline") is None


def test_labels_naming_a_test_task_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = json.loads(golden.labels_path(D).read_text())
    leak = split_ids(D, "test")[0]
    labels["tasks"][leak] = {"workflows": ["human_transfer"], "gold": "", "flag": ""}
    (tmp_path / D).mkdir()
    (tmp_path / D / "labels.json").write_text(json.dumps(labels))
    monkeypatch.setattr(golden, "WORKFLOWS_DIR", tmp_path)
    with pytest.raises(ValueError, match="outside train"):
        golden.golden_set(D)


def test_the_mapping_rules_reproduce_the_train_labels_but_two() -> None:
    labels = json.loads(golden.labels_path(D).read_text())["tasks"]
    tasks = {t["id"]: t for t in read_task_extract(D)["tasks"]}
    off = [
        t
        for t in split_ids(D, "train")
        if golden.mapped_workflows(tasks[t]) != set(labels[t]["workflows"])
    ]
    # both are workflows with no write of their own: a closure stopped at its dispute check,
    # and a decline that turned out to be fraud
    assert off == ["task_046", "task_088"]
    # in gold's order too, but one: gold reissues the card before the PIN reset its label puts first
    order = [
        t
        for t in split_ids(D, "train")
        if t not in off and golden.ordered_workflows(tasks[t]) != labels[t]["workflows"]
    ]
    assert order == ["task_090"]


def test_the_sealed_counts_carry_no_test_id_and_regenerate_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    committed = json.loads(golden.sealed_counts_path(D).read_text())
    text = golden.sealed_counts_path(D).read_text()
    assert not any(t in text for t in split_ids(D, "test"))
    (tmp_path / D).mkdir()
    (tmp_path / D / "labels.json").write_text(golden.labels_path(D).read_text())
    monkeypatch.setattr(golden, "WORKFLOWS_DIR", tmp_path)
    assert golden.seal_test_counts(D) == committed
    assert committed["n_test"] == 37 and committed["covered"] + committed["needs_new"] <= 37


def test_the_api_serves_the_golden_set_and_each_datasets_agents() -> None:
    c = TestClient(create_app())
    kinds = c.get("/api/agents").json()["kinds"]
    assert kinds[D] == ["answering", "workflow_rag"] and kinds["airline"] == ["answering", "judge"]
    g = c.get(f"/api/workflows/{D}/golden").json()
    assert len(g["entries"]) == 60 and g["test"]["n_test"] == 37
    # test is held out of optimisation, not hidden: each test question with its rule-mapped workflows
    test = g["test_entries"]
    assert sorted(e["task_id"] for e in test) == sorted(split_ids(D, "test"))
    assert not {e["task_id"] for e in test} & {e["task_id"] for e in g["entries"]}
    assert {e["split"] for e in test} == {"test"}
    assert {e["split"] for e in g["entries"]} == {"train"}
    # its workflows add up to the sealed counts, and it carries no hand label
    per_wf: dict[str, int] = {}
    for e in test:
        for w in e["workflows"]:
            per_wf[w] = per_wf.get(w, 0) + 1
    assert per_wf == g["test"]["per_workflow"]
    assert sum(bool(e["new_tools"]) for e in test) == g["test"]["needs_new"]
    assert all(not e["info"] and not e["gold"] and not e["flag"] for e in test)
    assert c.get("/api/workflows/airline/golden").status_code == 404
    assert c.get("/api/workflows/nowhere/golden").status_code == 404


def test_each_workflow_names_the_knowledge_base_procedure_that_defines_it() -> None:
    """The workflow names are ours; the procedure documents behind them are the knowledge base's."""
    g = golden.golden_set(D)
    assert g is not None
    without = [w["name"] for w in g["workflows"] if not w["kb_docs"]]
    # a card recommendation has no procedure document: it is built from each card's own terms
    assert without == ["credit_card_recommendation"]
    assert all(d["title"] for w in g["workflows"] for d in w["kb_docs"])
    dispute = next(w for w in g["workflows"] if w["name"] == "debit_card_dispute")
    assert dispute["kb_docs"][0]["title"] == "Internal: Filing a Debit Card Transaction Dispute"


def test_each_question_lists_the_information_its_workflows_need() -> None:
    """The second of the three scored parts: facts the customer must supply, held-back ones marked."""
    labels = json.loads(golden.labels_path(D).read_text())
    fields = {w: {f["field"] for f in v["info_fields"]} for w, v in labels["workflows"].items()}
    fields["verification"] = {f["field"] for f in labels["verification_fields"]}
    g = golden.golden_set(D)
    assert g is not None
    for e in g["entries"]:
        own = [i for i in e["info"] if i["workflow"] != "verification"]
        assert own, f"{e['task_id']} names no information"
        for i in e["info"]:
            assert i["workflow"] in set(e["workflows"]) | {"verification"}
            assert i["field"] in fields[i["workflow"]] | {"other"}
            assert isinstance(i["held_back"], bool) and i["value"]
    # task_001's customer mentions the Rho-Bank+ subscription only if asked
    e001 = next(e for e in g["entries"] if e["task_id"] == "task_001")
    assert any(i["field"] == "rho_bank_plus" and i["held_back"] for i in e001["info"])
