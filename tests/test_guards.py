"""s09's guards on the optimiser: the read fence, the code surfaces' imports, customer data, the halves."""

from __future__ import annotations

from pathlib import Path

from tau2_loop.config import ROOT
from tau2_loop.data.splits import cut_halves, halves, read_split, read_task_extract
from tau2_loop.loop.guards import (
    bash_fence_reason,
    fence_reason,
    import_violations,
    leak_values,
    leaks,
)


def test_the_fence_closes_runs_data_loop_and_vendor_but_not_what_the_prompt_names() -> None:
    trace = (ROOT / "runs" / "r1" / "traces" / "t.json").resolve()
    partner = ROOT / "agents" / "banking_knowledge" / "v2"
    assert fence_reason("data/tasks/banking_knowledge.json", set(), []) is not None
    assert fence_reason(str(ROOT / "loop" / "airline" / "ledger.jsonl"), set(), []) is not None
    assert fence_reason("vendor/tau2-bench/data/tau2/domains", set(), []) is not None
    assert fence_reason("runs/r1/traces/t.json", {trace}, []) is None  # a trace the prompt cites
    assert fence_reason("runs/r1/results.jsonl", {trace}, []) is not None
    assert fence_reason("agents/banking_knowledge/v1/system.md", set(), []) is None
    assert "another challenger" in (
        fence_reason(str(partner / "checks.py"), set(), [partner]) or ""
    )
    assert fence_reason("", set(), []) is not None  # a search with no path covers the repo
    assert bash_fence_reason("cat data/splits/banking_knowledge.json", []) is not None
    assert bash_fence_reason("uv run python -c 'import json'", []) is None
    assert bash_fence_reason("ls agents/banking_knowledge/v2", [partner]) is not None


def test_the_fence_refuses_the_root_and_any_ancestor_of_a_closed_folder() -> None:
    partner = ROOT / "agents" / "banking_knowledge" / "v2"
    for path in (".", str(ROOT), "agents/..", str(ROOT.parent)):
        reason = fence_reason(path, set(), [])
        assert reason and "inside your version" in reason
    assert fence_reason("agents/banking_knowledge", set(), [partner]) is not None
    assert fence_reason("agents/banking_knowledge/v1", set(), [partner]) is None
    assert fence_reason("agents/banking_knowledge/v1/system.md", set(), []) is None


def test_a_code_surface_imports_only_the_allow_list() -> None:
    ok = "import re\nfrom datetime import date\n\ndef check_write(n, a, s):\n    return None\n"
    assert import_violations(ok) == []
    bad = "import os\nimport urllib.request\n\ndef f():\n    return open('x').read()\n"
    assert import_violations(bad) == ["imports os", "imports urllib", "calls open()"]
    assert import_violations("def f(:\n")[0].startswith("does not parse")


def test_customer_data_is_found_and_generic_words_are_not() -> None:
    vals = leak_values("airline")
    assert vals and all(len(v) >= 3 for v in vals)
    customer = sorted(v for v in vals if "_" in v and any(c.isdigit() for c in v))[0]
    assert leaks({"checks.py": f"if uid == '{customer}':\n    pass"}, vals) == [
        f"checks.py: {customer}"
    ]
    assert leaks({"system.md": "Confirm the reservation before any write."}, vals) == []


def test_banking_keeps_its_halves_on_file_but_the_loop_reads_all_of_train() -> None:
    """s09 dealt banking's train into read and gate halves; since its gate moved to test (3 Oct
    2026) the loop uses neither, and the split keeps them as the record."""
    s = read_split("banking_knowledge")
    read, gate = s["halves"]["read"], s["halves"]["gate"]
    assert halves("banking_knowledge") is None
    assert len(read) == len(gate) == 30 and not set(read) & set(gate)
    assert sorted(read + gate) == sorted(s["train"])
    # split v2's halves, reproducible from its train list, are kept whole inside v3's
    v2 = cut_halves(s["v2"]["train"])
    assert read[:24] == v2["read"] and gate[:24] == v2["gate"]
    assert halves("airline") is None


def test_committed_versions_pass_the_leak_guard() -> None:
    from tau2_loop.agent.versions import list_versions

    for domain in ("airline", "retail", "telecom", "banking_knowledge"):
        vals = leak_values(domain)
        for v in list_versions(domain):
            assert leaks(v.files(), vals) == [], f"{domain}/{v.name}"


def test_paths_resolve_from_the_repo_root(tmp_path: Path) -> None:
    assert fence_reason(str(tmp_path / "anything"), set(), []) is None


def test_a_discoverable_tool_is_named_freely_and_the_ids_in_its_arguments_are_guarded() -> None:
    """Banking's first cycle was rejected for naming `open_bank_account_4821`, a tool the knowledge
    base documents for every customer (expected in train and test tasks alike), not a customer's
    value. Its arguments travel as JSON text, whose ids the guard used to miss whole."""
    from tau2_loop.loop.guards import _strings

    vals = leak_values("banking_knowledge")
    for tool in (
        "open_bank_account_4821",
        "close_bank_account_7392",
        "get_user_dispute_history_7291",
    ):
        assert tool not in vals
        assert leaks({"system.md": f"Unlock {tool} before you call it."}, vals) == []
    # an id that appears only inside a discoverable call's JSON arguments is a customer's
    assert ("account_id", "chk_1") in _strings(
        {"agent_tool_name": "close_bank_account_7392", "arguments": '{"account_id": "chk_1"}'}
    )
    assert _strings({"ids": ["a1b2c", "d3e4f"]}) == [("ids", "a1b2c"), ("ids", "d3e4f")]
    inner = {
        v
        for t in read_task_extract("banking_knowledge")["tasks"]
        for a in t["evaluation_criteria"]["actions"]
        if a["name"] == "call_discoverable_agent_tool"
        for k, v in _strings(a.get("arguments") or {})
        if k.endswith("_id") and any(c.isdigit() for c in v) and len(v) >= 5
    }
    assert inner and inner <= vals
