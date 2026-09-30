"""s09's guards on the optimiser: the read fence, the code surfaces' imports, customer data, the halves."""

from __future__ import annotations

from pathlib import Path

from tau2_loop.config import ROOT
from tau2_loop.data.splits import cut_halves, halves, read_split
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


def test_banking_is_halved_into_disjoint_read_and_gate_halves_of_train() -> None:
    h = halves("banking_knowledge")
    assert h is not None
    read, gate = h
    train = read_split("banking_knowledge")["train"]
    assert len(read) == len(gate) == 24 and not set(read) & set(gate)
    assert sorted(read + gate) == sorted(train)
    assert cut_halves(train) == read_split("banking_knowledge")["halves"]  # reproducible
    assert halves("airline") is None


def test_banking_test_is_capped_at_25_and_nothing_held_back_reaches_train() -> None:
    s = read_split("banking_knowledge")
    held = s["test_cap"]["held_back"]
    assert len(s["test"]) == 25 and len(held) == 24 and s["reserve_n"] == 24
    assert set(s["v1"]["test"]) <= set(s["test"])  # every task that was test on split v1 still is
    assert not set(held) & (set(s["train"]) | set(s["test"]))


def test_committed_versions_pass_the_leak_guard() -> None:
    from tau2_loop.agent.versions import list_versions

    for domain in ("airline", "retail", "telecom", "banking_knowledge"):
        vals = leak_values(domain)
        for v in list_versions(domain):
            assert leaks(v.files(), vals) == [], f"{domain}/{v.name}"


def test_paths_resolve_from_the_repo_root(tmp_path: Path) -> None:
    assert fence_reason(str(tmp_path / "anything"), set(), []) is None
