"""loop/<domain>/versions.md: every version in build order, the ones the Optimise tab leaves off
between the ones it draws. Rendered from synthetic histories, and checked against the committed log."""

from __future__ import annotations

import re
from typing import Any

from tau2_loop.loop.history import load_history
from tau2_loop.loop.versions_log import path, render, spans


def _v(name: str, full: bool, **extra: Any) -> dict[str, Any]:
    run = {"run_id": f"{name}_train", "passed": 30, "n": 60, "cut": 3, "of": 60}
    return {
        "version": name,
        "model": "claude-sdk/claude-sonnet-5",
        "effort": "medium",
        "made": {
            "kind": "workflow tools",
            "cycle": None,
            "source": "v8",
            "detail": "workflows: r3 → r5",
        },
        "train": run if full else {**run, "passed": 1, "n": 2, "cut": None},
        "test": None,
        "verdict": "by hand" if full else "not gated",
        "fixed": None,
        "broke": None,
        "full": full,
        "why": None if full else f"s22: {name}'s check",
        **extra,
    }


def test_spans() -> None:
    assert spans(["v5", "v6", "v9", "v10", "v11"]) == "v5, v6, v9–v11"
    assert spans(["v1", "v2", "v3", "v7", "v8", "v23"]) == "v1–v3, v7, v8, v23"


def test_the_versions_left_off_sit_between_the_ones_it_draws() -> None:
    h = {
        "domain": "banking_knowledge",
        "champion": "v8",
        "versions": [
            _v("v8", True),
            _v("v9", False),
            _v("v10", False),
            _v("v23", True),
            _v("v24", False),
        ],
    }
    md = render(h)
    heads = re.findall(r"^#+ .+$", md, flags=re.M)
    assert heads == [
        "# banking_knowledge: every answering-agent version, in build order",
        "## v8 · on the Optimise tab · champion",
        "## Between v8 and v23 · 2 versions left off the tab",
        "### v9",
        "### v10",
        "## v23 · on the Optimise tab",
        "## After v23 · 1 version left off the tab",
        "### v24",
    ]
    assert (
        "draws the 2 of 5 versions played on every task of their split: v8, v23. The other 3 (v9, v10, v24)"
        in md
    )
    assert "train 1 of 2 played, of the split's 60 · no test run" in md
    assert "| v9 | no | workflow tools | v8 | 1 of 2 played | — | not gated |" in md
    assert "- **Why:** s22: v9's check" in md


def test_the_committed_banking_log_has_every_version_the_tab_leaves_off() -> None:
    """Fails when a version is added or played without `make versions-log DOMAIN=banking_knowledge`."""
    h = load_history("banking_knowledge")
    off = [v["version"] for v in h["versions"] if not v["full"]]
    logged = re.findall(r"^### (v\d+)$", path("banking_knowledge").read_text(), flags=re.M)
    assert logged == off and len(off) == 22  # v28 (s30) and v29 (s32) joined it
