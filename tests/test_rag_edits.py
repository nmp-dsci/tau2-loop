"""r4: a merge written as edits on the library's newest version of one job. No model."""

from __future__ import annotations

from typing import Any

import pytest

from tau2_loop.workflows import edits, rubric

RULE = "Eligibility: verified, age 18 or over, no more than 4 personal checking accounts"
BASE: dict[str, Any] = {
    "job": "opening_personal_checking_accounts",
    "aliases": ["open a checking account"],
    "when": {"quote": "To open a personal checking account:", "doc": "doc_a"},
    "tools": [
        {"tool": "get_user_information_by_name", "who": "agent", "step": "lookup"},
        {"tool": "open_bank_account_4821", "who": "agent", "step": "open"},
    ],
    "info": [{"field": "identity_details", "from": "customer", "ask": "Your name?"}],
    "steps": [
        {"id": "lookup", "by": "model", "call": "get_user_information_by_name"},
        {
            "id": "pick",
            "by": "model",
            "do": "recommend the account",
            "quote": "Fees vary.",
            "doc": "d",
        },
        {"id": "open", "by": "harness", "call": "open_bank_account_4821"},
    ],
    "rules": [
        {"rule": RULE, "quote": "3. The customer does not exceed 4 accounts.", "doc": "doc_a"},
        {
            "rule": "Minors open a Light Green account",
            "quote": "Under 18: Light Green.",
            "doc": "b",
        },
    ],
    "done_when": "the account is open",
}


def entry(*ops: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return {
        "job": "opening_personal_checking_accounts",
        "into": ["opening_personal_checking_accounts"],
        "edits": list(ops),
        **kw,
    }


def test_edits_change_only_what_they_name_and_are_the_changelog() -> None:
    new_step = {"id": "atm_fees", "by": "model", "do": "compare ATM fees", "if": "they travel"}
    job = edits.apply(
        entry(
            {"op": "add", "list": "steps", "after": "pick", "item": new_step, "why": "travel"},
            {
                "op": "change",
                "list": "rules",
                "key": "Eligibility: verified",
                "fields": {"doc": "c"},
            },
            {"op": "add", "list": "aliases", "item": "lowest ATM fee checking account"},
            {"op": "set", "field": "done_when", "value": "open, and the fees explained"},
            aliases=["choose a checking account"],
        ),
        BASE,
    )
    assert [s["id"] for s in job["steps"]] == ["lookup", "pick", "atm_fees", "open"]
    assert job["rules"][0] == {**BASE["rules"][0], "doc": "c"}  # the rest of the rule is kept
    assert job["rules"][1] == BASE["rules"][1] and job["tools"] == BASE["tools"]
    assert job["done_when"] == "open, and the fees explained"
    assert job["aliases"] == [
        "open a checking account",
        "lowest ATM fee checking account",
        "choose a checking account",
    ]
    assert [(c["change"], c["what"]) for c in job["changelog"]] == [
        ("added", "steps atm_fees"),
        ("changed", f"rules {RULE}"),
        ("added", "lowest ATM fee checking account"),
        ("changed", "done_when"),
    ]
    assert job["into"] == ["opening_personal_checking_accounts"] and "edits" not in job
    assert BASE["steps"][1]["id"] == "pick" and len(BASE["steps"]) == 3  # the base is not touched


def test_a_removal_names_what_it_removes_so_d8_counts_it_but_a_silent_change_is_caught() -> None:
    removed = edits.apply(
        entry(
            {
                "op": "remove",
                "list": "rules",
                "key": "Minors open a Light",
                "why": "a document says",
            }
        ),
        BASE,
    )
    assert "Under 18: Light Green." in removed["changelog"][0]["what"]
    assert rubric.lost(removed, [BASE]) == {"steps": [], "tools": [], "quotes": []}
    # a step replaced without its quote, and the why does not say so: D8 still sees the loss
    changed = edits.apply(
        entry(
            {
                "op": "change",
                "list": "steps",
                "key": "pick",
                "item": {"id": "pick", "by": "model", "do": "recommend"},
                "why": "shorter",
            }
        ),
        BASE,
    )
    assert rubric.lost(changed, [BASE])["quotes"] == ["Fees vary."]


@pytest.mark.parametrize(
    ("op", "says"),
    [
        ({"op": "remove", "list": "steps", "key": "nope"}, "no single item of steps is 'nope'"),
        ({"op": "change", "list": "rules", "key": "M", "fields": {}}, "no single item of rules"),
        ({"op": "add", "list": "steps", "item": {"id": "pick"}}, "already has 'pick'"),
        ({"op": "add", "list": "steps", "item": {"do": "x"}}, "needs its id"),
        ({"op": "set", "field": "steps", "value": []}, "set takes a `field`"),
        ({"op": "rename", "list": "steps"}, "is not add, change, remove or set"),
        ({"op": "add", "list": "documents", "item": {}}, "`list` must be one of"),
        ({"op": "change", "list": "steps", "key": "pick"}, "the `fields` it sets"),
    ],
)
def test_an_edit_that_names_nothing_or_breaks_the_job_is_refused(
    op: dict[str, Any], says: str
) -> None:
    with pytest.raises(edits.EditError, match=says.replace("`", ".").replace("(", ".")):
        edits.apply(entry(op), BASE)


def test_edits_apply_to_one_library_job() -> None:
    assert edits.base_name(entry()) == "opening_personal_checking_accounts"
    assert edits.base_name(entry(into=["a", "b"])) is None
    assert edits.base_name(entry(into=[])) is None
