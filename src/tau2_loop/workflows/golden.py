"""workflow_rag's golden set (s16): each train question's workflows, in gold's order, with gold's calls.

The labels are hand-written from each train task's gold answer (`data/workflows/<domain>/labels.json`,
draft until a person confirms them). Test's gold calls are mapped to the same workflows by fixed rules
(`RULES`, which must reproduce the train labels). Test is held out of optimisation, not hidden (the
person's call, 6 Oct 2026): the viewer shows each test question's mapped workflows, but no file holds
them, and `test_counts.json` keeps counts only, never a test task id, argument or text.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tau2_loop.config import DATA_DIR, DATASET_AGENTS
from tau2_loop.data import documents as kb
from tau2_loop.data.splits import read_task_extract, split_ids

WORKFLOWS_DIR = DATA_DIR / "workflows"
CALL_ARG_CHARS = 60

# (gold call prefix, workflow): what marks a workflow in a gold answer
RULES: tuple[tuple[str, str], ...] = (
    ("apply_for_credit_card", "credit_card_recommendation"),
    ("submit_cash_back_dispute", "cash_back_dispute"),
    ("file_credit_card_transaction_dispute", "credit_card_dispute"),
    ("order_replacement_credit_card", "credit_card_replacement"),
    ("log_credit_card_closure_reason", "credit_card_closure"),
    ("close_credit_card_account", "credit_card_closure"),
    ("apply_statement_credit", "credit_card_closure"),
    ("submit_credit_limit_increase_request", "credit_limit_increase"),
    ("approve_credit_limit_increase", "credit_limit_increase"),
    ("deny_credit_limit_increase", "credit_limit_increase"),
    ("pay_credit_card_from_checking", "card_balance_payment"),
    ("transfer:customer_demands_after_unavailable_offer_refusal", "unavailable_offer_refusal"),
    ("submit_transaction", "referral_bonus_status"),
    ("open_bank_account", "bank_account_opening"),
    ("close_bank_account", "bank_account_closure"),
    ("transfer_funds_between_bank_accounts", "fund_transfer"),
    ("deposit_check", "check_deposit"),
    ("apply_checking_account_credit", "atm_fee_refund"),
    ("apply_savings_account_credit", "savings_interest_correction"),
    ("submit_interest_discrepancy_report", "savings_interest_correction"),
    ("submit_referral", "referral_submission"),
    ("freeze_debit_card", "lost_stolen_debit_cards"),
    ("file_debit_card_transaction_dispute", "debit_card_dispute"),
    ("close_debit_card", "debit_card_reissue"),
    ("order_debit_card", "debit_card_reissue"),
    ("unfreeze_debit_card", "debit_card_decline"),
    ("clear_debit_card_fraud_alert", "debit_card_decline"),
    ("request_temporary_debit_card_limit_increase", "debit_card_decline"),
    ("reset_debit_card_pin", "debit_card_decline"),
    ("activate_debit_card", "debit_card_activation"),
    ("initial_transfer_to_human_agent", "payment_incident_transfer"),
    ("transfer:", "human_transfer"),
    ("request_human_agent_transfer", "human_transfer"),
)


def labels_path(domain: str) -> Path:
    return WORKFLOWS_DIR / domain / "labels.json"


def sealed_counts_path(domain: str) -> Path:
    return WORKFLOWS_DIR / domain / "test_counts.json"


def has_workflow_rag(domain: str) -> bool:
    return "workflow_rag" in DATASET_AGENTS.get(domain, ())


def _short(v: Any) -> Any:
    s = v if isinstance(v, str) else json.dumps(v)
    return s if len(s) <= CALL_ARG_CHARS else s[: CALL_ARG_CHARS - 1] + "…"


def gold_calls(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Gold's calls as a person reads them: a discoverable tool by its own name and arguments.

    The unlock before each discoverable agent tool is a prerequisite, not a step, so it is left out.
    """
    out: list[dict[str, Any]] = []
    for a in (task.get("evaluation_criteria") or {}).get("actions") or []:
        name, args = a.get("name", ""), a.get("arguments") or {}
        if name == "unlock_discoverable_agent_tool":
            continue
        if name in ("call_discoverable_agent_tool", "call_discoverable_user_tool"):
            name = args.get("agent_tool_name") or args.get("discoverable_tool_name") or name
            try:
                args = json.loads(args.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
        elif name == "give_discoverable_user_tool":
            name = f"give {args.get('discoverable_tool_name', '')}"
            args = {}
        by = "customer" if a.get("requestor") == "user" else "agent"
        out.append({"by": by, "tool": name, "args": {k: _short(v) for k, v in args.items()}})
    return out


def _situation(task: dict[str, Any]) -> str:
    text = str(((task.get("user_scenario") or {}).get("instructions")) or "")
    m = re.search(r"\*\*Your situation:\*\*(.*?)(?:\n\n|\*\*)", text, re.S)
    s = (m.group(1) if m else text).strip()
    return re.sub(r"\s+", " ", s)[:700]


def golden_set(domain: str) -> dict[str, Any] | None:
    """The golden set for the viewer: train's hand labels, test's rule-mapped workflows, and test's
    counts. None where the dataset has no workflow_rag agent or no labels yet.

    A test entry has the workflows `RULES` read off its gold calls, in gold's order, and the tools no
    train answer uses; the facts, the summary and the flag are hand labels, so it has none.
    """
    p = labels_path(domain)
    if not has_workflow_rag(domain) or not p.is_file():
        return None
    labels = json.loads(p.read_text())
    train = set(split_ids(domain, "train"))
    unknown = set(labels["tasks"]) - train
    if unknown:  # the house rule: test never enters an open file
        raise ValueError(f"labels name tasks outside train: {sorted(unknown)}")
    tasks = {t["id"]: t for t in read_task_extract(domain)["tasks"]}
    sealed = (
        json.loads(sealed_counts_path(domain).read_text())
        if sealed_counts_path(domain).is_file()
        else None
    )
    test_n = (sealed or {}).get("per_workflow", {})
    entries = []
    for tid in sorted(labels["tasks"], key=lambda t: int(re.sub(r"\D", "", t) or 0)):
        lab, task = labels["tasks"][tid], tasks[tid]
        calls = gold_calls(task)
        entries.append(
            {
                "task_id": tid,
                "workflows": lab["workflows"],
                "gold": lab["gold"],
                "flag": lab["flag"],
                "situation": _situation(task),
                "calls": calls,
                # the three things the RAG agent is scored on: the workflows above, the
                # information the customer must supply, and the documents the task names
                "info": lab.get("info") or [],
                "distractors": lab.get("distractors") or [],
                "required_documents": kb.described(
                    domain, list(task.get("required_documents") or [])
                ),
                "split": "train",
            }
        )
    known = {_base(c) for t in train for c in _call_names(tasks[t])}
    test_entries = [
        {
            "task_id": tid,
            "workflows": ordered_workflows(tasks[tid]),
            "gold": "",
            "flag": "",
            "situation": _situation(tasks[tid]),
            "calls": gold_calls(tasks[tid]),
            "info": [],
            "distractors": [],
            "required_documents": kb.described(
                domain, list(tasks[tid].get("required_documents") or [])
            ),
            "split": "test",
            "new_tools": sorted({c for c in _call_names(tasks[tid]) if _base(c) not in known}),
        }
        for tid in sorted(split_ids(domain, "test"), key=lambda t: int(re.sub(r"\D", "", t) or 0))
    ]
    uses = Counter(w for e in entries for w in e["workflows"])
    # a workflow's name is ours; what defines it is the knowledge base's own procedure document
    workflows = [
        {
            "name": w,
            **v,
            "kb_docs": kb.described(domain, list(v.get("kb_docs") or [])),
            "train": uses.get(w, 0),
            "test": test_n.get(w, 0),
        }
        for w, v in labels["workflows"].items()
    ]
    return {
        "domain": domain,
        "status": labels.get("status", "draft"),
        "source": labels.get("source", ""),
        "workflows": workflows,
        "verification_fields": labels.get("verification_fields") or [],
        "entries": entries,
        "test": sealed,
        "test_entries": test_entries,
    }


def _call_names(task: dict[str, Any]) -> list[str]:
    out = []
    for a in (task.get("evaluation_criteria") or {}).get("actions") or []:
        n, g = a.get("name", ""), a.get("arguments") or {}
        if n == "unlock_discoverable_agent_tool":
            continue
        if n == "call_discoverable_agent_tool":
            out.append(g.get("agent_tool_name", ""))
        elif n in ("call_discoverable_user_tool", "give_discoverable_user_tool"):
            out.append(g.get("discoverable_tool_name", ""))
        elif n == "transfer_to_human_agents":
            out.append(f"transfer:{g.get('reason', '')}")
        else:
            out.append(n)
    return [re.sub(r"_\d{4}$", "", c) for c in out]


def _base(c: str) -> str:
    """A call's tool, with every transfer counted as one tool whatever its reason."""
    return "transfer" if c.startswith("transfer:") else c


def mapped_workflows(task: dict[str, Any]) -> set[str]:
    """The workflows `RULES` read off a gold answer."""
    found = {w for c in _call_names(task) for p, w in RULES if c.startswith(p)}
    if (
        "lost_stolen_debit_cards" in found
    ):  # a stolen wallet's close, reorder and unfreeze belong to it
        found -= {"debit_card_reissue", "debit_card_decline"}
    if "payment_incident_transfer" in found:  # the incident protocol ends in its own transfer
        found.discard("human_transfer")
    return found


def ordered_workflows(task: dict[str, Any]) -> list[str]:
    """`mapped_workflows` in gold's order: each where its first call comes."""
    found, out = mapped_workflows(task), []
    for c in _call_names(task):
        for p, w in RULES:
            if c.startswith(p) and w in found and w not in out:
                out.append(w)
    return out


def seal_test_counts(domain: str) -> dict[str, Any]:
    """Map test's gold answers to workflows and keep only the counts; writes `test_counts.json`."""
    labels = json.loads(labels_path(domain).read_text())
    tasks = {t["id"]: t for t in read_task_extract(domain)["tasks"]}
    train, test = split_ids(domain, "train"), split_ids(domain, "test")
    mismatch = [
        t for t in train if mapped_workflows(tasks[t]) != set(labels["tasks"][t]["workflows"])
    ]

    known = {_base(c) for t in train for c in _call_names(tasks[t])}
    per_wf: Counter[str] = Counter()
    new_tools: Counter[str] = Counter()
    per_task: Counter[int] = Counter()
    covered = needs_new = 0
    for t in test:
        ws = mapped_workflows(tasks[t])
        unseen = sorted({c for c in _call_names(tasks[t]) if _base(c) not in known})
        per_wf.update(ws)
        per_task[len(ws)] += 1
        new_tools.update(unseen)
        needs_new += bool(unseen)
        covered += bool(ws) and not unseen
    out = {
        "n_test": len(test),
        "train_reproduced": len(train) - len(mismatch),
        "train_mismatches": mismatch,
        "per_workflow": dict(per_wf),
        "covered": covered,
        "needs_new": needs_new,
        "new_tools": dict(new_tools),
        "workflows_per_task": {str(k): v for k, v in sorted(per_task.items())},
    }
    if any(t in json.dumps(out) for t in test):
        raise AssertionError("a test task id would leave the sealed set")
    sealed_counts_path(domain).write_text(json.dumps(out, indent=1) + "\n")
    return out
