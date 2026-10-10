"""Conversation memory for the banking_knowledge v4 agent.

remember(state, name, arguments, result) runs after every tool result and every user
message (name == "user"). It records the facts checks.py and guidance.py read:

* current_time: get_current_time's exact "YYYY-MM-DD HH:MM:SS TZ" (log_verification, tenure).
* accounts: account_id -> {type, level, status, date_opened} from get_all_user_accounts_*.
* debit_cards: card_id -> account_id from debit-card reads and dispute/close args.
* called: discoverable tools called (name without the 4-digit suffix) -> list of parsed args.
* replacement_reads: credit_card_account_id -> raw result of get_pending_replacement_orders.
* dispute_history: raw results of get_user_dispute_history.
* writes this conversation: credited accounts, debit disputes filed, credit-card dispute ids
  filed, savings / business checking opened, debit cards closed (card_id -> account_id).
* from user messages: human_requests, wants_savings, wants_business_checking, typed_call.
* leads: tool names (name_1234) seen in knowledge-base results, and those looked up.
"""

import json
import re

_TOOL_NAME = re.compile(r"\b([a-z][a-z_]*[a-z]_\d{4})\b")
_SUFFIX = re.compile(r"_\d{4}$")
_TIME = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?: [A-Z]{2,4})?)")
_HUMAN = re.compile(
    r"\b(human|real person|live person|representative|supervisor)\b|"
    r"\b(?:your|a|the|to)\s+manager\b|\btransfer (?:me|this|my call)\b|"
    r"\b(?:speak|talk) (?:to|with) (?:an? |another |a real |a live )?(?:agent|someone|somebody)\b",
    re.I,
)
_WANTS_SAVINGS = re.compile(
    r"\b(open|opening|start|get|set up|setup|new|want|like)\b[^.?!]{0,60}\bsavings\b", re.I
)
_WANTS_BIZ_CHK = re.compile(r"\bbusiness (checking|account)\b", re.I)
_TYPED_CALL = re.compile(
    r"call_discoverable_user_tool|<function_calls>|\"name\"\s*:|\[tool result|"
    r"Executed: [a-z_]+_\d{4}|\"tool_calls\"",
    re.I,
)
_BUSINESS_CLASSES = (
    "navy blue", "cobalt blue", "hunter green", "lime green", "world blue",
    "true blue", "beige", "sky blue",
)
_FAIL = re.compile(r"\b(error|failed|cannot|not allowed|ineligible|denied|invalid|unable)\b", re.I)


def base(tool: str) -> str:
    return _SUFFIX.sub("", str(tool or ""))


def parse_args(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _records(text: str):
    """Split 'N. Record ID: x\\n   key: value' blocks into dicts."""
    out = []
    for block in re.split(r"\n\s*\d+\.\s+Record ID:", "\n" + text):
        rec = {}
        for m in re.finditer(r"^\s*([a-z_]+):\s*(.+?)\s*$", block, re.M):
            rec.setdefault(m.group(1), m.group(2))
        if rec:
            out.append(rec)
    # JSON results, if any
    try:
        val = json.loads(text)
        items = val if isinstance(val, list) else (val.get("accounts") or val.get("cards") or [val]) if isinstance(val, dict) else []
        for it in items:
            if isinstance(it, dict):
                out.append({k: str(v) for k, v in it.items()})
    except (ValueError, TypeError, AttributeError):
        pass
    return out


def _ok(result: str) -> bool:
    return not _FAIL.search(result[:300])


def remember(state: dict, name: str, arguments: dict, result: str) -> None:
    try:
        _remember(state, name, arguments or {}, result if isinstance(result, str) else str(result or ""))
    except Exception:
        pass


def _remember(state, name, arguments, result):
    state.setdefault("called", {})
    state.setdefault("accounts", {})
    state.setdefault("debit_cards", {})
    state.setdefault("leads_seen", [])
    state.setdefault("looked_up", set())
    state["step"] = state.get("step", 0) + 1

    if name == "user":
        text = result or ""
        state["last_user"] = text
        state["last_user_human"] = bool(_HUMAN.search(text))
        if state["last_user_human"]:
            state["human_requests"] = state.get("human_requests", 0) + 1
        if _WANTS_SAVINGS.search(text):
            state["wants_savings"] = True
        if _WANTS_BIZ_CHK.search(text) and re.search(r"\b(open|new|need|want|set up)\b", text, re.I):
            state["wants_business_checking"] = True
        state["typed_call"] = bool(_TYPED_CALL.search(text))
        return

    # knowledge-base leads
    if name in ("KB_search_bm25", "KB_search_dense", "shell"):
        q = json.dumps(arguments)
        for t in _TOOL_NAME.findall(q):
            state["looked_up"].add(t)
        if name == "shell" and re.search(r"\bgrep\b", q):
            for t in re.findall(r"[a-z_]+_\d{4}|[a-z_]{6,}", q):
                state["looked_up"].add(t)
        for t in _TOOL_NAME.findall(result):
            if t not in state["leads_seen"]:
                state["leads_seen"].append(t)
        return

    if name.startswith("get_user_information"):
        state["user_looked_up"] = True
        return

    if name == "get_current_time":
        m = _TIME.search(result)
        if m:
            state["current_time"] = m.group(1)
        return

    if name in ("unlock_discoverable_agent_tool", "give_discoverable_user_tool"):
        t = arguments.get("agent_tool_name") or arguments.get("discoverable_tool_name")
        if t:
            state["looked_up"].add(t)
            state.setdefault("unlocked", set()).add(base(t))
        return

    if name != "call_discoverable_agent_tool":
        return

    tool = str(arguments.get("agent_tool_name") or "")
    state["looked_up"].add(tool)
    b = base(tool)
    args = parse_args(arguments.get("arguments"))
    ok = _ok(result)
    if ok:
        state["called"].setdefault(b, []).append(args)

    if b == "get_all_user_accounts_by_user_id":
        for r in _records(result):
            aid = r.get("account_id")
            if not aid or aid.startswith("cc_") or "card_type" in r:
                continue
            state["accounts"][aid] = {
                "type": (r.get("class") or r.get("account_type") or "").lower(),
                "level": r.get("level") or r.get("account_class") or "",
                "status": (r.get("status") or "").upper(),
                "date_opened": r.get("date_opened") or "",
            }
    elif b == "get_debit_cards_by_account_id":
        for r in _records(result):
            cid = r.get("card_id")
            if cid:
                state["debit_cards"][cid] = r.get("account_id") or args.get("account_id") or ""
    elif b == "get_pending_replacement_orders":
        cc = args.get("credit_card_account_id") or ""
        state.setdefault("replacement_reads", {})[cc] = result
    elif b == "get_user_dispute_history":
        state.setdefault("dispute_history", []).append(result)
    elif b == "file_credit_card_transaction_dispute" and ok:
        for d in re.findall(r"\bdsp_[0-9a-zA-Z]+", result):
            state.setdefault("cc_disputes_filed", set()).add(d)
        state["cc_dispute_count"] = state.get("cc_dispute_count", 0) + 1
    elif b == "file_debit_card_transaction_dispute" and ok:
        state["debit_disputes_filed"] = state.get("debit_disputes_filed", 0) + 1
        if args.get("card_id") and args.get("account_id"):
            state["debit_cards"][args["card_id"]] = args["account_id"]
    elif b in ("apply_checking_account_credit", "apply_savings_account_credit") and ok:
        if args.get("account_id"):
            state.setdefault("credited", set()).add(args["account_id"])
    elif b == "open_bank_account" and ok:
        typ = str(args.get("account_type") or "").lower()
        cls = str(args.get("account_class") or "").lower()
        m = re.search(r"account(?:_id)?[:\s]+([A-Za-z0-9_]+)", result)
        if "savings" in typ:
            state["opened_savings"] = True
        if "business" in typ or any(c in cls for c in _BUSINESS_CLASSES):
            state["opened_business_checking"] = True
        if m and "checking" in typ:
            state["accounts"].setdefault(m.group(1), {
                "type": "checking", "level": args.get("account_class") or "",
                "status": "OPEN", "date_opened": "",
            })
    elif b == "close_bank_account" and ok:
        aid = args.get("account_id")
        if aid in state["accounts"]:
            state["accounts"][aid]["status"] = "CLOSED"
    elif b == "close_debit_card" and ok:
        cid = args.get("card_id") or ""
        state.setdefault("closed_cards", {})[cid] = state["debit_cards"].get(cid, "")
