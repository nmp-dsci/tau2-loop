"""Conversation memory for the banking_knowledge v4 agent.

remember(state, name, arguments, result) runs after every tool result and every user
message (name == "user"). It records the facts checks.py and guidance.py read:

* current_time: get_current_time's exact "YYYY-MM-DD HH:MM:SS TZ" (log_verification, tenure).
* accounts: account_id -> {type, level, status, date_opened} from get_all_user_accounts_*.
* debit_cards: card_id -> account_id from debit-card reads and dispute/close args.
* called: discoverable tools called (name without the 4-digit suffix) -> list of parsed args.
* tools_called (v27): every tool that ran, discoverable or not, by its base name (for A3).
* replacement_reads: credit_card_account_id -> raw result of get_pending_replacement_orders.
* dispute_history: raw results of get_user_dispute_history.
* writes this conversation: credited accounts, debit disputes filed, credit-card dispute ids
  filed, savings / business checking opened, debit cards closed (card_id -> account_id).
* from user messages: human_requests, wants_savings, wants_business_checking, typed_call.
* leads: tool names (name_1234) seen in knowledge-base results, and those looked up.
* v24: opened_accounts (type, class without its '(checking)' qualifier, id); cli_exists after a
  CLI submission says a similar request exists, until an approve/deny; cc_txn_amounts and
  cc_txn_type by transaction_id; cc_accounts (id -> card type); open_cc_disputes and
  cc_close_blocked (cards failing logistics_003 Step 1 req. 1-2); wants_cc_close, user_design.
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
_CLASS_SUFFIX = re.compile(r"\s*\((?:personal\s+|business\s+)?(?:savings|checking)[^)]*\)\s*$", re.I)
_CLI_EXISTS = re.compile(r"already exist|similar request", re.I)
_WANTS_CC_CLOSE = (
    re.compile(r"\bclos\w*\b[^.?!]{0,60}\b(card|credit)\b", re.I),
    re.compile(r"\b(card|credit)\b[^.?!]{0,60}\bclos", re.I),
)
_USER_DESIGN = re.compile(r"\b(design|metallic|custom|classic)\b|premium (design|look|card design)", re.I)
_FINAL_DISPUTE = {"RESOLVED", "CLOSED", "DENIED", "REJECTED", "WITHDRAWN", "APPROVED", "COMPLETED"}
_NON_FINAL = re.compile(r"\b(pending|processing|shipped|ordered|in[_ ]transit)\b", re.I)
_NO_ORDERS = re.compile(r"\bno (pending |outstanding )?(replacement )?(card )?orders?\b|\[\]|found 0|0 record", re.I)


def norm_class(cls) -> str:
    """account_class compared without a '(checking)'/'(savings)' qualifier, case-insensitive."""
    return _CLASS_SUFFIX.sub("", str(cls or "")).strip().lower()


def _amount(v):
    try:
        return float(str(v).replace("$", "").replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _recompute_cc_blocked(state):
    """Credit cards whose closure fails logistics_003 Step 1 requirement 1 or 2."""
    accts = state.get("cc_accounts", {})
    by_type = {}
    for aid, typ in accts.items():
        by_type.setdefault(typ.strip().lower(), []).append(aid)
    blocked = set()
    for d in state.get("open_cc_disputes", []):
        typ = state.get("cc_txn_type", {}).get(d.get("transaction_id") or "")
        if typ and by_type.get(typ.strip().lower()):
            blocked.update(by_type[typ.strip().lower()])
        elif len(accts) == 1:
            blocked.update(accts)
    for cc, r in state.get("replacement_reads", {}).items():
        if cc and _NON_FINAL.search(r) and not _NO_ORDERS.search(r):
            blocked.add(cc)
    state["cc_close_blocked"] = blocked


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
    if name != "user" and _ok(result):  # v27 (A3): every tool that ran, by its base name
        t = arguments.get("agent_tool_name") if name == "call_discoverable_agent_tool" else name
        if t:
            state.setdefault("tools_called", set()).add(base(str(t)))

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
        for rx in _WANTS_CC_CLOSE:
            m = rx.search(text)
            if m and not re.search(r"\bdebit\b", m.group(0), re.I):
                state["wants_cc_close"] = True
        if _USER_DESIGN.search(text):
            state["user_design"] = True
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

    if name == "get_credit_card_transactions_by_user":
        for r in _records(result):
            tid = r.get("transaction_id")
            if not tid:
                continue
            amt = _amount(r.get("transaction_amount"))
            if amt is not None:
                state.setdefault("cc_txn_amounts", {})[tid] = amt
            if r.get("credit_card_type"):
                state.setdefault("cc_txn_type", {})[tid] = r["credit_card_type"]
        _recompute_cc_blocked(state)
        return

    if name == "get_credit_card_accounts_by_user":
        for r in _records(result):
            aid = r.get("account_id")
            if aid and r.get("card_type"):
                state.setdefault("cc_accounts", {})[aid] = r["card_type"]
        _recompute_cc_blocked(state)
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
        _recompute_cc_blocked(state)
    elif b == "get_user_dispute_history":
        state.setdefault("dispute_history", []).append(result)
        opened = []
        for r in _records(result):
            if not (r.get("dispute_id") or "").strip():
                continue
            if (r.get("status") or "").strip().upper() in _FINAL_DISPUTE:
                continue
            opened.append({"dispute_id": r["dispute_id"], "transaction_id": r.get("transaction_id") or ""})
        state["open_cc_disputes"] = opened
        _recompute_cc_blocked(state)
    elif b == "submit_credit_limit_increase_request":
        if _CLI_EXISTS.search(result):
            state["cli_exists"] = True
    elif b in ("approve_credit_limit_increase", "deny_credit_limit_increase") and ok:
        state["cli_exists"] = False
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
        mid = re.search(r"Account ID:\s*([A-Za-z0-9_]+)", result)
        if mid:
            state.setdefault("opened_accounts", []).append(
                {"type": typ.strip(), "norm_class": norm_class(cls), "id": mid.group(1)})
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
