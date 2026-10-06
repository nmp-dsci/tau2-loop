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
v5 additions (checks H, B, transfer, cc_015, cc_004, check_reply, guidance):
* txn_reads: account_ids whose get_bank_account_transactions returned (doc _025 req 4-5).
* card_state: card_id -> "FROZEN"/"ACTIVE"/"CLOSED" from freeze/unfreeze/close results here.
* wants_freeze, wants_cc_close, wants_reco (+ reco_cats), offer_raised, referral_talk: user words.
* given: user tools given; verified; accounts_read.
* resolved_txns: transaction ids a tool result shows RESOLVED or APPROVED (doc cc_004).
* already_exists: the last discoverable-tool result saying a record already exists (logistics_007).
* products_listed / products_read: catalog products seen in ls/search results and opened in full.
* cc_hist: {transaction_id: submitted_at} of credit-card disputes in get_user_dispute_history.
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
_FREEZE_ASK = re.compile(r"(?<!un)\bfreez|\block (?:down |up )?(?:my|the|all|it|them|those|these)\b", re.I)
_CC_CLOSE = re.compile(r"\b(close|closing|cancel+(?:ing)?|shut down|get rid of)\b[^.?!]{0,80}", re.I)
_CC_WORDS = re.compile(r"credit|rewards card|ecocard|crypto|zoom|diamond elite", re.I)
_RECO = re.compile(r"\b(recommend|best|which (?:one|account|card|of)|maximi[sz]e|highest|optimal|most (?:money|cash|interest|return|bonus)|right (?:one |account |card )?for me|better (?:account|card|option|perks))", re.I)
_OFFER = re.compile(r"\b(offer|promo(?:tion|tional)?s?|flyers?|mailers?|letter|advertis\w*|coupon)\b", re.I)
_REFERRAL = re.compile(r"\brefer(?:ral|ring|red)?s?\b", re.I)
_EXISTS = re.compile(r"already exist", re.I)
_DOC = re.compile(r"doc_(business_checking_accounts|business_savings_accounts|business_credit_cards|checking_accounts|savings_accounts|credit_cards)_([a-z0-9_\-().]+?)_(\d{3})\b")
_GENERIC_PRODUCT = re.compile(r"general|logistics|replacements|virtual_card|automatic_sweep|joint_business", re.I)
_RESOLVED = re.compile(r"\b(RESOLVED|APPROVED)\b")
_TXN = re.compile(r"\b(b?txn_[0-9A-Za-z_]+)")


def _norm_product(p: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")


def _products(text: str):
    out = []
    for cat, prod, _n in _DOC.findall(text or ""):
        if _GENERIC_PRODUCT.search(prod):
            continue
        out.append((cat, _norm_product(prod)))
    return out


def _reco_cats(text: str):
    t = text.lower()
    biz = "business" in t
    cats = set()
    if "saving" in t:
        cats.add("business_savings_accounts" if biz else "savings_accounts")
    if "checking" in t or _REFERRAL.search(t):
        cats.add("business_checking_accounts" if biz else "checking_accounts")
    if re.search(r"credit cards?|rewards card|travel insurance|cash ?back card", t) and "debit" not in t:
        cats.add("business_credit_cards" if biz else "credit_cards")
    return cats


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
    head = (result or "").strip()
    if re.match(r"(error|failed|failure)\b", head, re.I):
        return False
    if re.search(r"success", head[:120], re.I):
        return True  # e.g. "Debit Card Closed Successfully ... cannot be reactivated"
    return not _FAIL.search(head[:300])


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
        if _FREEZE_ASK.search(text):
            state["wants_freeze"] = True
        for m in _CC_CLOSE.finditer(text):
            clause = m.group(0)
            if _CC_WORDS.search(clause) and not re.search(r"debit", clause, re.I):
                state["wants_cc_close"] = True
        if _RECO.search(text):
            state["wants_reco"] = True
            state.setdefault("reco_cats", set()).update(_reco_cats(text))
        elif state.get("wants_reco"):
            state.setdefault("reco_cats", set()).update(_reco_cats(text))
        if _OFFER.search(text):
            state["offer_raised"] = True
        if _REFERRAL.search(text):
            state["referral_talk"] = True
            state["wants_reco"] = state.get("wants_reco") or bool(_RECO.search(text))
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
        listed = state.setdefault("products_listed", {})
        for cat, prod in _products(result):
            listed.setdefault(cat, [])
            if prod not in listed[cat]:
                listed[cat].append(prod)
        if name == "shell":
            cmd = str(arguments.get("command") or q)
            if re.search(r"\b(cat|head|sed|less|more|tail|awk)\b|for f in", cmd):
                read = state.setdefault("products_read", set())
                for cat, prod in _products(cmd):
                    read.add((cat, prod))
                # "cat doc_x_blue_account_*.md" style globs
                for cat, prod in re.findall(r"doc_(business_checking_accounts|business_savings_accounts|business_credit_cards|checking_accounts|savings_accounts|credit_cards)_([a-z0-9_\-]+?)_?\*", cmd):
                    read.add((cat, _norm_product(prod)))
        return

    if name == "log_verification" and _ok(result):
        state["verified"] = True
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
            if name == "give_discoverable_user_tool" and _ok(result):
                state.setdefault("given", set()).add(t)
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

    # records shown RESOLVED / APPROVED in any discoverable-tool result (doc cc_004)
    for block in re.split(r"\n\s*\d+\.\s+Record ID:|\},\s*\{", result):
        if _RESOLVED.search(block):
            for t in _TXN.findall(block):
                state.setdefault("resolved_txns", set()).add(t)
    # a result saying the record already exists (logistics_007: carry on with that record)
    if _EXISTS.search(result[:300]):
        state["already_exists"] = {"tool": b, "step": state["step"], "result": result[:160]}
    elif ok and not b.startswith("get_"):
        state.pop("already_exists", None)

    if b == "get_all_user_accounts_by_user_id":
        if ok:
            state["accounts_read"] = True
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
        hist = state.setdefault("cc_hist", {})
        for r in _records(result):
            t = r.get("transaction_id")
            if t:
                hist[t] = r.get("submitted_at") or r.get("date") or ""
    elif b == "get_bank_account_transactions":
        if ok and args.get("account_id"):
            state.setdefault("txn_reads", set()).add(args["account_id"])
    elif b in ("freeze_debit_card", "unfreeze_debit_card") and ok:
        cid = args.get("card_id") or ""
        if cid:
            state.setdefault("card_state", {})[cid] = "FROZEN" if b == "freeze_debit_card" else "ACTIVE"
            if b == "freeze_debit_card":
                state.setdefault("frozen_here", set()).add(cid)
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
        state.setdefault("card_state", {})[cid] = "CLOSED"
