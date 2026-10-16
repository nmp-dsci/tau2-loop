"""Write and reply checks for the banking_knowledge v4 agent.

check_write(name, arguments, state) -> sentence | None, before every write tool call.
Each rule is decided from facts in `state` (memory.py). When a fact is missing the call
goes through. A rule blocks the same target at most twice, then lets the call through, so
an agent that has a reason to proceed is never stuck in a loop.

  A log_verification: time_verified must be get_current_time's value (policy rule 4).
  B credit-card closure flow (close, log reason, flag, statement credit, payment while a
    closure is in progress): dispute history and pending replacements must be read first
    (doc credit_card_replacements_005, credit_card_account_logistics_003 Step 1). No close
    while that card's replacement read shows a non-final order.
  C CLI approve/deny: both reads first (logistics_007 Step 2). A 'pending_disputes' denial
    whose only open disputes were filed in this conversation is refused.
  D debit disputes: from the second filing on, get_debit_dispute_status must have been
    read (doc _031 pre-filing 4). customer_max_liability_amount must be 50, 500, -1, or the
    disputed amount when it is between 50 and 500 (doc _031/_032 liability tiers).
  E order_debit_card: delivery/design fees by tier (doc _029; tiers from doc _005); ENTRY
    tier has STANDARD only and a 48-hour wait after a card closed in this conversation.
  F one apply_checking/savings_account_credit per account per conversation (doc _017).
  B0 v24: pay/log reason/statement credit/flag/close on a card with an open dispute or a non-final
    replacement while the customer asked to close a credit card (logistics_003 Step 1).
  D2 liability is a tier only (50, 500, -1); D3 yes_shared PIN gets no provisional credit (doc _032).
  E5 a non-CLASSIC card_design the customer never chose. H a second open of the same account.
  I a transfer while a CLI request is already on record (logistics_007). J a partial_refund of the
    whole transaction is full_refund (doc cc_014).
  G close_bank_account: not while a requested savings account would lose its 14-day
    checking tenure (doc _002 item 5), nor while a requested business checking account is
    unopened (doc _003 item 4: no CLOSED accounts).

check_reply(text, state): a reply that answers a repeated (2nd+) request for a human by demanding identity
details is sent back once: a transfer needs no identity verification.
"""

import datetime as _dt
import json
import re

_SUFFIX = re.compile(r"_\d{4}$")
_CLASS_SUFFIX = re.compile(r"\s*\((?:personal\s+|business\s+)?(?:savings|checking)[^)]*\)\s*$", re.I)

_TIER_BY_CLASS = {
    "light blue account": "ENTRY",
    "light green account": "ENTRY",
    "green fee-free account": "ENTRY",
    "blue account": "MID",
    "green account": "MID",
    "green account (checking)": "MID",
    "evergreen account": "PREMIUM",
    "bluest account": "ELITE",
}
# doc _029: delivery_fee by tier and option (None = option not available)
_DELIVERY = {
    "ENTRY": {"STANDARD": 0},
    "MID": {"STANDARD": 0, "EXPEDITED": 15},
    "PREMIUM": {"STANDARD": 0, "EXPEDITED": 0, "RUSH": 35},
    "ELITE": {"STANDARD": 0, "EXPEDITED": 0, "RUSH": 0},
}
_DESIGN = {
    "ENTRY": {"CLASSIC": 0, "PREMIUM": 10, "CUSTOM": 25},
    "MID": {"CLASSIC": 0, "PREMIUM": 10, "CUSTOM": 25},
    "PREMIUM": {"CLASSIC": 0, "PREMIUM": 0, "CUSTOM": 15},
    "ELITE": {"CLASSIC": 0, "PREMIUM": 0, "CUSTOM": 0},
}
_NON_FINAL = re.compile(r"\b(pending|processing|shipped|ordered|in[_ ]transit)\b", re.I)
_NO_ORDERS = re.compile(r"\bno (pending |outstanding )?(replacement )?(card )?orders?\b|\[\]|found 0|0 record", re.I)
_CLOSURE_TOOLS = ("close_credit_card_account", "log_credit_card_closure_reason", "get_closure_reason_history")


def _base(tool) -> str:
    return _SUFFIX.sub("", str(tool or ""))


def _args(raw):
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {}
    except (ValueError, TypeError):
        return {}


def _num(v):
    try:
        return float(str(v).replace("$", "").replace(",", ""))
    except (ValueError, TypeError):
        return None


def _once(state, key, message):
    """Block a given target at most twice; afterwards let it through."""
    blocks = state.setdefault("_blocks", {})
    if blocks.get(key, 0) >= 2:
        return None
    blocks[key] = blocks.get(key, 0) + 1
    return message


def _date(s):
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(str(s).strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def check_write(name: str, arguments: dict, state: dict):
    try:
        return _check(name, arguments or {}, state)
    except Exception:
        return None


def _check(name, arguments, state):
    called = state.get("called", {})

    # I ── transfer while a CLI request is already on record (logistics_007 Step 4)
    if name == "transfer_to_human_agents":
        if state.get("cli_exists"):
            return _once(state, "I", "Not transferred: the credit limit increase request is already on record ('similar request "
                         "may already exist' means it was received). Per logistics_007, re-check eligibility on the account as "
                         "it stands now and approve or deny it; approve_credit_limit_increase_5847 needs no request id.")
        return None

    # A ── log_verification
    if name == "log_verification":
        now = state.get("current_time")
        tv = str(arguments.get("time_verified") or "").strip()
        if not now:
            return _once(state, "A", "Not logged: call get_current_time first and copy its value exactly into time_verified (never assume today's date).")
        if tv != now:
            return _once(state, "A", f"Not logged: time_verified must be exactly get_current_time's value, '{now}'.")
        return None

    if name != "call_discoverable_agent_tool":
        return None
    tool = _base(arguments.get("agent_tool_name"))
    a = _args(arguments.get("arguments"))

    # B0 ── no closure step on a card that fails logistics_003 Step 1 requirement 1 or 2
    if tool in ("pay_credit_card_from_checking", "log_credit_card_closure_reason", "apply_statement_credit",
                "apply_credit_card_account_flag", "close_credit_card_account") and state.get("wants_cc_close"):
        cc = a.get("credit_card_account_id") or ""
        if cc and cc in set(state.get("cc_close_blocked", set())):
            return _once(state, "B0:" + tool + cc, "Not run: this card has an open dispute or a replacement not yet delivered, "
                         "so closure requirement 1 or 2 fails. Per logistics_003 Step 1, explain what must be resolved and "
                         "take no further closure step on it (balance payment, reason log, retention).")

    # B ── credit-card closure flow
    closure_flow = bool(
        set(state.get("unlocked", set())) & {"close_credit_card_account", "log_credit_card_closure_reason", "get_closure_reason_history"}
        or any(t in called for t in _CLOSURE_TOOLS)
    )
    gated = tool in ("close_credit_card_account", "log_credit_card_closure_reason") or (
        closure_flow and tool in ("apply_statement_credit", "apply_credit_card_account_flag", "pay_credit_card_from_checking")
    )
    if gated:
        cc = a.get("credit_card_account_id") or ""
        missing = []
        if "get_user_dispute_history" not in called:
            missing.append("get_user_dispute_history_7291 (pending disputes)")
        reads = state.get("replacement_reads", {})
        if cc and cc not in reads:
            missing.append(f"get_pending_replacement_orders_5765 for {cc}")
        if missing:
            return _once(state, "B:" + tool + cc, "Not run: before any closure-flow step on a credit card, check closure eligibility in order "
                         "(doc credit_card_account_logistics_003 Step 1; replacements_005): call " + " and ".join(missing)
                         + ". Do not rely on the customer's word.")
        if tool == "close_credit_card_account" and cc in reads:
            r = reads[cc]
            if _NON_FINAL.search(r) and not _NO_ORDERS.search(r):
                return _once(state, "Bc:" + cc, "Not closed: the pending replacement read for this card shows a non-final order; "
                             "per replacements_005 the account cannot be closed until it is delivered or cancelled. Tell the customer.")

    # C ── CLI decision
    if tool in ("approve_credit_limit_increase", "deny_credit_limit_increase"):
        cc = a.get("credit_card_account_id") or ""
        missing = []
        if "get_user_dispute_history" not in called:
            missing.append("get_user_dispute_history_7291")
        if cc and cc not in state.get("replacement_reads", {}):
            missing.append(f"get_pending_replacement_orders_5765 for {cc}")
        if missing:
            return _once(state, "C:" + cc, "Not run: logistics_007 Step 2 requires every eligibility check before the decision, "
                         "including pending disputes and pending replacement cards: call " + " and ".join(missing) + " first.")
        if tool == "deny_credit_limit_increase" and str(a.get("denial_reason")) == "pending_disputes":
            mine = set(state.get("cc_disputes_filed", set()))
            if state.get("cc_dispute_count") or mine:
                hist = " ".join(state.get("dispute_history", []))
                others = set(re.findall(r"\bdsp_[0-9a-zA-Z]+", hist)) - mine
                if not others:
                    return _once(state, "C2:" + cc, "Not denied: the only open disputes are ones filed in this conversation. "
                                 "Decide the credit limit increase on the account as it stood before them, and with several "
                                 "requests order them so one does not block another.")

    # D ── debit card disputes
    if tool == "file_debit_card_transaction_dispute":
        if state.get("debit_disputes_filed", 0) >= 1 and "get_debit_dispute_status" not in called:
            return _once(state, "D1", "Not filed: before filing another dispute, call get_debit_dispute_status_7483 to count the "
                         "account's open disputes against its tier limit (doc _031 pre-filing 4: Entry 2, Mid 3, Premium 4, Elite 5).")
        liab = _num(a.get("customer_max_liability_amount"))
        if liab is not None and liab not in (50.0, 500.0, -1.0):
            return _once(state, "D2:" + str(a.get("transaction_id")), "Not filed: customer_max_liability_amount is a Regulation E "
                         "tier (doc _031), never the disputed amount: 50 when reported within 2 business days of the statement "
                         "showing the transaction (a charge since the last statement is in time), 500 within 60 days, -1 after 60 days.")
        pce = a.get("provisional_credit_eligible")
        if str(a.get("pin_compromised") or "").strip().lower() == "yes_shared" and (pce is True or str(pce).strip().lower() == "true"):
            return _once(state, "D3:" + str(a.get("transaction_id")), "Not filed: doc _032 says provisional credit is not required "
                         "when the customer shared their PIN voluntarily (pin_compromised 'yes_shared'); set "
                         "provisional_credit_eligible false.")

    # J ── a 'partial' refund of the whole charge is a full refund (doc cc_014 item 13)
    if tool == "file_credit_card_transaction_dispute" and str(a.get("resolution_requested") or "").strip().lower() == "partial_refund":
        part = _num(a.get("partial_refund_amount"))
        whole = state.get("cc_txn_amounts", {}).get(a.get("transaction_id") or "")
        if part is not None and whole is not None and part >= whole - 0.005:
            return _once(state, "J:" + str(a.get("transaction_id")), "Not filed: the refund asked for is the whole transaction "
                         f"(${whole:,.2f}); doc cc_014 calls that 'full_refund'. Use resolution_requested 'full_refund' and omit "
                         "partial_refund_amount.")

    # H ── the same account opened twice (helper strips '(checking)'/'(savings)')
    if tool == "open_bank_account":
        typ = str(a.get("account_type") or "").strip().lower()
        cls = _CLASS_SUFFIX.sub("", str(a.get("account_class") or "")).strip().lower()
        for o in state.get("opened_accounts", []):
            if o.get("type") == typ and o.get("norm_class") == cls and cls:
                return _once(state, "H:" + typ + cls, f"Not opened: this conversation already opened this {typ} account "
                             f"(Account ID {o.get('id')}). The system records classes under the base product name, without "
                             "'(checking)' or '(savings)', so that account is the requested product. Do not open it again; "
                             "go on to the next step.")

    # E ── debit card order fees by tier
    if tool == "order_debit_card":
        acct = state.get("accounts", {}).get(a.get("account_id") or "", {})
        tier = _TIER_BY_CLASS.get(str(acct.get("level") or "").strip().lower())
        des0 = str(a.get("card_design") or "").strip().upper()
        if des0 and des0 != "CLASSIC" and not state.get("user_design"):
            return _once(state, "E5", "Not ordered: the customer has not chosen a card design. Ask them (doc _029: CLASSIC, "
                         "PREMIUM, CUSTOM, with any design fee for their tier) before ordering.")
        if tier:
            opt = str(a.get("delivery_option") or "").upper()
            des = str(a.get("card_design") or "").upper()
            dfee, gfee = _num(a.get("delivery_fee")), _num(a.get("design_fee"))
            if opt and opt not in _DELIVERY[tier]:
                return _once(state, "E1:" + opt, f"Not ordered: this is a {tier}-tier account; doc _029 allows "
                             + "/".join(_DELIVERY[tier]) + " delivery only. Offer an allowed option.")
            if opt in _DELIVERY[tier] and dfee is not None and dfee != _DELIVERY[tier][opt]:
                return _once(state, "E2", f"Not ordered: for a {tier}-tier account doc _029 sets delivery_fee {_DELIVERY[tier][opt]:g} for {opt}.")
            if des in _DESIGN[tier] and gfee is not None and gfee != _DESIGN[tier][des]:
                return _once(state, "E3", f"Not ordered: for a {tier}-tier account doc _029 sets design_fee {_DESIGN[tier][des]:g} for {des}.")
            if tier == "ENTRY" and a.get("account_id") in set(state.get("closed_cards", {}).values()):
                return _once(state, "E4", "Not ordered: ENTRY tier has a 48-hour waiting period after a card closure before a "
                             "replacement can be ordered (doc _029). Tell the customer when they can order.")

    # F ── one credit per account
    if tool in ("apply_checking_account_credit", "apply_savings_account_credit"):
        if a.get("account_id") and a.get("account_id") in state.get("credited", set()):
            return _once(state, "F:" + str(a.get("account_id")), "Not applied: this account was already credited in this conversation; "
                         "only one credit per account is allowed (doc _017), combining all corrections net into a single amount.")

    # G ── close_bank_account ordering
    if tool == "close_bank_account":
        aid = a.get("account_id") or ""
        accts = state.get("accounts", {})
        if state.get("wants_business_checking") and not state.get("opened_business_checking"):
            return _once(state, "G1", "Not closed: the customer also asked for a business checking account, which requires no "
                         "CLOSED accounts (doc _003). Open it first, then close.")
        if state.get("wants_savings") and not state.get("opened_savings") and accts.get(aid, {}).get("type") == "checking":
            now = _date(str(state.get("current_time") or "")[:10])
            if now:
                keep = [x for k, x in accts.items() if k != aid and x.get("type") == "checking"
                        and x.get("status") == "OPEN" and _date(x.get("date_opened"))
                        and (now - _date(x.get("date_opened"))).days >= 14]
                if not keep:
                    return _once(state, "G2", "Not closed: the customer asked to open a savings account, which needs a checking "
                                 "account held at least 14 days (doc _002); this close would leave none. Open the savings "
                                 "account first, then close this one (or confirm the customer no longer wants savings).")
    return None


_ID_ASK = re.compile(
    r"\b(verify your identity|verification|date of birth|\bDOB\b|phone number|email address|"
    r"your address|two of the following)\b",
    re.I,
)


_INVENTED = re.compile(
    r"\bcan(?:not|'t|\u2019t) tell you to\b|\bcan(?:not|'t|\u2019t) confirm (?:that|it) would\b|"
    r"\bcan(?:not|'t|\u2019t) (?:look at|share|access) (?:her|his|their)\b|\bnot permitted\b|\bnot allowed\b|"
    r"\bagainst (?:our )?policy\b|\bprohibited\b",
    re.I,
)


def check_reply(text: str, state: dict):
    try:
        if state.get("last_user_human") and state.get("human_requests", 0) >= 2 and isinstance(text, str) and _ID_ASK.search(text):
            return _once(state, "R:" + str(state.get("step")), "A transfer to a human needs no identity verification. Follow the KB "
                         "transfer guidance for this scenario (search it), or policy rule 6: help first, and transfer once the "
                         "customer has asked for a human 4 times.")
        if isinstance(text, str) and not state.get("_invented_sent") and _INVENTED.search(text):
            state["_invented_sent"] = True
            return ("Cite the policy or document sentence that states this restriction, or drop it (policy guideline 1: "
                    "invent no restriction). A customer acting with their own tool needs no permission from you.")
    except Exception:
        return None
    return None
