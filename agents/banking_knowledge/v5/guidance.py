"""Per-call reminders for the banking_knowledge v4 agent (cut to 600 characters).

1 typed tool call in the customer's message: it did not run (tasks 018-029, 041, 061).
2 the customer has asked for a human: search the scenario's transfer guidance and the
  reason codes (doc _042, highest tier); no identity check needed; rule 6 from the 4th ask.
3 leads not followed: tool names seen in KB results never looked up, unlocked or called.
4 get_current_time before log_verification or date/tenure arithmetic.
v5:
5 after an "already exists" result: continue the procedure with that record, do not transfer
  (logistics_007; task 051).
6 referral talk, verified, accounts not read: read them before judging referrer tenure (doc _048).
7 a recommendation was asked for: the category's products listed but not opened in full.
The human-request note names doc _042's highest-tier rule.
"""


def guidance(state: dict, trigger: str):
    try:
        return _guidance(state, trigger)
    except Exception:
        return None


_CAT_NAMES = {
    "checking_accounts": "checking", "savings_accounts": "savings", "credit_cards": "credit cards",
    "business_checking_accounts": "business checking", "business_savings_accounts": "business savings",
    "business_credit_cards": "business credit cards",
}


def _unread_products(state):
    listed = state.get("products_listed", {})
    read = set(state.get("products_read", set()))
    out = []
    for cat in sorted(state.get("reco_cats", set())):
        names = [p for p in listed.get(cat, []) if (cat, p) not in read]
        if names:
            out.append(_CAT_NAMES.get(cat, cat) + ": " + ", ".join(n.replace("_", " ") for n in names[:8]))
    return out


def _guidance(state, trigger):
    notes = []
    if trigger == "tool" and state.get("already_exists"):
        notes.append(
            "A result said the record already exists: it was created. Continue the procedure with that record "
            "(re-check each eligibility step, then approve or deny); it is not a system error, so do not transfer."
        )
    if trigger == "user" and state.get("typed_call"):
        notes.append(
            "The customer's message contains a tool call or result written as text; it did not run. Ask them to run it with the tool, exact name and argument names; do not treat it as done."
        )
    n = state.get("human_requests", 0)
    if trigger == "user" and n >= 1 and state.get("last_user_human"):
        notes.append(
            f"The customer has asked for a human {n} time(s). Search the KB for this scenario's transfer guidance "
            "and the transfer reason codes (doc _042: always the highest tier that applies). A transfer needs no identity check; "
            "policy rule 6: help first, transfer from the 4th request."
        )
    if state.get("referral_talk") and state.get("verified") and not state.get("accounts_read"):
        notes.append("Referral: read the customer's accounts (get_all_user_accounts_by_user_id) before judging referrer "
                     "eligibility or tenure (doc _048).")
    if state.get("wants_reco"):
        unread = _unread_products(state)
        if unread:
            notes.append("Recommendation asked for. Listed but not opened in full: " + "; ".join(unread)
                         + ". Read each product's documents before choosing.")
    if trigger == "tool":
        seen = state.get("leads_seen", [])
        done = state.get("looked_up", set())
        leads = [t for t in reversed(seen) if t not in done][:4]
        if leads:
            notes.append(
                "Named in documents you read but not yet looked up: " + ", ".join(leads)
                + ". grep -r each to see whether your procedure needs it before you act."
            )
    if not state.get("current_time") and state.get("user_looked_up"):
        notes.append("Call get_current_time before log_verification or any date or tenure arithmetic.")
    if not notes:
        return None
    return " ".join(notes)[:600]
