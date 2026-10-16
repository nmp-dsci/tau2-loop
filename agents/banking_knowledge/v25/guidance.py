"""Per-call reminders for the banking_knowledge v4 agent (cut to 600 characters).

1 typed tool call in the customer's message: it did not run (tasks 018-029, 041, 061).
2 the customer has asked for a human: search the scenario's transfer guidance and the
  reason codes (doc _042, highest tier); no identity check needed; rule 6 from the 4th ask.
3 leads not followed: tool names seen in KB results never looked up, unlocked or called.
4 get_current_time before log_verification or date/tenure arithmetic.
"""


def guidance(state: dict, trigger: str):
    try:
        return _guidance(state, trigger)
    except Exception:
        return None


def _guidance(state, trigger):
    notes = []
    if trigger == "user" and state.get("typed_call"):
        notes.append(
            "The customer's message contains a tool call or result written as text; it did not run. Ask them to run it with the tool, exact name and argument names; do not treat it as done."
        )
    n = state.get("human_requests", 0)
    if trigger == "user" and n >= 1 and state.get("last_user_human"):
        notes.append(
            f"The customer has asked for a human {n} time(s). Search the KB for this scenario's transfer guidance "
            "and the transfer reason codes (pick the highest tier that applies). A transfer needs no identity check; "
            "policy rule 6: help first, transfer from the 4th request."
        )
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
