"""Deterministic hooks for the airline v2 agent.

Only one hook is used: extra_context, which re-states the policy's own
"current time" line at the very end of the system prompt. Transcripts from
the champion (v0) showed the model overriding the policy's stated current
time (2024-05-15 in train) with some other notion of "today" (often the
real-world date) and then wrongly concluding that in-policy flight dates had
"already been flown", causing it to refuse actions or transfer to a human
agent instead of completing the task (e.g. tasks 11, 22, 24, 44, 14, 23, 25,
39 in the train failures). Repeating the policy's own current-time line right
after the policy (a recency-favoured position) reduces that drift. The line
is extracted from the policy text itself, never hard-coded, so this
generalises to whatever date the test split's policy states.

This is the same hook used by the held v1 challenger, which showed the
reminder itself does not cause harm; the one v1 regression (task 19) traced
to extra confirmation turns added in system.md, not to this hook, so it is
kept unchanged here.
"""

import re

_CURRENT_TIME_RE = re.compile(r"The current time is[^.\n]*\.", re.IGNORECASE)


def extra_context(policy: str) -> str:
    match = _CURRENT_TIME_RE.search(policy)
    if not match:
        return ""
    stmt = match.group(0).strip()
    return (
        "\n<reminder>\n"
        f"Reminder: {stmt} Use exactly this date/time as \"today\" for every judgment "
        "about whether a flight date is in the past or upcoming. Do not substitute any "
        "other date. A flight dated on or after this current time has not been flown yet.\n"
        "</reminder>\n"
    )
