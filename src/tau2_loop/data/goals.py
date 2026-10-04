"""A one-line customer goal, derived from a task's scenario, where tau2 gives no real purpose.

banking_knowledge's tasks all carry the placeholder purpose `Task: <id>`, so the Evals table had
nothing to say about what each customer wants. The scenario says it, in one of three shapes:

- prose: "You are playing the role of a customer …. Your character is …. You're looking for a
  credit card to use for your everyday purchases. …";
- labelled sections: `**Your character:**`, `**Your situation:**`, `**Your goal:**`, then the
  scripted lines (`**Opening:**`, `**If asked …:**`, `**Verification info:**`, list items);
- a character section and a `## Conversation Flow` script, with the goal only in the character's
  later sentences or the customer's opening line.

The rule: drop the boilerplate first sentence and the character sentence; a `Your goal` section's
first sentence is the goal; else the first narrative sentence that states what the customer wants
(`looking for`, `want`, `need`, `would like`, `calling`), one about "you" first; else the
character's later sentences by the same test; else the opening line (its first wanting sentence,
else its first full sentence); else a scripted item by the same test; else the first narrative
sentence left. It is derived text, never tau2's: the viewer labels it so.
"""

from __future__ import annotations

import re
from typing import Any

BOILERPLATE = re.compile(
    r"^\s*You are playing the role of a customer contacting a customer service representative"
    r" agent\.\s*",
    re.I,
)
WANTS = re.compile(
    r"\b(looking for|looking to|wants?|wanted|needs?|would like|'d like|calling)\b", re.I
)
PLACEHOLDER = re.compile(r"^Task: \S+$")
LABEL = re.compile(r"^\*\*([^*]+?)\*\*:?\s*")
CHARACTER = re.compile(r"^Your character\b")
ABOUT_YOU = re.compile(r"^You\b")
ITEM = re.compile(r"^\s*(?:[-*•]|\d+[a-z]?[.)])\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])[\"”]?\s+(?=[A-Z\"'(*“])")
QUOTES = "\"“”' "
GREETING = re.compile(r"^(?:Hi|Hello|Hey)(?: there)?[!,.]?\s+", re.I)
NARRATIVE = ("", "your situation", "your goal")
MIN_OPENING_WORDS = 6
MAX_CHARS = 240

Row = tuple[str, str, str]  # (section label, kind, sentence)


def is_placeholder(purpose: str | None) -> bool:
    """tau2's stand-in purpose (`Task: task_001`), or none at all."""
    return not purpose or bool(PLACEHOLDER.match(purpose.strip()))


def _label(text: str) -> tuple[str, str]:
    m = LABEL.match(text)
    if not m:
        return "", text
    name = re.sub(r"\s*\(.*?\)", "", m.group(1)).rstrip(": ").strip().lower()
    return name, text[m.end() :]


def _rows(text: str) -> list[Row]:
    """Every sentence in reading order. Kind is `prose`, `character` (the character section after
    its first sentence), `item` (a list item, kept whole) or `script` (prose after a markdown
    heading such as `## Conversation Flow`, which is what the customer says when asked)."""
    out: list[Row] = []
    script = False
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if para.startswith("#"):
            script = True
        if not para or para.startswith(("#", "---")):
            continue
        label, body = _label(para)
        character = label.startswith("your character")
        first = True
        for line in body.split("\n"):
            line = line.strip()
            if not line or line.startswith(("#", "---")):
                continue
            if ITEM.match(line):
                sub, rest = _label(ITEM.sub("", line))
                out.append((sub or label, "item", rest))
                continue
            sub, rest = _label(line)  # a labelled line inside a paragraph: `**Opening:** "…"`
            for s in SENTENCE_END.split(rest if sub else line):
                s = s.strip()
                if not s:
                    continue
                if (character and first) or CHARACTER.match(s):
                    first = False  # the character sentence: who they are, never what they want
                    continue
                first = False
                kind = "character" if character else "script" if script else "prose"
                out.append((sub or label, kind, s))
    return out


def _first_wanting(cands: list[tuple[int, str]]) -> tuple[int, str] | None:
    """The first sentence that says what the customer wants, one about "you" before any other."""
    wanting = [(i, s) for i, s in cands if WANTS.search(s)]
    return next((c for c in wanting if ABOUT_YOU.match(c[1])), wanting[0] if wanting else None)


def _opening(rows: list[Row]) -> str | None:
    """The line the customer is scripted to open with: its first wanting sentence, else its first
    sentence of `MIN_OPENING_WORDS` words or more, in quotes."""
    said: list[str] = []
    for label, kind, s in rows:
        if not label.startswith("opening"):
            continue
        for part in SENTENCE_END.split(s) if kind == "item" else [s]:
            q = GREETING.sub("", part.strip(QUOTES)).strip(QUOTES)
            if q:
                said.append(q[0].upper() + q[1:])
    full = [q for q in said if len(q.split()) >= MIN_OPENING_WORDS]
    pick = next((q for q in full if WANTS.search(q)), full[0] if full else None)
    return f"“{pick}”" if pick else None


def _clip(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= MAX_CHARS:
        return s
    return s[: MAX_CHARS - 1].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def customer_goal(instructions: Any) -> str | None:
    """The customer's goal in one sentence, or None when the scenario is not prose (tau2's
    structured scenarios carry a `reason_for_call`, which already says it)."""
    if not isinstance(instructions, str) or not instructions.strip():
        return None
    rows = _rows(BOILERPLATE.sub("", instructions))
    narrative = [
        (i, label, s)
        for i, (label, kind, s) in enumerate(rows)
        if kind == "prose" and label in NARRATIVE and not s.startswith(('"', "“"))
    ]
    pick = next(((i, s) for i, label, s in narrative if label == "your goal"), None)
    pick = pick or _first_wanting([(i, s) for i, _, s in narrative])
    pick = pick or _first_wanting([(i, s) for i, (_, k, s) in enumerate(rows) if k == "character"])
    if pick is None:
        opening = _opening(rows)
        if opening:
            return _clip(opening)
        pick = _first_wanting([(i, s) for i, (_, k, s) in enumerate(rows) if k == "item"])
    if pick is None and narrative:
        pick = (narrative[0][0], narrative[0][2])
    if pick is None:
        return None
    i, s = pick
    if s.endswith(":"):  # "You've saved up $20,000 and want to:" and then the list that says what
        items = []
        for _, kind, x in rows[i + 1 :]:
            if kind != "item":
                break
            items.append(x.rstrip(".;"))
        if items:
            s = f"{s} {'; '.join(items)}"
    return _clip(s)
