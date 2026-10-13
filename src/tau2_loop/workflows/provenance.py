"""Every clause from its source (s23, r18): which steps a reviewer reads, and what a session read.

s23 found that the rubric checked a quote exists (D1), never that it supports its step: all eight of
v23's misread workflow failures carried a real quote, found word for word, under a clause it does
not say. r18 gives a step one `basis` entry per deciding clause ({clause, quote, doc}, or
{clause, source: "none", why} for one no document gives) and adds three checks:

  D14 clauses   a step whose clauses may not follow from its quotes is read by a reviewer, a
                model in its own context that sees no gold and no outcome, kept steps as well
                as new ones (M1–M3). Code picks the steps; the reviewer judges them.
  D15 read      every document the session printed in full is cited by a job or set aside with
                its reason (M4). r8 printed doc 037 and r13 gold_account_009; neither was cited.
  D16 named     (`toolspec.named_titles`) a called tool's description that names a document's
                title is followed to that document, or the document is set aside (M5).

Code picks the steps a reviewer reads with four checks measured on r17's 1,290 steps (s23,
`why_misread.md`): a value the documents define missing from the step's quotes (C2, 11% of
steps), a hedged quote under an imperative (C3, 3%), more deciding clauses than quotes (C4, 33%),
and a condition no quote or cited document shares a word with (C7, 21%). Together they flag all
six defects behind the eight failures. None of them is a gate: the reviewer's "no" is what goes
back to the writer, once. No gold is read here.
"""

from __future__ import annotations

import json
import re
from functools import cache
from typing import Any

from tau2_loop.data import documents as kb

ENUM = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")  # snake_case values: pin_purchase, full_refund
DECIDE = re.compile(
    r"\b(if|unless|only|when|never|do not|don't|drop|leave .{0,20} out|stays?|otherwise|instead)\b",
    re.I,
)
HEDGE = re.compile(r"\b(may|might|can|could|possibly|eligible for)\b", re.I)
IMPER = re.compile(r"\b(add|apply|always|use|give|credit|include)\b", re.I)
STOP = frozenset(
    [
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "is",
        "are",
        "be",
        "it",
        "its",
        "this",
        "that",
        "as",
        "if",
        "when",
        "then",
        "each",
        "any",
        "all",
        "not",
        "no",
        "only",
        "do",
        "does",
        "customer",
        "their",
        "they",
        "have",
        "been",
        "what",
    ]
)
PRINTS = re.compile(r"\b(cat|head|tail|sed|awk|less)\b")
DOC_ID = re.compile(r"\bdoc_[A-Za-z0-9_()\-]+")
CONTEXT_CHARS = 250  # the reviewer sees this much of the document either side of a quote


@cache
def values(domain: str) -> frozenset[str]:
    """The value vocabulary: snake_case tokens a document writes in quotes ('pin_purchase')."""
    out: set[str] = set()
    for i in kb.read_index(domain):
        d = kb.document(domain, i)
        if d:
            out |= set(re.findall(r"['\"`]([a-z]+(?:_[a-z]+)+)['\"`]", d["content"]))
    return frozenset(out)


def basis(step: dict[str, Any]) -> list[dict[str, Any]]:
    return [b for b in step.get("basis") or [] if isinstance(b, dict) and b.get("clause")]


def _quotes(step: dict[str, Any]) -> list[dict[str, str]]:
    """The step's own quote and its basis quotes, each with the document it cites."""
    out = (
        [{"quote": str(step["quote"]), "doc": str(step.get("doc") or "")}]
        if step.get("quote")
        else []
    )
    out += [
        {"quote": str(b["quote"]), "doc": str(b.get("doc") or "")}
        for b in basis(step)
        if b.get("quote")
    ]
    return out


def _text(domain: str, doc: str, policy: str) -> str:
    from tau2_loop.workflows.rag_agent import POLICY_REFS, resolve_doc

    if doc.strip().lower() in POLICY_REFS:
        return policy
    i = resolve_doc(domain, doc) if doc else None
    d = kb.document(domain, i) if i else None
    return str(d["content"]) if d else ""


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z_]{3,}", t.lower()) if w not in STOP}


def step_flags(domain: str, step: dict[str, Any], policy: str = "") -> dict[str, Any]:
    """C2, C3, C4 and C7 for one step; empty when none fires."""
    do, cond = str(step.get("do") or ""), str(step.get("if") or "")
    body = f"{do} {cond}"
    qs = _quotes(step)
    qtext = " ".join(q["quote"] for q in qs)
    src = (qtext + " " + " ".join(_text(domain, q["doc"], policy) for q in qs)).lower()
    out: dict[str, Any] = {}
    missing = sorted({v for v in ENUM.findall(body) if v in values(domain) and v not in qtext})
    if missing:
        out["C2"] = missing
    if HEDGE.search(qtext) and IMPER.search(do):
        out["C3"] = True
    sentences = [x for x in re.split(r"(?<=[.;:])\s+|\n", body) if x.strip()]
    deciding = [x for x in sentences if DECIDE.search(x)]
    accounted = len(qs) + sum(1 for b in basis(step) if not b.get("quote"))
    if len(deciding) > accounted:
        out["C4"] = {"deciding": len(deciding), "quotes": accounted}
    unsourced = [
        cl.strip()
        for cl in re.split(r",|;|\band\b|\bor\b", cond)
        if (w := _words(cl)) and not any(x[:6] in src for x in w)
    ]
    if unsourced:
        out["C7"] = unsourced
    return out


def flagged(domain: str, job: dict[str, Any], policy: str = "") -> list[dict[str, Any]]:
    """Every step of a job (kept or new, M3) whose clauses a reviewer must read, with why."""
    out = []
    for st in job.get("steps") or []:
        if isinstance(st, dict) and st.get("id") and (f := step_flags(domain, st, policy)):
            out.append({"step": str(st["id"]), "flags": f})
    return out


def _context(domain: str, quote: str, doc: str, policy: str) -> str:
    text = _text(domain, doc, policy)
    if not text:
        return ""
    sq = re.sub(r"\s+", " ", text)
    q = re.sub(r"\s+", " ", quote.strip(" .…\"'"))
    i = sq.find(q)
    if i < 0:
        return ""
    return sq[max(0, i - CONTEXT_CHARS) : i + len(q) + CONTEXT_CHARS]


def review_request(
    domain: str, prompt: str, job: dict[str, Any], picks: list[dict[str, Any]], policy: str = ""
) -> str:
    """The reviewer's message: its instructions (the version's `review.md`), then each picked
    step with its clauses, its quotes and the text around each quote."""
    steps = {str(s.get("id")): s for s in job.get("steps") or [] if isinstance(s, dict)}
    parts = []
    for p in picks:
        st = steps.get(p["step"])
        if st is None:
            continue
        lines = [f"### step {p['step']}", f"do: {st.get('do') or ''}"]
        if st.get("if"):
            lines.append(f"if: {st['if']}")
        for q in _quotes(st):
            lines.append(f'quote ({q["doc"]}): "{q["quote"]}"')
            ctx = _context(domain, q["quote"], q["doc"], policy)
            if ctx:
                lines.append(f"  around it: …{ctx}…")
        for b in basis(st):
            if not b.get("quote"):
                lines.append(f'marked as no document\'s: "{b["clause"]}" ({b.get("why") or ""})')
            else:
                lines.append(
                    f'clause "{b["clause"]}" rests on "{b["quote"]}" ({b.get("doc") or ""})'
                )
        lines.append(f"why code picked it: {json.dumps(p['flags'], ensure_ascii=False)}")
        parts.append("\n".join(lines))
    return prompt.format(job=job.get("job", ""), steps="\n\n".join(parts))


def parse_review(text: str | None) -> list[dict[str, str]]:
    """The reviewer's verdicts, [{step, clause, verdict, unsupported, why}]; [] if unreadable."""
    from tau2_loop.workflows.rag_agent import _json_object

    obj = _json_object(text, lambda v: isinstance(v.get("verdicts"), list))
    if obj is None:
        return []
    out = []
    for v in obj["verdicts"]:
        if (
            isinstance(v, dict)
            and v.get("step")
            and v.get("verdict") in ("entails", "partly", "no")
        ):
            out.append(
                {
                    k: str(v.get(k) or "")
                    for k in ("step", "clause", "verdict", "unsupported", "why")
                }
            )
    return out


def review_reasons(verdicts: list[dict[str, str]]) -> list[str]:
    """The send-back lines for the reviewer's "no" verdicts (D14)."""
    return [
        f'step {v["step"]}: the reviewer finds "{v["clause"][:160]}" not supported by its quote'
        + (f" ({v['why'][:200]})" if v.get("why") else "")
        + (f"; unsupported: {v['unsupported'][:120]}" if v.get("unsupported") else "")
        + ". Quote the passage that says it, rewrite the clause to what the passage says, or "
        'give it a basis entry with source "none" and make it a question for the customer'
        for v in verdicts
        if v["verdict"] == "no"
    ]


# ── D15: what the session read in full, against what it cites ──
def read_in_full(domain: str, messages: list[dict[str, Any]]) -> list[str]:
    """Documents the session printed with the shell (cat, head, sed …), in the order it read them."""
    from tau2_loop.workflows.rag_agent import resolve_doc

    out: list[str] = []
    for m in messages:
        for tc in m.get("tool_calls") or [] if m.get("role") == "assistant" else []:
            fn = tc.get("function") or {}
            if fn.get("name") != "shell":
                continue
            try:
                cmd = str(json.loads(fn.get("arguments") or "{}").get("command", ""))
            except (json.JSONDecodeError, AttributeError):
                continue
            if not PRINTS.search(cmd):
                continue
            for ref in DOC_ID.findall(cmd):
                i = resolve_doc(domain, ref.removesuffix(".md"))
                if i and i not in out:
                    out.append(i)
    return out


def set_aside(domain: str, wf: dict[str, Any] | None) -> dict[str, str]:
    """The documents a reply sets aside, {doc id: why}."""
    from tau2_loop.workflows.rag_agent import resolve_doc

    out: dict[str, str] = {}
    for s in (wf or {}).get("set_aside") or []:
        if isinstance(s, dict) and s.get("doc"):
            i = resolve_doc(domain, str(s["doc"]).removesuffix(".md"))
            out[i or str(s["doc"])] = str(s.get("why") or "")
    return out


def job_cited(domain: str, wf: dict[str, Any] | None) -> set[str]:
    """Documents a quote in one of the reply's jobs cites (its `documents` list does not count)."""
    from tau2_loop.workflows.rag_agent import quotes, resolve_doc

    out: set[str] = set()
    for q in quotes({"jobs": (wf or {}).get("jobs") or []}):
        i = resolve_doc(domain, q["doc"]) if q["doc"] else None
        if i:
            out.add(i)
    return out


def unaccounted(
    domain: str, messages: list[dict[str, Any]], wf: dict[str, Any] | None
) -> list[str]:
    """D15: documents printed in full that no job cites and the reply does not set aside."""
    done = job_cited(domain, wf) | set(set_aside(domain, wf))
    return [d for d in read_in_full(domain, messages) if d not in done]


def read_list(domain: str, docs: list[str]) -> str:
    """The merge turn's list of documents read and not cited, each with its title."""
    lines = []
    for d in docs:
        doc = kb.document(domain, d)
        lines.append(f"- {d}: {doc['title'] if doc else ''}")
    return "\n".join(lines)
