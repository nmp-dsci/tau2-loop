"""What the tools need, read from tau2's code and the documents, for rubric D9 and D10 (s22, r5).

D9: every step that makes a write names every required parameter of its tool. The parameters
come from tau2's own signatures and docstrings (the text the answering agent sees when it unlocks
a tool), so a send-back can say what each one is. Debit disputes left out
`customer_max_liability_amount` and the agent filled it in wrongly (s22 D2).

D10: a write a document lets only some statuses take (close: ACTIVE or PENDING) cannot be made on
a card the job can meet in another status (FROZEN) unless the job also calls the tool the
documents give for that status, the one that leaves it in an allowed one (unfreeze: FROZEN, then
ACTIVE). The requirements and the effects are parsed from the documents' own requirement and step
sections, never written here per tool. r3's closing workflow quoted "must currently be in ACTIVE
or PENDING status" twice, then told the agent not to unfreeze, so every close of a frozen card
failed (s22 D1).

Read only: vendor/tau2-bench is never edited. A domain other than banking_knowledge, or the demo
image with no tau2, has no spec and both checks pass.
"""

from __future__ import annotations

import inspect
import json
import re
from functools import cache
from typing import Any

from tau2_loop.data import documents as kb

STATUS = r"(ACTIVE|PENDING|FROZEN|CLOSED|OPEN|SUSPENDED)"
TOOL = re.compile(r"\b([a-z][a-z_]*_\d{4})\b")
REQUIRES = re.compile(
    r"must (?:currently )?be (?:in |an? )?("
    + STATUS
    + r"(?:(?:,| or|/| and)\s*"
    + STATUS
    + r")*)\s+status"
)
EFFECT_VERB = {"freeze": "FROZEN", "unfreeze": "ACTIVE", "activate": "ACTIVE", "close": "CLOSED"}
NOT_A_CALL = ("unlock_discoverable_agent_tool",)


@cache
def tools(domain: str) -> dict[str, dict[str, Any]]:
    """Each tool: its required parameters with their descriptions, and whether it writes."""
    if domain != "banking_knowledge":
        return {}
    try:
        from tau2.domains.banking_knowledge import tools as t2
    except ImportError:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for cls, user in ((t2.KnowledgeTools, False), (t2.KnowledgeUserTools, True)):
        for name, fn in inspect.getmembers(cls, inspect.isfunction):
            if not getattr(fn, "__tool__", False):
                continue
            sig = inspect.signature(fn).parameters
            required = [
                p for p, v in sig.items() if p != "self" and v.default is inspect.Parameter.empty
            ]
            try:
                params = t2.parse_discoverable_tool_docstring(fn).get("parameters") or {}
            except Exception:  # noqa: BLE001 - a docstring tau2 cannot parse still has a signature
                params = {}
            kind = getattr(fn, "__tool_type__", None)
            out[name] = {
                "required": required,
                "about": {p: str((params.get(p) or {}).get("description") or "") for p in required},
                "write": getattr(kind, "value", "") == "write",
                "user": user,
            }
    return out


@cache
def describe(domain: str, name: str) -> str | None:
    """A tool's whole description as tau2 writes it, the text the answering agent reads when it
    unlocks the tool: what it does and what each parameter and value means (s22, r13). None for a
    name tau2 does not have."""
    if domain != "banking_knowledge" or not isinstance(name, str):
        return None
    try:
        from tau2.domains.banking_knowledge import tools as t2
    except ImportError:
        return None
    for cls, who in ((t2.KnowledgeTools, "the agent"), (t2.KnowledgeUserTools, "the customer")):
        fn = getattr(cls, name.strip(), None)
        if fn is not None and getattr(fn, "__tool__", False):
            params = ", ".join(p for p in inspect.signature(fn).parameters if p != "self")
            return f"{name.strip()}({params}), run by {who}:\n{inspect.getdoc(fn) or ''}"
    return None


def descriptions(domain: str, jobs: list[dict[str, Any]]) -> str:
    """Every tool the jobs call, each with its whole description (r13's merge and decide turns)."""
    from tau2_loop.workflows import rubric

    names = sorted({t for j in jobs if isinstance(j, dict) for t in rubric.tools_named(j)})
    found = [(n, describe(domain, n)) for n in names]
    return "\n\n".join(f"### {n}\n{d}" for n, d in found if d)


@cache
def status_rules(domain: str) -> dict[str, dict[str, Any]]:
    """Each tool a document's requirement section governs: the statuses it allows, the status it
    leaves (from "is now active" in the steps, else its verb), and the document."""
    if not kb.available(domain):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(kb.source_dir(domain).glob("*.json")):
        content = str(json.loads(path.read_text()).get("content") or "")
        secs = re.split(r"\n(?=#{2,3} )", content)
        for i, sec in enumerate(secs):
            head = sec.split("\n", 1)[0].lower()
            if "requirement" not in head and "eligib" not in head:
                continue
            allowed: set[str] = set()
            for m in REQUIRES.finditer(sec):
                allowed |= set(re.findall(STATUS, m.group(1)))
            if not allowed:
                continue
            named: set[str] = set()
            effect = None
            for nxt in secs[i + 1 : i + 3]:  # the steps section that follows names the tool
                if "step" in nxt.split("\n", 1)[0].lower():
                    named |= set(TOOL.findall(nxt))
                    now = re.search(r"\bis now (active|frozen|closed|pending)\b", nxt, re.I)
                    effect = effect or (now.group(1).upper() if now else None)
            for t in named:
                r = out.setdefault(t, {"allowed": set(), "effect": None, "doc": path.stem})
                r["allowed"] |= allowed
                r["effect"] = r["effect"] or effect or EFFECT_VERB.get(t.split("_")[0])
    return out


def step_tool(job: dict[str, Any], k: int) -> str | None:
    from tau2_loop.workflows.rubric import step_tool as _st

    return _st(job, k)


def _given(args: Any) -> set[str]:
    """The parameters a step's call names: its own keys and those of an `arguments` object."""
    if not isinstance(args, dict):
        return set()
    keys = {k for k in args if k not in ("agent_tool_name", "arguments")}
    inner = args.get("arguments")
    if isinstance(inner, dict):
        keys |= set(inner)
    elif isinstance(inner, str):
        keys |= set(re.findall(r'"(\w+)"\s*:', inner))
    return keys


def missing_params(domain: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    """D9: each write step whose call leaves out a required parameter of its tool."""
    spec = tools(domain)
    steps = [s for s in job.get("steps") or [] if isinstance(s, dict)]
    out = []
    for k, s in enumerate(steps):
        t = step_tool(job, k)
        if s.get("call") in NOT_A_CALL or t not in spec:
            continue
        info = spec[t]
        if not info["write"] or info["user"]:
            continue
        miss = [p for p in info["required"] if p not in _given(s.get("args"))]
        if miss:
            out.append(
                {
                    "step": s.get("id"),
                    "tool": t,
                    "missing": miss,
                    "about": {p: info["about"].get(p, "") for p in miss},
                }
            )
    return out


def unmet_status(domain: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    """D10: each write step that needs a status the job may not have, when the documents give a
    tool that leaves the card in a status it allows and the job never calls it, nor quotes a
    document forbidding it."""
    rules = status_rules(domain)
    steps = [s for s in job.get("steps") or [] if isinstance(s, dict)]
    calls = [
        (k, step_tool(job, k)) for k in range(len(steps)) if steps[k].get("call") not in NOT_A_CALL
    ]
    called = {t for _, t in calls}
    made = {rules[t]["effect"] for _, t in calls if t in rules}
    acts_on: dict[str, set[str]] = {}
    for name, rule in rules.items():
        if len(rule["allowed"]) == 1:
            acts_on.setdefault(next(iter(rule["allowed"])), set()).add(name)
    quoted = [
        (str(x.get("doc") or ""), str(x.get("quote") or "").lower())
        for x in [*steps, *(job.get("rules") or [])]
        if isinstance(x, dict)
    ]
    out = []
    for k, t in calls:
        if t is None or t not in rules:
            continue
        r, s = rules[t], steps[k]
        own = set(re.findall(r"\b" + STATUS + r"\b", f"{s.get('do', '')} {s.get('if', '')}"))
        for st in sorted((own | made) - r["allowed"] - {"CLOSED", r["effect"]}):
            remedy = sorted(
                m for m in acts_on.get(st, set()) if rules[m]["effect"] in r["allowed"] and m != t
            )
            if not remedy or set(remedy) & called:
                continue
            verb = remedy[0].split("_")[0][:6]
            own_docs = {r["doc"], rules[remedy[0]]["doc"]}
            if any(
                d in own_docs and re.search(r"\b(not|never|cannot)\b[^.]{0,60}" + verb, q)
                for d, q in quoted
            ):
                continue  # the write's or the remedy's own procedure forbids it, as quoted
            out.append(
                {
                    "step": s.get("id"),
                    "tool": t,
                    "needs": sorted(r["allowed"]),
                    "meets": st,
                    "remedy": remedy,
                    "doc": r["doc"],
                    "remedy_doc": rules[remedy[0]]["doc"],
                    "leaves": rules[remedy[0]]["effect"],
                }
            )
    return out


def reasons(c: dict[str, Any], which: tuple[str, ...]) -> list[str]:
    """The send-back lines for the checks a version gates on (`rubric:` in its yaml)."""
    why = []
    if "D9" in which:
        for m in c.get("D9") or []:
            about = "; ".join(f"{p}: {d}" if d else p for p, d in m["about"].items())
            why.append(
                f"step {m['step']} calls {m['tool']} without {', '.join(m['missing'])}: give each "
                f"and where its value comes from, from the documents ({about})"
            )
    if "D10" in which:
        for m in c.get("D10") or []:
            why.append(
                f"step {m['step']} calls {m['tool']}, which {m['doc']} allows only for a card in "
                f"{' or '.join(m['needs'])} status, but this job can meet a {m['meets']} card and never "
                f"calls {' or '.join(m['remedy'])}, which {m['remedy_doc']} gives for a {m['meets']} card "
                f"and leaves it {m['leaves']}: write the step that meets the requirement before "
                f"{m['step']}, or quote the document that forbids it"
            )
    return why
