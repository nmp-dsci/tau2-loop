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

D13: a job that calls or rules out a tool whose own description says its rules are in the knowledge
base ("The proper transfer reason enum can be found in the knowledge base") cites at least one of
the documents that name that tool. r14 and r15 kept a referral workflow's ban on every transfer
without reading doc 042's reason codes, though the transfer tool's description points there (s22
D7, r16).

M6 (s23, r18): a tool's description also lists the error replies it can return, parsed from
tau2's source (`error_replies`). r17's credit limit workflow sent a resubmission to a human because
nothing it read said that the same request always returns "A similar request may already exist."

D16 (s23, r18): a job that calls a tool whose description names an idea that is a document's
title ("Regulation E reporting timing rules" → "Understanding Regulation E…") cites that document
or sets it aside (`named_titles`). Doc 037's liability tiers were read by 17 of the 18 sessions
that wrote the debit dispute workflow and cited by none.

Read only: vendor/tau2-bench is never edited. A domain other than banking_knowledge, or the demo
image with no tau2, has no spec and both checks pass.
"""

from __future__ import annotations

import ast
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
POINTER = re.compile(
    r"(?:found|described|listed|defined|documented) in the knowledge base"
    r"|search (?:it|the knowledge base)",
    re.I,
)


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
def describe(domain: str, name: str, errors: bool = False) -> str | None:
    """A tool's whole description as tau2 writes it, the text the answering agent reads when it
    unlocks the tool: what it does and what each parameter and value means (s22, r13). None for a
    name tau2 does not have. `errors` (r18) adds the error replies the tool can return."""
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
            text = f"{name.strip()}({params}), run by {who}:\n{inspect.getdoc(fn) or ''}"
            replies = error_replies(domain).get(name.strip(), ()) if errors else ()
            if replies:
                text += "\nReplies it can return besides its result:\n" + "\n".join(
                    f"- {r}" for r in replies[:ERROR_LINES]
                )
            return text
    return None


ERROR_LINES = 12


def _rendered(node: ast.expr | None) -> str | None:
    """An "Error…" string a return gives: a literal, an f-string with its values shown by name,
    or the first item of a returned tuple."""
    if isinstance(node, ast.Tuple) and node.elts:
        node = node.elts[0]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(
            str(v.value) if isinstance(v, ast.Constant) else "{" + ast.unparse(v.value) + "}"
            for v in node.values
            if isinstance(v, (ast.Constant, ast.FormattedValue))
        )
    else:
        return None
    return text if text.startswith("Error") else None


def _returns(fn: ast.FunctionDef) -> list[str]:
    out = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and (t := _rendered(n.value)) and t not in out:
            out.append(t)
    return out


@cache
def error_replies(domain: str) -> dict[str, tuple[str, ...]]:
    """M6 (s23, r18): each tool's error replies, the `return "Error: …"` strings in its body and in
    the helpers it calls by name (one level), parsed from tau2's source, never run. These are the
    replies the answering agent gets; they are the environment's behaviour, not gold."""
    if domain != "banking_knowledge":
        return {}
    try:
        from tau2.domains.banking_knowledge import tools as t2

        tree = ast.parse(inspect.getsource(t2))
    except (ImportError, OSError, TypeError, SyntaxError):
        return {}
    funcs: dict[str, ast.FunctionDef] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef):
            funcs.setdefault(n.name, n)
    out: dict[str, tuple[str, ...]] = {}
    for name in tools(domain):
        fn = funcs.get(name)
        if fn is None:
            continue
        found = _returns(fn)
        for c in ast.walk(fn):
            if not isinstance(c, ast.Call):
                continue
            callee = (
                c.func.attr
                if isinstance(c.func, ast.Attribute) and isinstance(c.func.value, ast.Name)
                else c.func.id
                if isinstance(c.func, ast.Name)
                else None
            )
            helper = (
                funcs.get(callee)
                if callee and callee != name and callee not in tools(domain)
                else None
            )
            for t in _returns(helper) if helper else []:
                if t not in found:
                    found.append(t)
        if found:
            out[name] = tuple(found)
    return out


def named(domain: str, job: dict[str, Any]) -> set[str]:
    """Every tool tau2 has that a job names anywhere as a word: a step's call, its arguments, its
    words, and its `tools` list, the tools it rules out included (s22, r17)."""
    text = json.dumps(job, ensure_ascii=False)
    return {n for n in tools(domain) if re.search(rf"\b{re.escape(n)}\b", text)}


def descriptions(
    domain: str, jobs: list[dict[str, Any]], every_named: bool = False, errors: bool = False
) -> str:
    """Every tool the jobs call, each with its whole description (r13's merge and decide turns).
    `every_named` (r17) adds every tool they name, so a tool a job rules out, such as a customer's
    own tool, is described too: r16's merge never showed `submit_referral` once a job listed it as
    not used, because only step calls and numbered names were collected."""
    from tau2_loop.workflows import rubric

    dicts = [j for j in jobs if isinstance(j, dict)]
    names = {t for j in dicts for t in rubric.tools_named(j)}
    if every_named:
        names |= {t for j in dicts for t in named(domain, j)}
    found = [(n, describe(domain, n, errors)) for n in sorted(names)]
    return "\n\n".join(f"### {n}\n{d}" for n, d in found if d)


@cache
def pointer_tools(domain: str) -> dict[str, str]:
    """Each tool whose own description says its rules are in the knowledge base, with those words
    (D13). The unlock tool's pointer is to a tool name, not to rules, and calls nothing."""
    out: dict[str, str] = {}
    for name in tools(domain):
        m = POINTER.search(describe(domain, name) or "")
        if m and name not in NOT_A_CALL:
            out[name] = m.group(0)
    return out


@cache
def naming_docs(domain: str, tool: str) -> tuple[str, ...]:
    """The documents that name a tool: where its description's pointer leads."""
    if not kb.available(domain):
        return ()
    return tuple(
        sorted(p.stem for p in kb.source_dir(domain).glob("*.json") if tool in p.read_text())
    )


def _norm_doc(doc: str) -> str:
    return re.sub(r"[()]", "_", doc.strip()).removesuffix(".md")


def _cited(job: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(job, dict):
        for k, v in job.items():
            if k == "doc" and isinstance(v, str):
                out.add(_norm_doc(v))
            else:
                out |= _cited(v)
    elif isinstance(job, list):
        for v in job:
            out |= _cited(v)
    return out


def unread_pointers(domain: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    """D13: each pointer tool the job calls or rules out while citing none of the documents that
    name it, with the documents to read. s30: the changelog is not the job: code writes every
    tool a merge left out into it, and that named a removed tool as one the job rules out."""
    text = json.dumps({k: v for k, v in job.items() if k != "changelog"}, ensure_ascii=False)
    cited = _cited(job)
    out = []
    for tool, words in pointer_tools(domain).items():
        if tool not in text:
            continue
        docs = naming_docs(domain, tool)
        if docs and not cited & {_norm_doc(d) for d in docs}:
            out.append({"tool": tool, "pointer": words, "docs": list(docs)})
    return out


TITLE_IDEA = re.compile(
    r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?: (?:[A-Z][a-z]+|[A-Z]{1,}|of|and))*(?: [A-Z])?\b"
)
NOT_AN_IDEA = ("use ", "the ", "must ", "this ", "if ", "for ")


def named_titles(
    domain: str, job: dict[str, Any], aside: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    """D16 (s23, r18): each idea a tool the job calls names in its description that is the title of
    one to three documents, when the job cites none of them and the reply sets none aside."""
    from tau2_loop.workflows import rubric

    index = kb.read_index(domain)
    titles = {d: str(v.get("title") or "") for d, v in index.items()}
    have = _cited(job) | {_norm_doc(a) for a in aside}
    out, seen = [], set()
    for tool in sorted(rubric.tools_named(job)):
        desc = describe(domain, tool) or ""
        for idea in sorted(set(TITLE_IDEA.findall(desc))):
            if len(idea.split()) < 2 or idea.lower().startswith(NOT_AN_IDEA):
                continue
            docs = sorted(d for d, t in titles.items() if idea in t)
            if not 0 < len(docs) <= 3 or have & {_norm_doc(d) for d in docs}:
                continue
            if (idea, tuple(docs)) not in seen:
                seen.add((idea, tuple(docs)))
                out.append({"tool": tool, "idea": idea, "docs": docs})
    return out


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
    made = {rules[t]["effect"] for _, t in calls if t in rules and rules[t]["effect"]}  # a PIN change makes none
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
    if "D16" in which:
        for m in c.get("D16") or []:
            why.append(
                f"this job calls {m['tool']}, whose own description names '{m['idea']}', the title "
                f"of {', '.join(m['docs'])}, and cites none of them: read it and quote the passage "
                "that decides this job's case, or list it in `set_aside` with why it does not apply"
            )
    if "D13" in which:
        for m in c.get("D13") or []:
            why.append(
                f"this job calls or rules out {m['tool']}, whose own description says its rules "
                f"are '{m['pointer']}', but it cites none of the documents that name it: "
                f"{', '.join(m['docs'])}. Read them and quote the passage that decides this job's "
                "case, then keep, rewrite or remove each step that calls or rules out the tool"
            )
    return why
