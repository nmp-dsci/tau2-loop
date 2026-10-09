"""workflow_rag's rubric (s20): what a workflow is judged on, checked without a model.

The rubric is written into r2's prompt (`rag_agents/<domain>/r2/rag_agent.md`), so the writer
knows it; this module measures the checks code can make. On r1's library against v7's train
outcomes, D2 (every tool the cited documents name is in the workflow) separated passing from
failing conversations best (AUC 0.69, s20); the schema checks every workflow already meets
carry no signal.

  D1 grounded         every quote is found in the document it cites
  D2 tools            every tool the cited documents name appears in the workflow
  D3 conditions       every step that changes the database says when it runs
  D4 lookups          every check names the read that answers it
  D5 starts           the job says who starts it: the customer asks, or the agent may offer
  D6 schema           each customer fact has its question; done_when is set
  D8 nothing lost     a merge keeps every step, tool and quote of the version it merged, or its
                      changelog names the change (compared with that version, never with gold)
  D9 parameters       every write step names every required parameter of its tool (s22)
  D10 statuses        a write a document allows only in some statuses is not made on a card the
                      job can meet in another, unless it calls the documents' tool for it (s22)
  D12 re-read         a merge accounts for every step and rule of the versions it merged: kept
                      (named in its `kept`, its quote saying it as written), rewritten, or removed;
                      none stays only because the merge did not look at it (s22, r7)

D9, D10 and D12 are reported for every version and gated only by one that names them (r5's
`rubric: [D10]`, r7's `[D9, D10, D12]`). D8 and D1's "no new quote that is not found" are the merge gate (`gate()`): a merged version
that fails them after one send-back is not taken, and the library keeps the previous one.
Gold is never read here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from tau2_loop.data import documents as kb

# a discoverable tool's name ends in four digits; three user tools carry none
TOOL = re.compile(
    r"\b[a-z]+(?:_[a-z0-9]+)*_\d{4}\b|\b(?:get_card_last_4_digits|apply_for_credit_card|get_referral_link)\b"
)
READS = ("get_", "list_", "check_", "verify_", "find_", "search_", "calculate_", "KB_")
NOT_WRITES = {
    "log_verification",
    "unlock_discoverable_agent_tool",
    "call_discoverable_user_tool",
    "get_current_time",
    "transfer_to_human_agents",
}
CHECK_WORDS = re.compile(r"\b(check|verify|confirm|eligib|ensure)\w*", re.I)
# what a check reads, by the word that names it, and a fragment of the tool that reads it
CHECK_NOUNS = {
    "dispute": "dispute",
    "replacement": "replacement",
    "balance": "account",
    "transaction": "transaction",
    "closure": "closure",
    "payment history": "payment",
}
QUOTE_KEY = 60


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("‑", "-")).strip().lower()


def step_tool(job: dict[str, Any], k: int) -> str | None:
    """The tool step `k` runs, a discoverable one by its own name; a gift as `give <tool>`."""
    steps = job.get("steps") or []
    st = steps[k]
    args = st.get("args") or {}
    call = st.get("call")
    if call == "give_discoverable_user_tool":
        t = args.get("discoverable_tool_name")
        if not t:  # the tool is often named only in the next steps or the words
            for nx in steps[k : k + 3]:
                t = (nx.get("args") or {}).get("discoverable_tool_name") or next(
                    iter(TOOL.findall(str(nx.get("do", "")))), None
                )
                if t:
                    break
        return f"give {t}" if t else "give"
    t = args.get("agent_tool_name") or call
    return str(t) if t else None


def is_write(tool: str | None) -> bool:
    if not tool or tool in NOT_WRITES:
        return False
    return tool.startswith("give") or not tool.startswith(READS)


def tools_named(job: dict[str, Any]) -> set[str]:
    """Every tool a workflow names anywhere: a step's call, its arguments, its words, `tools`."""
    text = json.dumps(job, ensure_ascii=False)
    names = set(TOOL.findall(text))
    for st in job.get("steps") or []:
        if isinstance(st.get("call"), str) and st["call"]:
            names.add(st["call"])
    return names


def cited_tools(domain: str, job: dict[str, Any]) -> set[str]:
    """The tools the documents a workflow cites name."""
    from tau2_loop.workflows.rag_agent import cited_docs

    out: set[str] = set()
    for d in cited_docs(domain, {"jobs": [job]}):
        doc = kb.document(domain, d)
        if doc:
            out |= set(TOOL.findall(doc["content"]))
    return out


def _quote_rows(domain: str, job: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    from tau2_loop.workflows.rag_agent import quote_check

    rows: list[dict[str, Any]] = quote_check(domain, {"jobs": [job]}, policy)["rows"]
    return rows


def check(
    domain: str,
    job: dict[str, Any],
    previous: list[dict[str, Any]] | None = None,
    policy: str = "",
    kept: Any = None,
) -> dict[str, Any]:
    """Every check code can make on one job; `previous` are the versions it merged, if any, and
    `kept` what the merge says it re-read and kept as it was (r7)."""
    steps = [s for s in job.get("steps") or [] if isinstance(s, dict)]
    q = _quote_rows(domain, job, policy)
    writes = [k for k in range(len(steps)) if is_write(step_tool(job, k))]
    conditioned = [k for k in writes if steps[k].get("if") or steps[k].get("by") == "customer"]
    calls = " ".join(f"{s.get('call')} {json.dumps(s.get('args') or {})}" for s in steps)
    checks, named = [], 0
    for s in steps:
        if s.get("call") or not CHECK_WORDS.search(str(s.get("do", ""))):
            continue
        frags = [t for w, t in CHECK_NOUNS.items() if w in str(s.get("do", "")).lower()]
        if not frags:
            continue
        checks.append(s.get("id"))
        named += bool(s.get("lookup")) or all(f in calls for f in frags)
    want = cited_tools(domain, job)
    have = tools_named(job)
    infos = [
        i for i in job.get("info") or [] if isinstance(i, dict) and i.get("from") == "customer"
    ]
    starts = job.get("starts")
    out: dict[str, Any] = {
        "D1": {"found": sum(r["found"] for r in q), "n": len(q)},
        "D2": {"named": sorted(want & have), "missing": sorted(want - have), "n": len(want)},
        "D3": {"with_condition": len(conditioned), "n": len(writes)},
        "D4": {"named": named, "n": len(checks)},
        "D5": {"set": bool(starts)},
        "D6": {
            "asks": sum(bool(i.get("ask")) for i in infos),
            "n": len(infos),
            "done_when": bool(job.get("done_when")),
        },
    }
    from tau2_loop.workflows import toolspec

    out["D9"] = toolspec.missing_params(domain, job)  # s22: reported always, gated by r5
    out["D10"] = toolspec.unmet_status(domain, job)
    if previous:
        out["D8"] = lost(job, previous)
        out["D12"] = unread(job, previous, kept)
        before = {
            _squash(r["quote"])[:QUOTE_KEY]
            for p in previous
            for r in _quote_rows(domain, p, policy)
        }
        out["D1"]["new_unfound"] = [
            r["quote"][:120]
            for r in q
            if not r["found"] and _squash(r["quote"])[:QUOTE_KEY] not in before
        ]
    else:
        out["D1"]["new_unfound"] = [r["quote"][:120] for r in q if not r["found"]]
    out["score"] = score(out)
    return out


def _share(a: int, n: int) -> float:
    return a / n if n else 1.0


def score(c: dict[str, Any]) -> dict[str, float]:
    """Each check as a share in [0, 1]: 1.0 when there was nothing to check."""
    return {
        "D1": _share(c["D1"]["found"], c["D1"]["n"]),
        "D2": _share(len(c["D2"]["named"]), c["D2"]["n"]),
        "D3": _share(c["D3"]["with_condition"], c["D3"]["n"]),
        "D4": _share(c["D4"]["named"], c["D4"]["n"]),
        "D5": 1.0 if c["D5"]["set"] else 0.0,
        "D6": _share(c["D6"]["asks"], c["D6"]["n"]) * (1.0 if c["D6"]["done_when"] else 0.5),
    }


def lost(job: dict[str, Any], previous: list[dict[str, Any]]) -> dict[str, list[str]]:
    """What the merged `job` dropped from the versions it merged and its changelog does not name."""
    log = " ".join(
        _squash(f"{e.get('what', '')} {e.get('why', '')}")
        for e in job.get("changelog") or []
        if isinstance(e, dict) and e.get("change") in ("removed", "changed")
    )
    now_ids = {s.get("id") for s in job.get("steps") or [] if isinstance(s, dict)}
    now_tools = tools_named(job)
    now_quotes = {_squash(q["quote"])[:QUOTE_KEY] for q in _quotes(job)}
    out: dict[str, list[str]] = {"steps": [], "tools": [], "quotes": []}
    for p in previous:
        gone_named = []  # steps the changelog removes by name take their tools and quotes along
        for s in p.get("steps") or []:
            sid = s.get("id") if isinstance(s, dict) else None
            if sid and sid not in now_ids:
                if _squash(sid) in log:
                    gone_named.append(s)
                else:
                    out["steps"].append(str(sid))
        covered = {"steps": gone_named}
        for t in sorted(tools_named(p) - now_tools - tools_named(covered)):
            if _squash(t) not in log:
                out["tools"].append(t)
        along = {_squash(q["quote"])[:QUOTE_KEY] for q in _quotes(covered)}
        for qt in _quotes(p):
            k = _squash(qt["quote"])[:QUOTE_KEY]
            if k and k not in now_quotes and k not in along and k[:40] not in log:
                out["quotes"].append(qt["quote"][:120])
    return {k: sorted(set(v)) for k, v in out.items()}


def _named(key: str, names: list[str]) -> bool:
    """`key` is one of `names`: exactly, or (a rule named by its start) by 12 or more characters."""
    k = _squash(key)
    return any(k == n or (len(n) >= 12 and k.startswith(n)) for n in names)


def unread(job: dict[str, Any], previous: list[dict[str, Any]], kept: Any) -> dict[str, list[str]]:
    """Steps and rules of the merged versions that are still in `job` exactly as they were, and that
    the merge does not name in `kept` (D12, r7). A step it rewrote or removed was looked at; one
    left untouched counts as re-read only when the merge says it kept it."""
    kept = kept if isinstance(kept, dict) else {}
    names = {
        lst: [_squash(x) for x in kept.get(lst) or [] if isinstance(x, str) and x.strip()]
        for lst in ("steps", "rules")
    }
    now_steps = {s.get("id"): s for s in job.get("steps") or [] if isinstance(s, dict)}
    now_rules = {
        _squash(r.get("rule", "")): r for r in job.get("rules") or [] if isinstance(r, dict)
    }
    out: dict[str, list[str]] = {"steps": [], "rules": []}
    for p in previous:
        for st in p.get("steps") or []:
            sid = st.get("id") if isinstance(st, dict) else None
            if sid and now_steps.get(sid) == st and not _named(str(sid), names["steps"]):
                out["steps"].append(str(sid))
        for r in p.get("rules") or []:
            text = str(r.get("rule", "")) if isinstance(r, dict) else ""
            if text and now_rules.get(_squash(text)) == r and not _named(text, names["rules"]):
                out["rules"].append(text)
    return {k: list(dict.fromkeys(v)) for k, v in out.items()}


def _quotes(o: Any) -> list[dict[str, str]]:
    from tau2_loop.workflows.rag_agent import quotes

    return quotes(o)


def gate(c: dict[str, Any], extra: tuple[str, ...] = ()) -> list[str]:
    """Why a merged version may not be taken: empty when it may. `extra` are the checks a version
    also gates on (r5: D10; r6: D9, `workflows/toolspec.py`; r7: D12)."""
    from tau2_loop.workflows import toolspec

    why = []
    for k, items in (c.get("D8") or {}).items():
        if items:
            why.append(f"lost {k} the changelog does not name: {', '.join(items[:6])}")
    if c["D1"].get("new_unfound"):
        why.append(
            "quotes not found in the documents they cite: " + " | ".join(c["D1"]["new_unfound"][:4])
        )
    if "D12" in extra:
        u = c.get("D12") or {}
        if u.get("steps") or u.get("rules"):
            parts = [f"steps {', '.join(u['steps'][:12])}"] if u.get("steps") else []
            parts += (
                [f"rules {' | '.join(r[:60] for r in u['rules'][:8])}"] if u.get("rules") else []
            )
            n = len(u.get("steps") or []) + len(u.get("rules") or [])
            why.append(
                f"{n} items of the version you merged were not re-read ({'; '.join(parts)}"
                f"{' …' if n > 20 else ''}): keep each only if its quote and document say it as "
                "written, and name it in `kept`; rewrite it with the passage it follows if the "
                "documents say otherwise; remove it, saying what you searched, if no document "
                "supports it"
            )
    return [*why, *toolspec.reasons(c, extra)]


def library_report(domain: str, rag_version: str) -> dict[str, Any]:
    """The rubric over a RAG version's whole library: per job and the mean of each check."""
    from tau2_loop.workflows import library

    rows = []
    for e in library.entries(domain, rag_version):
        c = check(domain, e["workflow"])
        rows.append(
            {
                "job": e["job"],
                "session": e["session"],
                **c["score"],
                "missing_tools": c["D2"]["missing"],
            }
        )
    keys = ("D1", "D2", "D3", "D4", "D5", "D6")
    mean = {k: round(sum(r[k] for r in rows) / len(rows), 3) if rows else 0.0 for k in keys}
    below = {k: sum(r[k] < 1 for r in rows) for k in keys}
    return {"rag": rag_version, "n": len(rows), "mean": mean, "below": below, "jobs": rows}
