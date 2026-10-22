"""Three ways to merge a draft into the library's job (s30, `.lavish/s30_workflow-merge-opt-plan.html`).

r18 settled a conflict between the draft and the library by keeping the library's step, r19 by
keeping the draft's; neither asked the documents. 11 of v27's 24 failures trace to a merge: 8 lost
a right rule in r19's merge, 3 kept a wrong one through every merge. Each option below is a merge
setting (`RagAgent.merge_mode`, `merge_account`), never a change to r19, and the merge bench
(`workflows/bench.py`) replays one job's merges under each:

  A account  (`merge_account`, turn mode) r19's merge, plus: every old step whose quotes are found
             is carried, named as covered by a step of the new job, or named as replaced by a
             passage. One still unaccounted after the send-back is carried over as it was.
  B audit    (`merge_mode: audit`) an auditor in its own context reads the library's version with
             the full documents it cites and marks each step valid, rewrite or invalid. The merge
             reads the audited version; a valid step leaves only when it is accounted for as in A.
  C ledger   (`merge_mode: ledger`) no research conversation: both versions are broken into
             claims, grouped by case and field; a verifier settles each claim and each slot with
             two values against the documents; a writer writes the job from the claims that
             survive, and code checks every survivor is in the job and no rejected claim is.

All three judge a clause as stated, derived, a question or unsupported (the derived standard), and
get rubric D19 and D20. No gold is read here.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from tau2_loop.data import documents as kb

Call = Callable[[list[dict[str, Any]], str], "str | None"]  # (messages, stage) -> the reply
CODE = "logged by code (s30 merge bench)"
STEP_MARKS = ("audit", "claims")  # what an option adds to a step for the merge; never stored


# ── shared: documents in full, the fields a merge reply carries beside its jobs ──


def _doc_ids(domain: str, jobs: list[dict[str, Any]]) -> list[str]:
    from tau2_loop.workflows.rag_agent import POLICY_REFS, quotes, resolve_doc

    out: list[str] = []
    for q in quotes({"jobs": jobs}):
        ref = str(q.get("doc") or "").strip()
        i = "policy" if ref.lower() in POLICY_REFS else resolve_doc(domain, ref) if ref else None
        if i and i not in out:
            out.append(i)
    return out


def documents_text(domain: str, jobs: list[dict[str, Any]], policy: str) -> str:
    """Every document the jobs cite, in full, in the order they are first cited."""
    parts = []
    for i in _doc_ids(domain, jobs):
        if i == "policy":
            parts.append(f"### policy: the answering agent's policy\n{policy}")
            continue
        d = kb.document(domain, i)
        if d:
            parts.append(f"### {i}: {d['title']}\n{d['content']}")
    return "\n\n".join(parts)


def tools_text(domain: str, jobs: list[dict[str, Any]]) -> str:
    from tau2_loop.workflows import toolspec

    return toolspec.descriptions(domain, jobs, every_named=True, errors=True)


def take_fields(job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """A merged job without what an option's reply carries beside it: `accounted` (A, B),
    `conflicts` (B), and the marks on its steps (`audit`, `claims`). Returns the job and them."""
    job = dict(job)
    extra: dict[str, Any] = {}
    for k in ("accounted", "conflicts"):
        if k in job:
            extra[k] = job.pop(k)
    marks: dict[str, dict[str, Any]] = {}
    steps = []
    for st in job.get("steps") or []:
        if isinstance(st, dict) and any(m in st for m in STEP_MARKS):
            marks[str(st.get("id"))] = {m: st[m] for m in STEP_MARKS if m in st}
            st = {k: v for k, v in st.items() if k not in STEP_MARKS}
        steps.append(st)
    if marks:
        job["steps"] = steps
        extra["marks"] = marks
    rules = []
    for r in job.get("rules") or []:
        if isinstance(r, dict) and "claims" in r:
            extra.setdefault("rule_claims", {})[str(r.get("rule", ""))[:80]] = r["claims"]
            r = {k: v for k, v in r.items() if k != "claims"}
        rules.append(r)
    if "rule_claims" in extra:
        job["rules"] = rules
    return job, extra


def join_opts(*opts: dict[str, Any] | None) -> dict[str, Any]:
    """s31: the option fields of several replies to one merge as one (a second reply written as
    edits on the first): their lists joined in order, their step marks merged."""
    out: dict[str, Any] = {}
    for o in opts:
        for k, v in (o or {}).items():
            if isinstance(v, list):
                if v:
                    out[k] = [*(out.get(k) or []), *v]
            elif isinstance(v, dict):
                out[k] = {**(out.get(k) or {}), **v}
            else:
                out[k] = v
    return out


def edit_accounts(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """s31 (O1): the account a `remove` edit of a step gives, read as `accounted` reads it: its
    `covered_by` step, or the `quote` and `doc` of the passage that decides the step's case."""
    out: list[dict[str, Any]] = []
    for e in entry.get("edits") or []:
        if not isinstance(e, dict) or e.get("op") != "remove" or e.get("list") != "steps":
            continue
        sid = str(e.get("key") or "")
        if not sid:
            continue
        if e.get("covered_by"):
            out.append({"step": sid, "covered_by": str(e["covered_by"])})
        elif e.get("quote"):
            out.append(
                {"step": sid, "replaced_by": {"quote": e["quote"], "doc": e.get("doc") or ""}}
            )
    return out


# ── A and B: an old step leaves only with an account ──


def _found(domain: str, quote: str, doc: str, policy: str) -> bool:
    from tau2_loop.workflows.rag_agent import quote_check

    rows = quote_check(domain, {"jobs": [{"steps": [{"quote": quote, "doc": doc}]}]}, policy)[
        "rows"
    ]
    return bool(rows) and all(r["found"] for r in rows)


def sourced_steps(domain: str, job: dict[str, Any], policy: str) -> dict[str, dict[str, Any]]:
    """A's protected steps: each step with at least one quote, every one found in its document."""
    from tau2_loop.workflows.rag_agent import quote_check

    out = {}
    for st in job.get("steps") or []:
        if not isinstance(st, dict) or not st.get("id"):
            continue
        rows = quote_check(domain, {"jobs": [{"steps": [st]}]}, policy)["rows"]
        if rows and all(r["found"] for r in rows):
            out[str(st["id"])] = st
    return out


def accounts(extra: dict[str, Any] | None) -> list[dict[str, Any]]:
    v = (extra or {}).get("accounted") or []
    return [a for a in v if isinstance(a, dict) and a.get("step")] if isinstance(v, list) else []


def unaccounted(
    domain: str,
    job: dict[str, Any],
    protected: dict[str, dict[str, Any]],
    accounted: list[dict[str, Any]],
    policy: str,
) -> list[str]:
    """The protected old steps the job neither keeps (by id) nor accounts for: covered by a step it
    has, or replaced by a passage found in its document."""
    ids = {str(s.get("id")) for s in job.get("steps") or [] if isinstance(s, dict)}
    ok = set(ids)
    for a in accounted:
        sid = str(a["step"])
        if str(a.get("covered_by") or "") in ids:
            ok.add(sid)
        rb = a.get("replaced_by")
        if (
            isinstance(rb, dict)
            and rb.get("quote")
            and _found(domain, str(rb["quote"]), str(rb.get("doc") or ""), policy)
        ):
            ok.add(sid)
    return [sid for sid in protected if sid not in ok]


def account_lines(missing: list[str], protected: dict[str, dict[str, Any]]) -> list[str]:
    return [
        f"step {sid} of the library's version stands on its documents and your job no longer has "
        f'it ("{str(protected[sid].get("do") or "")[:120]}"): carry it, or name it in `accounted` '
        'as {"step": "' + sid + '", "covered_by": "<the step of your job that handles its case>"} '
        'or {"step": "' + sid + '", "replaced_by": {"quote": "<the passage that decides its case '
        'otherwise>", "doc": "<doc id>"}}'
        for sid in missing
    ]


def carry(
    job: dict[str, Any], order: list[str], by_id: dict[str, dict[str, Any]], ids: list[str]
) -> list[dict[str, str]]:
    """Put each old step (`by_id`: as A read it, or as B's audit left it) back where it stood in
    the old `order`: after the nearest earlier old step the job still has, else at the start.
    Returns the changelog lines."""
    order = [*order, *(i for i in ids if i not in order)]
    log = []
    for sid in sorted(ids, key=order.index):
        steps = list(job.get("steps") or [])
        have = [str(s.get("id")) for s in steps if isinstance(s, dict)]
        before = [o for o in order[: order.index(sid)] if o in have]
        at = have.index(before[-1]) + 1 if before else 0
        steps.insert(at, by_id[sid])
        job["steps"] = steps
        log.append(
            {
                "change": "added",
                "what": f"step {sid}",
                "why": f"carried over as it was: it stands on its documents and the merge left "
                f"it out without naming what covers or replaces it; {CODE}",
            }
        )
    return log


# ── B: the audit ──


def _numbered_rules(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": f"R{k}", **r} for k, r in enumerate(job.get("rules") or [], 1) if isinstance(r, dict)
    ]


def audit_request(domain: str, prompt: str, job: dict[str, Any], policy: str) -> str:
    body = {"steps": job.get("steps") or [], "rules": _numbered_rules(job)}
    return prompt.format(
        job=job.get("job", ""),
        version=json.dumps(body, ensure_ascii=False, indent=1),
        documents=documents_text(domain, [job], policy) or "(none cited)",
        tools=tools_text(domain, [job]) or "(none)",
    )


def parse_audit(text: str | None) -> dict[str, dict[str, Any]] | None:
    """{"steps": {id: verdict}, "rules": {"R<n>": verdict}}; None when the reply has neither."""
    from tau2_loop.workflows.rag_agent import _json_object

    obj = _json_object(text, lambda v: isinstance(v.get("steps"), list))
    if obj is None:
        return None
    out: dict[str, dict[str, Any]] = {"steps": {}, "rules": {}}
    for kind, key in (("steps", "step"), ("rules", "rule")):
        for v in obj.get(kind) or []:
            if (
                isinstance(v, dict)
                and v.get(key)
                and v.get("verdict")
                in (
                    "valid",
                    "rewrite",
                    "invalid",
                )
            ):
                out[kind][str(v[key])] = v
    return out


def audited(job: dict[str, Any], verdicts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The library's version as the audit leaves it: invalid steps and rules removed, rewritten
    ones replaced (same id), each kept step marked for the merge. Returns {shown, protected,
    removed, unaudited}: `protected` are the valid and rewritten steps as the merge reads them."""
    shown = {k: v for k, v in job.items() if k not in ("steps", "rules")}
    steps, protected, removed, unaudited = [], {}, [], []
    for st in job.get("steps") or []:
        if not isinstance(st, dict):
            continue
        sid = str(st.get("id"))
        v = verdicts["steps"].get(sid)
        if v is None:
            steps.append(st)
            unaudited.append(sid)
            continue
        if v["verdict"] == "invalid":
            removed.append({"step": sid, "why": str(v.get("why") or "")})
            continue
        new = v.get("rewrite") if v["verdict"] == "rewrite" else None
        new = {**new, "id": sid} if isinstance(new, dict) and new.get("do") else st
        protected[sid] = new
        mark = "valid" if new is st else "rewritten"
        steps.append({**new, "audit": mark})
    rules = []
    for r in _numbered_rules(job):
        rid = r.pop("id")
        v = verdicts["rules"].get(rid)
        if v is not None and v["verdict"] == "invalid":
            removed.append({"rule": str(r.get("rule", ""))[:160], "why": str(v.get("why") or "")})
            continue
        new = v.get("rewrite") if v is not None and v["verdict"] == "rewrite" else None
        rules.append(new if isinstance(new, dict) and new.get("rule") else r)
    shown["steps"], shown["rules"] = steps, rules
    return {"shown": shown, "protected": protected, "removed": removed, "unaudited": unaudited}


def removed_text(audits: dict[str, dict[str, Any]]) -> str:
    lines = []
    for job, a in audits.items():
        for x in a["removed"]:
            what = f"step {x['step']}" if "step" in x else f'rule "{x["rule"]}"'
            lines.append(f"- {job}: {what}: {x['why'][:240]}")
    if not lines:
        return ""
    return (
        "\n\n## What the audit removed from the library's version, with why\n\n"
        + "\n".join(lines)
        + "\n\nThese stay out unless your research found a passage that supports one: then write "
        "it with that quote."
    )


def audit_heads(
    domain: str,
    prompt: str,
    entries: list[dict[str, Any]],
    policy: str,
    call: Call,
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, dict[str, Any]]:
    """B's first call for each library job the merge holds, in its own context."""
    out = {}
    for e in entries:
        wf = e["workflow"]
        reply = call(
            [{"role": "user", "content": audit_request(domain, prompt, wf, policy)}], "audit"
        )
        v = parse_audit(reply)
        if v is None:
            emit(
                {
                    "kind": "note",
                    "text": f"The audit of {e['job']} was unreadable; it is shown as it was.",
                }
            )
            continue
        a = audited(wf, v)
        out[e["job"]] = {**a, "verdicts": v}
        emit(
            {
                "kind": "audited",
                "job": e["job"],
                "valid": sum(x["verdict"] == "valid" for x in v["steps"].values()),
                "rewrite": sum(x["verdict"] == "rewrite" for x in v["steps"].values()),
                "invalid": sum(x["verdict"] == "invalid" for x in v["steps"].values()),
                "unaudited": a["unaudited"],
            }
        )
    return out


# ── C: the claim ledger ──


def label_versions(
    old: list[dict[str, Any]], draft: dict[str, Any] | None
) -> tuple[str, dict[str, str]]:
    """Both sides with every step and rule given an origin id: `L:<id>` (`L2:` for a second
    library job), `L:R<n>`, `D:<id>`, `D:R<n>`. Returns the text and {origin: kind}."""
    origins: dict[str, str] = {}
    parts = []
    sides = [(("L" if len(old) == 1 else f"L{k}"), j) for k, j in enumerate(old, 1)]
    if draft is not None:
        sides.append(("D", draft))
    for tag, j in sides:
        steps = []
        for st in j.get("steps") or []:
            if isinstance(st, dict) and st.get("id"):
                oid = f"{tag}:{st['id']}"
                origins[oid] = "step"
                steps.append({**st, "id": oid})
        rules = []
        for k, r in enumerate(j.get("rules") or [], 1):
            if isinstance(r, dict):
                oid = f"{tag}:R{k}"
                origins[oid] = "rule"
                rules.append({"id": oid, **r})
        who = "the library's version" if tag.startswith("L") else "the draft"
        body = {k: v for k, v in j.items() if k not in ("steps", "rules", "changelog")}
        parts.append(
            f"## {who} ({tag}): {j.get('job', '')}\n"
            + json.dumps({**body, "steps": steps, "rules": rules}, ensure_ascii=False, indent=1)
        )
    return "\n\n".join(parts), origins


def parse_claims(text: str | None) -> list[dict[str, Any]] | None:
    from tau2_loop.workflows.rag_agent import _json_object

    obj = _json_object(text, lambda v: isinstance(v.get("claims"), list))
    if obj is None:
        return None
    out = []
    for c in obj["claims"]:
        if isinstance(c, dict) and c.get("id") and c.get("origin"):
            o = c["origin"]
            c = {**c, "id": str(c["id"]), "origin": [o] if isinstance(o, str) else list(o)}
            out.append(c)
    return out


def missing_origins(claims: list[dict[str, Any]], origins: dict[str, str]) -> list[str]:
    seen = {str(o) for c in claims for o in c["origin"]}
    return [o for o in origins if o not in seen]


def mark_quotes(domain: str, claims: list[dict[str, Any]], policy: str) -> None:
    for c in claims:
        if c.get("quote"):
            c["quote_found"] = _found(domain, str(c["quote"]), str(c.get("doc") or ""), policy)


def _norm(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower().rstrip(".")


def slots(claims: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Claims grouped by case and field, `<case> · <field>`."""
    out: dict[str, list[dict[str, Any]]] = {}
    for c in claims:
        out.setdefault(f"{_norm(c.get('case'))} · {_norm(c.get('field'))}", []).append(c)
    return out


def flagged(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    """The slots whose claims give two or more different values."""
    return [k for k, cs in groups.items() if len({_norm(c.get("value")) for c in cs}) > 1]


def _claim_line(c: dict[str, Any], kind: bool = False) -> str:
    q = (
        f' · quote ({c.get("doc") or "?"}{"" if c.get("quote_found", True) else ", NOT FOUND in it"}): "{c["quote"]}"'
        if c.get("quote")
        else " · no quote"
    )
    k = f" · {c['kind']}" if kind and c.get("kind") else ""
    d = f" · derivation: {c['derivation']}" if kind and c.get("derivation") else ""
    if kind and c.get("vquote") and c.get("vquote") != c.get("quote"):
        d += f' · the passage that decides it ({c.get("vdoc") or "?"}): "{c["vquote"]}"'
    return (
        f"- {c['id']} [{c.get('from', '')} {', '.join(c['origin'])}]{k}: {c.get('value', '')}{q}{d}"
    )


def slots_text(groups: dict[str, list[dict[str, Any]]], flags: list[str]) -> str:
    parts = []
    for k, cs in groups.items():
        head = f"### {k}" + ("  ← two or more values: settle it" if k in flags else "")
        parts.append(head + "\n" + "\n".join(_claim_line(c) for c in cs))
    return "\n\n".join(parts)


def parse_verdicts(
    text: str | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]] | None:
    from tau2_loop.workflows.rag_agent import _json_object

    obj = _json_object(text, lambda v: isinstance(v.get("claims"), list))
    if obj is None:
        return None
    kinds = ("stated", "derived", "question", "unsupported")
    claims = {
        str(v["id"]): v
        for v in obj["claims"]
        if isinstance(v, dict) and v.get("id") and v.get("kind") in kinds
    }
    settled = [s for s in obj.get("slots") or [] if isinstance(s, dict) and s.get("slot")]
    return claims, settled


def survivors(
    claims: list[dict[str, Any]],
    verdicts: dict[str, dict[str, Any]],
    settled: list[dict[str, Any]],
    flags: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The claims the job is written from, each with its kind, and the rejected ones with why.
    A claim survives when it is stated, derived or a question, and, in a slot with two values,
    when the verifier kept it (or made the slot a question)."""
    keep: dict[str, set[str]] = {}
    ask: set[str] = set()
    for s in settled:
        k = _norm(s["slot"])
        ids = s.get("keep") or []
        keep[k] = {str(i) for i in ([ids] if isinstance(ids, str) else ids)}
        if s.get("question"):
            ask.add(k)
    groups = slots(claims)
    slot_of = {c["id"]: k for k, cs in groups.items() for c in cs}
    out, rejected = [], {}
    for c in claims:
        v = verdicts.get(c["id"])
        if v is None:
            rejected[c["id"]] = "the verifier gave it no verdict"
            continue
        if v["kind"] == "unsupported":
            rejected[c["id"]] = str(v.get("why") or "unsupported")
            continue
        k = slot_of[c["id"]]
        if (
            k in flags
            and _norm(k) in keep
            and c["id"] not in keep[_norm(k)]
            and _norm(k) not in ask
        ):
            rejected[c["id"]] = f"its slot ({k}) was settled for another value"
            continue
        kind = "question" if _norm(k) in ask else v["kind"]
        out.append(
            {
                **c,
                "kind": kind,
                "derivation": v.get("derivation") or "",
                "why": v.get("why") or "",
                "vquote": v.get("quote") or "",
                "vdoc": v.get("doc") or "",
            }
        )
    return out, rejected


def claims_in(job: dict[str, Any]) -> dict[str, list[str]]:
    """{claim id: [where]} for every claim a written job's steps and rules carry."""
    out: dict[str, list[str]] = {}
    for st in job.get("steps") or []:
        if isinstance(st, dict):
            for c in st.get("claims") or []:
                out.setdefault(str(c), []).append(f"step {st.get('id')}")
    for k, r in enumerate(job.get("rules") or [], 1):
        if isinstance(r, dict):
            for c in r.get("claims") or []:
                out.setdefault(str(c), []).append(f"rule {k}")
    return out


def claims_from_opt(opt: dict[str, Any] | None) -> dict[str, list[str]]:
    """{claim id: [where]} from the marks `take_fields` lifted off a written job."""
    out: dict[str, list[str]] = {}
    for sid, m in ((opt or {}).get("marks") or {}).items():
        for c in m.get("claims") or []:
            out.setdefault(str(c), []).append(f"step {sid}")
    for rule, cs in ((opt or {}).get("rule_claims") or {}).items():
        for c in cs or []:
            out.setdefault(str(c), []).append(f'rule "{rule[:40]}"')
    return out


def coverage_lines(
    have: dict[str, list[str]], surv: list[dict[str, Any]], rejected: dict[str, str]
) -> list[str]:
    """C's check: every surviving claim is in the job, no rejected claim is."""
    missing = [c for c in surv if c["id"] not in have]
    used = [i for i in have if i in rejected]
    lines = [
        f"claim {c['id']} survived and no step or rule carries it ({str(c.get('value', ''))[:140]}): "
        "write it into the step it decides and list its id in that step's `claims`"
        for c in missing
    ]
    lines += [
        f"claim {i} was rejected ({rejected[i][:140]}) and {', '.join(have[i])} carries it: take it out"
        for i in used
    ]
    return lines


def ledger_job(
    domain: str,
    prompts: dict[str, str],
    name: str,
    old: list[dict[str, Any]],
    draft: dict[str, Any] | None,
    policy: str,
    schema: str,
    call: Call,
    emit: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """C for one job: extract, verify, write, each in its own context. Returns the written job
    (claims still on its steps) and the ledger, which keeps the writer's conversation."""
    text, origins = label_versions(old, draft)
    msgs = [{"role": "user", "content": prompts["extract"].format(job=name, versions=text)}]
    claims = parse_claims(call(msgs, "extract"))
    gap = missing_origins(claims or [], origins) if claims is not None else list(origins)
    if claims is None or gap:
        msgs += [
            {"role": "assistant", "content": json.dumps({"claims": claims or []})},
            {
                "role": "user",
                "content": (
                    "These steps and rules have no claim: "
                    + ", ".join(gap[:80])
                    + ". Reply with the whole JSON again, every claim, with a claim for each of them."
                    if claims is not None
                    else "Reply with one JSON object only, in the shape asked for."
                ),
            },
        ]
        claims = parse_claims(call(msgs, "extract")) or claims or []
    mark_quotes(domain, claims, policy)
    groups = slots(claims)
    flags = flagged(groups)
    emit(
        {
            "kind": "claims",
            "job": name,
            "claims": len(claims),
            "slots": len(groups),
            "flagged": len(flags),
            "no_claim": missing_origins(claims, origins),
        }
    )
    jobs = [*old, *([draft] if draft else [])]
    vtext = prompts["verify"].format(
        job=name,
        slots=slots_text(groups, flags),
        documents=documents_text(domain, jobs, policy) or "(none cited)",
        tools=tools_text(domain, jobs) or "(none)",
    )
    vmsgs = [{"role": "user", "content": vtext}]
    got = parse_verdicts(call(vmsgs, "verify"))
    verdicts, settled = got if got is not None else ({}, [])
    unjudged = [c["id"] for c in claims if c["id"] not in verdicts]
    if unjudged:
        vmsgs += [
            {
                "role": "assistant",
                "content": json.dumps({"claims": list(verdicts.values()), "slots": settled}),
            },
            {
                "role": "user",
                "content": "These claims have no verdict: "
                + ", ".join(unjudged[:120])
                + ". Reply with the whole JSON again, a verdict for every claim.",
            },
        ]
        again = parse_verdicts(call(vmsgs, "verify"))
        if again is not None:
            verdicts = {**verdicts, **again[0]}
            settled = again[1] or settled
    surv, rejected = survivors(claims, verdicts, settled, flags)
    emit(
        {
            "kind": "verified",
            "job": name,
            "survive": len(surv),
            "rejected": len(rejected),
            "kinds": {
                k: sum(c["kind"] == k for c in surv) for k in ("stated", "derived", "question")
            },
        }
    )
    meta = {
        k: v
        for k, v in {
            "names": sorted({str(j.get("job")) for j in jobs if j.get("job")}),
            "aliases": sorted(
                {a for j in old for a in j.get("aliases") or [] if isinstance(a, str)}
            ),
            "when": [j.get("when") for j in jobs if j.get("when")],
            "starts": [j.get("starts") for j in jobs if j.get("starts")],
            "info": [i for j in jobs for i in j.get("info") or []],
            "tools": [t for j in jobs for t in j.get("tools") or []],
            "done_when": [j.get("done_when") for j in jobs if j.get("done_when")],
        }.items()
    }
    wtext = prompts["write"].format(
        job=name,
        claims="\n".join(_claim_line(c, kind=True) for c in surv),
        settled=json.dumps(settled, ensure_ascii=False, indent=1) if settled else "(none)",
        meta=json.dumps(meta, ensure_ascii=False, indent=1),
        tools=tools_text(domain, jobs) or "(none)",
        schema=schema,
    )
    wmsgs = [{"role": "user", "content": wtext}]
    reply = call(wmsgs, "write")
    wmsgs.append({"role": "assistant", "content": reply or ""})
    from tau2_loop.workflows.rag_agent import parse_workflow

    wf = parse_workflow(reply)
    job = next((j for j in (wf or {}).get("jobs") or [] if isinstance(j, dict)), None)
    if job is not None:
        job = {**job, "job": name}
    ledger = {
        "claims": claims,
        "slots": {k: [c["id"] for c in cs] for k, cs in groups.items()},
        "flagged": flags,
        "verdicts": verdicts,
        "settled": settled,
        "survive": [c["id"] for c in surv],
        "rejected": rejected,
        "_surv": surv,
        "_writer": wmsgs,
    }
    return job, ledger


def ledger_changelog(ledger: dict[str, Any]) -> list[dict[str, str]]:
    """What the ledger left out of the job, each with the verifier's reason."""
    by_id = {c["id"]: c for c in ledger["claims"]}
    return [
        {
            "change": "removed",
            "what": f"claim {i} ({', '.join(by_id[i]['origin'])}): {str(by_id[i].get('value', ''))[:120]}",
            "why": f"{why[:200]}; {CODE}",
        }
        for i, why in ledger["rejected"].items()
        if i in by_id
    ]


def schema_text(rag_prompt: str) -> str:
    """The reading rules and the job's JSON shape, cut from the researcher's own instructions."""
    a = rag_prompt.find("## Reading the documents into rules")
    b = rag_prompt.find("# What the answering agent can do")
    c = rag_prompt.find("# What you write")
    d = rag_prompt.find("# How every workflow is judged")
    rules = rag_prompt[a:b] if 0 <= a < b else ""
    shape = rag_prompt[c:d] if 0 <= c < d else ""
    return (rules + "\n" + shape).replace("{{", "{").replace("}}", "}")


def option_lines(
    domain: str,
    row: dict[str, Any],
    protect: dict[str, dict[str, dict[str, Any]]] | None,
    ledgers: dict[str, dict[str, Any]],
    policy: str,
) -> list[str]:
    """An option's own send-back lines for one merged job: A's and B's unaccounted old steps, C's
    claims missing from the job or rejected and in it."""
    if row.get("blocked"):
        return []
    names = [e["job"] for e in row["previous"]]
    lines: list[str] = []
    if protect:
        guard: dict[str, dict[str, Any]] = {}
        for n in names:
            guard |= protect.get(n) or {}
        missing = unaccounted(domain, row["job"], guard, accounts(row.get("opt")), policy)
        lines += account_lines(missing, guard)
    led = next((ledgers[n] for n in names if n in ledgers), None)
    if led is not None:
        lines += coverage_lines(claims_from_opt(row.get("opt")), led["_surv"], led["rejected"])
    return lines


def post_audit(
    domain: str,
    prompt: str,
    rows: list[dict[str, Any]],
    policy: str,
    gate_checks: tuple[str, ...],
    call: Call,
    emit: Callable[[dict[str, Any]], None],
) -> None:
    """D's last word on each merged job, after the send-back: the auditor reads it with the whole
    documents it cites, its rewrites replace the clauses they fix, and an invalid step goes only if
    the job still passes the gate without it (the reviewer removed r19 086's filing step over one
    clause, and the same removal refused A's and C's merges). Changes are written to the changelog."""
    from tau2_loop.workflows import rubric

    for r in rows:
        if r.get("blocked") or not r.get("previous"):
            continue
        job = r["job"]
        reply = call(
            [{"role": "user", "content": audit_request(domain, prompt, job, policy)}], "post_audit"
        )
        v = parse_audit(reply)
        if v is None:
            emit(
                {
                    "kind": "note",
                    "text": f"The post-merge audit of {job.get('job')} was unreadable; the job is kept as merged.",
                }
            )
            continue
        a = audited(job, v)
        new_steps = [{k: x for k, x in st.items() if k != "audit"} for st in a["shown"]["steps"]]
        prev = [e["workflow"] for e in r["previous"]]
        before = rubric.gate(rubric.check(domain, job, prev, policy), gate_checks, draft_base=True)
        trial = {**job, "steps": new_steps, "rules": a["shown"]["rules"]}
        after = rubric.gate(rubric.check(domain, trial, prev, policy), gate_checks, draft_base=True)
        kept_back: list[str] = []
        if len(after) > len(before):  # an invalid step the job needs stays, as it was merged
            steps = [st for st in job.get("steps") or [] if isinstance(st, dict)]
            calls = {str(st.get("id")): rubric.step_tool(job, k) for k, st in enumerate(steps)}
            gone = [x["step"] for x in a["removed"] if "step" in x]
            failing = " ".join(after)
            need = [g for g in gone if g in failing or (calls.get(g) or "\0") in failing]
            audited_steps = {str(st.get("id")): st for st in new_steps}

            def keep(
                back: list[str],
                steps: list[dict[str, Any]] = steps,
                audited_steps: dict[str, dict[str, Any]] = audited_steps,
            ) -> list[dict[str, Any]]:
                return [
                    audited_steps.get(str(st.get("id")), st)
                    for st in steps
                    if str(st.get("id")) in audited_steps or str(st.get("id")) in back
                ]

            new_steps = keep(need)
            trial = {**job, "steps": new_steps, "rules": a["shown"]["rules"]}
            again = rubric.gate(
                rubric.check(domain, trial, prev, policy), gate_checks, draft_base=True
            )
            if len(again) > len(before):  # not enough: every step the audit removed stays
                need = gone
                new_steps = keep(need)
                trial = {**job, "steps": new_steps, "rules": a["shown"]["rules"]}
            kept_back = need
        log = [
            {
                "change": "changed",
                "what": f"step {sid}",
                "why": f"rewritten by the post-merge audit: {str(v['steps'][sid].get('why') or '')[:200]}; {CODE}",
            }
            for sid, x in v["steps"].items()
            if x["verdict"] == "rewrite"
        ] + [
            {
                "change": "removed",
                "what": f"step {x['step']}" if "step" in x else f"rule {x['rule'][:80]}",
                "why": f"the post-merge audit finds it unsupported: {x['why'][:200]}; {CODE}",
            }
            for x in a["removed"]
            if x.get("step") not in kept_back
        ]
        job["steps"], job["rules"] = trial["steps"], trial["rules"]
        job["changelog"] = [*(job.get("changelog") or []), *log]
        r["post_audit"] = {
            "verdicts": v,
            "removed": [x for x in a["removed"] if x.get("step") not in kept_back],
            "kept_for_the_gate": kept_back,
        }
        emit(
            {
                "kind": "post_audited",
                "job": job.get("job"),
                "rewrite": sum(x["verdict"] == "rewrite" for x in v["steps"].values()),
                "invalid": sum(x["verdict"] == "invalid" for x in v["steps"].values()),
                "kept_for_the_gate": kept_back,
            }
        )
