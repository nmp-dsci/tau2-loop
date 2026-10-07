"""The workflow library an answering agent looks up (s16, v6): what workflow_rag already wrote.

The library is every job in the workflows that a RAG-agent version wrote on train questions
(`rag_agent_runs/`, sessions with `source: question` that ended `done`), the latest session's
version of each job winning. A workflow researched mid-conversation (`request_workflow`) is
never added: the conversation could be a test one, and a workflow from it would be a cached
answer. Nothing here calls a model except `request()`, which runs a RAG-agent session.

The two functions behind v6's harness tools return text for the model to read:
`find()` ranks the library's jobs against the agent's one-line request by shared words (no
model, so the same request always gets the same answer) and gives the best match whole;
`request()` asks workflow_rag to research the request and returns what it wrote.

Since s20 a version may name a `seed` (r2: r1): its library starts from the seed's, and each of
its sessions merges the question into the jobs it shares, so a later version contains the
earlier one instead of replacing it; a merged job's `aliases` retire the names it absorbed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tau2_loop.workflows import rag_agent

STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "customer",
        "wants",
        "would",
        "like",
        "help",
    ]
)
SHOWN = 8  # jobs listed per lookup


def _stem(w: str) -> str:
    """A crude stem, so close, closing and closure meet; short words are left whole."""
    return re.sub(r"(ings?|ures?|ed|es|s|e)$", "", w) if len(w) > 4 else w


def _words(text: str) -> list[str]:
    return [
        _stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 1
    ]


def _job_text(job: dict[str, Any]) -> str:
    parts = [str(job.get("job", "")).replace("_", " ")]
    when = job.get("when") or {}
    parts.append(str(when.get("quote", "")) if isinstance(when, dict) else "")
    for st in job.get("steps") or []:
        if isinstance(st, dict):
            parts.append(str(st.get("do", "")))
    return " ".join(parts)


def _seed(domain: str, rag_version: str) -> str | None:
    """The version whose library this one starts from (s20: r2 starts from r1's), if any."""
    try:
        return rag_agent.load(domain, rag_version).seed
    except rag_agent.RagAgentError:
        return None


def entries(domain: str, rag_version: str, root: Path | None = None) -> list[dict[str, Any]]:
    """One entry per job, from the latest finished train-question session that wrote it.

    A version with a `seed` (s20, r2) starts from the seed's library, and each of its sessions
    wrote merged versions: a job replaces its own name and every name it lists in `aliases`
    or merged `into` it, so two names for one procedure become one job."""
    root = root or rag_agent.RAG_RUNS_DIR
    seed = _seed(domain, rag_version)
    out: dict[str, dict[str, Any]] = (
        {e["job"]: e for e in entries(domain, seed, root)} if seed and seed != rag_version else {}
    )
    if not root.is_dir():
        return list(out.values())
    for d in sorted(root.iterdir()):  # oldest first, so a later session's job replaces it
        meta_p, out_p = d / "run.json", d / "output.json"
        if not (meta_p.is_file() and out_p.is_file()):
            continue
        meta = json.loads(meta_p.read_text())
        if (
            meta.get("domain") != domain
            or meta.get("agent") != rag_version
            or meta.get("status") != "done"
            or meta.get("source", "question") != "question"
        ):
            continue
        wf = json.loads(out_p.read_text())
        for job in wf.get("jobs") or []:
            if isinstance(job, dict) and job.get("job"):
                for old in merged_names(job):
                    out.pop(old, None)
                out[str(job["job"])] = {
                    "job": str(job["job"]),
                    "workflow": job,
                    "session": meta["id"],
                    "task_id": meta.get("task_id"),
                }
    return list(out.values())


def merged_names(job: dict[str, Any]) -> list[str]:
    """The other names a merged job stands for: its `aliases` and the jobs merged `into` it."""
    names: list[str] = []
    for k in ("aliases", "into"):
        v = job.get(k) or []
        for n in [v] if isinstance(v, str) else v:
            if isinstance(n, str) and n and n != job.get("job") and n not in names:
                names.append(n)
    return names


def by_name(lib: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """A library job by its name or one of its aliases."""
    name = name.strip()
    return next((e for e in lib if e["job"] == name or name in merged_names(e["workflow"])), None)


def rank(request: str, lib: list[dict[str, Any]]) -> list[tuple[float, bool, dict[str, Any]]]:
    """Jobs by the request's words they share, best first; ties keep name order. A shared word
    weighs by how few of the library's jobs use it (idf), so "cash back" outweighs "credit card",
    which nearly every banking job says; a word of the job's own name counts twice. A job is a
    match only when the request shares a word of its name: a lookup for closing a card must not
    offer the cash-back dispute because both say "credit card"."""
    want = set(_words(request))
    texts = [
        (
            set(_words(" ".join([e["job"], *merged_names(e["workflow"])]).replace("_", " "))),
            set(_words(_job_text(e["workflow"]))),
        )
        for e in lib
    ]
    df: dict[str, int] = {}
    for name, body in texts:
        for w in name | body:
            df[w] = df.get(w, 0) + 1

    def weight(words: set[str]) -> float:
        return sum(math.log(1 + len(lib) / df[w]) for w in words)

    scored = []
    for e, (name, body) in zip(lib, texts, strict=True):
        named = want & name
        score = round((2 * weight(named) + weight(want & body)) / math.sqrt(len(want) or 1), 3)
        scored.append((score, bool(named), e))
    return sorted(scored, key=lambda x: (not x[1], -x[0], x[2]["job"]))


def find(domain: str, rag_version: str, request: str, job: str = "") -> str:
    """`find_workflow`'s result: the library's jobs ranked for the request, and one in full."""
    lib = entries(domain, rag_version)
    if not lib:
        return (
            "The workflow library is empty. Call request_workflow to have workflow_rag research "
            "this request, or work from the knowledge base."
        )
    if job:
        e = by_name(lib, job)
        if e is None:
            names = ", ".join(x["job"] for x in lib)
            return f"No job named {job!r} in the library. Its jobs: {names}."
        return _full(e)
    ranked = rank(request, lib)
    n = f"{len(lib)} workflow{'' if len(lib) == 1 else 's'}"
    lines = [f"The library holds {n}. Ranked for your request:"]
    for score, named, e in ranked[:SHOWN]:
        when = (e["workflow"].get("when") or {}).get("quote", "")
        tag = f"match {score:g}" if named else "not a match"
        lines.append(f"- {e['job']} ({tag}): {str(when)[:160]}")
    _, named, best = ranked[0]
    if not named:
        lines.append(
            "\nNo job's name shares a word with your request, so none is shown. If none of these "
            "is the customer's job, call request_workflow to have workflow_rag research it."
        )
        return "\n".join(lines)
    lines.append(
        "\nThe best match, in full. Follow it only if it is the customer's job; for another job "
        "in the list call find_workflow with job=<name>, and if none fits call request_workflow."
    )
    return "\n".join(lines) + "\n\n" + _full(best)


def _full(e: dict[str, Any]) -> str:
    return (
        f"Workflow {e['job']} (researched by workflow_rag, session {e['session']}):\n"
        + json.dumps(e["workflow"], ensure_ascii=False, indent=1)
    )


def request(
    domain: str,
    rag_version: str,
    text: str,
    *,
    conversation: str | None = None,
    on_start: Callable[[rag_agent.SessionMeta], None] | None = None,
) -> tuple[str, str]:
    """`request_workflow`'s result and the session id: workflow_rag researches the request now."""
    meta = rag_agent.start_request(
        domain, text, rag_version, conversation=conversation, on_start=on_start
    )
    got = rag_agent.session(domain, meta.id)
    wf = got.get("output")
    if not wf:
        return (
            f"workflow_rag could not write a workflow ({meta.error or 'no workflow'}). Work from "
            "the knowledge base with your search tools.",
            meta.id,
        )
    return (
        f"workflow_rag researched your request (session {meta.id}) and wrote:\n"
        + json.dumps(wf, ensure_ascii=False, indent=1),
        meta.id,
    )
