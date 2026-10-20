"""workflow_rag's RAG agent (s16): one agentic-RAG session per train question, live in the viewer.

A session reads one customer's train script, researches the knowledge base with the answering
agent's own knowledge tools (v4's `alltools_minilm`: BM25, dense search and the read-only shell)
and writes the workflow the answering agent would follow, as JSON. The model runs on the
subscription through `llm.core.answer`, one SDK session per turn with native tool calls, exactly
as the answering agent does; this module runs each call in tau2's banking environment and feeds
the result back, until the model writes the workflow or reaches the version's step limit.

What it never sees: gold. Its prompt is the version's `rag_agent.md`, the answering agent's
policy and tool schemas, and the customer's script. Gold (the task's actions and required
documents) is read only by `score()`, after the session, to show how close it came. Only train
questions can be started; a test id is refused before anything runs. The shell is tau2's
best-effort sandbox, so a command that names the task files is refused here as well.

A version lives in `rag_agents/<domain>/<rN>/` (prompt + frozen settings, like `agents/`); each
session in `rag_agent_runs/<id>/` (`run.json`, `events.jsonl`, `output.json`, `score.json`), kept
like a run folder: never edited once it ends.

s20 (r2): a version with `merge: sequential` follows its draft with a merge. The harness shows it
the library's current version of each drafted job (`merge.md`), it writes the next versions with a
changelog, and `rubric.gate` checks each without a model (nothing lost that the changelog does not
name, no quote that is not found); a failure is sent back once, and a merge that still fails is not
taken, so the library keeps the version it had. `make rag-build` runs the train questions one at
a time in id order. Gold stays out of every message the model sees.

s21 (r3): a version with `merge: concurrent` is built by several workers at once. Research takes
no lock; after the draft one short turn names the library jobs it will write (`decide.md`), the
store locks them all at once (`store.FileStore`: a worker waits, and logs who holds what), the
merge reads their newest versions, and the commit is refused unless every version it read is
still the head. Every message the harness sends is saved as a `user` event, so a session's
conversation rebuilds turn for turn.

s23 (r18): every clause from its source (`workflows/provenance.py`). A step gives each deciding
clause its own quote (`basis`); code picks the steps whose clauses may not follow from their quotes,
kept steps included, and a reviewer reads them in its own context (`review.md`, a separate model
call saved as a `review` event, no gold, no outcome); the documents the session printed in full and
cites nowhere are listed in the merge turn and must be cited or set aside (`set_aside`); a called
tool's description that names a document's title is followed; and a tool's description lists its
error replies. The reviewer's "no", the unaccounted documents and the named titles go back to the
writer once with the gate's reasons, and are recorded, never blocking: the gate is still D1, D8
and the version's `rubric` checks.
"""

from __future__ import annotations

import json
import re
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from tau2_loop.config import ROOT
from tau2_loop.data import documents as kb
from tau2_loop.data.splits import read_task_extract, split_ids

RAG_AGENTS_DIR = ROOT / "rag_agents"
# what precedes the question in the agent's first message: a train script, or a live request
SCRIPT_OPENING = "The customer's situation, as their script tells it:"
REQUEST_OPENING = (
    "An answering agent is talking to a customer now and asks for the procedure. Its request, in "
    "its own words (you do not see the customer):"
)
RAG_RUNS_DIR = ROOT / "rag_agent_runs"
QUOTE_CHARS = 300
WRITE_AGAIN = "Reply with the workflow as one JSON object only, in the shape the instructions give."
TOOLS_CLOSED = "no more tool calls in this turn; reply with the JSON asked for."
# a shell command may not reach for the task files, whatever tau2's sandbox allows
SHELL_REFUSED = re.compile(r"task|evaluation_criteria|\.\.|~|^\s*/|\s/", re.I)
DOC_ID = re.compile(r"\bdoc_[A-Za-z0-9_()\-]+")
# how a quote cites the answering agent's policy rather than a knowledge-base document
POLICY_REFS = frozenset({"policy", "the policy", "agent policy", "customer service policy"})


class RagAgentError(ValueError):
    """A request a session cannot serve: an unknown version, a non-train question."""


@dataclass(frozen=True)
class RagAgent:
    name: str
    domain: str
    model: str
    effort: str
    retrieval: str
    tool_mode: str
    max_steps: int
    result_chars: int
    prompt: str  # rag_agent.md, unformatted
    # s20: r2 starts from r1's library (`seed`) and merges each question into it (`merge`)
    seed: str | None = None
    merge: str | None = None  # "sequential" (r2) or "concurrent" (s21, r3): merged into the library
    merge_steps: int = 12  # model turns for the merge, its research included
    merge_prompt: str = ""  # merge.md, unformatted: the harness's message after the draft
    # s21: a concurrent build's workers, the order it takes questions in, and which it skips
    workers: int = 1
    order: str = "id"  # "id" or "shuffled" (by `order_seed`, so the order can be rebuilt)
    order_seed: int = 0
    queue: str = ""  # "after_seed": skip every question the seed version already merged
    decide_prompt: str = ""  # decide.md, unformatted: which library jobs it will write
    # r4 (8 Oct 2026): what r3's merges spent. A turn that must write keeps the tools listed (a call
    # is refused, not run), so the prompt cache reads the conversation back: dropping them changed
    # the prompt's first block, and r3's decide turns read 3% from cache. A merge may write a job
    # as edits on the version it read (`workflows/edits.py`) instead of the whole job again.
    keep_tools: bool = False
    merge_write: str = "full"  # "full" (r2, r3) or "edits"
    # s22 (r5): rubric checks the merge gate adds to D8 and D1 (D9 parameters, D10 statuses)
    rubric: tuple[str, ...] = ()
    # s22 (r13): workflow_rag reads each tool's own description, as the answering agent does when it
    # unlocks it: a `tool_description` research tool, and the descriptions of every tool a job
    # calls in the decide and merge turns. r12 never saw that 'savings' means personal savings.
    tool_docs: bool = False
    # s22 (r17): those turns describe every tool a job names, the ones its `tools` list rules out
    # included. Before it only step calls and numbered names were collected, so once r14 ruled out
    # the customer's `submit_referral`, no later merge showed its description again.
    describe_named: bool = False
    # s23 (r18): a tool's description lists its error replies (M6), and a reviewer reads the clauses
    # code picks (M2): (model, effort) and its instructions, `review.md`
    tool_errors: bool = False
    reviewer: tuple[str, str] | None = None
    review_prompt: str = ""
    # s25 (r19, the person's call, 15 Oct 2026): a better merge on the same research. `drafts_from`
    # names the version whose sessions a re-merge takes: each one's research is rebuilt from its
    # events, its draft and decision are reused, and only the merge runs again. `merge_base:
    # draft` makes the draft the base, an old step carried only when it covers a case the draft
    # does not (M1, M2), and what it drops is logged by code, never a rejection (M3).
    # `review_binding` removes a step the reviewer still finds unsupported after the send-back (R1).
    drafts_from: str | None = None
    merge_base: str = "library"  # "library" (r2-r18) or "draft" (r19)
    review_binding: bool = False
    # s30 (the merge bench, `workflows/merge_opts.py`): `merge_account` (option A) carries an old
    # step whose quotes are found unless the merge names what covers or replaces it; `merge_mode`
    # "audit" (B) has an auditor read the library's version against its documents before the merge
    # turn, "ledger" (C) merges by claims in fresh contexts instead of a merge turn. `stages` is
    # the (model, effort) of those calls; their instructions are audit.md, extract.md, verify.md
    # and write.md.
    merge_account: bool = False
    merge_mode: str = "turn"
    stages: tuple[str, str] | None = None
    stage_prompts: tuple[tuple[str, str], ...] = ()
    # s30 (option D): after the send-back the auditor reads the merged job against its whole
    # documents and rewrites what they do not support, in place of the reviewer's binding removal
    post_audit: bool = False


def versions(domain: str) -> list[str]:
    root = RAG_AGENTS_DIR / domain
    if not root.is_dir():
        return []
    return sorted(
        (p.name for p in root.iterdir() if (p / "rag_agent.yaml").is_file()),
        key=lambda n: int(re.sub(r"\D", "", n) or 0),
    )


def load(domain: str, name: str) -> RagAgent:
    d = RAG_AGENTS_DIR / domain / name
    if not re.fullmatch(r"r\d+", name) or not (d / "rag_agent.yaml").is_file():
        raise RagAgentError(f"no RAG agent {name} for {domain}")
    cfg = yaml.safe_load((d / "rag_agent.yaml").read_text())
    return RagAgent(
        name=name,
        domain=domain,
        model=str(cfg["model"]),
        effort=str(cfg["effort"]),
        retrieval=str(cfg["retrieval"]),
        tool_mode=str(cfg.get("tool_mode", "native_parallel")),
        max_steps=int(cfg.get("max_steps", 30)),
        result_chars=int(cfg.get("result_chars", 20000)),
        prompt=(d / "rag_agent.md").read_text(),
        seed=cfg.get("seed"),
        merge=cfg.get("merge"),
        merge_steps=int(cfg.get("merge_steps", 12)),
        merge_prompt=(d / "merge.md").read_text() if (d / "merge.md").is_file() else "",
        workers=int(cfg.get("workers", 1)),
        order=str(cfg.get("order", "id")),
        order_seed=int(cfg.get("order_seed", 0)),
        queue=str(cfg.get("queue") or ""),
        decide_prompt=(d / "decide.md").read_text() if (d / "decide.md").is_file() else "",
        keep_tools=bool(cfg.get("keep_tools", False)),
        merge_write=str(cfg.get("merge_write", "full")),
        rubric=tuple(str(x) for x in cfg.get("rubric") or ()),
        tool_docs=bool(cfg.get("tool_docs", False)),
        describe_named=bool(cfg.get("describe_named", False)),
        tool_errors=bool(cfg.get("tool_errors", False)),
        reviewer=(
            (str(cfg["reviewer"]["model"]), str(cfg["reviewer"]["effort"]))
            if isinstance(cfg.get("reviewer"), dict)
            else None
        ),
        review_prompt=(d / "review.md").read_text() if (d / "review.md").is_file() else "",
        drafts_from=cfg.get("drafts_from"),
        merge_base=str(cfg.get("merge_base", "library")),
        review_binding=bool(cfg.get("review_binding", False)),
        **merge_settings(cfg, d),
    )


STAGE_FILES = ("audit", "extract", "verify", "write")


def merge_settings(cfg: dict[str, Any], d: Path) -> dict[str, Any]:
    """s30: the merge options' settings from a yaml and the folder beside it (an rN or a bench
    option, `workflows/bench.py`)."""
    st = cfg.get("stages")
    return {
        "merge_account": bool(cfg.get("merge_account", False)),
        "merge_mode": str(cfg.get("merge_mode", "turn")),
        "post_audit": bool(cfg.get("post_audit", False)),
        "stages": (str(st["model"]), str(st["effort"])) if isinstance(st, dict) else None,
        "stage_prompts": tuple(
            (s, (d / f"{s}.md").read_text()) for s in STAGE_FILES if (d / f"{s}.md").is_file()
        ),
    }


def train_question(domain: str, task_id: str, split: str = "train") -> str:
    """The customer's whole train script; a test or unknown id is refused, unless the caller names
    the test split (`make rag-build SPLIT=test`, the person's call for r3 on 8 Oct 2026)."""
    if split not in ("train", "test"):
        raise RagAgentError(f"split is train or test, not {split}")
    if task_id not in set(split_ids(domain, split)):
        raise RagAgentError(
            f"{task_id} is not a train question: test is sealed"
            if split == "train"
            else f"{task_id} is not a {split} question"
        )
    for t in read_task_extract(domain)["tasks"]:
        if t["id"] == task_id:
            return str(((t.get("user_scenario") or {}).get("instructions")) or "").strip()
    raise RagAgentError(f"{task_id} is not in the task extract")


# ── the environment: tau2's banking tools, built once per process ──

_env_lock = threading.Lock()
_envs: dict[str, Any] = {}


def environment(retrieval: str) -> Any:
    """tau2's banking environment with the given retrieval variant, built once and shared:
    sessions call only its read-only knowledge tools."""
    with _env_lock:
        env = _envs.get(retrieval)
        if env is None:
            from tau2_loop.config import quiet_tau2

            quiet_tau2()
            from tau2_loop.eval.retrieval import register
            from tau2_loop.llm import scrub_injected_key

            register()
            from tau2.domains.banking_knowledge.environment import get_environment

            scrub_injected_key()  # tau2's dotenv search may have pulled a key in
            env = get_environment(retrieval_variant=retrieval)
            _envs[retrieval] = env
        return env


def _schema(t: Any) -> dict[str, Any]:
    return dict(t.openai_schema)


def tool_lists(env: Any, retrieval: str) -> tuple[list[dict[str, Any]], str, str]:
    """(the knowledge tools the agent calls, the answering agent's other tools as text, the
    customer's tools as text)."""
    from tau2_loop.eval.retrieval import variant_tools

    kb_names = set(variant_tools(retrieval))
    callable_ = [_schema(t) for t in env.get_tools() if t.name in kb_names]

    def listed(tools: list[Any]) -> str:
        rows = []
        for t in tools:
            fn = _schema(t)["function"]
            params = ", ".join((fn.get("parameters") or {}).get("properties", {}))
            first = ((fn.get("description") or "").strip().splitlines() or [""])[0]
            rows.append(f"- `{fn['name']}({params})`: {first}")
        return "\n".join(rows)

    agent = [t for t in env.get_tools() if t.name not in kb_names]
    return callable_, listed(agent), listed(list(env.get_user_tools()))


TOOL_DOC = "tool_description"
TOOL_DOC_TEXT = (
    "The whole description of one of the answering agent's or the customer's tools, as the "
    "answering agent reads it when it unlocks the tool: what it does, and what each "
    "parameter and each allowed value means."
)
TOOL_DOC_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_DOC,
        "description": TOOL_DOC_TEXT,
        "parameters": {
            "type": "object",
            "properties": {"tool_name": {"type": "string", "description": "the tool's exact name"}},
            "required": ["tool_name"],
        },
    },
}


def tool_doc_reply(domain: str, args: dict[str, Any], errors: bool = False) -> str:
    from tau2_loop.workflows import toolspec

    name = str(args.get("tool_name") or "").strip()
    return toolspec.describe(domain, name, errors) or f"No tool is named {name!r}."


def system_prompt(agent: RagAgent, env: Any) -> tuple[str, list[dict[str, Any]]]:
    tools, agent_tools, user_tools = tool_lists(env, agent.retrieval)
    text = agent.prompt.format(policy=env.policy, agent_tools=agent_tools, user_tools=user_tools)
    return text, [*tools, TOOL_DOC_SCHEMA] if agent.tool_docs else tools


# ── pure helpers, unit-tested ──


def shell_refusal(command: str) -> str | None:
    """Why a shell command is refused, or None. The knowledge base folder is flat, so nothing
    a search needs reaches outside it."""
    m = SHELL_REFUSED.search(command)
    if m:
        return (
            f"Refused: the command contains {m.group(0)!r}. Commands run inside the knowledge "
            "base folder only, on its documents."
        )
    return None


def _json_object(text: str | None, ok: Callable[[dict[str, Any]], bool]) -> dict[str, Any] | None:
    """The JSON object in a reply that `ok` accepts: the whole text, else the outermost {...}."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    for cand in (s, s[s.find("{") : s.rfind("}") + 1] if "{" in s else ""):
        try:
            v = json.loads(cand)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict) and ok(v):
            return v
    return None


def parse_workflow(text: str | None) -> dict[str, Any] | None:
    """The workflow JSON in a final reply: an object with a `jobs` list."""
    return _json_object(text, lambda v: isinstance(v.get("jobs"), list))


def _norm_id(s: str) -> str:
    return re.sub(r"[()]", "_", s.strip()).removesuffix(".md")


def resolve_doc(domain: str, ref: str) -> str | None:
    """A cited document as its id: the id itself, or the shell's file name for it."""
    index = kb.read_index(domain)
    if ref in index:
        return ref
    want = _norm_id(ref)
    return next((i for i in index if _norm_id(i) == want), None)


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("‑", "-")).strip()


def quotes(o: Any) -> list[dict[str, str]]:
    """Every {quote, doc} pair anywhere in a workflow."""
    out: list[dict[str, str]] = []
    if isinstance(o, dict):
        if isinstance(o.get("quote"), str) and o.get("quote"):
            out.append({"quote": o["quote"], "doc": str(o.get("doc") or "")})
        for v in o.values():
            out.extend(quotes(v))
    elif isinstance(o, list):
        for v in o:
            out.extend(quotes(v))
    return out


def quote_check(domain: str, wf: dict[str, Any], policy: str = "") -> dict[str, Any]:
    """Each quote looked up in the document it cites, whitespace folded; no model. A quote that
    cites the policy (not a knowledge-base document) is looked up in the policy the agent read."""
    rows = []
    for q in quotes(wf):
        if q["doc"].strip().lower() in POLICY_REFS:
            doc_id: str | None = "policy"
            text = policy or None
        else:
            doc_id = resolve_doc(domain, q["doc"]) if q["doc"] else None
            d = kb.document(domain, doc_id) if doc_id else None
            text = d["content"] if d else None
        found = text is not None and _squash(q["quote"].strip(" .…\"'")) in _squash(text)
        rows.append({**q, "doc_id": doc_id, "found": found, "long": len(q["quote"]) > QUOTE_CHARS})
    return {"n": len(rows), "found": sum(r["found"] for r in rows), "rows": rows}


def cited_docs(domain: str, wf: dict[str, Any]) -> list[str]:
    refs = [str(d.get("id") or "") for d in wf.get("documents") or [] if isinstance(d, dict)]
    refs += [q["doc"] for q in quotes(wf)]
    out: list[str] = []
    for r in refs:
        i = resolve_doc(domain, r) if r else None
        if i and i not in out:
            out.append(i)
    return out


def seen_docs(domain: str, events: list[dict[str, Any]]) -> list[str]:
    """Documents any tool result showed: a search hit's id, or a file name the shell printed."""
    out: list[str] = []
    for e in events:
        if e.get("kind") != "tool":
            continue
        for ref in DOC_ID.findall(str(e.get("content") or "")):
            i = resolve_doc(domain, ref)
            if i and i not in out:
                out.append(i)
    return out


def workflow_tools(wf: dict[str, Any]) -> list[str]:
    """The tools a workflow's steps call, a discoverable one by its own name."""
    out: list[str] = []
    for job in wf.get("jobs") or []:
        for st in job.get("steps") or [] if isinstance(job, dict) else []:
            c = st.get("call") if isinstance(st, dict) else None
            if isinstance(c, str) and c and c not in out:
                out.append(c)
    return out


def _gold_tool(name: str) -> str:
    return name.removeprefix("give ")


def score(
    domain: str,
    task_id: str,
    wf: dict[str, Any] | None,
    events: list[dict[str, Any]],
    policy: str = "",
) -> dict[str, Any]:
    """How close the session came to the golden entry: read only here, after the session."""
    from tau2_loop.workflows.golden import golden_set

    gs = golden_set(domain) or {"entries": []}
    entry = next((e for e in gs["entries"] if e["task_id"] == task_id), None)
    required = [d["id"] for d in (entry or {}).get("required_documents", [])]
    seen = seen_docs(domain, events)
    out: dict[str, Any] = {
        "golden": None
        if entry is None
        else {
            "workflows": entry["workflows"],
            "info": entry["info"],
            "calls": entry["calls"],
            "required_documents": entry["required_documents"],
        },
        "documents_seen": {
            "required": len(required),
            "hit": [d for d in required if d in seen],
            "n": len(seen),
        },
    }
    if wf is None:
        return out
    cited = cited_docs(domain, wf)
    gold_tools = sorted({_gold_tool(c["tool"]) for c in (entry or {}).get("calls", [])})
    tools = workflow_tools(wf)
    out.update(
        {
            "documents_referenced": {
                "required": len(required),
                "hit": [d for d in required if d in cited],
                "extra": [d for d in cited if d not in required],
            },
            "quotes": quote_check(domain, wf, policy),
            "tools": {
                "gold": gold_tools,
                "hit": [t for t in gold_tools if t in tools],
                "extra": [t for t in tools if t not in gold_tools],
            },
            "jobs": [str(j.get("job")) for j in wf.get("jobs") or [] if isinstance(j, dict)],
        }
    )
    return out


# ── sessions ──


@dataclass
class SessionMeta:
    id: str
    domain: str
    agent: str
    task_id: str
    model: str
    effort: str
    retrieval: str
    started: str
    status: str = "running"  # running | done | failed | interrupted
    steps: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: int = 0
    error: str | None = None
    # "question": started on a train question (the demo); "conversation": an answering agent's
    # `request_workflow` mid-conversation, which reads only the agent's words and is never cached
    source: str = "question"
    conversation: str | None = None  # the live conversation that asked, when one did
    # the question's split: train, or test only when a build is told to (r3, 8 Oct 2026, the
    # person's call: test researched into the library, so a test score that reads it is not held out)
    split: str = "train"
    # s25: a re-merge's source session ("remerge" sessions only), whose research it replays
    drafts_of: str | None = None


_live: dict[str, SessionMeta] = {}
_live_lock = threading.Lock()


def session_dir(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9TZ]+_[a-z_]+_r\d+_(task_\d+|req_[0-9a-f]{6})", session_id):
        raise RagAgentError(f"no session {session_id}")
    return RAG_RUNS_DIR / session_id


def _append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_meta(d: Path, meta: SessionMeta) -> None:
    (d / "run.json").write_text(json.dumps(asdict(meta), indent=1) + "\n")


Ask = Callable[[list[dict[str, Any]], list[dict[str, Any]] | None, str, str, str], Any]
RunTool = Callable[[str, dict[str, Any]], str]


def default_ask(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    model: str,
    effort: str,
    tool_mode: str,
) -> Any:
    from tau2_loop.llm import core

    return core.answer(messages, tools, model, effort=effort, tool_mode=tool_mode)


_env_run_locks: dict[int, threading.Lock] = {}


def env_runner(env: Any) -> RunTool:
    """Run a knowledge tool in `env`. Sessions share one environment, and several run at once in
    a concurrent build (s21), so its calls take one lock per environment, not one per session."""
    from tau2.data_model.message import ToolCall

    with _env_lock:
        lock = _env_run_locks.setdefault(id(env), threading.Lock())

    def run(name: str, args: dict[str, Any]) -> str:
        with lock:
            msg = env.get_response(
                ToolCall(id="rag", name=name, arguments=args, requestor="assistant")
            )
        return str(msg.content or "")

    return run


@dataclass(frozen=True)
class Replay:
    """s25 (r19): where a re-merge starts: a source session's research conversation rebuilt from
    its events, its draft and its decision. No research turn runs; only the merge does."""

    source: str
    messages: list[dict[str, Any]]
    draft: dict[str, Any]
    decision: dict[str, list[str]]


def run_session(
    d: Path,
    meta: SessionMeta,
    agent: RagAgent,
    system: str,
    tools: list[dict[str, Any]],
    question: str,
    ask: Ask,
    run_tool: RunTool,
    policy: str = "",
    opening: str = SCRIPT_OPENING,
    replay: Replay | None = None,
) -> SessionMeta:
    """The agentic loop: ask, run every tool call, feed the results back, until the model writes
    the workflow. Every step is appended to `events.jsonl` as it happens, which the viewer polls.
    A re-merge (`replay`, s25) starts from another session's research and runs only the merge."""
    events_path = d / "events.jsonl"
    events: list[dict[str, Any]] = []
    started = time.time()

    def emit(row: dict[str, Any]) -> None:
        row = {"t": int((time.time() - started) * 1000), **row}
        events.append(row)
        _append(events_path, row)

    names = {t["function"]["name"] for t in tools}
    messages: list[dict[str, Any]] = (
        list(replay.messages)
        if replay is not None
        else [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"{opening}\n\n{question}",
            },
        ]
    )
    emit(
        {
            "kind": "start",
            "question": question,
            **({"replayed_from": replay.source} if replay is not None else {}),
        }
    )

    def say(text: str) -> None:
        """A message from the harness: in the conversation and, in full, in the events (s21)."""
        messages.append({"role": "user", "content": text})
        emit({"kind": "user", "step": meta.steps + 1, "text": text})

    def turns(
        budget: int,
        write_now: str,
        parse: Callable[[str | None], Any] = parse_workflow,
        again: str = WRITE_AGAIN,
    ) -> Any:
        """Ask, run every tool call and feed the results back until the model writes the JSON;
        past `budget` model turns it is told to write with no tools, and asked once more if its
        reply is not the JSON. With `budget` 0, `write_now` is the question itself. A version that
        keeps the tools (r4) still lists them then, so the prompt is the cached one, and refuses a
        call instead of running it."""
        first = meta.steps
        nudged = retried = False
        while meta.steps - first < budget + 3:
            over = meta.steps - first >= budget
            if over and not nudged:
                say(write_now)
                nudged = True
            offer = tools if agent.keep_tools or not over else None
            ans = ask(messages, offer, meta.model, meta.effort, agent.tool_mode)
            meta.steps += 1
            res = getattr(ans, "result", None)
            if res is not None:
                meta.input_tokens += int(res.input_tokens or 0)
                meta.output_tokens += int(res.output_tokens or 0)
            emit(
                {
                    "kind": "model",
                    "step": meta.steps,
                    "text": ans.content,
                    "calls": [
                        {"id": c["id"], "name": c["name"], "args": c["arguments"]}
                        for c in ans.tool_calls
                    ],
                    "input_tokens": getattr(res, "input_tokens", None),
                    "output_tokens": getattr(res, "output_tokens", None),
                    "cache_read": getattr(res, "cache_read", None),
                    "ms": getattr(res, "duration_ms", None),
                }
            )
            if ans.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": c["id"],
                                "type": "function",
                                "function": {
                                    "name": c["name"],
                                    "arguments": json.dumps(c["arguments"]),
                                },
                            }
                            for c in ans.tool_calls
                        ],
                    }
                )
                for c in ans.tool_calls:
                    args = c["arguments"] if isinstance(c["arguments"], dict) else {}
                    refused = None
                    if over:
                        refused = f"Refused: {TOOLS_CLOSED}"
                    elif c["name"] not in names:
                        refused = f"Refused: {c['name']} is not one of your tools."
                    elif c["name"] == "shell":
                        refused = shell_refusal(str(args.get("command", "")))
                    out = refused or (
                        tool_doc_reply(meta.domain, args, agent.tool_errors)
                        if c["name"] == TOOL_DOC
                        else run_tool(c["name"], args)
                    )
                    cut = len(out) > agent.result_chars
                    if cut:
                        out = (
                            out[: agent.result_chars]
                            + f"\n[cut at {agent.result_chars:,} of {len(out):,} characters]"
                        )
                    meta.tool_calls += 1
                    emit(
                        {
                            "kind": "tool",
                            "step": meta.steps,
                            "id": c["id"],
                            "name": c["name"],
                            "args": args,
                            "content": out,
                            "refused": bool(refused),
                            "cut": cut,
                        }
                    )
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                _write_meta(d, meta)
                continue
            messages.append({"role": "assistant", "content": ans.content or ""})
            got = parse(ans.content)
            if got is not None or retried:
                return got
            retried = True  # one more chance to write it as JSON
            emit({"kind": "note", "text": "The reply was not the JSON asked for; asked once more."})
            say(again)
        return None

    def review(text: str) -> str | None:
        """s23 (r18): one reviewer call in its own context, with no tools and no gold; the
        message and the reply are saved as a `review` event."""
        if agent.reviewer is None:
            return None
        from tau2_loop.llm import core

        model, effort = agent.reviewer
        ans = ask(
            [{"role": "user", "content": text}],
            None,
            core.resolve_model(model),
            effort,
            agent.tool_mode,
        )
        res = getattr(ans, "result", None)
        if res is not None:
            meta.input_tokens += int(res.input_tokens or 0)
            meta.output_tokens += int(res.output_tokens or 0)
        emit(
            {
                "kind": "review",
                "text": text,
                "reply": ans.content,
                "model": model,
                "effort": effort,
                "input_tokens": getattr(res, "input_tokens", None),
                "output_tokens": getattr(res, "output_tokens", None),
                "ms": getattr(res, "duration_ms", None),
            }
        )
        return str(ans.content) if ans.content is not None else None

    def call(msgs: list[dict[str, Any]], stage: str) -> str | None:
        """s30: one call of a merge option's stage (audit, extract, verify, write) in its own
        context, no tools; the last message and the reply are saved as a `stage` event."""
        from tau2_loop.llm import core

        model, effort = agent.stages or (agent.model, agent.effort)
        ans = ask(msgs, None, core.resolve_model(model), effort, agent.tool_mode)
        res = getattr(ans, "result", None)
        if res is not None:
            meta.input_tokens += int(res.input_tokens or 0)
            meta.output_tokens += int(res.output_tokens or 0)
        emit(
            {
                "kind": "stage",
                "stage": stage,
                "messages": len(msgs),
                "text": msgs[-1]["content"],
                "reply": ans.content,
                "model": model,
                "effort": effort,
                "input_tokens": getattr(res, "input_tokens", None),
                "output_tokens": getattr(res, "output_tokens", None),
                "ms": getattr(res, "duration_ms", None),
            }
        )
        return str(ans.content) if ans.content is not None else None

    wf: dict[str, Any] | None = None
    try:
        if replay is not None:
            wf = replay.draft
            emit(
                {
                    "kind": "note",
                    "text": f"The research is {replay.source}'s, replayed from its events "
                    f"({len(replay.messages)} messages); its draft and decision are reused, and "
                    "only the merge runs.",
                }
            )
        else:
            wf = turns(
                agent.max_steps,
                "Your research steps are used up. Write the workflow JSON now from what you have.",
            )
        if wf is not None and agent.merge and meta.source in ("question", "remerge"):
            if replay is not None:
                wf = _merge_concurrent(
                    meta,
                    agent,
                    wf,
                    messages,
                    turns,
                    say,
                    emit,
                    policy,
                    review,
                    replay.decision,
                    call,
                )
            else:
                if agent.merge == "concurrent":
                    wf = _merge_concurrent(
                        meta, agent, wf, messages, turns, say, emit, policy, review, None, call
                    )
                else:
                    wf = _merge(meta, agent, wf, messages, turns, say, emit, policy, review)
        meta.status = "done" if wf is not None else "failed"
        if wf is None:
            meta.error = "no workflow JSON in the final reply"
    except Exception as e:  # noqa: BLE001 - a failed session is recorded, not raised into the server
        meta.status = "failed"
        meta.error = f"{type(e).__name__}: {e}"[:500]
        emit({"kind": "error", "text": meta.error, "trace": traceback.format_exc()[-2000:]})
    meta.duration_ms = int((time.time() - started) * 1000)
    if wf is not None:
        (d / "output.json").write_text(json.dumps(wf, indent=1, ensure_ascii=False) + "\n")
    try:
        sc = score(meta.domain, meta.task_id, wf, events, policy)
        (d / "score.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False) + "\n")
    except Exception as e:  # noqa: BLE001
        emit({"kind": "error", "text": f"scoring failed: {type(e).__name__}: {e}"[:500]})
    emit({"kind": "done", "status": meta.status})
    _write_meta(d, meta)
    return meta


# ── s20: the merge (r2); s21: the merge under locks (r3) ──

MERGE_CANDIDATES = 4  # library jobs offered for each drafted job
MERGE_WRITE_NOW = "Your merge steps are used up. Write the merged workflow JSON now."
DECIDE_AGAIN = (
    'Reply with one JSON object only: {"decision": {"<drafted job>": ["<library job>", ...]}}.'
)


def merge_candidates(
    draft: dict[str, Any], lib: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """For each drafted job, the library jobs it may be: its own name or an alias first, then the
    lookup's best matches by name (no model, so a rerun offers the same)."""
    from tau2_loop.workflows import library

    out: dict[str, list[dict[str, Any]]] = {}
    for job in draft.get("jobs") or []:
        if not isinstance(job, dict) or not job.get("job"):
            continue
        name = str(job["job"])
        picks: list[dict[str, Any]] = []
        same = library.by_name(lib, name)
        if same is not None:
            picks.append(same)
        when = job.get("when") or {}
        request = (
            f"{name.replace('_', ' ')} {when.get('quote', '') if isinstance(when, dict) else ''}"
        )
        for _, named, e in library.rank(request, lib):
            if len(picks) >= MERGE_CANDIDATES:
                break
            if named and e not in picks:
                picks.append(e)
        out[name] = picks
    return out


def _versions_text(entries: list[dict[str, Any]], changed: dict[str, str] | None = None) -> str:
    changed = changed or {}
    return "\n\n".join(
        f"### {e['job']} (current version, session {e['session']})"
        + (f" · {changed[e['job']]}" if e["job"] in changed else "")
        + "\n"
        + json.dumps(e["workflow"], ensure_ascii=False, indent=1)
        for e in entries
    )


def _unique(cands: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for es in cands.values():
        for e in es:
            seen.setdefault(e["job"], e)
    return list(seen.values())


def _tool_docs_text(
    agent: RagAgent, cands: dict[str, list[dict[str, Any]]], draft: dict[str, Any] | None
) -> str:
    """r13: every tool the drafted jobs and the library candidates call, described in full; r17:
    every tool they name."""
    if not agent.tool_docs:
        return ""
    from tau2_loop.workflows import toolspec

    jobs = [*((draft or {}).get("jobs") or []), *(e["workflow"] for e in _unique(cands))]
    text = toolspec.descriptions(
        agent.domain, jobs, every_named=agent.describe_named, errors=agent.tool_errors
    )
    which = "call or name" if agent.describe_named else "call"
    return (
        f"\n\n## The tools these jobs {which}, as the answering agent reads them when it unlocks "
        "them\n\n"
        "Each tool's own description is the authority on what its parameters and values mean "
        "(reading rule 10).\n\n" + text
        if text
        else ""
    )


def _read_text(
    agent: RagAgent, messages: list[dict[str, Any]], draft: dict[str, Any] | None
) -> str:
    """s23 (r18, D15): the documents the session printed in full that its draft cites nowhere."""
    if "D15" not in agent.rubric:
        return ""
    from tau2_loop.workflows import provenance

    docs = provenance.unaccounted(agent.domain, messages, draft)
    if not docs:
        return ""
    return (
        "\n\n## Documents you printed in full that your draft cites nowhere (D15)\n\n"
        + provenance.read_list(agent.domain, docs)
        + "\n\nFor each, quote the passage in the step it decides, or list it in `set_aside` "
        '({"doc": "<id>", "why": "<why it does not bear on these jobs>"}). Two documents that '
        "give one figure or requirement are both cited (reading rule 18)."
    )


def merge_message(
    agent: RagAgent, cands: dict[str, list[dict[str, Any]]], draft: dict[str, Any] | None = None
) -> str:
    """The harness's message after the draft: the library's candidates in full, and (r13) the
    descriptions of the tools the jobs call."""
    pairs = "\n".join(
        f"- {name}: {', '.join(e['job'] for e in es) or 'no library job matches'}"
        for name, es in cands.items()
    )
    return agent.merge_prompt.format(
        pairs=pairs, current=_versions_text(_unique(cands)) or "(none)"
    ) + _tool_docs_text(agent, cands, draft)


def decide_message(
    agent: RagAgent, cands: dict[str, list[dict[str, Any]]], draft: dict[str, Any] | None = None
) -> str:
    """s21: before it merges, the candidates in full and the question which it will write."""
    pairs = "\n".join(
        f"- {name}: {', '.join(e['job'] for e in es) or 'no library job matches'}"
        for name, es in cands.items()
    )
    return agent.decide_prompt.format(
        pairs=pairs, current=_versions_text(_unique(cands)) or "(none)"
    ) + _tool_docs_text(agent, cands, draft)


def parse_decision(text: str | None, drafted: list[str]) -> dict[str, list[str]] | None:
    """The decide turn's reply, {"decision": {drafted job: [library jobs]}}, for every drafted
    job (one it left out is new); None when the reply holds no such object."""
    obj = _json_object(
        text, lambda v: isinstance(v.get("decision"), dict) or any(k in v for k in drafted)
    )
    if obj is None:
        return None
    dec = obj["decision"] if isinstance(obj.get("decision"), dict) else obj
    out: dict[str, list[str]] = {}
    for name in drafted:
        v = dec.get(name, [])
        v = [v] if isinstance(v, str) else v if isinstance(v, list) else []
        out[name] = [str(n) for n in v if isinstance(n, str) and n.strip()]
    return out


def _advice(
    domain: str,
    job: dict[str, Any],
    c: dict[str, Any],
    merged: dict[str, Any],
    policy: str,
    gate_checks: tuple[str, ...],
    review: Callable[[str], str | None] | None,
    review_prompt: str,
) -> tuple[list[str], dict[str, Any]]:
    """s23 (r18): what goes back to the writer once without blocking: the reviewer's "no" on a
    clause code picked (D14, every step of the job, kept ones included), and an idea a called
    tool's description names that is a document's title (D16). Returns the lines and the record."""
    from tau2_loop.workflows import provenance, toolspec

    lines: list[str] = []
    record: dict[str, Any] = {}
    if "D16" in gate_checks:
        aside = frozenset(provenance.set_aside(domain, merged))
        c["D16"] = toolspec.named_titles(domain, job, aside)
        lines += toolspec.reasons({"D16": c["D16"]}, ("D16",))
    if "D14" in gate_checks and review is not None and review_prompt:
        picks = provenance.flagged(domain, job, policy)
        verdicts = (
            provenance.parse_review(
                review(provenance.review_request(domain, review_prompt, job, picks, policy))
            )
            if picks
            else []
        )
        record = {"picked": picks, "verdicts": verdicts}
        c["D14"] = {
            "picked": len(picks),
            "no": sum(v["verdict"] == "no" for v in verdicts),
            "partly": sum(v["verdict"] == "partly" for v in verdicts),
        }
        lines += provenance.review_reasons(verdicts)
    if "D18" in gate_checks:  # s25 (r19): a named period says where it starts and ends
        c["D18"] = provenance.unbounded_periods(job)
        lines += [
            f"step {sid}: it names a period or window without saying where it starts and ends; "
            "say it from the records the step reads (reading rule 5)"
            for sid in c["D18"]
        ]
    if "D19" in gate_checks:  # s30: a step id the job points to is one of its steps
        from tau2_loop.workflows import rubric

        lines += rubric.structure_reasons(c, ("D19",))
    return lines, record


def _assess(
    domain: str,
    merged: dict[str, Any],
    lib: list[dict[str, Any]],
    policy: str,
    held: set[str] | None = None,
    gate_checks: tuple[str, ...] = (),
    review: Callable[[str], str | None] | None = None,
    review_prompt: str = "",
    draft_base: bool = False,
) -> list[dict[str, Any]]:
    """Each merged job with the library versions it merged and its checks; `why` lists what
    stops it being taken. Under locks (`held`), a job may write only a name it holds and merge
    only library jobs it holds. A job written as edits (r4) is first applied to the version of the
    one job it names in `into`; edits that do not apply are sent back, then not taken. A version
    that gates on more checks (`gate_checks`, r5: D9, D10) sends a job back for them too; a new
    job that still fails them is taken as written, a merge keeps the previous version. A job's
    `kept` (r7: the steps and rules it re-read and kept as they were) is checked, not stored."""
    from tau2_loop.workflows import edits, library, rubric, toolspec

    rows: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}
    for job in merged.get("jobs") or []:
        if not isinstance(job, dict) or not job.get("job"):
            continue
        written = None
        job = dict(job)
        kept = job.pop("kept", None)
        from tau2_loop.workflows import merge_opts

        job, opt = merge_opts.take_fields(job)  # s30: an option's accounts and step marks
        if edits.is_edits(job):
            name = edits.base_name(job)
            base = library.by_name(lib, name) if name else None
            written = {"edits": len(job["edits"]), "chars": edits.size(job)}
            try:
                if base is None:
                    raise edits.EditError(
                        "edits apply to the newest version of one library job: name it, and only "
                        "it, in `into`; a job that merges several, or a new job, is written in full"
                    )
                job = edits.apply(job, base["workflow"])
            except edits.EditError as err:
                rows.append(
                    {
                        "job": job,
                        "previous": [base] if base else [],
                        "checks": {},
                        "why": [f"edits: {err}"],
                        "blocked": True,
                        "written": written,
                    }
                )
                continue
        names = [str(job["job"]), *library.merged_names(job)]
        prev: list[dict[str, Any]] = []
        for n in names:
            e = library.by_name(lib, n)
            if e is not None and e not in prev:
                prev.append(e)
        c = rubric.check(domain, job, [e["workflow"] for e in prev], policy, kept)
        why = (
            rubric.gate(c, gate_checks, draft_base=draft_base)
            if prev
            else [
                *(
                    [
                        "quotes not found in the documents they cite: "
                        + " | ".join(c["D1"]["new_unfound"][:4])
                    ]
                    if c["D1"]["new_unfound"]
                    else []
                ),
                *toolspec.reasons(c, gate_checks),
            ]
        )
        blocked = []
        if held is not None:
            if str(job["job"]) not in held:
                blocked.append(
                    f"writes {job['job']}, a name it did not name, so the library is not holding "
                    f"it; keep a name it holds: {', '.join(sorted(held)) or 'nothing'}"
                )
            outside = sorted({e["job"] for e in prev} - held)
            if outside:
                blocked.append(
                    f"merges {', '.join(outside)}, which it did not name, so the library is not "
                    f"holding it; it holds {', '.join(sorted(held)) or 'nothing'}"
                )
            twice = [e["job"] for e in prev if e["job"] in claimed]
            if twice:
                blocked.append(
                    f"{', '.join(twice)} is merged by {claimed[twice[0]]} in this reply too: "
                    "write one version of it"
                )
            for e in prev:
                claimed.setdefault(e["job"], str(job["job"]))
        advise, reviewed = _advice(
            domain, job, c, merged, policy, gate_checks, review, review_prompt
        )
        rows.append(
            {
                "job": job,
                "previous": prev,
                "checks": c,
                "why": [*why, *blocked],
                "advise": advise,
                "review": reviewed,
                "blocked": bool(blocked),
                "written": written,
                "kept": kept,
                "opt": opt,
            }
        )
    return rows


def _send_back(
    agent: RagAgent,
    rows: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
    extra: list[str] | None = None,
) -> dict[str, Any] | None:
    failing = [r for r in rows if r["why"] or r.get("advise")]
    lines = "\n".join(
        f"- {r['job']['job']}: {'; '.join([*r['why'], *(r.get('advise') or [])])}" for r in failing
    )
    if extra:
        lines = "\n".join([*([lines] if lines else []), *(f"- {x}" for x in extra)])
    emit({"kind": "note", "text": f"The merge failed its checks; sent back once.\n{lines}"})
    say(
        "The harness checked your merge and these jobs fail:\n"
        f"{lines}\n\n"
        + (
            "A step of the library's version that stands on its documents stays unless a step "
            "of yours now handles its case or a passage decides its case otherwise: name which in "
            "`accounted`. After this reply, one still unaccounted is carried into the job as it "
            "was, a quote that is still not found loses its place, and "
            + (
                "an auditor reads the job against its whole documents and rewrites what they do "
                "not support."
                if agent.post_audit
                else "a step the reviewer still finds unsupported is removed."
            )
            if agent.merge_account or agent.merge_mode == "audit"
            else "The draft is the base: a step of the versions you merged stays only when it covers a "
            "case no step of the draft covers, its quote is found in the document it cites, and "
            "its clauses follow from that quote. You need not account for what you leave out: "
            "code logs it. After this reply, a quote that is still not found loses its place, and "
            + (
                "a step the reviewer still finds unsupported is removed."
                if agent.review_binding
                else "the job is taken."
            )
            if agent.merge_base == "draft"
            else "A merged job accounts for every step and rule of the versions it merged: kept "
            "(named in `kept`), rewritten or removed, each with the passage that decides it, "
            "and every quote is copied from the document it cites."
            if "D12" in agent.rubric
            else "A merged job keeps every step, tool and quote of the versions it "
            "merged unless its changelog names the change (removed or changed, with why), and "
            "every quote is copied from the document it cites."
        )
        + (
            " A clause the reviewer finds unsupported is quoted from the passage that says it, "
            'rewritten to what the passage says, or given a basis entry with source "none" and '
            "made a question for the customer; a document is cited or set aside."
            if "D14" in agent.rubric or "D15" in agent.rubric
            else ""
        )
        + " Reply with the whole JSON again, fixed."
        + (
            " A job written as edits is applied again to the library's newest version, not to "
            "your last reply: write all of its edits."
            if agent.merge_write == "edits"
            else ""
        )
    )
    again: dict[str, Any] | None = turns(max(agent.merge_steps // 2, 4), MERGE_WRITE_NOW)
    return again


def _take(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """What the library takes: a merge that still fails keeps the previous version; a new job is
    taken as written unless it was blocked (s21: outside its lock). Returns taken, the merge
    records and the rejected."""
    from tau2_loop.workflows import library

    taken: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for r in rows:
        job = dict(r["job"])
        prev_names = [e["job"] for e in r["previous"]]
        keep = not r.get("blocked") and not (r["why"] and r["previous"])
        aliases = [n for n in [*library.merged_names(job), *prev_names] if n != job["job"]]
        records.append(
            {
                "job": job["job"],
                "into": prev_names,
                "previous_sessions": [e["session"] for e in r["previous"]],
                "changelog": job.get("changelog") or [],
                "checks": r["checks"],
                "failed": r["why"],
                **({"advice": r["advise"]} if r.get("advise") else {}),
                **({"review": r["review"]} if r.get("review") else {}),
                "taken": keep,
                **({"written": r["written"]} if r.get("written") else {}),
                **({"kept": r["kept"]} if r.get("kept") else {}),
                **({"settled": r["settled"]} if r.get("settled") else {}),
                **({"option": r["opt"]} if r.get("opt") else {}),  # s30: accounts, marks
                **({"audit": r["audit"]} if r.get("audit") else {}),
                **({"ledger": r["ledger"]} if r.get("ledger") else {}),
                **({"post_audit": r["post_audit"]} if r.get("post_audit") else {}),
            }
        )
        job.pop("changelog", None)
        job.pop("into", None)
        job.pop("edits", None)
        if aliases:
            job["aliases"] = sorted(set(aliases))
        (taken if keep else rejected).append(job)
    return taken, records, rejected


CODE_LOGGED = "logged by code (r19's draft-first merge)"


def _unquote(domain: str, job: dict[str, Any], policy: str) -> list[dict[str, str]]:
    """s25 (r19, M3): each quote the job still carries that its document does not hold loses its
    place. The clause stays, without the quote: a basis entry becomes source "none", a step,
    rule or other item is marked `unquoted`. Returns the changelog lines code adds for them."""
    rows = quote_check(domain, {"jobs": [job]}, policy)["rows"]
    bad = {_squash(r["quote"].strip(" .…\"'")) for r in rows if not r["found"]}
    log: list[dict[str, str]] = []

    def walk(o: Any, in_log: bool) -> None:
        if isinstance(o, dict):
            q = o.get("quote")
            if isinstance(q, str) and q and _squash(q.strip(" .…\"'")) in bad:
                doc = str(o.pop("doc", "") or "")
                o.pop("quote")
                if not in_log:
                    note = f"its quote was not found in {doc or 'the document it cited'}"
                    if "clause" in o:
                        o["source"], o["why"] = "none", note
                    else:
                        o["unquoted"] = note
                    log.append(
                        {
                            "change": "changed",
                            "what": f"quote not found: {q[:120]}",
                            "why": f"{note}; {CODE_LOGGED}",
                        }
                    )
            for k, v in o.items():
                walk(v, in_log or k == "changelog")
        elif isinstance(o, list):
            for v in o:
                walk(v, in_log)

    walk(job, False)
    return log


def _settle(
    domain: str,
    rows: list[dict[str, Any]],
    policy: str,
    gate_checks: tuple[str, ...],
    binding: bool,
    protect: dict[str, dict[str, dict[str, Any]]] | None = None,
    audited: bool = False,
) -> None:
    """s25 (r19): a draft-based merge's last word, after its one send-back. A quote still not
    found loses its place, not the merge (M3); a step the reviewer still finds unsupported is
    removed (R1, `binding`); what the job leaves out of the versions it merged is written into its
    changelog by code (M3). The checks are then run again on what is left.

    s30: `protect` (options A and B) holds, per library job, the old steps that leave only with an
    account; one the merge still leaves out unaccounted is carried over as it was. Under the audit
    (B, `audited`) the reviewer's "no" does not remove a step the auditor found valid and the merge
    kept as the audit left it: the auditor read it against the whole documents."""
    from tau2_loop.workflows import merge_opts, rubric, toolspec

    for r in rows:
        if r.get("blocked"):
            continue
        job = r["job"]
        guard: dict[str, dict[str, Any]] = {}
        for e in r["previous"]:
            guard |= (protect or {}).get(str(e.get("job"))) or {}
        log = _unquote(domain, job, policy)
        reviewed_out: set[str] = set()
        if binding:
            no = {
                v["step"]
                for v in (r.get("review") or {}).get("verdicts") or []
                if v.get("verdict") == "no"
            }
            if audited:
                no = {
                    sid
                    for sid in no
                    if not any(
                        isinstance(st, dict) and st.get("id") == sid and st == guard.get(sid)
                        for st in job.get("steps") or []
                    )
                }
            gone = [
                st for st in job.get("steps") or [] if isinstance(st, dict) and st.get("id") in no
            ]
            reviewed_out = {str(st.get("id")) for st in gone}
            if gone:
                job["steps"] = [st for st in job.get("steps") or [] if st not in gone]
                for st in gone:
                    why = next(
                        (
                            v.get("why") or v.get("clause")
                            for v in r["review"]["verdicts"]
                            if v["step"] == st["id"]
                        ),
                        "",
                    )
                    log.append(
                        {
                            "change": "removed",
                            "what": f"step {st['id']}",
                            "why": f"the reviewer still finds it unsupported after the send-back: {str(why)[:200]}; {CODE_LOGGED}",
                        }
                    )
        prev = [e["workflow"] for e in r["previous"]]
        if guard and prev:  # s30 (A, B): no protected step leaves without an account
            left = {k: v for k, v in guard.items() if k not in reviewed_out}
            missing = merge_opts.unaccounted(
                domain, job, left, merge_opts.accounts(r.get("opt")), policy
            )
            if missing:
                order = [
                    str(s.get("id"))
                    for p in prev
                    for s in p.get("steps") or []
                    if isinstance(s, dict)
                ]
                log += merge_opts.carry(job, order, guard, missing)
        if prev:
            for kind, items in rubric.lost(job, prev).items():
                for x in items:
                    log.append(
                        {
                            "change": "removed",
                            "what": f"{kind[:-1]} {x}",
                            "why": f"not carried into the draft-based version; {CODE_LOGGED}",
                        }
                    )
        if not log:
            continue
        job["changelog"] = [*(job.get("changelog") or []), *log]
        c = rubric.check(domain, job, prev, policy, r.get("kept"))
        r["checks"] = {**r["checks"], **c}
        r["why"] = (
            rubric.gate(c, gate_checks, draft_base=True)
            if prev
            else toolspec.reasons(c, gate_checks)
        )
        r["settled"] = log


def _unaccounted_lines(
    agent: RagAgent, messages: list[dict[str, Any]], merged: dict[str, Any] | None
) -> list[str]:
    """s23 (r18, D15): the send-back line for documents printed in full, cited by no job and not
    set aside; empty when every one is accounted for or the version does not check it."""
    if "D15" not in agent.rubric or merged is None:
        return []
    from tau2_loop.workflows import provenance

    docs = provenance.unaccounted(agent.domain, messages, merged)
    if not docs:
        return []
    return [
        "you printed these documents in full and no job cites them or sets them aside: "
        + ", ".join(docs[:20])
        + (" …" if len(docs) > 20 else "")
        + ". Quote each in the step it decides, or list it in `set_aside` with why it does not "
        "bear on these jobs"
    ]


def _accounting(
    agent: RagAgent, messages: list[dict[str, Any]], merged: dict[str, Any] | None
) -> dict[str, Any]:
    """s23 (r18): the session's `set_aside` and what is still unaccounted, kept in its output."""
    if "D15" not in agent.rubric and "D16" not in agent.rubric:
        return {}
    from tau2_loop.workflows import provenance

    return {
        "set_aside": [
            {"doc": d, "why": w} for d, w in provenance.set_aside(agent.domain, merged).items()
        ],
        "unaccounted": provenance.unaccounted(agent.domain, messages, merged),
    }


def _merge(
    meta: SessionMeta,
    agent: RagAgent,
    draft: dict[str, Any],
    messages: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
    policy: str,
    review: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Merge the drafted jobs into the library's current versions, check each merge without a
    model, send failures back once, and take only what passes (s20). A merged job that still
    fails keeps the library's previous version; a new job is taken as written."""
    from tau2_loop.workflows import library

    lib = library.entries(meta.domain, agent.name)  # as the last session left it
    cands = merge_candidates(draft, lib)
    emit({"kind": "merge", "candidates": {k: [e["job"] for e in v] for k, v in cands.items()}})
    say(merge_message(agent, cands, draft) + _read_text(agent, messages, draft))
    merged = turns(agent.merge_steps, MERGE_WRITE_NOW)
    rev = (review, agent.review_prompt)
    rows = (
        _assess(meta.domain, merged, lib, policy, None, agent.rubric, *rev)
        if merged is not None
        else []
    )
    extra = _unaccounted_lines(agent, messages, merged)
    if merged is not None and (any(r["why"] or r.get("advise") for r in rows) or extra):
        again = _send_back(agent, rows, turns, say, emit, extra)
        if again is not None:
            merged, rows = again, _assess(meta.domain, again, lib, policy, None, agent.rubric, *rev)
    if merged is None:  # no merge written: keep only the draft's jobs no library job matches
        merged = {
            **draft,
            "jobs": [j for j in draft.get("jobs") or [] if not cands.get(str(j.get("job")))],
        }
        rows = _assess(meta.domain, merged, lib, policy, gate_checks=agent.rubric)
    taken, records, rejected = _take(rows)
    emit(
        {
            "kind": "merged",
            "taken": [j["job"] for j in taken],
            "rejected": [j["job"] for j in rejected],
        }
    )
    return {
        "jobs": taken,
        "documents": merged.get("documents") or draft.get("documents") or [],
        "open_questions": merged.get("open_questions") or draft.get("open_questions") or [],
        **_accounting(agent, messages, merged),
        "merge": records,
        "rejected": rejected,
        "draft": draft.get("jobs") or [],
    }


def _holds(decision: dict[str, list[str]], lib: list[dict[str, Any]]) -> set[str]:
    """The jobs a decision writes, as the library names them now: every library job it named
    (through an alias if it was renamed), a library job with the drafted job's own name, and the
    drafted job's own name, which the merge may keep while it retires the jobs it merged (a new
    name is locked like any other, so two workers never create it at once)."""
    from tau2_loop.workflows import library

    out: set[str] = set()
    for drafted, names in decision.items():
        out |= {e["job"] for e in (library.by_name(lib, n) for n in [*names, drafted]) if e}
        out.add(drafted)
    return out


def _ledger_merge(
    meta: SessionMeta,
    agent: RagAgent,
    draft: dict[str, Any],
    decision: dict[str, list[str]],
    mine: list[dict[str, Any]],
    policy: str,
    call: Callable[[list[dict[str, Any]], str], str | None],
    emit: Callable[[dict[str, Any]], None],
    ledgers: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """s30 (option C): each drafted job merged into the library jobs it names by the claim
    ledger, in fresh contexts; a drafted job that names none is new and taken as drafted."""
    from tau2_loop.workflows import library, merge_opts

    prompts = dict(agent.stage_prompts)
    schema = merge_opts.schema_text(agent.prompt)
    by_name = {str(j.get("job")): j for j in draft.get("jobs") or [] if isinstance(j, dict)}
    jobs = []
    for drafted, names in decision.items():
        d = by_name.get(drafted)
        olds: list[dict[str, Any]] = []
        for n in names:
            e = library.by_name(mine, n)
            if e is not None and e not in olds:
                olds.append(e)
        if not olds:
            if d is not None:
                jobs.append(d)
            continue
        name = olds[0]["job"]
        job, ledger = merge_opts.ledger_job(
            meta.domain, prompts, name, [e["workflow"] for e in olds], d, policy, schema, call, emit
        )
        ledgers[name] = ledger
        if job is not None:
            jobs.append({**job, "into": [e["job"] for e in olds]})
    if not jobs:
        return None
    return {**{k: v for k, v in draft.items() if k != "jobs"}, "jobs": jobs}


def _ledger_send_back(
    rows: list[dict[str, Any]],
    ledgers: dict[str, dict[str, Any]],
    merged: dict[str, Any],
    call: Callable[[list[dict[str, Any]], str], str | None],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    """s30 (C): the one send-back goes to the writer's own conversation, job by job."""
    jobs = list(merged.get("jobs") or [])
    changed = False
    for r in rows:
        names = [e["job"] for e in r["previous"]]
        name = next((n for n in names if n in ledgers), None)
        lines = [*r["why"], *(r.get("advise") or [])]
        if name is None or not lines:
            continue
        emit(
            {
                "kind": "note",
                "text": f"The merge failed its checks; sent back once.\n- {name}: "
                + "; ".join(lines),
            }
        )
        conv = ledgers[name]["_writer"]
        conv.append(
            {
                "role": "user",
                "content": "The harness checked the job you wrote and it fails:\n"
                + "\n".join(f"- {x}" for x in lines)
                + "\n\nEvery claim that survived is carried by a step or rule that lists its id "
                "in `claims`, and no rejected claim is. A clause the reviewer finds unsupported is "
                "quoted from the passage that says it, rewritten to what the passage says, or made "
                "a question for the customer. Reply with the whole JSON again, fixed.",
            }
        )
        reply = call(conv, "write")
        conv.append({"role": "assistant", "content": reply or ""})
        job = next(
            (j for j in (parse_workflow(reply) or {}).get("jobs") or [] if isinstance(j, dict)),
            None,
        )
        if job is None:
            continue
        job = {**job, "job": name, "into": names}
        jobs = [job if str(j.get("job")) == name else j for j in jobs]
        changed = True
    return {**merged, "jobs": jobs} if changed else None


def _merge_concurrent(
    meta: SessionMeta,
    agent: RagAgent,
    draft: dict[str, Any],
    messages: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
    policy: str,
    review: Callable[[str], str | None] | None = None,
    decision: dict[str, list[str]] | None = None,
    call: Callable[[list[dict[str, Any]], str], str | None] | None = None,
) -> dict[str, Any]:
    """s21 (r3): the merge while other workers merge too. Name the jobs it will write, lock them
    all at once (waiting, and logging who holds them, if another worker does), read their newest
    versions, merge and check as r2 does, and commit under the version check. Only the jobs it
    holds may be merged; a merge of another is sent back once, then not taken. A re-merge (s25)
    brings the source session's `decision`, so no decide turn runs. s30: `call` runs a merge
    option's stages in their own contexts (`merge_opts`)."""
    from tau2_loop.workflows import library, merge_opts, store

    st = library.store_for(meta.domain, agent.name)
    snap = st.entries()
    cands = merge_candidates(draft, snap)
    emit({"kind": "merge", "candidates": {k: [e["job"] for e in v] for k, v in cands.items()}})
    drafted = list(cands)
    replayed = decision is not None
    if decision is None:
        decision = turns(
            0,
            decide_message(agent, cands, draft),
            parse=lambda t: parse_decision(t, drafted),
            again=DECIDE_AGAIN,
        )
    else:
        decision = {k: list(decision.get(k) or []) for k in drafted}
    if decision is None:  # no usable answer: hold every candidate, which r2 would have shown it
        decision = {k: [e["job"] for e in v] for k, v in cands.items()}
        emit({"kind": "note", "text": "No usable decision; the library holds every candidate."})
    emit({"kind": "decided", "decision": decision, **({"replayed": True} if replayed else {})})

    want = _holds(decision, snap)
    asked = time.time()

    def waiting(way: list[dict[str, Any]]) -> None:
        emit(
            {
                "kind": "wait",
                "jobs": sorted(want),
                "in_the_way": [
                    {
                        "task": w.get("task"),
                        "session": w["holder"],
                        "state": w["state"],
                        "jobs": w["jobs"],
                    }
                    for w in way
                ],
            }
        )

    lock = st.lock(sorted(want), meta.id, meta.task_id, on_wait=waiting)
    for _ in range(3):  # a job renamed or created while it waited joins the lock
        more = _holds(decision, st.entries()) - set(lock.jobs)
        if not more:
            break
        st.release(lock)
        want |= more
        emit(
            {
                "kind": "note",
                "text": f"The library changed while it waited; it also holds {', '.join(sorted(more))}.",
            }
        )
        lock = st.lock(sorted(want), meta.id, meta.task_id, on_wait=waiting)
    waited_ms = int((time.time() - asked) * 1000)
    committed: dict[str, int] = {}
    error = None
    with st.holding(lock) as hold:
        heads = st.entries()
        mine = [e for e in heads if e["job"] in set(lock.jobs)]
        before = {e["job"]: e["version"] for e in snap}
        changed = {
            e[
                "job"
            ]: f"changed since you read it above: version {e['version']}, from {e.get('task_id')}"
            for e in mine
            if before.get(e["job"]) != e["version"]
        }
        emit(
            {
                "kind": "locked",
                "token": lock.token,
                "jobs": list(lock.jobs),
                "versions": {e["job"]: e["version"] for e in mine},
                "changed": sorted(changed),
                "waited_ms": waited_ms,
            }
        )
        pairs = "\n".join(
            f"- {name}: {', '.join(n for n in names) or 'a new job'}"
            for name, names in decision.items()
        )
        held = set(lock.jobs)
        base = agent.merge_base == "draft"
        audits: dict[str, dict[str, Any]] = {}
        ledgers: dict[str, dict[str, Any]] = {}
        if agent.merge_mode == "ledger" and call is not None:  # s30 (C): no merge turn
            merged = _ledger_merge(meta, agent, draft, decision, mine, policy, call, emit, ledgers)
        else:
            shown = mine
            if agent.merge_mode == "audit" and call is not None:  # s30 (B): audited first
                audits = merge_opts.audit_heads(
                    meta.domain, dict(agent.stage_prompts)["audit"], mine, policy, call, emit
                )
                shown = [
                    {**e, "workflow": audits[e["job"]]["shown"]} if e["job"] in audits else e
                    for e in mine
                ]
            say(
                agent.merge_prompt.format(
                    pairs=pairs, current=_versions_text(shown, changed) or "(none)"
                )
                + merge_opts.removed_text(audits)
                + _tool_docs_text(agent, {"held": mine}, draft)
                + _read_text(agent, messages, draft)
            )
            merged = turns(agent.merge_steps, MERGE_WRITE_NOW)
        protect = (
            {j: a["protected"] for j, a in audits.items()}
            if audits
            else {
                e["job"]: merge_opts.sourced_steps(meta.domain, e["workflow"], policy) for e in mine
            }
            if agent.merge_account
            else None
        )

        def assess(m: dict[str, Any], **kw: Any) -> list[dict[str, Any]]:
            rs = _assess(meta.domain, m, heads, policy, held, agent.rubric, draft_base=base, **kw)
            for r in rs:  # s30: an option's own lines join the reviewer's in the one send-back
                r["advise"] = [
                    *(r.get("advise") or []),
                    *merge_opts.option_lines(meta.domain, r, protect, ledgers, policy),
                ]
            return rs

        rows = (
            assess(merged, review=review, review_prompt=agent.review_prompt)
            if merged is not None
            else []
        )
        extra = [] if ledgers else _unaccounted_lines(agent, messages, merged)
        if merged is not None and (any(r["why"] or r.get("advise") for r in rows) or extra):
            again = (
                _ledger_send_back(rows, ledgers, merged, call, emit)
                if ledgers and call is not None
                else _send_back(agent, rows, turns, say, emit, extra)
            )
            if again is not None:
                merged, rows = (
                    again,
                    assess(again, review=review, review_prompt=agent.review_prompt),
                )
        if merged is None:  # nothing written: only the draft's new jobs, and only those it holds
            merged = {
                **draft,
                "jobs": [
                    j
                    for j in draft.get("jobs") or []
                    if not cands.get(str(j.get("job"))) and j.get("job") in held
                ],
            }
            rows = _assess(meta.domain, merged, heads, policy, held, agent.rubric, draft_base=base)
        if agent.post_audit and call is not None:  # s30 (D): the auditor's last word
            merge_opts.post_audit(
                meta.domain,
                dict(agent.stage_prompts)["audit"],
                rows,
                policy,
                agent.rubric,
                call,
                emit,
            )
        if base:
            _settle(
                meta.domain, rows, policy, agent.rubric, agent.review_binding, protect, bool(audits)
            )
        for r in rows:  # s30: what the option did, kept in the merge record
            names = [e["job"] for e in r["previous"]]
            if audits:
                r["audit"] = {
                    j: {k: audits[j][k] for k in ("verdicts", "removed", "unaudited")}
                    for j in names
                    if j in audits
                }
            led = next((ledgers[j] for j in names if j in ledgers), None)
            if led is not None:
                r["ledger"] = {k: v for k, v in led.items() if not k.startswith("_")}
                r["job"]["changelog"] = [
                    *(r["job"].get("changelog") or []),
                    *merge_opts.ledger_changelog(led),
                ]
        taken, records, rejected = _take(rows)
        writes = [
            {
                "job": j["job"],
                "workflow": j,
                "reads": {e["job"]: e["version"] for e in r["previous"]},
                "session": meta.id,
                "task": meta.task_id,
            }
            for j, r in zip(
                taken,
                [r for r, rec in zip(rows, records, strict=True) if rec["taken"]],
                strict=True,
            )
        ]
        try:
            if hold.lost:
                raise store.LockLostError(f"lock {lock.token} expired while it merged")
            committed = st.commit(hold.lock, writes) if writes else {}
        except (
            store.LockLostError,
            store.CommitRefusedError,
        ) as e:  # nothing written; the library is as it was
            error = f"{type(e).__name__}: {e}"
            emit({"kind": "error", "text": f"commit refused, nothing written: {error}"})
            rejected, taken = [*rejected, *taken], []
            for rec in records:
                rec["taken"] = False
    emit({"kind": "committed", "token": lock.token, "versions": committed})
    emit(
        {
            "kind": "merged",
            "taken": [j["job"] for j in taken],
            "rejected": [j["job"] for j in rejected],
        }
    )
    return {
        "jobs": taken,
        "documents": merged.get("documents") or draft.get("documents") or [],
        "open_questions": merged.get("open_questions") or draft.get("open_questions") or [],
        **_accounting(agent, messages, merged),
        "merge": records,
        "rejected": rejected,
        "draft": draft.get("jobs") or [],
        "decision": decision,
        "lock": {
            "token": lock.token,
            "jobs": list(lock.jobs),
            "waited_ms": waited_ms,
            "read": {e["job"]: e["version"] for e in mine},
            "committed": committed,
            "error": error,
        },
    }


def _launch(
    domain: str,
    agent: RagAgent,
    session_tail: str,
    question: str,
    opening: str,
    model: str | None,
    effort: str | None,
    *,
    task_id: str = "",
    source: str = "question",
    conversation: str | None = None,
    split: str = "train",
    ask: Ask | None,
    run_tool: RunTool | None,
    env: Any,
    background: bool,
    on_start: Callable[[SessionMeta], None] | None = None,
    source_session: dict[str, Any] | None = None,
) -> SessionMeta:
    from tau2_loop.llm import core

    model = model or agent.model
    effort = effort or agent.effort
    if effort not in core.EFFORTS:
        raise RagAgentError(f"effort is one of {', '.join(core.EFFORTS)}")
    if core.resolve_model(model) not in core.MODELS.values():
        raise RagAgentError(f"model is one of {', '.join(core.MODELS)}")
    if ask is None:
        from tau2_loop.llm import require_live

        require_live()  # refuses the demo image and a key that would bill per token
    env = env if env is not None else environment(agent.retrieval)
    system, tools = system_prompt(agent, env)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = SessionMeta(
        id=f"{stamp}_{domain}_{agent.name}_{session_tail}",
        domain=domain,
        agent=agent.name,
        task_id=task_id,
        model=core.resolve_model(model),
        effort=effort,
        retrieval=agent.retrieval,
        started=stamp,
        source=source,
        conversation=conversation,
        split=split,
        drafts_of=source_session["meta"]["id"] if source_session else None,
    )
    replay = _replay(source_session, system, opening) if source_session else None
    d = session_dir(meta.id)
    d.mkdir(parents=True, exist_ok=False)
    (d / "system.md").write_text(system)
    _write_meta(d, meta)
    with _live_lock:
        _live[meta.id] = meta
    if on_start is not None:
        on_start(meta)
    args = (
        d,
        meta,
        agent,
        system,
        tools,
        question,
        ask or default_ask,
        run_tool or env_runner(env),
        str(getattr(env, "policy", "") or ""),
        opening,
        replay,
    )
    if not background:
        return run_session(*args)
    threading.Thread(target=run_session, args=args, daemon=True, name=f"rag-{meta.id}").start()
    return meta


def start(
    domain: str,
    task_id: str,
    agent_name: str = "r1",
    model: str | None = None,
    effort: str | None = None,
    *,
    ask: Ask | None = None,
    run_tool: RunTool | None = None,
    env: Any = None,
    background: bool = True,
    split: str = "train",
) -> SessionMeta:
    """Start a session on a train question (a test one only when `split` says so, which only the
    build does); returns at once, the session running in a thread."""
    agent = load(domain, agent_name)
    question = train_question(domain, task_id, split)
    return _launch(
        domain,
        agent,
        task_id,
        question,
        SCRIPT_OPENING,
        model,
        effort,
        task_id=task_id,
        split=split,
        ask=ask,
        run_tool=run_tool,
        env=env,
        background=background,
    )


# ── s25 (r19, the person's call, 15 Oct 2026): merge again, on the same research ──

REPLAY_STOP = frozenset({"merge", "decided", "locked"})


def replay_messages(
    events: list[dict[str, Any]], system: str, opening: str = SCRIPT_OPENING
) -> list[dict[str, Any]]:
    """A session's research conversation, rebuilt from its events as the model saw it, up to the
    draft (the first merge event): every model turn, tool result and harness message. The system
    prompt is the re-merging version's, so its reading rules are the ones the merge reads."""
    start = next(e for e in events if e["kind"] == "start")
    out: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{opening}\n\n{start['question']}"},
    ]
    for e in events:
        if e["kind"] in REPLAY_STOP:
            break
        if e["kind"] == "model" and e.get("calls"):
            out.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in e["calls"]
                    ],
                }
            )
        elif e["kind"] == "model":
            out.append({"role": "assistant", "content": e.get("text") or ""})
        elif e["kind"] == "tool":
            out.append({"role": "tool", "tool_call_id": e["id"], "content": e["content"]})
        elif e["kind"] == "user":
            out.append({"role": "user", "content": e["text"]})
    return out


def _replay(src: dict[str, Any], system: str, opening: str) -> Replay:
    """A re-merge's start from a finished session (`session()`'s record): its research rebuilt, its
    draft as the research's last reply wrote it, and the decision it made."""
    events = src["events"]
    messages = replay_messages(events, system, opening)
    last = next(
        (m["content"] for m in reversed(messages) if m["role"] == "assistant" and m["content"]),
        None,
    )
    draft = parse_workflow(last)
    if draft is None:
        out = src.get("output") or {}
        draft = {"jobs": out.get("draft") or [], "documents": out.get("documents") or []}
    decision = (src.get("output") or {}).get("decision")
    if not isinstance(decision, dict):
        raise RagAgentError(f"{src['meta']['id']} recorded no decision to merge by")
    return Replay(src["meta"]["id"], messages, draft, decision)


def remerge_queue(domain: str, agent_name: str) -> list[str]:
    """The sessions a re-merging version (`drafts_from`) still has to merge again: every finished
    session of that version that reached its merge, in the order it took its lock (the order its
    merges reached the library), less those this version has merged already."""
    agent = load(domain, agent_name)
    if not agent.drafts_from:
        raise RagAgentError(f"{agent_name} merges no other version's drafts")
    done = {
        m.get("drafts_of")
        for m in sessions(domain)
        if m.get("agent") == agent_name and m.get("status") == "done" and not _refused(m["id"])
    }
    order: list[tuple[float, str]] = []
    for m in sessions(domain):
        if (
            m.get("agent") != agent.drafts_from
            or m.get("status") != "done"
            or m.get("source", "question") != "question"
        ):
            continue
        events = session(domain, m["id"])["events"]
        lock = next((e for e in events if e["kind"] == "locked"), None)
        if lock is None:
            continue  # it never reached a merge: nothing to merge again
        t0 = datetime.strptime(m["started"], "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC).timestamp()
        order.append((t0 + lock["t"] / 1000, m["id"]))
    return [sid for _, sid in sorted(order) if sid not in done]


def remerge(
    domain: str,
    agent_name: str,
    source_id: str,
    *,
    ask: Ask | None = None,
    run_tool: RunTool | None = None,
    env: Any = None,
    background: bool = False,
) -> SessionMeta:
    """Merge one session of `drafts_from` again with this version's merge: no research turn and no
    decide turn run, so the two versions' libraries differ only by how they merge."""
    agent = load(domain, agent_name)
    src = session(domain, source_id)
    sm = src["meta"]
    if sm.get("agent") != agent.drafts_from:
        raise RagAgentError(f"{source_id} is not a session of {agent.drafts_from}")
    return _launch(
        domain,
        agent,
        str(sm["task_id"]),
        src["events"][0].get("question") or "",
        SCRIPT_OPENING,
        None,  # this version's model and effort
        None,
        task_id=str(sm["task_id"]),
        source="remerge",
        split=str(sm.get("split") or "train"),
        ask=ask,
        run_tool=run_tool,
        env=env,
        background=background,
        source_session=src,
    )


def _refused(session_id: str) -> bool:
    """s21: the session's commit was refused (its lock lost, or a name it did not hold)."""
    p = RAG_RUNS_DIR / session_id / "output.json"
    if not p.is_file():
        return False
    lock = json.loads(p.read_text()).get("lock") or {}
    return bool(lock.get("error"))


def build_queue(
    domain: str, agent_name: str, only: list[str] | None = None, split: str = "train"
) -> list[str]:
    """The train questions a merging version still has to research (s20): each question once, so
    a stopped build resumes where it was. A test id is refused, unless `split` is test (r3, the
    person's call, 8 Oct 2026), and then only test ids are queued. In id order, or (s21) shuffled by
    the version's `order_seed`, the same order on every resume; a version with `queue:
    after_seed` also skips every question its seed's library already merged: the seed's, and its
    seed's while that one merged too (r3: r2's; r4: r3's and r2's, not r1's, which merged
    nothing). A concurrent session whose commit was refused wrote nothing, so its question is
    queued again."""
    if split not in ("train", "test"):
        raise RagAgentError(f"split is train or test, not {split}")
    ids = split_ids(domain, split)
    want = list(only) if only else list(ids)
    bad = [t for t in want if t not in set(ids)]
    if bad:
        raise RagAgentError(
            f"not train questions: {', '.join(bad)}: test is sealed"
            if split == "train"
            else f"not {split} questions: {', '.join(bad)}"
        )
    agent = load(domain, agent_name)
    by = {agent_name}
    seed = agent.seed if agent.queue == "after_seed" else None
    while seed and seed not in by:
        above = load(domain, seed)
        if not above.merge:
            break
        by.add(seed)
        seed = above.seed
    done = {
        m.get("task_id")
        for m in sessions(domain)
        if m.get("agent") in by
        and m.get("status") == "done"
        and m.get("source", "question") == "question"
        and not _refused(m["id"])
    }
    ordered = sorted(ids, key=lambda t: int(re.sub(r"\D", "", t) or 0))
    if agent.order == "shuffled":
        import random

        random.Random(agent.order_seed).shuffle(ordered)
    rank = {t: i for i, t in enumerate(ordered)}
    return sorted((t for t in dict.fromkeys(want) if t not in done), key=lambda t: rank[t])


def start_request(
    domain: str,
    request: str,
    agent_name: str = "r1",
    *,
    conversation: str | None = None,
    on_start: Callable[[SessionMeta], None] | None = None,
    ask: Ask | None = None,
    run_tool: RunTool | None = None,
    env: Any = None,
) -> SessionMeta:
    """Research an answering agent's request, mid-conversation, and return when it is written.

    It reads only the request (the agent's words), never a task's script or gold, and its
    workflow is not cached into the library: the conversation could be a test one."""
    import secrets

    agent = load(domain, agent_name)
    text = request.strip()[:4000]
    if not text:
        raise RagAgentError("an empty request")
    return _launch(
        domain,
        agent,
        f"req_{secrets.token_hex(3)}",
        text,
        REQUEST_OPENING,
        None,
        None,
        source="conversation",
        conversation=conversation,
        ask=ask,
        run_tool=run_tool,
        env=env,
        background=False,
        on_start=on_start,
    )


STALE_S = 900  # a session another process runs (`make rag-build`) that has written nothing for this long has stopped


def _read_meta(d: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((d / "run.json").read_text())
    with _live_lock:
        live = meta["id"] in _live
    if meta["status"] == "running" and not live:
        # another process may be running it (s20: the build); it is running while it still writes
        last = max(
            (p.stat().st_mtime for p in (d / "events.jsonl", d / "run.json") if p.is_file()),
            default=0,
        )
        if time.time() - last > STALE_S:
            meta["status"] = "interrupted"  # the server restarted under it
    return meta


def sessions(domain: str) -> list[dict[str, Any]]:
    if not RAG_RUNS_DIR.is_dir():
        return []
    out = []
    for d in sorted(RAG_RUNS_DIR.iterdir(), reverse=True):
        if (d / "run.json").is_file() and f"_{domain}_" in d.name:
            out.append(_read_meta(d))
    return out


def session(domain: str, session_id: str) -> dict[str, Any]:
    d = session_dir(session_id)
    if not (d / "run.json").is_file() or f"_{domain}_" not in session_id:
        raise RagAgentError(f"no session {session_id}")
    events = []
    if (d / "events.jsonl").is_file():
        events = [
            json.loads(line)
            for line in (d / "events.jsonl").read_text().splitlines()
            if line.strip()
        ]

    def opt(name: str) -> Any:
        p = d / name
        return json.loads(p.read_text()) if p.is_file() else None

    return {
        "meta": _read_meta(d),
        "events": events,
        "output": opt("output.json"),
        "score": opt("score.json"),
    }


CHANGE_FILES = ("rag_agent.md", "merge.md", "decide.md", "review.md", "rag_agent.yaml")


def changes(domain: str, name: str) -> dict[str, Any] | None:
    """A version's change record (s22, r5 on): `based_on` (the version whose prompts and settings
    it copied), `seed` (whose library it starts from), why, and what changed in each file."""
    p = RAG_AGENTS_DIR / domain / name / "changes.json"
    return json.loads(p.read_text()) if p.is_file() else None


PROMPT_FILES = ("rag_agent.md", "decide.md", "merge.md", "review.md")


def describe(domain: str, name: str) -> dict[str, Any]:
    """A version as the Agent tab shows it: settings, prompts, the tools it calls and, from r5 on,
    what changed from the version it was based on. `files` (sha and size of each prompt) lets the
    tab mark a prompt that differs from the parent's, and `library` is its library now."""
    import hashlib

    from tau2_loop.eval.retrieval import variant_tool_rows
    from tau2_loop.workflows import library as wf_library

    agent = load(domain, name)
    record = changes(domain, name)
    d = RAG_AGENTS_DIR / domain / name
    cfg = yaml.safe_load((d / "rag_agent.yaml").read_text())
    files = {
        f: {"sha": hashlib.sha256(p.read_bytes()).hexdigest()[:12], "chars": len(p.read_text())}
        for f in PROMPT_FILES
        if (p := d / f).is_file()
    }
    tool_doc = {
        "name": TOOL_DOC,
        "description": TOOL_DOC_TEXT,
        "type": "read",
        "mutates": False,
        "harness": True,
    }
    st = wf_library.store_for(domain, name) if agent.merge == "concurrent" else None
    return {
        "name": agent.name,
        "based_on": record.get("based_on") if record else None,
        "changes": record,
        "written": str(cfg.get("written") or ""),
        "input": str(cfg.get("input") or "script"),
        "model": agent.model,
        "effort": agent.effort,
        "retrieval": agent.retrieval,
        "tool_mode": agent.tool_mode,
        "max_steps": agent.max_steps,
        "merge_steps": agent.merge_steps,
        "result_chars": agent.result_chars,
        "seed": agent.seed,
        "merge": agent.merge,
        "workers": agent.workers,
        "order": agent.order,
        "order_seed": agent.order_seed,
        "queue": agent.queue,
        "keep_tools": agent.keep_tools,
        "merge_write": agent.merge_write,
        "rubric": list(agent.rubric),
        "tool_docs": agent.tool_docs,
        "describe_named": agent.describe_named,
        "tool_errors": agent.tool_errors,
        "reviewer": list(agent.reviewer) if agent.reviewer else None,
        "prompt": agent.prompt,
        "decide_prompt": agent.decide_prompt,
        "merge_prompt": agent.merge_prompt,
        "review_prompt": agent.review_prompt,
        "files": files,
        "library": {
            "workflows": len(wf_library.entries(domain, name)),
            "commits": len(st.commits()) if st else None,
        },
        "tools": [*variant_tool_rows(agent.retrieval), *([tool_doc] if agent.tool_docs else [])],
    }
