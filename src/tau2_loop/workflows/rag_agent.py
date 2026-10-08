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
    )


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


def system_prompt(agent: RagAgent, env: Any) -> tuple[str, list[dict[str, Any]]]:
    tools, agent_tools, user_tools = tool_lists(env, agent.retrieval)
    text = agent.prompt.format(policy=env.policy, agent_tools=agent_tools, user_tools=user_tools)
    return text, tools


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
) -> SessionMeta:
    """The agentic loop: ask, run every tool call, feed the results back, until the model writes
    the workflow. Every step is appended to `events.jsonl` as it happens, which the viewer polls."""
    events_path = d / "events.jsonl"
    events: list[dict[str, Any]] = []
    started = time.time()

    def emit(row: dict[str, Any]) -> None:
        row = {"t": int((time.time() - started) * 1000), **row}
        events.append(row)
        _append(events_path, row)

    names = {t["function"]["name"] for t in tools}
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{opening}\n\n{question}",
        },
    ]
    emit({"kind": "start", "question": question})

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
                    out = refused or run_tool(c["name"], args)
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

    wf: dict[str, Any] | None = None
    try:
        wf = turns(
            agent.max_steps,
            "Your research steps are used up. Write the workflow JSON now from what you have.",
        )
        if wf is not None and agent.merge and meta.source == "question":
            merge = _merge_concurrent if agent.merge == "concurrent" else _merge
            wf = merge(meta, agent, wf, messages, turns, say, emit, policy)
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


def merge_message(agent: RagAgent, cands: dict[str, list[dict[str, Any]]]) -> str:
    """The harness's message after the draft: the library's candidates in full."""
    pairs = "\n".join(
        f"- {name}: {', '.join(e['job'] for e in es) or 'no library job matches'}"
        for name, es in cands.items()
    )
    return agent.merge_prompt.format(
        pairs=pairs, current=_versions_text(_unique(cands)) or "(none)"
    )


def decide_message(agent: RagAgent, cands: dict[str, list[dict[str, Any]]]) -> str:
    """s21: before it merges, the candidates in full and the question which it will write."""
    pairs = "\n".join(
        f"- {name}: {', '.join(e['job'] for e in es) or 'no library job matches'}"
        for name, es in cands.items()
    )
    return agent.decide_prompt.format(
        pairs=pairs, current=_versions_text(_unique(cands)) or "(none)"
    )


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


def _assess(
    domain: str,
    merged: dict[str, Any],
    lib: list[dict[str, Any]],
    policy: str,
    held: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Each merged job with the library versions it merged and its checks; `why` lists what
    stops it being taken. Under locks (`held`), a job may write only a name it holds and merge
    only library jobs it holds. A job written as edits (r4) is first applied to the version of the
    one job it names in `into`; edits that do not apply are sent back, then not taken."""
    from tau2_loop.workflows import edits, library, rubric

    rows: list[dict[str, Any]] = []
    claimed: dict[str, str] = {}
    for job in merged.get("jobs") or []:
        if not isinstance(job, dict) or not job.get("job"):
            continue
        written = None
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
        c = rubric.check(domain, job, [e["workflow"] for e in prev], policy)
        why = (
            rubric.gate(c)
            if prev
            else [
                f"quotes not found in the documents they cite: {' | '.join(c['D1']['new_unfound'][:4])}"
            ]
            if c["D1"]["new_unfound"]
            else []
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
        rows.append(
            {
                "job": job,
                "previous": prev,
                "checks": c,
                "why": [*why, *blocked],
                "blocked": bool(blocked),
                "written": written,
            }
        )
    return rows


def _send_back(
    agent: RagAgent,
    rows: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
) -> dict[str, Any] | None:
    failing = [r for r in rows if r["why"]]
    lines = "\n".join(f"- {r['job']['job']}: {'; '.join(r['why'])}" for r in failing)
    emit({"kind": "note", "text": f"The merge failed its checks; sent back once.\n{lines}"})
    say(
        "The harness checked your merge and these jobs fail:\n"
        f"{lines}\n\nA merged job keeps every step, tool and quote of the versions it "
        "merged unless its changelog names the change (removed or changed, with why), and "
        "every quote is copied from the document it cites. Reply with the whole JSON again, "
        "fixed."
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
                "taken": keep,
                **({"written": r["written"]} if r.get("written") else {}),
            }
        )
        job.pop("changelog", None)
        job.pop("into", None)
        job.pop("edits", None)
        if aliases:
            job["aliases"] = sorted(set(aliases))
        (taken if keep else rejected).append(job)
    return taken, records, rejected


def _merge(
    meta: SessionMeta,
    agent: RagAgent,
    draft: dict[str, Any],
    messages: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
    policy: str,
) -> dict[str, Any]:
    """Merge the drafted jobs into the library's current versions, check each merge without a
    model, send failures back once, and take only what passes (s20). A merged job that still
    fails keeps the library's previous version; a new job is taken as written."""
    from tau2_loop.workflows import library

    lib = library.entries(meta.domain, agent.name)  # as the last session left it
    cands = merge_candidates(draft, lib)
    emit({"kind": "merge", "candidates": {k: [e["job"] for e in v] for k, v in cands.items()}})
    say(merge_message(agent, cands))
    merged = turns(agent.merge_steps, MERGE_WRITE_NOW)
    rows = _assess(meta.domain, merged, lib, policy) if merged is not None else []
    if merged is not None and any(r["why"] for r in rows):
        again = _send_back(agent, rows, turns, say, emit)
        if again is not None:
            merged, rows = again, _assess(meta.domain, again, lib, policy)
    if merged is None:  # no merge written: keep only the draft's jobs no library job matches
        merged = {
            **draft,
            "jobs": [j for j in draft.get("jobs") or [] if not cands.get(str(j.get("job")))],
        }
        rows = _assess(meta.domain, merged, lib, policy)
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


def _merge_concurrent(
    meta: SessionMeta,
    agent: RagAgent,
    draft: dict[str, Any],
    messages: list[dict[str, Any]],
    turns: Callable[..., Any],
    say: Callable[[str], None],
    emit: Callable[[dict[str, Any]], None],
    policy: str,
) -> dict[str, Any]:
    """s21 (r3): the merge while other workers merge too. Name the jobs it will write, lock them
    all at once (waiting, and logging who holds them, if another worker does), read their newest
    versions, merge and check as r2 does, and commit under the version check. Only the jobs it
    holds may be merged; a merge of another is sent back once, then not taken."""
    from tau2_loop.workflows import library, store

    st = library.store_for(meta.domain, agent.name)
    snap = st.entries()
    cands = merge_candidates(draft, snap)
    emit({"kind": "merge", "candidates": {k: [e["job"] for e in v] for k, v in cands.items()}})
    drafted = list(cands)
    decision = turns(
        0,
        decide_message(agent, cands),
        parse=lambda t: parse_decision(t, drafted),
        again=DECIDE_AGAIN,
    )
    if decision is None:  # no usable answer: hold every candidate, which r2 would have shown it
        decision = {k: [e["job"] for e in v] for k, v in cands.items()}
        emit({"kind": "note", "text": "No usable decision; the library holds every candidate."})
    emit({"kind": "decided", "decision": decision})

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
        say(
            agent.merge_prompt.format(
                pairs=pairs, current=_versions_text(mine, changed) or "(none)"
            )
        )
        held = set(lock.jobs)
        merged = turns(agent.merge_steps, MERGE_WRITE_NOW)
        rows = _assess(meta.domain, merged, heads, policy, held) if merged is not None else []
        if merged is not None and any(r["why"] for r in rows):
            again = _send_back(agent, rows, turns, say, emit)
            if again is not None:
                merged, rows = again, _assess(meta.domain, again, heads, policy, held)
        if merged is None:  # nothing written: only the draft's new jobs, and only those it holds
            merged = {
                **draft,
                "jobs": [
                    j
                    for j in draft.get("jobs") or []
                    if not cands.get(str(j.get("job"))) and j.get("job") in held
                ],
            }
            rows = _assess(meta.domain, merged, heads, policy, held)
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
    )
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


def describe(domain: str, name: str) -> dict[str, Any]:
    """A version as the Agent tab shows it: settings, prompt and the tools it calls."""
    from tau2_loop.eval.retrieval import variant_tool_rows

    agent = load(domain, name)
    return {
        "name": agent.name,
        "model": agent.model,
        "effort": agent.effort,
        "retrieval": agent.retrieval,
        "tool_mode": agent.tool_mode,
        "max_steps": agent.max_steps,
        "result_chars": agent.result_chars,
        "seed": agent.seed,
        "merge": agent.merge,
        "workers": agent.workers,
        "order": agent.order,
        "queue": agent.queue,
        "keep_tools": agent.keep_tools,
        "merge_write": agent.merge_write,
        "prompt": agent.prompt,
        "tools": variant_tool_rows(agent.retrieval),
    }
